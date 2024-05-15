import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict, defaultdict
import pathlib
import cv2

from cleanrl.my_files import utils
from cleanrl.my_files import encoder_decoder as coding
from cleanrl.my_files import causal_disentanglement as causal


class iCITRIS(nn.Module):
    """ The main module implementing iCITRIS-VAE """

    def __init__(self, c_hid, num_latents,
                 num_causal_vars,
                 run_name,
                 width=64,
                 graph_learning_method="ENCO",
                 c_in=3,
                 lambda_sparse=0.0,
                 num_graph_samples=8,
                 act_fn='silu',
                 beta_classifier=4.0,
                 beta_mi_estimator=0.0,
                 beta_t1=1.0,
                 var_names=None,
                 autoregressive_prior=False,
                 cluster_logging=False,
                 ):
        """
        Parameters
        ----------
        c_hid : int
                Hidden dimensionality to use in the network
        num_latents : int
                      Number of latent variables in the VAE
        lr : float
             Learning rate to use for training
        num_causal_vars : int
                          Number of causal variables
        warmup : int
                 Number of learning rate warmup steps
        max_iters : int
                    Number of max. training iterations. Needed for
                    cosine annealing of the learning rate.
        width : int
                    Width to be used in calculating the number of convolutional blocks with int(np.log2(width) - 2)
        graph_learning_method : str
                                Which graph learning method to use in the prior.
                                Options: ENCO, NOTEARS
        graph_lr : float
                   Learning rate of the graph parameters
        c_in : int
               Number of input channels (3 for RGB)
        lambda_sprase : float
                        Regularizer for encouraging sparse graphs
        lambda_reg : float
                     Regularizer for promoting intervention-independent information to be modeled
                     in psi(0)
        num_graph_samples : int
                            Number of graph samples to use in ENCO's gradient estimation
        beta_classifier : float
                          Weight of the target classifier in training
        beta_mi_estimator : float
                            Weight of the mutual information estimator in training
        causal_encoder_checkpoint : str
                                    Path to the checkpoint of a Causal-Encoder model to use for
                                    the triplet evaluation.
        act_fn : str
                 Activation function to use in the encoder and decoder network.
        no_encoder_decoder : bool
                             If True, no encoder or decoder are initialized. Used for CITRIS-NF
        var_names : Optional[List[str]]
                    Names of the causal variables, for plotting and logging
        autoregressive_prior : bool
                               If True, the prior per causal variable is autoregressive
        use_flow_prior : bool
                         If True, use a NF prior in the VAE.
        cluster_logging : bool
                          If True, the logging will be reduced to a minimum
        """
        super().__init__()
        act_fn_func = utils.get_act_fn(act_fn)

        self.cluster_logging = cluster_logging
        self.beta_mi_estimator = beta_mi_estimator
        self.beta_classifier = beta_classifier
        self.beta_t1 = beta_t1

        # Encoder-Decoder init
        self.encoder = coding.Encoder(num_latents=num_latents,
                                      c_hid=c_hid,
                                      c_in=c_in,
                                      width=width,
                                      act_fn=act_fn_func,
                                      variational=True)
        self.decoder = coding.Decoder(num_latents=num_latents,
                                      c_hid=c_hid,
                                      c_out=c_in,
                                      width=width,
                                      num_blocks=1,
                                      act_fn=act_fn_func)
        # Prior
        self.prior = utils.InstantaneousPrior(num_latents=num_latents,
                                              c_hid=c_hid,
                                              num_blocks=num_causal_vars,
                                              shared_inputs=num_latents,
                                              num_graph_samples=num_graph_samples,
                                              lambda_sparse=lambda_sparse,
                                              graph_learning_method=graph_learning_method,
                                              autoregressive=autoregressive_prior)
        self.intv_classifier = causal.InstantaneousTargetClassifier(
            num_latents=num_latents,
            num_blocks=num_causal_vars,
            c_hid=c_hid * 2,
            num_layers=1,
            act_fn=nn.SiLU,
            var_names=var_names,
            momentum_model=0.9,
            gumbel_temperature=1.0,
            use_normalization=True,
            use_conditional_targets=True)
        self.mi_estimator = causal.MIEstimator(num_latents=num_latents,
                                               num_blocks=num_causal_vars,
                                               c_hid=c_hid,
                                               var_names=var_names,
                                               momentum_model=0.9,
                                               gumbel_temperature=1.0)
        self.mi_scheduler = utils.SineWarmupScheduler(warmup=50000,
                                                      start_factor=0.004,
                                                      end_factor=1.0,
                                                      offset=20000)
        self.matrix_exp_scheduler = utils.SineWarmupScheduler(warmup=100000,
                                                              start_factor=-6,
                                                              end_factor=4,
                                                              offset=10000)
        self.causal_encoder = None
        self.all_val_dists = defaultdict(list)
        self.all_v_dicts = []
        self.prior_t1 = self.prior

    def encode(self, x, random=True):
        # Map input to encoding, e.g. for correlation metrics
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        return z_mean, z_sample, z_logstd

    def decode(self, z_sample):
        return self.decoder(z_sample)

    def get_loss(self, batch, target, global_step, epoch, data_loader):
        """ Main training method for calculating the loss """
        imgs = batch.cuda()
        target = target.cuda().flatten(0, 1)
        labels = imgs

        # En- and decode every element
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        z_sample = z_sample.unflatten(0, imgs.shape[:2])
        z_sample[:, 0] = z_mean.unflatten(0, imgs.shape[:2])[:, 0]
        z_sample = z_sample.flatten(0, 1)

        test = 1

        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:, 1:].flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in
                                             [z_sample, z_mean, z_logstd, x_rec]]

        # Calculate KL divergence between every pair of frames
        kld = self.prior.forward(z_sample=z_sample[:, 1:].flatten(0, 1),
                                 z_mean=z_mean[:, 1:].flatten(0, 1),
                                 z_logstd=z_logstd[:, 1:].flatten(0, 1),
                                 target=target.flatten(0, 1),
                                 z_shared=z_sample[:, :-1].flatten(0, 1),
                                 matrix_exp_factor=np.exp(self.matrix_exp_scheduler.get_factor(global_step)))
        kld = kld.unflatten(0, (imgs.shape[0], -1))

        # Calculate reconstruction loss
        rec_loss = F.mse_loss(x_rec, labels[:, 1:], reduction='none').sum(dim=list(range(2, len(x_rec.shape))))
        # Combine to full loss
        loss = (kld * self.beta_t1 + rec_loss).mean()
        # Target classifier
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  logger=None,
                                                  target=target,
                                                  transition_prior=self.prior)
        loss = loss + loss_model + loss_z * self.beta_classifier
        # Mutual information estimator
        scheduler_factor = self.mi_scheduler.get_factor(global_step)
        loss_model_mi, loss_z_mi = self.mi_estimator(z_sample=z_sample,
                                                     logger=None,
                                                     target=target,
                                                     transition_prior=self.prior,
                                                     instant_prob=scheduler_factor)

        loss = loss + loss_model_mi + loss_z_mi * self.beta_mi_estimator * (1.0 + 2.0 * scheduler_factor)
        # For stabilizing the mean, since it is unconstrained
        loss_z_reg = (z_sample.mean(dim=[0, 1]) ** 2 + z_sample.std(dim=[0, 1]).log() ** 2).mean()
        loss = loss + 0.1 * loss_z_reg
        logging = {'kld': kld.mean(), 'rec_loss_t1': rec_loss.mean(), 'intv_classifier_z': loss_z,
                   'mi_estimator_model': loss_model_mi, 'mi_estimator_z': loss_z_mi,
                   'mi_estimator_factor': scheduler_factor, 'reg_loss': loss_z_reg}

        if epoch == 0 and data_loader == 0:
            # Save image of an original + reconstruction every policy rollout
            path = str(pathlib.Path().absolute()) + '/my_files/data/input_reconstructions'

            original_img = imgs[-1].permute(0, 2, 3, 1).cpu().detach().numpy()
            rec_img = x_rec[-1].permute(0, 2, 3, 1).cpu().detach().numpy()
            for i in range(len(original_img)):
                cv2.imwrite(f"{path}/{global_step}_{i}_original.png", cv2.cvtColor(255*original_img[i],
                                                                                   cv2.COLOR_RGB2BGR))
            for i in range(len(rec_img)):
                rec_loss_current = round(rec_loss[-1][i].item())
                cv2.imwrite(f"{path}/{global_step}_{i+1}_recloss{rec_loss_current}.png",
                            cv2.cvtColor(255*rec_img[i], cv2.COLOR_RGB2BGR))

        return loss, logging

    def get_causal_rep(self, x):
        """
        Encode an image to latent space, retrieve the current latent to causal assignment and apply it to get the
        minimal causal variables
        :param x: batch of unflattened image tensors
        :return:
        """
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        # Get latent assignment to causal vars in binary
        # TODO: Usually the output of the encoder is provided as input to the neural net, as the encoder is trained
        #  using the causal logic
        # # TODO: Move the latent assignment from binary to probability and test that for training
        target_assignment = self.prior.get_target_assignment(hard=False)
        # Assign latent vals to their respective causal var
        latent_causal_assignment = [target_assignment * z_sample[i][:, None] for i in range(len(z_sample))]
        latent_causal_assignment = torch.stack(latent_causal_assignment, dim=0)

        return latent_causal_assignment

    def set_train(self, training=True):
        if training:
            self.encoder.train()
            self.decoder.train()
            self.intv_classifier.train()
            self.prior.train()
            self.mi_estimator.train()
        else:
            self.encoder.eval()
            self.decoder.eval()
            self.intv_classifier.eval()
            self.prior.eval()
            self.mi_estimator.eval()

    def clip_gradients(self):
        nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.intv_classifier.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.prior.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.mi_estimator.parameters(), 0.5)

    def get_params(self):
        """
        Get the relevant parameters to feed to the optimizer
        """
        graph_params, counter_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if name.startswith('prior.enco') or name.startswith('prior.notears'):
                graph_params.append(param)
            elif name.startswith('intv_classifier') or name.startswith('mi_estimator'):
                counter_params.append(param)
            else:
                other_params.append(param)

        return graph_params, counter_params, other_params
