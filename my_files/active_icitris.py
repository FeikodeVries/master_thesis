import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import numpy as np
from collections import OrderedDict, defaultdict
from torch.utils.tensorboard import SummaryWriter

from cleanrl.my_files import utils
from cleanrl.my_files import encoder_decoder as coding
from cleanrl.my_files import flow_modules as flow
from cleanrl.my_files import causal_disentanglement as causal


class active_iCITRISVAE(pl.LightningModule):
    """ The main module implementing iCITRIS-VAE """

    def __init__(self, c_hid, num_latents, lr,
                 num_causal_vars,
                 run_name,
                 counter,
                 warmup=100, max_iters=100000,
                 img_width=64,
                 graph_learning_method="ENCO",
                 graph_lr=5e-4,
                 c_in=3,
                 lambda_sparse=0.0,
                 lambda_reg=0.01,
                 num_graph_samples=8,
                 causal_encoder_checkpoint=None,
                 act_fn='silu',
                 beta_classifier=2.0,
                 beta_mi_estimator=2.0,
                 no_encoder_decoder=False,
                 var_names=None,
                 autoregressive_prior=False,
                 use_flow_prior=False,
                 cluster_logging=False,
                 **kwargs):
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
        img_width : int
                    Width of the input image (assumed to be equal to height)
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
        self.save_hyperparameters()
        act_fn_func = utils.get_act_fn(self.hparams.act_fn)

        self.run_name = f"{run_name}__citris"
        self.writer = SummaryWriter(f"runs/{run_name}")
        # TODO: Find out what determines the maximum amount of epochs
        self.current_counter = counter * 29
        print(f"Current counter is: {self.current_counter}")

        # Encoder-Decoder init
        if not self.hparams.no_encoder_decoder:
            self.encoder = coding.Encoder(num_latents=self.hparams.num_latents,
                                   c_hid=self.hparams.c_hid,
                                   c_in=self.hparams.c_in,
                                   width=self.hparams.img_width,
                                   act_fn=act_fn_func,
                                   variational=True)
            self.decoder = coding.Decoder(num_latents=self.hparams.num_latents,
                                   c_hid=self.hparams.c_hid,
                                   c_out=self.hparams.c_in,
                                   width=self.hparams.img_width,
                                   num_blocks=1,
                                   act_fn=act_fn_func)
        # Prior
        self.prior = utils.InstantaneousPrior(num_latents=self.hparams.num_latents,
                                        c_hid=self.hparams.c_hid,
                                        num_blocks=self.hparams.num_causal_vars,
                                        shared_inputs=self.hparams.num_latents,
                                        num_graph_samples=self.hparams.num_graph_samples,
                                        lambda_sparse=self.hparams.lambda_sparse,
                                        graph_learning_method=self.hparams.graph_learning_method,
                                        autoregressive=self.hparams.autoregressive_prior)
        self.intv_classifier = causal.InstantaneousTargetClassifier(
            num_latents=self.hparams.num_latents,
            num_blocks=self.hparams.num_causal_vars,
            c_hid=self.hparams.c_hid * 2,
            num_layers=1,
            act_fn=nn.SiLU,
            var_names=self.hparams.var_names,
            momentum_model=0.9,
            gumbel_temperature=1.0,
            use_normalization=True,
            use_conditional_targets=True)
        self.mi_estimator = causal.MIEstimator(num_latents=self.hparams.num_latents,
                                               num_blocks=self.hparams.num_causal_vars,
                                               c_hid=self.hparams.c_hid,
                                               var_names=self.hparams.var_names,
                                               momentum_model=0.9,
                                               gumbel_temperature=1.0)
        if self.hparams.use_flow_prior:
            self.flow = flow.AutoregNormalizingFlow(self.hparams.num_latents,
                                               num_flows=4,
                                               act_fn=nn.SiLU,
                                               hidden_per_var=16)

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

    def forward(self, x):
        # Full encoding and decoding of samples
        z_mean, z_logstd = self.encoder(x)
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        x_rec = self.decoder(z_sample)
        return x_rec, z_sample, z_mean, z_logstd

    def encode(self, x, random=True):
        # Map input to encoding, e.g. for correlation metrics
        z_mean, z_logstd = self.encoder(x)
        if random:
            z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        else:
            z_sample = z_mean
        if self.hparams.use_flow_prior:
            z_sample, _ = self.flow(z_sample)
        return z_sample

    def configure_optimizers(self):
        # We use different learning rates for the target classifier (higher lr for faster learning).
        graph_params, counter_params, other_params = [], [], []
        for name, param in self.named_parameters():
            if name.startswith('prior.enco') or name.startswith('prior.notears'):
                graph_params.append(param)
            elif name.startswith('intv_classifier') or name.startswith('mi_estimator'):
                counter_params.append(param)
            else:
                other_params.append(param)
        optimizer = optim.AdamW(
            [{'params': graph_params, 'lr': self.hparams.graph_lr, 'weight_decay': 0.0, 'eps': 1e-8},
             {'params': counter_params, 'lr': 2 * self.hparams.lr, 'weight_decay': 1e-4},
             {'params': other_params}], lr=self.hparams.lr, weight_decay=0.0)
        lr_scheduler = utils.CosineWarmupScheduler(optimizer,
                                             warmup=[200 * self.hparams.warmup, 2 * self.hparams.warmup,
                                                     2 * self.hparams.warmup],
                                             offset=[10000, 0, 0],
                                             max_iters=self.hparams.max_iters)
        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def _get_loss(self, batch, mode='train'):
        """ Main training method for calculating the loss """
        if len(batch) == 2:
            imgs, target = batch
            labels = imgs
        else:
            imgs, labels, target = batch
        # En- and decode every element
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))
        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        z_sample = z_sample.unflatten(0, imgs.shape[:2])
        z_sample[:, 0] = z_mean.unflatten(0, imgs.shape[:2])[:, 0]
        z_sample = z_sample.flatten(0, 1)
        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:, 1:].flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in
                                             [z_sample, z_mean, z_logstd, x_rec]]

        if self.hparams.use_flow_prior:
            init_nll = -utils.gaussian_log_prob(z_mean[:, 1:], z_logstd[:, 1:], z_sample[:, 1:]).sum(dim=-1)
            z_sample, ldj = self.flow(z_sample.flatten(0, 1))
            z_sample = z_sample.unflatten(0, (imgs.shape[0], -1))
            ldj = ldj.unflatten(0, (imgs.shape[0], -1))[:, 1:]
            out_nll = self.prior.forward(z_sample=z_sample[:, 1:].flatten(0, 1),
                                         target=target.flatten(0, 1),
                                         z_shared=z_sample[:, :-1].flatten(0, 1),
                                         matrix_exp_factor=np.exp(
                                             self.matrix_exp_scheduler.get_factor(self.global_step)))
            out_nll = out_nll.unflatten(0, (imgs.shape[0], -1))
            p_z = out_nll
            p_z_x = init_nll - ldj
            kld = -(p_z_x - p_z)
            kld = kld.unflatten(0, (imgs.shape[0], -1))
        else:
            # Calculate KL divergence between every pair of frames
            kld = self.prior.forward(z_sample=z_sample[:, 1:].flatten(0, 1),
                                     z_mean=z_mean[:, 1:].flatten(0, 1),
                                     z_logstd=z_logstd[:, 1:].flatten(0, 1),
                                     target=target.flatten(0, 1),
                                     z_shared=z_sample[:, :-1].flatten(0, 1),
                                     matrix_exp_factor=np.exp(self.matrix_exp_scheduler.get_factor(self.global_step)))
            kld = kld.unflatten(0, (imgs.shape[0], -1))

        # Calculate reconstruction loss
        rec_loss = F.mse_loss(x_rec, labels[:, 1:], reduction='none').sum(dim=list(range(2, len(x_rec.shape))))
        # Combine to full loss
        loss = (kld * self.hparams.beta_t1 + rec_loss).mean()
        # Target classifier
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  logger=self if not self.hparams.cluster_logging else None,
                                                  target=target,
                                                  transition_prior=self.prior)
        loss = loss + loss_model + loss_z * self.hparams.beta_classifier
        # Mutual information estimator
        scheduler_factor = self.mi_scheduler.get_factor(self.global_step)
        loss_model_mi, loss_z_mi = self.mi_estimator(z_sample=z_sample,
                                                     logger=None,
                                                     target=target,
                                                     transition_prior=self.prior,
                                                     instant_prob=scheduler_factor)

        loss = loss + loss_model_mi + loss_z_mi * self.hparams.beta_mi_estimator * (1.0 + 2.0 * scheduler_factor)
        # For stabilizing the mean, since it is unconstrained
        loss_z_reg = (z_sample.mean(dim=[0, 1]) ** 2 + z_sample.std(dim=[0, 1]).log() ** 2).mean()

        loss = loss + 0.1 * loss_z_reg

        # Logging
        self.writer.add_scalar("icitris/kld", kld.mean(), (self.current_counter + self.global_step))
        self.writer.add_scalar("icitris/rec_loss_t1",  rec_loss.mean(), (self.current_counter + self.global_step))
        self.writer.add_scalar("icitris/intv_classifier_z", loss_z, (self.current_counter + self.global_step))
        self.writer.add_scalar("icitris/mi_estimator_model", loss_model_mi, (self.current_counter + self.global_step))
        self.writer.add_scalar("icitris/mi_estimator_z", loss_z_mi, (self.current_counter + self.global_step))
        self.writer.add_scalar("icitris/mi_estimator_factor", scheduler_factor, (self.current_counter + self.global_step))
        self.writer.add_scalar("icitris/reg_loss", loss_z_reg, (self.current_counter + self.global_step))

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._get_loss(batch, mode='train')
        self.writer.add_scalar("icitris/train_loss", loss, (self.current_counter + self.global_step))
        if self.global_step == 29:
            self.i += 1
            print(f"Last training_loss: {loss}")
        return loss

    def training_epoch_end(self, *args, **kwargs):
        super().training_epoch_end(*args, **kwargs)
        self.prior.check_trainability()

    def get_causal_rep(self, img):
        """
        Encode an image to latent space, retrieve the current latent to causal assignment and apply it to get the
        minimal causal variables
        :param img:
        :return:
        """
        z_mean, z_logstd = self.encoder(img)
        # Get latent assignment to causal vars in binary
        target_assignment = self.prior.get_target_assignment(hard=True)
        # Assign latent vals to their respective causal var
        latent_causal_assignment = target_assignment * z_mean[0][:, None]

        return latent_causal_assignment
