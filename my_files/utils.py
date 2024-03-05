import os.path
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from cleanrl.my_files.datasets import ReacherDataset
import pathlib
import torch.utils.data as data
from pytorch_lightning.callbacks import ModelCheckpoint
from collections import defaultdict
import scipy.linalg


# METHODS: ENCODER-DECODER
class Encoder(nn.Module):
    """
    Convolution encoder network
    We use a stack of convolutions with strides in every second convolution to reduce
    dimensionality. For the datasets in question, the network showed to be sufficient.
    """

    def __init__(self, c_hid, num_latents,
                 c_in=3,
                 width=32,
                 act_fn=lambda: nn.SiLU(),
                 use_batch_norm=True,
                 variational=True):
        super().__init__()
        num_layers = int(np.log2(width) - 2)
        NormLayer = nn.BatchNorm2d if use_batch_norm else nn.InstanceNorm2d
        self.scale_factor = nn.Parameter(torch.zeros(num_latents,))
        self.variational = variational
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(c_in if l_idx == 0 else c_hid,
                              c_hid,
                              kernel_size=3,
                              padding=1,
                              stride=2,
                              bias=False),
                    PositionLayer(c_hid) if l_idx == 0 else nn.Identity(),
                    NormLayer(c_hid),
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1, bias=False),
                    NormLayer(c_hid),
                    act_fn()
                ) for l_idx in range(num_layers)
            ],
            nn.Flatten(),
            nn.Linear(4*4*c_hid, 4*c_hid),
            nn.LayerNorm(4*c_hid),
            act_fn(),
            nn.Linear(4*c_hid, (2*num_latents if self.variational else num_latents))
        )

    def forward(self, img):
        feats = self.net(img)
        if self.variational:
            mean, log_std = feats.chunk(2, dim=-1)
            s = F.softplus(self.scale_factor)
            log_std = torch.tanh(log_std / s) * s  # Stabilizing the prediction
            return mean, log_std
        else:
            return feats


class Decoder(nn.Module):
    """
    Convolutional decoder network
    We use a ResNet-based decoder network with upsample layers to increase the
    dimensionality stepwise. We add positional information in the ResNet blocks
    for improved position-awareness, similar to setups like SlotAttention.
    """

    def __init__(self, c_hid, num_latents,
                 num_labels=-1,
                 width=32,
                 act_fn=lambda: nn.SiLU(),
                 use_batch_norm=True,
                 num_blocks=1,
                 c_out=-1):
        super().__init__()
        if num_labels > 1:
            out_act = nn.Identity()
        else:
            num_labels = 3 if c_out <= 0 else c_out
            out_act = nn.Tanh()
        NormLayer = nn.BatchNorm2d if use_batch_norm else nn.InstanceNorm2d
        self.width = width
        self.linear = nn.Sequential(
            nn.Linear(num_latents, 4 * c_hid),
            nn.LayerNorm(4 * c_hid),
            act_fn(),
            nn.Linear(4 * c_hid, 4 * 4 * c_hid)
        )
        num_layers = int(np.log2(width) - 2)
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                    *[ResidualBlock(nn.Sequential(
                        NormLayer(c_hid),
                        act_fn(),
                        nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
                        PositionLayer(c_hid),
                        NormLayer(c_hid),
                        act_fn(),
                        nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1)
                    )) for _ in range(num_blocks)]
                ) for _ in range(num_layers)
            ],
            NormLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, 1),
            PositionLayer(c_hid),
            NormLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, num_labels, 1),
            out_act
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class ResidualBlock(nn.Module):
    """ Simple module for residual blocks """

    def __init__(self, net, skip_connect=None):
        super().__init__()
        self.net = net
        self.skip_connect = skip_connect if skip_connect is not None else nn.Identity()

    def forward(self, x):
        return self.skip_connect(x) + self.net(x)


class PositionLayer(nn.Module):
    """ Module for adding position features to images """

    def __init__(self, hidden_dim):
        super().__init__()
        self.pos_embed = nn.Linear(2, hidden_dim)

    def forward(self, x):
        pos = create_pos_grid(x.shape[2:], x.device)
        pos = self.pos_embed(pos)
        pos = pos.permute(2, 0, 1)[None]
        x = x + pos
        return x


# METHODS: MODULES
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    """ Learning rate scheduler with Cosine annealing and warmup """

    def __init__(self, optimizer, warmup, max_iters, min_factor=0.05, offset=0):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_factor = min_factor
        self.offset = offset
        super().__init__(optimizer)
        if isinstance(self.warmup, list) and not isinstance(self.offset, list):
            self.offset = [self.offset for _ in self.warmup]

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        if isinstance(lr_factor, list):
            return [base_lr * f for base_lr, f in zip(self.base_lrs, lr_factor)]
        else:
            return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        lr_factor = lr_factor * (1 - self.min_factor) + self.min_factor
        if isinstance(self.warmup, list):
            new_lr_factor = []
            for o, w in zip(self.offset, self.warmup):
                e = max(0, epoch - o)
                l = lr_factor * ((e * 1.0 / w) if e <= w and w > 0 else 1)
                new_lr_factor.append(l)
            lr_factor = new_lr_factor
        else:
            epoch = max(0, epoch - self.offset)
            if epoch <= self.warmup and self.warmup > 0:
                lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class SineWarmupScheduler(object):
    """ Warmup scheduler used for KL divergence, if chosen """

    def __init__(self, warmup, start_factor=0.1, end_factor=1.0, offset=0):
        super().__init__()
        self.warmup = warmup
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.offset = offset

    def get_factor(self, step):
        step = step - self.offset
        if step >= self.warmup:
            return self.end_factor
        elif step < 0:
            return self.start_factor
        else:
            v = self.start_factor + (self.end_factor - self.start_factor) * 0.5 * (1 - np.cos(np.pi * step / self.warmup))
            return v


class MultivarLinear(nn.Module):

    def __init__(self, input_dims, output_dims, extra_dims, bias=True):
        """
        Linear layer, which effectively applies N independent linear layers in parallel.

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        output_dims : int
                      Number of output dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.extra_dims = extra_dims

        self.weight = nn.Parameter(torch.zeros(*extra_dims, output_dims, input_dims))
        if bias:
            self.bias = nn.Parameter(torch.zeros(*extra_dims, output_dims))

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        # Shape preparation
        x_extra_dims = x.shape[1:-1]
        if len(x_extra_dims) > 0:
            for i in range(len(x_extra_dims)):
                assert x_extra_dims[-(i + 1)] == self.extra_dims[-(i + 1)], \
                    "Shape mismatch: X=%s, Layer=%s" % (str(x.shape), str(self.extra_dims))
        for _ in range(len(self.extra_dims) - len(x_extra_dims)):
            x = x.unsqueeze(dim=1)

        # Unsqueeze
        x = x.unsqueeze(dim=-1)
        weight = self.weight.unsqueeze(dim=0)

        # Linear layer
        out = torch.matmul(weight, x).squeeze(dim=-1)

        # Bias
        if hasattr(self, 'bias'):
            bias = self.bias.unsqueeze(dim=0)
            out = out + bias
        return out


class MultivarLayerNorm(nn.Module):

    def __init__(self, input_dims, extra_dims):
        """
        Normalization layer with the same properties as MultivarLinear.

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.extra_dims = np.prod(extra_dims)

        self.norm = nn.GroupNorm(self.extra_dims, self.input_dims * self.extra_dims)

    def forward(self, x):
        shape = x.shape
        out = self.norm(x.flatten(1, -1))
        out = out.reshape(shape)
        return out


class MultivarStableTanh(nn.Module):

    def __init__(self, input_dims, extra_dims, init_bias=0.0):
        """
        Stabilizing Tanh layer like in flows.

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.extra_dims = np.prod(extra_dims)

        self.scale_factors = nn.Parameter(torch.zeros(self.extra_dims, self.input_dims).fill_(init_bias))

    def forward(self, x):
        sc = torch.exp(self.scale_factors)[None]
        out = torch.tanh(x / sc) * sc
        return out


class AutoregLinear(nn.Module):

    def __init__(self, num_vars, inp_per_var, out_per_var, diagonal=False,
                 no_act_fn_init=False,
                 init_std_factor=1.0,
                 init_bias_factor=1.0,
                 init_first_block_zeros=False):
        """
        Autoregressive linear layer, where the weight matrix is correspondingly masked.

        Parameters
        ----------
        num_vars : int
                   Number of autoregressive variables/steps.
        inp_per_var : int
                      Number of inputs per autoregressive variable.
        out_per_var : int
                      Number of outputs per autoregressvie variable.
        diagonal : bool
                   If True, the n-th output depends on the n-th input.
                   If False, the n-th output only depends on the inputs 1 to n-1
        """
        super().__init__()
        self.linear = nn.Linear(num_vars * inp_per_var,
                                num_vars * out_per_var)
        mask = torch.zeros_like(self.linear.weight.data)
        init_kwargs = {}
        if no_act_fn_init:  # Set kaiming to init for linear act fn
            init_kwargs['nonlinearity'] = 'leaky_relu'
            init_kwargs['a'] = 1.0
        for out_var in range(num_vars):
            out_start_idx = out_var * out_per_var
            out_end_idx = (out_var + 1) * out_per_var
            inp_end_idx = (out_var + (1 if diagonal else 0)) * inp_per_var
            if inp_end_idx > 0:
                mask[out_start_idx:out_end_idx, :inp_end_idx] = 1.0
                if out_var == 0 and init_first_block_zeros:
                    self.linear.weight.data[out_start_idx:out_end_idx, :inp_end_idx].fill_(0.0)
                else:
                    nn.init.kaiming_uniform_(self.linear.weight.data[out_start_idx:out_end_idx, :inp_end_idx],
                                             **init_kwargs)
        self.linear.weight.data.mul_(init_std_factor)
        self.linear.bias.data.mul_(init_bias_factor)
        self.register_buffer('mask', mask)

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


class TanhScaled(nn.Module):
    """ Tanh activation function with scaling factor """

    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        assert self.scale > 0, 'Only positive scales allowed.'

    def forward(self, x):
        return torch.tanh(x / self.scale) * self.scale


# METHODS: PRIOR
class InstantaneousPrior(nn.Module):
    """ The prior of iCITRIS """

    def __init__(self, num_latents,
                 c_hid,
                 num_blocks,
                 shared_inputs=0,
                 lambda_sparse=0.02,
                 gumbel_temperature=1.0,
                 num_graph_samples=8,
                 graph_learning_method='ENCO',
                 autoregressive=False):
        """
        For simplicity, this prior considers that zpsi{0} is an empty set.
        However, it can be easily extended by adding an extra causal dimension
        with set orientations.

        Parameters
        ----------
        num_latents : int
                      Number of latent variables
        c_hid : int
                Hidden dimension in the networks
        shared_inputs : int
                        Number of dimensions to condition the prior on.
                        Usually number of latents, i.e. the previous time step.
        lambda_sparse : float
                        Sparsity regularizer in the causal discovery method
        gumbel_temperature : float
                             Temperature for the gumbel softmax operations
        num_graph_samples : int
                            Number of graph samples to use in ENCO's gradient estimation
        graph_learning_method : str
                                Method to use for graph learning. Options: ENCO, NOTEARS
        autoregressive : bool
                         If True, the prior within a causal variable is autoregressive or not
        """
        super().__init__()
        self.num_latents = num_latents
        self.num_blocks = num_blocks
        self.shared_inputs = shared_inputs
        self.gumbel_temperature = gumbel_temperature
        self.graph_learning_method = graph_learning_method
        self.lambda_sparse = lambda_sparse
        self.autoregressive = autoregressive
        self.num_graph_samples = num_graph_samples

        self.target_params = nn.Parameter(torch.zeros(num_latents, num_blocks))
        self.net = nn.Sequential(
            MultivarLinear(self.num_latents * 2 + self.num_blocks + self.shared_inputs, c_hid, [self.num_latents]),
            MultivarLayerNorm(c_hid, [self.num_latents]),
            nn.SiLU(),
            MultivarLinear(c_hid, c_hid, [self.num_latents]),
            MultivarLayerNorm(c_hid, [self.num_latents]),
            nn.SiLU(),
            MultivarLinear(c_hid, 2,
                           [self.num_latents]),
            MultivarStableTanh(2, [self.num_latents])
        )
        self.net[-2].weight.data.fill_(0.0)

        if self.graph_learning_method == 'ENCO':
            self.enco_theta = nn.Parameter(torch.zeros(num_blocks, num_blocks))
            self.enco_theta.data.masked_fill_(torch.eye(self.enco_theta.shape[0], dtype=torch.bool), -9e15)
            self.enco_gamma = nn.Parameter(self.enco_theta.data.clone())
            self.enco_gamma.data.masked_fill_(~torch.eye(self.enco_theta.shape[0], dtype=torch.bool), 4)
        elif self.graph_learning_method == 'NOTEARS':
            self.notears_params = nn.Parameter(torch.zeros(num_blocks, num_blocks))
            self.notears_params.data.masked_fill_(torch.eye(self.notears_params.shape[0], dtype=torch.bool), -9e15)
            self.num_graph_samples = 1
        else:
            assert False, f'Unknown graph learning method \"{self.graph_learning_method}\"'

        self.train_graph = True
        self.gradient_efficient = True

    def forward(self, z_sample, target, z_mean=None, z_logstd=None, num_graph_samples=-1, z_shared=None,
                matrix_exp_factor=0.0):
        """
        Calculate the NLL or KLD of samples under an instantaneous prior.

        Parameters
        ----------
        z_sample : torch.FloatTensor, shape [Batch_size, num_latents] or [Batch_size, num_samples, num_latents]
                   The latents of the time step for which the NLL/KLD should be calculated, i.e. z^t+1
        target : torch.FloatTensor, shape [Batch_size, num_causal_vars] or torch.LongTensor, shape [Batch_size]
                 The intervention targets to consider on C^t+1
        z_mean : Optional, torch.FloatTensor, shape [Batch_size, num_latents]
                 If given, the prior calculates the KL divergence to the distribution of a Gaussian
                 with mean z_mean and standard deviation exp(z_logstd)
        z_logstd : Optional, torch.FloatTensor, shape [Batch_size, num_latents]
                   See z_mean
        num_graph_samples : int
                            Number of graph samples to consider for estimating the gradients of
                            ENCO. If smaller than zero, the default number of samples in
                            self.num_graph_samples is used.
        z_shared : Optional, torch.FloatTensor, shape [Batch_size, shared_inputs]
                   The additional latents to consider in the prior (usually z^t)
        matrix_exp_factor : float
                            The weight of the acyclicity regularizer of NOTEARS
        """
        batch_size = z_sample.shape[0]
        if len(z_sample.shape) == 2:
            z_sample = z_sample[:, None]  # Add a 'sample' dimension
            if z_shared is not None:
                z_shared = z_shared[:, None]
        num_z_samples = z_sample.shape[1]
        num_graph_samples = self.num_graph_samples if num_graph_samples <= 0 else num_graph_samples

        # Target preparation
        if len(target.shape) == 1:
            target_oh = F.one_hot(target, num_classes=self.num_blocks)
        else:
            target_oh = target
        target_probs = torch.softmax(self.target_params, dim=-1)
        target_samples = F.gumbel_softmax(self.target_params[None].expand(target.shape[0], -1, -1),
                                          tau=self.gumbel_temperature, hard=True)

        # Graph sampling
        if self.train_graph:  # When the graph parameters converged, we can stop the graph training for efficiency
            if self.graph_learning_method == 'ENCO':
                edge_prob = torch.sigmoid(self.enco_theta.detach()) * torch.sigmoid(self.enco_gamma.detach())
                adj_matrix = torch.bernoulli(edge_prob[None, None].expand(batch_size, num_graph_samples, -1, -1))
            else:
                edge_logits = torch.stack([torch.zeros_like(self.notears_params), self.notears_params], dim=-1)
                adj_matrix = F.gumbel_softmax(edge_logits[None, None].expand(batch_size, num_graph_samples, -1, -1, -1),
                                              tau=self.gumbel_temperature, hard=True)[..., 1]
        else:
            adj_matrix = self.get_adj_matrix(hard=True)[None, None].expand(batch_size, num_graph_samples, -1, -1)
        # adj_matrix: shape [batch_size, num_graph_samples, causal_vars (parents), causal_vars (children)]

        """
        The general idea is to run the MLP for every latent variable for every possible causal assignment of it.
        The sample assignment of target_samples is only used for inputs.
        """
        # Map the causal -> causal adjacency matrix to latent -> causal
        latent_to_causal = (target_samples[:, None, :, :, None] * adj_matrix[:, :, None, :, :]).sum(dim=-2)
        # latent_to_causal: shape [batch, num_graph_samples, latent_vars, causal_vars]

        # Transpose for mask because adj[i,j] means that i->j
        latent_mask = latent_to_causal.transpose(-2, -1)
        # latent_mask: shape [batch, num_graph_samples, causal_vars, latent_vars]

        # Expand mask to run MLP for every latent variable for every possible causal assignment
        # Make sure that no latent variable sees itself
        latent_eye_matrix = torch.eye(latent_mask.shape[-1], device=latent_mask.device)
        latent_mask = latent_mask[:, :, :, None, :] * (1 - latent_eye_matrix)[None, None, None, :, :]
        latent_mask = latent_mask[:, None].expand(-1, num_z_samples, -1, -1, -1, -1)
        # latent_mask: shape [batch, num_z_samples, num_graph_samples, causal_vars, latent_vars (MLP for each latent), latent_vars (input to NN)]

        # Mask out inputs for those causal variables that have been intervened upon
        latent_mask = latent_mask * (1 - target_oh[:, None, None, :, None, None])
        if self.autoregressive:
            # Add autoregressive-style variables
            ones_tril = target_samples.new_ones(self.num_latents, self.num_latents).tril(diagonal=-1)
            extra_mask = target_samples.transpose(1, 2)[:, None, None, :, None, :] * ones_tril[None, None, None, None,
                                                                                     :, :]
            latent_mask = latent_mask + extra_mask

        # Prepare targets as input to the MLPs (-1 for masked elements)
        target_eye_matrix = torch.eye(target_oh.shape[-1], device=latent_mask.device)[None, None, :, None, :]
        target_input = target_oh[:, None, None, None, None, :].expand(-1, num_z_samples, num_graph_samples, -1,
                                                                      self.num_latents,
                                                                      -1) * target_eye_matrix  # - (1 - target_eye_matrix)
        # target_input: shape [batch, num_z_samples, num_graph_samples, causal_vars, latent_vars, num_targets]

        # Prepared shared inputs if any
        if z_shared is None:
            z_shared = []
        else:
            # z_shared: shape [batch, num_z_samples, shared_inputs] -> [batch, num_z_samples, num_graph_samples, causal_vars, latent_vars, shared_inputs]
            z_shared = z_shared[:, :, None, None, None, :].expand(-1, -1, num_graph_samples, self.num_blocks,
                                                                  self.num_latents, -1)
            z_shared = [z_shared]

        # Obtain predictions from network
        z_inp = torch.cat(
            [z_sample[:, :, None, None, None, :] * latent_mask, latent_mask * 2 - 1, target_input] + z_shared, dim=-1)
        s = 1
        # For efficiency, we run 1 sample differentiably for the distribution parameters,
        # and all without gradients since ENCO doesn't need gradients through the networks
        if self.gradient_efficient and self.train_graph and num_graph_samples > 1:
            z_inp_flat = z_inp[:, :, :s].flatten(0, 3)
            preds = self.net(z_inp_flat)
            preds = preds.unflatten(0, (batch_size, num_z_samples, s, self.num_blocks))
            with torch.no_grad():
                z_inp_flat_nograd = z_inp[:, :, s:].detach().flatten(0, 3)
                if self.num_latents <= 16:
                    preds_nograd = self.net(z_inp_flat_nograd)
                else:
                    preds_nograd = torch.cat([self.net(z_part) for z_part in z_inp_flat_nograd.chunk(2, dim=0)], dim=0)
                preds_nograd = preds_nograd.unflatten(0, (
                batch_size, num_z_samples, num_graph_samples - s, self.num_blocks)).detach()
            preds = torch.cat([preds, preds_nograd], dim=2)
        else:
            z_inp_flat = z_inp.flatten(0, 3)
            preds = self.net(z_inp_flat)
            preds = preds.unflatten(0, (batch_size, num_z_samples, num_graph_samples, self.num_blocks))
        prior_mean, prior_logstd = preds.unbind(dim=-1)
        # prior_mean: shape [batch, num_graph_sampels, causal_vars, latent_vars]

        # Calculate KL divergence or Gaussian Log Prob if we have samples
        if z_mean is not None and z_logstd is not None:
            if len(z_mean.shape) == 2:
                z_mean = z_mean[:, None]
                z_logstd = z_logstd[:, None]
            kld = kl_divergence(z_mean[:, :, None, None], z_logstd[:, :, None, None], prior_mean, prior_logstd).mean(
                dim=1)
        else:
            kld = -gaussian_log_prob(prior_mean[:, None], prior_logstd[:, None], z_sample[:, :, None, None, None, :])
            kld = kld.mean(dim=[1, 2])  # Mean samples over samples
        # kld: shape [batch, num_graph_samples, causal_vars, latent_vars]

        # VAE KLD
        if not self.gradient_efficient:
            kld_vae = kld.mean(dim=1)  # Mean over graph samples
        else:
            kld_vae = kld[:, :s].mean(dim=1)
        # kld_vae: shape [batch, causal_vars, latent_vars]
        kld_vae = (kld_vae * (target_probs.transpose(0, 1)[None] + 1e-4)).sum(dim=1)  # Weighted mean over causal vars
        kld_vae = kld_vae.sum(dim=-1)  # Sum over latents
        # kld_vae: shape [batch]

        if self.training and self.train_graph:
            if self.graph_learning_method == 'ENCO':
                # ENCO gradients
                with torch.no_grad():
                    # Get theta and gamma probabilities for gradients
                    orient_probs = torch.sigmoid(self.enco_theta)
                    gamma_probs = torch.sigmoid(self.enco_gamma)

                    if len(adj_matrix.shape) == 3:  # Expand adj matrix by batch size if not sampled with it
                        adj_matrix = adj_matrix[None].expand(batch_size, -1, -1, -1)
                    # The NLL of causal var is the sum of all latent vars weighted by their probability to belong
                    # to the corresponding causal variable
                    kld_causal = (kld[:, :, :, :] * target_probs.transpose(0, 1)[None, None]).sum(
                        dim=-1)  # Sum over latents
                    self.kld_per_causal = kld_causal.detach()
                    kld_causal = kld_causal[:, :, None, :]  # Expand since we only consider children NLL

                    # Standard ENCO gradients. Note that we take the positive and negative average per batch element
                    num_pos = adj_matrix.sum(dim=1)
                    num_neg = adj_matrix.shape[1] - num_pos
                    mask = ((num_pos > 0) * (num_neg > 0)).float()
                    pos_grads = (kld_causal * adj_matrix).sum(dim=1) / num_pos.clamp_(min=1e-5)
                    neg_grads = (kld_causal * (1 - adj_matrix)).sum(dim=1) / num_neg.clamp_(min=1e-5)
                    gamma_grads = mask * gamma_probs * (1 - gamma_probs) * orient_probs * (
                                pos_grads - neg_grads + self.lambda_sparse)
                    theta_grads = mask * orient_probs * (1 - orient_probs) * gamma_probs * (pos_grads - neg_grads)

                    gamma_grads = gamma_grads * (1 - target_oh[:, None, :])  # Ignore gradients for intervened vars
                    gamma_grads[:, torch.arange(gamma_grads.shape[1]), torch.arange(gamma_grads.shape[2])] = 0.

                    theta_grads = theta_grads * target_oh[:, :, None]  # Only gradients for intervened vars
                    theta_grads = theta_grads * (
                                1 - target_oh[:, :, None] * target_oh[:, None, :])  # Mask out intervened to intervened
                    theta_grads = theta_grads - theta_grads.transpose(1,
                                                                      2)  # theta_ij = -theta_ji, and implicitly theta_ii=0

                    self.gamma_grads = gamma_grads.mean(dim=0).detach()
                    self.theta_grads = theta_grads.mean(dim=0).detach()
                # Hook gradients to ENCO parameters on the backward pass
                if kld_vae.requires_grad:
                    kld_vae.register_hook(lambda *args, **kwargs: update_enco_params(self, *args, **kwargs))
            elif self.graph_learning_method == 'NOTEARS':
                # Gradients already through the Gumbel Softmax
                edge_probs = torch.sigmoid(self.notears_params)
                sparsity_regularizer = edge_probs.sum() * self.lambda_sparse
                matrix_exponential = torch.trace(torch.matrix_exp(edge_probs)) - edge_probs.shape[0]
                kld_vae = kld_vae + sparsity_regularizer + matrix_exponential * matrix_exp_factor
        else:
            if self.graph_learning_method == 'ENCO':
                self.gamma_grads = None
                self.theta_grads = None

        return kld_vae

    def logging(self, logger):
        if self.graph_learning_method == 'ENCO':
            for i in range(self.enco_theta.shape[0]):
                for j in range(self.enco_theta.shape[1]):
                    if i == j:
                        continue
                    logger.log(f'enco_theta_{i}_to_{j}', self.enco_theta[i, j], on_step=False, on_epoch=True)
                    logger.log(f'enco_gamma_{i}_to_{j}', self.enco_gamma[i, j], on_step=False, on_epoch=True)
                    if self.enco_theta.grad is not None:
                        logger.log(f'enco_theta_{i}_to_{j}_grads', self.enco_theta.grad[i, j], on_step=False,
                                   on_epoch=True)
                    if self.enco_gamma.grad is not None:
                        logger.log(f'enco_gamma_{i}_to_{j}_grads', self.enco_gamma.grad[i, j], on_step=False,
                                   on_epoch=True)
        elif self.graph_learning_method == 'NOTEARS':
            for i in range(self.notears_params.shape[0]):
                for j in range(self.notears_params.shape[1]):
                    if i == j:
                        continue
                    logger.log(f'notears_{i}_to_{j}', self.notears_params[i, j], on_step=False, on_epoch=True)
        soft_adj = self.get_adj_matrix(hard=False)
        logger.log('matrix_exponential', torch.trace(torch.matrix_exp(soft_adj)))
        max_grad = (soft_adj * (1 - soft_adj)).max()
        logger.log('adj_matrix_max_grad', max_grad)

    def check_trainability(self):
        soft_adj = self.get_adj_matrix(hard=False)
        max_grad = (soft_adj * (1 - soft_adj)).max().item()
        if max_grad < 1e-3 and self.num_graph_samples > 1:
            print('Freezing graph due to minimal gradients')
            self.num_graph_samples = 1
            self.train_graph = False

    def get_adj_matrix(self, hard=False, for_training=False):
        if hard or (for_training and not self.train_graph):
            if self.graph_learning_method == 'ENCO':
                adj_matrix = torch.logical_and(self.enco_theta > 0.0, self.enco_gamma > 0.0).float()
            elif self.graph_learning_method == 'NOTEARS':
                adj_matrix = (self.notears_params > 0.0).float()
        else:
            if self.graph_learning_method == 'ENCO':
                adj_matrix = torch.sigmoid(self.enco_theta) * torch.sigmoid(self.enco_gamma)
            elif self.graph_learning_method == 'NOTEARS':
                adj_matrix = torch.sigmoid(self.notears_params)
        return adj_matrix.detach()

    def get_target_assignment(self, hard=False):
        """
        Psi Assignment function
        :param hard:
        :return:
        """
        # Returns psi, either 'hard' (one-hot, e.g. for triplet eval) or 'soft' (probabilities, e.g. for debug)
        if not hard:
            return torch.softmax(self.target_params, dim=-1)
        else:
            return F.one_hot(torch.argmax(self.target_params, dim=-1), num_classes=self.target_params.shape[-1])


# METHOD: CLASSIFIER
class TargetClassifier(nn.Module):
    """
    The target classifier for guiding towards better disentanglement of the causal variables in latent space.
    """

    def __init__(self, num_latents,
                 c_hid,
                 num_blocks,
                 momentum_model=0.0,
                 var_names=None,
                 num_layers=1,
                 act_fn=lambda: nn.SiLU(),
                 gumbel_temperature=1.0,
                 use_normalization=True,
                 use_conditional_targets=False):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latents of the encoder.
        c_hid : int
                Hidden dimensions to use in the target classifier network
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        momentum_model : float
                         During training, we need to run the target classifier once without
                         gradients, which we implement by using a separate, frozen network.
                         Following common practice in GANs, we also consider using momentum for
                         the frozen parameters here. In general, we did not experience it to
                         have a considerable effect on the performance.
        var_names : list
                    List of names of the causal variables. Used for logging.
        num_layers : int
                     Number of layers to use in the network.
        act_fn : function returning nn.Module
                 Activation function to use in the network.
        gumbel_temperature : float
                             Temperature to use for the Gumbel Softmax sampling.
        use_normalization : bool
                            Whether to use LayerNorm in the target classifier or not.
        use_conditional_targets : bool
                                  If True, we record conditional targets p(I^t+1_i|I^t+1_j).
                                  Needed when intervention targets are confounded.
        """
        super().__init__()
        self.momentum_model = momentum_model
        self.gumbel_temperature = gumbel_temperature
        self.num_blocks = num_blocks
        self.use_conditional_targets = use_conditional_targets
        self.dist_steps = 0.0

        # Network creation
        norm = lambda c: (nn.LayerNorm(c) if use_normalization else nn.Identity())
        layers = [nn.Linear(3 * num_latents, 2 * c_hid), norm(2 * c_hid), act_fn()]
        inp_dim = 2 * c_hid
        for _ in range(num_layers - 1):
            layers += [nn.Linear(inp_dim, c_hid), norm(c_hid), act_fn()]
            inp_dim = c_hid
        layers += [nn.Linear(inp_dim, num_blocks)]
        self.classifiers = nn.Sequential(*layers)
        self.classifiers[-1].weight.data.fill_(0.0)
        self.exp_classifiers = deepcopy(self.classifiers)
        for p in self.exp_classifiers.parameters():
            p.requires_grad_(False)

        # Buffers for recording p(I^t+1_i) / p(I^t+1_i|I^t+1_j) in the training data
        self.register_buffer('avg_dist', torch.zeros(num_blocks, 2).fill_(0.5))
        if use_conditional_targets:
            self.register_buffer('avg_cond_dist', torch.zeros(num_blocks, num_blocks, 2, 2).fill_(0.5))
            self.register_buffer('dist_cond_steps', torch.zeros(num_blocks, 2).fill_(1))

        # Variable names for logging
        self.var_names = var_names
        if self.var_names is not None:
            if len(self.var_names) <= num_blocks:
                self.var_names = self.var_names + ['No variable']
            if len(self.var_names) <= num_blocks + 1:
                self.var_names = self.var_names + ['All variables']

    @torch.no_grad()
    def _step_exp_avg(self):
        # Update frozen model with momentum on new params
        for p1, p2 in zip(self.classifiers.parameters(), self.exp_classifiers.parameters()):
            p2.data.mul_(self.momentum_model).add_(p1.data * (1 - self.momentum_model))

    @torch.no_grad()
    def _update_dist(self, target):
        # Add target tensor to average of target distributions
        if self.dist_steps < 1e6:  # At this time we should have a pretty good average
            target = target.float()
            avg_target = target.mean(dim=[0, 1])
            new_dist = torch.stack([1 - avg_target, avg_target], dim=-1)
            self.avg_dist.mul_(self.dist_steps / (self.dist_steps + 1)).add_(new_dist * (1. / (self.dist_steps + 1)))

            if hasattr(self, 'avg_cond_dist'):
                target_sums = target.sum(dim=[0, 1])
                target_prod = (target[..., None, :] * target[..., :, None]).sum(dim=[0, 1])

                one_cond_one = target_prod / target_sums[None, :].clamp(min=1e-5)
                zero_cond_one = 1 - one_cond_one
                inv_sum = (target.shape[0] * target.shape[1] - target_sums)
                one_cond_zero = (target_sums[:, None] - target_prod) / inv_sum[None, :].clamp(min=1e-5)
                zero_cond_zero = 1 - one_cond_zero
                new_cond_steps = torch.stack([target.shape[0] * target.shape[1] - target_sums, target_sums], dim=-1)
                update_factor = (self.dist_cond_steps / (self.dist_cond_steps + new_cond_steps))[None, :, :, None]
                cond_dist = torch.stack([zero_cond_zero, one_cond_zero, zero_cond_one, one_cond_one], dim=-1).unflatten(
                    -1, (2, 2))
                self.avg_cond_dist.mul_(update_factor).add_(cond_dist * (1 - update_factor))
                self.dist_cond_steps.add_(new_cond_steps)
            self.dist_steps += 1

    def _tag_to_str(self, t):
        # Helper function for finding correct logging names for causal variable indices
        if self.var_names is None or len(self.var_names) <= t:
            return str(t)
        else:
            return f'[{self.var_names[t]}]'

    def forward(self, z_sample, target, transition_prior, logger=None):
        """
        Calculates the loss for the target classifier (predict all intervention targets from all sets of latents)
        and the latents + psi (predict only its respective intervention target from a set of latents).

        Parameters
        ----------
        z_sample : torch.FloatTensor, shape [batch_size, time_steps, num_latents]
                   The sets of latents for which the loss should be calculated. If time steps is 2, we
                   use z^t=z_sample[:,0], z^t+1=z_sample[:,1]. If time steps is larger than 2, we apply
                   it for every possible pair over time.
        target : torch.FloatTensor, shape [batch_size, time_steps-1, num_blocks]
                 The intervention targets I^t+1
        transition_prior : TransitionPrior
                           The transition prior of the model. Needed for obtaining the parameters of psi.
        """
        if self.training:
            self._step_exp_avg()
            self._update_dist(target)
        # Joint preparations
        batch_size = z_sample.shape[0]
        time_steps = z_sample.shape[1] - 1
        num_classes = target.shape[-1]
        # Sample latent-to-causal variable assignments
        target_assignment = F.gumbel_softmax(
            transition_prior.target_params[None].expand(batch_size, time_steps, -1, -1),
            tau=self.gumbel_temperature, hard=True)
        if target_assignment.shape[-1] == num_classes:
            target_assignment = torch.cat(
                [target_assignment, target_assignment.new_zeros(target_assignment.shape[:-1] + (1,))], dim=-1)
        target_assignment = torch.cat(
            [target_assignment, target_assignment.new_ones(target_assignment.shape[:-1] + (1,))], dim=-1)
        num_targets = target_assignment.shape[-1]
        target_assignment = target_assignment.permute(0, 1, 3, 2)
        z_sample = z_sample[..., None, :].expand(-1, -1, num_targets, -1)
        exp_target = target[..., None, :].expand(-1, -1, num_targets, -1).flatten(0, 2).float()

        # We consider 2 + [number of causal variables] sets of latents:
        # (1) one per causal variable, (2) the noise/'no-causal-var' slot psi(0), (3) all latents
        # The latter is helpful to encourage the VAE in the first iterations to just put everything in the latent space
        # We create a mask below for which intervention targets are supposed to be predicted from which set of latents
        loss_mask = torch.cat(
            [torch.eye(num_classes, dtype=torch.bool, device=target.device),  # Latent to causal variables
             torch.zeros(1, num_classes, dtype=torch.bool, device=target.device),  # 'No-causal-var' slot
             torch.ones(1, num_classes, dtype=torch.bool, device=target.device)  # 'All-var' slot
             ], dim=0)
        loss_mask = loss_mask[None].expand(batch_size * time_steps, -1, -1).flatten(0, 1)

        # Model step => Cross entropy loss on targets for all sets of latents
        z_sample_model = z_sample.detach()
        target_assignment_det = target_assignment.detach()
        model_inps = torch.cat(
            [z_sample_model[:, :-1, :], z_sample_model[:, 1:, :] * target_assignment_det, target_assignment_det],
            dim=-1)
        model_inps = model_inps.flatten(0, 2)
        model_pred = self.classifiers(model_inps)
        loss_model = F.binary_cross_entropy_with_logits(model_pred, exp_target, reduction='none')
        loss_model = num_targets * time_steps * loss_model.mean()

        # Log target classification accuracies
        if logger is not None:
            with torch.no_grad():
                acc = ((model_pred > 0.0).float() == exp_target).float().unflatten(0, (
                batch_size * time_steps, num_targets)).mean(dim=0)
                for b in range(num_targets):
                    for c in range(num_classes):
                        logger.log(f'target_classifier_block{self._tag_to_str(b)}_class{self._tag_to_str(c)}',
                                   acc[b, c], on_step=False, on_epoch=True)

        # Latent step => Cross entropy loss on true targets for respective sets of latents, and cross entropy loss on marginal (conditional) accuracy otherwise.
        z_inps = torch.cat([z_sample[:, :-1, :], z_sample[:, 1:, :] * target_assignment, target_assignment], dim=-1)
        z_inps = z_inps.flatten(0, 2)
        z_pred = self.exp_classifiers(z_inps)
        z_pred_unflatten = z_pred.unflatten(0, (batch_size * time_steps, num_targets))
        if hasattr(self, 'avg_cond_dist'):
            avg_dist_labels = self.avg_cond_dist[None, None, :, :, 1, 1] * target[:, :, None, :] + self.avg_cond_dist[
                                                                                                   None, None, :, :, 0,
                                                                                                   1] * (
                                          1 - target[:, :, None, :])
            avg_dist_labels = avg_dist_labels.permute(0, 1, 3, 2).flatten(0, 1)
            avg_dist_labels = torch.cat(
                [avg_dist_labels, self.avg_dist[None, None, :, 1].expand(avg_dist_labels.shape[0], 2, -1)], dim=1)
            avg_dist_labels = avg_dist_labels.flatten(0, 1)
        else:
            avg_dist_labels = self.avg_dist[None, :, 1]
        z_targets = torch.where(loss_mask, exp_target, avg_dist_labels)
        loss_z = F.binary_cross_entropy_with_logits(z_pred, z_targets, reduction='none')
        loss_mask = loss_mask.float()
        pos_weight = num_classes  # Beneficial to weight the cross entropy loss for true target higher, especially for many causal variables
        loss_z = loss_z * (pos_weight * loss_mask + (1 - loss_mask))
        loss_z = loss_z.mean()
        loss_z = num_targets * time_steps * loss_z

        return loss_model, loss_z


class InstantaneousTargetClassifier(TargetClassifier):
    """
    The target classifier for guiding towards better disentanglement of the causal variables in latent space.
    This is adapted for potentially instantaneous effects, since parents are needed to predict the interventions as well
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        target_counter = torch.zeros((2,) * self.num_blocks, dtype=torch.float32)
        torch_range = torch.arange(2 ** self.num_blocks, dtype=torch.float32)
        target_counter_mask = torch.stack(
            [torch.div(torch_range, 2 ** i, rounding_mode='floor') % 2 for i in range(self.num_blocks - 1, -1, -1)],
            dim=-1)
        self.register_buffer('target_counter', target_counter)
        self.register_buffer('target_counter_mask', target_counter_mask)
        self.register_buffer('graph_structures', target_counter_mask.clone())
        self.register_buffer('target_counter_prob',
                             torch.zeros(2 ** self.num_blocks, 2 ** self.num_blocks, self.num_blocks))
        self.register_buffer('two_exp_range', torch.Tensor([2 ** i for i in range(self.num_blocks - 1, -1, -1)]))

    @torch.no_grad()
    def _update_dist(self, target):
        super()._update_dist(target)
        if self.dist_steps < 1e6 and self.use_conditional_targets:
            target = target.flatten(0, -2)
            unique_targets, target_counts = torch.unique(target, dim=0, return_counts=True)
            self.target_counter[unique_targets.long().unbind(dim=-1)] += target_counts

            target_equal = (self.target_counter_mask[None, :] == self.target_counter_mask[:, None])
            mask = torch.logical_or(target_equal[None, :, :], self.graph_structures[:, None, None] == 0)
            mask = mask.all(dim=-1)
            mask = mask.reshape((mask.shape[0],) + (2,) * self.num_blocks + mask.shape[2:])
            masked_counter = mask * self.target_counter[None, ..., None]
            all_probs = []
            for i in range(self.num_blocks):
                counter_sum = masked_counter.sum(dim=[j + 1 for j in range(self.num_blocks) if i != j])
                counter_prob = counter_sum[:, 1] / counter_sum.sum(dim=1).clamp_(min=1e-5)
                all_probs.append(counter_prob)
            self.target_counter_prob = torch.stack(all_probs, dim=-1)

    def forward(self, z_sample, target, transition_prior, logger=None, add_anc_prob=0.0):
        """
        Calculates the loss for the target classifier (predict all intervention targets from all sets of latents)
        and the latents + psi (predict only its respective intervention target from a set of latents).

        Parameters
        ----------
        z_sample : torch.FloatTensor, shape [batch_size, time_steps, num_latents]
                   The sets of latents for which the loss should be calculated. If time steps is 2, we
                   use z^t=z_sample[:,0], z^t+1=z_sample[:,1]. If time steps is larger than 2, we apply
                   it for every possible pair over time.
        target : torch.FloatTensor, shape [batch_size, num_blocks]
                 The intervention targets I^t+1
        transition_prior : TransitionPrior
                           The transition prior of the model. Needed for obtaining the parameters of psi.
        """
        if self.training:
            self._step_exp_avg()
            self._update_dist(target)

        # Joint preparations
        batch_size = z_sample.shape[0]
        time_steps = z_sample.shape[1] - 1
        num_classes = target.shape[-1]
        num_latents = z_sample.shape[-1]
        # Sample latent-to-causal variable assignments
        target_assignment = F.gumbel_softmax(
            transition_prior.target_params[None].expand(batch_size, time_steps, -1, -1),
            tau=self.gumbel_temperature, hard=True)
        if target_assignment.shape[-1] == num_classes:  # No variable slot
            target_assignment = torch.cat(
                [target_assignment, target_assignment.new_zeros(target_assignment.shape[:-1] + (1,))], dim=-1)
        target_assignment = torch.cat(
            [target_assignment, target_assignment.new_ones(target_assignment.shape[:-1] + (1,))],
            dim=-1)  # All latents slot
        num_targets = target_assignment.shape[-1]

        target_assignment = target_assignment.transpose(-2, -1)  # shape [batch, time_steps, num_targets, latent_vars]
        z_sample = z_sample[..., None, :].expand(-1, -1, num_targets, -1)
        exp_target = target[..., None, :].expand(-1, -1, num_targets, -1).flatten(0, 2).float()

        # Sample adjacency matrices
        graph_probs = transition_prior.get_adj_matrix(hard=False, for_training=True)
        graph_samples = torch.bernoulli(graph_probs[None, None].expand(batch_size, time_steps, -1, -1))
        if add_anc_prob > 0.0:  # Once the graph is fixed, add ancestors to the mix here
            graph_samples_anc = add_ancestors_to_adj_matrix(graph_samples, remove_diag=True, exclude_cycles=True)
            graph_samples = torch.where(
                torch.rand(*graph_samples.shape[:2], 1, 1, device=graph_samples.device) < add_anc_prob,
                graph_samples_anc, graph_samples)
        # Add self-connections since we want to identify interventions from itself as well
        graph_samples_eye = graph_samples + torch.eye(graph_samples.shape[-1], device=graph_samples.device)[None, None]
        latent_to_causal = (
                    target_assignment[:, :, :graph_probs.shape[0], :, None] * graph_samples_eye[:, :, :, None, :]).sum(
            dim=-3)
        latent_mask = latent_to_causal.transpose(-2, -1)  # shape: [batch_size, time_steps, causal_vars, latent_vars]
        latent_mask = torch.cat([latent_mask] +
                                ([latent_mask.new_zeros(batch_size, time_steps, 1, num_latents)] if (
                                            latent_mask.shape[2] == num_classes) else []) +
                                [latent_mask.new_ones(batch_size, time_steps, 1, num_latents)],
                                dim=-2)  # shape [batch, time_steps, num_targets, latent_vars]
        # Model step => Cross entropy loss on targets for all sets of latents
        z_sample_model = z_sample.detach()
        latent_mask_det = latent_mask.detach()
        model_inps = torch.cat([z_sample_model[:, :-1], z_sample_model[:, 1:] * latent_mask_det, latent_mask_det],
                               dim=-1)
        model_inps = model_inps.flatten(0, 2)
        model_pred = self.classifiers(model_inps)
        loss_model = F.binary_cross_entropy_with_logits(model_pred, exp_target, reduction='none')
        loss_model = num_targets * loss_model.mean()

        # Log target classification accuracies
        if logger is not None:
            with torch.no_grad():
                acc = ((model_pred > 0.0).float() == exp_target).float().unflatten(0, (
                batch_size * time_steps, num_targets)).mean(dim=0)
                for b in range(num_targets):
                    for c in range(num_classes):
                        logger.log(f'target_classifier_block{self._tag_to_str(b)}_class{self._tag_to_str(c)}',
                                   acc[b, c], on_step=False, on_epoch=True)

        # We consider 2 + [number of causal variables] sets of latents:
        # (1) one per causal variable plus its children, (2) the noise/'no-causal-var' slot psi(0), (3) all latents
        # The latter is helpful to encourage the VAE in the first iterations to just put everything in the latent space
        # We create a mask below for which intervention targets are supposed to be predicted from which set of latents
        loss_mask = torch.eye(num_classes, dtype=torch.float32, device=target.device)
        loss_mask = loss_mask[None, None].expand(batch_size, time_steps, -1, -1)
        loss_mask = loss_mask - graph_samples.transpose(-2, -1)
        loss_mask = torch.cat([loss_mask,
                               torch.zeros(batch_size, time_steps, 1, num_classes, dtype=torch.float32,
                                           device=target.device),  # 'No-causal-var' slot
                               torch.ones(batch_size, time_steps, 1, num_classes, dtype=torch.float32,
                                          device=target.device)  # 'All-var' slot
                               ], dim=2)
        loss_mask = loss_mask.flatten(0, 2)

        # Latent step => Cross entropy loss on true targets for respective sets of latents, and cross entropy loss on marginal (conditional) accuracy otherwise.
        z_inps = torch.cat([z_sample[:, :-1], z_sample[:, 1:] * latent_mask, latent_mask], dim=-1)
        z_inps = z_inps.flatten(0, 2)
        z_pred = self.exp_classifiers(z_inps)
        z_pred_unflatten = z_pred.unflatten(0, (batch_size * time_steps, num_targets))
        if hasattr(self, 'avg_cond_dist'):
            with torch.no_grad():
                # Add labels for all variables
                pred_logits = z_pred_unflatten.detach()[:, :-2]
                pred_probs = F.logsigmoid(pred_logits)
                pred_neg_probs = F.logsigmoid(-pred_logits)
                target_flat = target.flatten(0, 1)
                graphs_flat = graph_samples_eye.flatten(0, 1)
                graph_idxs = (graphs_flat * self.two_exp_range[None, :, None]).sum(dim=1).long()
                avg_dist_labels = self.target_counter_prob[graph_idxs]

                target_weights = torch.where(self.target_counter_mask[None, None] == 0, pred_neg_probs[:, :, None],
                                             pred_probs[:, :, None]).sum(dim=-1)
                target_weights = torch.softmax(target_weights, dim=-1)
                avg_dist_labels = (avg_dist_labels * target_weights[..., None]).sum(dim=-2)

                # Add labels for no and all variables
                avg_dist_labels = torch.cat([avg_dist_labels,
                                             self.avg_dist[None, None, :, 1].expand(avg_dist_labels.shape[0], 2, -1)],
                                            dim=1)
                avg_dist_labels = avg_dist_labels.flatten(0, 1)
        else:
            avg_dist_labels = self.avg_dist[None, :, 1]
        z_targets = torch.where(loss_mask == 1, exp_target, avg_dist_labels)
        loss_z = F.binary_cross_entropy_with_logits(z_pred, z_targets, reduction='none')
        pos_weight = num_classes  # Beneficial to weight the cross entropy loss for true target higher, especially for many causal variables
        loss_z = loss_z * (pos_weight * (loss_mask == 1).float() + 1 * (loss_mask == 0).float())
        loss_z = loss_z.mean()
        loss_z = num_targets * loss_z

        return loss_model, loss_z


class MIEstimator(nn.Module):
    """
    The MI estimator for guiding towards better disentanglement of the causal variables in latent space.
    """

    def __init__(self, num_latents, c_hid, num_blocks, momentum_model=0.0, var_names=None, num_layers=1,
                 act_fn=lambda: nn.SiLU(), gumbel_temperature=1.0, use_normalization=True):
        """
        Parameters
        ----------
        num_latents : int
                      Number of latents of the encoder.
        c_hid : int
                Hidden dimensions to use in the target classifier network
        num_blocks : int
                     Number of blocks to group the latent dimensions into. In other words,
                     it is the number of causal variables plus 1 (psi(0) - unintervened information).
        momentum_model : float
                         During training, we need to run the target classifier once without
                         gradients, which we implement by using a separate, frozen network.
                         Following common practice in GANs, we also consider using momentum for
                         the frozen parameters here. In general, we did not experience it to
                         have a considerable effect on the performance.
        var_names : list
                    List of names of the causal variables. Used for logging.
        num_layers : int
                     Number of layers to use in the network.
        act_fn : function returning nn.Module
                 Activation function to use in the network.
        gumbel_temperature : float
                             Temperature to use for the Gumbel Softmax sampling.
        use_normalization : bool
                            Whether to use LayerNorm in the target classifier or not.
        """
        super().__init__()
        self.momentum_model = momentum_model
        self.gumbel_temperature = gumbel_temperature
        self.num_comparisons = 1
        self.num_blocks = num_blocks
        self.num_latents = num_latents
        self.c_hid = c_hid * 2

        # Network creation
        self.mi_estimator = nn.Sequential(
            nn.Linear(num_latents * 3 + num_blocks, self.c_hid),
            nn.LayerNorm(self.c_hid),
            act_fn(),
            nn.Linear(self.c_hid, self.c_hid),
            nn.LayerNorm(self.c_hid),
            act_fn(),
            nn.Linear(self.c_hid, 1, bias=False)
        )
        self.mi_estimator[-1].weight.data.fill_(0.0)
        self.exp_mi_estimator = deepcopy(self.mi_estimator)
        for p in self.exp_mi_estimator.parameters():
            p.requires_grad_(False)

        # Variable names for logging
        self.var_names = var_names
        # if self.var_names is not None:
        #     if len(self.var_names) <= num_blocks:
        #         self.var_names = self.var_names + ['No variable']
        #     if len(self.var_names) <= num_blocks + 1:
        #         self.var_names = self.var_names + ['All variables']

    @torch.no_grad()
    def _step_exp_avg(self):
        # Update frozen model with momentum on new params
        for p1, p2 in zip(self.mi_estimator.parameters(), self.exp_mi_estimator.parameters()):
            p2.data.mul_(self.momentum_model).add_(p1.data * (1 - self.momentum_model))

    def _tag_to_str(self, t):
        # Helper function for finding correct logging names for causal variable indices
        if self.var_names is None or len(self.var_names) <= t:
            return str(t)
        else:
            return f'[{self.var_names[t]}]'

    def forward(self, z_sample, target, transition_prior, logger=None, instant_prob=None):
        """
        Calculates the loss for the mutual information estimator.

        Parameters
        ----------
        z_sample : torch.FloatTensor, shape [batch_size, time_steps, num_latents]
                   The sets of latents for which the loss should be calculated. If time steps is 2, we
                   use z^t=z_sample[:,0], z^t+1=z_sample[:,1]. If time steps is larger than 2, we apply
                   it for every possible pair over time.
        target : torch.FloatTensor, shape [batch_size, time_steps-1, num_blocks]
                 The intervention targets I^t+1
        transition_prior : TransitionPrior
                           The transition prior of the model. Needed for obtaining the parameters of psi.
        """
        if self.training:
            self._step_exp_avg()

        target = target.flatten(0, 1)
        z_sample_0 = z_sample[:, :-1].flatten(0, 1)
        z_sample_1 = z_sample[:, 1:].flatten(0, 1)

        # Find samples for which certain variables have been intervened upon
        with torch.no_grad():
            idxs = [torch.where(target[:, i] == 1)[0] for i in range(self.num_blocks)]
            idxs_stack = torch.cat(idxs, dim=0)
            batch_size = idxs_stack.shape[0]
            i_batch_sizes = [dx.shape[0] for dx in idxs]
            i_batch_sizes_cumsum = np.cumsum(i_batch_sizes)
            intv_target = torch.zeros_like(idxs_stack)
            for i in range(1, self.num_blocks):
                intv_target[i_batch_sizes_cumsum[i - 1]:i_batch_sizes_cumsum[i]] = i
            intv_target_onehot = F.one_hot(intv_target, num_classes=self.num_blocks)

        # Sample graphs and latent->causal assignments
        target_assignment = F.gumbel_softmax(transition_prior.target_params[None].expand(batch_size, -1, -1),
                                             tau=self.gumbel_temperature, hard=True)
        graph_probs = transition_prior.get_adj_matrix(hard=False, for_training=True)
        graph_samples = torch.bernoulli(graph_probs[None].expand(batch_size, -1, -1))
        if instant_prob is not None:  # Mask out instant parents with probability
            graph_mask = (torch.rand(batch_size, graph_probs.shape[1],
                                     device=graph_samples.device) < instant_prob).float()
            graph_samples = graph_samples * graph_mask[:, None, :]
        graph_samples = graph_samples - torch.eye(graph_samples.shape[1], device=graph_samples.device)[
            None]  # Self-connection (-1), parents (1), others (0)
        latent_mask = (target_assignment[:, :, :, None] * graph_samples[:, None, :, :]).sum(dim=-2).transpose(1, 2)

        # Prepare positive pairs
        z_sample_sel_0 = z_sample_0[idxs_stack]
        z_sample_sel_1 = z_sample_1[idxs_stack]
        latent_mask_sel = latent_mask[torch.arange(intv_target.shape[0], dtype=torch.long), intv_target]
        latent_mask_sel_abs = latent_mask_sel.abs()
        inp_sel = torch.cat([z_sample_sel_0,
                             z_sample_sel_1 * latent_mask_sel_abs,
                             latent_mask_sel,
                             intv_target_onehot], dim=-1)

        # Prepare negative pairs
        inp_alts = []
        for _ in range(self.num_comparisons):
            alter_idxs = torch.cat([torch.randperm(i_batch_sizes[i], device=idxs_stack.device) + (
                0 if i == 0 else i_batch_sizes_cumsum[i - 1]) for i in range(self.num_blocks)], dim=0)
            z_sample_alt_1 = z_sample_sel_1[alter_idxs]
            inp_alt = torch.cat([z_sample_sel_0,
                                 torch.where(latent_mask_sel == -1, z_sample_alt_1,
                                             z_sample_sel_1) * latent_mask_sel_abs,
                                 latent_mask_sel,
                                 intv_target_onehot], dim=-1)
            inp_alts.append(inp_alt)
        joint_inp = torch.stack([inp_sel] + inp_alts, dim=1)

        # Binary classifier as mutual information estimator
        model_out = self.mi_estimator(joint_inp.detach()).squeeze(dim=-1)
        z_out = self.exp_mi_estimator(joint_inp).squeeze(dim=-1)
        loss_model = -model_out[:, 0] + torch.logsumexp(model_out, dim=1)  # Same to -F.log_softmax(z_out, dim=1)[:,0]
        loss_z = -F.log_softmax(z_out, dim=1)[:, -1]  # Alternative is mean over last dimension

        # Finalize loss
        loss_model = loss_model.mean()
        loss_z = loss_z.mean()
        reg_loss = 0.001 * (model_out ** 2).mean()  # To keep outputs in a reasonable range
        loss_model = loss_model + reg_loss

        return loss_model, loss_z


# METHOD: FLOW
class AutoregNormalizingFlow(nn.Module):
    """
    Base class for the autoregressive normalizing flow
    We use a combination of affine autoregressive coupling layers,
    activation normalization, and invertible 1x1 convolutions /
    orthogonal invertible transformations.
    """

    def __init__(self, num_vars, num_flows, act_fn, hidden_per_var=16, zero_init=False, use_scaling=True,
                 use_1x1_convs=True, init_std_factor=0.2):
        super().__init__()
        self.flows = nn.ModuleList([])
        transform_layer = lambda num_vars: AffineFlow(num_vars, use_scaling=use_scaling)
        for i in range(num_flows):
            self.flows.append(ActNormFlow(num_vars))
            if i > 0:
                if use_1x1_convs:
                    self.flows.append(OrthogonalFlow(num_vars))
                else:
                    self.flows.append(ReverseSeqFlow())
            self.flows.append(AutoregressiveFlow(num_vars,
                                                 hidden_per_var=hidden_per_var,
                                                 act_fn=act_fn,
                                                 init_std_factor=(0 if zero_init else init_std_factor),
                                                 transform_layer=transform_layer))

    def forward(self, x):
        ldj = x.new_zeros(x.shape[0], )
        for flow in self.flows:
            x, ldj = flow(x, ldj)
        return x, ldj

    def reverse(self, x):
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x


class AffineFlow(nn.Module):
    """ Affine transformation """

    def __init__(self, num_vars, use_scaling=True, hard_limit=-1):
        super().__init__()
        self.num_vars = num_vars
        self.hard_limit = hard_limit
        self.use_scaling = use_scaling
        if self.use_scaling and self.hard_limit <= 0:
            self.scaling = nn.Parameter(torch.zeros(num_vars, ))

    def get_num_outputs(self):
        return 2

    def _get_affine_params(self, out):
        if isinstance(out, (list, tuple)):
            t, s = out
        else:
            t, s = out.unflatten(-1, (-1, 2)).unbind(dim=-1)
        if self.use_scaling:
            if self.hard_limit > 0:
                s = s - torch.max(s - self.hard_limit, torch.zeros_like(s)).detach()
                s = s + torch.max(-self.hard_limit - s, torch.zeros_like(s)).detach()
            else:
                sc = torch.tanh(self.scaling.exp()[None] / 3.0) * 3.0
                s = torch.tanh(s / sc.clamp(min=1.0)) * sc
        else:
            s = s * 0.0
        return t, s

    def forward(self, x, out, ldj):
        t, s = self._get_affine_params(out)
        x = (x + t) * s.exp()
        ldj = ldj - s.sum(dim=1)
        return x, ldj

    def reverse(self, x, out):
        t, s = self._get_affine_params(out)
        x = x * (-s).exp() - t
        return x


class AutoregressiveFlow(nn.Module):
    """ Autoregressive flow with arbitrary invertible transformation """

    def __init__(self, num_vars, hidden_per_var=16,
                 act_fn=lambda: nn.SiLU(),
                 init_std_factor=0.2,
                 transform_layer=AffineFlow):
        super().__init__()
        self.transformation = transform_layer(num_vars)
        self.net = nn.Sequential(
            AutoregLinear(num_vars, 1, hidden_per_var, diagonal=False),
            act_fn(),
            AutoregLinear(num_vars, hidden_per_var, hidden_per_var, diagonal=True),
            act_fn(),
            AutoregLinear(num_vars, hidden_per_var, self.transformation.get_num_outputs(), diagonal=True,
                          no_act_fn_init=True,
                          init_std_factor=init_std_factor,
                          init_bias_factor=0.0,
                          init_first_block_zeros=True)
        )

    def forward(self, x, ldj):
        out = self.net(x)
        x, ldj = self.transformation(x, out, ldj)
        return x, ldj

    def reverse(self, x):
        inp = x * 0.0
        for i in range(x.shape[1]):
            out = self.net(inp)
            x_new = self.transformation.reverse(x, out)
            inp[:, i] = x_new[:, i]
        return x_new


class ActNormFlow(nn.Module):
    """ Activation normalization """

    def __init__(self, num_vars):
        super().__init__()
        self.num_vars = num_vars
        self.data_init = False

        self.bias = nn.Parameter(torch.zeros(self.num_vars, ))
        self.scales = nn.Parameter(torch.zeros(self.num_vars, ))
        self.affine_flow = AffineFlow(self.num_vars, hard_limit=3.0)

    def get_num_outputs(self):
        return 2

    def forward(self, x, ldj):
        if self.training and not self.data_init:
            self.data_init_forward(x)
        x, ldj = self.affine_flow(x, [self.bias[None], self.scales[None]], ldj)
        return x, ldj

    def reverse(self, x):
        x = self.affine_flow.reverse(x, [self.bias[None], self.scales[None]])
        return x

    @torch.no_grad()
    def data_init_forward(self, input_data):
        if (self.bias != 0.0).any():
            self.data_init = True
            return

        batch_size = input_data.shape[0]

        self.bias.data = -input_data.mean(dim=0)
        self.scales.data = -input_data.std(dim=0).log()
        self.data_init = True

        out, _ = self.forward(input_data, input_data.new_zeros(batch_size, ))
        print(f"[INFO - ActNorm] New mean: {out.mean().item():4.2f}")
        print(f"[INFO - ActNorm] New variance {out.std(dim=0).mean().item():4.2f}")


class ReverseSeqFlow(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, ldj):
        return torch.flip(x, dims=(-1,)), ldj

    def reverse(self, x):
        return self.forward(x, None)[0]


class OrthogonalFlow(nn.Module):
    """ Invertible 1x1 convolution / orthogonal flow """

    def __init__(self, num_vars, LU_decomposed=True):
        super().__init__()
        self.num_vars = num_vars
        self.LU_decomposed = LU_decomposed

        # Initialize with a random orthogonal matrix
        w_init = np.random.randn(self.num_vars, self.num_vars)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)

        if not self.LU_decomposed:
            self.weight = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)
        else:
            # LU decomposition can slightly speed up the inverse
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_init.shape, dtype=np.float32), -1)
            eye = np.eye(*w_init.shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)), requires_grad=True)
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)), requires_grad=True)
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)), requires_grad=True)
            self.register_buffer('l_mask', torch.Tensor(l_mask))
            self.register_buffer('eye', torch.Tensor(eye))

        self.eval_dict = defaultdict(lambda: self._get_default_inner_dict())

    def _get_default_inner_dict(self):
        return {"weight": None, "inv_weight": None, "sldj": None}

    def _get_weight(self, device_name, inverse=False):
        if self.training or self._is_eval_dict_empty(device_name):
            if not self.LU_decomposed:
                weight = self.weight
                sldj = torch.slogdet(weight)[1]
            else:
                l, log_s, u = self.l, self.log_s, self.u
                l = l * self.l_mask + self.eye
                u = u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(log_s))
                weight = torch.matmul(self.p, torch.matmul(l, u))
                sldj = log_s.sum()

        if not self.training:
            if self._is_eval_dict_empty(device_name):
                self.eval_dict[device_name]["weight"] = weight.detach()
                self.eval_dict[device_name]["sldj"] = sldj.detach()
                self.eval_dict[device_name]["inv_weight"] = torch.inverse(weight.double()).float().detach()
            else:
                weight, sldj = self.eval_dict[device_name]["weight"], self.eval_dict[device_name]["sldj"]
        elif not self._is_eval_dict_empty(device_name):
            self._empty_eval_dict(device_name)

        if inverse:
            if self.training:
                weight = torch.inverse(weight.double()).float()
            else:
                weight = self.eval_dict[device_name]["inv_weight"]

        return weight, sldj

    def _is_eval_dict_empty(self, device_name=None):
        if device_name is not None:
            return not (device_name in self.eval_dict)
        else:
            return len(self.eval_dict) == 0

    def _empty_eval_dict(self, device_name=None):
        if device_name is not None:
            self.eval_dict.pop(device_name)
        else:
            self.eval_dict = defaultdict(lambda: self._get_default_inner_dict())

    def forward(self, x, ldj):
        weight, sldj = self._get_weight(device_name=str(x.device), inverse=False)
        ldj = ldj - sldj
        z = torch.matmul(x, weight)
        return z, ldj

    def reverse(self, x):
        weight, sldj = self._get_weight(device_name=str(x.device), inverse=True)
        z = torch.matmul(x, weight)
        return z


# FUNCTIONS
def create_pos_grid(shape, device, stack_dim=-1):
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, shape[0], device=device),
                                  torch.linspace(-1, 1, shape[1], device=device),
                                  indexing='ij')
    pos = torch.stack([pos_x, pos_y], dim=stack_dim)
    return pos


def get_act_fn(act_fn_name):
    """ Map activation function string to activation function """
    act_fn_name = act_fn_name.lower()
    if act_fn_name == 'elu':
        act_fn_func = nn.ELU
    elif act_fn_name == 'silu':
        act_fn_func = nn.SiLU
    elif act_fn_name == 'leakyrelu':
        act_fn_func = lambda: nn.LeakyReLU(negative_slope=0.05, inplace=True)
    elif act_fn_name == 'relu':
        act_fn_func = nn.ReLU
    else:
        assert False, f'Unknown activation function \"{act_fn_name}\"'
    return act_fn_func


def load_datasets(args, env_name):
    pl.seed_everything(args.seed)
    print('Loading data...')

    # Extend for different models
    if env_name == 'Reacher':
        DataClass = ReacherDataset
        dataset_args = {}
        test_args = lambda train_set: {'causal_vars': train_set.target_names}
    else:
        pass
    folder = str(pathlib.Path(__file__).parent.resolve()) + '/data/'

    train_data = DataClass(data_folder=folder, split='train', single_image=False, seq_len=2, **dataset_args)
    val_data = DataClass(data_folder=folder, split='val_indep', single_image=True, **dataset_args, **test_args(train_data))
    train_loader = data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                   pin_memory=True, drop_last=True, num_workers=args.num_workers)

    print(f'Training dataset size: {len(train_data)} / {len(train_loader)}')
    print(f'Val correlation dataset size: {len(val_data)}')

    datasets = {
        'train': train_data,
        'val': val_data
    }
    data_loaders = {
        'train': train_loader
    }
    return datasets, data_loaders, env_name.lower()


def train_model(model_class, train_loader, max_epochs=200, check_val_every_n_epoch=1, load_pretrained=True,
                gradient_clip_val=1, cluster=False, seed=42, save_last_model=False, data_dir=None,
                val_track_metric='val_loss', callback_kwargs=None, files_to_save=None, **kwargs):
    trainer_args = {}
    trainer_args['enable_progress_bar'] = False
    trainer_args['log_every_n_steps'] = 2
    root_dir = str(pathlib.Path(__file__).parent.resolve()) + f'/data/model_checkpoints/active_iCITRIS/'
    checkpoint_callback = ModelCheckpoint(dirpath=root_dir, save_last=True)

    trainer = pl.Trainer(default_root_dir=root_dir, callbacks=[checkpoint_callback], accelerator='auto', max_epochs=max_epochs,
                         check_val_every_n_epoch=1, gradient_clip_val=gradient_clip_val,
                         **trainer_args)
    trainer.logger._default_hp_metric = None

    pretrained_filename = root_dir + 'last.ckpt'
    pl.seed_everything(seed)
    if load_pretrained and os.path.isfile(pretrained_filename):
        print('model found')
        model = model_class.load_from_checkpoint(pretrained_filename)
    else:
        model = model_class(**kwargs)

    trainer.fit(model, train_loader)


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="../data/")
    parser.add_argument('--causal_encoder_checkpoint', type=str,
                        default="../data/model_checkpoints/active_iCITRIS/CausalEncoder.ckpt")
    parser.add_argument('--cluster', action="store_true")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--exclude_vars', type=str, nargs='+', default=None)
    parser.add_argument('--exclude_objects', type=int, nargs='+', default=None)
    parser.add_argument('--coarse_vars', action='store_true')
    parser.add_argument('--data_img_width', type=int, default=-1)
    parser.add_argument('--seq_len', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--imperfect_interventions', action='store_true')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=-1)
    parser.add_argument('--logger_name', type=str, default='')
    parser.add_argument('--files_to_save', type=str, nargs='+', default='')
    return parser


def update_enco_params(self, *args, **kwargs):
    if self.gamma_grads is not None:
        self.enco_gamma.grad = self.gamma_grads.detach()
    if self.theta_grads is not None:
        self.enco_theta.grad = self.theta_grads.detach()


# FUNCTIONS: MODULES
def kl_divergence(mean1, log_std1, mean2=None, log_std2=None):
    """ Returns the KL divergence between two Gaussian distributions """
    if mean2 is None:
        mean2 = torch.zeros_like(mean1)
    if log_std2 is None:
        log_std2 = torch.zeros_like(log_std1)

    var1, var2 = (2*log_std1).exp(), (2*log_std2).exp()
    KLD = (log_std2 - log_std1) + (var1 + (mean1 - mean2) ** 2) / (2 * var2) - 0.5
    return KLD


def gaussian_log_prob(mean, log_std, samples):
    """ Returns the log probability of a specified Gaussian for a tensor of samples """
    if len(samples.shape) == len(mean.shape)+1:
        mean = mean[...,None]
    if len(samples.shape) == len(log_std.shape)+1:
        log_std = log_std[...,None]
    return - log_std - 0.5 * np.log(2*np.pi) - 0.5 * ((samples - mean) / log_std.exp())**2


@torch.no_grad()
def add_ancestors_to_adj_matrix(adj_matrix, remove_diag=True, exclude_cycles=False):
    adj_matrix = adj_matrix.bool()
    orig_adj_matrix = adj_matrix
    eye_matrix = torch.eye(adj_matrix.shape[-1], device=adj_matrix.device, dtype=torch.bool).reshape(
        (1,) * (len(adj_matrix.shape) - 2) + (-1, adj_matrix.shape[-1]))
    changed = True
    while changed:
        new_anc = torch.logical_and(adj_matrix[..., None], adj_matrix[..., None, :, :]).any(dim=-2)
        new_anc = torch.logical_or(adj_matrix, new_anc)
        changed = not (new_anc == adj_matrix).all().item()
        adj_matrix = new_anc
    if exclude_cycles:
        is_diagonal = torch.logical_and(adj_matrix, eye_matrix).any(dim=-2, keepdims=True)
        adj_matrix = torch.where(is_diagonal, orig_adj_matrix, adj_matrix)

    if remove_diag:
        adj_matrix = torch.logical_and(adj_matrix, ~eye_matrix)
    return adj_matrix.float()


def mask_actions(actions, learning=bool):
    # Randomly mask actions
    mask = np.random.choice([0, 1], actions.shape)
    if learning:
        mask = torch.from_numpy(mask).to(device="cuda:0")
        actions = actions * mask
        return actions
    else:
        return np.multiply(actions, mask)


