from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
                sum_scale = target.shape[0] * target.shape[1]
                target_sums = target.sum(dim=[0, 1]) / sum_scale
                # target_sums = target
                target_prod = (target[..., None, :] * target[..., :, None]).sum(dim=[0, 1]) / sum_scale
                # target_prod = (target[..., None, :] * target[..., :, None])

                one_cond_one = target_prod / target_sums[None, :].clamp(min=1e-5)
                zero_cond_one = 1 - one_cond_one
                inv_sum = (target.shape[0] * target.shape[1] - target_sums)
                one_cond_zero = (target_sums[:, None] - target_prod) / inv_sum[None, :].clamp(min=1e-5)
                zero_cond_zero = 1 - one_cond_zero
                new_cond_steps = torch.stack([target.shape[0] * target.shape[1] - target_sums, target_sums], dim=-1)
                update_factor = (self.dist_cond_steps / (self.dist_cond_steps + new_cond_steps))[None, :, :, None]
                cond_dist = torch.stack([zero_cond_zero, one_cond_zero,
                                         zero_cond_one, one_cond_one], dim=-1).unflatten(-1, (2, 2))
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
            [z_sample_model[:, :-1, :], z_sample_model[:, 1:, :] * target_assignment_det, target_assignment_det], dim=-1)
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
            avg_dist_labels = self.avg_cond_dist[None, None, :, :, 1, 1] * target[:, :, None, :] + \
                              self.avg_cond_dist[None, None, :, :, 0, 1] * (1 - target[:, :, None, :])
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
            [target_assignment, target_assignment.new_ones(target_assignment.shape[:-1] + (1,))], dim=-1)  # All latents slot
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
        latent_to_causal = (target_assignment[:, :, :graph_probs.shape[0], :, None] * graph_samples_eye[:, :, :, None, :]).sum(dim=-3)
        latent_mask = latent_to_causal.transpose(-2, -1)  # shape: [batch_size, time_steps, causal_vars, latent_vars]
        latent_mask = torch.cat([latent_mask] + ([latent_mask.new_zeros(batch_size, time_steps, 1, num_latents)] if
                                                 (latent_mask.shape[2] == num_classes) else []) +
                                [latent_mask.new_ones(batch_size, time_steps, 1, num_latents)],
                                dim=-2)  # shape [batch, time_steps, num_targets, latent_vars]
        # Model step => Cross entropy loss on targets for all sets of latents
        z_sample_model = z_sample.detach()
        latent_mask_det = latent_mask.detach()
        model_inps = torch.cat([z_sample_model[:, :-1], z_sample_model[:, 1:] * latent_mask_det, latent_mask_det], dim=-1)
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
        # Beneficial to weight the cross entropy loss for true target higher, especially for many causal variables
        pos_weight = num_classes
        loss_z = loss_z * (pos_weight * (loss_mask == 1).float() + 1 * (loss_mask == 0).float())
        loss_z = loss_z.mean()
        loss_z = num_targets * loss_z

        return loss_model, loss_z


class MIEstimator(nn.Module):
    """
    The MI estimator for guiding towards better disentanglement of the causal variables in latent space.
    """

    def __init__(self, num_latents,
                 c_hid,
                 num_blocks,
                 momentum_model=0.0,
                 var_names=None,
                 num_layers=1,
                 act_fn=lambda: nn.SiLU(),
                 gumbel_temperature=1.0,
                 use_normalization=True):
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
        if self.var_names is not None:
            if len(self.var_names) <= num_blocks:
                self.var_names = self.var_names + ['No variable']
            if len(self.var_names) <= num_blocks + 1:
                self.var_names = self.var_names + ['All variables']

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