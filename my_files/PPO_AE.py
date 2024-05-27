import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from my_files.encoder_decoder import make_encoder, make_decoder
from my_files import utils, actor_critic


class PPO_AE(object):
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            decoder_type='pixel',
            decoder_lr=1e-3,
            decoder_update_freq=1,
            decoder_latent_lambda=0.0,
            decoder_weight_lambda=0.0,
            num_layers=4,
            num_filters=32
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(action_shape))).to('cuda')

        self.logger = {}

        self.actor = actor_critic.Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic = actor_critic.Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        if decoder_type != 'identity':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(actor_critic.weight_init)

            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    def get_value(self, obs, action, detach=False):
        return self.critic(obs, action, detach_encoder=detach)

    def get_action_and_value(self, x, action=None, dropout_prob=0.1):
        # TODO: actor output is probably causing issues with the PPO system
        action_mean, _, _, _ = self.actor(x, compute_log_pi=False, detach_encoder=True)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            # detach = False
            # action_mean, _, _, _ = self.actor(x, compute_log_pi=False)
            # action_logstd = self.actor_logstd.expand_as(action_mean)
            # action_std = torch.exp(action_logstd)
            # probs = Normal(action_mean, action_std)
            action = probs.sample()
        # else:
        #     detach = True
        #     action_mean, _, _, _ = self.actor(x, detach_encoder=True)
        #     action_logstd = self.actor_logstd.expand_as(action_mean)
        #     action_std = torch.exp(action_logstd)
        #     probs = Normal(action_mean, action_std)

        return torch.flatten(action), probs.log_prob(action).sum(1), probs.entropy().sum(1), \
               self.get_value(x, action, detach=True)
        # action_mean = self.actor(x, compute_log_pi=False, detach_encoder=True)
        # action_logstd = self.actor_logstd.expand_as(action_mean)
        # action_std = torch.exp(action_logstd)
        # probs = Normal(action_mean, action_std)
        # if action is None:
        #     action = probs.sample()
        #     # action = utils.mask_actions(action, dropout_prob=dropout_prob)
        #
        # return torch.flatten(action), probs.log_prob(action).sum(1), probs.entropy().sum(1), \
        #        self.get_value(x, action, detach=action is not None)

    def policy_loss(self, newlogprob, b_logprobs, mb_inds, clip_coef, norm_adv, clipfracs, b_advantages):
        logratio = newlogprob - b_logprobs[mb_inds]

        # TODO: Clip logratio to range, the max. approx_kl becomes 0.3 * 0.1
        logratio = torch.min(logratio, logratio.new_full(logratio.size(), 0.3))
        logratio = torch.max(logratio, logratio.new_full(logratio.size(), -0.3))

        ratio = logratio.exp()

        with torch.no_grad():
            # calculate approx_kl http://joschu.net/blog/kl-approx.html
            old_approx_kl = (-logratio).mean()
            approx_kl = ((ratio - 1) - logratio).mean()
            clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

        mb_advantages = b_advantages[mb_inds]
        if norm_adv:
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Policy loss
        pg_loss1 = -mb_advantages * ratio
        pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        return old_approx_kl, approx_kl, pg_loss

    def value_loss(self, newvalue, b_returns, b_values, mb_inds, clip_coef, clip_vloss):
        newvalue = newvalue.view(-1)
        if clip_vloss:
            v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
            v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -clip_coef, clip_coef,)
            v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
            v_loss = 0.5 * v_loss_max.mean()
        else:
            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

        return v_loss

    def rec_loss(self, obs, target_obs):
        h = self.critic.encoder(obs)

        # if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
        #    target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(h)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * h.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss

        return loss, rec_obs

    def update_ppo(self, loss, clip_norm_ac, clip_norm_dec):
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(self.actor.parameters(), clip_norm_ac)
        nn.utils.clip_grad_norm_(self.critic.parameters(), clip_norm_ac)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), clip_norm_dec)

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
