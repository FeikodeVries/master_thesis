# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
import time
import cProfile, pstats
import gymnasium as gym
# import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
#from PIL import Image
from shimmy.registration import DM_CONTROL_SUITE_ENVS
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import pathlib
import torch.nn.functional as F
import cv2
from collections import defaultdict
import pathlib

from my_files import custom_env_wrappers as mywrapper
from my_files.encoder_decoder import make_encoder, make_decoder
from my_files import causal_utils
from my_files import shared_utils
from my_files.shared_utils import layer_init, weight_init
from my_files import causal_disentanglement as causal
from my_files.datahandling import load_data_new
import my_files.CURL as curl


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help="""the name of this experiment""")
    parser.add_argument('--seed', type=int, default=35,
                        help="""seed of the experiment""")
    parser.add_argument('--torch_deterministic', type=bool, default=True,
                        help="""if toggled, `torch.backends.cudnn.deterministic=False`""")
    parser.add_argument('--cuda', type=bool, default=True,
                        help="""if toggled, cuda will be enabled by default""")
    parser.add_argument('--track', type=bool, default=False,
                        help="""if toggled, this experiment will be tracked with Weights and Biases""")
    parser.add_argument('--wandb_project', type=str, default='',
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default='',
                        help="the entity (team) of wandb's project")
    parser.add_argument('--capture_video', action='store_true',
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument('--save_model', action='store_false',
                        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument('--upload_model', type=bool, default=False,
                        help="whether to upload the saved model to huggingface")
    parser.add_argument('--hf_entity', type=str, default="",
                        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument('--env_id', type=str, default="dm_control/walker-walk-v0",
                        help="the id of the environment")
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument('--num_envs', type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument('--num_steps', type=int, default=2048,
                        help="the number of steps to run in each environment per policy rollout (Always 1000 for DMC)")
    parser.add_argument('--anneal_lr', action='store_false',
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument('--num_minibatches', type=int, default=16,
                        help="the number of mini-batches (Default is 32)")
    parser.add_argument('--update_epochs', type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument('--norm_adv', action='store_false',
                        help="Toggles advantages normalization")
    parser.add_argument('--clip_coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--clip_vloss', action='store_false',
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument('--ent_coef', type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument('--target_kl', type=float, default=0.0,
                        help="the target KL divergence threshold --> used 0.05 previously")

    # MY ARGUMENTS (Base system)
    parser.add_argument('--eval_representation', action='store_true',
                        help='')
    parser.add_argument('--eval_seed', type=int, default=16,
                        help='')
    parser.add_argument('--model_loc', type=str, default='/baselines/causal_baselines/from_scratch/',
                        help='On the cluster it is: /home/fvs660/cleanrl/cleanrl/my_files/saved_models/')
    parser.add_argument('--state_baseline', action='store_true',
                        help="whether to train base PPO on states or on pixels")
    parser.add_argument('--pixel_baseline', action='store_true',
                        help="whether to train base PPO directly on pixels")
    parser.add_argument('--action_repeat', type=int, default=2,
                        help="for how many frames to hold an action, 1 the default, regular action")
    parser.add_argument('--framestack', type=int, default=3,
                        help="How many frames to stack in a rolling manner")
    parser.add_argument('--save_img', action='store_true',
                        help="whether to save reconstructions of the images")
    parser.add_argument('--hidden_dims', type=int, default=1024,
                        help="how many hidden dimensions for the actor-critic networks")
    parser.add_argument('--img_size', type=int, default=84,
                        help="image size of the images used to train the PPO agent")
    parser.add_argument('--latent_dims', type=int, default=50,
                        help="how large the latent representation should be")
    parser.add_argument('--action_in_critic', action='store_true',
                        help="whether to add the action to the critic")
    parser.add_argument('--is_vae', action='store_true',
                        help="whether to use a VAE or an AE")
    parser.add_argument('--beta', type=float, default=1.0,
                        help="the coefficient for the kld on the VAE loss (1.0 for causal, else 1e-8)")
    parser.add_argument('--encoder_lr', type=float, default=1e-3,
                        help="learning rate for the encoder")
    parser.add_argument('--ae_freeze', type=int, default=2,
                        help="")
    parser.add_argument('--encoderinput_noise', type=float, default=0.05,
                        help="")

    # MY ARGUMENTS (Causal)
    parser.add_argument('--causal', action='store_false',
                        help="whether to use the causal vae to train PPO")
    parser.add_argument('--causal_hidden_dims', type=int, default=32,
                        help='')
    parser.add_argument('--hard_rep', action='store_false')
    parser.add_argument('--counter_lr', type=float, default=1e-3)
    parser.add_argument('--graph_lr', type=float, default=5e-4)
    parser.add_argument('--lambda_sparse', type=float, default=0.02,
                        help="regularizer for encouraging sparse graphs")
    parser.add_argument('--num_graph_samples', type=int, default=8,
                        help='number of graph samples to use in ENCO gradient estimation')
    parser.add_argument('--autoregressive_prior', action='store_true',
                        help='whether the prior per causal variable is autoregressive')
    parser.add_argument('--intervention_prob', type=float, default=0.2,
                        help="how often to intervene on the actions to be able to disentangle the latent space")
    parser.add_argument('--log_std_min', type=int, default=-10,
                        help='')
    parser.add_argument('--log_std_max', type=int, default=2,
                        help='')
    parser.add_argument('--beta_classifier', type=float, default=4.0,
                        help='In iCITRIS this value is 4 for the causal pinball set')
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0,
                        help='Default is 2.0')
    parser.add_argument('--warmup', type=int, default=100,
                        help='')

    # MY ARGUMENTS (CURL)
    parser.add_argument('--curl', action='store_true',
                        help='')
    parser.add_argument('--curl_latent_dims', type=int, default=128,
                        help='')
    parser.add_argument('--pre_transform_img_size', type=int, default=100,
                        help='')
    parser.add_argument('--curl_encoder_update_freq', type=int, default=4,
                        help='')
    parser.add_argument('--encoder_tau', type=float, default=0.005,
                        help='')

    # MAKE AT RUNTIME
    parser.add_argument('--experiment_name', type=str, default='default',
                        help="name of the experiment")
    parser.add_argument('--batch_size', type=int, default=0,
                        help="the batch size (computed in runtime)")
    parser.add_argument('--minibatch_size', type=int, default=0,
                        help="the mini-batch size (computed in runtime)")
    parser.add_argument('--num_iterations', type=int, default=0,
                        help="the number of iterations (computed in runtime)")

    return parser.parse_args()


def make_env(env_id):
    def thunk():
        if args.state_baseline:
            env = gym.make(env_id)
            env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        else:
            env = gym.make(env_id, render_mode="rgb_array", render_kwargs={'camera_id': 0})
            env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = gym.wrappers.RecordEpisodeStatistics(env)

            # Stack the frames in a rolling manner
            env = gym.wrappers.AddRenderObservation(env, render_only=True)
            env = mywrapper.ResizeObservation(env, shape=(args.img_size if not args.curl else args.pre_transform_img_size))
            env = mywrapper.FrameSkip(env, frameskip=args.action_repeat, causal=args.causal,
                                      intervention_prob=args.intervention_prob, intervention_freeze=args.ae_freeze,
                                      rollout_size=args.num_steps)
            if not args.causal:
                env = mywrapper.FrameStack(env, k=args.framestack)
            else:
                env = mywrapper.NormObsSize(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeReward(env, gamma=args.gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


class Agent(nn.Module):
    """
    Combines the PPO agent with the citris implementation to allow for the gradients to flow through the value function
    """
    def __init__(self, envs):
        super().__init__()
        input_dims = np.array(envs.single_observation_space.shape).prod() if (args.state_baseline or args.pixel_baseline) else args.latent_dims
        input_dims = input_dims if not args.causal else (args.latent_dims * np.prod(envs.single_action_space.shape))

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dims, args.hidden_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dims, args.hidden_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dims, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dims, args.hidden_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dims, args.hidden_dims)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dims, np.prod(envs.single_action_space.shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        if not (args.state_baseline or args.pixel_baseline):
            self.encoder = make_encoder(encoder_type='pixel', obs_shape=envs.single_observation_space.shape,
                                        feature_dim=args.latent_dims, input_noise=args.encoderinput_noise,
                                        num_layers=4, num_filters=32,
                                        variational=args.is_vae, curl=True if args.curl else False).to(device)

            self.decoder = make_decoder(decoder_type='pixel', obs_shape=envs.single_observation_space.shape,
                                        action_size=np.prod(envs.single_action_space.shape),
                                        feature_dim=args.latent_dims, num_layers=4, num_filters=32,
                                        causal=args.causal).to(device)
            if not args.eval_representation:
                self.encoder.apply(weight_init)
                self.decoder.apply(weight_init)

            self.decoder_latent_lambda = 1e-6

        if args.causal:
            self.prior = causal_utils.InstantaneousPrior(num_latents=args.latent_dims,
                                                         c_hid=args.causal_hidden_dims,
                                                         num_blocks=np.prod(envs.single_action_space.shape),
                                                         shared_inputs=args.latent_dims,
                                                         num_graph_samples=args.num_graph_samples,
                                                         lambda_sparse=args.lambda_sparse,
                                                         graph_learning_method="ENCO",
                                                         autoregressive=args.autoregressive_prior)

            self.intv_classifier = causal.InstantaneousTargetClassifier(
                num_latents=args.latent_dims,
                num_blocks=np.prod(envs.single_action_space.shape),
                c_hid=args.causal_hidden_dims*2,
                num_layers=1,
                act_fn=nn.SiLU,
                momentum_model=0.9,
                gumbel_temperature=1.0,
                use_normalization=True,
                use_conditional_targets=True
            )
            self.mi_estimator = causal.MIEstimator(num_latents=args.latent_dims,
                                                   num_blocks=np.prod(envs.single_action_space.shape),
                                                   c_hid=args.causal_hidden_dims,
                                                   momentum_model=0.9,
                                                   gumbel_temperature=1.0)
            self.mi_scheduler = causal_utils.SineWarmupScheduler(warmup=50000,
                                                                 start_factor=0.004,
                                                                 end_factor=1.0,
                                                                 offset=20000)
            self.matrix_exp_scheduler = causal_utils.SineWarmupScheduler(warmup=100000,
                                                                         start_factor=-6,
                                                                         end_factor=4,
                                                                         offset=10000)
            self.causal_encoder = None
            self.all_val_dists = defaultdict(list)
            self.all_v_dicts = []
            self.prior_t1 = self.prior

        if args.curl:
            self.encoder_target = make_encoder(encoder_type='pixel', obs_shape=envs.single_observation_space.shape,
                                               feature_dim=args.latent_dims, input_noise=args.encoderinput_noise,
                                               num_layers=4, num_filters=32,
                                               variational=False, curl=True).to(device)
            self.encoder_target.load_state_dict(self.encoder.state_dict())
            self.CURL = curl.CURL(args.latent_dims, args.curl_latent_dims, self.encoder, self.encoder_target,
                                  output_type='continuous').to(device)
            self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()

    # SHARED methods
    def train(self, training=True):
        self.actor.train(training)
        self.critic.train(training)
        if not (args.state_baseline or args.pixel_baseline):
            self.decoder.train(training)
            self.encoder.train(training)
        elif args.causal:
            self.prior.train(training)
            self.intv_classifier.train(training)
            self.mi_estimator.train(training)
        elif args.curl:
            self.CURL.train(training)
            self.encoder_target.train(training)

    def get_action_and_value(self, x, action=None):
        # Prevent the gradient from flowing through the policy
        if args.state_baseline or args.pixel_baseline:
            # Streamline the rest of the code
            latent = x
            if args.pixel_baseline:
                latent = torch.flatten(x, start_dim=1)
            action_mean = self.actor(latent)
        else:
            if args.causal:
                latent = self.get_causal_rep(x)
            else:
                if args.curl:
                    x = curl.center_crop_image(x, args.img_size)
                latent = self.encoder(x)
            action_mean = self.actor(latent.detach())

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        intervention = torch.empty(envs.single_action_space.shape)
        if action is None:
            action = probs.sample()
            # if iteration % args.ae_freeze == 0 and args.causal and not args.eval_representation:
            #     intervention = torch.empty(envs.single_action_space.shape)
            # action, intervention = causal_utils.mask_actions(action, device, args.intervention_prob)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(latent), intervention

    # AE methods
    def rec_loss(self, obs, target_obs):
        latent = self.encoder(obs)

        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = shared_utils.preprocess_obs(target_obs*255.)

        rec_obs = self.decoder(latent)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * latent.pow(2).sum(1)).mean()

        rec_loss = rec_loss + self.decoder_latent_lambda * latent_loss
        return rec_loss, rec_obs, target_obs

    # CURL methods
    def curl_loss(self, obs_anchor, obs_pos):
        z_a = self.CURL.encode(obs_anchor)
        z_pos = self.CURL.encode(obs_pos, ema=True)

        logits = self.CURL.compute_logits(z_a, z_pos)
        labels = torch.arange(logits.shape[0]).long().to(device)
        curl_loss = self.cross_entropy_loss(logits, labels)

        return curl_loss

    # CAUSAL methods
    def get_causal_rep(self, obs):
        z_mean, z_logstd = self.encoder(obs)

        # constrain log_std inside [log_std_min, log_std_max]
        latent_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()

        # Get latent assignment to causal vars
        target_assignment = self.prior.get_target_assignment(hard=args.hard_rep)
        # Assign latent vals to their respective causal var
        latent_causal_assignment = [target_assignment * latent_sample[i][:, None] for i in range(len(latent_sample))]
        latent_causal_assignment = torch.flatten(torch.stack(latent_causal_assignment, dim=0), start_dim=1)

        return latent_causal_assignment

    def causal_loss(self, imgs, intervention_targets, actions, global_step):
        target = intervention_targets

        # En- and decode every element
        z_mean, z_logstd = self.encoder(imgs.flatten(0, 1))

        z_sample = z_mean + torch.randn_like(z_mean) * z_logstd.exp()
        z_sample = z_sample.unflatten(0, imgs.shape[:2])
        z_sample[:, 0] = z_mean.unflatten(0, imgs.shape[:2])[:, 0]
        z_sample = z_sample.flatten(0, 1)

        # Preprocess obs --> normalise around 0
        imgs = shared_utils.preprocess_obs(imgs*255.)

        x_rec = self.decoder(z_sample.unflatten(0, imgs.shape[:2])[:, 1:].flatten(0, 1))
        z_sample, z_mean, z_logstd, x_rec = [t.unflatten(0, (imgs.shape[0], -1)) for t in [z_sample, z_mean, z_logstd, x_rec]]

        # Calculate KL divergence between every pair of frames
        kld = self.prior.forward(z_sample=z_sample[:, 1:].flatten(0, 1),
                                 z_mean=z_mean[:, 1:].flatten(0, 1),
                                 action=actions[:, :-1].flatten(0, 1),
                                 z_logstd=z_logstd[:, 1:].flatten(0, 1),
                                 target=target.flatten(0, 1),
                                 z_shared=z_sample[:, :-1].flatten(0, 1),
                                 matrix_exp_factor=np.exp(self.matrix_exp_scheduler.get_factor(global_step)))
        kld = kld.unflatten(0, (imgs.shape[0], -1))
        # Calculate reconstruction loss
        log_rec_loss = F.mse_loss(x_rec, imgs[:, 1:])
        rec_loss = F.mse_loss(x_rec, imgs[:, 1:], reduction='none').sum(dim=list(range(2, len(x_rec.shape))))

        # Combine to full loss
        loss = (kld * args.beta + rec_loss).mean()

        # Target classifier
        loss_model, loss_z = self.intv_classifier(z_sample=z_sample,
                                                  action=actions,
                                                  logger=None,
                                                  target=intervention_targets,
                                                  transition_prior=self.prior)

        loss = loss + loss_model + loss_z * args.beta_classifier

        # Mutual information estimator
        scheduler_factor = self.mi_scheduler.get_factor(global_step)
        loss_model_mi, loss_z_mi = self.mi_estimator(z_sample=z_sample,
                                                     action=actions,
                                                     logger=None,
                                                     target=intervention_targets,
                                                     transition_prior=self.prior,
                                                     instant_prob=scheduler_factor)

        loss = loss + loss_model_mi + loss_z_mi * args.beta_mi_estimator * (1.0 + 2.0 * scheduler_factor)

        # For stabilizing the mean, since it is unconstrained
        loss_z_reg = (z_sample.mean(dim=[0, 1]) ** 2 + z_sample.std(dim=[0, 1]).log() ** 2).mean()
        loss = loss + 0.1 * loss_z_reg

        logging = {'kld': kld.mean(), 'ae_rec_loss': log_rec_loss, 'rec_loss': rec_loss.mean(), 'intv_classifier_z': loss_z,
                   'mi_estimator_model': loss_model_mi, 'mi_estimator_z': loss_z_mi,
                   'mi_estimator_factor': scheduler_factor, 'reg_loss': loss_z_reg, 'causal_loss': loss}

        return loss, x_rec, imgs, logging


if __name__ == "__main__":
    args = parser()
    # profiler = cProfile.Profile()
    # profiler.enable()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.state_baseline:
        args.set_train = False
        args.save_img = False
        args.is_vae = False
        args.causal = False
        args.hidden_dims = 64
        args.framestack = 1
        args.action_repeat = 1
        args.clip_coef = 0.2

    elif args.causal:
        args.is_vae = True
        args.state_baseline = False

    elif args.curl:
        args.causal = False
        args.is_vae = False
        args.state_baseline = False

    args.experiment_name = f'runs/action_included/{args.env_id}_curl{args.curl}_causal{args.causal}_pixel{args.pixel_baseline}_eval{args.eval_representation}_seed{args.seed}'
    # args.experiment_name = f'baselines/causal_baselines/retrained/walk_binary/dmc_causal_walk_seed{args.seed}'
    if args.curl:
        retrain_name = 'curl'
    elif args.causal:
        retrain_name = 'causal'
    else:
        retrain_name = 'ae'
    run_name = f"{args.experiment_name}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    assert device != 'cuda', "No GPU available"

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id) for _ in range(args.num_envs)])
    agent = Agent(envs).to(device)

    test = envs.metadata
    optimizer_ppo, optimizer_rep, agent_params = shared_utils.make_optimizers(args, agent)

    # Load trained model and evaluate
    if args.eval_representation and not (args.state_baseline or args.pixel_baseline):
        print("Loading model...")
        # model_path = args.model_loc + retrain_name + f'/seed{args.eval_seed}/ppo_causal.cleanrl_model'
        model_path = str(pathlib.Path(__file__).parent.resolve()) + args.model_loc + f"walk_binary/dmc_causal_walk_seed{args.eval_seed}/ppo_causal.cleanrl_model"
        trained_model = torch.load(model_path)
        # Load only the representation params
        representation_params = {key: value for key, value in trained_model['model_state_dict'].items() if
                                 not key.startswith('actor') and not key.startswith('critic')}
        agent.load_state_dict(representation_params, strict=False)
        optimizer_rep.load_state_dict(trained_model['optimizer_ae_state_dict'])

        # Freeze all parameters except for the PPO actor-critic networks
        params = [param for name, param in agent.named_parameters() if not name.startswith('actor') and not name.startswith('critic')]
        for param in params:
            param.requires_grad = False

        graph = agent.prior.get_adj_matrix()
        test = 1

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    interventions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)

    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # RL TRAINING
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        agent.train(training=True)

        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer_ppo.param_groups[0]["lr"] = lrnow

        print(f"Interventions active: {iteration % args.ae_freeze == 0}")

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # print(f"STEP: {step} GLOBAL_STEP: {global_step}")

            # print(f"TEST: args.num_envs{args.num_envs}")

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, intervention = agent.get_action_and_value(x=next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # # TODO: View the DM_control physics values -->
            # #  don't change the action if the intervention takes place,
            # #  instead set velocity of the joint to 0 and then perform the action
            # env_test = envs.envs[0].env.env.env.env.env.env.env.env.env.env.env
            # env_intervened = env_test._env.physics

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "intervention" in infos:
                interventions[step] = torch.from_numpy(infos["intervention"][0][3:])

            if "episode" in infos:
                if args.curl or args.causal and not (args.state_baseline or args.pixel_baseline):
                    # Only save evaluation policy rollouts
                    if iteration % args.ae_freeze != 0:
                        writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)
                    print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                else:
                    print(f"global_step={global_step}, episodic_return={infos['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"], global_step)

        # AGENT EVAL
        agent.train(training=False)
        # bootstrap value if not done
        with torch.no_grad():
            if args.state_baseline or args.pixel_baseline:
                bootstrap_obs = next_obs
                if args.pixel_baseline:
                    bootstrap_obs = torch.flatten(bootstrap_obs)
                next_value = agent.critic(bootstrap_obs).reshape(1, -1)
            else:
                bootstrap_obs = next_obs
                if args.causal:
                    latent = agent.get_causal_rep(next_obs)
                else:
                    if args.curl:
                        bootstrap_obs = curl.center_crop_image(next_obs, args.img_size)
                    latent = agent.encoder(bootstrap_obs)
                next_value = agent.critic(latent).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_interventions = interventions.reshape((-1,) + envs.single_action_space.shape)
        b_interventions = torch.abs(b_interventions - 1)  # --> Flips the intervention

        if args.curl:
            pos = torch.clone(b_obs)
            obs_anchor = curl.random_crop(b_obs, args.img_size)
            obs_pos = curl.random_crop(pos, args.img_size)

        if args.causal:
            _, data_loaders, _ = load_data_new(args, b_obs, b_interventions, b_actions, args.env_id, seq_len=args.framestack)
        else:
            data_loaders = None

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            print(f"EPOCH {epoch}")
            minbatch = 0
            for i in (data_loaders['train'] if args.causal else range(0, args.batch_size, args.minibatch_size)):
                if args.causal:
                    img_pairs = i[0]
                    targets = i[1]
                    stacked_actions = i[2]
                    mb_inds = np.array(i[3])
                    # count = 0
                    # for pairs in img_pairs:
                    #     img = (pairs[0].permute(1,2,0)*255).detach().cpu().numpy().astype(np.uint8)
                    #     save_img = Image.fromarray(img)
                    #     save_img.save(f'runs/images/epoch{epoch}_minbatch{minbatch}_img{count}.png')
                    #     count += 1
                else:
                    end = i + args.minibatch_size
                    mb_inds = b_inds[i:end]
                minbatch += 1

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]

                # Clamp logratio to range
                logratio = torch.min(logratio, logratio.new_full(logratio.size(), 1))
                logratio = torch.max(logratio, logratio.new_full(logratio.size(), -1))

                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                # Calculate representation loss
                # Freeze autoencoder update to allow PPO to adapt to the representations
                if iteration % args.ae_freeze == 0 and not args.eval_representation:
                    if args.causal:
                        loss_representation, rec_obs, target_obs, logging = agent.causal_loss(img_pairs, targets, stacked_actions, global_step)
                    elif args.curl:
                        loss_representation = agent.curl_loss(obs_anchor[mb_inds], obs_pos[mb_inds])
                        if iteration % args.curl_encoder_update_freq == 0:
                            curl.soft_update_params(agent.encoder, agent.encoder_target, args.encoder_tau)
                    elif not (args.state_baseline or args.pixel_baseline):
                        loss_representation, rec_obs, target_obs = agent.rec_loss(b_obs[mb_inds], b_obs[mb_inds])

                loss_ppo = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # TODO: Prevent PPO from learning from iterations with interventions on the velocity
                if args.causal:
                    if iteration % args.ae_freeze != 0:
                        optimizer_ppo.zero_grad()
                        loss_ppo.backward()
                        nn.utils.clip_grad_norm_(agent_params, args.max_grad_norm)
                        optimizer_ppo.step()
                else:
                    optimizer_ppo.zero_grad()
                    loss_ppo.backward()
                    nn.utils.clip_grad_norm_(agent_params, args.max_grad_norm)
                    optimizer_ppo.step()

                if iteration % args.ae_freeze == 0 and not (args.state_baseline or args.eval_representation or args.pixel_baseline):
                    non_agent_params = [param for name, param in agent.named_parameters() if not name.startswith('actor') and not name.startswith('critic')]
                    optimizer_rep.zero_grad()
                    loss_representation.backward()
                    nn.utils.clip_grad_norm_(non_agent_params, args.max_grad_norm)
                    optimizer_rep.step()

            if args.target_kl != 0.0 and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer_ppo.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)

        if iteration % args.ae_freeze == 0 and not (args.state_baseline or args.causal
                                                    or args.eval_representation or args.pixel_baseline):
            writer.add_scalar("losses/representation_loss", loss_representation, global_step)

        if iteration % args.ae_freeze == 0 and args.causal and not args.eval_representation:
            for key, value in logging.items():
                loss_name = 'causal'
                writer.add_scalar(f"{loss_name}/{key}", value, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('tottime')
        # stats.print_stats()

    if args.save_model:
        model_path = f"{run_name}/{args.exp_name}.cleanrl_model"
        if args.state_baseline or args.pixel_baseline:
            torch.save({'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer_ppo.state_dict()}, model_path)
        else:
            torch.save({'model_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer_ppo.state_dict(),
                        'optimizer_ae_state_dict': optimizer_rep.state_dict()}, model_path)
        # torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    print(f"End of training, took: {(time.time() - start_time) / 60} minutes")
    envs.close()
    writer.close()

    # TODO: Add action to all systems:
    #   - Look into the interventions and see if I should give them another value? --> Instead of using an action
    #   to represent an intervention, directly alter the velocity? (set it to 0 and then give it a random value?)
    #   When doing this, be very careful about having the PPO learning from the iterations with interventions active, as it can drastically fling the robot

    # TODO:
    #   Test new setup


    # TODO: Default setups:
    #  State baseline: is_vae = false, action_repeat = 1, lr_anneal = true, clip_coef = 0.2
    #  AE: is_vae = false, action_repeat = 2, framestack = 3, lr_anneal = true, clip_coef = 0.1, num_minibatch = 16
    #  Causal: is_vae = true, causal = true, action_repeat = 2, framestack = 2, beta = 1.0, beta_classifier = 4, causal_hidden_dims = 32
    #  lr_anneal = true, clip_coef = 0.1,  num_minibatch = 16
    #  IMPORTANT:
    #  - LR_Annealing is really useful for stabilising training and improving performance
    #  - Target_kl to be set at 0.05?
