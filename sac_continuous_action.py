# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import argparse
from shimmy.registration import DM_CONTROL_SUITE_ENVS

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help="""the name of this experiment""")
    parser.add_argument('--seed', type=int, default=77,
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
    parser.add_argument('--target_kl', type=float, default=0.05,
                        help="the target KL divergence threshold")

    # MY ARGUMENTS (Base system)
    parser.add_argument('--eval_representation', action='store_true',)
    parser.add_argument('--model_loc', type=str, default='/baselines/ae_baselines/dmc_walk_ae')

    parser.add_argument('--state_baseline', action='store_true',
                        help="whether to train base PPO on states or on pixels")
    parser.add_argument('--action_repeat', type=int, default=2,
                        help="for how many frames to hold an action, 1 the default, regular action")
    parser.add_argument('--framestack', type=int, default=3,
                        help="How many frames to stack in a rolling manner")
    parser.add_argument('--save_img', action='store_true',
                        help="whether to save reconstructions of the images")
    parser.add_argument('--hidden_dims', type=int, default=1024,
                        help="how many hidden dimensions for the actor-critic networks")
    parser.add_argument('--activation_function', type=str, default='tanh',
                        help="activation function to use for the actor-critic system")
    parser.add_argument('--img_size', type=int, default=84,
                        help="image size of the images used to train the PPO agent")
    parser.add_argument('--latent_dims', type=int, default=50,
                        help="how large the latent representation should be")
    parser.add_argument('--action_in_critic', action='store_true',
                        help="whether to add the action to the critic")
    parser.add_argument('--is_vae', action='store_true',
                        help="whether to use a VAE or an AE")
    parser.add_argument('--beta', type=float, default=1e-8,
                        help="the coefficient for the kld on the VAE loss")
    parser.add_argument('--encoder_lr', type=float, default=1e-3,
                        help="learning rate for the encoder")
    parser.add_argument('--ae_freeze', type=int, default=2,
                        help="")
    parser.add_argument('--encoderinput_noise', type=float, default=0.05,
                        help="")

    # MY ARGUMENTS (Causal)
    parser.add_argument('--causal', action='store_true',
                        help="whether to use the causal vae to train PPO")
    parser.add_argument('--causal_coef', type=float, default=1.0)
    parser.add_argument('--counter_lr', type=float, default=1e-3)
    parser.add_argument('--graph_lr', type=float, default=5e-4)
    parser.add_argument('--lambda_sparse', type=float, default=0.02,
                        help="regularizer for encouraging sparse graphs")
    parser.add_argument('--num_graph_samples', type=int, default=8,
                        help='number of graph samples to use in ENCO gradient estimation')
    parser.add_argument('--autoregressive_prior', action='store_true',
                        help='whether the prior per causal variable is autoregressive')
    parser.add_argument('--action_dropout', type=float, default=0.1,
                        help="how often to intervene on the actions to be able to disentangle the latent space")
    parser.add_argument('--log_std_min', type=int, default=-10,
                        help='')
    parser.add_argument('--log_std_max', type=int, default=2,
                        help='')
    parser.add_argument('--beta_classifier', type=float, default=2.0,
                        help='Default is 2.0')
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0,
                        help='Default is 2.0')
    parser.add_argument('--warmup', type=int, default=100,
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


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = parser()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
    writer = SummaryWriter(f"runs/{run_name}")
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

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()