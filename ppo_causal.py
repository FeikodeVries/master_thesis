# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import argparse
import os
import random
import time
from dataclasses import dataclass
import cProfile, pstats
# import gymnasium as gym
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import pathlib
import torch.nn.functional as F
import cv2
from my_files import custom_env_wrappers as mywrapper
from my_files.encoder_decoder import make_encoder, make_decoder
import my_files.actor_critic as actor_critic

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__)[: -len(".py")],
                        help="""the name of this experiment""")
    parser.add_argument('--seed', type=int, default=0,
                        help="""seed of the experiment""")
    parser.add_argument('--torch_deterministic', type=bool, default=True,
                        help="""if toggled, `torch.backends.cudnn.deterministic=False`""")
    parser.add_argument('--cuda', type=bool, default=True,
                        help="""if toggled, cuda will be enabled by default""")
    parser.add_argument('--track', type=bool, default=False,
                        help="""if toggled, this experiment will be tracked with Weights and Biases""")
    parser.add_argument('--wandb_project', type=str, default=None,
                        help="the wandb's project name")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    parser.add_argument('--capture_video', action='store_true',
                        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument('--save_model', action='store_false',
                        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument('--load_model', type=bool, default=False,
                        help="whether to load a model to continue training")
    parser.add_argument('--upload_model', type=bool, default=False,
                        help="whether to upload the saved model to huggingface")
    parser.add_argument('--hf_entity', type=str, default="",
                        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument('--env_id', type=str, default="Walker2d-v4",
                        help="the id of the environment")
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument('--num_envs', type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument('--num_steps', type=int, default=1000,
                        help="the number of steps to run in each environment per policy rollout (Always 1000 for DMC)")
    parser.add_argument('--anneal_lr', action='store_false',
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument('--gae_lambda', type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument('--num_minibatches', type=int, default=32,
                        help="the number of mini-batches (Default is 32)")
    parser.add_argument('--update_epochs', type=int, default=10,
                        help="the K epochs to update the policy")
    parser.add_argument('--norm_adv', action='store_false',
                        help="Toggles advantages normalization")
    parser.add_argument('--clip_coef', type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--clip_vloss', action='store_false',
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument('--ent_coef', type=float, default=0.0,
                        help="coefficient of the entropy")
    parser.add_argument('--vf_coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument('--target_kl', type=float, default=0.01,
                        help="the target KL divergence threshold")

    # MY ARGUMENTS
    parser.add_argument('--state_baseline', action='store_false',
                        help="whether to train base PPO on states or on pixels")
    parser.add_argument('--action_repeat', type=int, default=1,
                        help="for how many frames to hold an action, 1 the default, regular action")
    parser.add_argument('--framestack', type=int, default=3,
                        help="How many frames to stack in a rolling manner")
    parser.add_argument('--action_dropout', type=float, default=0.1,
                        help="how often to intervene on the actions to be able to disentangle the latent space")
    parser.add_argument('--save_img', action='store_true',
                        help="whether to save reconstructions of the images")
    parser.add_argument('--hidden_dims', type=int, default=64,
                        help="how many hidden dimensions for the actor-critic networks")
    parser.add_argument('--activation_function', type=str, default='tanh',
                        help="activation function to use for the actor-critic system")
    parser.add_argument('--img_size', type=int, default=84,
                        help="image size of the images used to train the PPO agent")
    parser.add_argument('--latent_dims', type=int, default=50,
                        help="how large the latent representation should be")
    parser.add_argument('--action_in_critic', action='store_true',
                        help="whether to add the action to the critic")
    parser.add_argument('--set_train', action='store_true',
                        help="whether to do model.train()")
    parser.add_argument('--is_vae', action='store_false',
                        help="whether to use a VAE or an AE")
    parser.add_argument('--beta', type=float, default=1e-8,
                        help="the coefficient for the kld on the VAE loss")
    parser.add_argument('--encoder_lr', type=float, default=1e-3,
                        help="learning rate for the encoder")
    parser.add_argument('--rpo', action='store_true',
                        help="whether to use RPO")
    parser.add_argument('--rpo_alpha', type=float, default=0.5,
                        help="rpo alpha")

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


def make_env():
    def thunk():
        if args.state_baseline:
            env = gym.make('dmc:Cartpole-swingup-v1')
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        else:
            env = gym.make('dmc:Walker-run-v1', from_pixels=True, channels_first=True, frame_skip=2)
            env = mywrapper.FrameStack(env, k=args.framestack)
        env.seed(args.seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """
    Combines the PPO agent with the citris implementation to allow for the gradients to flow through the value function
    """
    def __init__(self, envs):
        super().__init__()
        if args.activation_function == 'tanh':
            act_fn = nn.Tanh()
        elif args.activation_function == 'relu':
            act_fn = nn.ReLU()
        else:
            act_fn = nn.SiLU()
        additional_critic_init = np.prod(envs.single_action_space.shape) if args.action_in_critic else 0
        input_dims = np.array(envs.single_observation_space.shape).prod() if args.state_baseline else args.latent_dims

        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dims + additional_critic_init, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dims, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, np.prod(envs.single_action_space.shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        if not args.state_baseline:
            self.encoder = make_encoder(encoder_type='pixel', obs_shape=envs.single_observation_space.shape,
                                        feature_dim=args.latent_dims, num_layers=4, num_filters=32,
                                        variational=args.is_vae).to(device)

            self.decoder = make_decoder(decoder_type='pixel', obs_shape=envs.single_observation_space.shape,
                                        feature_dim=args.latent_dims, num_layers=4, num_filters=32).to(device)

            self.encoder.apply(actor_critic.weight_init)
            self.decoder.apply(actor_critic.weight_init)
            self.decoder_latent_lambda = 1e-6

        if args.set_train:
            self.train()

    def train(self, training=True):
        self.actor.train(training)
        self.critic.train(training)
        self.decoder.train(training)
        self.encoder.train(training)

    def get_value(self, obs, action):
        input = torch.cat([obs, action], dim=1) if args.action_in_critic else obs
        return self.critic(input)

    def encode_imgs(self, obs):
        if args.is_vae:
            z_mean, z_logstd = self.encoder(obs)
            latent = z_mean + torch.randn_like(z_mean) * z_logstd.exp()

            kld = torch.mean(-0.5 * torch.sum(1 + z_logstd - z_mean ** 2 - z_logstd.exp(), dim=1), dim=0)
            return kld, latent
        else:
            latent = self.encoder(obs)
            return _, latent

    def get_action_and_value(self, x, action=None):
        # Prevent the gradient from flowing through the policy
        if args.state_baseline:
            # Streamline the rest of the code
            latent = x
            action_mean = self.actor(latent)
        else:
            _, latent = self.encode_imgs(x)
            action_mean = self.actor(latent.detach())
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            # TODO: Should dropout be done in the policy rollout or update? --> I believe in the rollout
            # action = utils.mask_actions(action, dropout_prob=args.dropout_prob)
        else:
            if args.rpo:
                # sample again to add stochasticity, for the policy update
                z = torch.FloatTensor(action_mean.shape).uniform_(-args.rpo_alpha, args.rpo_alpha).to(device)
                action_mean = action_mean + z
                probs = Normal(action_mean, action_std)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(latent, action)

    def rec_loss(self, obs, target_obs):
        kld, latent = self.encode_imgs(obs)
        rec_obs = self.decoder(latent)
        rec_loss = F.mse_loss(target_obs, rec_obs)
        if args.is_vae:
            rec_loss = kld * args.beta + rec_loss
        else:
            # add L2 penalty on latent representation
            # see https://arxiv.org/pdf/1903.12436.pdf
            latent_loss = (0.5 * latent.pow(2).sum(1)).mean()

            rec_loss = rec_loss + self.decoder_latent_lambda * latent_loss
        return rec_loss, rec_obs, target_obs


if __name__ == "__main__":
    args = parser()
    # args = tyro.cli(Args)
    # profiler = cProfile.Profile()
    # profiler.enable()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # args.experiment_name = f'repeat{args.action_repeat}_chid{args.hidden_dims}_activation{args.activation_function}_' \
    #                        f'latent{args.latent_dims}_aIC{args.action_in_critic}_setTrain{args.set_train}_beta{args.beta}_' \
    #                        f'batches{args.num_minibatches}_anneal{args.anneal_lr}_rpo{args.rpo}_seed{args.seed}'
    args.experiment_name = "dmc_test_cartpole_swingup_klclip"
    run_name = f"{args.experiment_name}"
    if args.state_baseline:
        args.set_train = False
        args.anneal_lr = False
        args.save_img = False

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env() for i in range(args.num_envs)])
    agent = Agent(envs).to(device)

    agent_params = [param for name, param in agent.named_parameters() if name.startswith('actor') or name.startswith('critic')]
    if args.state_baseline:
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    else:
        optimizer = optim.Adam([{'params': agent.encoder.parameters(), 'lr': args.encoder_lr},  # Encoder params
                                {'params': agent.decoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': 1e-6},  # Decoder params
                                {'params': agent_params, 'lr': args.learning_rate, 'eps': 1e-5}], lr=args.learning_rate)

    if args.load_model:
        path = str(pathlib.Path(__file__).parent.resolve()) + f'/runs/{run_name}/ppo_continuous_action.cleanrl_model'
        if os.path.isfile(path):
            print("Loading pretrained RL model")
            agent.load_state_dict(torch.load(path))
        else:
            print("Pretrained RL model not found")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    gae = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # RESET THE ENVIRONMENT
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # RL TRAINING
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.set_train:
            agent.train(training=True)
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        total_reward = 0
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(x=next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, infos = envs.step(action.cpu().numpy())
            total_reward += reward
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        print(f"global_step={global_step}, episodic_return={total_reward}")
        writer.add_scalar("charts/episodic_return", total_reward, global_step)

        # AGENT EVAL
        if args.set_train:
            agent.train(training=False)
        # bootstrap value if not done
        with torch.no_grad():
            # TODO: This allows for integrating the action into the critic network
            _, _, _, next_value = agent.get_action_and_value(x=next_obs)
            next_value = next_value.reshape(1, -1)
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

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(x=b_obs[mb_inds],
                                                                              action=b_actions[mb_inds])

                logratio = newlogprob - b_logprobs[mb_inds]
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
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                if args.state_baseline:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                else:
                    rec_loss, rec_obs, target_obs = agent.rec_loss(b_obs[mb_inds], b_obs[mb_inds])
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + rec_loss

                if args.save_img:
                    if epoch == 0 and start == 0:
                        # Save image of an original + reconstruction every policy rollout
                        path = str(pathlib.Path().absolute()) + '/my_files/data/input_reconstructions'

                        original_imgs = torch.split(target_obs[0], 3)
                        original_imgs = [img.permute(1, 2, 0).cpu().detach().numpy() for img in original_imgs]
                        rec_imgs = torch.split(rec_obs[0], 3)

                        rec_imgs = [img.permute(1, 2, 0).cpu().detach().numpy() for img in rec_imgs]

                        cv2.imwrite(f"{path}/{global_step}_{0}_original.png", cv2.cvtColor(255*original_imgs[0],
                                                                                           cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{path}/{global_step}_{1}_original.png", cv2.cvtColor(255*original_imgs[1],
                                                                                           cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{path}/{global_step}_{2}_original.png", cv2.cvtColor(255*original_imgs[2],
                                                                                           cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{path}/{global_step}_{0}_recimg_{rec_loss}.png", cv2.cvtColor(255*rec_imgs[0],
                                                                                           cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{path}/{global_step}_{1}_recimg_{rec_loss}.png", cv2.cvtColor(255*rec_imgs[1],
                                                                                           cv2.COLOR_RGB2BGR))
                        cv2.imwrite(f"{path}/{global_step}_{2}_recimg_{rec_loss}.png", cv2.cvtColor(255*rec_imgs[2],
                                                                                           cv2.COLOR_RGB2BGR))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl != 0.0 and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        if not args.state_baseline:
            writer.add_scalar("losses/rec_loss", rec_loss, global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('tottime')
        # stats.print_stats()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    print(f"End of training, took: {(time.time() - start_time) / 60} minutes")
    envs.close()
    writer.close()

    #  TODO: Current setup: default PPO, framestack, action_repeat==2, no set_train, no logclipping

    # TODO: Different setups to test:
    #  - RPO with default settings
    #  - Current system with default settings now that L2 penalty is removed
    #  - All different action repeats with default settings
    #  - Different encoder lrs
    #  - Different B values for VAE
    #  - Ent coefficient with the correct values

    #  DEFAULT SETTINGS:
    #  - num_minibatches: 16
    #  - learning_annealing: yes
    #  - hidden_dims: 64
    #  - activation: tanh
    #  - action in critic: no
    #  - VAE: true
    #  - annealing: true


    #   - encoder_lr: 1e-3, 1e-4, 1e-5
    #   - beta: 1e-8, 1e-5, 1e-3


