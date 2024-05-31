# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
# import cProfile, pstats
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import pathlib
import torch.nn.functional as F
# import cv2
from my_files import custom_env_wrappers as mywrapper
from my_files.encoder_decoder import make_encoder, make_decoder
import my_files.actor_critic as actor_critic
from my_files.utils import InstantaneousPrior

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 3
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    load_model: bool = False
    """whether to load a model to continue training"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Walker2d-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4  # TODO: Default is 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32  # TODO: Default(32) Could be that the running average from BatchNorm
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0  # TODO: See if this could help --> 0.01 could be useful
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # MY ARGS
    action_repeat: int = 1
    """for how many frames to hold an action, 1 the default, regular action"""
    framestack: int = 3
    """How many frames to stack in a rolling manner"""
    action_dropout: float = 0.1
    """how often to intervene on the actions to be able to disentangle the latent space"""
    save_img: bool = False
    """whether to save reconstructions of the images"""
    hidden_dims: int = 64
    """how many hidden dimensions for the actor-critic networks"""
    activation_function = 'tanh'
    """activation function to use for the actor-critic system"""
    img_size: int = 84
    """image size of the images used to train the PPO agent"""
    latent_dims: int = 50
    """how large the latent representation should be"""
    action_in_critic: bool = False
    """whether to add the action to the critic"""
    set_train: bool = False
    """whether to do model.train()"""
    is_vae: bool = False
    """whether to use a VAE or an AE"""
    beta: float = 1.0
    """the coefficient for the kld on the VAE loss"""
    experiment_name: str = 'test_default_PPO+AE'

    """
    Default PPO+AE:
    AE:
    - action_repeat = 1
    - framestack = 3
    - hidden_dims = 64
    - act_fn = tanh
    - latent_dims = 50
    - action + critic = False
    - set_train = False
    - img_size = 84
    - ent_coef = 0
    - beta = 1.0
    PPO:
    - lr = 3e-4
    - num_minibatches = 32
    - num_steps = 2048
    - anneal_lr = True
    """

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)

        # Stack the frames in a rolling manner
        env = gym.wrappers.PixelObservationWrapper(env, pixels_only=True)
        env = mywrapper.ResizeObservationandFrameSkip(env, shape=args.img_size, frame_skip=args.action_repeat)
        env = mywrapper.FrameStack(env, k=args.framestack)

        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
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

        self.critic = nn.Sequential(
            layer_init(nn.Linear(args.latent_dims + additional_critic_init, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(args.latent_dims, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, args.hidden_dims)), act_fn,
            layer_init(nn.Linear(args.hidden_dims, np.prod(envs.single_action_space.shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))

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
        _, latent = self.encode_imgs(x)
        action_mean = self.actor(latent.detach())
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            # TODO: Should dropout be done in the policy rollout or update? --> I believe in the rollout
            # action = utils.mask_actions(action, dropout_prob=args.dropout_prob)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(latent, action)

    def rec_loss(self, obs, target_obs):
        kld, latent = self.encode_imgs(obs)
        rec_obs = self.decoder(latent)
        rec_loss = F.mse_loss(target_obs, rec_obs)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * latent.pow(2).sum(1)).mean()
        rec_loss = rec_loss + self.decoder_latent_lambda * latent_loss
        if args.is_vae:
            rec_loss = kld * args.beta + rec_loss
        return rec_loss, rec_obs, target_obs


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.experiment_name}__{args.seed}"

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

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    agent = Agent(envs).to(device)

    agent_params = [param for name, param in agent.named_parameters() if name.startswith('actor') or name.startswith('critic')]

    optimizer = optim.Adam([{'params': agent.encoder.parameters(), 'lr': 1e-3},  # Encoder params
                            {'params': agent.decoder.parameters(), 'lr': 1e-3, 'weight_decay': 1e-6},  # Decoder params
                            {'params': agent_params, 'eps': 1e-5}], lr=args.learning_rate)

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

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
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
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # RL TRAINING
    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            if args.set_train:
                agent.train(training=True)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(x=next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # # TODO: Do action repeat --> might be causing problems still
            # reward = 0
            # for i in range(args.action_repeat):
            #     next_obs, r, terminations, truncations, infos = envs.step(action.cpu().numpy())
            #     reward += r
            #     if terminations or truncations:
            #         break

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.array(bool(np.logical_or(terminations, truncations)))
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # AGENT EVAL
        if args.set_train:
            agent.train(training=False)
        # bootstrap value if not done
        with torch.no_grad():
            # TODO: This allows for integrating the action into the critic network
            _, _, _, next_value = agent.get_action_and_value(x=next_obs)
            next_value = next_value.reshape(1, -1)
            # next_value = agent.get_value(agent.encoder(next_obs)).reshape(1, -1)
            # TODO: This perhaps resulted in better return
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

                rec_loss, rec_obs, target_obs = agent.rec_loss(b_obs[mb_inds], b_obs[mb_inds])

                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + rec_loss

                # if args.save_img:
                #     if epoch == 0 and start == 0:
                #         # Save image of an original + reconstruction every policy rollout
                #         path = str(pathlib.Path().absolute()) + '/my_files/data/input_reconstructions'
                #
                #         original_imgs = torch.split(target_obs[0], 3)
                #         original_imgs = [img.permute(1, 2, 0).cpu().detach().numpy() for img in original_imgs]
                #         rec_imgs = torch.split(rec_obs[0], 3)
                #
                #         rec_imgs = [img.permute(1, 2, 0).cpu().detach().numpy() for img in rec_imgs]
                #
                #         cv2.imwrite(f"{path}/{global_step}_{0}_original.png", cv2.cvtColor(255 * original_imgs[0],
                #                                                                            cv2.COLOR_RGB2BGR))
                #         cv2.imwrite(f"{path}/{global_step}_{1}_original.png", cv2.cvtColor(255 * original_imgs[1],
                #                                                                            cv2.COLOR_RGB2BGR))
                #         cv2.imwrite(f"{path}/{global_step}_{2}_original.png", cv2.cvtColor(255 * original_imgs[2],
                #                                                                            cv2.COLOR_RGB2BGR))
                #         cv2.imwrite(f"{path}/{global_step}_{0}_recimg_{rec_loss}.png", cv2.cvtColor(255 * rec_imgs[0],
                #                                                                            cv2.COLOR_RGB2BGR))
                #         cv2.imwrite(f"{path}/{global_step}_{1}_recimg_{rec_loss}.png", cv2.cvtColor(255 * rec_imgs[1],
                #                                                                            cv2.COLOR_RGB2BGR))
                #         cv2.imwrite(f"{path}/{global_step}_{2}_recimg_{rec_loss}.png", cv2.cvtColor(255 * rec_imgs[2],
                #                                                                            cv2.COLOR_RGB2BGR))

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
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
    #  - hidden dims 1064, 512, 256, 128, 64
    #  - actions in critic: yes, no
    #  - learning annealing enabled, disabled
    #  - action repeat 1, 2, 3, 4
    #  - batch size 64, 128
    #  - activation function: relu, tanh, silu
    #  - latent dims: 50, 64, 96
    #  - ent_coef: 0, 0.001, 0.01, 0.1

