# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
import cProfile, pstats
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import pathlib
from gymnasium import spaces
from statistics import mean
import torch.nn.functional as F

from my_files.datahandling import load_data_new
import my_files.datahandling as dh
from my_files.new_active_icitris import iCITRIS
from my_files import custom_env_wrappers as mywrapper
from my_files.encoder_decoder import make_encoder, make_decoder
import my_files.utils as utils
import my_files.actor_critic as actor_critic
from gymnasium.wrappers import PixelObservationWrapper, ResizeObservation, FrameStack
from my_files.PPO_AE import PPO_AE

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
    num_steps: int = 512
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 128  # TODO: Default(32) Could be that the running average from BatchNorm is unstable due to very small minibatches?
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

LOG_FREQ = 10000
IMG_SIZE = 84
# TODO: IMG_Size has to have a whole number as output when calculating x**n = IMG_SIZE / n
#  Otherwise there are issues with decoder resizing, other possible image sizes are: 64: n=4, 96: n=6


def make_env(env_id, capture_video, run_name, gamma):
    # def thunk():
    # if capture_video and idx == 0:
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    else:
        env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)

    # TODO: Stack the frames in a rolling manner
    env = gym.wrappers.PixelObservationWrapper(env, pixels_only=True)
    env = mywrapper.ResizeObservation(env, shape=IMG_SIZE)
    env = mywrapper.FrameStack(env, k=args_citris.framestack)
    env = gym.wrappers.ClipAction(env)

    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

    # return thunk


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
        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod()+(3*IMG_SIZE*IMG_SIZE), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 1), std=1.0),
        # )
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod()+(3*IMG_SIZE*IMG_SIZE), 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, 64)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.1),
        # )

        # self.critic = nn.Sequential(
        #     layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod()+(3*IMG_SIZE*IMG_SIZE), 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 128)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(128, 128)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(128, 1), std=1.0),
        # )
        # self.actor_mean = nn.Sequential(
        #     layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod()+(3*IMG_SIZE*IMG_SIZE), 256)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(256, 128)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(128, 128)),
        #     nn.Tanh(),
        #     layer_init(nn.Linear(128, np.prod(envs.single_action_space.shape)), std=0.1),
        # )

        # TODO: Sometimes, the actor MLP returns NAN?
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod(), 1024)), nn.ReLU(),
            # layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod(), 1024)), nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)), nn.ReLU(),
            layer_init(nn.Linear(1024, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod(), 1024)), nn.ReLU(),
            # layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod(), 1024)), nn.ReLU(),
            layer_init(nn.Linear(1024, 1024)), nn.ReLU(),
            layer_init(nn.Linear(1024, np.prod(envs.action_space.shape)), std=0.01),
        )

        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.action_space.shape)))
        self.causal_model = iCITRIS(c_hid=args_citris.c_hid, width=IMG_SIZE, num_latents=args_citris.num_latents,
                                    obs_shape=envs.observation_space.shape, c_in=3 if args_citris.rgb else 1,
                                    num_causal_vars=args_citris.num_causal_vars,
                                    run_name=run_name, lambda_sparse=args_citris.lambda_sparse,
                                    act_fn=args_citris.act_fn, beta_classifier=args_citris.beta_classifier,
                                    beta_mi_estimator=args_citris.beta_mi_estimator, beta_t1=args_citris.beta_t1,
                                    autoregressive_prior=args_citris.autoregressive_prior,
                                    action_shape=envs.action_space.shape)

    def get_causal_rep_from_img(self, x):
        """
        :param x: Unflattened image tensor to be used by iCITRIS
        :return: Flattened causal representation
        """
        # final_processed = None
        # for i, j in enumerate(x):
        #     processed = self.unflatten_and_process(j)
        #     final_processed = processed if final_processed is None else torch.cat((final_processed, processed))

        causal_rep = self.causal_model.get_causal_rep(x)
        causal_rep = torch.flatten(causal_rep, start_dim=1)

        return causal_rep

    def unflatten_and_process(self, x):
        """
        :param x: Flattened image tensor
        :return: Unflattened image tensor, to be used by iCITRIS
        """
        # Unflatten image
        # unflatten_x = spaces.unflatten(UNFLATTEN_SPACE, x.detach().cpu())
        # Push the image into the correct shape for icitris to use
        #obs = np.array([obs['pixels']])
        img = torch.from_numpy(np.array(obs)).float()
        img = img.permute(0, 3, 1, 2)

        return img.to(device)

    def get_value(self, x):
        causal_rep = self.get_causal_rep_from_img(x)
        # combined_rep = torch.concat((causal_rep, x), dim=1)
        return self.critic(causal_rep)

    def get_action_and_value(self, x, action=None, dropout_prob=0.1):
        # Prevent the gradient from flowing through the policy
        # TODO: Make the process the same as in SAC+AE
        x = x.unsqueeze(0)
        # causal_rep = self.get_causal_rep_from_img(x).detach()
        latent = self.causal_model.encoder(x).detach()
        # combined_rep = torch.concat((causal_rep, x), dim=1)  # Give the causal data as context for the pixel data
        action_mean = self.actor_mean(latent)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
            # TODO: Should dropout be done in the policy rollout or update? --> I believe in the rollout
            action = utils.mask_actions(action, dropout_prob=dropout_prob)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}"

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

    # MYCODE
    parser = utils.get_default_parser()
    parser.add_argument('--model', type=str, default='active_iCITRISVAE')
    parser.add_argument('--c_hid', type=int, default=32)                # TODO: Optimise hyperparam --> Default: 32
    parser.add_argument('--pretraining_size', type=float, default=0)
    parser.add_argument('--dropout_pretraining', type=float, default=0.7)
    parser.add_argument('--dropout_update', type=float, default=0.01)    # TODO: Optimise hyperparam (default: .1)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)    # TODO: Optimise hyperparam
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=50)          # TODO: Optimise hyperparam --> Default: 32
    parser.add_argument('--classifier_lr', type=float, default=4e-3)
    parser.add_argument('--classifier_momentum', type=float, default=0.0)
    parser.add_argument('--classifier_gumbel_temperature', type=float, default=1.0)
    parser.add_argument('--classifier_use_normalization', action='store_true')
    parser.add_argument('--classifier_use_conditional_targets', action='store_true')
    parser.add_argument('--kld_warmup', type=int, default=0)
    parser.add_argument('--beta_t1', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lambda_reg', type=float, default=0.0)
    parser.add_argument('--autoregressive_prior', action='store_true')
    parser.add_argument('--beta_classifier', type=float, default=4.0)
    parser.add_argument('--beta_mi_estimator', type=float, default=0.0)
    parser.add_argument('--lambda_sparse', type=float, default=0.02)
    parser.add_argument('--mi_estimator_comparisons', type=int, default=1)
    parser.add_argument('--graph_learning_method', type=str, default="ENCO")
    parser.add_argument('--graph_lr', type=float, default=5e-4)

    parser.add_argument('--num_causal_vars', type=int, default=6)
    parser.add_argument('--resume_training', type=bool, default=False)
    parser.add_argument('--rgb', type=bool, default=True)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--framestack', type=int, default=3)
    parser.add_argument('--max_iters', type=int, default=100000)
    # TODO: There is a bug in rgb TRUE, it absolutely tanks performance

    datahandler = dh.DataHandling()

    args_citris = parser.parse_args()
    model_args = vars(args_citris)
    args.num_pretraining = args_citris.pretraining_size // args.batch_size
    CAUSAL_OBSERVATION = spaces.Box(shape=(args_citris.num_latents, args_citris.num_causal_vars),
                                    low=-float("inf"), high=float("inf"), dtype=np.float32)

    # END MYCODE

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # TODO: Removed the possibility of syncing multiple environments
    envs = make_env(args.env_id, args.capture_video, run_name, args.gamma)

    # agent = Agent(envs).to(device)
    agent = PPO_AE(obs_shape=envs.observation_space.shape, action_shape=envs.action_space.shape,
                   device='cuda', hidden_dim=1024)

    if args.load_model:
        path = str(pathlib.Path(__file__).parent.resolve()) + f'/runs/{run_name}/ppo_continuous_action.cleanrl_model'
        if os.path.isfile(path):
            print("Loading pretrained RL model")
            agent.load_state_dict(torch.load(path))
        else:
            print("Pretrained RL model not found")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps,) + envs.observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps,) + envs.observation_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps,)).to(device)
    rewards = torch.zeros((args.num_steps,)).to(device)
    dones = torch.zeros((args.num_steps,)).to(device)
    values = torch.zeros((args.num_steps,)).to(device)

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

    # # Get the different params from the causal model
    # citris_graph_params, citris_counter_params, citris_other_params = agent.causal_model.get_params()
    # # Remove the causal model params from the PPO agent
    # agent_params = [param for name, param in agent.named_parameters() if not name.startswith('causal_model')]
    #
    # # TODO: Potentially use AdamW --> optimise the values of the learning rate scheduler
    # optimizer = optim.Adam([{'params': agent_params, 'lr': args.learning_rate, 'eps': 1e-5},
    #                         {'params': citris_graph_params, 'lr': args_citris.graph_lr, 'weight_decay': 0.0, 'eps': 1e-8},
    #                         {'params': citris_counter_params, 'lr': 2 * args_citris.lr, 'weight_decay': 1e-4},
    #                         {'params': citris_other_params}], lr=args.learning_rate, weight_decay=0.0)
    # scheduler = utils.CosineWarmupScheduler(optimizer,
    #                                         warmup=[200 * args_citris.warmup, 2 * args_citris.warmup,
    #                                                  2 * args_citris.warmup],
    #                                         offset=[10000, 0, 0],
    #                                         max_iters=args_citris.max_iters)

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
            # optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # ALGO LOGIC: action logic
            with torch.no_grad():
                # TODO: Do action repeat, only do a new action every so often so the results of a
                #   certain action can be clearer to the system
                if (step % args_citris.action_repeat) == 0:
                    action, logprob, _, value = agent.get_action_and_value(x=next_obs,
                                                                           dropout_prob=args_citris.dropout_update)
                else:
                    action = actions[step - 1]
                    logprob = logprobs[step - 1]
                    value = values[step-1][None, :]

                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        agent.causal_model.set_train(training=False)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
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
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # OBTAIN THE INTERVENTIONS FROM THE ACTION VALUES
        # TODO: These might need to be the other way around as the 0 values are the intervened ones --> if all zeros,
        #   the system will return zeros
        interventions = torch.where(actions != 0, 1.0, 0.0)

        # unflatten_obs = None
        # for ob in obs:
        #     processed = agent.unflatten_and_process(ob)
        #     unflatten_obs = processed if unflatten_obs is None else torch.concat((unflatten_obs, processed))

        datasets, data_loaders, data_name = load_data_new(args_citris, img_data=obs,
                                                          interventions=interventions, env_name=args.env_id,
                                                          seq_len=args_citris.seq_len)

        # Optimizing the policy and value network
        clipfracs = []
        for epoch in range(args.update_epochs):
            print(f'UPDATE EPOCH {epoch}')
            data_loader = 0
            citris_track_logs = {'kld': [], 'rec_loss_t1': [], 'intv_classifier_z': [], 'mi_estimator_model': [],
                                 'mi_estimator_z': [], 'mi_estimator_factor': [], 'reg_loss': []}
            citris_track_loss = []
            for x in data_loaders['train']:
                img_pairs = x[0]
                targets = x[1]
                mb_inds = np.array(x[2])

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(x=b_obs[mb_inds],
                                                                              action=b_actions[mb_inds],
                                                                              dropout_prob=args_citris.dropout_update)
                logratio = newlogprob - b_logprobs[mb_inds]

                # TODO: Clip the logratio to a range of -0.02 and 0.02
                logratio = torch.min(logratio, logratio.new_full(logratio.size(), 0.02))
                logratio = torch.max(logratio, logratio.new_full(logratio.size(), -0.02))

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
                # MYCODE
                # Calculate citris loss
                citris_loss, citris_logs = agent.causal_model.get_loss(batch=img_pairs,
                                                                       target=targets,
                                                                       global_step=global_step,
                                                                       epoch=epoch,
                                                                       data_loader=data_loader)
                # Save all intermediate values for more accurate tracking of performance
                for key, value in citris_logs.items():
                    if key == 'mi_estimator_factor':
                        citris_track_logs[key].append(value)
                    else:
                        citris_track_logs[key].append(value.item())
                citris_track_loss.append(citris_loss.item())
                # To prevent an overflow of saved images
                data_loader += 1

                # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + citris_loss
                loss = citris_loss
                agent.causal_model.set_train(training=True)
                optimizer.zero_grad()
                loss.backward()
                # TODO: This probably already stabilises the child models, but I'm not fully sure
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                # agent.causal_model.clip_gradients()
                optimizer.step()

            # TODO: This should be correct for the scheduler
            scheduler.step()
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

        # CITRIS LOGGING
        for key, value in citris_track_logs.items():
            writer.add_scalar(f"icitris/{key}", mean(value), global_step)
        writer.add_scalar("icitris/train_loss", mean(citris_track_loss), global_step)

        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # profiler.disable()
        # stats = pstats.Stats(profiler).sort_stats('tottime')
        # stats.print_stats()
        # test = 0

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    print(f"End of training, took: {(time.time() - start_time) / 60} minutes")
    envs.close()
    writer.close()

    #  TODO: Current setup: 80x80, bw, lr1e3, dropout 0.01, learning rate scheduler enabled, mi_coef = 0,
    #   target classifier coef = 4, no target classifier loss, no prior (kld) loss
