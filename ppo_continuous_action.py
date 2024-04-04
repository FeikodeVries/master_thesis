# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import pathlib
from torchvision.utils import make_grid
from my_files import custom_env_wrappers as mywrapper
import pytorch_lightning as pl
from gymnasium import spaces


from my_files.datahandling import load_datasets
import my_files.datahandling as dh
from my_files.active_icitris import active_iCITRISVAE
from my_files import custom_env_wrappers as mywrapper
from my_files.datasets import ReacherDataset
import my_files.utils as utils
import matplotlib.pyplot as plt

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
    total_timesteps: int = 50000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
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
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
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


UNFLATTEN_SPACE = None


def make_env(env_id, idx, capture_video, run_name, num_latents, num_causal, gamma, batch_size=100):
    def thunk():
        global UNFLATTEN_SPACE
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = mywrapper.CausalWrapper(env, shape=64, rgb=True, batch_size=batch_size, latents=num_latents,
                                      causal_vars=num_causal, causal=True)
        UNFLATTEN_SPACE = env.observation_space
        env = gym.wrappers.FlattenObservation(env)

        env = mywrapper.ActionWrapper(env, batch_size=batch_size, causal=True)
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
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(CAUSAL_OBSERVATION.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.causal_model = active_iCITRISVAE(c_hid=args_citris.c_hid, num_latents=args_citris.num_latents,
                                              lr=args_citris.lr, num_causal_vars=args_citris.num_causal_vars,
                                              run_name=run_name, counter=0)

    def update_causal_model(self):
        # Use pretrained model
        root_dir = str(pathlib.Path(__file__).resolve()) + f'/my_files/data/model_checkpoints/active_iCITRIS/'
        pretrained_filename = root_dir + 'last.ckpt'
        # Update stored model with new model
        if os.path.isfile(pretrained_filename):
            self.causal_model = active_iCITRISVAE.load_from_checkpoint(pretrained_filename)

    def get_causal_rep_from_img(self, x):
        """
        :param x: Unflattened image tensor to be used by iCITRIS
        :return: Flattened causal representation
        """
        final_processed = None
        for i, j in enumerate(x):
            img = self.unflatten_and_process(j)
            causal_rep = self.causal_model.get_causal_rep(img)
            causal_rep_flat = torch.flatten(causal_rep)[None, :]
            if final_processed is None:
                final_processed = causal_rep_flat
            else:
                final_processed = torch.cat((final_processed, causal_rep_flat))

        return final_processed

    def unflatten_and_process(self, x):
        """
        :param x: Flattened image tensor
        :return: Unflattened image tensor, to be used by iCITRIS
        """
        # Unflatten image
        unflatten_x = spaces.unflatten(UNFLATTEN_SPACE, x.detach().cpu())
        # Push the image into the correct shape for icitris to use
        obs = np.array([unflatten_x['pixels']])
        img = torch.from_numpy(np.array(obs)).float()
        img = img.permute(0, 3, 1, 2)

        return img.to(device)

    def get_value(self, x):
        causal_rep = self.get_causal_rep_from_img(x)
        return self.critic(causal_rep)

    def get_action_and_value(self, x, current_step, training_size, action=None, masking=False):
        # Prevent the gradient from flowing through the policy
        causal_rep = self.get_causal_rep_from_img(x).detach()
        action_mean = self.actor_mean(causal_rep)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        if masking:
            action = utils.mask_actions(action, current_step=current_step, training_size=training_size)

        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.get_value(x)


if __name__ == "__main__":
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
    parser.add_argument('--c_hid', type=int, default=32)
    parser.add_argument('--pretraining_size', type=float, default=50000)
    parser.add_argument('--update_interval', type=int, default=5)
    parser.add_argument('--decoder_num_blocks', type=int, default=1)
    parser.add_argument('--act_fn', type=str, default='silu')
    parser.add_argument('--num_latents', type=int, default=32)
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
    parser.add_argument('--beta_classifier', type=float, default=2.0)
    parser.add_argument('--beta_mi_estimator', type=float, default=2.0)
    parser.add_argument('--lambda_sparse', type=float, default=0.02)
    parser.add_argument('--mi_estimator_comparisons', type=int, default=1)
    parser.add_argument('--graph_learning_method', type=str, default="ENCO")

    parser.add_argument('--num_causal_vars', type=int, default=6)
    parser.add_argument('--resume_training', type=bool, default=False)

    datahandler = dh.DataHandling()

    args_citris = parser.parse_args()
    model_args = vars(args_citris)
    citris_counter = 0
    last_loss = None
    args.num_pretraining = args_citris.pretraining_size // args.batch_size
    CAUSAL_OBSERVATION = spaces.Box(shape=(args_citris.num_latents, args_citris.num_causal_vars),
                                    low=-float("inf"), high=float("inf"), dtype=np.float32)

    if args_citris.resume_training:
        root_dir = str(pathlib.Path(__file__).parent.resolve()) + f'/my_files/data/model_checkpoints/active_iCITRIS/'
        pretrained_filename = root_dir + 'last.ckpt'
        # Update stored model with new model
        if os.path.isfile(pretrained_filename):
            print('Retrieving causal representation...')
            citris = active_iCITRISVAE.load_from_checkpoint(pretrained_filename)
        else:
            print('Causal representation not found')
    else:
        citris = active_iCITRISVAE(c_hid=args_citris.c_hid, num_latents=args_citris.num_latents, lr=args_citris.lr,
                                   num_causal_vars=args_citris.num_causal_vars, run_name=run_name, counter=0)
    pl.seed_everything(args.seed)
    # END MYCODE

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args_citris.num_latents, args_citris.num_causal_vars,
                  args.gamma, batch_size=args.num_steps) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    if args.load_model:
        path = str(pathlib.Path(__file__).parent.resolve()) + f'/runs/{run_name}/ppo_continuous_action.cleanrl_model'
        if os.path.isfile(path):
            print("Loading pretrained RL model")
            agent.load_state_dict(torch.load(path))
        else:
            print("Pretrained RL model not found")
    # TODO: agent.parameters not taking into account the citris params
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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

    # CITRIS PRETRAINING
    if args.num_pretraining > 0:
        for iteration in range(1, args.num_pretraining + 1):
            for step in range(0, args.num_steps):
                print(f"Current pretraining step: {global_step}")
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # Randomised pretraining
                    # action = envs.action_space.sample()
                    # action = utils.mask_actions(action, current_step=global_step, training_size=args.total_timesteps)
                    # action = torch.from_numpy(action).float().to(device)
                    action, logprob, _, value = agent.get_action_and_value(x=next_obs, current_step=global_step,
                                                                           training_size=args.total_timesteps)
                actions[step] = action

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # MYCODE
            # Update VAE to better disentangle causal representation
            datasets, data_loaders, data_name = load_datasets(args_citris, 'Walker')

            model_args['run_name'] = run_name
            model_args['num_causal_vars'] = datasets['train'].num_vars()
            model_args['counter'] = citris_counter
            model_args['max_epochs'] = args_citris.max_epochs
            model_args['len_dataloader'] = len(data_loaders['train'])

            utils.train_model(model_class=active_iCITRISVAE, train_loader=data_loaders['train'], **model_args)
            agent.update_causal_model()
            citris_counter += 1
            # END MYCODE

        # Update the trained values for the RL to use
        agent.update_causal_model()
        path = pathlib.Path(__file__).parent.resolve()
        last_loss = torch.load(f'{path}/my_files/data/last_loss.pt')

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
            # ALGO LOGIC: action logic
            with torch.no_grad():
                if iteration % args_citris.update_interval == 0:
                    # Get data from this iteration to update the causal representation
                    action, logprob, _, value = agent.get_action_and_value(x=next_obs, current_step=global_step,
                                                                           training_size=args.total_timesteps,
                                                                           masking=True)
                else:
                    action, logprob, _, value = agent.get_action_and_value(x=next_obs, current_step=global_step,
                                                                           training_size=args.total_timesteps,
                                                                           masking=False)
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

        if iteration % args_citris.update_interval == 0:
            # Update VAE to better disentangle causal representation
            datasets, data_loaders, data_name = load_datasets(args_citris, 'Walker')

            model_args['run_name'] = run_name
            model_args['num_causal_vars'] = datasets['train'].num_vars()
            model_args['counter'] = citris_counter
            model_args['max_epochs'] = 5
            model_args['len_dataloader'] = len(data_loaders['train'])

            utils.train_model(model_class=active_iCITRISVAE, train_loader=data_loaders['train'], **model_args)
            agent.update_causal_model()
            path = pathlib.Path(__file__).parent.resolve()
            last_loss = torch.load(f'{path}/my_files/data/last_loss.pt')
            citris_counter += 1
            # END MYCODE

        # TODO: add needed citris params to PPO
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
                                                                              current_step=global_step,
                                                                              training_size=args.total_timesteps,
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

                # In testing, if pretraining hasn't been performed be sure to leave out last_loss
                if last_loss is None:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                else:
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + (0.00125 * last_loss)
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    print(f"End of training, took: {(time.time() - start_time) / 60} minutes")
    envs.close()
    writer.close()
