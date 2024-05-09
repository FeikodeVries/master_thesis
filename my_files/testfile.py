# import torch
#
# x = torch.rand((2, 12288))
#
# for i, j in enumerate(x):
#     print(j.shape)
import torch
from collections import defaultdict
# from statistics import mean
# test1 = {'a': 1, 'b':2}
# test2 = {'a': 3, 'b':4}
#
# ds = [test1, test2]
# d = {}
# for k in test1.keys():
#   d[k] = list(d[k] for d in ds)
#
# for k in test1.keys():
#     d[k] = mean(d[k])
#
# print(d)
from torch.distributions.normal import Normal

probs = Normal(-1, 1)
action = torch.tensor([0, 0, 0, 0, 0, 0])
probs.log_prob(action).sum(1)

# NON FUNCTIONAL BUT USEFUL RESUME TRAINING CODE
# if args_citris.resume_training:
#     root_dir = str(pathlib.Path(__file__).parent.resolve()) + f'/my_files/data/model_checkpoints/active_iCITRIS/'
#     pretrained_filename = root_dir + 'last.ckpt'
#     # Update stored model with new model
#     if os.path.isfile(pretrained_filename):
#         print('Retrieving causal representation...')
#         citris = active_iCITRISVAE.load_from_checkpoint(pretrained_filename)
#     else:
#         print('Causal representation not found')
# else:
#     citris = active_iCITRISVAE(c_hid=args_citris.c_hid, num_latents=args_citris.num_latents, lr=args_citris.lr,
#                                num_causal_vars=args_citris.num_causal_vars, run_name=run_name, counter=0)


# Lightning Training code
# # Update VAE to better disentangle causal representation
# datasets, data_loaders, data_name = load_datasets(args_citris, 'Walker')
#
# model_args['run_name'] = run_name
# model_args['num_causal_vars'] = datasets['train'].num_vars()
# model_args['counter'] = citris_counter
# model_args['max_epochs'] = 5
# model_args['len_dataloader'] = len(data_loaders['train'])
#
# utils.train_model(model_class=active_iCITRISVAE, train_loader=data_loaders['train'], **model_args)
# agent.update_causal_model()
# path = pathlib.Path(__file__).parent.resolve()
# last_loss = torch.load(f'{path}/my_files/data/last_loss.pt')
# citris_counter += 1
# # END MYCODE

# # CITRIS PRETRAINING
# if args.num_pretraining > 0:
#     for iteration in range(1, args.num_pretraining + 1):
#         for step in range(0, args.num_steps):
#             print(f"Current pretraining step: {global_step}")
#             global_step += args.num_envs
#             obs[step] = next_obs
#             dones[step] = next_done
#             # ALGO LOGIC: action logic
#             with torch.no_grad():
#                 # Randomised pretraining
#                 action, logprob, _, value = agent.get_action_and_value(x=next_obs, current_step=global_step,
#                                                                        training_size=args.total_timesteps,
#                                                                        dropout_prob=args_citris.dropout_pretraining)
#             actions[step] = action
#
#             # TRY NOT TO MODIFY: execute the game and log data.
#             next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
#             next_done = np.logical_or(terminations, truncations)
#             rewards[step] = torch.tensor(reward).to(device).view(-1)
#             next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
#
#         # # MYCODE
#         # # Update VAE to better disentangle causal representation
#         # datasets, data_loaders, data_name = load_datasets(args_citris, 'Walker')
#         #
#         # model_args['run_name'] = run_name
#         # model_args['num_causal_vars'] = datasets['train'].num_vars()
#         # model_args['counter'] = citris_counter
#         # model_args['max_epochs'] = args_citris.max_epochs
#         # model_args['len_dataloader'] = len(data_loaders['train'])
#         #
#         # utils.train_model(model_class=active_iCITRISVAE, train_loader=data_loaders['train'], **model_args)
#         # agent.update_causal_model()
#         # citris_counter += 1
#         # # END MYCODE
#
#     # Update the trained values for the RL to use
#     # agent.update_causal_model()
#     # path = pathlib.Path(__file__).parent.resolve()
#     # last_loss = torch.load(f'{path}/my_files/data/last_loss.pt')

# def update_causal_model(self):
#     # Use pretrained model
#     root_dir = str(pathlib.Path(__file__).resolve()) + f'/my_files/data/model_checkpoints/active_iCITRIS/'
#     pretrained_filename = root_dir + 'last.ckpt'
#     # Update stored model with new model
#     if os.path.isfile(pretrained_filename):
#         self.causal_model = active_iCITRISVAE.load_from_checkpoint(pretrained_filename)

# # if self.causal:
# if self.t < self.batch_size:
#     if len(self.observations) == 0:
#         self.observations = np.array([pixel_observation['pixels']], dtype=np.float32)
#     else:
#         self.observations = np.concatenate((self.observations,
#                                             np.array([pixel_observation['pixels']])), axis=0,
#                                            dtype=np.float32)
# elif self.batch_size == self.t:
#     self.datahandling.batch_update_npz(self.observations)
#     self.observations = []
#     self.t = 0
# self.t += 1