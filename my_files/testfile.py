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