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


for i in range(1):
    print(i)

# OLD CODE: ACTOR CRITIC WITH OWN ENCODERS
# self.actor = actor_critic.Actor(
        #     envs.single_observation_space.shape, envs.single_action_space.shape,
        #     1024, 'pixel', 50, -10, 2, 4, 32).to(device)
        # self.critic = actor_critic.Critic(
        #     envs.single_observation_space.shape, envs.single_action_space.shape, 1024, 'pixel', 50, 4, 32).to(device)
# agent_params = [param for name, param in self.named_parameters() if name.startswith('actor_logstd')]
        # critic_params = [param for name, param in self.critic.named_parameters() if name.startswith('Q')]

        # optimizers
        # self.optimizer = torch.optim.Adam(agent_params, lr=args.learning_rate, eps=1e-5)
        # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
        # self.critic_optimizer = torch.optim.Adam(critic_params, lr=args.learning_rate, betas=(0.9, 0.999))
        # self.encoder_optimizer = torch.optim.Adam(self.critic.encoder.parameters(), lr=1e-3)
        # self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=1e-3, weight_decay=1e-6)

        # self.causal_model = iCITRIS(c_hid=args_citris.c_hid, width=IMG_SIZE, num_latents=args_citris.num_latents,
        #                             obs_shape=envs.observation_space.shape, c_in=3 if args_citris.rgb else 1,
        #                             num_causal_vars=args_citris.num_causal_vars,
        #                             run_name=run_name, lambda_sparse=args_citris.lambda_sparse,
        #                             act_fn=args_citris.act_fn, beta_classifier=args_citris.beta_classifier,
        #                             beta_mi_estimator=args_citris.beta_mi_estimator, beta_t1=args_citris.beta_t1,
        #                             autoregressive_prior=args_citris.autoregressive_prior,
        #                             action_shape=envs.action_space.shape)

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