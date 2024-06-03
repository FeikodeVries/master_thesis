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
from dm_control.suite import ALL_TASKS
print(*ALL_TASKS, sep="\n")

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

# iCITRIS VAE
# TODO: Replace the iCITRIS encoder and decoder with the SAC+AE versions
# # METHODS: ENCODER-DECODER
# class Encoder(nn.Module):
#     """
#     Convolution encoder network
#     We use a stack of convolutions with strides in every second convolution to reduce
#     dimensionality. For the datasets in question, the network showed to be sufficient.
#     """
#     # TODO: If num_layers is not processed with image width, the system will crash due to mismatching shapes
#     def __init__(self, c_hid, num_latents,
#                  c_in=3,
#                  width=32,
#                  act_fn=lambda: nn.SiLU(),
#                  use_batch_norm=True,
#                  variational=True):
#         super().__init__()
#         num_layers = int(np.log2(width) - 2)
#         NormLayer = nn.BatchNorm2d if use_batch_norm else nn.InstanceNorm2d
#         self.scale_factor = nn.Parameter(torch.zeros(num_latents,))
#         self.variational = variational
#         self.net = nn.Sequential(
#             *[
#                 nn.Sequential(
#                     nn.Conv2d(c_in if l_idx == 0 else c_hid,
#                               c_hid,
#                               kernel_size=3,
#                               padding=1,
#                               stride=2,
#                               bias=False),
#                     PositionLayer(c_hid) if l_idx == 0 else nn.Identity(),
#                     NormLayer(c_hid),
#                     act_fn(),
#                     nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1, bias=False),
#                     NormLayer(c_hid),
#                     act_fn()
#                 ) for l_idx in range(num_layers)
#             ],
#             nn.Flatten(),
#             nn.Linear(5*5*c_hid, 4*c_hid),  # TODO: the 5x5 is hardcoded to 80x80 image size
#             nn.LayerNorm(4*c_hid),
#             act_fn(),
#             nn.Linear(4*c_hid, (2*num_latents if self.variational else num_latents))
#         )
#
#     def forward(self, img):
#         # self.test()
#         # test_img = self.test_net(img)
#         feats = self.net(img)
#         if self.variational:
#             mean, log_std = feats.chunk(2, dim=-1)
#             s = F.softplus(self.scale_factor)
#             log_std = torch.tanh(log_std / s) * s  # Stabilizing the prediction
#             return mean, log_std
#         else:
#             return feats
#
#
# class Decoder(nn.Module):
#     """
#     Convolutional decoder network
#     We use a ResNet-based decoder network with upsample layers to increase the
#     dimensionality stepwise. We add positional information in the ResNet blocks
#     for improved position-awareness, similar to setups like SlotAttention.
#     """
#
#     def __init__(self, c_hid, num_latents,
#                  num_labels=-1,
#                  width=32,
#                  act_fn=lambda: nn.SiLU(),
#                  use_batch_norm=True,
#                  num_blocks=1,
#                  c_out=-1):
#         super().__init__()
#         if num_labels > 1:
#             out_act = nn.Identity()
#         else:
#             num_labels = 3 if c_out <= 0 else c_out
#             out_act = nn.Tanh()
#         NormLayer = nn.BatchNorm2d if use_batch_norm else nn.InstanceNorm2d
#         self.width = width
#         self.linear = nn.Sequential(
#             nn.Linear(num_latents, 4 * c_hid),
#             nn.LayerNorm(4 * c_hid),
#             act_fn(),
#             nn.Linear(4 * c_hid, 5*5*c_hid)  # TODO: 5x5 is hardcoded to support 80x80 image size
#         )
#         num_layers = int(np.log2(width) - 2)
#         self.net = nn.Sequential(
#             *[
#                 nn.Sequential(
#                     nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
#                     *[ResidualBlock(nn.Sequential(
#                         NormLayer(c_hid),
#                         act_fn(),
#                         nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
#                         PositionLayer(c_hid, decoding=True),
#                         NormLayer(c_hid),
#                         act_fn(),
#                         nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1)
#                     )) for _ in range(num_blocks)]
#                 ) for _ in range(num_layers)
#             ],
#             NormLayer(c_hid),
#             act_fn(),
#             nn.Conv2d(c_hid, c_hid, 1),
#             PositionLayer(c_hid, decoding=True),
#             NormLayer(c_hid),
#             act_fn(),
#             nn.Conv2d(c_hid, num_labels, 1),
#             out_act
#         )
#
#     def forward(self, x):
#         x = self.linear(x)
#         x = x.reshape(x.shape[0], -1, 5, 5)  # TODO: 5x5 is needed
#         x = self.net(x)
#         return x
#
#
# class ResidualBlock(nn.Module):
#     """ Simple module for residual blocks """
#
#     def __init__(self, net, skip_connect=None):
#         super().__init__()
#         self.net = net
#         self.skip_connect = skip_connect if skip_connect is not None else nn.Identity()
#
#     def forward(self, x):
#         return self.skip_connect(x) + self.net(x)
#
#
# class PositionLayer(nn.Module):
#     """ Module for adding position features to images """
#
#     def __init__(self, hidden_dim, decoding=False):
#         super().__init__()
#         self.pos_embed = nn.Linear(2, hidden_dim)
#         self.decoding = decoding
#
#     def forward(self, x):
#         pos = create_pos_grid(x.shape[2:], x.device)
#         pos = self.pos_embed(pos)
#         pos = pos.permute(2, 0, 1)[None]
#         x = x + pos
#         if self.decoding:
#             test = 1
#         return x
#
#
# def create_pos_grid(shape, device, stack_dim=-1):
#     pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, shape[0], device=device),
#                                   torch.linspace(-1, 1, shape[1], device=device),
#                                   indexing='ij')
#     pos = torch.stack([pos_x, pos_y], dim=stack_dim)
#     return pos

