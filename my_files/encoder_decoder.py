import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

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


def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, variational=True):
        super().__init__()
        self.variational = variational
        self.scale_factor = nn.Parameter(torch.zeros(feature_dim, ))

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList([nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)])
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = torch.tanh(h_norm)
        self.outputs['tanh'] = out
        # TODO: Make Variational
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class PixelDecoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32):
        super().__init__()

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.out_dim = OUT_DIM[num_layers]

        self.fc = nn.Linear(feature_dim, num_filters * self.out_dim * self.out_dim)

        self.deconvs = nn.ModuleList()

        for i in range(self.num_layers - 1):
            self.deconvs.append(nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1))
        self.deconvs.append(
            nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1)
        )

        self.outputs = dict()

    def forward(self, h):
        h = torch.relu(self.fc(h))
        self.outputs['fc'] = h

        deconv = h.view(-1, self.num_filters, self.out_dim, self.out_dim)
        self.outputs['deconv1'] = deconv

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            self.outputs['deconv%s' % (i + 1)] = deconv

        obs = self.deconvs[-1](deconv)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param(
                'train_decoder/deconv%s' % (i + 1), self.deconvs[i], step
            )
        L.log_param('train_decoder/fc', self.fc, step)


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}
_AVAILABLE_DECODERS = {'pixel': PixelDecoder}


def make_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert encoder_type in _AVAILABLE_ENCODERS

    return _AVAILABLE_ENCODERS[encoder_type](obs_shape, feature_dim, num_layers, num_filters)


def make_decoder(decoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert decoder_type in _AVAILABLE_DECODERS

    return _AVAILABLE_DECODERS[decoder_type](obs_shape, feature_dim, num_layers, num_filters)


