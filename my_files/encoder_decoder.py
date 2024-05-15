import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


# METHODS: ENCODER-DECODER
class Encoder(nn.Module):
    """
    Convolution encoder network
    We use a stack of convolutions with strides in every second convolution to reduce
    dimensionality. For the datasets in question, the network showed to be sufficient.
    """
    # TODO: If num_layers is not processed with image width, the system will crash due to mismatching shapes
    def __init__(self, c_hid, num_latents,
                 c_in=3,
                 width=32,
                 act_fn=lambda: nn.SiLU(),
                 use_batch_norm=True,
                 variational=True):
        super().__init__()
        num_layers = int(np.log2(width) - 2)
        NormLayer = nn.BatchNorm2d if use_batch_norm else nn.InstanceNorm2d
        self.scale_factor = nn.Parameter(torch.zeros(num_latents,))
        self.variational = variational
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(c_in if l_idx == 0 else c_hid,
                              c_hid,
                              kernel_size=3,
                              padding=1,
                              stride=2,
                              bias=False),
                    PositionLayer(c_hid) if l_idx == 0 else nn.Identity(),
                    NormLayer(c_hid),
                    act_fn(),
                    nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1, bias=False),
                    NormLayer(c_hid),
                    act_fn()
                ) for l_idx in range(num_layers)
            ],
            nn.Flatten(),
            nn.Linear(5*5*c_hid, 4*c_hid),  # TODO: the 5x5 is hardcoded to 80x80 image size
            nn.LayerNorm(4*c_hid),
            act_fn(),
            nn.Linear(4*c_hid, (2*num_latents if self.variational else num_latents))
        )

    def forward(self, img):
        # self.test()
        # test_img = self.test_net(img)
        feats = self.net(img)
        if self.variational:
            mean, log_std = feats.chunk(2, dim=-1)
            s = F.softplus(self.scale_factor)
            log_std = torch.tanh(log_std / s) * s  # Stabilizing the prediction
            return mean, log_std
        else:
            return feats


class Decoder(nn.Module):
    """
    Convolutional decoder network
    We use a ResNet-based decoder network with upsample layers to increase the
    dimensionality stepwise. We add positional information in the ResNet blocks
    for improved position-awareness, similar to setups like SlotAttention.
    """

    def __init__(self, c_hid, num_latents,
                 num_labels=-1,
                 width=32,
                 act_fn=lambda: nn.SiLU(),
                 use_batch_norm=True,
                 num_blocks=1,
                 c_out=-1):
        super().__init__()
        if num_labels > 1:
            out_act = nn.Identity()
        else:
            num_labels = 3 if c_out <= 0 else c_out
            out_act = nn.Tanh()
        NormLayer = nn.BatchNorm2d if use_batch_norm else nn.InstanceNorm2d
        self.width = width
        self.linear = nn.Sequential(
            nn.Linear(num_latents, 4 * c_hid),
            nn.LayerNorm(4 * c_hid),
            act_fn(),
            nn.Linear(4 * c_hid, 5*5*c_hid)  # TODO: 5x5 is hardcoded to support 80x80 image size
        )
        num_layers = int(np.log2(width) - 2)
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=True),
                    *[ResidualBlock(nn.Sequential(
                        NormLayer(c_hid),
                        act_fn(),
                        nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
                        PositionLayer(c_hid, decoding=True),
                        NormLayer(c_hid),
                        act_fn(),
                        nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1)
                    )) for _ in range(num_blocks)]
                ) for _ in range(num_layers)
            ],
            NormLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, c_hid, 1),
            PositionLayer(c_hid, decoding=True),
            NormLayer(c_hid),
            act_fn(),
            nn.Conv2d(c_hid, num_labels, 1),
            out_act
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 5, 5)  # TODO: 5x5 is needed
        x = self.net(x)
        return x


class ResidualBlock(nn.Module):
    """ Simple module for residual blocks """

    def __init__(self, net, skip_connect=None):
        super().__init__()
        self.net = net
        self.skip_connect = skip_connect if skip_connect is not None else nn.Identity()

    def forward(self, x):
        return self.skip_connect(x) + self.net(x)


class PositionLayer(nn.Module):
    """ Module for adding position features to images """

    def __init__(self, hidden_dim, decoding=False):
        super().__init__()
        self.pos_embed = nn.Linear(2, hidden_dim)
        self.decoding = decoding

    def forward(self, x):
        pos = create_pos_grid(x.shape[2:], x.device)
        pos = self.pos_embed(pos)
        pos = pos.permute(2, 0, 1)[None]
        x = x + pos
        if self.decoding:
            test = 1
        return x


def create_pos_grid(shape, device, stack_dim=-1):
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, shape[0], device=device),
                                  torch.linspace(-1, 1, shape[1], device=device),
                                  indexing='ij')
    pos = torch.stack([pos_x, pos_y], dim=stack_dim)
    return pos
