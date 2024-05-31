import torch
import torch.nn as nn
import torch.nn.functional as F

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


OUT_DIM = {2: 39, 4: 35, 6: 31}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, variational=False):
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
        self.fc = nn.Linear(num_filters * out_dim * out_dim,
                            2 * self.feature_dim if self.variational else self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
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
        # TODO: Should this last layernorm be done for a vae?
        # h_norm = self.ln(h_fc)
        # self.outputs['ln'] = h_norm
        if self.variational:
            mean, log_std = h_fc.chunk(2, dim=-1)
            s = F.softplus(self.scale_factor)
            log_std = torch.tanh(log_std / s) * s  # Stabilizing the prediction
            return mean, log_std
        else:
            h_norm = self.ln(h_fc)
            self.outputs['ln'] = h_norm

            out = torch.tanh(h_norm)
            self.outputs['tanh'] = out

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


def make_encoder(encoder_type, obs_shape, feature_dim, num_layers, num_filters, variational):
    assert encoder_type in _AVAILABLE_ENCODERS

    return _AVAILABLE_ENCODERS[encoder_type](obs_shape, feature_dim, num_layers, num_filters, variational)


def make_decoder(decoder_type, obs_shape, feature_dim, num_layers, num_filters):
    assert decoder_type in _AVAILABLE_DECODERS

    return _AVAILABLE_DECODERS[decoder_type](obs_shape, feature_dim, num_layers, num_filters)


