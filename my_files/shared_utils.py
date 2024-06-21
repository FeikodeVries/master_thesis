import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def make_optimizers(args, agent):
    # Define optimizers
    if args.state_baseline or args.pixel_baseline:
        agent_params = agent.parameters()
        optimizer_ppo = optim.Adam(agent_params, lr=args.learning_rate, eps=1e-5)

        return optimizer_ppo, None, agent_params

    elif args.curl:
        agent_params = [param for name, param in agent.named_parameters() if
                        name.startswith('actor') or name.startswith('critic')]
        curl_params = [param for name, param in agent.CURL.named_parameters() if not name.startswith('encoder')]

        optimizer_ppo = optim.Adam(agent_params, lr=args.learning_rate, eps=1e-5)
        optimizer_rep = optim.Adam(
            [{'params': curl_params},
             {'params': agent.encoder.parameters(), 'weight_decay': 1e-6}], lr=args.encoder_lr)

    elif args.causal:
        graph_params, counter_params, agent_params, other_params = [], [], [], []

        for name, param in agent.named_parameters():
            if name.startswith('prior.enco') or name.startswith('prior.notears'):
                graph_params.append(param)
            elif name.startswith('intv_classifier') or name.startswith('mi_estimator'):
                counter_params.append(param)
            elif name.startswith('actor') or name.startswith('critic'):
                agent_params.append(param)
            elif not (name.startswith('encoder') or name.startswith('decoder')):
                other_params.append(param)

        optimizer_ppo = optim.Adam(agent_params, lr=args.learning_rate, eps=1e-5)
        optimizer_rep = optim.Adam(
            [{'params': graph_params, 'lr': args.graph_lr, 'weight_decay': 0.0, 'eps': 1e-8},
             {'params': counter_params, 'lr': 2 * args.counter_lr, 'weight_decay': 1e-4},
             {'params': agent.encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': 1e-6},
             {'params': agent.decoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': 1e-6},
             {'params': other_params, 'lr': args.counter_lr, 'weight_decay': 0.0}], lr=args.learning_rate)

    else:
        agent_params = [param for name, param in agent.named_parameters() if
                        name.startswith('actor') or name.startswith('critic')]
        optimizer_rep = optim.Adam([{'params': agent.encoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': 1e-6},
                                    {'params': agent.decoder.parameters(), 'lr': args.encoder_lr, 'weight_decay': 1e-6},
                                    ], lr=args.learning_rate)

        optimizer_ppo = optim.Adam(agent_params, lr=args.learning_rate, eps=1e-5)

    return optimizer_ppo, optimizer_rep, agent_params
