"""Glia learn to play video games"""
import fire
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from glia import gn
from glia.exp.atari_wrappers import create_atari


class DQGlia(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()

        # -------------------------------------------------------------------
        # Vision
        self.conv = nn.Sequential(
            nn.Conv2d(in_features[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_features = self.conv_layer_size(in_features)

        # -------------------------------------------------------------------
        # Decision

        # TODO. shrink d before AGN layer. Neural decode should not be
        # there.

        # Build glia decision layer
        glia1 = []
        for s in reversed(range(512 + 2, conv_features, 2)):
            glia1.append(gn.Gather(s, bias=False))
            glia1.append(torch.nn.ELU())
        self.fc1 = nn.Sequential(*glia1)

        # Linear neurons (decoder?)
        self.fc2 = nn.Linear(512, num_actions)

    def conv_layer_size(self, shape):
        o = self.conv(torch.Tensor(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256  # Done at the last minute for mem
        fx = self.conv(fx).view(fx.size()[0], -1)
        fx = F.elu(self.fc1(fx))  # RELU is the std
        fx = self.fc2(fx)

        return fx


class DQNet(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()

        # Vision
        self.conv = nn.Sequential(
            nn.Conv2d(in_features[0], 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        conv_features = self.conv_layer_size(in_features)

        # Decision
        self.fc1 = nn.Linear(conv_features, 512)  # TODO nonlin
        self.fc2 = nn.Linear(512, num_actions)

    def conv_layer_size(self, shape):
        o = self.conv(torch.Tensor(torch.zeros(1, *shape)))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256  # Done at the last minute for mem
        fx = self.conv(fx).view(fx.size()[0], -1)
        fx = F.elu(self.fc1(fx))  # RELU is the std
        fx = self.fc2(fx)

        return fx


def train(model):
    return model


def test(model):
    pass


def main(env_id, epochs=10, episode_life=True, glia=False):
    # Workaround for DataLoader
    torch.multiprocessing.set_start_method('spawn')

    # Init gym
    env = create_atari(env_id, episode_life=episode_life)

    in_features = None  # TODO
    num_actions = None

    # Init model
    if glia:
        model = DQGlia(in_features, num_actions)
    else:
        model = DQNet(in_features, num_actions)

    # Need a memory.

    # Train loop
    for epoch in range(1, epochs + 1):
        model = train(model)


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire(main)
