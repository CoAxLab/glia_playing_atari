"""Glia learn to play video games"""
import fire
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from glia import gn
from glia 

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


def main(env_id, glia=False):
    # Init gym

    # Need a memory.

    
    pass


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire(main)
