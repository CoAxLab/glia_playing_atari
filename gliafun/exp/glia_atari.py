"""Glia learn to play video games"""
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from gliafun import gn


class DQGlia(nn.Module):
    def __init__(self, in_features, num_actions):
        # --------------------------------------------------------------------
        # Build visual neuronal network
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # --------------------------------------------------------------------
        # Build glia decision layer
        glia1 = []
        for s in reversed(range(num_actions + 2, 448, 2)):
            glia1.append(gn.GliaShrink(s, bias=False))
            glia1.append(torch.nn.Tanh())
        self.head = nn.Sequential(*glia1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNet(nn.Module):
    def __init__(self, in_features, num_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, num_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def main(glia=False):
    pass


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire(main)
