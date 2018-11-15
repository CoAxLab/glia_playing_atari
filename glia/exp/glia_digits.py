#!/usr/bin/env python
"""Glia learn to see (digits)

Code modified from:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
from __future__ import print_function
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from glia import gn
from random import shuffle


class PerceptronGlia(nn.Module):
    """A minst digit perceptron, made only of glia layers"""

    def __init__(self, num_hidden_n=24, random_neurons=False):
        super().__init__()
        # --------------------------------------------------------------------
        # Low d neuron projection:
        #
        # This strond d reduction is bio motivated. 'thousands of
        # neuron contact a single astrocyte (from SFN2018 presentation;
        # need cite).
        #
        # fc0: 784 -> num_hidden_n
        self.fc0 = nn.Linear(784, num_hidden_n)

        # Turn off learning; it's a random neural projection only!
        if random_neurons:
            for p in self.fc0.parameters():
                p.requires_grad = False

        # --------------------------------------------------------------------
        # Start GLIA learning
        # fc1: num_hidden_n -> num_hidden_n
        self.fc1 = nn.Sequential(gn.Slide(num_hidden_n), nn.ELU())

        # fc2: Linear readout, 256 -> 10
        glia2 = []
        for s in reversed(range(10 + 2, num_hidden_n, 2)):
            glia2.append(gn.Gather(s, bias=False))

            # Linear on the last output
            if s > 12:
                glia2.append(torch.nn.ELU())

        self.fc2 = nn.Sequential(*glia2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.elu(self.fc0(x))
        x = self.fc1(x)  # ELU implicit
        x = self.fc2(x)  # ELU then Linear @ final layer.

        return F.log_softmax(x, dim=1)


class PerceptronNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = nn.Linear(784, 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvGlia(nn.Module):
    """A minsy digit perceptron."""

    def __init__(self, random_neurons=False):
        super().__init__()

        # --------------------------------------------------------------------
        # Build visual neurons
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        # Turn off learning for visual neurons; neurons only do random proj
        if random_neurons:
            for p in self.conv1.parameters():
                p.requires_grad = False
            for p in self.conv2.parameters():
                p.requires_grad = False
            for p in self.conv2_drop.parameters():
                p.requires_grad = False

        # Low d projection into glia
        self.fc1 = nn.Linear(320, 20)

        if random_neurons:
            for p in self.fc1.parameters():
                p.requires_grad = False

        # --------------------------------------------------------------------
        # Build glia decision layers
        glia1 = []
        for s in reversed(range(10 + 2, 22, 2)):
            glia1.append(gn.Gather(s, bias=False))

            # Linear for the last set; matches the DigitNet.
            if s > 12:
                glia1.append(torch.nn.ELU())

        self.fc2 = nn.Sequential(*glia1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.tanh(F.max_pool2d(self.conv1(x), 2))
        x = F.tanh(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


def train(model,
          device,
          train_loader,
          optimizer,
          epoch,
          log_interval=10,
          debug=False):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Get batch data
        data, target = data.to(device), target.to(device)

        # Get return
        output = model(data)

        # Learn
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Log
        if (batch_idx % log_interval == 0) and debug:
            pred = output.max(1, keepdim=True)[1]
            print(">>> Example target[:5]: {}".format(target[:5].tolist()))
            print(">>> Example output[:5]: {}".format(pred[:5, 0].tolist()))
            print(
                '>>> Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, progress=False, debug=False):
    # Test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # Log
    if debug or progress:
        print('>>> Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.
              format(test_loss, correct, len(test_loader.dataset),
                     100. * correct / len(test_loader.dataset)))

    correct = correct / len(test_loader.dataset)
    return test_loss, correct


def main(glia=False,
         conv=True,
         random_neurons=False,
         num_hidden_n=24,
         batch_size=64,
         test_batch_size=1000,
         epochs=10,
         lr=0.01,
         use_cuda=False,
         device_num=0,
         seed=1,
         log_interval=50,
         progress=False,
         debug=False):
    """Glia learn to see (digits)"""
    # ------------------------------------------------------------------------
    # Training settings
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(device_num)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            '../data',
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307, ), (0.3081, ))
            ])),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs)

    # ------------------------------------------------------------------------
    # Model init
    if glia:
        if conv:
            model = ConvGlia(random_neurons=random_neurons).to(device)
        else:
            model = PerceptronGlia(
                num_hidden_n=num_hidden_n,
                random_neurons=random_neurons).to(device)
    else:
        if conv:
            model = ConvNet().to(device)
        else:
            model = PerceptronNet().to(device)

    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # ------------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            log_interval=log_interval,
            debug=debug)
        test_loss, correct = test(
            model, device, test_loader, debug=debug, progress=progress)

    print(">>> After training:")
    print(">>> Loss: {:.5f}, Correct: {:.2f}".format(test_loss, 100 * correct))


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire(main)
