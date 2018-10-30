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


class SkipInputGlia(nn.Module):
    """A minst digit glai perceptron, w/ 'skipped input'. """

    def __init__(self, skip_features=56, random_skip=False):
        super().__init__()

        if (784 % skip_features) != 0:
            raise ValueError("skip_features must evenly divide 785")

        self.skip_features = skip_features
        self.num_skip = int(784 / skip_features)
        self.random_skip = random_skip

        # Skip inputs
        # fc0: 784 -> skip_features
        self.fc0 = []
        for _ in range(self.num_skip):
            self.fc0.append(
                nn.Sequential(
                    gn.Slide(skip_features), nn.BatchNorm1d(skip_features),
                    nn.ELU()))

        # fc1
        self.fc1 = nn.Sequential(gn.Slide(skip_features), nn.ELU())

        # fc2: Linear readout, skip_features -> 10
        glia2 = []
        for s in reversed(range(10 + 2, skip_features, 2)):
            glia2.append(gn.Gather(s, bias=False))
            # Linear on the last output
            if s > 12:
                glia2.append(torch.nn.ELU())

        self.fc2 = nn.Sequential(*glia2)

    def forward(self, x):
        x = x.view(-1, 784)

        # Create skip index
        skip_index = list(range(784))
        if self.random_skip:
            shuffle(skip_index)

        # fc0: skip input
        i = 0
        x_last = torch.zeros(x.shape[0], self.skip_features)
        for fc in self.fc0:
            skip = skip_index[i:(i + self.skip_features)]
            x_last = fc(x[:, skip] + x_last)
            i += self.skip_features
        x = x_last

        # fc1: nonlin slide
        x = self.fc1(x)

        # fc2: 'lin' decode
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class PerceptronGlia(nn.Module):
    """A minst digit perceptron, made only of glia layers"""

    def __init__(self):
        super().__init__()

        # fc0: 784 -> 256
        glia1 = []
        for s in reversed(range(256 + 2, 784, 2)):
            glia1.append(gn.Gather(s, bias=False))
            glia1.append(nn.ELU())
        self.fc0 = nn.Sequential(*glia1)

        # fc1: 256 -> 256
        self.fc1 = nn.Sequential(gn.Slide(256), nn.ELU())

        # fc2: Linear readout, 256 -> 10
        glia2 = []
        for s in reversed(range(10 + 2, 256, 2)):
            glia2.append(gn.Gather(s, bias=False))

            # Linear on the last output
            if s > 12:
                glia2.append(torch.nn.ELU())

        self.fc2 = nn.Sequential(*glia2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
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

    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------
        # Build visual neurons
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        # --------------------------------------------------------------------
        # Build glia decision layers

        # CHEATING FOR TRAINSPEED.
        self.fc1 = nn.Linear(320, 20)

        # GLIA make the final decision
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
         skip=True,
         batch_size=64,
         test_batch_size=1000,
         epochs=10,
         lr=0.01,
         momentum=0.5,
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
            model = ConvGlia().to(device)
        else:
            if skip:
                model = SkipInputGlia(random=False).to(device)
            else:
                model = PerceptronGlia().to(device)
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
