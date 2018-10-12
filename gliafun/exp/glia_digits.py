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

from gliafun import gn


class DigitGlia(nn.Module):
    def __init__(self):
        super().__init__()

        # --------------------------------------------------------------------
        # Build visual neurons
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        # --------------------------------------------------------------------
        # Build glia decision layers

        # Shrink from 320 -> 50
        # glia1 = []
        # for s in reversed(range(50 + 2, 320, 2)):
        #     glia1.append(gn.GliaShrink(s, bias=False))
        #     glia1.append(torch.nn.ReLU())
        # self.fc1 = nn.Sequential(*glia1)

        # To slow to have two glia layers... 'cheating' time
        self.fc1 = nn.Linear(320, 50)

        # Shrink from 50 -> 10
        glia2 = []
        for s in reversed(range(10 + 2, 52, 2)):
            glia2.append(gn.GliaShrink(s, bias=False))
            if s > 10:  # Last glia cells should be linear
                glia2.append(torch.nn.Tanh())
        self.fc2 = nn.Sequential(*glia2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)  # nonlin is built in
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


class DigitNet(nn.Module):
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
        # x = F.dropout(x, training=self.training)
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
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx % log_interval == 0) and debug:
            pred = output.max(1, keepdim=True)[1]
            print(">>> Example target[:5]: {}".format(target[:5].tolist()))
            print(">>> Example output[:5]: {}".format(pred[:5, 0].tolist()))
            print(
                '>>> Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, debug=False):
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
    if debug:
        print(
            '\n>>> Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.
            format(test_loss, correct, len(test_loader.dataset),
                   100. * correct / len(test_loader.dataset)))

    correct = correct / len(test_loader.dataset)
    return test_loss, correct


def main(glia=False,
         batch_size=64,
         test_batch_size=1000,
         epochs=10,
         lr=0.01,
         momentum=0.5,
         use_cuda=False,
         seed=1,
         log_interval=50,
         debug=False):
    """Glia learn to see (digits)"""
    # ------------------------------------------------------------------------
    # Training settings
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")

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
        model = DigitGlia().to(device)
    else:
        model = DigitNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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
        test_loss, correct = test(model, device, test_loader, debug=debug)

    print(">>> After training:")
    print(">>> Loss: {:.5f}, Correct: {:.2f}".format(test_loss, 100 * correct))


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire(main)
