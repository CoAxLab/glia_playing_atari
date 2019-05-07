#!/usr/bin/env python
"""Glia learn to see (digits)

Code modified from:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""
from __future__ import print_function
import os
import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

from glia import gn
from random import shuffle


class VAE(nn.Module):
    """A MINST-shaped VAE."""

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


class PerceptronNet(nn.Module):
    """A minst digit perceptron.

    Note: assumes input is from a VAE.
    """

    def __init__(self, z_features=20):
        super().__init__()
        self.z_features = z_features

        self.fc1 = nn.Linear(self.z_features, self.z_features)
        self.fc2 = nn.Linear(self.z_features, 10)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class PerceptronGlia(nn.Module):
    """A minst digit perceptron.

    Note: assumes input is from a VAE.
    """

    def __init__(self, z_features=20, activation_function='Softmax'):
        # --------------------------------------------------------------------
        # Init
        super().__init__()
        self.z_features = z_features

        # Lookup activation function (a class)
        AF = getattr(nn, activation_function)

        # --------------------------------------------------------------------
        # Def fc1:
        glia1 = []
        for s in reversed(range(12, self.z_features, 2)):
            glia1.append(gn.Gather(s))
            glia1.append(gn.Slide(s - 2))
            # Linear on the last output, for digit decode
            if s > 12:
                glia1.append(AF())
        self.glia1 = nn.Sequential(*glia1)

    def forward(self, x):
        x = self.glia1(x)

        return F.log_softmax(x, dim=1)


class TravelingWave(nn.Module):
    """A minst digit traveling wave of computation.

    Note: assumes input is from a VAE.
    """

    def __init__(self,
                 z_features=20,
                 wave_size=40,
                 activation_function='Softmax'):
        # --------------------------------------------------------------------
        # Init
        super().__init__()
        self.z_features = z_features
        self.wave_size = wave_size

        # Lookup activation function (a class)
        AF = getattr(nn, activation_function)

        # --------------------------------------------------------------------
        # Def growing wave:
        glia1 = []
        for s in range(self.z_features, self.wave_size, 2):
            glia1.append(gn.Spread(s))
            glia1.append(gn.Slide(s + 2))
            glia1.append(AF())
        self.glia1 = nn.Sequential(*glia1)

        # --------------------------------------------------------------------
        # Def gather:
        glia2 = []
        for s in reversed(range(12, self.wave_size + 2, 2)):
            glia2.append(gn.Gather(s))
            glia2.append(gn.Slide(s - 2))
            # Linear on the last output, for digit decode
            if s > 12:
                glia2.append(AF())
        self.glia2 = nn.Sequential(*glia2)

    def forward(self, x):
        x = self.glia1(x)
        x = self.glia2(x)

        return F.log_softmax(x, dim=1)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train_vae(model,
              device,
              train_loader,
              optimizer,
              epoch,
              log_interval=10,
              progress=False,
              debug=False):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if debug and (batch_idx % log_interval == 0):
            print('>>> VAE train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                  format(epoch, batch_idx * len(data),
                         len(train_loader.dataset),
                         100. * batch_idx / len(train_loader),
                         loss.item() / len(data)))


def test_vae(model,
             device,
             test_loader,
             epoch,
             batch_size,
             progress=False,
             debug=False,
             data_path="."):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar, _ = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat(
                    [data[:n],
                     recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(
                    comparison.cpu(),
                    os.path.join(data_path,
                                 "reconstruction_{}.png".format(epoch)),
                    nrow=n)

    test_loss /= len(test_loader.dataset)
    if progress or debug:
        print('>>> VAE test loss: {:.4f}'.format(test_loss))
    return test_loss


def train(model,
          model_vae,
          device,
          train_loader,
          optimizer,
          epoch,
          log_interval=10,
          progress=False,
          debug=False):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # Get batch data
        data, target = data.to(device), target.to(device)

        # Classify
        _, _, _, z = model_vae(data)
        output = model(z)

        # Learn
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Log
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()

        if (batch_idx % log_interval == 0) and debug:
            print(">>> Train example target[:5]: {}".format(
                target[:5].tolist()))
            print(">>> Train example output[:5]: {}".format(
                pred[:5, 0].tolist()))
            print(
                '>>> Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    if progress or debug:
        print('>>> Train loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))


def test(model, model_vae, device, test_loader, progress=False, debug=False):
    # Test
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Get batch data
            data, target = data.to(device), target.to(device)

            # Classify
            _, _, _, z = model_vae(data)
            output = model(z)

            # Scores
            test_loss += F.nll_loss(
                output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(
                1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # Log
    if debug or progress:
        print('>>> Test loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    correct = correct / len(test_loader.dataset)
    return test_loss, correct


def run(glia=False,
        batch_size=128,
        test_batch_size=128,
        epochs=10,
        lr=0.01,
        wave_size=None,
        vae_path=None,
        lr_vae=1e-3,
        use_cuda=False,
        device_num=0,
        seed=1,
        save=None,
        log_interval=50,
        progress=False,
        debug=False,
        data_path=None):
    """Glia learn to see (digits)"""
    # ------------------------------------------------------------------------
    # Training settings
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.cuda.set_device(device_num)

    if data_path is None:
        data_path = "data"

    # ------------------------------------------------------------------------
    # Get and pre-process data
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_path,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            data_path,
            train=False,
            transform=transforms.ToTensor(),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs)

    # ------------------------------------------------------------------------
    # Decision model
    # Init
    # Init
    if vae_path is None:
        model_vae = VAE().to(device)
        optimizer_vae = optim.Adam(model_vae.parameters(), lr=lr_vae)
    else:
        model_vae = None  ## TODO load me

    if glia:
        if wave_size is not None:
            model = TravelingWave(wave_size=wave_size).to(device)
        else:
            model = PerceptronGlia().to(device)
    else:
        model = PerceptronNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learn classes
    for epoch in range(1, epochs + 1):
        # Learn z?
        if vae_path is None:
            train_vae(
                model_vae,
                device,
                train_loader,
                optimizer_vae,
                epoch,
                log_interval=log_interval,
                debug=debug,
                progress=progress)

            test_loss = test_vae(
                model_vae,
                device,
                test_loader,
                epoch,
                test_batch_size,
                debug=debug,
                progress=progress,
                data_path=data_path)

        # Glia learn
        train(
            model,
            model_vae,
            device,
            train_loader,
            optimizer,
            epoch,
            log_interval=log_interval,
            progress=progress,
            debug=debug)

        test_loss, correct = test(
            model,
            model_vae,
            device,
            test_loader,
            debug=debug,
            progress=progress)

    print(">>> After training:")
    print(">>> Loss: {:.5f}, Correct: {:.2f}".format(test_loss, 100 * correct))

    if save is not None:
        state = dict(
            model_dict=model.state_dict(),
            vae_dict=model_vae.state_dict(),
            glia=glia,
            wave_size=wave_size,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            epochs=epochs,
            lr=lr,
            vae_path=vae_path,
            lr_vae=lr_vae,
            use_cuda=use_cuda,
            device_num=device_num,
            seed=seed)
        torch.save(state, save + ".pytorch")


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire(run)
