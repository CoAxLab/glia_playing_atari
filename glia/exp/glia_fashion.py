#!/usr/bin/env python
"""Glia learn to see (clothes)

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

import numpy as np
from sklearn.random_projection import gaussian_random_matrix
from sklearn.random_projection import sparse_random_matrix

from glia import gn
from random import shuffle
from copy import deepcopy


class GP(nn.Module):
    """A Gaussian Random Projection"""
    def __init__(self, n_features=784, n_components=20, random_state=None):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.decode = torch.Tensor(
            gaussian_random_matrix(self.n_components,
                                   self.n_features,
                                   random_state=random_state)).float()

    def forward(self, x):
        z = torch.einsum("bi,ji->bj", x.reshape(x.shape[0], 784), self.decode)
        return None, None, None, z  # Mimic VAE fwd return


class SP(nn.Module):
    """A Sparse Random Projection"""
    def __init__(self, n_features=784, n_components=20, random_state=None):
        super().__init__()
        self.n_features = n_features
        self.n_components = n_components
        self.decode = torch.Tensor(
            sparse_random_matrix(self.n_components,
                                 self.n_features,
                                 random_state=random_state).todense()).float()

    def forward(self, x):
        z = torch.einsum("bi,ji->bj", x.reshape(x.shape[0], 784), self.decode)
        return None, None, None, z  # Mimic VAE fwd return


class VAE(nn.Module):
    """A MINST-shaped VAE."""
    def __init__(self, z_features=20):
        super().__init__()
        self.z_features = z_features
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, self.z_features)
        self.fc22 = nn.Linear(400, self.z_features)
        self.fc3 = nn.Linear(self.z_features, 400)
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
    def __init__(self, z_features=20, activation_function='Softmax'):
        super().__init__()
        self.z_features = z_features

        # Lookup activation function (a class)
        AF = getattr(nn, activation_function)
        self.AF = AF(dim=1)
        self.fc1 = nn.Linear(self.z_features, self.z_features)
        self.fc2 = nn.Linear(self.z_features, 10)

    def forward(self, x):
        x = self.AF(self.fc1(x))
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
        if z_features < 12:
            raise ValueError("z_features must be >= 12.")

        self.z_features = z_features

        # Lookup activation function (a class)
        AF = getattr(nn, activation_function)

        # --------------------------------------------------------------------
        # Def fc1:
        glia1 = []
        for s in reversed(range(12, self.z_features + 2, 2)):
            glia1.append(gn.Gather(s))
            glia1.append(gn.Slide(s - 2))
            # Linear on the last output, for digit decode
            if s > 12:
                glia1.append(AF(dim=1))
        self.glia1 = nn.Sequential(*glia1)

    def forward(self, x):
        x = self.glia1(x)

        return F.log_softmax(x, dim=1)


class PerceptronLeak(nn.Module):
    """A minst digit perceptron, with blured connections.
    """
    def __init__(self, z_features=20, activation_function='Softmax', sigma=1):
        # --------------------------------------------------------------------
        # Init
        super().__init__()
        if z_features < 12:
            raise ValueError("z_features must be >= 12.")

        self.z_features = z_features
        self.sigma = sigma

        # Lookup activation function (a class)
        AF = getattr(nn, activation_function)

        # --------------------------------------------------------------------
        # Def fc1:
        glia1 = []
        for s in reversed(range(12, self.z_features + 2, 2)):
            glia1.append(gn.Gather(s))
            glia1.append(gn.Leak(s - 2, sigma=sigma))
            glia1.append(gn.Slide(s - 2))
            glia1.append(gn.Leak(s - 2, sigma=sigma))

            # Linear on the last output, for digit decode
            if s > 12:
                glia1.append(AF(dim=1))
        self.glia1 = nn.Sequential(*glia1)

    def forward(self, x):
        x = self.glia1(x)

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

    return train_loss


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
                save_image(comparison.cpu(),
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
            print(">>> Train example output[:5]: {}".format(pred[:5,
                                                                 0].tolist()))
            print(
                '>>> Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

    if progress or debug:
        print('>>> Train loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
            loss, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))

    return loss


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


def run_VAE_only(batch_size=128,
                 test_batch_size=128,
                 num_epochs=500,
                 lr_vae=1e-3,
                 z_features=20,
                 use_gpu=False,
                 device_num=None,
                 seed_value=1,
                 save=None,
                 log_interval=50,
                 progress=False,
                 debug=False,
                 data_path=None):
    """Train (only) a VAE."""

    # ------------------------------------------------------------------------
    # Training settings
    torch.manual_seed(seed_value)
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if device_num is not None:
            torch.cuda.set_device(device_num)

    if data_path is None:
        data_path = "data"

    z_features = int(z_features)

    # ------------------------------------------------------------------------
    # Get and pre-process data
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
        data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    ),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
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
    model_vae = VAE(z_features=z_features).to(device)
    optimizer_vae = optim.Adam(model_vae.parameters(), lr=lr_vae)

    if debug:
        print(f">>> z_features: {z_features}")
        print(model_vae)

    # Learn classes
    train_loss = []
    test_loss = []
    for epoch in range(1, num_epochs + 1):
        # Learn z?
        loss = train_vae(model_vae,
                         device,
                         train_loader,
                         optimizer_vae,
                         epoch,
                         log_interval=log_interval,
                         debug=debug,
                         progress=progress)
        # Log
        train_loss.append(deepcopy(float(loss)))

        test_loss = test_vae(model_vae,
                             device,
                             test_loader,
                             epoch,
                             test_batch_size,
                             debug=debug,
                             progress=progress,
                             data_path=data_path)

        # Log
        test_loss.append(deepcopy(float(loss)))

    print(">>> Training complete")
    print(">>> VAE loss: {:.5f}".format(test_loss))

    state = dict(vae_dict=model_vae.cpu().state_dict(),
                 batch_size=batch_size,
                 test_batch_size=test_batch_size,
                 num_epochs=num_epochs,
                 lr_vae=lr_vae,
                 use_gpu=use_gpu,
                 device_num=device_num,
                 train_loss=train_loss,
                 test_loss=test_loss,
                 seed_value=seed_value)

    if save is not None:
        torch.save(state, save + ".pytorch")
    else:
        return state


def run_VAE(glia=False,
            sigma=0,
            batch_size=128,
            test_batch_size=128,
            num_epochs=10,
            lr=0.005,
            vae_path=None,
            lr_vae=1e-3,
            z_features=20,
            activation_function='Softmax',
            use_gpu=False,
            device_num=None,
            seed_value=1,
            save=None,
            log_interval=50,
            progress=False,
            debug=False,
            data_path=None):
    """Glia learn to see (clothes)"""
    # ------------------------------------------------------------------------
    # Training settings
    if seed_value is not None:
        torch.manual_seed(seed_value)

    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if device_num is not None:
            torch.cuda.set_device(device_num)

    if data_path is None:
        data_path = "data"

    z_features = int(z_features)

    # ------------------------------------------------------------------------
    # Get and pre-process data
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
        data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    ),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
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
    if vae_path is None:
        model_vae = VAE(z_features=z_features).to(device)
        optimizer_vae = optim.Adam(model_vae.parameters(), lr=lr_vae)
    else:
        saved = torch.load(vae_path)
        model_vae = VAE(z_features=z_features)
        model_vae.load_state_dict(saved["vae_dict"])
        model_vae.eval()
        model_vae.to(device)

        if debug:
            print(saved["vae_dict"])
            print(f">>> Loaded VAE from {vae_path}")

    if glia:
        if sigma > 0:
            model = PerceptronLeak(z_features=z_features,
                                   activation_function=activation_function,
                                   sigma=sigma)
        else:
            model = PerceptronGlia(
                z_features=z_features,
                activation_function=activation_function).to(device)
    else:
        model = PerceptronNet(
            z_features=z_features,
            activation_function=activation_function).to(device)

    # -
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if debug:
        print(f">>> z_features: {z_features}")
        print(model_vae)
        print(model)

    # Learn classes
    vae_loss = []  # log losses
    train_loss = []
    test_loss = []
    test_correct = []
    for epoch in range(1, num_epochs + 1):
        # Learn z?
        if vae_path is None:
            train_vae(model_vae,
                      device,
                      train_loader,
                      optimizer_vae,
                      epoch,
                      log_interval=log_interval,
                      debug=debug,
                      progress=progress)

            loss = test_vae(model_vae,
                            device,
                            test_loader,
                            epoch,
                            test_batch_size,
                            debug=debug,
                            progress=progress,
                            data_path=data_path)
            # Log
            vae_loss.append(deepcopy(float(loss)))

        # Glia learn
        loss = train(model,
                     model_vae,
                     device,
                     train_loader,
                     optimizer,
                     epoch,
                     log_interval=log_interval,
                     progress=progress,
                     debug=debug)
        # Log
        train_loss.append(deepcopy(float(loss)))

        loss, correct = test(model,
                             model_vae,
                             device,
                             test_loader,
                             debug=debug,
                             progress=progress)
        # Log
        test_loss.append(deepcopy(float(loss)))
        test_correct.append(deepcopy(float(correct)))

    print(">>> Training complete")
    print(">>> Loss: {:.5f}, Correct: {:.2f}".format(test_loss[-1],
                                                     100 * correct))

    state = dict(model_dict=model.cpu().state_dict(),
                 vae_dict=model_vae.cpu().state_dict(),
                 glia=glia,
                 batch_size=batch_size,
                 test_batch_size=test_batch_size,
                 num_epochs=num_epochs,
                 lr=lr,
                 vae_path=vae_path,
                 lr_vae=lr_vae,
                 use_gpu=use_gpu,
                 device_num=device_num,
                 vae_loss=vae_loss,
                 train_loss=train_loss,
                 test_loss=test_loss,
                 test_correct=test_correct,
                 correct=correct,
                 seed_value=seed_value)

    if save is not None:
        torch.save(state, save + ".pytorch")
    else:
        return state


def run_RP(glia=False,
           batch_size=128,
           test_batch_size=128,
           num_epochs=10,
           random_projection='SP',
           lr=0.005,
           z_features=20,
           activation_function='Softmax',
           use_gpu=False,
           device_num=None,
           seed_value=1,
           save=None,
           log_interval=50,
           progress=False,
           debug=False,
           data_path=None):
    """Glia learn to see (clothes)"""
    # ------------------------------------------------------------------------
    # Training settings
    prng = np.random.RandomState(seed_value)

    if seed_value is not None:
        torch.manual_seed(seed_value)

    # -
    device = torch.device("cuda" if use_gpu else "cpu")
    if use_gpu:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if device_num is not None:
            torch.cuda.set_device(device_num)

    if data_path is None:
        data_path = "data"

    z_features = int(z_features)

    # ------------------------------------------------------------------------
    # Get and pre-process data
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
    train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
        data_path,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    ),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               **kwargs)
    test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST(
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
    if random_projection == 'SP':
        # Perceptrons assume 20; might not be ideal
        model_rp = SP(n_features=784,
                      n_components=z_features,
                      random_state=prng)
    elif random_projection == 'GP':
        model_rp = GP(n_features=784,
                      n_components=z_features,
                      random_state=prng)
    else:
        raise ValueError("random_projection must be GP or SP")

    if glia:
        model = PerceptronGlia(
            z_features=z_features,
            activation_function=activation_function).to(device)
    else:
        model = PerceptronNet(
            z_features=z_features,
            activation_function=activation_function).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Learn classes
    train_loss = []
    test_loss = []
    test_correct = []
    for epoch in range(1, num_epochs + 1):

        # Glia learn
        loss = train(model,
                     model_rp,
                     device,
                     train_loader,
                     optimizer,
                     epoch,
                     log_interval=log_interval,
                     progress=progress,
                     debug=debug)
        # Log
        train_loss.append(deepcopy(float(loss)))

        loss, correct = test(model,
                             model_rp,
                             device,
                             test_loader,
                             debug=debug,
                             progress=progress)

        # Log
        test_loss.append(deepcopy(float(loss)))
        test_correct.append(deepcopy(float(correct)))

    print(">>> Training complete")
    print(">>> Loss: {:.5f}, Digit correct: {:.2f}".format(
        test_loss[-1], 100 * correct))

    state = dict(model_dict=model.cpu().state_dict(),
                 model_rp=random_projection,
                 glia=glia,
                 batch_size=batch_size,
                 test_batch_size=test_batch_size,
                 num_epochs=num_epochs,
                 lr=lr,
                 use_gpu=use_gpu,
                 device_num=device_num,
                 train_loss=train_loss,
                 test_loss=test_loss,
                 test_correct=test_correct,
                 correct=correct,
                 seed=seed_value)

    if save is not None:
        torch.save(state, save + ".pytorch")
    else:
        return state


# ----------------------------------------------------------------------------
if __name__ == '__main__':
    fire.Fire({'VAE': run_VAE, 'VAE_only': run_VAE_only, 'RP': run_RP})
