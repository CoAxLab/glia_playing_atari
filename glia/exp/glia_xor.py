#!/usr/bin/env python
"""Glia learn to do logic. 

Code modified from: https://gist.githubusercontent.com/RichardKelley/17ef5f2291c273de11540c33dc1bfbf2/raw/9a9049a24a61b4db29b2ed0b040b609d9ab1b007/pytorch_xor.py. 
"""
import fire

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

from glia import gn


class XorGlia(nn.Module):
    def __init__(self):
        super().__init__()
        # self.fc1 = nn.Sequential(
        #     gn.Spread(2, bias=False), torch.nn.Softmax(),
        #     gn.Spread(4, bias=False), torch.nn.Softmax(),
        #     gn.Spread(6, bias=False), torch.nn.Softmax(),
        #     gn.Spread(8, bias=False), torch.nn.Softmax(),
        #     gn.Slide(10, bias=False), torch.nn.Softmax(),
        #     gn.Slide(10, bias=False), torch.nn.Softmax(),
        #     gn.Gather(10, bias=False), torch.nn.Softmax(),
        #     gn.Gather(8, bias=False), torch.nn.Softmax(),
        #     gn.Gather(6, bias=False), torch.nn.Softmax(),
        #     gn.Gather(4, bias=False), torch.nn.Softmax(),
        #     gn.Gather(2, bias=False))
        self.fc1 = nn.Sequential(
            gn.Spread(2, bias=False), torch.nn.Softmax(),
            gn.Slide(4, bias=False), torch.nn.Softmax(),
            gn.Spread(4, bias=False), torch.nn.Softmax(),
            gn.Slide(6, bias=False), torch.nn.Softmax(),
            gn.Spread(6, bias=False), torch.nn.Softmax(),
            gn.Slide(8, bias=False), torch.nn.Softmax(),
            gn.Spread(8, bias=False), torch.nn.Softmax(),
            gn.Slide(10, bias=False), torch.nn.Softmax(),
            gn.Gather(10, bias=False), torch.nn.Softmax(),
            gn.Gather(8, bias=False), torch.nn.Softmax(),
            gn.Gather(6, bias=False), torch.nn.Softmax(),
            gn.Gather(4, bias=False), torch.nn.Softmax(),
            gn.Gather(2, bias=False))

    def forward(self, x):
        x = self.fc1(x)

        return x


class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.softmax(self.fc1(x))
        x = self.fc2(x)
        return x


def main(glia=True, epochs=3000, log_interval=50, debug=False):
    """Glia learns logic"""

    if glia:
        m = XorGlia()
    else:
        m = XorNet()

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(m.parameters(), lr=1e-3)

    # input-output pairs
    pairs = [(np.asarray([0.0, 0.0]), [0.0]), (np.asarray([0.0, 1.0]), [1.0]),
             (np.asarray([1.0, 0.0]), [1.0]), (np.asarray([1.0, 1.0]), [0.0])]

    state_matrix = np.vstack([x[0] for x in pairs])
    label_matrix = np.vstack([x[1] for x in pairs])

    for i in range(epochs):
        for batch_ind in range(4):
            # wrap the data in variables
            minibatch_state_var = torch.Tensor(state_matrix)
            minibatch_label_var = torch.Tensor(label_matrix)

            # forward pass
            y_pred = m(minibatch_state_var)

            # compute and print loss
            loss = loss_fn(y_pred, minibatch_label_var)
            if (i % log_interval == 0) and debug:
                print(">>> f(0,0) = {:.2f}".format(
                    float(m(Variable(torch.Tensor([0.0, 0.0]).unsqueeze(0))))))
                print(">>> f(0,1) = {:.2f}".format(
                    float(m(Variable(torch.Tensor([0.0, 1.0]).unsqueeze(0))))))
                print(">>> f(1,0) = {:.2f}".format(
                    float(m(Variable(torch.Tensor([1.0, 0.0]).unsqueeze(0))))))
                print(">>> f(1,1) = {:.2f}".format(
                    float(m(Variable(torch.Tensor([1.0, 1.0]).unsqueeze(0))))))

                print(">>> i{}, batch {}, loss {}".format(
                    i, batch_ind, loss.data[0]))

            # reset gradients
            optimizer.zero_grad()

            # backwards pass
            loss.backward()

            # step the optimizer - update the weights
            optimizer.step()

    print(">>> After training:")
    print(">>> f(0,0) = {:.2f}".format(
        float(m(Variable(torch.Tensor([0.0, 0.0]).unsqueeze(0))))))
    print(">>> f(0,1) = {:.2f}".format(
        float(m(Variable(torch.Tensor([0.0, 1.0]).unsqueeze(0))))))
    print(">>> f(1,0) = {:.2f}".format(
        float(m(Variable(torch.Tensor([1.0, 0.0]).unsqueeze(0))))))
    print(">>> f(1,1) = {:.2f}".format(
        float(m(Variable(torch.Tensor([1.0, 1.0]).unsqueeze(0))))))


if __name__ == "__main__":
    fire.Fire(main)
