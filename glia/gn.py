"""Artificial Glia Netorks"""
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module


class Base(Module):
    """Base Glia class. DO NOT USE DIRECTLY."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)

    def forward(self, input):
        pass


class Slide(Base):
    """Nearest neighbor Ca++ signal propagation.
    
    If the input size is n, the output size is n.
    
    Params
    ------
    in_features : int
        Number of input features.
    bias : bool
        Add a bias (leave as False).
    """

    def __init__(self, in_features, bias=True):
        # Init out
        out_features = in_features
        super().__init__(in_features, out_features, bias=bias)

        # Init params
        self.weight = Parameter(
            torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        # print(f"Start slide: ({self.in_features},{self.out_features})")

        # Make batch compatible
        batch_size = input.shape[0]
        output = torch.zeros(batch_size,
                             self.out_features)  #, requires_grad=True)

        # Nearest-neighbor linear transform
        # i, j = 0, 3
        i = self.out_features - 1
        j = (i + 1) % self.out_features
        k = (j + 1) % self.out_features

        # Calculating running sums on the output
        for n in range(self.in_features):
            # Create mask
            mask = torch.zeros(self.weight.size()).detach()
            mask[[i, j, k], n] = 1
            output = output + torch.einsum('bi,ji -> bj', input,
                                           mask * self.weight)

            # Add bias?
            if self.bias is not None:
                mask = torch.zeros(self.bias.size()).detach()
                mask[[i, j, k]] = 1
                output = output + torch.einsum('bj,j->bj', output,
                                               mask * self.bias)

            # Update index
            i = (i + 1) % self.out_features
            j = (j + 1) % self.out_features
            k = (k + 1) % self.out_features

        # print("End layer")
        return output


class Spread(Base):
    """Nearest neighbor Ca++ signal propagation.
    
    If the input size is n, the output size is n + 2.
    
    Params
    ------
    in_features : int
        Number of input features.
    bias : bool
        Add a bias (leave as False).
    """

    def __init__(self, in_features, bias=True):
        # Init out
        out_features = in_features + 2
        super().__init__(in_features, out_features, bias=bias)

        # Init params
        self.weight = Parameter(
            torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        # Make batch compatible
        batch_size = input.shape[0]
        output = torch.zeros(batch_size,
                             self.out_features)  #, requires_grad=True)

        # Nearest-neighbor linear transform
        i, j = 0, 3

        # Calculating running sums on the output
        for n in range(self.in_features):
            # Create mask
            mask = torch.zeros(self.weight.size()).detach()
            mask[i:j, n] = 1
            output = output + torch.einsum('bi,ji -> bj', input,
                                           mask * self.weight)

            # Add bias?
            if self.bias is not None:
                mask = torch.zeros(self.bias.size()).detach()
                mask[i:j] = 1
                output = output + torch.einsum('bj,j->bj', output,
                                               mask * self.bias)

            # # Calc update
            # update = input[:, n].unsqueeze(1) * self.weight[i:j, n].unsqueeze(
            #     1).t()

            # # And apply it
            # output[:, i:j] += update
            # if self.bias is not None:
            #     output[:, i:j] += self.bias[i:j]

            # Update index
            i += 1
            j += 1

        return output


class Gather(Base):
    """Nearest neighbor Ca++ signal contraction.
    
    If the input size is n, the output size is max(n - 2, 1). 
    
    Params
    ------
    in_features : int
        Number of input features.
    bias : bool
        Add a bias (leave as False).
    """

    def __init__(self, in_features, bias=True):
        # Init out
        out_features = max(in_features - 2, 1)
        super().__init__(in_features, out_features, bias=bias)

        # Init params
        self.weight = Parameter(
            torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, input):
        # print(f"Start gather: ({self.in_features},{self.out_features})")

        # Make batch compatible
        batch_size = input.shape[0]
        output = torch.zeros(batch_size,
                             self.out_features)  #, requires_grad=True)

        # Nearest-neighbor linear transform

        # Handling the the out_features=1 edge case
        # i, j = 0, min(3, self.out_features + 1)
        i = self.out_features - 1
        j = (i + 1) % self.out_features
        k = (j + 1) % self.out_features

        # Calculating running sums on the output
        for n in range(max(self.in_features - 2, 1)):
            # print((i, j, k))

            # Create mask
            mask = torch.zeros(self.weight.size()).detach()
            mask[[i, j, k], n] = 1
            output = output + torch.einsum('bi,ji->bj', input,
                                           mask * self.weight)

            # Add bias?
            if self.bias is not None:
                mask = torch.zeros(self.bias.size()).detach()
                mask[[i, j, k]] = 1
                output = output + torch.einsum('bj,j->bj', output,
                                               mask * self.bias)

            # # Calc update
            # update = input[:, n].unsqueeze(
            #     1) * self.weight[[i, j, k], n].unsqueeze(1).t()

            # # and apply it
            # output[:, [i, j, k]] += update
            # if self.bias is not None:
            #     output[:, [i, j, k]] += self.bias[[i, j, k]]

            # Update index
            i = (i + 1) % self.out_features
            j = (j + 1) % self.out_features
            k = (k + 1) % self.out_features

        # print("End layer")
        return output
