"""Artificial Glia Netorks"""
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module


class Base(Module):
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


# class GliaGrow(Base):
#     def __init__(self, in_features, bias=True):
#         self.step = 2  # Actual growth
#         self.pad = 2  # Projection dummy
#         out_features = in_features + self.step + (2 * self.pad)

#         super().__init__(in_features, out_features, bias=bias)

#         # Init params
#         self.weight = Parameter(
#             torch.Tensor(self.out_features, self.in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(self.out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()

#     def forward(self, input):
#         # Make batch compatible
#         output = torch.zeros(input.shape[0], self.out_features)

#         # Nearest-neighbor linear transform
#         i, j = 0, 3
#         for n in range(self.in_features):
#             if self.bias is not None:
#                 output[:, i:j] += F.linear(
#                     input[:, i:j], self.weight[n, i:j].unsqueeze(1).t(),
#                     self.bias[i:j])
#             else:
#                 output[:, i:j] += F.linear(
#                     input[:, i:j], self.weight[n, i:j].unsqueeze(1).t(),
#                     self.bias)

#             # Update index
#             i += 1
#             j += 1

#         output = output[:, self.pad:-self.pad]  # Strip pad

#         return output


class GliaGrow(Base):
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
        for n in range(self.in_features):
            # Calc update
            update = input[:, n].unsqueeze(1) * self.weight[i:j, n].unsqueeze(
                1).t()

            # And apply it
            output[:, i:j] += update

            if self.bias is not None:
                output[:, i:j] += self.bias[i:j]

            # Update index
            i += 1
            j += 1

        return output


class GliaShrink(Base):
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
        # Make batch compatible
        batch_size = input.shape[0]
        output = torch.zeros(batch_size,
                             self.out_features)  #, requires_grad=True)

        # Nearest-neighbor linear transform

        # Set initial index, handling the the out_features=1
        # edge case
        i, j = 0, min(3, self.out_features + 1)

        # Iter over input, calculating running sums on the output
        # with the i,j index
        for n in range(max(self.in_features - 2, 1)):
            # Calc update
            update = input[:, n].unsqueeze(1) * self.weight[i:j, n].unsqueeze(
                1).t()

            # And apply it
            output[:, i:j] += update

            if self.bias is not None:
                output[:, i:j] += self.bias[i:j]

            # Update index
            i += 1
            j += 1

        return output
