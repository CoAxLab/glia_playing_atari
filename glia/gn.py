"""Artificial Glia Netorks"""
import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module
from torch import nn


class Base(Module):
    """Base Glia class. DO NOT USE DIRECTLY.q"""

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

            # Update index
            i = (i + 1) % self.out_features
            j = (j + 1) % self.out_features
            k = (k + 1) % self.out_features

        # print("End layer")
        return output


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d, 2d or 3d tensor. 
    
    Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    
    Params
    ------
        channels (int, sequence): Number of channels of the input tensors. Output will have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 1 (spatial).
    
    Note
    ----
        This code is based on:
        https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
    """

    def __init__(self, channels, kernel_size, sigma, dim=1):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                f'Only 1, 2 and 3 dimensions are supported. Received {dim}.')

    def forward(self, input):
        """Apply gaussian filter to input."""
        return self.conv(input, weight=self.weight, groups=self.groups)
