"""Artificial Glia Netorks"""
import math
import numbers
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn import Dropout
from torch.distributions.bernoulli import Bernoulli
from torch import nn
from kornia.filters import GaussianBlur2d


class WeightNoise(Module):
    def __init__(self, sigma=0.1):
        if sigma < 0:
            raise ValueError("sigma must be positive.")
        self.sigma = sigma

    def forward(self, m):
        """Add Normal noise to a layer's wieghts.

        NOTE: this class leads to in-place perturbations of the layers
              in a model! Copy your model first?

        Params
        ------
        sigma : float, positive
            Noise level, sampled from N(0, sigma)

        Usage
        -----
        Use an instance of this class as a functional 
        with model.apply(). For example,
            `noise = WeightNoise(1)`
            `model.apply(noise)`
        """

        with torch.no_grad():
            if hasattr(m, 'weight'):
                m.weight.add_(torch.randn(m.weight.size()) * self.sigma)


class WeightLoss(Module):
    def __init__(self, p):
        if p < 0:
            raise ValueError("p must be positive")
        if p > 1:
            raise ValueError("p must be less than one")
        self.p
        self.bernoulli = Bernoulli(self.p).sample  # a functional def

    def forward(self, m):
        """Zero a layer's wieghts using the Bernoulli dist.

        NOTE: this class leads to in-place perturbations of the layers
              in a model! Copy your model first?

        Params
        ------
        p : float, (0,1)
            Prob on individual wieght loss

        Usage
        -----
        Use an instance of this class as a functional 
        with model.apply(). For example,
            `wloss = WeightLoss(p=0.1)`
            `model.apply(wloss)`
        """
        with torch.no_grad():
            m.weight.mult_(self.bernoulli(m.weight.size()))


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


# TODO figure out how to make GaussianBlur play well with z
class Leak(nn.Module):
    """Model transmitter leak with Guassian blur"""

    def __init__(self, in_features, sigma=1, kernel_size=3):

        super().__init__()
        self.in_features = in_features
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.GaussianBlur2d = GaussianBlur2d(
            sigma=(self.sigma, self.sigma),
            kernel_size=(self.kernel_size, self.kernel_size),
            border_type='constant')

    def forward(self, input):
        # The only grad-compatible fn I can find in the ecosystem
        # expects 2d images. So we re-format input to look 2d.
        # Apply the blur/NT leak. Then format back again.
        # Expected format: BxCxHxW
        x = input.float().view(-1, 1, self.in_features, 1)
        output = self.GaussianBlur2d(x)
        return output.view(*input.shape)
