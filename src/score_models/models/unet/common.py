""" 
Copyright zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


class Identity(nn.Module):
    """Identity module."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return input


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass of the Swish activation function.

        :param x: Input tensor
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: Output tensor
        """
        return x * torch.sigmoid(x)


class DownSample(nn.Module):
    """Downsample block."""

    def __init__(self, in_ch: int) -> None:
        """Constructor of the DownSample block.

        :param in_ch: Number of input channels
        """
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self):
        """Initialize the weights of the DownSample block."""
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass of the DownSample block.

        :param x: Input tensor
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: Output tensor
        """
        return self.main(x)


class UpSample(nn.Module):
    """Upsample block."""

    def __init__(self, in_ch: int) -> None:
        """Constructor of the UpSample block.

        :param in_ch: Number of input channels
        """
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self):
        """Initialize the weights of the UpSample block."""
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass of the UpSample block.

        :param x: Input tensor
        :param args: Additional arguments
        :param kwargs: Additional keyword arguments
        :return: Output tensor
        """
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.main(x)
