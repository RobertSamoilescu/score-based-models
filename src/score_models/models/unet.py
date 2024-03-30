""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class ConditionalBatchNorm2d(nn.Module):
    """Conditional Batch Normalization 2d Layer."""

    def __init__(self, L: int, num_features: int) -> None:
        """Constructor of the Conditional Batch Normalization 1d Layer.

        :param L: Number of batch normalization layers
        :param num_features: Number of features in the input tensor
        """
        super().__init__()
        self.L = L
        self.num_features = num_features
        self.bn = nn.ModuleList([nn.BatchNorm2d(self.num_features) for _ in range(self.L)])

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """Forward pass of the Conditional Batch Normalization 1d Layer.
        Passes the input tensor through the i-th batch normalization layer.

        :return: Output tensor
        """
        return self.bn[i](x)


class DoubleConv(nn.Module):
    """ Double convolution block. """

    def __init__(self, L: int, in_channels: int, out_channels: int, mid_channels: Optional[int] = None) -> None:
        """ Constructor of the Double Convolution block.
        
        :param L: Number of batch normalization layers
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param mid_channels: Number of channels in the intermediate layer
        """
        super().__init__()

        if not mid_channels:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = ConditionalBatchNorm2d(L, mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = ConditionalBatchNorm2d(L, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor, i: int) -> Tensor:
        x = self.relu1(self.bn1(self.conv1(x), i))
        return self.relu2(self.bn2(self.conv2(x), i))


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, L: int, in_channels: int, out_channels: int) -> None:
        """ Constructor of the Downscaling block.
        
        :param L: Number of batch normalization layers
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        """
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = DoubleConv(L, in_channels, out_channels)

    def forward(self, x: Tensor, i: int) -> Tensor:
        return self.maxpool(self.conv(x, i))


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, L: int, in_channels: int, out_channels: int, bilinear: bool =True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(L, in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(L, in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor, i: int) -> Tensor:
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, i)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, L: int, n_channels: int, n_classes: int, bilinear: bool = False) -> None:
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(L, n_channels, 64)
        self.down1 = Down(L, 64, 128)
        self.down2 = Down(L, 128, 256)
        self.down3 = Down(L, 256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(L, 512, 1024 // factor)
        self.up1 = Up(L, 1024, 512 // factor, bilinear)
        self.up2 = Up(L, 512, 256 // factor, bilinear)
        self.up3 = Up(L, 256, 128 // factor, bilinear)
        self.up4 = Up(L, 128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: Tensor, i: int) -> Tensor:
        x1 = self.inc(x, i)
        x2 = self.down1(x1, i)
        x3 = self.down2(x2, i)
        x4 = self.down3(x3, i)
        x5 = self.down4(x4, i)
        x = self.up1(x5, x4, i)
        x = self.up2(x, x3, i)
        x = self.up3(x, x2, i)
        x = self.up4(x, x1, i)
        return self.outc(x)
