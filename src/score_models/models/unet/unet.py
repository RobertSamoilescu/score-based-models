""" 
Copyright zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
"""

import abc
from functools import partial
from typing import List

import torch
from torch import nn
from torch.nn import init

from score_models.models.unet.common import DownSample, Swish, UpSample
from score_models.models.unet.res_block import CondResBlock, ResBlock
from score_models.models.unet.time_embeddings import TimeEmbedding


class UNetBase(nn.Module):
    """UNet base class."""

    def __init__(
        self, T: int, in_ch: int, ch: int, ch_mult: List[int], attn: List[int], num_res_blocks: int, dropout: bool
    ) -> None:
        """Constructor of the UNet model.

        :param T: Number of time embeddings
        :param in_ch: Number of input channels
        :param ch: Number of channels
        :param ch_mult: Multiplier for the number of channels
        :param attn: List of indices of the layers where attention is used
        :param num_res_blocks: Number of residual blocks
        :param dropout: Dropout rate
        """
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        ResBlock = self.get_resblock_class()

        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(
                    ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn))
                )
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList(
            [
                ResBlock(now_ch, now_ch, tdim, dropout, attn=True),
                ResBlock(now_ch, now_ch, tdim, dropout, attn=False),
            ]
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult

            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, attn=(i in attn))
                )
                now_ch = out_ch

            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0
        self.tail = nn.Sequential(nn.GroupNorm(32, now_ch), Swish(), nn.Conv2d(now_ch, in_ch, 3, stride=1, padding=1))
        self.initialize()

    def initialize(self):
        """Initialize the weights of the UNet model."""
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)
        init.xavier_uniform_(self.tail[-1].weight, gain=1e-5)
        init.zeros_(self.tail[-1].bias)

    @abc.abstractmethod
    def get_resblock_class(self):
        """Get the ResBlock class."""
        pass


class UNet(UNetBase):
    """UNet model."""

    def get_resblock_class(self):
        return ResBlock

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet

        :param x: Input tensor
        :param t: Time tensor
        :return: Output tensor
        """
        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, temb)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, temb)

        h = self.tail(h)
        assert len(hs) == 0
        return h


class CondUNet(UNetBase):
    """Conditional UNet model."""

    def __init__(
        self,
        T: int,
        in_ch: int,
        ch: int,
        ch_mult: List[int],
        cond_ch: int,
        attn: List[int],
        num_res_blocks: int,
        dropout: bool,
    ) -> None:
        """Constructor of the Conditional UNet model.

        :param T, in_ch, ch, ch_mult, attn, num_res_blocks, dropout: See UNetBase
        :param cond_ch: Number of conditional channels
        """
        self.cond_ch = cond_ch
        super().__init__(T, in_ch, ch, ch_mult, attn, num_res_blocks, dropout)

    def get_resblock_class(self):
        return partial(CondResBlock, cond_ch=self.cond_ch)

    def forward(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UNet

        :param x: Input tensor
        :param t: Time tensor
        :return: Output tensor
        """
        # Timestep embedding
        temb = self.time_embedding(t)

        # Downsampling
        h = self.head(x)
        hs = [h]

        for layer in self.downblocks:
            h = layer(h, y, temb)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, y, temb)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, CondResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, y, temb)

        h = self.tail(h)
        assert len(hs) == 0
        return h
