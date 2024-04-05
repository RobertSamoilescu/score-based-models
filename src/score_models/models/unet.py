""" 
Copyright zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
"""

import math
from typing import List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Swish activation function.

        :param x: Input tensor
        :return: Output tensor
        """
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """Time embedding module."""

    def __init__(self, T: int, d_model: int, dim: int) -> None:
        """Constructor of the TimeEmbedding module.

        :param T: Number of time embeddings
        :param d_model: Number of features in the input tensor
        :param dim: Number of features in the output tensor
        """
        assert d_model % 2 == 0

        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]

        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)

        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        """Initialize the weights of the TimeEmbedding module."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TimeEmbedding module.

        :param t: Time tensor
        :return: Output tensor
        """
        return self.timembedding(t)


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

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DownSample block.

        :param x: Input tensor
        :param temb: Time embedding tensor
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

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the UpSample block.

        :param x: Input tensor
        :param temb: Time embedding tensor
        :return: Output tensor
        """
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.main(x)


class AttnBlock(nn.Module):
    """Attention block."""

    def __init__(self, in_ch: int) -> None:
        """Constructor of the Attention block.

        :param in_ch: Number of input channels
        """
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        """Initialize the weights of the Attention block."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention block.

        :param x: Input tensor
        :return: Output tensor
        """
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    """Residual block."""

    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float, attn: bool = False) -> None:
        """Constructor of the ResBlock.

        :param in_ch: Number of input channels
        :param out_ch: Number of output channels
        :param tdim: Number of features in the time embedding tensor
        :param dropout: Dropout rate
        :param attn: Whether to use attention
        """
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()  # type: ignore[assignment]

        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()  # type: ignore[assignment]

        self.initialize()

    def initialize(self):
        """Initialize the weights of the ResBlock."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)
        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResBlock.

        :param x: Input tensor
        :param temb: Time embedding tensor
        :return: Output tensor
        """
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        return self.attn(h)


class UNet(nn.Module):
    """UNet model."""

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
