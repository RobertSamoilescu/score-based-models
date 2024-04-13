""" 
Copyright zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init


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

    def forward(self, x: Tensor) -> Tensor:
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


class CondAttnBlock(nn.Module):
    """Conditional Attention block."""

    def __init__(self, in_ch: int, cond_ch: int) -> None:
        """Constructor of the Attention block.

        :param in_ch: Number of input channels
        """
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Linear(cond_ch, in_ch)
        self.proj_v = nn.Linear(cond_ch, in_ch)

        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self):
        """Initialize the weights of the Attention block."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)
        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Attention block.

        :param x: Input tensor
        :param y: Condition tensor
        :return: Output tensor
        """
        B, C, H, W = x.shape
        _, M, D = y.shape

        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(y)
        v = self.proj_v(y)

        # shape: B, H*W, C
        q = q.permute(0, 2, 3, 1).view(B, H * W, C)  # B, H*W, C
        # shape: B, C, M
        k = k.permute(0, 2, 1)  # B, C, M
        # shape: (B, H*W, C) x (B, C, M) -> (B, H*W, M)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, M]
        w = F.softmax(w, dim=-1)

        # shape: (B, H*W, M) x (B, M, C) -> (B, H*W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)
        return x + h
