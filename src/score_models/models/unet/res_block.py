import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

from score_models.models.unet.attention import AttnBlock, CondAttnBlock
from score_models.models.unet.common import Identity, Swish


class ResBlockBase(nn.Module):
    """Residual block base class."""

    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float) -> None:
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
            self.shortcut = Identity()  # type: ignore[assignment]

    def initialize(self):
        """Initialize the weights of the ResBlock."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

        init.xavier_uniform_(self.block2[-1].weight, gain=1e-5)


class ResBlock(ResBlockBase):
    """Residual block."""

    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float, attn: bool) -> None:
        """Constructor of the ResBlock.

        :param in_ch, out_ch, tdim, dropout: See `ResBlockBase`
        :param attn: Whether to use attention
        """
        super().__init__(in_ch=in_ch, out_ch=out_ch, tdim=tdim, dropout=dropout)
        self.attn = AttnBlock(out_ch) if attn else Identity()  # type: ignore[assignment]
        self.initialize()

    def forward(self, x: Tensor, temb: Tensor) -> Tensor:
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


class CondResBlock(ResBlockBase):
    """Conditional residual block."""

    def __init__(self, in_ch: int, out_ch: int, tdim: int, dropout: float, attn: bool, cond_ch: int) -> None:
        """Constructor of the ResBlock.

        :param in_ch, out_ch, tdim, dropout, attn: See `ResBlockBase`
        :param cond_ch: Number of channels in the conditional tensor
        """
        super().__init__(in_ch=in_ch, out_ch=out_ch, tdim=tdim, dropout=dropout)
        self.attn = CondAttnBlock(out_ch, cond_ch) if attn else Identity()  # type: ignore[assignment]
        self.initialize()

    def forward(self, x: torch.Tensor, y: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        """Forward pass of the ResBlock.

        :param x: Input tensor
        :param y: Conditional tensor
        :param temb: Time embedding tensor
        :return: Output tensor
        """
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)
        h = h + self.shortcut(x)
        return self.attn(h, y)
