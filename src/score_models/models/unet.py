from typing import List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init


class Swish(nn.Module):
    """Swish activation function."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class DownSample(nn.Module):
    """Downsample block."""

    def __init__(self, in_ch: int) -> None:
        """Initialize the DownSample block.

        :param in_ch: Number of input channels.
        """
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)
        self.initialize()

    def initialize(self) -> None:
        """Initialize the weights of the DownSample block."""
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)  # type: ignore[arg-type]

    def forward(self, x: Tensor, i: int):
        """Forward pass of the DownSample block.

        :param x: Input tensor.
        :param i: Conditional index. Not used.
        :return: Output tensor.
        """
        return self.main(x)


class UpSample(nn.Module):
    """Upsample block."""

    def __init__(self, in_ch: int) -> None:
        """Initialize the UpSample block.

        :param in_ch: Number of input channels.
        """
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self) -> None:
        """Initialize the weights of the UpSample block."""
        init.xavier_uniform_(self.main.weight)
        init.zeros_(self.main.bias)  # type: ignore[arg-type]

    def forward(self, x: Tensor, i: int) -> None:
        """Forward pass of the UpSample block.

        :param x: Input tensor.
        :param i: Conditional index. Not used.
        :return: Output tensor.
        """
        _, _, H, W = x.shape
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.main(x)


class ConditionalGroupNorm(nn.Module):
    """Conditional Group Normalization."""

    def __init__(self, num_groups: int, num_channels: int, L: int = 10) -> None:
        """Initialize the ConditionalGroupNorm.

        :param num_groups: Number of groups.
        :param num_channels: Number of channels.
        :param L: Number of (conditional) layers.
        """
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.L = L
        self.gns = nn.ModuleList([nn.GroupNorm(num_groups, num_channels) for _ in range(self.L)])

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Forward pass of the ConditionalGroupNorm.

        :param x: Input tensor.
        :param i: Conditional index.
        :return: Output tensor.
        """
        return self.gns[i](x)


class ConditionalIdentity(nn.Module):
    """Conditional Identity block."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the ConditionalIdentity block."""
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Forward pass of the ConditionalIdentity block.

        :param x: Input tensor.
        :param i: Conditional index. Not used.
        :return: Output tensor.
        """
        return x


class AttnBlock(nn.Module):
    """Attention block."""

    def __init__(self, in_ch: int, L: int = 10) -> None:
        """Initialize the Attention block.

        :param in_ch: Number of input channels.
        :param L: Number of (conditional) layers.
        """
        super().__init__()
        self.group_norm = ConditionalGroupNorm(32, in_ch, L)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.initialize()

    def initialize(self) -> None:
        """Initialize the weights of the Attention block."""
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            init.xavier_uniform_(module.weight)
            init.zeros_(module.bias)  # type: ignore[arg-type]

        init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Forward pass of the Attention block.

        :param x: Input tensor.
        :param i: Conditional index.
        :return: Output tensor.
        """
        B, C, H, W = x.shape
        h = self.group_norm(x, i)
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

    def __init__(self, in_ch: int, out_ch: int, L: int, dropout: float, attn=False) -> None:
        """Initialize the ResBlock.

        :param in_ch: Number of input channels.
        :param out_ch: Number of output channels.
        :param L: Number of (conditional) layers.
        :param dropout: Dropout rate.
        :param attn: Whether to use attention block.
        """
        super().__init__()
        self.cond_gn1 = ConditionalGroupNorm(32, in_ch, L)
        self.swish1 = Swish()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)

        self.cond_gn2 = ConditionalGroupNorm(32, out_ch, L)
        self.swish2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1)

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()  # type: ignore[assignment]

        if attn:
            self.attn = AttnBlock(out_ch, L)
        else:
            self.attn = ConditionalIdentity()  # type: ignore[assignment]

        self.initialize()

    def initialize(self) -> None:
        """Initialize the weights of the ResBlock."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)  # type: ignore[arg-type]

        init.xavier_uniform_(self.conv2.weight, gain=1e-5)

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Forward pass of the ResBlock.

        :param x: Input tensor.
        :param i: Conditional index.
        :return: Output tensor.
        """
        h = self.conv1(self.swish1(self.cond_gn1(x, i)))
        h = self.conv2(self.dropout(self.swish2(self.cond_gn2(h, i))))
        h = h + self.shortcut(x)
        return self.attn(h, i)


class UNet(nn.Module):
    """UNet model."""

    def __init__(
        self, in_ch: int, ch: int, ch_mult: List[int], L: int, attn: List[int], num_res_blocks: int, dropout: float
    ) -> None:
        """Initialize the UNet.

        :param in_ch: Number of input channels.
        :param ch: Number of channels.
        :param ch_mult: List of channel multipliers.
        :param L: Number of (conditional) layers.
        :param attn: Index of attention block.
        :param num_res_blocks: Number of residual blocks.
        :param dropout: Dropout rate.
        """
        super().__init__()
        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        self.head = nn.Conv2d(in_ch, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        chs = [ch]  # record output channel when dowmsample for upsample
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult

            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, L=L, dropout=dropout, attn=(i in attn)))
                now_ch = out_ch
                chs.append(now_ch)

            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList(
            [
                ResBlock(now_ch, now_ch, L, dropout, attn=True),
                ResBlock(now_ch, now_ch, L, dropout, attn=False),
            ]
        )

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult

            for _ in range(num_res_blocks + 1):
                self.upblocks.append(
                    ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, L=L, dropout=dropout, attn=(i in attn))
                )
                now_ch = out_ch

            if i != 0:
                self.upblocks.append(UpSample(now_ch))

        assert len(chs) == 0

        self.cond_gn = ConditionalGroupNorm(32, now_ch, L)
        self.swish = Swish()
        self.conv = nn.Conv2d(now_ch, in_ch, 3, stride=1, padding=1)
        self.initialize()

    def initialize(self) -> None:
        """Initialize the weights of the UNet."""
        init.xavier_uniform_(self.head.weight)
        init.zeros_(self.head.bias)  # type: ignore[arg-type]
        init.xavier_uniform_(self.conv.weight, gain=1e-5)
        init.zeros_(self.conv.bias)  # type: ignore[arg-type]

    def forward(self, x: Tensor, i: int) -> Tensor:
        """Forward pass of the UNet.

        :param x: Input tensor.
        :param i: Conditional index.
        :return: Output tensor.
        """
        # Downsampling
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, i)
            hs.append(h)

        # Middle
        for layer in self.middleblocks:
            h = layer(h, i)

        # Upsampling
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = layer(h, i)

        h = self.conv(self.swish(self.cond_gn(h, i)))
        assert len(hs) == 0
        return h
