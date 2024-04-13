""" 
Copyright zoubohao/DenoisingDiffusionProbabilityModel-ddpm-
https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-/blob/main/Diffusion/Model.py
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init

from score_models.models.unet.common import Swish


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

    def forward(self, t: Tensor) -> Tensor:
        """Forward pass of the TimeEmbedding module.

        :param t: Time tensor
        :return: Output tensor
        """
        return self.timembedding(t)
