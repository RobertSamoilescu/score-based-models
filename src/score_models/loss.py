import torch
import torch.nn as nn
from torch import Tensor


class DenoisingScoreMatching(nn.Module):
    """Denoising Score Matching Loss."""

    def __init__(self, sigma: float):
        """Constructor of the Denoising Score Matching Loss.

        :param sigma: Standard deviation of the noise
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, x: Tensor, x_tilde: Tensor, score: Tensor) -> Tensor:
        """Computes the score loss for 1D tensors.

        :param x: Input tensor
        :param x_tilde: Noisy input tensor
        :param score: Score tensor
        :return: denoising score matching loss
        """
        dim = tuple(range(1, len(x.shape)))
        return 0.5 * torch.mean(torch.norm(self.sigma * score + (x_tilde - x) / self.sigma, dim=dim) ** 2)
