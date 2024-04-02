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
        x = x.view(x.size(0), -1)
        x_tilde = x_tilde.view(x_tilde.size(0), -1)
        score = score.view(score.size(0), -1)
        loss = torch.sum((self.sigma * score + (x_tilde - x) / self.sigma) ** 2, dim=-1)
        return torch.mean(loss, dim=0)