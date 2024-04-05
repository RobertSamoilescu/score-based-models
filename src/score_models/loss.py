import torch
import torch.nn as nn
from torch import Tensor


class DenoisingScoreMatching(nn.Module):
    """Denoising Score Matching Loss."""

    def __init__(self):
        super().__init__()

    def forward(self, eps: Tensor, score: Tensor, sigmas) -> Tensor:
        """Computes the score loss for 1D tensors.

        :param eps: noise
        :param score: score tensor
        :param sigmas: standard deviations for noise
        :return: denoising score matching loss
        """
        eps = eps.view(eps.size(0), -1)
        score = score.view(score.size(0), -1)
        sigmas = sigmas.view(sigmas.size(0), -1)
        return torch.mean(torch.sum((sigmas * score + eps / sigmas) ** 2, dim=-1), dim=0)
