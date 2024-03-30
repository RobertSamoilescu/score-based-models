import abc
from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from score_models.loss import DenoisingScoreMatching


class TrainStep:
    """Abstract class for the Train Step."""

    @abc.abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass


class TrainStepDenoisingScoreMatching(TrainStep):
    def __init__(self, score_model: nn.Module, sigmas: List[float]) -> None:
        """Constructor of the Train Step Denoising Score Matching.

        :param score_model: Score model
        :param sigmas: List of sigmas
        """
        self.score_model = score_model
        self.sigmas = sigmas

        # define loss functions
        self.loss_fns = [DenoisingScoreMatching(sigma) for sigma in sigmas]

    def __call__(self, x: Tensor) -> Tensor:
        """Computes the loss for the Denoising Score Matching.

        :param x: batch of input tensors
        :return: loss
        """
        loss = torch.tensor(0.0, device=x.device)

        for i, sigma in enumerate(self.sigmas):
            x_tilde = x + sigma * torch.randn_like(x)
            score = self.score_model(x_tilde, i)
            loss += self.loss_fns[i](x, x_tilde, score)

        return loss / len(self.sigmas)
