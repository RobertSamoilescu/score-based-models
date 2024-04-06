from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from score_models.loss import DenoisingScoreMatching
from score_models.train_steps.base_train_step import TrainStep


class TrainStepNCSN(TrainStep):
    def __init__(self, score_model: nn.Module, sigmas: List[float]) -> None:
        """Constructor of the train step for the Noise Conditional Score Networks(NCSN).

        :param score_model: Score model
        :param sigmas: List of sigmas
        """
        self.score_model = score_model
        self.sigmas = torch.tensor(sigmas)

        # define loss functions
        self.loss_fn = DenoisingScoreMatching()

    def __call__(self, x: Tensor) -> Tensor:
        """Computes the loss for the Denoising Score Matching.

        :param x: batch of input tensors
        :return: loss
        """
        # sample sigmas
        batch_size = x.shape[0]
        indices = torch.randint(0, len(self.sigmas), (batch_size,))
        sigmas = self.sigmas[indices]

        # send tensors to device
        indices = indices.to(x.device)
        sigmas = sigmas.to(x.device)

        # reshape sigmas to match x
        sigmas = sigmas.reshape(-1, *(1,) * (len(x.shape) - 1))

        # construct noisy example and compute score
        eps = sigmas * torch.randn_like(x)
        score = self.score_model(x + eps, indices)

        # compute and return loss
        return self.loss_fn(eps=eps, score=score, sigmas=sigmas)
