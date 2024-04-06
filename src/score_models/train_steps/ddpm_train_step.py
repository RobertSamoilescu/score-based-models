from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from score_models.loss import DenoiseLoss
from score_models.train_steps.base_train_step import TrainStep


class TrainStepDDPM(TrainStep):
    def __init__(self, score_model: nn.Module, alphas_bar: List[float]) -> None:
        self.score_model = score_model
        self.alphas_bar = torch.tensor(alphas_bar)

        # define loss functions
        self.loss_fn = DenoiseLoss()

    def __call__(self, x: Tensor) -> Tensor:
        """Computes the loss for the Denoising Score Matching.

        :param x: batch of input tensors
        :return: loss
        """
        # sample sigmas
        batch_size = x.shape[0]
        indices = torch.randint(0, len(self.alphas_bar), (batch_size,))
        alphas_bar = self.alphas_bar[indices]

        # send tensors to device
        indices = indices.to(x.device)
        alphas_bar = alphas_bar.to(x.device)

        # reshape sigmas to match x
        alphas_bar = alphas_bar.reshape(-1, *(1,) * (len(x.shape) - 1))

        # sample noise, construct noisy example and predict noise
        # from noisy example
        eps = torch.randn_like(x)
        x_tilde = torch.sqrt(alphas_bar) * x + torch.sqrt(1 - alphas_bar) * eps
        eps_pred = self.score_model(x_tilde, indices)

        # compute and return loss
        return self.loss_fn(eps_pred=eps_pred, eps=eps)
