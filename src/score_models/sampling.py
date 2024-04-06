from typing import List

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def annealed_langevin_dynamics(
    x: torch.Tensor,
    score_model: nn.Module,
    sigmas: List[float],
    eps: float = 0.1,
    T: int = 100,
) -> torch.Tensor:
    """Annealed Langevin Dynamics sampling.

    :param x: Input tensor
    :param score_model: Score model
    :param sigmas: List of sigmas
    :param eps: Step size
    :param T: Number of steps
    :param r: Range of the input
    :return: Sampled tensor
    """
    score_model.eval()
    L = len(sigmas)

    for i in range(L):
        alpha_i = eps * (sigmas[i] / sigmas[L - 1]) ** 2
        indices = i * torch.ones((x.shape[0],), dtype=torch.long).cuda()

        for _ in range(T):
            z_t = torch.randn_like(x)
            x = x + 0.5 * alpha_i * score_model(x, indices) + np.sqrt(alpha_i) * z_t

    return x


@torch.no_grad()
def ddpm_sampling(
    x: torch.Tensor,
    score_model: nn.Module,
    alphas: List[float],
    alphas_bar: List[float],
    sigmas: List[float],
    T: int = 1_000,
) -> torch.Tensor:
    """DDPM sampling.

    :param x: Input tensor
    :param score_model: Score model
    :param alphas: List of alphas
    :param alphas_bar: List of alphas_bar
    :param sigmas: List of sigmas
    :param T: Number of steps
    """
    score_model.eval()

    for t in range(T - 1, -1, -1):
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        ts = t * torch.ones(x.shape[0], device=x.device, dtype=torch.long)
        x = (x - (1 - alphas[t]) / np.sqrt(1 - alphas_bar[t]) * score_model(x, ts)) / np.sqrt(alphas[t]) + sigmas[t] * z

    return x
