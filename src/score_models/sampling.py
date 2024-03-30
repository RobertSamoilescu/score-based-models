from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn


@torch.no_grad()
def annealed_langevin_dynamics(
    score_model: nn.Module, input_size: Tuple[int, ...], sigmas: List[float], eps: float = 0.1, T: int = 100
) -> torch.Tensor:
    """Annealed Langevin Dynamics sampling.

    :param score_model: Score model
    :param sigmas: List of sigmas
    :param eps: Step size
    :param T: Number of steps
    :return: Sampled tensor
    """
    score_model.eval()
    r1, r2 = -8, 8

    x = (r2 - r1) * torch.rand(input_size).cuda() + r1
    L = len(sigmas)

    for i in range(L):
        alpha_i = eps * sigmas[i] / sigmas[L - 1]

        for _ in range(T):
            z_t = torch.randn(input_size).cuda()
            x = x + 0.5 * alpha_i * score_model(x, i) + np.sqrt(alpha_i) * z_t

    return x
