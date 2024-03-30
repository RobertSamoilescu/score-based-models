from typing import Iterator

import numpy as np
import torch
from torch import Tensor


def guassian_mixture(
    N: int = 1000,
    p0: float = 0.5,
    mean_0: Tensor = torch.tensor([5, 5]),
    mean_1: Tensor = torch.tensor([-5, -5]),
) -> Tensor:
    """Generates a mixture of two 2D Gaussian distributions.

    :param N: Number of samples
    :param p0: Probability of the first Gaussian
    :param mean_0: Mean of the first Gaussian
    :param mean_1: Mean of the second Gaussian
    :return: Sampled tensor
    """
    size_0 = int(N * p0)
    size_1 = N - size_0

    Z_0 = torch.randn(size_0, 2) + mean_0.unsqueeze(0)
    Z_1 = torch.randn(size_1, 2) + mean_1.unsqueeze(0)

    X_train = torch.cat([Z_0, Z_1], dim=0)
    return X_train[torch.randperm(X_train.size()[0])]


def dataloader(X_train: torch.Tensor, batch_size: int = 100, device: str = "cuda") -> Iterator[torch.Tensor]:
    len_train = X_train.shape[0]
    n_iter = int(np.ceil(len_train / batch_size))

    for i in range(n_iter):
        istart, iend = i * batch_size, min((i + 1) * batch_size, len_train)
        yield X_train[istart:iend].to(device)
