from typing import List

import numpy as np


def get_sigmas(L: int, sigma_min: float, sigma_max: float) -> List[float]:
    """Returns a list of a geometric series of L sigmas ordered from largest to smallest.
    Used for NCSN.

    :param L: Number of sigmas
    :param sigma_min: Minimum sigma
    :param sigma_max: Maximum sigma
    :return: List of sigmas
    """
    r = (sigma_max / sigma_min) ** (1 / (L - 1))
    return [sigma_min * r**i for i in range(L)][::-1]


def get_alphas_bar(beta_min: float = 1e-4, beta_max: float = 2e-2, T: int = 1_000) -> np.ndarray:
    """Returns a list of alphas_bar values for a given range of betas.
    Used for DDPM.

    :param beta_min: Minimum beta
    :param beta_max: Maximum beta
    :param T: Number of alphas
    :return: Array of alphas_bar values
    """
    betas = np.linspace(beta_min, beta_max, T)
    alphas = 1 - betas
    return np.cumprod(alphas)
