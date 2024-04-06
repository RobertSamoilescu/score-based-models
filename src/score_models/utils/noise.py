from typing import List, Dict

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


def get_betas(beta_min: float = 1e-4, beta_max: float = 2e-2, T: int = 1_000) -> Dict[str, List[float]]:
    """Returns a dictionary of betas, alphas, alphas_bar and sigmas.
    Used for DDPM.

    :param beta_min: Minimum beta
    :param beta_max: Maximum beta
    :param T: Number of alphas
    :return: Dictionary of betas, alphas, alphas_bar and sigmas
    """
    betas = np.linspace(beta_min, beta_max, T)
    alphas = 1 - betas
    alphas_bar = np.cumprod(alphas)
    sigmas = np.sqrt(betas)
    
    return {
        "betas": betas.tolist(),
        "alphas": alphas.tolist(),
        "alphas_bar": alphas_bar.tolist(),
        "sigmas": sigmas.tolist(),
    }