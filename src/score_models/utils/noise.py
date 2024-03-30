from typing import List


def get_sigmas(L: int, sigma_min: float, sigma_max: float) -> List[float]:
    """Returns a list of a geometric series of L sigmas ordered from largest to smallest.

    :param L: Number of sigmas
    :param sigma_min: Minimum sigma
    :param sigma_max: Maximum sigma
    :return: List of sigmas
    """
    r = (sigma_max / sigma_min) ** (1 / (L - 1))
    return [sigma_min * r**i for i in range(L)][::-1]
