import abc

from torch import Tensor


class TrainStep:
    """Abstract class for the Train Step."""

    @abc.abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass
