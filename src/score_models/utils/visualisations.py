from typing import Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def _plot_gradient_field(
    tensor_func: Callable,
    score_model: nn.Module,
    i: Optional[int],
    x_range: Tuple[int, int],
    y_range: Tuple[int, int],
    spacing: float = 0.1,
    ax=None,
) -> None:
    """Helper function to plot the gradient field.

    :param tensor_func: Tensor function to get the gradients
    :param score_model: Score model
    :param i: Sigma index
    :param x_range: X range
    :param y_range: Y range
    :param spacing: Spacing
    :param ax: Axis
    """
    if ax is None:
        ax = plt.gca()

    x = np.arange(x_range[0], x_range[1], spacing)
    y = np.arange(y_range[0], y_range[1], spacing)
    X, Y = np.meshgrid(x, y)
    U, V = tensor_func(score_model, i, X, Y)

    ax.quiver(X, Y, U, V, scale=30)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")


def _tensor_func(score_model: nn.Module, i: int, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Helper function to get the gradients.

    :param score_model: Score model
    :param i: Sigma index
    :param x: X tensor
    :param y: Y tensor
    :return: Gradients
    """
    shape = x.shape
    x_pt = torch.from_numpy(x).reshape(-1, 1).float()
    y_pt = torch.from_numpy(y).reshape(-1, 1).float()
    X = torch.cat((x_pt, y_pt), dim=-1).to("cuda")
    indices = i * torch.ones((X.shape[0],), dtype=torch.long).to("cuda")

    score_model.eval()
    with torch.no_grad():
        Psi = score_model(X, indices) if i is not None else score_model(X)

    df_dx = Psi[:, 0].cpu().numpy().reshape(shape)
    df_dy = Psi[:, 1].cpu().numpy().reshape(shape)
    return df_dx, df_dy


def plot_gradient_field(
    score_model: nn.Module,
    i: Optional[int] = None,
    x_range: Tuple[int, int] = (-10, 10),
    y_range: Tuple[int, int] = (-10, 10),
    datapoints: Optional[Tensor] = None,
) -> None:
    """Plot the gradient field of the score model.

    :param score_model: Score model
    :param i: Sigma index. Default is None
    :param x_range: X range
    :param y_range: Y range
    :param datapoints: Data points
    """
    _, ax = plt.subplots()
    _plot_gradient_field(
        tensor_func=_tensor_func, score_model=score_model, i=i, x_range=x_range, y_range=y_range, ax=ax, spacing=0.7
    )

    # Add scatter plot of data points
    if datapoints is not None:
        datapoints_np = datapoints.numpy()
        ax.scatter(datapoints_np[:, 0], datapoints_np[:, 1], color="red", label="datapoints", s=4)
        ax.legend()

    plt.title("Gradient Field")
    plt.show()
