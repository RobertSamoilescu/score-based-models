import textwrap
from typing import Callable, List, Optional, Tuple, Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
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
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot the gradient field of the score model.

    :param score_model: Score model
    :param i: Sigma index. Default is None
    :param x_range: X range
    :param y_range: Y range
    :param datapoints: Data points
    """
    if ax is None:
        _, ax = plt.subplots()

    _plot_gradient_field(
        tensor_func=_tensor_func, score_model=score_model, i=i, x_range=x_range, y_range=y_range, ax=ax, spacing=0.7
    )

    # Add scatter plot of data points
    if datapoints is not None:
        datapoints_np = datapoints.numpy()
        ax.scatter(datapoints_np[:, 0], datapoints_np[:, 1], color="red", label="datapoints", s=4)
        ax.legend()

    ax.set_title("Gradient Field")
    return ax


def show_torch_images(
    imgs: Union[torch.Tensor, List[torch.Tensor]], titles: Optional[List[str]] = None, width_factor: float = 10.0
) -> None:
    """Display a list of images.

    :param imgs: list of images
    """
    if not isinstance(imgs, list):
        imgs = [imgs]

    nrows = 1 if titles is None else 2
    fig, axs = plt.subplots(nrows=nrows, ncols=len(imgs), squeeze=False)

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if titles is not None:
            wrapped_text = textwrap.fill(titles[i], width=int(fig.get_figwidth() * width_factor // len(imgs)))
            axs[1, i].text(0.5, 0.5, wrapped_text, ha="center", va="center", fontsize=10)
            axs[1, i].axis("off")
            # axs[1, i].set_aspect('equal')

    # Adjust spacing to remove padding between subplots
    plt.subplots_adjust(wspace=0, hspace=0)


def tensors_to_gif(
    tensor_list: List[torch.Tensor], filename: str, duration=2.0, value_range: Tuple[int, int] = (-1, 1)
) -> None:
    """Save a list of tensors as a GIF file.

    :param tensor_list: List of tensors
    :param filename: Name of the GIF file
    :param duration: Duration of each frame in seconds
    """
    images = []

    for tensor in tensor_list:
        tensor = torch.clamp(tensor, -1, 1)
        image = torchvision.utils.make_grid(tensor, nrow=8, normalize=True, value_range=value_range)
        image = image.permute(1, 2, 0).numpy()
        # Ensure pixel values are in the range [0, 255] and of integer type
        image = (image * 255).astype("uint8")
        images.append(image)

    # Write images to GIF using imageio
    imageio.mimsave(filename, images, format="GIF", duration=duration, loop=0)  # type: ignore[call-overload]
