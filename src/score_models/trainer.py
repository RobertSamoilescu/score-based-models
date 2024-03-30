import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from score_models.train_steps import TrainStep


def trainer(
    train_step: TrainStep,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    num_epochs: int = 100,
    device: str = "cuda",
) -> nn.Module:
    """Trains the model using the train_step function.

    :param train_step: Function to compute the loss
    :param model: Model to train
    :param train_loader: DataLoader for the training data
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param num_epochs: Number of epochs
    """
    for epoch in range(num_epochs):
        model.train()

        for i, x in enumerate(train_loader):
            loss = train_step(x.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = epoch * len(train_loader) + i
            if step % 100 == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

    return model
