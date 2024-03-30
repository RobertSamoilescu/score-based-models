import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from score_models.train_steps import TrainStep


def trainer(
    train_step: TrainStep,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    num_steps: int = 100,
    device: str = "cuda",
    log_every: int = 100,
) -> nn.Module:
    """Trains the model using the train_step function.

    :param train_step: Function to compute the loss
    :param model: Model to train
    :param train_loader: DataLoader for the training data
    :param optimizer: Optimizer
    :param criterion: Loss function
    :param num_epochs: Number of epochs
    """
    model.train()
    generator = iter(train_loader)

    for step in tqdm(range(num_steps)):
        try:
            x = next(generator)
        except StopIteration:
            generator = iter(train_loader)
            x = next(generator)

        loss = train_step(x.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    return model
