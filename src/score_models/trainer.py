import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional

from score_models.train_steps import TrainStep


def save_checkpoint(
    step: int,
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,

):
    path = f"{checkpoint_dir}/{step}.pt"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    ckpt = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict(),

    torch.save(ckpt, path)


def trainer(
    train_step: TrainStep,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    num_steps: int = 100,
    device: str = "cuda",
    log_every: int = 100,
    save_every: int = 1000,
    checkpoint_dir: str = "checkpoints",
) -> nn.Module:
    """Trains the model using the train_step function.

    :param train_step: Function to compute the loss
    :param model: Model to train
    :param train_loader: DataLoader for the training data
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param criterion: Loss function
    :param num_epochs: Number of epochs
    :param device: Device to use
    """
    model.train()
    generator = iter(train_loader)

    for step in tqdm(range(1, num_steps + 1)):
        try:
            x = next(generator)
        except StopIteration:
            generator = iter(train_loader)
            x = next(generator)

        # compute the loss
        loss = train_step(x.to(device))

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # scheduler step
        if scheduler is not None:
            scheduler.step()

        # log the loss
        if step % log_every == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # save the model
        if step % save_every == 0:
            save_checkpoint(step, checkpoint_dir, model, optimizer, scheduler)

    save_checkpoint(num_steps, checkpoint_dir, model, optimizer, scheduler)
    return model
