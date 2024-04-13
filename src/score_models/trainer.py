import os
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore[import-untyped]

from score_models.train_steps.ncsn_train_steps import TrainStep


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
        ckpt["scheduler_state_dict"] = (scheduler.state_dict(),)

    torch.save(ckpt, path)


def compute_running_loss(running_loss: Optional[float], loss: float):
    if running_loss is None:
        return loss

    return 0.9 * running_loss + 0.1 * loss


def trainer(
    train_step: TrainStep,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    num_steps: int = 100,
    log_every: int = 100,
    save_every: int = 1000,
    checkpoint_dir: str = "checkpoints",
    batch_preprocessor: Optional[Callable] = None,
) -> nn.Module:
    """Trains the model using the train_step function.

    :param train_step: Function to compute the loss
    :param model: Model to train
    :param train_loader: DataLoader for the training data
    :param optimizer: Optimizer
    :param scheduler: Scheduler
    :param num_steps: Number of steps
    :param log_every: Log statistics every n steps
    :param save_every: Save model checkpoint every n steps
    :param checkpoint_dir: Directory to save model checkpoints
    :param batch_preprocessor: Preprocessor for the batch
    """
    model.train()
    generator = iter(train_loader)
    running_loss = None

    for step in tqdm(range(1, num_steps + 1)):
        try:
            x = next(generator)
        except StopIteration:
            generator = iter(train_loader)
            x = next(generator)

        if batch_preprocessor is not None:
            x = batch_preprocessor(x)

        # compute the loss
        loss = train_step(*x)

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # scheduler step
        if scheduler is not None:
            scheduler.step()

        # update the running loss
        running_loss = compute_running_loss(running_loss, loss.item())

        # log the loss
        if step % log_every == 0 or step == 1:
            print(f"Step {step}, Loss: {running_loss:.4f}. LR: {optimizer.param_groups[0]['lr'] * 1e4:.4f}e-4")

        # save the model
        if step % save_every == 0:
            save_checkpoint(step, checkpoint_dir, model, optimizer, scheduler)

    save_checkpoint(num_steps, checkpoint_dir, model, optimizer, scheduler)
    return model
