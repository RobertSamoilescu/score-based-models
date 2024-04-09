import argparse
from typing import Callable, Tuple

import torch
import torch.optim as optim
import torchvision
from score_models.models.unet import UNet
from score_models.train_steps.ncsn_train_steps import TrainStepNCSN
from score_models.trainer import trainer
from score_models.utils.noise import get_sigmas
from score_models.utils.warmup_scheduler import WarmUpScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms


def _get_dataloader_cifar10(batch_size: int, shuffle: bool = True, num_workers: int = 0) -> Tuple[DataLoader, Callable]:
    # Define transformations to be applied to the data
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # Convert PIL image or numpy array to tensor
        ]
    )

    # Download and load the CIFAR10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

    # Create a DataLoader for the CIFAR10 training dataset
    return (
        DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),  # type: ignore[arg-type]
        lambda x: x[0],
    )


def get_dataloader(
    dataset: str, batch_size: int, shuffle: bool = True, num_workers: int = 0
) -> Tuple[DataLoader, Callable]:
    if dataset == "cifar10":
        return _get_dataloader_cifar10(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    else:
        raise ValueError(f"Dataset {dataset} not supported")


def train(args: argparse.Namespace) -> None:
    """Train Noise Conditional Score Networks on CIFAR10 dataset.

    :param args: Arguments
    """

    # define score model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    score_model = UNet(
        T=args.L,
        in_ch=args.in_ch,
        ch=args.ch,
        ch_mult=args.ch_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
    ).to(device)

    # compile the model
    score_model = torch.compile(score_model)  # type: ignore[assignment]

    # load dataset
    train_loader, batch_preprocessor = get_dataloader(
        dataset=args.dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # define optimizer
    optimizer = optim.AdamW(score_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define scheduler
    warmup_steps = 1000
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps - warmup_steps)
    warmup_scheduler = WarmUpScheduler(
        optimizer=optimizer,
        lr_scheduler=scheduler,
        warmup_steps=warmup_steps,
        warmup_start_lr=0.0,
        warmup_mode="linear",
    )

    # define train step (i.e., criterion)
    sigmas = get_sigmas(L=args.L, sigma_min=args.sigma_min, sigma_max=args.sigma_max)
    train_step = TrainStepNCSN(score_model=score_model, sigmas=sigmas)

    # run training loop
    score_model = trainer(  # type: ignore[assignment]
        train_step=train_step,
        model=score_model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=warmup_scheduler,  # type: ignore[arg-type]
        device=device,
        num_steps=args.num_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
        batch_preprocessor=batch_preprocessor,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Noise Conditional Score Networks on CIFAR10")
    # dataset arguments
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10"], help="Dataset to train on")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    # training arguments
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_steps", type=int, default=200_000, help="Number of training steps")
    parser.add_argument("--log_every", type=int, default=100, help="Log training statistics every n steps")
    parser.add_argument("--save_every", type=int, default=10_000, help="Save model checkpoint every n steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/ncsn_cifar10/", help="Directory to save model checkpoints"
    )
    # noise arguments
    parser.add_argument("--L", type=int, default=10, help="Number of noise scales")
    parser.add_argument("--sigma_min", type=float, default=0.01, help="Minimum sigma for noise")
    parser.add_argument("--sigma_max", type=float, default=1.0, help="Maximum sigma for noise")
    # model arguments
    parser.add_argument("--in_ch", type=int, default=3, help="Number of input channels")
    parser.add_argument("--ch", type=int, default=128, help="Number of channels in the model")
    parser.add_argument("--ch_mult", type=int, nargs="+", default=[1, 2, 3, 4], help="Channel multiplier")
    parser.add_argument("--attn", type=int, nargs="+", default=[2], help="Attention layers")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Number of residual blocks")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout probability")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
