import argparse

import torch
import torch.optim as optim
import torchvision
from score_models.models.unet import UNet
from score_models.train_steps.ddpm_train_step import TrainStepDDPM
from score_models.trainer import trainer
from score_models.utils.noise import get_betas
from score_models.utils.warmup_scheduler import WarmUpScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms


def get_dataloader(batch_size: int, shuffle: bool = True) -> DataLoader:
    # Define transformations to be applied to the data
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),  # Convert PIL image or numpy array to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize the image between -1 and 1
        ]
    )

    # Download and load the CIFAR10 training dataset
    train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)

    # Create a DataLoader for the CIFAR10 training dataset
    images = [image for image, _ in train_dataset]
    return DataLoader(dataset=images, batch_size=batch_size, shuffle=shuffle, num_workers=4)  # type: ignore[arg-type]


def train(args: argparse.Namespace) -> None:
    """Train Noise Conditional Score Networks on CIFAR10 dataset.

    :param args: Arguments
    """

    # define score model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    score_model = UNet(
        T=args.T,
        in_ch=args.in_ch,
        ch=args.ch,
        ch_mult=args.ch_mult,
        attn=args.attn,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
    ).to(device)

    # load dataset
    train_loader = get_dataloader(batch_size=args.batch_size)

    # define optimizer
    optimizer = optim.AdamW(score_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define scheduler
    warmup_steps = int(0.1 * args.num_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_steps - warmup_steps)
    warmup_scheduler = WarmUpScheduler(
        optimizer=optimizer,
        lr_scheduler=scheduler,
        warmup_steps=warmup_steps,
        warmup_start_lr=0.0,
        warmup_mode="linear",
    )

    # define train step (i.e., criterion)
    betas = get_betas(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    train_step = TrainStepDDPM(score_model=score_model, alphas_bar=betas["alphas_bar"])

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
    )


def main():
    parser = argparse.ArgumentParser(description="Train Noise Conditional Score Networks on CIFAR10")
    # training arguments
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num_steps", type=int, default=800_000, help="Number of training steps")
    parser.add_argument("--log_every", type=int, default=100, help="Log training statistics every n steps")
    parser.add_argument("--save_every", type=int, default=50_000, help="Save model checkpoint every n steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints/ddpm_cifar10/", help="Directory to save model checkpoints"
    )
    # noise arguments
    parser.add_argument("--T", type=int, default=10, help="Number of diffusion steps")
    parser.add_argument("--beta_min", type=float, default=1e-4, help="Minimum beta for noise")
    parser.add_argument("--beta_max", type=float, default=2e-2, help="Maximum beta for noise")
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
