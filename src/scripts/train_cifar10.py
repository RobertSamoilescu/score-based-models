import torchvision
from torchvision import transforms

import torch.optim as optim
from torch.utils.data import DataLoader

from score_models.models.unet import UNet
from score_models.trainer import trainer
from score_models.train_steps import TrainStepDenoisingScoreMatching
from score_models.utils.noise import get_sigmas


def get_dataloader(batch_size: int, shuffle: bool = True):
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
    images = [image for image, _ in train_dataset]
    return DataLoader(dataset=images, batch_size=batch_size, shuffle=shuffle)


def train():
    
    # network constants
    L = 10
    in_ch = 3
    ch = 128
    ch_mult = [1, 2, 3, 4]
    attn = [2]
    num_res_blocks = 2
    dropout = 0.15

    # optimization constants
    checkpoint_dir = "checkpoints/cifar10_1e-4_mean/"
    device = "cuda"
    batch_size = 128
    num_steps = 200_000
    log_every = 100
    save_every = 10_000
    lr = 1e-4

    sigma_min = 0.01
    sigma_max = 1.0

    # define score model
    score_model = UNet(
        T=L,
        in_ch=in_ch,
        ch=ch,
        ch_mult=ch_mult,
        attn=attn,
        num_res_blocks=num_res_blocks,
        dropout=dropout,
    ).to(device)

    # load dataset
    train_loader = get_dataloader(batch_size=batch_size)

    # define optimizer
    optimizer = optim.AdamW(score_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps)

    # define train step (i.e., criterion)
    sigmas = get_sigmas(L=L, sigma_min=sigma_min, sigma_max=sigma_max)
    train_step = TrainStepDenoisingScoreMatching(score_model=score_model, sigmas=sigmas)

    # run training loop
    score_model = trainer(
        train_step=train_step,
        model=score_model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_steps=num_steps,
        log_every=log_every,
        save_every=save_every,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    train()
