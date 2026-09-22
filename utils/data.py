import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=64):

    transform = transforms.ToTensor()

    # =========================
    # Load 60,000 training images
    # =========================

    full_train_dataset = datasets.MNIST(
        root="./MNIST DATASET",
        train=True,
        transform=transform,
        download=True
    )

    # =========================
    # Load 10,000 test images
    # =========================

    test_dataset = datasets.MNIST(
        root="./MNIST DATASET",
        train=False,
        transform=transform,
        download=True
    )

    # =========================
    # Split train / validation
    # =========================

    train_size = 50000
    valid_size = 10000

    train_dataset, valid_dataset = random_split(
        full_train_dataset,
        [train_size, valid_size]
    )

    # =========================
    # DataLoader
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, valid_loader, test_loader