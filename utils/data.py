import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=64):
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(root="./MNIST DATASET", train=True, transform=transform, download=True)

    test_dataset = datasets.MNIST(root="./MNIST DATASET", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

    return train_loader,  test_loader