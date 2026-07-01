"""
data_loader.py — Phase 1: Download CIFAR-10 and set up data loaders.

Usage:
    python src/data_loader.py

This script:
  1. Downloads the CIFAR-10 dataset into data/raw/
  2. Applies basic transforms (ToTensor + Normalize)
  3. Creates train and test DataLoaders
  4. Prints dataset statistics and class distribution
"""

import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# CIFAR-10 class names (index → label)
CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# CIFAR-10 channel-wise mean and std (precomputed on training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ---------------------------------------------------------------------------
# Dataset & Loader helpers
# ---------------------------------------------------------------------------
def get_datasets():
    """Download (if needed) and return CIFAR-10 train and test datasets."""
    train_dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR,
        train=False,
        download=True,
        transform=test_transform,
    )
    return train_dataset, test_dataset


def get_dataloaders(batch_size: int = 64, num_workers: int = 2):
    """Return DataLoaders for the train and test splits."""
    train_dataset, test_dataset = get_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Exploration utilities
# ---------------------------------------------------------------------------
def print_dataset_stats(train_dataset, test_dataset):
    """Print basic statistics about the datasets."""
    print("=" * 60)
    print("CIFAR-10 Dataset Statistics")
    print("=" * 60)
    print(f"  Training samples : {len(train_dataset)}")
    print(f"  Test samples     : {len(test_dataset)}")
    print(f"  Image shape      : {train_dataset[0][0].shape}  (C, H, W)")
    print(f"  Number of classes : {len(CIFAR10_CLASSES)}")
    print(f"  Classes           : {CIFAR10_CLASSES}")
    print()

    # Class distribution
    train_labels = np.array(train_dataset.targets)
    test_labels  = np.array(test_dataset.targets)

    print("  Class Distribution (train / test):")
    for idx, name in enumerate(CIFAR10_CLASSES):
        train_count = int((train_labels == idx).sum())
        test_count  = int((test_labels == idx).sum())
        print(f"    {idx:2d}. {name:12s}  {train_count:5d} / {test_count:5d}")
    print("=" * 60)


def save_sample_grid(dataset, save_path: str, n: int = 25):
    """Save a grid of n sample images (un-normalised) to save_path."""
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    fig.suptitle("CIFAR-10 Sample Images", fontsize=14)

    for i, ax in enumerate(axes.flat):
        if i >= n or i >= len(dataset):
            ax.axis("off")
            continue

        img, label = dataset[i]
        # Undo normalisation for display
        img = img.clone()
        for ch in range(3):
            img[ch] = img[ch] * CIFAR10_STD[ch] + CIFAR10_MEAN[ch]
        img = img.clamp(0, 1)

        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(CIFAR10_CLASSES[label], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Sample grid saved to: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n[Phase 1] Downloading and loading CIFAR-10...\n")

    train_dataset, test_dataset = get_datasets()
    print_dataset_stats(train_dataset, test_dataset)

    # Create data loaders and verify a batch
    train_loader, test_loader = get_dataloaders(batch_size=64)
    images, labels = next(iter(train_loader))
    print(f"\n  Sample batch — images: {images.shape}, labels: {labels.shape}")

    # Save a visual grid
    grid_path = os.path.join(PROCESSED_DATA_DIR, "sample_grid.png")
    save_sample_grid(train_dataset, grid_path)

    print("\n✅  Phase 1 complete — dataset is ready.\n")


if __name__ == "__main__":
    main()
