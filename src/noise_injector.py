"""
noise_injector.py -- Inject controlled label noise into CIFAR-10 training set.

Usage:
    python src/noise_injector.py
    python src/noise_injector.py --noise-rate 0.15 --seed 42

This script:
    1. Loads the original CIFAR-10 training set (50,000 images + labels)
    2. Randomly selects a fraction of samples and flips their labels
       to a DIFFERENT random class
    3. Saves the complete noisy dataset to data/processed/noisy_cifar10.pt
       (contains both images and corrupted labels as PyTorch tensors)
    4. Saves noise metadata to data/processed/noise/noisy_labels.json
    5. Prints a summary of the corruption

After running this script, every downstream script (data_auditor.py,
downstream_validation.py, etc.) will automatically use the noisy dataset
via data_loader.get_datasets().
"""

import os
import sys
import json
import argparse
from collections import Counter

import numpy as np
import torch
import torchvision

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    RAW_DATA_DIR,
    CIFAR10_CLASSES,
    PROCESSED_DATA_DIR,
    _ensure_cifar10_downloaded,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
NOISE_DIR = os.path.join(PROCESSED_DATA_DIR, "noise")
NOISY_PT_PATH = os.path.join(PROCESSED_DATA_DIR, "noisy_cifar10.pt")


def inject_noise(noise_rate: float = 0.15, seed: int = 42) -> dict:
    """
    Corrupt a fraction of CIFAR-10 training labels and save the full
    noisy dataset (images + corrupted labels) as a PyTorch .pt file.

    Each corrupted sample's label is flipped to a uniformly random
    DIFFERENT class (never the original class).

    Args:
        noise_rate: Fraction of labels to corrupt (0.0 - 1.0)
        seed: Random seed for reproducibility

    Returns:
        dict with metadata about the corruption
    """
    print("=" * 65)
    print(" Noise Injection: Corrupting CIFAR-10 Labels")
    print("=" * 65)

    # Load original CIFAR-10 (raw, no transform)
    _ensure_cifar10_downloaded(RAW_DATA_DIR)
    dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR, train=True, download=False,
    )
    original_labels = np.array(dataset.targets, dtype=np.int64)
    n_samples = len(original_labels)
    n_classes = 10
    n_corrupt = int(n_samples * noise_rate)

    print(f"\n  Total samples:    {n_samples:,}")
    print(f"  Noise rate:       {noise_rate:.0%}")
    print(f"  Samples to corrupt: {n_corrupt:,}")

    # Select indices to corrupt
    rng = np.random.default_rng(seed)
    corrupted_indices = sorted(rng.choice(n_samples, size=n_corrupt, replace=False).tolist())

    # Flip labels to a different random class
    noisy_labels = original_labels.copy()
    for idx in corrupted_indices:
        original_cls = original_labels[idx]
        # Pick a random class that is NOT the original
        new_cls = rng.integers(0, n_classes - 1)
        if new_cls >= original_cls:
            new_cls += 1
        noisy_labels[idx] = new_cls

    # Build original_labels mapping (only for corrupted samples)
    original_map = {str(idx): int(original_labels[idx]) for idx in corrupted_indices}

    # Stats
    print(f"\n  Per-class corruption breakdown:")
    print(f"  {'Class':>12s} | {'Original':>8s} | {'After Noise':>11s} | {'Corrupted':>9s}")
    print(f"  {'-' * 12}-+-{'-' * 8}-+-{'-' * 11}-+-{'-' * 9}")
    orig_counts = Counter(original_labels.tolist())
    noisy_counts = Counter(noisy_labels.tolist())
    for cls in range(n_classes):
        corrupted_in_class = sum(1 for i in corrupted_indices if original_labels[i] == cls)
        print(f"  {CIFAR10_CLASSES[cls]:>12s} | {orig_counts[cls]:>8,} | "
              f"{noisy_counts[cls]:>11,} | {corrupted_in_class:>9,}")

    # Verify no label stayed the same
    n_actually_changed = sum(1 for i in corrupted_indices if noisy_labels[i] != original_labels[i])
    assert n_actually_changed == n_corrupt, "Some labels were not changed!"
    print(f"\n  Verified: all {n_corrupt:,} labels were changed to a different class.")

    # -----------------------------------------------------------------------
    # Save the COMPLETE noisy dataset as a PyTorch .pt file
    # -----------------------------------------------------------------------
    # dataset.data is a numpy array of shape (50000, 32, 32, 3), uint8
    images_data = dataset.data  # numpy array (N, H, W, C)
    noisy_labels_tensor = torch.tensor(noisy_labels, dtype=torch.long)

    noisy_dataset = {
        "data": images_data,                     # np.ndarray (50000, 32, 32, 3)
        "targets": noisy_labels_tensor,           # torch.Tensor (50000,)
        "noise_rate": noise_rate,
        "seed": seed,
        "n_corrupted": n_corrupt,
        "corrupted_indices": corrupted_indices,
    }

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    torch.save(noisy_dataset, NOISY_PT_PATH)
    pt_size = os.path.getsize(NOISY_PT_PATH) / (1024 * 1024)
    print(f"\n  Saved noisy dataset: {NOISY_PT_PATH}")
    print(f"  File size: {pt_size:.1f} MB")

    # -----------------------------------------------------------------------
    # Also save the JSON metadata (for compatibility and auditing)
    # -----------------------------------------------------------------------
    result = {
        "noisy_labels": noisy_labels.tolist(),
        "corrupted_indices": corrupted_indices,
        "original_labels": original_map,
        "noise_rate": noise_rate,
        "seed": seed,
        "n_corrupted": n_corrupt,
        "n_total": n_samples,
    }

    os.makedirs(NOISE_DIR, exist_ok=True)
    json_path = os.path.join(NOISE_DIR, "noisy_labels.json")
    with open(json_path, "w") as f:
        json.dump(result, f)
    print(f"  Saved metadata: {json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Inject label noise into CIFAR-10")
    parser.add_argument("--noise-rate", type=float, default=0.15, help="Fraction of labels to corrupt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    inject_noise(noise_rate=args.noise_rate, seed=args.seed)

    print(f"\n{'=' * 65}")
    print(f"  Noise injection complete.")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
