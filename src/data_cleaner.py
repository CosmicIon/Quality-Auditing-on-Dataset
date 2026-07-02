"""
data_cleaner.py — Create a cleaned CIFAR-10 training subset.

Usage:
    python src/data_cleaner.py          # standalone: prints cleaning stats

This module reads the flagged sample indices produced by Phase 2
(data_auditor.py) and provides helper functions to build a "cleaned"
training set that excludes suspected mislabeled and outlier samples.
"""

import os
import sys
import json
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Subset

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_datasets, CIFAR10_CLASSES, PROCESSED_DATA_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIT_DIR = os.path.join(PROCESSED_DATA_DIR, "audit")
FLAGGED_PATH = os.path.join(AUDIT_DIR, "flagged_indices.json")


# ═══════════════════════════════════════════════════════════════════════════
# Core API
# ═══════════════════════════════════════════════════════════════════════════
def load_flagged_indices() -> dict:
    """
    Load the flagged sample indices produced by the data auditor.

    Returns:
        dict with keys:
            mislabeled_indices – list[int]
            outlier_indices    – list[int]
            total_mislabeled   – int
            total_outliers     – int
    """
    if not os.path.isfile(FLAGGED_PATH):
        raise FileNotFoundError(
            f"Audit results not found at {FLAGGED_PATH}. "
            "Run `python src/data_auditor.py` first (Phase 2)."
        )
    with open(FLAGGED_PATH, "r") as f:
        data = json.load(f)
    return data


def get_clean_indices(total_samples: int = 50_000) -> np.ndarray:
    """
    Return sorted array of indices to KEEP (i.e. not flagged).

    The flagged set is the union of mislabeled and outlier indices.
    """
    flagged = load_flagged_indices()
    bad_set = set(flagged["mislabeled_indices"]) | set(flagged["outlier_indices"])
    clean = sorted(set(range(total_samples)) - bad_set)
    return np.array(clean, dtype=np.int64)


def get_clean_dataset():
    """
    Return a torch Subset of CIFAR-10 training data with flagged samples
    removed.

    Returns:
        clean_dataset – torch.utils.data.Subset
    """
    train_dataset, _ = get_datasets()
    clean_idx = get_clean_indices(len(train_dataset))
    return Subset(train_dataset, clean_idx.tolist())


def get_clean_indices_per_class(target_per_class: int = 5000) -> dict:
    """
    Compute per-class statistics after cleaning.

    Returns:
        dict mapping class_index -> {
            "name":          str,
            "original":      int,   # always 5000 for CIFAR-10
            "clean":         int,   # remaining after removal
            "removed":       int,
            "deficit":       int,   # how many synthetic samples needed
        }
    """
    train_dataset, _ = get_datasets()
    all_labels = np.array(train_dataset.targets)
    clean_idx = get_clean_indices(len(train_dataset))
    clean_labels = all_labels[clean_idx]

    clean_counts = Counter(clean_labels.tolist())
    stats = {}
    for cls_idx in range(len(CIFAR10_CLASSES)):
        clean_count = clean_counts.get(cls_idx, 0)
        removed = target_per_class - clean_count
        stats[cls_idx] = {
            "name": CIFAR10_CLASSES[cls_idx],
            "original": target_per_class,
            "clean": clean_count,
            "removed": removed,
            "deficit": max(0, removed),
        }
    return stats


def print_cleaning_summary():
    """Print a formatted summary of the data cleaning results."""
    flagged = load_flagged_indices()
    bad_set = set(flagged["mislabeled_indices"]) | set(flagged["outlier_indices"])
    overlap = set(flagged["mislabeled_indices"]) & set(flagged["outlier_indices"])

    print("=" * 65)
    print(" Data Cleaning Summary")
    print("=" * 65)
    print(f"  Mislabeled samples flagged : {flagged['total_mislabeled']:,}")
    print(f"  Outlier samples flagged    : {flagged['total_outliers']:,}")
    print(f"  Overlap (both flags)       : {len(overlap):,}")
    print(f"  Unique samples removed     : {len(bad_set):,}")
    print(f"  Remaining clean samples    : {50_000 - len(bad_set):,}")
    print()

    stats = get_clean_indices_per_class()
    print("  Per-class breakdown:")
    print(f"  {'Class':<12s} {'Original':>8s} {'Clean':>8s} {'Removed':>8s} {'Deficit':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    total_deficit = 0
    for cls_idx in range(len(CIFAR10_CLASSES)):
        s = stats[cls_idx]
        print(f"  {s['name']:<12s} {s['original']:>8,} {s['clean']:>8,} "
              f"{s['removed']:>8,} {s['deficit']:>8,}")
        total_deficit += s["deficit"]
    print(f"  {'TOTAL':<12s} {'50,000':>8s} {50_000 - len(bad_set):>8,} "
          f"{len(bad_set):>8,} {total_deficit:>8,}")
    print("=" * 65)
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print_cleaning_summary()
