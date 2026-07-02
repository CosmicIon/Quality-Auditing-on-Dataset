"""
synthesize_data.py — Generate synthetic CIFAR-10 images to fill the deficit
                     left by removing flagged samples.

Usage:
    python src/synthesize_data.py
    python src/synthesize_data.py --checkpoint path/to/model.pt --gen-batch 64

This script:
    1. Loads the trained EMA model from the final checkpoint
    2. Computes per-class deficit (5000 - clean_count)
    3. Generates the exact number of synthetic images needed per class
    4. Saves the synthetic images, per-class visualizations, and a report
    5. Creates the final augmented dataset (clean reals + synthetics)
"""

import os
import sys
import time
import argparse
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    PROCESSED_DATA_DIR,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    get_datasets,
)
from src.data_cleaner import (
    get_clean_indices,
    get_clean_indices_per_class,
    print_cleaning_summary,
)
from src.synthesis.model import UNet
from src.synthesis.diffusion import GaussianDiffusion

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SYNTHESIS_DIR = os.path.join(PROCESSED_DATA_DIR, "synthesis")
CHECKPOINT_DIR = os.path.join(SYNTHESIS_DIR, "checkpoints")
PER_CLASS_DIR = os.path.join(SYNTHESIS_DIR, "per_class_samples")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 normalization: output is in [0, 1]."""
    mean = torch.tensor(CIFAR10_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def _save_class_grid(images: torch.Tensor, class_name: str, save_path: str, n: int = 25):
    """Save a grid of generated images for one class."""
    n = min(n, len(images))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.8 * rows))
    fig.suptitle(f"Generated: {class_name}", fontsize=14, fontweight="bold")

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue
        img = _unnormalize(images[i]).cpu().permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main synthesis
# ═══════════════════════════════════════════════════════════════════════════
def synthesize(args):
    print("=" * 65)
    print(" Phase 3: Synthetic Data Generation")
    print("=" * 65)

    # Print data cleaning summary
    print_cleaning_summary()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # --- Load model ---
    model = UNet().to(device)
    diffusion = GaussianDiffusion(num_timesteps=1000)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_path = os.path.join(CHECKPOINT_DIR, "ema_model_final.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {ckpt_path}. "
            "Run `python src/train_generator.py` first."
        )

    print(f"  Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # --- Compute per-class deficit ---
    class_stats = get_clean_indices_per_class()
    total_to_generate = sum(s["deficit"] for s in class_stats.values())
    print(f"\n  Total synthetic images to generate: {total_to_generate:,}")

    # --- Generate per class ---
    os.makedirs(PER_CLASS_DIR, exist_ok=True)
    all_synthetic_images = []
    all_synthetic_labels = []

    start_time = time.time()
    for cls_idx in range(10):
        deficit = class_stats[cls_idx]["deficit"]
        name = class_stats[cls_idx]["name"]

        if deficit == 0:
            print(f"  Class {cls_idx} ({name}): no deficit — skipping")
            continue

        print(f"\n  Generating {deficit:,} images for class {cls_idx} ({name}) ...")

        class_images = []
        remaining = deficit
        while remaining > 0:
            batch = min(remaining, args.gen_batch)
            labels = torch.full((batch,), cls_idx, dtype=torch.long, device=device)
            shape = (batch, 3, 32, 32)

            samples = diffusion.p_sample_loop(
                model, shape, labels, device=device, progress=True
            )
            class_images.append(samples.cpu())
            remaining -= batch

        class_images = torch.cat(class_images, dim=0)[:deficit]
        all_synthetic_images.append(class_images)
        all_synthetic_labels.extend([cls_idx] * deficit)

        # Save per-class visualization
        grid_path = os.path.join(PER_CLASS_DIR, f"class_{cls_idx}_{name}.png")
        _save_class_grid(class_images, name, grid_path)
        print(f"  Saved grid: {grid_path}")

    elapsed = time.time() - start_time
    print(f"\n  Generation complete in {elapsed / 60:.1f} minutes")

    # --- Save synthetic images ---
    if all_synthetic_images:
        synthetic_images = torch.cat(all_synthetic_images, dim=0)
        synthetic_labels = torch.tensor(all_synthetic_labels, dtype=torch.long)
    else:
        synthetic_images = torch.empty(0, 3, 32, 32)
        synthetic_labels = torch.empty(0, dtype=torch.long)

    synth_path = os.path.join(SYNTHESIS_DIR, "synthetic_images.pt")
    torch.save({"images": synthetic_images, "labels": synthetic_labels}, synth_path)
    print(f"  Synthetic data saved: {synth_path}")

    # --- Build augmented dataset (clean reals + synthetics) ---
    print("\n  Building augmented dataset ...")
    train_dataset, _ = get_datasets()
    clean_idx = get_clean_indices(len(train_dataset))

    # Extract clean images and labels as tensors
    clean_images = []
    clean_labels = []
    for idx in tqdm(clean_idx, desc="  Loading clean images", leave=False):
        img, lbl = train_dataset[int(idx)]
        clean_images.append(img)
        clean_labels.append(lbl)

    clean_images = torch.stack(clean_images)
    clean_labels = torch.tensor(clean_labels, dtype=torch.long)

    # Combine
    augmented_images = torch.cat([clean_images, synthetic_images], dim=0)
    augmented_labels = torch.cat([clean_labels, synthetic_labels], dim=0)

    aug_path = os.path.join(SYNTHESIS_DIR, "augmented_dataset.pt")
    torch.save({"images": augmented_images, "labels": augmented_labels}, aug_path)
    print(f"  Augmented dataset saved: {aug_path}")
    print(f"    Clean real images:   {len(clean_images):,}")
    print(f"    Synthetic images:    {len(synthetic_images):,}")
    print(f"    Total augmented:     {len(augmented_images):,}")

    # --- Generate synthesis report ---
    _generate_report(class_stats, len(clean_images), len(synthetic_images), elapsed)

    print(f"\n{'=' * 65}")
    print(f" Phase 3 complete!")
    print(f"{'=' * 65}\n")


# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════
def _generate_report(class_stats, n_clean, n_synthetic, gen_time):
    """Generate a Markdown synthesis report."""
    lines = []
    lines.append("# Phase 3: Controlled Synthesis Report\n")
    lines.append("*Generated automatically by `synthesize_data.py`*\n")
    lines.append("---\n")

    # Summary
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Clean real images | {n_clean:,} |")
    lines.append(f"| Synthetic images generated | {n_synthetic:,} |")
    lines.append(f"| Total augmented dataset | {n_clean + n_synthetic:,} |")
    lines.append(f"| Generation time | {gen_time / 60:.1f} minutes |")
    lines.append(f"| Generative model | Conditional DDPM (U-Net) |")
    lines.append(f"| Diffusion steps | 1000 |")
    lines.append("")

    # Per-class breakdown
    lines.append("## Per-Class Breakdown\n")
    lines.append("| Class | Original | Removed | Clean | Synthetic | Final |")
    lines.append("|---|---|---|---|---|---|")
    for cls_idx in range(10):
        s = class_stats[cls_idx]
        final = s["clean"] + s["deficit"]
        lines.append(
            f"| {s['name']} | {s['original']:,} | {s['removed']:,} | "
            f"{s['clean']:,} | {s['deficit']:,} | {final:,} |"
        )
    lines.append("")

    # Per-class sample grids
    lines.append("## Generated Sample Grids\n")
    for cls_idx in range(10):
        name = CIFAR10_CLASSES[cls_idx]
        deficit = class_stats[cls_idx]["deficit"]
        if deficit > 0:
            lines.append(f"### {name} ({deficit:,} generated)\n")
            lines.append(f"![{name}](per_class_samples/class_{cls_idx}_{name}.png)\n")
    lines.append("")

    # Files
    lines.append("## Output Files\n")
    lines.append("| File | Description |")
    lines.append("|---|---|")
    lines.append("| `synthetic_images.pt` | All generated synthetic images + labels |")
    lines.append("| `augmented_dataset.pt` | Clean real + synthetic images (final dataset) |")
    lines.append("| `per_class_samples/` | Visual grids of generated images per class |")
    lines.append("| `training_loss.png` | DDPM training loss curve |")
    lines.append("| `checkpoints/` | Model checkpoints during training |")
    lines.append("")

    report_path = os.path.join(SYNTHESIS_DIR, "synthesis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Synthesis report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic CIFAR-10 images")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to EMA model checkpoint (default: checkpoints/ema_model_final.pt)"
    )
    parser.add_argument(
        "--gen-batch", type=int, default=64,
        help="Batch size for generation (lower = less VRAM)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    synthesize(args)
