"""
train_generator.py — Train the class-conditional DDPM on cleaned CIFAR-10.

Usage:
    python src/train_generator.py                        # defaults (100 epochs)
    python src/train_generator.py --epochs 1 --batch-size 32  # quick smoke test

This script:
    1. Loads the cleaned CIFAR-10 training subset (excluding flagged samples)
    2. Trains a class-conditional U-Net denoiser with DDPM
    3. Maintains an EMA copy of the model weights for better generation
    4. Saves checkpoints + sample grids every N epochs
    5. Plots a training loss curve
"""

import os
import sys
import copy
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import PROCESSED_DATA_DIR, CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD
from src.data_cleaner import get_clean_dataset, print_cleaning_summary
from src.synthesis.model import UNet
from src.synthesis.diffusion import GaussianDiffusion

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SYNTHESIS_DIR = os.path.join(PROCESSED_DATA_DIR, "synthesis")
CHECKPOINT_DIR = os.path.join(SYNTHESIS_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(SYNTHESIS_DIR, "samples")


# ═══════════════════════════════════════════════════════════════════════════
# EMA helper
# ═══════════════════════════════════════════════════════════════════════════
class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


# ═══════════════════════════════════════════════════════════════════════════
# Sample grid generation
# ═══════════════════════════════════════════════════════════════════════════
def _unnormalize(img: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 normalization: output is in [0, 1]."""
    mean = torch.tensor(CIFAR10_MEAN, device=img.device).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=img.device).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def save_sample_grid(
    model: nn.Module,
    diffusion: GaussianDiffusion,
    device: torch.device,
    save_path: str,
    n_per_class: int = 4,
):
    """
    Generate a grid of samples (n_per_class per class) and save as PNG.
    Grid layout: 10 columns (classes) × n_per_class rows.
    """
    model.eval()
    num_classes = 10
    total = num_classes * n_per_class

    # Build class labels: [0,0,..,0, 1,1,..,1, ..., 9,9,..,9]
    labels = torch.arange(num_classes).repeat_interleave(n_per_class).to(device)
    shape = (total, 3, 32, 32)

    samples = diffusion.p_sample_loop(model, shape, labels, device=device, progress=True)
    samples = _unnormalize(samples).cpu()

    # Arrange as grid
    fig, axes = plt.subplots(n_per_class, num_classes, figsize=(20, 2.2 * n_per_class))
    fig.suptitle("Generated CIFAR-10 Samples (1 column per class)", fontsize=14, fontweight="bold")

    for cls in range(num_classes):
        axes[0, cls].set_title(CIFAR10_CLASSES[cls], fontsize=9)
        for row in range(n_per_class):
            idx = cls * n_per_class + row
            img = samples[idx].permute(1, 2, 0).numpy()
            axes[row, cls].imshow(img)
            axes[row, cls].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Sample grid saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════
def train(args):
    # Print cleaning summary first
    print_cleaning_summary()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # --- Data ---
    clean_dataset = get_clean_dataset()
    loader = DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    print(f"  Clean dataset size: {len(clean_dataset):,}")
    print(f"  Batches per epoch:  {len(loader):,}")

    # --- Model & Diffusion ---
    model = UNet().to(device)
    diffusion = GaussianDiffusion(num_timesteps=1000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  U-Net parameters:   {n_params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # --- EMA ---
    ema = EMA(model, decay=0.9999)

    # --- Create directories ---
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    # --- Training ---
    print(f"\n{'=' * 65}")
    print(f" Phase 3: Training Conditional DDPM ({args.epochs} epochs)")
    print(f"{'=' * 65}\n")

    epoch_losses = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(loader, desc=f"  Epoch {epoch:>3d}/{args.epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            # Random timesteps
            t = torch.randint(0, diffusion.T, (images.shape[0],), device=device)

            loss = diffusion.training_loss(model, images, t, labels)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema.update(model)

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = running_loss / len(loader)
        epoch_losses.append(avg_loss)
        lr_now = scheduler.get_last_lr()[0]
        elapsed = time.time() - start_time

        print(f"  Epoch {epoch:>3d}/{args.epochs}  |  loss={avg_loss:.4f}  |  "
              f"lr={lr_now:.2e}  |  elapsed={elapsed:.0f}s")

        # --- Periodic checkpoint + samples ---
        if epoch % args.save_every == 0 or epoch == args.epochs:
            # Save checkpoint
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

            # Generate sample grid (using EMA model)
            sample_path = os.path.join(SAMPLE_DIR, f"epoch_{epoch}_samples.png")
            save_sample_grid(ema.shadow, diffusion, device, sample_path)

    # --- Save final EMA model ---
    final_path = os.path.join(CHECKPOINT_DIR, "ema_model_final.pt")
    torch.save(ema.state_dict(), final_path)
    print(f"\n  Final EMA model saved: {final_path}")

    # --- Training loss curve ---
    loss_plot_path = os.path.join(SYNTHESIS_DIR, "training_loss.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, color="steelblue", linewidth=1.5)
    ax.set_title("DDPM Training Loss", fontsize=13)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=120)
    plt.close()
    print(f"  Training loss curve saved: {loss_plot_path}")

    total_time = time.time() - start_time
    print(f"\n{'=' * 65}")
    print(f" Training complete in {total_time / 60:.1f} minutes")
    print(f"{'=' * 65}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional DDPM on cleaned CIFAR-10")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
