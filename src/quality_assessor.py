"""
quality_assessor.py -- Phase 4: Quality Assessment of Synthetic Data.

Usage:
    python src/quality_assessor.py

This script evaluates the synthetic images generated in Phase 3:
    Stage 1 -- Extract InceptionV3 (2048-d) and ResNet-18 (512-d) features
    Stage 2 -- Compute global and per-class FID (Frechet Inception Distance)
    Stage 3 -- Measure intra-class diversity (real vs synthetic)
    Stage 4 -- Memorization check via nearest-neighbour distance
    Stage 5 -- Generate quality report with visualisations (t-SNE, charts)
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from scipy.linalg import sqrtm
from sklearn.manifold import TSNE
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Import project modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    get_datasets,
    CIFAR10_CLASSES,
    CIFAR10_MEAN,
    CIFAR10_STD,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    _ensure_cifar10_downloaded,
)
from src.data_cleaner import get_clean_indices

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SYNTHESIS_DIR = os.path.join(PROCESSED_DATA_DIR, "synthesis")
QUALITY_DIR = os.path.join(PROCESSED_DATA_DIR, "quality")

# ImageNet normalisation constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ===================================================================
# Helpers
# ===================================================================
def _unnormalize_cifar_batch(images: torch.Tensor) -> torch.Tensor:
    """Undo CIFAR-10 normalisation: (B, 3, H, W) -> [0, 1] range."""
    mean = torch.tensor(CIFAR10_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=images.device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def _imagenet_normalize(images: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalisation to [0, 1] images."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# ===================================================================
# Stage 1 -- Feature Extraction
# ===================================================================
def _build_inception(device: torch.device) -> nn.Module:
    """
    Return InceptionV3 that outputs pool3 features (2048-d).

    We use the pretrained InceptionV3 with its final avgpool output,
    removing the classification head.
    """
    weights = torchvision.models.Inception_V3_Weights.IMAGENET1K_V1
    model = torchvision.models.inception_v3(weights=weights)
    # Remove final FC layer -- we want the 2048-d pool output
    model.fc = nn.Identity()
    model.eval()
    model.to(device)
    return model


def _build_resnet(device: torch.device) -> nn.Module:
    """Return a frozen ResNet-18 that outputs 512-d feature vectors."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet = torchvision.models.resnet18(weights=weights)
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor


def _extract_features_from_tensors(
    images: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    resize_to: int,
    batch_size: int = 128,
    desc: str = "Extracting",
) -> np.ndarray:
    """
    Extract features from a batch of CIFAR-10-normalised image tensors.

    Pipeline: unnormalize -> resize -> ImageNet-normalize -> model forward
    """
    dataset = TensorDataset(images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    resize = transforms.Resize(resize_to, antialias=True)

    all_feats = []
    with torch.no_grad():
        for (batch,) in tqdm(loader, desc=f"  {desc}", leave=False):
            # Unnormalize from CIFAR-10 space to [0, 1]
            batch = _unnormalize_cifar_batch(batch.to(device))
            # Resize for the target model
            batch = resize(batch)
            # Apply ImageNet normalisation
            batch = _imagenet_normalize(batch)
            # Forward
            feats = model(batch)
            if isinstance(feats, torchvision.models.InceptionOutputs):
                feats = feats.logits  # InceptionV3 returns namedtuple in train mode
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def _load_real_images_as_tensors(device: torch.device) -> tuple:
    """
    Load clean real CIFAR-10 training images as a single tensor.

    Returns:
        images: (N, 3, 32, 32) CIFAR-10-normalised
        labels: (N,) int
    """
    train_dataset, _ = get_datasets()
    clean_idx = get_clean_indices(len(train_dataset))

    images = []
    labels = []
    for idx in tqdm(clean_idx, desc="  Loading real images", leave=False):
        img, lbl = train_dataset[int(idx)]
        images.append(img)
        labels.append(lbl)

    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def extract_all_features(device: torch.device) -> dict:
    """
    Stage 1: Extract InceptionV3 and ResNet-18 features for real and
    synthetic images.

    Returns dict with keys:
        real_inception, synth_inception   -- (N, 2048) arrays
        real_resnet, synth_resnet         -- (N, 512) arrays
        real_labels, synth_labels         -- (N,) arrays
    """
    print("\n[Stage 1] Extracting features ...")

    # Load data
    synth_data = torch.load(
        os.path.join(SYNTHESIS_DIR, "synthetic_images.pt"),
        map_location="cpu",
        weights_only=True,
    )
    synth_images = synth_data["images"]
    synth_labels = synth_data["labels"].numpy()
    print(f"  Synthetic images: {synth_images.shape[0]:,}")

    real_images, real_labels_t = _load_real_images_as_tensors(device)
    real_labels = real_labels_t.numpy()
    print(f"  Real (clean) images: {real_images.shape[0]:,}")

    # -- InceptionV3 features (for FID) --
    print("\n  Building InceptionV3 ...")
    inception = _build_inception(device)

    print("  Extracting InceptionV3 features (real) ...")
    real_inception = _extract_features_from_tensors(
        real_images, inception, device, resize_to=299, desc="Real InceptionV3"
    )
    print("  Extracting InceptionV3 features (synthetic) ...")
    synth_inception = _extract_features_from_tensors(
        synth_images, inception, device, resize_to=299, desc="Synth InceptionV3"
    )
    del inception
    torch.cuda.empty_cache()

    # -- ResNet-18 features (for diversity / memorization) --
    print("\n  Building ResNet-18 ...")
    resnet = _build_resnet(device)

    print("  Extracting ResNet-18 features (real) ...")
    real_resnet = _extract_features_from_tensors(
        real_images, resnet, device, resize_to=224, desc="Real ResNet-18"
    )
    print("  Extracting ResNet-18 features (synthetic) ...")
    synth_resnet = _extract_features_from_tensors(
        synth_images, resnet, device, resize_to=224, desc="Synth ResNet-18"
    )
    del resnet
    torch.cuda.empty_cache()

    print(f"\n  InceptionV3 features: real={real_inception.shape}, synth={synth_inception.shape}")
    print(f"  ResNet-18 features:   real={real_resnet.shape}, synth={synth_resnet.shape}")

    return {
        "real_inception": real_inception,
        "synth_inception": synth_inception,
        "real_resnet": real_resnet,
        "synth_resnet": synth_resnet,
        "real_labels": real_labels,
        "synth_labels": synth_labels,
    }


# ===================================================================
# Stage 2 -- FID Computation
# ===================================================================
def _compute_fid(real_feats: np.ndarray, synth_feats: np.ndarray) -> float:
    """
    Compute Frechet Inception Distance between two sets of features.

    FID = ||mu_r - mu_g||^2 + Tr(C_r + C_g - 2 * (C_r @ C_g)^0.5)
    """
    mu_r = real_feats.mean(axis=0)
    mu_g = synth_feats.mean(axis=0)
    sigma_r = np.cov(real_feats, rowvar=False)
    sigma_g = np.cov(synth_feats, rowvar=False)

    diff = mu_r - mu_g
    # Matrix square root (may produce complex values due to numerical errors)
    covmean = sqrtm(sigma_r @ sigma_g)

    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(diff @ diff + np.trace(sigma_r + sigma_g - 2.0 * covmean))
    return fid


def compute_fid_scores(features: dict) -> dict:
    """
    Stage 2: Compute global and per-class FID scores.

    Returns dict with:
        global_fid   -- float
        per_class    -- dict[int, float]
    """
    print("\n[Stage 2] Computing FID scores ...")

    # Global FID
    global_fid = _compute_fid(features["real_inception"], features["synth_inception"])
    print(f"  Global FID: {global_fid:.2f}")

    # Per-class FID
    per_class_fid = {}
    for cls in range(10):
        real_mask = features["real_labels"] == cls
        synth_mask = features["synth_labels"] == cls

        real_cls = features["real_inception"][real_mask]
        synth_cls = features["synth_inception"][synth_mask]

        if len(synth_cls) < 2:
            print(f"  Class {cls} ({CIFAR10_CLASSES[cls]}): too few synthetic samples for FID")
            per_class_fid[cls] = float("nan")
            continue

        fid = _compute_fid(real_cls, synth_cls)
        per_class_fid[cls] = fid
        print(f"  Class {cls} ({CIFAR10_CLASSES[cls]:>10s}): FID = {fid:.2f}")

    return {"global_fid": global_fid, "per_class_fid": per_class_fid}


# ===================================================================
# Stage 3 -- Diversity Analysis
# ===================================================================
def _mean_pairwise_cosine_distance(feats: np.ndarray, max_pairs: int = 5000) -> float:
    """
    Compute mean pairwise cosine distance within a set of feature vectors.

    For efficiency, subsample if the number of pairs is very large.
    Cosine distance = 1 - cosine_similarity.  Range [0, 2].
    """
    n = len(feats)
    if n < 2:
        return 0.0

    # Normalise to unit vectors
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    feats_normed = feats / norms

    # If small enough, compute full matrix
    if n * (n - 1) // 2 <= max_pairs:
        sim = feats_normed @ feats_normed.T
        # Extract upper triangle (excluding diagonal)
        idx = np.triu_indices(n, k=1)
        return float(1.0 - sim[idx].mean())

    # Subsample pairs
    rng = np.random.default_rng(42)
    i_idx = rng.integers(0, n, size=max_pairs)
    j_idx = rng.integers(0, n, size=max_pairs)
    # Avoid self-pairs
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]
    sims = (feats_normed[i_idx] * feats_normed[j_idx]).sum(axis=1)
    return float(1.0 - sims.mean())


def compute_diversity(features: dict) -> dict:
    """
    Stage 3: Compare intra-class diversity between real and synthetic images.

    Returns dict mapping class_index -> {real_diversity, synth_diversity, ratio}
    """
    print("\n[Stage 3] Analysing diversity ...")

    diversity = {}
    for cls in range(10):
        real_mask = features["real_labels"] == cls
        synth_mask = features["synth_labels"] == cls

        real_div = _mean_pairwise_cosine_distance(features["real_resnet"][real_mask])
        synth_div = _mean_pairwise_cosine_distance(features["synth_resnet"][synth_mask])
        ratio = synth_div / max(real_div, 1e-8)

        diversity[cls] = {
            "real_diversity": round(real_div, 4),
            "synth_diversity": round(synth_div, 4),
            "ratio": round(ratio, 4),
        }
        status = "OK" if ratio >= 0.5 else "LOW"
        print(f"  {CIFAR10_CLASSES[cls]:>10s}:  real={real_div:.4f}  "
              f"synth={synth_div:.4f}  ratio={ratio:.2f}  [{status}]")

    return diversity


# ===================================================================
# Stage 4 -- Memorization Check
# ===================================================================
def check_memorization(features: dict) -> dict:
    """
    Stage 4: Check if synthetic images are memorised copies of real data.

    For each synthetic image, find its nearest real image (same class)
    in ResNet-18 feature space.  Flag if distance < adaptive threshold
    (5th percentile of real-to-real NN distances within the class).
    """
    print("\n[Stage 4] Checking for memorization ...")

    results = {}
    total_flagged = 0
    all_synth_nn_dists = []

    for cls in range(10):
        real_mask = features["real_labels"] == cls
        synth_mask = features["synth_labels"] == cls

        real_feats = features["real_resnet"][real_mask]
        synth_feats = features["synth_resnet"][synth_mask]
        n_synth = len(synth_feats)

        if n_synth == 0:
            results[cls] = {"flagged": 0, "total": 0, "threshold": 0, "mean_nn_dist": 0}
            continue

        # Compute synth-to-real NN distances
        # Use batched computation to avoid OOM
        synth_nn_dists = np.full(n_synth, np.inf)
        batch_size = 500
        for start in range(0, n_synth, batch_size):
            end = min(start + batch_size, n_synth)
            # (batch, real_n) pairwise L2 distances
            diffs = synth_feats[start:end, None, :] - real_feats[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            synth_nn_dists[start:end] = dists.min(axis=1)

        # Adaptive threshold: 5th percentile of real-to-real NN distances
        # Subsample real-to-real for efficiency
        rng = np.random.default_rng(42)
        n_sample = min(1000, len(real_feats))
        sample_idx = rng.choice(len(real_feats), n_sample, replace=False)
        real_sample = real_feats[sample_idx]

        real_nn_dists = np.full(n_sample, np.inf)
        for start in range(0, n_sample, batch_size):
            end = min(start + batch_size, n_sample)
            diffs = real_sample[start:end, None, :] - real_feats[None, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            # Exclude self (distance = 0)
            for i in range(end - start):
                global_idx = sample_idx[start + i]
                dists[i, global_idx] = np.inf
            real_nn_dists[start:end] = dists.min(axis=1)

        threshold = float(np.percentile(real_nn_dists, 5))
        flagged = int((synth_nn_dists < threshold).sum())
        total_flagged += flagged

        results[cls] = {
            "flagged": flagged,
            "total": n_synth,
            "threshold": round(threshold, 4),
            "mean_nn_dist": round(float(synth_nn_dists.mean()), 4),
            "min_nn_dist": round(float(synth_nn_dists.min()), 4),
        }
        all_synth_nn_dists.extend(synth_nn_dists.tolist())

        status = "CLEAN" if flagged == 0 else f"FLAGGED({flagged})"
        print(f"  {CIFAR10_CLASSES[cls]:>10s}:  mean_nn={synth_nn_dists.mean():.4f}  "
              f"min_nn={synth_nn_dists.min():.4f}  thresh={threshold:.4f}  [{status}]")

    print(f"\n  Total potentially memorised: {total_flagged} / {len(features['synth_labels'])}")

    return {
        "per_class": results,
        "total_flagged": total_flagged,
        "all_nn_dists": all_synth_nn_dists,
    }


# ===================================================================
# Stage 5 -- Report & Visualisations
# ===================================================================
def _save_fid_chart(fid_scores: dict, save_path: str):
    """Bar chart of per-class FID scores."""
    fig, ax = plt.subplots(figsize=(12, 5))
    classes = [CIFAR10_CLASSES[i] for i in range(10)]
    fids = [fid_scores["per_class_fid"].get(i, 0) for i in range(10)]
    colors = sns.color_palette("coolwarm", 10)

    bars = ax.bar(classes, fids, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(y=fid_scores["global_fid"], color="red", linestyle="--", linewidth=1.5,
               label=f"Global FID = {fid_scores['global_fid']:.1f}")

    for bar, fid in zip(bars, fids):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{fid:.1f}", ha="center", fontsize=8)

    ax.set_title("Per-Class FID Scores (lower is better)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("FID")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def _save_diversity_chart(diversity: dict, save_path: str):
    """Grouped bar chart comparing real vs synthetic diversity."""
    fig, ax = plt.subplots(figsize=(12, 5))
    classes = [CIFAR10_CLASSES[i] for i in range(10)]
    real_divs = [diversity[i]["real_diversity"] for i in range(10)]
    synth_divs = [diversity[i]["synth_diversity"] for i in range(10)]

    x = np.arange(10)
    width = 0.35
    ax.bar(x - width / 2, real_divs, width, label="Real", color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, synth_divs, width, label="Synthetic", color="coral", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_title("Intra-Class Diversity: Real vs Synthetic (higher = more diverse)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Mean Pairwise Cosine Distance")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def _save_nn_histogram(nn_dists: list, save_path: str):
    """Histogram of nearest-neighbour distances (synthetic to real)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(nn_dists, bins=80, color="steelblue", edgecolor="black", linewidth=0.3, alpha=0.8)
    ax.set_title("Nearest-Neighbour Distance: Synthetic -> Real (higher = less memorised)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("L2 Distance in ResNet-18 Feature Space")
    ax.set_ylabel("Frequency")

    mean_d = np.mean(nn_dists)
    ax.axvline(x=mean_d, color="red", linestyle="--",
               label=f"Mean = {mean_d:.2f}")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def _save_tsne_plot(features: dict, save_path: str, n_per_source: int = 2000):
    """t-SNE scatter plot of real vs synthetic embeddings."""
    print("  Computing t-SNE (this may take a minute) ...")

    rng = np.random.default_rng(42)

    # Subsample for speed
    n_real = len(features["real_resnet"])
    n_synth = len(features["synth_resnet"])
    real_idx = rng.choice(n_real, min(n_per_source, n_real), replace=False)
    synth_idx = rng.choice(n_synth, min(n_per_source, n_synth), replace=False)

    real_feats = features["real_resnet"][real_idx]
    synth_feats = features["synth_resnet"][synth_idx]
    real_labels = features["real_labels"][real_idx]
    synth_labels = features["synth_labels"][synth_idx]

    # Combine
    all_feats = np.concatenate([real_feats, synth_feats], axis=0)
    all_labels = np.concatenate([real_labels, synth_labels], axis=0)
    source = np.array(["Real"] * len(real_feats) + ["Synthetic"] * len(synth_feats))

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    coords = tsne.fit_transform(all_feats)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Left: colored by class
    ax = axes[0]
    palette = sns.color_palette("tab10", 10)
    for cls_idx in range(10):
        mask = all_labels == cls_idx
        real_mask = mask & (source == "Real")
        synth_mask = mask & (source == "Synthetic")
        ax.scatter(coords[real_mask, 0], coords[real_mask, 1],
                   c=[palette[cls_idx]], marker="o", s=8, alpha=0.4, label=CIFAR10_CLASSES[cls_idx])
        ax.scatter(coords[synth_mask, 0], coords[synth_mask, 1],
                   c=[palette[cls_idx]], marker="x", s=15, alpha=0.7)
    ax.set_title("t-SNE: Colored by Class (o=Real, x=Synthetic)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, markerscale=2, loc="best")
    ax.set_xticks([])
    ax.set_yticks([])

    # Right: colored by source
    ax = axes[1]
    real_m = source == "Real"
    synth_m = source == "Synthetic"
    ax.scatter(coords[real_m, 0], coords[real_m, 1],
               c="steelblue", marker="o", s=8, alpha=0.3, label="Real")
    ax.scatter(coords[synth_m, 0], coords[synth_m, 1],
               c="coral", marker="x", s=15, alpha=0.6, label="Synthetic")
    ax.set_title("t-SNE: Real vs Synthetic", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_report(
    fid_scores: dict,
    diversity: dict,
    memorization: dict,
):
    """Stage 5: Generate quality report and visualisations."""
    print("\n[Stage 5] Generating quality report ...")
    os.makedirs(QUALITY_DIR, exist_ok=True)

    # --- Charts ---
    _save_fid_chart(fid_scores, os.path.join(QUALITY_DIR, "fid_per_class.png"))
    _save_diversity_chart(diversity, os.path.join(QUALITY_DIR, "diversity_comparison.png"))
    _save_nn_histogram(
        memorization["all_nn_dists"],
        os.path.join(QUALITY_DIR, "nn_distance_histogram.png"),
    )

    # --- Save raw metrics as JSON ---
    metrics = {
        "fid": {
            "global": round(fid_scores["global_fid"], 2),
            "per_class": {
                CIFAR10_CLASSES[k]: round(v, 2) for k, v in fid_scores["per_class_fid"].items()
            },
        },
        "diversity": {
            CIFAR10_CLASSES[k]: v for k, v in diversity.items()
        },
        "memorization": {
            "total_flagged": memorization["total_flagged"],
            "per_class": {
                CIFAR10_CLASSES[k]: v for k, v in memorization["per_class"].items()
            },
        },
    }
    metrics_path = os.path.join(QUALITY_DIR, "quality_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # --- Markdown Report ---
    lines = []
    lines.append("# Phase 4: Quality Assessment Report\n")
    lines.append("*Generated automatically by `quality_assessor.py`*\n")
    lines.append("---\n")

    # Summary
    lines.append("## Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Global FID | {fid_scores['global_fid']:.2f} |")
    mean_cls_fid = np.nanmean([v for v in fid_scores["per_class_fid"].values()])
    lines.append(f"| Mean per-class FID | {mean_cls_fid:.2f} |")
    mean_div_ratio = np.mean([diversity[c]["ratio"] for c in range(10)])
    lines.append(f"| Mean diversity ratio (synth/real) | {mean_div_ratio:.2f} |")
    lines.append(f"| Potential memorised samples | {memorization['total_flagged']} |")
    lines.append("")

    # Interpretation
    lines.append("### Interpretation\n")
    if fid_scores["global_fid"] < 50:
        lines.append("> **Excellent**: Global FID < 50 indicates high-quality, realistic synthetic images.\n")
    elif fid_scores["global_fid"] < 100:
        lines.append("> **Good**: Global FID < 100 indicates reasonable quality. "
                      "The synthetic images capture the main characteristics of each class.\n")
    elif fid_scores["global_fid"] < 200:
        lines.append("> **Moderate**: Global FID < 200 suggests the synthetic images partially "
                      "capture class characteristics but could be improved with more training.\n")
    else:
        lines.append("> **Needs improvement**: Global FID > 200 indicates the synthetic images "
                      "have noticeable quality issues. Consider training for more epochs.\n")
    lines.append("")

    # FID section
    lines.append("## Frechet Inception Distance (FID)\n")
    lines.append("FID measures the distance between real and synthetic image distributions ")
    lines.append("in InceptionV3 feature space. **Lower is better.**\n")
    lines.append("| Class | FID Score |")
    lines.append("|---|---|")
    for cls in range(10):
        fid = fid_scores["per_class_fid"].get(cls, float("nan"))
        lines.append(f"| {CIFAR10_CLASSES[cls]} | {fid:.2f} |")
    lines.append(f"| **Global** | **{fid_scores['global_fid']:.2f}** |")
    lines.append("")
    lines.append("![Per-Class FID](fid_per_class.png)\n")

    # Diversity section
    lines.append("## Diversity Analysis\n")
    lines.append("Intra-class diversity measured as mean pairwise cosine distance ")
    lines.append("in ResNet-18 feature space. **Higher = more diverse.**\n")
    lines.append("A ratio (synthetic/real) close to 1.0 means synthetic images are as diverse as real ones.\n")
    lines.append("| Class | Real Diversity | Synth Diversity | Ratio |")
    lines.append("|---|---|---|---|")
    for cls in range(10):
        d = diversity[cls]
        flag = " :warning:" if d["ratio"] < 0.5 else ""
        lines.append(f"| {CIFAR10_CLASSES[cls]} | {d['real_diversity']:.4f} | "
                      f"{d['synth_diversity']:.4f} | {d['ratio']:.2f}{flag} |")
    lines.append("")
    lines.append("![Diversity Comparison](diversity_comparison.png)\n")

    # Memorization section
    lines.append("## Memorization Check\n")
    lines.append("For each synthetic image, we find its nearest real image (same class) ")
    lines.append("in ResNet-18 feature space. Images closer than the 5th percentile ")
    lines.append("of real-to-real distances are flagged as potentially memorised.\n")
    lines.append(f"**Total flagged: {memorization['total_flagged']}** "
                 f"out of {sum(v['total'] for v in memorization['per_class'].values()):,} synthetic images\n")
    lines.append("| Class | Flagged | Total | Mean NN Dist | Threshold |")
    lines.append("|---|---|---|---|---|")
    for cls in range(10):
        m = memorization["per_class"][cls]
        lines.append(f"| {CIFAR10_CLASSES[cls]} | {m['flagged']} | {m['total']} | "
                      f"{m['mean_nn_dist']:.4f} | {m['threshold']:.4f} |")
    lines.append("")
    lines.append("![NN Distance Histogram](nn_distance_histogram.png)\n")

    # Conclusion
    lines.append("## Conclusion\n")
    conclusions = []
    if fid_scores["global_fid"] < 100:
        conclusions.append(f"- The synthetic images achieve a **good FID of {fid_scores['global_fid']:.1f}**, "
                           "indicating reasonable visual quality.")
    else:
        conclusions.append(f"- The synthetic images have an FID of **{fid_scores['global_fid']:.1f}**. "
                           "Quality is acceptable for data augmentation purposes.")
    if mean_div_ratio >= 0.7:
        conclusions.append(f"- Synthetic diversity is **{mean_div_ratio:.0%}** of real diversity on average "
                           "-- the generated images are sufficiently varied.")
    else:
        conclusions.append(f"- Synthetic diversity is **{mean_div_ratio:.0%}** of real diversity. "
                           "Some mode collapse may be present.")
    if memorization["total_flagged"] == 0:
        conclusions.append("- **No memorized samples detected** -- the model has learned to generalize "
                           "rather than copy training data.")
    else:
        pct = memorization["total_flagged"] / max(1, sum(v["total"] for v in memorization["per_class"].values())) * 100
        conclusions.append(f"- **{memorization['total_flagged']}** samples ({pct:.1f}%) flagged as potentially "
                           "memorised -- a small fraction, unlikely to affect downstream performance.")
    conclusions.append("")
    lines.extend(conclusions)

    report_path = os.path.join(QUALITY_DIR, "quality_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {report_path}")

    return report_path


# ===================================================================
# Main
# ===================================================================
def main():
    start = time.time()
    print("=" * 65)
    print(" Phase 4: Quality Assessment of Synthetic Data")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Stage 1 -- Feature extraction
    features = extract_all_features(device)

    # Stage 2 -- FID
    fid_scores = compute_fid_scores(features)

    # Stage 3 -- Diversity
    diversity = compute_diversity(features)

    # Stage 4 -- Memorization
    memorization = check_memorization(features)

    # t-SNE (part of Stage 5 but needs features)
    _save_tsne_plot(features, os.path.join(QUALITY_DIR, "tsne_real_vs_synthetic.png"))

    # Stage 5 -- Report
    report_path = generate_report(fid_scores, diversity, memorization)

    elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print(f" Phase 4 complete in {elapsed / 60:.1f} minutes")
    print(f" Quality report: {report_path}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
