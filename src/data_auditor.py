"""
data_auditor.py — Phase 2: Automated Data Quality Auditing for CIFAR-10.

Usage:
    python src/data_auditor.py

This script:
  1. Extracts deep features from CIFAR-10 training images using a pretrained ResNet-18
  2. Generates cross-validated predicted probabilities for Cleanlab
  3. Detects likely mislabeled samples using Cleanlab (confident learning)
  4. Detects outlier/anomalous images using Isolation Forest
  5. Analyses class imbalance
  6. Generates an automated Markdown audit report with visualisations
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import IsolationForest
# pyrefly: ignore [missing-import]
from cleanlab.filter import find_label_issues
# pyrefly: ignore [missing-import]
from cleanlab.rank import get_label_quality_scores
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
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
AUDIT_DIR = os.path.join(PROCESSED_DATA_DIR, "audit")

# ImageNet normalisation constants (for ResNet feature extraction)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════════════════
# Stage 1 — Feature Extraction
# ═══════════════════════════════════════════════════════════════════════════
def _build_feature_extractor(device: torch.device) -> nn.Module:
    """Return a frozen ResNet-18 that outputs 512-d feature vectors."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    resnet = torchvision.models.resnet18(weights=weights)
    # Remove the final classification head → output is 512-d
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
    feature_extractor.eval()
    feature_extractor.to(device)
    return feature_extractor


def _get_imagenet_transform():
    """Transform for CIFAR-10 images to match ImageNet ResNet input."""
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def extract_features(device: torch.device, batch_size: int = 256) -> tuple:
    """
    Extract 512-d ResNet-18 features for all CIFAR-10 training images.

    Returns:
        features  – np.ndarray of shape (N, 512)
        labels    – np.ndarray of shape (N,)
    """
    print("\n[Stage 1] Extracting ResNet-18 features ...")

    # Load raw CIFAR-10 with ImageNet-compatible transform
    from src.data_loader import RAW_DATA_DIR, _ensure_cifar10_downloaded
    _ensure_cifar10_downloaded(RAW_DATA_DIR)

    dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR,
        train=True,
        download=False,
        transform=_get_imagenet_transform(),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_feature_extractor(device)
    all_features = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Extracting features"):
            images = images.to(device)
            feats = model(images).cpu().numpy()
            all_features.append(feats)
            all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Check for injected noise — if present, override labels
    noisy_path = os.path.join(PROCESSED_DATA_DIR, "noise", "noisy_labels.json")
    if os.path.isfile(noisy_path):
        with open(noisy_path) as f:
            noisy_data = json.load(f)
        labels = np.array(noisy_data["noisy_labels"], dtype=labels.dtype)
        print(f"  [NOISE] Loaded noisy labels ({noisy_data['noise_rate']*100:.0f}% corrupted)")

    print(f"  Feature matrix shape: {features.shape}")
    return features, labels


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 — Cross-Validated Predicted Probabilities
# ═══════════════════════════════════════════════════════════════════════════
def compute_pred_probs(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Train a logistic-regression classifier on the ResNet features and
    return out-of-fold predicted probabilities via 3-fold stratified CV.
    """
    print("\n[Stage 2] Computing cross-validated predicted probabilities ...")
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    pred_probs = cross_val_predict(
        clf, features, labels, cv=cv, method="predict_proba", n_jobs=-1
    )
    print(f"  pred_probs shape: {pred_probs.shape}")
    return pred_probs


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 — Label Issue Detection (Cleanlab)
# ═══════════════════════════════════════════════════════════════════════════
def detect_label_issues(labels: np.ndarray, pred_probs: np.ndarray) -> dict:
    """
    Use Cleanlab's confident-learning to find likely mislabelled samples.

    Returns a dict with:
        issue_indices  – indices of flagged samples
        quality_scores – per-sample label quality score (0=worst, 1=best)
        suggested_labels – the most likely correct label per flagged sample
    """
    print("\n[Stage 3] Detecting label issues with Cleanlab ...")

    issue_mask = find_label_issues(labels, pred_probs, return_indices_ranked_by="self_confidence")
    quality_scores = get_label_quality_scores(labels, pred_probs)

    # Suggested label = argmax of predicted probabilities for flagged samples
    suggested_labels = pred_probs.argmax(axis=1)

    # issue_mask from find_label_issues with return_indices_ranked_by returns
    # ranked indices directly (not a boolean mask)
    issue_indices = issue_mask

    print(f"  Total label issues found: {len(issue_indices)}")
    print(f"  Label quality score range: [{quality_scores.min():.4f}, {quality_scores.max():.4f}]")

    return {
        "issue_indices": issue_indices,
        "quality_scores": quality_scores,
        "suggested_labels": suggested_labels,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 4 — Outlier / Anomaly Detection
# ═══════════════════════════════════════════════════════════════════════════
def detect_outliers(features: np.ndarray, contamination: float = 0.05) -> dict:
    """
    Fit an Isolation Forest on the feature embeddings to find anomalous images.

    Returns a dict with:
        outlier_indices – indices flagged as outliers
        anomaly_scores  – per-sample anomaly score (lower = more anomalous)
    """
    print(f"\n[Stage 4] Detecting outliers (contamination={contamination}) ...")
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
        n_jobs=-1,
    )
    preds = iso_forest.fit_predict(features)           # +1 = inlier, -1 = outlier
    anomaly_scores = iso_forest.decision_function(features)

    outlier_indices = np.where(preds == -1)[0]
    print(f"  Outliers found: {len(outlier_indices)}")
    return {
        "outlier_indices": outlier_indices,
        "anomaly_scores": anomaly_scores,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 — Class Imbalance Analysis
# ═══════════════════════════════════════════════════════════════════════════
def analyse_class_imbalance(labels: np.ndarray) -> dict:
    """Compute class distribution stats."""
    print("\n[Stage 5] Analysing class imbalance ...")
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    distribution = {
        CIFAR10_CLASSES[int(c)]: int(cnt)
        for c, cnt in zip(unique, counts)
    }
    imbalance_ratio = float(counts.max()) / float(counts.min())
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.4f}")
    return {
        "distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
        "total_samples": total,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6 — Report Generation & Visualisations
# ═══════════════════════════════════════════════════════════════════════════
def _unnormalize_cifar(img_tensor: torch.Tensor) -> np.ndarray:
    """Undo CIFAR-10 normalisation and return HWC uint8 numpy array."""
    img = img_tensor.clone()
    for ch in range(3):
        img[ch] = img[ch] * CIFAR10_STD[ch] + CIFAR10_MEAN[ch]
    img = img.clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _save_flagged_grid(
    dataset, indices, labels, suggested, title, save_path, n=25
):
    """Save a grid of flagged sample images."""
    n = min(n, len(indices))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.8 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
    for i, ax in enumerate(axes_flat):
        if i >= n:
            ax.axis("off")
            continue
        idx = indices[i]
        img, given_label = dataset[idx]
        img_np = _unnormalize_cifar(img)
        ax.imshow(img_np)
        given_name = CIFAR10_CLASSES[given_label]
        sugg_name = CIFAR10_CLASSES[suggested[idx]] if suggested is not None else ""
        if suggested is not None and given_label != suggested[idx]:
            ax.set_title(f"#{idx}\n{given_name} → {sugg_name}", fontsize=8, color="red")
        else:
            ax.set_title(f"#{idx}\n{given_name}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def _save_class_distribution_chart(distribution: dict, save_path: str):
    """Bar chart of class distribution."""
    fig, ax = plt.subplots(figsize=(10, 5))
    classes = list(distribution.keys())
    counts = list(distribution.values())
    colors = sns.color_palette("viridis", len(classes))
    ax.bar(classes, counts, color=colors)
    ax.set_title("CIFAR-10 Training Set — Class Distribution", fontsize=13)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    for i, (cls, cnt) in enumerate(zip(classes, counts)):
        ax.text(i, cnt + 30, str(cnt), ha="center", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def _save_quality_score_histogram(scores: np.ndarray, save_path: str):
    """Histogram of per-sample label quality scores."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=100, color="steelblue", edgecolor="black", linewidth=0.3)
    ax.set_title("Label Quality Score Distribution", fontsize=13)
    ax.set_xlabel("Label Quality Score (0 = worst, 1 = best)")
    ax.set_ylabel("Frequency")
    ax.axvline(x=np.percentile(scores, 5), color="red", linestyle="--",
               label=f"5th percentile ({np.percentile(scores, 5):.3f})")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_report(
    label_results: dict,
    outlier_results: dict,
    imbalance_results: dict,
    labels: np.ndarray,
):
    """Generate the full audit report + visualisations."""
    print("\n[Stage 6] Generating audit report ...")
    os.makedirs(AUDIT_DIR, exist_ok=True)

    # --- Load the original (CIFAR-normalised) dataset for visualisations ---
    train_dataset, _ = get_datasets()

    # --- Visualisations ---
    # 1. Flagged mislabeled samples grid
    mislabel_grid_path = os.path.join(AUDIT_DIR, "mislabeled_samples.png")
    _save_flagged_grid(
        train_dataset,
        label_results["issue_indices"],
        labels,
        label_results["suggested_labels"],
        "Top Suspected Mislabeled Samples",
        mislabel_grid_path,
        n=25,
    )

    # 2. Outlier samples grid
    outlier_grid_path = os.path.join(AUDIT_DIR, "outlier_samples.png")
    # Sort outliers by anomaly score (most anomalous first)
    sorted_outlier_idx = outlier_results["outlier_indices"][
        np.argsort(outlier_results["anomaly_scores"][outlier_results["outlier_indices"]])
    ]
    _save_flagged_grid(
        train_dataset,
        sorted_outlier_idx,
        labels,
        None,
        "Top Outlier / Anomalous Samples",
        outlier_grid_path,
        n=25,
    )

    # 3. Class distribution chart
    dist_chart_path = os.path.join(AUDIT_DIR, "class_distribution.png")
    _save_class_distribution_chart(imbalance_results["distribution"], dist_chart_path)

    # 4. Quality score histogram
    quality_hist_path = os.path.join(AUDIT_DIR, "quality_score_histogram.png")
    _save_quality_score_histogram(label_results["quality_scores"], quality_hist_path)

    # --- Save flagged indices as JSON ---
    flagged_data = {
        "mislabeled_indices": [int(i) for i in label_results["issue_indices"]],
        "outlier_indices": [int(i) for i in outlier_results["outlier_indices"]],
        "total_mislabeled": len(label_results["issue_indices"]),
        "total_outliers": len(outlier_results["outlier_indices"]),
    }
    flagged_path = os.path.join(AUDIT_DIR, "flagged_indices.json")
    with open(flagged_path, "w") as f:
        json.dump(flagged_data, f, indent=2)
    print(f"  Saved: {flagged_path}")

    # --- Markdown Report ---
    report_lines = []
    report_lines.append("# CIFAR-10 Data Audit Report\n")
    report_lines.append(f"*Generated automatically by `data_auditor.py`*\n")
    report_lines.append("---\n")

    # Summary
    report_lines.append("## Summary\n")
    report_lines.append(f"| Metric | Value |")
    report_lines.append(f"|---|---|")
    report_lines.append(f"| Total training samples | {imbalance_results['total_samples']:,} |")
    report_lines.append(f"| Suspected mislabeled samples | {len(label_results['issue_indices']):,} |")
    report_lines.append(f"| Outlier samples | {len(outlier_results['outlier_indices']):,} |")
    report_lines.append(f"| Class imbalance ratio (max/min) | {imbalance_results['imbalance_ratio']:.4f} |")
    report_lines.append(f"| Mean label quality score | {label_results['quality_scores'].mean():.4f} |")
    report_lines.append(f"| Min label quality score | {label_results['quality_scores'].min():.4f} |")
    report_lines.append("")

    # Class distribution
    report_lines.append("## Class Distribution\n")
    report_lines.append("| Class | Count | Percentage |")
    report_lines.append("|---|---|---|")
    total = imbalance_results["total_samples"]
    for cls, cnt in imbalance_results["distribution"].items():
        pct = cnt / total * 100
        report_lines.append(f"| {cls} | {cnt:,} | {pct:.1f}% |")
    report_lines.append("")
    report_lines.append(f"![Class Distribution](class_distribution.png)\n")

    # Label quality
    report_lines.append("## Label Quality Analysis\n")
    report_lines.append(f"Cleanlab's confident-learning algorithm identified "
                        f"**{len(label_results['issue_indices']):,}** samples with suspected label issues.\n")
    report_lines.append(f"![Quality Score Histogram](quality_score_histogram.png)\n")

    # Top mislabeled
    report_lines.append("### Top Suspected Mislabeled Samples\n")
    report_lines.append("The grid below shows the top 25 samples most likely to be mislabeled. "
                        "Red titles show `given_label → suggested_label`.\n")
    report_lines.append(f"![Mislabeled Samples](mislabeled_samples.png)\n")

    # Top 10 mislabeled table
    report_lines.append("| Rank | Index | Given Label | Suggested Label | Quality Score |")
    report_lines.append("|---|---|---|---|---|")
    top_n = min(10, len(label_results["issue_indices"]))
    for rank, idx in enumerate(label_results["issue_indices"][:top_n], 1):
        given = CIFAR10_CLASSES[labels[idx]]
        suggested = CIFAR10_CLASSES[label_results["suggested_labels"][idx]]
        score = label_results["quality_scores"][idx]
        report_lines.append(f"| {rank} | {idx} | {given} | {suggested} | {score:.4f} |")
    report_lines.append("")

    # Outliers
    report_lines.append("## Outlier / Anomaly Detection\n")
    report_lines.append(f"Isolation Forest flagged **{len(outlier_results['outlier_indices']):,}** "
                        f"samples as anomalous (contamination=0.05).\n")
    report_lines.append(f"![Outlier Samples](outlier_samples.png)\n")

    # Conclusion
    report_lines.append("## Conclusion\n")
    if len(label_results["issue_indices"]) > 0:
        report_lines.append(f"- **{len(label_results['issue_indices']):,}** samples have suspected label issues "
                            f"and should be reviewed or removed before training.")
    if len(outlier_results["outlier_indices"]) > 0:
        report_lines.append(f"- **{len(outlier_results['outlier_indices']):,}** outlier samples were detected "
                            f"that deviate significantly from their class distribution.")
    if imbalance_results["imbalance_ratio"] < 1.1:
        report_lines.append("- The dataset is **well-balanced** across all 10 classes.")
    else:
        report_lines.append(f"- The dataset has a class imbalance ratio of "
                            f"**{imbalance_results['imbalance_ratio']:.2f}** — consider resampling.")
    report_lines.append("")

    report_path = os.path.join(AUDIT_DIR, "audit_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"  Saved: {report_path}")

    return report_path


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    start = time.time()
    print("=" * 65)
    print(" Phase 2: Automated Data Quality Auditing")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Stage 1 — Feature extraction
    features, labels = extract_features(device)

    # Stage 2 — Cross-validated predicted probabilities
    pred_probs = compute_pred_probs(features, labels)

    # Stage 3 — Label issue detection
    label_results = detect_label_issues(labels, pred_probs)

    # Stage 4 — Outlier detection
    outlier_results = detect_outliers(features, contamination=0.05)

    # Stage 5 — Class imbalance analysis
    imbalance_results = analyse_class_imbalance(labels)

    # Stage 6 — Report generation
    report_path = generate_report(label_results, outlier_results, imbalance_results, labels)

    elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print(f" Phase 2 complete in {elapsed:.1f}s")
    print(f" Audit report: {report_path}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
