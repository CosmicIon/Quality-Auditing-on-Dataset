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
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import scipy.linalg
from scipy.spatial.distance import mahalanobis
import cv2
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
# Stage 4 — Ambiguous/Confusing Images (Prediction Entropy)
# ═══════════════════════════════════════════════════════════════════════════
def detect_ambiguous_images(pred_probs: np.ndarray, threshold_percentile: int = 95) -> dict:
    """
    Flag images where the classifier is most uncertain using prediction entropy.
    """
    print(f"\n[Stage 4] Detecting ambiguous images (entropy percentile >= {threshold_percentile}) ...")
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10), axis=1)
    threshold = np.percentile(entropy, threshold_percentile)
    ambiguous_indices = np.where(entropy >= threshold)[0]
    
    # Sort by entropy (most ambiguous first)
    sorted_ambiguous = ambiguous_indices[np.argsort(entropy[ambiguous_indices])[::-1]]
    
    print(f"  Ambiguous images found: {len(sorted_ambiguous)}")
    return {
        "ambiguous_indices": sorted_ambiguous,
        "entropy_scores": entropy,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5 — Blurry Images (Laplacian Variance)
# ═══════════════════════════════════════════════════════════════════════════
def detect_blurry_images(dataset_data: np.ndarray, threshold_percentile: int = 5) -> dict:
    """
    Flag images that lack high-frequency details (blurry) using Laplacian variance.
    dataset_data: np.ndarray of shape (N, H, W, 3) in uint8 format (CIFAR-10 data)
    """
    print(f"\n[Stage 5] Detecting blurry images (Laplacian variance percentile <= {threshold_percentile}) ...")
    
    laplacian_vars = np.zeros(len(dataset_data))
    for i, img_rgb in enumerate(dataset_data):
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        laplacian_vars[i] = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        
    threshold = np.percentile(laplacian_vars, threshold_percentile)
    blurry_indices = np.where(laplacian_vars <= threshold)[0]
    
    # Sort by blurriness (lowest variance first = most blurry)
    sorted_blurry = blurry_indices[np.argsort(laplacian_vars[blurry_indices])]
    
    print(f"  Blurry images found: {len(sorted_blurry)}")
    return {
        "blurry_indices": sorted_blurry,
        "laplacian_vars": laplacian_vars,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6 — Suspicious / Outlier Images (Per-Class Mahalanobis Distance)
# ═══════════════════════════════════════════════════════════════════════════
def detect_outliers_mahalanobis(features: np.ndarray, labels: np.ndarray, threshold_percentile: int = 97) -> dict:
    """
    Per-class Mahalanobis distance to find images that are outliers within their own class.
    """
    print(f"\n[Stage 6] Detecting per-class outliers (Mahalanobis distance percentile >= {threshold_percentile}) ...")
    
    mahalanobis_dists = np.zeros(len(features))
    unique_labels = np.unique(labels)
    
    for cls in unique_labels:
        cls_idx = np.where(labels == cls)[0]
        cls_features = features[cls_idx]
        
        # Compute mean and covariance for the class
        mu = np.mean(cls_features, axis=0)
        cov = np.cov(cls_features, rowvar=False)
        
        # Add small ridge to diagonal for numerical stability (invertibility)
        cov += np.eye(cov.shape[0]) * 1e-5
        
        try:
            inv_cov = scipy.linalg.inv(cov)
        except scipy.linalg.LinAlgError:
            inv_cov = scipy.linalg.pinv(cov)
            
        # Compute Mahalanobis distance for all samples in this class
        diff = cls_features - mu
        # (N, D) @ (D, D) -> (N, D) * (N, D) -> sum axis 1 -> (N,)
        dists = np.sqrt(np.sum(np.dot(diff, inv_cov) * diff, axis=1))
        
        mahalanobis_dists[cls_idx] = dists
        
    # Global threshold based on per-class distances 
    outlier_indices = []
    for cls in unique_labels:
        cls_idx = np.where(labels == cls)[0]
        cls_dists = mahalanobis_dists[cls_idx]
        threshold = np.percentile(cls_dists, threshold_percentile)
        outliers_in_cls = cls_idx[cls_dists >= threshold]
        outlier_indices.extend(outliers_in_cls)
        
    outlier_indices = np.array(outlier_indices)
    
    # Sort globally by distance (highest first = most anomalous)
    sorted_outliers = outlier_indices[np.argsort(mahalanobis_dists[outlier_indices])[::-1]]
    
    print(f"  Outliers found (per-class Mahalanobis): {len(sorted_outliers)}")
    return {
        "outlier_indices": sorted_outliers,
        "anomaly_scores": mahalanobis_dists,  # Higher is more anomalous here
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 7 — Duplicate / Near-Duplicate Detection (Cosine Similarity)
# ═══════════════════════════════════════════════════════════════════════════
def detect_duplicates(features: np.ndarray, labels: np.ndarray, similarity_threshold: float = 0.98) -> dict:
    """
    Find duplicate/near-duplicate image pairs using cosine similarity in feature space.
    Computed within each class to save time.
    """
    print(f"\n[Stage 7] Detecting duplicate/near-duplicate pairs (cosine similarity >= {similarity_threshold}) ...")
    
    duplicate_pairs = []
    unique_labels = np.unique(labels)
    
    for cls in unique_labels:
        cls_idx = np.where(labels == cls)[0]
        cls_features = features[cls_idx]
        
        # Compute pairwise cosine similarity
        sim_matrix = cosine_similarity(cls_features)
        
        # Find upper triangle indices where similarity > threshold
        np.fill_diagonal(sim_matrix, 0)
        i_idx, j_idx = np.where(sim_matrix >= similarity_threshold)
        
        # Keep only upper triangle pairs
        valid_pairs_mask = i_idx < j_idx
        i_idx = i_idx[valid_pairs_mask]
        j_idx = j_idx[valid_pairs_mask]
        
        for idx_i, idx_j in zip(i_idx, j_idx):
            global_i = cls_idx[idx_i]
            global_j = cls_idx[idx_j]
            sim = sim_matrix[idx_i, idx_j]
            duplicate_pairs.append((int(global_i), int(global_j), float(sim)))
            
    # Sort pairs by similarity (highest first)
    duplicate_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Get unique indices involved in duplicates
    duplicate_indices = list(set([p[0] for p in duplicate_pairs] + [p[1] for p in duplicate_pairs]))
    
    print(f"  Duplicate pairs found: {len(duplicate_pairs)} (involving {len(duplicate_indices)} unique images)")
    return {
        "duplicate_pairs": duplicate_pairs,
        "duplicate_indices": duplicate_indices,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 8 — Minority / Weak Clusters (K-Means)
# ═══════════════════════════════════════════════════════════════════════════
def analyse_weak_clusters(features: np.ndarray, labels: np.ndarray, n_clusters: int = 5) -> dict:
    """
    K-Means clustering within each class to find minority sub-groups.
    """
    print(f"\n[Stage 8] Analysing weak clusters (K={n_clusters} per class) ...")
    
    cluster_results = {}
    unique_labels = np.unique(labels)
    
    for cls in unique_labels:
        cls_idx = np.where(labels == cls)[0]
        cls_features = features[cls_idx]
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(cls_features)
        
        unique, counts = np.unique(cluster_labels, return_counts=True)
        
        cls_name = CIFAR10_CLASSES[int(cls)]
        cluster_results[cls_name] = {
            "cluster_sizes": {int(c): int(cnt) for c, cnt in zip(unique, counts)},
            "sample_indices": {int(c): cls_idx[np.where(cluster_labels == c)[0]] for c in unique},
        }
        
    weak_threshold = (len(features) / len(unique_labels)) / n_clusters * 0.25
    weak_clusters_found = 0
    
    for cls_name, res in cluster_results.items():
        for c, size in res["cluster_sizes"].items():
            if size < weak_threshold:
                weak_clusters_found += 1
                
    print(f"  Weak clusters found (size < {int(weak_threshold)}): {weak_clusters_found}")
    
    return {
        "cluster_results": cluster_results,
        "weak_threshold": int(weak_threshold),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Stage 9 — Class Imbalance Analysis
# ═══════════════════════════════════════════════════════════════════════════
def analyse_class_imbalance(labels: np.ndarray) -> dict:
    """Compute class distribution stats."""
    print("\n[Stage 9] Analysing class imbalance ...")
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
    dataset, indices, labels, suggested, title, save_path, n=25, scores=None
):
    """Save a grid of flagged sample images."""
    n = min(n, len(indices))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
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
        
        title_text = f"#{idx}\nGiven: {given_name}"
        if suggested is not None and given_label != suggested[idx]:
            title_text += f"\nShould be: {sugg_name}"
            color = "red"
        else:
            color = "black"
            
        if scores is not None:
            title_text += f"\nConf: {scores[idx]:.4f}"
            
        ax.set_title(title_text, fontsize=8, color=color)
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


def _save_duplicate_pairs_grid(dataset, duplicate_pairs, labels, save_path, n=10):
    """Save a grid showing pairs of duplicate images side-by-side."""
    n = min(n, len(duplicate_pairs))
    if n == 0:
        return
    fig, axes = plt.subplots(n, 2, figsize=(6, 2.5 * n))
    fig.suptitle("Top Duplicate / Near-Duplicate Pairs", fontsize=14, fontweight="bold")
    
    if n == 1:
        axes = [axes]
        
    for i in range(n):
        idx1, idx2, sim = duplicate_pairs[i]
        
        img1, l1 = dataset[idx1]
        img2, l2 = dataset[idx2]
        
        ax1, ax2 = axes[i]
        
        ax1.imshow(_unnormalize_cifar(img1))
        ax1.set_title(f"#{idx1} ({CIFAR10_CLASSES[l1]})", fontsize=10)
        ax1.axis("off")
        
        ax2.imshow(_unnormalize_cifar(img2))
        ax2.set_title(f"#{idx2} ({CIFAR10_CLASSES[l2]})\nSim: {sim:.4f}", fontsize=10)
        ax2.axis("off")
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


def generate_report(
    label_results: dict,
    ambiguous_results: dict,
    blurry_results: dict,
    outlier_results: dict,
    duplicate_results: dict,
    cluster_results: dict,
    imbalance_results: dict,
    labels: np.ndarray,
):
    """Generate the full enhanced audit report + visualisations."""
    print("\n[Stage 10] Generating enhanced audit report ...")
    os.makedirs(AUDIT_DIR, exist_ok=True)

    # --- Load the original (CIFAR-normalised) dataset for visualisations ---
    train_dataset, _ = get_datasets()

    # --- Visualisations ---
    # 1. Flagged mislabeled samples grid
    mislabel_grid_path = os.path.join(AUDIT_DIR, "mislabeled_samples.png")
    _save_flagged_grid(
        train_dataset, label_results["issue_indices"], labels,
        label_results["suggested_labels"], "Top Suspected Mislabeled Samples",
        mislabel_grid_path, n=25, scores=label_results["quality_scores"]
    )

    # 2. Ambiguous samples grid
    ambiguous_grid_path = os.path.join(AUDIT_DIR, "ambiguous_samples.png")
    _save_flagged_grid(
        train_dataset, ambiguous_results["ambiguous_indices"], labels, None,
        "Top Ambiguous/Confusing Samples", ambiguous_grid_path, n=25
    )

    # 3. Blurry samples grid
    blurry_grid_path = os.path.join(AUDIT_DIR, "blurry_samples.png")
    _save_flagged_grid(
        train_dataset, blurry_results["blurry_indices"], labels, None,
        "Top Blurry/Featureless Samples", blurry_grid_path, n=25
    )

    # 4. Outlier samples grid
    outlier_grid_path = os.path.join(AUDIT_DIR, "outlier_samples.png")
    _save_flagged_grid(
        train_dataset, outlier_results["outlier_indices"], labels, None,
        "Top Outlier / Anomalous Samples", outlier_grid_path, n=25
    )

    # 5. Duplicate pairs grid
    duplicate_grid_path = os.path.join(AUDIT_DIR, "duplicate_samples.png")
    _save_duplicate_pairs_grid(
        train_dataset, duplicate_results["duplicate_pairs"], labels,
        duplicate_grid_path, n=10
    )

    # 6. Class distribution chart
    dist_chart_path = os.path.join(AUDIT_DIR, "class_distribution.png")
    _save_class_distribution_chart(imbalance_results["distribution"], dist_chart_path)

    # 7. Quality score histogram
    quality_hist_path = os.path.join(AUDIT_DIR, "quality_score_histogram.png")
    _save_quality_score_histogram(label_results["quality_scores"], quality_hist_path)

    # --- Save flagged indices as JSON ---
    flagged_data = {
        "mislabeled_indices": [int(i) for i in label_results["issue_indices"]],
        "ambiguous_indices": [int(i) for i in ambiguous_results["ambiguous_indices"]],
        "blurry_indices": [int(i) for i in blurry_results["blurry_indices"]],
        "outlier_indices": [int(i) for i in outlier_results["outlier_indices"]],
        "duplicate_indices": [int(i) for i in duplicate_results["duplicate_indices"]],
        "total_mislabeled": len(label_results["issue_indices"]),
        "total_ambiguous": len(ambiguous_results["ambiguous_indices"]),
        "total_blurry": len(blurry_results["blurry_indices"]),
        "total_outliers": len(outlier_results["outlier_indices"]),
        "total_duplicates": len(duplicate_results["duplicate_indices"]),
    }
    flagged_path = os.path.join(AUDIT_DIR, "flagged_indices.json")
    with open(flagged_path, "w") as f:
        json.dump(flagged_data, f, indent=2)
    print(f"  Saved: {flagged_path}")

    # --- Markdown Report ---
    report_lines = []
    report_lines.append("# CIFAR-10 Enhanced Data Audit Report\n")
    report_lines.append(f"*Generated automatically by `data_auditor.py`*\n")
    report_lines.append("---\n")

    # Summary
    report_lines.append("## Summary\n")
    report_lines.append(f"| Metric | Value |")
    report_lines.append(f"|---|---|")
    report_lines.append(f"| Total training samples | {imbalance_results['total_samples']:,} |")
    report_lines.append(f"| Suspected mislabeled samples | {len(label_results['issue_indices']):,} |")
    report_lines.append(f"| Ambiguous/Confusing samples | {len(ambiguous_results['ambiguous_indices']):,} |")
    report_lines.append(f"| Blurry/Featureless samples | {len(blurry_results['blurry_indices']):,} |")
    report_lines.append(f"| Outlier samples | {len(outlier_results['outlier_indices']):,} |")
    report_lines.append(f"| Duplicate/Near-Duplicate samples | {len(duplicate_results['duplicate_indices']):,} |")
    report_lines.append(f"| Class imbalance ratio (max/min) | {imbalance_results['imbalance_ratio']:.4f} |")
    report_lines.append("")
    
    unique_issues = set(label_results["issue_indices"]).union(
        ambiguous_results["ambiguous_indices"],
        blurry_results["blurry_indices"],
        outlier_results["outlier_indices"],
        duplicate_results["duplicate_indices"]
    )
    report_lines.append(f"> **Total unique problematic samples:** {len(unique_issues):,} "
                        f"({len(unique_issues)/imbalance_results['total_samples']*100:.1f}% of dataset)\n")

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
    report_lines.append(f"![Mislabeled Samples](mislabeled_samples.png)\n")
    
    # Ambiguous Images
    report_lines.append("## Ambiguous / Confusing Images\n")
    report_lines.append(f"High prediction entropy flagged **{len(ambiguous_results['ambiguous_indices']):,}** "
                        f"samples where the model is highly uncertain.\n")
    report_lines.append(f"![Ambiguous Samples](ambiguous_samples.png)\n")

    # Blurry Images
    report_lines.append("## Blurry Images\n")
    report_lines.append(f"Laplacian variance detection flagged **{len(blurry_results['blurry_indices']):,}** "
                        f"samples that lack high-frequency detail.\n")
    report_lines.append(f"![Blurry Samples](blurry_samples.png)\n")

    # Outliers
    report_lines.append("## Outlier / Anomaly Detection\n")
    report_lines.append(f"Per-class Mahalanobis distance flagged **{len(outlier_results['outlier_indices']):,}** "
                        f"samples as anomalies within their respective classes.\n")
    report_lines.append(f"![Outlier Samples](outlier_samples.png)\n")

    # Duplicates
    report_lines.append("## Duplicate / Near-Duplicate Detection\n")
    report_lines.append(f"Cosine similarity in feature space found **{len(duplicate_results['duplicate_pairs']):,}** "
                        f"highly similar pairs (involving {len(duplicate_results['duplicate_indices']):,} unique images).\n")
    if len(duplicate_results['duplicate_pairs']) > 0:
        report_lines.append(f"![Duplicate Samples](duplicate_samples.png)\n")

    # Weak Clusters
    report_lines.append("## Weak / Minority Clusters\n")
    report_lines.append(f"K-Means clustering within classes identified sub-groups that are underrepresented "
                        f"(threshold: < {cluster_results['weak_threshold']} samples).\n")
    for cls_name, res in cluster_results["cluster_results"].items():
        weak_in_cls = [(c, s) for c, s in res["cluster_sizes"].items() if s < cluster_results["weak_threshold"]]
        if weak_in_cls:
            report_lines.append(f"- **{cls_name}**: Found {len(weak_in_cls)} weak cluster(s).")
            for c, s in weak_in_cls:
                report_lines.append(f"  - Cluster {c}: {s} samples")
    report_lines.append("")

    # Conclusion
    report_lines.append("## Conclusion\n")
    report_lines.append(f"A total of **{len(unique_issues):,}** unique samples have been flagged across all detection stages. "
                        f"These should be addressed (removed, reviewed, or augmented) before downstream training.")

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
    print(" Phase 2: Enhanced Automated Data Quality Auditing")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Load raw dataset for blur detection
    train_dataset, _ = get_datasets()
    dataset_data = train_dataset.data

    # Stage 1 — Feature extraction
    features, labels = extract_features(device)

    # Stage 2 — Cross-validated predicted probabilities
    pred_probs = compute_pred_probs(features, labels)

    # Stage 3 — Wrong Labels (Cleanlab)
    label_results = detect_label_issues(labels, pred_probs)
    
    # Stage 4 — Ambiguous/Confusing Images (Entropy)
    ambiguous_results = detect_ambiguous_images(pred_probs, threshold_percentile=95)
    
    # Stage 5 — Blurry Images (Laplacian Variance)
    blurry_results = detect_blurry_images(dataset_data, threshold_percentile=5)

    # Stage 6 — Suspicious / Outliers (Mahalanobis)
    outlier_results = detect_outliers_mahalanobis(features, labels, threshold_percentile=97)
    
    # Stage 7 — Duplicates / Near-Duplicates (Cosine Similarity)
    duplicate_results = detect_duplicates(features, labels, similarity_threshold=0.98)
    
    # Stage 8 — Weak / Minority Clusters (K-Means)
    cluster_results = analyse_weak_clusters(features, labels, n_clusters=5)

    # Stage 9 — Class imbalance analysis
    imbalance_results = analyse_class_imbalance(labels)

    # Stage 10 — Report generation
    report_path = generate_report(
        label_results, ambiguous_results, blurry_results, outlier_results, 
        duplicate_results, cluster_results, imbalance_results, labels
    )

    elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print(f" Phase 2 complete in {elapsed:.1f}s")
    print(f" Audit report: {report_path}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
