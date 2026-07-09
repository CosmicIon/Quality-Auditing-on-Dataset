# Enhanced Data Auditing Pipeline

**Goal**: Upgrade `data_auditor.py` from 3 detection methods to 7, producing a comprehensive audit report that covers all real-world data quality issues.

---

## Professor's Suggestions — Cross-Verification & My Recommendations

### 0. Wrong Labels — Cleanlab (Already Done) ✅
**Professor**: Use Cleanlab. **My verdict**: Perfect. No changes needed.

### 1. Confusing / Ambiguous Images — Prediction Uncertainty ✅✅
**Professor**: Use prediction uncertainty. **My verdict**: Excellent suggestion. I recommend a **two-pronged approach**:

- **Prediction entropy** (from the existing `pred_probs`): If the classifier outputs near-uniform probabilities across classes (high entropy), the image is inherently ambiguous — the model genuinely can't tell what it is. This catches partially visible objects, images at class boundaries (e.g., a pickup truck vs an automobile).
- **Laplacian variance** (image-level blur detection): This is a classical computer vision technique — compute the variance of the Laplacian filter on the grayscale image. Low variance = blurry/featureless. This catches motion blur, out-of-focus images, and low-contrast images that entropy alone would miss.

Using both gives us two separate categories: "confusing" (model can't decide) vs "blurry" (low visual quality).

### 2. Suspicious / Outlier Images — Embedding-based Detection ✅✅
**Professor**: Use embedding-based outlier detection or entropy. **My verdict**: Good suggestion. But I recommend upgrading from the current global Isolation Forest to **per-class Mahalanobis distance**:

- **Current approach** (Isolation Forest on all 50K samples): Finds images that are globally weird, but misses images that are outliers *within their own class* (e.g., a cartoon cat among real cats).
- **Better approach** (per-class Mahalanobis distance): For each class, compute the mean and covariance of its feature embeddings. Then measure how far each sample is from its class center. A cat image that's 5 standard deviations from the "cat" cluster is suspicious even if it's not a global outlier.

> [!TIP]
> Mahalanobis distance is more principled than Isolation Forest for Gaussian-distributed embeddings (which ResNet features approximately are). It's also easier to interpret: "this image is X standard deviations from its class center."

### 3. Duplicate / Near-Duplicate Detection — Image Similarity ✅✅
**Professor**: Use image similarity. **My verdict**: Absolutely essential for real-world datasets. I recommend:

- **Cosine similarity in ResNet-18 feature space**: We already have the 512-d features extracted. Compute pairwise cosine similarity within each class and flag pairs with similarity > 0.98 (near-duplicates) and > 0.9999 (exact duplicates).
- **Why not pixel hashing?** Perceptual hashing (pHash) is faster but misses semantically similar images with slight differences (crops, color shifts). Feature-space similarity catches these.

> [!IMPORTANT]
> Computing pairwise similarity for 50K images would be 1.25 billion pairs — too slow. We compute **within each class only** (5K × 5K = 12.5M pairs per class), which is fast and meaningful.

### 4. Minority / Weak Clusters — Clustering ✅✅
**Professor**: Use clustering. **My verdict**: Great suggestion. I recommend:

- **K-Means within each class** (k=5 sub-clusters per class): This reveals internal structure. For example, the "automobile" class might have sub-clusters for sedans, SUVs, sports cars, trucks. If one sub-cluster has only 50 images while others have 900+, that's a "weak cluster" that needs more samples.
- **Why K-Means over DBSCAN?** K-Means is more interpretable (fixed number of clusters, clear sizes), and we can visualize representative images from each cluster. DBSCAN is better when you don't know k, but for a fixed analysis report, K-Means gives cleaner results.

---

## New Pipeline Structure (8 Stages)

| Stage | Detection Type | Method | Status |
|---|---|---|---|
| 1 | Feature Extraction | ResNet-18 (pretrained) | Existing ✅ |
| 2 | Cross-Validated Probabilities | Logistic Regression + 3-fold CV | Existing ✅ |
| 3 | Wrong Labels | Cleanlab confident learning | Existing ✅ |
| 4 | Ambiguous/Confusing Images | Prediction entropy (high entropy = ambiguous) | **NEW** |
| 5 | Blurry/Low-Quality Images | Laplacian variance (low variance = blurry) | **NEW** |
| 6 | Suspicious/Outlier Images | Per-class Mahalanobis distance | **UPGRADED** (was global Isolation Forest) |
| 7 | Duplicate/Near-Duplicate Images | Cosine similarity in feature space | **NEW** |
| 8 | Minority/Weak Clusters | K-Means within each class | **NEW** |
| 9 | Class Imbalance | Count-based analysis | Existing ✅ |
| 10 | Report Generation | Enhanced Markdown + visualizations | **UPGRADED** |

---

## Proposed Changes

### [MODIFY] [data_auditor.py](file:///d:/CODING/github/Quality-Auditing-on-Dataset/src/data_auditor.py)

Major rewrite to add 4 new detection stages and upgrade 1 existing stage. The file will grow from ~508 lines to ~900+ lines. Key additions:

#### New Stage 4: Ambiguous Image Detection
```python
def detect_ambiguous_images(pred_probs: np.ndarray, threshold_percentile: int = 95):
    """
    Flag images where the classifier is most uncertain.
    High prediction entropy = model can't decide = ambiguous image.
    """
    entropy = -np.sum(pred_probs * np.log(pred_probs + 1e-10), axis=1)
    threshold = np.percentile(entropy, threshold_percentile)
    ambiguous_indices = np.where(entropy >= threshold)[0]
    return {"ambiguous_indices": ambiguous_indices, "entropy_scores": entropy}
```

#### New Stage 5: Blur Detection
```python
def detect_blurry_images(dataset, threshold_percentile: int = 5):
    """
    Compute Laplacian variance for each image.
    Low variance = blurry/featureless.
    """
    # For each image: grayscale → Laplacian filter → variance
    # Flag bottom 5% as blurry
```

#### Upgraded Stage 6: Per-Class Mahalanobis Outlier Detection
```python
def detect_outliers_mahalanobis(features, labels, threshold_percentile=97):
    """
    Per-class Mahalanobis distance — finds images that are outliers
    within their own class distribution.
    """
    # For each class: compute mean, covariance, Mahalanobis distance
    # Flag top 3% per class as outliers
```

#### New Stage 7: Duplicate Detection
```python
def detect_duplicates(features, labels, near_threshold=0.98):
    """
    Find duplicate/near-duplicate images using cosine similarity
    in feature space, computed within each class.
    """
    # Pairwise cosine similarity within each class
    # Flag pairs with similarity > threshold
```

#### New Stage 8: Weak Cluster Analysis
```python
def analyse_weak_clusters(features, labels, n_clusters=5):
    """
    K-Means clustering within each class to find minority sub-groups.
    """
    # For each class: run K-Means, report cluster sizes
    # Flag clusters below a size threshold
```

#### Enhanced Report
The report will grow from ~70 lines to ~200+ lines with new sections:
- Ambiguous images section (grid + entropy histogram)
- Blurry images section (grid + Laplacian score distribution)
- Duplicate pairs section (side-by-side image pairs)
- Cluster analysis section (per-class cluster sizes + representative image grids)
- Enhanced summary table with all issue counts
- Grand total of unique flagged samples (union of all issues)

### [MODIFY] [data_cleaner.py](file:///d:/CODING/github/Quality-Auditing-on-Dataset/src/data_cleaner.py)

Update to read new flagged categories from the enhanced `flagged_indices.json` (which will now contain `duplicate_indices`, `ambiguous_indices`, `blurry_indices` in addition to existing fields).

---

## New Dependencies

```
scipy          # Already installed (Mahalanobis distance uses scipy.spatial.distance)
opencv-python  # For Laplacian blur detection (cv2.Laplacian)
```

> [!NOTE]
> `opencv-python` is the only new dependency. `scipy` and `sklearn` are already installed.

---

## Verification Plan

### Smoke Test
- Run `python src/data_auditor.py` and verify all 8 stages complete without errors
- Verify the enhanced `audit_report.md` has all new sections
- Check that all visualization PNGs are generated

### Expected Results
- **Ambiguous images**: ~2,500 flagged (top 5% entropy)
- **Blurry images**: ~2,500 flagged (bottom 5% Laplacian variance)
- **Outliers**: ~1,500 flagged (per-class Mahalanobis, top 3%)
- **Duplicates**: Variable (depends on dataset, likely 100-500 near-duplicate pairs in CIFAR-10)
- **Weak clusters**: Identification of 2-3 underrepresented sub-groups per class

### Runtime
- Current runtime: ~5 minutes
- Expected new runtime: ~8-10 minutes (blur detection adds per-image computation, duplicate detection adds pairwise similarity)
