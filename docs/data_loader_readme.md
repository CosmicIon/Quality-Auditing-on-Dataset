# `data_loader.py` - Detailed Code Walkthrough

This document provides a comprehensive explanation of the `data_loader.py` script located in the `src/` directory. This script acts as Phase 1 of the machine learning pipeline, primarily responsible for downloading the CIFAR-10 dataset, preprocessing it, and setting up the PyTorch data loaders.

---

## Overview

The `data_loader.py` script executes the following high-level tasks:
1. **Downloads CIFAR-10** into the `data/raw/` directory, using robust mirrors in case the default PyTorch servers are slow or unresponsive.
2. **Applies Data Transformations**, specifically converting the raw images to PyTorch tensors and normalizing them based on precomputed CIFAR-10 channel statistics.
3. **Creates PyTorch DataLoaders** for both training and testing datasets, optimizing them with settings like `pin_memory` for GPU acceleration when available.
4. **Validates & Explores the Data** by printing dataset statistics (class distribution) and generating a visual grid of sample images saved to `data/processed/`.

---

## Code Flow and Key Snippets

### 1. Imports and Setup
The script begins by importing standard modules (`os`, `sys`), PyTorch modules (`torch`, `torchvision`, `transforms`, `DataLoader`), and visualization tools (`numpy`, `matplotlib`). 
> **Note:** It explicitly configures matplotlib to use the non-interactive `"Agg"` backend (`matplotlib.use("Agg")`), which ensures the script runs safely on headless servers without attempting to launch a GUI window.

### 2. Global Variables and Configurations
- **Paths:** It resolves the `PROJECT_ROOT`, `RAW_DATA_DIR`, and `PROCESSED_DATA_DIR` automatically relative to the script's location.
- **Classes:** `CIFAR10_CLASSES` stores the 10 label names (e.g., airplane, automobile, bird).
- **Normalization Constants:** `CIFAR10_MEAN` and `CIFAR10_STD` are hardcoded. These are the pre-calculated channel-wise means and standard deviations of the CIFAR-10 training set. 

### 3. Data Transformations (`train_transform` & `test_transform`)
Two transformation pipelines are created using `torchvision.transforms.Compose`. 
```python
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
```
1. `ToTensor()`: Converts PIL Images (or NumPy arrays) with shape (H x W x C) in the range [0, 255] to torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
2. `Normalize()`: Normalizes each tensor channel using the `CIFAR10_MEAN` and `CIFAR10_STD` to have a mean of 0 and standard deviation of 1.

### 4. Dataset Acquisition Helpers
- **`_CIFAR10_MIRRORS`:** A list of fallback URLs to download the dataset.
- **`_ensure_cifar10_downloaded(root)`:** A resilient helper function. It circumvents intermittent hangs associated with `torchvision`'s default download logic by downloading and extracting the tarball manually if not present.
- **`get_datasets()`:** Initializes `torchvision.datasets.CIFAR10` for both splits:
```python
def get_datasets():
    # Pre-download with mirror fallback so torchvision doesn't hang
    _ensure_cifar10_downloaded(RAW_DATA_DIR)

    train_dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR, train=True, download=False, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=RAW_DATA_DIR, train=False, download=False, transform=test_transform
    )
    return train_dataset, test_dataset
```

### 5. DataLoader Generation
`get_dataloaders()` wraps the datasets in PyTorch `DataLoader` objects, which handle batching, shuffling, and parallel data loading.
```python
def get_dataloaders(batch_size: int = 64, num_workers: int = 2):
    train_dataset, test_dataset = get_datasets()
    use_pin_memory = torch.cuda.is_available() # Check for GPU

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    # ... test_loader is identical but shuffle=False
```
Notice `pin_memory=use_pin_memory`: When a GPU is available, pinning memory significantly speeds up data transfer from CPU to GPU during training.

### 6. Exploration Utilities
- **`print_dataset_stats(train_dataset, test_dataset)`:** A debugging function that prints standard information such as total samples, image shapes, and counts per individual class.
- **`save_sample_grid(dataset, save_path, n=25)`:** Generates a 5x5 visual grid of images. Crucially, before displaying the images, it **un-normalizes** them (reverses the `Normalize` transform) to ensure the colors look correct when plotting:
```python
        # Undo normalisation for display
        img = img.clone()
        for ch in range(3):
            img[ch] = img[ch] * CIFAR10_STD[ch] + CIFAR10_MEAN[ch]
        img = img.clamp(0, 1)
```

### 7. Main Execution Flow (`main()`)
When the script is executed directly (`python src/data_loader.py`), the `main()` function ties everything together:
```python
def main():
    # 1. Load data
    train_dataset, test_dataset = get_datasets()
    print_dataset_stats(train_dataset, test_dataset)

    # 2. Setup loaders
    train_loader, test_loader = get_dataloaders(batch_size=64)
    images, labels = next(iter(train_loader)) # Pull a batch for verification

    # 3. Save a visual grid
    grid_path = os.path.join(PROCESSED_DATA_DIR, "sample_grid.png")
    save_sample_grid(train_dataset, grid_path)
```

---

## How to Run

To execute this phase independently, run the following command from the project root directory:

```bash
python src/data_loader.py
```
