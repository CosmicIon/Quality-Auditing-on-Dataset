# Quality Auditing on Dataset

## Overview
This project focuses on **Data-Centric AI**: improving machine learning models by systematically elevating the quality of the dataset itself, rather than just tweaking the model architecture. 

It provides an automated, end-to-end pipeline designed to thoroughly audit datasets, detect multiple forms of data degradation (label noise, blurriness, outliers, etc.), clean the data, and selectively augment underrepresented classes using generative models (Diffusion).

**Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Key Features & Capabilities

Our enhanced **Data Auditing Pipeline** doesn't just look for one issue; it systematically scans the dataset across 6 different dimensions:
1. **Mislabeled Data**: Uses Confident Learning (`cleanlab`) to flag incorrect labels.
2. **Ambiguous/Confusing Images**: Uses Prediction Entropy to find images the model is highly uncertain about.
3. **Blurry Images**: Uses OpenCV Laplacian Variance to flag images lacking high-frequency details.
4. **Suspicious/Outlier Images**: Computes Per-Class Mahalanobis Distance to find severe anomalies.
5. **Duplicate Images**: Uses Feature-Space Cosine Similarity to find near-exact duplicates.
6. **Weak Clusters**: Uses K-Means clustering to analyze potential minority sub-groups within classes.

## Project Structure
```text
Quality-Auditing-on-Dataset/
├── data/
│   ├── raw/                # Raw downloaded datasets (CIFAR-10)
│   └── processed/          # Processed data, sample grids, tensors
│       ├── audit/          # Phase 2 audit reports & visual grids
│       ├── noise/          # Injected noisy labels metadata
│       └── validation/     # Phase 5 downstream validation outputs
├── docs/                   # Project documentation
├── scripts/                # Utility / helper scripts
│   └── md_to_pdf.py        # Converts Markdown reports to PDF
├── src/                    # Source code
│   ├── data_loader.py      # Phase 1: CIFAR-10 download & dynamic data loading
│   ├── noise_injector.py   # Optional: Injects 15% label noise (.pt generation)
│   ├── data_auditor.py     # Phase 2: The 6-stage automated data quality auditor
│   ├── data_cleaner.py     # Phase 3: Builds the cleaned subset removing flagged data
│   ├── synthesis/          # Phase 3: Generative models (DDPM / U-Net)
│   ├── train_generator.py  # Phase 3: Train the DDPM
│   ├── synthesize_data.py  # Phase 3: Generate synthetic images
│   ├── quality_assessor.py # Phase 4: Quality assessment of synthetic data
│   └── downstream_validation.py  # Phase 5: Validates performance gains on ResNet-18
├── README.md
├── TODO.md
└── requirements.txt
```

## Setup & Installation

### Prerequisites
- Python 3.10+
- PyTorch (CUDA recommended for feature extraction and downstream validation)

### Installation
```bash
# 1. Clone the repository
git clone https://github.com/CosmicIon/Quality-Auditing-on-Dataset.git
cd Quality-Auditing-on-Dataset

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# PowerShell (Windows):
.\.venv\Scripts\Activate.ps1
# CMD (Windows):
.\.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

## Running the Pipeline

The project is structured into 5 sequential phases:

### Phase 1: Data Setup & Optional Noise Injection
Download the dataset and prepare the environment. For academic validation, you can intentionally corrupt the dataset.
```bash
# Download CIFAR-10 and verify the data loaders
python src/data_loader.py

# (Optional) Inject 15% label noise to prove the auditor works. 
# This generates a physical noisy_cifar10.pt dataset that downstream scripts auto-detect.
python src/noise_injector.py
```

### Phase 2: Data Auditing
Run the comprehensive 6-stage auditor to find mislabeled, blurry, ambiguous, outlier, and duplicate images.
```bash
# Extracts ResNet-18 features and generates a detailed audit_report.md
python src/data_auditor.py
```

### Phase 3: Data Cleaning & Synthesis
Remove the bad data flagged by the auditor, and train a Diffusion model to synthesize replacements.
```bash
# Preview cleaning stats (removes bad samples based on the Phase 2 audit)
python src/data_cleaner.py           

# Train DDPM and generate synthetic images to replace removed data
python src/train_generator.py        
python src/synthesize_data.py        
```

### Phase 4: Quality Assessment
Assess the visual fidelity and diversity of the newly synthesized data.
```bash
python src/quality_assessor.py       # FID, diversity, memorization checks
```

### Phase 5: Downstream Validation
The ultimate test of data-centric AI. Train a downstream classifier (ResNet-18) on three dataset variants to prove that improving data quality yields higher accuracy.
```bash
# Compares model performance on Original (Noisy) vs Cleaned vs Cleaned+Augmented datasets
python src/downstream_validation.py --epochs 15
```

## Documentation
- [Project Overview](docs/PROJECT_OVERVIEW.md): Detailed problem statement, major issues addressed, and expected outcomes.
- [Pipeline Tasks](docs/PIPELINE_TASKS.md): Detailed breakdown of the automated pipeline tasks (Auditing, Synthesis, Quality Check, Validation).

## Core Philosophy
**Data-Centric AI**: We aim to prove that systematically improving data quality—by rigorously cleaning bad samples and intelligently augmenting minority classes—yields significant, undeniable improvements in downstream model accuracy, often far exceeding what can be achieved by tweaking model architectures alone.
