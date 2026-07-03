# Quality Auditing on Dataset

## Overview
This project focuses on the **Automatic Generation of Synthetic Data and Quality Auditing**. It provides an automated pipeline designed to detect label noise, bias, and out-of-distribution (OOD) samples in a specialized dataset. By leveraging generative models (such as Diffusion models), the system selectively augments underrepresented classes to improve downstream model performance.

**Dataset**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

## Project Structure
```
Quality-Auditing-on-Dataset/
├── data/
│   ├── raw/                # Raw downloaded datasets (CIFAR-10)
│   └── processed/          # Processed data, sample grids, tensors
│       ├── audit/          # Phase 2 audit outputs
│       ├── synthesis/      # Phase 3 synthesis outputs
│       ├── quality/        # Phase 4 quality assessment outputs
│       └── validation/     # Phase 5 downstream validation outputs
├── docs/                   # Project documentation
│   ├── PIPELINE_TASKS.md
│   └── PROJECT_OVERVIEW.md
├── notebooks/              # Exploratory Jupyter notebooks
├── scripts/                # Utility / helper scripts
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_loader.py      # Phase 1: CIFAR-10 download & data loaders
│   ├── data_auditor.py     # Phase 2: Automated data quality auditing
│   ├── data_cleaner.py     # Phase 3: Build cleaned training subset
│   ├── synthesis/          # Phase 3: Generative model
│   │   ├── model.py        #   Class-conditional U-Net
│   │   └── diffusion.py    #   Gaussian diffusion process
│   ├── train_generator.py  # Phase 3: Train the DDPM
│   ├── synthesize_data.py  # Phase 3: Generate synthetic images
│   ├── quality_assessor.py # Phase 4: Quality assessment of synthetic data
│   └── downstream_validation.py  # Phase 5: Downstream validation
├── tests/                  # Unit tests
├── .gitignore
├── README.md
├── TODO.md
└── requirements.txt
```


## Setup

### Prerequisites
- Python 3.13+

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

### Quick Start
```bash
# Phase 1 — Download CIFAR-10 and verify the data loaders
python src/data_loader.py

# Phase 2 — Run automated data quality auditing
python src/data_auditor.py

# Phase 3 — Train generative model and synthesize data
python src/data_cleaner.py           # Preview cleaning stats
python src/train_generator.py        # Train DDPM (~2-3 hrs on GPU)
python src/synthesize_data.py        # Generate synthetic images

# Phase 4 — Assess quality of synthetic data
python src/quality_assessor.py       # FID, diversity, memorization checks

# Phase 5 — Downstream validation (train classifier on 3 dataset variants)
python src/downstream_validation.py  # Compare Original vs Cleaned vs Augmented
```

## Documentation
- [Project Overview](docs/PROJECT_OVERVIEW.md): Detailed problem statement, major issues addressed, and expected outcomes.
- [Pipeline Tasks](docs/PIPELINE_TASKS.md): Detailed breakdown of the automated pipeline tasks (Auditing, Synthesis, Quality Check, Validation).
- [TODO](TODO.md): Project progress tracker.

## Core Philosophy
Data-centric AI: Proving that improving data quality (cleaning and targeted augmentation) yields better accuracy gains than architectural changes alone.
