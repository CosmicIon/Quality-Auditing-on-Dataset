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
├── docs/                   # Project documentation
│   ├── PIPELINE_TASKS.md
│   └── PROJECT_OVERVIEW.md
├── notebooks/              # Exploratory Jupyter notebooks
├── scripts/                # Utility / helper scripts
├── src/                    # Source code
│   ├── __init__.py
│   └── data_loader.py      # CIFAR-10 download & data loaders
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
# Download CIFAR-10 and verify the data loaders
python src/data_loader.py
```

## Documentation
- [Project Overview](docs/PROJECT_OVERVIEW.md): Detailed problem statement, major issues addressed, and expected outcomes.
- [Pipeline Tasks](docs/PIPELINE_TASKS.md): Detailed breakdown of the automated pipeline tasks (Auditing, Synthesis, Quality Check, Validation).
- [TODO](TODO.md): Project progress tracker.

## Core Philosophy
Data-centric AI: Proving that improving data quality (cleaning and targeted augmentation) yields better accuracy gains than architectural changes alone.
