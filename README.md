# CP219 Project 2 - Goose IDS Detection

## Overview

This repository contains the implementation of CP219 Project 2 for intrusion detection system analysis. The code generates all figures and tables presented in the accompanying report through five main tasks:

- **Task 1**: Exploratory Data Analysis (EDA)
- **Task 2**: Feature Characterization
- **Task 3**: Binary Detection Models
- **Task 4**: Multi-class Detection Models
- **Task 5**: Advanced analysis

All experiments are tracked using Weights & Biases (W&B) and outputs are saved locally in the `outputs/` directory.

---

## Prerequisites

- **Python**: Version specified in `pyproject.toml`
- **UV**: Modern Python package manager
- **Data**: Raw data files must be placed in `data/raw/` directory
- **W&B Account**: For experiment tracking (optional but recommended)

---

## Setup Instructions

### 1. Install UV Package Manager

If you don't have UV installed:

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### 2. Clone and Setup Environment

```bash
# Navigate to project directory
cd cp219_project2

# Create virtual environment with UV
uv venv

# Activate virtual environment
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate

# Install all dependencies from pyproject.toml
uv pip install -e .
# Install all dependencies from pyproject.toml
uv pip install -e ".[deep]"
```

### 3. Setup Weights & Biases (Optional)

```bash
# Login to W&B
wandb login

# Enter your API key from: https://wandb.ai/authorize
```

### 4. Verify Data Files

Ensure your data files are placed in the `data/raw/` directory. The exact filenames should match the configuration in `config/config.yaml`.

```bash
# Check data directory
ls data/raw/
```

---

## Reproducing Results

All tasks can be reproduced by running the corresponding scripts in the `experiments/` directory. Each script accepts command-line arguments for customization.

### General Command Structure

```bash
python experiments/run_task<N>.py --mode="<mode>" --wandb-name="<experiment_name>"
```

**Arguments:**
- `--mode`: Execution mode (e.g., "full", "core", "new", "core_new")
- `--wandb-name`: Name for the W&B experiment run

---

### Task 1: Exploratory Data Analysis

Generates distribution plots, correlation matrices, and summary statistics.

```bash
python experiments/run_task1.py --mode="full" --wandb-name="task1_eda"
```

**Outputs:**
- Figures: `outputs/figures/task1/`
- Tables: `outputs/tables/task1/`

**Generated Artifacts:**
- Data distribution visualizations
- Feature correlation heatmaps
- Class balance analysis
- Summary statistics tables

---

### Task 2: Feature Characterization

Performs feature importance analysis, dimensionality reduction, and feature engineering.

```bash
python experiments/run_task2.py --mode="full" --wandb-name="task2_characterization"
```

**Outputs:**
- Figures: `outputs/figures/task2/`
- Tables: `outputs/tables/task2/`

**Generated Artifacts:**
- Feature importance rankings
- PCA/t-SNE visualizations
- Feature distribution comparisons
- Statistical test results

---

### Task 3: Binary Detection Models

Trains and evaluates binary classification models (normal vs. attack).

```bash
python experiments/run_task3.py --mode="full" --wandb-name="task3_binary"
```

**Outputs:**
- Figures: `outputs/figures/task3/`
- Tables: `outputs/tables/task3/`
- Models: `outputs/models/task3/`

**Generated Artifacts:**
- Precision-Recall curves
- Confusion matrices
- Model performance comparison tables
- Trained model checkpoints

---

### Task 4: Multi-class Detection Models

Trains and evaluates multi-class classification models (attack type classification).

```bash
python experiments/run_task4.py --mode="full" --wandb-name="task4_multiclass"
```

**Outputs:**
- Figures: `outputs/figures/task4/`
- Tables: `outputs/tables/task4/`
- Models: `outputs/models/task4/`

**Generated Artifacts:**
- Multi-class confusion matrices
- Per-class performance metrics
- Model comparison tables
- Attack type classification results
- Trained model checkpoints

---

### Task 5: Unsupervised Change-point Anomaly Detection (UCAD)

Performs unsupervised anomaly detection using change-point detection methods.

```bash
python experiments/run_task5_ucad.py --mode="full" --wandb-name="task5_ucad"
```

**Outputs:**
- Figures: `outputs/figures/task5/`
- Tables: `outputs/tables/task5/`

**Generated Artifacts:**
- Change-point detection visualizations
- Anomaly score distributions
- Temporal analysis plots
- Unsupervised detection performance metrics

---

## Running All Tasks

To reproduce all results sequentially:

```bash
# Task 1: EDA
python experiments/run_task1.py --mode="full" --wandb-name="task1_eda"

# Task 2: Feature Characterization
python experiments/run_task2.py --mode="full" --wandb-name="task2_characterization"

# Task 3: Binary Detection
python experiments/run_task3.py --mode="full" --wandb-name="task3_binary"

# Task 4: Multi-class Detection
python experiments/run_task4.py --mode="full" --wandb-name="task4_multiclass"

# Task 5: UCAD
python experiments/run_task5_ucad.py --mode="full" --wandb-name="task5_ucad"
```

---

## Output Structure

```
outputs/
├── figures/
│   ├── task1/          # EDA visualizations
│   ├── task2/          # Feature characterization plots
│   ├── task3/          # Binary detection results
│   ├── task4/          # Multi-class detection results
│   └── task5/          # UCAD visualizations
├── tables/
│   ├── task1/          # Summary statistics
│   ├── task2/          # Feature analysis tables
│   ├── task3/          # Binary model performance
│   ├── task4/          # Multi-class model performance
│   └── task5/          # UCAD results
└── models/
    ├── task3/          # Trained binary models
    └── task4/          # Trained multi-class models
```

---

## Viewing Results

### Local Outputs

```bash
# View figures (macOS)
open outputs/figures/task1/*.png

# View figures (Linux)
xdg-open outputs/figures/task1/*.png

# View figures (Windows)
start outputs/figures/task1/*.png

# List all generated outputs
find outputs/ -type f
```

### Weights & Biases Dashboard

View all logged experiments and metrics:

1. Navigate to: https://wandb.ai/HPR-cp219/cp219-goose-ids
2. Browse runs by task name
3. Compare metrics across different runs
4. View logged figures and tables

---

## Configuration

All experiment parameters are configured in `config/config.yaml`. The configuration includes:

- Data paths and preprocessing parameters
- Model hyperparameters
- Training settings
- Output directories
- W&B project settings

**Note**: No modifications to `config/config.yaml` are required for standard reproduction. The default settings will reproduce all results from the report.

---


### Issue: UV command not found

```bash
# Add UV to PATH (macOS/Linux)
export PATH="$HOME/.cargo/bin:$PATH"

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc

# Verify installation
uv --version
```

### Issue: Data files not found

```bash
# Check data directory exists
ls data/raw/

# Verify config.yaml points to correct paths
cat config/config.yaml | grep "data"

# Ensure data files match expected names in config
```

### Issue: W&B authentication

```bash
# Re-authenticate
wandb login --relogin

# Check status
wandb status

# Run with debug mode
WANDB_DEBUG=true python experiments/run_task1.py --mode="full" --wandb-name="debug"

# Disable W&B (if needed)
export WANDB_MODE=disabled
```

### Issue: Out of memory

```bash
# Reduce batch size or use quick mode
python experiments/run_task3.py --mode="quick" --wandb-name="task3_quick"

# Monitor memory usage
# Adjust parameters in config/config.yaml if needed
```

### Issue: Missing dependencies

```bash
# Update all dependencies
uv pip install --upgrade -e ".[dev]"

# Check installed packages
uv pip list

# Verify specific package
uv pip show <package-name>
```

---

## Project Structure

```
cp219_project2/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   └── raw/                     # Raw data files (not in repo)
├── experiments/
│   ├── run_task1.py            # Task 1 runner
│   ├── run_task2.py            # Task 2 runner
│   ├── run_task3.py            # Task 3 runner
│   ├── run_task4.py            # Task 4 runner
│   └── run_task5_ucad.py       # Task 5 runner
├── src/
│   ├── tasks/                   # Task implementations
│   │   ├── task1_eda.py
│   │   ├── task2_characterization.py
│   │   ├── task3_binary.py
│   │   ├── task4_multiclass.py
│   │   └── task5_ucad.py
│   ├── models/                  # Model implementations
│   ├── utils/                   # Utility functions
│   └── preprocessing/           # Data preprocessing
├── outputs/                     # Generated outputs
│   ├── figures/
│   ├── tables/
│   └── models/
├── tests/                       # Unit tests
├── pyproject.toml              # Project dependencies
└── README.md                    # This file
```

---

## Additional Commands

```bash
# Install new package
uv pip install <package-name>

# Update specific package
uv pip install --upgrade <package-name>

# List installed packages
uv pip list

# Run tests (if available)
pytest tests/

# Format code
black src/

# Lint code
flake8 src/
```

---

## Notes on Reproducibility

- **Random Seeds**: All random seeds are set in `config/config.yaml` for reproducibility
- **Deterministic Operations**: Neural network operations use deterministic algorithms where possible
- **Environment**: Results may vary slightly across different hardware/OS due to numerical precision
- **Data**: Ensure data files in `data/raw/` match the expected format and version
- **Dependencies**: All versions are pinned in `pyproject.toml` for consistent results

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review W&B logs for detailed error messages
3. Verify configuration in `config/config.yaml`
4. Check that all dependencies are correctly installed

---

## Project Information

- **Course**: CP219
- **Project**: Project 2 - Goose IDS Detection
- **W&B Project**: HPR-cp219/cp219-goose-ids
- **Repository**: [Add your repository URL]

---

**Last Updated**: November 2025