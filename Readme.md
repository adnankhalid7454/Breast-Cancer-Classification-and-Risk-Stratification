# Breast Cancer Classification and Risk Stratification using M3T

A deep learning pipeline for 3D medical image analysis using the Multi-Modal Multi-Plane Transformer architecture for breast cancer classification and risk stratification.

You can download the model weights from the following link: https://doi.org/10.5281/zenodo.19711271

## 🎯 Project Overview

This project implements a 2D aggregated version of the M3T (Multi-Modal Multi-Plane Transformer) model for binary classification of breast cancer from 3D MRI volumes. The model processes pre-contrast, post-contrast, and subtraction MRI images to predict cancer risk.


## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (for GPU support)
- pip or conda

### Step 1: Clone the Repository
```bash
git clone https://github.com/adnankhalid7454/Breast-Cancer-Classification-and-Risk-Stratification.git
cd Breast-Cancer-Classification-and-Risk-Stratification
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n breast-cancer python=3.9
conda activate breast-cancer
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- torch >= 2.0.0
- pytorch-lightning >= 2.0.0
- torchio >= 0.18.0
- torchvision >= 0.15.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- numpy >= 1.21.0
- SimpleITK >= 2.1.0

## 📁 Project Structure

```
Breast-Cancer-Classification-and-Risk-Stratification/
│
├── models/
│   └── M3T_2D_aggregated.py          # M3T model architecture
│
├── data/
│   ├── data_processing.py             # Dataset classes and utilities
│   └── datamodule.py                  # PyTorch Lightning DataModule
│
├── scripts/
│   ├── testing.py                     # Testing/evaluation script
│   ├── train.py                       # Training script (if available)
│   └── inference.py                   # Inference script (optional)
│
├── configs/
│   └── config.yaml                    # Configuration file (optional)
│
├── README.md                          # This file
├── requirements.txt                   # Project dependencies
└── .gitignore                         # Git ignore file
```

## 📊 Data Preparation

### Data Organization

Organize your data in the following directory structure:

```
data/
├── breast_side_regions/
│   ├── patient_001/
│   │   ├── pre.nii.gz                 # Pre-contrast MRI
│   │   ├── post1.nii.gz               # Post-contrast MRI
│   │   └── sub.nii.gz                 # Subtraction image
│   ├── patient_002/
│   │   ├── pre.nii.gz
│   │   ├── post1.nii.gz
│   │   └── sub.nii.gz
│   └── ...
│
├── breast_side_labels.xlsx            # Labels file
└── train_test_splits.csv              # Train/test split file
```

### Labels File Format (breast_side_labels.xlsx)

| patient_id | label |
|-----------|-------|
| patient_001 | benign |
| patient_002 | malignant |
| patient_003 | benign |

### Train/Test Split File (train_test_splits.csv)

| train_split | test_split |
|-----------|-----------|
| patient_001 | patient_100 |
| patient_002 | patient_101 |
| | patient_102 |



### Inference on New Data

```python
import torch
from models.M3T_2D_aggregated import M3T
from data.data_processing import TumorDatasetTIO

# Load model
model = M3T(in_channels=3, n_classes=2, image_size=128)
checkpoint = torch.load("best_model.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Load single patient
dataset = TumorDatasetTIO(
    root_dir="path/to/patient_dir",
    labels={"patient_id": 0},
    spatial_size=(128, 128, 128)
)

# Predict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

with torch.no_grad():
    x, y = dataset[0]
    x = x.unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    
    print(f"Prediction: {pred.item()}")
    print(f"Confidence: {probs[0].max().item():.4f}")
```

## 🧪 Testing

### Run Full Test Suite

```bash
# Edit configuration in testing.py
python scripts/testing.py
```

### Test Configuration

Edit the following in `testing.py`:

```python
# Data paths
ROOT_PATIENTS = r"path/to/breast_side_regions"
LABELS_FILE = r"path/to/breast_side_labels.xlsx"
SPLIT_CSV = r"path/to/train_test_splits.csv"

# Model checkpoint
CHECKPOINT_PATH = r"path/to/best_model.pt"

# Output directory
SAVE_DIR = r"path/to/results"


### Key Hyperparameters

**Model Parameters** (in `M3T_2D_aggregated.py`):
```python
model = M3T(
    in_channels=3,           # Number of input channels (pre, post, sub)
    emb_size=256,            # Embedding dimension
    depth=8,                 # Number of transformer blocks
    n_classes=2,             # Binary classification
    image_size=128,          # Resized plane image size
    agg_mode="max",         # Aggregation: "max"
    num_heads=8,             # Attention heads
    dropout=0.1,             # Dropout rate
    forward_expansion=2,     # FFN expansion factor
)
```

## 📝 File Descriptions

### `models/M3T_2D_aggregated.py`
Implements the complete M3T architecture:
- `PlaneAggregation2D`: 3D to 2D plane conversion with ResNet50
- `EmbeddingLayer2D`: Token embeddings with positional encoding
- `MultiHeadAttention`: Self-attention mechanism
- `TransformerEncoder`: Stack of encoder blocks
- `ClassificationHead`: Binary classification output
- `M3T`: Complete model pipeline

### `data/data_processing.py`
Data utilities and dataset class:
- `TumorDatasetTIO`: PyTorch Dataset for 3D MRI volumes
- `build_tumor_datasets()`: Create train/val splits
- `_read_volume()`: Load NIfTI images
- `_make_transforms()`: Data augmentation pipeline
- `_read_labels_table()`: Parse label files
- `set_seed()`: Reproducibility

### `data/datamodule.py`
PyTorch Lightning integration:
- `custom_collate_fn()`: Batch collation with UIDs
- `DataModule`: LightningDataModule for train/val/test
- Weighted sampling for imbalanced data
- Multi-worker support with memory safety

### `scripts/testing.py`
Evaluation script:
- Load trained model checkpoint
- Evaluate on test set
- Compute metrics (AUC, F1, etc.)
- Save predictions and metrics
- Supports label mapping for readability


## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact by email.

## Acknowledgments

- DOI: 
---

