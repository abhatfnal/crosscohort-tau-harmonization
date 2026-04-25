
# 🧠 ADNI MRI Classification

Framework for training and evaluating deep learning models on the [ADNI MRI dataset](https://adni.loni.usc.edu/).  
Supports flexible configuration, modular 3D CNN architectures, SPM-based preprocessing, and automatic experiment tracking.

---

## 🗂️ Requirements
- Python 3.10
- Matlab
- [SPM]( https://www.fil.ion.ucl.ac.uk/spm/software/spm12/ )

---
## 🔧 Setup Instructions

### 1. Download and Unzip SPM (if not already downloaded)

You need SPM12 for SPM-based preprocessing. If not already installed, download and unzip with the commands below. Change the unzip path before executing.

```bash
wget -O spm.zip https://github.com/spm/spm/releases/download/25.01.02/spm_25.01.02.zip
unzip spm.zip -d /path/to/unzip/
```

- This will create `/path/to/unzip/spm`.
- You may change the URL to download a different release.
- 🔗 [SPM Releases on GitHub](https://github.com/spm/spm/releases)

---

### 2. Open and edit `env_setup.sh`

Set the correct paths and environment activation commands. Example:

```bash
# 1)  Paste below the commands to activate your environment.

source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh

conda activate adni_rcc
module load matlab

# 2) Set required environment variables

# Absolute path to SPM12 folder (WITHOUT final /)
export SPM_PATH="/path/to/unzip/spm"

# Absolute path to diagnosis file DXSUM_05Apr2025.csv
export DIAGNOSIS_FILE="/path/to/DXSUM_05Apr2025.csv"

```

### 3. Install Python Dependencies

Activate your environment using

```bash
source env_setup.sh

```
Install Python requirements with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart

### 1. Activate your environment 

```bash
source env_setup.sh
```

### 2. Run preprocessing

Open one preprocessing pipeline folder under the `data/` folder (recommended: `data/preprocessing_dicom_spm`)  and follow the instructions to produce the preprocessed data.

### 3. Run training with ResNet18 model

After having updated the paths to `trainval.csv` and `test.csv` in the desired configuation file, as described in the preprocessing instructions, from the project root directory run

```bash 
python train.py --config configs/training/resnet18.yaml --job
```

---

## 📁 Repository Structure

```
├── configs/
│   ├── schema.py              # Config schema & validation
│   └── training/*.yaml        # Training configs
├── data/
│   ├── preprocessing_dicom_spm/
│   │   ├── run.sh             # Runs SPM-based preprocessing
│   │   └── data/*.nii, *.csv  # Output images and splits
│   └── DXSUM_05Apr2025.csv    # ADNI subject metadata
├── experiments/               # Logs, metrics, models per run
├── models/                    # Custom + standard model architectures
├── train.py                   # Training CLI
├── utils/                     # Tools for splits, plotting, etc
└── README.md                   # This document
```

---

## ⚙️ Configuration System

YAML files under `configs/training/` define all model, data, and training settings.

You can **merge multiple configs**, and **override any field from the CLI**.

```bash
python train.py \
  --config configs/training/default.yaml \
  --config configs/training/custom_3dcnn.yaml \
  training.epochs=300 training.optimizer.lr=1e-4
```

### 🔢 Example Config (default.yaml)

```yaml
model:
  name: resnet18
  num_classes: 3
  in_channels: 1
  init_filters: 4

training:
  batch_size: 8
  epochs: 200
  optimizer:
    name: adam
    lr: 1e-3
  scheduler:
    name: CosineAnnealingLR
    t_max: 200
    lr_max: 1e-3
    lr_min: 2e-6

data:
  trainval_csv: ./data/.../trainval.csv
  test_csv: ./data/.../test.csv
  augmentation:
    random_crop:
      p: 1

cross_validation:
  method: kfold
  folds: 5
```

### 🧮 Config Reference

| Field | Description | Example |
|-------|-------------|---------|
| `model.name` | Model architecture | `resnet18`, `custom_3dcnn` |
| `training.epochs` | Number of training epochs | `200` |
| `data.trainval_csv` | Path to training CSV | `./data/.../trainval.csv` |
| `cross_validation.method` | CV method | `kfold`, `none` |

---

## 🧠 Model Registry

- Add a model:
  1. Subclass `models/base.py`
  2. Register it in `models/registry.py`

Built-ins: `simple_3dcnn`, `resnet18`, `custom_3dcnn`, `simple_3dcnn_gradcam`.

---

## 📊 Experiment Logging

Each training run creates a folder inside `experiments/`, named by timestamp or job ID.

Contents include:
- Merged YAML config
- `losses.csv` (train/val loss)
- `metrics.csv` (acc, AUC, precision/recall)
- Confusion matrices
- Model checkpoints (`.pth`)

---

## 🗂️ Preprocessing Pipelines

Each preprocessing folder contains:
- A `run.sh` script
- A `data/` subfolder with:
  - `.nii` MRI volumes
  - `trainval.csv`, `test.csv` with labels and paths

The default is `data/preprocessing_dicom_spm/`, which uses Matlab + SPM.

---

## 🛠️ Utilities

| Script | Purpose |
|--------|---------|
| `train_test_split.py` | Split dataset into train/val/test |
| `plot_training_log.py` | Plot loss/metric curves |
| `gradcam_visualize.py` | Visualize saliency maps |
| `plot_histogram.py` | Plot class distributions |

---

## 🏷️ Data Labels

| Label | Description |
|-------|-------------|
| 1 | CN (Cognitively Normal) |
| 2 | MCI (Mild Cognitive Impairment) |
| 3 | AD (Alzheimer’s Disease) |

Source: `DXSUM_05Apr2025.csv`

---

## 🧪 Example Commands

### ✅ Local training with ResNet18

```bash
python train.py \
  --config configs/training/resnet18.yaml \
  data.trainval_csv=./data/.../trainval.csv \
  data.test_csv=./data/.../test.csv
```

### ✅ Submit to RCC with custom overrides

```bash
python train.py \
  --config configs/training/custom_3dcnn.yaml \
  --job \
  training.epochs=300 \
  training.optimizer.lr=5e-4
```

---

## 🧯 Troubleshooting

| Problem | Solution |
|--------|----------|
| `matlab: command not found` | Ensure Matlab is installed and in PATH |
| `SPM not found` | Check `SPM_DIR` in `env_setup.sh` |
| `trainval.csv not found` | Run the preprocessing script to generate splits |
| `schema validation error` | Config is missing required fields — see `schema.py` |

---

## 📎 Resources

- [ADNI Data Portal](https://adni.loni.usc.edu/)
- [SPM12 Releases](https://github.com/spm/spm/releases)

---

For questions or contributions, please open an issue or pull request.
