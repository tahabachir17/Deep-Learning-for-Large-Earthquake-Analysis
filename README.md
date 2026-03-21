# 🌍 Deep Learning for Large Earthquake Analysis using HR-GNSS Data

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)](https://keras.io/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow)](https://mlflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Inference-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A deep learning pipeline for **earthquake magnitude estimation** using High-Rate GNSS (HR-GNSS) displacement time series. This project implements a Sequential Convolutional Neural Network (CNN) trained on synthetic rupture scenarios and validated on real seismic events from multiple tectonic regions worldwide.

> Based on: *Quinteros-Cartaya et al. (2024) — "Exploring a CNN model for earthquake magnitude estimation using HR-GNSS data", Journal of South American Earth Sciences.*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference API](#inference-api)
- [CI/CD Pipeline](#cicd-pipeline)
- [Results](#results)
- [Contributing](#contributing)
- [References](#references)

---

## 🔍 Overview

HR-GNSS instruments record continuous 3-component ground displacement at 1 Hz without saturation — even for the largest earthquakes (Mw 9+). This makes them uniquely suited for rapid magnitude estimation in early warning systems.

This project trains a CNN to estimate moment magnitude (Mw) directly from raw displacement time series across multiple GNSS stations, covering magnitudes from Mw 6.6 to 9.6.

**Key capabilities:**
- Magnitude estimation from 3-component (U, N, E) displacement waveforms
- Supports 3 dataset configurations (Case I, II, III) covering different station counts and window sizes
- Cross-region generalization tested on Cascadia, Chile, Mexico, Indonesia, and Costa Rica
- Real-time inference via REST API
- Full MLOps pipeline with DVC + MLflow + GitHub Actions

---

## 📁 Project Structure

```
earthquake-dl-hrgnss/
│
├── data/
│   ├── raw/
│   │   ├── gnss_rinex/          # Raw RINEX GNSS files
│   │   ├── synthetic_chile/     # 36,800 synthetic rupture scenarios (Lin et al. 2020)
│   │   └── real_catalog/        # Real earthquake waveforms (Melgar & Ruhl 2018)
│   ├── processed/
│   │   ├── tensors/
│   │   │   ├── case_i/          # 3 stations, Δ≤3°, 181 s window
│   │   │   ├── case_ii/         # 7 stations, Δ≤3°, 181 s window
│   │   │   └── case_iii/        # 7 stations, mixed Δ, 501 s window
│   │   └── splits/              # Train/val/test indices
│   └── external/                # Third-party reference datasets
│
├── src/
│   ├── data/
│   │   ├── download.py          # Data download utilities
│   │   ├── preprocess.py        # Detrending, filtering, alignment
│   │   ├── windowing.py         # Time window extraction (181 s / 501 s)
│   │   ├── station_selection.py # Azimuth diversity + epicentral distance filtering
│   │   ├── tensor_assembly.py   # Build Ns × Nt × 3 input tensors
│   │   ├── dataset.py           # PyDataset / tf.data pipeline
│   │   └── augmentation.py      # Noise injection for robustness training
│   ├── models/
│   │   ├── cnn.py               # Sequential CNN architecture
│   │   ├── layers.py            # Custom layer definitions
│   │   └── registry.py          # Model versioning and loading
│   ├── training/
│   │   ├── trainer.py           # Training loop orchestration
│   │   ├── callbacks.py         # EarlyStopping, LR schedule, MLflow logging
│   │   ├── scheduler.py         # Learning rate decay schedule
│   │   └── losses.py            # MSE loss + custom weighted variants
│   ├── evaluation/
│   │   ├── metrics.py           # RMS, MAE, per-magnitude error analysis
│   │   ├── evaluate.py          # Full evaluation runner
│   │   └── visualize.py         # Error distribution plots (matplotlib)
│   ├── inference/
│   │   ├── predict.py           # Single-event inference
│   │   ├── api.py               # FastAPI application
│   │   └── schema.py            # Pydantic request/response schemas
│   └── utils/
│       ├── logger.py            # Structured logging
│       ├── io.py                # File I/O helpers
│       ├── geo.py               # Geodetic utilities (epicentral distance, azimuth)
│       └── seed.py              # Reproducibility seed control
│
├── configs/
│   ├── case_i.yaml              # Case I hyperparameters
│   ├── case_ii.yaml             # Case II hyperparameters
│   ├── case_iii.yaml            # Case III hyperparameters
│   ├── train_defaults.yaml      # Shared training defaults
│   └── deploy.yaml              # Inference deployment config
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_architecture.ipynb
│   ├── 03_training_analysis.ipynb
│   ├── 04_evaluation_results.ipynb
│   └── 05_real_data_testing.ipynb
│
├── tests/
│   ├── unit/                    # Per-function unit tests
│   ├── integration/             # End-to-end pipeline tests
│   └── fixtures/conftest.py     # Shared pytest fixtures
│
├── scripts/
│   ├── download_data.py
│   ├── run_preprocessing.py
│   ├── train_all_cases.ps1
│   ├── evaluate_model.py
│   ├── export_model.py
│   └── ingest_realtime.py
│
├── docker/
│   ├── Dockerfile               # Training image
│   ├── Dockerfile.api           # Inference API image
│   └── docker-compose.yml
│
├── .github/workflows/
│   ├── ci.yml                   # Lint + tests on every PR
│   ├── train.yml                # Automated retraining pipeline
│   └── deploy.yml               # Model promotion + deployment
│
├── dvc.yaml                     # DVC pipeline DAG
├── params.yaml                  # Tracked hyperparameters
├── requirements.txt
├── pyproject.toml
└── Makefile
```

---

## 🧠 Model Architecture

The model is a **Sequential 2D Convolutional Neural Network** for regression. Input is a multi-station, multi-channel displacement tensor.

```
Input: (Ns × Nt × 3)   ← stations × time steps × [U, N, E]
         │
    ┌────▼────────────────────────────────────────┐
    │  Conv2D(12)  → Conv2D(24)  → Conv2D(36)     │  Kernel: (1,3) | Stride: (1,1)
    │  MaxPool2D   → Conv2D(64)  → Conv2D(128)    │  Pool:   (1,2)
    │  MaxPool2D   → Conv2D(256) → MaxPool2D      │  ReLU after each layer
    └────────────────────────────────────────────-┘
         │
      Flatten  →  Dense(128)  →  Dense(32)  →  Dense(1)
                  ReLU            ReLU         Linear output = Mw
```

**Key design choices:**
- Kernel `(1, 3)` processes only the time axis — station features remain independent through convolutions
- Cross-station information is merged only at the Dense layers
- No label normalization — output is directly in Mw units
- Max-norm regularization (`max_norm=3`) on Dense layers
- Loss: MSE | Optimizer: ADAM | LR: 0.01 with decay schedule

---

## 📊 Dataset

| Source | Type | Events | Mw Range | Region |
|--------|------|--------|----------|--------|
| Lin et al. (2020) | Synthetic | 36,800 | 6.6 – 9.6 | Chile subduction zone |
| Melgar et al. (2016) | Synthetic | 1 | 8.7 | Cascadia (cross-region test) |
| Melgar & Ruhl (2018) | Real | 6 | 7.6 – 8.8 | Chile, Mexico, Costa Rica, Indonesia |

**Dataset split (synthetic Chile):**
- Training: 72% (~24,888 events)
- Validation: 18% (~6,222 events)
- Testing: 10% (~3,457 events)

**Three input configurations:**

| Case | Stations | Distance | Window | Input Shape |
|------|----------|----------|--------|-------------|
| I  | 3 | Δ ≤ 3° | 181 s | (3 × 181 × 3) |
| II | 7 | Δ ≤ 3° | 181 s | (7 × 181 × 3) |
| III | 7 | Mixed Δ | 501 s | (7 × 501 × 3) |

Download the datasets:
- Synthetic Chile: [zenodo.org/record/4008690](https://doi.org/10.5281/zenodo.4008690)
- Real waveforms: [zenodo.org/record/1434374](https://doi.org/10.5281/zenodo.1434374)

---

## ⚙️ Installation

**Requirements:** Python 3.9+, Git, DVC

```bash
# Clone the repository
git clone https://github.com/tahabachir17/Deep-Learning-for-Large-Earthquake-Analysis.git
cd Deep-Learning-for-Large-Earthquake-Analysis

# Create and activate virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull data with DVC (requires configured remote storage)
dvc pull
```

**Or use Docker:**
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 🚀 Quick Start

```python
from src.models.cnn import build_cnn
from src.inference.predict import predict_magnitude
import numpy as np

# Load a pre-trained model (Case III)
model = build_cnn(case="III")
model.load_weights("checkpoints/case_iii_best.h5")

# Input: displacement tensor (1 earthquake, 7 stations, 501 time steps, 3 components)
waveform = np.load("data/processed/tensors/case_iii/sample.npy")  # shape: (1, 7, 501, 3)

# Predict magnitude
mw = predict_magnitude(model, waveform)
print(f"Estimated Mw: {mw:.2f}")
```

---

## 🏋️ Training

Train all three model variants:

```bash
# Using DVC pipeline (recommended — full reproducibility)
dvc repro

# Or train a specific case manually
python scripts/train_all_cases.ps1   # Windows PowerShell
# or
python -m src.training.trainer --config configs/case_iii.yaml
```

**Key training parameters** (from `params.yaml`):
```yaml
learning_rate: 0.01
batch_size: 128
max_epochs: 200
early_stopping_patience: 20
optimizer: adam
loss: mse
max_norm: 3
```

MLflow tracking UI:
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## 📈 Evaluation

```bash
python scripts/evaluate_model.py --case III --checkpoint checkpoints/case_iii_best.h5
```

**Results summary:**

| Test Set | Case I RMS | Case II RMS | Case III RMS |
|----------|-----------|------------|-------------|
| Chile synthetic (in-distribution) | 0.114 | 0.106 | **0.069** |
| Cascadia Mw 8.7 (cross-region) | 0.130 | 0.150 | **0.110** |
| Iquique 2014 Mw 8.1 (real) | 0.14 | **0.09** | 0.33 |
| Tehuantepec 2017 Mw 8.2 (real) | 0.17 | **0.10** | 0.10 |
| Maule 2010 Mw 8.8 (real) | 0.20 | 0.28 | **0.13** |

---

## 🌐 Inference API

Start the FastAPI inference server:

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

**POST** `/predict`
```json
{
  "waveform": [[[...]]],   // shape: (Ns, Nt, 3) — float array
  "case": "III"
}
```

**Response:**
```json
{
  "magnitude_mw": 8.14,
  "case": "III",
  "model_version": "v1.0.0"
}
```

API docs available at `http://localhost:8000/docs`

---

## 🔄 CI/CD Pipeline

| Workflow | Trigger | Actions |
|----------|---------|---------|
| `ci.yml` | Every PR | pytest, flake8, mypy, tensor shape checks |
| `train.yml` | Data/config change | DVC repro on GPU runner, MLflow logging |
| `deploy.yml` | RMS gate passed | Promote model in registry, build & push Docker image |

---

## 📚 References

```bibtex
@article{quinteros2024cnn,
  title   = {Exploring a CNN model for earthquake magnitude estimation using HR-GNSS data},
  author  = {Quinteros-Cartaya, Claudia and K{\"o}hler, Jonas and Li, Wei and Faber, Johannes and Srivastava, Nishtha},
  journal = {Journal of South American Earth Sciences},
  volume  = {136},
  pages   = {104815},
  year    = {2024},
  doi     = {10.1016/j.jsames.2024.104815}
}
```

- Lin et al. (2020): Chilean Subduction Zone Rupture Scenarios — [zenodo.4008690](https://doi.org/10.5281/zenodo.4008690)
- Melgar & Ruhl (2018): High-rate GNSS Displacement Waveforms — [zenodo.1434374](https://doi.org/10.5281/zenodo.1434374)
- Melgar et al. (2016): Cascadia Kinematic Rupture Scenarios — [10.1002/2016JB013314](https://doi.org/10.1002/2016JB013314)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please make sure all tests pass before submitting a PR:
```bash
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Developed at Frankfurt Institute for Advanced Studies (FIAS) · Funded by BMBF grant SAI 01IS20059
</p>