# Earthquake DL HR-GNSS

A modular Python project for estimating earthquake moment magnitude (Mw) from high-rate GNSS displacement tensors. The codebase refactors the original notebook workflow into reusable package modules for data loading, preprocessing, model training, real-event evaluation, and API inference.

The implementation is inspired by the public DL-HRGNSS project and the Quinteros-Cartaya et al. (2024) CNN workflow, but this repository is organized as a maintainable software project rather than a notebook-only reproduction.

## Supported Cases

| Case | Stations | Window | Input shape |
| --- | ---: | ---: | --- |
| Case I | 3 | 181 s | `(3, 181, 3)` |
| Case II | 7 | 181 s | `(7, 181, 3)` |
| Case III | 7 | 501 s | `(7, 501, 3)` |

The channel order is `U, N, E`, mapped from MiniSEED components `LXZ, LXN, LXE` for real-event evaluation.

## Project Layout

```text
configs/              Case and deployment configuration
scripts/              CLI entrypoints for training, evaluation, export, and ingestion
src/data/             File discovery, preprocessing, station selection, tensor assembly
src/models/           CNN architecture and model registry/loading helpers
src/training/         Training orchestration, callbacks, losses, schedules
src/evaluation/       Metrics, plotting, real-event evaluation pipeline
src/inference/        Prediction helpers and FastAPI app
src/utils/            Logging, I/O, geodesy, reproducibility helpers
tests/                Unit and integration tests
notebooks/            Exploratory notebooks retained as references only
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

TensorFlow and ObsPy are required for model training and MiniSEED evaluation. The lightweight unit tests exercise the core NumPy and geometry code.

## Run Tests

```bash
python -m pytest -q
```

## Train From NumPy Tensors

```bash
python scripts/train_model.py \
  --x-path data/GNSS_M3S_181/xdata.npy \
  --y-path data/GNSS_M3S_181/ydata.npy \
  --nst 3 \
  --nt 181 \
  --output-dir reports/case_i
```

Convenience targets are available through `Makefile`:

```bash
make train-case-i
make train-case-ii
make train-case-iii
```

## Evaluate Real HR-GNSS Events

Expected real-data layout:

```text
data/real_events/Nicoya2012/
  disp/
    STATION.LXE.mseed
    STATION.LXN.mseed
    STATION.LXZ.mseed
  Nicoya2012_disp.chan
```

Run evaluation:

```bash
python scripts/evaluate_model.py \
  --data-root data/real_events \
  --model-case-i checkpoints/GNSS_M3S_181/model_Standard.h5 \
  --model-case-ii checkpoints/GNSS_M7S_181/model_Standard.h5 \
  --output-csv reports/real_data_results.csv
```

The evaluator samples station combinations, assembles tensors, predicts Mw in batches, and writes per-combination errors plus distance statistics.

## Inference API

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

Health check: `GET /health`

Prediction: `POST /predict` with a waveform batch and model path.

## DVC Pipeline

`params.yaml` and `dvc.yaml` define reproducible stages for the three supported training cases. After placing the expected NumPy arrays under `data/`, run:

```bash
dvc repro
```

## Reference

Quinteros-Cartaya C., Koehler J., Li W., Faber J., Srivastava N. (2024). Exploring a CNN model for earthquake magnitude estimation using HR-GNSS data. Journal of South American Earth Sciences, 136, 104815.
