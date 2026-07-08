"""Inference helpers."""

from __future__ import annotations

import numpy as np

from src.models.registry import load_model


def predict_magnitude(model, waveform: np.ndarray, round_digits: int | None = None) -> np.ndarray:
    predictions = model.predict(np.asarray(waveform, dtype=np.float32), verbose=0).reshape(-1)
    if round_digits is not None:
        predictions = np.round(predictions, round_digits)
    return predictions


def load_and_predict(model_path: str, waveform: np.ndarray, nst: int, nt: int, round_digits: int | None = None) -> np.ndarray:
    model = load_model(model_path, nst=nst, nt=nt)
    return predict_magnitude(model, waveform, round_digits=round_digits)
