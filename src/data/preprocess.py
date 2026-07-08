"""Waveform preprocessing helpers for HR-GNSS displacement tensors."""

from __future__ import annotations

from typing import Literal

import numpy as np

NormalizationMode = Literal["none", "per_station_maxabs", "per_channel_maxabs"]


def enforce_length(values: np.ndarray, nt: int) -> np.ndarray:
    """Crop or right-pad a 1D trace to exactly ``nt`` samples."""
    if nt <= 0:
        raise ValueError("nt must be positive")
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size >= nt:
        return arr[:nt].astype(np.float32, copy=False)
    out = np.zeros(nt, dtype=np.float32)
    out[: arr.size] = arr
    return out


def normalize_station_tensor(tensor: np.ndarray, mode: str | None = "per_station_maxabs") -> np.ndarray:
    """Normalize a station tensor while preserving zeros and dtype stability."""
    arr = np.asarray(tensor, dtype=np.float32)
    if mode is None or mode == "none":
        return arr
    if mode == "per_station_maxabs":
        scale = float(np.max(np.abs(arr)))
        return arr if scale == 0.0 else arr / scale
    if mode == "per_channel_maxabs":
        scale = np.max(np.abs(arr), axis=0, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return arr / scale
    raise ValueError(f"Unknown normalization mode: {mode}")
