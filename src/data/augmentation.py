"""Simple augmentation helpers."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(x: np.ndarray, noise_std: float = 0.01, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(x, dtype=np.float32) + rng.normal(0.0, noise_std, size=np.asarray(x).shape).astype(np.float32)
