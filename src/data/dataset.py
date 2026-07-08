"""Dataset helpers for numpy-backed training tensors."""

from __future__ import annotations

import numpy as np


def load_numpy_dataset(x_path: str, y_path: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(x_path)
    y = np.load(y_path)
    return x, y
