"""Window extraction helpers."""

from __future__ import annotations

import numpy as np



def slice_time_window(arr: np.ndarray, nt: int, start: int = 0) -> np.ndarray:
    return np.asarray(arr)[start : start + nt]
