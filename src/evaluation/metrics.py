"""Evaluation metrics."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def rms(errors: Iterable[float]) -> float:
    arr = np.asarray(list(errors), dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else float("nan")


def mae(errors: Iterable[float]) -> float:
    arr = np.asarray(list(errors), dtype=float)
    return float(np.mean(np.abs(arr))) if len(arr) else float("nan")


def pct_within(errors: Iterable[float], threshold: float = 0.5) -> float:
    arr = np.asarray(list(errors), dtype=float)
    return float(np.mean(np.abs(arr) <= threshold) * 100.0) if len(arr) else float("nan")
