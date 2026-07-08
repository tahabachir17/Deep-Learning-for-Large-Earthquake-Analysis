"""Helpers for assembling model-ready station tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from src.data.preprocess import enforce_length, normalize_station_tensor


def stack_components(
    up: np.ndarray,
    north: np.ndarray,
    east: np.ndarray,
    nt: int,
    normalize: str | None = None,
) -> np.ndarray:
    """Return a single station tensor with shape ``(nt, 3)`` in U, N, E order."""
    tensor = np.stack(
        [
            enforce_length(up, nt),
            enforce_length(north, nt),
            enforce_length(east, nt),
        ],
        axis=-1,
    )
    return normalize_station_tensor(tensor, normalize)


def assemble_station_batch(station_codes: Sequence[str], tensor_cache: Mapping[str, np.ndarray]) -> np.ndarray:
    """Stack station tensors into shape ``(nst, nt, 3)``."""
    if not station_codes:
        raise ValueError("station_codes must contain at least one station")
    missing = [code for code in station_codes if code not in tensor_cache]
    if missing:
        raise KeyError(f"Missing station tensors: {', '.join(missing)}")
    batch = np.stack([np.asarray(tensor_cache[code], dtype=np.float32) for code in station_codes], axis=0)
    if batch.ndim != 3 or batch.shape[-1] != 3:
        raise ValueError(f"Expected station batch shape (nst, nt, 3), got {batch.shape}")
    return batch
