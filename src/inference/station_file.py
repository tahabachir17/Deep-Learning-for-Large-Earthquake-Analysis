"""Load ad-hoc station files for single-event magnitude prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.files import discover_station_components, find_chan_file, load_station_tensor, parse_chan_file
from src.data.preprocess import normalize_station_tensor
from src.data.station_selection import combo_distance_stats, draw_combinations, usable_stations
from src.data.tensor_assembly import assemble_station_batch, stack_components

CSV_COMPONENT_ALIASES = {
    "up": ("u", "up", "z", "lxz", "vertical"),
    "north": ("n", "north", "lxn"),
    "east": ("e", "east", "lxe"),
}


@dataclass(frozen=True)
class Origin:
    latitude: float
    longitude: float
    depth_km: float | None = None


@dataclass(frozen=True)
class MseedEventPlan:
    origin: Origin
    usable_station_codes: list[str]
    combinations: list[tuple[str, ...]]
    station_metadata: dict


def _read_csv_station(path: Path, nt: int, normalize: str | None) -> np.ndarray:
    try:
        import pandas as pd
    except Exception as exc:
        raise ImportError("pandas is required to read CSV station files.") from exc

    frame = pd.read_csv(path)
    lowered = {str(column).strip().lower(): column for column in frame.columns}
    selected = []
    missing = []
    for component, aliases in CSV_COMPONENT_ALIASES.items():
        column = next((lowered[alias] for alias in aliases if alias in lowered), None)
        if column is None:
            missing.append(component)
        else:
            selected.append(frame[column].to_numpy(dtype=np.float32))
    if missing:
        raise ValueError(
            f"CSV station file {path} must contain U/N/E columns. "
            f"Missing: {', '.join(missing)}"
        )
    return stack_components(selected[0], selected[1], selected[2], nt=nt, normalize=normalize)


def load_station_file(path: str | Path, nt: int, normalize: str | None = "per_station_maxabs") -> np.ndarray:
    """Load one station from CSV or NPY as shape ``(nt, 3)``."""
    station_path = Path(path)
    suffix = station_path.suffix.lower()
    if suffix == ".csv":
        return _read_csv_station(station_path, nt=nt, normalize=normalize)
    if suffix == ".npy":
        arr = np.load(station_path).astype(np.float32)
        if arr.ndim == 2 and arr.shape[-1] == 3:
            if arr.shape[0] >= nt:
                out = arr[:nt]
            else:
                out = np.zeros((nt, 3), dtype=np.float32)
                out[: arr.shape[0], :] = arr
            return normalize_station_tensor(out, normalize)
        raise ValueError(f"NPY station file {path} must have shape (nt, 3); got {arr.shape}")
    raise ValueError(f"Unsupported station file extension for {path}. Use .csv or .npy")


def load_event_tensor_from_station_files(
    station_files: Sequence[str | Path],
    nst: int,
    nt: int,
    normalize: str | None = "per_station_maxabs",
    allow_repeat_single_station: bool = False,
) -> np.ndarray:
    """Load station files and return model input shape ``(1, nst, nt, 3)``."""
    paths = list(station_files)
    if len(paths) == 1 and nst > 1 and allow_repeat_single_station:
        paths = paths * nst
    if len(paths) != nst:
        raise ValueError(f"Expected {nst} station files, got {len(paths)}")
    cache = {f"S{i:02d}": load_station_file(path, nt=nt, normalize=normalize) for i, path in enumerate(paths)}
    event = assemble_station_batch(list(cache), cache)
    return event[np.newaxis, ...]


def load_preassembled_tensor(path: str | Path, nst: int, nt: int, normalize: str | None = None) -> np.ndarray:
    """Load a preassembled tensor with shape ``(nst, nt, 3)`` or ``(1, nst, nt, 3)``."""
    arr = np.load(Path(path)).astype(np.float32)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    expected = (1, nst, nt, 3)
    if arr.shape != expected:
        raise ValueError(f"Expected tensor shape {expected}, got {arr.shape}")
    if normalize:
        arr = np.stack([normalize_station_tensor(station, normalize) for station in arr[0]], axis=0)[np.newaxis, ...]
    return arr


def load_event_tensor_from_mseed_dir(
    disp_dir: str | Path,
    station_codes: Sequence[str],
    nt: int,
    normalize: str | None = "per_station_maxabs",
) -> np.ndarray:
    """Load an event tensor from a MiniSEED disp directory and station codes."""
    station_files = discover_station_components(disp_dir)
    codes = [code.upper() for code in station_codes]
    missing = [code for code in codes if code not in station_files]
    if missing:
        raise KeyError(f"Missing complete LXE/LXN/LXZ files for stations: {', '.join(missing)}")
    cache = {code: load_station_tensor(station_files[code], nt=nt, normalize=normalize) for code in codes}
    return assemble_station_batch(codes, cache)[np.newaxis, ...]


def _resolve_chan_path(disp_dir: str | Path, chan_path: str | Path | None = None) -> Path:
    if chan_path is not None:
        return Path(chan_path)
    disp_path = Path(disp_dir)
    event_dir = disp_path.parent if disp_path.name.lower() == "disp" else disp_path
    return find_chan_file(event_dir)


def plan_mseed_event_combinations(
    disp_dir: str | Path,
    chan_path: str | Path | None,
    origin_lat: float,
    origin_lon: float,
    nst: int,
    origin_depth_km: float | None = None,
    seed: int = 42,
    max_radius_deg: float | None = None,
    max_combinations: int | None = 500,
) -> MseedEventPlan:
    """Discover stations, filter by origin distance, and draw model-sized station combinations."""
    station_files = discover_station_components(disp_dir)
    if not station_files:
        raise ValueError(f"No complete LXE/LXN/LXZ station triplets found in {disp_dir}")
    metadata = parse_chan_file(_resolve_chan_path(disp_dir, chan_path), set(station_files))
    origin = Origin(latitude=float(origin_lat), longitude=float(origin_lon), depth_km=origin_depth_km)
    codes = usable_stations(station_files.keys(), metadata, origin, max_radius_deg=max_radius_deg)
    combos = draw_combinations(codes, nst=nst, seed=seed, max_combinations=max_combinations)
    return MseedEventPlan(origin=origin, usable_station_codes=codes, combinations=combos, station_metadata=metadata)


def load_mseed_combination_batch(
    disp_dir: str | Path,
    combinations: Sequence[Sequence[str]],
    nt: int,
    normalize: str | None = "per_station_maxabs",
) -> np.ndarray:
    """Load many station combinations into shape ``(n_combinations, nst, nt, 3)``."""
    station_files = discover_station_components(disp_dir)
    unique_codes = sorted({code.upper() for combo in combinations for code in combo})
    tensor_cache = {
        code: load_station_tensor(station_files[code], nt=nt, normalize=normalize)
        for code in unique_codes
    }
    return np.stack([assemble_station_batch([code.upper() for code in combo], tensor_cache) for combo in combinations], axis=0).astype(np.float32)


def summarize_combination_predictions(
    predictions: Sequence[float],
    combinations: Sequence[Sequence[str]],
    station_metadata: dict,
    origin: Origin,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    """Return aggregate Mw statistics and per-combination prediction rows."""
    arr = np.asarray(predictions, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("No predictions to summarize")
    summary = {
        "estimated_magnitude_mw": float(np.median(arr)),
        "mean_magnitude_mw": float(np.mean(arr)),
        "std_magnitude_mw": float(np.std(arr)),
        "min_magnitude_mw": float(np.min(arr)),
        "max_magnitude_mw": float(np.max(arr)),
        "n_combinations": int(arr.size),
    }
    rows: list[dict[str, object]] = []
    for combo, prediction in zip(combinations, arr):
        rows.append(
            {
                "stations": ",".join(combo),
                "pred_mw": float(prediction),
                **combo_distance_stats(combo, station_metadata, origin),
            }
        )
    return summary, rows


