"""Real-event evaluation pipeline extracted from the notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.files import (
    discover_station_components,
    find_chan_file,
    get_disp_folder,
    load_station_tensor,
    parse_chan_file,
)
from src.data.station_selection import combo_distance_stats, draw_combinations, usable_stations
from src.data.tensor_assembly import assemble_station_batch
from src.models.registry import load_model

KNOWN_EVENT_COORDS = {
    "Nicoya2012": {"lat": 10.085, "lon": -85.315, "depth_km": 20.0, "magnitude": 7.6},
    "Mentawai2010": {"lat": -3.490, "lon": 100.080, "depth_km": 20.0, "magnitude": 7.7},
    "Iquique2014": {"lat": -19.610, "lon": -70.770, "depth_km": 25.0, "magnitude": 8.1},
    "Tehuantepec2017": {"lat": 14.760, "lon": -94.100, "depth_km": 47.0, "magnitude": 8.2},
    "Illapel2015": {"lat": -31.640, "lon": -71.740, "depth_km": 22.4, "magnitude": 8.3},
    "Maule2010": {"lat": -35.910, "lon": -72.730, "depth_km": 35.0, "magnitude": 8.8},
}


@dataclass(frozen=True)
class EventMeta:
    event_id: str
    latitude: float
    longitude: float
    depth_km: float
    magnitude: float


def _event_meta(event_folder: str | Path) -> EventMeta:
    event_name = Path(event_folder).resolve().name
    if event_name not in KNOWN_EVENT_COORDS:
        raise ValueError(f"Event '{event_name}' not found in KNOWN_EVENT_COORDS.")
    info = KNOWN_EVENT_COORDS[event_name]
    return EventMeta(
        event_id=event_name,
        latitude=float(info["lat"]),
        longitude=float(info["lon"]),
        depth_km=float(info["depth_km"]),
        magnitude=float(info["magnitude"]),
    )


def evaluate_event(
    event_folder: str,
    model_path: str,
    nst: int,
    case_label: str,
    nt: int = 181,
    normalize: Optional[str] = "per_station_maxabs",
    seed: int = 42,
    max_radius_deg: Optional[float] = None,
    max_combinations: Optional[int] = 500,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Evaluate one real event for one trained case model."""
    event_meta = _event_meta(event_folder)
    disp_folder = get_disp_folder(event_folder)
    station_files = discover_station_components(disp_folder)
    station_meta = parse_chan_file(find_chan_file(event_folder), set(station_files))
    codes = usable_stations(station_files.keys(), station_meta, event_meta, max_radius_deg)
    combos = draw_combinations(codes, nst, seed=seed, max_combinations=max_combinations)

    tensor_cache = {
        code: load_station_tensor(station_files[code], nt=nt, normalize=normalize)
        for code in codes
    }
    model = load_model(model_path, nst=nst, nt=nt)

    rows: list[dict[str, object]] = []
    for start in range(0, len(combos), batch_size):
        chunk = combos[start : start + batch_size]
        x_batch = np.stack(
            [assemble_station_batch(combo, tensor_cache) for combo in chunk],
            axis=0,
        ).astype(np.float32)
        predictions = np.round(model.predict(x_batch, verbose=0).reshape(-1).astype(float), 1)
        for combo, prediction in zip(chunk, predictions):
            error = float(prediction - event_meta.magnitude)
            rows.append(
                {
                    "case": case_label,
                    "nst": nst,
                    "nt": nt,
                    "event_id": event_meta.event_id,
                    "true_mw": event_meta.magnitude,
                    "pred_mw": float(prediction),
                    "error": error,
                    "abs_error": abs(error),
                    "stations": ",".join(combo),
                    "n_usable_stations": len(codes),
                    "n_combinations": len(combos),
                    **combo_distance_stats(combo, station_meta, event_meta),
                }
            )
    return pd.DataFrame(rows)


def evaluate_all_events(
    event_folders: list[str],
    model_path_case_i: str,
    model_path_case_ii: str,
    normalize: Optional[str] = "per_station_maxabs",
    seed: int = 42,
    max_radius_deg: Optional[float] = None,
    max_combinations: Optional[int] = 500,
    batch_size: int = 128,
) -> pd.DataFrame:
    """Evaluate Case I and Case II models across event folders."""
    frames: list[pd.DataFrame] = []
    for case_label, nst, model_path in [
        ("Case I (3 stations, 181 s)", 3, model_path_case_i),
        ("Case II (7 stations, 181 s)", 7, model_path_case_ii),
    ]:
        for event_folder in event_folders:
            frames.append(
                evaluate_event(
                    event_folder=event_folder,
                    model_path=model_path,
                    nst=nst,
                    case_label=case_label,
                    nt=181,
                    normalize=normalize,
                    seed=seed,
                    max_radius_deg=max_radius_deg,
                    max_combinations=max_combinations,
                    batch_size=batch_size,
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

