"""Station filtering, combination sampling, and geometry summaries."""

from __future__ import annotations

import random
from itertools import combinations
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from src.utils.geo import azimuth_deg, haversine_km, km_to_deg


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"Missing any of fields: {', '.join(names)}")


def draw_combinations(
    station_codes: Sequence[str],
    nst: int,
    seed: int = 42,
    max_combinations: Optional[int] = None,
) -> list[tuple[str, ...]]:
    """Generate reproducible station combinations, optionally capped."""
    if nst <= 0:
        raise ValueError("nst must be positive")
    codes = sorted({str(code).upper() for code in station_codes})
    if len(codes) < nst:
        raise ValueError(f"Only {len(codes)} stations available but {nst} required")
    combos = list(combinations(codes, nst))
    random.Random(seed).shuffle(combos)
    return combos[:max_combinations] if max_combinations is not None else combos


def usable_stations(
    station_codes: Iterable[str],
    station_meta: dict[str, Any],
    event_meta: Any,
    max_radius_deg: Optional[float] = None,
) -> list[str]:
    """Return stations with coordinates and optional epicentral distance filtering."""
    event_lat = float(_field(event_meta, "latitude", "lat"))
    event_lon = float(_field(event_meta, "longitude", "lon"))
    usable: list[str] = []
    for code in sorted({str(code).upper() for code in station_codes}):
        meta = station_meta.get(code)
        if meta is None:
            continue
        try:
            lat = float(_field(meta, "lat", "latitude"))
            lon = float(_field(meta, "lon", "longitude"))
        except (AttributeError, TypeError, ValueError):
            continue
        if max_radius_deg is not None:
            if km_to_deg(haversine_km(event_lat, event_lon, lat, lon)) > max_radius_deg:
                continue
        usable.append(code)
    return usable


def combo_distance_stats(combo: Sequence[str], station_meta: dict[str, Any], event_meta: Any) -> dict[str, float]:
    """Compute distance and azimuth summary statistics for a station combination."""
    event_lat = float(_field(event_meta, "latitude", "lat"))
    event_lon = float(_field(event_meta, "longitude", "lon"))
    distances_km: list[float] = []
    distances_deg: list[float] = []
    azimuths: list[float] = []
    for code in combo:
        meta = station_meta[str(code).upper()]
        lat = float(_field(meta, "lat", "latitude"))
        lon = float(_field(meta, "lon", "longitude"))
        distance_km = haversine_km(event_lat, event_lon, lat, lon)
        distances_km.append(distance_km)
        distances_deg.append(km_to_deg(distance_km))
        azimuths.append(azimuth_deg(event_lat, event_lon, lat, lon))
    return {
        "median_distance_deg": float(np.median(distances_deg)),
        "median_distance_km": float(np.median(distances_km)),
        "min_distance_deg": float(np.min(distances_deg)),
        "max_distance_deg": float(np.max(distances_deg)),
        "median_azimuth_deg": float(np.median(azimuths)),
    }
