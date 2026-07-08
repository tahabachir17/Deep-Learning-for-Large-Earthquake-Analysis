"""Filesystem readers for real HR-GNSS event folders."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.data.tensor_assembly import stack_components


@dataclass(frozen=True)
class StationMeta:
    code: str
    net: Optional[str] = None
    loc: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    elev: Optional[float] = None
    samplerate: Optional[float] = None
    gain: Optional[float] = None
    units: Optional[str] = None


def get_disp_folder(event_folder: str | Path) -> Path:
    path = Path(event_folder) / "disp"
    if not path.is_dir():
        raise FileNotFoundError(f"'disp' folder not found inside: {event_folder}")
    return path


def find_chan_file(event_folder: str | Path) -> Path:
    candidates = sorted(Path(event_folder).glob("*.chan"))
    if not candidates:
        raise FileNotFoundError(f"No .chan file found in: {event_folder}")
    preferred = [path for path in candidates if path.name.lower().endswith("_disp.chan")]
    return preferred[0] if preferred else candidates[0]


def discover_station_components(disp_folder: str | Path) -> dict[str, dict[str, Path]]:
    """Find stations with LXE, LXN, and LXZ MiniSEED components."""
    pattern = re.compile(r"^([^.]+)\.(LXE|LXN|LXZ)\.mseed$", re.IGNORECASE)
    station_files: dict[str, dict[str, Path]] = {}
    for path in Path(disp_folder).iterdir():
        match = pattern.match(path.name)
        if match is None:
            continue
        station = match.group(1).upper()
        component = match.group(2).upper()
        station_files.setdefault(station, {})[component] = path
    return {
        station: components
        for station, components in station_files.items()
        if {"LXE", "LXN", "LXZ"}.issubset(components)
    }


def parse_chan_file(chan_path: str | Path, valid_station_codes: set[str]) -> dict[str, StationMeta]:
    """Parse a Ruhl-style ``.chan`` metadata file."""
    valid = {code.upper() for code in valid_station_codes}
    metadata: dict[str, StationMeta] = {}
    with Path(chan_path).open(encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            station = parts[1].upper()
            if station not in valid or station in metadata:
                continue
            metadata[station] = StationMeta(
                code=station,
                net=parts[0],
                loc=parts[2],
                lat=float(parts[4]),
                lon=float(parts[5]),
                elev=float(parts[6]),
                samplerate=float(parts[7]),
                gain=float(parts[8]),
                units=parts[9],
            )
    return metadata


def read_mseed_trace(path: str | Path) -> np.ndarray:
    try:
        from obspy import read
    except Exception as exc:
        raise ImportError("ObsPy is required to read MiniSEED files.") from exc
    stream = read(str(path))
    if not stream:
        raise ValueError(f"No traces found in {path}")
    return np.asarray(stream[0].data, dtype=np.float32)


def load_station_tensor(
    component_paths: dict[str, str | Path],
    nt: int,
    normalize: str | None = "per_station_maxabs",
) -> np.ndarray:
    """Load one station as ``(nt, 3)`` in U, N, E order from LXZ, LXN, LXE."""
    return stack_components(
        up=read_mseed_trace(component_paths["LXZ"]),
        north=read_mseed_trace(component_paths["LXN"]),
        east=read_mseed_trace(component_paths["LXE"]),
        nt=nt,
        normalize=normalize,
    )
