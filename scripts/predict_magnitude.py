"""CLI for estimating Mw from station folders or prebuilt tensors."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.predict import load_and_predict, predict_magnitude
from src.inference.station_file import (
    load_event_tensor_from_mseed_dir,
    load_event_tensor_from_station_files,
    load_mseed_combination_batch,
    load_preassembled_tensor,
    plan_mseed_event_combinations,
    summarize_combination_predictions,
)
from src.models.registry import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate earthquake magnitude from HR-GNSS station data.")
    parser.add_argument("--model-path", required=True, help="Path to a trained Keras model or weights file.")
    parser.add_argument("--nst", type=int, required=True, help="Number of stations expected by the model, e.g. 3 or 7.")
    parser.add_argument("--nt", type=int, required=True, help="Time samples expected by the model, e.g. 181 or 501.")
    parser.add_argument("--normalize", default="per_station_maxabs", choices=["per_station_maxabs", "per_channel_maxabs", "none"], help="Station normalization mode.")
    parser.add_argument("--round-digits", type=int, default=1, help="Round predicted Mw to this many decimals.")

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--tensor-path", help="Preassembled .npy tensor with shape (nst, nt, 3) or (1, nst, nt, 3).")
    inputs.add_argument("--station-file", action="append", help="CSV/NPY station file. Repeat once per station.")
    inputs.add_argument("--mseed-dir", help="MiniSEED disp folder containing STATION.LXE/LXN/LXZ.mseed files.")

    parser.add_argument("--station", action="append", help="Station code to read from --mseed-dir. If omitted, stations are selected automatically from the .chan file.")
    parser.add_argument("--chan-path", help="Path to the event .chan metadata file. Defaults to the parent folder of --mseed-dir.")
    parser.add_argument("--origin-lat", type=float, help="Earthquake origin latitude. Required for automatic --mseed-dir station selection.")
    parser.add_argument("--origin-lon", type=float, help="Earthquake origin longitude. Required for automatic --mseed-dir station selection.")
    parser.add_argument("--origin-depth-km", type=float, default=None, help="Optional earthquake origin depth in kilometers; included in output metadata.")
    parser.add_argument("--max-radius-deg", type=float, default=None, help="Optional maximum epicentral distance in degrees for station selection.")
    parser.add_argument("--max-combinations", type=int, default=500, help="Maximum station combinations to evaluate when selecting automatically.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for station-combination sampling.")
    parser.add_argument("--output-combinations-csv", help="Optional CSV path for per-combination predictions.")
    parser.add_argument("--allow-repeat-single-station", action="store_true", help="Duplicate one station file to fill nst. Useful only for smoke tests, not scientific estimates.")
    return parser.parse_args()


def _write_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _predict_single_tensor(args: argparse.Namespace, waveform: np.ndarray) -> dict[str, object]:
    prediction = load_and_predict(
        args.model_path,
        waveform,
        nst=args.nst,
        nt=args.nt,
        round_digits=args.round_digits,
    )
    return {
        "estimated_magnitude_mw": float(prediction[0]),
        "input_shape": list(waveform.shape),
        "model_path": args.model_path,
    }


def main() -> None:
    args = parse_args()
    normalize = None if args.normalize == "none" else args.normalize

    if args.tensor_path:
        waveform = load_preassembled_tensor(args.tensor_path, nst=args.nst, nt=args.nt, normalize=normalize)
        print(json.dumps(_predict_single_tensor(args, waveform), indent=2))
        return

    if args.station_file:
        waveform = load_event_tensor_from_station_files(
            args.station_file,
            nst=args.nst,
            nt=args.nt,
            normalize=normalize,
            allow_repeat_single_station=args.allow_repeat_single_station,
        )
        print(json.dumps(_predict_single_tensor(args, waveform), indent=2))
        return

    if args.station:
        if len(args.station) != args.nst:
            raise SystemExit(f"Expected {args.nst} --station values, got {len(args.station)}")
        waveform = load_event_tensor_from_mseed_dir(args.mseed_dir, args.station, nt=args.nt, normalize=normalize)
        payload = _predict_single_tensor(args, waveform)
        payload["selected_stations"] = [station.upper() for station in args.station]
        print(json.dumps(payload, indent=2))
        return

    if args.origin_lat is None or args.origin_lon is None:
        raise SystemExit("--origin-lat and --origin-lon are required for automatic --mseed-dir station selection")

    plan = plan_mseed_event_combinations(
        disp_dir=args.mseed_dir,
        chan_path=args.chan_path,
        origin_lat=args.origin_lat,
        origin_lon=args.origin_lon,
        nst=args.nst,
        origin_depth_km=args.origin_depth_km,
        seed=args.seed,
        max_radius_deg=args.max_radius_deg,
        max_combinations=args.max_combinations,
    )
    waveform = load_mseed_combination_batch(
        args.mseed_dir,
        plan.combinations,
        nt=args.nt,
        normalize=normalize,
    )
    model = load_model(args.model_path, nst=args.nst, nt=args.nt)
    predictions = predict_magnitude(model, waveform, round_digits=args.round_digits)
    summary, rows = summarize_combination_predictions(
        predictions,
        plan.combinations,
        plan.station_metadata,
        plan.origin,
    )
    payload = {
        **summary,
        "origin": {"latitude": args.origin_lat, "longitude": args.origin_lon, "depth_km": args.origin_depth_km},
        "n_usable_stations": len(plan.usable_station_codes),
        "usable_stations": plan.usable_station_codes,
        "input_shape": list(waveform.shape),
        "model_path": args.model_path,
        "aggregation": "median over station combinations",
    }
    if args.output_combinations_csv:
        _write_rows_csv(args.output_combinations_csv, rows)
        payload["combinations_csv"] = args.output_combinations_csv
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
