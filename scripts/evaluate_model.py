"""CLI entrypoint for real-event evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.evaluate import evaluate_all_events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained GNSS magnitude models on real events.")
    parser.add_argument("--data-root", required=True, help="Directory containing one subdirectory per event.")
    parser.add_argument("--model-case-i", required=True, help="Path to the Case I trained model.")
    parser.add_argument("--model-case-ii", required=True, help="Path to the Case II trained model.")
    parser.add_argument("--output-csv", default="real_data_results.csv")
    parser.add_argument("--max-combinations", type=int, default=500)
    parser.add_argument("--max-radius-deg", type=float, default=None)
    parser.add_argument("--normalize", default="per_station_maxabs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_folders = [str(path) for path in sorted(Path(args.data_root).iterdir()) if path.is_dir()]
    results = evaluate_all_events(
        event_folders=event_folders,
        model_path_case_i=args.model_case_i,
        model_path_case_ii=args.model_case_ii,
        normalize=args.normalize,
        seed=args.seed,
        max_radius_deg=args.max_radius_deg,
        max_combinations=args.max_combinations,
        batch_size=args.batch_size,
    )
    Path(args.output_csv).write_text(results.to_csv(index=False), encoding="utf-8")


if __name__ == "__main__":
    main()
