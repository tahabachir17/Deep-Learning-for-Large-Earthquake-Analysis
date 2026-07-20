"""CLI entrypoint for model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import TrainingConfig, train_model


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except Exception as exc:
        raise ImportError("PyYAML is required to read training config files.") from exc
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return payload or {}


def _value(args: argparse.Namespace, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name)
    return config.get(name, default) if value is None else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GNSS CNN on prebuilt numpy tensors.")
    parser.add_argument("--config", help="YAML config containing data paths, shape, and training parameters.")
    parser.add_argument("--x-path", help="Path to xdata.npy")
    parser.add_argument("--y-path", help="Path to ydata.npy")
    parser.add_argument("--nst", type=int)
    parser.add_argument("--nt", type=int)
    parser.add_argument("--nc", type=int)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--decay-rate", type=float, dest="decay_rate")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", help="Root-level output folder for data_info/model/predictions artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_config(args.config)

    x_path = _value(args, config, "x_path")
    y_path = _value(args, config, "y_path")
    nst = _value(args, config, "nst")
    nt = _value(args, config, "nt")
    if x_path is None or y_path is None or nst is None or nt is None:
        raise SystemExit("Training requires --x-path, --y-path, --nst, and --nt, or a --config containing them.")

    output_dir = _value(args, config, "output_dir", "reports/training_run")
    training_config = TrainingConfig(
        nst=int(nst),
        nt=int(nt),
        nc=int(_value(args, config, "nc", 3)),
        batch_size=int(_value(args, config, "batch_size", 128)),
        epochs=int(_value(args, config, "epochs", 200)),
        learning_rate=float(_value(args, config, "learning_rate", 1e-2)),
        decay_rate=float(_value(args, config, "decay_rate", 0.9)),
        seed=int(_value(args, config, "seed", 2)),
        output_dir=str(output_dir),
    )

    x = np.load(Path(x_path))
    y = np.load(Path(y_path))
    metrics = train_model(x, y, training_config)
    payload = {
        "metrics": metrics,
        "output_dir": str(output_dir),
        "model_dir": str(Path(output_dir) / "model"),
        "predictions_dir": str(Path(output_dir) / "predictions"),
        "data_info_dir": str(Path(output_dir) / "data_info"),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
