"""CLI entrypoint for notebook-equivalent model training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.trainer import TrainingConfig, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GNSS CNN on prebuilt numpy tensors.")
    parser.add_argument("--x-path", required=True, help="Path to xdata.npy")
    parser.add_argument("--y-path", required=True, help="Path to ydata.npy")
    parser.add_argument("--nst", type=int, required=True)
    parser.add_argument("--nt", type=int, required=True)
    parser.add_argument("--nc", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--decay-rate", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x = np.load(Path(args.x_path))
    y = np.load(Path(args.y_path))
    config = TrainingConfig(
        nst=args.nst,
        nt=args.nt,
        nc=args.nc,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        decay_rate=args.decay_rate,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    metrics = train_model(x, y, config)
    print(metrics)


if __name__ == "__main__":
    main()
