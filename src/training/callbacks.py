"""Training callbacks."""

from __future__ import annotations

from src.utils.io import ensure_dir


def build_checkpoint_callback(output_dir: str):
    try:
        from tensorflow import keras
    except Exception as exc:
        raise ImportError("TensorFlow is required to build callbacks.") from exc

    path = ensure_dir(output_dir) / "cp.weights.h5"
    return keras.callbacks.ModelCheckpoint(
        filepath=str(path),
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
    )
