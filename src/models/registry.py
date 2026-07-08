"""Model loading helpers."""

from __future__ import annotations

from pathlib import Path

from src.models.cnn import build_cnn


def load_model(model_path: str, nst: int, nt: int):
    """Load a Keras model, a local weights file, or the reference checkpoint format."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        from tensorflow import keras
    except Exception as exc:
        raise ImportError("TensorFlow is required to load trained models.") from exc

    try:
        return keras.models.load_model(path, compile=False)
    except Exception as load_error:
        model = build_cnn(nst=nst, nt=nt)
        candidates = [
            path,
            path.parent / "cp.weights.h5",
            path.parent / "cp_Standard.ckpt",
        ]
        for candidate in candidates:
            if candidate.suffix == ".h5" and candidate.exists():
                try:
                    model.load_weights(candidate)
                    return model
                except Exception:
                    continue
            if candidate.with_suffix(".ckpt.index").exists() or Path(str(candidate) + ".index").exists():
                model.load_weights(str(candidate))
                return model
        raise FileNotFoundError(
            "Could not load full model or fallback weights near " f"{path}"
        ) from load_error

