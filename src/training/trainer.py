"""Training orchestration for the GNSS CNN."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.models.cnn import build_cnn
from src.training.callbacks import build_checkpoint_callback
from src.training.scheduler import build_exponential_decay
from src.utils.io import ensure_dir
from src.utils.seed import set_seed


@dataclass
class TrainingConfig:
    nst: int
    nt: int
    nc: int = 3
    batch_size: int = 128
    epochs: int = 200
    learning_rate: float = 1e-2
    decay_rate: float = 0.9
    seed: int = 2
    output_dir: str = "results"


def split_dataset(x: np.ndarray, y: np.ndarray, seed: int = 1) -> Tuple[np.ndarray, ...]:
    try:
        from sklearn.model_selection import train_test_split
    except Exception as exc:
        raise ImportError("scikit-learn is required for dataset splitting.") from exc

    indices = np.arange(len(y), dtype=int)
    x_tr1, x_test, ix_tr1, ix_test = train_test_split(x, indices, test_size=0.1, random_state=seed)
    x_train, x_val, ix_train, ix_val = train_test_split(x_tr1, ix_tr1, test_size=0.2, random_state=seed)
    y_train, y_val, y_test = y[ix_train], y[ix_val], y[ix_test]
    return x_train, x_val, x_test, y_train, y_val, y_test, ix_train, ix_val, ix_test


def train_model(x: np.ndarray, y: np.ndarray, config: TrainingConfig) -> Dict[str, float]:
    try:
        from tensorflow import keras
    except Exception as exc:
        raise ImportError("TensorFlow is required for training.") from exc

    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    data_info_dir = ensure_dir(output_dir / "data_info")
    model_dir = ensure_dir(output_dir / "model")
    pred_dir = ensure_dir(output_dir / "predictions")

    x_train, x_val, x_test, y_train, y_val, y_test, ix_train, ix_val, ix_test = split_dataset(x, y)
    np.save(data_info_dir / "index_datatrain.npy", ix_train)
    np.save(data_info_dir / "index_dataval.npy", ix_val)
    np.save(data_info_dir / "index_datatest.npy", ix_test)

    steps_per_epoch = max(1, len(x_train) // config.batch_size)
    lr_schedule = build_exponential_decay(config.learning_rate, steps_per_epoch, config.decay_rate)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    def lr_metric(_, __):
        return optimizer.learning_rate

    model = build_cnn(config.nst, config.nt, config.nc)
    model.compile(loss="mse", optimizer=optimizer, metrics=["mae", lr_metric])

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        verbose=2,
        callbacks=[build_checkpoint_callback(str(model_dir))],
        shuffle=True,
    )

    loss, mae_value, lr = model.evaluate(x_val, y_val, verbose=0)
    lr_value = float(lr.numpy()) if hasattr(lr, "numpy") else float(lr)
    np.savetxt(model_dir / "Validation_values.txt", (loss, mae_value, lr_value), fmt="%5.5f", header="loss, mae, lr")
    model.load_weights(model_dir / "cp.weights.h5")
    model.save(model_dir / "model.keras")

    with open(model_dir / "history.p", "wb") as handle:
        pickle.dump(history.history, handle)

    with open(model_dir / "report_model.txt", "w", encoding="utf-8") as handle:
        model.summary(print_fn=lambda line: handle.write(line + "\n"))

    y_pred = model.predict(x_test, verbose=0).reshape(len(y_test))
    y_pred_rounded = np.round(y_pred, 1)
    pred_error = y_pred_rounded - y_test
    abs_error = np.abs(pred_error)
    with open(pred_dir / "Results_Magnitude.dat", "w", encoding="utf-8") as handle:
        handle.write("Magnitude, Predicted Mag\n")
        for true_value, pred_value in zip(y_test, y_pred_rounded):
            handle.write(f"{true_value} {pred_value}\n")
    np.savetxt(
        pred_dir / "Predict_Eval.txt",
        (np.mean(abs_error), np.min(abs_error), np.max(abs_error), np.std(pred_error), np.sqrt((pred_error ** 2).mean())),
        fmt="%10.5f",
        header="mean_error, min_error, max_error, std_error, rms_error",
    )

    return {"loss": float(loss), "mae": float(mae_value), "lr": lr_value}
