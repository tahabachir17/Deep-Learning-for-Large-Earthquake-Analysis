"""CNN architecture from the notebook and paper."""

from __future__ import annotations


def _tf_imports():
    try:
        from tensorflow.keras.constraints import max_norm
        from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
        from tensorflow.keras.models import Sequential
    except Exception as exc:
        raise ImportError("TensorFlow is required to build the CNN model.") from exc
    return Sequential, Conv2D, Dense, Flatten, MaxPooling2D, max_norm


def build_cnn(nst: int, nt: int, nc: int = 3):
    Sequential, Conv2D, Dense, Flatten, MaxPooling2D, max_norm = _tf_imports()
    model = Sequential(name=f"GNSS_CNN_{nst}S_{nt}t")
    model.add(Conv2D(12, (1, 3), activation="relu", input_shape=(nst, nt, nc)))
    model.add(MaxPooling2D((1, 2)))
    model.add(Conv2D(24, (1, 3), activation="relu", padding="same"))
    model.add(Conv2D(32, (1, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((1, 2)))
    model.add(Conv2D(64, (1, 3), activation="relu", padding="same"))
    model.add(Conv2D(128, (1, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D((1, 2)))
    model.add(Conv2D(256, (1, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(128, activation="relu", kernel_constraint=max_norm(3)))
    model.add(Dense(32, activation="relu", kernel_constraint=max_norm(3)))
    model.add(Dense(1, activation="linear"))
    return model
