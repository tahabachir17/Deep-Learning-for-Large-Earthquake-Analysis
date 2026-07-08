"""Learning-rate schedules."""

from __future__ import annotations


def build_exponential_decay(initial_learning_rate: float, steps_per_epoch: int, decay_rate: float = 0.9):
    try:
        from tensorflow import keras
    except Exception as exc:
        raise ImportError("TensorFlow is required to build the scheduler.") from exc

    return keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=max(1, steps_per_epoch),
        decay_rate=decay_rate,
        staircase=True,
    )
