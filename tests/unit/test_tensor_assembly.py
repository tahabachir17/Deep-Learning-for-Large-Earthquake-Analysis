import numpy as np

from src.data.tensor_assembly import assemble_station_batch, stack_components


def test_stack_components_shape():
    tensor = stack_components(np.arange(3), np.arange(3), np.arange(3), nt=3)
    assert tensor.shape == (3, 3)


def test_assemble_station_batch_shape():
    cache = {"A": np.zeros((5, 3), dtype=np.float32), "B": np.ones((5, 3), dtype=np.float32)}
    out = assemble_station_batch(["A", "B"], cache)
    assert out.shape == (2, 5, 3)
