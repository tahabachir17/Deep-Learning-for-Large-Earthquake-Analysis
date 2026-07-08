import numpy as np

from src.data.preprocess import enforce_length, normalize_station_tensor


def test_enforce_length_pads():
    out = enforce_length(np.array([1, 2], dtype=np.float32), 4)
    assert out.tolist() == [1.0, 2.0, 0.0, 0.0]


def test_normalize_station_tensor_maxabs():
    tensor = np.array([[1.0, -2.0, 0.5]], dtype=np.float32)
    out = normalize_station_tensor(tensor, "per_station_maxabs")
    assert np.max(np.abs(out)) == 1.0
