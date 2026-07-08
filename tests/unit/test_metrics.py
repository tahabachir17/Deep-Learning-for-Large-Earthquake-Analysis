from src.evaluation.metrics import mae, pct_within, rms


def test_metrics():
    values = [-0.5, 0.5]
    assert rms(values) == 0.5
    assert mae(values) == 0.5
    assert pct_within(values, 0.5) == 100.0
