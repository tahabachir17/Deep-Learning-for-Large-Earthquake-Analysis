import numpy as np
import pandas as pd
import pytest

from src.inference.station_file import (
    load_event_tensor_from_station_files,
    load_preassembled_tensor,
    load_station_file,
    plan_mseed_event_combinations,
    summarize_combination_predictions,
)


def test_load_station_csv(tmp_path):
    path = tmp_path / "station.csv"
    pd.DataFrame({"U": [1.0, 2.0], "N": [0.5, 0.0], "E": [-1.0, 1.0]}).to_csv(path, index=False)
    tensor = load_station_file(path, nt=4, normalize="none")
    assert tensor.shape == (4, 3)
    assert tensor[0].tolist() == [1.0, 0.5, -1.0]
    assert tensor[-1].tolist() == [0.0, 0.0, 0.0]


def test_load_event_tensor_from_station_files(tmp_path):
    station_paths = []
    for idx in range(3):
        path = tmp_path / f"s{idx}.npy"
        np.save(path, np.ones((5, 3), dtype=np.float32) * idx)
        station_paths.append(path)
    tensor = load_event_tensor_from_station_files(station_paths, nst=3, nt=5, normalize="none")
    assert tensor.shape == (1, 3, 5, 3)


def test_load_event_tensor_requires_expected_station_count(tmp_path):
    path = tmp_path / "s0.npy"
    np.save(path, np.ones((5, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Expected 3 station files"):
        load_event_tensor_from_station_files([path], nst=3, nt=5)


def test_load_preassembled_tensor(tmp_path):
    path = tmp_path / "event.npy"
    np.save(path, np.zeros((3, 5, 3), dtype=np.float32))
    tensor = load_preassembled_tensor(path, nst=3, nt=5)
    assert tensor.shape == (1, 3, 5, 3)


def test_plan_mseed_event_combinations_uses_chan_metadata(tmp_path):
    event_dir = tmp_path / "Aegean2014"
    disp_dir = event_dir / "disp"
    disp_dir.mkdir(parents=True)
    for station in ["ALEX", "CANA", "IPSA", "LEMN"]:
        for channel in ["LXE", "LXN", "LXZ"]:
            (disp_dir / f"{station}.{channel}.mseed").write_bytes(b"")
    chan = event_dir / "Aegean2014_disp.chan"
    chan.write_text(
        "# net,sta,loc,chan,lat,lon,elev,samplerate,gain,units\n"
        "RK ALEX 00 LXE 40.8492 25.8534 0.0 1.00 1.000000e+04 counts/cm\n"
        "RK CANA 00 LXE 40.1112 26.4143 0.0 1.00 1.000000e+04 counts/cm\n"
        "RK IPSA 00 LXE 40.9175 26.3798 0.0 1.00 1.000000e+04 counts/cm\n"
        "RK LEMN 00 LXE 39.8972 25.1806 0.0 1.00 1.000000e+04 counts/cm\n",
        encoding="utf-8",
    )

    plan = plan_mseed_event_combinations(
        disp_dir=disp_dir,
        chan_path=None,
        origin_lat=40.3,
        origin_lon=25.7,
        nst=3,
        max_combinations=2,
    )

    assert len(plan.usable_station_codes) == 4
    assert len(plan.combinations) == 2
    assert all(len(combo) == 3 for combo in plan.combinations)


def test_summarize_combination_predictions():
    metadata = {
        "AAA": {"lat": 0.0, "lon": 0.0},
        "BBB": {"lat": 0.0, "lon": 1.0},
        "CCC": {"lat": 1.0, "lon": 0.0},
    }
    summary, rows = summarize_combination_predictions(
        predictions=[7.1, 7.3, 7.2],
        combinations=[("AAA", "BBB", "CCC"), ("AAA", "BBB", "CCC"), ("AAA", "BBB", "CCC")],
        station_metadata=metadata,
        origin={"latitude": 0.0, "longitude": 0.0},
    )
    assert summary["estimated_magnitude_mw"] == pytest.approx(7.2)
    assert len(rows) == 3
