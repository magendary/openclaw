from __future__ import annotations

import pandas as pd

from candlestick_v1.data import build_five_minute_snapshots
from candlestick_v1.model import ProbabilityTableModel


def test_build_five_minute_snapshots() -> None:
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range(
                "2025-01-01T00:00:00Z", periods=5, freq="1min"
            ),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "close": [101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )
    snapshots = build_five_minute_snapshots(frame)
    assert snapshots["elapsed_seconds"].tolist() == [60, 120, 180, 240]
    assert snapshots["final_up"].tolist() == [1, 1, 1, 1]
    assert snapshots["final_close"].iloc[0] == 105.0


def test_probability_table_direction() -> None:
    training = pd.DataFrame(
        {
            "elapsed_seconds": [60] * 8,
            "distance_from_open_bps": [-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0],
            "final_up": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    model = ProbabilityTableModel(
        bin_width_bps=1.0,
        min_samples=1,
        prior_strength=0.0,
        max_radius_bins=0,
    ).fit(training)

    down = model.predict_one(60, -2.2)
    up = model.predict_one(60, 2.2)
    assert down.p_up == 0.0
    assert up.p_up == 1.0
