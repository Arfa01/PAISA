from __future__ import annotations

import pandas as pd

from paisa.experiments.training_targets import add_next_session_target


def test_next_session_target_uses_next_open_to_next_close():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-02", "2026-01-05"],
            "ticker": ["OGDC", "OGDC", "OGDC"],
            "source": ["unit", "unit", "unit"],
            "open": [100.0, 110.0, 120.0],
            "close": [105.0, 100.0, 130.0],
            "volume": [10, 11, 12],
        }
    )
    result = add_next_session_target(frame, ticker="OGDC")

    first = result.iloc[0]
    assert first["target_date"].strftime("%Y-%m-%d") == "2026-01-02"
    assert first["next_open"] == 110.0
    assert first["next_close"] == 100.0
    assert first["target_next_session_up"] == 0

    second = result.iloc[1]
    assert second["target_date"].strftime("%Y-%m-%d") == "2026-01-05"
    assert second["target_next_session_up"] == 1
