from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from paisa.experiments.training_targets import (
    add_next_session_target,
    target_class_distribution,
)


def test_target_uses_next_observed_close_not_next_session_intraday_direction():
    original = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-05", "2026-01-06"],
            "ticker": ["OGDC", "OGDC", "OGDC"],
            "source": ["unit", "unit", "unit"],
            "open": [90.0, 200.0, 100.0],
            "close": [100.0, 150.0, 150.0],
            "volume": [10, 11, 12],
        }
    )
    before = original.copy(deep=True)

    result = add_next_session_target(original)

    assert_frame_equal(original, before)
    assert result["feature_date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
    ]
    assert result.loc[0, "target_date"].strftime("%Y-%m-%d") == "2026-01-05"
    # Monday is down from its own open, but up versus Friday's close. The
    # contract therefore assigns class 1.
    assert result.loc[0, "target_next_day_up"] == 1
    assert result.loc[1, "target_next_day_up"] == 0
    assert pd.isna(result.loc[2, "target_next_day_up"])
    assert pd.isna(result.loc[2, "target_date"])
    assert target_class_distribution(result) == {"0": 1, "1": 1}


def test_target_filters_one_symbol_and_sorts_without_calendar_day_assumptions():
    frame = pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-02", "2026-01-02"],
            "ticker": ["ogdc", "OGDC", "HBL"],
            "close": [101.0, 100.0, 50.0],
        }
    )
    result = add_next_session_target(frame, ticker="ogdc")
    assert result["ticker"].tolist() == ["OGDC", "OGDC"]
    assert result.loc[0, "target_date"].strftime("%Y-%m-%d") == "2026-01-05"
    assert result.loc[0, "target_next_day_up"] == 1


def test_target_rejects_duplicate_trading_dates_and_invalid_prices():
    duplicate = pd.DataFrame(
        {
            "date": ["2026-01-02", "2026-01-02"],
            "ticker": ["OGDC", "OGDC"],
            "close": [100.0, 101.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate trading dates"):
        add_next_session_target(duplicate)

    invalid = duplicate.iloc[:1].copy()
    invalid.loc[0, "close"] = float("inf")
    with pytest.raises(ValueError, match="finite numeric"):
        add_next_session_target(invalid)
