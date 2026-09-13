from __future__ import annotations

import pandas as pd

from paisa.preprocessing import standardize_price_frame


def test_standardize_accepts_common_provider_columns():
    raw = pd.DataFrame(
        {
            "Date": ["2021-01-01", "2021-01-04"],
            "Open Price": ["100.5", "101.5"],
            "Close Price": ["101.0", "100.75"],
            "Volume": ["1,000", "2,000"],
        }
    )
    out = standardize_price_frame(raw, ticker="OGDC", source="unit_test")
    assert list(out.columns) == ["date", "ticker", "open", "high", "low", "close", "volume", "source"]
    assert out.loc[0, "ticker"] == "OGDC"
    assert out.loc[0, "volume"] == 1000
