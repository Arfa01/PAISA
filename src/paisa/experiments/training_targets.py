from __future__ import annotations

import numpy as np
import pandas as pd


def add_next_session_target(features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Create the paper-trading-aligned next-session target for one ticker.

    Features for date t use only information available no later than the close of t.
    The target measures the next trading session movement from open(t+1) to close(t+1).
    """
    required = {"date", "ticker", "open", "close", "volume"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Input feature frame missing required columns: {sorted(missing)}")

    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame[frame["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    frame = frame.sort_values("date").reset_index(drop=True)

    if frame.empty:
        raise ValueError(f"No rows found for ticker {ticker}.")
    if frame["date"].duplicated().any():
        duplicates = frame.loc[frame["date"].duplicated(), "date"].dt.strftime("%Y-%m-%d").tolist()
        raise AssertionError(f"Duplicate dates found before target creation: {duplicates[:10]}")

    for column in ["open", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["target_date"] = frame["date"].shift(-1)
    frame["next_open"] = frame["open"].shift(-1)
    frame["next_close"] = frame["close"].shift(-1)
    frame["target_next_session_return"] = frame["next_close"] / frame["next_open"] - 1.0
    frame["target_next_session_up"] = np.where(
        frame["target_date"].notna(),
        (frame["next_close"] > frame["next_open"]).astype(int),
        np.nan,
    )

    known_target = frame["target_date"].notna()
    if not (frame.loc[known_target, "target_date"] > frame.loc[known_target, "date"]).all():
        raise AssertionError("target_date must be strictly later than feature date.")

    if frame.loc[known_target, ["next_open", "next_close"]].isna().any().any():
        raise AssertionError("Known target rows must include next_open and next_close audit values.")

    zero_return_rows = frame.loc[known_target & (frame["target_next_session_return"] == 0)]
    if not zero_return_rows.empty and not (zero_return_rows["target_next_session_up"] == 0).all():
        raise AssertionError("Zero next-session returns must be assigned class 0.")

    return frame


def target_class_distribution(frame: pd.DataFrame, target_col: str = "target_next_session_up") -> dict[str, int]:
    counts = frame[target_col].dropna().astype(int).value_counts().sort_index().to_dict()
    return {str(int(key)): int(value) for key, value in counts.items()}
