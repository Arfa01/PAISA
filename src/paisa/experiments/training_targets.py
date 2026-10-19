"""Trading-session-aligned targets for versioned model experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from paisa.experiments.experiment_config import DEFAULT_SYMBOL, TARGET_NAME


TARGET_RETURN_COLUMN = "target_next_day_return"


def add_next_session_target(
    features: pd.DataFrame,
    ticker: str = DEFAULT_SYMBOL,
) -> pd.DataFrame:
    """Attach the next-observed-session close-to-close target for one ticker.

    A row for trading day ``t`` receives class 1 exactly when the close on the
    next observed session is greater than the close on ``t``. The final row is
    kept with a missing target so unknown is not silently treated as class 0.
    Compatibility audit aliases from the earlier RF-only work are retained,
    but feature selection rejects all ``next_`` and ``target_`` columns.
    """

    required = {"date", "ticker", "close"}
    missing = sorted(required - set(features.columns))
    if missing:
        raise ValueError(f"Target input is missing required columns: {missing}")

    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("ticker must be a non-empty symbol")

    frame = features.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame = frame[frame["ticker"] == symbol].copy()
    if frame.empty:
        raise ValueError(f"No rows found for ticker {symbol}")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any():
        raise ValueError("date must not contain missing values")
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if frame["date"].duplicated().any():
        duplicates = (
            frame.loc[frame["date"].duplicated(keep=False), "date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
        raise ValueError(f"Duplicate trading dates for {symbol}: {duplicates}")

    close = pd.to_numeric(frame["close"], errors="coerce")
    if close.isna().any() or not np.isfinite(close.to_numpy(dtype=float)).all():
        raise ValueError("close must contain only finite numeric values")
    if (close <= 0).any():
        raise ValueError("close must be strictly positive")
    frame["close"] = close

    frame["feature_date"] = frame["date"]
    frame["target_date"] = frame["date"].shift(-1)
    next_close = close.shift(-1)
    frame[TARGET_RETURN_COLUMN] = (next_close - close) / close
    frame[TARGET_NAME] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    known = frame["target_date"].notna()
    frame.loc[known, TARGET_NAME] = (
        next_close.loc[known] > close.loc[known]
    ).astype("int64")

    # Audit fields retained for compatibility with the earlier training report.
    frame["next_close"] = next_close
    if "open" in frame.columns:
        frame["next_open"] = pd.to_numeric(frame["open"], errors="coerce").shift(-1)
    frame["target_next_session_return"] = frame[TARGET_RETURN_COLUMN]
    frame["target_next_session_up"] = frame[TARGET_NAME]

    if not (
        frame.loc[known, "target_date"] > frame.loc[known, "feature_date"]
    ).all():
        raise AssertionError("target_date must be later than feature_date")
    return frame


def target_class_distribution(
    frame: pd.DataFrame,
    target_col: str = TARGET_NAME,
) -> dict[str, int]:
    """Return stable string-keyed counts for known binary targets."""

    if target_col not in frame.columns:
        raise ValueError(f"Target column is missing: {target_col}")
    counts = (
        pd.to_numeric(frame[target_col], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    return {str(int(key)): int(value) for key, value in counts.items()}


__all__ = [
    "TARGET_RETURN_COLUMN",
    "add_next_session_target",
    "target_class_distribution",
]
