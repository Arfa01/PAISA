from __future__ import annotations

import pandas as pd


def chronological_split(
    dataset: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date, not random rows, to reduce time-series leakage."""
    if dataset.empty:
        raise ValueError("Cannot split an empty dataset.")
    if "date" not in dataset.columns:
        raise ValueError("Dataset must contain a date column.")

    working = dataset.copy()
    working["date"] = pd.to_datetime(working["date"])
    unique_dates = sorted(working["date"].dropna().unique())
    if len(unique_dates) < 10:
        raise ValueError("Need at least 10 unique trading dates for chronological split.")

    train_cut = max(1, int(len(unique_dates) * train_ratio))
    val_cut = max(train_cut + 1, int(len(unique_dates) * (train_ratio + validation_ratio)))
    val_cut = min(val_cut, len(unique_dates) - 1)

    train_dates = set(unique_dates[:train_cut])
    val_dates = set(unique_dates[train_cut:val_cut])
    test_dates = set(unique_dates[val_cut:])

    train = working[working["date"].isin(train_dates)].sort_values(["date", "ticker"])
    val = working[working["date"].isin(val_dates)].sort_values(["date", "ticker"])
    test = working[working["date"].isin(test_dates)].sort_values(["date", "ticker"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
