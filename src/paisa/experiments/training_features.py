from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from paisa.features import FEATURE_COLUMNS as PAISA_FEATURE_COLUMNS


@dataclass(frozen=True)
class FeatureSetSpec:
    name: str
    columns: list[str]
    metadata_label: str
    max_lookback_trading_days: int


FEATURE_SHORT = [
    "open",
    "close",
    "volume",
    "daily_return",
    "log_return",
    "volume_change",
]

FEATURE_WEEK = [
    *FEATURE_SHORT,
    "close_lag_1",
    "close_lag_5",
    "volume_lag_1",
    "ma_5",
    "return_volatility_5",
]

FEATURE_MONTH = list(PAISA_FEATURE_COLUMNS)

FEATURE_SETS: dict[str, FeatureSetSpec] = {
    "FEATURE_SHORT": FeatureSetSpec(
        name="FEATURE_SHORT",
        columns=FEATURE_SHORT,
        metadata_label="short context / current and immediate recent behaviour",
        max_lookback_trading_days=1,
    ),
    "FEATURE_WEEK": FeatureSetSpec(
        name="FEATURE_WEEK",
        columns=FEATURE_WEEK,
        metadata_label="approximately five-trading-day context",
        max_lookback_trading_days=5,
    ),
    "FEATURE_MONTH": FeatureSetSpec(
        name="FEATURE_MONTH",
        columns=FEATURE_MONTH,
        metadata_label="approximately twenty-trading-day context",
        max_lookback_trading_days=20,
    ),
}

AUDIT_COLUMNS = [
    "target_date",
    "next_open",
    "next_close",
    "target_next_session_return",
    "target_next_session_up",
]

FORBIDDEN_FEATURE_COLUMNS = set(AUDIT_COLUMNS) | {
    "target_next_day_return",
    "target_next_day_up",
}


def validate_feature_sets() -> None:
    month = set(FEATURE_MONTH)
    short = set(FEATURE_SHORT)
    week = set(FEATURE_WEEK)

    missing_short = short - month
    missing_week = week - month
    if missing_short or missing_week:
        raise AssertionError(
            "Short/week feature sets must be subsets of PAISA FEATURE_COLUMNS. "
            f"missing_short={sorted(missing_short)}, missing_week={sorted(missing_week)}"
        )

    for spec in FEATURE_SETS.values():
        forbidden = sorted(set(spec.columns) & FORBIDDEN_FEATURE_COLUMNS)
        if forbidden:
            raise AssertionError(f"{spec.name} illegally includes target/audit columns: {forbidden}")


def get_feature_columns(feature_set_name: str) -> list[str]:
    validate_feature_sets()
    if feature_set_name not in FEATURE_SETS:
        raise KeyError(f"Unknown feature set: {feature_set_name}")
    return list(FEATURE_SETS[feature_set_name].columns)


def assert_no_future_or_target_columns(feature_columns: list[str]) -> None:
    forbidden = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
    future_like = sorted(
        col
        for col in feature_columns
        if col.startswith("next_") or col.startswith("target_") or col == "target_date"
    )
    if forbidden or future_like:
        raise AssertionError(
            f"Feature columns contain leakage-prone target/future columns: "
            f"{sorted(set(forbidden + future_like))}"
        )


def build_common_modelling_rows(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Restrict all variants to rows where FEATURE_MONTH and primary target are complete.

    This creates the fair-comparison date set required by the experiment:
    short, week, and month context variants must receive the same dates and target values.
    """
    validate_feature_sets()
    required = [
        "date",
        "ticker",
        "source",
        *FEATURE_MONTH,
        *AUDIT_COLUMNS,
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Modelling frame is missing required columns: {missing}")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data["target_date"] = pd.to_datetime(data["target_date"], errors="raise")
    data = data[data["ticker"].astype(str).str.upper() == ticker.upper()].copy()
    data = data.sort_values("date").reset_index(drop=True)

    data = data.replace([np.inf, -np.inf], np.nan)
    complete_required = [*FEATURE_MONTH, "target_next_session_up", "target_next_session_return", "next_open", "next_close"]
    before = len(data)
    data = data.dropna(subset=complete_required).copy()
    removed = before - len(data)

    if data.empty:
        raise ValueError("No common modelling rows remain after FEATURE_MONTH/target completeness filter.")
    if data["date"].duplicated().any():
        raise AssertionError("Duplicated feature dates remain in common modelling rows.")

    data["target_next_session_up"] = data["target_next_session_up"].astype(int)
    if data["target_next_session_up"].nunique() < 2:
        raise AssertionError("Primary target must contain both classes after filtering.")

    for feature_set in FEATURE_SETS.values():
        missing_in_set = data[feature_set.columns].isna().sum().sum()
        if missing_in_set:
            raise AssertionError(f"{feature_set.name} still has missing values on common dates: {missing_in_set}")
        assert_no_future_or_target_columns(feature_set.columns)

    if not (data["target_date"] > data["date"]).all():
        raise AssertionError("Every target_date must be strictly later than the feature date.")

    data.attrs["common_rows_removed_for_month_feature_or_target_completeness"] = int(removed)
    return data.reset_index(drop=True)
