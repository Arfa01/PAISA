"""Canonical feature preparation and chronological target-date splitting."""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from paisa.experiments.experiment_config import (
    DEFAULT_SYMBOL,
    FEATURE_SET_ID,
    TARGET_NAME,
)
from paisa.features import FEATURE_COLUMNS as PAISA_FEATURE_COLUMNS


@dataclass(frozen=True, slots=True)
class FeatureSetSpec:
    name: str
    columns: tuple[str, ...]
    metadata_label: str
    max_lookback_trading_days: int


# Earlier names remain available for compatibility, while all required model
# variants deliberately select FEATURE_MONTH so dates/features are identical.
FEATURE_SHORT = (
    "open",
    "close",
    "volume",
    "daily_return",
    "log_return",
    "volume_change",
)
FEATURE_WEEK = (
    *FEATURE_SHORT,
    "close_lag_1",
    "close_lag_5",
    "volume_lag_1",
    "ma_5",
    "return_volatility_5",
)
FEATURE_MONTH = tuple(PAISA_FEATURE_COLUMNS)
TECHNICAL_FEATURE_COLUMNS = FEATURE_MONTH

FEATURE_SETS: Mapping[str, FeatureSetSpec] = {
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
        metadata_label="existing PAISA daily technical feature set",
        max_lookback_trading_days=20,
    ),
}

AUDIT_COLUMNS = (
    "feature_date",
    "target_date",
    "next_open",
    "next_close",
    "target_next_day_return",
    "target_next_day_up",
    "target_next_session_return",
    "target_next_session_up",
)
FORBIDDEN_FEATURE_COLUMNS = set(AUDIT_COLUMNS) | {
    "actual_class",
    "date",
    "ticker",
    "source",
}


def validate_feature_columns(feature_columns: Sequence[str]) -> tuple[str, ...]:
    """Validate ordered features and reject future/target/metadata fields."""

    columns = tuple(feature_columns)
    if not columns:
        raise ValueError("feature_columns must not be empty")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns must not contain duplicates")
    forbidden = [
        column
        for column in columns
        if column in FORBIDDEN_FEATURE_COLUMNS
        or column.startswith("target_")
        or column.startswith("next_")
    ]
    if forbidden:
        raise ValueError(
            f"Feature columns contain target, future, or metadata fields: {forbidden}"
        )
    unknown = [column for column in columns if column not in PAISA_FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"Feature columns must reuse the existing PAISA technical set: {unknown}"
        )
    return columns


def validate_feature_sets() -> None:
    month = set(FEATURE_MONTH)
    for spec in FEATURE_SETS.values():
        missing = set(spec.columns) - month
        if missing:
            raise AssertionError(
                f"{spec.name} must be a subset of PAISA FEATURE_COLUMNS: {sorted(missing)}"
            )
        validate_feature_columns(spec.columns)


def get_feature_columns(feature_set_name: str) -> list[str]:
    validate_feature_sets()
    try:
        return list(FEATURE_SETS[feature_set_name].columns)
    except KeyError as exc:
        raise KeyError(f"Unknown feature set: {feature_set_name}") from exc


def assert_no_future_or_target_columns(feature_columns: Sequence[str]) -> None:
    validate_feature_columns(feature_columns)


def prepare_model_frame(
    frame: pd.DataFrame,
    ticker: str = DEFAULT_SYMBOL,
    feature_columns: Sequence[str] = TECHNICAL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Return complete, ordered model rows without leakage-prone columns."""

    columns = validate_feature_columns(feature_columns)
    required = {"ticker", "feature_date", "target_date", TARGET_NAME, *columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {missing}")

    symbol = str(ticker).strip().upper()
    if not symbol:
        raise ValueError("ticker must be a non-empty symbol")
    working = frame.copy()
    working["ticker"] = working["ticker"].astype(str).str.strip().str.upper()
    working = working[working["ticker"] == symbol].copy()
    if working.empty:
        raise ValueError(f"No rows found for ticker {symbol}")

    working["feature_date"] = pd.to_datetime(
        working["feature_date"], errors="raise"
    )
    working["target_date"] = pd.to_datetime(working["target_date"], errors="coerce")
    for column in columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working[TARGET_NAME] = pd.to_numeric(working[TARGET_NAME], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan)
    working = working.dropna(
        subset=["feature_date", "target_date", TARGET_NAME, *columns]
    ).copy()
    if working.empty:
        raise ValueError("No complete model rows remain after feature/target filtering")
    if not (working["target_date"] > working["feature_date"]).all():
        raise ValueError("Every target_date must be later than feature_date")

    integer_target = working[TARGET_NAME].astype(int)
    if not (working[TARGET_NAME] == integer_target).all() or not set(
        integer_target.unique()
    ).issubset({0, 1}):
        raise ValueError(f"{TARGET_NAME} must contain only classes 0 and 1")
    working[TARGET_NAME] = integer_target

    working = working.sort_values(
        ["target_date", "feature_date"], kind="mergesort"
    ).reset_index(drop=True)
    if working["feature_date"].duplicated().any():
        raise ValueError("feature_date must be unique for one ticker")
    if working["target_date"].duplicated().any():
        raise ValueError("target_date must be unique for one ticker")

    metadata = ["feature_date", "target_date", "ticker"]
    if "source" in working.columns:
        metadata.append("source")
    return working[[*metadata, *columns, TARGET_NAME]].reset_index(drop=True)


def build_common_modelling_rows(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Prepare one fair-comparison row set while retaining audit aliases.

    This compatibility entrypoint now uses the full existing feature set for
    every required variant. It preserves non-feature audit columns for reports;
    the trainer must obtain estimator inputs only through
    :func:`get_feature_columns`.
    """

    data = frame.copy()
    if "feature_date" not in data.columns:
        if "date" not in data.columns:
            raise ValueError("Modelling frame requires feature_date or date")
        data["feature_date"] = data["date"]
    if TARGET_NAME not in data.columns:
        if "target_next_session_up" not in data.columns:
            raise ValueError(f"Modelling frame is missing {TARGET_NAME}")
        data[TARGET_NAME] = data["target_next_session_up"]

    before = len(data)
    prepared = prepare_model_frame(data, ticker=ticker)
    selected_dates = set(prepared["feature_date"])
    data["feature_date"] = pd.to_datetime(data["feature_date"], errors="raise")
    data["target_date"] = pd.to_datetime(data["target_date"], errors="coerce")
    data = data[data["feature_date"].isin(selected_dates)].copy()
    data = data.sort_values("feature_date", kind="mergesort").reset_index(drop=True)
    data["date"] = data["feature_date"]
    data[TARGET_NAME] = pd.to_numeric(data[TARGET_NAME], errors="raise").astype(int)
    data["target_next_session_up"] = data[TARGET_NAME]
    if "target_next_day_return" in data.columns:
        data["target_next_session_return"] = data["target_next_day_return"]
    data.attrs["common_rows_removed_for_month_feature_or_target_completeness"] = int(
        before - len(data)
    )
    return data


def _date_value(value: Any, field_name: str) -> date:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid date") from exc
    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must be a valid date")
    return timestamp.date()


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """Inclusive target-date ranges shared by every model variant."""

    train_target_start: date
    train_target_end: date
    validation_target_start: date
    validation_target_end: date
    test_target_start: date
    test_target_end: date

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(
                self,
                item.name,
                _date_value(getattr(self, item.name), item.name),
            )
        if self.train_target_start > self.train_target_end:
            raise ValueError("train target range is reversed")
        if self.validation_target_start > self.validation_target_end:
            raise ValueError("validation target range is reversed")
        if self.test_target_start > self.test_target_end:
            raise ValueError("test target range is reversed")
        if not (
            self.train_target_end < self.validation_target_start
            and self.validation_target_end < self.test_target_start
        ):
            raise ValueError(
                "train, validation, and test target ranges must be strictly ordered"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SplitBoundaries":
        return cls(
            train_target_start=value["train_target_start"],
            train_target_end=value["train_target_end"],
            validation_target_start=value["validation_target_start"],
            validation_target_end=value["validation_target_end"],
            test_target_start=value["test_target_start"],
            test_target_end=value["test_target_end"],
        )

    def to_dict(self) -> dict[str, str]:
        return {
            item.name: getattr(self, item.name).isoformat()
            for item in fields(self)
        }


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    """Chronological frames plus their reusable target-date boundaries."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: SplitBoundaries

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


def _target_dates(frame: pd.DataFrame) -> list[pd.Timestamp]:
    if frame.empty:
        raise ValueError("Cannot split an empty model frame")
    missing = sorted({"feature_date", "target_date"} - set(frame.columns))
    if missing:
        raise ValueError(f"Model frame is missing split columns: {missing}")
    targets = pd.to_datetime(frame["target_date"], errors="raise")
    features = pd.to_datetime(frame["feature_date"], errors="raise")
    if targets.isna().any() or features.isna().any():
        raise ValueError("Split dates must not be missing")
    if not (targets > features).all():
        raise ValueError("Every split row must have target_date after feature_date")
    if targets.duplicated().any():
        raise ValueError("target_date must be unique before chronological splitting")
    return sorted(pd.Timestamp(value) for value in targets.unique())


def split_with_boundaries(
    frame: pd.DataFrame,
    boundaries: SplitBoundaries,
) -> DatasetSplits:
    """Apply fixed target-date ranges, rejecting uncovered or overlapping rows."""

    working = frame.copy()
    working["feature_date"] = pd.to_datetime(working["feature_date"], errors="raise")
    working["target_date"] = pd.to_datetime(working["target_date"], errors="raise")
    _target_dates(working)

    ranges = {
        "train": (boundaries.train_target_start, boundaries.train_target_end),
        "validation": (
            boundaries.validation_target_start,
            boundaries.validation_target_end,
        ),
        "test": (boundaries.test_target_start, boundaries.test_target_end),
    }
    split_frames: dict[str, pd.DataFrame] = {}
    assigned = pd.Series(False, index=working.index)
    for name, (start, end) in ranges.items():
        mask = working["target_date"].dt.date.between(start, end, inclusive="both")
        if (assigned & mask).any():
            raise AssertionError("Target-date split ranges overlap")
        assigned |= mask
        selected = working.loc[mask].sort_values(
            ["target_date", "feature_date"], kind="mergesort"
        )
        if selected.empty:
            raise ValueError(f"{name} split must not be empty")
        split_frames[name] = selected.reset_index(drop=True)
    if not assigned.all():
        uncovered = (
            working.loc[~assigned, "target_date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
        raise ValueError(f"Split boundaries do not cover target dates: {uncovered}")

    train_end = split_frames["train"]["target_date"].max()
    validation_start = split_frames["validation"]["target_date"].min()
    validation_end = split_frames["validation"]["target_date"].max()
    test_start = split_frames["test"]["target_date"].min()
    if not (train_end < validation_start and validation_end < test_start):
        raise AssertionError("Target-date splits are not strictly chronological")
    return DatasetSplits(boundaries=boundaries, **split_frames)


def chronological_target_split(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    boundaries: SplitBoundaries | None = None,
) -> DatasetSplits:
    """Split by ordered target dates without shuffling or boundary leakage."""

    if boundaries is not None:
        return split_with_boundaries(frame, boundaries)
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")

    dates = _target_dates(frame)
    if len(dates) < 3:
        raise ValueError("Need at least three target dates for non-empty splits")
    train_count = min(max(1, int(len(dates) * train_ratio)), len(dates) - 2)
    remaining = len(dates) - train_count
    validation_count = min(
        max(1, int(len(dates) * validation_ratio)), remaining - 1
    )
    validation_end_index = train_count + validation_count - 1

    calculated = SplitBoundaries(
        train_target_start=dates[0].date(),
        train_target_end=dates[train_count - 1].date(),
        validation_target_start=dates[train_count].date(),
        validation_target_end=dates[validation_end_index].date(),
        test_target_start=dates[validation_end_index + 1].date(),
        test_target_end=dates[-1].date(),
    )
    return split_with_boundaries(frame, calculated)


__all__ = [
    "AUDIT_COLUMNS",
    "DatasetSplits",
    "FEATURE_MONTH",
    "FEATURE_SETS",
    "FEATURE_SHORT",
    "FEATURE_WEEK",
    "FORBIDDEN_FEATURE_COLUMNS",
    "FeatureSetSpec",
    "SplitBoundaries",
    "TECHNICAL_FEATURE_COLUMNS",
    "assert_no_future_or_target_columns",
    "build_common_modelling_rows",
    "chronological_target_split",
    "get_feature_columns",
    "prepare_model_frame",
    "split_with_boundaries",
    "validate_feature_columns",
    "validate_feature_sets",
]
