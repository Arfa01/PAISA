from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from paisa.experiments.training_features import (
    TECHNICAL_FEATURE_COLUMNS,
    SplitBoundaries,
    chronological_target_split,
    prepare_model_frame,
    split_with_boundaries,
    validate_feature_columns,
)
from paisa.features import FEATURE_COLUMNS


def _complete_model_frame(row_count: int = 12) -> pd.DataFrame:
    feature_dates = pd.bdate_range("2026-01-01", periods=row_count)
    target_dates = feature_dates + pd.offsets.BDay(1)
    data: dict[str, object] = {
        "feature_date": feature_dates,
        "target_date": target_dates,
        "ticker": ["OGDC"] * row_count,
        "source": ["unit"] * row_count,
        "target_next_day_up": [index % 2 for index in range(row_count)],
        "next_close": [999.0] * row_count,
    }
    for offset, column in enumerate(FEATURE_COLUMNS, start=1):
        data[column] = [float(offset + index) for index in range(row_count)]
    return pd.DataFrame(data)


def test_feature_order_reuses_v0_and_leakage_columns_are_rejected():
    assert TECHNICAL_FEATURE_COLUMNS == tuple(FEATURE_COLUMNS)
    assert validate_feature_columns(TECHNICAL_FEATURE_COLUMNS) == tuple(
        FEATURE_COLUMNS
    )
    for forbidden in ("target_next_day_up", "target_date", "next_close"):
        with pytest.raises(ValueError, match="target, future, or metadata"):
            validate_feature_columns([*FEATURE_COLUMNS, forbidden])


def test_prepare_model_frame_is_ordered_and_excludes_audit_future_values():
    frame = _complete_model_frame().sample(frac=1.0, random_state=7)
    prepared = prepare_model_frame(frame)

    assert prepared["target_date"].is_monotonic_increasing
    assert prepared.columns.tolist() == [
        "feature_date",
        "target_date",
        "ticker",
        "source",
        *FEATURE_COLUMNS,
        "target_next_day_up",
    ]
    assert "next_close" not in prepared.columns
    assert prepared["feature_date"].is_unique
    assert prepared["target_date"].is_unique


def test_chronological_split_has_no_overlap_and_assigns_by_target_date():
    prepared = prepare_model_frame(_complete_model_frame())
    splits = chronological_target_split(prepared)

    train_dates = set(splits.train["target_date"])
    validation_dates = set(splits.validation["target_date"])
    test_dates = set(splits.test["target_date"])
    assert not train_dates & validation_dates
    assert not train_dates & test_dates
    assert not validation_dates & test_dates
    assert train_dates | validation_dates | test_dates == set(
        prepared["target_date"]
    )
    assert max(train_dates) < min(validation_dates) < min(test_dates)
    assert all(
        frame["target_date"].is_monotonic_increasing
        for frame in splits.as_dict().values()
    )
    assert splits.boundaries.train_target_end < (
        splits.boundaries.validation_target_start
    )
    assert splits.boundaries.validation_target_end < (
        splits.boundaries.test_target_start
    )


def test_fixed_boundaries_reproduce_identical_dates_for_every_variant():
    prepared = prepare_model_frame(_complete_model_frame())
    first = chronological_target_split(prepared)
    changed_features = prepared.sample(frac=1.0, random_state=99).copy()
    changed_features.loc[:, FEATURE_COLUMNS] *= 2

    repeated = chronological_target_split(
        changed_features,
        boundaries=first.boundaries,
    )
    direct = split_with_boundaries(changed_features, first.boundaries)
    for split_name in ("train", "validation", "test"):
        expected_dates = first.as_dict()[split_name]["target_date"].tolist()
        assert repeated.as_dict()[split_name]["target_date"].tolist() == expected_dates
        assert direct.as_dict()[split_name]["target_date"].tolist() == expected_dates


def test_split_rejects_boundaries_that_leave_a_target_date_unassigned():
    prepared = prepare_model_frame(_complete_model_frame())
    original = chronological_target_split(prepared).boundaries
    incomplete = SplitBoundaries(
        train_target_start=original.train_target_start,
        train_target_end=original.train_target_end,
        validation_target_start=original.validation_target_start,
        validation_target_end=original.validation_target_end,
        test_target_start=original.test_target_start,
        test_target_end=original.test_target_end - timedelta(days=1),
    )
    with pytest.raises(ValueError, match="do not cover target dates"):
        split_with_boundaries(prepared, incomplete)
