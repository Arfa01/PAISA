"""Reusable comparison boundaries for LSTM and retained PAISA predictions.

The helpers in this module deliberately operate on the stable prediction and
experiment contracts.  They do not train a model, choose a trading policy, or
mutate any retained artifact.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from paisa.experiments.evaluation import (
    binary_classification_metrics,
    validate_prediction_frame,
)
from paisa.experiments.experiment_config import DEFAULT_SYMBOL
from paisa.trading.models import ExperimentConfig


PredictionKey = tuple[str, str, str]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True, slots=True)
class FairCommonDateAlignment:
    """Validated prediction frames restricted to their exact common keys."""

    aligned_predictions: Mapping[str, pd.DataFrame]
    registered_model_ids: Mapping[str, str]
    common_keys: tuple[PredictionKey, ...]
    common_target_dates: tuple[str, ...]
    original_counts: Mapping[str, int]
    common_counts: Mapping[str, int]
    dropped_target_dates: Mapping[str, tuple[str, ...]]

    def frame_for(self, variant_id: str) -> pd.DataFrame:
        """Return a defensive copy of one aligned prediction frame."""

        try:
            frame = self.aligned_predictions[variant_id]
        except KeyError as exc:
            raise KeyError(f"Unknown comparison variant: {variant_id}") from exc
        return frame.copy(deep=True)


def _validated_identifier(value: object, *, field_name: str) -> str:
    identifier = str(value).strip()
    if not identifier or identifier in {".", ".."} or not _SAFE_ID.fullmatch(identifier):
        raise ValueError(
            f"{field_name} must contain only letters, digits, '.', '_', and '-' "
            "and must start with a letter or digit"
        )
    return identifier


def _registered_model_id(frame: pd.DataFrame, *, variant_id: str) -> str:
    model_ids = frame["model_id"].drop_duplicates().tolist()
    if len(model_ids) != 1:
        raise ValueError(
            f"{variant_id} predictions must contain exactly one registered model_id"
        )
    return str(model_ids[0])


def _prediction_keys(frame: pd.DataFrame) -> list[PredictionKey]:
    return [
        (str(ticker), str(feature_date), str(target_date))
        for ticker, feature_date, target_date in frame[
            ["ticker", "feature_date", "target_date"]
        ].itertuples(index=False, name=None)
    ]


def align_predictions_on_common_dates(
    predictions_by_variant: Mapping[str, pd.DataFrame],
) -> FairCommonDateAlignment:
    """Validate and fairly align variants on exact prediction observation keys.

    The fair set is the intersection of ``(ticker, feature_date, target_date)``
    keys.  Labels must agree for every retained key so all variants are scored
    against the same observations.  Each input frame remains associated with
    its registered (run-specific) ``model_id``.
    """

    if not isinstance(predictions_by_variant, Mapping) or not predictions_by_variant:
        raise ValueError("predictions_by_variant must contain at least one variant")

    normalized_by_variant: dict[str, pd.DataFrame] = {}
    model_ids: dict[str, str] = {}
    original_counts: dict[str, int] = {}
    key_sets: dict[str, set[PredictionKey]] = {}

    for raw_variant_id, raw_frame in predictions_by_variant.items():
        variant_id = _validated_identifier(raw_variant_id, field_name="variant_id")
        if variant_id in normalized_by_variant:
            raise ValueError(f"Duplicate normalized variant_id: {variant_id}")
        if not isinstance(raw_frame, pd.DataFrame):
            raise TypeError(f"{variant_id} predictions must be a pandas DataFrame")

        normalized = validate_prediction_frame(raw_frame)
        normalized_by_variant[variant_id] = normalized
        model_ids[variant_id] = _registered_model_id(
            normalized, variant_id=variant_id
        )
        original_counts[variant_id] = int(len(normalized))
        key_sets[variant_id] = set(_prediction_keys(normalized))

    common_set = set.intersection(*(keys for keys in key_sets.values()))
    if not common_set:
        raise ValueError("Prediction variants have no common observation keys")
    common_keys = tuple(sorted(common_set))

    ordered_variants = sorted(normalized_by_variant)
    expected_actuals: dict[PredictionKey, int | None] = {}
    first_variant = ordered_variants[0]
    first_frame = normalized_by_variant[first_variant]
    for key, actual in zip(
        _prediction_keys(first_frame), first_frame["actual_class"], strict=True
    ):
        if key in common_set:
            expected_actuals[key] = None if pd.isna(actual) else int(actual)

    for variant_id in ordered_variants[1:]:
        frame = normalized_by_variant[variant_id]
        for key, actual in zip(
            _prediction_keys(frame), frame["actual_class"], strict=True
        ):
            if key not in common_set:
                continue
            candidate = None if pd.isna(actual) else int(actual)
            if candidate != expected_actuals[key]:
                raise ValueError(
                    "actual_class mismatch for common key "
                    f"{key!r}: {first_variant}={expected_actuals[key]!r}, "
                    f"{variant_id}={candidate!r}"
                )

    aligned: dict[str, pd.DataFrame] = {}
    common_counts: dict[str, int] = {}
    dropped_target_dates: dict[str, tuple[str, ...]] = {}
    for variant_id in ordered_variants:
        frame = normalized_by_variant[variant_id]
        keys = _prediction_keys(frame)
        retained = frame.loc[[key in common_set for key in keys]].copy()
        retained = retained.sort_values(
            ["ticker", "feature_date", "target_date"], kind="mergesort"
        ).reset_index(drop=True)
        aligned[variant_id] = retained
        common_counts[variant_id] = int(len(retained))
        dropped_target_dates[variant_id] = tuple(
            sorted({key[2] for key in key_sets[variant_id] - common_set})
        )

    return FairCommonDateAlignment(
        aligned_predictions=MappingProxyType(aligned),
        registered_model_ids=MappingProxyType(
            {variant_id: model_ids[variant_id] for variant_id in ordered_variants}
        ),
        common_keys=common_keys,
        common_target_dates=tuple(sorted({key[2] for key in common_keys})),
        original_counts=MappingProxyType(
            {variant_id: original_counts[variant_id] for variant_id in ordered_variants}
        ),
        common_counts=MappingProxyType(common_counts),
        dropped_target_dates=MappingProxyType(dropped_target_dates),
    )


def _validated_binary_values(
    values: Sequence[int] | pd.Series | np.ndarray, *, field_name: str
) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if result.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if pd.isna(result).any():
        raise ValueError(f"{field_name} must contain only 0 and 1")
    try:
        parsed = result.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only 0 and 1") from exc
    if not np.isin(parsed, [0, 1]).all() or not np.equal(result, parsed).all():
        raise ValueError(f"{field_name} must contain only 0 and 1")
    return parsed


def calibration_information(
    actual_class: Sequence[int] | pd.Series | np.ndarray,
    probability_up: Sequence[float] | pd.Series | np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Return fixed equal-width reliability bins and ECE.

    Empty bins are retained with ``None`` rates.  Brier score is intentionally
    not duplicated here because it is already emitted by
    :func:`binary_classification_metrics`.
    """

    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    actual = _validated_binary_values(actual_class, field_name="actual_class")
    try:
        probabilities = np.asarray(probability_up, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability_up must contain finite numeric values") from exc
    if probabilities.ndim != 1 or len(probabilities) != len(actual):
        raise ValueError(
            "probability_up must be one-dimensional and match actual_class"
        )
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("probability_up must contain only finite values in [0, 1]")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.searchsorted(edges, probabilities, side="right") - 1
    assignments = np.clip(assignments, 0, n_bins - 1)
    bins: list[dict[str, Any]] = []
    expected_calibration_error = 0.0
    row_count = int(len(actual))

    for bin_index in range(n_bins):
        mask = assignments == bin_index
        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_up_rate = float(actual[mask].mean())
            absolute_gap = abs(mean_probability - observed_up_rate)
            expected_calibration_error += (count / row_count) * absolute_gap
        else:
            mean_probability = None
            observed_up_rate = None
            absolute_gap = None
        bins.append(
            {
                "bin_index": bin_index,
                "lower_bound": float(edges[bin_index]),
                "upper_bound": float(edges[bin_index + 1]),
                "upper_bound_inclusive": bin_index == n_bins - 1,
                "count": count,
                "mean_probability_up": mean_probability,
                "observed_up_rate": observed_up_rate,
                "absolute_gap": absolute_gap,
            }
        )

    return {
        "method": "equal_width",
        "n_bins": n_bins,
        "row_count": row_count,
        "expected_calibration_error": float(expected_calibration_error),
        "bins": bins,
    }


def build_fixed_experiment_config(
    *,
    comparison_run_id: str,
    variant_id: str,
    registered_model_id: str,
    evaluation_dates: Mapping[str, str],
    created_at: str,
    symbol: str = DEFAULT_SYMBOL,
) -> dict[str, Any]:
    """Build one validated config with the frozen PAISA comparison rules."""

    run_id = _validated_identifier(comparison_run_id, field_name="comparison_run_id")
    variant = _validated_identifier(variant_id, field_name="variant_id")
    experiment_id = f"{run_id}-{variant.lower()}"
    raw_config: dict[str, Any] = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "model_id": registered_model_id,
        "portfolio_id": f"{experiment_id}-strategy",
        "benchmark_portfolio_ids": {
            "cash_only": f"{experiment_id}-cash-only",
            "buy_and_hold": f"{experiment_id}-buy-and-hold",
        },
        "symbol": symbol,
        "initial_cash": "100000.00",
        "thresholds": {
            "buy_probability_up_gte": 0.60,
            "sell_probability_up_lte": 0.40,
        },
        "maximum_allocation": 0.50,
        "transaction_cost": {"type": "proportional_bps", "value": 10},
        "evaluation_dates": dict(evaluation_dates),
        "execution_policy": {
            "signal_timing": "after_close",
            "execution_session": "next_trading_session",
            "reference_price_field": "open",
            "slippage_bps": 0,
            "missing_execution_price": "fail",
            "allocation_mode": "state_transition_to_maximum",
            "whole_share_rounding": "floor",
            "valuation_price_field": "close",
            "missing_valuation_price": "fail",
            "ending_position": "mark_to_market",
        },
        "metric_assumptions": {
            "annualization_factor": 252,
            "risk_free_rate_annual": 0.0,
        },
        "created_at": created_at,
    }
    return ExperimentConfig.from_mapping(raw_config).to_dict()


def build_comparison_metric_row(
    *,
    variant_id: str,
    predictions: pd.DataFrame,
    split_name: str = "common_test",
    label: str | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    """Recompute a comparison row from a validated prediction frame."""

    variant = _validated_identifier(variant_id, field_name="variant_id")
    normalized = validate_prediction_frame(predictions)
    model_id = _registered_model_id(normalized, variant_id=variant)
    if normalized["actual_class"].isna().any():
        raise ValueError("comparison metrics require actual_class for every prediction")

    metrics = binary_classification_metrics(
        normalized["actual_class"].astype(int).to_numpy(),
        normalized["predicted_class"].astype(int).to_numpy(),
        normalized["probability_up"].astype(float).to_numpy(),
    )
    return {
        "variant_id": variant,
        "model_id": model_id,
        "registered_model_id": model_id,
        "split": str(split_name),
        "label": label,
        "algorithm": algorithm,
        **metrics,
    }


__all__ = [
    "FairCommonDateAlignment",
    "align_predictions_on_common_dates",
    "build_comparison_metric_row",
    "build_fixed_experiment_config",
    "calibration_information",
]
