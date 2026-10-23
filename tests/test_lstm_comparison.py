from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from paisa.experiments.evaluation import binary_classification_metrics
from paisa.experiments.lstm_comparison import (
    align_predictions_on_common_dates,
    build_comparison_metric_row,
    build_fixed_experiment_config,
    calibration_information,
)
from paisa.trading.models import ExperimentConfig


def _prediction_frame(
    *,
    model_id: str,
    feature_dates: list[str] | None = None,
    target_dates: list[str] | None = None,
    probabilities: list[float] | None = None,
    actuals: list[int] | None = None,
) -> pd.DataFrame:
    feature_dates = feature_dates or ["2026-01-05", "2026-01-06", "2026-01-07"]
    target_dates = target_dates or ["2026-01-06", "2026-01-07", "2026-01-08"]
    probabilities = probabilities or [0.2, 0.8, 0.4]
    actuals = actuals or [0, 1, 0]
    probability_up = np.asarray(probabilities, dtype=float)
    return pd.DataFrame(
        {
            "schema_version": "1.0",
            "model_id": model_id,
            "ticker": "OGDC",
            "feature_date": feature_dates,
            "target_date": target_dates,
            "predicted_class": (probability_up > 0.5).astype(int),
            "probability_up": probability_up,
            "probability_down": 1.0 - probability_up,
            "actual_class": actuals,
        }
    )


def test_exact_common_date_alignment_retains_registered_models_and_is_sorted():
    first = _prediction_frame(model_id="registered-m005")
    second = _prediction_frame(
        model_id="registered-m006",
        probabilities=[0.3, 0.7, 0.1],
    ).iloc[::-1]

    result = align_predictions_on_common_dates({"M-006": second, "M-005": first})

    assert tuple(result.aligned_predictions) == ("M-005", "M-006")
    assert dict(result.registered_model_ids) == {
        "M-005": "registered-m005",
        "M-006": "registered-m006",
    }
    assert dict(result.original_counts) == {"M-005": 3, "M-006": 3}
    assert dict(result.common_counts) == {"M-005": 3, "M-006": 3}
    assert dict(result.dropped_target_dates) == {"M-005": (), "M-006": ()}
    assert result.common_target_dates == (
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    )
    assert result.frame_for("M-006")["target_date"].tolist() == [
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
    ]
    with pytest.raises(FrozenInstanceError):
        result.common_keys = ()  # type: ignore[misc]


def test_differing_dates_use_only_fair_intersection_and_report_drops():
    first = _prediction_frame(model_id="registered-m001")
    second = _prediction_frame(
        model_id="registered-m005",
        feature_dates=["2026-01-06", "2026-01-07", "2026-01-08"],
        target_dates=["2026-01-07", "2026-01-08", "2026-01-09"],
        probabilities=[0.7, 0.3, 0.8],
        actuals=[1, 0, 1],
    )

    result = align_predictions_on_common_dates({"M-001": first, "M-005": second})

    assert result.common_keys == (
        ("OGDC", "2026-01-06", "2026-01-07"),
        ("OGDC", "2026-01-07", "2026-01-08"),
    )
    assert dict(result.common_counts) == {"M-001": 2, "M-005": 2}
    assert dict(result.dropped_target_dates) == {
        "M-001": ("2026-01-06",),
        "M-005": ("2026-01-09",),
    }
    assert result.frame_for("M-001")["target_date"].tolist() == [
        "2026-01-07",
        "2026-01-08",
    ]


def test_common_date_alignment_rejects_actual_label_mismatch():
    first = _prediction_frame(model_id="registered-m001")
    second = _prediction_frame(
        model_id="registered-m005",
        actuals=[0, 0, 0],
    )

    with pytest.raises(ValueError, match="actual_class mismatch"):
        align_predictions_on_common_dates({"M-001": first, "M-005": second})


@pytest.mark.parametrize("invalid_probability", [np.nan, np.inf, -0.1, 1.1])
def test_common_date_alignment_rejects_nonfinite_or_out_of_range_probabilities(
    invalid_probability: float,
):
    invalid = _prediction_frame(model_id="registered-m005")
    invalid.loc[0, "probability_up"] = invalid_probability
    invalid.loc[0, "probability_down"] = 1.0 - invalid_probability

    with pytest.raises(ValueError, match="finite and in"):
        align_predictions_on_common_dates({"M-005": invalid})


def test_common_date_alignment_rejects_noncontract_schema():
    invalid = _prediction_frame(model_id="registered-m005").assign(split="test")

    with pytest.raises(ValueError, match="exactly match schema"):
        align_predictions_on_common_dates({"M-005": invalid})


def test_calibration_information_has_fixed_bins_empty_bins_and_ece():
    calibration = calibration_information(
        actual_class=np.array([0, 0, 1, 1]),
        probability_up=np.array([0.05, 0.25, 0.75, 0.95]),
        n_bins=5,
    )

    assert calibration["method"] == "equal_width"
    assert calibration["row_count"] == 4
    assert [item["count"] for item in calibration["bins"]] == [1, 1, 0, 1, 1]
    assert calibration["bins"][2]["mean_probability_up"] is None
    assert calibration["bins"][2]["observed_up_rate"] is None
    assert calibration["bins"][4]["upper_bound_inclusive"] is True
    assert calibration["expected_calibration_error"] == pytest.approx(0.15)
    assert "brier_score" not in calibration


@pytest.mark.parametrize("probability", [np.nan, np.inf, -0.01, 1.01])
def test_calibration_information_validates_probabilities(probability: float):
    with pytest.raises(ValueError, match="finite values in"):
        calibration_information([0], [probability])


def test_fixed_experiment_configs_share_rules_but_have_unique_portfolios():
    evaluation_dates = {
        "feature_start": "2026-01-05",
        "feature_end": "2026-01-30",
        "target_start": "2026-01-06",
        "target_end": "2026-02-02",
    }
    first = build_fixed_experiment_config(
        comparison_run_id="lstm-comparison-001",
        variant_id="M-005",
        registered_model_id="M-005-registered",
        evaluation_dates=evaluation_dates,
        created_at="2026-10-24T10:00:00Z",
    )
    second = build_fixed_experiment_config(
        comparison_run_id="lstm-comparison-001",
        variant_id="M-006",
        registered_model_id="M-006-registered",
        evaluation_dates=evaluation_dates,
        created_at="2026-10-24T10:00:00Z",
    )

    for config in (first, second):
        assert ExperimentConfig.from_mapping(config).to_dict() == config
        assert config["initial_cash"] == "100000.00"
        assert config["thresholds"] == {
            "buy_probability_up_gte": 0.6,
            "sell_probability_up_lte": 0.4,
        }
        assert config["maximum_allocation"] == 0.5
        assert config["transaction_cost"] == {
            "type": "proportional_bps",
            "value": 10.0,
        }
        assert config["execution_policy"]["slippage_bps"] == 0.0
        assert config["execution_policy"]["execution_session"] == "next_trading_session"
        assert config["execution_policy"]["reference_price_field"] == "open"
        assert config["execution_policy"]["valuation_price_field"] == "close"

    portfolio_ids = {
        first["portfolio_id"],
        first["benchmark_portfolio_ids"]["cash_only"],
        first["benchmark_portfolio_ids"]["buy_and_hold"],
        second["portfolio_id"],
        second["benchmark_portfolio_ids"]["cash_only"],
        second["benchmark_portfolio_ids"]["buy_and_hold"],
    }
    assert len(portfolio_ids) == 6
    assert first["experiment_id"] != second["experiment_id"]


def test_comparison_metric_row_recomputes_shared_binary_metrics():
    predictions = _prediction_frame(
        model_id="M-005-registered",
        probabilities=[0.1, 0.8, 0.7],
        actuals=[0, 1, 0],
    )
    expected = binary_classification_metrics(
        np.array([0, 1, 0]),
        np.array([0, 1, 1]),
        np.array([0.1, 0.8, 0.7]),
    )

    row = build_comparison_metric_row(
        variant_id="M-005",
        predictions=predictions,
        split_name="common_test",
        label="LSTM | price-only",
        algorithm="lstm",
    )

    assert row["variant_id"] == "M-005"
    assert row["model_id"] == "M-005-registered"
    assert row["registered_model_id"] == "M-005-registered"
    assert row["split"] == "common_test"
    assert row["label"] == "LSTM | price-only"
    assert row["algorithm"] == "lstm"
    for metric_name, expected_value in expected.items():
        assert row[metric_name] == expected_value
