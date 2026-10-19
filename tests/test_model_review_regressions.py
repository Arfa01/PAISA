from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import paisa.experiments.trainer as trainer_module
from paisa.experiments.evaluation import (
    binary_classification_metrics,
    make_reference_baselines,
    select_provisional_champion,
)
from paisa.experiments.trainer import run_training_stage


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _price_fixture(rows: int = 140) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100.0 + index * 0.08 + 2.0 * np.sin(index / 2.7)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=rows),
            "ticker": "OGDC",
            "open": close * (1.0 + 0.002 * np.cos(index)),
            "close": close,
            "volume": 100_000 + (index % 17) * 1_000,
            "source": "deterministic-fixture",
        }
    )


def _experiment_config(predictions: pd.DataFrame, model_id: str) -> dict:
    return {
        "schema_version": "1.0",
        "experiment_id": "review-cli-handoff",
        "model_id": model_id,
        "portfolio_id": "review-cli-strategy",
        "benchmark_portfolio_ids": {
            "cash_only": "review-cli-cash",
            "buy_and_hold": "review-cli-buy-hold",
        },
        "symbol": "OGDC",
        "initial_cash": "100000.00",
        "thresholds": {
            "buy_probability_up_gte": 0.60,
            "sell_probability_up_lte": 0.40,
        },
        "maximum_allocation": 0.80,
        "transaction_cost": {"type": "proportional_bps", "value": 10},
        "evaluation_dates": {
            "feature_start": predictions["feature_date"].iloc[0],
            "feature_end": predictions["feature_date"].iloc[-1],
            "target_start": predictions["target_date"].iloc[0],
            "target_end": predictions["target_date"].iloc[-1],
        },
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
        "created_at": "2026-10-19T00:00:00Z",
    }


def test_unsafe_symbol_is_rejected_before_any_filesystem_write(tmp_path: Path):
    output_root = tmp_path / "must-not-be-created"
    with pytest.raises(ValueError, match="filesystem-safe uppercase ticker"):
        run_training_stage(
            symbol="../../ESCAPED",
            prices_path=tmp_path / "missing.csv",
            output_dir=output_root,
            run_id="unsafe-symbol",
            model_ids=["M-001"],
            create_zip=False,
        )
    assert not output_root.exists()


def test_single_class_validation_is_ineligible_for_champion_selection():
    metrics = binary_classification_metrics(
        np.array([0, 0, 0]),
        np.array([0, 0, 0]),
        np.array([0.1, 0.2, 0.3]),
    )
    assert metrics["balanced_accuracy"] is None
    assert metrics["matthews_corrcoef"] is None
    assert metrics["roc_auc"] is None
    assert metrics["selection_eligible"] is False
    assert "single_class_actuals" in metrics["metric_warnings"]

    leaderboard = pd.DataFrame(
        [
            {
                "model_id": "M-001-fixture",
                "label": "one-class",
                "balanced_accuracy": None,
                "f1": 0.0,
                "accuracy": 1.0,
                "selection_eligible": False,
            }
        ]
    )
    with pytest.raises(ValueError, match="both actual classes"):
        select_provisional_champion(leaderboard)


def test_champion_ignores_ineligible_rows_even_when_accuracy_is_perfect():
    leaderboard = pd.DataFrame(
        [
            {
                "model_id": "M-001-one-class",
                "label": "ineligible",
                "balanced_accuracy": None,
                "f1": 0.0,
                "accuracy": 1.0,
                "selection_eligible": False,
            },
            {
                "model_id": "M-002-two-class",
                "label": "eligible",
                "balanced_accuracy": 0.55,
                "f1": 0.60,
                "accuracy": 0.60,
                "selection_eligible": True,
            },
        ]
    )
    champion = select_provisional_champion(leaderboard)
    assert champion["model_id"] == "M-002-two-class"


def test_reference_baselines_use_training_majority_and_feature_date_direction(
    monkeypatch,
):
    captured_predictions: list[np.ndarray] = []

    def capture_metrics(y_true, y_pred, probability_up):
        del y_true, probability_up
        captured_predictions.append(np.asarray(y_pred).copy())
        return {"accuracy": 0.0}

    monkeypatch.setattr(
        "paisa.experiments.evaluation.binary_classification_metrics",
        capture_metrics,
    )
    train = pd.DataFrame(
        {
            "target_next_day_up": [0, 0, 1],
            "daily_return": [1.0, 1.0, 1.0],
        }
    )
    evaluation = pd.DataFrame(
        {
            "target_next_day_up": [1, 0, 1],
            "daily_return": [-0.01, 0.00, 0.02],
        }
    )

    make_reference_baselines(evaluation, train)

    np.testing.assert_array_equal(captured_predictions[0], np.array([0, 0, 0]))
    np.testing.assert_array_equal(captured_predictions[1], np.array([0, 0, 1]))


def test_test_metrics_are_evaluated_only_after_validation_champion_is_frozen(
    tmp_path: Path,
    monkeypatch,
):
    prices_path = tmp_path / "prices.csv"
    _price_fixture().to_csv(prices_path, index=False)
    output_root = tmp_path / "artifacts"
    original_evaluate = trainer_module.evaluate_variant
    champion_seen_during_test: list[str] = []

    def evaluate_after_freeze(fitted, rows, split_name, symbol="OGDC"):
        evaluated = original_evaluate(fitted, rows, split_name, symbol)
        if split_name != "test":
            return evaluated

        stage_directories = list(output_root.glob(".validation-freeze-*"))
        assert len(stage_directories) == 1
        champion_path = (
            stage_directories[0] / "reports" / "provisional_champion.json"
        )
        assert champion_path.is_file()
        frozen_champion = json.loads(champion_path.read_text(encoding="utf-8"))
        champion_seen_during_test.append(frozen_champion["model_id"])

        # Force the validation champion to lose on the test leaderboard. The
        # already-frozen selection must remain unchanged.
        score = 0.0 if fitted.model_id == frozen_champion["model_id"] else 1.0
        metrics = dict(evaluated.metrics)
        for key in ("accuracy", "balanced_accuracy", "precision", "recall", "f1"):
            metrics[key] = score
        return trainer_module.VariantEvaluation(
            split_name=evaluated.split_name,
            predictions=evaluated.predictions,
            metrics=metrics,
        )

    monkeypatch.setattr(trainer_module, "evaluate_variant", evaluate_after_freeze)
    summary = run_training_stage(
        start="2025-01-01",
        prices_path=prices_path,
        output_dir=output_root,
        run_id="validation-freeze",
        create_zip=False,
        model_ids=["M-001", "M-002"],
    )

    assert champion_seen_during_test
    assert set(champion_seen_during_test) == {
        summary["provisional_champion"]["model_id"]
    }
    assert summary["test_leaderboard"][0]["model_id"] != summary[
        "provisional_champion"
    ]["model_id"]


def test_training_and_backtest_clis_accept_normal_multi_symbol_price_file(
    tmp_path: Path,
):
    ogdc = _price_fixture()
    hbl = ogdc.copy()
    hbl["ticker"] = "HBL"
    hbl["open"] = hbl["open"] * 1.5
    hbl["close"] = hbl["close"] * 1.5
    prices_path = tmp_path / "psx_prices.csv"
    pd.concat([ogdc, hbl], ignore_index=True).to_csv(prices_path, index=False)

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    artifacts_root = tmp_path / "artifacts"
    trained = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "train_ogdc_models.py"),
            "--prices",
            str(prices_path),
            "--output-dir",
            str(artifacts_root),
            "--run-id",
            "review-cli-train",
            "--model-id",
            "M-001",
            "--no-zip",
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(trained.stdout)
    prediction_path = Path(summary["prediction_paths"]["M-001"])
    predictions = pd.read_csv(prediction_path, dtype={"model_id": str})
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(_experiment_config(predictions, predictions["model_id"].iloc[0])),
        encoding="utf-8",
    )

    backtested = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_backtest.py"),
            "--predictions",
            str(prediction_path),
            "--prices",
            str(prices_path),
            "--config",
            str(config_path),
            "--runs-dir",
            str(tmp_path / "runs"),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(backtested.stdout)
    assert result["experiment_id"] == "review-cli-handoff"
    assert (tmp_path / "runs" / "review-cli-handoff" / "result.json").is_file()
