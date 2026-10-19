from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paisa.experiments.evaluation import PREDICTION_COLUMNS
from paisa.experiments.experiment_config import TARGET_NAME, get_variant
from paisa.experiments.runner import load_predictions, run_experiment
from paisa.experiments.trainer import (
    evaluate_variant,
    fit_variant,
    prepare_training_data,
    run_training_stage,
)
from paisa.experiments.training_features import TECHNICAL_FEATURE_COLUMNS


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


@pytest.fixture(scope="module")
def prepared():
    return prepare_training_data(_price_fixture(), symbol="OGDC")


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("model-training-run")
    prices_path = root / "prices.csv"
    _price_fixture().to_csv(prices_path, index=False)
    summary = run_training_stage(
        start="2025-01-01",
        prices_path=prices_path,
        output_dir=root / "artifacts",
        run_id="pytest-all-models",
        create_zip=False,
    )
    return root, prices_path, summary


@pytest.mark.parametrize("variant_id", ["M-001", "M-004"])
def test_training_is_deterministic_for_fixed_seed(prepared, variant_id):
    variant = get_variant(variant_id)
    first = fit_variant(variant, prepared.splits.train, random_state=42)
    second = fit_variant(variant, prepared.splits.train, random_state=42)

    first_predictions = evaluate_variant(
        first, prepared.splits.validation, "validation"
    ).predictions
    second_predictions = evaluate_variant(
        second, prepared.splits.validation, "validation"
    ).predictions
    np.testing.assert_allclose(
        first_predictions["probability_up"],
        second_predictions["probability_up"],
        rtol=0,
        atol=1e-15,
    )
    assert first_predictions["predicted_class"].equals(
        second_predictions["predicted_class"]
    )


def test_all_variants_share_dates_features_and_have_unique_registered_ids(completed_run):
    _, _, summary = completed_run
    registered_ids = summary["model_ids"]
    assert list(registered_ids) == ["M-001", "M-002", "M-003", "M-004"]
    assert len(set(registered_ids.values())) == 4

    expected_dates = None
    expected_features = list(TECHNICAL_FEATURE_COLUMNS)
    for variant_id, registered_id in registered_ids.items():
        model_dir = Path(summary["artifact_directory"]) / "models" / registered_id
        manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["variant_id"] == variant_id
        assert manifest["feature_set"]["columns"] == expected_features
        assert "NaN" not in (model_dir / "manifest.json").read_text(encoding="utf-8")
        audit = pd.read_csv(model_dir / "predictions_all_splits.csv")
        split_dates = {
            split: audit.loc[audit["split"] == split, "feature_date"].tolist()
            for split in ("train", "validation", "test")
        }
        expected_dates = split_dates if expected_dates is None else expected_dates
        assert split_dates == expected_dates


def test_prediction_csv_is_exact_backtesting_contract(completed_run):
    _, _, summary = completed_run
    for path_string in summary["prediction_paths"].values():
        path = Path(path_string)
        frame = pd.read_csv(path)
        assert frame.columns.tolist() == PREDICTION_COLUMNS
        records = load_predictions(path)
        assert len(records) == len(frame)
        assert all(record.ticker == "OGDC" for record in records)


def test_generated_prediction_csv_runs_directly_in_backtester(completed_run, tmp_path):
    _, prices_path, summary = completed_run
    prediction_path = Path(summary["prediction_paths"]["M-001"])
    predictions = pd.read_csv(prediction_path)
    model_id = str(predictions["model_id"].iloc[0])
    config = {
        "schema_version": "1.0",
        "experiment_id": "training-handoff-test",
        "model_id": model_id,
        "portfolio_id": "training-handoff-strategy",
        "benchmark_portfolio_ids": {
            "cash_only": "training-handoff-cash",
            "buy_and_hold": "training-handoff-buy-hold",
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
    config_path = tmp_path / "experiment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = run_experiment(
        predictions_path=prediction_path,
        prices_path=prices_path,
        config_path=config_path,
        runs_root=tmp_path / "runs",
    )
    assert result.config.model_id == model_id
    assert result.artifact_path.is_dir()


def test_training_cli_smoke_with_existing_prices(tmp_path):
    prices_path = tmp_path / "prices.csv"
    _price_fixture().to_csv(prices_path, index=False)
    output_root = tmp_path / "cli-artifacts"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "train_ogdc_models.py"),
            "--prices",
            str(prices_path),
            "--start",
            "2025-01-01",
            "--output-dir",
            str(output_root),
            "--run-id",
            "cli-smoke",
            "--random-state",
            "7",
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
    output = json.loads(completed.stdout)
    assert output["run_id"] == "cli-smoke"
    assert set(output["model_ids"]) == {"M-001"}
    assert Path(output["prediction_paths"]["M-001"]).is_file()
    registered_id = output["model_ids"]["M-001"]
    config_path = output_root / "cli-smoke" / "models" / registered_id / "config.json"
    configuration = json.loads(config_path.read_text(encoding="utf-8"))
    assert configuration["random_seed"] == 7
    assert configuration["model_configuration"]["random_state"] == 7
    assert "Traceback" not in completed.stderr


def test_prepared_target_is_current_close_to_next_close(prepared):
    rows = prepared.model_frame
    assert TARGET_NAME in rows
    assert not any(
        column.startswith("target_") or column.startswith("next_")
        for column in prepared.feature_columns
    )
