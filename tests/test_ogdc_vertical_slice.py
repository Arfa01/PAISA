from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from paisa.experiments.evaluation import PREDICTION_COLUMNS
from paisa.experiments.runner import run_experiment
from paisa.vertical_slice import (
    PriceAcquisition,
    build_combined_leaderboard,
    run_ogdc_vertical_slice,
    validate_real_price_file,
)


REQUIRED_OUTPUTS = {
    "manifest.json",
    "model_training_run_reference.json",
    "experiment_configs",
    "model_metrics.csv",
    "portfolio_metrics.csv",
    "combined_leaderboard.csv",
    "daily_equity_curves.csv",
    "model_metric_comparison.png",
    "portfolio_equity_comparison.png",
    "drawdown_comparison.png",
    "final_summary.json",
}
EXPECTED_VARIANTS = {"M-001", "M-002", "M-003", "M-004"}


def _price_fixture(
    rows: int = 160,
    *,
    source: str = "deterministic_fixture",
) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100.0 + index * 0.08 + 2.0 * np.sin(index / 2.7)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=rows),
            "ticker": "OGDC",
            "open": close * (1.0 + 0.002 * np.cos(index)),
            "high": np.nan,
            "low": np.nan,
            "close": close,
            "volume": 100_000 + (index % 17) * 1_000,
            "source": source,
        }
    )


def _fixture_acquisition(*, symbol: str, start: str, work_dir: Path) -> PriceAcquisition:
    assert symbol == "OGDC"
    assert start <= "2025-01-01"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    prices = _price_fixture()
    prices_path = work_dir / "OGDC_fixture_prices.csv"
    prices.to_csv(prices_path, index=False)
    return PriceAcquisition(
        prices_path=prices_path,
        source="deterministic_fixture",
        mode="fixture_test",
        date_start="2025-01-01",
        date_end=pd.Timestamp(prices["date"].max()).strftime("%Y-%m-%d"),
        row_count=len(prices),
        is_real_data=False,
        live_error=None,
    )


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class CompletedVerticalSlice:
    artifact_path: Path
    output_dir: Path
    model_output_dir: Path
    summary: dict[str, Any]
    backtest_calls: tuple[dict[str, Any], ...]


@pytest.fixture(scope="module")
def completed_vertical_slice(tmp_path_factory: pytest.TempPathFactory) -> CompletedVerticalSlice:
    root = tmp_path_factory.mktemp("ogdc-vertical-slice")
    output_dir = root / "vertical"
    model_output_dir = root / "model-training"
    calls: list[dict[str, Any]] = []

    def capture_backtest(**kwargs: Any):
        prediction_path = Path(kwargs["predictions_path"])
        prices_path = Path(kwargs["prices_path"])
        before = _sha256(prediction_path)
        completed = run_experiment(**kwargs)
        calls.append(
            {
                **kwargs,
                "prediction_sha256_before": before,
                "prediction_sha256_after": _sha256(prediction_path),
                "prices_sha256": _sha256(prices_path),
                "completed": completed,
            }
        )
        return completed

    summary = run_ogdc_vertical_slice(
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        cached_prices_path=root / "missing-cached-prices.csv",
        vertical_slice_run_id="pytest-smoke",
        acquire_fn=_fixture_acquisition,
        backtest_fn=capture_backtest,
    )
    return CompletedVerticalSlice(
        artifact_path=output_dir / "pytest-smoke",
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        summary=summary,
        backtest_calls=tuple(calls),
    )


def _experiment_configs(run: CompletedVerticalSlice) -> list[dict[str, Any]]:
    paths = sorted((run.artifact_path / "experiment_configs").glob("*.json"))
    assert {path.stem for path in paths} == EXPECTED_VARIANTS
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_all_four_experiments_use_identical_dates_market_prices_and_rules(
    completed_vertical_slice: CompletedVerticalSlice,
):
    configs = _experiment_configs(completed_vertical_slice)
    for field in (
        "initial_cash",
        "thresholds",
        "maximum_allocation",
        "transaction_cost",
        "evaluation_dates",
        "execution_policy",
        "metric_assumptions",
    ):
        assert all(config[field] == configs[0][field] for config in configs[1:])

    assert configs[0]["initial_cash"] == "100000.00"
    assert configs[0]["thresholds"] == {
        "buy_probability_up_gte": 0.6,
        "sell_probability_up_lte": 0.4,
    }
    assert configs[0]["maximum_allocation"] == 0.5
    assert configs[0]["transaction_cost"] == {
        "type": "proportional_bps",
        "value": 10.0,
    }
    assert configs[0]["execution_policy"]["slippage_bps"] == 0.0

    calls = completed_vertical_slice.backtest_calls
    assert len(calls) == 4
    assert len({Path(call["prices_path"]).resolve() for call in calls}) == 1
    assert len({call["prices_sha256"] for call in calls}) == 1
    common_market = tuple(
        (bar.date, bar.open, bar.close)
        for bar in calls[0]["completed"].strategy.market_bars
    )
    assert common_market
    for call in calls[1:]:
        assert tuple(
            (bar.date, bar.open, bar.close)
            for bar in call["completed"].strategy.market_bars
        ) == common_market

    date_pairs = []
    for call in calls:
        predictions = pd.read_csv(call["predictions_path"])
        date_pairs.append(
            list(zip(predictions["feature_date"], predictions["target_date"], strict=True))
        )
    assert all(pairs == date_pairs[0] for pairs in date_pairs[1:])


def test_each_model_experiment_has_a_unique_portfolio_id(
    completed_vertical_slice: CompletedVerticalSlice,
):
    configs = _experiment_configs(completed_vertical_slice)
    strategy_ids = [config["portfolio_id"] for config in configs]
    assert len(strategy_ids) == len(set(strategy_ids)) == 4

    all_ids = []
    for config in configs:
        all_ids.extend(
            [
                config["portfolio_id"],
                config["benchmark_portfolio_ids"]["cash_only"],
                config["benchmark_portfolio_ids"]["buy_and_hold"],
            ]
        )
    assert len(all_ids) == len(set(all_ids)) == 12


def test_orchestrator_refuses_to_overwrite_an_old_run(
    completed_vertical_slice: CompletedVerticalSlice,
):
    before = _tree_hashes(completed_vertical_slice.artifact_path)

    def acquisition_must_not_run(**_: Any) -> PriceAcquisition:
        raise AssertionError("collision must be rejected before acquisition")

    with pytest.raises(FileExistsError, match="overwrite|already exists|exists"):
        run_ogdc_vertical_slice(
            output_dir=completed_vertical_slice.output_dir,
            model_output_dir=completed_vertical_slice.model_output_dir,
            cached_prices_path=completed_vertical_slice.output_dir / "missing.csv",
            vertical_slice_run_id="pytest-smoke",
            acquire_fn=acquisition_must_not_run,
        )

    assert _tree_hashes(completed_vertical_slice.artifact_path) == before


def test_failed_model_prevents_a_false_successful_final_report(tmp_path: Path):
    def failed_training(**_: Any) -> dict[str, Any]:
        raise RuntimeError("M-003 training sentinel failure")

    with pytest.raises(RuntimeError, match="M-003 training sentinel failure"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="failed-model-run",
            acquire_fn=_fixture_acquisition,
            train_fn=failed_training,
        )

    final_run = tmp_path / "vertical" / "failed-model-run"
    assert not final_run.exists()
    assert not list((tmp_path / "vertical").glob("**/final_summary.json"))


def test_live_data_failure_never_uses_synthetic_cached_data(tmp_path: Path):
    cached = tmp_path / "psx_prices.csv"
    _price_fixture(source="sample").to_csv(cached, index=False)
    train_called = False

    def live_failure(**_: Any) -> PriceAcquisition:
        raise RuntimeError("live provider sentinel HTTP 403")

    def training_must_not_run(**_: Any) -> dict[str, Any]:
        nonlocal train_called
        train_called = True
        raise AssertionError("synthetic prices must not reach training")

    with pytest.raises(Exception) as failure:
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=cached,
            vertical_slice_run_id="reject-sample-cache",
            acquire_fn=live_failure,
            train_fn=training_must_not_run,
        )

    message = str(failure.value).lower()
    assert "live provider sentinel http 403" in message
    assert "sample" in message or "synthetic" in message or "real" in message
    assert train_called is False
    assert not (tmp_path / "vertical" / "reject-sample-cache").exists()


def test_cached_real_price_validation_confirms_ogdc_and_date_range(tmp_path: Path):
    cached = tmp_path / "psx_prices.csv"
    prices = _price_fixture(source="psx_dps")
    prices.to_csv(cached, index=False)

    validated = validate_real_price_file(cached, symbol="OGDC", start="2025-02-03")

    assert validated.source == "psx_dps"
    assert validated.date_start == "2025-02-03"
    assert validated.date_end == pd.Timestamp(prices["date"].max()).strftime("%Y-%m-%d")
    assert validated.row_count == int((prices["date"] >= "2025-02-03").sum())


def test_invalid_live_result_does_not_open_cached_fallback(tmp_path: Path):
    cached = tmp_path / "cached.csv"
    _price_fixture(source="psx_dps").to_csv(cached, index=False)
    invalid_live = tmp_path / "invalid-live.csv"
    invalid = _price_fixture(source="psx_dps")
    invalid.loc[0, "open"] = -1
    invalid.to_csv(invalid_live, index=False)
    train_called = False

    def invalid_acquisition(**_: Any) -> PriceAcquisition:
        return PriceAcquisition(
            prices_path=invalid_live,
            source="psx_dps",
            mode="live_psx_dps",
            is_real_data=True,
        )

    def training_must_not_run(**_: Any) -> dict[str, Any]:
        nonlocal train_called
        train_called = True
        raise AssertionError("invalid live data must fail before cached fallback")

    with pytest.raises(ValueError, match="strictly positive"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=cached,
            vertical_slice_run_id="invalid-live-result",
            acquire_fn=invalid_acquisition,
            train_fn=training_must_not_run,
        )

    assert train_called is False
    assert not (tmp_path / "vertical" / "invalid-live-result").exists()


def test_combined_leaderboard_calculations_and_separate_rankings_are_correct():
    model_metrics = pd.DataFrame(
        [
            {
                "model_id": "M-001",
                "registered_model_id": "registered-001",
                "algorithm": "random_forest",
                "validation_accuracy": 0.70,
                "validation_balanced_accuracy": 0.75,
                "validation_f1": 0.72,
                "test_accuracy": 0.51,
                "test_balanced_accuracy": 0.52,
                "test_f1": 0.50,
            },
            {
                "model_id": "M-002",
                "registered_model_id": "registered-002",
                "algorithm": "random_forest",
                "validation_accuracy": 0.60,
                "validation_balanced_accuracy": 0.65,
                "validation_f1": 0.64,
                "test_accuracy": 0.91,
                "test_balanced_accuracy": 0.90,
                "test_f1": 0.89,
            },
            {
                "model_id": "M-003",
                "registered_model_id": "registered-003",
                "algorithm": "random_forest",
                "validation_accuracy": 0.58,
                "validation_balanced_accuracy": 0.57,
                "validation_f1": 0.56,
                "test_accuracy": 0.59,
                "test_balanced_accuracy": 0.58,
                "test_f1": 0.57,
            },
            {
                "model_id": "M-004",
                "registered_model_id": "registered-004",
                "algorithm": "xgboost",
                "validation_accuracy": 0.55,
                "validation_balanced_accuracy": 0.54,
                "validation_f1": 0.53,
                "test_accuracy": 0.56,
                "test_balanced_accuracy": 0.55,
                "test_f1": 0.54,
            },
        ]
    )
    portfolio_metrics = pd.DataFrame(
        [
            {
                "model_id": "M-001",
                "registered_model_id": "registered-001",
                "portfolio_id": "portfolio-001",
                "portfolio_type": "model",
                "ending_portfolio_value": 110_000.0,
                "total_return": 0.10,
                "maximum_drawdown": 0.03,
                "volatility": 0.12,
                "sharpe_ratio": 1.1,
                "number_of_trades": 4,
                "fees_paid": 50.0,
                "result_versus_buy_and_hold": 0.02,
            },
            {
                "model_id": "M-002",
                "registered_model_id": "registered-002",
                "portfolio_id": "portfolio-002",
                "portfolio_type": "model",
                "ending_portfolio_value": 105_000.0,
                "total_return": 0.05,
                "maximum_drawdown": 0.01,
                "volatility": 0.08,
                "sharpe_ratio": 0.8,
                "number_of_trades": 2,
                "fees_paid": 25.0,
                "result_versus_buy_and_hold": -0.03,
            },
            {
                "model_id": "M-003",
                "registered_model_id": "registered-003",
                "portfolio_id": "portfolio-003",
                "portfolio_type": "model",
                "ending_portfolio_value": 103_000.0,
                "total_return": 0.03,
                "maximum_drawdown": 0.02,
                "volatility": 0.09,
                "sharpe_ratio": 0.6,
                "number_of_trades": 3,
                "fees_paid": 30.0,
                "result_versus_buy_and_hold": -0.05,
            },
            {
                "model_id": "M-004",
                "registered_model_id": "registered-004",
                "portfolio_id": "portfolio-004",
                "portfolio_type": "model",
                "ending_portfolio_value": 101_000.0,
                "total_return": 0.01,
                "maximum_drawdown": 0.04,
                "volatility": 0.11,
                "sharpe_ratio": 0.2,
                "number_of_trades": 5,
                "fees_paid": 60.0,
                "result_versus_buy_and_hold": -0.07,
            },
        ]
    )

    leaderboard = build_combined_leaderboard(model_metrics, portfolio_metrics)

    assert leaderboard["model_id"].tolist() == [
        "M-001",
        "M-002",
        "M-003",
        "M-004",
    ]
    first = leaderboard.set_index("model_id").loc["M-001"]
    second = leaderboard.set_index("model_id").loc["M-002"]
    assert first["validation_balanced_accuracy"] == pytest.approx(0.75)
    assert first["validation_f1"] == pytest.approx(0.72)
    assert first["ending_portfolio_value"] == pytest.approx(110_000.0)
    assert first["result_versus_buy_and_hold"] == pytest.approx(0.02)
    assert second["test_accuracy"] == pytest.approx(0.91)
    # Higher test accuracy does not reorder or replace the validation choice.
    assert leaderboard.iloc[0]["model_id"] == "M-001"


def test_model_predictions_reach_backtester_without_schema_conversion(
    completed_vertical_slice: CompletedVerticalSlice,
):
    calls = completed_vertical_slice.backtest_calls
    assert len(calls) == 4
    for call in calls:
        prediction_path = Path(call["predictions_path"])
        assert prediction_path.name == "predictions.csv"
        assert prediction_path.parent.parent.name == "models"
        assert call["prediction_sha256_before"] == call["prediction_sha256_after"]
        assert pd.read_csv(prediction_path).columns.tolist() == PREDICTION_COLUMNS

    reference = json.loads(
        (completed_vertical_slice.artifact_path / "model_training_run_reference.json").read_text(
            encoding="utf-8"
        )
    )
    referenced_paths = {
        Path(path).resolve() for path in reference["prediction_paths"].values()
    }
    assert {Path(call["predictions_path"]).resolve() for call in calls} == referenced_paths


def test_stale_prediction_metrics_prevent_backtesting(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    reference = json.loads(
        (completed_vertical_slice.artifact_path / "model_training_run_reference.json").read_text(
            encoding="utf-8"
        )
    )
    source_training = Path(reference["artifact_directory"])
    copied_training = tmp_path / "copied-training"
    shutil.copytree(source_training, copied_training)
    model_ids = dict(reference["model_ids"])
    prediction_paths = {
        variant_id: str(
            copied_training / "models" / registered_id / "predictions.csv"
        )
        for variant_id, registered_id in model_ids.items()
    }
    tampered_path = Path(prediction_paths["M-001"])
    tampered = pd.read_csv(tampered_path)
    tampered.loc[0, "actual_class"] = 1 - int(tampered.loc[0, "actual_class"])
    tampered.to_csv(tampered_path, index=False)
    run_manifest = json.loads(
        (copied_training / "run_manifest.json").read_text(encoding="utf-8")
    )

    def stale_training(**_: Any) -> dict[str, Any]:
        return {
            "run_id": run_manifest["run_id"],
            "invocation_id": run_manifest["invocation_id"],
            "artifact_directory": str(copied_training),
            "model_ids": model_ids,
            "prediction_paths": prediction_paths,
        }

    def backtest_must_not_run(**_: Any):
        raise AssertionError("stale prediction evidence must fail before backtesting")

    with pytest.raises(ValueError, match="class distribution|does not match predictions"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing-cache.csv",
            vertical_slice_run_id="stale-predictions",
            acquire_fn=_fixture_acquisition,
            train_fn=stale_training,
            backtest_fn=backtest_must_not_run,
        )


def test_complete_fixture_based_end_to_end_smoke_run(
    completed_vertical_slice: CompletedVerticalSlice,
):
    artifact = completed_vertical_slice.artifact_path
    assert artifact.is_dir()
    assert REQUIRED_OUTPUTS.issubset({path.name for path in artifact.iterdir()})
    assert {path.stem for path in (artifact / "experiment_configs").glob("*.json")} == EXPECTED_VARIANTS

    final_summary = json.loads((artifact / "final_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert completed_vertical_slice.summary == final_summary
    assert final_summary["status"] == "PASS"
    assert "not financial advice" in final_summary["disclaimer"].lower()
    assert final_summary["data_source"]["source"] == "deterministic_fixture"
    assert final_summary["data_source"]["mode"] == "fixture_test"
    assert final_summary["data_source"]["is_real_data"] is False
    assert manifest["status"] == "PASS"
    assert manifest["data"] == final_summary["data_source"]
    assert set(manifest["registered_model_ids"]) == EXPECTED_VARIANTS
    assert set(manifest["portfolio_ids"]) == EXPECTED_VARIANTS
    assert len(
        {
            portfolio_id
            for identities in manifest["portfolio_ids"].values()
            for portfolio_id in identities.values()
        }
    ) == 12
    for key in (
        "best_validation_prediction_model",
        "best_test_period_simulated_portfolio",
        "lowest_drawdown_portfolio",
        "buy_and_hold_result",
        "any_model_portfolio_beat_buy_and_hold",
    ):
        assert key in final_summary
    assert "distinct measurements" in " ".join(final_summary["warnings"])
    assert sum(final_summary["common_split_counts"].values()) > 0
    for split_name, split_count in final_summary["common_split_counts"].items():
        assert sum(final_summary["class_distributions"][split_name].values()) == split_count

    model_metrics = pd.read_csv(artifact / "model_metrics.csv")
    portfolio_metrics = pd.read_csv(artifact / "portfolio_metrics.csv")
    leaderboard = pd.read_csv(artifact / "combined_leaderboard.csv")
    equity = pd.read_csv(artifact / "daily_equity_curves.csv")
    assert set(model_metrics["model_id"]) == EXPECTED_VARIANTS
    assert set(leaderboard["model_id"]) == EXPECTED_VARIANTS
    assert {
        "model_id",
        "registered_model_id",
        "portfolio_id",
        "algorithm",
        "validation_balanced_accuracy",
        "validation_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "ending_portfolio_value",
        "total_return",
        "maximum_drawdown",
        "volatility",
        "sharpe_ratio",
        "number_of_trades",
        "fees_paid",
        "result_versus_buy_and_hold",
    }.issubset(leaderboard.columns)
    assert len(portfolio_metrics) == 6
    assert portfolio_metrics["portfolio_id"].nunique() == 6
    assert equity["portfolio_id"].nunique() == 6

    buy_hold_return = float(
        portfolio_metrics.loc[
            portfolio_metrics["portfolio_type"].eq("buy_and_hold"), "total_return"
        ].iloc[0]
    )
    model_portfolios = portfolio_metrics.loc[
        portfolio_metrics["portfolio_type"].eq("model")
    ].set_index("model_id")
    model_scores = model_metrics.set_index("model_id")
    for row in leaderboard.to_dict(orient="records"):
        portfolio = model_portfolios.loc[row["model_id"]]
        model = model_scores.loc[row["model_id"]]
        assert row["ending_portfolio_value"] == pytest.approx(
            float(portfolio["ending_portfolio_value"])
        )
        assert row["result_versus_buy_and_hold"] == pytest.approx(
            float(portfolio["total_return"]) - buy_hold_return
        )
        assert row["validation_balanced_accuracy"] == pytest.approx(
            float(model["validation_balanced_accuracy"])
        )

    for filename in (
        "model_metric_comparison.png",
        "portfolio_equity_comparison.png",
        "drawdown_comparison.png",
    ):
        assert (artifact / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
