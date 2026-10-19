"""File-oriented orchestration for immutable portfolio experiments."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from paisa.backtesting.benchmarks import BenchmarkOutcome, run_buy_and_hold, run_cash_only
from paisa.backtesting.engine import BacktestEngine, BacktestOutcome
from paisa.backtesting.metrics import (
    calculate_portfolio_metrics,
    calculate_prediction_metrics,
)
from paisa.backtesting.reports import write_experiment_report
from paisa.experiments.portfolio_registry import PortfolioRegistry
from paisa.trading.models import ExperimentConfig, MarketBar, PredictionRecord
from paisa.trading.portfolio import assert_independent_ledgers


PREDICTION_COLUMNS = {
    "schema_version",
    "model_id",
    "ticker",
    "feature_date",
    "target_date",
    "predicted_class",
    "probability_up",
    "probability_down",
    "actual_class",
}
PRICE_COLUMNS = {"date", "ticker", "open", "close"}


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """Completed experiment plus its immutable artifact location."""

    artifact_path: Path
    config: ExperimentConfig
    strategy: BacktestOutcome
    cash_only: BenchmarkOutcome
    buy_and_hold: BenchmarkOutcome
    metrics: Mapping[str, Any]


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, dict):
        raise ValueError("Experiment configuration root must be a JSON object.")
    return ExperimentConfig.from_mapping(raw)


def _read_rows(path: str | Path, required_columns: set[str]) -> list[dict[str, str]]:
    source_path = Path(path)
    with source_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source_path}")
        missing = required_columns - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV {source_path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return [dict(row) for row in reader]


def load_predictions(path: str | Path) -> tuple[PredictionRecord, ...]:
    rows = _read_rows(path, PREDICTION_COLUMNS)
    return tuple(PredictionRecord.from_mapping(row) for row in rows)


def load_market_bars(path: str | Path) -> tuple[MarketBar, ...]:
    rows = _read_rows(path, PRICE_COLUMNS)
    return tuple(MarketBar.from_mapping(row) for row in rows)


def _metric_options(config: ExperimentConfig) -> dict[str, Any]:
    return {
        "annualization_factor": config.metric_assumptions.annualization_factor,
        "risk_free_rate_annual": float(config.metric_assumptions.risk_free_rate_annual),
    }


def build_experiment_metrics(
    config: ExperimentConfig,
    strategy: BacktestOutcome,
    cash_only: BenchmarkOutcome,
    buy_and_hold: BenchmarkOutcome,
) -> dict[str, Any]:
    """Combine portfolio, prediction, and benchmark contract metrics."""

    metric_options = _metric_options(config)
    strategy_metrics = calculate_portfolio_metrics(
        strategy.snapshots, strategy.trades, **metric_options
    )
    cash_metrics = calculate_portfolio_metrics(
        cash_only.snapshots, cash_only.trades, **metric_options
    )
    buy_hold_metrics = calculate_portfolio_metrics(
        buy_and_hold.snapshots, buy_and_hold.trades, **metric_options
    )
    excess_cash = strategy_metrics["total_return"] - cash_metrics["total_return"]
    excess_buy_hold = strategy_metrics["total_return"] - buy_hold_metrics["total_return"]

    return {
        "schema_version": config.schema_version,
        "experiment_id": config.experiment_id,
        "model_id": config.model_id,
        "portfolio_id": config.portfolio_id,
        "prediction_metrics": calculate_prediction_metrics(strategy.predictions),
        **strategy_metrics,
        "comparison_with_buy_and_hold": excess_buy_hold,
        "benchmark_comparison": {
            "cash_only": {
                "portfolio_id": config.benchmark_portfolio_ids.cash_only,
                "snapshots_path": "benchmarks/cash_only_snapshots.csv",
                "ending_equity": cash_metrics["ending_equity"],
                "total_return": cash_metrics["total_return"],
            },
            "buy_and_hold": {
                "portfolio_id": config.benchmark_portfolio_ids.buy_and_hold,
                "snapshots_path": "benchmarks/buy_and_hold_snapshots.csv",
                "trades_path": "benchmarks/buy_and_hold_trades.csv",
                "ending_equity": buy_hold_metrics["ending_equity"],
                "total_return": buy_hold_metrics["total_return"],
            },
            "excess_return_vs_cash_only": excess_cash,
            "excess_return_vs_buy_and_hold": excess_buy_hold,
        },
    }


def run_experiment(
    *,
    predictions_path: str | Path,
    prices_path: str | Path,
    config_path: str | Path,
    runs_root: str | Path = "runs",
) -> ExperimentRun:
    """Load, validate, simulate, and atomically save one experiment."""

    config = load_experiment_config(config_path)
    registry = PortfolioRegistry(runs_root)
    portfolios = registry.register(config)
    try:
        assert_independent_ledgers(
            (portfolios.strategy, portfolios.cash_only, portfolios.buy_and_hold)
        )

        predictions = load_predictions(predictions_path)
        market_bars = load_market_bars(prices_path)
        strategy = BacktestEngine(config).run(
            predictions, market_bars, ledger=portfolios.strategy
        )
        cash_only = run_cash_only(config, market_bars, ledger=portfolios.cash_only)
        buy_and_hold = run_buy_and_hold(
            config, market_bars, ledger=portfolios.buy_and_hold
        )

        strategy_market = tuple(
            (bar.date, bar.open, bar.close) for bar in strategy.market_bars
        )
        for benchmark in (cash_only, buy_and_hold):
            benchmark_market = tuple(
                (bar.date, bar.open, bar.close) for bar in benchmark.market_bars
            )
            if benchmark_market != strategy_market:
                raise AssertionError("Every portfolio must receive identical market observations.")

        metrics = build_experiment_metrics(config, strategy, cash_only, buy_and_hold)
        artifact_path = write_experiment_report(
            config=config,
            strategy=strategy,
            cash_only=cash_only,
            buy_and_hold=buy_and_hold,
            metrics=metrics,
            runs_root=runs_root,
        )
    except Exception:
        registry.release(config)
        raise
    registry.finalize(config)
    return ExperimentRun(
        artifact_path=artifact_path,
        config=config,
        strategy=strategy,
        cash_only=cash_only,
        buy_and_hold=buy_and_hold,
        metrics=metrics,
    )
