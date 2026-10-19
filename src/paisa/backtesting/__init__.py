"""Backtesting engines, benchmarks, metrics, and immutable reports."""

from paisa.backtesting.benchmarks import BenchmarkOutcome, run_buy_and_hold, run_cash_only
from paisa.backtesting.engine import BacktestEngine, BacktestOutcome
from paisa.backtesting.metrics import (
    calculate_daily_returns,
    calculate_drawdowns,
    calculate_portfolio_metrics,
    calculate_prediction_metrics,
)

__all__ = [
    "BacktestEngine",
    "BacktestOutcome",
    "BenchmarkOutcome",
    "calculate_daily_returns",
    "calculate_drawdowns",
    "calculate_portfolio_metrics",
    "calculate_prediction_metrics",
    "run_buy_and_hold",
    "run_cash_only",
]
