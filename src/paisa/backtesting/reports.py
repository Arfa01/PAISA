"""Atomic, immutable artifact writers and headless deterministic charts."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from paisa.backtesting.benchmarks import BenchmarkOutcome
from paisa.backtesting.engine import BacktestOutcome
from paisa.backtesting.metrics import calculate_drawdowns
from paisa.trading.models import ExperimentConfig, PortfolioSnapshot


PREDICTION_FIELDS = [
    "schema_version",
    "model_id",
    "ticker",
    "feature_date",
    "target_date",
    "predicted_class",
    "probability_up",
    "probability_down",
    "actual_class",
]
SIGNAL_FIELDS = [
    "schema_version",
    "signal_id",
    "portfolio_id",
    "model_id",
    "ticker",
    "signal_date",
    "target_date",
    "action",
    "probability_up",
    "target_allocation",
    "reason",
]
TRADE_FIELDS = [
    "schema_version",
    "trade_id",
    "portfolio_id",
    "signal_date",
    "execution_date",
    "action",
    "quantity",
    "reference_price",
    "execution_price",
    "fee",
    "cash_after_trade",
    "shares_after_trade",
]
SNAPSHOT_FIELDS = [
    "schema_version",
    "portfolio_id",
    "date",
    "cash",
    "shares",
    "market_price",
    "position_value",
    "total_equity",
    "realized_profit_loss",
    "unrealized_profit_loss",
]


def _json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def _write_csv(path: Path, fieldnames: Sequence[str], records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def _plot_equity_curve(
    path: Path,
    strategy: Sequence[PortfolioSnapshot],
    cash_only: Sequence[PortfolioSnapshot],
    buy_and_hold: Sequence[PortfolioSnapshot],
) -> None:
    figure = Figure(figsize=(9, 5), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    for label, snapshots, color in (
        ("Strategy", strategy, "#1f77b4"),
        ("Cash only", cash_only, "#7f7f7f"),
        ("Buy and hold", buy_and_hold, "#2ca02c"),
    ):
        axis.plot(
            [snapshot.date for snapshot in snapshots],
            [float(snapshot.total_equity) for snapshot in snapshots],
            label=label,
            color=color,
            linewidth=1.8,
        )
    axis.set_title("OGDC simulated portfolio equity")
    axis.set_xlabel("Trading session")
    axis.set_ylabel("Equity (PKR)")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.text(0.5, 0.01, "Academic simulation — not financial advice", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(str(path), metadata={"Software": "PAISA"})


def _plot_drawdowns(
    path: Path,
    strategy: Sequence[PortfolioSnapshot],
    cash_only: Sequence[PortfolioSnapshot],
    buy_and_hold: Sequence[PortfolioSnapshot],
) -> None:
    figure = Figure(figsize=(9, 5), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    for label, snapshots, color in (
        ("Strategy", strategy, "#1f77b4"),
        ("Cash only", cash_only, "#7f7f7f"),
        ("Buy and hold", buy_and_hold, "#2ca02c"),
    ):
        drawdowns = calculate_drawdowns(snapshots)
        axis.plot(
            [item["date"] for item in drawdowns],
            [item["drawdown"] for item in drawdowns],
            label=label,
            color=color,
            linewidth=1.8,
        )
    axis.set_title("OGDC simulated portfolio drawdown")
    axis.set_xlabel("Trading session")
    axis.set_ylabel("Drawdown")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend()
    axis.tick_params(axis="x", rotation=30)
    figure.text(0.5, 0.01, "Academic simulation — not financial advice", ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(str(path), metadata={"Software": "PAISA"})


def write_experiment_report(
    *,
    config: ExperimentConfig,
    strategy: BacktestOutcome,
    cash_only: BenchmarkOutcome,
    buy_and_hold: BenchmarkOutcome,
    metrics: Mapping[str, Any],
    runs_root: str | Path = "runs",
) -> Path:
    """Publish a complete run directory atomically, refusing all collisions."""

    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / config.experiment_id
    if destination.exists():
        raise FileExistsError(f"Experiment artifact already exists: {destination}")

    staging = Path(tempfile.mkdtemp(prefix=f".{config.experiment_id}.tmp-", dir=root))
    try:
        benchmarks_dir = staging / "benchmarks"
        benchmarks_dir.mkdir()

        (staging / "config.json").write_text(
            _json_text(config.to_dict()), encoding="utf-8"
        )
        _write_csv(
            staging / "predictions.csv",
            PREDICTION_FIELDS,
            (prediction.to_dict() for prediction in strategy.predictions),
        )
        _write_csv(
            staging / "signals.csv",
            SIGNAL_FIELDS,
            (signal.to_dict() for signal in strategy.signals),
        )
        _write_csv(
            staging / "trades.csv",
            TRADE_FIELDS,
            (trade.to_dict() for trade in strategy.trades),
        )
        _write_csv(
            staging / "portfolio_snapshots.csv",
            SNAPSHOT_FIELDS,
            (snapshot.to_dict() for snapshot in strategy.snapshots),
        )
        metrics_text = _json_text(metrics)
        (staging / "metrics.json").write_text(metrics_text, encoding="utf-8")
        # ARCHITECTURE_CONTRACTS names result.json as the stable API artifact;
        # metrics.json is retained as the sprint's requested human-facing name.
        (staging / "result.json").write_text(metrics_text, encoding="utf-8")

        _write_csv(
            benchmarks_dir / "cash_only_snapshots.csv",
            SNAPSHOT_FIELDS,
            (snapshot.to_dict() for snapshot in cash_only.snapshots),
        )
        _write_csv(
            benchmarks_dir / "buy_and_hold_snapshots.csv",
            SNAPSHOT_FIELDS,
            (snapshot.to_dict() for snapshot in buy_and_hold.snapshots),
        )
        _write_csv(
            benchmarks_dir / "buy_and_hold_trades.csv",
            TRADE_FIELDS,
            (trade.to_dict() for trade in buy_and_hold.trades),
        )
        _plot_equity_curve(
            staging / "equity_curve.png",
            strategy.snapshots,
            cash_only.snapshots,
            buy_and_hold.snapshots,
        )
        _plot_drawdowns(
            staging / "drawdown.png",
            strategy.snapshots,
            cash_only.snapshots,
            buy_and_hold.snapshots,
        )

        if destination.exists():
            raise FileExistsError(f"Experiment artifact already exists: {destination}")
        os.rename(staging, destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination
