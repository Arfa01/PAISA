from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from paisa.experiments.runner import run_experiment


PREDICTION_HEADER = (
    "schema_version",
    "model_id",
    "ticker",
    "feature_date",
    "target_date",
    "predicted_class",
    "probability_up",
    "probability_down",
    "actual_class",
)
PRICE_HEADER = ("date", "ticker", "open", "close")


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _inputs(root: Path, *, experiment_id: str = "ogdc-run-one", portfolio_id: str = "strategy-one") -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / "predictions.csv"
    prices = root / "prices.csv"
    config = root / "config.json"
    _write_csv(
        predictions,
        PREDICTION_HEADER,
        [
            ("1.0", "model-ogdc-v1", "OGDC", "2026-01-05", "2026-01-06", "1", "0.70", "0.30", "1"),
            ("1.0", "model-ogdc-v1", "OGDC", "2026-01-06", "2026-01-07", "0", "0.30", "0.70", "0"),
            ("1.0", "model-ogdc-v1", "OGDC", "2026-01-07", "2026-01-08", "1", "0.65", "0.35", "1"),
        ],
    )
    _write_csv(
        prices,
        PRICE_HEADER,
        [
            ("2026-01-05", "OGDC", "100.00", "101.00"),
            ("2026-01-06", "OGDC", "102.00", "103.00"),
            ("2026-01-07", "OGDC", "99.00", "98.00"),
            ("2026-01-08", "OGDC", "97.00", "100.00"),
        ],
    )
    raw_config = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "model_id": "model-ogdc-v1",
        "portfolio_id": portfolio_id,
        "benchmark_portfolio_ids": {
            "cash_only": f"{experiment_id}-cash",
            "buy_and_hold": f"{experiment_id}-buy-hold",
        },
        "symbol": "OGDC",
        "initial_cash": "100000.00",
        "thresholds": {"buy_probability_up_gte": 0.60, "sell_probability_up_lte": 0.40},
        "maximum_allocation": 0.80,
        "transaction_cost": {"type": "proportional_bps", "value": 10},
        "evaluation_dates": {
            "feature_start": "2026-01-05",
            "feature_end": "2026-01-07",
            "target_start": "2026-01-06",
            "target_end": "2026-01-08",
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
        "metric_assumptions": {"annualization_factor": 252, "risk_free_rate_annual": 0.0},
        "created_at": "2026-01-01T00:00:00Z",
    }
    config.write_text(json.dumps(raw_config, indent=2) + "\n", encoding="utf-8")
    return predictions, prices, config


def _run(inputs: tuple[Path, Path, Path], runs_root: Path):
    predictions, prices, config = inputs
    return run_experiment(
        predictions_path=predictions,
        prices_path=prices,
        config_path=config,
        runs_root=runs_root,
    )


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_runner_writes_complete_normalized_experiment_artifacts(tmp_path: Path):
    inputs = _inputs(tmp_path)
    run = _run(inputs, tmp_path / "runs")
    artifact = run.artifact_path

    assert artifact == tmp_path / "runs" / "ogdc-run-one"
    assert {path.relative_to(artifact).as_posix() for path in artifact.rglob("*")} == {
        "config.json",
        "predictions.csv",
        "signals.csv",
        "trades.csv",
        "portfolio_snapshots.csv",
        "metrics.json",
        "result.json",
        "equity_curve.png",
        "drawdown.png",
        "benchmarks",
        "benchmarks/cash_only_snapshots.csv",
        "benchmarks/buy_and_hold_snapshots.csv",
        "benchmarks/buy_and_hold_trades.csv",
    }
    assert (artifact / "equity_curve.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (artifact / "drawdown.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    normalized_config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    assert normalized_config["schema_version"] == "1.0"
    assert normalized_config["initial_cash"] == "100000.00"
    assert normalized_config["symbol"] == "OGDC"
    with (artifact / "predictions.csv").open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert tuple(reader.fieldnames or ()) == PREDICTION_HEADER
        predictions = list(reader)
    assert len(predictions) == 3
    assert [row["feature_date"] for row in predictions] == [
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
    ]
    assert predictions[0] == {
        "schema_version": "1.0",
        "model_id": "model-ogdc-v1",
        "ticker": "OGDC",
        "feature_date": "2026-01-05",
        "target_date": "2026-01-06",
        "predicted_class": "1",
        "probability_up": "0.7",
        "probability_down": "0.3",
        "actual_class": "1",
    }
    result = json.loads((artifact / "result.json").read_text(encoding="utf-8"))
    assert result["experiment_id"] == "ogdc-run-one"
    assert result["model_id"] == "model-ogdc-v1"
    assert result["portfolio_id"] == "strategy-one"
    assert result["benchmark_comparison"]["cash_only"]["snapshots_path"] == (
        "benchmarks/cash_only_snapshots.csv"
    )
    assert result["benchmark_comparison"]["buy_and_hold"]["trades_path"] == (
        "benchmarks/buy_and_hold_trades.csv"
    )


def test_runner_refuses_existing_experiment_and_preserves_all_bytes(tmp_path: Path):
    inputs = _inputs(tmp_path)
    root = tmp_path / "runs"
    run = _run(inputs, root)
    before = _tree_bytes(run.artifact_path)

    with pytest.raises((FileExistsError, ValueError)):
        _run(inputs, root)

    assert _tree_bytes(run.artifact_path) == before


def test_runner_rejects_portfolio_id_reuse_by_another_experiment(tmp_path: Path):
    root = tmp_path / "runs"
    first_inputs = _inputs(tmp_path / "first", experiment_id="experiment-first", portfolio_id="shared-ledger")
    _run(first_inputs, root)

    second_inputs = _inputs(tmp_path / "second", experiment_id="experiment-second", portfolio_id="shared-ledger")
    with pytest.raises(ValueError, match="portfolio_id is already registered"):
        _run(second_inputs, root)
    assert not (root / "experiment-second").exists()


def test_runner_serialized_outputs_are_deterministic_across_roots(tmp_path: Path):
    inputs = _inputs(tmp_path)
    first = _run(inputs, tmp_path / "runs-a")
    second = _run(inputs, tmp_path / "runs-b")

    first_bytes = _tree_bytes(first.artifact_path)
    second_bytes = _tree_bytes(second.artifact_path)
    assert first_bytes.keys() == second_bytes.keys()
    assert first_bytes == second_bytes
    # Make the intended byte-for-byte comparison explicit for the binary reports.
    for name in ("equity_curve.png", "drawdown.png"):
        assert hashlib.sha256(first_bytes[name]).digest() == hashlib.sha256(second_bytes[name]).digest()
