from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

import pytest

from paisa.backtesting.reports import SIGNAL_FIELDS, SNAPSHOT_FIELDS, TRADE_FIELDS
from paisa.experiments.runner import (
    load_portfolio_snapshots,
    load_signals,
    load_trades,
)
from paisa.trading.models import PortfolioSnapshot, SignalRecord, TradeRecord


def _signal_row() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "signal_id": "signal-001",
        "portfolio_id": "portfolio-001",
        "model_id": "model-001",
        "ticker": "OGDC",
        "signal_date": "2026-01-05",
        "target_date": "2026-01-06",
        "action": "HOLD",
        "probability_up": "0.55",
        "target_allocation": "",
        "reason": "probability is between the configured thresholds",
    }


def _trade_row() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "trade_id": "trade-001",
        "portfolio_id": "portfolio-001",
        "signal_date": "2026-01-05",
        "execution_date": "2026-01-06",
        "action": "BUY",
        "quantity": "2",
        "reference_price": "100.00",
        "execution_price": "100.00",
        "fee": "0.20",
        "cash_after_trade": "799.80",
        "shares_after_trade": "2",
    }


def _snapshot_row() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "portfolio_id": "portfolio-001",
        "date": "2026-01-06",
        "cash": "799.80",
        "shares": "2",
        "market_price": "101.00",
        "position_value": "202.00",
        "total_equity": "1001.80",
        "realized_profit_loss": "0.00",
        "unrealized_profit_loss": "1.80",
    }


def _write_report(
    path: Path, fields: list[str], rows: list[dict[str, Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: row.get(field, "") for field in fields} for row in rows
        )


def test_persisted_ledger_loaders_round_trip_report_records(tmp_path: Path):
    signals_path = tmp_path / "signals.csv"
    trades_path = tmp_path / "trades.csv"
    snapshots_path = tmp_path / "portfolio_snapshots.csv"
    _write_report(signals_path, SIGNAL_FIELDS, [_signal_row()])
    _write_report(trades_path, TRADE_FIELDS, [_trade_row()])
    _write_report(snapshots_path, SNAPSHOT_FIELDS, [_snapshot_row()])

    signals = load_signals(signals_path)
    trades = load_trades(trades_path)
    snapshots = load_portfolio_snapshots(snapshots_path)

    assert len(signals) == len(trades) == len(snapshots) == 1
    assert signals[0].target_allocation is None
    assert trades[0].quantity == trades[0].shares_after_trade == 2
    assert snapshots[0].shares == 2
    assert SignalRecord.from_dict(signals[0].to_dict()) == signals[0]
    assert TradeRecord.from_dict(trades[0].to_dict()) == trades[0]
    assert PortfolioSnapshot.from_dict(snapshots[0].to_dict()) == snapshots[0]


@pytest.mark.parametrize(
    ("loader", "fields", "row"),
    [
        (load_signals, SIGNAL_FIELDS, _signal_row()),
        (load_trades, TRADE_FIELDS, _trade_row()),
        (load_portfolio_snapshots, SNAPSHOT_FIELDS, _snapshot_row()),
    ],
)
def test_persisted_ledger_loaders_require_the_exact_ordered_report_schema(
    tmp_path: Path,
    loader: Callable[[Path], tuple[Any, ...]],
    fields: list[str],
    row: dict[str, Any],
):
    missing_path = tmp_path / f"missing-{fields[1]}.csv"
    _write_report(missing_path, fields[:-1], [row])
    with pytest.raises(ValueError, match="expected"):
        loader(missing_path)

    reordered = [fields[1], fields[0], *fields[2:]]
    reordered_path = tmp_path / f"reordered-{fields[1]}.csv"
    _write_report(reordered_path, reordered, [row])
    with pytest.raises(ValueError, match="expected"):
        loader(reordered_path)

    extra = [*fields, "unexpected"]
    extra_path = tmp_path / f"extra-{fields[1]}.csv"
    _write_report(extra_path, extra, [{**row, "unexpected": "value"}])
    with pytest.raises(ValueError, match="expected"):
        loader(extra_path)


def test_signal_loader_rejects_semantically_invalid_persisted_rows(tmp_path: Path):
    path = tmp_path / "signals.csv"
    invalid = {**_signal_row(), "target_allocation": "0.50"}
    _write_report(path, SIGNAL_FIELDS, [invalid])

    with pytest.raises(ValueError, match="HOLD requires target_allocation=null"):
        load_signals(path)


@pytest.mark.parametrize("field", ["quantity", "shares_after_trade"])
def test_trade_loader_rejects_noncanonical_integer_text(
    tmp_path: Path, field: str
):
    path = tmp_path / f"trades-{field}.csv"
    invalid = {**_trade_row(), field: "2.0"}
    _write_report(path, TRADE_FIELDS, [invalid])

    with pytest.raises(ValueError, match=rf"{field} must be an integer"):
        load_trades(path)


def test_snapshot_loader_rejects_an_unreconciled_ledger_row(tmp_path: Path):
    path = tmp_path / "portfolio_snapshots.csv"
    invalid = {**_snapshot_row(), "total_equity": "999.99"}
    _write_report(path, SNAPSHOT_FIELDS, [invalid])

    with pytest.raises(ValueError, match=r"total_equity must equal cash \+ position_value"):
        load_portfolio_snapshots(path)
