"""Independent cash-only and buy-and-hold benchmark portfolios."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

from paisa.backtesting.engine import evaluation_market_bars
from paisa.trading.execution import execute_signal
from paisa.trading.models import (
    Action,
    ExperimentConfig,
    MarketBar,
    PortfolioSnapshot,
    SignalRecord,
    TradeRecord,
    stable_id,
)
from paisa.trading.portfolio import PortfolioLedger


@dataclass(frozen=True, slots=True)
class BenchmarkOutcome:
    """One benchmark's reconciled immutable in-memory records."""

    portfolio_id: str
    market_bars: tuple[MarketBar, ...]
    trades: tuple[TradeRecord, ...]
    snapshots: tuple[PortfolioSnapshot, ...]


def _fresh_ledger(
    ledger: PortfolioLedger | None, portfolio_id: str, config: ExperimentConfig
) -> PortfolioLedger:
    result = ledger or PortfolioLedger(portfolio_id, config.initial_cash)
    if result.portfolio_id != portfolio_id:
        raise ValueError("Benchmark ledger portfolio_id does not match its configured ID.")
    if result.cash != config.initial_cash or result.shares != 0:
        raise ValueError("Benchmark ledger must be fresh, flat, and independently funded.")
    return result


def run_cash_only(
    config: ExperimentConfig,
    market_bars: Sequence[MarketBar],
    *,
    ledger: PortfolioLedger | None = None,
) -> BenchmarkOutcome:
    """Keep initial cash unchanged while valuing every evaluation session."""

    bars = evaluation_market_bars(config, market_bars)
    portfolio_id = config.benchmark_portfolio_ids.cash_only
    cash = _fresh_ledger(ledger, portfolio_id, config)
    snapshots = tuple(cash.snapshot(bar.date, bar.close) for bar in bars)
    return BenchmarkOutcome(portfolio_id, bars, (), snapshots)


def run_buy_and_hold(
    config: ExperimentConfig,
    market_bars: Sequence[MarketBar],
    *,
    ledger: PortfolioLedger | None = None,
) -> BenchmarkOutcome:
    """Buy once at the first eligible open, then mark the position to market."""

    bars = evaluation_market_bars(config, market_bars)
    portfolio_id = config.benchmark_portfolio_ids.buy_and_hold
    portfolio = _fresh_ledger(ledger, portfolio_id, config)
    snapshots: list[PortfolioSnapshot] = [portfolio.snapshot(bars[0].date, bars[0].close)]
    trades: list[TradeRecord] = []

    first_execution = next(
        (bar for bar in bars if bar.date == config.evaluation_dates.target_start), None
    )
    if first_execution is None:
        raise ValueError("Buy-and-hold target_start is not an observed market session.")
    signal = SignalRecord(
        signal_id=stable_id(
            "signal",
            portfolio_id,
            "buy-and-hold",
            config.symbol,
            config.evaluation_dates.feature_start.isoformat(),
            config.evaluation_dates.target_start.isoformat(),
            Action.BUY.value,
        ),
        portfolio_id=portfolio_id,
        model_id=config.model_id,
        ticker=config.symbol,
        signal_date=config.evaluation_dates.feature_start,
        target_date=config.evaluation_dates.target_start,
        action=Action.BUY,
        probability_up=Decimal("1"),
        target_allocation=config.maximum_allocation,
        reason="buy_and_hold_entry",
    )

    for bar in bars[1:]:
        if bar.date == first_execution.date:
            trade = execute_signal(signal, bar, portfolio, config)
            if trade is not None:
                trades.append(trade)
        snapshots.append(portfolio.snapshot(bar.date, bar.close))

    return BenchmarkOutcome(portfolio_id, bars, tuple(trades), tuple(snapshots))
