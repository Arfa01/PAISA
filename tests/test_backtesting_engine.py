from dataclasses import replace
from datetime import date
from decimal import Decimal

import pytest

from paisa.backtesting.benchmarks import run_buy_and_hold, run_cash_only
from paisa.backtesting.engine import BacktestEngine
from paisa.trading.models import (
    Action,
    BenchmarkPortfolioIds,
    EvaluationDates,
    ExperimentConfig,
    MarketBar,
    PredictionRecord,
    TransactionCostConfig,
)
from paisa.trading.portfolio import PortfolioLedger


def _config(*, allocation="0.6", fee_bps="100"):
    return ExperimentConfig(
        experiment_id="exp-engine-tests",
        model_id="model-engine-tests",
        portfolio_id="strategy-engine-tests",
        benchmark_portfolio_ids=BenchmarkPortfolioIds("cash-engine-tests", "hold-engine-tests"),
        evaluation_dates=EvaluationDates(
            feature_start=date(2025, 2, 7),  # Friday
            feature_end=date(2025, 2, 11),
            target_start=date(2025, 2, 10),  # next observed session, Monday
            target_end=date(2025, 2, 12),
        ),
        created_at="2025-02-01T00:00:00Z",
        initial_cash=Decimal("1000.00"),
        maximum_allocation=Decimal(allocation),
        transaction_cost=TransactionCostConfig(value=Decimal(fee_bps)),
    )


def _bars():
    return (
        MarketBar("OGDC", date(2025, 2, 7), Decimal("90"), Decimal("95")),
        MarketBar("OGDC", date(2025, 2, 10), Decimal("100"), Decimal("110")),
        MarketBar("OGDC", date(2025, 2, 11), Decimal("120"), Decimal("115")),
        MarketBar("OGDC", date(2025, 2, 12), Decimal("200"), Decimal("210")),
    )


def _predictions():
    return (
        PredictionRecord(
            "model-engine-tests", "OGDC", date(2025, 2, 7), date(2025, 2, 10),
            1, Decimal("0.9"), Decimal("0.1"), actual_class=0,
        ),
        PredictionRecord(
            "model-engine-tests", "OGDC", date(2025, 2, 10), date(2025, 2, 11),
            0, Decimal("0.1"), Decimal("0.9"), actual_class=1,
        ),
        PredictionRecord(
            "model-engine-tests", "OGDC", date(2025, 2, 11), date(2025, 2, 12),
            1, Decimal("0.9"), Decimal("0.1"), actual_class=0,
        ),
    )


def test_backtest_preserves_nonnegative_balances_and_only_sells_owned_shares():
    cfg = _config()
    ledger = PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)

    outcome = BacktestEngine(cfg).run(_predictions(), _bars(), ledger=ledger)

    assert outcome.trades
    assert all(snapshot.cash >= 0 and snapshot.shares >= 0 for snapshot in outcome.snapshots)
    assert all(trade.cash_after_trade >= 0 and trade.shares_after_trade >= 0 for trade in outcome.trades)
    # The policy sells the whole position; its sale quantity must match the shares owned before it.
    bought = None
    for trade in outcome.trades:
        if trade.action is Action.BUY:
            assert trade.shares_after_trade == trade.quantity
            bought = trade.shares_after_trade
        else:
            assert bought is not None
            assert trade.quantity == bought
            assert trade.shares_after_trade == 0
            bought = None


def test_duplicate_prediction_dates_are_rejected_without_duplicate_trades():
    cfg = _config()
    predictions = _predictions()
    with pytest.raises(ValueError, match="Duplicate prediction"):
        BacktestEngine(cfg).run((*predictions, predictions[0]), _bars())


def test_future_market_price_mutation_cannot_change_earlier_trades():
    cfg = _config()
    original = BacktestEngine(cfg).run(_predictions(), _bars())
    changed_bars = (*_bars()[:2], replace(_bars()[2], open=Decimal("9999"), close=Decimal("8888")), _bars()[3])
    changed = BacktestEngine(cfg).run(_predictions(), changed_bars)

    original_early = tuple(t for t in original.trades if t.execution_date == date(2025, 2, 10))
    changed_early = tuple(t for t in changed.trades if t.execution_date == date(2025, 2, 10))
    assert original_early == changed_early


def test_signals_execute_on_exact_next_observed_session_across_weekend():
    cfg = _config()
    outcome = BacktestEngine(cfg).run(_predictions(), _bars())
    sessions = {bar.date for bar in outcome.market_bars}

    assert outcome.signals[0].signal_date == date(2025, 2, 7)
    assert outcome.signals[0].target_date == date(2025, 2, 10)
    assert (outcome.signals[0].target_date - outcome.signals[0].signal_date).days == 3
    assert all(
        signal.target_date
        == next(bar.date for bar in outcome.market_bars if bar.date > signal.signal_date)
        and signal.target_date in sessions
        for signal in outcome.signals
    )
    assert all(
        trade.execution_date == next(
            bar.date for bar in outcome.market_bars if bar.date > trade.signal_date
        )
        for trade in outcome.trades
    )


def test_strategy_and_benchmarks_share_market_bars_but_not_portfolio_state():
    cfg = _config()
    strategy = BacktestEngine(cfg).run(_predictions(), _bars())
    cash_ledger = PortfolioLedger(cfg.benchmark_portfolio_ids.cash_only, cfg.initial_cash)
    hold_ledger = PortfolioLedger(cfg.benchmark_portfolio_ids.buy_and_hold, cfg.initial_cash)
    cash = run_cash_only(cfg, _bars(), ledger=cash_ledger)
    hold = run_buy_and_hold(cfg, _bars(), ledger=hold_ledger)

    assert strategy.market_bars == cash.market_bars == hold.market_bars
    assert cash_ledger is not hold_ledger
    assert strategy.trades
    assert cash.trades == ()
    assert hold.trades
    assert strategy.trades[0].portfolio_id != hold.trades[0].portfolio_id
    assert cash_ledger.cash == cfg.initial_cash and cash_ledger.shares == 0
    assert hold_ledger.cash < cfg.initial_cash and hold_ledger.shares > 0


def test_cash_only_benchmark_is_exact_initial_cash_throughout():
    cfg = _config()
    outcome = run_cash_only(cfg, _bars())

    assert outcome.trades == ()
    assert [snapshot.cash for snapshot in outcome.snapshots] == [cfg.initial_cash] * len(_bars())
    assert [snapshot.shares for snapshot in outcome.snapshots] == [0] * len(_bars())
    assert [snapshot.total_equity for snapshot in outcome.snapshots] == [cfg.initial_cash] * len(_bars())


def test_buy_and_hold_applies_allocation_and_fees_exactly():
    cfg = _config(allocation="0.6", fee_bps="100")
    outcome = run_buy_and_hold(cfg, _bars())

    trade = outcome.trades[0]
    assert trade.execution_date == date(2025, 2, 10)
    assert trade.reference_price == Decimal("100.00")
    assert trade.quantity == 6  # floor(60% of 1,000 / 100)
    assert trade.fee == Decimal("6.00")
    assert trade.cash_after_trade == Decimal("394.00")
    final = outcome.snapshots[-1]
    assert final.shares == 6
    assert final.position_value == Decimal("1260.00")
    assert final.total_equity == Decimal("1654.00")


def test_repeated_engine_runs_with_fresh_ledgers_are_identical():
    cfg = _config()

    first = BacktestEngine(cfg).run(
        _predictions(), _bars(), ledger=PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
    )
    second = BacktestEngine(cfg).run(
        _predictions(), _bars(), ledger=PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
    )

    assert first == second
