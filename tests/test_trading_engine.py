from datetime import date
from decimal import Decimal

import pytest

from paisa.trading import (
    Action,
    BenchmarkPortfolioIds,
    DecisionInput,
    EvaluationDates,
    ExecutionEngine,
    ExecutionPolicyConfig,
    ExperimentConfig,
    MarketBar,
    PortfolioLedger,
    SignalRecord,
    TargetAllocationPolicy,
    TransactionCostConfig,
    execute_signal,
)


def config(
    *,
    cash="1000.00",
    allocation="1",
    fee_bps="0",
    slippage_bps="0",
):
    return ExperimentConfig(
        experiment_id="exp-one",
        model_id="model-one",
        portfolio_id="portfolio-one",
        benchmark_portfolio_ids=BenchmarkPortfolioIds("cash-one", "hold-one"),
        evaluation_dates=EvaluationDates(
            feature_start=date(2025, 1, 1),
            feature_end=date(2025, 1, 2),
            target_start=date(2025, 1, 2),
            target_end=date(2025, 1, 3),
        ),
        created_at="2025-01-01T00:00:00Z",
        initial_cash=Decimal(cash),
        maximum_allocation=Decimal(allocation),
        transaction_cost=TransactionCostConfig(value=Decimal(fee_bps)),
        execution_policy=ExecutionPolicyConfig(slippage_bps=Decimal(slippage_bps)),
    )


def signal(action, *, signal_id="signal-one", probability="0.8", target="1"):
    return SignalRecord(
        signal_id=signal_id,
        portfolio_id="portfolio-one",
        model_id="model-one",
        ticker="OGDC",
        signal_date=date(2025, 1, 1),
        target_date=date(2025, 1, 2),
        action=action,
        probability_up=Decimal(probability),
        target_allocation=Decimal(target) if target is not None else None,
        reason="test",
    )


def bar(day=date(2025, 1, 2), *, opening="100.00", close="105.00"):
    return MarketBar(ticker="OGDC", date=day, open=Decimal(opening), close=Decimal(close))


def test_threshold_boundaries_and_neutral_band():
    policy = TargetAllocationPolicy.from_config(config())
    cases = (("0.60", Action.BUY), ("0.50", Action.HOLD), ("0.40", Action.HOLD))
    ledger = PortfolioLedger("portfolio-one")
    for probability, expected in cases:
        decision = DecisionInput(
            "model-one", "OGDC", date(2025, 1, 1), date(2025, 1, 2), Decimal(probability)
        )
        assert policy.decide(decision, ledger.view()).action is expected
    held = PortfolioLedger("portfolio-two")
    sell_policy = TargetAllocationPolicy.from_config(config())
    decision = DecisionInput(
        "model-one", "OGDC", date(2025, 1, 1), date(2025, 1, 2), Decimal("0.40")
    )
    assert sell_policy.decide(decision, held.view()).reason == "already_in_cash"


def test_buy_sizing_accounts_for_slippage_and_fee_and_snapshot_marks():
    cfg = config(cash="1000.00", allocation="0.5", fee_bps="100", slippage_bps="100")
    ledger = PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
    buy = signal(Action.BUY, target="0.5")
    trade = execute_signal(buy, bar(), ledger, cfg)
    assert trade is not None
    assert trade.reference_price == Decimal("100.00")
    assert trade.execution_price == Decimal("101.00")
    assert trade.quantity == 4
    assert trade.fee == Decimal("4.04")
    assert trade.cash_after_trade == Decimal("591.96")
    assert ledger.view().shares == 4
    snapshot = ledger.snapshot(date(2025, 1, 2), Decimal("105.00"))
    assert snapshot.position_value == Decimal("420.00")
    assert snapshot.total_equity == Decimal("1011.96")


def test_no_negative_cash_and_small_budget_buy_is_skipped():
    cfg = config(cash="50.00", fee_bps="500")
    ledger = PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
    assert ExecutionEngine(cfg).execute(signal(Action.BUY), bar(), ledger) is None
    assert ledger.cash == Decimal("50.00")
    assert ledger.shares == 0


def test_oversell_rejected_and_full_episode_accounting():
    cfg = config(cash="500.00")
    empty = PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
    with pytest.raises(ValueError, match="cannot sell more shares"):
        empty.apply_sell_all(
            signal(Action.SELL, probability="0.2", target="0"),
            execution_date=date(2025, 1, 2),
            reference_price="100",
            execution_price="100",
            fee="0",
        )

    ledger = PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
    opened = execute_signal(signal(Action.BUY), bar(opening="50", close="50"), ledger, cfg)
    assert opened and opened.quantity == 10
    # A second buy signal while invested is a policy HOLD, so there is no pyramid.
    decision = DecisionInput(
        "model-one", "OGDC", date(2025, 1, 2), date(2025, 1, 3), Decimal("0.9")
    )
    next_signal = TargetAllocationPolicy.from_config(cfg).decide(decision, ledger.view())
    assert next_signal.action is Action.HOLD

    sell = SignalRecord(
        signal_id="signal-sell",
        portfolio_id=cfg.portfolio_id,
        model_id="model-one",
        ticker="OGDC",
        signal_date=date(2025, 1, 2),
        target_date=date(2025, 1, 3),
        action=Action.SELL,
        probability_up=Decimal("0.2"),
        target_allocation=Decimal("0"),
        reason="test sell",
    )
    closed = execute_signal(
        sell,
        bar(date(2025, 1, 3), opening="60", close="59"),
        ledger,
        cfg,
    )
    assert closed and closed.quantity == 10
    assert ledger.shares == 0
    assert ledger.closed_episode_pnls == (Decimal("100.00"),)
    # The policy correctly turns another sell threshold into HOLD when flat.
    policy = TargetAllocationPolicy.from_config(cfg)
    flat_sell = policy.decide(
        DecisionInput(
            "model-one", "OGDC", date(2025, 1, 3), date(2025, 1, 4), Decimal("0.1")
        ),
        ledger.view(),
    )
    assert flat_sell.action is Action.HOLD
    assert flat_sell.reason == "already_in_cash"


def test_trade_ids_are_deterministic_and_signal_is_single_use():
    cfg = config(cash="500.00", allocation="0.5")
    def run_once():
        ledger = PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
        return execute_signal(signal(Action.BUY, target="0.5"), bar(opening="50"), ledger, cfg)

    first, second = run_once(), run_once()
    assert first and second
    assert first.trade_id == second.trade_id
    assert first.trade_id.startswith("trade-")

    ledger = PortfolioLedger(cfg.portfolio_id, cfg.initial_cash)
    buy = signal(Action.BUY, target="0.5")
    execute_signal(buy, bar(opening="50"), ledger, cfg)
    with pytest.raises(ValueError, match="already been executed"):
        execute_signal(buy, bar(opening="50"), ledger, cfg)
