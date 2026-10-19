from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest

from paisa.backtesting.metrics import (
    calculate_closed_episode_pnls,
    calculate_daily_returns,
    calculate_drawdowns,
    calculate_portfolio_metrics,
)
from paisa.trading.models import Action, PortfolioSnapshot, TradeRecord


def _snapshot(
    day: str,
    *,
    cash: str,
    shares: int,
    market_price: str,
    realized: str = "0",
    unrealized: str = "0",
) -> PortfolioSnapshot:
    price = Decimal(market_price)
    position = Decimal(shares) * price
    cash_value = Decimal(cash)
    return PortfolioSnapshot(
        portfolio_id="portfolio-metrics",
        date=date.fromisoformat(day),
        cash=cash_value,
        shares=shares,
        market_price=price,
        position_value=position,
        total_equity=cash_value + position,
        realized_profit_loss=Decimal(realized),
        unrealized_profit_loss=Decimal(unrealized),
    )


def _trade(
    trade_id: str,
    signal_day: str,
    execution_day: str,
    action: Action,
    *,
    quantity: int,
    price: str,
    fee: str = "0",
    cash_after: str,
    shares_after: int,
) -> TradeRecord:
    return TradeRecord(
        trade_id=trade_id,
        portfolio_id="portfolio-metrics",
        signal_date=date.fromisoformat(signal_day),
        execution_date=date.fromisoformat(execution_day),
        action=action,
        quantity=quantity,
        reference_price=Decimal(price),
        execution_price=Decimal(price),
        fee=Decimal(fee),
        cash_after_trade=Decimal(cash_after),
        shares_after_trade=shares_after,
    )


def _canonical_snapshots() -> list[PortfolioSnapshot]:
    return [
        _snapshot("2026-01-05", cash="1000", shares=0, market_price="10"),
        _snapshot("2026-01-06", cash="500", shares=50, market_price="11", unrealized="50"),
        _snapshot("2026-01-07", cash="500", shares=50, market_price="12", unrealized="100"),
        _snapshot("2026-01-08", cash="900", shares=0, market_price="8", realized="-100"),
        _snapshot("2026-01-09", cash="450", shares=50, market_price="9", realized="-100"),
        _snapshot("2026-01-12", cash="950", shares=0, market_price="10", realized="-50"),
    ]


def _canonical_trades() -> list[TradeRecord]:
    return [
        _trade(
            "trade-1",
            "2026-01-05",
            "2026-01-06",
            Action.BUY,
            quantity=50,
            price="10",
            cash_after="500",
            shares_after=50,
        ),
        _trade(
            "trade-2",
            "2026-01-07",
            "2026-01-08",
            Action.SELL,
            quantity=50,
            price="8",
            cash_after="900",
            shares_after=0,
        ),
        _trade(
            "trade-3",
            "2026-01-08",
            "2026-01-09",
            Action.BUY,
            quantity=50,
            price="9",
            cash_after="450",
            shares_after=50,
        ),
        _trade(
            "trade-4",
            "2026-01-09",
            "2026-01-12",
            Action.SELL,
            quantity=50,
            price="10",
            cash_after="950",
            shares_after=0,
        ),
    ]


def test_portfolio_metrics_match_deterministic_known_curve():
    snapshots = _canonical_snapshots()
    trades = _canonical_trades()

    first = calculate_portfolio_metrics(snapshots, trades)
    second = calculate_portfolio_metrics(snapshots, trades)

    assert first == second
    assert first["initial_equity"] == "1000.00"
    assert first["ending_equity"] == "950.00"
    assert first["total_return"] == pytest.approx(-0.05)
    assert first["maximum_drawdown"] == pytest.approx(2 / 11)
    assert first["volatility"] == pytest.approx(1.6022198624009514)
    assert first["sharpe_ratio"] == pytest.approx(-0.9010226358528869)
    assert first["number_of_trades"] == 4
    assert first["win_rate"] == pytest.approx(0.5)
    assert first["average_gain"] == "50.00"
    assert first["average_loss"] == "-100.00"
    assert first["turnover"] == pytest.approx(1.8813559322033897)
    assert first["fees_paid"] == "0.00"
    assert first["time_in_market"] == pytest.approx(3 / 5)

    assert [item["date"] for item in first["daily_returns"]] == [
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
    ]
    assert [item["return"] for item in first["daily_returns"]] == pytest.approx(
        [0.05, 1 / 21, -2 / 11, 0.0, 1 / 18]
    )
    assert max(item["drawdown"] for item in calculate_drawdowns(snapshots)) == pytest.approx(2 / 11)


def test_volatility_and_sharpe_edge_cases():
    opening = _snapshot("2026-01-05", cash="1000", shares=0, market_price="10")
    one_snapshot = calculate_portfolio_metrics([opening], [])

    assert one_snapshot["daily_returns"] == []
    assert one_snapshot["volatility"] is None
    assert one_snapshot["sharpe_ratio"] is None
    assert one_snapshot["maximum_drawdown"] == 0.0
    assert one_snapshot["turnover"] == 0.0
    assert one_snapshot["time_in_market"] == 0.0

    constant = [
        opening,
        _snapshot("2026-01-06", cash="1000", shares=0, market_price="11"),
        _snapshot("2026-01-07", cash="1000", shares=0, market_price="12"),
    ]
    constant_metrics = calculate_portfolio_metrics(constant, [])
    assert constant_metrics["volatility"] == 0.0
    assert constant_metrics["sharpe_ratio"] is None
    assert constant_metrics["win_rate"] is None
    assert constant_metrics["average_gain"] is None
    assert constant_metrics["average_loss"] is None


def test_closed_episode_metrics_include_buy_and_sell_fees():
    trades = [
        _trade(
            "fee-trade-1",
            "2026-01-05",
            "2026-01-06",
            Action.BUY,
            quantity=10,
            price="10",
            fee="1",
            cash_after="899",
            shares_after=10,
        ),
        _trade(
            "fee-trade-2",
            "2026-01-06",
            "2026-01-07",
            Action.SELL,
            quantity=10,
            price="12",
            fee="1",
            cash_after="1018",
            shares_after=0,
        ),
        _trade(
            "fee-trade-3",
            "2026-01-07",
            "2026-01-08",
            Action.BUY,
            quantity=10,
            price="10",
            fee="1",
            cash_after="917",
            shares_after=10,
        ),
        _trade(
            "fee-trade-4",
            "2026-01-08",
            "2026-01-09",
            Action.SELL,
            quantity=10,
            price="8",
            fee="1",
            cash_after="996",
            shares_after=0,
        ),
    ]

    assert calculate_closed_episode_pnls(trades) == [Decimal("18.00"), Decimal("-22.00")]

    metrics = calculate_portfolio_metrics(_canonical_snapshots(), trades)
    assert metrics["win_rate"] == pytest.approx(0.5)
    assert metrics["average_gain"] == "18.00"
    assert metrics["average_loss"] == "-22.00"
    assert metrics["fees_paid"] == "4.00"


def test_metrics_reject_non_chronological_snapshots_and_trades():
    snapshots = [
        _snapshot("2026-01-06", cash="1000", shares=0, market_price="10"),
        _snapshot("2026-01-05", cash="1000", shares=0, market_price="10"),
    ]
    with pytest.raises(ValueError, match="strictly increasing dates"):
        calculate_daily_returns(snapshots)

    trades = [
        _trade(
            "late-buy",
            "2026-01-09",
            "2026-01-10",
            Action.BUY,
            quantity=10,
            price="10",
            cash_after="900",
            shares_after=10,
        ),
        _trade(
            "early-sell",
            "2026-01-07",
            "2026-01-08",
            Action.SELL,
            quantity=10,
            price="11",
            cash_after="1010",
            shares_after=0,
        ),
    ]
    with pytest.raises(ValueError, match="strictly increasing execution dates"):
        calculate_closed_episode_pnls(trades)
