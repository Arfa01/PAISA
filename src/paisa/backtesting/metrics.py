"""Deterministic portfolio metrics for PAISA backtest snapshots and trades."""

from __future__ import annotations

import math
import statistics
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Sequence

from paisa.trading.models import PortfolioSnapshot, PredictionRecord, TradeRecord


MONEY_QUANTUM = Decimal("0.01")


def calculate_prediction_metrics(
    predictions: Sequence[PredictionRecord],
) -> dict[str, Any]:
    """Return contract prediction metrics for rows whose label is known.

    Labels are consumed only here, after simulation.  Policy and execution
    inputs intentionally have no ``actual_class`` field.
    """

    known = [prediction for prediction in predictions if prediction.actual_class is not None]
    support = len(known)
    true_positive = sum(
        prediction.predicted_class == 1 and prediction.actual_class == 1
        for prediction in known
    )
    true_negative = sum(
        prediction.predicted_class == 0 and prediction.actual_class == 0
        for prediction in known
    )
    false_positive = sum(
        prediction.predicted_class == 1 and prediction.actual_class == 0
        for prediction in known
    )
    false_negative = sum(
        prediction.predicted_class == 0 and prediction.actual_class == 1
        for prediction in known
    )
    accuracy = (true_positive + true_negative) / support if support else 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "positive_label": 1,
        "average": "binary",
        "zero_division": 0,
    }


def _money(value: Decimal) -> Decimal:
    """Quantize a monetary value using the repository accounting convention."""

    return value.quantize(MONEY_QUANTUM, rounding=ROUND_HALF_UP)


def _money_string(value: Decimal) -> str:
    """Serialize money as a fixed-scale PKR string."""

    return format(_money(value), ".2f")


def _iso_date(value: date) -> str:
    return value.isoformat()


def _validate_snapshots(snapshots: Sequence[PortfolioSnapshot]) -> None:
    previous_date: date | None = None
    for snapshot in snapshots:
        if previous_date is not None and snapshot.date <= previous_date:
            raise ValueError("Portfolio snapshots must have unique, strictly increasing dates.")
        if not snapshot.total_equity.is_finite() or snapshot.total_equity <= 0:
            raise ValueError("Portfolio snapshot equity must be finite and strictly positive.")
        if not isinstance(snapshot.shares, int) or isinstance(snapshot.shares, bool) or snapshot.shares < 0:
            raise ValueError("Portfolio snapshot shares must be a non-negative integer.")
        previous_date = snapshot.date


def calculate_daily_returns(snapshots: Sequence[PortfolioSnapshot]) -> list[dict[str, Any]]:
    """Return dated simple returns for consecutive chronological snapshots.

    The opening snapshot has no return.  Each subsequent item is associated
    with the later snapshot's date and uses ``equity_t / equity_(t-1) - 1``.
    """

    _validate_snapshots(snapshots)
    records: list[dict[str, Any]] = []
    for previous, current in zip(snapshots, snapshots[1:]):
        value = (current.total_equity / previous.total_equity) - Decimal("1")
        records.append({"date": _iso_date(current.date), "return": float(value)})
    return records


def calculate_drawdowns(snapshots: Sequence[PortfolioSnapshot]) -> list[dict[str, Any]]:
    """Return dated peak-to-trough drawdown fractions for every snapshot."""

    _validate_snapshots(snapshots)
    records: list[dict[str, Any]] = []
    running_peak: Decimal | None = None
    for snapshot in snapshots:
        if running_peak is None or snapshot.total_equity > running_peak:
            running_peak = snapshot.total_equity
        drawdown = Decimal("1") - (snapshot.total_equity / running_peak)
        records.append({"date": _iso_date(snapshot.date), "drawdown": float(drawdown)})
    return records


def _trade_action(trade: TradeRecord) -> str:
    raw_action = getattr(trade.action, "value", trade.action)
    return str(raw_action).upper()


def _validate_trade(trade: TradeRecord, previous_date: date | None) -> None:
    if previous_date is not None and trade.execution_date <= previous_date:
        raise ValueError("Trades must have unique, strictly increasing execution dates.")
    if not isinstance(trade.quantity, int) or isinstance(trade.quantity, bool) or trade.quantity <= 0:
        raise ValueError("Trade quantity must be a strictly positive integer.")
    if not trade.execution_price.is_finite() or trade.execution_price <= 0:
        raise ValueError("Trade execution price must be finite and strictly positive.")
    if not trade.fee.is_finite() or trade.fee < 0:
        raise ValueError("Trade fee must be finite and non-negative.")


def calculate_closed_episode_pnls(trades: Sequence[TradeRecord]) -> list[Decimal]:
    """Calculate fee-aware P/L for each fully closed long-only episode.

    A supported episode consists of one buy from zero shares followed by one
    sale of the complete position.  An open final episode is intentionally
    excluded because sprint backtests mark it to market rather than force a
    liquidation.
    """

    episode_pnls: list[Decimal] = []
    open_quantity = 0
    open_cost = Decimal("0")
    previous_date: date | None = None

    for trade in trades:
        _validate_trade(trade, previous_date)
        action = _trade_action(trade)
        notional = _money(Decimal(trade.quantity) * trade.execution_price)

        if action == "BUY":
            if open_quantity != 0:
                raise ValueError("Adding to an open position is not supported in schema 1.0.")
            if trade.shares_after_trade != trade.quantity:
                raise ValueError("A BUY trade must leave exactly its purchased quantity in shares.")
            open_quantity = trade.quantity
            open_cost = _money(notional + trade.fee)
        elif action == "SELL":
            if open_quantity == 0:
                raise ValueError("A SELL trade cannot close a position that is not open.")
            if trade.quantity != open_quantity or trade.shares_after_trade != 0:
                raise ValueError("A SELL trade must close the complete open position.")
            proceeds = _money(notional - trade.fee)
            episode_pnls.append(_money(proceeds - open_cost))
            open_quantity = 0
            open_cost = Decimal("0")
        else:
            raise ValueError(f"Unsupported trade action: {action!r}.")

        previous_date = trade.execution_date

    return episode_pnls


def calculate_portfolio_metrics(
    snapshots: Sequence[PortfolioSnapshot],
    trades: Sequence[TradeRecord],
    *,
    annualization_factor: int = 252,
    risk_free_rate_annual: float = 0.0,
) -> dict[str, Any]:
    """Calculate deterministic strategy metrics from one independent ledger.

    Monetary outputs use fixed-scale strings.  Ratio and rate outputs are JSON-
    compatible floats.  Volatility is the annualized sample standard deviation
    of daily returns.  Sharpe uses the contract's geometrically converted daily
    risk-free rate and is null when there are fewer than two returns or when
    sample volatility is zero.

    Turnover is total absolute fill notional divided by mean snapshot equity.
    Time in market is the fraction of post-opening snapshots whose ending share
    balance is positive; it is zero when no post-opening snapshot exists.
    """

    if not snapshots:
        raise ValueError("At least one portfolio snapshot is required.")
    if (
        not isinstance(annualization_factor, int)
        or isinstance(annualization_factor, bool)
        or annualization_factor <= 0
    ):
        raise ValueError("annualization_factor must be a positive integer.")
    if not math.isfinite(float(risk_free_rate_annual)) or risk_free_rate_annual <= -1:
        raise ValueError("risk_free_rate_annual must be finite and greater than -1.")

    _validate_snapshots(snapshots)
    daily_return_records = calculate_daily_returns(snapshots)
    drawdown_records = calculate_drawdowns(snapshots)
    returns = [record["return"] for record in daily_return_records]

    volatility: float | None = None
    sharpe_ratio: float | None = None
    if len(returns) >= 2:
        daily_standard_deviation = statistics.stdev(returns)
        volatility = daily_standard_deviation * math.sqrt(annualization_factor)
        if daily_standard_deviation != 0:
            daily_risk_free_rate = (1.0 + float(risk_free_rate_annual)) ** (1.0 / annualization_factor) - 1.0
            sharpe_ratio = (
                math.sqrt(annualization_factor)
                * (statistics.fmean(returns) - daily_risk_free_rate)
                / daily_standard_deviation
            )

    episode_pnls = calculate_closed_episode_pnls(trades)
    gains = [pnl for pnl in episode_pnls if pnl > 0]
    losses = [pnl for pnl in episode_pnls if pnl < 0]
    win_rate = (
        len(gains) / len(episode_pnls)
        if episode_pnls
        else None
    )
    average_gain = _money(sum(gains, Decimal("0")) / len(gains)) if gains else None
    average_loss = _money(sum(losses, Decimal("0")) / len(losses)) if losses else None

    fill_notional = sum(
        (_money(Decimal(trade.quantity) * trade.execution_price) for trade in trades),
        Decimal("0"),
    )
    mean_equity = sum((snapshot.total_equity for snapshot in snapshots), Decimal("0")) / len(snapshots)
    if mean_equity <= 0:
        raise ValueError("Mean snapshot equity must be strictly positive.")

    fees_paid = _money(sum((trade.fee for trade in trades), Decimal("0")))
    post_opening_snapshots = snapshots[1:]
    time_in_market = (
        sum(snapshot.shares > 0 for snapshot in post_opening_snapshots) / len(post_opening_snapshots)
        if post_opening_snapshots
        else 0.0
    )

    initial_equity = snapshots[0].total_equity
    ending_equity = snapshots[-1].total_equity
    total_return = (ending_equity / initial_equity) - Decimal("1")

    return {
        "initial_equity": _money_string(initial_equity),
        "ending_equity": _money_string(ending_equity),
        "total_return": float(total_return),
        "daily_returns": daily_return_records,
        "maximum_drawdown": max((record["drawdown"] for record in drawdown_records), default=0.0),
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "number_of_trades": len(trades),
        "win_rate": win_rate,
        "average_gain": _money_string(average_gain) if average_gain is not None else None,
        "average_loss": _money_string(average_loss) if average_loss is not None else None,
        "turnover": float(fill_notional / mean_equity),
        "fees_paid": _money_string(fees_paid),
        "time_in_market": time_in_market,
    }
