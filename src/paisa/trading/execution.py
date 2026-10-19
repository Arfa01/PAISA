"""Deterministic schema-1.0 execution for policy signals and market bars."""

from __future__ import annotations

from decimal import Decimal, ROUND_FLOOR

from paisa.trading.models import (
    Action,
    BPS_DENOMINATOR,
    ExperimentConfig,
    MarketBar,
    SignalRecord,
    TradeRecord,
    money,
)
from paisa.trading.portfolio import PortfolioLedger


class ExecutionEngine:
    """Execute a signal on its scheduled market bar using its experiment rules.

    A HOLD or a BUY that rounds to zero whole shares creates no trade. Bars
    must be the signal's target session; callers own the separate daily
    snapshot cadence.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        self.config = config

    def execute(
        self,
        signal: SignalRecord,
        bar: MarketBar,
        portfolio: PortfolioLedger,
    ) -> TradeRecord | None:
        """Apply an eligible signal to ``portfolio`` and return its trade."""

        if not isinstance(signal, SignalRecord):
            raise TypeError("signal must be a SignalRecord")
        if not isinstance(bar, MarketBar):
            raise TypeError("bar must be a MarketBar")
        if not isinstance(portfolio, PortfolioLedger):
            raise TypeError("portfolio must be a PortfolioLedger")
        allowed_portfolio_ids = {
            self.config.portfolio_id,
            self.config.benchmark_portfolio_ids.cash_only,
            self.config.benchmark_portfolio_ids.buy_and_hold,
        }
        if signal.portfolio_id != portfolio.portfolio_id:
            raise ValueError("signal belongs to a different portfolio ledger")
        if portfolio.portfolio_id not in allowed_portfolio_ids:
            raise ValueError("portfolio is not registered by the experiment config")
        if signal.ticker != self.config.symbol or bar.ticker != signal.ticker:
            raise ValueError("signal and market bar ticker must match experiment symbol")
        if bar.date != signal.execution_date:
            raise ValueError("market bar date must equal the signal execution_date")
        if signal.action is Action.HOLD:
            return None

        policy = self.config.execution_policy
        reference_price = bar.open
        slippage_fraction = policy.slippage_bps / BPS_DENOMINATOR
        if signal.action is Action.BUY:
            execution_price = money(
                reference_price * (Decimal("1") + slippage_fraction),
                field_name="execution_price",
            )
        else:
            execution_price = money(
                reference_price * (Decimal("1") - slippage_fraction),
                field_name="execution_price",
            )
        if execution_price <= 0:
            raise ValueError("slippage-adjusted execution price must be positive")

        fee_rate = self.config.transaction_cost.value / BPS_DENOMINATOR
        if signal.action is Action.BUY:
            equity = portfolio.pre_trade_equity(reference_price)
            cash_limit = (portfolio.cash / (execution_price * (Decimal("1") + fee_rate)))
            allocation_limit = (
                self.config.maximum_allocation * equity / execution_price
            )
            quantity = min(
                int(cash_limit.to_integral_value(rounding=ROUND_FLOOR)),
                int(allocation_limit.to_integral_value(rounding=ROUND_FLOOR)),
            )
            while quantity > 0:
                notional = money(Decimal(quantity) * execution_price)
                fee = money(Decimal(quantity) * execution_price * fee_rate)
                if (
                    money(notional + fee) <= portfolio.cash
                    and notional <= self.config.maximum_allocation * equity
                ):
                    break
                quantity -= 1
            if quantity == 0:
                return None
            return portfolio.apply_buy(
                signal,
                execution_date=bar.date,
                quantity=quantity,
                reference_price=reference_price,
                execution_price=execution_price,
                fee=fee,
            )

        fee = money(Decimal(portfolio.shares) * execution_price * fee_rate)
        return portfolio.apply_sell_all(
            signal,
            execution_date=bar.date,
            reference_price=reference_price,
            execution_price=execution_price,
            fee=fee,
        )


def execute_signal(
    signal: SignalRecord,
    bar: MarketBar,
    portfolio: PortfolioLedger,
    config: ExperimentConfig,
) -> TradeRecord | None:
    """Convenience function equivalent to ``ExecutionEngine(config).execute``."""

    return ExecutionEngine(config).execute(signal, bar, portfolio)
