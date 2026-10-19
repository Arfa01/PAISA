"""Independent, invariant-enforcing portfolio ledger for PAISA simulations."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Iterable

from paisa.trading.models import (
    Action,
    PortfolioSnapshot,
    PortfolioView,
    SignalRecord,
    TradeRecord,
    money,
    stable_id,
)


def _as_date(value: date | str, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO YYYY-MM-DD date") from exc


class PortfolioLedger:
    """Own all mutable accounting state for exactly one simulated portfolio.

    A ledger supports the schema-1.0 full-position lifecycle only: a BUY may
    open a position from zero shares, and a SELL closes every held share.  The
    public ``trades`` and ``snapshots`` collections are immutable tuple views so
    state cannot leak between experiments.
    """

    def __init__(
        self,
        portfolio_id: str,
        initial_cash: Decimal | str | int = Decimal("100000.00"),
    ) -> None:
        opening_cash = money(initial_cash, field_name="initial_cash")
        if opening_cash <= 0:
            raise ValueError("initial_cash must be strictly positive")
        validated = PortfolioView(portfolio_id=portfolio_id, cash=opening_cash, shares=0)
        self.portfolio_id = validated.portfolio_id
        self.initial_cash = opening_cash
        self.cash = opening_cash
        self.shares = 0
        self.open_cost_basis = Decimal("0.00")
        self.realized_profit_loss = Decimal("0.00")
        self._trades: list[TradeRecord] = []
        self._snapshots: list[PortfolioSnapshot] = []
        self._closed_episode_pnls: list[Decimal] = []
        self._processed_signal_ids: set[str] = set()
        self._trade_dates: set[date] = set()

    @property
    def trades(self) -> tuple[TradeRecord, ...]:
        return tuple(self._trades)

    @property
    def snapshots(self) -> tuple[PortfolioSnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def closed_episode_pnls(self) -> tuple[Decimal, ...]:
        return tuple(self._closed_episode_pnls)

    @property
    def fees_paid(self) -> Decimal:
        return money(sum((trade.fee for trade in self._trades), Decimal("0")))

    def view(self) -> PortfolioView:
        """Return the price-free state that a decision policy is allowed to see."""

        return PortfolioView(portfolio_id=self.portfolio_id, cash=self.cash, shares=self.shares)

    def pre_trade_equity(self, reference_price: Decimal | str | int) -> Decimal:
        """Value current holdings at an execution-session reference price."""

        price = money(reference_price, field_name="reference_price")
        if price <= 0:
            raise ValueError("reference_price must be positive")
        return money(self.cash + Decimal(self.shares) * price)

    def apply_buy(
        self,
        signal: SignalRecord,
        *,
        execution_date: date | str,
        quantity: int,
        reference_price: Decimal | str | int,
        execution_price: Decimal | str | int,
        fee: Decimal | str | int,
        trade_id: str | None = None,
    ) -> TradeRecord:
        """Open a position and return the immutable reconciled trade record."""

        self._validate_signal_for_trade(signal, Action.BUY)
        if self.shares != 0:
            raise ValueError("schema 1.0 forbids adding to an open position")
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity <= 0:
            raise ValueError("quantity must be a positive whole number")
        execution_day = self._validate_new_execution_date(execution_date)
        reference = money(reference_price, field_name="reference_price")
        execution = money(execution_price, field_name="execution_price")
        charged_fee = money(fee, field_name="fee")
        if reference <= 0 or execution <= 0 or charged_fee < 0:
            raise ValueError("prices must be positive and fee non-negative")

        notional = money(Decimal(quantity) * execution)
        total_cost = money(notional + charged_fee)
        cash_after = money(self.cash - total_cost)
        if cash_after < 0:
            raise ValueError("buy would make cash negative")

        identifier = trade_id or stable_id(
            "trade",
            self.portfolio_id,
            signal.signal_id,
            execution_day.isoformat(),
            Action.BUY.value,
            quantity,
            execution,
            charged_fee,
        )
        record = TradeRecord(
            trade_id=identifier,
            portfolio_id=self.portfolio_id,
            signal_date=signal.signal_date,
            execution_date=execution_day,
            action=Action.BUY,
            quantity=quantity,
            reference_price=reference,
            execution_price=execution,
            fee=charged_fee,
            cash_after_trade=cash_after,
            shares_after_trade=quantity,
        )

        self.cash = cash_after
        self.shares = quantity
        self.open_cost_basis = total_cost
        self._append_trade(record, signal.signal_id)
        self._assert_invariants()
        return record

    def apply_sell_all(
        self,
        signal: SignalRecord,
        *,
        execution_date: date | str,
        reference_price: Decimal | str | int,
        execution_price: Decimal | str | int,
        fee: Decimal | str | int,
        trade_id: str | None = None,
    ) -> TradeRecord:
        """Close the complete open position and realize its fee-aware P/L."""

        self._validate_signal_for_trade(signal, Action.SELL)
        if self.shares <= 0:
            raise ValueError("portfolio cannot sell more shares than it owns")
        execution_day = self._validate_new_execution_date(execution_date)
        reference = money(reference_price, field_name="reference_price")
        execution = money(execution_price, field_name="execution_price")
        charged_fee = money(fee, field_name="fee")
        if reference <= 0 or execution <= 0 or charged_fee < 0:
            raise ValueError("prices must be positive and fee non-negative")

        quantity = self.shares
        notional = money(Decimal(quantity) * execution)
        net_proceeds = money(notional - charged_fee)
        cash_after = money(self.cash + net_proceeds)
        if cash_after < 0:
            raise ValueError("sell fee would make cash negative")
        episode_profit_loss = money(net_proceeds - self.open_cost_basis)
        realized_after = money(self.realized_profit_loss + episode_profit_loss)

        identifier = trade_id or stable_id(
            "trade",
            self.portfolio_id,
            signal.signal_id,
            execution_day.isoformat(),
            Action.SELL.value,
            quantity,
            execution,
            charged_fee,
        )
        record = TradeRecord(
            trade_id=identifier,
            portfolio_id=self.portfolio_id,
            signal_date=signal.signal_date,
            execution_date=execution_day,
            action=Action.SELL,
            quantity=quantity,
            reference_price=reference,
            execution_price=execution,
            fee=charged_fee,
            cash_after_trade=cash_after,
            shares_after_trade=0,
        )

        self.cash = cash_after
        self.shares = 0
        self.open_cost_basis = Decimal("0.00")
        self.realized_profit_loss = realized_after
        self._closed_episode_pnls.append(episode_profit_loss)
        self._append_trade(record, signal.signal_id)
        self._assert_invariants()
        return record

    def snapshot(
        self,
        session_date: date | str,
        market_price: Decimal | str | int,
    ) -> PortfolioSnapshot:
        """Append exactly one chronological end-of-session mark-to-market state."""

        session = _as_date(session_date, "session_date")
        if self._snapshots and session <= self._snapshots[-1].date:
            raise ValueError("portfolio snapshots must be strictly chronological and unique")
        price = money(market_price, field_name="market_price")
        if price <= 0:
            raise ValueError("market_price must be positive")
        position_value = money(Decimal(self.shares) * price)
        total_equity = money(self.cash + position_value)
        unrealized = (
            money(position_value - self.open_cost_basis)
            if self.shares > 0
            else Decimal("0.00")
        )
        record = PortfolioSnapshot(
            portfolio_id=self.portfolio_id,
            date=session,
            cash=self.cash,
            shares=self.shares,
            market_price=price,
            position_value=position_value,
            total_equity=total_equity,
            realized_profit_loss=self.realized_profit_loss,
            unrealized_profit_loss=unrealized,
        )
        self._snapshots.append(record)
        self._assert_invariants()
        return record

    mark_to_market = snapshot

    def _validate_signal_for_trade(self, signal: SignalRecord, expected: Action) -> None:
        if not isinstance(signal, SignalRecord):
            raise TypeError("signal must be a SignalRecord")
        if signal.portfolio_id != self.portfolio_id:
            raise ValueError("signal belongs to a different portfolio")
        if signal.action is not expected:
            raise ValueError(f"{expected.value} execution requires a {expected.value} signal")
        if signal.signal_id in self._processed_signal_ids:
            raise ValueError("signal has already been executed")

    def _validate_new_execution_date(self, value: date | str) -> date:
        execution_day = _as_date(value, "execution_date")
        if execution_day in self._trade_dates:
            raise ValueError("a portfolio may execute at most one trade per session")
        if self._snapshots and execution_day <= self._snapshots[-1].date:
            raise ValueError("trade execution must occur after the latest portfolio snapshot")
        return execution_day

    def _append_trade(self, trade: TradeRecord, signal_id: str) -> None:
        if any(existing.trade_id == trade.trade_id for existing in self._trades):
            raise ValueError(f"duplicate trade_id: {trade.trade_id}")
        self._trades.append(trade)
        self._processed_signal_ids.add(signal_id)
        self._trade_dates.add(trade.execution_date)

    def _assert_invariants(self) -> None:
        self.cash = money(self.cash, field_name="cash")
        self.open_cost_basis = money(self.open_cost_basis, field_name="open_cost_basis")
        self.realized_profit_loss = money(
            self.realized_profit_loss, field_name="realized_profit_loss"
        )
        if self.cash < 0:
            raise AssertionError("portfolio invariant violated: cash is negative")
        if isinstance(self.shares, bool) or not isinstance(self.shares, int) or self.shares < 0:
            raise AssertionError("portfolio invariant violated: shares are invalid")
        if self.shares == 0 and self.open_cost_basis != Decimal("0.00"):
            raise AssertionError("portfolio invariant violated: flat ledger has a cost basis")
        if self.shares > 0 and self.open_cost_basis <= 0:
            raise AssertionError("portfolio invariant violated: open position lacks a cost basis")


def assert_independent_ledgers(ledgers: Iterable[PortfolioLedger]) -> None:
    """Validate that a group of experiment/benchmark ledgers owns unique IDs."""

    portfolio_ids: set[str] = set()
    object_ids: set[int] = set()
    for ledger in ledgers:
        if not isinstance(ledger, PortfolioLedger):
            raise TypeError("all ledgers must be PortfolioLedger instances")
        if ledger.portfolio_id in portfolio_ids or id(ledger) in object_ids:
            raise ValueError("portfolio ledgers must be independent and have unique IDs")
        portfolio_ids.add(ledger.portfolio_id)
        object_ids.add(id(ledger))
