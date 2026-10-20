"""Validated domain records for PAISA's long-only trading simulator.

The trading layer deliberately uses :class:`~decimal.Decimal` for every price
and PKR amount.  Boundary records are immutable; the only mutable object in the
package is the portfolio ledger in :mod:`paisa.trading.portfolio`.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"
MONEY_QUANTUM = Decimal("0.01")
BPS_DENOMINATOR = Decimal("10000")
PROBABILITY_TOLERANCE = Decimal("0.000001")

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def decimal_value(value: Any, *, field_name: str = "value") -> Decimal:
    """Return a finite Decimal without accepting booleans or binary artefacts."""

    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} must be a finite decimal number")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite decimal number") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def money(value: Any, *, field_name: str = "money") -> Decimal:
    """Quantize a PKR amount or price to paisa using ROUND_HALF_UP."""

    return decimal_value(value, field_name=field_name).quantize(
        MONEY_QUANTUM, rounding=ROUND_HALF_UP
    )


def ratio_value(
    value: Any,
    *,
    field_name: str = "ratio",
    minimum: Decimal | None = None,
    maximum: Decimal | None = None,
    minimum_inclusive: bool = True,
    maximum_inclusive: bool = True,
) -> Decimal:
    """Parse a finite ratio and optionally validate its range."""

    result = decimal_value(value, field_name=field_name)
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ValueError(f"{field_name} must be {operator} {minimum}")
    if maximum is not None:
        invalid = result > maximum if maximum_inclusive else result >= maximum
        if invalid:
            operator = "<=" if maximum_inclusive else "<"
            raise ValueError(f"{field_name} must be {operator} {maximum}")
    return result


def fixed_money(value: Decimal) -> str:
    """Serialize a Decimal PKR value with exactly two fractional digits."""

    return format(money(value), ".2f")


def stable_id(prefix: str, *parts: object) -> str:
    """Create a deterministic, filesystem-safe identifier from stable inputs."""

    prefix = _validated_id(prefix, "prefix")
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()}"


def _validated_schema(value: object) -> str:
    schema = str(value)
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema!r}; expected {SCHEMA_VERSION!r}"
        )
    return schema


def _validated_id(value: object, field_name: str) -> str:
    identifier = str(value).strip()
    if not identifier or identifier in {".", ".."} or not _SAFE_ID.fullmatch(identifier):
        raise ValueError(
            f"{field_name} must contain only letters, digits, '.', '_', and '-' "
            "and must start with a letter or digit"
        )
    return identifier


def _validated_ticker(value: object, field_name: str = "ticker") -> str:
    ticker = str(value).strip().upper()
    if not ticker or not re.fullmatch(r"[A-Z0-9._-]+", ticker):
        raise ValueError(f"{field_name} must be a non-empty uppercase symbol")
    return ticker


def _date_value(value: object, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO YYYY-MM-DD date") from exc


def _optional_class(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Decimal) and value.is_nan():
        return None
    if isinstance(value, bool):
        raise ValueError("actual_class must be 0, 1, or null")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual_class must be 0, 1, or null") from exc
    if parsed not in (0, 1) or str(value).strip() not in {"0", "1", "0.0", "1.0"}:
        raise ValueError("actual_class must be 0, 1, or null")
    return parsed


def _required_class(value: object, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be 0 or 1")
    try:
        parsed = decimal_value(value, field_name=field_name)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be 0 or 1") from exc
    if parsed not in (Decimal("0"), Decimal("1")):
        raise ValueError(f"{field_name} must be 0 or 1")
    return int(parsed)


def _integer_value(value: object, field_name: str) -> int:
    """Parse a canonical integer without accepting booleans or decimal text."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
        raise ValueError(f"{field_name} must be an integer")
    return int(text)


def _action_value(value: object) -> "Action":
    if isinstance(value, Action):
        return value
    try:
        return Action(str(value).strip().upper())
    except ValueError as exc:
        raise ValueError("action must be BUY, SELL, or HOLD") from exc


def _utc_timestamp(value: object) -> str:
    timestamp = str(value).strip()
    if not timestamp.endswith("Z"):
        raise ValueError("created_at must be an ISO 8601 UTC timestamp ending in 'Z'")
    try:
        parsed = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("created_at must be a valid ISO 8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at must be in UTC")
    return timestamp


class Action(str, Enum):
    """A policy decision.  HOLD signals never create trade records."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """A versioned prediction at the model/trading boundary."""

    model_id: str
    ticker: str
    feature_date: date
    target_date: date
    predicted_class: int
    probability_up: Decimal
    probability_down: Decimal
    actual_class: int | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validated_schema(self.schema_version))
        object.__setattr__(self, "model_id", _validated_id(self.model_id, "model_id"))
        object.__setattr__(self, "ticker", _validated_ticker(self.ticker))
        object.__setattr__(self, "feature_date", _date_value(self.feature_date, "feature_date"))
        object.__setattr__(self, "target_date", _date_value(self.target_date, "target_date"))
        if self.target_date <= self.feature_date:
            raise ValueError("target_date must be later than feature_date")
        object.__setattr__(
            self, "predicted_class", _required_class(self.predicted_class, "predicted_class")
        )
        up = ratio_value(
            self.probability_up,
            field_name="probability_up",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        down = ratio_value(
            self.probability_down,
            field_name="probability_down",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if abs(up + down - Decimal("1")) > PROBABILITY_TOLERANCE:
            raise ValueError("probability_up and probability_down must sum to 1 within 1e-6")
        expected_class = 1 if up > down else 0
        if self.predicted_class != expected_class:
            raise ValueError(
                "predicted_class must be 1 when probability_up > probability_down and 0 otherwise"
            )
        object.__setattr__(self, "probability_up", up)
        object.__setattr__(self, "probability_down", down)
        object.__setattr__(self, "actual_class", _optional_class(self.actual_class))

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PredictionRecord":
        return cls(
            schema_version=row.get("schema_version", SCHEMA_VERSION),
            model_id=row["model_id"],
            ticker=row["ticker"],
            feature_date=row["feature_date"],
            target_date=row["target_date"],
            predicted_class=row["predicted_class"],
            probability_up=row["probability_up"],
            probability_down=row["probability_down"],
            actual_class=row.get("actual_class"),
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "ticker": self.ticker,
            "feature_date": self.feature_date.isoformat(),
            "target_date": self.target_date.isoformat(),
            "predicted_class": self.predicted_class,
            "probability_up": float(self.probability_up),
            "probability_down": float(self.probability_down),
            "actual_class": self.actual_class,
        }

@dataclass(frozen=True, slots=True)
class DecisionInput:
    """Leakage-safe policy input; intentionally excludes labels and prices."""

    model_id: str
    ticker: str
    feature_date: date
    target_date: date
    probability_up: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _validated_id(self.model_id, "model_id"))
        object.__setattr__(self, "ticker", _validated_ticker(self.ticker))
        object.__setattr__(self, "feature_date", _date_value(self.feature_date, "feature_date"))
        object.__setattr__(self, "target_date", _date_value(self.target_date, "target_date"))
        if self.target_date <= self.feature_date:
            raise ValueError("target_date must be later than feature_date")
        object.__setattr__(
            self,
            "probability_up",
            ratio_value(
                self.probability_up,
                field_name="probability_up",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            ),
        )

    @classmethod
    def from_prediction(cls, prediction: PredictionRecord) -> "DecisionInput":
        return cls(
            model_id=prediction.model_id,
            ticker=prediction.ticker,
            feature_date=prediction.feature_date,
            target_date=prediction.target_date,
            probability_up=prediction.probability_up,
        )


@dataclass(frozen=True, slots=True)
class MarketBar:
    """The execution and valuation fields for one observed trading session."""

    ticker: str
    date: date
    open: Decimal
    close: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticker", _validated_ticker(self.ticker))
        object.__setattr__(self, "date", _date_value(self.date, "date"))
        opening = money(self.open, field_name="open")
        closing = money(self.close, field_name="close")
        if opening <= 0 or closing <= 0:
            raise ValueError("market open and close prices must be positive")
        object.__setattr__(self, "open", opening)
        object.__setattr__(self, "close", closing)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MarketBar":
        return cls(ticker=row["ticker"], date=row["date"], open=row["open"], close=row["close"])

    from_dict = from_mapping


@dataclass(frozen=True, slots=True)
class PortfolioView:
    """Read-only portfolio state supplied to a decision policy."""

    portfolio_id: str
    cash: Decimal
    shares: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "portfolio_id", _validated_id(self.portfolio_id, "portfolio_id"))
        cash = money(self.cash, field_name="cash")
        if cash < 0:
            raise ValueError("cash cannot be negative")
        if isinstance(self.shares, bool) or not isinstance(self.shares, int) or self.shares < 0:
            raise ValueError("shares must be a non-negative whole number")
        object.__setattr__(self, "cash", cash)


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """A post-close policy decision scheduled for its target session."""

    signal_id: str
    portfolio_id: str
    model_id: str
    ticker: str
    signal_date: date
    target_date: date
    action: Action
    probability_up: Decimal
    target_allocation: Decimal | None
    reason: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validated_schema(self.schema_version))
        object.__setattr__(self, "signal_id", _validated_id(self.signal_id, "signal_id"))
        object.__setattr__(self, "portfolio_id", _validated_id(self.portfolio_id, "portfolio_id"))
        object.__setattr__(self, "model_id", _validated_id(self.model_id, "model_id"))
        object.__setattr__(self, "ticker", _validated_ticker(self.ticker))
        object.__setattr__(self, "signal_date", _date_value(self.signal_date, "signal_date"))
        object.__setattr__(self, "target_date", _date_value(self.target_date, "target_date"))
        if self.target_date <= self.signal_date:
            raise ValueError("target_date must be later than signal_date")
        action = _action_value(self.action)
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "probability_up",
            ratio_value(
                self.probability_up,
                field_name="probability_up",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            ),
        )
        target = self.target_allocation
        if target is not None:
            target = ratio_value(
                target,
                field_name="target_allocation",
                minimum=Decimal("0"),
                maximum=Decimal("1"),
            )
        if action is Action.BUY and (target is None or target <= 0):
            raise ValueError("BUY requires a positive target_allocation")
        if action is Action.SELL and target != Decimal("0"):
            raise ValueError("SELL requires target_allocation=0")
        if action is Action.HOLD and target is not None:
            raise ValueError("HOLD requires target_allocation=null")
        object.__setattr__(self, "target_allocation", target)
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("reason must not be empty")
        object.__setattr__(self, "reason", reason)

    @property
    def execution_date(self) -> date:
        """Alias for the scheduled target session used by the executor."""

        return self.target_date

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "SignalRecord":
        target_allocation = row["target_allocation"]
        if target_allocation is None or (
            isinstance(target_allocation, str) and not target_allocation.strip()
        ):
            target_allocation = None
        return cls(
            schema_version=row["schema_version"],
            signal_id=row["signal_id"],
            portfolio_id=row["portfolio_id"],
            model_id=row["model_id"],
            ticker=row["ticker"],
            signal_date=row["signal_date"],
            target_date=row["target_date"],
            action=row["action"],
            probability_up=row["probability_up"],
            target_allocation=target_allocation,
            reason=row["reason"],
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "signal_id": self.signal_id,
            "portfolio_id": self.portfolio_id,
            "model_id": self.model_id,
            "ticker": self.ticker,
            "signal_date": self.signal_date.isoformat(),
            "target_date": self.target_date.isoformat(),
            "action": self.action.value,
            "probability_up": float(self.probability_up),
            "target_allocation": (
                None if self.target_allocation is None else float(self.target_allocation)
            ),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TradeRecord:
    """An immutable executed BUY or SELL record."""

    trade_id: str
    portfolio_id: str
    signal_date: date
    execution_date: date
    action: Action
    quantity: int
    reference_price: Decimal
    execution_price: Decimal
    fee: Decimal
    cash_after_trade: Decimal
    shares_after_trade: int
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validated_schema(self.schema_version))
        object.__setattr__(self, "trade_id", _validated_id(self.trade_id, "trade_id"))
        object.__setattr__(self, "portfolio_id", _validated_id(self.portfolio_id, "portfolio_id"))
        object.__setattr__(self, "signal_date", _date_value(self.signal_date, "signal_date"))
        object.__setattr__(self, "execution_date", _date_value(self.execution_date, "execution_date"))
        if self.execution_date <= self.signal_date:
            raise ValueError("execution_date must be later than signal_date")
        action = _action_value(self.action)
        if action is Action.HOLD:
            raise ValueError("HOLD is not an executed trade")
        object.__setattr__(self, "action", action)
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("quantity must be a positive whole number")
        reference = money(self.reference_price, field_name="reference_price")
        execution = money(self.execution_price, field_name="execution_price")
        fee = money(self.fee, field_name="fee")
        cash = money(self.cash_after_trade, field_name="cash_after_trade")
        if reference <= 0 or execution <= 0:
            raise ValueError("reference_price and execution_price must be positive")
        if fee < 0 or cash < 0:
            raise ValueError("fee and cash_after_trade cannot be negative")
        if (
            isinstance(self.shares_after_trade, bool)
            or not isinstance(self.shares_after_trade, int)
            or self.shares_after_trade < 0
        ):
            raise ValueError("shares_after_trade must be a non-negative whole number")
        object.__setattr__(self, "reference_price", reference)
        object.__setattr__(self, "execution_price", execution)
        object.__setattr__(self, "fee", fee)
        object.__setattr__(self, "cash_after_trade", cash)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TradeRecord":
        return cls(
            schema_version=row["schema_version"],
            trade_id=row["trade_id"],
            portfolio_id=row["portfolio_id"],
            signal_date=row["signal_date"],
            execution_date=row["execution_date"],
            action=row["action"],
            quantity=_integer_value(row["quantity"], "quantity"),
            reference_price=row["reference_price"],
            execution_price=row["execution_price"],
            fee=row["fee"],
            cash_after_trade=row["cash_after_trade"],
            shares_after_trade=_integer_value(
                row["shares_after_trade"], "shares_after_trade"
            ),
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trade_id": self.trade_id,
            "portfolio_id": self.portfolio_id,
            "signal_date": self.signal_date.isoformat(),
            "execution_date": self.execution_date.isoformat(),
            "action": self.action.value,
            "quantity": self.quantity,
            "reference_price": fixed_money(self.reference_price),
            "execution_price": fixed_money(self.execution_price),
            "fee": fixed_money(self.fee),
            "cash_after_trade": fixed_money(self.cash_after_trade),
            "shares_after_trade": self.shares_after_trade,
        }


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    """A reconciled ledger state after execution and close valuation."""

    portfolio_id: str
    date: date
    cash: Decimal
    shares: int
    market_price: Decimal
    position_value: Decimal
    total_equity: Decimal
    realized_profit_loss: Decimal
    unrealized_profit_loss: Decimal
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validated_schema(self.schema_version))
        object.__setattr__(self, "portfolio_id", _validated_id(self.portfolio_id, "portfolio_id"))
        object.__setattr__(self, "date", _date_value(self.date, "date"))
        if isinstance(self.shares, bool) or not isinstance(self.shares, int) or self.shares < 0:
            raise ValueError("shares must be a non-negative whole number")
        cash = money(self.cash, field_name="cash")
        market = money(self.market_price, field_name="market_price")
        position = money(self.position_value, field_name="position_value")
        equity = money(self.total_equity, field_name="total_equity")
        realized = money(self.realized_profit_loss, field_name="realized_profit_loss")
        unrealized = money(self.unrealized_profit_loss, field_name="unrealized_profit_loss")
        if cash < 0 or market <= 0 or position < 0 or equity < 0:
            raise ValueError("cash/position/equity must be non-negative and market_price positive")
        expected_position = money(Decimal(self.shares) * market)
        if position != expected_position:
            raise ValueError("position_value must equal shares * market_price")
        if equity != money(cash + position):
            raise ValueError("total_equity must equal cash + position_value")
        if self.shares == 0 and unrealized != Decimal("0.00"):
            raise ValueError("unrealized_profit_loss must be zero while flat")
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "market_price", market)
        object.__setattr__(self, "position_value", position)
        object.__setattr__(self, "total_equity", equity)
        object.__setattr__(self, "realized_profit_loss", realized)
        object.__setattr__(self, "unrealized_profit_loss", unrealized)

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "PortfolioSnapshot":
        return cls(
            schema_version=row["schema_version"],
            portfolio_id=row["portfolio_id"],
            date=row["date"],
            cash=row["cash"],
            shares=_integer_value(row["shares"], "shares"),
            market_price=row["market_price"],
            position_value=row["position_value"],
            total_equity=row["total_equity"],
            realized_profit_loss=row["realized_profit_loss"],
            unrealized_profit_loss=row["unrealized_profit_loss"],
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "portfolio_id": self.portfolio_id,
            "date": self.date.isoformat(),
            "cash": fixed_money(self.cash),
            "shares": self.shares,
            "market_price": fixed_money(self.market_price),
            "position_value": fixed_money(self.position_value),
            "total_equity": fixed_money(self.total_equity),
            "realized_profit_loss": fixed_money(self.realized_profit_loss),
            "unrealized_profit_loss": fixed_money(self.unrealized_profit_loss),
        }


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    buy_probability_up_gte: Decimal = Decimal("0.60")
    sell_probability_up_lte: Decimal = Decimal("0.40")

    def __post_init__(self) -> None:
        buy = ratio_value(
            self.buy_probability_up_gte,
            field_name="buy_probability_up_gte",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        sell = ratio_value(
            self.sell_probability_up_lte,
            field_name="sell_probability_up_lte",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
        )
        if sell >= buy:
            raise ValueError("sell threshold must be strictly below buy threshold")
        object.__setattr__(self, "buy_probability_up_gte", buy)
        object.__setattr__(self, "sell_probability_up_lte", sell)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ThresholdConfig":
        return cls(
            buy_probability_up_gte=value["buy_probability_up_gte"],
            sell_probability_up_lte=value["sell_probability_up_lte"],
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, float]:
        return {
            "buy_probability_up_gte": float(self.buy_probability_up_gte),
            "sell_probability_up_lte": float(self.sell_probability_up_lte),
        }


@dataclass(frozen=True, slots=True)
class TransactionCostConfig:
    type: str = "proportional_bps"
    value: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if self.type != "proportional_bps":
            raise ValueError("transaction_cost.type must be 'proportional_bps'")
        value = decimal_value(self.value, field_name="transaction_cost.value")
        if value < 0:
            raise ValueError("transaction_cost.value cannot be negative")
        object.__setattr__(self, "value", value)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TransactionCostConfig":
        return cls(type=str(value["type"]), value=value["value"])

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "value": float(self.value)}


@dataclass(frozen=True, slots=True)
class EvaluationDates:
    feature_start: date
    feature_end: date
    target_start: date
    target_end: date

    def __post_init__(self) -> None:
        for field_name in ("feature_start", "feature_end", "target_start", "target_end"):
            object.__setattr__(self, field_name, _date_value(getattr(self, field_name), field_name))
        if self.feature_start > self.feature_end:
            raise ValueError("feature_start must be on or before feature_end")
        if self.target_start > self.target_end:
            raise ValueError("target_start must be on or before target_end")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationDates":
        return cls(
            feature_start=value["feature_start"],
            feature_end=value["feature_end"],
            target_start=value["target_start"],
            target_end=value["target_end"],
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, str]:
        return {
            "feature_start": self.feature_start.isoformat(),
            "feature_end": self.feature_end.isoformat(),
            "target_start": self.target_start.isoformat(),
            "target_end": self.target_end.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ExecutionPolicyConfig:
    signal_timing: str = "after_close"
    execution_session: str = "next_trading_session"
    reference_price_field: str = "open"
    slippage_bps: Decimal = Decimal("0")
    missing_execution_price: str = "fail"
    allocation_mode: str = "state_transition_to_maximum"
    whole_share_rounding: str = "floor"
    valuation_price_field: str = "close"
    missing_valuation_price: str = "fail"
    ending_position: str = "mark_to_market"

    def __post_init__(self) -> None:
        expected = {
            "signal_timing": "after_close",
            "execution_session": "next_trading_session",
            "reference_price_field": "open",
            "missing_execution_price": "fail",
            "allocation_mode": "state_transition_to_maximum",
            "whole_share_rounding": "floor",
            "valuation_price_field": "close",
            "missing_valuation_price": "fail",
            "ending_position": "mark_to_market",
        }
        for field_name, expected_value in expected.items():
            if getattr(self, field_name) != expected_value:
                raise ValueError(f"{field_name} must be {expected_value!r} in schema 1.0")
        slippage = ratio_value(
            self.slippage_bps,
            field_name="slippage_bps",
            minimum=Decimal("0"),
            maximum=BPS_DENOMINATOR,
            maximum_inclusive=False,
        )
        object.__setattr__(self, "slippage_bps", slippage)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExecutionPolicyConfig":
        return cls(
            signal_timing=str(value["signal_timing"]),
            execution_session=str(value["execution_session"]),
            reference_price_field=str(value["reference_price_field"]),
            slippage_bps=value["slippage_bps"],
            missing_execution_price=str(value["missing_execution_price"]),
            allocation_mode=str(value["allocation_mode"]),
            whole_share_rounding=str(value["whole_share_rounding"]),
            valuation_price_field=str(value["valuation_price_field"]),
            missing_valuation_price=str(value["missing_valuation_price"]),
            ending_position=str(value["ending_position"]),
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_timing": self.signal_timing,
            "execution_session": self.execution_session,
            "reference_price_field": self.reference_price_field,
            "slippage_bps": float(self.slippage_bps),
            "missing_execution_price": self.missing_execution_price,
            "allocation_mode": self.allocation_mode,
            "whole_share_rounding": self.whole_share_rounding,
            "valuation_price_field": self.valuation_price_field,
            "missing_valuation_price": self.missing_valuation_price,
            "ending_position": self.ending_position,
        }


@dataclass(frozen=True, slots=True)
class MetricAssumptions:
    annualization_factor: int = 252
    risk_free_rate_annual: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        if (
            isinstance(self.annualization_factor, bool)
            or not isinstance(self.annualization_factor, int)
            or self.annualization_factor <= 0
        ):
            raise ValueError("annualization_factor must be a positive integer")
        risk_free = decimal_value(self.risk_free_rate_annual, field_name="risk_free_rate_annual")
        if risk_free <= Decimal("-1"):
            raise ValueError("risk_free_rate_annual must be greater than -1")
        object.__setattr__(self, "risk_free_rate_annual", risk_free)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MetricAssumptions":
        return cls(
            annualization_factor=value["annualization_factor"],
            risk_free_rate_annual=value["risk_free_rate_annual"],
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "annualization_factor": self.annualization_factor,
            "risk_free_rate_annual": float(self.risk_free_rate_annual),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkPortfolioIds:
    cash_only: str
    buy_and_hold: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "cash_only", _validated_id(self.cash_only, "cash_only portfolio_id"))
        object.__setattr__(
            self, "buy_and_hold", _validated_id(self.buy_and_hold, "buy_and_hold portfolio_id")
        )
        if self.cash_only == self.buy_and_hold:
            raise ValueError("cash-only and buy-and-hold portfolio IDs must differ")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkPortfolioIds":
        return cls(cash_only=value["cash_only"], buy_and_hold=value["buy_and_hold"])

    from_dict = from_mapping

    def to_dict(self) -> dict[str, str]:
        return {"cash_only": self.cash_only, "buy_and_hold": self.buy_and_hold}


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentConfig:
    """Canonical immutable experiment configuration from contract schema 1.0."""

    experiment_id: str
    model_id: str
    portfolio_id: str
    benchmark_portfolio_ids: BenchmarkPortfolioIds
    evaluation_dates: EvaluationDates
    created_at: str
    symbol: str = "OGDC"
    initial_cash: Decimal = Decimal("100000.00")
    thresholds: ThresholdConfig = field(default_factory=ThresholdConfig)
    maximum_allocation: Decimal = Decimal("1")
    transaction_cost: TransactionCostConfig = field(default_factory=TransactionCostConfig)
    execution_policy: ExecutionPolicyConfig = field(default_factory=ExecutionPolicyConfig)
    metric_assumptions: MetricAssumptions = field(default_factory=MetricAssumptions)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _validated_schema(self.schema_version))
        object.__setattr__(self, "experiment_id", _validated_id(self.experiment_id, "experiment_id"))
        object.__setattr__(self, "model_id", _validated_id(self.model_id, "model_id"))
        object.__setattr__(self, "portfolio_id", _validated_id(self.portfolio_id, "portfolio_id"))
        object.__setattr__(self, "symbol", _validated_ticker(self.symbol, "symbol"))
        initial_cash = money(self.initial_cash, field_name="initial_cash")
        if initial_cash <= 0:
            raise ValueError("initial_cash must be strictly positive")
        object.__setattr__(self, "initial_cash", initial_cash)
        maximum = ratio_value(
            self.maximum_allocation,
            field_name="maximum_allocation",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
            minimum_inclusive=False,
        )
        object.__setattr__(self, "maximum_allocation", maximum)
        object.__setattr__(self, "created_at", _utc_timestamp(self.created_at))
        if not isinstance(self.thresholds, ThresholdConfig):
            raise TypeError("thresholds must be a ThresholdConfig")
        if not isinstance(self.benchmark_portfolio_ids, BenchmarkPortfolioIds):
            raise TypeError("benchmark_portfolio_ids must be BenchmarkPortfolioIds")
        if not isinstance(self.evaluation_dates, EvaluationDates):
            raise TypeError("evaluation_dates must be EvaluationDates")
        if not isinstance(self.transaction_cost, TransactionCostConfig):
            raise TypeError("transaction_cost must be TransactionCostConfig")
        if not isinstance(self.execution_policy, ExecutionPolicyConfig):
            raise TypeError("execution_policy must be ExecutionPolicyConfig")
        if not isinstance(self.metric_assumptions, MetricAssumptions):
            raise TypeError("metric_assumptions must be MetricAssumptions")
        ids = {
            self.portfolio_id,
            self.benchmark_portfolio_ids.cash_only,
            self.benchmark_portfolio_ids.buy_and_hold,
        }
        if len(ids) != 3:
            raise ValueError("strategy and benchmark portfolio IDs must all be distinct")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExperimentConfig":
        return cls(
            schema_version=value["schema_version"],
            experiment_id=value["experiment_id"],
            model_id=value["model_id"],
            portfolio_id=value["portfolio_id"],
            benchmark_portfolio_ids=BenchmarkPortfolioIds.from_mapping(
                value["benchmark_portfolio_ids"]
            ),
            symbol=value["symbol"],
            initial_cash=value.get("initial_cash", "100000.00"),
            thresholds=ThresholdConfig.from_mapping(value["thresholds"]),
            maximum_allocation=value["maximum_allocation"],
            transaction_cost=TransactionCostConfig.from_mapping(
                value["transaction_cost"]
            ),
            evaluation_dates=EvaluationDates.from_mapping(value["evaluation_dates"]),
            execution_policy=ExecutionPolicyConfig.from_mapping(
                value["execution_policy"]
            ),
            metric_assumptions=MetricAssumptions.from_mapping(
                value["metric_assumptions"]
            ),
            created_at=value["created_at"],
        )

    from_dict = from_mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "model_id": self.model_id,
            "portfolio_id": self.portfolio_id,
            "benchmark_portfolio_ids": self.benchmark_portfolio_ids.to_dict(),
            "symbol": self.symbol,
            "initial_cash": fixed_money(self.initial_cash),
            "thresholds": self.thresholds.to_dict(),
            "maximum_allocation": float(self.maximum_allocation),
            "transaction_cost": self.transaction_cost.to_dict(),
            "evaluation_dates": self.evaluation_dates.to_dict(),
            "execution_policy": self.execution_policy.to_dict(),
            "metric_assumptions": self.metric_assumptions.to_dict(),
            "created_at": self.created_at,
        }
