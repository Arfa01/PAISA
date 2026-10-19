"""Leakage-safe conversion of model probabilities into portfolio signals."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from paisa.trading.models import (
    Action,
    DecisionInput,
    ExperimentConfig,
    PortfolioView,
    SignalRecord,
    ThresholdConfig,
    ratio_value,
    stable_id,
)


@dataclass(frozen=True, slots=True)
class ConfidenceThresholdPolicy:
    """Classify an up probability with an inclusive buy/sell neutral band.

    The input type intentionally has no actual label or price fields.  Equality
    at either configured threshold takes the corresponding BUY/SELL branch.
    """

    thresholds: ThresholdConfig

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, ThresholdConfig):
            raise TypeError("thresholds must be a ThresholdConfig")

    def classify(self, decision: DecisionInput) -> Action:
        if not isinstance(decision, DecisionInput):
            raise TypeError("decision must be a leakage-safe DecisionInput")
        if decision.probability_up >= self.thresholds.buy_probability_up_gte:
            return Action.BUY
        if decision.probability_up <= self.thresholds.sell_probability_up_lte:
            return Action.SELL
        return Action.HOLD


@dataclass(frozen=True, slots=True)
class TargetAllocationPolicy:
    """Turn a confidence action into the schema-1.0 position transition.

    Schema 1.0 supports full position episodes: BUY transitions a flat ledger
    toward ``maximum_allocation``; SELL transitions an invested ledger to zero;
    and it never pyramids or creates a redundant sell while already in cash.
    """

    confidence_policy: ConfidenceThresholdPolicy
    maximum_allocation: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.confidence_policy, ConfidenceThresholdPolicy):
            raise TypeError("confidence_policy must be a ConfidenceThresholdPolicy")
        maximum = ratio_value(
            self.maximum_allocation,
            field_name="maximum_allocation",
            minimum=Decimal("0"),
            maximum=Decimal("1"),
            minimum_inclusive=False,
        )
        object.__setattr__(self, "maximum_allocation", maximum)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "TargetAllocationPolicy":
        """Construct the policy bound by an immutable experiment config."""

        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        return cls(
            confidence_policy=ConfidenceThresholdPolicy(config.thresholds),
            maximum_allocation=config.maximum_allocation,
        )
    def decide(self, decision: DecisionInput, portfolio: PortfolioView) -> SignalRecord:
        """Return one deterministic signal without consulting prices or labels."""

        if not isinstance(decision, DecisionInput):
            raise TypeError("decision must be a leakage-safe DecisionInput")
        if not isinstance(portfolio, PortfolioView):
            raise TypeError("portfolio must be a PortfolioView")

        confidence_action = self.confidence_policy.classify(decision)
        if confidence_action is Action.BUY and portfolio.shares == 0:
            action = Action.BUY
            target: Decimal | None = self.maximum_allocation
            reason = "buy_threshold_reached"
        elif confidence_action is Action.BUY:
            action = Action.HOLD
            target = None
            reason = "already_in_position"
        elif confidence_action is Action.SELL and portfolio.shares > 0:
            action = Action.SELL
            target = Decimal("0")
            reason = "sell_threshold_reached"
        elif confidence_action is Action.SELL:
            action = Action.HOLD
            target = None
            reason = "already_in_cash"
        else:
            action = Action.HOLD
            target = None
            reason = "inside_neutral_band"

        signal_id = stable_id(
            "signal",
            portfolio.portfolio_id,
            decision.model_id,
            decision.ticker,
            decision.feature_date.isoformat(),
            decision.target_date.isoformat(),
            action.value,
        )
        return SignalRecord(
            signal_id=signal_id,
            portfolio_id=portfolio.portfolio_id,
            model_id=decision.model_id,
            ticker=decision.ticker,
            signal_date=decision.feature_date,
            target_date=decision.target_date,
            action=action,
            probability_up=decision.probability_up,
            target_allocation=target,
            reason=reason,
        )
