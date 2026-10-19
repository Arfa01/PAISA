"""Chronological, leakage-safe orchestration for one strategy portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

from paisa.trading.execution import execute_signal
from paisa.trading.models import (
    DecisionInput,
    ExperimentConfig,
    MarketBar,
    PortfolioSnapshot,
    PredictionRecord,
    SignalRecord,
    TradeRecord,
)
from paisa.trading.policy import TargetAllocationPolicy
from paisa.trading.portfolio import PortfolioLedger


@dataclass(frozen=True, slots=True)
class BacktestOutcome:
    """The deterministic in-memory outcome for a single strategy ledger."""

    predictions: tuple[PredictionRecord, ...]
    market_bars: tuple[MarketBar, ...]
    signals: tuple[SignalRecord, ...]
    trades: tuple[TradeRecord, ...]
    snapshots: tuple[PortfolioSnapshot, ...]


def evaluation_market_bars(
    config: ExperimentConfig, market_bars: Sequence[MarketBar]
) -> tuple[MarketBar, ...]:
    """Validate and return the exact observed evaluation-session sequence."""

    if not market_bars:
        raise ValueError("At least one market bar is required.")

    seen: set[tuple[str, date]] = set()
    selected: list[MarketBar] = []
    for bar in market_bars:
        if not isinstance(bar, MarketBar):
            raise TypeError("market_bars must contain MarketBar records")
        key = (bar.ticker, bar.date)
        if key in seen:
            raise ValueError(f"Duplicate market bar for {bar.ticker} on {bar.date}.")
        seen.add(key)
        if bar.ticker != config.symbol:
            raise ValueError(
                f"Market ticker {bar.ticker!r} does not match configured symbol "
                f"{config.symbol!r}."
            )
        if config.evaluation_dates.feature_start <= bar.date <= config.evaluation_dates.target_end:
            selected.append(bar)

    selected.sort(key=lambda item: item.date)
    if not selected:
        raise ValueError("No market bars fall inside the evaluation window.")
    if selected[0].date != config.evaluation_dates.feature_start:
        raise ValueError("A valuation bar is required on evaluation feature_start.")
    if selected[-1].date != config.evaluation_dates.target_end:
        raise ValueError("A valuation bar is required on evaluation target_end.")

    session_dates = [bar.date for bar in selected]
    if config.evaluation_dates.target_start not in session_dates:
        raise ValueError("target_start must be an observed trading session.")
    if config.evaluation_dates.feature_end not in session_dates:
        raise ValueError("feature_end must be an observed trading session.")

    feature_start_index = session_dates.index(config.evaluation_dates.feature_start)
    if feature_start_index + 1 >= len(session_dates):
        raise ValueError("The evaluation window must include a next trading session.")
    if session_dates[feature_start_index + 1] != config.evaluation_dates.target_start:
        raise ValueError("target_start must be the next observed session after feature_start.")

    feature_end_index = session_dates.index(config.evaluation_dates.feature_end)
    if feature_end_index + 1 >= len(session_dates):
        raise ValueError("feature_end must have an observed next trading session.")
    if session_dates[feature_end_index + 1] != config.evaluation_dates.target_end:
        raise ValueError("target_end must be the next observed session after feature_end.")
    return tuple(selected)


def validate_predictions(
    config: ExperimentConfig,
    predictions: Sequence[PredictionRecord],
    market_bars: Sequence[MarketBar],
) -> tuple[PredictionRecord, ...]:
    """Validate identity, uniqueness, windows, and next-session alignment."""

    if not predictions:
        raise ValueError("At least one prediction record is required.")

    sessions = {bar.date: index for index, bar in enumerate(market_bars)}
    unique_keys: set[tuple[str, str, date, date]] = set()
    feature_dates: set[date] = set()
    target_dates: set[date] = set()
    validated: list[PredictionRecord] = []

    for prediction in predictions:
        if not isinstance(prediction, PredictionRecord):
            raise TypeError("predictions must contain PredictionRecord records")
        if prediction.model_id != config.model_id:
            raise ValueError("Prediction model_id does not match the experiment configuration.")
        if prediction.ticker != config.symbol:
            raise ValueError("Prediction ticker does not match the experiment symbol.")

        key = (
            prediction.model_id,
            prediction.ticker,
            prediction.feature_date,
            prediction.target_date,
        )
        if key in unique_keys:
            raise ValueError(f"Duplicate prediction key for feature date {prediction.feature_date}.")
        if prediction.feature_date in feature_dates:
            raise ValueError(f"Duplicate prediction feature date {prediction.feature_date}.")
        if prediction.target_date in target_dates:
            raise ValueError(f"Duplicate prediction target date {prediction.target_date}.")
        unique_keys.add(key)
        feature_dates.add(prediction.feature_date)
        target_dates.add(prediction.target_date)

        dates = config.evaluation_dates
        if not dates.feature_start <= prediction.feature_date <= dates.feature_end:
            raise ValueError("Prediction feature_date falls outside the configured feature window.")
        if not dates.target_start <= prediction.target_date <= dates.target_end:
            raise ValueError("Prediction target_date falls outside the configured target window.")
        if prediction.feature_date not in sessions or prediction.target_date not in sessions:
            raise ValueError("Prediction dates must both be observed market sessions.")
        feature_index = sessions[prediction.feature_date]
        if feature_index + 1 >= len(market_bars):
            raise ValueError("Prediction feature_date has no next observed trading session.")
        expected_target = market_bars[feature_index + 1].date
        if prediction.target_date != expected_target:
            raise ValueError(
                "Prediction target_date must be the next observed trading session; "
                f"expected {expected_target}, got {prediction.target_date}."
            )
        validated.append(prediction)

    validated.sort(key=lambda item: (item.feature_date, item.target_date))
    return tuple(validated)


class BacktestEngine:
    """Run a single strategy without training models or consulting labels."""

    def __init__(self, config: ExperimentConfig):
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        self.config = config
        self.policy = TargetAllocationPolicy.from_config(config)

    def run(
        self,
        predictions: Sequence[PredictionRecord],
        market_bars: Sequence[MarketBar],
        *,
        ledger: PortfolioLedger | None = None,
    ) -> BacktestOutcome:
        """Run chronologically, executing each signal only at its target open."""

        bars = evaluation_market_bars(self.config, market_bars)
        records = validate_predictions(self.config, predictions, bars)
        strategy = ledger or PortfolioLedger(
            portfolio_id=self.config.portfolio_id,
            initial_cash=self.config.initial_cash,
        )
        if strategy.portfolio_id != self.config.portfolio_id:
            raise ValueError("Strategy ledger portfolio_id does not match the configuration.")
        if strategy.cash != self.config.initial_cash or strategy.shares != 0:
            raise ValueError("Strategy ledger must be a fresh flat ledger with configured cash.")

        by_target = {prediction.target_date: prediction for prediction in records}
        signals: list[SignalRecord] = []
        trades: list[TradeRecord] = []
        snapshots: list[PortfolioSnapshot] = []

        opening = bars[0]
        snapshots.append(strategy.snapshot(opening.date, opening.close))

        for bar in bars[1:]:
            prediction = by_target.get(bar.date)
            if prediction is not None:
                # DecisionInput deliberately excludes actual_class and all market prices.
                signal = self.policy.decide(
                    DecisionInput.from_prediction(prediction), strategy.view()
                )
                signals.append(signal)
                trade = execute_signal(signal, bar, strategy, self.config)
                if trade is not None:
                    trades.append(trade)
            snapshots.append(strategy.snapshot(bar.date, bar.close))

        return BacktestOutcome(
            predictions=records,
            market_bars=bars,
            signals=tuple(signals),
            trades=tuple(trades),
            snapshots=tuple(snapshots),
        )
