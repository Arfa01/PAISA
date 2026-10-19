from datetime import date
from decimal import Decimal

import pytest

from paisa.experiments.portfolio_registry import PortfolioRegistry
from paisa.trading.models import (
    BenchmarkPortfolioIds,
    EvaluationDates,
    ExperimentConfig,
    PredictionRecord,
)


def _config(experiment_id: str, strategy_id: str) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        model_id="registry-model",
        portfolio_id=strategy_id,
        benchmark_portfolio_ids=BenchmarkPortfolioIds(
            f"{experiment_id}-cash", f"{experiment_id}-hold"
        ),
        evaluation_dates=EvaluationDates(
            feature_start=date(2026, 1, 5),
            feature_end=date(2026, 1, 5),
            target_start=date(2026, 1, 6),
            target_end=date(2026, 1, 6),
        ),
        created_at="2026-01-01T00:00:00Z",
    )


def test_disk_reservation_prevents_concurrent_portfolio_id_reuse(tmp_path):
    first = _config("registry-first", "shared-strategy")
    second = _config("registry-second", "shared-strategy")
    first_registry = PortfolioRegistry(tmp_path)
    first_registry.register(first)

    second_registry = PortfolioRegistry(tmp_path)
    with pytest.raises(ValueError, match="portfolio_id is already registered"):
        second_registry.register(second)

    first_registry.release(first)
    assert not (tmp_path / ".paisa-registry").exists()


def test_experiment_mapping_requires_nested_contract_fields_but_defaults_cash():
    complete = _config("mapping-test", "mapping-strategy").to_dict()
    del complete["initial_cash"]
    parsed = ExperimentConfig.from_mapping(complete)
    assert parsed.initial_cash == Decimal("100000.00")

    del complete["thresholds"]["buy_probability_up_gte"]
    with pytest.raises(KeyError):
        ExperimentConfig.from_mapping(complete)


def test_prediction_class_rejects_fractional_numeric_values():
    with pytest.raises(ValueError, match="predicted_class must be 0 or 1"):
        PredictionRecord(
            model_id="registry-model",
            ticker="OGDC",
            feature_date=date(2026, 1, 5),
            target_date=date(2026, 1, 6),
            predicted_class=1.8,
            probability_up=Decimal("0.8"),
            probability_down=Decimal("0.2"),
        )
