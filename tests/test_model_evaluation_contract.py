from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paisa.experiments.evaluation import (
    PREDICTION_COLUMNS,
    binary_classification_metrics,
    make_prediction_frame,
    validate_prediction_frame,
)
from paisa.experiments.runner import load_predictions


def _target_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_date": ["2026-01-05", "2026-01-06", "2026-01-07"],
            "target_date": ["2026-01-06", "2026-01-07", "2026-01-08"],
            "target_next_day_up": [0, 1, 0],
        }
    )


def test_prediction_frame_is_exact_backtesting_contract_and_tie_maps_down(tmp_path):
    predictions = make_prediction_frame(
        _target_rows(),
        ticker="ogdc",
        model_id="M-001-fixture",
        split_name="test",
        y_pred=np.array([1, 1, 0]),
        probability_up=np.array([0.5, 0.75, 0.2]),
    )

    assert predictions.columns.tolist() == PREDICTION_COLUMNS
    assert predictions["schema_version"].tolist() == ["1.0"] * 3
    assert predictions["ticker"].tolist() == ["OGDC"] * 3
    assert predictions["predicted_class"].tolist() == [0, 1, 0]

    path = tmp_path / "predictions.csv"
    predictions.to_csv(path, index=False)
    records = load_predictions(path)
    assert len(records) == 3
    assert records[0].model_id == "M-001-fixture"
    assert records[0].predicted_class == 0
    assert str(records[1].probability_up) == "0.75"


def test_prediction_validation_rejects_duplicates_and_extra_columns():
    frame = make_prediction_frame(
        _target_rows(),
        ticker="OGDC",
        model_id="M-001-fixture",
        probability_up=np.array([0.4, 0.6, 0.3]),
    )
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="uniqueness key"):
        validate_prediction_frame(duplicate)

    with_extra = frame.assign(split="test")
    with pytest.raises(ValueError, match="exactly match"):
        validate_prediction_frame(with_extra)


def test_binary_metrics_include_complete_contract_and_are_deterministic():
    args = (
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        np.array([0.1, 0.6, 0.8, 0.7]),
    )
    first = binary_classification_metrics(*args)
    second = binary_classification_metrics(*args)

    assert first == second
    assert first["support"] == 4
    assert first["positive_label"] == 1
    assert first["average"] == "binary"
    assert first["zero_division"] == 0
    assert first["class_distribution"] == {"0": 2, "1": 2}
    assert first["confusion_matrix"] == [[1, 1], [0, 2]]
    assert first["roc_auc"] is not None
    for key in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "matthews_corrcoef",
        "brier_score",
    ):
        assert isinstance(first[key], float)


def test_roc_auc_is_null_when_only_one_actual_class_is_present():
    metrics = binary_classification_metrics(
        np.array([0, 0]), np.array([0, 0]), np.array([0.1, 0.2])
    )
    assert metrics["roc_auc"] is None
    assert metrics["class_distribution"] == {"0": 2, "1": 0}


def test_experiment_package_lazy_exports_preserve_public_api():
    from paisa.experiments import ModelArtifactRegistry, PortfolioRegistry, run_experiment

    assert ModelArtifactRegistry.__name__ == "ModelArtifactRegistry"
    assert PortfolioRegistry.__name__ == "PortfolioRegistry"
    assert callable(run_experiment)
