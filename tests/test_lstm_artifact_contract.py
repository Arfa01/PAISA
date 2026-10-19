from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from paisa.experiments.artifacts import validate_model_manifest
from paisa.experiments.evaluation import validate_prediction_frame
from paisa.experiments.lstm import (
    LSTMInferenceBundle,
    LSTMTrainingConfig,
    build_lstm_sequence_batch,
    register_fitted_lstm,
    train_and_register_lstm,
)
from paisa.experiments.registry import ModelArtifactRegistry
from paisa.experiments.trainer import prepare_training_data


class _FakeOptimizer:
    def get_config(self) -> dict[str, float]:
        return {"learning_rate": 0.001}


class _FakeKerasModel:
    """Small serializer/predictor double for the TensorFlow-free contract test."""

    optimizer = _FakeOptimizer()
    loss = "binary_crossentropy"

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"fake-keras-v3-model")

    def predict(self, values: np.ndarray, *, verbose: int = 0) -> np.ndarray:
        del verbose
        scores = np.linspace(0.2, 0.8, num=len(values), dtype=float)
        return scores.reshape(-1, 1)

    def get_config(self) -> dict[str, object]:
        return {"name": "contract-test-lstm", "layers": [{"class_name": "LSTM"}]}


def _prepared_training_data(rows: int = 180):
    dates = pd.bdate_range("2025-01-02", periods=rows)
    movement = np.where(np.arange(rows) % 2 == 0, 1.8, -1.1)
    close = 100.0 + np.cumsum(movement)
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "OGDC",
            "open": close - 0.25,
            "close": close,
            "volume": 1_000_000 + np.arange(rows) * 1_000,
            "source": "lstm-contract-fixture",
        }
    )
    return prepare_training_data(prices, symbol="OGDC")


def test_sequence_batch_uses_prior_context_without_future_rows():
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    scaler = StandardScaler().fit(
        prepared.splits.train[list(prepared.feature_columns)]
    )

    batch = build_lstm_sequence_batch(
        prepared.model_frame,
        prepared.splits.validation,
        scaler=scaler,
        feature_columns=prepared.feature_columns,
        sequence_length=config.sequence_length,
        require_every_target=True,
    )

    assert len(batch.rows) == len(prepared.splits.validation)
    assert batch.rows.iloc[0]["feature_date"] == prepared.splits.validation.iloc[0][
        "feature_date"
    ]
    first_target_date = pd.Timestamp(batch.rows.iloc[0]["feature_date"])
    assert batch.window_feature_dates[0][-1] == first_target_date
    assert all(date <= first_target_date for date in batch.window_feature_dates[0])
    assert min(batch.window_feature_dates[0]) < first_target_date


def test_lstm_bundle_uses_registry_manifest_and_shared_prediction_contract(tmp_path):
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    scaler = StandardScaler().fit(
        prepared.splits.train[list(prepared.feature_columns)]
    )
    model_id = "LSTM-CONTRACT-TEST"

    result = register_fitted_lstm(
        prepared=prepared,
        model=_FakeKerasModel(),
        scaler=scaler,
        config=config,
        models_root=tmp_path / "models",
        repo_root=Path(__file__).resolve().parents[1],
        model_id=model_id,
        provider="fixture",
        history={"loss": [0.7], "val_loss": [0.8]},
    )

    registry = ModelArtifactRegistry(tmp_path / "models")
    loaded, manifest = registry.load(model_id)
    assert isinstance(loaded, LSTMInferenceBundle)
    assert loaded.keras_model_bytes == b"fake-keras-v3-model"
    assert loaded.feature_columns == prepared.feature_columns
    assert loaded.sequence_length == 5
    assert hasattr(loaded.scaler, "mean_")
    assert manifest == validate_model_manifest(manifest)
    assert manifest["algorithm"] == "lstm"
    assert manifest["artifact_path"] == f"{model_id}/lstm_bundle.joblib"
    assert manifest["feature_set"]["transforms"][0]["name"] == "standard_scaler"
    assert manifest["feature_set"]["transforms"][0]["fit_scope"] == "train_only"
    assert manifest["model_configuration"]["fit"]["shuffle"] is False
    assert manifest["library_versions"]["tensorflow"] is None

    predictions_path = result.model_dir / "predictions.csv"
    predictions = validate_prediction_frame(pd.read_csv(predictions_path))
    assert list(predictions.columns) == [
        "schema_version",
        "model_id",
        "ticker",
        "feature_date",
        "target_date",
        "predicted_class",
        "probability_up",
        "probability_down",
        "actual_class",
    ]
    assert set(predictions["model_id"]) == {model_id}
    assert len(predictions) == len(prepared.splits.test)
    assert json.loads((result.model_dir / "metrics.json").read_text())["test"] == manifest[
        "metrics"
    ]["test"]


def test_lstm_registration_rejects_scaler_fitted_beyond_training_split(tmp_path):
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    leaky_scaler = StandardScaler().fit(
        prepared.model_frame[list(prepared.feature_columns)]
    )

    with pytest.raises(ValueError, match="training split only"):
        register_fitted_lstm(
            prepared=prepared,
            model=_FakeKerasModel(),
            scaler=leaky_scaler,
            config=config,
            models_root=tmp_path / "models",
            repo_root=Path(__file__).resolve().parents[1],
            model_id="LSTM-LEAKY-SCALER",
            provider="fixture",
        )

    assert not (tmp_path / "models").exists()


def test_tensorflow_lstm_bundle_round_trip_smoke(tmp_path):
    tf = pytest.importorskip("tensorflow", reason="optional TensorFlow is not installed")
    prepared = _prepared_training_data(rows=140)
    config = LSTMTrainingConfig(
        sequence_length=3,
        units=2,
        dropout=0.0,
        epochs=1,
        batch_size=16,
        patience=0,
        verbose=0,
    )

    result = train_and_register_lstm(
        prepared=prepared,
        models_root=tmp_path / "models",
        repo_root=Path(__file__).resolve().parents[1],
        config=config,
        model_id="LSTM-TENSORFLOW-SMOKE",
        provider="fixture",
    )
    loaded, manifest = ModelArtifactRegistry(tmp_path / "models").load(
        "LSTM-TENSORFLOW-SMOKE"
    )
    probabilities = loaded.predict_probability_up(
        prepared.model_frame,
        prepared.splits.test,
        tensorflow_module=tf,
    )

    assert manifest["algorithm"] == "lstm"
    assert probabilities.shape == (len(prepared.splits.test),)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    validate_prediction_frame(result.predictions)
