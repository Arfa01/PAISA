from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from paisa.experiments.artifacts import validate_model_manifest
from paisa.experiments.evaluation import validate_prediction_frame
from paisa.experiments.lstm import (
    DETERMINISM_NUMERIC_CAVEAT,
    LSTMInferenceBundle,
    LSTMPreprocessor,
    LSTMTrainingConfig,
    build_lstm_model,
    build_lstm_sequence_batch,
    lstm_runtime_metadata,
    register_fitted_lstm,
    set_lstm_determinism,
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

    def __init__(self) -> None:
        self.predict_calls: list[int] = []

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(b"fake-keras-v3-model")

    def predict(self, values: np.ndarray, *, verbose: int = 0) -> np.ndarray:
        del verbose
        self.predict_calls.append(len(values))
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


def _fake_tensorflow(
    *,
    deterministic_error: Exception | None = None,
    logical_device_types: tuple[str, ...] = ("CPU", "GPU"),
):
    state: dict[str, object] = {"seed": None, "determinism_calls": 0}

    def set_random_seed(seed: int) -> None:
        state["seed"] = seed

    def enable_op_determinism() -> None:
        state["determinism_calls"] = int(state["determinism_calls"]) + 1
        if deterministic_error is not None:
            raise deterministic_error

    devices = [
        SimpleNamespace(name="/physical_device:CPU:0", device_type="CPU"),
        SimpleNamespace(name="/physical_device:GPU:0", device_type="GPU"),
    ]
    logical_devices = [
        SimpleNamespace(name=f"/device:{device_type}:0", device_type=device_type)
        for device_type in logical_device_types
    ]
    tensorflow = SimpleNamespace(
        __version__="test-tensorflow",
        keras=SimpleNamespace(utils=SimpleNamespace(set_random_seed=set_random_seed)),
        config=SimpleNamespace(
            experimental=SimpleNamespace(
                enable_op_determinism=enable_op_determinism
            ),
            list_physical_devices=lambda: devices,
            list_logical_devices=lambda: logical_devices,
        ),
    )
    return tensorflow, state


class _FakeSequential:
    def __init__(self, layers: list[dict[str, object]]) -> None:
        self.layers = layers
        self.compile_arguments: dict[str, object] = {}

    def compile(self, **kwargs: object) -> None:
        self.compile_arguments = kwargs

    def count_params(self) -> int:
        return 1_234


def _fake_tensorflow_for_model_builder():
    def input_layer(*, shape: tuple[int, int]) -> dict[str, object]:
        return {"kind": "Input", "shape": shape}

    def lstm_layer(units: int, **kwargs: object) -> dict[str, object]:
        return {"kind": "LSTM", "units": units, **kwargs}

    def dense_layer(units: int, **kwargs: object) -> dict[str, object]:
        return {"kind": "Dense", "units": units, **kwargs}

    def adam(*, learning_rate: float) -> dict[str, object]:
        return {"kind": "Adam", "learning_rate": learning_rate}

    return SimpleNamespace(
        keras=SimpleNamespace(
            layers=SimpleNamespace(
                Input=input_layer,
                LSTM=lstm_layer,
                Dense=dense_layer,
            ),
            regularizers=SimpleNamespace(l2=lambda value: ("l2", value)),
            optimizers=SimpleNamespace(Adam=adam),
            Sequential=_FakeSequential,
        )
    )


def test_lstm_preprocessor_fits_median_then_scaler_in_immutable_order():
    train_rows = pd.DataFrame(
        {
            "close": [10.0, np.nan, 30.0, 40.0],
            "open": [1.0, 2.0, np.nan, 4.0],
        }
    )
    feature_columns = ("open", "close")

    preprocessor = LSTMPreprocessor.fit(train_rows, feature_columns)
    expected_imputed = np.array(
        [[1.0, 10.0], [2.0, 30.0], [2.0, 30.0], [4.0, 40.0]]
    )

    assert preprocessor.feature_columns == feature_columns
    np.testing.assert_allclose(preprocessor.imputer.statistics_, [2.0, 30.0])
    np.testing.assert_allclose(
        preprocessor.scaler.mean_, expected_imputed.mean(axis=0)
    )
    np.testing.assert_allclose(
        preprocessor.transform(train_rows[["close", "open"]]),
        preprocessor.scaler.transform(expected_imputed),
    )
    with pytest.raises(FrozenInstanceError):
        preprocessor.feature_columns = ("close", "open")
    with pytest.raises(ValueError, match="missing feature columns"):
        preprocessor.transform(train_rows[["open"]])


def test_lstm_training_config_serializes_regularization_and_class_weights():
    config = LSTMTrainingConfig(
        units=64,
        recurrent_dropout=0.15,
        kernel_l2=0.001,
        recurrent_l2=0.002,
        second_layer_units=16,
        class_weight={0: 0.75, 1: 1.25},
        verbose=0,
    )

    serialized = config.to_dict()
    assert serialized["units"] == 64
    assert serialized["recurrent_dropout"] == 0.15
    assert serialized["kernel_l2"] == 0.001
    assert serialized["recurrent_l2"] == 0.002
    assert serialized["second_layer_units"] == 16
    assert serialized["class_weight"] == {"0": 0.75, "1": 1.25}
    assert serialized["verbose"] == 0
    with pytest.raises(TypeError):
        config.class_weight[0] = 99.0
    with pytest.raises(ValueError, match="both binary classes"):
        LSTMTrainingConfig(class_weight={1: 1.0})
    with pytest.raises(ValueError, match="non-negative"):
        LSTMTrainingConfig(recurrent_l2=-0.1)
    with pytest.raises(ValueError, match="positive integer"):
        LSTMTrainingConfig(second_layer_units=0)


def test_lstm_determinism_report_records_success_and_failure_with_fake_tf():
    tensorflow, state = _fake_tensorflow()
    first_report = set_lstm_determinism(71, tensorflow_module=tensorflow)
    first_python = random.random()
    first_numpy = np.random.random()
    second_report = set_lstm_determinism(71, tensorflow_module=tensorflow)

    assert random.random() == first_python
    assert np.random.random() == first_numpy
    assert state == {"seed": 71, "determinism_calls": 2}
    assert first_report.to_dict() == second_report.to_dict()
    assert first_report.deterministic_ops_enabled is True
    assert first_report.deterministic_ops_error is None
    assert "CPU/GPU" in first_report.numeric_reproducibility_caveat

    failing_tensorflow, _ = _fake_tensorflow(
        deterministic_error=RuntimeError("unsupported")
    )
    failure_report = set_lstm_determinism(
        9, tensorflow_module=failing_tensorflow
    )
    assert failure_report.deterministic_ops_enabled is False
    assert failure_report.deterministic_ops_error == "RuntimeError: unsupported"
    assert failure_report.deterministic_ops_requested is True


def test_lstm_model_builder_uses_controlled_config_and_runtime_metadata():
    tensorflow = _fake_tensorflow_for_model_builder()
    config = LSTMTrainingConfig(
        sequence_length=20,
        units=64,
        dropout=0.25,
        recurrent_dropout=0.1,
        kernel_l2=0.001,
        recurrent_l2=0.002,
        second_layer_units=16,
        learning_rate=0.0005,
    )

    model = build_lstm_model(config, 19, tensorflow_module=tensorflow)
    recurrent_layers = [layer for layer in model.layers if layer["kind"] == "LSTM"]

    assert model.layers[0] == {"kind": "Input", "shape": (20, 19)}
    assert [layer["units"] for layer in recurrent_layers] == [64, 16]
    assert recurrent_layers[0]["return_sequences"] is True
    assert recurrent_layers[1]["return_sequences"] is False
    assert recurrent_layers[0]["recurrent_dropout"] == 0.1
    assert recurrent_layers[0]["kernel_regularizer"] == ("l2", 0.001)
    assert recurrent_layers[0]["recurrent_regularizer"] == ("l2", 0.002)
    assert model.compile_arguments["optimizer"] == {
        "kind": "Adam",
        "learning_rate": 0.0005,
    }

    runtime_tf, _ = _fake_tensorflow()
    metadata = lstm_runtime_metadata(
        tensorflow_module=runtime_tf, model=model
    )
    assert metadata["tensorflow_version"] == "test-tensorflow"
    assert metadata["device"] == "GPU"
    assert metadata["device_selection_source"] == "logical_devices"
    assert metadata["parameter_count"] == 1_234
    assert metadata["numeric_reproducibility_caveat"] == DETERMINISM_NUMERIC_CAVEAT

    cpu_only_tf, _ = _fake_tensorflow(logical_device_types=("CPU",))
    cpu_metadata = lstm_runtime_metadata(
        tensorflow_module=cpu_only_tf, model=model
    )
    assert {item["device_type"] for item in cpu_metadata["physical_devices"]} == {
        "CPU",
        "GPU",
    }
    assert cpu_metadata["effective_devices"] == [
        {"name": "/device:CPU:0", "device_type": "CPU"}
    ]
    assert cpu_metadata["device"] == "CPU"


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
    assert loaded.preprocessor is None
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


def test_lstm_preprocessor_registration_saves_components_and_checksums(tmp_path):
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    preprocessor = LSTMPreprocessor.fit(
        prepared.splits.train, prepared.feature_columns
    )
    model_id = "LSTM-PREPROCESSOR-CONTRACT"

    result = register_fitted_lstm(
        prepared=prepared,
        model=_FakeKerasModel(),
        preprocessor=preprocessor,
        config=config,
        models_root=tmp_path / "models",
        repo_root=Path(__file__).resolve().parents[1],
        model_id=model_id,
        provider="fixture",
    )

    loaded, manifest = ModelArtifactRegistry(tmp_path / "models").load(model_id)
    assert isinstance(loaded.preprocessor, LSTMPreprocessor)
    assert loaded.preprocessor.feature_columns == prepared.feature_columns
    assert [
        transform["name"] for transform in manifest["feature_set"]["transforms"]
    ] == ["median_imputer", "standard_scaler"]
    assert all(
        transform["fit_scope"] == "train_only"
        for transform in manifest["feature_set"]["transforms"]
    )

    preprocessing_dir = result.model_dir / "preprocessing"
    assert isinstance(
        joblib.load(preprocessing_dir / "imputer.joblib"),
        type(preprocessor.imputer),
    )
    assert isinstance(joblib.load(preprocessing_dir / "scaler.joblib"), StandardScaler)
    saved_preprocessor = joblib.load(preprocessing_dir / "preprocessor.joblib")
    assert saved_preprocessor.feature_columns == prepared.feature_columns
    assert json.loads((preprocessing_dir / "feature_columns.json").read_text()) == list(
        prepared.feature_columns
    )

    checksum_manifest = json.loads(
        (preprocessing_dir / "checksums.json").read_text()
    )
    assert checksum_manifest == manifest["preprocessing_artifacts"]
    for item in checksum_manifest["files"]:
        payload = (result.model_dir / item["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]


def test_lstm_registration_reuses_validated_frozen_probabilities_and_extras(tmp_path):
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    scaler = StandardScaler().fit(
        prepared.splits.train[list(prepared.feature_columns)]
    )
    frozen_test_probabilities = np.linspace(
        0.1, 0.9, num=len(prepared.splits.test)
    )
    model = _FakeKerasModel()

    result = register_fitted_lstm(
        prepared=prepared,
        model=model,
        scaler=scaler,
        config=config,
        models_root=tmp_path / "models",
        repo_root=Path(__file__).resolve().parents[1],
        model_id="LSTM-FROZEN-PROBABILITIES",
        provider="fixture",
        probabilities_by_split={"test": frozen_test_probabilities},
        manifest_extra_fields={
            "variant_id": "M-005",
            "invocation_id": "INVOCATION-TEST",
        },
        extra_files={"reports/frozen_evaluation.json": {"evaluated_once": True}},
    )

    assert len(model.predict_calls) == 2
    np.testing.assert_allclose(
        result.predictions["probability_up"], frozen_test_probabilities
    )
    assert result.manifest["variant_id"] == "M-005"
    assert result.manifest["invocation_id"] == "INVOCATION-TEST"
    assert json.loads(
        (result.model_dir / "reports" / "frozen_evaluation.json").read_text()
    ) == {"evaluated_once": True}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("wrong_length", "probability count"),
        ("not_finite", "finite and in"),
        ("out_of_range", "finite and in"),
    ],
)
def test_lstm_registration_rejects_invalid_precomputed_probabilities(
    tmp_path, case, message
):
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    scaler = StandardScaler().fit(
        prepared.splits.train[list(prepared.feature_columns)]
    )
    test_count = len(prepared.splits.test)
    probabilities = {
        "wrong_length": [0.5],
        "not_finite": [np.nan] * test_count,
        "out_of_range": [1.1] * test_count,
    }[case]

    with pytest.raises(ValueError, match=message):
        register_fitted_lstm(
            prepared=prepared,
            model=_FakeKerasModel(),
            scaler=scaler,
            config=config,
            models_root=tmp_path / "models",
            repo_root=Path(__file__).resolve().parents[1],
            model_id=f"LSTM-INVALID-PROBABILITIES-{case}",
            provider="fixture",
            probabilities_by_split={"test": probabilities},
        )


def test_lstm_registration_rejects_protected_manifest_and_file_overrides(tmp_path):
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    scaler = StandardScaler().fit(
        prepared.splits.train[list(prepared.feature_columns)]
    )
    common = {
        "prepared": prepared,
        "model": _FakeKerasModel(),
        "scaler": scaler,
        "config": config,
        "models_root": tmp_path / "models",
        "repo_root": Path(__file__).resolve().parents[1],
        "provider": "fixture",
    }

    with pytest.raises(ValueError, match="protected fields"):
        register_fitted_lstm(
            **common,
            model_id="LSTM-PROTECTED-MANIFEST",
            manifest_extra_fields={"metrics": {}},
        )
    with pytest.raises(ValueError, match="protected contract files"):
        register_fitted_lstm(
            **common,
            model_id="LSTM-PROTECTED-FILE",
            extra_files={"predictions.csv": "replacement"},
        )


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


def test_lstm_registration_rejects_preprocessor_fitted_beyond_training_split(
    tmp_path,
):
    prepared = _prepared_training_data()
    config = LSTMTrainingConfig(sequence_length=5, epochs=1, batch_size=8)
    leaky_preprocessor = LSTMPreprocessor.fit(
        prepared.model_frame, prepared.feature_columns
    )

    with pytest.raises(ValueError, match="training split only"):
        register_fitted_lstm(
            prepared=prepared,
            model=_FakeKerasModel(),
            preprocessor=leaky_preprocessor,
            config=config,
            models_root=tmp_path / "models",
            repo_root=Path(__file__).resolve().parents[1],
            model_id="LSTM-LEAKY-PREPROCESSOR",
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
