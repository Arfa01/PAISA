from __future__ import annotations

from dataclasses import make_dataclass
import io
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

import paisa.experiments.lstm as lstm_module
from paisa.experiments.experiment_config import TARGET_NAME
from paisa.experiments.lstm import (
    LSTMInferenceBundle,
    LSTMPreprocessor,
    build_lstm_sequence_batch,
)


FEATURE_COLUMNS = ("open", "close")


def _model_rows(row_count: int = 8) -> pd.DataFrame:
    feature_dates = pd.bdate_range("2025-01-02", periods=row_count)
    target_dates = pd.bdate_range("2025-01-03", periods=row_count)
    close = 100.0 + np.arange(row_count, dtype=float)
    return pd.DataFrame(
        {
            "feature_date": feature_dates,
            "target_date": target_dates,
            "open": close - 0.5,
            "close": close,
            TARGET_NAME: np.ones(row_count, dtype=int),
        }
    )


def _scaler(rows: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(rows[list(FEATURE_COLUMNS)])


class _LoadedModel:
    def predict(self, values: np.ndarray, *, verbose: int = 0) -> np.ndarray:
        del verbose
        return np.full((len(values), 1), 0.625, dtype=float)


def _fake_tensorflow(expected_model_bytes: bytes) -> SimpleNamespace:
    def load_model(path, *, compile: bool):
        assert compile is False
        assert path.read_bytes() == expected_model_bytes
        return _LoadedModel()

    return SimpleNamespace(
        keras=SimpleNamespace(models=SimpleNamespace(load_model=load_model))
    )


def _legacy_bundle_payload(rows: pd.DataFrame) -> bytes:
    """Create the exact pre-preprocessor slotted pickle shape in memory."""

    legacy_class = make_dataclass(
        "LSTMInferenceBundle",
        [
            ("keras_model_bytes", bytes),
            ("scaler", StandardScaler),
            ("feature_columns", tuple[str, ...]),
            ("sequence_length", int),
            ("keras_format", str, "keras-v3"),
        ],
        slots=True,
    )
    legacy_class.__module__ = lstm_module.__name__
    legacy_class.__qualname__ = "LSTMInferenceBundle"
    legacy_bundle = legacy_class(
        b"legacy-keras-v3-model",
        _scaler(rows),
        FEATURE_COLUMNS,
        3,
    )

    payload = io.BytesIO()
    current_class = lstm_module.LSTMInferenceBundle
    lstm_module.LSTMInferenceBundle = legacy_class
    try:
        joblib.dump(legacy_bundle, payload)
    finally:
        lstm_module.LSTMInferenceBundle = current_class
    return payload.getvalue()


def test_legacy_slotted_bundle_migrates_and_runs_scaler_only_inference():
    rows = _model_rows()
    payload = _legacy_bundle_payload(rows)

    loaded = joblib.load(io.BytesIO(payload))

    assert type(loaded) is LSTMInferenceBundle
    assert loaded.preprocessor is None
    assert loaded.feature_columns == FEATURE_COLUMNS
    assert loaded.sequence_length == 3
    probability_up = loaded.predict_probability_up(
        rows,
        rows.iloc[-2:].copy(),
        tensorflow_module=_fake_tensorflow(b"legacy-keras-v3-model"),
    )
    np.testing.assert_allclose(probability_up, [0.625, 0.625])


def test_current_bundle_round_trip_preserves_preprocessor():
    rows = _model_rows()
    preprocessor = LSTMPreprocessor.fit(rows, FEATURE_COLUMNS)
    bundle = LSTMInferenceBundle(
        keras_model_bytes=b"current-keras-v3-model",
        scaler=preprocessor.scaler,
        feature_columns=FEATURE_COLUMNS,
        sequence_length=3,
        preprocessor=preprocessor,
    )
    payload = io.BytesIO()
    joblib.dump(bundle, payload)
    payload.seek(0)

    loaded = joblib.load(payload)

    assert isinstance(loaded.preprocessor, LSTMPreprocessor)
    assert loaded.preprocessor.feature_columns == FEATURE_COLUMNS
    np.testing.assert_allclose(
        loaded.preprocessor.transform(rows), preprocessor.transform(rows)
    )


def test_sequence_batch_rejects_target_date_that_disagrees_with_all_rows():
    rows = _model_rows()
    forged = rows.iloc[-1:][["feature_date", "target_date"]].copy()
    forged.loc[:, "target_date"] = pd.Timestamp("2099-01-02")

    with pytest.raises(ValueError, match="target_date does not match all_rows"):
        build_lstm_sequence_batch(
            rows,
            forged,
            scaler=_scaler(rows),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=3,
            require_every_target=True,
        )


def test_sequence_batch_rejects_target_class_that_disagrees_with_all_rows():
    rows = _model_rows()
    forged = rows.iloc[-1:].copy()
    forged.loc[:, TARGET_NAME] = 0

    with pytest.raises(ValueError, match=f"{TARGET_NAME} does not match all_rows"):
        build_lstm_sequence_batch(
            rows,
            forged,
            scaler=_scaler(rows),
            feature_columns=FEATURE_COLUMNS,
            sequence_length=3,
            require_every_target=True,
        )


def test_sequence_batch_ignores_unselected_history_rows_without_labels():
    rows = _model_rows()
    rows.loc[0, TARGET_NAME] = np.nan
    selected = rows.iloc[-1:].copy()

    batch = build_lstm_sequence_batch(
        rows,
        selected,
        scaler=_scaler(rows),
        feature_columns=FEATURE_COLUMNS,
        sequence_length=3,
        require_every_target=True,
    )

    np.testing.assert_array_equal(batch.targets, [1])


def test_sequence_batch_preserves_label_free_feature_context_inference():
    rows = _model_rows()
    feature_context = rows.drop(columns=["target_date", TARGET_NAME])
    selected = rows.iloc[-2:][["feature_date", "target_date"]].copy()

    batch = build_lstm_sequence_batch(
        feature_context,
        selected,
        scaler=_scaler(feature_context),
        feature_columns=FEATURE_COLUMNS,
        sequence_length=3,
        require_every_target=True,
    )

    assert len(batch.values) == 2
    assert batch.targets is None
    pd.testing.assert_frame_equal(batch.rows, selected.reset_index(drop=True))
