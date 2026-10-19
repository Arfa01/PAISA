"""Optional TensorFlow LSTM training with the standard immutable artifact contract.

TensorFlow is imported only by :func:`train_and_register_lstm` or when a saved
bundle is used for inference. Importing this module therefore remains safe for
the normal CPU-only PAISA installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
import random
import tempfile
from typing import Any, Mapping, Sequence, TYPE_CHECKING
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from paisa.experiments.artifacts import (
    code_provenance,
    package_versions,
    to_jsonable,
    utc_now_z,
    validate_model_manifest,
)
from paisa.experiments.evaluation import (
    binary_classification_metrics,
    make_prediction_frame,
    validate_prediction_frame,
)
from paisa.experiments.experiment_config import (
    DEFAULT_PROVIDER,
    FEATURE_SET_ID,
    PREDICTION_HORIZON,
    RANDOM_SEED,
    TARGET_NAME,
    TARGET_POSITIVE_CLASS_CONDITION,
)
from paisa.experiments.registry import ModelArtifactRegistry
from paisa.experiments.training_features import validate_feature_columns

if TYPE_CHECKING:
    from paisa.experiments.trainer import PreparedTrainingData


@dataclass(frozen=True, slots=True)
class LSTMTrainingConfig:
    """Complete, serializable configuration for the optional LSTM."""

    sequence_length: int = 20
    units: int = 32
    dropout: float = 0.20
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    patience: int = 6
    random_seed: int = RANDOM_SEED
    verbose: int = 1

    def __post_init__(self) -> None:
        integer_fields = {
            "sequence_length": self.sequence_length,
            "units": self.units,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise ValueError("patience must be a non-negative integer")
        if self.patience < 0:
            raise ValueError("patience must be a non-negative integer")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1, or 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "units": self.units,
            "dropout": float(self.dropout),
            "learning_rate": float(self.learning_rate),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": self.patience,
                "restore_best_weights": True,
            },
            "random_seed": self.random_seed,
            "shuffle": False,
            "deterministic_ops_requested": True,
        }


@dataclass(frozen=True, slots=True)
class LSTMSequenceBatch:
    """Sequences and their aligned target rows for one chronological split."""

    values: np.ndarray
    rows: pd.DataFrame
    targets: np.ndarray | None
    window_feature_dates: tuple[tuple[pd.Timestamp, ...], ...]


@dataclass(slots=True)
class LSTMInferenceBundle:
    """One portable inference unit containing both Keras bytes and its scaler."""

    keras_model_bytes: bytes
    scaler: StandardScaler
    feature_columns: tuple[str, ...]
    sequence_length: int
    keras_format: str = "keras-v3"

    def __post_init__(self) -> None:
        self.feature_columns = validate_feature_columns(self.feature_columns)
        if not self.keras_model_bytes:
            raise ValueError("keras_model_bytes must not be empty")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            raise ValueError("scaler must be fitted before it is bundled")
        if self.keras_format != "keras-v3":
            raise ValueError("keras_format must be 'keras-v3'")

    def load_keras_model(self, *, tensorflow_module: Any | None = None) -> Any:
        """Reconstruct the Keras model without making TensorFlow a core dependency."""

        tf = tensorflow_module or _require_tensorflow()
        with tempfile.TemporaryDirectory(prefix="paisa-lstm-load-") as temp_dir:
            model_path = Path(temp_dir) / "model.keras"
            model_path.write_bytes(self.keras_model_bytes)
            return tf.keras.models.load_model(model_path, compile=False)

    def predict_probability_up(
        self,
        all_rows: pd.DataFrame,
        target_rows: pd.DataFrame | None = None,
        *,
        tensorflow_module: Any | None = None,
        verbose: int = 0,
    ) -> np.ndarray:
        """Scale raw feature rows, build past-only windows, and predict class 1."""

        selected = all_rows if target_rows is None else target_rows
        batch = build_lstm_sequence_batch(
            all_rows,
            selected,
            scaler=self.scaler,
            feature_columns=self.feature_columns,
            sequence_length=self.sequence_length,
            require_every_target=target_rows is not None,
        )
        model = self.load_keras_model(tensorflow_module=tensorflow_module)
        return _predict_probability_up(model, batch.values, verbose=verbose)


@dataclass(frozen=True, slots=True)
class LSTMTrainingResult:
    """Paths and in-memory evidence emitted by one immutable LSTM registration."""

    model_id: str
    model_dir: Path
    manifest: dict[str, Any]
    metrics: dict[str, dict[str, Any]]
    predictions: pd.DataFrame
    predictions_all_splits: pd.DataFrame
    history: dict[str, Any]


def _require_tensorflow() -> Any:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by optional runtime
        raise RuntimeError(
            "The optional LSTM workflow requires TensorFlow. Install the notebook's "
            "documented TensorFlow constraint; it is intentionally absent from the "
            "core CPU requirements."
        ) from exc
    return tf


def build_lstm_sequence_batch(
    all_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    *,
    scaler: StandardScaler,
    feature_columns: Sequence[str],
    sequence_length: int,
    require_every_target: bool,
) -> LSTMSequenceBatch:
    """Build windows ending at each target feature date using no future row.

    ``all_rows`` supplies historical context across split boundaries. Target
    membership comes exclusively from ``target_rows``. This preserves every
    validation/test target once enough earlier training history exists.
    """

    columns = validate_feature_columns(feature_columns)
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise ValueError("sequence_length must be a positive integer")
    if sequence_length <= 0:
        raise ValueError("sequence_length must be a positive integer")
    if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
        raise ValueError("scaler must be fitted before sequence construction")

    required_all = {"feature_date", *columns}
    required_targets = {"feature_date", "target_date"}
    missing_all = sorted(required_all - set(all_rows.columns))
    missing_targets = sorted(required_targets - set(target_rows.columns))
    if missing_all:
        raise ValueError(f"all_rows is missing required columns: {missing_all}")
    if missing_targets:
        raise ValueError(f"target_rows is missing required columns: {missing_targets}")

    history = all_rows.copy()
    history["feature_date"] = pd.to_datetime(history["feature_date"], errors="raise")
    history = history.sort_values("feature_date", kind="mergesort").reset_index(drop=True)
    if history["feature_date"].duplicated().any():
        raise ValueError("all_rows feature_date must be unique")
    raw_frame = history[list(columns)].apply(pd.to_numeric, errors="raise")
    raw_values = raw_frame.to_numpy(float)
    if not np.isfinite(raw_values).all():
        raise ValueError("LSTM features must be finite before scaling")
    scaled_values = np.asarray(scaler.transform(raw_frame), dtype=np.float32)
    if not np.isfinite(scaled_values).all():
        raise ValueError("Scaled LSTM features must be finite")

    selected = target_rows.copy()
    selected["feature_date"] = pd.to_datetime(selected["feature_date"], errors="raise")
    selected["target_date"] = pd.to_datetime(selected["target_date"], errors="raise")
    selected = selected.sort_values("feature_date", kind="mergesort").reset_index(drop=True)
    if selected.empty:
        raise ValueError("target_rows must not be empty")
    if selected["feature_date"].duplicated().any():
        raise ValueError("target_rows feature_date must be unique")
    if not (selected["target_date"] > selected["feature_date"]).all():
        raise ValueError("Every LSTM target_date must be later than feature_date")

    history_index = {
        timestamp: index for index, timestamp in enumerate(history["feature_date"])
    }
    sequences: list[np.ndarray] = []
    retained_indexes: list[int] = []
    window_dates: list[tuple[pd.Timestamp, ...]] = []
    missing_context: list[str] = []
    for selected_index, row in selected.iterrows():
        feature_date = pd.Timestamp(row["feature_date"])
        if feature_date not in history_index:
            raise ValueError(f"Target feature_date is absent from all_rows: {feature_date.date()}")
        end_index = history_index[feature_date]
        start_index = end_index - sequence_length + 1
        if start_index < 0:
            missing_context.append(feature_date.strftime("%Y-%m-%d"))
            continue
        dates = tuple(
            pd.Timestamp(value)
            for value in history.iloc[start_index : end_index + 1]["feature_date"]
        )
        if len(dates) != sequence_length or dates[-1] != feature_date:
            raise AssertionError("LSTM sequence alignment failed")
        if any(value > feature_date for value in dates):
            raise AssertionError("LSTM sequence contains a future feature row")
        sequences.append(scaled_values[start_index : end_index + 1])
        retained_indexes.append(selected_index)
        window_dates.append(dates)

    if require_every_target and missing_context:
        raise ValueError(
            "Historical context is insufficient for target feature dates: "
            f"{missing_context[:3]}"
        )
    if not sequences:
        raise ValueError("No LSTM sequences remain after applying the lookback")

    retained = selected.iloc[retained_indexes].reset_index(drop=True)
    targets: np.ndarray | None = None
    if TARGET_NAME in retained.columns:
        target_values = pd.to_numeric(retained[TARGET_NAME], errors="raise").to_numpy()
        parsed = target_values.astype(int)
        if not np.equal(target_values, parsed).all() or not np.isin(parsed, [0, 1]).all():
            raise ValueError(f"{TARGET_NAME} must contain only classes 0 and 1")
        targets = parsed.astype(np.int32)
    return LSTMSequenceBatch(
        values=np.asarray(sequences, dtype=np.float32),
        rows=retained,
        targets=targets,
        window_feature_dates=tuple(window_dates),
    )


def _serialize_keras_model(model: Any) -> bytes:
    with tempfile.TemporaryDirectory(prefix="paisa-lstm-save-") as temp_dir:
        model_path = Path(temp_dir) / "model.keras"
        model.save(model_path)
        if not model_path.is_file():
            raise RuntimeError("Keras did not create the expected .keras model file")
        payload = model_path.read_bytes()
    if not payload:
        raise RuntimeError("The serialized Keras model is empty")
    return payload


def _predict_probability_up(model: Any, values: np.ndarray, *, verbose: int) -> np.ndarray:
    probability = np.asarray(model.predict(values, verbose=verbose), dtype=float).reshape(-1)
    if len(probability) != len(values):
        raise ValueError("The LSTM prediction count does not match the sequence count")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("LSTM probabilities must be finite and in [0, 1]")
    return probability


def _date_string(values: pd.Series, which: str) -> str:
    dates = pd.to_datetime(values, errors="raise")
    value = dates.min() if which == "min" else dates.max()
    return value.strftime("%Y-%m-%d")


def _assert_train_fitted_scaler(
    prepared: "PreparedTrainingData", scaler: StandardScaler
) -> None:
    columns = list(prepared.feature_columns)
    expected = StandardScaler().fit(prepared.splits.train[columns])
    for attribute in ("mean_", "scale_", "var_"):
        actual_value = np.asarray(getattr(scaler, attribute, None), dtype=float)
        expected_value = np.asarray(getattr(expected, attribute), dtype=float)
        if actual_value.shape != expected_value.shape or not np.allclose(
            actual_value, expected_value, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("The LSTM scaler must be fit on the training split only")


def _model_configuration(model: Any, config: LSTMTrainingConfig) -> dict[str, Any]:
    optimizer = getattr(model, "optimizer", None)
    optimizer_configuration = (
        optimizer.get_config() if optimizer is not None and hasattr(optimizer, "get_config") else {}
    )
    loss = getattr(model, "loss", "binary_crossentropy")
    if not isinstance(loss, (str, int, float, bool, type(None), dict, list, tuple)):
        loss = str(loss)
    return to_jsonable(
        {
            "keras_model": model.get_config(),
            "compile": {
                "optimizer_class": optimizer.__class__.__name__ if optimizer is not None else None,
                "optimizer_configuration": optimizer_configuration,
                "loss": loss,
                "metrics": ["accuracy"],
            },
            "fit": config.to_dict(),
        }
    )


def _library_versions(tensorflow_version: str | None) -> dict[str, str | None]:
    versions = package_versions()
    versions["tensorflow"] = tensorflow_version
    try:
        versions["keras"] = metadata.version("keras") if tensorflow_version else None
    except metadata.PackageNotFoundError:
        versions["keras"] = None
    return versions


def _tensorflow_version_for_model(model: Any) -> str | None:
    module = model.__class__.__module__
    if not (module.startswith("keras") or module.startswith("tensorflow")):
        return None
    try:
        return metadata.version("tensorflow")
    except metadata.PackageNotFoundError:
        return None


def register_fitted_lstm(
    *,
    prepared: "PreparedTrainingData",
    model: Any,
    scaler: StandardScaler,
    config: LSTMTrainingConfig,
    models_root: str | Path,
    repo_root: str | Path,
    model_id: str,
    provider: str = DEFAULT_PROVIDER,
    requested_start: str | None = None,
    requested_end: str | None = None,
    history: Mapping[str, Any] | None = None,
) -> LSTMTrainingResult:
    """Register a fitted LSTM and scaler under the active model contract."""

    columns = validate_feature_columns(prepared.feature_columns)
    _assert_train_fitted_scaler(prepared, scaler)
    batches = {
        "train": build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.train,
            scaler=scaler,
            feature_columns=columns,
            sequence_length=config.sequence_length,
            require_every_target=False,
        ),
        "validation": build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.validation,
            scaler=scaler,
            feature_columns=columns,
            sequence_length=config.sequence_length,
            require_every_target=True,
        ),
        "test": build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.test,
            scaler=scaler,
            feature_columns=columns,
            sequence_length=config.sequence_length,
            require_every_target=True,
        ),
    }

    prediction_frames: dict[str, pd.DataFrame] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for split_name, batch in batches.items():
        if batch.targets is None:
            raise ValueError(f"The {split_name} rows do not contain {TARGET_NAME}")
        probability_up = _predict_probability_up(model, batch.values, verbose=0)
        predictions = make_prediction_frame(
            rows=batch.rows,
            ticker=prepared.symbol,
            model_id=model_id,
            probability_up=probability_up,
            target_column=TARGET_NAME,
        )
        prediction_frames[split_name] = validate_prediction_frame(predictions)
        metrics[split_name] = binary_classification_metrics(
            batch.targets,
            prediction_frames[split_name]["predicted_class"].to_numpy(dtype=int),
            prediction_frames[split_name]["probability_up"].to_numpy(dtype=float),
        )

    all_predictions = validate_prediction_frame(
        pd.concat(prediction_frames.values(), ignore_index=True)
    )
    test_predictions = prediction_frames["test"]
    keras_bytes = _serialize_keras_model(model)
    bundle = LSTMInferenceBundle(
        keras_model_bytes=keras_bytes,
        scaler=scaler,
        feature_columns=columns,
        sequence_length=config.sequence_length,
    )

    repository = Path(repo_root).resolve()
    provenance = code_provenance(
        repository,
        source_paths=[
            Path(__file__).resolve(),
            repository / "src" / "paisa" / "experiments" / "evaluation.py",
            repository / "src" / "paisa" / "experiments" / "training_features.py",
            repository / "src" / "paisa" / "experiments" / "training_targets.py",
        ],
    )
    price_start = _date_string(prepared.prices["date"], "min")
    price_end = _date_string(prepared.prices["date"], "max")
    training_dates = {
        "data_feature_start": _date_string(prepared.model_frame["feature_date"], "min"),
        "data_feature_end": _date_string(prepared.model_frame["feature_date"], "max"),
        "train_target_start": _date_string(batches["train"].rows["target_date"], "min"),
        "train_target_end": _date_string(batches["train"].rows["target_date"], "max"),
        "validation_target_start": _date_string(
            batches["validation"].rows["target_date"], "min"
        ),
        "validation_target_end": _date_string(
            batches["validation"].rows["target_date"], "max"
        ),
        "test_target_start": _date_string(batches["test"].rows["target_date"], "min"),
        "test_target_end": _date_string(batches["test"].rows["target_date"], "max"),
    }
    manifest = {
        "schema_version": "1.0",
        "model_id": model_id,
        "algorithm": "lstm",
        "model_configuration": _model_configuration(model, config),
        "feature_set": {
            "feature_set_id": FEATURE_SET_ID,
            "columns": list(columns),
            "feature_code_version": provenance["source_fingerprint"],
            "transforms": [
                {
                    "name": "standard_scaler",
                    "parameters": {
                        "with_mean": bool(scaler.with_mean),
                        "with_std": bool(scaler.with_std),
                        "mean": to_jsonable(scaler.mean_),
                        "scale": to_jsonable(scaler.scale_),
                        "variance": to_jsonable(scaler.var_),
                        "samples_seen": to_jsonable(scaler.n_samples_seen_),
                    },
                    "fit_scope": "train_only",
                }
            ],
        },
        "training_dates": training_dates,
        "target_definition": {
            "name": TARGET_NAME,
            "positive_class_condition": TARGET_POSITIVE_CLASS_CONDITION,
            "class_labels": {"0": "down_or_flat", "1": "up"},
        },
        "prediction_horizon": dict(PREDICTION_HORIZON),
        "random_seed": config.random_seed,
        "metrics": metrics,
        "created_at": utc_now_z(),
        "symbols": [prepared.symbol],
        "data_provenance": {
            "provider": provider,
            "requested_symbols": [prepared.symbol],
            "included_symbols": [prepared.symbol],
            "requested_start": requested_start or price_start,
            "requested_end": requested_end or price_end,
            "actual_feature_start": training_dates["data_feature_start"],
            "actual_feature_end": training_dates["data_feature_end"],
            "dataset_sha256": prepared.dataset_sha256,
            "input_schema_version": "1.0",
            "price_adjustment": "unknown",
            "dividends_included": False,
            "corporate_actions_policy": "not_modeled",
        },
        "class_mapping": {
            "estimator_classes": [0, 1],
            "probability_fields_by_class": {
                "0": "probability_down",
                "1": "probability_up",
            },
        },
        **provenance,
        "library_versions": _library_versions(_tensorflow_version_for_model(model)),
    }
    registry = ModelArtifactRegistry(models_root)
    model_dir = registry.register(
        model_id,
        bundle,
        manifest,
        artifact_filename="lstm_bundle.joblib",
        extra_files={
            "predictions.csv": test_predictions,
            "predictions_all_splits.csv": all_predictions,
            "metrics.json": metrics,
            "training_history.json": dict(history or {}),
        },
    )
    completed_manifest = validate_model_manifest(registry.manifest(model_id))
    return LSTMTrainingResult(
        model_id=model_id,
        model_dir=model_dir,
        manifest=completed_manifest,
        metrics=metrics,
        predictions=test_predictions,
        predictions_all_splits=all_predictions,
        history=to_jsonable(dict(history or {})),
    )


def train_and_register_lstm(
    *,
    prepared: "PreparedTrainingData",
    models_root: str | Path,
    repo_root: str | Path,
    config: LSTMTrainingConfig | None = None,
    model_id: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    requested_start: str | None = None,
    requested_end: str | None = None,
) -> LSTMTrainingResult:
    """Fit the optional LSTM and atomically register its complete bundle."""

    resolved_config = config or LSTMTrainingConfig()
    resolved_model_id = model_id or f"LSTM-{uuid4().hex}"
    tf = _require_tensorflow()
    tf.keras.backend.clear_session()
    random.seed(resolved_config.random_seed)
    np.random.seed(resolved_config.random_seed)
    tf.keras.utils.set_random_seed(resolved_config.random_seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    columns = validate_feature_columns(prepared.feature_columns)
    scaler = StandardScaler()
    scaler.fit(prepared.splits.train[list(columns)])
    train_batch = build_lstm_sequence_batch(
        prepared.model_frame,
        prepared.splits.train,
        scaler=scaler,
        feature_columns=columns,
        sequence_length=resolved_config.sequence_length,
        require_every_target=False,
    )
    validation_batch = build_lstm_sequence_batch(
        prepared.model_frame,
        prepared.splits.validation,
        scaler=scaler,
        feature_columns=columns,
        sequence_length=resolved_config.sequence_length,
        require_every_target=True,
    )
    if train_batch.targets is None or validation_batch.targets is None:
        raise ValueError("Training and validation rows must contain targets")

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(resolved_config.sequence_length, len(columns))
            ),
            tf.keras.layers.LSTM(
                resolved_config.units,
                dropout=resolved_config.dropout,
            ),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=resolved_config.learning_rate
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=resolved_config.patience,
            restore_best_weights=True,
        )
    ]
    fitted_history = model.fit(
        train_batch.values,
        train_batch.targets,
        validation_data=(validation_batch.values, validation_batch.targets),
        epochs=resolved_config.epochs,
        batch_size=resolved_config.batch_size,
        shuffle=False,
        callbacks=callbacks,
        verbose=resolved_config.verbose,
    )
    return register_fitted_lstm(
        prepared=prepared,
        model=model,
        scaler=scaler,
        config=resolved_config,
        models_root=models_root,
        repo_root=repo_root,
        model_id=resolved_model_id,
        provider=provider,
        requested_start=requested_start,
        requested_end=requested_end,
        history=fitted_history.history,
    )


__all__ = [
    "LSTMInferenceBundle",
    "LSTMSequenceBatch",
    "LSTMTrainingConfig",
    "LSTMTrainingResult",
    "build_lstm_sequence_batch",
    "register_fitted_lstm",
    "train_and_register_lstm",
]
