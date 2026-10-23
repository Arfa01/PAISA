"""Optional TensorFlow LSTM training with the standard immutable artifact contract.

TensorFlow is imported only by :func:`train_and_register_lstm` or when a saved
bundle is used for inference. Importing this module therefore remains safe for
the normal CPU-only PAISA installation.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import time
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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
    recurrent_dropout: float = 0.0
    kernel_l2: float = 0.0
    recurrent_l2: float = 0.0
    second_layer_units: int | None = None
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32
    patience: int = 6
    random_seed: int = RANDOM_SEED
    class_weight: Mapping[int, float] | None = None
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
        for name, value in {
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout,
        }.items():
            if not np.isfinite(value) or not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        for name, value in {
            "kernel_l2": self.kernel_l2,
            "recurrent_l2": self.recurrent_l2,
        }.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.second_layer_units is not None and (
            isinstance(self.second_layer_units, bool)
            or not isinstance(self.second_layer_units, int)
            or self.second_layer_units <= 0
        ):
            raise ValueError("second_layer_units must be null or a positive integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.class_weight is not None:
            if not isinstance(self.class_weight, Mapping) or not self.class_weight:
                raise ValueError("class_weight must be null or a non-empty mapping")
            parsed_weights: dict[int, float] = {}
            for raw_class, raw_weight in self.class_weight.items():
                if isinstance(raw_class, bool):
                    raise ValueError("class_weight keys must be binary class IDs 0 and 1")
                try:
                    class_id = int(raw_class)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "class_weight keys must be binary class IDs 0 and 1"
                    ) from exc
                if class_id not in (0, 1) or str(raw_class).strip() not in {
                    "0",
                    "1",
                }:
                    raise ValueError("class_weight keys must be binary class IDs 0 and 1")
                weight = float(raw_weight)
                if not np.isfinite(weight) or weight <= 0:
                    raise ValueError("class_weight values must be positive and finite")
                parsed_weights[class_id] = weight
            if set(parsed_weights) != {0, 1}:
                raise ValueError("class_weight must define both binary classes 0 and 1")
            object.__setattr__(
                self,
                "class_weight",
                MappingProxyType(dict(sorted(parsed_weights.items()))),
            )
        if self.verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1, or 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "units": self.units,
            "dropout": float(self.dropout),
            "recurrent_dropout": float(self.recurrent_dropout),
            "kernel_l2": float(self.kernel_l2),
            "recurrent_l2": float(self.recurrent_l2),
            "second_layer_units": self.second_layer_units,
            "learning_rate": float(self.learning_rate),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "early_stopping": {
                "monitor": "val_loss",
                "patience": self.patience,
                "restore_best_weights": True,
            },
            "random_seed": self.random_seed,
            "class_weight": (
                None
                if self.class_weight is None
                else {
                    str(class_id): float(weight)
                    for class_id, weight in self.class_weight.items()
                }
            ),
            "verbose": self.verbose,
            "shuffle": False,
            "deterministic_ops_requested": True,
        }


@dataclass(frozen=True, slots=True)
class LSTMPreprocessor:
    """Immutable feature order with train-fitted median imputation and scaling."""

    feature_columns: tuple[str, ...]
    imputer: SimpleImputer
    scaler: StandardScaler

    def __post_init__(self) -> None:
        columns = validate_feature_columns(self.feature_columns)
        object.__setattr__(self, "feature_columns", columns)
        if self.imputer.strategy != "median":
            raise ValueError("LSTM imputer strategy must be 'median'")
        if not hasattr(self.imputer, "statistics_"):
            raise ValueError("LSTM imputer must be fitted")
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            raise ValueError("LSTM scaler must be fitted")
        feature_count = len(columns)
        imputer_statistics = np.asarray(self.imputer.statistics_, dtype=float)
        scaler_mean = np.asarray(self.scaler.mean_, dtype=float)
        scaler_scale = np.asarray(self.scaler.scale_, dtype=float)
        if (
            imputer_statistics.shape != (feature_count,)
            or scaler_mean.shape != (feature_count,)
            or scaler_scale.shape != (feature_count,)
        ):
            raise ValueError(
                "LSTM preprocessor fitted feature count must match feature_columns"
            )
        if not np.isfinite(imputer_statistics).all():
            raise ValueError("LSTM imputer statistics must be finite")
        if not np.isfinite(scaler_mean).all() or not np.isfinite(scaler_scale).all():
            raise ValueError("LSTM scaler parameters must be finite")

    @staticmethod
    def _numeric_frame(
        rows: pd.DataFrame, columns: Sequence[str], *, field_name: str
    ) -> pd.DataFrame:
        if not isinstance(rows, pd.DataFrame):
            raise TypeError(f"{field_name} must be a pandas DataFrame")
        missing = sorted(set(columns) - set(rows.columns))
        if missing:
            raise ValueError(f"{field_name} is missing feature columns: {missing}")
        try:
            numeric = rows.loc[:, list(columns)].apply(pd.to_numeric, errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} feature values must be numeric") from exc
        values = numeric.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"{field_name} feature values must not contain infinity")
        return numeric

    @classmethod
    def fit(
        cls,
        train_rows: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> "LSTMPreprocessor":
        """Fit both learned transforms on the declared training rows only."""

        columns = validate_feature_columns(feature_columns)
        numeric = cls._numeric_frame(train_rows, columns, field_name="train_rows")
        imputer = SimpleImputer(strategy="median")
        imputed = np.asarray(imputer.fit_transform(numeric), dtype=float)
        if imputed.shape != (len(numeric), len(columns)) or not np.isfinite(
            imputed
        ).all():
            raise ValueError(
                "Median imputation must retain every ordered feature with finite values"
            )
        scaler = StandardScaler().fit(imputed)
        return cls(feature_columns=columns, imputer=imputer, scaler=scaler)

    fit_train_only = fit

    def transform(self, rows: pd.DataFrame) -> np.ndarray:
        """Transform rows using the immutable stored feature order."""

        numeric = self._numeric_frame(
            rows, self.feature_columns, field_name="preprocessing rows"
        )
        imputed = np.asarray(self.imputer.transform(numeric), dtype=float)
        if imputed.shape != (len(numeric), len(self.feature_columns)):
            raise ValueError("LSTM imputer changed the ordered feature count")
        transformed = np.asarray(self.scaler.transform(imputed), dtype=float)
        if transformed.shape != imputed.shape or not np.isfinite(transformed).all():
            raise ValueError("LSTM preprocessing output must be finite and shape-preserving")
        return transformed


@dataclass(frozen=True, slots=True)
class LSTMSequenceBatch:
    """Sequences and their aligned target rows for one chronological split."""

    values: np.ndarray
    rows: pd.DataFrame
    targets: np.ndarray | None
    window_feature_dates: tuple[tuple[pd.Timestamp, ...], ...]


DETERMINISM_NUMERIC_CAVEAT = (
    "Deterministic operations are requested where TensorFlow supports them, but "
    "accelerator kernels and platform or library differences may still produce "
    "small numeric differences; bit-identical CPU/GPU results are not promised."
)


@dataclass(frozen=True, slots=True)
class LSTMDeterminismReport:
    """Auditable outcome of configuring Python, NumPy, and TensorFlow seeds."""

    python_seed: int
    numpy_seed: int
    tensorflow_seed: int
    deterministic_ops_requested: bool
    deterministic_ops_enabled: bool
    deterministic_ops_error: str | None
    numeric_reproducibility_caveat: str = DETERMINISM_NUMERIC_CAVEAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_seed": self.python_seed,
            "numpy_seed": self.numpy_seed,
            "tensorflow_seed": self.tensorflow_seed,
            "deterministic_ops_requested": self.deterministic_ops_requested,
            "deterministic_ops_enabled": self.deterministic_ops_enabled,
            "deterministic_ops_error": self.deterministic_ops_error,
            "numeric_reproducibility_caveat": self.numeric_reproducibility_caveat,
        }


def set_lstm_determinism(
    seed: int = RANDOM_SEED,
    *,
    tensorflow_module: Any | None = None,
) -> LSTMDeterminismReport:
    """Seed every LSTM runtime and request deterministic TensorFlow operations."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    tf = tensorflow_module or _require_tensorflow()
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    deterministic_enabled = False
    deterministic_error: str | None = None
    try:
        tf.config.experimental.enable_op_determinism()
        deterministic_enabled = True
    except Exception as exc:  # pragma: no cover - backend/platform dependent
        deterministic_error = f"{type(exc).__name__}: {exc}"
    return LSTMDeterminismReport(
        python_seed=seed,
        numpy_seed=seed,
        tensorflow_seed=seed,
        deterministic_ops_requested=True,
        deterministic_ops_enabled=deterministic_enabled,
        deterministic_ops_error=deterministic_error,
    )


def _regularizer(tf: Any, coefficient: float) -> Any | None:
    return tf.keras.regularizers.l2(float(coefficient)) if coefficient > 0 else None


def build_lstm_model(
    config: LSTMTrainingConfig,
    feature_count: int,
    *,
    tensorflow_module: Any | None = None,
) -> Any:
    """Construct and compile the configured small recurrent classifier."""

    if isinstance(feature_count, bool) or not isinstance(feature_count, int):
        raise ValueError("feature_count must be a positive integer")
    if feature_count <= 0:
        raise ValueError("feature_count must be a positive integer")
    tf = tensorflow_module or _require_tensorflow()
    kernel_regularizer = _regularizer(tf, config.kernel_l2)
    recurrent_regularizer = _regularizer(tf, config.recurrent_l2)
    recurrent_options = {
        "dropout": float(config.dropout),
        "recurrent_dropout": float(config.recurrent_dropout),
        "kernel_regularizer": kernel_regularizer,
        "recurrent_regularizer": recurrent_regularizer,
    }
    layers: list[Any] = [
        tf.keras.layers.Input(shape=(config.sequence_length, feature_count))
    ]
    if config.second_layer_units is None:
        layers.append(tf.keras.layers.LSTM(config.units, **recurrent_options))
    else:
        layers.extend(
            [
                tf.keras.layers.LSTM(
                    config.units,
                    return_sequences=True,
                    **recurrent_options,
                ),
                tf.keras.layers.LSTM(
                    config.second_layer_units,
                    return_sequences=False,
                    **recurrent_options,
                ),
            ]
        )
    layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))
    model = tf.keras.Sequential(layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def lstm_runtime_metadata(
    *, tensorflow_module: Any, model: Any | None = None
) -> dict[str, Any]:
    """Describe TensorFlow-visible devices and the fitted model size.

    Physical devices describe installed hardware, including accelerators hidden
    with ``set_visible_devices``.  The reported effective device must therefore
    come from TensorFlow's logical device view (or its configured visible-device
    view on older/test runtimes), not the physical inventory.
    """

    try:
        physical_devices = list(tensorflow_module.config.list_physical_devices())
    except Exception:
        physical_devices = []

    effective_devices: list[Any]
    device_selection_source: str
    try:
        effective_devices = list(tensorflow_module.config.list_logical_devices())
        device_selection_source = "logical_devices"
    except (AttributeError, TypeError):
        try:
            effective_devices = list(tensorflow_module.config.get_visible_devices())
            device_selection_source = "visible_devices"
        except (AttributeError, TypeError):
            # Compatibility for lightweight TensorFlow doubles and older
            # runtimes that expose only the physical-device query.
            effective_devices = list(physical_devices)
            device_selection_source = "physical_devices_fallback"
    except Exception:
        # A configured visible-device query is still more truthful than the
        # physical inventory if logical device initialization is unavailable.
        try:
            effective_devices = list(tensorflow_module.config.get_visible_devices())
            device_selection_source = "visible_devices"
        except Exception:
            effective_devices = list(physical_devices)
            device_selection_source = "physical_devices_fallback"

    physical_device_records = [
        {
            "name": str(getattr(device, "name", "unknown")),
            "device_type": str(getattr(device, "device_type", "unknown")).upper(),
        }
        for device in physical_devices
    ]
    effective_device_records = [
        {
            "name": str(getattr(device, "name", "unknown")),
            "device_type": str(getattr(device, "device_type", "unknown")).upper(),
        }
        for device in effective_devices
    ]
    device_types = {device["device_type"] for device in effective_device_records}
    selected_type = next(
        (
            candidate
            for candidate in ("TPU", "GPU", "CPU")
            if candidate in device_types
        ),
        "UNKNOWN",
    )
    parameter_count: int | None = None
    if model is not None and hasattr(model, "count_params"):
        parameter_count = int(model.count_params())
        if parameter_count < 0:
            raise ValueError("Keras model parameter count must be non-negative")
    return {
        "tensorflow_version": getattr(tensorflow_module, "__version__", None),
        "device": selected_type,
        "device_selection_source": device_selection_source,
        "physical_devices": physical_device_records,
        "effective_devices": effective_device_records,
        "parameter_count": parameter_count,
        "numeric_reproducibility_caveat": DETERMINISM_NUMERIC_CAVEAT,
    }


@dataclass(slots=True)
class LSTMInferenceBundle:
    """One portable inference unit containing both Keras bytes and its scaler."""

    keras_model_bytes: bytes
    scaler: StandardScaler
    feature_columns: tuple[str, ...]
    sequence_length: int
    keras_format: str = "keras-v3"
    preprocessor: LSTMPreprocessor | None = None

    def __post_init__(self) -> None:
        self.feature_columns = validate_feature_columns(self.feature_columns)
        if not self.keras_model_bytes:
            raise ValueError("keras_model_bytes must not be empty")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            raise ValueError("scaler must be fitted before it is bundled")
        if self.preprocessor is not None:
            if self.preprocessor.feature_columns != self.feature_columns:
                raise ValueError(
                    "Bundled preprocessor feature order must match feature_columns"
                )
            for attribute in ("mean_", "scale_", "var_"):
                bundled = np.asarray(getattr(self.scaler, attribute), dtype=float)
                prepared = np.asarray(
                    getattr(self.preprocessor.scaler, attribute), dtype=float
                )
                if bundled.shape != prepared.shape or not np.allclose(
                    bundled, prepared, rtol=0.0, atol=0.0
                ):
                    raise ValueError(
                        "Bundled compatibility scaler must match the preprocessor scaler"
                    )
        if self.keras_format != "keras-v3":
            raise ValueError("keras_format must be 'keras-v3'")

    def __setstate__(self, state: Any) -> None:
        """Restore current and legacy slotted bundles from joblib state.

        Bundles written before the train-only imputer was introduced have no
        ``preprocessor`` slot in their pickle state.  Python can otherwise
        deserialize such an object while leaving the new slot unset, causing
        inference to fail only when the attribute is first accessed.
        """

        if isinstance(state, tuple):
            if len(state) != 2:
                raise ValueError("LSTM bundle pickle state must contain two items")
            dictionary_state, slot_state = state
            if dictionary_state is not None and not isinstance(
                dictionary_state, Mapping
            ):
                raise TypeError("LSTM bundle dictionary state must be a mapping")
            if not isinstance(slot_state, Mapping):
                raise TypeError("LSTM bundle slot state must be a mapping")
            restored = dict(dictionary_state or {})
            restored.update(slot_state)
        elif isinstance(state, Mapping):
            restored = dict(state)
        else:
            raise TypeError("LSTM bundle pickle state must be a mapping or pair")

        field_names = (
            "keras_model_bytes",
            "scaler",
            "feature_columns",
            "sequence_length",
            "keras_format",
            "preprocessor",
        )
        unknown = sorted(set(restored) - set(field_names))
        if unknown:
            raise ValueError(f"LSTM bundle pickle state has unknown fields: {unknown}")
        restored.setdefault("preprocessor", None)
        missing = sorted(set(field_names) - set(restored))
        if missing:
            raise ValueError(f"LSTM bundle pickle state is missing fields: {missing}")

        for field_name in field_names:
            object.__setattr__(self, field_name, restored[field_name])
        self.__post_init__()

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
            preprocessor=self.preprocessor,
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
            "The optional LSTM workflow requires TensorFlow. Install the constraint "
            "documented in requirements-lstm.txt in the separate experiment "
            "environment; it is intentionally absent from the core CPU requirements."
        ) from exc
    return tf


def build_lstm_sequence_batch(
    all_rows: pd.DataFrame,
    target_rows: pd.DataFrame,
    *,
    scaler: StandardScaler | None = None,
    preprocessor: LSTMPreprocessor | None = None,
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
    if preprocessor is not None:
        if preprocessor.feature_columns != columns:
            raise ValueError(
                "preprocessor feature order must exactly match feature_columns"
            )
        if scaler is not None:
            for attribute in ("mean_", "scale_", "var_"):
                provided = np.asarray(getattr(scaler, attribute, None), dtype=float)
                expected = np.asarray(
                    getattr(preprocessor.scaler, attribute), dtype=float
                )
                if provided.shape != expected.shape or not np.allclose(
                    provided, expected, rtol=0.0, atol=0.0
                ):
                    raise ValueError(
                        "scaler must match the supplied LSTM preprocessor scaler"
                    )
    elif scaler is None or not hasattr(scaler, "mean_") or not hasattr(
        scaler, "scale_"
    ):
        raise ValueError(
            "A fitted scaler or LSTM preprocessor is required for sequence construction"
        )

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
    if "target_date" in history.columns:
        history["target_date"] = pd.to_datetime(
            history["target_date"], errors="raise"
        )
    history = history.sort_values("feature_date", kind="mergesort").reset_index(drop=True)
    if history["feature_date"].duplicated().any():
        raise ValueError("all_rows feature_date must be unique")
    raw_frame = history[list(columns)].apply(pd.to_numeric, errors="raise")
    raw_values = raw_frame.to_numpy(float)
    if np.isinf(raw_values).any():
        raise ValueError("LSTM features must not contain infinity")
    if preprocessor is None and not np.isfinite(raw_values).all():
        raise ValueError(
            "LSTM features must be finite before scaler-only preprocessing"
        )
    if preprocessor is not None:
        transformed = preprocessor.transform(history)
    else:
        assert scaler is not None
        transformed = scaler.transform(raw_frame)
    scaled_values = np.asarray(transformed, dtype=np.float32)
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

    selected_has_targets = TARGET_NAME in selected.columns
    if selected_has_targets:
        required_authoritative = {"target_date", TARGET_NAME}
        missing_authoritative = sorted(
            required_authoritative - set(history.columns)
        )
        if missing_authoritative:
            raise ValueError(
                "all_rows is missing authoritative target metadata required for "
                f"labelled target_rows: {missing_authoritative}"
            )
        selected_target_values = pd.to_numeric(
            selected[TARGET_NAME], errors="raise"
        ).to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(selected_target_values).all():
            raise ValueError(
                f"target_rows {TARGET_NAME} must contain only classes 0 and 1"
            )
        parsed_selected_targets = selected_target_values.astype(np.int32)
        if (
            not np.equal(selected_target_values, parsed_selected_targets).all()
            or not np.isin(parsed_selected_targets, [0, 1]).all()
        ):
            raise ValueError(
                f"target_rows {TARGET_NAME} must contain only classes 0 and 1"
            )
        selected[TARGET_NAME] = selected_target_values.astype(np.int32)

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
        authoritative_row = history.iloc[end_index]
        if "target_date" in history.columns and pd.Timestamp(
            row["target_date"]
        ) != pd.Timestamp(authoritative_row["target_date"]):
            raise ValueError(
                "target_rows target_date does not match all_rows for feature_date "
                f"{feature_date.date()}"
            )
        if selected_has_targets:
            authoritative_target = pd.to_numeric(
                pd.Series([authoritative_row[TARGET_NAME]]), errors="raise"
            ).to_numpy(dtype=float, na_value=np.nan)
            if (
                not np.isfinite(authoritative_target).all()
                or not np.isin(authoritative_target, [0.0, 1.0]).all()
            ):
                raise ValueError(
                    f"all_rows {TARGET_NAME} must contain class 0 or 1 for selected "
                    f"feature_date {feature_date.date()}"
                )
            if int(row[TARGET_NAME]) != int(authoritative_target[0]):
                raise ValueError(
                    f"target_rows {TARGET_NAME} does not match all_rows for "
                    f"feature_date {feature_date.date()}"
                )
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


def _assert_train_fitted_preprocessor(
    prepared: "PreparedTrainingData", preprocessor: LSTMPreprocessor
) -> None:
    columns = validate_feature_columns(prepared.feature_columns)
    if preprocessor.feature_columns != columns:
        raise ValueError(
            "The LSTM preprocessor feature order must match prepared training data"
        )
    expected = LSTMPreprocessor.fit(prepared.splits.train, columns)
    comparisons = (
        (preprocessor.imputer, expected.imputer, "statistics_"),
        (preprocessor.scaler, expected.scaler, "mean_"),
        (preprocessor.scaler, expected.scaler, "scale_"),
        (preprocessor.scaler, expected.scaler, "var_"),
    )
    for actual_transform, expected_transform, attribute in comparisons:
        actual_value = np.asarray(getattr(actual_transform, attribute, None), dtype=float)
        expected_value = np.asarray(getattr(expected_transform, attribute), dtype=float)
        if actual_value.shape != expected_value.shape or not np.allclose(
            actual_value, expected_value, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(
                "The LSTM imputer and scaler must be fit on the training split only"
            )


def _preprocessing_transforms(
    scaler: StandardScaler,
    preprocessor: LSTMPreprocessor | None,
) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = []
    if preprocessor is not None:
        transforms.append(
            {
                "name": "median_imputer",
                "parameters": {
                    "strategy": "median",
                    "statistics": to_jsonable(preprocessor.imputer.statistics_),
                    "n_features_in": int(preprocessor.imputer.n_features_in_),
                },
                "fit_scope": "train_only",
            }
        )
    transforms.append(
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
    )
    return transforms


def _joblib_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    joblib.dump(value, buffer)
    payload = buffer.getvalue()
    if not payload:
        raise RuntimeError("Joblib produced an empty preprocessing artifact")
    return payload


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")


def _preprocessing_artifacts(
    preprocessor: LSTMPreprocessor,
) -> tuple[dict[str, bytes | dict[str, Any]], dict[str, Any]]:
    payloads: dict[str, bytes] = {
        "preprocessing/preprocessor.joblib": _joblib_bytes(preprocessor),
        "preprocessing/imputer.joblib": _joblib_bytes(preprocessor.imputer),
        "preprocessing/scaler.joblib": _joblib_bytes(preprocessor.scaler),
        "preprocessing/feature_columns.json": _json_bytes(
            list(preprocessor.feature_columns)
        ),
    }
    files = [
        {
            "path": path,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for path, payload in sorted(payloads.items())
    ]
    checksums = {"algorithm": "sha256", "files": files}
    return {
        **payloads,
        "preprocessing/checksums.json": checksums,
    }, checksums


def _model_configuration(model: Any, config: LSTMTrainingConfig) -> dict[str, Any]:
    optimizer = getattr(model, "optimizer", None)
    optimizer_configuration = (
        optimizer.get_config()
        if optimizer is not None and hasattr(optimizer, "get_config")
        else {}
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


def _validated_probability_up(
    values: Sequence[float] | np.ndarray,
    *,
    split_name: str,
    expected_count: int,
) -> np.ndarray:
    """Validate caller-supplied frozen probabilities without predicting again."""

    try:
        probability = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Precomputed {split_name} probabilities must be numeric"
        ) from exc
    if probability.ndim == 2 and probability.shape[1:] == (1,):
        probability = probability[:, 0]
    elif probability.ndim != 1:
        raise ValueError(
            f"Precomputed {split_name} probabilities must be one-dimensional"
        )
    if len(probability) != expected_count:
        raise ValueError(
            f"Precomputed {split_name} probability count {len(probability)} does "
            f"not match the sequence count {expected_count}"
        )
    if not np.isfinite(probability).all() or (
        (probability < 0.0) | (probability > 1.0)
    ).any():
        raise ValueError(
            f"Precomputed {split_name} probabilities must be finite and in [0, 1]"
        )
    return probability


def _scaler_state_matches(left: StandardScaler, right: StandardScaler) -> bool:
    for attribute in ("mean_", "scale_", "var_"):
        left_value = np.asarray(getattr(left, attribute, None), dtype=float)
        right_value = np.asarray(getattr(right, attribute, None), dtype=float)
        if left_value.shape != right_value.shape or not np.allclose(
            left_value, right_value, rtol=0.0, atol=0.0
        ):
            return False
    return True


def register_fitted_lstm(
    *,
    prepared: "PreparedTrainingData",
    model: Any,
    scaler: StandardScaler | None = None,
    preprocessor: LSTMPreprocessor | None = None,
    config: LSTMTrainingConfig,
    models_root: str | Path,
    repo_root: str | Path,
    model_id: str,
    provider: str = DEFAULT_PROVIDER,
    requested_start: str | None = None,
    requested_end: str | None = None,
    input_schema_version: str = "1.0",
    data_provenance_extra_fields: Mapping[str, Any] | None = None,
    source_paths: Iterable[str | Path] | None = None,
    history: Mapping[str, Any] | None = None,
    probabilities_by_split: Mapping[
        str, Sequence[float] | np.ndarray
    ] | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    manifest_extra_fields: Mapping[str, Any] | None = None,
    extra_files: Mapping[str, Any] | None = None,
) -> LSTMTrainingResult:
    """Register a fitted LSTM and its train-only preprocessing contract.

    ``probabilities_by_split`` may contain any subset of ``train``,
    ``validation``, and ``test``. Supplied arrays are validated and reused, so
    callers that already performed their single frozen-test evaluation do not
    invoke the model on that split again.
    """

    columns = validate_feature_columns(prepared.feature_columns)
    normalized_provider = str(provider).strip()
    normalized_input_schema = str(input_schema_version).strip()
    if not normalized_provider:
        raise ValueError("provider must be non-empty")
    if not normalized_input_schema:
        raise ValueError("input_schema_version must be non-empty")
    required_data_provenance_fields = {
        "provider",
        "requested_symbols",
        "included_symbols",
        "requested_start",
        "requested_end",
        "actual_feature_start",
        "actual_feature_end",
        "dataset_sha256",
        "input_schema_version",
        "price_adjustment",
        "dividends_included",
        "corporate_actions_policy",
    }
    supplied_data_provenance = dict(data_provenance_extra_fields or {})
    if any(
        not isinstance(key, str) or not key.strip()
        for key in supplied_data_provenance
    ):
        raise ValueError(
            "data_provenance_extra_fields keys must be non-empty strings"
        )
    provenance_collisions = sorted(
        required_data_provenance_fields.intersection(supplied_data_provenance)
    )
    if provenance_collisions:
        raise ValueError(
            "data_provenance_extra_fields cannot override required fields: "
            f"{provenance_collisions}"
        )
    price_start = _date_string(prepared.prices["date"], "min")
    price_end = _date_string(prepared.prices["date"], "max")
    effective_requested_start = requested_start or price_start
    effective_requested_end = requested_end or price_end
    for field_name, raw_value in (
        ("requested_start", effective_requested_start),
        ("requested_end", effective_requested_end),
    ):
        normalized = str(raw_value)
        try:
            parsed = pd.Timestamp(normalized)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be an ISO date") from exc
        if parsed.strftime("%Y-%m-%d") != normalized:
            raise ValueError(f"{field_name} must use exact YYYY-MM-DD form")
    if pd.Timestamp(effective_requested_start) > pd.Timestamp(effective_requested_end):
        raise ValueError("requested_start must not be later than requested_end")
    if preprocessor is not None:
        _assert_train_fitted_preprocessor(prepared, preprocessor)
        if scaler is not None and not _scaler_state_matches(
            scaler, preprocessor.scaler
        ):
            raise ValueError(
                "The compatibility scaler must match the supplied LSTM preprocessor"
            )
        effective_scaler = preprocessor.scaler
    else:
        if scaler is None:
            raise ValueError("A fitted scaler or LSTM preprocessor is required")
        _assert_train_fitted_scaler(prepared, scaler)
        effective_scaler = scaler

    supplied_probabilities = dict(probabilities_by_split or {})
    unknown_probability_splits = sorted(
        set(supplied_probabilities) - {"train", "validation", "test"}
    )
    if unknown_probability_splits:
        raise ValueError(
            "probabilities_by_split contains unknown splits: "
            f"{unknown_probability_splits}"
        )
    batches = {
        "train": build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.train,
            scaler=effective_scaler,
            preprocessor=preprocessor,
            feature_columns=columns,
            sequence_length=config.sequence_length,
            require_every_target=False,
        ),
        "validation": build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.validation,
            scaler=effective_scaler,
            preprocessor=preprocessor,
            feature_columns=columns,
            sequence_length=config.sequence_length,
            require_every_target=True,
        ),
        "test": build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.test,
            scaler=effective_scaler,
            preprocessor=preprocessor,
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
        if split_name in supplied_probabilities:
            probability_up = _validated_probability_up(
                supplied_probabilities[split_name],
                split_name=split_name,
                expected_count=len(batch.values),
            )
        else:
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
        scaler=effective_scaler,
        feature_columns=columns,
        sequence_length=config.sequence_length,
        preprocessor=preprocessor,
    )

    repository = Path(repo_root).resolve()
    default_source_paths = [
        Path(__file__).resolve(),
        repository / "src" / "paisa" / "experiments" / "evaluation.py",
        repository / "src" / "paisa" / "experiments" / "training_features.py",
        repository / "src" / "paisa" / "experiments" / "training_targets.py",
    ]
    provenance = code_provenance(
        repository,
        source_paths=(default_source_paths if source_paths is None else source_paths),
    )
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
    required_data_provenance: dict[str, Any] = {
        "provider": normalized_provider,
        "requested_symbols": [prepared.symbol],
        "included_symbols": [prepared.symbol],
        "requested_start": effective_requested_start,
        "requested_end": effective_requested_end,
        "actual_feature_start": training_dates["data_feature_start"],
        "actual_feature_end": training_dates["data_feature_end"],
        "dataset_sha256": prepared.dataset_sha256,
        "input_schema_version": normalized_input_schema,
        "price_adjustment": "unknown",
        "dividends_included": False,
        "corporate_actions_policy": "not_modeled",
    }
    data_provenance = {
        **required_data_provenance,
        **to_jsonable(supplied_data_provenance),
    }

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "model_id": model_id,
        "algorithm": "lstm",
        "model_configuration": _model_configuration(model, config),
        "feature_set": {
            "feature_set_id": FEATURE_SET_ID,
            "columns": list(columns),
            "feature_code_version": provenance["source_fingerprint"],
            "transforms": _preprocessing_transforms(
                effective_scaler, preprocessor
            ),
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
        "data_provenance": data_provenance,
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
    registered_extra_files: dict[str, Any] = {
        "predictions.csv": test_predictions,
        "predictions_all_splits.csv": all_predictions,
        "metrics.json": metrics,
        "training_history.json": dict(history or {}),
    }
    if training_metadata is not None:
        manifest["training_metadata"] = to_jsonable(dict(training_metadata))
        registered_extra_files["training_metadata.json"] = dict(training_metadata)
    if preprocessor is not None:
        preprocessing_files, preprocessing_checksums = _preprocessing_artifacts(
            preprocessor
        )
        manifest["preprocessing_artifacts"] = preprocessing_checksums
        registered_extra_files.update(preprocessing_files)

    supplied_manifest_fields = dict(manifest_extra_fields or {})
    if any(
        not isinstance(key, str) or not key.strip()
        for key in supplied_manifest_fields
    ):
        raise ValueError("manifest_extra_fields keys must be non-empty strings")
    protected_manifest_fields = set(manifest) | {
        "artifact_path",
        "artifact_sha256",
    }
    manifest_collisions = sorted(
        protected_manifest_fields.intersection(supplied_manifest_fields)
    )
    if manifest_collisions:
        raise ValueError(
            "manifest_extra_fields cannot override protected fields: "
            f"{manifest_collisions}"
        )
    manifest.update(to_jsonable(supplied_manifest_fields))

    supplied_extra_files = dict(extra_files or {})
    if any(not isinstance(path, str) or not path.strip() for path in supplied_extra_files):
        raise ValueError("extra_files paths must be non-empty strings")
    file_collisions = sorted(
        set(registered_extra_files).intersection(supplied_extra_files)
    )
    if file_collisions:
        raise ValueError(
            "extra_files cannot replace protected contract files: "
            f"{file_collisions}"
        )
    registered_extra_files.update(supplied_extra_files)

    registry = ModelArtifactRegistry(models_root)
    model_dir = registry.register(
        model_id,
        bundle,
        manifest,
        artifact_filename="lstm_bundle.joblib",
        extra_files=registered_extra_files,
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
    manifest_extra_fields: Mapping[str, Any] | None = None,
    extra_files: Mapping[str, Any] | None = None,
) -> LSTMTrainingResult:
    """Fit the optional LSTM and atomically register its complete bundle."""

    resolved_config = config or LSTMTrainingConfig()
    resolved_model_id = model_id or f"LSTM-{uuid4().hex}"
    tf = _require_tensorflow()
    tf.keras.backend.clear_session()
    determinism = set_lstm_determinism(
        resolved_config.random_seed, tensorflow_module=tf
    )

    columns = validate_feature_columns(prepared.feature_columns)
    preprocessor = LSTMPreprocessor.fit(prepared.splits.train, columns)
    train_batch = build_lstm_sequence_batch(
        prepared.model_frame,
        prepared.splits.train,
        preprocessor=preprocessor,
        feature_columns=columns,
        sequence_length=resolved_config.sequence_length,
        require_every_target=False,
    )
    validation_batch = build_lstm_sequence_batch(
        prepared.model_frame,
        prepared.splits.validation,
        preprocessor=preprocessor,
        feature_columns=columns,
        sequence_length=resolved_config.sequence_length,
        require_every_target=True,
    )
    if train_batch.targets is None or validation_batch.targets is None:
        raise ValueError("Training and validation rows must contain targets")

    model = build_lstm_model(
        resolved_config,
        len(columns),
        tensorflow_module=tf,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=resolved_config.patience,
        restore_best_weights=True,
    )
    started_at = time.perf_counter()
    fitted_history = model.fit(
        train_batch.values,
        train_batch.targets,
        validation_data=(validation_batch.values, validation_batch.targets),
        epochs=resolved_config.epochs,
        batch_size=resolved_config.batch_size,
        shuffle=False,
        callbacks=[early_stopping],
        class_weight=(
            None
            if resolved_config.class_weight is None
            else dict(resolved_config.class_weight)
        ),
        verbose=resolved_config.verbose,
    )
    training_duration_seconds = time.perf_counter() - started_at
    history = to_jsonable(dict(fitted_history.history))
    epochs_completed = max(
        (len(values) for values in history.values() if isinstance(values, list)),
        default=0,
    )
    validation_losses = history.get("val_loss", [])
    best_epoch = (
        int(np.argmin(np.asarray(validation_losses, dtype=float))) + 1
        if validation_losses
        else None
    )
    stopped_epoch = int(getattr(early_stopping, "stopped_epoch", 0))
    training_metadata = {
        "determinism": determinism.to_dict(),
        "runtime": lstm_runtime_metadata(tensorflow_module=tf, model=model),
        "training_duration_seconds": float(training_duration_seconds),
        "epochs_requested": resolved_config.epochs,
        "epochs_completed": epochs_completed,
        "best_validation_epoch": best_epoch,
        "early_stopping_epoch": stopped_epoch + 1 if stopped_epoch > 0 else None,
        "batch_size": resolved_config.batch_size,
        "class_weight": (
            None
            if resolved_config.class_weight is None
            else dict(resolved_config.class_weight)
        ),
    }
    return register_fitted_lstm(
        prepared=prepared,
        model=model,
        scaler=preprocessor.scaler,
        preprocessor=preprocessor,
        config=resolved_config,
        models_root=models_root,
        repo_root=repo_root,
        model_id=resolved_model_id,
        provider=provider,
        requested_start=requested_start,
        requested_end=requested_end,
        history=history,
        training_metadata=training_metadata,
        manifest_extra_fields=manifest_extra_fields,
        extra_files=extra_files,
    )


__all__ = [
    "DETERMINISM_NUMERIC_CAVEAT",
    "LSTMDeterminismReport",
    "LSTMInferenceBundle",
    "LSTMPreprocessor",
    "LSTMSequenceBatch",
    "LSTMTrainingConfig",
    "LSTMTrainingResult",
    "build_lstm_model",
    "build_lstm_sequence_batch",
    "lstm_runtime_metadata",
    "register_fitted_lstm",
    "set_lstm_determinism",
    "train_and_register_lstm",
]
