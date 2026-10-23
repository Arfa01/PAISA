"""Immutable orchestration for the two controlled price-only LSTM variants."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import platform
import re
import shutil
import stat
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from paisa.evidence.artifact_security import (
    atomic_rename_noreplace,
    prepare_output_root,
    reject_symlink_components,
)
from paisa.experiments.artifacts import (
    TrainingRunEvidence,
    assert_training_run_artifacts_unchanged,
    load_model_artifact,
    make_zip_from_directory,
    seal_training_run_artifacts,
    sha256_file,
    to_jsonable,
    utc_now_z,
    validate_training_run_artifacts,
    write_json,
)
from paisa.experiments.evaluation import (
    binary_classification_metrics,
    select_provisional_champion,
    validate_prediction_frame,
)
from paisa.experiments.experiment_config import (
    DEFAULT_SYMBOL,
    MODEL_VARIANTS_BY_ID,
    RANDOM_SEED,
    SOURCE_FREQUENCY,
    TARGET_POSITIVE_CLASS_CONDITION,
)
from paisa.experiments.lstm import (
    LSTMPreprocessor,
    LSTMSequenceBatch,
    LSTMTrainingConfig,
    build_lstm_model,
    build_lstm_sequence_batch,
    lstm_runtime_metadata,
    register_fitted_lstm,
    set_lstm_determinism,
)
from paisa.experiments.lstm_comparison import (
    align_predictions_on_common_dates,
    build_comparison_metric_row,
    build_fixed_experiment_config,
    calibration_information,
)
from paisa.experiments.runner import ExperimentRun, run_experiment
from paisa.experiments.trainer import PreparedTrainingData, prepare_training_data


ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_BASELINE_VARIANTS = ("M-001", "M-002", "M-003", "M-004")
_CANONICAL_TARGET_COUNTS = {"train": 853, "validation": 182, "test": 184}
_CANONICAL_SEQUENCE_COUNTS = {"train": 834, "validation": 182, "test": 184}
_REPRODUCTION_RTOL = 1e-6
_REPRODUCTION_ATOL = 1e-7
_SOURCE_FINGERPRINT_INPUTS = (
    "scripts/run_lstm_price_only_experiments.py",
    "src/paisa/backtesting/benchmarks.py",
    "src/paisa/backtesting/engine.py",
    "src/paisa/backtesting/metrics.py",
    "src/paisa/backtesting/reports.py",
    "src/paisa/experiments/artifacts.py",
    "src/paisa/experiments/evaluation.py",
    "src/paisa/experiments/experiment_config.py",
    "src/paisa/experiments/lstm.py",
    "src/paisa/experiments/lstm_comparison.py",
    "src/paisa/experiments/lstm_runner.py",
    "src/paisa/experiments/portfolio_registry.py",
    "src/paisa/experiments/registry.py",
    "src/paisa/experiments/runner.py",
    "src/paisa/experiments/trainer.py",
    "src/paisa/experiments/training_features.py",
    "src/paisa/experiments/training_targets.py",
    "src/paisa/features.py",
    "src/paisa/trading/execution.py",
    "src/paisa/trading/models.py",
    "src/paisa/trading/policy.py",
    "src/paisa/trading/portfolio.py",
)


@dataclass(frozen=True, slots=True)
class LSTMExperimentVariant:
    """One fixed recurrent experiment identity and complete fit configuration."""

    variant_id: str
    label: str
    training_config: LSTMTrainingConfig

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.variant_id):
            raise ValueError("variant_id must be filesystem-safe")
        if not self.label.strip():
            raise ValueError("label must be non-empty")
        if self.training_config.sequence_length != 20:
            raise ValueError("LSTM comparison sequence_length must be 20")
        if self.training_config.units != 64:
            raise ValueError("LSTM comparison units must be 64")
        if self.training_config.second_layer_units is not None:
            raise ValueError("LSTM comparison variants must use one recurrent layer")
        if self.training_config.class_weight is not None:
            raise ValueError("LSTM comparison class_weight must be null")


LSTM_VARIANTS: tuple[LSTMExperimentVariant, ...] = (
    LSTMExperimentVariant(
        variant_id="M-005",
        label=(
            "LSTM | 20-session sequence | baseline recurrent configuration | frozen"
        ),
        training_config=LSTMTrainingConfig(
            sequence_length=20,
            units=64,
            dropout=0.20,
            recurrent_dropout=0.0,
            kernel_l2=0.0,
            recurrent_l2=0.0,
            second_layer_units=None,
            learning_rate=0.001,
            epochs=50,
            batch_size=32,
            patience=6,
            random_seed=RANDOM_SEED,
            class_weight=None,
            verbose=1,
        ),
    ),
    LSTMExperimentVariant(
        variant_id="M-006",
        label=(
            "LSTM | 20-session sequence | regularized recurrent configuration | frozen"
        ),
        training_config=LSTMTrainingConfig(
            sequence_length=20,
            units=64,
            dropout=0.40,
            recurrent_dropout=0.10,
            kernel_l2=0.0001,
            recurrent_l2=0.0001,
            second_layer_units=None,
            learning_rate=0.001,
            epochs=50,
            batch_size=32,
            patience=6,
            random_seed=RANDOM_SEED,
            class_weight=None,
            verbose=1,
        ),
    ),
)
LSTM_VARIANTS_BY_ID: Mapping[str, LSTMExperimentVariant] = MappingProxyType(
    {variant.variant_id: variant for variant in LSTM_VARIANTS}
)


@dataclass(frozen=True, slots=True)
class DirectoryInventoryEvidence:
    """Checksum baseline for a retained vertical-slice directory."""

    directory: Path
    manifest_sha256: str
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class BaselineDataProvenance:
    """Checksum-bound acquisition fields inherited from M-001 through M-004."""

    provider: str
    requested_start: str
    requested_end: str
    input_schema_version: str
    training_run_id: str
    training_run_manifest_sha256: str
    source_files: tuple[tuple[str, str], ...]

    def registration_extra_fields(self) -> dict[str, Any]:
        return {
            "upstream_baseline": {
                "training_run_id": self.training_run_id,
                "training_run_manifest_sha256": self.training_run_manifest_sha256,
                "source_files": [
                    {"path": path, "sha256": checksum}
                    for path, checksum in self.source_files
                ],
            }
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "requested_start": self.requested_start,
            "requested_end": self.requested_end,
            "input_schema_version": self.input_schema_version,
            **self.registration_extra_fields(),
        }


@dataclass(slots=True)
class _FittedState:
    variant: LSTMExperimentVariant
    registered_model_id: str
    model: Any
    preprocessor: LSTMPreprocessor
    train_batch: LSTMSequenceBatch
    validation_batch: LSTMSequenceBatch
    test_batch: LSTMSequenceBatch | None
    probabilities_by_split: dict[str, np.ndarray]
    metrics_by_split: dict[str, dict[str, Any]]
    calibration_by_split: dict[str, dict[str, Any]]
    history: dict[str, Any]
    training_metadata: dict[str, Any]


def _safe_run_id(value: object) -> str:
    run_id = str(value).strip()
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or digit and contain only letters, "
            "digits, '.', '_', and '-'"
        )
    return run_id


def _load_json(path: Path, *, field_name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field_name} is missing or not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _safe_relative(value: object, *, field_name: str) -> str:
    raw = str(value)
    if "\\" in raw:
        raise ValueError(f"{field_name} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{field_name} must be a safe relative path")
    return path.as_posix()


def _regular_files(root: Path, *, exclude: set[str]) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = item.relative_to(root).as_posix()
        if item.is_symlink():
            raise ValueError(f"Retained artifacts must not contain symlinks: {relative}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError(f"Retained artifact is not a regular file: {relative}")
        if relative not in exclude:
            files[relative] = item
    return files


def validate_vertical_slice_artifacts(
    directory: str | Path,
) -> DirectoryInventoryEvidence:
    """Validate the retained vertical slice against its exact SHA-256 inventory."""

    candidate = Path(directory)
    if candidate.is_symlink():
        raise ValueError("Baseline vertical-slice directory must not be a symlink")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError(f"Baseline vertical-slice directory is missing: {root}")
    manifest_path = root / "manifest.json"
    manifest = _load_json(manifest_path, field_name="Vertical-slice manifest")
    raw_entries = manifest.get("artifacts")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Vertical-slice manifest artifacts must be a non-empty array")

    parsed: list[tuple[str, str]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"path", "sha256"}:
            raise ValueError(
                "Every vertical-slice artifact entry must contain exactly path and sha256"
            )
        relative = _safe_relative(
            raw_entry["path"], field_name=f"artifacts[{index}].path"
        )
        checksum = str(raw_entry["sha256"])
        if not _SHA256.fullmatch(checksum):
            raise ValueError(f"Invalid vertical-slice SHA-256 for {relative}")
        parsed.append((relative, checksum))
    if parsed != sorted(parsed) or len({path for path, _ in parsed}) != len(parsed):
        raise ValueError("Vertical-slice artifact inventory must be sorted and unique")

    actual = _regular_files(root, exclude={"manifest.json"})
    if set(actual) != {path for path, _ in parsed}:
        expected = {path for path, _ in parsed}
        raise ValueError(
            "Vertical-slice artifact paths must exactly match its manifest; "
            f"missing={sorted(expected - set(actual))}, "
            f"unexpected={sorted(set(actual) - expected)}"
        )
    for relative, expected in parsed:
        actual_checksum = sha256_file(actual[relative])
        if actual_checksum != expected:
            raise ValueError(
                f"Vertical-slice artifact checksum mismatch: {relative}"
            )
    return DirectoryInventoryEvidence(
        directory=root,
        manifest_sha256=sha256_file(manifest_path),
        files=tuple(parsed),
    )


def assert_vertical_slice_artifacts_unchanged(
    evidence: DirectoryInventoryEvidence,
) -> None:
    """Reassert the exact retained vertical-slice baseline."""

    current = validate_vertical_slice_artifacts(evidence.directory)
    if (
        current.manifest_sha256 != evidence.manifest_sha256
        or current.files != evidence.files
    ):
        raise RuntimeError("Baseline vertical-slice artifacts changed during the run")


def _validate_baseline_relationship(
    training: TrainingRunEvidence,
    vertical: DirectoryInventoryEvidence,
) -> None:
    reference = _load_json(
        vertical.directory / "model_training_run_reference.json",
        field_name="Vertical-slice training-run reference",
    )
    if str(reference.get("run_manifest_sha256")) != training.run_manifest_sha256:
        raise ValueError(
            "Baseline vertical slice does not reference the selected training manifest"
        )
    referenced_id = str(reference.get("artifact_directory", "")).rstrip("/").split("/")[-1]
    if referenced_id != training.run_id:
        raise ValueError(
            "Baseline vertical slice does not reference the selected training run_id"
        )


def _validate_output_separation(
    output_root: Path,
    training_root: Path,
    vertical_root: Path,
) -> None:
    """Reject output paths physically equal to or nested below a baseline.

    Walking existing ancestors with ``samefile`` keeps the check correct on
    case-insensitive filesystems even when the caller spells a path using a
    different case. Nonexistent descendants are still rejected before they
    can be created inside a retained baseline.
    """

    for baseline in (training_root, vertical_root):
        candidate = output_root
        while True:
            if os.path.lexists(candidate):
                try:
                    if os.path.samefile(candidate, baseline):
                        raise ValueError(
                            "output_dir must not be inside a retained baseline directory"
                        )
                except FileNotFoundError:
                    pass
            if candidate.parent == candidate:
                break
            candidate = candidate.parent


def _baseline_price_checksum(evidence: TrainingRunEvidence) -> str:
    expected = dict(evidence.artifact_sha256)
    relative = "frozen_input/OGDC_prices.csv"
    if relative not in expected:
        raise ValueError(
            "Baseline training inventory is missing frozen_input/OGDC_prices.csv"
        )
    return expected[relative]


def _freeze_verified_prices(
    source_path: Path,
    stage_dir: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Copy the already-authorized price bytes once and verify the snapshot."""

    if not _SHA256.fullmatch(str(expected_sha256)):
        raise ValueError("Baseline price checksum must be a lowercase SHA-256 digest")
    destination_dir = stage_dir / "frozen_input"
    destination_dir.mkdir(parents=True, exist_ok=False)
    destination = destination_dir / "OGDC_prices.csv"
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source_path, flags)
    digest = hashlib.sha256()
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError("prices_path must resolve to a regular file")
        source_file = os.fdopen(descriptor, "rb")
        descriptor = -1
        with source_file as source, destination.open("xb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    actual = digest.hexdigest()
    if actual != expected_sha256 or sha256_file(destination) != expected_sha256:
        raise ValueError(
            "prices_path does not match the checksum-bound baseline prices"
        )
    return destination


def _entry_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.lstat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
    )


def _publish_staged_zip(
    staged_zip: Path, final_zip: Path
) -> None:
    """Atomically publish a complete sibling ZIP without replacement."""

    staged_identity = _entry_identity(staged_zip)
    if not stat.S_ISREG(staged_identity[2]) or staged_zip.is_symlink():
        raise ValueError("Staged ZIP must be a regular non-symlink file")
    atomic_rename_noreplace(staged_zip, final_zip)
    published_identity = _entry_identity(final_zip)
    if published_identity != staged_identity:
        raise RuntimeError("Published ZIP identity changed during publication")


def _publish_staged_artifacts(
    *,
    stage_dir: Path,
    staged_zip: Path | None,
    final_run_dir: Path,
    final_zip: Path | None,
) -> None:
    """Publish ZIP first and use the run directory as the final commit point."""

    if os.path.lexists(final_run_dir):
        raise FileExistsError(
            f"Run directory appeared during staging: {final_run_dir}"
        )
    if final_zip is not None and os.path.lexists(final_zip):
        raise FileExistsError(
            f"ZIP appeared during staging; refusing to overwrite: {final_zip}"
        )
    if (staged_zip is None) != (final_zip is None):
        raise ValueError("Staged and final ZIP paths must either both be set or omitted")
    stage_identity = _entry_identity(stage_dir)
    if not stat.S_ISDIR(stage_identity[2]) or stage_dir.is_symlink():
        raise ValueError("Staged run must be a real directory")

    if staged_zip is not None:
        assert final_zip is not None
        _publish_staged_zip(staged_zip, final_zip)
    # This is deliberately the last publication operation. A visible run
    # directory therefore always has its advertised ZIP. If this rename fails
    # or the process stops first, the complete ZIP is retained as an orphan;
    # automatic deletion would introduce a check/unlink ownership race.
    atomic_rename_noreplace(stage_dir, final_run_dir)


def _exact_iso_date(value: object, *, field_name: str) -> str:
    normalized = str(value)
    try:
        parsed = pd.Timestamp(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO date") from exc
    if parsed.strftime("%Y-%m-%d") != normalized:
        raise ValueError(f"{field_name} must use exact YYYY-MM-DD form")
    return normalized


def _load_baseline_data_provenance(
    evidence: TrainingRunEvidence,
) -> BaselineDataProvenance:
    """Read common acquisition fields only from checksum-inventoried baselines."""

    expected = dict(evidence.artifact_sha256)
    data_manifest_path = "data_manifest.json"
    if data_manifest_path not in expected:
        raise ValueError("Baseline inventory is missing data_manifest.json")
    data_manifest_file = evidence.run_directory / data_manifest_path
    if sha256_file(data_manifest_file) != expected[data_manifest_path]:
        raise ValueError("Baseline data_manifest.json checksum changed")
    data_manifest = _load_json(
        data_manifest_file, field_name="Baseline data manifest"
    )

    model_ids = dict(evidence.model_ids)
    if set(model_ids) != set(_BASELINE_VARIANTS):
        raise ValueError("Baseline provenance requires exactly M-001 through M-004")
    source_files: list[tuple[str, str]] = [
        (data_manifest_path, expected[data_manifest_path])
    ]
    common: tuple[str, str, str, str] | None = None
    for variant_id in _BASELINE_VARIANTS:
        registered_id = model_ids[variant_id]
        relative = f"models/{registered_id}/manifest.json"
        checksum = expected.get(relative)
        if checksum is None:
            raise ValueError(
                f"Baseline inventory is missing {variant_id} model manifest"
            )
        model_manifest_file = evidence.run_directory / relative
        if sha256_file(model_manifest_file) != checksum:
            raise ValueError(f"Baseline {variant_id} model manifest checksum changed")
        model_manifest = _load_json(
            model_manifest_file,
            field_name=f"Baseline {variant_id} model manifest",
        )
        raw = model_manifest.get("data_provenance")
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"Baseline {variant_id} model manifest lacks data_provenance"
            )
        provider = str(raw.get("provider", "")).strip()
        input_schema = str(raw.get("input_schema_version", "")).strip()
        if not provider or not input_schema:
            raise ValueError(
                f"Baseline {variant_id} provider and input schema must be non-empty"
            )
        requested_start = _exact_iso_date(
            raw.get("requested_start"),
            field_name=f"Baseline {variant_id} requested_start",
        )
        requested_end = _exact_iso_date(
            raw.get("requested_end"),
            field_name=f"Baseline {variant_id} requested_end",
        )
        if pd.Timestamp(requested_start) > pd.Timestamp(requested_end):
            raise ValueError(
                f"Baseline {variant_id} requested_start is later than requested_end"
            )
        candidate = (provider, requested_start, requested_end, input_schema)
        if common is None:
            common = candidate
        elif candidate != common:
            raise ValueError(
                "Baseline model manifests disagree on provider, requested range, "
                "or input schema"
            )
        source_files.append((relative, checksum))

    assert common is not None
    provider, requested_start, requested_end, input_schema = common
    manifest_provider = str(data_manifest.get("provider", "")).strip()
    if manifest_provider != provider:
        raise ValueError(
            "Baseline data and model manifests disagree on the data provider"
        )
    manifest_requested_start = _exact_iso_date(
        data_manifest.get("requested_start"),
        field_name="Baseline data manifest requested_start",
    )
    if manifest_requested_start != requested_start:
        raise ValueError(
            "Baseline data and model manifests disagree on requested_start"
        )
    return BaselineDataProvenance(
        provider=provider,
        requested_start=requested_start,
        requested_end=requested_end,
        input_schema_version=input_schema,
        training_run_id=evidence.run_id,
        training_run_manifest_sha256=evidence.run_manifest_sha256,
        source_files=tuple(source_files),
    )


def _source_fingerprint_paths(repo_root: Path) -> tuple[Path, ...]:
    paths = tuple(repo_root / relative for relative in _SOURCE_FINGERPRINT_INPUTS)
    missing = [
        path.relative_to(repo_root).as_posix() for path in paths if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"LSTM source-fingerprint inputs are missing: {missing}"
        )
    return paths


def _split_target_dates(frame: pd.DataFrame) -> list[str]:
    return pd.to_datetime(frame["target_date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    ).tolist()


def _validate_prepared_against_baseline(
    prepared: PreparedTrainingData,
    training: TrainingRunEvidence,
    prices_path: Path,
) -> dict[str, Any]:
    if sha256_file(prices_path) != _baseline_price_checksum(training):
        raise ValueError("prices_path does not match the checksum-bound baseline prices")
    data_manifest = _load_json(
        training.run_directory / "data_manifest.json",
        field_name="Baseline data manifest",
    )
    if str(data_manifest.get("dataset_sha256")) != prepared.dataset_sha256:
        raise ValueError("Prepared model rows do not match the baseline dataset checksum")
    split_manifest = _load_json(
        training.run_directory / "split_manifest.json",
        field_name="Baseline split manifest",
    )
    raw_counts = split_manifest.get("row_counts")
    raw_dates = split_manifest.get("target_dates")
    if not isinstance(raw_counts, Mapping) or not isinstance(raw_dates, Mapping):
        raise ValueError("Baseline split manifest is missing counts or target dates")
    for name, frame in prepared.splits.as_dict().items():
        if int(raw_counts.get(name, -1)) != len(frame):
            raise ValueError(f"Prepared {name} rows do not match the baseline split count")
        if list(raw_dates.get(name, [])) != _split_target_dates(frame):
            raise ValueError(f"Prepared {name} target dates do not match the baseline")
    return split_manifest


def _load_baseline_predictions(
    evidence: TrainingRunEvidence,
) -> dict[str, pd.DataFrame]:
    model_ids = dict(evidence.model_ids)
    prediction_paths = dict(evidence.prediction_paths)
    if set(model_ids) != set(_BASELINE_VARIANTS) or set(prediction_paths) != set(
        _BASELINE_VARIANTS
    ):
        raise ValueError("Baseline training run must contain exactly M-001 through M-004")
    frames: dict[str, pd.DataFrame] = {}
    for variant_id in _BASELINE_VARIANTS:
        frame = validate_prediction_frame(
            pd.read_csv(evidence.run_directory / prediction_paths[variant_id])
        )
        if frame["model_id"].nunique() != 1 or frame["model_id"].iloc[0] != model_ids[
            variant_id
        ]:
            raise ValueError(f"{variant_id} baseline prediction identity is inconsistent")
        frames[variant_id] = frame
    return frames


def _tensorflow_module(tensorflow_module: Any | None) -> Any:
    if tensorflow_module is not None:
        return tensorflow_module
    try:
        return importlib.import_module("tensorflow")
    except ImportError as exc:
        raise RuntimeError(
            "The LSTM experiment runner requires optional TensorFlow; it never "
            "downloads dependencies or models."
        ) from exc


def _configure_device(tf: Any, requested: str) -> None:
    normalized = str(requested).strip().lower()
    if normalized not in {"auto", "cpu", "gpu", "metal"}:
        raise ValueError("device must be one of: auto, cpu, gpu, metal")
    list_devices = getattr(getattr(tf, "config", None), "list_physical_devices", None)
    devices = list_devices() if callable(list_devices) else []
    gpu_devices = [
        item
        for item in devices
        if str(getattr(item, "device_type", "")).upper() == "GPU"
    ]
    get_visible = getattr(getattr(tf, "config", None), "get_visible_devices", None)
    if callable(get_visible):
        visible_gpu_devices = list(get_visible("GPU"))
    else:
        visible_gpu_devices = gpu_devices
    if normalized == "cpu":
        set_visible = getattr(getattr(tf, "config", None), "set_visible_devices", None)
        if callable(set_visible):
            set_visible([], "GPU")
    elif normalized == "gpu" and not visible_gpu_devices:
        raise RuntimeError("Requested gpu device is not visible to TensorFlow")
    elif normalized == "metal":
        if platform.system() != "Darwin":
            raise RuntimeError(
                "Requested metal device is supported only on macOS (Darwin)"
            )
        if not visible_gpu_devices:
            raise RuntimeError(
                "Requested metal device requires a TensorFlow-visible GPU on macOS"
            )


def _registered_model_id(
    variant: LSTMExperimentVariant,
    prepared: PreparedTrainingData,
    run_id: str,
    invocation_id: str,
) -> str:
    payload = json.dumps(
        {
            "variant_id": variant.variant_id,
            "configuration": variant.training_config.to_dict(),
            "dataset_sha256": prepared.dataset_sha256,
            "features": list(prepared.feature_columns),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    semantic = hashlib.sha256(payload).hexdigest()[:16]
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    return f"{variant.variant_id}-{semantic}-{run_digest}-{invocation_id}"


def _validated_probability_up(
    model: Any, values: np.ndarray, *, variant_id: str, split_name: str
) -> np.ndarray:
    result = np.asarray(model.predict(values, verbose=0), dtype=float)
    if result.ndim == 2 and result.shape[1:] == (1,):
        result = result[:, 0]
    elif result.ndim != 1:
        raise ValueError(f"{variant_id} {split_name} probabilities must be one-dimensional")
    if len(result) != len(values):
        raise ValueError(f"{variant_id} {split_name} probability count is incorrect")
    if not np.isfinite(result).all() or ((result < 0) | (result > 1)).any():
        raise ValueError(
            f"{variant_id} {split_name} probabilities must be finite and in [0, 1]"
        )
    return result


def _split_metrics(batch: LSTMSequenceBatch, probability_up: np.ndarray) -> dict[str, Any]:
    if batch.targets is None:
        raise ValueError("LSTM comparison batches must retain their binary targets")
    predicted = (probability_up > (1.0 - probability_up)).astype(int)
    return binary_classification_metrics(batch.targets, predicted, probability_up)


def _leaderboard_row(
    state: _FittedState, *, split_name: str
) -> dict[str, Any]:
    return {
        "model_id": state.variant.variant_id,
        "variant_id": state.variant.variant_id,
        "registered_model_id": state.registered_model_id,
        "label": state.variant.label,
        "algorithm": "lstm",
        **state.metrics_by_split[split_name],
    }


def _ordered_leaderboard(states: Sequence[_FittedState], split_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [_leaderboard_row(state, split_name=split_name) for state in states]
    ).sort_values(
        ["balanced_accuracy", "f1", "accuracy", "model_id"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _append_event(
    events: list[dict[str, Any]], event: str, **details: Any
) -> None:
    events.append(
        {
            "sequence": len(events) + 1,
            "occurred_at": utc_now_z(),
            "event": event,
            **to_jsonable(details),
        }
    )


def _history_rows(states: Sequence[_FittedState]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for state in states:
        sequence_values = {
            key: value
            for key, value in state.history.items()
            if isinstance(value, list)
        }
        epochs = max((len(value) for value in sequence_values.values()), default=0)
        for epoch_index in range(epochs):
            record: dict[str, Any] = {
                "variant_id": state.variant.variant_id,
                "registered_model_id": state.registered_model_id,
                "epoch": epoch_index + 1,
            }
            for name, values in sorted(sequence_values.items()):
                record[name] = values[epoch_index] if epoch_index < len(values) else None
            records.append(record)
    return pd.DataFrame(records)


def _confusion_frame(metrics_by_split: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        matrix = metrics_by_split[split_name]["confusion_matrix"]
        for actual_class in (0, 1):
            records.append(
                {
                    "split": split_name,
                    "actual_class": actual_class,
                    "predicted_down": int(matrix[actual_class][0]),
                    "predicted_up": int(matrix[actual_class][1]),
                }
            )
    return pd.DataFrame(records)


def _save_root_preprocessing(stage_dir: Path, state: _FittedState) -> None:
    destination = stage_dir / "preprocessing" / state.variant.variant_id
    destination.mkdir(parents=True, exist_ok=False)
    joblib.dump(state.preprocessor, destination / "preprocessor.joblib")
    joblib.dump(state.preprocessor.imputer, destination / "imputer.joblib")
    joblib.dump(state.preprocessor.scaler, destination / "scaler.joblib")
    write_json(
        {
            "variant_id": state.variant.variant_id,
            "registered_model_id": state.registered_model_id,
            "fit_scope": "training_rows_only",
            "feature_columns": list(state.preprocessor.feature_columns),
        },
        destination / "metadata.json",
    )


def _probability_sha256(values: np.ndarray) -> str:
    normalized = np.asarray(values, dtype="<f8")
    if normalized.ndim != 1:
        raise ValueError("Probability checksum input must be one-dimensional")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()


def _verify_validation_save_load_reproduction(
    *,
    state: _FittedState,
    prepared: PreparedTrainingData,
    model_dir: Path,
    models_root: Path,
    tf: Any,
) -> dict[str, Any]:
    """Reload the registered bundle and reproduce validation only, never test."""

    loaded_bundle, loaded_manifest = load_model_artifact(
        model_dir, trusted_root=models_root
    )
    cached = np.asarray(
        state.probabilities_by_split["validation"], dtype=float
    )
    reproduced = np.asarray(
        loaded_bundle.predict_probability_up(
            prepared.model_frame,
            state.validation_batch.rows,
            tensorflow_module=tf,
            verbose=0,
        ),
        dtype=float,
    )
    if reproduced.ndim != 1 or reproduced.shape != cached.shape:
        raise AssertionError(
            f"{state.variant.variant_id} reloaded validation probability shape changed"
        )
    if not np.isfinite(reproduced).all() or (
        (reproduced < 0.0) | (reproduced > 1.0)
    ).any():
        raise AssertionError(
            f"{state.variant.variant_id} reloaded validation probabilities are invalid"
        )
    exact_match = bool(np.array_equal(cached, reproduced))
    tolerance_match = bool(
        np.allclose(
            cached,
            reproduced,
            rtol=_REPRODUCTION_RTOL,
            atol=_REPRODUCTION_ATOL,
            equal_nan=False,
        )
    )
    max_absolute_error = float(np.max(np.abs(cached - reproduced)))
    if not tolerance_match:
        raise AssertionError(
            f"{state.variant.variant_id} saved bundle did not reproduce cached "
            f"validation probabilities; max_absolute_error={max_absolute_error}"
        )
    return {
        "schema_version": "1.0",
        "variant_id": state.variant.variant_id,
        "registered_model_id": state.registered_model_id,
        "artifact_sha256": loaded_manifest["artifact_sha256"],
        "split": "validation",
        "row_count": int(len(cached)),
        "cached_probability_sha256": _probability_sha256(cached),
        "reloaded_probability_sha256": _probability_sha256(reproduced),
        "exact_match": exact_match,
        "tolerance_match": tolerance_match,
        "relative_tolerance": _REPRODUCTION_RTOL,
        "absolute_tolerance": _REPRODUCTION_ATOL,
        "max_absolute_error": max_absolute_error,
        "test_split_inference_performed": False,
        "status": "PASS",
    }


def _fit_until_validation_freeze(
    *,
    prepared: PreparedTrainingData,
    stage_dir: Path,
    run_id: str,
    invocation_id: str,
    tf: Any,
    requested_device: str,
    events: list[dict[str, Any]],
) -> tuple[list[_FittedState], dict[str, Any]]:
    """Fit both variants and persist the validation-only freeze before test use."""

    states: list[_FittedState] = []
    checkpoint_root = stage_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=False)
    for variant in LSTM_VARIANTS:
        config = variant.training_config
        tf.keras.backend.clear_session()
        determinism = set_lstm_determinism(
            config.random_seed, tensorflow_module=tf
        )
        preprocessor = LSTMPreprocessor.fit(
            prepared.splits.train, prepared.feature_columns
        )
        train_batch = build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.train,
            preprocessor=preprocessor,
            feature_columns=prepared.feature_columns,
            sequence_length=config.sequence_length,
            require_every_target=False,
        )
        validation_batch = build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.validation,
            preprocessor=preprocessor,
            feature_columns=prepared.feature_columns,
            sequence_length=config.sequence_length,
            require_every_target=True,
        )
        if train_batch.targets is None or validation_batch.targets is None:
            raise ValueError("Training and validation sequences must contain targets")

        model = build_lstm_model(
            config,
            len(prepared.feature_columns),
            tensorflow_module=tf,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.patience,
            restore_best_weights=True,
        )
        checkpoint_path = checkpoint_root / f"{variant.variant_id}.weights.h5"
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        )
        _append_event(events, "training_started", variant_id=variant.variant_id)
        started = time.perf_counter()
        fitted_history = model.fit(
            train_batch.values,
            train_batch.targets,
            validation_data=(validation_batch.values, validation_batch.targets),
            epochs=config.epochs,
            batch_size=config.batch_size,
            shuffle=False,
            callbacks=[early_stopping, model_checkpoint],
            class_weight=None,
            verbose=config.verbose,
        )
        duration = time.perf_counter() - started
        if not checkpoint_path.is_file() or checkpoint_path.is_symlink():
            raise RuntimeError(
                f"Validation checkpoint was not saved for {variant.variant_id}"
            )
        model.load_weights(checkpoint_path)
        history = to_jsonable(dict(fitted_history.history))
        train_probability = _validated_probability_up(
            model,
            train_batch.values,
            variant_id=variant.variant_id,
            split_name="train",
        )
        validation_probability = _validated_probability_up(
            model,
            validation_batch.values,
            variant_id=variant.variant_id,
            split_name="validation",
        )
        train_metrics = _split_metrics(train_batch, train_probability)
        validation_metrics = _split_metrics(validation_batch, validation_probability)
        validation_losses = history.get("val_loss", [])
        epochs_completed = max(
            (len(value) for value in history.values() if isinstance(value, list)),
            default=0,
        )
        best_epoch = (
            int(np.argmin(np.asarray(validation_losses, dtype=float))) + 1
            if validation_losses
            else None
        )
        stopped_epoch = int(getattr(early_stopping, "stopped_epoch", 0))
        runtime = lstm_runtime_metadata(tensorflow_module=tf, model=model)
        training_metadata = {
            "variant_id": variant.variant_id,
            "label": variant.label,
            "requested_device": requested_device,
            "determinism": determinism.to_dict(),
            "runtime": runtime,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "tensorflow_version": getattr(tf, "__version__", None),
            "training_duration_seconds": float(duration),
            "epochs_requested": config.epochs,
            "epochs_completed": epochs_completed,
            "best_validation_epoch": best_epoch,
            "early_stopping_epoch": stopped_epoch + 1 if stopped_epoch > 0 else None,
            "batch_size": config.batch_size,
            "optimizer": "Adam",
            "learning_rate": config.learning_rate,
            "class_weight": None,
            "shuffle": False,
            "sequence_length": config.sequence_length,
            "architecture": {
                "recurrent_layers": 1,
                "units": config.units,
                "output": "Dense(1, activation='sigmoid')",
            },
            "parameter_count": runtime.get("parameter_count"),
            "early_stopping": {
                "monitor": "val_loss",
                "mode": "min",
                "restore_best_weights": True,
                "patience": config.patience,
            },
            "model_checkpoint": {
                "monitor": "val_loss",
                "mode": "min",
                "save_best_only": True,
                "save_weights_only": True,
                "path": checkpoint_path.relative_to(stage_dir).as_posix(),
                "sha256": sha256_file(checkpoint_path),
                "loaded_after_fit": True,
            },
        }
        state = _FittedState(
            variant=variant,
            registered_model_id=_registered_model_id(
                variant, prepared, run_id, invocation_id
            ),
            model=model,
            preprocessor=preprocessor,
            train_batch=train_batch,
            validation_batch=validation_batch,
            test_batch=None,
            probabilities_by_split={
                "train": train_probability,
                "validation": validation_probability,
            },
            metrics_by_split={
                "train": train_metrics,
                "validation": validation_metrics,
            },
            calibration_by_split={
                "train": calibration_information(
                    train_batch.targets, train_probability
                ),
                "validation": calibration_information(
                    validation_batch.targets, validation_probability
                ),
            },
            history=history,
            training_metadata=training_metadata,
        )
        states.append(state)
        _append_event(
            events,
            "validation_evaluation_completed",
            variant_id=variant.variant_id,
            test_predictions_performed=0,
        )

    validation_leaderboard = _ordered_leaderboard(states, "validation")
    validation_leaderboard.to_csv(
        stage_dir / "validation_leaderboard.csv", index=False
    )
    preferred = select_provisional_champion(validation_leaderboard)
    freeze_decision = {
        "schema_version": "1.0",
        "frozen_at": utc_now_z(),
        "selection_data": "validation_only",
        "test_metrics_consulted": False,
        "validation_preferred_variant": preferred["model_id"],
        "selection_rule": preferred["selection_rule"],
        "validation_metrics": {
            "balanced_accuracy": preferred["validation_balanced_accuracy"],
            "f1": preferred["validation_f1"],
            "accuracy": preferred["validation_accuracy"],
        },
    }
    write_json(freeze_decision, stage_dir / "validation_freeze_decision.json")
    _append_event(
        events,
        "validation_selection_frozen",
        validation_preferred_variant=preferred["model_id"],
        test_metrics_consulted=False,
    )
    write_json(
        {"schema_version": "1.0", "events": events},
        stage_dir / "selection_events.json",
    )
    return states, freeze_decision


def _evaluate_test_once(
    *,
    states: Sequence[_FittedState],
    prepared: PreparedTrainingData,
    events: list[dict[str, Any]],
    stage_dir: Path,
) -> pd.DataFrame:
    """Evaluate each already-frozen model on test exactly once."""

    for state in states:
        config = state.variant.training_config
        _append_event(
            events,
            "test_evaluation_started",
            variant_id=state.variant.variant_id,
        )
        test_batch = build_lstm_sequence_batch(
            prepared.model_frame,
            prepared.splits.test,
            preprocessor=state.preprocessor,
            feature_columns=prepared.feature_columns,
            sequence_length=config.sequence_length,
            require_every_target=True,
        )
        if test_batch.targets is None:
            raise ValueError("Test sequences must contain targets")
        test_probability = _validated_probability_up(
            state.model,
            test_batch.values,
            variant_id=state.variant.variant_id,
            split_name="test",
        )
        state.test_batch = test_batch
        state.probabilities_by_split["test"] = test_probability
        state.metrics_by_split["test"] = _split_metrics(
            test_batch, test_probability
        )
        state.calibration_by_split["test"] = calibration_information(
            test_batch.targets, test_probability
        )
        _append_event(
            events,
            "test_evaluation_completed",
            variant_id=state.variant.variant_id,
            prediction_calls_for_test=1,
        )
    write_json(
        {"schema_version": "1.0", "events": events},
        stage_dir / "selection_events.json",
    )
    leaderboard = _ordered_leaderboard(states, "test")
    leaderboard.to_csv(stage_dir / "test_leaderboard.csv", index=False)
    return leaderboard


def _register_states(
    *,
    states: Sequence[_FittedState],
    prepared: PreparedTrainingData,
    stage_dir: Path,
    repo_root: Path,
    invocation_id: str,
    baseline_provenance: BaselineDataProvenance,
    tf: Any,
    events: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    dict[str, pd.DataFrame],
    dict[str, dict[str, Any]],
]:
    """Register both fitted models without invoking inference again."""

    models_root = stage_dir / "models"
    models_root.mkdir(parents=True, exist_ok=False)
    registered_results: dict[str, Any] = {}
    predictions: dict[str, pd.DataFrame] = {}
    reproduction_checks: dict[str, dict[str, Any]] = {}
    source_paths = _source_fingerprint_paths(repo_root)

    for state in states:
        if state.test_batch is None or set(state.probabilities_by_split) != {
            "train",
            "validation",
            "test",
        }:
            raise RuntimeError("All frozen probabilities must exist before registration")
        extra_files = {
            "variant_config.json": {
                "variant_id": state.variant.variant_id,
                "registered_model_id": state.registered_model_id,
                "label": state.variant.label,
                "algorithm": "lstm",
                "configuration": state.variant.training_config.to_dict(),
            },
            "confusion_matrix.csv": _confusion_frame(state.metrics_by_split),
            "calibration.json": state.calibration_by_split,
        }
        result = register_fitted_lstm(
            prepared=prepared,
            model=state.model,
            scaler=state.preprocessor.scaler,
            preprocessor=state.preprocessor,
            config=state.variant.training_config,
            models_root=models_root,
            repo_root=repo_root,
            model_id=state.registered_model_id,
            provider=baseline_provenance.provider,
            requested_start=baseline_provenance.requested_start,
            requested_end=baseline_provenance.requested_end,
            input_schema_version=baseline_provenance.input_schema_version,
            data_provenance_extra_fields=(
                baseline_provenance.registration_extra_fields()
            ),
            source_paths=source_paths,
            history=state.history,
            probabilities_by_split=state.probabilities_by_split,
            training_metadata=state.training_metadata,
            manifest_extra_fields={
                "variant_id": state.variant.variant_id,
                "invocation_id": invocation_id,
                "label": state.variant.label,
                "source_fingerprint_inputs": list(_SOURCE_FINGERPRINT_INPUTS),
            },
            extra_files=extra_files,
        )
        if result.model_id != state.registered_model_id:
            raise AssertionError("Registered LSTM model identity changed unexpectedly")
        if result.metrics != state.metrics_by_split:
            raise AssertionError("Registered LSTM metrics changed from frozen probabilities")
        registered_results[state.variant.variant_id] = result
        predictions[state.variant.variant_id] = validate_prediction_frame(
            result.predictions
        )
        _save_root_preprocessing(stage_dir, state)
        reproduction_checks[state.variant.variant_id] = (
            _verify_validation_save_load_reproduction(
                state=state,
                prepared=prepared,
                model_dir=result.model_dir,
                models_root=models_root,
                tf=tf,
            )
        )
        write_json(
            reproduction_checks[state.variant.variant_id],
            result.model_dir / "validation_reproduction_check.json",
        )
        _append_event(
            events,
            "validation_save_load_reproduction_completed",
            variant_id=state.variant.variant_id,
            split="validation",
            passed=True,
            test_inference_performed=False,
        )
    write_json(
        {
            "schema_version": "1.0",
            "checks": reproduction_checks,
            "test_split_inference_performed": False,
        },
        stage_dir / "validation_reproduction_checks.json",
    )
    write_json(
        {"schema_version": "1.0", "events": events},
        stage_dir / "selection_events.json",
    )
    return registered_results, predictions, reproduction_checks


def _sequence_inventory(
    prepared: PreparedTrainingData,
    states: Sequence[_FittedState],
) -> tuple[dict[str, int], list[str]]:
    counts_by_state: list[dict[str, int]] = []
    dropped_by_state: list[list[str]] = []
    original_train_dates = _split_target_dates(prepared.splits.train)
    for state in states:
        if state.test_batch is None:
            raise RuntimeError("Test batches must exist before split evidence is written")
        counts = {
            "train": len(state.train_batch.rows),
            "validation": len(state.validation_batch.rows),
            "test": len(state.test_batch.rows),
        }
        retained_train_dates = set(_split_target_dates(state.train_batch.rows))
        dropped = [
            value for value in original_train_dates if value not in retained_train_dates
        ]
        counts_by_state.append(counts)
        dropped_by_state.append(dropped)
    if any(value != counts_by_state[0] for value in counts_by_state[1:]):
        raise AssertionError("M-005 and M-006 must have identical sequence counts")
    if any(value != dropped_by_state[0] for value in dropped_by_state[1:]):
        raise AssertionError("M-005 and M-006 must drop identical training dates")

    target_counts = {
        name: len(frame) for name, frame in prepared.splits.as_dict().items()
    }
    sequence_counts = counts_by_state[0]
    dropped_dates = dropped_by_state[0]
    if target_counts == _CANONICAL_TARGET_COUNTS:
        if sequence_counts != _CANONICAL_SEQUENCE_COUNTS:
            raise AssertionError(
                "Canonical sequence counts must be 834 train, 182 validation, 184 test"
            )
        if len(dropped_dates) != 19:
            raise AssertionError("Canonical LSTM training must report 19 dropped dates")
    return sequence_counts, dropped_dates


def _write_data_and_split_evidence(
    *,
    prepared: PreparedTrainingData,
    states: Sequence[_FittedState],
    stage_dir: Path,
    prices_path: Path,
    training_evidence: TrainingRunEvidence,
    vertical_evidence: DirectoryInventoryEvidence,
    baseline_provenance: BaselineDataProvenance,
    baseline_split_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sequence_counts, dropped_dates = _sequence_inventory(prepared, states)
    target_counts = {
        name: int(len(frame)) for name, frame in prepared.splits.as_dict().items()
    }
    target_dates = {
        name: _split_target_dates(frame)
        for name, frame in prepared.splits.as_dict().items()
    }
    retained_dates = {
        "train": _split_target_dates(states[0].train_batch.rows),
        "validation": _split_target_dates(states[0].validation_batch.rows),
        "test": _split_target_dates(states[0].test_batch.rows),
    }
    split_manifest = {
        "schema_version": "1.0",
        "method": "baseline_chronological_target_date_with_past_only_sequences",
        "sequence_length": 20,
        "target_row_counts": target_counts,
        "sequence_counts": sequence_counts,
        "row_counts": sequence_counts,
        "dropped_target_dates": {
            "train": dropped_dates,
            "validation": [],
            "test": [],
        },
        "dropped_target_date_count": {
            "train": len(dropped_dates),
            "validation": 0,
            "test": 0,
        },
        "target_dates": retained_dates,
        "original_target_dates": target_dates,
        "split_membership_field": "target_date",
        "historical_context_policy": (
            "Validation and test may use earlier feature rows as context; each "
            "sequence ends at its own feature_date and contains no future row."
        ),
    }
    data_manifest = {
        "schema_version": "1.0",
        "symbol": prepared.symbol,
        "source_frequency": SOURCE_FREQUENCY,
        "prices_path": prices_path.relative_to(stage_dir).as_posix(),
        "prices_sha256": sha256_file(prices_path),
        "price_rows": int(len(prepared.prices)),
        "model_rows": int(len(prepared.model_frame)),
        "dataset_sha256": prepared.dataset_sha256,
        "feature_columns": list(prepared.feature_columns),
        "feature_count": len(prepared.feature_columns),
        "target_definition": TARGET_POSITIVE_CLASS_CONDITION,
        "baseline_training_run_id": training_evidence.run_id,
        "baseline_training_manifest_sha256": training_evidence.run_manifest_sha256,
        "baseline_vertical_manifest_sha256": vertical_evidence.manifest_sha256,
        "baseline_data_provenance": baseline_provenance.to_dict(),
    }
    split_reference = {
        "schema_version": "1.0",
        "baseline_training_run_id": training_evidence.run_id,
        "baseline_training_invocation_id": training_evidence.invocation_id,
        "baseline_training_manifest_sha256": training_evidence.run_manifest_sha256,
        "baseline_vertical_slice": vertical_evidence.directory.name,
        "baseline_vertical_manifest_sha256": vertical_evidence.manifest_sha256,
        "baseline_split_method": baseline_split_manifest.get("method"),
        "baseline_boundaries": baseline_split_manifest.get("boundaries"),
        "baseline_target_row_counts": target_counts,
        "lstm_sequence_counts": sequence_counts,
        "dropped_training_target_dates": dropped_dates,
        "validation_target_dates": retained_dates["validation"],
        "test_target_dates": retained_dates["test"],
    }
    write_json(data_manifest, stage_dir / "data_manifest.json")
    write_json(split_manifest, stage_dir / "split_manifest.json")
    write_json(split_reference, stage_dir / "split_reference.json")
    return data_manifest, split_manifest, split_reference


def _variant_details(variant_id: str) -> tuple[str, str]:
    if variant_id in LSTM_VARIANTS_BY_ID:
        variant = LSTM_VARIANTS_BY_ID[variant_id]
        return variant.label, "lstm"
    variant = MODEL_VARIANTS_BY_ID[variant_id]
    return variant.label, variant.algorithm


def _evaluation_dates(frame: pd.DataFrame) -> dict[str, str]:
    feature_dates = pd.to_datetime(frame["feature_date"], errors="raise")
    target_dates = pd.to_datetime(frame["target_date"], errors="raise")
    return {
        "feature_start": feature_dates.min().strftime("%Y-%m-%d"),
        "feature_end": feature_dates.max().strftime("%Y-%m-%d"),
        "target_start": target_dates.min().strftime("%Y-%m-%d"),
        "target_end": target_dates.max().strftime("%Y-%m-%d"),
    }


def _run_fair_comparison(
    *,
    predictions_by_variant: Mapping[str, pd.DataFrame],
    stage_dir: Path,
    run_id: str,
    prices_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, ExperimentRun], dict[str, Any]]:
    alignment = align_predictions_on_common_dates(predictions_by_variant)
    comparison_predictions = stage_dir / "comparison_predictions"
    experiment_configs = stage_dir / "experiment_configs"
    comparison_predictions.mkdir(parents=True, exist_ok=False)
    experiment_configs.mkdir(parents=True, exist_ok=False)

    fair_evidence = {
        "schema_version": "1.0",
        "alignment_key": ["ticker", "feature_date", "target_date"],
        "common_count": len(alignment.common_keys),
        "common_keys": [
            {
                "ticker": ticker,
                "feature_date": feature_date,
                "target_date": target_date,
            }
            for ticker, feature_date, target_date in alignment.common_keys
        ],
        "common_target_dates": list(alignment.common_target_dates),
        "registered_model_ids": dict(alignment.registered_model_ids),
        "original_counts": dict(alignment.original_counts),
        "common_counts": dict(alignment.common_counts),
        "dropped_target_dates": {
            key: list(value) for key, value in alignment.dropped_target_dates.items()
        },
    }
    write_json(fair_evidence, stage_dir / "fair_common_dates.json")

    metric_rows: list[dict[str, Any]] = []
    calibration: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    backtests: dict[str, ExperimentRun] = {}
    created_at = utc_now_z()
    for variant_id in sorted(alignment.aligned_predictions):
        frame = alignment.frame_for(variant_id)
        prediction_path = comparison_predictions / f"{variant_id}.csv"
        frame.to_csv(prediction_path, index=False)
        label, algorithm = _variant_details(variant_id)
        metric_rows.append(
            build_comparison_metric_row(
                variant_id=variant_id,
                predictions=frame,
                split_name="fair_common_test",
                label=label,
                algorithm=algorithm,
            )
        )
        calibration[variant_id] = calibration_information(
            frame["actual_class"].astype(int).to_numpy(),
            frame["probability_up"].astype(float).to_numpy(),
        )
        config = build_fixed_experiment_config(
            comparison_run_id=run_id,
            variant_id=variant_id,
            registered_model_id=alignment.registered_model_ids[variant_id],
            evaluation_dates=_evaluation_dates(frame),
            created_at=created_at,
            symbol=DEFAULT_SYMBOL,
        )
        configs[variant_id] = config
        config_path = experiment_configs / f"{variant_id}.json"
        write_json(config, config_path)
        backtests[variant_id] = run_experiment(
            predictions_path=prediction_path,
            prices_path=prices_path,
            config_path=config_path,
            runs_root=stage_dir / "backtests",
        )

    portfolio_ids = [
        portfolio_id
        for config in configs.values()
        for portfolio_id in (
            config["portfolio_id"],
            config["benchmark_portfolio_ids"]["cash_only"],
            config["benchmark_portfolio_ids"]["buy_and_hold"],
        )
    ]
    if len(portfolio_ids) != len(set(portfolio_ids)):
        raise AssertionError("Every comparison portfolio identity must be unique")

    metric_frame = pd.DataFrame(metric_rows).sort_values(
        ["balanced_accuracy", "f1", "accuracy", "variant_id"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    metric_frame.to_csv(stage_dir / "comparison_metrics.csv", index=False)
    write_json(calibration, stage_dir / "comparison_calibration.json")

    portfolio_rows: list[dict[str, Any]] = []
    for variant_id, completed in sorted(backtests.items()):
        metrics = completed.metrics
        buy_hold = metrics["benchmark_comparison"]["buy_and_hold"]
        portfolio_rows.append(
            {
                "variant_id": variant_id,
                "registered_model_id": alignment.registered_model_ids[variant_id],
                "portfolio_id": completed.config.portfolio_id,
                "ending_portfolio_value": metrics["ending_equity"],
                "total_return": metrics["total_return"],
                "maximum_drawdown": metrics["maximum_drawdown"],
                "volatility": metrics["volatility"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "number_of_trades": metrics["number_of_trades"],
                "fees_paid": metrics["fees_paid"],
                "buy_and_hold_portfolio_id": completed.config.benchmark_portfolio_ids.buy_and_hold,
                "buy_and_hold_ending_value": buy_hold["ending_equity"],
                "buy_and_hold_total_return": buy_hold["total_return"],
                "result_versus_buy_and_hold": metrics[
                    "comparison_with_buy_and_hold"
                ],
            }
        )
    portfolio_frame = pd.DataFrame(portfolio_rows).sort_values(
        ["total_return", "variant_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    portfolio_frame.to_csv(stage_dir / "portfolio_metrics.csv", index=False)
    combined = metric_frame.merge(
        portfolio_frame,
        on=["variant_id", "registered_model_id"],
        how="inner",
        validate="one_to_one",
    )
    combined.to_csv(stage_dir / "comparison_leaderboard.csv", index=False)
    _write_comparison_plots(stage_dir, metric_frame, portfolio_frame, backtests)
    return metric_frame, portfolio_frame, backtests, fair_evidence


def _write_comparison_plots(
    stage_dir: Path,
    metric_frame: pd.DataFrame,
    portfolio_frame: pd.DataFrame,
    backtests: Mapping[str, ExperimentRun],
) -> None:
    plot_root = stage_dir / "comparison_plots"
    plot_root.mkdir(parents=True, exist_ok=False)

    figure = Figure(figsize=(10, 5), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(
        metric_frame["variant_id"],
        metric_frame["balanced_accuracy"].fillna(0.0),
        color="#4c78a8",
    )
    axis.set_title("Fair common-date test balanced accuracy")
    axis.set_ylabel("Balanced accuracy")
    axis.set_ylim(0, 1)
    axis.grid(axis="y", alpha=0.25)
    figure.text(
        0.5, 0.01, ACADEMIC_DISCLAIMER, ha="center", fontsize=8
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(
        str(plot_root / "prediction_metric_comparison.png"),
        metadata={"Software": "PAISA"},
    )

    equity_records: list[dict[str, Any]] = []
    figure = Figure(figsize=(11, 6), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    for variant_id, completed in sorted(backtests.items()):
        dates = [snapshot.date for snapshot in completed.strategy.snapshots]
        equity = [float(snapshot.total_equity) for snapshot in completed.strategy.snapshots]
        axis.plot(dates, equity, label=variant_id, linewidth=1.4)
        equity_records.extend(
            {
                "variant_id": variant_id,
                "date": date.isoformat(),
                "total_equity": value,
            }
            for date, value in zip(dates, equity, strict=True)
        )
    axis.set_title("Fair common-date simulated portfolio equity")
    axis.set_xlabel("Trading session")
    axis.set_ylabel("Equity (PKR)")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=3)
    figure.autofmt_xdate()
    figure.text(0.5, 0.01, ACADEMIC_DISCLAIMER, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(
        str(plot_root / "portfolio_equity_comparison.png"),
        metadata={"Software": "PAISA"},
    )
    pd.DataFrame(equity_records).to_csv(
        stage_dir / "daily_equity_curves.csv", index=False
    )

    figure = Figure(figsize=(10, 5), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(
        portfolio_frame["variant_id"],
        portfolio_frame["total_return"],
        color="#59a14f",
    )
    axis.axhline(
        float(portfolio_frame["buy_and_hold_total_return"].iloc[0]),
        color="#e15759",
        linestyle="--",
        label="Buy and hold",
    )
    axis.set_title("Fair common-date simulated total return")
    axis.set_ylabel("Total return")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.text(0.5, 0.01, ACADEMIC_DISCLAIMER, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(
        str(plot_root / "portfolio_return_comparison.png"),
        metadata={"Software": "PAISA"},
    )


def _rank_records(frame: pd.DataFrame, columns: Sequence[str]) -> list[dict[str, Any]]:
    return to_jsonable(frame.loc[:, list(columns)].to_dict(orient="records"))


def _complete_lstm_split_metrics(
    states: Sequence[_FittedState], split_name: str
) -> dict[str, dict[str, Any]]:
    return {
        state.variant.variant_id: {
            "registered_model_id": state.registered_model_id,
            **state.metrics_by_split[split_name],
            "calibration": state.calibration_by_split[split_name],
        }
        for state in states
    }


def _complete_lstm_portfolio_metrics(
    portfolio_metrics: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    selected = portfolio_metrics[
        portfolio_metrics["variant_id"].isin(LSTM_VARIANTS_BY_ID)
    ]
    return {
        str(record["variant_id"]): {
            key: value for key, value in record.items() if key != "variant_id"
        }
        for record in to_jsonable(selected.to_dict(orient="records"))
    }


def _build_final_summary(
    *,
    run_id: str,
    invocation_id: str,
    final_run_dir: Path,
    zip_path: Path | None,
    freeze_decision: Mapping[str, Any],
    validation_leaderboard: pd.DataFrame,
    test_leaderboard: pd.DataFrame,
    comparison_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
    fair_evidence: Mapping[str, Any],
    states: Sequence[_FittedState],
    training_evidence: TrainingRunEvidence,
    vertical_evidence: DirectoryInventoryEvidence,
    baseline_provenance: BaselineDataProvenance,
    reproduction_checks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    lstm_rows = comparison_metrics[
        comparison_metrics["variant_id"].isin(LSTM_VARIANTS_BY_ID)
    ]
    baseline_rows = comparison_metrics[
        comparison_metrics["variant_id"].isin(_BASELINE_VARIANTS)
    ]
    lstm_scores = [
        float(value)
        for value in lstm_rows["balanced_accuracy"].dropna().tolist()
    ]
    baseline_scores = [
        float(value)
        for value in baseline_rows["balanced_accuracy"].dropna().tolist()
    ]
    any_lstm_beat_baselines = bool(
        lstm_scores
        and baseline_scores
        and max(lstm_scores) > max(baseline_scores)
    )
    lstm_portfolios = portfolio_metrics[
        portfolio_metrics["variant_id"].isin(LSTM_VARIANTS_BY_ID)
    ]
    any_lstm_beat_buy_hold = bool(
        (
            lstm_portfolios["total_return"].astype(float)
            > lstm_portfolios["buy_and_hold_total_return"].astype(float)
        ).any()
    )
    training_metadata = {
        state.variant.variant_id: state.training_metadata for state in states
    }
    model_ids = {
        state.variant.variant_id: state.registered_model_id for state in states
    }
    return to_jsonable(
        {
            "schema_version": "1.1",
            "status": "PASS",
            "run_id": run_id,
            "invocation_id": invocation_id,
            "artifact_directory": final_run_dir.name,
            "zip_path": zip_path.name if zip_path is not None else None,
            "path_base": "output_root",
            "symbol": DEFAULT_SYMBOL,
            "model_ids": model_ids,
            "validation_preferred_variant": freeze_decision[
                "validation_preferred_variant"
            ],
            "validation_selection": dict(freeze_decision),
            "lstm_metrics": {
                "validation": _complete_lstm_split_metrics(states, "validation"),
                "post_freeze_lstm_test": _complete_lstm_split_metrics(
                    states, "test"
                ),
            },
            "shared_test_holdout_context": {
                "selection_frozen_before_lstm_test_inference": True,
                "globally_untouched_project_holdout": False,
                "previously_exposed_through_variants": [
                    "M-001",
                    "M-002",
                    "M-003",
                    "M-004",
                ],
                "shared_with_variants": [
                    "M-001",
                    "M-002",
                    "M-003",
                    "M-004",
                    "M-005",
                    "M-006",
                ],
                "disclosure": (
                    "The same dated holdout was exposed previously through "
                    "M-001 through M-004 results. It is a post-freeze LSTM test, "
                    "not a globally untouched project holdout."
                ),
            },
            "lstm_portfolio_metrics": _complete_lstm_portfolio_metrics(
                portfolio_metrics
            ),
            "validation_lstm_ranking": _rank_records(
                validation_leaderboard,
                (
                    "variant_id",
                    "registered_model_id",
                    "balanced_accuracy",
                    "f1",
                    "accuracy",
                ),
            ),
            "post_freeze_lstm_test_ranking": _rank_records(
                test_leaderboard,
                (
                    "variant_id",
                    "registered_model_id",
                    "balanced_accuracy",
                    "f1",
                    "accuracy",
                    "brier_score",
                ),
            ),
            "fair_common_test_ranking_all_six": _rank_records(
                comparison_metrics,
                (
                    "variant_id",
                    "registered_model_id",
                    "balanced_accuracy",
                    "f1",
                    "accuracy",
                    "matthews_corrcoef",
                    "roc_auc",
                    "brier_score",
                ),
            ),
            "portfolio_ranking_all_six": _rank_records(
                portfolio_metrics,
                (
                    "variant_id",
                    "portfolio_id",
                    "ending_portfolio_value",
                    "total_return",
                    "maximum_drawdown",
                    "volatility",
                    "sharpe_ratio",
                    "number_of_trades",
                    "fees_paid",
                    "result_versus_buy_and_hold",
                ),
            ),
            "any_lstm_beat_all_m001_to_m004_on_fair_test_balanced_accuracy": (
                any_lstm_beat_baselines
            ),
            "any_lstm_portfolio_beat_buy_and_hold": any_lstm_beat_buy_hold,
            "buy_and_hold_result": {
                "ending_portfolio_value": portfolio_metrics[
                    "buy_and_hold_ending_value"
                ].iloc[0],
                "total_return": portfolio_metrics[
                    "buy_and_hold_total_return"
                ].iloc[0],
            },
            "fair_common_date_evidence": {
                "common_count": fair_evidence["common_count"],
                "common_target_dates": fair_evidence["common_target_dates"],
                "dropped_target_dates": fair_evidence["dropped_target_dates"],
            },
            "training_metadata": training_metadata,
            "validation_save_load_reproduction": dict(reproduction_checks),
            "baseline_evidence": {
                "training_run_id": training_evidence.run_id,
                "training_manifest_sha256": training_evidence.run_manifest_sha256,
                "vertical_slice_id": vertical_evidence.directory.name,
                "vertical_manifest_sha256": vertical_evidence.manifest_sha256,
                "data_provenance": baseline_provenance.to_dict(),
            },
            "selection_policy": (
                "Validation metrics select the preferred LSTM. Test metrics and "
                "portfolio simulations are reported separately and do not rename "
                "a champion."
            ),
            "limitations": [
                "One stock and one historical period do not establish general superiority.",
                "Corporate-action adjustment is unknown and dividends are not modeled.",
                "Results depend on fixed thresholds, fees, and the chosen split regime.",
                "Accelerator and library differences may cause small numeric differences.",
                "Simulated results do not predict future performance.",
            ],
            "disclaimer": ACADEMIC_DISCLAIMER,
        }
    )


def _write_artifact_manifest(stage_dir: Path) -> dict[str, Any]:
    excluded = {"run_manifest.json", "artifact_manifest.json"}
    files = _regular_files(stage_dir, exclude=excluded)
    manifest = {
        "algorithm": "sha256",
        "root": ".",
        "excludes": sorted(excluded),
        "files": [
            {"path": relative, "sha256": sha256_file(files[relative])}
            for relative in sorted(files)
        ],
    }
    write_json(manifest, stage_dir / "artifact_manifest.json")
    return manifest


def _run_manifest(
    *,
    run_id: str,
    invocation_id: str,
    registered_results: Mapping[str, Any],
    freeze_decision: Mapping[str, Any],
    training_evidence: TrainingRunEvidence,
    vertical_evidence: DirectoryInventoryEvidence,
    baseline_provenance: BaselineDataProvenance,
) -> dict[str, Any]:
    manifests = [
        registered_results[variant.variant_id].manifest for variant in LSTM_VARIANTS
    ]
    provenance_fields = ("code_version", "dirty_tree", "source_fingerprint")
    provenance = {field: manifests[0][field] for field in provenance_fields}
    for manifest in manifests[1:]:
        if any(manifest[field] != provenance[field] for field in provenance_fields):
            raise ValueError("LSTM model provenance must agree at run level")
    source_inputs = manifests[0].get("source_fingerprint_inputs")
    if source_inputs != list(_SOURCE_FINGERPRINT_INPUTS) or any(
        manifest.get("source_fingerprint_inputs") != source_inputs
        for manifest in manifests[1:]
    ):
        raise ValueError("LSTM source-fingerprint inputs must agree at run level")

    model_records: list[dict[str, Any]] = []
    prediction_paths: dict[str, str] = {}
    model_ids: dict[str, str] = {}
    for variant in LSTM_VARIANTS:
        result = registered_results[variant.variant_id]
        model_records.append(
            {
                "model_id": result.model_id,
                "variant_id": variant.variant_id,
                "invocation_id": invocation_id,
                "manifest_path": f"models/{result.model_id}/manifest.json",
                "artifact_sha256": result.manifest["artifact_sha256"],
            }
        )
        prediction_paths[variant.variant_id] = (
            f"models/{result.model_id}/predictions.csv"
        )
        model_ids[variant.variant_id] = result.model_id
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "invocation_id": invocation_id,
        "created_at": utc_now_z(),
        **provenance,
        "source_fingerprint_inputs": source_inputs,
        "symbol": DEFAULT_SYMBOL,
        "source_frequency": SOURCE_FREQUENCY,
        "target_definition": TARGET_POSITIVE_CLASS_CONDITION,
        "random_seed": RANDOM_SEED,
        "data_manifest": "data_manifest.json",
        "split_manifest": "split_manifest.json",
        "split_reference": "split_reference.json",
        "artifact_manifest": "artifact_manifest.json",
        "models": model_records,
        "model_ids": model_ids,
        "prediction_paths": prediction_paths,
        "provisional_champion": dict(freeze_decision),
        "selection_policy": "validation metrics only; test evaluated after freeze",
        "baseline_training_run": {
            "run_id": training_evidence.run_id,
            "manifest_sha256": training_evidence.run_manifest_sha256,
            "data_provenance": baseline_provenance.to_dict(),
        },
        "baseline_vertical_slice": {
            "run_id": vertical_evidence.directory.name,
            "manifest_sha256": vertical_evidence.manifest_sha256,
        },
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


def run_lstm_price_only_experiments(
    *,
    prices_path: str | Path,
    output_dir: str | Path,
    run_id: str,
    baseline_training_run: str | Path,
    baseline_vertical_slice: str | Path,
    create_zip: bool = True,
    device: str = "auto",
    tensorflow_module: Any | None = None,
) -> dict[str, Any]:
    """Run, compare, backtest, and atomically publish M-005 and M-006.

    The function performs no network access. Both retained baselines and the
    supplied prices must already exist and pass their checksum contracts.
    """

    resolved_run_id = _safe_run_id(run_id)
    repo_root = Path(__file__).resolve().parents[3]
    source_candidate = Path(prices_path).expanduser()
    if source_candidate.is_symlink():
        raise ValueError("prices_path must not be a symlink")
    source_prices = source_candidate.resolve(strict=True)
    if not source_prices.is_file():
        raise FileNotFoundError(f"prices_path must be a regular file: {source_prices}")

    training_evidence = validate_training_run_artifacts(baseline_training_run)
    vertical_evidence = validate_vertical_slice_artifacts(baseline_vertical_slice)
    _validate_baseline_relationship(training_evidence, vertical_evidence)

    output_candidate = reject_symlink_components(
        Path(output_dir).expanduser(),
        label="LSTM experiment output root",
    )
    _validate_output_separation(
        output_candidate, training_evidence.run_directory, vertical_evidence.directory
    )
    output_root = prepare_output_root(
        output_candidate,
        repository_root=repo_root,
        label="LSTM experiment output root",
    )
    _validate_output_separation(
        output_root, training_evidence.run_directory, vertical_evidence.directory
    )
    final_run_dir = output_root / resolved_run_id
    zip_path = output_root / f"{resolved_run_id}.zip" if create_zip else None
    if os.path.lexists(final_run_dir):
        raise FileExistsError(
            f"Run directory already exists; refusing to overwrite: {final_run_dir}"
        )
    if zip_path is not None and os.path.lexists(zip_path):
        raise FileExistsError(f"ZIP already exists; refusing to overwrite: {zip_path}")

    stage_parent = Path(
        tempfile.mkdtemp(prefix=f".{resolved_run_id}-", dir=output_root)
    )
    stage_dir = stage_parent / resolved_run_id
    stage_dir.mkdir()
    staged_zip = (
        stage_parent / f"{resolved_run_id}.zip" if zip_path is not None else None
    )
    committed = False
    events: list[dict[str, Any]] = []

    try:
        frozen_prices = _freeze_verified_prices(
            source_prices,
            stage_dir,
            expected_sha256=_baseline_price_checksum(training_evidence),
        )
        prices = pd.read_csv(frozen_prices)
        prepared = prepare_training_data(prices, symbol=DEFAULT_SYMBOL)
        baseline_split_manifest = _validate_prepared_against_baseline(
            prepared, training_evidence, frozen_prices
        )
        baseline_provenance = _load_baseline_data_provenance(training_evidence)
        baseline_predictions = _load_baseline_predictions(training_evidence)
        tf = _tensorflow_module(tensorflow_module)
        _configure_device(tf, device)
        invocation_id = str(uuid.uuid4())
        states, freeze_decision = _fit_until_validation_freeze(
            prepared=prepared,
            stage_dir=stage_dir,
            run_id=resolved_run_id,
            invocation_id=invocation_id,
            tf=tf,
            requested_device=device,
            events=events,
        )
        validation_leaderboard = pd.read_csv(
            stage_dir / "validation_leaderboard.csv"
        )
        test_leaderboard = _evaluate_test_once(
            states=states,
            prepared=prepared,
            events=events,
            stage_dir=stage_dir,
        )
        registered_results, lstm_predictions, reproduction_checks = _register_states(
            states=states,
            prepared=prepared,
            stage_dir=stage_dir,
            repo_root=repo_root,
            invocation_id=invocation_id,
            baseline_provenance=baseline_provenance,
            tf=tf,
            events=events,
        )
        history = _history_rows(states)
        history.to_csv(stage_dir / "training_history.csv", index=False)
        _write_data_and_split_evidence(
            prepared=prepared,
            states=states,
            stage_dir=stage_dir,
            prices_path=frozen_prices,
            training_evidence=training_evidence,
            vertical_evidence=vertical_evidence,
            baseline_provenance=baseline_provenance,
            baseline_split_manifest=baseline_split_manifest,
        )

        all_predictions = {**baseline_predictions, **lstm_predictions}
        comparison_metrics, portfolio_metrics, _backtests, fair_evidence = (
            _run_fair_comparison(
                predictions_by_variant=all_predictions,
                stage_dir=stage_dir,
                run_id=resolved_run_id,
                prices_path=frozen_prices,
            )
        )
        summary = _build_final_summary(
            run_id=resolved_run_id,
            invocation_id=invocation_id,
            final_run_dir=final_run_dir,
            zip_path=zip_path,
            freeze_decision=freeze_decision,
            validation_leaderboard=validation_leaderboard,
            test_leaderboard=test_leaderboard,
            comparison_metrics=comparison_metrics,
            portfolio_metrics=portfolio_metrics,
            fair_evidence=fair_evidence,
            states=states,
            training_evidence=training_evidence,
            vertical_evidence=vertical_evidence,
            baseline_provenance=baseline_provenance,
            reproduction_checks=reproduction_checks,
        )
        write_json(summary, stage_dir / "final_summary.json")
        _write_artifact_manifest(stage_dir)
        run_manifest = _run_manifest(
            run_id=resolved_run_id,
            invocation_id=invocation_id,
            registered_results=registered_results,
            freeze_decision=freeze_decision,
            training_evidence=training_evidence,
            vertical_evidence=vertical_evidence,
            baseline_provenance=baseline_provenance,
        )
        seal_training_run_artifacts(stage_dir, run_manifest)
        validate_training_run_artifacts(
            stage_dir,
            expected_run_id=resolved_run_id,
            expected_invocation_id=invocation_id,
        )
        if staged_zip is not None:
            make_zip_from_directory(stage_dir, staged_zip)
        assert_training_run_artifacts_unchanged(training_evidence)
        assert_vertical_slice_artifacts_unchanged(vertical_evidence)
        _publish_staged_artifacts(
            stage_dir=stage_dir,
            staged_zip=staged_zip,
            final_run_dir=final_run_dir,
            final_zip=zip_path,
        )
        committed = True
        # Publication has committed. Cleanup must not turn a successful atomic
        # commit into a reported failure or trigger rollback of public output.
        try:
            stage_parent.rmdir()
        except OSError:
            pass
    except Exception:
        if not committed and stage_parent.exists():
            shutil.rmtree(stage_parent)
        assert_training_run_artifacts_unchanged(training_evidence)
        assert_vertical_slice_artifacts_unchanged(vertical_evidence)
        raise
    return {
        **summary,
        "run_id": resolved_run_id,
        "artifact_directory": str(final_run_dir),
        "zip_path": str(zip_path) if zip_path is not None else None,
        "path_base": "absolute",
        "validation_preferred_variant": freeze_decision[
            "validation_preferred_variant"
        ],
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


__all__ = [
    "ACADEMIC_DISCLAIMER",
    "DirectoryInventoryEvidence",
    "LSTMExperimentVariant",
    "LSTM_VARIANTS",
    "LSTM_VARIANTS_BY_ID",
    "assert_vertical_slice_artifacts_unchanged",
    "run_lstm_price_only_experiments",
    "validate_vertical_slice_artifacts",
]
