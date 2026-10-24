from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib import metadata
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd

from paisa.experiments.training_features import validate_feature_columns


MANIFEST_FILENAME = "manifest.json"
DEFAULT_ARTIFACT_FILENAME = "model.joblib"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MAX_REGISTERED_MANIFEST_BYTES = 1024 * 1024
_MAX_REGISTERED_MODEL_BYTES = 256 * 1024 * 1024
_REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "model_id",
    "algorithm",
    "model_configuration",
    "feature_set",
    "training_dates",
    "target_definition",
    "prediction_horizon",
    "random_seed",
    "metrics",
    "artifact_path",
    "created_at",
    "symbols",
    "data_provenance",
    "class_mapping",
    "code_version",
    "dirty_tree",
    "source_fingerprint",
    "library_versions",
    "artifact_sha256",
}


def make_run_id(repo_root: Path, prefix: str = "TRN") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    sha = get_git_commit_sha(repo_root, short=True) or "nogit"
    return f"{prefix}-{timestamp}-{sha}"


def utc_now_z() -> str:
    """Return an ISO 8601 UTC timestamp in the contract's ``Z`` form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def get_git_commit_sha(repo_root: Path, short: bool = False) -> str | None:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        result = subprocess.run(args, cwd=repo_root, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def git_tree_is_dirty(repo_root: Path) -> bool:
    """Report tracked or untracked changes, treating an unreadable tree as dirty."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return True
    return bool(result.stdout.strip())


def source_fingerprint(paths: Iterable[str | Path], *, root: str | Path) -> str:
    """Hash a deterministic sequence of source paths and file contents."""

    base = Path(root).resolve()
    selected = sorted({Path(path).resolve() for path in paths}, key=str)
    if not selected:
        raise ValueError("At least one source path is required for a fingerprint.")

    digest = hashlib.sha256()
    for path in selected:
        if not path.is_file():
            raise FileNotFoundError(f"Source fingerprint path is not a file: {path}")
        try:
            relative = path.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Source path is outside fingerprint root: {path}") from exc
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def code_provenance(
    repo_root: str | Path,
    *,
    source_paths: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Build the manifest's commit, dirty-tree, and source-fingerprint fields."""

    repo = Path(repo_root).resolve()
    commit = get_git_commit_sha(repo) or "unknown"
    dirty = git_tree_is_dirty(repo)
    if dirty:
        selected = list(source_paths or sorted((repo / "src" / "paisa").rglob("*.py")))
        fingerprint = source_fingerprint(selected, root=repo)
    else:
        fingerprint = commit
    return {
        "code_version": commit,
        "dirty_tree": dirty,
        "source_fingerprint": fingerprint,
    }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_jsonable(data), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def write_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value
    if isinstance(value, (np.ndarray,)):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def detect_runtime_environment() -> dict[str, Any]:
    env = platform.system().lower() or "unknown"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        env = "colab"
    elif "KAGGLE_URL_BASE" in os.environ:
        env = "kaggle"

    cpu_info = platform.processor() or platform.machine()
    ram_gb = None
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = round(pages * page_size / (1024**3), 2)
    except Exception:
        ram_gb = None

    gpu_available = False
    gpu_note = "No CUDA GPU detected through torch; the CPU training pipeline remains supported."
    try:
        import torch  # type: ignore

        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            gpu_note = (
                f"CUDA GPU detected by torch: {torch.cuda.get_device_name(0)}. "
                "Random Forest remains CPU-based; XGBoost is configured for CPU hist."
            )
    except Exception:
        pass

    return {
        "runtime_environment": env,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_info,
        "ram_gb": ram_gb,
        "gpu_available": gpu_available,
        "gpu_note": gpu_note,
        "random_forest_compute_note": "scikit-learn RandomForestClassifier uses CPU in this run.",
        "xgboost_compute_note": "XGBoost uses the deterministic CPU hist tree method in this run.",
    }


def package_versions() -> dict[str, str | None]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "scikit-learn": "scikit-learn",
        "joblib": "joblib",
        "matplotlib": "matplotlib",
        "xgboost": "xgboost",
    }
    versions: dict[str, str | None] = {}
    for key, distribution in packages.items():
        try:
            versions[key] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[key] = None
    return versions


def make_zip_from_directory(source_dir: Path, zip_path: Path) -> Path:
    ensure_dir(zip_path.parent)
    if zip_path.exists():
        raise FileExistsError(f"ZIP already exists; refusing to overwrite: {zip_path}")
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(source_dir.rglob("*")):
            if item.is_file():
                archive.write(item, arcname=item.relative_to(source_dir.parent))
    return zip_path


def _safe_relative_path(value: str | Path, *, field_name: str) -> PurePosixPath:
    raw = str(value)
    if "\\" in raw:
        raise ValueError(f"{field_name} must use POSIX separators")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} must be a safe relative POSIX path")
    return path


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _require_fields(
    value: Mapping[str, Any], required: set[str], *, field_name: str
) -> None:
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{field_name} is missing required fields: {missing}")


def _iso_date(value: Any, *, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO YYYY-MM-DD date") from exc


def _string_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty array")
    values = [str(item) for item in value]
    if any(not item.strip() for item in values):
        raise ValueError(f"{field_name} entries must be non-empty strings")
    return values


def validate_model_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete model artifact contract without loading code."""

    value = deepcopy(dict(_require_mapping(manifest, field_name="manifest")))
    _require_fields(value, _REQUIRED_MANIFEST_FIELDS, field_name="manifest")
    if value["schema_version"] != "1.0":
        raise ValueError("manifest.schema_version must be '1.0'")

    model_id = str(value["model_id"])
    if not _SAFE_ID.fullmatch(model_id):
        raise ValueError("manifest.model_id is not filesystem-safe")
    if value["algorithm"] not in {"random_forest", "xgboost", "lstm"}:
        raise ValueError("manifest.algorithm must be random_forest, xgboost, or lstm")
    _require_mapping(value["model_configuration"], field_name="model_configuration")

    feature_set = _require_mapping(value["feature_set"], field_name="feature_set")
    _require_fields(
        feature_set,
        {"feature_set_id", "columns", "feature_code_version", "transforms"},
        field_name="feature_set",
    )
    if not str(feature_set["feature_set_id"]).strip() or not str(
        feature_set["feature_code_version"]
    ).strip():
        raise ValueError("feature_set ID and code version must be non-empty")
    columns = _string_list(feature_set["columns"], field_name="feature_set.columns")
    try:
        validate_feature_columns(columns)
    except ValueError as exc:
        raise ValueError(f"feature_set.columns are invalid: {exc}") from exc
    transforms = feature_set["transforms"]
    if not isinstance(transforms, list):
        raise ValueError("feature_set.transforms must be an array")
    for index, transform in enumerate(transforms):
        transform_value = _require_mapping(
            transform, field_name=f"feature_set.transforms[{index}]"
        )
        _require_fields(
            transform_value,
            {"name", "parameters", "fit_scope"},
            field_name=f"feature_set.transforms[{index}]",
        )
        if not str(transform_value["name"]).strip():
            raise ValueError("Transform names must be non-empty")
        _require_mapping(
            transform_value["parameters"],
            field_name=f"feature_set.transforms[{index}].parameters",
        )
        if transform_value["fit_scope"] != "train_only":
            raise ValueError("Every learned transform must declare fit_scope='train_only'")

    training_dates = _require_mapping(value["training_dates"], field_name="training_dates")
    training_date_fields = {
        "data_feature_start",
        "data_feature_end",
        "train_target_start",
        "train_target_end",
        "validation_target_start",
        "validation_target_end",
        "test_target_start",
        "test_target_end",
    }
    _require_fields(training_dates, training_date_fields, field_name="training_dates")
    parsed_dates = {
        key: _iso_date(training_dates[key], field_name=f"training_dates.{key}")
        for key in training_date_fields
    }
    if parsed_dates["data_feature_start"] > parsed_dates["data_feature_end"]:
        raise ValueError("data feature coverage dates are reversed")
    split_ranges = [
        (parsed_dates["train_target_start"], parsed_dates["train_target_end"]),
        (
            parsed_dates["validation_target_start"],
            parsed_dates["validation_target_end"],
        ),
        (parsed_dates["test_target_start"], parsed_dates["test_target_end"]),
    ]
    if any(start > end for start, end in split_ranges):
        raise ValueError("Every target-date split must be internally ordered")
    if not (
        split_ranges[0][1] < split_ranges[1][0]
        and split_ranges[1][1] < split_ranges[2][0]
    ):
        raise ValueError("Target-date splits must be chronological and non-overlapping")

    target = _require_mapping(value["target_definition"], field_name="target_definition")
    _require_fields(
        target,
        {"name", "positive_class_condition", "class_labels"},
        field_name="target_definition",
    )
    if target["name"] != "target_next_day_up":
        raise ValueError("target_definition.name must be target_next_day_up")
    if target["positive_class_condition"] != "close[target_date] > close[feature_date]":
        raise ValueError("target_definition positive-class condition does not match the contract")
    if target["class_labels"] != {"0": "down_or_flat", "1": "up"}:
        raise ValueError("target_definition.class_labels do not match the contract")

    if value["prediction_horizon"] != {"value": 1, "unit": "trading_session"}:
        raise ValueError("prediction_horizon must be one trading session")
    if isinstance(value["random_seed"], bool) or not isinstance(value["random_seed"], int):
        raise ValueError("random_seed must be an integer")

    metrics = _require_mapping(value["metrics"], field_name="metrics")
    _require_fields(metrics, {"train", "validation", "test"}, field_name="metrics")
    metric_fields = {
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "matthews_corrcoef",
        "roc_auc",
        "brier_score",
        "confusion_matrix",
        "class_distribution",
        "observed_classes",
        "complete_class_coverage",
        "selection_eligible",
        "metric_warnings",
        "support",
        "positive_label",
        "average",
        "zero_division",
    }
    for split in ("train", "validation", "test"):
        split_metrics = _require_mapping(metrics[split], field_name=f"metrics.{split}")
        _require_fields(split_metrics, metric_fields, field_name=f"metrics.{split}")
        for metric in (
            "accuracy",
            "precision",
            "recall",
            "f1",
            "brier_score",
        ):
            try:
                score = float(split_metrics[metric])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metrics.{split}.{metric} must be numeric") from exc
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"metrics.{split}.{metric} must be in [0, 1]")
        support = split_metrics["support"]
        if isinstance(support, bool) or not isinstance(support, int) or support <= 0:
            raise ValueError(f"metrics.{split}.support must be a positive integer")
        matrix = split_metrics["confusion_matrix"]
        if (
            not isinstance(matrix, list)
            or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 2 for row in matrix)
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for row in matrix
                for count in row
            )
            or sum(count for row in matrix for count in row) != support
        ):
            raise ValueError(
                f"metrics.{split}.confusion_matrix must be a non-negative 2x2 "
                "integer matrix summing to support"
            )
        distribution = _require_mapping(
            split_metrics["class_distribution"],
            field_name=f"metrics.{split}.class_distribution",
        )
        if set(distribution) != {"0", "1"} or any(
            isinstance(count, bool) or not isinstance(count, int) or count < 0
            for count in distribution.values()
        ) or sum(distribution.values()) != support:
            raise ValueError(
                f"metrics.{split}.class_distribution must contain non-negative "
                "integer counts for classes 0 and 1 summing to support"
            )
        observed_classes = [
            class_id for class_id in (0, 1) if distribution[str(class_id)] > 0
        ]
        if split_metrics["observed_classes"] != observed_classes:
            raise ValueError(
                f"metrics.{split}.observed_classes must match class_distribution"
            )
        complete_coverage = len(observed_classes) == 2
        if split_metrics["complete_class_coverage"] is not complete_coverage:
            raise ValueError(
                f"metrics.{split}.complete_class_coverage does not match "
                "class_distribution"
            )
        if split_metrics["selection_eligible"] is not complete_coverage:
            raise ValueError(
                f"metrics.{split}.selection_eligible must require both actual classes"
            )
        warnings = split_metrics["metric_warnings"]
        if not isinstance(warnings, list) or any(
            not isinstance(item, str) for item in warnings
        ):
            raise ValueError(f"metrics.{split}.metric_warnings must be an array of strings")
        if complete_coverage:
            if "single_class_actuals" in warnings:
                raise ValueError(
                    f"metrics.{split}.metric_warnings contradict complete class coverage"
                )
            for metric, lower, upper in (
                ("balanced_accuracy", 0.0, 1.0),
                ("matthews_corrcoef", -1.0, 1.0),
                ("roc_auc", 0.0, 1.0),
            ):
                try:
                    score = float(split_metrics[metric])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"metrics.{split}.{metric} must be numeric when both classes are observed"
                    ) from exc
                if not np.isfinite(score) or not lower <= score <= upper:
                    raise ValueError(
                        f"metrics.{split}.{metric} must be in [{lower}, {upper}]"
                    )
        else:
            if "single_class_actuals" not in warnings:
                raise ValueError(
                    f"metrics.{split}.metric_warnings must flag single_class_actuals"
                )
            for metric in ("balanced_accuracy", "matthews_corrcoef", "roc_auc"):
                if split_metrics[metric] is not None:
                    raise ValueError(
                        f"metrics.{split}.{metric} must be null for single-class actuals"
                    )
        if (
            split_metrics["positive_label"] != 1
            or split_metrics["average"] != "binary"
            or split_metrics["zero_division"] != 0
        ):
            raise ValueError(f"metrics.{split} classification conventions are invalid")

    artifact_path = _safe_relative_path(value["artifact_path"], field_name="artifact_path")
    if artifact_path.parts[0] != model_id:
        raise ValueError("artifact_path must begin with its model_id directory")
    if not _SHA256.fullmatch(str(value["artifact_sha256"])):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")

    timestamp = str(value["created_at"])
    if not timestamp.endswith("Z"):
        raise ValueError("created_at must be an ISO 8601 UTC timestamp ending in Z")
    try:
        parsed_timestamp = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("created_at must be a valid ISO 8601 UTC timestamp") from exc
    if parsed_timestamp.utcoffset() != timezone.utc.utcoffset(parsed_timestamp):
        raise ValueError("created_at must be in UTC")

    symbols = _string_list(value["symbols"], field_name="symbols")
    if any(symbol != symbol.upper() for symbol in symbols):
        raise ValueError("symbols must be uppercase")

    provenance = _require_mapping(value["data_provenance"], field_name="data_provenance")
    provenance_fields = {
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
    _require_fields(provenance, provenance_fields, field_name="data_provenance")
    if not str(provenance["provider"]).strip() or not str(
        provenance["input_schema_version"]
    ).strip():
        raise ValueError("Data provider and input schema version must be non-empty")
    requested_symbols = _string_list(
        provenance["requested_symbols"], field_name="data_provenance.requested_symbols"
    )
    included_symbols = _string_list(
        provenance["included_symbols"], field_name="data_provenance.included_symbols"
    )
    if any(symbol != symbol.upper() for symbol in requested_symbols + included_symbols):
        raise ValueError("Data-provenance symbols must be uppercase")
    if not set(included_symbols).issubset(requested_symbols):
        raise ValueError("included_symbols must be a subset of requested_symbols")
    for field in ("requested_start", "requested_end", "actual_feature_start", "actual_feature_end"):
        _iso_date(provenance[field], field_name=f"data_provenance.{field}")
    if not _SHA256.fullmatch(str(provenance["dataset_sha256"])):
        raise ValueError("data_provenance.dataset_sha256 must be a lowercase SHA-256 digest")
    if provenance["price_adjustment"] not in {"adjusted", "unadjusted", "unknown"}:
        raise ValueError("data_provenance.price_adjustment is invalid")
    if not isinstance(provenance["dividends_included"], bool):
        raise ValueError("data_provenance.dividends_included must be boolean")
    if not str(provenance["corporate_actions_policy"]).strip():
        raise ValueError("data_provenance.corporate_actions_policy must be non-empty")

    class_mapping = _require_mapping(value["class_mapping"], field_name="class_mapping")
    _require_fields(
        class_mapping,
        {"estimator_classes", "probability_fields_by_class"},
        field_name="class_mapping",
    )
    if class_mapping["estimator_classes"] != [0, 1] or class_mapping[
        "probability_fields_by_class"
    ] != {"0": "probability_down", "1": "probability_up"}:
        raise ValueError("class_mapping does not match the binary prediction contract")

    for field in ("code_version", "source_fingerprint"):
        if not str(value[field]).strip():
            raise ValueError(f"{field} must be non-empty")
    if not isinstance(value["dirty_tree"], bool):
        raise ValueError("dirty_tree must be boolean")
    _require_mapping(value["library_versions"], field_name="library_versions")
    return value


def _write_extra_file(path: Path, value: Any) -> None:
    ensure_dir(path.parent)
    if isinstance(value, pd.DataFrame):
        value.to_csv(path, index=False)
    elif isinstance(value, bytes):
        path.write_bytes(value)
    elif isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    elif isinstance(value, Path):
        copy_if_exists(value, path)
    else:
        write_json(value, path)


def write_model_artifact(
    models_root: str | Path,
    model_id: str,
    artifact: Any,
    manifest: Mapping[str, Any],
    *,
    artifact_filename: str = DEFAULT_ARTIFACT_FILENAME,
    extra_files: Mapping[str, Any] | None = None,
) -> Path:
    """Atomically register one immutable model directory.

    ``artifact_path`` is persisted relative to ``models_root`` as required by
    the architecture contract.  The completed destination is published only
    after the binary, extra files, checksum, and manifest have all been written.
    """

    identifier = str(model_id).strip()
    if not _SAFE_ID.fullmatch(identifier):
        raise ValueError(
            "model_id must start with a letter or digit and contain only letters, "
            "digits, '.', '_', and '-'"
        )

    filename = _safe_relative_path(artifact_filename, field_name="artifact_filename")
    root = ensure_dir(Path(models_root).resolve())
    destination = root / identifier
    if destination.exists():
        raise FileExistsError(
            f"Model directory already exists; refusing to overwrite: {destination}"
        )

    provided = deepcopy(dict(manifest))
    if str(provided.get("schema_version", "")) != "1.0":
        raise ValueError("Model manifest must declare schema_version '1.0'")
    if "model_id" in provided and str(provided["model_id"]) != identifier:
        raise ValueError("Manifest model_id does not match the registered model_id")
    canonical_artifact_path = PurePosixPath(identifier, *filename.parts).as_posix()
    if "artifact_path" in provided and provided["artifact_path"] != canonical_artifact_path:
        raise ValueError("Manifest artifact_path does not match the registry destination")

    staging = Path(tempfile.mkdtemp(prefix=f".{identifier}.", suffix=".tmp", dir=root))
    try:
        binary_path = staging.joinpath(*filename.parts)
        ensure_dir(binary_path.parent)
        joblib.dump(artifact, binary_path)
        checksum = sha256_file(binary_path)
        if "artifact_sha256" in provided and provided["artifact_sha256"] != checksum:
            raise ValueError("Provided artifact_sha256 does not match the serialized artifact")

        for relative_name, value in (extra_files or {}).items():
            relative = _safe_relative_path(relative_name, field_name="extra file path")
            if relative.as_posix() in {filename.as_posix(), MANIFEST_FILENAME}:
                raise ValueError(f"Extra file would replace a reserved artifact file: {relative}")
            _write_extra_file(staging.joinpath(*relative.parts), value)

        completed_manifest = {
            **provided,
            "model_id": identifier,
            "artifact_path": canonical_artifact_path,
            "artifact_sha256": checksum,
        }
        completed_manifest = validate_model_manifest(completed_manifest)
        write_json(completed_manifest, staging / MANIFEST_FILENAME)
        if destination.exists():
            raise FileExistsError(
                f"Model directory appeared during registration: {destination}"
            )
        staging.rename(destination)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return destination


def _registered_model_directory(
    model_dir: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> Path:
    """Resolve one real model directory within an optional registry boundary."""

    candidate = Path(model_dir)
    if candidate.is_symlink():
        raise ValueError(f"Registered model directory must not be a symlink: {candidate}")
    try:
        directory = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Registered model directory is missing: {candidate}") from exc
    if not directory.is_dir():
        raise FileNotFoundError(f"Registered model directory is missing: {candidate}")

    if trusted_root is not None:
        root = Path(trusted_root).resolve(strict=True)
        try:
            relative = directory.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                "Registered model directory resolves outside the trusted registry root"
            ) from exc
        if not relative.parts:
            raise ValueError("A registered model directory cannot be the registry root")
    return directory


def _reject_symlink_components(path: Path, *, root: Path, field_name: str) -> None:
    """Reject symlinks at every path component below a resolved trusted root."""

    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} must remain inside its model directory") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{field_name} must not contain symlinks: {current}")


def _open_regular_file(path: Path, *, field_name: str):
    """Open a regular file without following its final symlink component."""

    if path.is_symlink():
        raise ValueError(f"{field_name} must not be a symlink: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise ValueError(f"{field_name} must not be a symlink: {path}") from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"{field_name} must be a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except Exception:
        os.close(descriptor)
        raise


@dataclass(frozen=True, slots=True)
class TrainingRunEvidence:
    """Immutable checksum baseline for one complete training-run directory."""

    run_directory: Path
    run_id: str
    invocation_id: str
    run_manifest_sha256: str
    artifact_sha256: tuple[tuple[str, str], ...]
    code_version: str
    dirty_tree: bool
    source_fingerprint: str
    model_ids: tuple[tuple[str, str], ...]
    prediction_paths: tuple[tuple[str, str], ...]


_TRAINING_RUN_MANIFEST = "run_manifest.json"
_TRAINING_INVENTORY_FIELDS = {"algorithm", "root", "excludes", "files"}
_TRAINING_INVENTORY_ENTRY_FIELDS = {"path", "sha256"}
_NESTED_CHECKSUM_FIELDS = {"algorithm", "files"}


def _training_run_directory(run_dir: str | Path) -> Path:
    candidate = Path(run_dir)
    if candidate.is_symlink():
        raise ValueError(f"Training-run directory must not be a symlink: {candidate}")
    try:
        root = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Training-run directory is missing: {candidate}") from exc
    if not root.is_dir():
        raise FileNotFoundError(f"Training-run directory is missing: {candidate}")
    return root


def _regular_file_sha256(path: Path, *, field_name: str) -> str:
    digest = hashlib.sha256()
    with _open_regular_file(path, field_name=field_name) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_regular_json(path: Path, *, field_name: str) -> tuple[dict[str, Any], str]:
    with _open_regular_file(path, field_name=field_name) as handle:
        payload = handle.read()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field_name} must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _training_payload_files(root: Path) -> dict[str, Path]:
    """Return every regular run payload, rejecting links and special files."""

    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"Training-run artifacts must not contain symlinks: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(
                f"Training-run artifacts must be regular files or directories: {relative}"
            )
        if relative == _TRAINING_RUN_MANIFEST:
            continue
        files[relative] = path
    return files


def _build_training_inventory(root: Path) -> dict[str, Any]:
    files = _training_payload_files(root)
    return {
        "algorithm": "sha256",
        "root": ".",
        "excludes": [_TRAINING_RUN_MANIFEST],
        "files": [
            {
                "path": relative,
                "sha256": _regular_file_sha256(
                    files[relative], field_name=f"Training artifact {relative}"
                ),
            }
            for relative in sorted(files)
        ],
    }


def _validate_run_source_provenance(manifest: Mapping[str, Any]) -> tuple[str, bool, str]:
    code_version = str(manifest.get("code_version", "")).strip()
    dirty_tree = manifest.get("dirty_tree")
    fingerprint = str(manifest.get("source_fingerprint", "")).strip()
    if not code_version:
        raise ValueError("Training run code_version must be non-empty")
    if not isinstance(dirty_tree, bool):
        raise ValueError("Training run dirty_tree must be boolean")
    if not fingerprint:
        raise ValueError("Training run source_fingerprint must be non-empty")
    if dirty_tree and not _SHA256.fullmatch(fingerprint):
        raise ValueError(
            "A dirty training run source_fingerprint must be a lowercase SHA-256 digest"
        )
    return code_version, dirty_tree, fingerprint


def _parse_training_inventory(manifest: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    inventory = _require_mapping(
        manifest.get("artifact_inventory"), field_name="artifact_inventory"
    )
    if set(inventory) != _TRAINING_INVENTORY_FIELDS:
        raise ValueError(
            "artifact_inventory must contain exactly algorithm, root, excludes, and files"
        )
    if inventory["algorithm"] != "sha256" or inventory["root"] != ".":
        raise ValueError("artifact_inventory must use SHA-256 relative to the run root")
    if inventory["excludes"] != [_TRAINING_RUN_MANIFEST]:
        raise ValueError("artifact_inventory may exclude only run_manifest.json")
    raw_files = inventory["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("artifact_inventory.files must be a non-empty array")

    parsed: list[tuple[str, str]] = []
    for index, entry in enumerate(raw_files):
        item = _require_mapping(entry, field_name=f"artifact_inventory.files[{index}]")
        if set(item) != _TRAINING_INVENTORY_ENTRY_FIELDS:
            raise ValueError(
                "Every artifact_inventory file must contain exactly path and sha256"
            )
        relative = _safe_relative_path(
            item["path"], field_name=f"artifact_inventory.files[{index}].path"
        ).as_posix()
        if relative == _TRAINING_RUN_MANIFEST:
            raise ValueError("run_manifest.json must not be included in its own inventory")
        checksum = str(item["sha256"])
        if not _SHA256.fullmatch(checksum):
            raise ValueError(
                f"artifact_inventory checksum must be lowercase SHA-256: {relative}"
            )
        parsed.append((relative, checksum))
    if parsed != sorted(parsed, key=lambda item: item[0]):
        raise ValueError("artifact_inventory files must be sorted by path")
    if len({path for path, _checksum in parsed}) != len(parsed):
        raise ValueError("artifact_inventory must not contain duplicate paths")
    return tuple(parsed)


def _parse_nested_checksum_inventory(
    value: Any,
    *,
    field_name: str,
) -> tuple[tuple[str, str], ...]:
    """Validate one advertised nested SHA-256 inventory.

    The run-level inventory proves the bytes that were published. Nested
    inventories additionally make semantic claims about those bytes, so their
    advertised digests must be parsed and reconciled separately rather than
    trusted merely because the containing JSON file is inventoried.
    """

    inventory = _require_mapping(value, field_name=field_name)
    if set(inventory) != _NESTED_CHECKSUM_FIELDS:
        raise ValueError(f"{field_name} must contain exactly algorithm and files")
    if inventory["algorithm"] != "sha256":
        raise ValueError(f"{field_name} must use SHA-256")
    raw_files = inventory["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"{field_name}.files must be a non-empty array")

    parsed: list[tuple[str, str]] = []
    for index, entry in enumerate(raw_files):
        item = _require_mapping(
            entry, field_name=f"{field_name}.files[{index}]"
        )
        if set(item) != _TRAINING_INVENTORY_ENTRY_FIELDS:
            raise ValueError(
                f"Every {field_name} file must contain exactly path and sha256"
            )
        relative = _safe_relative_path(
            item["path"], field_name=f"{field_name}.files[{index}].path"
        ).as_posix()
        checksum = str(item["sha256"])
        if not _SHA256.fullmatch(checksum):
            raise ValueError(
                f"{field_name} checksum must be lowercase SHA-256: {relative}"
            )
        parsed.append((relative, checksum))

    if parsed != sorted(parsed, key=lambda item: item[0]):
        raise ValueError(f"{field_name} files must be sorted by path")
    if len({path for path, _checksum in parsed}) != len(parsed):
        raise ValueError(f"{field_name} must not contain duplicate paths")
    return tuple(parsed)


def _inventoried_json(
    root: Path,
    relative: str,
    expected_sha256: Mapping[str, str],
    *,
    field_name: str,
) -> dict[str, Any]:
    path = root.joinpath(*PurePosixPath(relative).parts)
    value, checksum = _read_regular_json(path, field_name=field_name)
    if checksum != expected_sha256[relative]:
        raise ValueError(f"Training artifact checksum mismatch: {relative}")
    return value


def _validate_model_nested_checksum_claims(
    *,
    root: Path,
    registered_id: str,
    model_manifest: Mapping[str, Any],
    expected_sha256: Mapping[str, str],
) -> None:
    """Reconcile optional model-level checksum claims with published bytes."""

    model_prefix = PurePosixPath("models", registered_id)
    if "preprocessing_artifacts" in model_manifest:
        field_name = f"Model {registered_id} preprocessing_artifacts"
        advertised = _parse_nested_checksum_inventory(
            model_manifest["preprocessing_artifacts"], field_name=field_name
        )
        checksum_manifest_relative = PurePosixPath(
            model_prefix, "preprocessing", "checksums.json"
        ).as_posix()
        if checksum_manifest_relative not in expected_sha256:
            raise ValueError(
                f"{field_name} must have a checksum-inventoried checksums.json"
            )
        checksum_manifest = _inventoried_json(
            root,
            checksum_manifest_relative,
            expected_sha256,
            field_name=f"Preprocessing checksum manifest {registered_id}",
        )
        if checksum_manifest != model_manifest["preprocessing_artifacts"]:
            raise ValueError(
                f"{field_name} disagrees with preprocessing/checksums.json"
            )

        preprocessing_prefix = PurePosixPath(model_prefix, "preprocessing")
        checksum_manifest_path = PurePosixPath(
            preprocessing_prefix, "checksums.json"
        )
        prefix_length = len(model_prefix.parts)
        inventoried_payloads: set[str] = set()
        for relative in expected_sha256:
            path = PurePosixPath(relative)
            if (
                path != checksum_manifest_path
                and path.parts[: len(preprocessing_prefix.parts)]
                == preprocessing_prefix.parts
            ):
                inventoried_payloads.add(
                    PurePosixPath(*path.parts[prefix_length:]).as_posix()
                )
        advertised_paths = {relative for relative, _checksum in advertised}
        if advertised_paths != inventoried_payloads:
            missing_claims = sorted(inventoried_payloads - advertised_paths)
            unexpected_claims = sorted(advertised_paths - inventoried_payloads)
            raise ValueError(
                f"{field_name} paths must exactly match the run-inventoried "
                "preprocessing payload files; "
                f"missing_claims={missing_claims}, "
                f"unexpected_claims={unexpected_claims}"
            )

        for relative, advertised_checksum in advertised:
            nested_path = PurePosixPath(relative)
            if (
                nested_path.parts[0] != "preprocessing"
                or nested_path == PurePosixPath("preprocessing", "checksums.json")
            ):
                raise ValueError(
                    f"{field_name} paths must identify preprocessing payload files"
                )
            run_relative = PurePosixPath(model_prefix, *nested_path.parts).as_posix()
            if expected_sha256.get(run_relative) != advertised_checksum:
                raise ValueError(
                    f"{field_name} checksum disagrees with the run inventory: "
                    f"{relative}"
                )

    if "training_metadata" not in model_manifest:
        return
    field_name = f"Model {registered_id} training_metadata"
    training_metadata = _require_mapping(
        model_manifest["training_metadata"], field_name=field_name
    )
    metadata_relative = PurePosixPath(
        model_prefix, "training_metadata.json"
    ).as_posix()
    if metadata_relative not in expected_sha256:
        raise ValueError(f"{field_name} must have an inventoried training_metadata.json")
    saved_metadata = _inventoried_json(
        root,
        metadata_relative,
        expected_sha256,
        field_name=f"Training metadata {registered_id}",
    )
    if saved_metadata != training_metadata:
        raise ValueError(f"{field_name} disagrees with training_metadata.json")

    if "model_checkpoint" not in training_metadata:
        return
    checkpoint = _require_mapping(
        training_metadata["model_checkpoint"],
        field_name=f"{field_name}.model_checkpoint",
    )
    _require_fields(
        checkpoint,
        {"path", "sha256"},
        field_name=f"{field_name}.model_checkpoint",
    )
    checkpoint_relative = _safe_relative_path(
        checkpoint["path"], field_name=f"{field_name}.model_checkpoint.path"
    ).as_posix()
    if PurePosixPath(checkpoint_relative).parts[0] != "checkpoints":
        raise ValueError(
            f"{field_name}.model_checkpoint.path must identify a checkpoint payload"
        )
    checkpoint_sha256 = str(checkpoint["sha256"])
    if not _SHA256.fullmatch(checkpoint_sha256):
        raise ValueError(
            f"{field_name}.model_checkpoint.sha256 must be lowercase SHA-256"
        )
    if expected_sha256.get(checkpoint_relative) != checkpoint_sha256:
        raise ValueError(
            f"{field_name} checkpoint checksum disagrees with the run inventory: "
            f"{checkpoint_relative}"
        )


def seal_training_run_artifacts(
    run_dir: str | Path,
    run_manifest: Mapping[str, Any] | None = None,
) -> TrainingRunEvidence:
    """Write the deterministic payload inventory and validate the sealed run.

    Supplying no manifest intentionally reseals an owned, unpublished run after
    a controlled metadata rewrite. Callers must never use resealing to accept
    changes made by untrusted callbacks.
    """

    root = _training_run_directory(run_dir)
    manifest_path = root / _TRAINING_RUN_MANIFEST
    if manifest_path.is_symlink():
        raise ValueError("Training run manifest must not be a symlink")
    if run_manifest is None:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Training run manifest is missing: {manifest_path}")
        value, _checksum = _read_regular_json(
            manifest_path, field_name="Training run manifest"
        )
    else:
        value = deepcopy(dict(run_manifest))
    _validate_run_source_provenance(value)
    value["artifact_inventory"] = _build_training_inventory(root)
    write_json(value, manifest_path)
    return validate_training_run_artifacts(root)


def validate_training_run_artifacts(
    run_dir: str | Path,
    *,
    expected_run_id: str | None = None,
    expected_invocation_id: str | None = None,
) -> TrainingRunEvidence:
    """Validate a complete training run without deserializing executable code."""

    root = _training_run_directory(run_dir)
    manifest_path = root / _TRAINING_RUN_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Training run manifest is missing: {manifest_path}")
    manifest, run_manifest_sha256 = _read_regular_json(
        manifest_path, field_name="Training run manifest"
    )
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Training run schema_version must be '1.0'")
    run_id = str(manifest.get("run_id", "")).strip()
    invocation_id = str(manifest.get("invocation_id", "")).strip()
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError("Training run_id is missing or not filesystem-safe")
    if not invocation_id:
        raise ValueError("Training invocation_id must be non-empty")
    if expected_run_id is not None and run_id != str(expected_run_id):
        raise ValueError("Training run_id does not match the expected run")
    if expected_invocation_id is not None and invocation_id != str(
        expected_invocation_id
    ):
        raise ValueError("Training invocation_id does not match the expected invocation")
    code_version, dirty_tree, fingerprint = _validate_run_source_provenance(manifest)

    inventory = _parse_training_inventory(manifest)
    expected_sha256 = dict(inventory)
    actual_files = _training_payload_files(root)
    expected_paths = set(expected_sha256)
    actual_paths = set(actual_files)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise ValueError(
            "Training artifact paths must exactly match the inventory; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for relative, expected in inventory:
        actual = _regular_file_sha256(
            actual_files[relative], field_name=f"Training artifact {relative}"
        )
        if actual != expected:
            raise ValueError(
                f"Training artifact checksum mismatch: {relative}; "
                f"expected {expected}, got {actual}"
            )

    for field, required_path in (
        ("data_manifest", "data_manifest.json"),
        ("split_manifest", "split_manifest.json"),
    ):
        relative = _safe_relative_path(
            manifest.get(field, ""), field_name=f"Training run {field}"
        ).as_posix()
        if relative != required_path or relative not in expected_sha256:
            raise ValueError(
                f"Training run {field} must identify its checksum-inventoried root file"
            )

    models = manifest.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Training run models must be a non-empty array")
    registered_manifest_paths: set[str] = set()
    variants: set[str] = set()
    registered_ids: set[str] = set()
    registered_by_variant: dict[str, str] = {}
    for index, record in enumerate(models):
        item = _require_mapping(record, field_name=f"models[{index}]")
        registered_id = str(item.get("model_id", ""))
        variant_id = str(item.get("variant_id", ""))
        if not _SAFE_ID.fullmatch(registered_id) or not _SAFE_ID.fullmatch(variant_id):
            raise ValueError("Training run model and variant IDs must be filesystem-safe")
        if registered_id in registered_ids or variant_id in variants:
            raise ValueError("Training run model and variant IDs must be unique")
        registered_ids.add(registered_id)
        variants.add(variant_id)
        registered_by_variant[variant_id] = registered_id
        if item.get("invocation_id") != invocation_id:
            raise ValueError("Training model record invocation_id does not match the run")
        relative = _safe_relative_path(
            item.get("manifest_path", ""), field_name=f"models[{index}].manifest_path"
        ).as_posix()
        expected_relative = f"models/{registered_id}/{MANIFEST_FILENAME}"
        if relative != expected_relative or relative not in expected_sha256:
            raise ValueError("Training model manifest path is invalid or not inventoried")
        registered_manifest_paths.add(relative)
        model_manifest = validate_model_manifest(
            _inventoried_json(
                root,
                relative,
                expected_sha256,
                field_name=f"Model manifest {registered_id}",
            )
        )
        if model_manifest["model_id"] != registered_id:
            raise ValueError("Training model manifest identity does not match its record")
        if model_manifest.get("variant_id") != variant_id:
            raise ValueError("Training model manifest variant does not match its record")
        if model_manifest.get("invocation_id") != invocation_id:
            raise ValueError("Training model manifest invocation_id does not match the run")
        for field, expected in (
            ("code_version", code_version),
            ("dirty_tree", dirty_tree),
            ("source_fingerprint", fingerprint),
        ):
            if model_manifest.get(field) != expected:
                raise ValueError(
                    f"Training model manifest {field} does not match run-level provenance"
                )
        if item.get("artifact_sha256") != model_manifest["artifact_sha256"]:
            raise ValueError("Training model binary checksum disagrees with its run record")
        artifact_relative = PurePosixPath(
            "models", *PurePosixPath(model_manifest["artifact_path"]).parts
        ).as_posix()
        if expected_sha256.get(artifact_relative) != model_manifest["artifact_sha256"]:
            raise ValueError("Training model binary is not transitively checksum-bound")
        _validate_model_nested_checksum_claims(
            root=root,
            registered_id=registered_id,
            model_manifest=model_manifest,
            expected_sha256=expected_sha256,
        )

    observed_manifest_paths = {
        relative
        for relative in expected_paths
        if len(PurePosixPath(relative).parts) == 3
        and PurePosixPath(relative).parts[0] == "models"
        and PurePosixPath(relative).name == MANIFEST_FILENAME
    }
    if observed_manifest_paths != registered_manifest_paths:
        raise ValueError("Training model manifests do not exactly match registered models")

    prediction_paths = _require_mapping(
        manifest.get("prediction_paths"), field_name="prediction_paths"
    )
    if set(prediction_paths) != variants:
        raise ValueError("Training prediction paths must exactly match registered variants")
    validated_prediction_paths: dict[str, str] = {}
    for variant_id, raw_path in prediction_paths.items():
        relative = _safe_relative_path(
            raw_path, field_name=f"prediction_paths.{variant_id}"
        ).as_posix()
        expected_path = (
            f"models/{registered_by_variant[str(variant_id)]}/predictions.csv"
        )
        if relative != expected_path or relative not in expected_sha256:
            raise ValueError(
                "Training prediction path does not identify its registered model's "
                "checksum-inventoried predictions.csv"
            )
        validated_prediction_paths[str(variant_id)] = relative

    return TrainingRunEvidence(
        run_directory=root,
        run_id=run_id,
        invocation_id=invocation_id,
        run_manifest_sha256=run_manifest_sha256,
        artifact_sha256=inventory,
        code_version=code_version,
        dirty_tree=dirty_tree,
        source_fingerprint=fingerprint,
        model_ids=tuple(sorted(registered_by_variant.items())),
        prediction_paths=tuple(sorted(validated_prediction_paths.items())),
    )


def assert_training_run_artifacts_unchanged(evidence: TrainingRunEvidence) -> None:
    """Reassert a previously captured run checksum baseline without resealing."""

    root = _training_run_directory(evidence.run_directory)
    manifest_path = root / _TRAINING_RUN_MANIFEST
    current_manifest_sha256 = _regular_file_sha256(
        manifest_path, field_name="Training run manifest"
    )
    if current_manifest_sha256 != evidence.run_manifest_sha256:
        raise RuntimeError("Training run manifest changed after validation")

    expected = dict(evidence.artifact_sha256)
    actual_files = _training_payload_files(root)
    if set(actual_files) != set(expected):
        raise RuntimeError("Training run artifact set changed after validation")
    for relative, expected_sha256 in evidence.artifact_sha256:
        actual_sha256 = _regular_file_sha256(
            actual_files[relative], field_name=f"Training artifact {relative}"
        )
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"Training artifact changed after validation: {relative}"
            )


def _read_registered_manifest(
    model_dir: str | Path,
    *,
    trusted_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    directory = _registered_model_directory(model_dir, trusted_root=trusted_root)
    manifest_path = directory / MANIFEST_FILENAME
    _reject_symlink_components(
        manifest_path,
        root=directory,
        field_name="Registered model manifest",
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Registered model manifest is missing: {manifest_path}")

    with _open_regular_file(
        manifest_path,
        field_name="Registered model manifest",
    ) as manifest_handle:
        if (
            os.fstat(manifest_handle.fileno()).st_size
            > _MAX_REGISTERED_MANIFEST_BYTES
        ):
            raise ValueError("Registered model manifest exceeds its size limit")
        payload = manifest_handle.read(_MAX_REGISTERED_MANIFEST_BYTES + 1)
    if len(payload) > _MAX_REGISTERED_MANIFEST_BYTES:
        raise ValueError("Registered model manifest exceeds its size limit")
    if expected_manifest_sha256 is not None:
        expected_manifest = str(expected_manifest_sha256).lower()
        if not _SHA256.fullmatch(expected_manifest):
            raise ValueError(
                "expected_manifest_sha256 must be a lowercase SHA-256 digest"
            )
        if hashlib.sha256(payload).hexdigest() != expected_manifest:
            raise ValueError(
                "Registered model manifest checksum does not match its trust anchor"
            )
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Registered model manifest must contain valid UTF-8 JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError("Model manifest must be a JSON object")
    raw = validate_model_manifest(raw)
    if str(raw.get("model_id", "")) != directory.name:
        raise ValueError("Manifest model_id does not match its immutable directory")
    return directory, raw


def read_model_manifest(
    model_dir: str | Path,
    *,
    trusted_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read and validate a registered manifest without deserializing its model."""

    _, manifest = _read_registered_manifest(model_dir, trusted_root=trusted_root)
    return manifest


def load_model_artifact(
    model_dir: str | Path,
    *,
    trusted_root: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
    expected_artifact_sha256: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Checksum and load trusted model bytes without reopening their path.

    Optional trust-anchor digests are checked against the same manifest bytes
    and artifact handle used before deserialization, closing a swap window for
    callers that pin an immutable external inventory.
    """

    directory, raw = _read_registered_manifest(
        model_dir,
        trusted_root=trusted_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )

    relative = _safe_relative_path(raw.get("artifact_path", ""), field_name="artifact_path")
    model_root = directory.parent
    artifact_candidate = model_root.joinpath(*relative.parts)
    _reject_symlink_components(
        artifact_candidate,
        root=directory,
        field_name="Registered model artifact",
    )
    try:
        artifact_path = artifact_candidate.resolve(strict=True)
    except ValueError as exc:
        raise ValueError("artifact_path must resolve inside its model directory") from exc
    try:
        artifact_path.relative_to(directory)
    except ValueError as exc:
        raise ValueError("artifact_path must resolve inside its model directory") from exc
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {artifact_path}")

    expected = str(raw.get("artifact_sha256", "")).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
    if expected_artifact_sha256 is not None:
        trust_anchor = str(expected_artifact_sha256).lower()
        if not _SHA256.fullmatch(trust_anchor):
            raise ValueError("expected_artifact_sha256 must be a lowercase SHA-256 digest")
        if expected != trust_anchor:
            raise ValueError("Model artifact checksum does not match its trust anchor")
    with _open_regular_file(
        artifact_path,
        field_name="Registered model artifact",
    ) as artifact_handle:
        if (
            os.fstat(artifact_handle.fileno()).st_size
            > _MAX_REGISTERED_MODEL_BYTES
        ):
            raise ValueError("Registered model artifact exceeds its size limit")
        payload = artifact_handle.read(_MAX_REGISTERED_MODEL_BYTES + 1)
    if len(payload) > _MAX_REGISTERED_MODEL_BYTES:
        raise ValueError("Registered model artifact exceeds its size limit")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Model artifact checksum mismatch for {directory.name}: "
            f"expected {expected}, got {actual}"
        )
    # Deserialize the immutable bytes that were checksum-validated. Loading
    # from the live descriptor would allow an in-place mutation between the
    # digest pass and executable deserialization.
    artifact = joblib.load(io.BytesIO(payload))
    return artifact, raw


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Expected file does not exist: {src}")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def split_rows_by_dates(frame: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    date_set = set(dates)
    data = frame.copy()
    data["_date_str"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    result = data[data["_date_str"].isin(date_set)].drop(columns=["_date_str"]).copy()
    return result.sort_values("date").reset_index(drop=True)
