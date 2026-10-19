from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping

import joblib
import numpy as np
import pandas as pd


MANIFEST_FILENAME = "manifest.json"
DEFAULT_ARTIFACT_FILENAME = "model.joblib"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
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
    if len(columns) != len(set(columns)):
        raise ValueError("feature_set.columns must be ordered and unique")
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
            "balanced_accuracy",
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
        try:
            matthews = float(split_metrics["matthews_corrcoef"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"metrics.{split}.matthews_corrcoef must be numeric"
            ) from exc
        if not np.isfinite(matthews) or not -1 <= matthews <= 1:
            raise ValueError(
                f"metrics.{split}.matthews_corrcoef must be in [-1, 1]"
            )
        roc_auc = split_metrics["roc_auc"]
        if roc_auc is not None:
            try:
                roc_auc_score = float(roc_auc)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metrics.{split}.roc_auc must be numeric or null") from exc
            if not np.isfinite(roc_auc_score) or not 0 <= roc_auc_score <= 1:
                raise ValueError(f"metrics.{split}.roc_auc must be null or in [0, 1]")
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


def load_model_artifact(model_dir: str | Path) -> tuple[Any, dict[str, Any]]:
    """Load a trusted registered artifact only after path and checksum checks."""

    directory = Path(model_dir).resolve()
    manifest_path = directory / MANIFEST_FILENAME
    if not directory.is_dir() or not manifest_path.is_file():
        raise FileNotFoundError(f"Registered model manifest is missing: {manifest_path}")

    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Model manifest must be a JSON object")
    raw = validate_model_manifest(raw)
    if str(raw.get("model_id", "")) != directory.name:
        raise ValueError("Manifest model_id does not match its immutable directory")

    relative = _safe_relative_path(raw.get("artifact_path", ""), field_name="artifact_path")
    model_root = directory.parent
    artifact_path = model_root.joinpath(*relative.parts).resolve()
    try:
        artifact_path.relative_to(directory)
    except ValueError as exc:
        raise ValueError("artifact_path must resolve inside its model directory") from exc
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Model artifact is missing: {artifact_path}")

    expected = str(raw.get("artifact_sha256", "")).lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise ValueError("artifact_sha256 must be a lowercase SHA-256 digest")
    actual = sha256_file(artifact_path)
    if actual != expected:
        raise ValueError(
            f"Model artifact checksum mismatch for {directory.name}: expected {expected}, got {actual}"
        )
    return joblib.load(artifact_path), raw


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
