"""Immutable SHAP evidence for the four PAISA tree price models.

The producing workflow is deliberately model-aware: it validates and loads the
four executable model artifacts only from the trusted training registry, then
publishes a portable explanation pack that contains no executable model.  The
offline verifier replays the derived tables solely from the long-form local
explanation CSVs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import struct
import tempfile
from typing import Any
import uuid
import zipfile
import zlib

import numpy as np
import pandas as pd

from paisa.evidence.artifact_security import (
    atomic_rename_noreplace,
    prepare_output_root,
    reject_symlink_components,
    remove_staging_tree,
)
from paisa.evidence.diagnostics import contains_sensitive_text
from paisa.experiments.artifacts import (
    TrainingRunEvidence,
    assert_training_run_artifacts_unchanged,
    code_provenance,
    load_model_artifact,
    sha256_file,
    validate_training_run_artifacts,
    write_json,
)
from paisa.experiments.training_features import (
    TECHNICAL_FEATURE_COLUMNS,
    validate_feature_columns,
)
from paisa.experiments.unified_comparison import (
    assert_vertical_slice_artifacts_unchanged,
    validate_vertical_slice_artifacts,
    verify_unified_comparison_pack,
)


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "paisa_tree_shap_explainability"
ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
HOLDOUT_TERM = "shared project holdout"
TICKER = "OGDC"
ROW_COUNT = 184
SHAP_PIN = "0.54.1"
PACKAGE_VERSION_KEYS = (
    "python",
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "joblib",
    "matplotlib",
    "shap",
)

TRAINING_RUN_ID = "TRN-20261020-191815-624662-nogit"
TRAINING_RUN_SHA256 = (
    "8d5113bc8b67f43a76639d7397b905217bda1f94321183f0438e33448f689023"
)
COMPARISON_RUN_ID = "CMP-20261024-OGDC-price-only-02"
COMPARISON_MANIFEST_SHA256 = (
    "e4b0b12bb565fa6df1f23b7e8855f84569ffcefb795c36c4bfeb4a1e2c82849d"
)
VERTICAL_SLICE_RUN_ID = "VSL-20261020-OGDC-main-02"
VERTICAL_SLICE_MANIFEST_SHA256 = (
    "2ce2f7581a5baff8487b09ae695213745c704f11715e6bd5abd0e99009337f31"
)
MODEL_ROWS_SHA256 = (
    "5ba9ce1c99af8873163c78ae9466c9a7de8042bb1deb3a56eeac525f12deb08e"
)
SPLIT_MANIFEST_SHA256 = (
    "cdc8d406884a41e04c3716082b75c76a8e09f402d38843589ca7c5e5a0cb6360"
)
DATE_ALIGNMENT_SHA256 = (
    "74c8bb247f6d26d30bdf0accc43fa40b3d1e6e2c04c082679d20ff35ecace462"
)
TRAINING_SOURCE_FINGERPRINT = "3b82245209925d7a4678fe7b276a3e58b5dac73d"
HOLDOUT_ROWS_SHA256 = (
    "730755ddb7993d829ddd4d7f5afaec4d5ca5617311d642f4a33722d251a5a85a"
)
BACKGROUND_TARGET_DATES_SHA256 = (
    "ecc4a54cbf66d52a45f5a0a723a04a82d682fac8192fe906865f5f082862a4bf"
)
TRANSFORMED_HOLDOUT_SHA256 = (
    "4add3d0aa5269e3df8f66f36311a9a53321066b67f38e1d715d847991ed26a46"
)
TRANSFORMED_BACKGROUND_SHA256 = (
    "ea1242cd06841ca64a74774d329aab838d4e62677c97689f62760bd4f2ae48b7"
)
TARGET_DEFINITION = "close[target_date] > close[feature_date]"

EXPECTED_MODELS: dict[str, dict[str, str]] = {
    "M-001": {
        "registered_model_id": (
            "M-001-6ce88dc384de975e-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
        "algorithm": "random_forest",
        "estimator_class": "RandomForestClassifier",
        "output_space": "probability",
        "model_binary_sha256": "d6a39ffd5fdacfd99b4a9762f3a80e0fe3b83ebb06a1af0bd882153732e07e22",
        "model_manifest_sha256": "b2d5ca84c775a28d68e8c00ee5f1572fb2667115ba1ae63b227c2238b2ed07a9",
        "comparison_prediction_sha256": "f86c2845ec640da1af43ed2702aef08484751bc1fd926bbb65381883fbbe616d",
    },
    "M-002": {
        "registered_model_id": (
            "M-002-5a0160ff1b7a8043-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
        "algorithm": "random_forest",
        "estimator_class": "RandomForestClassifier",
        "output_space": "probability",
        "model_binary_sha256": "3d4ddc1a0b01b96e7731b331211962f4df65803de01921d1b5fe2bbde404d77d",
        "model_manifest_sha256": "8218c30a666b9d94fe0b1269718fffd0540881f1044794ad2edb91885d497bad",
        "comparison_prediction_sha256": "9d59ee4f41ba038ecf745c0af5f8da086000b4a3b3e9f7c2ee712685cc236773",
    },
    "M-003": {
        "registered_model_id": (
            "M-003-23efd4aed873bf08-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
        "algorithm": "random_forest",
        "estimator_class": "RandomForestClassifier",
        "output_space": "probability",
        "model_binary_sha256": "35dba486e62612e4b6048e1386d80ee391939142c4f1ce5c29d09912986c8c4a",
        "model_manifest_sha256": "da26731de4b61a1a260036b9b9932617201574ee620751bdf31b4d750153a725",
        "comparison_prediction_sha256": "d12e3982ea595b22d2f07404f633dc49d721621ee668d8005c803cb4cc1c08cc",
    },
    "M-004": {
        "registered_model_id": (
            "M-004-6c60cc92f77a495a-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
        "algorithm": "xgboost",
        "estimator_class": "XGBClassifier",
        "output_space": "probability",
        "model_binary_sha256": "70054bd6fa879966c3fee50b7507c9c102ecbba2ca06c0767ab8fc67c1f6e21b",
        "model_manifest_sha256": "9c58c9147d4ce5c58ea0901c58c5379e552b629547acca4253fcacae1b8339b4",
        "comparison_prediction_sha256": "58251d7ef4a5cac20c386eae8dca8bc72bbf81f10b03e29c524c40083c564c00",
    },
}

MODEL_ORDER = tuple(EXPECTED_MODELS)
EXPECTED_FEATURE_COLUMNS = tuple(TECHNICAL_FEATURE_COLUMNS)
# SHAP's interventional probability transform accumulates small floating-point
# error across trees and the 100-row background.  These are strict absolute
# probability-space tolerances, selected per estimator family and still fail
# the official run when exceeded.
ADDITIVITY_TOLERANCES = {
    "random_forest": 5e-5,
    "xgboost": 1e-6,
}
PREDICTION_REPLAY_TOLERANCE = 1e-6
BACKGROUND_ROW_COUNT = 100
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_RUN_ID = re.compile(r"XAI-\d{8}-\d{6}-OGDC-tree-v1(?:-[a-z0-9][a-z0-9-]*)?")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_ZIP_FILES = 128
_MAX_ZIP_BYTES = 128 * 1024 * 1024
_MAX_ARTIFACT_FILE_BYTES = 32 * 1024 * 1024
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_PNG_PIXELS = 16 * 1024 * 1024

LOCAL_COLUMNS = (
    "schema_version",
    "model_id",
    "registered_model_id",
    "algorithm",
    "ticker",
    "feature_date",
    "target_date",
    "predicted_class",
    "probability_up",
    "probability_down",
    "actual_class",
    "output_space",
    "expected_value",
    "direct_model_output",
    "reconstructed_model_output",
    "additivity_error",
    "feature_name",
    "feature_value",
    "shap_value",
    "absolute_contribution_rank",
    "contribution_direction",
)

GLOBAL_COLUMNS = (
    "schema_version",
    "model_id",
    "registered_model_id",
    "algorithm",
    "output_space",
    "feature_name",
    "mean_absolute_shap_value",
    "signed_mean_shap_value",
    "median_absolute_shap_value",
    "maximum_absolute_shap_value",
    "feature_rank",
    "share_of_total_absolute_attribution",
    "positive_contribution_count",
    "negative_contribution_count",
)

AUDIT_CATEGORIES = (
    "highest_confidence_class_1",
    "highest_confidence_class_0",
    "correct_positive",
    "correct_negative",
    "false_positive",
    "false_negative",
    "largest_absolute_additivity_error",
)

AUDIT_COLUMNS = (
    "schema_version",
    "model_id",
    "registered_model_id",
    "algorithm",
    "category",
    "available",
    "unavailability_reason",
    "feature_date",
    "target_date",
    "predicted_class",
    "actual_class",
    "probability_up",
    "probability_down",
    "output_space",
    "absolute_additivity_error",
)

FORBIDDEN_FEATURE_NAMES = {
    "actual_class",
    "target_next_day_up",
    "target_date",
    "feature_date",
    "ticker",
    "source",
}

FORBIDDEN_CLAIMS = (
    "caused the stock to rise",
    "caused the market movement",
    "guaranteed an increase",
    "economically caused the prediction",
    "proves profitability",
    "financial advice",  # the exact disclaimer is allowlisted separately
)

REQUIRED_FIXED_FILES = {
    "run_manifest.json",
    "source_run_references.json",
    "explainer_contract.json",
    "feature_matrix_contract.json",
    "additivity_report.json",
    "global_feature_importance.csv",
    "cross_model_feature_rankings.csv",
    "audit_sample.csv",
    "quality_report.json",
    "final_summary.json",
    "artifact_manifest.json",
}

QUALITY_CHECKS = (
    "authoritative_source_ids",
    "source_integrity_roots",
    "trusted_model_loading",
    "model_binary_checksums",
    "registered_model_ids",
    "algorithm_identities",
    "ordered_feature_columns",
    "target_and_horizon",
    "estimator_class_mapping",
    "exact_184_date_holdout",
    "actual_class_alignment",
    "no_target_or_future_features",
    "original_fitted_preprocessing_only",
    "finite_shap_values",
    "additivity",
    "deterministic_audit_sample",
    "portable_relative_content",
    "no_executable_model_payload",
    "forbidden_claim_policy",
)


@dataclass(frozen=True, slots=True)
class SourceEvidence:
    training: TrainingRunEvidence
    vertical: Any
    comparison_root: Path
    comparison_manifest_sha256: str
    training_root: Path
    vertical_root: Path


@dataclass(frozen=True, slots=True)
class ModelExplanation:
    variant_id: str
    registered_model_id: str
    algorithm: str
    estimator_class: str
    output_space: str
    expected_value: float
    tolerance: float
    feature_columns: tuple[str, ...]
    transformed_matrix_sha256: str
    local_rows: pd.DataFrame
    contract: dict[str, Any]


def _safe_id(value: object, *, field: str) -> str:
    text = str(value)
    if not _SAFE_ID.fullmatch(text):
        raise ValueError(f"{field} must be a filesystem-safe identifier")
    return text


def _safe_run_id(value: object) -> str:
    text = _safe_id(value, field="run_id")
    if not _RUN_ID.fullmatch(text):
        raise ValueError(
            "run_id must match XAI-YYYYMMDD-HHMMSS-OGDC-tree-v1 with an optional suffix"
        )
    return text


def make_explainability_run_id(now: datetime | None = None) -> str:
    instant = now or datetime.now(timezone.utc)
    return f"XAI-{instant.strftime('%Y%m%d-%H%M%S')}-OGDC-tree-v1"


def _safe_relative(value: object, *, field: str) -> str:
    text = str(value)
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or text != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in text
    ):
        raise ValueError(f"{field} must be a normalized relative POSIX path")
    return text


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_regular_bytes(
    path: Path,
    *,
    field: str,
    max_bytes: int,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    """Read one non-symlink regular file through a stable descriptor."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FileNotFoundError(
            f"{field} is missing or not a readable regular file"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{field} must be a regular file")
        if before.st_size > max_bytes:
            raise ValueError(f"{field} exceeds its size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if len(payload) > max_bytes:
        raise ValueError(f"{field} exceeds its size limit")
    identity = _stat_identity(before)
    if _stat_identity(after) != identity or len(payload) != before.st_size:
        raise RuntimeError(f"{field} changed while it was being read")
    try:
        current = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{field} changed while it was being read") from exc
    if not stat.S_ISREG(current.st_mode) or _stat_identity(current) != identity:
        raise RuntimeError(f"{field} changed while it was being read")
    return payload, identity


def _decode_json_object(payload: bytes, *, field: str) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant is forbidden: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    payload, _ = _read_regular_bytes(path, field=field, max_bytes=_MAX_JSON_BYTES)
    return _decode_json_object(payload, field=field)


def _read_verified_bytes(
    path: Path,
    *,
    field: str,
    expected_sha256: str,
    max_bytes: int = _MAX_ARTIFACT_FILE_BYTES,
) -> bytes:
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError(f"{field} expected SHA-256 is invalid")
    payload, _ = _read_regular_bytes(path, field=field, max_bytes=max_bytes)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"{field} fingerprint is not authoritative")
    return payload


def _regular_files(root: Path, *, exclude: set[str] | None = None) -> dict[str, Path]:
    excluded = exclude or set()
    result: dict[str, Path] = {}
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"Explainability artifacts must not contain symlinks: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"Explainability artifact must be a regular file: {relative}")
        if relative not in excluded:
            result[relative] = candidate
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _frame_sha256(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    ).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _versions() -> dict[str, str]:
    packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "xgboost": "xgboost",
        "joblib": "joblib",
        "matplotlib": "matplotlib",
        "shap": "shap",
    }
    versions: dict[str, str] = {"python": platform.python_version()}
    for label, distribution in packages.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Missing explainability runtime dependency: {distribution}"
            ) from exc
    if versions["shap"] != SHAP_PIN:
        raise RuntimeError(
            f"Official explainability runs require shap=={SHAP_PIN}; "
            f"found {versions['shap']}"
        )
    return versions


def _class_index(classes: Sequence[Any], positive_class: int = 1) -> int:
    try:
        class_values = list(classes)
    except TypeError as exc:
        raise ValueError("Estimator class order must be exactly [0, 1]") from exc
    normalized: list[int] = []
    for value in class_values:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("Estimator class order must be exactly [0, 1]")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Estimator class order must be exactly [0, 1]") from exc
        if value != integer:
            raise ValueError("Estimator class order must be exactly [0, 1]")
        normalized.append(integer)
    if normalized != [0, 1]:
        raise ValueError("Estimator class order must be exactly [0, 1]")
    return normalized.index(positive_class)


def normalize_expected_value(value: Any, *, class_index: int) -> float:
    """Return one finite expected value across SHAP's binary output conventions."""

    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        result = float(array)
    else:
        flat = array.reshape(-1)
        if flat.size not in {1, 2}:
            raise ValueError(
                "SHAP expected value must be scalar or one binary-class pair"
            )
        if flat.size == 1:
            result = float(flat[0])
        elif class_index < flat.size:
            result = float(flat[class_index])
        else:
            raise ValueError("SHAP expected value has no positive-class entry")
    if not math.isfinite(result):
        raise ValueError("SHAP expected value must be finite")
    return result


def normalize_positive_class_shap(
    values: Any,
    *,
    row_count: int,
    feature_count: int,
    class_index: int,
) -> np.ndarray:
    """Normalize list, 2-D, and class-dimension SHAP outputs for class 1."""

    if class_index not in {0, 1}:
        raise ValueError("Binary SHAP class index must be 0 or 1")
    raw = getattr(values, "values", values)
    if isinstance(raw, (list, tuple)):
        if len(raw) != 2 or class_index not in {0, 1}:
            raise ValueError("Binary SHAP list output must contain exactly two classes")
        result = np.asarray(raw[class_index], dtype=float)
    else:
        array = np.asarray(raw, dtype=float)
        if array.ndim == 2:
            result = array
        elif array.ndim == 3:
            candidates: list[np.ndarray] = []
            if array.shape[:2] == (row_count, feature_count) and array.shape[2] == 2:
                candidates.append(array[:, :, class_index])
            if array.shape[1:] == (row_count, feature_count) and array.shape[0] == 2:
                candidates.append(array[class_index, :, :])
            if (
                array.shape[0] == row_count
                and array.shape[1] == 2
                and array.shape[2] == feature_count
            ):
                candidates.append(array[:, class_index, :])
            unique = {
                (candidate.shape, candidate.tobytes()): candidate for candidate in candidates
            }
            if len(unique) != 1:
                raise ValueError("SHAP class-dimension output shape is ambiguous or invalid")
            result = next(iter(unique.values()))
        else:
            raise ValueError("SHAP values must be a list, 2-D array, or 3-D array")
    if result.shape != (row_count, feature_count):
        raise ValueError(
            "Normalized SHAP values do not match the expected row/feature shape"
        )
    if not np.isfinite(result).all():
        raise ValueError("SHAP values must all be finite")
    return result


def _path_inside(path: Path, root: Path, *, field: str) -> Path:
    lexical = reject_symlink_components(path, label=field)
    trusted = reject_symlink_components(root, label=f"{field} trusted root")
    try:
        trusted_resolved = trusted.resolve(strict=True)
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"{field} is unavailable") from exc
    if trusted.is_symlink() or lexical.is_symlink():
        raise ValueError(f"{field} cannot be a symlink")
    try:
        relative = resolved.relative_to(trusted_resolved)
    except ValueError as exc:
        raise ValueError(f"{field} resolves outside its configured trusted root") from exc
    if len(relative.parts) != 1:
        raise ValueError(f"{field} must be one immutable run below its trusted root")
    return resolved


def _validate_sources(
    *,
    model_training_run: str | Path,
    comparison_run: str | Path,
    vertical_slice_run: str | Path,
    trusted_model_training_root: str | Path,
) -> SourceEvidence:
    training_root = _path_inside(
        Path(model_training_run),
        Path(trusted_model_training_root),
        field="Model-training run",
    )
    if training_root.name != TRAINING_RUN_ID:
        raise ValueError("Model-training run identity is not authoritative")
    training = validate_training_run_artifacts(
        training_root, expected_run_id=TRAINING_RUN_ID
    )
    if training.run_manifest_sha256 != TRAINING_RUN_SHA256:
        raise ValueError("Model-training integrity root is not authoritative")
    if training.source_fingerprint != TRAINING_SOURCE_FINGERPRINT:
        raise ValueError("Model-training source fingerprint is not authoritative")
    if dict(training.model_ids) != {
        model_id: contract["registered_model_id"]
        for model_id, contract in EXPECTED_MODELS.items()
    }:
        raise ValueError("Training run does not contain the four expected model identities")

    comparison_candidate = reject_symlink_components(
        Path(comparison_run), label="Unified comparison run"
    )
    vertical_candidate = reject_symlink_components(
        Path(vertical_slice_run), label="Vertical-slice run"
    )
    comparison_root = comparison_candidate.resolve(strict=True)
    vertical_root = vertical_candidate.resolve(strict=True)
    if comparison_root.name != COMPARISON_RUN_ID:
        raise ValueError("Unified comparison identity is not authoritative")
    if vertical_root.name != VERTICAL_SLICE_RUN_ID:
        raise ValueError("Vertical-slice identity is not authoritative")
    comparison_result = verify_unified_comparison_pack(comparison_root)
    if comparison_result.get("status") != "PASS":
        raise ValueError("Unified comparison pack did not verify PASS")
    comparison_sha = sha256_file(comparison_root / "manifest.json")
    if comparison_sha != COMPARISON_MANIFEST_SHA256:
        raise ValueError("Unified comparison integrity root is not authoritative")
    vertical = validate_vertical_slice_artifacts(vertical_root)
    if sha256_file(vertical_root / "manifest.json") != VERTICAL_SLICE_MANIFEST_SHA256:
        raise ValueError("Vertical-slice integrity root is not authoritative")
    if vertical.directory.name != VERTICAL_SLICE_RUN_ID:
        raise ValueError("Vertical-slice manifest identity is inconsistent")
    source_roots = _read_json(
        comparison_root / "source_roots.json", field="Comparison source roots"
    )
    sources = source_roots.get("sources")
    if not isinstance(sources, list):
        raise ValueError("Comparison source roots are malformed")
    source_by_role = {
        str(record.get("role")): record
        for record in sources
        if isinstance(record, Mapping)
    }
    expected_bindings = {
        "baseline_training": (TRAINING_RUN_ID, TRAINING_RUN_SHA256),
        "baseline_vertical_slice": (
            VERTICAL_SLICE_RUN_ID,
            VERTICAL_SLICE_MANIFEST_SHA256,
        ),
    }
    for role, (run_id, digest) in expected_bindings.items():
        record = source_by_role.get(role)
        if (
            not isinstance(record, Mapping)
            or record.get("run_id") != run_id
            or record.get("integrity_root_sha256") != digest
        ):
            raise ValueError("Comparison source-run binding is inconsistent")
    return SourceEvidence(
        training=training,
        vertical=vertical,
        comparison_root=comparison_root,
        comparison_manifest_sha256=comparison_sha,
        training_root=training_root,
        vertical_root=vertical_root,
    )


def _assert_sources_unchanged(evidence: SourceEvidence) -> None:
    assert_training_run_artifacts_unchanged(evidence.training)
    assert_vertical_slice_artifacts_unchanged(evidence.vertical)
    if sha256_file(evidence.comparison_root / "manifest.json") != evidence.comparison_manifest_sha256:
        raise RuntimeError("Unified comparison changed after source validation")
    verify_unified_comparison_pack(evidence.comparison_root)


def _load_holdout(
    evidence: SourceEvidence,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    alignment = _decode_json_object(
        _read_verified_bytes(
            evidence.comparison_root / "date_alignment.json",
            field="Comparison date alignment",
            expected_sha256=DATE_ALIGNMENT_SHA256,
            max_bytes=_MAX_JSON_BYTES,
        ),
        field="Comparison date alignment",
    )
    holdout = alignment.get("shared_project_holdout")
    if not isinstance(holdout, Mapping):
        raise ValueError("Comparison shared project holdout is missing")
    common_keys = holdout.get("common_keys")
    if (
        holdout.get("terminology") != HOLDOUT_TERM
        or holdout.get("common_count") != ROW_COUNT
        or not isinstance(common_keys, list)
        or len(common_keys) != ROW_COUNT
    ):
        raise ValueError("Comparison does not declare the exact 184-row holdout")
    if _canonical_json_sha256(common_keys) != HOLDOUT_ROWS_SHA256:
        raise ValueError("Comparison holdout rows are not the authoritative 184 keys")
    key_frame = pd.DataFrame(common_keys)
    required_key_columns = ["ticker", "feature_date", "target_date", "actual_class"]
    if set(key_frame.columns) != set(required_key_columns):
        raise ValueError("Comparison holdout key schema is invalid")
    key_frame = key_frame.loc[:, required_key_columns]
    if set(key_frame["ticker"].astype(str)) != {TICKER}:
        raise ValueError("Explainability holdout must contain only OGDC")
    if key_frame["feature_date"].duplicated().any() or key_frame["target_date"].duplicated().any():
        raise ValueError("Explainability holdout dates must be unique")
    feature_dates = key_frame["feature_date"].astype(str).tolist()
    target_dates = key_frame["target_date"].astype(str).tolist()
    if feature_dates != sorted(feature_dates) or target_dates != sorted(target_dates):
        raise ValueError("Explainability holdout dates must be chronological")
    if feature_dates[0] != "2025-12-15" or feature_dates[-1] != "2026-09-14":
        raise ValueError("Explainability feature-date coverage is invalid")
    if target_dates[0] != "2025-12-16" or target_dates[-1] != "2026-09-15":
        raise ValueError("Explainability target-date coverage is invalid")

    model_rows_path = evidence.training_root / "frozen_input" / "OGDC_model_rows.csv"
    model_rows = pd.read_csv(
        io.BytesIO(
            _read_verified_bytes(
                model_rows_path,
                field="Frozen modelling rows",
                expected_sha256=MODEL_ROWS_SHA256,
            )
        ),
        dtype={"ticker": "string"},
    )
    model_rows["feature_date"] = model_rows["feature_date"].astype(str)
    model_rows["target_date"] = model_rows["target_date"].astype(str)
    indexed = model_rows.set_index("target_date", drop=False)
    if any(value not in indexed.index for value in target_dates):
        raise ValueError("Frozen modelling rows do not cover the full holdout")
    selected = indexed.loc[target_dates].reset_index(drop=True)
    if selected["feature_date"].tolist() != feature_dates:
        raise ValueError("Frozen feature dates disagree with the comparison")
    if selected["ticker"].astype(str).tolist() != [TICKER] * ROW_COUNT:
        raise ValueError("Frozen modelling rows contain a wrong ticker")
    if selected["target_next_day_up"].astype(int).tolist() != key_frame[
        "actual_class"
    ].astype(int).tolist():
        raise ValueError("Frozen actual classes disagree with the comparison")

    predictions: dict[str, pd.DataFrame] = {}
    for variant_id, model_contract in EXPECTED_MODELS.items():
        path = (
            evidence.comparison_root
            / "evidence"
            / "predictions"
            / variant_id
            / "predictions_all_splits.csv"
        )
        frame = pd.read_csv(
            io.BytesIO(
                _read_verified_bytes(
                    path,
                    field=f"{variant_id} comparison predictions",
                    expected_sha256=model_contract["comparison_prediction_sha256"],
                )
            )
        )
        frame["feature_date"] = frame["feature_date"].astype(str)
        frame["target_date"] = frame["target_date"].astype(str)
        frame = frame.set_index("target_date", drop=False)
        if any(value not in frame.index for value in target_dates):
            raise ValueError(f"{variant_id} predictions do not cover the full holdout")
        frame = frame.loc[target_dates].reset_index(drop=True)
        if len(frame) != ROW_COUNT or frame["target_date"].tolist() != target_dates:
            raise ValueError(f"{variant_id} prediction dates are invalid")
        if frame["feature_date"].tolist() != feature_dates:
            raise ValueError(f"{variant_id} feature dates disagree with the holdout")
        if frame["actual_class"].astype(int).tolist() != key_frame[
            "actual_class"
        ].astype(int).tolist():
            raise ValueError(f"{variant_id} actual classes disagree with the holdout")
        if set(frame["ticker"].astype(str)) != {TICKER}:
            raise ValueError(f"{variant_id} predictions contain a wrong ticker")
        if set(frame["model_id"].astype(str)) != {
            model_contract["registered_model_id"]
        }:
            raise ValueError(f"{variant_id} registered model ID is invalid")
        probabilities = frame[["probability_down", "probability_up"]].to_numpy(float)
        if (
            not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0)
        ):
            raise ValueError(f"{variant_id} stored probabilities are invalid")
        predictions[variant_id] = frame

    split_manifest = _decode_json_object(
        _read_verified_bytes(
            evidence.training_root / "split_manifest.json",
            field="Training split manifest",
            expected_sha256=SPLIT_MANIFEST_SHA256,
            max_bytes=_MAX_JSON_BYTES,
        ),
        field="Training split manifest",
    )
    split_target_dates = split_manifest.get("target_dates")
    if not isinstance(split_target_dates, Mapping):
        raise ValueError("Training split manifest target dates are missing")
    train_dates = split_target_dates.get("train")
    if not isinstance(train_dates, list) or len(train_dates) != 853:
        raise ValueError("Training background source must contain exactly 853 rows")
    train_dates = [str(value) for value in train_dates]
    if any(value not in indexed.index for value in train_dates):
        raise ValueError("Frozen modelling rows do not cover the training background")
    train_rows = indexed.loc[train_dates].reset_index(drop=True)
    background_indices = np.linspace(
        0, len(train_rows) - 1, num=BACKGROUND_ROW_COUNT, dtype=int
    )
    if len(set(background_indices.tolist())) != BACKGROUND_ROW_COUNT:
        raise ValueError("Deterministic background selection produced duplicate rows")
    background = train_rows.iloc[background_indices].reset_index(drop=True)

    return selected, background, predictions, {
        "feature_dates": feature_dates,
        "target_dates": target_dates,
        "holdout_rows": common_keys,
        "background_target_dates": background["target_date"].astype(str).tolist(),
        "background_policy": (
            "100 chronological training rows selected at evenly spaced deterministic "
            "indices; fitted preprocessing only"
        ),
    }


def _model_pipeline_parts(pipeline: Any) -> tuple[Any, Any]:
    named_steps = getattr(pipeline, "named_steps", None)
    if not isinstance(named_steps, Mapping) or tuple(named_steps) != ("imputer", "model"):
        raise ValueError("Trusted model must be the fitted imputer/model pipeline")
    imputer = named_steps["imputer"]
    estimator = named_steps["model"]
    if not hasattr(imputer, "transform") or not hasattr(estimator, "predict_proba"):
        raise ValueError("Trusted pipeline does not expose fitted inference components")
    return imputer, estimator


def _raw_margin(estimator: Any, transformed: np.ndarray) -> np.ndarray:
    try:
        margin = estimator.predict(transformed, output_margin=True)
    except TypeError:
        booster = estimator.get_booster()
        import xgboost as xgb

        margin = booster.predict(xgb.DMatrix(transformed), output_margin=True)
    result = np.asarray(margin, dtype=float).reshape(-1)
    if result.shape != (len(transformed),) or not np.isfinite(result).all():
        raise ValueError("XGBoost raw-margin output is invalid")
    return result


def _explain_model(
    *,
    variant_id: str,
    pipeline: Any,
    manifest: Mapping[str, Any],
    model_rows: pd.DataFrame,
    background_rows: pd.DataFrame,
    predictions: pd.DataFrame,
) -> ModelExplanation:
    contract = EXPECTED_MODELS[variant_id]
    if manifest.get("variant_id") != variant_id:
        raise ValueError(f"{variant_id} stable model ID is invalid")
    if manifest.get("model_id") != contract["registered_model_id"]:
        raise ValueError(f"{variant_id} registered model ID is invalid")
    if manifest.get("algorithm") != contract["algorithm"]:
        raise ValueError(f"{variant_id} algorithm identity is invalid")
    if manifest.get("target_definition", {}).get("positive_class_condition") != TARGET_DEFINITION:
        raise ValueError(f"{variant_id} target definition is invalid")
    if manifest.get("prediction_horizon") != {"value": 1, "unit": "trading_session"}:
        raise ValueError(f"{variant_id} prediction horizon is invalid")
    class_mapping = manifest.get("class_mapping")
    if not isinstance(class_mapping, Mapping):
        raise ValueError(f"{variant_id} class mapping is missing")
    manifest_classes = class_mapping.get("estimator_classes")
    if manifest_classes != [0, 1]:
        raise ValueError(f"{variant_id} manifest class mapping must be [0, 1]")
    feature_set = manifest.get("feature_set")
    if not isinstance(feature_set, Mapping):
        raise ValueError(f"{variant_id} feature-set manifest is missing")
    feature_columns = tuple(validate_feature_columns(feature_set.get("columns", [])))
    if feature_columns != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(f"{variant_id} ordered feature columns are not authoritative")
    if set(feature_columns) & FORBIDDEN_FEATURE_NAMES or any(
        name.startswith(("target_", "next_")) for name in feature_columns
    ):
        raise ValueError(f"{variant_id} feature matrix contains leakage columns")
    if any(column not in model_rows.columns for column in feature_columns):
        raise ValueError(f"{variant_id} feature columns are absent from frozen input")
    raw_features = model_rows.loc[:, feature_columns].copy()
    if "actual_class" in raw_features or "target_next_day_up" in raw_features:
        raise ValueError("Evaluation labels cannot enter the explainer input")

    imputer, estimator = _model_pipeline_parts(pipeline)
    estimator_class = type(estimator).__name__
    if estimator_class != contract["estimator_class"]:
        raise ValueError(f"{variant_id} estimator class is invalid")
    pipeline_classes = getattr(pipeline, "classes_", None)
    estimator_classes = getattr(estimator, "classes_", None)
    positive_index = _class_index(pipeline_classes)
    try:
        estimator_positive_index = _class_index(estimator_classes)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{variant_id} fitted estimator class order is invalid"
        ) from exc
    if estimator_positive_index != positive_index:
        raise ValueError(f"{variant_id} fitted estimator class order is invalid")
    transformed = np.asarray(imputer.transform(raw_features), dtype=float)
    if transformed.shape != (ROW_COUNT, len(feature_columns)):
        raise ValueError(f"{variant_id} fitted preprocessing changed the feature shape")
    if not np.isfinite(transformed).all():
        raise ValueError(f"{variant_id} transformed feature matrix must be finite")
    transformed_frame = pd.DataFrame(transformed, columns=feature_columns)
    transformed_matrix_sha256 = _frame_sha256(transformed_frame)
    if transformed_matrix_sha256 != TRANSFORMED_HOLDOUT_SHA256:
        raise ValueError(f"{variant_id} transformed holdout fingerprint is invalid")
    background_features = background_rows.loc[:, feature_columns].copy()
    if "actual_class" in background_features or "target_next_day_up" in background_features:
        raise ValueError("Evaluation labels cannot enter the SHAP background matrix")
    transformed_background = np.asarray(
        imputer.transform(background_features), dtype=float
    )
    if transformed_background.shape != (BACKGROUND_ROW_COUNT, len(feature_columns)):
        raise ValueError(f"{variant_id} background preprocessing shape is invalid")
    if not np.isfinite(transformed_background).all():
        raise ValueError(f"{variant_id} SHAP background matrix must be finite")
    background_matrix_sha256 = _frame_sha256(
        pd.DataFrame(transformed_background, columns=feature_columns)
    )
    if background_matrix_sha256 != TRANSFORMED_BACKGROUND_SHA256:
        raise ValueError(f"{variant_id} transformed background fingerprint is invalid")

    probabilities = np.asarray(pipeline.predict_proba(raw_features), dtype=float)
    if probabilities.shape != (ROW_COUNT, 2) or not np.isfinite(probabilities).all():
        raise ValueError(f"{variant_id} model probabilities are invalid")
    stored_up = predictions["probability_up"].to_numpy(float)
    stored_down = predictions["probability_down"].to_numpy(float)
    if not np.allclose(
        probabilities[:, positive_index],
        stored_up,
        atol=PREDICTION_REPLAY_TOLERANCE,
        rtol=0,
    ):
        raise ValueError(f"{variant_id} reproduced positive probabilities disagree")
    if not np.allclose(
        probabilities[:, 1 - positive_index],
        stored_down,
        atol=PREDICTION_REPLAY_TOLERANCE,
        rtol=0,
    ):
        raise ValueError(f"{variant_id} reproduced negative probabilities disagree")
    predicted = (stored_up > stored_down).astype(int)
    if not np.array_equal(predicted, predictions["predicted_class"].to_numpy(int)):
        raise ValueError(f"{variant_id} stored predicted classes disagree")

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is required for real explanation runs; install requirements-explainability.txt"
        ) from exc
    explainer = shap.TreeExplainer(
        estimator,
        data=transformed_background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    raw_values = explainer.shap_values(transformed, check_additivity=True)
    shap_values = normalize_positive_class_shap(
        raw_values,
        row_count=ROW_COUNT,
        feature_count=len(feature_columns),
        class_index=positive_index,
    )
    expected_value = normalize_expected_value(
        explainer.expected_value, class_index=positive_index
    )
    output_space = contract["output_space"]
    if output_space != "probability":  # pragma: no cover - immutable contract guard
        raise AssertionError("Unsupported output space")
    direct_output = probabilities[:, positive_index]
    tolerance = ADDITIVITY_TOLERANCES[contract["algorithm"]]
    reconstructed = expected_value + shap_values.sum(axis=1)
    signed_error = reconstructed - direct_output
    if not np.isfinite(reconstructed).all() or not np.isfinite(signed_error).all():
        raise ValueError(f"{variant_id} additivity values must be finite")
    if np.any(np.abs(signed_error) > tolerance):
        maximum = float(np.max(np.abs(signed_error)))
        raise ValueError(
            f"{variant_id} SHAP additivity failed: max error {maximum} exceeds {tolerance}"
        )

    rows: list[dict[str, Any]] = []
    for row_index in range(ROW_COUNT):
        order = sorted(
            range(len(feature_columns)),
            key=lambda index: (-abs(float(shap_values[row_index, index])), feature_columns[index]),
        )
        ranks = {feature_index: rank for rank, feature_index in enumerate(order, start=1)}
        for feature_index, feature_name in enumerate(feature_columns):
            contribution = float(shap_values[row_index, feature_index])
            direction = (
                "increased_model_output"
                if contribution > 0
                else "decreased_model_output"
                if contribution < 0
                else "neutral"
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_id": variant_id,
                    "registered_model_id": contract["registered_model_id"],
                    "algorithm": contract["algorithm"],
                    "ticker": TICKER,
                    "feature_date": str(predictions.iloc[row_index]["feature_date"]),
                    "target_date": str(predictions.iloc[row_index]["target_date"]),
                    "predicted_class": int(predictions.iloc[row_index]["predicted_class"]),
                    "probability_up": float(stored_up[row_index]),
                    "probability_down": float(stored_down[row_index]),
                    "actual_class": int(predictions.iloc[row_index]["actual_class"]),
                    "output_space": output_space,
                    "expected_value": expected_value,
                    "direct_model_output": float(direct_output[row_index]),
                    "reconstructed_model_output": float(reconstructed[row_index]),
                    "additivity_error": float(signed_error[row_index]),
                    "feature_name": feature_name,
                    "feature_value": float(transformed[row_index, feature_index]),
                    "shap_value": contribution,
                    "absolute_contribution_rank": ranks[feature_index],
                    "contribution_direction": direction,
                }
            )
    local = pd.DataFrame(rows, columns=LOCAL_COLUMNS)
    contract_payload = {
        "schema_version": SCHEMA_VERSION,
        "model_id": variant_id,
        "registered_model_id": contract["registered_model_id"],
        "explainer_type": "shap.TreeExplainer",
        "algorithm": contract["algorithm"],
        "estimator_class": estimator_class,
        "class_explained": 1,
        "estimator_class_order": [0, 1],
        "expected_value": expected_value,
        "output_space": output_space,
        "feature_perturbation": "interventional",
        "background_data_policy": (
            "100 evenly spaced rows from the chronological training split; "
            "original fitted median imputer only"
        ),
        "background_row_count": BACKGROUND_ROW_COUNT,
        "background_matrix_sha256": background_matrix_sha256,
        "shap_version": SHAP_PIN,
        "feature_count": len(feature_columns),
        "row_count": ROW_COUNT,
        "additivity_tolerance": tolerance,
    }
    return ModelExplanation(
        variant_id=variant_id,
        registered_model_id=contract["registered_model_id"],
        algorithm=contract["algorithm"],
        estimator_class=estimator_class,
        output_space=output_space,
        expected_value=expected_value,
        tolerance=tolerance,
        feature_columns=feature_columns,
        transformed_matrix_sha256=transformed_matrix_sha256,
        local_rows=local,
        contract=contract_payload,
    )


def _prediction_rows(local: pd.DataFrame) -> pd.DataFrame:
    """Collapse long-form feature rows to one checked row per prediction."""

    metadata = [
        "schema_version",
        "model_id",
        "registered_model_id",
        "algorithm",
        "ticker",
        "feature_date",
        "target_date",
        "predicted_class",
        "probability_up",
        "probability_down",
        "actual_class",
        "output_space",
        "expected_value",
        "direct_model_output",
        "reconstructed_model_output",
        "additivity_error",
    ]
    records: list[dict[str, Any]] = []
    for (model_id, target_date), group in local.groupby(
        ["model_id", "target_date"], sort=False, dropna=False
    ):
        record: dict[str, Any] = {}
        for column in metadata:
            values = group[column].drop_duplicates().tolist()
            if len(values) != 1:
                raise ValueError(
                    f"Local explanation metadata changes within {model_id}/{target_date}"
                )
            record[column] = values[0]
        feature_names = group["feature_name"].astype(str).tolist()
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("A local explanation contains duplicate feature names")
        ranks = sorted(group["absolute_contribution_rank"].astype(int).tolist())
        if ranks != list(range(1, len(group) + 1)):
            raise ValueError("Absolute contribution ranks must be one complete permutation")
        recomputed_order = sorted(
            zip(
                group["feature_name"].astype(str),
                group["shap_value"].astype(float),
                group["absolute_contribution_rank"].astype(int),
                strict=True,
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )
        if [item[2] for item in recomputed_order] != list(range(1, len(group) + 1)):
            raise ValueError("Absolute contribution ranks are not deterministic")
        ordered_values = group.sort_values("feature_name", kind="stable")[
            "shap_value"
        ].astype(float)
        shap_sum = float(math.fsum(ordered_values.tolist()))
        reconstructed = float(record["expected_value"]) + shap_sum
        if not math.isclose(
            reconstructed,
            float(record["reconstructed_model_output"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Reconstructed model output does not equal base plus SHAP sum")
        error = reconstructed - float(record["direct_model_output"])
        if not math.isclose(
            error,
            float(record["additivity_error"]),
            rel_tol=0,
            abs_tol=1e-12,
        ):
            raise ValueError("Persisted signed additivity error is inconsistent")
        record["absolute_additivity_error"] = abs(error)
        record["feature_count"] = len(group)
        records.append(record)
    result = pd.DataFrame(records)
    return result.sort_values(["model_id", "target_date"], kind="stable").reset_index(
        drop=True
    )


def compute_global_importance(local: pd.DataFrame) -> pd.DataFrame:
    """Recompute the canonical global feature table from local explanation rows."""

    rows: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        model_rows = local.loc[local["model_id"] == model_id]
        if model_rows.empty:
            raise ValueError(f"Missing local explanations for {model_id}")
        total_absolute = float(model_rows["shap_value"].astype(float).abs().sum())
        if not math.isfinite(total_absolute) or total_absolute <= 0:
            raise ValueError(f"{model_id} total absolute attribution must be positive")
        model_records: list[dict[str, Any]] = []
        for feature_name, group in model_rows.groupby("feature_name", sort=False):
            values = group["shap_value"].astype(float).to_numpy()
            absolute = np.abs(values)
            model_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "model_id": model_id,
                    "registered_model_id": str(group["registered_model_id"].iloc[0]),
                    "algorithm": str(group["algorithm"].iloc[0]),
                    "output_space": str(group["output_space"].iloc[0]),
                    "feature_name": str(feature_name),
                    "mean_absolute_shap_value": float(np.mean(absolute)),
                    "signed_mean_shap_value": float(np.mean(values)),
                    "median_absolute_shap_value": float(np.median(absolute)),
                    "maximum_absolute_shap_value": float(np.max(absolute)),
                    "share_of_total_absolute_attribution": float(
                        np.sum(absolute) / total_absolute
                    ),
                    "positive_contribution_count": int(np.count_nonzero(values > 0)),
                    "negative_contribution_count": int(np.count_nonzero(values < 0)),
                }
            )
        model_records.sort(
            key=lambda record: (
                -record["mean_absolute_shap_value"],
                record["feature_name"],
            )
        )
        for rank, record in enumerate(model_records, start=1):
            record["feature_rank"] = rank
            rows.append(record)
    return pd.DataFrame(rows, columns=GLOBAL_COLUMNS)


def compute_cross_model_rankings(global_frame: pd.DataFrame) -> pd.DataFrame:
    """Build a unit-labelled feature-rank comparison without pooling magnitudes."""

    features = sorted(global_frame["feature_name"].astype(str).unique())
    columns = ["schema_version", "feature_name"]
    for model_id in MODEL_ORDER:
        prefix = model_id.replace("-", "_")
        columns.extend(
            [
                f"{prefix}_rank",
                f"{prefix}_mean_absolute_shap_value",
                f"{prefix}_output_space",
            ]
        )
    columns.extend(
        [
            "random_forest_mean_rank",
            "random_forest_rank_range",
            "all_model_rank_range",
            "all_model_rank_standard_deviation",
        ]
    )
    rows: list[dict[str, Any]] = []
    for feature_name in features:
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "feature_name": feature_name,
        }
        ranks: list[float] = []
        rf_ranks: list[float] = []
        for model_id in MODEL_ORDER:
            match = global_frame.loc[
                (global_frame["model_id"] == model_id)
                & (global_frame["feature_name"] == feature_name)
            ]
            if len(match) != 1:
                raise ValueError("Every feature must appear once per explained model")
            row = match.iloc[0]
            rank = float(row["feature_rank"])
            prefix = model_id.replace("-", "_")
            record[f"{prefix}_rank"] = int(rank)
            record[f"{prefix}_mean_absolute_shap_value"] = float(
                row["mean_absolute_shap_value"]
            )
            record[f"{prefix}_output_space"] = str(row["output_space"])
            ranks.append(rank)
            if model_id != "M-004":
                rf_ranks.append(rank)
        record["random_forest_mean_rank"] = float(np.mean(rf_ranks))
        record["random_forest_rank_range"] = int(max(rf_ranks) - min(rf_ranks))
        record["all_model_rank_range"] = int(max(ranks) - min(ranks))
        record["all_model_rank_standard_deviation"] = float(np.std(ranks, ddof=0))
        rows.append(record)
    rows.sort(
        key=lambda record: (
            record["random_forest_mean_rank"],
            record["feature_name"],
        )
    )
    return pd.DataFrame(rows, columns=columns)


def _select_audit_rows(local: pd.DataFrame) -> pd.DataFrame:
    predictions = _prediction_rows(local)
    output: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        rows = predictions.loc[predictions["model_id"] == model_id].copy()
        if len(rows) != ROW_COUNT:
            raise ValueError(f"{model_id} must have exactly 184 prediction rows")
        selectors: dict[str, tuple[pd.Series, str, bool]] = {
            "highest_confidence_class_1": (
                rows["predicted_class"].astype(int) == 1,
                "probability_up",
                False,
            ),
            "highest_confidence_class_0": (
                rows["predicted_class"].astype(int) == 0,
                "probability_down",
                False,
            ),
            "correct_positive": (
                (rows["predicted_class"].astype(int) == 1)
                & (rows["actual_class"].astype(int) == 1),
                "probability_up",
                False,
            ),
            "correct_negative": (
                (rows["predicted_class"].astype(int) == 0)
                & (rows["actual_class"].astype(int) == 0),
                "probability_down",
                False,
            ),
            "false_positive": (
                (rows["predicted_class"].astype(int) == 1)
                & (rows["actual_class"].astype(int) == 0),
                "probability_up",
                False,
            ),
            "false_negative": (
                (rows["predicted_class"].astype(int) == 0)
                & (rows["actual_class"].astype(int) == 1),
                "probability_down",
                False,
            ),
            "largest_absolute_additivity_error": (
                pd.Series(True, index=rows.index),
                "absolute_additivity_error",
                False,
            ),
        }
        for category in AUDIT_CATEGORIES:
            mask, sort_column, ascending = selectors[category]
            candidates = rows.loc[mask].sort_values(
                [sort_column, "target_date"],
                ascending=[ascending, True],
                kind="stable",
            )
            base = {
                "schema_version": SCHEMA_VERSION,
                "model_id": model_id,
                "registered_model_id": EXPECTED_MODELS[model_id][
                    "registered_model_id"
                ],
                "algorithm": EXPECTED_MODELS[model_id]["algorithm"],
                "category": category,
            }
            if candidates.empty:
                output.append(
                    {
                        **base,
                        "available": False,
                        "unavailability_reason": "no_prediction_matches_category",
                        "feature_date": "",
                        "target_date": "",
                        "predicted_class": "",
                        "actual_class": "",
                        "probability_up": "",
                        "probability_down": "",
                        "output_space": EXPECTED_MODELS[model_id]["output_space"],
                        "absolute_additivity_error": "",
                    }
                )
                continue
            selected = candidates.iloc[0]
            output.append(
                {
                    **base,
                    "available": True,
                    "unavailability_reason": "",
                    "feature_date": str(selected["feature_date"]),
                    "target_date": str(selected["target_date"]),
                    "predicted_class": int(selected["predicted_class"]),
                    "actual_class": int(selected["actual_class"]),
                    "probability_up": float(selected["probability_up"]),
                    "probability_down": float(selected["probability_down"]),
                    "output_space": str(selected["output_space"]),
                    "absolute_additivity_error": float(
                        selected["absolute_additivity_error"]
                    ),
                }
            )
    return pd.DataFrame(output, columns=AUDIT_COLUMNS)


def _rank_agreements(global_frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for left_index, left_id in enumerate(MODEL_ORDER):
        left = global_frame.loc[global_frame["model_id"] == left_id].set_index(
            "feature_name"
        )
        for right_id in MODEL_ORDER[left_index + 1 :]:
            right = global_frame.loc[
                global_frame["model_id"] == right_id
            ].set_index("feature_name")
            features = sorted(set(left.index) & set(right.index))
            if not features:
                raise ValueError("Models have no common feature ranks")
            left_ranks = left.loc[features, "feature_rank"].to_numpy(float)
            right_ranks = right.loc[features, "feature_rank"].to_numpy(float)
            correlation = float(np.corrcoef(left_ranks, right_ranks)[0, 1])
            top_left = set(
                left.sort_values(["feature_rank"]).head(5).index.astype(str)
            )
            top_right = set(
                right.sort_values(["feature_rank"]).head(5).index.astype(str)
            )
            records.append(
                {
                    "left_model_id": left_id,
                    "right_model_id": right_id,
                    "spearman_rank_correlation": correlation,
                    "top_five_overlap": sorted(top_left & top_right),
                    "top_five_overlap_count": len(top_left & top_right),
                    "output_spaces": [
                        EXPECTED_MODELS[left_id]["output_space"],
                        EXPECTED_MODELS[right_id]["output_space"],
                    ],
                    "raw_magnitudes_directly_comparable": (
                        EXPECTED_MODELS[left_id]["output_space"]
                        == EXPECTED_MODELS[right_id]["output_space"]
                    ),
                }
            )
    return records


def _model_summary(
    model_id: str,
    *,
    local: pd.DataFrame,
    global_frame: pd.DataFrame,
    audit: pd.DataFrame,
    agreements: list[dict[str, Any]],
) -> dict[str, Any]:
    model_global = global_frame.loc[global_frame["model_id"] == model_id].copy()
    model_global = model_global.sort_values("feature_rank", kind="stable")
    positive = model_global.loc[model_global["signed_mean_shap_value"] > 0].sort_values(
        ["signed_mean_shap_value", "feature_name"],
        ascending=[False, True],
        kind="stable",
    )
    negative = model_global.loc[model_global["signed_mean_shap_value"] < 0].sort_values(
        ["signed_mean_shap_value", "feature_name"],
        ascending=[True, True],
        kind="stable",
    )

    def feature_records(frame: pd.DataFrame, value_column: str, count: int = 5) -> list[dict[str, Any]]:
        return [
            {
                "feature_name": str(row["feature_name"]),
                "value": float(row[value_column]),
                "output_space": str(row["output_space"]),
            }
            for _, row in frame.head(count).iterrows()
        ]

    audit_examples: list[dict[str, Any]] = []
    model_audit = audit.loc[audit["model_id"] == model_id]
    model_local = local.loc[local["model_id"] == model_id]
    for _, audit_row in model_audit.iterrows():
        example: dict[str, Any] = {
            "category": str(audit_row["category"]),
            "available": bool(audit_row["available"]),
        }
        if not bool(audit_row["available"]):
            example["unavailability_reason"] = str(
                audit_row["unavailability_reason"]
            )
        else:
            target_date = str(audit_row["target_date"])
            contributors = model_local.loc[
                model_local["target_date"].astype(str) == target_date
            ].sort_values("absolute_contribution_rank", kind="stable")
            example.update(
                {
                    "feature_date": str(audit_row["feature_date"]),
                    "target_date": target_date,
                    "top_contributors": [
                        {
                            "feature_name": str(row["feature_name"]),
                            "shap_value": float(row["shap_value"]),
                            "direction": str(row["contribution_direction"]),
                        }
                        for _, row in contributors.head(5).iterrows()
                    ],
                }
            )
        audit_examples.append(example)

    pairwise = [
        record
        for record in agreements
        if model_id in {record["left_model_id"], record["right_model_id"]}
    ]
    ordered_agreements = sorted(
        pairwise,
        key=lambda record: (-record["spearman_rank_correlation"], record["left_model_id"], record["right_model_id"]),
    )
    ordered_disagreements = sorted(
        pairwise,
        key=lambda record: (record["spearman_rank_correlation"], record["left_model_id"], record["right_model_id"]),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "model_id": model_id,
        "registered_model_id": EXPECTED_MODELS[model_id]["registered_model_id"],
        "algorithm": EXPECTED_MODELS[model_id]["algorithm"],
        "output_space": EXPECTED_MODELS[model_id]["output_space"],
        "top_features_by_mean_absolute_attribution": feature_records(
            model_global, "mean_absolute_shap_value"
        ),
        "global_top_positive_contributors": feature_records(
            positive, "signed_mean_shap_value"
        ),
        "global_top_negative_contributors": feature_records(
            negative, "signed_mean_shap_value"
        ),
        "strongest_rank_agreements": ordered_agreements[:3],
        "strongest_rank_disagreements": ordered_disagreements[:3],
        "local_audit_examples": audit_examples,
        "interpretation_policy": (
            "Attributions describe associations with this model output; they do not "
            "establish economic causality or future performance."
        ),
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


def _additivity_report(local: pd.DataFrame, contracts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    predictions = _prediction_rows(local)
    contract_by_model = {str(record["model_id"]): record for record in contracts}
    models: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        rows = predictions.loc[predictions["model_id"] == model_id]
        tolerance = float(contract_by_model[model_id]["additivity_tolerance"])
        absolute = rows["absolute_additivity_error"].astype(float).to_numpy()
        pass_mask = absolute <= tolerance
        models.append(
            {
                "model_id": model_id,
                "registered_model_id": EXPECTED_MODELS[model_id][
                    "registered_model_id"
                ],
                "output_space": EXPECTED_MODELS[model_id]["output_space"],
                "mean_absolute_error": float(np.mean(absolute)),
                "maximum_absolute_error": float(np.max(absolute)),
                "row_count_passing_tolerance": int(np.count_nonzero(pass_mask)),
                "row_count_failing_tolerance": int(np.count_nonzero(~pass_mask)),
                "numerical_tolerance": tolerance,
            }
        )
    status = "PASS" if all(item["row_count_failing_tolerance"] == 0 for item in models) else "FAIL"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "signed_error_definition": "reconstructed_model_output - direct_model_output",
        "models": models,
    }


def _source_references(
    evidence: SourceEvidence,
    *,
    explanations: Sequence[ModelExplanation],
) -> dict[str, Any]:
    training_manifest = _read_json(
        evidence.training_root / "run_manifest.json", field="Training run manifest"
    )
    model_manifest_by_id: dict[str, dict[str, Any]] = {}
    for record in training_manifest["models"]:
        registered_id = str(record["model_id"])
        model_manifest_by_id[registered_id] = _read_json(
            evidence.training_root / str(record["manifest_path"]),
            field=f"Model manifest {registered_id}",
        )
    model_records: list[dict[str, Any]] = []
    for explanation in explanations:
        manifest = model_manifest_by_id[explanation.registered_model_id]
        prediction_path = (
            evidence.comparison_root
            / "evidence"
            / "predictions"
            / explanation.variant_id
            / "predictions_all_splits.csv"
        )
        prediction_sha256 = sha256_file(prediction_path)
        expected = EXPECTED_MODELS[explanation.variant_id]
        if prediction_sha256 != expected["comparison_prediction_sha256"]:
            raise ValueError(
                f"{explanation.variant_id} comparison prediction fingerprint changed"
            )
        model_records.append(
            {
                "model_id": explanation.variant_id,
                "registered_model_id": explanation.registered_model_id,
                "algorithm": explanation.algorithm,
                "model_binary_sha256": str(manifest["artifact_sha256"]),
                "model_manifest_sha256": sha256_file(
                    evidence.training_root
                    / "models"
                    / explanation.registered_model_id
                    / "manifest.json"
                ),
                "comparison_prediction_sha256": prediction_sha256,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": [
            {
                "role": "model_training",
                "run_id": TRAINING_RUN_ID,
                "integrity_root_name": "run_manifest.json",
                "integrity_root_sha256": TRAINING_RUN_SHA256,
                "source_fingerprint": evidence.training.source_fingerprint,
                "dataset_sha256": MODEL_ROWS_SHA256,
            },
            {
                "role": "unified_comparison",
                "run_id": COMPARISON_RUN_ID,
                "integrity_root_name": "manifest.json",
                "integrity_root_sha256": COMPARISON_MANIFEST_SHA256,
            },
            {
                "role": "vertical_slice",
                "run_id": VERTICAL_SLICE_RUN_ID,
                "integrity_root_name": "manifest.json",
                "integrity_root_sha256": VERTICAL_SLICE_MANIFEST_SHA256,
            },
        ],
        "models": model_records,
        "portable_paths": True,
        "model_binaries_included": False,
    }


def _feature_matrix_contract(
    *,
    holdout_contract: Mapping[str, Any],
    explanations: Sequence[ModelExplanation],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "ticker": TICKER,
        "holdout_terminology": HOLDOUT_TERM,
        "row_count": ROW_COUNT,
        "feature_date_count": len(holdout_contract["feature_dates"]),
        "target_date_count": len(holdout_contract["target_dates"]),
        "feature_dates": list(holdout_contract["feature_dates"]),
        "target_dates": list(holdout_contract["target_dates"]),
        "holdout_rows": list(holdout_contract["holdout_rows"]),
        "holdout_rows_sha256": HOLDOUT_ROWS_SHA256,
        "feature_value_space": "output of the original fitted median imputer",
        "preprocessing_refit": False,
        "background_policy": holdout_contract["background_policy"],
        "background_target_dates": list(holdout_contract["background_target_dates"]),
        "background_target_dates_sha256": BACKGROUND_TARGET_DATES_SHA256,
        "source_dataset_sha256": MODEL_ROWS_SHA256,
        "models": [
            {
                "model_id": item.variant_id,
                "registered_model_id": item.registered_model_id,
                "ordered_feature_columns": list(item.feature_columns),
                "feature_count": len(item.feature_columns),
                "row_count": ROW_COUNT,
                "transformed_matrix_sha256": item.transformed_matrix_sha256,
                "target_columns_in_matrix": False,
                "actual_class_in_matrix": False,
                "future_values_in_matrix": False,
            }
            for item in explanations
        ],
    }


def _final_summary(
    *,
    run_id: str,
    versions: Mapping[str, str],
    global_frame: pd.DataFrame,
    additivity: Mapping[str, Any],
    audit: pd.DataFrame,
    agreements: list[dict[str, Any]],
) -> dict[str, Any]:
    model_records: list[dict[str, Any]] = []
    additivity_by_model = {
        str(record["model_id"]): record for record in additivity["models"]
    }
    for model_id in MODEL_ORDER:
        top = global_frame.loc[global_frame["model_id"] == model_id].sort_values(
            "feature_rank", kind="stable"
        )
        categories = audit.loc[audit["model_id"] == model_id]
        model_records.append(
            {
                "model_id": model_id,
                "registered_model_id": EXPECTED_MODELS[model_id][
                    "registered_model_id"
                ],
                "algorithm": EXPECTED_MODELS[model_id]["algorithm"],
                "output_space": EXPECTED_MODELS[model_id]["output_space"],
                "row_count": ROW_COUNT,
                "feature_count": int(len(top)),
                "maximum_absolute_additivity_error": additivity_by_model[model_id][
                    "maximum_absolute_error"
                ],
                "mean_absolute_additivity_error": additivity_by_model[model_id][
                    "mean_absolute_error"
                ],
                "top_global_features": top.head(5)["feature_name"].astype(str).tolist(),
                "audit_categories": [
                    {
                        "category": str(row["category"]),
                        "available": bool(row["available"]),
                    }
                    for _, row in categories.iterrows()
                ],
            }
        )
    ordered_agreements = sorted(
        agreements,
        key=lambda record: (
            -record["spearman_rank_correlation"],
            record["left_model_id"],
            record["right_model_id"],
        ),
    )
    ordered_disagreements = sorted(
        agreements,
        key=lambda record: (
            record["spearman_rank_correlation"],
            record["left_model_id"],
            record["right_model_id"],
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "run_id": run_id,
        "status": str(additivity["status"]),
        "ticker": TICKER,
        "holdout_terminology": HOLDOUT_TERM,
        "shap_version": versions["shap"],
        "models": model_records,
        "cross_model_rank_agreements": ordered_agreements[:3],
        "cross_model_rank_disagreements": ordered_disagreements[:3],
        "output_space_policy": (
            "Every model is explained in positive-class probability space. "
            "Raw attribution magnitudes are still retained by model and are not "
            "averaged into a universal importance score."
        ),
        "limitations": [
            "The 184 dates are a previously exposed shared project holdout, not a fresh independent evaluation.",
            "Attributions describe fitted-model behavior and do not establish economic causality.",
            "Evidence covers one stock, one period, one feature family, and four fitted tree configurations.",
            "Feature importance does not establish profitability or future performance.",
        ],
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


def _plot_global_importance(frame: pd.DataFrame, path: Path, *, model_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = frame.loc[frame["model_id"] == model_id].sort_values(
        "feature_rank", kind="stable"
    )
    selected = selected.head(15).iloc[::-1]
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(
        selected["feature_name"].astype(str),
        selected["mean_absolute_shap_value"].astype(float),
        color="#2a6f97",
    )
    axis.set_xlabel("Mean absolute SHAP value (class-1 probability space)")
    axis.set_title(f"{model_id} global feature attribution")
    axis.grid(axis="x", alpha=0.25)
    figure.text(
        0.5,
        0.01,
        ACADEMIC_DISCLAIMER,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=160, metadata={"Software": "PAISA"})
    plt.close(figure)


def _plot_cross_model_ranks(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = frame.sort_values(
        ["random_forest_mean_rank", "feature_name"], kind="stable"
    )
    rank_columns = [f"{model_id.replace('-', '_')}_rank" for model_id in MODEL_ORDER]
    matrix = ordered[rank_columns].to_numpy(float)
    figure, axis = plt.subplots(figsize=(8, 8))
    image = axis.imshow(matrix, aspect="auto", cmap="viridis_r")
    axis.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    axis.set_yticks(
        range(len(ordered)), ordered["feature_name"].astype(str).tolist()
    )
    axis.set_title("Cross-model feature ranks (lower is more influential)")
    figure.colorbar(image, ax=axis, label="Feature rank")
    figure.text(
        0.5,
        0.01,
        ACADEMIC_DISCLAIMER,
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path, dpi=160, metadata={"Software": "PAISA"})
    plt.close(figure)


def _make_tree_private(root: Path) -> None:
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if candidate.is_symlink():
            raise ValueError("Explainability staging tree cannot contain symlinks")
        if candidate.is_dir():
            candidate.chmod(0o700)
        elif candidate.is_file():
            candidate.chmod(0o600)
        else:
            raise ValueError("Explainability staging tree contains a special file")
    root.chmod(0o700)


def _write_artifact_manifest(stage_dir: Path, *, run_id: str) -> dict[str, Any]:
    files = _regular_files(stage_dir, exclude={"artifact_manifest.json"})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "run_id": run_id,
        "status": "PASS",
        "inventory": {
            "algorithm": "sha256",
            "root": ".",
            "excludes": ["artifact_manifest.json"],
            "files": [
                {"path": relative, "sha256": sha256_file(files[relative])}
                for relative in sorted(files)
            ],
        },
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    write_json(manifest, stage_dir / "artifact_manifest.json")
    return manifest


def _contains_absolute_path(text: str) -> bool:
    patterns = (
        r"(?:^|[^A-Za-z0-9_.:-])(?<!:/)/(?:/)?[A-Za-z0-9._~-]",
        r"(?:^|[^A-Za-z0-9_.-])[A-Za-z]:[\\/]",
        r"\\\\[^\\\s]+\\[^\\\s]+",
        r"file://",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _validate_png_payload(payload: bytes, *, relative: str) -> None:
    signature = b"\x89PNG\r\n\x1a\n"
    if len(payload) > _MAX_ARTIFACT_FILE_BYTES or not payload.startswith(signature):
        raise ValueError(f"Plot is not a valid PNG payload: {relative}")
    offset = len(signature)
    chunks: list[tuple[bytes, bytes]] = []
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise ValueError(f"Plot has a truncated PNG chunk: {relative}")
        length = struct.unpack_from(">I", payload, offset)[0]
        chunk_end = offset + 12 + length
        if length > _MAX_ARTIFACT_FILE_BYTES or chunk_end > len(payload):
            raise ValueError(f"Plot has an invalid PNG chunk length: {relative}")
        chunk_type = payload[offset + 4 : offset + 8]
        chunk_data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack_from(">I", payload, offset + 8 + length)[0]
        observed_crc = zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF
        if expected_crc != observed_crc:
            raise ValueError(f"Plot has an invalid PNG chunk checksum: {relative}")
        chunks.append((chunk_type, chunk_data))
        offset = chunk_end
        if chunk_type == b"IEND":
            break
    if offset != len(payload):
        raise ValueError(f"Plot contains bytes after PNG IEND: {relative}")
    if not chunks or chunks[0][0] != b"IHDR" or chunks[-1] != (b"IEND", b""):
        raise ValueError(f"Plot has an invalid PNG chunk sequence: {relative}")
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        raise ValueError(f"Plot has an invalid PNG header: {relative}")
    width, height, bit_depth, color_type, compression, filtering, interlace = (
        struct.unpack(">IIBBBBB", ihdr)
    )
    bytes_per_pixel = {4: 2, 6: 4}.get(color_type)
    if (
        width == 0
        or height == 0
        or width * height > _MAX_PNG_PIXELS
        or bit_depth != 8
        or bytes_per_pixel is None
        or (compression, filtering, interlace) != (0, 0, 0)
    ):
        raise ValueError(f"Plot has an unsupported PNG image format: {relative}")
    allowed = {b"IHDR", b"tEXt", b"pHYs", b"IDAT", b"IEND"}
    types = [chunk_type for chunk_type, _ in chunks]
    if any(chunk_type not in allowed for chunk_type in types):
        raise ValueError(f"Plot contains a non-canonical PNG chunk: {relative}")
    if types.count(b"IHDR") != 1 or types.count(b"IEND") != 1:
        raise ValueError(f"Plot has duplicate structural PNG chunks: {relative}")
    if types.count(b"tEXt") > 1 or types.count(b"pHYs") > 1:
        raise ValueError(f"Plot has duplicate metadata PNG chunks: {relative}")
    for chunk_type, chunk_data in chunks:
        if chunk_type == b"tEXt" and chunk_data != b"Software\x00PAISA":
            raise ValueError(f"Plot has unexpected PNG text metadata: {relative}")
        if chunk_type == b"pHYs" and (
            len(chunk_data) != 9
            or chunk_data[-1] != 1
            or struct.unpack(">II", chunk_data[:8])[0] == 0
            or struct.unpack(">II", chunk_data[:8])[1] == 0
        ):
            raise ValueError(f"Plot has invalid PNG resolution metadata: {relative}")
    idat_indices = [index for index, value in enumerate(types) if value == b"IDAT"]
    if not idat_indices or idat_indices != list(
        range(idat_indices[0], idat_indices[-1] + 1)
    ):
        raise ValueError(f"Plot has a non-contiguous PNG image stream: {relative}")
    if any(
        value not in {b"tEXt", b"pHYs"}
        for value in types[1 : idat_indices[0]]
    ) or any(value != b"IEND" for value in types[idat_indices[-1] + 1 :]):
        raise ValueError(f"Plot has a non-canonical PNG chunk order: {relative}")
    compressed = b"".join(
        chunk_data for chunk_type, chunk_data in chunks if chunk_type == b"IDAT"
    )
    expected_image_bytes = height * (1 + width * bytes_per_pixel)
    decompressor = zlib.decompressobj()
    try:
        image_bytes = decompressor.decompress(compressed, expected_image_bytes + 1)
    except zlib.error as exc:
        raise ValueError(f"Plot has an invalid PNG image stream: {relative}") from exc
    if (
        len(image_bytes) != expected_image_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError(f"Plot has an invalid PNG image size: {relative}")
    scanline_bytes = 1 + width * bytes_per_pixel
    if any(image_bytes[row * scanline_bytes] > 4 for row in range(height)):
        raise ValueError(f"Plot has an invalid PNG scanline filter: {relative}")


def _assert_portable_content(pack_root: Path) -> None:
    executable_suffixes = {
        ".joblib",
        ".pkl",
        ".pickle",
        ".bin",
        ".pt",
        ".pth",
        ".h5",
        ".onnx",
    }
    for relative, path in _regular_files(pack_root).items():
        if path.suffix.lower() in executable_suffixes:
            raise ValueError("Portable explanation pack contains an executable model")
        if path.suffix.lower() == ".png":
            payload, _ = _read_regular_bytes(
                path,
                field=f"Portable plot {relative}",
                max_bytes=_MAX_ARTIFACT_FILE_BYTES,
            )
            _validate_png_payload(payload, relative=relative)
            continue
        try:
            payload, _ = _read_regular_bytes(
                path,
                field=f"Portable artifact {relative}",
                max_bytes=_MAX_ARTIFACT_FILE_BYTES,
            )
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Portable artifact is not UTF-8 text: {relative}") from exc
        if _contains_absolute_path(text):
            raise ValueError(f"Portable artifact contains an absolute host path: {relative}")
        if contains_sensitive_text(text):
            raise ValueError(
                f"Portable artifact contains host-path or credential data: {relative}"
            )
        lowered = text.lower()
        for phrase in FORBIDDEN_CLAIMS:
            if phrase == "financial advice":
                without_disclaimer = lowered.replace(ACADEMIC_DISCLAIMER.lower(), "")
                if phrase in without_disclaimer:
                    raise ValueError("Explanation pack contains prohibited advice wording")
            elif phrase in lowered:
                raise ValueError(f"Explanation pack contains prohibited causal claim: {phrase}")


def _create_deterministic_zip(source_dir: Path, zip_path: Path, *, run_id: str) -> str:
    if os.path.lexists(zip_path):
        raise FileExistsError("Explainability ZIP already exists")
    files = _regular_files(source_dir)
    with zipfile.ZipFile(
        zip_path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(
                f"{run_id}/{relative}", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            archive.writestr(info, files[relative].read_bytes(), compresslevel=9)
    return sha256_file(zip_path)


def _expected_layout() -> set[str]:
    expected = set(REQUIRED_FIXED_FILES)
    expected.update(f"local_explanations/{model_id}.csv" for model_id in MODEL_ORDER)
    expected.update(f"summaries/{model_id}.json" for model_id in MODEL_ORDER)
    expected.update(
        f"plots/{model_id}-global-importance.png" for model_id in MODEL_ORDER
    )
    expected.add("plots/cross-model-rank-comparison.png")
    return expected


def _validate_output_separation(output_root: Path, sources: SourceEvidence) -> None:
    candidate = output_root.resolve(strict=False)
    source_roots = (
        sources.training_root,
        sources.comparison_root,
        sources.vertical_root,
    )
    for source in source_roots:
        resolved_source = source.resolve(strict=True)
        try:
            candidate.relative_to(resolved_source)
        except ValueError:
            pass
        else:
            raise ValueError("Explainability output cannot be inside a source run")
        try:
            resolved_source.relative_to(candidate)
        except ValueError:
            pass
        else:
            raise ValueError("Explainability output cannot contain a source run")


def _publish_staged_file(
    staged: Path,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[int, int, int]:
    metadata = staged.lstat()
    if staged.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Staged portable artifact must be a regular file")
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)
    expected_digest = expected_sha256 or sha256_file(staged)
    if sha256_file(staged) != expected_digest:
        raise RuntimeError("Staged portable artifact changed before publication")
    moved = False
    try:
        atomic_rename_noreplace(staged, destination)
        moved = True
        published = destination.lstat()
        if (published.st_dev, published.st_ino, published.st_size) != identity:
            raise RuntimeError("Portable artifact identity changed during publication")
        if sha256_file(destination) != expected_digest:
            raise RuntimeError("Portable artifact content changed during publication")
    except Exception:
        if moved:
            _rollback_published_file(destination, identity=identity)
        raise
    return identity


def _rollback_published_file(
    destination: Path, *, identity: tuple[int, int, int]
) -> None:
    """Remove only the exact file inode published by this failed build."""

    try:
        metadata = destination.lstat()
    except FileNotFoundError:
        return
    observed = (metadata.st_dev, metadata.st_ino, metadata.st_size)
    if destination.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("Refusing to roll back a replaced publication target")
    if observed != identity:
        raise RuntimeError("Refusing to roll back a changed publication target")
    destination.unlink()


def build_tree_shap_explainability(
    *,
    model_training_run: str | Path,
    comparison_run: str | Path,
    vertical_slice_run: str | Path,
    output_dir: str | Path,
    run_id: str | None = None,
    trusted_model_training_root: str | Path | None = None,
    create_zip: bool = True,
) -> dict[str, Any]:
    """Build and atomically publish one immutable four-model SHAP evidence pack."""

    resolved_run_id = _safe_run_id(run_id or make_explainability_run_id())
    repo_root = Path(__file__).resolve().parents[3]
    trusted_root = Path(
        trusted_model_training_root
        or os.environ.get(
            "PAISA_MODEL_TRAINING_ROOT", repo_root / "artifacts" / "model_training"
        )
    )
    versions = _versions()
    sources = _validate_sources(
        model_training_run=model_training_run,
        comparison_run=comparison_run,
        vertical_slice_run=vertical_slice_run,
        trusted_model_training_root=trusted_root,
    )
    output_candidate = reject_symlink_components(
        Path(output_dir), label="Explainability output root"
    )
    _validate_output_separation(output_candidate, sources)
    output_root = prepare_output_root(
        output_candidate,
        repository_root=repo_root,
        label="Explainability output root",
    )
    _validate_output_separation(output_root, sources)

    final_dir = output_root / resolved_run_id
    final_zip = output_root / f"{resolved_run_id}.zip"
    final_sidecar = output_root / f"{resolved_run_id}.zip.sha256"
    # A run ID owns the directory and both portable sibling names even when a
    # caller opts out of ZIP creation; divergent artifacts under one ID are
    # never allowed.
    targets = (final_dir, final_zip, final_sidecar)
    for target in targets:
        if os.path.lexists(target):
            raise FileExistsError("Explainability output already exists; refusing collision")

    stage_parent = Path(
        tempfile.mkdtemp(prefix=f".{resolved_run_id}.staging-", dir=output_root)
    )
    stage_dir = stage_parent / resolved_run_id
    stage_dir.mkdir(mode=0o700)
    staged_zip = stage_parent / f"{resolved_run_id}.zip"
    staged_sidecar = stage_parent / f"{resolved_run_id}.zip.sha256"
    published = False
    zip_sha256: str | None = None
    try:
        model_rows, background_rows, prediction_frames, holdout_contract = _load_holdout(
            sources
        )
        explanations: list[ModelExplanation] = []
        model_root = sources.training_root / "models"
        for variant_id in MODEL_ORDER:
            expected_model = EXPECTED_MODELS[variant_id]
            registered_id = expected_model["registered_model_id"]
            model_directory = model_root / registered_id
            pipeline, manifest = load_model_artifact(
                model_directory,
                trusted_root=model_root,
                expected_manifest_sha256=expected_model["model_manifest_sha256"],
                expected_artifact_sha256=expected_model["model_binary_sha256"],
            )
            explanations.append(
                _explain_model(
                    variant_id=variant_id,
                    pipeline=pipeline,
                    manifest=manifest,
                    model_rows=model_rows,
                    background_rows=background_rows,
                    predictions=prediction_frames[variant_id],
                )
            )
            assert_training_run_artifacts_unchanged(sources.training)

        local = pd.concat(
            [item.local_rows for item in explanations], ignore_index=True
        )
        global_frame = compute_global_importance(local)
        cross_frame = compute_cross_model_rankings(global_frame)
        audit = _select_audit_rows(local)
        contracts = [item.contract for item in explanations]
        additivity = _additivity_report(local, contracts)
        if additivity["status"] != "PASS":
            raise ValueError("Official explanation run failed the additivity contract")
        agreements = _rank_agreements(global_frame)

        (stage_dir / "local_explanations").mkdir(mode=0o700)
        (stage_dir / "summaries").mkdir(mode=0o700)
        (stage_dir / "plots").mkdir(mode=0o700)
        for item in explanations:
            _write_csv(
                item.local_rows,
                stage_dir / "local_explanations" / f"{item.variant_id}.csv",
            )
        _write_csv(global_frame, stage_dir / "global_feature_importance.csv")
        _write_csv(cross_frame, stage_dir / "cross_model_feature_rankings.csv")
        _write_csv(audit, stage_dir / "audit_sample.csv")
        write_json(
            _source_references(sources, explanations=explanations),
            stage_dir / "source_run_references.json",
        )
        write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "models": contracts,
                "output_space_rule": (
                    "Attributions and direct outputs are positive-class probabilities. "
                    "No raw-margin or log-odds values are mixed into this run."
                ),
            },
            stage_dir / "explainer_contract.json",
        )
        write_json(
            _feature_matrix_contract(
                holdout_contract=holdout_contract,
                explanations=explanations,
            ),
            stage_dir / "feature_matrix_contract.json",
        )
        write_json(additivity, stage_dir / "additivity_report.json")
        for model_id in MODEL_ORDER:
            write_json(
                _model_summary(
                    model_id,
                    local=local,
                    global_frame=global_frame,
                    audit=audit,
                    agreements=agreements,
                ),
                stage_dir / "summaries" / f"{model_id}.json",
            )
            _plot_global_importance(
                global_frame,
                stage_dir / "plots" / f"{model_id}-global-importance.png",
                model_id=model_id,
            )
        _plot_cross_model_ranks(
            cross_frame, stage_dir / "plots" / "cross-model-rank-comparison.png"
        )

        write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "PASS",
                "checks": [
                    {"check": check, "status": "PASS"} for check in QUALITY_CHECKS
                ],
                "disclaimer": ACADEMIC_DISCLAIMER,
            },
            stage_dir / "quality_report.json",
        )
        final_summary = _final_summary(
            run_id=resolved_run_id,
            versions=versions,
            global_frame=global_frame,
            additivity=additivity,
            audit=audit,
            agreements=agreements,
        )
        write_json(final_summary, stage_dir / "final_summary.json")
        provenance = code_provenance(repo_root)
        write_json(
            {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": ARTIFACT_TYPE,
                "run_id": resolved_run_id,
                "status": "PASS",
                "created_at": datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                ),
                "ticker": TICKER,
                "holdout_terminology": HOLDOUT_TERM,
                "model_ids": list(MODEL_ORDER),
                "registered_model_ids": [
                    EXPECTED_MODELS[model_id]["registered_model_id"]
                    for model_id in MODEL_ORDER
                ],
                "package_versions": versions,
                "background_selection": holdout_contract["background_policy"],
                "random_seed": None,
                "code_provenance": provenance,
                "source_references": "source_run_references.json",
                "explainer_contract": "explainer_contract.json",
                "feature_matrix_contract": "feature_matrix_contract.json",
                "final_summary": "final_summary.json",
                "disclaimer": ACADEMIC_DISCLAIMER,
            },
            stage_dir / "run_manifest.json",
        )
        _write_artifact_manifest(stage_dir, run_id=resolved_run_id)
        _make_tree_private(stage_dir)
        _assert_portable_content(stage_dir)
        staged_result = verify_explainability_pack(stage_dir)
        if staged_result["status"] != "PASS":  # pragma: no cover - defensive
            raise ValueError("Staged explainability pack did not verify PASS")

        if create_zip:
            zip_sha256 = _create_deterministic_zip(
                stage_dir, staged_zip, run_id=resolved_run_id
            )
            staged_sidecar.write_text(
                f"{zip_sha256}  {resolved_run_id}.zip\n", encoding="ascii"
            )
            staged_zip.chmod(0o600)
            staged_sidecar.chmod(0o600)
            zipped = verify_explainability_pack(
                staged_zip, checksum=staged_sidecar
            )
            if zipped["status"] != "PASS":  # pragma: no cover - defensive
                raise ValueError("Staged explainability ZIP did not verify PASS")
            bound = verify_explainability_pack(
                stage_dir, checksum=staged_sidecar
            )
            if bound["status"] != "PASS":  # pragma: no cover - defensive
                raise ValueError(
                    "Staged explainability directory and ZIP did not verify PASS"
                )

        # Bind the final private directory bytes to the authenticated ZIP before
        # the last source-integrity gate.  That source revalidation is kept as
        # the final expensive operation before exclusive publication.
        if create_zip:
            verify_explainability_pack(stage_dir, checksum=staged_sidecar)
        else:
            verify_explainability_pack(stage_dir)
        publish_targets = (final_dir, final_zip, final_sidecar)
        if any(os.path.lexists(target) for target in publish_targets):
            raise FileExistsError("Explainability output appeared during staging")
        _assert_sources_unchanged(sources)
        published_files: list[tuple[Path, tuple[int, int, int]]] = []
        try:
            if create_zip:
                # An orphan sidecar is inert; a published ZIP always follows its
                # authenticator. Exact-inode rollback covers a later collision.
                sidecar_sha256 = sha256_file(staged_sidecar)
                published_files.append(
                    (
                        final_sidecar,
                        _publish_staged_file(
                            staged_sidecar,
                            final_sidecar,
                            expected_sha256=sidecar_sha256,
                        ),
                    )
                )
                if zip_sha256 is None:  # pragma: no cover - defensive
                    raise RuntimeError("Explainability ZIP digest is unavailable")
                published_files.append(
                    (
                        final_zip,
                        _publish_staged_file(
                            staged_zip,
                            final_zip,
                            expected_sha256=zip_sha256,
                        ),
                    )
                )
            atomic_rename_noreplace(stage_dir, final_dir)
        except Exception:
            for destination, identity in reversed(published_files):
                _rollback_published_file(destination, identity=identity)
            raise
        published = True
        stage_parent.rmdir()
        return {
            "status": "PASS",
            "run_id": resolved_run_id,
            "artifact_directory": final_dir.name,
            "zip": final_zip.name if create_zip else None,
            "zip_sha256": zip_sha256,
            "models": list(MODEL_ORDER),
            "shap_version": versions["shap"],
            "disclaimer": ACADEMIC_DISCLAIMER,
        }
    finally:
        if not published and stage_parent.exists():
            remove_staging_tree(stage_parent, output_root=output_root)


def _parse_inventory(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping) or set(value) != {
        "algorithm",
        "root",
        "excludes",
        "files",
    }:
        raise ValueError("Explainability inventory schema is invalid")
    if (
        value["algorithm"] != "sha256"
        or value["root"] != "."
        or value["excludes"] != ["artifact_manifest.json"]
        or not isinstance(value["files"], list)
    ):
        raise ValueError("Explainability inventory contract is invalid")
    parsed: list[tuple[str, str]] = []
    for index, raw in enumerate(value["files"]):
        if not isinstance(raw, Mapping) or set(raw) != {"path", "sha256"}:
            raise ValueError("Explainability inventory entry schema is invalid")
        relative = _safe_relative(raw["path"], field=f"inventory.files[{index}].path")
        digest = str(raw["sha256"])
        if not _SHA256.fullmatch(digest):
            raise ValueError("Explainability inventory digest is invalid")
        parsed.append((relative, digest))
    if [path for path, _ in parsed] != sorted(path for path, _ in parsed):
        raise ValueError("Explainability inventory paths must be sorted")
    if len(parsed) != len({path for path, _ in parsed}):
        raise ValueError("Explainability inventory paths must be unique")
    return tuple(parsed)


def _assert_frames_close(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    numeric_columns: set[str],
    label: str,
) -> None:
    if list(actual.columns) != list(expected.columns) or actual.shape != expected.shape:
        raise ValueError(f"{label} schema or shape does not match recomputation")
    for column in actual.columns:
        if column in numeric_columns:
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.allclose(
                np.nan_to_num(left),
                np.nan_to_num(right),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(f"{label} numeric column {column} is inconsistent")
        else:
            left = actual[column].fillna("").astype(str).tolist()
            right = expected[column].fillna("").astype(str).tolist()
            if left != right:
                raise ValueError(f"{label} column {column} is inconsistent")


def _semantic_equal(actual: Any, expected: Any, *, label: str) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise ValueError(f"{label} does not match recomputation")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            raise ValueError(f"{label} numeric type is invalid")
        if not math.isfinite(float(actual)) or not math.isclose(
            float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12
        ):
            raise ValueError(f"{label} numeric value does not match recomputation")
        return
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(f"{label} object schema does not match recomputation")
        for key in expected:
            _semantic_equal(actual[key], expected[key], label=f"{label}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{label} array does not match recomputation")
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _semantic_equal(
                actual_item, expected_item, label=f"{label}[{index}]"
            )
        return
    raise TypeError(f"Unsupported semantic comparison type: {type(expected)!r}")


def _validate_source_references(payload: Mapping[str, Any]) -> None:
    if set(payload) != {
        "schema_version",
        "sources",
        "models",
        "portable_paths",
        "model_binaries_included",
    }:
        raise ValueError("Source reference schema is invalid")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["portable_paths"] is not True
        or payload["model_binaries_included"] is not False
    ):
        raise ValueError("Source reference portability contract is invalid")
    sources = payload["sources"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Source references must contain exactly three sources")
    expected_sources = {
        "model_training": (TRAINING_RUN_ID, TRAINING_RUN_SHA256),
        "unified_comparison": (COMPARISON_RUN_ID, COMPARISON_MANIFEST_SHA256),
        "vertical_slice": (
            VERTICAL_SLICE_RUN_ID,
            VERTICAL_SLICE_MANIFEST_SHA256,
        ),
    }
    roles = [str(record.get("role")) for record in sources if isinstance(record, Mapping)]
    if len(roles) != len(sources) or set(roles) != set(expected_sources):
        raise ValueError("Source references must contain each authoritative role once")
    for record in sources:
        if not isinstance(record, Mapping):
            raise ValueError("Source reference must be an object")
        role = str(record.get("role"))
        if role not in expected_sources:
            raise ValueError("Unexpected source role")
        expected_id, expected_digest = expected_sources[role]
        if (
            record.get("run_id") != expected_id
            or record.get("integrity_root_sha256") != expected_digest
        ):
            raise ValueError("Source run fingerprint is not authoritative")
        if role == "model_training":
            if set(record) != {
                "role",
                "run_id",
                "integrity_root_name",
                "integrity_root_sha256",
                "source_fingerprint",
                "dataset_sha256",
            }:
                raise ValueError("Model-training source reference schema is invalid")
            if (
                record.get("integrity_root_name") != "run_manifest.json"
                or record.get("source_fingerprint") != TRAINING_SOURCE_FINGERPRINT
                or record.get("dataset_sha256") != MODEL_ROWS_SHA256
            ):
                raise ValueError("Source dataset fingerprint is not authoritative")
        elif set(record) != {
            "role",
            "run_id",
            "integrity_root_name",
            "integrity_root_sha256",
        } or record.get("integrity_root_name") != "manifest.json":
            raise ValueError(f"{role} source reference schema is invalid")
    models = payload["models"]
    if (
        not isinstance(models, list)
        or any(not isinstance(record, Mapping) for record in models)
        or [record.get("model_id") for record in models] != list(MODEL_ORDER)
    ):
        raise ValueError("Source model identities are invalid")
    for record in models:
        if set(record) != {
            "model_id",
            "registered_model_id",
            "algorithm",
            "model_binary_sha256",
            "model_manifest_sha256",
            "comparison_prediction_sha256",
        }:
            raise ValueError("Source model reference schema is invalid")
        model_id = str(record["model_id"])
        expected = EXPECTED_MODELS[model_id]
        if (
            record.get("registered_model_id") != expected["registered_model_id"]
            or record.get("algorithm") != expected["algorithm"]
            or record.get("model_binary_sha256") != expected["model_binary_sha256"]
            or record.get("model_manifest_sha256") != expected["model_manifest_sha256"]
            or record.get("comparison_prediction_sha256")
            != expected["comparison_prediction_sha256"]
        ):
            raise ValueError(f"{model_id} source-model fingerprint is invalid")


def _validate_feature_contract(payload: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    required = {
        "schema_version",
        "ticker",
        "holdout_terminology",
        "row_count",
        "feature_date_count",
        "target_date_count",
        "feature_dates",
        "target_dates",
        "holdout_rows",
        "holdout_rows_sha256",
        "feature_value_space",
        "preprocessing_refit",
        "background_policy",
        "background_target_dates",
        "background_target_dates_sha256",
        "source_dataset_sha256",
        "models",
    }
    if set(payload) != required:
        raise ValueError("Feature-matrix contract schema is invalid")
    feature_dates = payload["feature_dates"]
    target_dates = payload["target_dates"]
    holdout_rows = payload["holdout_rows"]
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["ticker"] != TICKER
        or payload["holdout_terminology"] != HOLDOUT_TERM
        or payload["row_count"] != ROW_COUNT
        or payload["feature_date_count"] != ROW_COUNT
        or payload["target_date_count"] != ROW_COUNT
        or payload["preprocessing_refit"] is not False
        or payload["source_dataset_sha256"] != MODEL_ROWS_SHA256
        or payload["holdout_rows_sha256"] != HOLDOUT_ROWS_SHA256
        or payload["feature_value_space"]
        != "output of the original fitted median imputer"
        or payload["background_policy"]
        != (
            "100 chronological training rows selected at evenly spaced deterministic "
            "indices; fitted preprocessing only"
        )
        or not isinstance(feature_dates, list)
        or not isinstance(target_dates, list)
        or not isinstance(holdout_rows, list)
        or len(feature_dates) != ROW_COUNT
        or len(target_dates) != ROW_COUNT
        or len(holdout_rows) != ROW_COUNT
    ):
        raise ValueError("Feature-matrix holdout contract is invalid")
    if _canonical_json_sha256(holdout_rows) != HOLDOUT_ROWS_SHA256:
        raise ValueError("Feature-matrix rows do not match the authoritative holdout")
    if feature_dates != sorted(feature_dates) or target_dates != sorted(target_dates):
        raise ValueError("Feature-matrix dates must be chronological")
    if (
        feature_dates[0] != "2025-12-15"
        or feature_dates[-1] != "2026-09-14"
        or target_dates[0] != "2025-12-16"
        or target_dates[-1] != "2026-09-15"
    ):
        raise ValueError("Feature-matrix date boundaries are invalid")
    for index, row in enumerate(holdout_rows):
        if not isinstance(row, Mapping) or set(row) != {
            "ticker",
            "feature_date",
            "target_date",
            "actual_class",
        }:
            raise ValueError("Feature-matrix holdout row schema is invalid")
        if (
            row["ticker"] != TICKER
            or row["feature_date"] != feature_dates[index]
            or row["target_date"] != target_dates[index]
            or row["actual_class"] not in {0, 1}
        ):
            raise ValueError("Feature-matrix holdout row is inconsistent")
    background_dates = payload["background_target_dates"]
    if (
        not isinstance(background_dates, list)
        or len(background_dates) != BACKGROUND_ROW_COUNT
        or background_dates != sorted(background_dates)
        or len(set(background_dates)) != BACKGROUND_ROW_COUNT
        or any(value >= target_dates[0] for value in background_dates)
        or payload["background_target_dates_sha256"]
        != BACKGROUND_TARGET_DATES_SHA256
        or _canonical_json_sha256(background_dates)
        != BACKGROUND_TARGET_DATES_SHA256
    ):
        raise ValueError("SHAP background dates are not a deterministic training subset")
    models = payload["models"]
    if (
        not isinstance(models, list)
        or any(not isinstance(record, Mapping) for record in models)
        or [record.get("model_id") for record in models] != list(MODEL_ORDER)
    ):
        raise ValueError("Feature-matrix model list is invalid")
    columns_by_model: dict[str, tuple[str, ...]] = {}
    for record in models:
        if set(record) != {
            "model_id",
            "registered_model_id",
            "ordered_feature_columns",
            "feature_count",
            "row_count",
            "transformed_matrix_sha256",
            "target_columns_in_matrix",
            "actual_class_in_matrix",
            "future_values_in_matrix",
        }:
            raise ValueError("Feature-matrix model schema is invalid")
        model_id = str(record["model_id"])
        columns = tuple(validate_feature_columns(record.get("ordered_feature_columns", [])))
        if (
            record.get("registered_model_id")
            != EXPECTED_MODELS[model_id]["registered_model_id"]
            or record.get("feature_count") != len(columns)
            or record.get("row_count") != ROW_COUNT
            or record.get("target_columns_in_matrix") is not False
            or record.get("actual_class_in_matrix") is not False
            or record.get("future_values_in_matrix") is not False
            or record.get("transformed_matrix_sha256")
            != TRANSFORMED_HOLDOUT_SHA256
            or set(columns) & FORBIDDEN_FEATURE_NAMES
            or any(name.startswith(("target_", "next_")) for name in columns)
            or columns != EXPECTED_FEATURE_COLUMNS
        ):
            raise ValueError(f"{model_id} feature-matrix contract is invalid")
        columns_by_model[model_id] = columns
    return columns_by_model


def _validate_explainer_contract(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if set(payload) != {"schema_version", "models", "output_space_rule"}:
        raise ValueError("Explainer contract schema is invalid")
    models = payload["models"]
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["output_space_rule"]
        != (
            "Attributions and direct outputs are positive-class probabilities. "
            "No raw-margin or log-odds values are mixed into this run."
        )
        or not isinstance(models, list)
    ):
        raise ValueError("Explainer contract is invalid")
    if [record.get("model_id") for record in models] != list(MODEL_ORDER):
        raise ValueError("Explainer contract model order is invalid")
    for record in models:
        model_id = str(record["model_id"])
        expected = EXPECTED_MODELS[model_id]
        required = {
            "schema_version",
            "model_id",
            "registered_model_id",
            "explainer_type",
            "algorithm",
            "estimator_class",
            "class_explained",
            "estimator_class_order",
            "expected_value",
            "output_space",
            "feature_perturbation",
            "background_data_policy",
            "background_row_count",
            "background_matrix_sha256",
            "shap_version",
            "feature_count",
            "row_count",
            "additivity_tolerance",
        }
        if set(record) != required:
            raise ValueError(f"{model_id} explainer contract schema is invalid")
        if (
            record["schema_version"] != SCHEMA_VERSION
            or record["registered_model_id"] != expected["registered_model_id"]
            or record["explainer_type"] != "shap.TreeExplainer"
            or record["algorithm"] != expected["algorithm"]
            or record["estimator_class"] != expected["estimator_class"]
            or record["class_explained"] != 1
            or record["estimator_class_order"] != [0, 1]
            or record["output_space"] != "probability"
            or record["feature_perturbation"] != "interventional"
            or record["background_data_policy"]
            != (
                "100 evenly spaced rows from the chronological training split; "
                "original fitted median imputer only"
            )
            or record["background_row_count"] != BACKGROUND_ROW_COUNT
            or record["shap_version"] != SHAP_PIN
            or record["row_count"] != ROW_COUNT
            or record["feature_count"] != len(EXPECTED_FEATURE_COLUMNS)
            or float(record["additivity_tolerance"])
            != ADDITIVITY_TOLERANCES[expected["algorithm"]]
            or record["background_matrix_sha256"]
            != TRANSFORMED_BACKGROUND_SHA256
            or not math.isfinite(float(record["expected_value"]))
            or not 0 <= float(record["expected_value"]) <= 1
        ):
            raise ValueError(f"{model_id} explainer contract is invalid")
    return [dict(record) for record in models]


def _read_and_validate_local(
    pack_root: Path,
    *,
    columns_by_model: Mapping[str, tuple[str, ...]],
    feature_contract: Mapping[str, Any],
    explainer_contracts: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    holdout_rows = feature_contract["holdout_rows"]
    feature_dates = [str(row["feature_date"]) for row in holdout_rows]
    target_dates = [str(row["target_date"]) for row in holdout_rows]
    actual_classes = [int(row["actual_class"]) for row in holdout_rows]
    contracts = {str(record["model_id"]): record for record in explainer_contracts}
    feature_records = {
        str(record["model_id"]): record for record in feature_contract["models"]
    }
    frames: list[pd.DataFrame] = []
    numeric_columns = {
        "predicted_class",
        "probability_up",
        "probability_down",
        "actual_class",
        "expected_value",
        "direct_model_output",
        "reconstructed_model_output",
        "additivity_error",
        "feature_value",
        "shap_value",
        "absolute_contribution_rank",
    }
    for model_id in MODEL_ORDER:
        path = pack_root / "local_explanations" / f"{model_id}.csv"
        frame = pd.read_csv(path, keep_default_na=False)
        if tuple(frame.columns) != LOCAL_COLUMNS:
            raise ValueError(f"{model_id} local explanation columns are invalid")
        feature_columns = columns_by_model[model_id]
        expected_count = ROW_COUNT * len(feature_columns)
        if len(frame) != expected_count:
            raise ValueError(f"{model_id} local explanation row count is invalid")
        if frame.duplicated(
            ["model_id", "ticker", "feature_date", "target_date", "feature_name"]
        ).any():
            raise ValueError(f"{model_id} local explanation keys must be unique")
        if (
            set(frame["schema_version"].astype(str)) != {SCHEMA_VERSION}
            or set(frame["model_id"].astype(str)) != {model_id}
            or set(frame["registered_model_id"].astype(str))
            != {EXPECTED_MODELS[model_id]["registered_model_id"]}
            or set(frame["algorithm"].astype(str))
            != {EXPECTED_MODELS[model_id]["algorithm"]}
            or set(frame["ticker"].astype(str)) != {TICKER}
            or set(frame["output_space"].astype(str)) != {"probability"}
        ):
            raise ValueError(f"{model_id} local explanation identity is invalid")
        for column in numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError(f"{model_id} {column} values must be finite")
            frame[column] = values
        for column in ("predicted_class", "actual_class", "absolute_contribution_rank"):
            values = frame[column].to_numpy(float)
            if not np.array_equal(values, np.floor(values)):
                raise ValueError(f"{model_id} {column} values must be integers")
        if not frame["predicted_class"].isin([0.0, 1.0]).all() or not frame[
            "actual_class"
        ].isin([0.0, 1.0]).all():
            raise ValueError(f"{model_id} class values are invalid")
        if (
            (frame["probability_up"] < 0).any()
            or (frame["probability_up"] > 1).any()
            or (frame["probability_down"] < 0).any()
            or (frame["probability_down"] > 1).any()
            or not np.allclose(
                frame["probability_up"] + frame["probability_down"],
                1.0,
                atol=1e-6,
                rtol=0,
            )
        ):
            raise ValueError(f"{model_id} probability fields are invalid")
        expected_direction = np.where(
            frame["shap_value"] > 0,
            "increased_model_output",
            np.where(
                frame["shap_value"] < 0, "decreased_model_output", "neutral"
            ),
        )
        if frame["contribution_direction"].astype(str).tolist() != expected_direction.tolist():
            raise ValueError(f"{model_id} contribution directions are invalid")
        if set(frame["feature_name"].astype(str)) != set(feature_columns):
            raise ValueError(f"{model_id} local feature set is invalid")
        if set(frame["feature_name"].astype(str)) & FORBIDDEN_FEATURE_NAMES:
            raise ValueError(f"{model_id} local explanation contains leakage fields")
        matrix_rows: list[list[float]] = []
        for row_index, target_date in enumerate(target_dates):
            group = frame.loc[frame["target_date"].astype(str) == target_date]
            if len(group) != len(feature_columns):
                raise ValueError(f"{model_id} does not explain every feature/date")
            if group["feature_name"].astype(str).tolist() != list(feature_columns):
                raise ValueError(f"{model_id} local feature ordering is invalid")
            if (
                set(group["feature_date"].astype(str)) != {feature_dates[row_index]}
                or set(group["actual_class"].astype(int)) != {actual_classes[row_index]}
            ):
                raise ValueError(f"{model_id} holdout alignment is invalid")
            matrix_rows.append(group["feature_value"].astype(float).tolist())
        matrix = pd.DataFrame(matrix_rows, columns=feature_columns)
        if (
            _frame_sha256(matrix)
            != feature_records[model_id]["transformed_matrix_sha256"]
        ):
            raise ValueError(f"{model_id} transformed feature matrix is inconsistent")
        predictions = _prediction_rows(frame)
        if predictions["target_date"].astype(str).tolist() != target_dates:
            raise ValueError(f"{model_id} local target dates are invalid")
        expected_class = (
            predictions["probability_up"].astype(float)
            > predictions["probability_down"].astype(float)
        ).astype(int)
        if not np.array_equal(
            expected_class.to_numpy(), predictions["predicted_class"].astype(int).to_numpy()
        ):
            raise ValueError(f"{model_id} local predicted classes are invalid")
        if not np.allclose(
            predictions["direct_model_output"].astype(float),
            predictions["probability_up"].astype(float),
            atol=PREDICTION_REPLAY_TOLERANCE,
            rtol=0,
        ):
            raise ValueError(f"{model_id} direct output must equal class-1 probability")
        contract = contracts[model_id]
        if not np.allclose(
            predictions["expected_value"].astype(float),
            float(contract["expected_value"]),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError(f"{model_id} expected values disagree with the contract")
        if (
            predictions["absolute_additivity_error"].astype(float)
            > float(contract["additivity_tolerance"])
        ).any():
            raise ValueError(f"{model_id} additivity exceeds its declared tolerance")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _verify_pack_directory(pack_root: Path) -> dict[str, Any]:
    if pack_root.is_symlink() or not pack_root.is_dir():
        raise ValueError("Explainability pack must be a real directory")
    all_files = _regular_files(pack_root)
    if not all_files or len(all_files) > _MAX_ZIP_FILES:
        raise ValueError("Explainability directory has an invalid file count")
    total_size = 0
    for relative, path in all_files.items():
        size = path.stat(follow_symlinks=False).st_size
        if size > _MAX_ARTIFACT_FILE_BYTES:
            raise ValueError(
                f"Explainability artifact exceeds its size limit: {relative}"
            )
        total_size += size
        if total_size > _MAX_ZIP_BYTES:
            raise ValueError("Explainability directory exceeds its total size limit")
    manifest = _read_json(
        pack_root / "artifact_manifest.json", field="Explainability artifact manifest"
    )
    if set(manifest) != {
        "schema_version",
        "artifact_type",
        "run_id",
        "status",
        "inventory",
        "disclaimer",
    }:
        raise ValueError("Explainability artifact manifest schema is invalid")
    run_id = _safe_run_id(manifest.get("run_id"))
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
        or manifest["status"] != "PASS"
        or manifest["disclaimer"] != ACADEMIC_DISCLAIMER
        or pack_root.name != run_id
    ):
        raise ValueError("Explainability manifest identity or status is invalid")
    inventory_entries = _parse_inventory(manifest["inventory"])
    inventory = dict(inventory_entries)
    actual_files = {
        relative: path
        for relative, path in all_files.items()
        if relative != "artifact_manifest.json"
    }
    if set(actual_files) != set(inventory):
        missing = sorted(set(inventory) - set(actual_files))
        unexpected = sorted(set(actual_files) - set(inventory))
        raise ValueError(
            "Explainability inventory must exactly match files; "
            f"missing={missing}, unexpected={unexpected}"
        )
    complete_files = set(actual_files) | {"artifact_manifest.json"}
    if complete_files != _expected_layout():
        raise ValueError("Explainability artifact layout is incomplete or unexpected")
    for relative, expected_sha in inventory_entries:
        if sha256_file(actual_files[relative]) != expected_sha:
            raise ValueError(f"Explainability artifact checksum mismatch: {relative}")
    _assert_portable_content(pack_root)

    source_references = _read_json(
        pack_root / "source_run_references.json", field="Source run references"
    )
    _validate_source_references(source_references)
    feature_contract = _read_json(
        pack_root / "feature_matrix_contract.json", field="Feature matrix contract"
    )
    columns_by_model = _validate_feature_contract(feature_contract)
    explainer_payload = _read_json(
        pack_root / "explainer_contract.json", field="Explainer contract"
    )
    explainer_contracts = _validate_explainer_contract(explainer_payload)
    for contract in explainer_contracts:
        model_id = str(contract["model_id"])
        if int(contract["feature_count"]) != len(columns_by_model[model_id]):
            raise ValueError("Explainer and feature-matrix feature counts disagree")

    local = _read_and_validate_local(
        pack_root,
        columns_by_model=columns_by_model,
        feature_contract=feature_contract,
        explainer_contracts=explainer_contracts,
    )
    expected_global = compute_global_importance(local)
    actual_global = pd.read_csv(pack_root / "global_feature_importance.csv")
    global_numeric = {
        "mean_absolute_shap_value",
        "signed_mean_shap_value",
        "median_absolute_shap_value",
        "maximum_absolute_shap_value",
        "feature_rank",
        "share_of_total_absolute_attribution",
        "positive_contribution_count",
        "negative_contribution_count",
    }
    _assert_frames_close(
        actual_global,
        expected_global,
        numeric_columns=global_numeric,
        label="Global feature importance",
    )
    expected_cross = compute_cross_model_rankings(expected_global)
    actual_cross = pd.read_csv(pack_root / "cross_model_feature_rankings.csv")
    cross_numeric = {
        column
        for column in expected_cross.columns
        if column.endswith("_rank")
        or column.endswith("_rank_range")
        or column.endswith("_rank_standard_deviation")
        or column.endswith("_mean_absolute_shap_value")
    }
    _assert_frames_close(
        actual_cross,
        expected_cross,
        numeric_columns=cross_numeric,
        label="Cross-model feature rankings",
    )
    expected_audit = _select_audit_rows(local)
    actual_audit = pd.read_csv(pack_root / "audit_sample.csv", keep_default_na=False)
    _assert_frames_close(
        actual_audit,
        expected_audit,
        numeric_columns={
            "predicted_class",
            "actual_class",
            "probability_up",
            "probability_down",
            "absolute_additivity_error",
        },
        label="Audit sample",
    )
    expected_additivity = _additivity_report(local, explainer_contracts)
    actual_additivity = _read_json(
        pack_root / "additivity_report.json", field="Additivity report"
    )
    _semantic_equal(actual_additivity, expected_additivity, label="additivity_report")
    if actual_additivity["status"] != "PASS":
        raise ValueError("Explainability additivity status must be PASS")

    agreements = _rank_agreements(expected_global)
    for model_id in MODEL_ORDER:
        actual_summary = _read_json(
            pack_root / "summaries" / f"{model_id}.json",
            field=f"{model_id} narrative summary",
        )
        expected_summary = _model_summary(
            model_id,
            local=local,
            global_frame=expected_global,
            audit=expected_audit,
            agreements=agreements,
        )
        _semantic_equal(
            actual_summary, expected_summary, label=f"summaries.{model_id}"
        )

    run_manifest = _read_json(
        pack_root / "run_manifest.json", field="Explainability run manifest"
    )
    required_run_manifest = {
        "schema_version",
        "artifact_type",
        "run_id",
        "status",
        "created_at",
        "ticker",
        "holdout_terminology",
        "model_ids",
        "registered_model_ids",
        "package_versions",
        "background_selection",
        "random_seed",
        "code_provenance",
        "source_references",
        "explainer_contract",
        "feature_matrix_contract",
        "final_summary",
        "disclaimer",
    }
    package_versions = run_manifest.get("package_versions")
    code_record = run_manifest.get("code_provenance")
    if (
        set(run_manifest) != required_run_manifest
        or run_manifest.get("schema_version") != SCHEMA_VERSION
        or run_manifest.get("artifact_type") != ARTIFACT_TYPE
        or run_manifest.get("run_id") != run_id
        or run_manifest.get("status") != "PASS"
        or run_manifest.get("ticker") != TICKER
        or run_manifest.get("holdout_terminology") != HOLDOUT_TERM
        or run_manifest.get("model_ids") != list(MODEL_ORDER)
        or run_manifest.get("registered_model_ids")
        != [EXPECTED_MODELS[model_id]["registered_model_id"] for model_id in MODEL_ORDER]
        or not isinstance(package_versions, Mapping)
        or set(package_versions) != set(PACKAGE_VERSION_KEYS)
        or any(
            not isinstance(package_versions[key], str) or not package_versions[key]
            for key in PACKAGE_VERSION_KEYS
        )
        or package_versions["shap"] != SHAP_PIN
        or run_manifest.get("background_selection")
        != feature_contract["background_policy"]
        or run_manifest.get("random_seed") is not None
        or not isinstance(code_record, Mapping)
        or set(code_record)
        != {"code_version", "dirty_tree", "source_fingerprint"}
        or not isinstance(code_record.get("code_version"), str)
        or not code_record.get("code_version")
        or not isinstance(code_record.get("source_fingerprint"), str)
        or not code_record.get("source_fingerprint")
        or not isinstance(code_record.get("dirty_tree"), bool)
        or run_manifest.get("source_references") != "source_run_references.json"
        or run_manifest.get("explainer_contract") != "explainer_contract.json"
        or run_manifest.get("feature_matrix_contract")
        != "feature_matrix_contract.json"
        or run_manifest.get("final_summary") != "final_summary.json"
        or run_manifest.get("disclaimer") != ACADEMIC_DISCLAIMER
    ):
        raise ValueError("Explainability run manifest is inconsistent")
    created_at = str(run_manifest.get("created_at", ""))
    if not created_at.endswith("Z"):
        raise ValueError("Explainability created_at must be UTC and end in Z")
    try:
        created_timestamp = datetime.fromisoformat(created_at[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError("Explainability created_at must be ISO 8601") from exc
    if created_timestamp.utcoffset() != timezone.utc.utcoffset(created_timestamp):
        raise ValueError("Explainability created_at must be UTC")

    expected_final = _final_summary(
        run_id=run_id,
        versions=run_manifest["package_versions"],
        global_frame=expected_global,
        additivity=expected_additivity,
        audit=expected_audit,
        agreements=agreements,
    )
    actual_final = _read_json(
        pack_root / "final_summary.json", field="Explainability final summary"
    )
    _semantic_equal(actual_final, expected_final, label="final_summary")
    quality = _read_json(pack_root / "quality_report.json", field="Quality report")
    if (
        set(quality) != {"schema_version", "status", "checks", "disclaimer"}
        or quality.get("schema_version") != SCHEMA_VERSION
        or quality.get("status") != "PASS"
        or quality.get("disclaimer") != ACADEMIC_DISCLAIMER
        or not isinstance(quality.get("checks"), list)
        or quality["checks"]
        != [{"check": check, "status": "PASS"} for check in QUALITY_CHECKS]
    ):
        raise ValueError("Explainability quality report is invalid")
    return {
        "status": "PASS",
        "run_id": run_id,
        "verified_file_count": len(complete_files),
        "models": list(MODEL_ORDER),
        "row_count_per_model": ROW_COUNT,
        "shap_version": SHAP_PIN,
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


def _read_checksum_sidecar(sidecar: Path, *, zip_path: Path) -> str:
    if sidecar.name != f"{zip_path.name}.sha256":
        raise ValueError("Checksum sidecar name must match the ZIP filename")
    try:
        payload, _ = _read_regular_bytes(
            sidecar, field="Checksum sidecar", max_bytes=512
        )
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Checksum sidecar must contain ASCII") from exc
    match = re.fullmatch(
        rf"([0-9a-f]{{64}})  {re.escape(zip_path.name)}\n", text
    )
    if match is None:
        raise ValueError("Checksum sidecar must contain one strict SHA-256 record")
    return match.group(1)


def _validated_zip_members(
    archive: zipfile.ZipFile,
) -> tuple[str, list[zipfile.ZipInfo]]:
    if archive.comment:
        raise ValueError("Explainability ZIP must not contain an archive comment")
    members = archive.infolist()
    if not members or len(members) > _MAX_ZIP_FILES:
        raise ValueError("Explainability ZIP has an invalid member count")
    names = [member.filename for member in members]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError(
            "Explainability ZIP members must be sorted and duplicate-free"
        )
    total_size = 0
    roots: set[str] = set()
    for index, member in enumerate(members):
        name = _safe_relative(member.filename, field=f"ZIP member {index}")
        if name != member.filename or member.is_dir():
            raise ValueError(
                "Explainability ZIP may contain only normalized regular files"
            )
        path = PurePosixPath(name)
        if len(path.parts) < 2:
            raise ValueError(
                "Every ZIP member must be under one explainability directory"
            )
        roots.add(path.parts[0])
        expected_external_attr = (stat.S_IFREG | 0o600) << 16
        if (
            member.flag_bits != 0
            or member.compress_type != zipfile.ZIP_DEFLATED
            or member.date_time != (1980, 1, 1, 0, 0, 0)
            or member.create_system != 3
            or member.create_version != 20
            or member.extract_version != 20
            or member.external_attr != expected_external_attr
            or member.extra
            or member.comment
            or member.volume != 0
            or member.internal_attr != 0
        ):
            raise ValueError(f"ZIP member metadata is not canonical: {name}")
        if member.file_size > _MAX_ARTIFACT_FILE_BYTES:
            raise ValueError(f"ZIP member exceeds its size limit: {name}")
        total_size += member.file_size
        if total_size > _MAX_ZIP_BYTES:
            raise ValueError("Explainability ZIP exceeds the uncompressed size limit")
        if member.file_size and (
            member.compress_size == 0
            or member.file_size / member.compress_size > 1000
        ):
            raise ValueError("Explainability ZIP has an unsafe compression ratio")
    if len(roots) != 1:
        raise ValueError(
            "Explainability ZIP must contain exactly one top-level directory"
        )
    run_id = _safe_run_id(next(iter(roots)))
    return run_id, members


def _validate_canonical_zip_bytes(
    payload: bytes, members: Sequence[zipfile.ZipInfo]
) -> None:
    if len(payload) < 22 or payload[-22:-18] != b"PK\x05\x06":
        raise ValueError("Explainability ZIP has no canonical end record")
    (
        signature,
        disk_number,
        central_disk,
        entries_on_disk,
        entry_count,
        central_size,
        central_offset,
        comment_length,
    ) = struct.unpack_from("<4s4H2LH", payload, len(payload) - 22)
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_disk != 0
        or entries_on_disk != len(members)
        or entry_count != len(members)
        or comment_length != 0
        or central_offset + central_size != len(payload) - 22
    ):
        raise ValueError("Explainability ZIP end record is not canonical")

    cursor = 0
    for member in members:
        if member.header_offset != cursor or cursor + 30 > central_offset:
            raise ValueError("Explainability ZIP local-header offsets are invalid")
        (
            local_signature,
            extract_version,
            flags,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
        ) = struct.unpack_from("<4s5H3L2H", payload, cursor)
        filename = member.filename.encode("utf-8")
        name_start = cursor + 30
        name_end = name_start + filename_length
        data_start = name_end + extra_length
        data_end = data_start + compressed_size
        if (
            local_signature != b"PK\x03\x04"
            or extract_version != 20
            or flags != 0
            or compression != zipfile.ZIP_DEFLATED
            or modified_time != 0
            or modified_date != 33
            or crc != member.CRC
            or compressed_size != member.compress_size
            or uncompressed_size != member.file_size
            or filename_length != len(filename)
            or extra_length != 0
            or name_end > central_offset
            or payload[name_start:name_end] != filename
            or data_end > central_offset
        ):
            raise ValueError(
                f"Explainability ZIP local header is not canonical: {member.filename}"
            )
        decompressor = zlib.decompressobj(wbits=-15)
        try:
            uncompressed = decompressor.decompress(
                payload[data_start:data_end], member.file_size + 1
            )
        except zlib.error as exc:
            raise ValueError(
                f"Explainability ZIP member stream is invalid: {member.filename}"
            ) from exc
        if (
            len(uncompressed) != member.file_size
            or not decompressor.eof
            or decompressor.unused_data
            or decompressor.unconsumed_tail
            or (zlib.crc32(uncompressed) & 0xFFFFFFFF) != member.CRC
        ):
            raise ValueError(
                f"Explainability ZIP member stream is not canonical: {member.filename}"
            )
        cursor = data_end
    if cursor != central_offset:
        raise ValueError("Explainability ZIP contains bytes outside canonical members")

    cursor = central_offset
    central_end = central_offset + central_size
    for member in members:
        if cursor + 46 > central_end:
            raise ValueError("Explainability ZIP central directory is truncated")
        (
            central_signature,
            made_by,
            extract_version,
            flags,
            compression,
            modified_time,
            modified_date,
            crc,
            compressed_size,
            uncompressed_size,
            filename_length,
            extra_length,
            comment_length,
            disk_start,
            internal_attr,
            external_attr,
            local_offset,
        ) = struct.unpack_from("<4s6H3L5H2L", payload, cursor)
        filename = member.filename.encode("utf-8")
        name_start = cursor + 46
        name_end = name_start + filename_length
        if (
            central_signature != b"PK\x01\x02"
            or made_by != (3 << 8) | 20
            or extract_version != 20
            or flags != 0
            or compression != zipfile.ZIP_DEFLATED
            or modified_time != 0
            or modified_date != 33
            or crc != member.CRC
            or compressed_size != member.compress_size
            or uncompressed_size != member.file_size
            or filename_length != len(filename)
            or extra_length != 0
            or comment_length != 0
            or disk_start != 0
            or internal_attr != 0
            or external_attr != member.external_attr
            or local_offset != member.header_offset
            or name_end > central_end
            or payload[name_start:name_end] != filename
        ):
            raise ValueError(
                f"Explainability ZIP central entry is not canonical: {member.filename}"
            )
        cursor = name_end
    if cursor != central_end:
        raise ValueError("Explainability ZIP central directory has trailing records")


def _assert_file_identity(
    path: Path,
    identity: tuple[int, int, int, int, int],
    *,
    field: str,
) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise RuntimeError(f"{field} changed during verification") from exc
    if not stat.S_ISREG(metadata.st_mode) or _stat_identity(metadata) != identity:
        raise RuntimeError(f"{field} changed during verification")


def _verify_zip_archive(
    zip_path: Path, *, expected_sha256: str
) -> tuple[dict[str, Any], dict[str, str]]:
    payload, identity = _read_regular_bytes(
        zip_path,
        field="Explainability ZIP",
        max_bytes=_MAX_ZIP_BYTES,
    )
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("Explainability ZIP checksum does not match its sidecar")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            run_id, members = _validated_zip_members(archive)
            _validate_canonical_zip_bytes(payload, members)
            if zip_path.name != f"{run_id}.zip":
                raise ValueError(
                    "Explainability ZIP filename must match its immutable run ID"
                )
            with tempfile.TemporaryDirectory(prefix="paisa-explainability-verify-") as temp:
                temp_root = Path(temp).resolve(strict=True)
                for member in members:
                    relative = PurePosixPath(member.filename)
                    destination = temp_root.joinpath(*relative.parts).resolve(
                        strict=False
                    )
                    try:
                        destination.relative_to(temp_root)
                    except ValueError as exc:
                        raise ValueError(
                            "Explainability ZIP member escapes the extraction root"
                        ) from exc
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member, "r") as source, destination.open(
                        "xb"
                    ) as target:
                        shutil.copyfileobj(source, target)
                pack_root = temp_root / run_id
                result = _verify_pack_directory(pack_root)
                expected_names = [
                    f"{run_id}/{relative}"
                    for relative in sorted(_regular_files(pack_root))
                ]
                if [member.filename for member in members] != expected_names:
                    raise ValueError(
                        "ZIP member list does not match the verified explainability pack"
                    )
                content_sha256 = {
                    relative: sha256_file(path)
                    for relative, path in _regular_files(pack_root).items()
                }
                _assert_file_identity(
                    zip_path, identity, field="Explainability ZIP"
                )
                return result, content_sha256
    except zipfile.BadZipFile as exc:
        raise ValueError("Explainability ZIP is malformed or corrupt") from exc


def _snapshot_pack_directory(
    source: Path, snapshot_parent: Path
) -> tuple[Path, dict[str, str], tuple[int, int, int, int, int]]:
    _safe_run_id(source.name)
    try:
        source_metadata = source.lstat()
    except OSError as exc:
        raise FileNotFoundError("Explainability directory is unavailable") from exc
    if not stat.S_ISDIR(source_metadata.st_mode):
        raise ValueError("Explainability pack must be a real directory")
    root_identity = _stat_identity(source_metadata)
    source_files = _regular_files(source)
    if not source_files or len(source_files) > _MAX_ZIP_FILES:
        raise ValueError("Explainability directory has an invalid file count")
    snapshot = snapshot_parent / source.name
    snapshot.mkdir(mode=0o700)
    content_sha256: dict[str, str] = {}
    total_size = 0
    for relative, path in source_files.items():
        payload, _ = _read_regular_bytes(
            path,
            field=f"Explainability artifact {relative}",
            max_bytes=_MAX_ARTIFACT_FILE_BYTES,
        )
        total_size += len(payload)
        if total_size > _MAX_ZIP_BYTES:
            raise ValueError("Explainability directory exceeds its total size limit")
        destination = snapshot.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(payload)
        content_sha256[relative] = hashlib.sha256(payload).hexdigest()
    return snapshot, content_sha256, root_identity


def _assert_directory_unchanged(
    source: Path,
    *,
    root_identity: tuple[int, int, int, int, int],
    content_sha256: Mapping[str, str],
) -> None:
    try:
        root_metadata = source.lstat()
    except OSError as exc:
        raise RuntimeError("Explainability directory changed during verification") from exc
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or _stat_identity(root_metadata) != root_identity
    ):
        raise RuntimeError("Explainability directory changed during verification")
    current_files = _regular_files(source)
    if set(current_files) != set(content_sha256):
        raise RuntimeError("Explainability directory changed during verification")
    for relative, path in current_files.items():
        payload, _ = _read_regular_bytes(
            path,
            field=f"Explainability artifact {relative}",
            max_bytes=_MAX_ARTIFACT_FILE_BYTES,
        )
        if hashlib.sha256(payload).hexdigest() != content_sha256[relative]:
            raise RuntimeError("Explainability directory changed during verification")
    try:
        final_root_metadata = source.lstat()
    except OSError as exc:
        raise RuntimeError("Explainability directory changed during verification") from exc
    if _stat_identity(final_root_metadata) != root_identity:
        raise RuntimeError("Explainability directory changed during verification")


def verify_explainability_pack(
    pack: str | Path,
    *,
    checksum: str | Path | None = None,
) -> dict[str, Any]:
    """Verify an explainability directory or deterministic ZIP entirely offline."""

    candidate = reject_symlink_components(
        Path(pack).expanduser(), label="Explainability pack"
    )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError("Explainability pack does not exist") from exc

    checksum_path: Path | None = None
    zip_sha256: str | None = None
    if resolved.is_dir():
        with tempfile.TemporaryDirectory(
            prefix="paisa-explainability-snapshot-"
        ) as temp:
            snapshot_parent = Path(temp).resolve(strict=True)
            snapshot, directory_content, root_identity = _snapshot_pack_directory(
                resolved, snapshot_parent
            )
            result = _verify_pack_directory(snapshot)
            if checksum is not None:
                sibling_zip = resolved.parent / f"{resolved.name}.zip"
                checksum_path = reject_symlink_components(
                    Path(checksum).expanduser(),
                    label="Explainability checksum sidecar",
                ).resolve(strict=True)
                zip_sha256 = _read_checksum_sidecar(
                    checksum_path, zip_path=sibling_zip
                )
                zip_result, zip_content = _verify_zip_archive(
                    sibling_zip, expected_sha256=zip_sha256
                )
                if zip_result["run_id"] != result["run_id"]:
                    raise ValueError("Directory and ZIP explainability run IDs disagree")
                if zip_content != directory_content:
                    raise ValueError(
                        "Explainability directory contents do not match the checksummed ZIP"
                    )
            _assert_directory_unchanged(
                resolved,
                root_identity=root_identity,
                content_sha256=directory_content,
            )
    elif resolved.is_file() and resolved.suffix == ".zip":
        requested_sidecar = Path(checksum).expanduser() if checksum is not None else None
        automatic_sidecar = resolved.with_name(f"{resolved.name}.sha256")
        sidecar_candidate = requested_sidecar or automatic_sidecar
        if requested_sidecar is None and not automatic_sidecar.exists():
            raise FileNotFoundError(
                "Explainability ZIP verification requires its SHA-256 sidecar"
            )
        checksum_path = reject_symlink_components(
            sidecar_candidate, label="Explainability checksum sidecar"
        ).resolve(strict=True)
        zip_sha256 = _read_checksum_sidecar(checksum_path, zip_path=resolved)
        result, _ = _verify_zip_archive(
            resolved, expected_sha256=zip_sha256
        )
    else:
        raise ValueError("Pack must be an explainability directory or .zip file")

    return {
        **result,
        "pack": resolved.name,
        "path_base": "input_parent",
        "checksum_verified": checksum_path is not None,
        "zip_sha256": zip_sha256,
        "verification_scope": "portable_evidence",
    }


__all__ = [
    "ACADEMIC_DISCLAIMER",
    "ADDITIVITY_TOLERANCES",
    "ARTIFACT_TYPE",
    "EXPECTED_MODELS",
    "HOLDOUT_TERM",
    "MODEL_ORDER",
    "ROW_COUNT",
    "SHAP_PIN",
    "build_tree_shap_explainability",
    "compute_cross_model_rankings",
    "compute_global_importance",
    "make_explainability_run_id",
    "normalize_expected_value",
    "normalize_positive_class_shap",
    "verify_explainability_pack",
]
