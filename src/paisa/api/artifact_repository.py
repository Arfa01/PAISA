"""Secure, read-only access to immutable PAISA result artifacts.

The repository validates both the vertical-slice checksum inventory and its
referenced model-training integrity root.  It never deserializes fitted model
artifacts and never accepts a client-selected filename.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from paisa.experiments.artifacts import (
    TrainingRunEvidence,
    validate_training_run_artifacts,
)


SCHEMA_VERSION = "1.0"
ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
ACTUAL_CLASS_LIMITATION = (
    "actual_class is evaluation-only and is never used for signals, execution, "
    "or portfolio accounting."
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,199}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MONEY = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{2}$")
_MODEL_VARIANTS = ("M-001", "M-002", "M-003", "M-004")
_TRADE_COLUMNS = (
    "schema_version",
    "trade_id",
    "portfolio_id",
    "signal_date",
    "execution_date",
    "action",
    "quantity",
    "reference_price",
    "execution_price",
    "fee",
    "cash_after_trade",
    "shares_after_trade",
)
_SNAPSHOT_COLUMNS = (
    "schema_version",
    "portfolio_id",
    "date",
    "cash",
    "shares",
    "market_price",
    "position_value",
    "total_equity",
    "realized_profit_loss",
    "unrealized_profit_loss",
)


class ArtifactRepositoryError(RuntimeError):
    """Base class for safe API-facing repository failures."""


class InvalidIdentifierError(ArtifactRepositoryError):
    pass


class RunNotFoundError(ArtifactRepositoryError):
    pass


class ExperimentNotFoundError(ArtifactRepositoryError):
    pass


class RunNotCompletedError(ArtifactRepositoryError):
    pass


class ArtifactIntegrityError(ArtifactRepositoryError):
    pass


class DefaultRunNotConfiguredError(ArtifactRepositoryError):
    pass


@dataclass(frozen=True, slots=True)
class RepositorySettings:
    repository_root: Path
    vertical_slice_root: Path
    model_training_root: Path
    default_vertical_slice_run_id: str | None


@dataclass(frozen=True, slots=True)
class ValidatedRun:
    run_id: str
    directory: Path
    manifest: dict[str, Any]
    summary: dict[str, Any]
    training_reference: dict[str, Any]
    training_evidence: TrainingRunEvidence
    artifact_paths: frozenset[str]
    vertical_signature: tuple[tuple[Any, ...], ...]
    training_signature: tuple[tuple[Any, ...], ...]


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _configured_root(value: str | None, *, default: str, repository_root: Path) -> Path:
    raw = Path(value or default).expanduser()
    return (raw if raw.is_absolute() else repository_root / raw).resolve()


def settings_from_environment() -> RepositorySettings:
    repository_root = _repository_root()
    default_run = os.getenv("PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID")
    if default_run is not None:
        default_run = default_run.strip() or None
    return RepositorySettings(
        repository_root=repository_root,
        vertical_slice_root=_configured_root(
            os.getenv("PAISA_VERTICAL_SLICE_ROOT"),
            default="artifacts/vertical_slice",
            repository_root=repository_root,
        ),
        model_training_root=_configured_root(
            os.getenv("PAISA_MODEL_TRAINING_ROOT"),
            default="artifacts/model_training",
            repository_root=repository_root,
        ),
        default_vertical_slice_run_id=default_run,
    )


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number is prohibited: {value}")


def _assert_finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite JSON numbers are prohibited")
    if isinstance(value, Mapping):
        for nested in value.values():
            _assert_finite_json(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_finite_json(nested)


def _open_regular_bytes(path: Path, *, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError) as exc:
        raise ValueError(f"{label} is missing or is not a regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise ValueError(f"{label} changed while it was being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = _open_regular_bytes(path, label=label)
    try:
        value = json.loads(payload, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _assert_finite_json(value)
    return value


def _sha256(path: Path, *, label: str) -> str:
    return hashlib.sha256(_open_regular_bytes(path, label=label)).hexdigest()


def _validate_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise InvalidIdentifierError(f"{field} must be a string")
    identifier = value.strip()
    if value != identifier:
        raise InvalidIdentifierError(f"{field} must not contain surrounding whitespace")
    if not _SAFE_ID.fullmatch(identifier):
        raise InvalidIdentifierError(
            f"{field} must start with a letter or digit and contain only letters, "
            "digits, '.', '_', and '-'"
        )
    return identifier


def _safe_relative_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    raw = value
    if not raw or "\\" in raw:
        raise ValueError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(raw)
    if path.is_absolute() or raw != path.as_posix():
        raise ValueError(f"{field} must be a normalized relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} contains an unsafe path component")
    return path.as_posix()


def _tree_signature(root: Path) -> tuple[tuple[Any, ...], ...]:
    if root.is_symlink():
        raise ValueError("Artifact directory must not be a symlink")
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("Artifact directory is missing") from exc
    if not resolved.is_dir():
        raise ValueError("Artifact root must be a directory")
    records: list[tuple[Any, ...]] = []
    for path in sorted(resolved.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(resolved).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Artifact tree contains a symbolic link: {relative}")
        if not stat.S_ISDIR(metadata.st_mode) and not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Artifact tree contains a special file: {relative}")
        records.append(
            (
                relative,
                stat.S_IFMT(metadata.st_mode),
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            )
        )
    return tuple(records)


def _regular_payload_files(root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"Artifact tree contains a symbolic link: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Artifact tree contains a special file: {relative}")
        if relative != "manifest.json":
            files[relative] = path
    return files


def _parse_vertical_inventory(manifest: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Vertical-slice manifest artifacts must be a non-empty array")
    parsed: list[tuple[str, str]] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"Invalid vertical artifact inventory entry {index}")
        relative = _safe_relative_path(entry["path"], field=f"artifacts[{index}].path")
        if relative == "manifest.json":
            raise ValueError("manifest.json cannot checksum itself")
        digest = str(entry["sha256"])
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"Invalid SHA-256 for vertical artifact {relative}")
        parsed.append((relative, digest))
    if parsed != sorted(parsed, key=lambda item: item[0]):
        raise ValueError("Vertical artifact inventory must be sorted by path")
    if len({path for path, _digest in parsed}) != len(parsed):
        raise ValueError("Vertical artifact inventory contains duplicate paths")
    return tuple(parsed)


def _require_mapping(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return dict(value)


def _require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _optional_string(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, field=field)


def _schema_version(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    version = _require_string(value, field=field)
    if version != SCHEMA_VERSION:
        raise ValueError(f"{field} must equal {SCHEMA_VERSION}")
    return version


def _sha256_string(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    digest = _require_string(value, field=field)
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _boolean(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _iso_date(value: Any, *, field: str) -> str:
    text = _require_string(value, field=field)
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc
    return parsed.isoformat()


def _iso_timestamp(value: Any, *, field: str) -> str:
    text = _require_string(value, field=field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if "T" not in text or parsed.tzinfo is None:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    return text


def _optional_iso_datetime(value: Any, *, field: str) -> str | None:
    """Validate optional producer timestamps, including legacy naive ISO values."""

    if value is None:
        return None
    text = _require_string(value, field=field)
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO timestamp") from exc
    if "T" not in text:
        raise ValueError(f"{field} must be an ISO timestamp")
    return text


def _producer_metadata(value: Any, *, field: str) -> dict[str, str] | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return _require_string(value, field=field)
    producer = _require_mapping(value, field=field)
    normalized: dict[str, str] = {}
    if "component" in producer:
        normalized["component"] = _require_string(
            producer["component"], field=f"{field}.component"
        )
    if "schema_version" in producer:
        normalized["schema_version"] = str(
            _schema_version(
                producer["schema_version"], field=f"{field}.schema_version"
            )
        )
    if "status" in producer:
        normalized["status"] = _require_string(
            producer["status"], field=f"{field}.status"
        )
    return normalized


def _finite_float(value: Any, *, field: str, nullable: bool = False) -> float | None:
    if value is None or str(value).strip().lower() in {"", "nan", "none", "null"}:
        if nullable:
            return None
        raise ValueError(f"{field} is required")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _integer(value: Any, *, field: str) -> int:
    text = str(value).strip()
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer") from exc
    if str(parsed) != text and text not in {f"+{parsed}", f"-{abs(parsed)}"}:
        raise ValueError(f"{field} must be an integer")
    return parsed


def _money(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        if nullable:
            return None
        raise ValueError(f"{field} is required")
    text = str(value).strip()
    if not _MONEY.fullmatch(text):
        raise ValueError(f"{field} must be a fixed-scale PKR value")
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field} must be finite")
    return format(parsed, ".2f")


def _paginate(rows: list[dict[str, Any]], *, limit: int, offset: int) -> dict[str, Any]:
    return {
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "rows": rows[offset : offset + limit],
    }


class ArtifactRepository:
    """Validate and project immutable result artifacts into API-safe values."""

    def __init__(self, settings: RepositorySettings):
        self.settings = settings
        self._cache: dict[str, ValidatedRun] = {}
        self._lock = threading.RLock()

    @property
    def default_run_id(self) -> str:
        value = self.settings.default_vertical_slice_run_id
        if value is None:
            raise DefaultRunNotConfiguredError(
                "PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID is not configured"
            )
        return _validate_identifier(value, field="default run ID")

    def _run_directory(self, run_id: str) -> Path:
        safe_id = _validate_identifier(run_id, field="run_id")
        root = self.settings.vertical_slice_root
        candidate = root / safe_id
        if candidate.is_symlink():
            raise ArtifactIntegrityError("Vertical-slice run directory is a symbolic link")
        if not candidate.exists():
            raise RunNotFoundError(f"Vertical-slice run '{safe_id}' was not found")
        try:
            resolved_root = root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except FileNotFoundError as exc:
            raise RunNotFoundError(f"Vertical-slice run '{safe_id}' was not found") from exc
        except ValueError as exc:
            raise ArtifactIntegrityError("Vertical-slice run escapes its configured root") from exc
        if not resolved.is_dir():
            raise RunNotFoundError(f"Vertical-slice run '{safe_id}' was not found")
        return resolved

    def _training_directory(self, training_run_id: str) -> Path:
        safe_id = _validate_identifier(training_run_id, field="training_run_id")
        root = self.settings.model_training_root
        candidate = root / safe_id
        if candidate.is_symlink():
            raise ArtifactIntegrityError("Training-run directory is a symbolic link")
        try:
            resolved_root = root.resolve(strict=True)
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(resolved_root)
        except (FileNotFoundError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"Referenced training run '{safe_id}' is unavailable or outside its root"
            ) from exc
        if not resolved.is_dir():
            raise ArtifactIntegrityError(f"Referenced training run '{safe_id}' is unavailable")
        return resolved

    def _cached_run(self, run_id: str, directory: Path) -> ValidatedRun | None:
        cached = self._cache.get(run_id)
        if cached is None:
            return None
        try:
            vertical_signature = _tree_signature(directory)
            training_signature = _tree_signature(cached.training_evidence.run_directory)
        except ValueError:
            return None
        if (
            vertical_signature == cached.vertical_signature
            and training_signature == cached.training_signature
        ):
            return cached
        return None

    def _validate_training_reference(
        self,
        *,
        manifest: Mapping[str, Any],
        reference: Mapping[str, Any],
    ) -> tuple[TrainingRunEvidence, tuple[tuple[Any, ...], ...]]:
        _schema_version(
            reference.get("schema_version"), field="training reference schema_version"
        )
        training_run_id = _require_string(
            reference.get("training_run_id"), field="training_run_id"
        )
        training_invocation_id = _require_string(
            reference.get("training_invocation_id"), field="training_invocation_id"
        )
        if manifest.get("model_training_run_id") != training_run_id:
            raise ValueError("Vertical and training-reference run IDs disagree")
        training_directory = self._training_directory(training_run_id)
        signature_before = _tree_signature(training_directory)
        try:
            evidence = validate_training_run_artifacts(
                training_directory,
                expected_run_id=training_run_id,
                expected_invocation_id=training_invocation_id,
            )
        except Exception as exc:
            raise ValueError("Referenced training run failed integrity validation") from exc
        signature_after = _tree_signature(training_directory)
        if signature_before != signature_after:
            raise ValueError("Referenced training run changed during validation")

        if reference.get("run_manifest_sha256") != evidence.run_manifest_sha256:
            raise ValueError("Training run manifest digest does not match its reference")
        source = _require_mapping(
            reference.get("training_source_provenance"),
            field="training_source_provenance",
        )
        if source != {
            "code_version": evidence.code_version,
            "dirty_tree": evidence.dirty_tree,
            "source_fingerprint": evidence.source_fingerprint,
        }:
            raise ValueError("Training source provenance does not match the validated run")

        inventory = _require_mapping(
            reference.get("training_artifact_inventory"),
            field="training_artifact_inventory",
        )
        if (
            inventory.get("algorithm") != "sha256"
            or inventory.get("root") != "."
            or inventory.get("excludes") != ["run_manifest.json"]
        ):
            raise ValueError("Training reference inventory metadata is invalid")
        files = inventory.get("files")
        if not isinstance(files, list):
            raise ValueError("Training reference inventory files must be an array")
        reference_inventory = tuple(
            (
                _safe_relative_path(item.get("path"), field="training inventory path"),
                str(item.get("sha256")),
            )
            for item in files
            if isinstance(item, Mapping)
        )
        if len(reference_inventory) != len(files) or reference_inventory != evidence.artifact_sha256:
            raise ValueError("Training reference inventory does not match the validated run")

        model_ids = _require_mapping(reference.get("model_ids"), field="model_ids")
        if dict(evidence.model_ids) != {str(key): str(value) for key, value in model_ids.items()}:
            raise ValueError("Training model identities do not match the validated run")
        registered = _require_mapping(
            manifest.get("registered_model_ids"), field="registered_model_ids"
        )
        if {str(key): str(value) for key, value in registered.items()} != dict(
            evidence.model_ids
        ):
            raise ValueError("Vertical model identities do not match the training run")

        prediction_sha = _require_mapping(
            reference.get("prediction_sha256"), field="prediction_sha256"
        )
        if set(prediction_sha) != {variant for variant, _path in evidence.prediction_paths}:
            raise ValueError("Training prediction checksum keys are incomplete")
        inventoried = dict(evidence.artifact_sha256)
        for variant, relative in evidence.prediction_paths:
            if prediction_sha.get(variant) != inventoried.get(relative):
                raise ValueError("Training prediction checksum does not match its inventory")
        raw_prediction_paths = _require_mapping(
            reference.get("prediction_paths"), field="prediction_paths"
        )
        if set(raw_prediction_paths) != set(prediction_sha):
            raise ValueError("Training prediction path keys are incomplete")
        return evidence, signature_after

    def _validate_run(self, run_id: str) -> ValidatedRun:
        safe_id = _validate_identifier(run_id, field="run_id")
        directory = self._run_directory(safe_id)
        with self._lock:
            cached = self._cached_run(safe_id, directory)
            if cached is not None:
                return cached
            try:
                signature_before = _tree_signature(directory)
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    raise RunNotCompletedError(
                        f"Vertical-slice run '{safe_id}' has no completed manifest"
                    )
                manifest = _read_json(manifest_path, label="vertical-slice manifest")
                if manifest.get("schema_version") != SCHEMA_VERSION:
                    raise ValueError("Unsupported vertical-slice schema_version")
                if manifest.get("vertical_slice_run_id") != safe_id:
                    raise ValueError("Vertical-slice manifest identity mismatch")
                if manifest.get("status") != "PASS":
                    raise RunNotCompletedError(
                        f"Vertical-slice run '{safe_id}' is not a completed PASS"
                    )
                variants = manifest.get("model_variants")
                if variants != list(_MODEL_VARIANTS):
                    raise ValueError(
                        "Vertical-slice run must declare the four ordered model variants"
                    )
                registered_models = _require_mapping(
                    manifest.get("registered_model_ids"),
                    field="registered_model_ids",
                )
                if set(registered_models) != set(_MODEL_VARIANTS):
                    raise ValueError("Registered model identities are incomplete")
                validated_model_ids = [
                    _validate_identifier(registered_models[variant], field="model_id")
                    for variant in _MODEL_VARIANTS
                ]
                if len(set(validated_model_ids)) != len(_MODEL_VARIANTS):
                    raise ValueError("Registered model identities must be unique")
                declared_portfolios = _require_mapping(
                    manifest.get("portfolio_ids"), field="portfolio_ids"
                )
                if set(declared_portfolios) != set(_MODEL_VARIANTS):
                    raise ValueError("Portfolio identities are incomplete")

                inventory = _parse_vertical_inventory(manifest)
                expected = dict(inventory)
                actual = _regular_payload_files(directory)
                if set(actual) != set(expected):
                    raise ValueError("Vertical-slice artifact set does not match its manifest")
                for relative, expected_digest in inventory:
                    if _sha256(actual[relative], label=f"artifact {relative}") != expected_digest:
                        raise ValueError(f"Vertical-slice checksum mismatch: {relative}")

                required = {
                    "final_summary.json",
                    "model_training_run_reference.json",
                    "model_metrics.csv",
                    "portfolio_metrics.csv",
                    "combined_leaderboard.csv",
                    "daily_equity_curves.csv",
                    *(f"experiment_configs/{variant}.json" for variant in _MODEL_VARIANTS),
                }
                if not required.issubset(expected):
                    raise ValueError("Vertical-slice run is missing required result artifacts")

                experiment_ids: list[str] = []
                expected_experiment_payloads = {
                    "config.json",
                    "predictions.csv",
                    "signals.csv",
                    "trades.csv",
                    "portfolio_snapshots.csv",
                    "metrics.json",
                    "result.json",
                    "equity_curve.png",
                    "drawdown.png",
                    "benchmarks/cash_only_snapshots.csv",
                    "benchmarks/buy_and_hold_snapshots.csv",
                    "benchmarks/buy_and_hold_trades.csv",
                }
                for variant in _MODEL_VARIANTS:
                    root_config = _read_json(
                        actual[f"experiment_configs/{variant}.json"],
                        label=f"normalized experiment config {variant}",
                    )
                    config = self._configuration(root_config)
                    experiment_id = _validate_identifier(
                        config["experiment_id"], field="experiment_id"
                    )
                    experiment_ids.append(experiment_id)
                    if config["model_id"] != str(registered_models[variant]):
                        raise ValueError("Experiment model identity mismatch")
                    prefix = f"experiments/{experiment_id}/"
                    observed_payloads = {
                        relative.removeprefix(prefix)
                        for relative in expected
                        if relative.startswith(prefix)
                    }
                    if observed_payloads != expected_experiment_payloads:
                        raise ValueError("Experiment artifact set is incomplete")
                    experiment_config = _read_json(
                        actual[f"{prefix}config.json"], label="experiment config"
                    )
                    result = _read_json(
                        actual[f"{prefix}result.json"], label="experiment result"
                    )
                    metrics = _read_json(
                        actual[f"{prefix}metrics.json"], label="experiment metrics"
                    )
                    if experiment_config != root_config or metrics != result:
                        raise ValueError("Persisted experiment evidence disagrees")
                    if (
                        result.get("experiment_id") != experiment_id
                        or result.get("model_id") != config["model_id"]
                        or result.get("portfolio_id") != config["portfolio_id"]
                    ):
                        raise ValueError("Experiment result identities disagree")
                    declared = _require_mapping(
                        declared_portfolios.get(variant),
                        field=f"portfolio_ids.{variant}",
                    )
                    if declared != {
                        "strategy": config["portfolio_id"],
                        "cash_only": config["benchmark_portfolio_ids"]["cash_only"],
                        "buy_and_hold": config["benchmark_portfolio_ids"]["buy_and_hold"],
                    }:
                        raise ValueError("Experiment portfolio identities disagree")
                if len(set(experiment_ids)) != len(_MODEL_VARIANTS):
                    raise ValueError("Experiment identities must be unique")
                summary = _read_json(
                    directory / "final_summary.json", label="final summary"
                )
                reference = _read_json(
                    directory / "model_training_run_reference.json",
                    label="model-training reference",
                )
                if (
                    summary.get("schema_version") != SCHEMA_VERSION
                    or summary.get("vertical_slice_run_id") != safe_id
                    or summary.get("status") != "PASS"
                ):
                    raise ValueError("Final summary is not a completed matching run")
                disclaimer = _require_string(
                    manifest.get("disclaimer"), field="manifest disclaimer"
                )
                if disclaimer != ACADEMIC_DISCLAIMER or disclaimer != _require_string(
                    summary.get("disclaimer"), field="summary disclaimer"
                ):
                    raise ValueError("Academic-simulation disclaimer is missing or inconsistent")

                training_evidence, training_signature = self._validate_training_reference(
                    manifest=manifest, reference=reference
                )
                signature_after = _tree_signature(directory)
                if signature_before != signature_after:
                    raise ValueError("Vertical-slice run changed during validation")
                validated = ValidatedRun(
                    run_id=safe_id,
                    directory=directory,
                    manifest=manifest,
                    summary=summary,
                    training_reference=reference,
                    training_evidence=training_evidence,
                    artifact_paths=frozenset(expected),
                    vertical_signature=signature_after,
                    training_signature=training_signature,
                )
                self._cache[safe_id] = validated
                return validated
            except (RunNotCompletedError, RunNotFoundError):
                raise
            except InvalidIdentifierError as exc:
                raise ArtifactIntegrityError(
                    f"Vertical-slice run '{safe_id}' contains an unsafe artifact identity"
                ) from exc
            except ArtifactRepositoryError:
                raise
            except Exception as exc:
                raise ArtifactIntegrityError(
                    f"Vertical-slice run '{safe_id}' failed integrity validation"
                ) from exc

    def _artifact_path(self, run: ValidatedRun, relative: str) -> Path:
        safe_relative = _safe_relative_path(relative, field="artifact path")
        if safe_relative not in run.artifact_paths:
            raise ArtifactIntegrityError("Requested repository artifact is not inventoried")
        candidate = run.directory.joinpath(*PurePosixPath(safe_relative).parts)
        if candidate.is_symlink():
            raise ArtifactIntegrityError("Repository artifact is a symbolic link")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(run.directory)
        except (FileNotFoundError, ValueError) as exc:
            raise ArtifactIntegrityError("Repository artifact escapes its run root") from exc
        if not resolved.is_file():
            raise ArtifactIntegrityError("Repository artifact is not a regular file")
        return resolved

    def _json_artifact(self, run: ValidatedRun, relative: str) -> dict[str, Any]:
        try:
            return _read_json(
                self._artifact_path(run, relative), label=f"artifact {relative}"
            )
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Inventoried artifact '{relative}' could not be read safely"
            ) from exc

    def _csv_artifact(
        self,
        run: ValidatedRun,
        relative: str,
        *,
        required_columns: Iterable[str] = (),
    ) -> list[dict[str, str]]:
        try:
            payload = _open_regular_bytes(
                self._artifact_path(run, relative), label=f"artifact {relative}"
            )
            text = payload.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError("CSV header is missing or contains duplicates")
            missing = set(required_columns) - set(reader.fieldnames)
            if missing:
                raise ValueError(f"CSV header is missing required columns: {sorted(missing)}")
            rows = [dict(row) for row in reader]
            if any(None in row for row in rows):
                raise ValueError("CSV row has more values than its header")
            return rows
        except Exception as exc:
            raise ArtifactIntegrityError(
                f"Inventoried artifact '{relative}' could not be parsed safely"
            ) from exc

    def _context(self, run: ValidatedRun) -> dict[str, Any]:
        data = _require_mapping(run.manifest.get("data"), field="manifest data")
        raw_range = _require_mapping(
            data.get("actual_ogdc_date_range"), field="actual_ogdc_date_range"
        )
        summary_warnings = run.summary.get("warnings", [])
        if not isinstance(summary_warnings, list):
            raise ArtifactIntegrityError("Final-summary warnings must be an array")
        warnings = [_require_string(item, field="warning") for item in summary_warnings]
        limitations = [ACTUAL_CLASS_LIMITATION]
        migration_path = "input_snapshot/migration_manifest.json"
        if migration_path in run.artifact_paths:
            migration = self._json_artifact(run, migration_path)
            raw_limitations = migration.get("limitations", [])
            if isinstance(raw_limitations, list):
                limitations.extend(
                    _require_string(item, field="migration limitation")
                    for item in raw_limitations
                )
        limitations = list(dict.fromkeys(limitations))
        return {
            "schema_version": str(run.manifest["schema_version"]),
            "run_id": run.run_id,
            "source_mode": _require_string(
                data.get("acquisition_mode", data.get("mode")), field="source mode"
            ),
            "data_date_range": {
                "start": _iso_date(raw_range.get("start"), field="data date start"),
                "end": _iso_date(raw_range.get("end"), field="data date end"),
            },
            "warnings": warnings,
            "limitations": limitations,
            "disclaimer": _require_string(
                run.manifest.get("disclaimer"), field="disclaimer"
            ),
        }

    def get_context(self, run_id: str) -> dict[str, Any]:
        return deepcopy(self._context(self._validate_run(run_id)))

    def list_completed_runs(self) -> list[dict[str, Any]]:
        root = self.settings.vertical_slice_root
        if not root.exists():
            return []
        if root.is_symlink() or not root.is_dir():
            raise ArtifactIntegrityError("Configured vertical-slice root is not a directory")
        rows: list[dict[str, Any]] = []
        for candidate in sorted(root.iterdir(), key=lambda item: item.name):
            if not candidate.is_dir() or candidate.is_symlink():
                continue
            try:
                run = self._validate_run(candidate.name)
                context = self._context(run)
                experiments = self.list_experiments(run.run_id)
            except (ArtifactRepositoryError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "schema_version": context["schema_version"],
                    "run_id": run.run_id,
                    "status": "PASS",
                    "symbol": _require_string(run.manifest.get("symbol"), field="symbol"),
                    "completed_at": _iso_timestamp(
                        run.manifest.get("created_at"), field="created_at"
                    ),
                    "source_mode": context["source_mode"],
                    "data_date_range": context["data_date_range"],
                    "model_training_run_id": _require_string(
                        run.manifest.get("model_training_run_id"),
                        field="model_training_run_id",
                    ),
                    "model_count": len(run.manifest.get("model_variants", [])),
                    "experiment_count": len(experiments),
                }
            )
        return rows

    def load_run_detail(self, run_id: str) -> dict[str, Any]:
        run = self._validate_run(run_id)
        context = self._context(run)
        evaluation = _require_mapping(
            run.manifest.get("common_test_dates"), field="common_test_dates"
        )
        assumptions = _require_mapping(
            run.manifest.get("shared_experiment_assumptions"),
            field="shared_experiment_assumptions",
        )
        experiments = self.list_experiments(run.run_id)
        return {
            **context,
            "status": "PASS",
            "symbol": _require_string(run.manifest.get("symbol"), field="symbol"),
            "created_at": _iso_timestamp(
                run.manifest.get("created_at"), field="created_at"
            ),
            "evaluation_dates": self._evaluation_dates(evaluation),
            "model_training_run_id": _require_string(
                run.manifest.get("model_training_run_id"), field="model_training_run_id"
            ),
            "model_variants": [str(item) for item in run.manifest.get("model_variants", [])],
            "registered_model_ids": {
                str(key): str(value)
                for key, value in _require_mapping(
                    run.manifest.get("registered_model_ids"),
                    field="registered_model_ids",
                ).items()
            },
            "experiment_ids": [row["experiment_id"] for row in experiments],
            "shared_experiment_assumptions": {
                "initial_cash_pkr": _money(
                    assumptions.get("initial_cash_pkr"), field="initial_cash_pkr"
                ),
                "buy_probability_up_gte": _finite_float(
                    assumptions.get("buy_probability_up_gte"),
                    field="buy_probability_up_gte",
                ),
                "sell_probability_up_lte": _finite_float(
                    assumptions.get("sell_probability_up_lte"),
                    field="sell_probability_up_lte",
                ),
                "maximum_allocation": _finite_float(
                    assumptions.get("maximum_allocation"), field="maximum_allocation"
                ),
                "long_only": _boolean(
                    assumptions.get("long_only"), field="long_only"
                ),
                "whole_shares": _boolean(
                    assumptions.get("whole_shares"), field="whole_shares"
                ),
                "transaction_cost_bps": _finite_float(
                    assumptions.get("transaction_cost_bps"), field="transaction_cost_bps"
                ),
                "slippage_bps": _finite_float(
                    assumptions.get("slippage_bps"), field="slippage_bps"
                ),
                "signal_timing": _require_string(
                    assumptions.get("signal_timing"), field="signal_timing"
                ),
                "execution": _require_string(
                    assumptions.get("execution"), field="execution"
                ),
                "valuation": _require_string(
                    assumptions.get("valuation"), field="valuation"
                ),
                "ending_position": _require_string(
                    assumptions.get("ending_position"), field="ending_position"
                ),
            },
        }

    @staticmethod
    def _evaluation_dates(value: Mapping[str, Any]) -> dict[str, str]:
        return {
            key: _iso_date(value.get(key), field=key)
            for key in ("feature_start", "feature_end", "target_start", "target_end")
        }

    def load_summary(self, run_id: str) -> dict[str, Any]:
        run = self._validate_run(run_id)
        value = run.summary
        best_validation = _require_mapping(
            value.get("best_validation_prediction_model"),
            field="best_validation_prediction_model",
        )
        best_portfolio = _require_mapping(
            value.get("best_test_period_simulated_portfolio"),
            field="best_test_period_simulated_portfolio",
        )
        lowest_drawdown = _require_mapping(
            value.get("lowest_drawdown_portfolio"), field="lowest_drawdown_portfolio"
        )
        buy_hold = _require_mapping(
            value.get("buy_and_hold_result"), field="buy_and_hold_result"
        )
        training_data = _require_mapping(
            value.get("training_data"), field="training_data"
        )
        evaluation = _require_mapping(
            value.get("common_test_dates"), field="common_test_dates"
        )
        return {
            **self._context(run),
            "status": "PASS",
            "symbol": _require_string(value.get("symbol"), field="symbol"),
            "completed_at": _iso_timestamp(
                value.get("completed_at"), field="completed_at"
            ),
            "model_training_run_id": _require_string(
                value.get("model_training_run_id"), field="model_training_run_id"
            ),
            "evaluation_dates": self._evaluation_dates(evaluation),
            "best_validation_prediction_model": {
                "model_id": _require_string(
                    best_validation.get("model_id"), field="best model_id"
                ),
                "registered_model_id": _require_string(
                    best_validation.get("registered_model_id"),
                    field="best registered_model_id",
                ),
                "algorithm": _require_string(
                    best_validation.get("algorithm"), field="best algorithm"
                ),
                "validation_balanced_accuracy": _finite_float(
                    best_validation.get("validation_balanced_accuracy"),
                    field="validation_balanced_accuracy",
                    nullable=True,
                ),
                "validation_f1": _finite_float(
                    best_validation.get("validation_f1"), field="validation_f1"
                ),
                "selection_basis": _require_string(
                    best_validation.get("selection_basis"), field="selection_basis"
                ),
            },
            "best_test_period_simulated_portfolio": {
                "model_id": _require_string(
                    best_portfolio.get("model_id"), field="best portfolio model_id"
                ),
                "portfolio_id": _require_string(
                    best_portfolio.get("portfolio_id"), field="best portfolio_id"
                ),
                "ending_portfolio_value": _money(
                    best_portfolio.get("ending_portfolio_value"),
                    field="ending_portfolio_value",
                ),
                "total_return": _finite_float(
                    best_portfolio.get("total_return"), field="total_return"
                ),
                "sharpe_ratio": _finite_float(
                    best_portfolio.get("sharpe_ratio"),
                    field="sharpe_ratio",
                    nullable=True,
                ),
                "selection_basis": _require_string(
                    best_portfolio.get("selection_basis"), field="selection_basis"
                ),
            },
            "lowest_drawdown_portfolio": {
                "model_id": _require_string(
                    lowest_drawdown.get("model_id"), field="lowest drawdown model_id"
                ),
                "portfolio_id": _require_string(
                    lowest_drawdown.get("portfolio_id"),
                    field="lowest drawdown portfolio_id",
                ),
                "maximum_drawdown": _finite_float(
                    lowest_drawdown.get("maximum_drawdown"), field="maximum_drawdown"
                ),
                "total_return": _finite_float(
                    lowest_drawdown.get("total_return"), field="total_return"
                ),
            },
            "buy_and_hold_result": {
                "portfolio_id": _require_string(
                    buy_hold.get("portfolio_id"), field="buy-and-hold portfolio_id"
                ),
                "ending_portfolio_value": _money(
                    buy_hold.get("ending_portfolio_value"),
                    field="buy-and-hold ending value",
                ),
                "total_return": _finite_float(
                    buy_hold.get("total_return"), field="buy-and-hold total_return"
                ),
                "maximum_drawdown": _finite_float(
                    buy_hold.get("maximum_drawdown"),
                    field="buy-and-hold maximum_drawdown",
                ),
                "fees_paid": _money(
                    buy_hold.get("fees_paid"), field="buy-and-hold fees_paid"
                ),
            },
            "any_model_portfolio_beat_buy_and_hold": _boolean(
                value.get("any_model_portfolio_beat_buy_and_hold"),
                field="any_model_portfolio_beat_buy_and_hold",
            ),
            "selection_policy": _require_string(
                value.get("selection_policy"), field="selection_policy"
            ),
            "target_interpretation": _require_string(
                value.get("target_interpretation"), field="target_interpretation"
            ),
            "portfolio_return_interpretation": _require_string(
                value.get("portfolio_return_interpretation"),
                field="portfolio_return_interpretation",
            ),
            "training_data": {
                "provider": _require_string(
                    training_data.get("provider"), field="training provider"
                ),
                "actual_feature_start": _iso_date(
                    training_data.get("actual_feature_start"),
                    field="actual_feature_start",
                ),
                "actual_feature_end": _iso_date(
                    training_data.get("actual_feature_end"), field="actual_feature_end"
                ),
                "dataset_sha256": _require_string(
                    training_data.get("dataset_sha256"), field="dataset_sha256"
                ),
            },
        }

    @staticmethod
    def _require_columns(
        rows: list[dict[str, str]], required: Iterable[str], *, artifact: str
    ) -> None:
        if not rows:
            raise ArtifactIntegrityError(f"{artifact} must contain at least one row")
        missing = set(required) - set(rows[0])
        if missing:
            raise ArtifactIntegrityError(
                f"{artifact} is missing required columns: {sorted(missing)}"
            )

    def load_model_metrics(self, run_id: str) -> list[dict[str, Any]]:
        run = self._validate_run(run_id)
        source = self._csv_artifact(run, "model_metrics.csv")
        fields = (
            "model_id",
            "registered_model_id",
            "algorithm",
            "validation_accuracy",
            "validation_balanced_accuracy",
            "validation_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1",
            "train_count",
            "validation_count",
            "test_count",
            "train_class_0",
            "train_class_1",
            "validation_class_0",
            "validation_class_1",
            "test_class_0",
            "test_class_1",
        )
        self._require_columns(source, fields, artifact="model_metrics.csv")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            prefix = f"model_metrics[{index}]"
            rows.append(
                {
                    "model_id": _require_string(row["model_id"], field=f"{prefix}.model_id"),
                    "registered_model_id": _require_string(
                        row["registered_model_id"], field=f"{prefix}.registered_model_id"
                    ),
                    "algorithm": _require_string(
                        row["algorithm"], field=f"{prefix}.algorithm"
                    ),
                    "validation_accuracy": _finite_float(
                        row["validation_accuracy"], field=f"{prefix}.validation_accuracy"
                    ),
                    "validation_balanced_accuracy": _finite_float(
                        row["validation_balanced_accuracy"],
                        field=f"{prefix}.validation_balanced_accuracy",
                        nullable=True,
                    ),
                    "validation_f1": _finite_float(
                        row["validation_f1"], field=f"{prefix}.validation_f1"
                    ),
                    "test_accuracy": _finite_float(
                        row["test_accuracy"], field=f"{prefix}.test_accuracy"
                    ),
                    "test_balanced_accuracy": _finite_float(
                        row["test_balanced_accuracy"],
                        field=f"{prefix}.test_balanced_accuracy",
                        nullable=True,
                    ),
                    "test_f1": _finite_float(row["test_f1"], field=f"{prefix}.test_f1"),
                    **{
                        field: _integer(row[field], field=f"{prefix}.{field}")
                        for field in (
                            "train_count",
                            "validation_count",
                            "test_count",
                            "train_class_0",
                            "train_class_1",
                            "validation_class_0",
                            "validation_class_1",
                            "test_class_0",
                            "test_class_1",
                        )
                    },
                }
            )
        rows.sort(key=lambda item: item["model_id"])
        if len({row["model_id"] for row in rows}) != len(rows):
            raise ArtifactIntegrityError("model_metrics.csv contains duplicate model IDs")
        expected_models = {
            str(key): str(value)
            for key, value in _require_mapping(
                run.manifest.get("registered_model_ids"), field="registered_model_ids"
            ).items()
        }
        observed_models = {
            row["model_id"]: row["registered_model_id"] for row in rows
        }
        if observed_models != expected_models:
            raise ArtifactIntegrityError(
                "model_metrics.csv identities do not match the run manifest"
            )
        return rows

    def load_portfolio_metrics(self, run_id: str) -> list[dict[str, Any]]:
        run = self._validate_run(run_id)
        source = self._csv_artifact(run, "portfolio_metrics.csv")
        fields = (
            "portfolio_type",
            "model_id",
            "portfolio_id",
            "ending_portfolio_value",
            "total_return",
            "maximum_drawdown",
            "volatility",
            "sharpe_ratio",
            "number_of_trades",
            "fees_paid",
            "result_versus_buy_and_hold",
        )
        self._require_columns(source, fields, artifact="portfolio_metrics.csv")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            prefix = f"portfolio_metrics[{index}]"
            rows.append(
                {
                    "portfolio_type": _require_string(
                        row["portfolio_type"], field=f"{prefix}.portfolio_type"
                    ),
                    "model_id": _require_string(row["model_id"], field=f"{prefix}.model_id"),
                    "portfolio_id": _require_string(
                        row["portfolio_id"], field=f"{prefix}.portfolio_id"
                    ),
                    "ending_portfolio_value": _money(
                        row["ending_portfolio_value"],
                        field=f"{prefix}.ending_portfolio_value",
                    ),
                    "total_return": _finite_float(
                        row["total_return"], field=f"{prefix}.total_return"
                    ),
                    "maximum_drawdown": _finite_float(
                        row["maximum_drawdown"], field=f"{prefix}.maximum_drawdown"
                    ),
                    "volatility": _finite_float(
                        row["volatility"], field=f"{prefix}.volatility", nullable=True
                    ),
                    "sharpe_ratio": _finite_float(
                        row["sharpe_ratio"], field=f"{prefix}.sharpe_ratio", nullable=True
                    ),
                    "number_of_trades": _integer(
                        row["number_of_trades"], field=f"{prefix}.number_of_trades"
                    ),
                    "fees_paid": _money(row["fees_paid"], field=f"{prefix}.fees_paid"),
                    "result_versus_buy_and_hold": _finite_float(
                        row["result_versus_buy_and_hold"],
                        field=f"{prefix}.result_versus_buy_and_hold",
                    ),
                }
            )
        order = {"model": 0, "cash_only": 1, "buy_and_hold": 2}
        rows.sort(
            key=lambda item: (
                order.get(item["portfolio_type"], 99),
                item["model_id"],
                item["portfolio_id"],
            )
        )
        declared = _require_mapping(
            run.manifest.get("portfolio_ids"), field="portfolio_ids"
        )
        observed_models = {
            row["model_id"]: row["portfolio_id"]
            for row in rows
            if row["portfolio_type"] == "model"
        }
        expected_models = {
            str(variant): str(_require_mapping(ids, field="portfolio IDs")["strategy"])
            for variant, ids in declared.items()
        }
        cash_rows = [row for row in rows if row["portfolio_type"] == "cash_only"]
        buy_hold_rows = [
            row for row in rows if row["portfolio_type"] == "buy_and_hold"
        ]
        first_variant = sorted(declared)[0] if declared else ""
        first_ids = (
            _require_mapping(declared[first_variant], field="benchmark portfolio IDs")
            if first_variant
            else {}
        )
        if (
            observed_models != expected_models
            or len(cash_rows) != 1
            or len(buy_hold_rows) != 1
            or cash_rows[0]["portfolio_id"] != first_ids.get("cash_only")
            or buy_hold_rows[0]["portfolio_id"] != first_ids.get("buy_and_hold")
        ):
            raise ArtifactIntegrityError(
                "portfolio_metrics.csv identities do not match the run manifest"
            )
        return rows

    def load_leaderboard(self, run_id: str) -> list[dict[str, Any]]:
        run = self._validate_run(run_id)
        source = self._csv_artifact(run, "combined_leaderboard.csv")
        fields = (
            "model_id",
            "registered_model_id",
            "algorithm",
            "validation_balanced_accuracy",
            "validation_f1",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_f1",
            "portfolio_id",
            "ending_portfolio_value",
            "total_return",
            "maximum_drawdown",
            "volatility",
            "sharpe_ratio",
            "number_of_trades",
            "fees_paid",
            "result_versus_buy_and_hold",
        )
        self._require_columns(source, fields, artifact="combined_leaderboard.csv")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            prefix = f"combined_leaderboard[{index}]"
            rows.append(
                {
                    "model_id": _require_string(row["model_id"], field=f"{prefix}.model_id"),
                    "registered_model_id": _require_string(
                        row["registered_model_id"], field=f"{prefix}.registered_model_id"
                    ),
                    "algorithm": _require_string(
                        row["algorithm"], field=f"{prefix}.algorithm"
                    ),
                    "validation_balanced_accuracy": _finite_float(
                        row["validation_balanced_accuracy"],
                        field=f"{prefix}.validation_balanced_accuracy",
                        nullable=True,
                    ),
                    "validation_f1": _finite_float(
                        row["validation_f1"], field=f"{prefix}.validation_f1"
                    ),
                    "test_accuracy": _finite_float(
                        row["test_accuracy"], field=f"{prefix}.test_accuracy"
                    ),
                    "test_balanced_accuracy": _finite_float(
                        row["test_balanced_accuracy"],
                        field=f"{prefix}.test_balanced_accuracy",
                        nullable=True,
                    ),
                    "test_f1": _finite_float(row["test_f1"], field=f"{prefix}.test_f1"),
                    "portfolio_id": _require_string(
                        row["portfolio_id"], field=f"{prefix}.portfolio_id"
                    ),
                    "ending_portfolio_value": _money(
                        row["ending_portfolio_value"],
                        field=f"{prefix}.ending_portfolio_value",
                    ),
                    "total_return": _finite_float(
                        row["total_return"], field=f"{prefix}.total_return"
                    ),
                    "maximum_drawdown": _finite_float(
                        row["maximum_drawdown"], field=f"{prefix}.maximum_drawdown"
                    ),
                    "volatility": _finite_float(
                        row["volatility"], field=f"{prefix}.volatility", nullable=True
                    ),
                    "sharpe_ratio": _finite_float(
                        row["sharpe_ratio"], field=f"{prefix}.sharpe_ratio", nullable=True
                    ),
                    "number_of_trades": _integer(
                        row["number_of_trades"], field=f"{prefix}.number_of_trades"
                    ),
                    "fees_paid": _money(row["fees_paid"], field=f"{prefix}.fees_paid"),
                    "result_versus_buy_and_hold": _finite_float(
                        row["result_versus_buy_and_hold"],
                        field=f"{prefix}.result_versus_buy_and_hold",
                    ),
                }
            )
        expected_models = {
            str(key): str(value)
            for key, value in _require_mapping(
                run.manifest.get("registered_model_ids"), field="registered_model_ids"
            ).items()
        }
        if {
            row["model_id"]: row["registered_model_id"] for row in rows
        } != expected_models:
            raise ArtifactIntegrityError(
                "combined_leaderboard.csv identities do not match the run manifest"
            )
        # The producer writes the leaderboard in its documented deterministic
        # validation ranking. Preserve that immutable ordering rather than
        # recreating a selection policy in the delivery layer.
        return rows

    def load_equity_curves(
        self, run_id: str, *, limit: int, offset: int
    ) -> dict[str, Any]:
        rows = self._equity_rows(run_id)
        return _paginate(rows, limit=limit, offset=offset)

    def load_drawdowns(self, run_id: str, *, limit: int, offset: int) -> dict[str, Any]:
        rows = [
            {
                key: row[key]
                for key in (
                    "date",
                    "series_id",
                    "portfolio_type",
                    "model_id",
                    "portfolio_id",
                    "drawdown",
                )
            }
            for row in self._equity_rows(run_id)
        ]
        return _paginate(rows, limit=limit, offset=offset)

    def _equity_rows(self, run_id: str) -> list[dict[str, Any]]:
        run = self._validate_run(run_id)
        source = self._csv_artifact(run, "daily_equity_curves.csv")
        fields = (
            "date",
            "series_id",
            "portfolio_type",
            "model_id",
            "portfolio_id",
            "cash",
            "shares",
            "market_price",
            "total_equity",
            "drawdown",
        )
        self._require_columns(source, fields, artifact="daily_equity_curves.csv")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            prefix = f"daily_equity_curves[{index}]"
            model_id = str(row["model_id"]).strip() or None
            rows.append(
                {
                    "date": _iso_date(row["date"], field=f"{prefix}.date"),
                    "series_id": _require_string(
                        row["series_id"], field=f"{prefix}.series_id"
                    ),
                    "portfolio_type": _require_string(
                        row["portfolio_type"], field=f"{prefix}.portfolio_type"
                    ),
                    "model_id": model_id,
                    "portfolio_id": _require_string(
                        row["portfolio_id"], field=f"{prefix}.portfolio_id"
                    ),
                    "cash": _money(row["cash"], field=f"{prefix}.cash"),
                    "shares": _integer(row["shares"], field=f"{prefix}.shares"),
                    "market_price": _money(
                        row["market_price"], field=f"{prefix}.market_price"
                    ),
                    "total_equity": _money(
                        row["total_equity"], field=f"{prefix}.total_equity"
                    ),
                    "drawdown": _finite_float(
                        row["drawdown"], field=f"{prefix}.drawdown"
                    ),
                }
            )
        declared = _require_mapping(
            run.manifest.get("portfolio_ids"), field="portfolio_ids"
        )
        allowed_series: set[tuple[str, str, str | None, str]] = set()
        for variant in _MODEL_VARIANTS:
            identities = _require_mapping(
                declared.get(variant), field=f"portfolio_ids.{variant}"
            )
            allowed_series.add(
                (variant, "model", variant, _require_string(
                    identities.get("strategy"), field=f"{variant} strategy portfolio_id"
                ))
            )
        benchmark_ids = _require_mapping(
            declared.get(_MODEL_VARIANTS[0]), field="benchmark portfolio IDs"
        )
        allowed_series.update(
            {
                (
                    "CASH-ONLY",
                    "cash_only",
                    None,
                    _require_string(
                        benchmark_ids.get("cash_only"), field="cash-only portfolio_id"
                    ),
                ),
                (
                    "BUY-AND-HOLD",
                    "buy_and_hold",
                    None,
                    _require_string(
                        benchmark_ids.get("buy_and_hold"),
                        field="buy-and-hold portfolio_id",
                    ),
                ),
            }
        )
        observed_series = {
            (
                row["series_id"],
                row["portfolio_type"],
                row["model_id"],
                row["portfolio_id"],
            )
            for row in rows
        }
        if observed_series != allowed_series:
            raise ArtifactIntegrityError(
                "daily_equity_curves.csv identities do not match the run manifest"
            )
        if len(
            {(row["date"], row["series_id"], row["portfolio_id"]) for row in rows}
        ) != len(rows):
            raise ArtifactIntegrityError(
                "daily_equity_curves.csv contains duplicate dated series rows"
            )
        rows.sort(key=lambda item: (item["date"], item["series_id"], item["portfolio_id"]))
        return rows

    def _experiment_ids(self, run: ValidatedRun) -> list[str]:
        ids: list[str] = []
        for relative in run.artifact_paths:
            parts = PurePosixPath(relative).parts
            if len(parts) == 3 and parts[0] == "experiments" and parts[2] == "config.json":
                try:
                    ids.append(_validate_identifier(parts[1], field="experiment_id"))
                except InvalidIdentifierError as exc:
                    raise ArtifactIntegrityError(
                        "Inventoried experiment directory has an unsafe identifier"
                    ) from exc
        if len(set(ids)) != len(ids):
            raise ArtifactIntegrityError("Vertical-slice run has duplicate experiment IDs")
        return sorted(ids)

    def _experiment_files(
        self, run: ValidatedRun, experiment_id: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        safe_id = _validate_identifier(experiment_id, field="experiment_id")
        if safe_id not in self._experiment_ids(run):
            raise ExperimentNotFoundError(
                f"Experiment '{safe_id}' was not found in run '{run.run_id}'"
            )
        prefix = f"experiments/{safe_id}/"
        observed = {
            relative.removeprefix(prefix)
            for relative in run.artifact_paths
            if relative.startswith(prefix)
        }
        expected = {
            "config.json",
            "predictions.csv",
            "signals.csv",
            "trades.csv",
            "portfolio_snapshots.csv",
            "metrics.json",
            "result.json",
            "equity_curve.png",
            "drawdown.png",
            "benchmarks/cash_only_snapshots.csv",
            "benchmarks/buy_and_hold_snapshots.csv",
            "benchmarks/buy_and_hold_trades.csv",
        }
        if observed != expected:
            raise ArtifactIntegrityError(
                f"Experiment '{safe_id}' does not contain the complete fixed artifact set"
            )
        config = self._json_artifact(run, f"experiments/{safe_id}/config.json")
        result = self._json_artifact(run, f"experiments/{safe_id}/result.json")
        metrics = self._json_artifact(run, f"experiments/{safe_id}/metrics.json")
        if config.get("experiment_id") != safe_id or result.get("experiment_id") != safe_id:
            raise ArtifactIntegrityError("Experiment identity does not match its directory")
        if metrics != result:
            raise ArtifactIntegrityError("Experiment result.json and metrics.json disagree")
        registered = _require_mapping(
            run.manifest.get("registered_model_ids"), field="registered_model_ids"
        )
        matching_variants = [
            str(variant)
            for variant, registered_id in registered.items()
            if str(registered_id) == str(config.get("model_id"))
        ]
        if len(matching_variants) != 1:
            raise ArtifactIntegrityError(
                "Experiment model does not map to exactly one registered variant"
            )
        variant_id = matching_variants[0]
        root_config_path = f"experiment_configs/{variant_id}.json"
        if root_config_path not in run.artifact_paths:
            raise ArtifactIntegrityError("Experiment's normalized root config is missing")
        if self._json_artifact(run, root_config_path) != config:
            raise ArtifactIntegrityError(
                "Experiment config disagrees with its normalized root config"
            )
        portfolio_ids = _require_mapping(
            run.manifest.get("portfolio_ids"), field="portfolio_ids"
        )
        declared = _require_mapping(
            portfolio_ids.get(variant_id), field=f"portfolio_ids.{variant_id}"
        )
        benchmarks = _require_mapping(
            config.get("benchmark_portfolio_ids"), field="benchmark_portfolio_ids"
        )
        if declared != {
            "strategy": config.get("portfolio_id"),
            "cash_only": benchmarks.get("cash_only"),
            "buy_and_hold": benchmarks.get("buy_and_hold"),
        }:
            raise ArtifactIntegrityError(
                "Experiment portfolio identities disagree with the run manifest"
            )
        return config, result

    def _variant_and_algorithm(
        self, run_id: str, registered_model_id: str
    ) -> tuple[str, str]:
        for row in self.load_model_metrics(run_id):
            if row["registered_model_id"] == registered_model_id:
                return row["model_id"], row["algorithm"]
        raise ArtifactIntegrityError(
            "Experiment model identity is absent from the model metrics table"
        )

    def _configuration(self, value: Mapping[str, Any]) -> dict[str, Any]:
        benchmark_ids = _require_mapping(
            value.get("benchmark_portfolio_ids"), field="benchmark_portfolio_ids"
        )
        thresholds = _require_mapping(value.get("thresholds"), field="thresholds")
        transaction_cost = _require_mapping(
            value.get("transaction_cost"), field="transaction_cost"
        )
        evaluation = _require_mapping(
            value.get("evaluation_dates"), field="evaluation_dates"
        )
        execution = _require_mapping(
            value.get("execution_policy"), field="execution_policy"
        )
        metric_assumptions = _require_mapping(
            value.get("metric_assumptions"), field="metric_assumptions"
        )
        schema_version = _require_string(
            value.get("schema_version"), field="config schema_version"
        )
        if schema_version != SCHEMA_VERSION:
            raise ArtifactIntegrityError("Unsupported experiment config schema_version")
        return {
            "schema_version": schema_version,
            "experiment_id": _require_string(
                value.get("experiment_id"), field="experiment_id"
            ),
            "model_id": _require_string(value.get("model_id"), field="model_id"),
            "portfolio_id": _require_string(
                value.get("portfolio_id"), field="portfolio_id"
            ),
            "benchmark_portfolio_ids": {
                "cash_only": _require_string(
                    benchmark_ids.get("cash_only"), field="cash_only portfolio_id"
                ),
                "buy_and_hold": _require_string(
                    benchmark_ids.get("buy_and_hold"), field="buy_and_hold portfolio_id"
                ),
            },
            "symbol": _require_string(value.get("symbol"), field="symbol"),
            "initial_cash": _money(value.get("initial_cash"), field="initial_cash"),
            "thresholds": {
                "buy_probability_up_gte": _finite_float(
                    thresholds.get("buy_probability_up_gte"),
                    field="buy_probability_up_gte",
                ),
                "sell_probability_up_lte": _finite_float(
                    thresholds.get("sell_probability_up_lte"),
                    field="sell_probability_up_lte",
                ),
            },
            "maximum_allocation": _finite_float(
                value.get("maximum_allocation"), field="maximum_allocation"
            ),
            "transaction_cost": {
                "type": _require_string(
                    transaction_cost.get("type"), field="transaction cost type"
                ),
                "value": _finite_float(
                    transaction_cost.get("value"), field="transaction cost value"
                ),
            },
            "evaluation_dates": self._evaluation_dates(evaluation),
            "execution_policy": {
                field: (
                    _finite_float(execution.get(field), field=field)
                    if field == "slippage_bps"
                    else _require_string(execution.get(field), field=field)
                )
                for field in (
                    "signal_timing",
                    "execution_session",
                    "reference_price_field",
                    "slippage_bps",
                    "missing_execution_price",
                    "allocation_mode",
                    "whole_share_rounding",
                    "valuation_price_field",
                    "missing_valuation_price",
                    "ending_position",
                )
            },
            "metric_assumptions": {
                "annualization_factor": _integer(
                    metric_assumptions.get("annualization_factor"),
                    field="annualization_factor",
                ),
                "risk_free_rate_annual": _finite_float(
                    metric_assumptions.get("risk_free_rate_annual"),
                    field="risk_free_rate_annual",
                ),
            },
            "created_at": _iso_timestamp(value.get("created_at"), field="created_at"),
        }

    @staticmethod
    def _prediction_metrics(value: Any) -> dict[str, Any]:
        metrics = _require_mapping(value, field="prediction_metrics")
        return {
            field: _finite_float(metrics.get(field), field=f"prediction {field}")
            for field in ("accuracy", "precision", "recall", "f1")
        } | {
            "support": _integer(metrics.get("support"), field="prediction support"),
            "positive_label": _integer(
                metrics.get("positive_label"), field="prediction positive_label"
            ),
            "average": _require_string(
                metrics.get("average"), field="prediction average"
            ),
            "zero_division": _integer(
                metrics.get("zero_division"), field="prediction zero_division"
            ),
        }

    @staticmethod
    def _benchmark_result(value: Any, *, field: str) -> dict[str, Any]:
        result = _require_mapping(value, field=field)
        return {
            "portfolio_id": _require_string(
                result.get("portfolio_id"), field=f"{field}.portfolio_id"
            ),
            "ending_equity": _money(
                result.get("ending_equity"), field=f"{field}.ending_equity"
            ),
            "total_return": _finite_float(
                result.get("total_return"), field=f"{field}.total_return"
            ),
        }

    def _result(self, value: Mapping[str, Any]) -> dict[str, Any]:
        comparison = _require_mapping(
            value.get("benchmark_comparison"), field="benchmark_comparison"
        )
        schema_version = _require_string(
            value.get("schema_version"), field="result schema_version"
        )
        if schema_version != SCHEMA_VERSION:
            raise ArtifactIntegrityError("Unsupported experiment result schema_version")
        return {
            "schema_version": schema_version,
            "experiment_id": _require_string(
                value.get("experiment_id"), field="result experiment_id"
            ),
            "model_id": _require_string(value.get("model_id"), field="result model_id"),
            "portfolio_id": _require_string(
                value.get("portfolio_id"), field="result portfolio_id"
            ),
            "prediction_metrics": self._prediction_metrics(
                value.get("prediction_metrics")
            ),
            "initial_equity": _money(
                value.get("initial_equity"), field="initial_equity"
            ),
            "ending_equity": _money(value.get("ending_equity"), field="ending_equity"),
            "total_return": _finite_float(value.get("total_return"), field="total_return"),
            "maximum_drawdown": _finite_float(
                value.get("maximum_drawdown"), field="maximum_drawdown"
            ),
            "volatility": _finite_float(
                value.get("volatility"), field="volatility", nullable=True
            ),
            "sharpe_ratio": _finite_float(
                value.get("sharpe_ratio"), field="sharpe_ratio", nullable=True
            ),
            "number_of_trades": _integer(
                value.get("number_of_trades"), field="number_of_trades"
            ),
            "win_rate": _finite_float(
                value.get("win_rate"), field="win_rate", nullable=True
            ),
            "average_gain": _money(
                value.get("average_gain"), field="average_gain", nullable=True
            ),
            "average_loss": _money(
                value.get("average_loss"), field="average_loss", nullable=True
            ),
            "turnover": _finite_float(value.get("turnover"), field="turnover"),
            "fees_paid": _money(value.get("fees_paid"), field="fees_paid"),
            "time_in_market": _finite_float(
                value.get("time_in_market"), field="time_in_market"
            ),
            "comparison_with_buy_and_hold": _finite_float(
                value.get("comparison_with_buy_and_hold"),
                field="comparison_with_buy_and_hold",
            ),
            "benchmark_comparison": {
                "cash_only": self._benchmark_result(
                    comparison.get("cash_only"), field="cash_only"
                ),
                "buy_and_hold": self._benchmark_result(
                    comparison.get("buy_and_hold"), field="buy_and_hold"
                ),
                "excess_return_vs_cash_only": _finite_float(
                    comparison.get("excess_return_vs_cash_only"),
                    field="excess_return_vs_cash_only",
                ),
                "excess_return_vs_buy_and_hold": _finite_float(
                    comparison.get("excess_return_vs_buy_and_hold"),
                    field="excess_return_vs_buy_and_hold",
                ),
            },
        }

    def list_experiments(self, run_id: str) -> list[dict[str, Any]]:
        run = self._validate_run(run_id)
        rows: list[dict[str, Any]] = []
        for experiment_id in self._experiment_ids(run):
            raw_config, raw_result = self._experiment_files(run, experiment_id)
            config = self._configuration(raw_config)
            result = self._result(raw_result)
            if (
                result["model_id"] != config["model_id"]
                or result["portfolio_id"] != config["portfolio_id"]
            ):
                raise ArtifactIntegrityError(
                    "Experiment configuration and result identities disagree"
                )
            variant_id, algorithm = self._variant_and_algorithm(
                run.run_id, config["model_id"]
            )
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "variant_id": variant_id,
                    "registered_model_id": config["model_id"],
                    "algorithm": algorithm,
                    "portfolio_id": config["portfolio_id"],
                    "symbol": config["symbol"],
                    "evaluation_dates": config["evaluation_dates"],
                    "ending_equity": result["ending_equity"],
                    "total_return": result["total_return"],
                    "maximum_drawdown": result["maximum_drawdown"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "number_of_trades": result["number_of_trades"],
                }
            )
        rows.sort(key=lambda item: item["experiment_id"])
        return rows

    def load_experiment(self, run_id: str, experiment_id: str) -> dict[str, Any]:
        run = self._validate_run(run_id)
        raw_config, raw_result = self._experiment_files(run, experiment_id)
        config = self._configuration(raw_config)
        result = self._result(raw_result)
        if result["model_id"] != config["model_id"]:
            raise ArtifactIntegrityError("Experiment model identities disagree")
        variant_id, algorithm = self._variant_and_algorithm(run.run_id, config["model_id"])
        return {
            **self._context(run),
            "variant_id": variant_id,
            "algorithm": algorithm,
            "configuration": config,
            "result": result,
            "actual_class_usage": ACTUAL_CLASS_LIMITATION,
        }

    @staticmethod
    def _parse_trades(source: list[dict[str, str]], *, artifact: str) -> list[dict[str, Any]]:
        if not source:
            return []
        ArtifactRepository._require_columns(source, _TRADE_COLUMNS, artifact=artifact)
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            prefix = f"{artifact}[{index}]"
            rows.append(
                {
                    "schema_version": _schema_version(
                        row["schema_version"], field=f"{prefix}.schema_version"
                    ),
                    "trade_id": _require_string(row["trade_id"], field=f"{prefix}.trade_id"),
                    "portfolio_id": _require_string(
                        row["portfolio_id"], field=f"{prefix}.portfolio_id"
                    ),
                    "signal_date": _iso_date(
                        row["signal_date"], field=f"{prefix}.signal_date"
                    ),
                    "execution_date": _iso_date(
                        row["execution_date"], field=f"{prefix}.execution_date"
                    ),
                    "action": _require_string(row["action"], field=f"{prefix}.action"),
                    "quantity": _integer(row["quantity"], field=f"{prefix}.quantity"),
                    "reference_price": _money(
                        row["reference_price"], field=f"{prefix}.reference_price"
                    ),
                    "execution_price": _money(
                        row["execution_price"], field=f"{prefix}.execution_price"
                    ),
                    "fee": _money(row["fee"], field=f"{prefix}.fee"),
                    "cash_after_trade": _money(
                        row["cash_after_trade"], field=f"{prefix}.cash_after_trade"
                    ),
                    "shares_after_trade": _integer(
                        row["shares_after_trade"], field=f"{prefix}.shares_after_trade"
                    ),
                }
            )
        rows.sort(key=lambda item: (item["execution_date"], item["trade_id"]))
        if len({row["trade_id"] for row in rows}) != len(rows):
            raise ArtifactIntegrityError(f"{artifact} contains duplicate trade IDs")
        return rows

    @staticmethod
    def _parse_snapshots(
        source: list[dict[str, str]], *, artifact: str
    ) -> list[dict[str, Any]]:
        ArtifactRepository._require_columns(source, _SNAPSHOT_COLUMNS, artifact=artifact)
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(source):
            prefix = f"{artifact}[{index}]"
            rows.append(
                {
                    "schema_version": _schema_version(
                        row["schema_version"], field=f"{prefix}.schema_version"
                    ),
                    "portfolio_id": _require_string(
                        row["portfolio_id"], field=f"{prefix}.portfolio_id"
                    ),
                    "date": _iso_date(row["date"], field=f"{prefix}.date"),
                    "cash": _money(row["cash"], field=f"{prefix}.cash"),
                    "shares": _integer(row["shares"], field=f"{prefix}.shares"),
                    "market_price": _money(
                        row["market_price"], field=f"{prefix}.market_price"
                    ),
                    "position_value": _money(
                        row["position_value"], field=f"{prefix}.position_value"
                    ),
                    "total_equity": _money(
                        row["total_equity"], field=f"{prefix}.total_equity"
                    ),
                    "realized_profit_loss": _money(
                        row["realized_profit_loss"],
                        field=f"{prefix}.realized_profit_loss",
                    ),
                    "unrealized_profit_loss": _money(
                        row["unrealized_profit_loss"],
                        field=f"{prefix}.unrealized_profit_loss",
                    ),
                }
            )
        rows.sort(key=lambda item: (item["date"], item["portfolio_id"]))
        if len({(row["date"], row["portfolio_id"]) for row in rows}) != len(rows):
            raise ArtifactIntegrityError(f"{artifact} contains duplicate dated snapshots")
        return rows

    def load_trades(
        self, run_id: str, experiment_id: str, *, limit: int, offset: int
    ) -> dict[str, Any]:
        run = self._validate_run(run_id)
        safe_id = _validate_identifier(experiment_id, field="experiment_id")
        raw_config, _raw_result = self._experiment_files(run, safe_id)
        config = self._configuration(raw_config)
        relative = f"experiments/{safe_id}/trades.csv"
        rows = self._parse_trades(
            self._csv_artifact(run, relative, required_columns=_TRADE_COLUMNS),
            artifact=relative,
        )
        if any(row["portfolio_id"] != config["portfolio_id"] for row in rows):
            raise ArtifactIntegrityError("Trade portfolio identity disagrees with experiment config")
        return _paginate(rows, limit=limit, offset=offset)

    def load_snapshots(
        self, run_id: str, experiment_id: str, *, limit: int, offset: int
    ) -> dict[str, Any]:
        run = self._validate_run(run_id)
        safe_id = _validate_identifier(experiment_id, field="experiment_id")
        raw_config, _raw_result = self._experiment_files(run, safe_id)
        config = self._configuration(raw_config)
        relative = f"experiments/{safe_id}/portfolio_snapshots.csv"
        rows = self._parse_snapshots(
            self._csv_artifact(run, relative, required_columns=_SNAPSHOT_COLUMNS),
            artifact=relative,
        )
        if any(row["portfolio_id"] != config["portfolio_id"] for row in rows):
            raise ArtifactIntegrityError(
                "Snapshot portfolio identity disagrees with experiment config"
            )
        return _paginate(rows, limit=limit, offset=offset)

    def load_benchmarks(
        self, run_id: str, experiment_id: str, *, limit: int, offset: int
    ) -> dict[str, Any]:
        run = self._validate_run(run_id)
        safe_id = _validate_identifier(experiment_id, field="experiment_id")
        raw_config, raw_result = self._experiment_files(run, safe_id)
        config = self._configuration(raw_config)
        result = self._result(raw_result)
        base = f"experiments/{safe_id}/benchmarks"
        cash_rows = self._parse_snapshots(
            self._csv_artifact(
                run,
                f"{base}/cash_only_snapshots.csv",
                required_columns=_SNAPSHOT_COLUMNS,
            ),
            artifact=f"{base}/cash_only_snapshots.csv",
        )
        buy_hold_rows = self._parse_snapshots(
            self._csv_artifact(
                run,
                f"{base}/buy_and_hold_snapshots.csv",
                required_columns=_SNAPSHOT_COLUMNS,
            ),
            artifact=f"{base}/buy_and_hold_snapshots.csv",
        )
        buy_hold_trades = self._parse_trades(
            self._csv_artifact(
                run,
                f"{base}/buy_and_hold_trades.csv",
                required_columns=_TRADE_COLUMNS,
            ),
            artifact=f"{base}/buy_and_hold_trades.csv",
        )
        cash_result = result["benchmark_comparison"]["cash_only"]
        buy_hold_result = result["benchmark_comparison"]["buy_and_hold"]
        if (
            cash_result["portfolio_id"]
            != config["benchmark_portfolio_ids"]["cash_only"]
            or buy_hold_result["portfolio_id"]
            != config["benchmark_portfolio_ids"]["buy_and_hold"]
            or any(
                row["portfolio_id"] != cash_result["portfolio_id"]
                for row in cash_rows
            )
            or any(
                row["portfolio_id"] != buy_hold_result["portfolio_id"]
                for row in buy_hold_rows
            )
            or any(
                row["portfolio_id"] != buy_hold_result["portfolio_id"]
                for row in buy_hold_trades
            )
        ):
            raise ArtifactIntegrityError("Benchmark identities disagree with experiment config")
        return {
            "cash_only": {
                "portfolio_id": cash_result["portfolio_id"],
                "ending_equity": cash_result["ending_equity"],
                "total_return": cash_result["total_return"],
                **_paginate(cash_rows, limit=limit, offset=offset),
            },
            "buy_and_hold": {
                "portfolio_id": buy_hold_result["portfolio_id"],
                "ending_equity": buy_hold_result["ending_equity"],
                "total_return": buy_hold_result["total_return"],
                **_paginate(buy_hold_rows, limit=limit, offset=offset),
            },
            "buy_and_hold_trades": {
                "portfolio_id": buy_hold_result["portfolio_id"],
                **_paginate(buy_hold_trades, limit=limit, offset=offset),
            },
        }

    def load_source(self, run_id: str) -> dict[str, Any]:
        run = self._validate_run(run_id)
        context = self._context(run)
        raw = _require_mapping(run.manifest.get("data"), field="manifest data")
        pipeline: dict[str, Any] | None = None
        pipeline_path = "input_snapshot/pipeline_metadata.json"
        if pipeline_path in run.artifact_paths:
            evidence = self._json_artifact(run, pipeline_path)
            prices = evidence.get("prices_artifact")
            safe_prices: dict[str, Any] | None = None
            if isinstance(prices, Mapping):
                safe_prices = {}
                if "schema_version" in prices:
                    safe_prices["schema_version"] = _schema_version(
                        prices["schema_version"],
                        field="pipeline prices schema_version",
                    )
                for key in ("status", "provider", "frequency"):
                    if key in prices:
                        safe_prices[key] = _require_string(
                            prices[key], field=f"pipeline prices {key}"
                        )
                if "sha256" in prices:
                    safe_prices["sha256"] = _sha256_string(
                        prices["sha256"], field="pipeline prices sha256"
                    )
                if "row_count" in prices:
                    safe_prices["row_count"] = _integer(
                        prices["row_count"], field="pipeline prices row_count"
                    )
                symbols = prices.get("symbols")
                if isinstance(symbols, Mapping):
                    normalized_symbols: dict[str, dict[str, Any]] = {}
                    for symbol, record in symbols.items():
                        symbol_id = _require_string(symbol, field="pipeline symbol")
                        record_mapping = _require_mapping(
                            record, field=f"pipeline symbol {symbol_id}"
                        )
                        normalized_symbols[symbol_id] = {
                            "date_start": _iso_date(
                                record_mapping.get("date_start"),
                                field=f"pipeline symbol {symbol_id} date_start",
                            ),
                            "date_end": _iso_date(
                                record_mapping.get("date_end"),
                                field=f"pipeline symbol {symbol_id} date_end",
                            ),
                            "row_count": _integer(
                                record_mapping.get("row_count"),
                                field=f"pipeline symbol {symbol_id} row_count",
                            ),
                            "source": _require_string(
                                record_mapping.get("source"),
                                field=f"pipeline symbol {symbol_id} source",
                            ),
                        }
                    safe_prices["symbols"] = normalized_symbols
                _assert_finite_json(safe_prices)
            requested = evidence.get("symbols_requested", [])
            if not isinstance(requested, list):
                raise ValueError("pipeline symbols_requested must be an array")
            pipeline = {
                "schema_version": _schema_version(
                    evidence.get("schema_version"),
                    field="pipeline schema_version",
                    nullable=True,
                ),
                "producer": _producer_metadata(
                    evidence.get("producer"), field="pipeline producer"
                ),
                "generated_at": _optional_iso_datetime(
                    evidence.get("generated_at"), field="pipeline generated_at"
                ),
                "provider": _optional_string(
                    evidence.get("provider"), field="pipeline provider"
                ),
                "symbols_requested": [
                    _require_string(item, field="pipeline requested symbol")
                    for item in requested
                ],
                "prices_artifact": safe_prices,
            }

        migration: dict[str, Any] | None = None
        migration_path = "input_snapshot/migration_manifest.json"
        if migration_path in run.artifact_paths:
            evidence = self._json_artifact(run, migration_path)
            raw_limitations = evidence.get("limitations", [])
            if not isinstance(raw_limitations, list):
                raise ValueError("migration limitations must be an array")
            migration = {
                "schema_version": _schema_version(
                    evidence.get("schema_version"),
                    field="migration schema_version",
                    nullable=True,
                ),
                "status": _optional_string(
                    evidence.get("status"), field="migration status"
                ),
                "producer": _producer_metadata(
                    evidence.get("producer"), field="migration producer"
                ),
                "migration_strategy": _optional_string(
                    evidence.get("migration_strategy"),
                    field="migration strategy",
                ),
                "provenance_strength": _optional_string(
                    evidence.get("provenance_strength"),
                    field="migration provenance_strength",
                ),
                "limitations": [
                    _require_string(item, field="migration limitation")
                    for item in raw_limitations
                ],
            }
        return {
            **context,
            "source": {
                "source": _require_string(raw.get("source"), field="source"),
                "acquisition_mode": context["source_mode"],
                "is_real_data": _boolean(
                    raw.get("is_real_data"), field="is_real_data"
                ),
                "live_fetch_succeeded": _boolean(
                    raw.get("live_fetch_succeeded"), field="live_fetch_succeeded"
                ),
                "live_acquisition_error_present": bool(
                    raw.get("live_acquisition_error")
                ),
                "original_provider": _require_string(
                    raw.get("original_provider"), field="original_provider"
                ),
                "legacy_metadata_preserved": _boolean(
                    raw.get("legacy_metadata_preserved"),
                    field="legacy_metadata_preserved",
                ),
                "producer_component": _optional_string(
                    raw.get("producer_component"), field="producer_component"
                ),
                "row_count": _integer(raw.get("row_count"), field="row_count"),
                "input_prices_sha256": _sha256_string(
                    raw.get("input_prices_sha256"), field="input_prices_sha256"
                ),
                "frozen_prices_sha256": _sha256_string(
                    raw.get("frozen_prices_sha256"), field="frozen_prices_sha256"
                ),
                "provenance_sha256": _sha256_string(
                    raw.get("provenance_sha256"),
                    field="provenance_sha256",
                    nullable=True,
                ),
                "migration_strategy": _optional_string(
                    raw.get("migration_strategy"), field="migration_strategy"
                ),
                "provenance_strength": _optional_string(
                    raw.get("provenance_strength"), field="provenance_strength"
                ),
            },
            "pipeline_evidence": pipeline,
            "migration_evidence": migration,
        }


@lru_cache(maxsize=16)
def _repository_for_settings(
    repository_root: str,
    vertical_slice_root: str,
    model_training_root: str,
    default_run_id: str | None,
) -> ArtifactRepository:
    return ArtifactRepository(
        RepositorySettings(
            repository_root=Path(repository_root),
            vertical_slice_root=Path(vertical_slice_root),
            model_training_root=Path(model_training_root),
            default_vertical_slice_run_id=default_run_id,
        )
    )


def get_artifact_repository() -> ArtifactRepository:
    """Return a cache partitioned by the current artifact-root configuration."""

    settings = settings_from_environment()
    return _repository_for_settings(
        str(settings.repository_root),
        str(settings.vertical_slice_root),
        str(settings.model_training_root),
        settings.default_vertical_slice_run_id,
    )


def clear_artifact_repository_cache() -> None:
    """Clear repository instances for tests or explicit process reconfiguration."""

    _repository_for_settings.cache_clear()
