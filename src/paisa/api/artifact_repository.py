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
from dataclasses import dataclass, replace
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit

from paisa.api.classification_metrics import validate_confusion_derived_metrics
from paisa.experiments.artifacts import (
    TrainingRunEvidence,
    validate_training_run_artifacts,
)
from paisa.experiments.unified_authority import (
    AUTHORITATIVE_COMPARISON_MANIFESTS,
    AUTHORITATIVE_SOURCE_ROOTS,
    expected_evidence_bindings,
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
_SUPPORTS_SECURE_DIR_FD = (
    hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.scandir in os.supports_fd
)
_MODEL_VARIANTS = ("M-001", "M-002", "M-003", "M-004")
_UNIFIED_MODEL_VARIANTS = (
    "M-001",
    "M-002",
    "M-003",
    "M-004",
    "M-005",
    "M-006",
)
_UNIFIED_REFERENCE_BASELINES = (
    "majority-class-baseline",
    "previous-direction-baseline",
)
_UNIFIED_BENCHMARKS = ("cash-only", "buy-and-hold")
_UNIFIED_HOLDOUT_TERM = "shared project holdout"
_UNIFIED_ARTIFACT_TYPE = "paisa_unified_price_model_comparison"
_UNIFIED_AUTHORED_FILES = {
    "comparison_contract.json",
    "date_alignment.json",
    "derived/reference_baseline_predictions.csv",
    "final_summary.json",
    "metrics.json",
    "rankings.json",
    "report.md",
    "source_roots.json",
}
_UNIFIED_PORTFOLIO_FILES = (
    "config.json",
    "signals.csv",
    "portfolio_snapshots.csv",
    "trades.csv",
    "benchmarks/cash_only_snapshots.csv",
    "benchmarks/buy_and_hold_snapshots.csv",
    "benchmarks/buy_and_hold_trades.csv",
)
_UNIFIED_EVIDENCE_FILES = {
    "evidence/market_data/baseline_psx_prices.csv",
    "evidence/market_data/lstm_ogdc_prices.csv",
    "evidence/source_manifests/baseline_training_run_manifest.json",
    "evidence/source_manifests/baseline_vertical_manifest_certificate.json",
    "evidence/source_manifests/lstm_artifact_manifest.json",
    "evidence/source_manifests/lstm_run_manifest.json",
    "evidence/splits/baseline_split_manifest.json",
    "evidence/splits/lstm_split_manifest.json",
    *(
        f"evidence/predictions/{variant}/predictions_all_splits.csv"
        for variant in _UNIFIED_MODEL_VARIANTS
    ),
    *(
        f"evidence/portfolios/{variant}/{relative}"
        for variant in _UNIFIED_MODEL_VARIANTS
        for relative in _UNIFIED_PORTFOLIO_FILES
    ),
}
_UNIFIED_EXPECTED_FILES = _UNIFIED_AUTHORED_FILES | _UNIFIED_EVIDENCE_FILES
_UNIFIED_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_UNIFIED_MAX_CLASSIFICATION_COUNT = 10_000_000
_UNIFIED_MAX_TREE_ENTRIES = 128
_UNIFIED_MAX_TREE_DEPTH = 8
_EXECUTABLE_MODEL_SUFFIXES = {
    ".joblib",
    ".pkl",
    ".pickle",
    ".keras",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
}
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


class StockNotFoundError(ArtifactRepositoryError):
    pass


class InvalidDateRangeError(ArtifactRepositoryError):
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
    unified_comparison_root: Path | None = None
    default_unified_comparison_run_id: str | None = None
    unified_comparison_source_authority: Mapping[str, Mapping[str, str]] | None = None
    unified_comparison_manifest_authority: Mapping[str, str] | None = None


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


@dataclass(frozen=True, slots=True)
class ValidatedUnifiedComparison:
    comparison_id: str
    manifest: dict[str, Any]
    summary: dict[str, Any]
    contract: dict[str, Any]
    sources: dict[str, Any]
    date_alignment: dict[str, Any]
    metrics: dict[str, Any]
    rankings: dict[str, Any]
    artifact_paths: frozenset[str]
    signature: tuple[tuple[Any, ...], ...]


@dataclass(slots=True)
class _PinnedArtifactTree:
    """One descriptor-pinned view of a comparison run's regular files."""

    descriptors: dict[str, int]
    signature: tuple[tuple[Any, ...], ...]

    def close(self) -> None:
        for descriptor in self.descriptors.values():
            os.close(descriptor)
        self.descriptors.clear()

    def read(
        self, relative: str, *, label: str, max_bytes: int | None = None
    ) -> bytes:
        descriptor = self.descriptors.get(relative)
        if descriptor is None:
            raise ValueError(f"{label} is missing or has an unsafe path")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return _read_regular_descriptor(
            descriptor, label=label, max_bytes=max_bytes
        )


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _configured_root(value: str | None, *, default: str, repository_root: Path) -> Path:
    raw = Path(value or default).expanduser()
    return (raw if raw.is_absolute() else repository_root / raw).resolve()


def _configured_lexical_root(
    value: str | None, *, default: str, repository_root: Path
) -> Path:
    """Keep path components visible so later validation can reject symlinks."""

    raw = Path(value or default).expanduser()
    candidate = raw if raw.is_absolute() else repository_root / raw
    return Path(os.path.abspath(os.fspath(candidate)))


def settings_from_environment() -> RepositorySettings:
    repository_root = _repository_root()
    default_run = os.getenv("PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID")
    if default_run is not None:
        default_run = default_run.strip() or None
    default_comparison = os.getenv("PAISA_DEFAULT_UNIFIED_COMPARISON_RUN_ID")
    if default_comparison is not None:
        default_comparison = default_comparison.strip() or None
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
        unified_comparison_root=_configured_lexical_root(
            os.getenv("PAISA_UNIFIED_COMPARISON_ROOT"),
            default="artifacts/model_comparisons",
            repository_root=repository_root,
        ),
        default_unified_comparison_run_id=default_comparison,
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


def _read_regular_descriptor(
    descriptor: int, *, label: str, max_bytes: int | None = None
) -> bytes:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file")
    if max_bytes is not None and before.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the allowed size")
    chunks: list[bytes] = []
    byte_count = 0
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            break
        byte_count += len(chunk)
        if max_bytes is not None and byte_count > max_bytes:
            raise ValueError(f"{label} exceeds the allowed size")
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


def _open_regular_bytes(
    path: Path, *, label: str, max_bytes: int | None = None
) -> bytes:
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
        return _read_regular_descriptor(
            descriptor, label=label, max_bytes=max_bytes
        )
    finally:
        os.close(descriptor)


def _open_regular_bytes_beneath(
    root: Path,
    relative: str,
    *,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    """Open a regular file without following any path-component symlink."""

    safe_relative = _safe_relative_path(relative, field="artifact path")
    parts = PurePosixPath(safe_relative).parts
    if not _SUPPORTS_SECURE_DIR_FD:  # pragma: no cover - non-POSIX fallback
        candidate = root
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError(f"{label} has a symbolic-link path component")
        try:
            candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"{label} escapes its artifact root") from exc
        return _open_regular_bytes(candidate, label=label, max_bytes=max_bytes)

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        directory_flags |= os.O_CLOEXEC
        file_flags |= os.O_CLOEXEC

    descriptors: list[int] = []
    try:
        current = os.open(root, directory_flags)
        descriptors.append(current)
        for part in parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        artifact = os.open(parts[-1], file_flags, dir_fd=current)
        descriptors.append(artifact)
        return _read_regular_descriptor(
            artifact, label=label, max_bytes=max_bytes
        )
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise ValueError(f"{label} is missing or has an unsafe path") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _directory_open_flags() -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _regular_open_flags() -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        flags |= os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return flags


def _open_absolute_directory_descriptor(path: Path) -> int:
    """Open every absolute-path component without following links.

    The returned descriptor pins the selected directory even if a local writer
    later renames an ancestor.  Platforms without descriptor-relative,
    no-follow traversal fail closed instead of reverting to check-then-open.
    """

    if not _SUPPORTS_SECURE_DIR_FD:
        raise NotImplementedError(
            "Secure configured-root descriptor traversal is unavailable"
        )
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute() or not absolute.anchor:
        raise ValueError("Configured artifact root must be absolute")
    flags = _directory_open_flags()
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory_descriptor(parent: int, name: str) -> int:
    descriptor = os.open(name, _directory_open_flags(), dir_fd=parent)
    try:
        metadata = os.fstat(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise ValueError("Artifact path component is not a directory")
    return descriptor


def _signature_record(relative: str, metadata: os.stat_result) -> tuple[Any, ...]:
    return (
        relative,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _capture_pinned_artifact_tree(root_descriptor: int) -> _PinnedArtifactTree:
    """Capture a run tree and retain every regular-file descriptor.

    Names are traversed only relative to already-open directories.  Files are
    read later through the retained descriptors, so a rename cannot redirect a
    checksum or JSON read to a different tree.
    """

    root_metadata = os.fstat(root_descriptor)
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("Artifact root must be a directory")
    records: list[tuple[Any, ...]] = [_signature_record(".", root_metadata)]
    descriptors: dict[str, int] = {}
    entry_count = 0

    def visit(directory_descriptor: int, prefix: PurePosixPath) -> None:
        nonlocal entry_count
        names: list[str] = []
        with os.scandir(directory_descriptor) as entries:
            for entry in entries:
                entry_count += 1
                if entry_count > _UNIFIED_MAX_TREE_ENTRIES:
                    raise ValueError(
                        "Unified comparison artifact tree exceeds the entry limit"
                    )
                name = entry.name
                if not name or "/" in name or name in {".", ".."}:
                    raise ValueError(
                        "Artifact tree contains an unsafe entry name"
                    )
                names.append(name)
        for name in sorted(names):
            relative_path = prefix / name
            if len(relative_path.parts) > _UNIFIED_MAX_TREE_DEPTH:
                raise ValueError(
                    "Unified comparison artifact tree exceeds the depth limit"
                )
            relative = relative_path.as_posix()
            observed = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(observed.st_mode):
                raise ValueError(
                    f"Artifact tree contains a symbolic link: {relative}"
                )
            if stat.S_ISDIR(observed.st_mode):
                child = _open_child_directory_descriptor(
                    directory_descriptor, name
                )
                try:
                    metadata = os.fstat(child)
                    records.append(_signature_record(relative, metadata))
                    visit(child, relative_path)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(observed.st_mode):
                raise ValueError(
                    f"Artifact tree contains a special file: {relative}"
                )
            descriptor = os.open(
                name, _regular_open_flags(), dir_fd=directory_descriptor
            )
            try:
                metadata = os.fstat(descriptor)
            except BaseException:
                os.close(descriptor)
                raise
            if not stat.S_ISREG(metadata.st_mode):
                os.close(descriptor)
                raise ValueError(
                    f"Artifact tree contains a non-regular file: {relative}"
                )
            records.append(_signature_record(relative, metadata))
            descriptors[relative] = descriptor

    try:
        visit(root_descriptor, PurePosixPath())
        return _PinnedArtifactTree(
            descriptors=descriptors,
            signature=tuple(sorted(records, key=lambda item: item[0])),
        )
    except BaseException:
        for descriptor in descriptors.values():
            os.close(descriptor)
        raise


def _descriptor_tree_signature(
    root_descriptor: int,
) -> tuple[tuple[Any, ...], ...]:
    captured = _capture_pinned_artifact_tree(root_descriptor)
    try:
        return captured.signature
    finally:
        captured.close()


def _decode_json(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains a duplicate JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not valid finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    _assert_finite_json(value)
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    return _decode_json(_open_regular_bytes(path, label=label), label=label)


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
    if path.is_absolute() or PureWindowsPath(raw).drive or raw != path.as_posix():
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


def _positive_decimal(value: Any, *, field: str, nullable: bool = False) -> str | None:
    text = "" if value is None else str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        if nullable:
            return None
        raise ValueError(f"{field} is required")
    if not re.fullmatch(r"[0-9]{1,100}(?:\.[0-9]{1,18})?", text):
        raise ValueError(
            f"{field} must be a positive fixed-point decimal with at most "
            "100 integer digits and 18 decimal places"
        )
    try:
        parsed = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"{field} must be a decimal") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError(f"{field} must be a finite, strictly positive decimal")
    decimal_places = max(2, -parsed.as_tuple().exponent)
    return format(parsed, f".{decimal_places}f")


def _paginate(rows: list[dict[str, Any]], *, limit: int, offset: int) -> dict[str, Any]:
    return {
        "total": len(rows),
        "limit": limit,
        "offset": offset,
        "rows": rows[offset : offset + limit],
    }


def _strict_float(value: Any, *, field: str, nullable: bool = False) -> float | None:
    """Validate persisted JSON numbers without accepting numeric strings."""

    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _bounded_float(
    value: Any,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    nullable: bool = False,
) -> float | None:
    """Validate a finite persisted metric against its semantic domain."""

    result = _strict_float(value, field=field, nullable=nullable)
    if result is None:
        return None
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be greater than or equal to {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be less than or equal to {maximum}")
    return result


def _nonnegative_money(
    value: Any, *, field: str, nullable: bool = False
) -> str | None:
    result = _money(value, field=field, nullable=nullable)
    if result is not None and Decimal(result) < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


_PUBLIC_NETWORK_URL = re.compile(
    r"\b(?:https?|wss?|ftp)://[^\s\"'<>]+",
    re.IGNORECASE,
)
_PUBLIC_ABSOLUTE_PATH = re.compile(
    r"(?:"
    r"(?:^|[^A-Za-z0-9_/])/{1,2}(?=\S)"
    r"|(?:^|[^A-Za-z0-9_\\])\\\\(?=\S)"
    r"|(?:^|[^A-Za-z0-9_])[A-Za-z]:[\\/]"
    r"|(?:^|[^A-Za-z0-9_])file:"
    r")",
    re.IGNORECASE,
)
_PUBLIC_EXECUTABLE_MODEL_REFERENCE = re.compile(
    r"\.(?:joblib|pkl|pickle|keras|h5|hdf5|onnx|pt|pth|pb|ckpt|safetensors)"
    r"(?=$|[^A-Za-z0-9])",
    re.IGNORECASE,
)


def _public_url_contains_prohibited_artifact_text(value: str) -> bool:
    """Inspect URL metadata without treating an ordinary URL path as local."""

    for match in _PUBLIC_NETWORK_URL.finditer(value):
        parsed = urlsplit(match.group(0))
        metadata = unquote(f"{parsed.query} {parsed.fragment}")
        decoded_path = unquote(parsed.path)
        if (
            _PUBLIC_ABSOLUTE_PATH.search(metadata)
            or _PUBLIC_EXECUTABLE_MODEL_REFERENCE.search(metadata)
            or _PUBLIC_EXECUTABLE_MODEL_REFERENCE.search(decoded_path)
        ):
            return True
    return False


def _validate_public_text_tree(value: Any, *, field: str) -> None:
    """Reject local paths and executable-model references from public text."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            _validate_public_text_tree(child, field=f"{field}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_public_text_tree(child, field=f"{field}[{index}]")
        return
    if not isinstance(value, str):
        return
    without_network_urls = _PUBLIC_NETWORK_URL.sub("", value)
    if (
        _public_url_contains_prohibited_artifact_text(value)
        or _PUBLIC_ABSOLUTE_PATH.search(without_network_urls)
    ):
        raise ValueError(f"{field} must not expose an absolute filesystem path")
    if _PUBLIC_EXECUTABLE_MODEL_REFERENCE.search(value):
        raise ValueError(f"{field} must not expose an executable model reference")


def _strict_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a JSON integer")
    return value


def _nonnegative_integer(value: Any, *, field: str) -> int:
    result = _strict_integer(value, field=field)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _string_array(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return [_require_string(item, field=f"{field}[{index}]") for index, item in enumerate(value)]


def _reject_symlink_path_components(path: Path, *, label: str) -> None:
    """Reject existing symbolic-link components without resolving them first."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            raise ArtifactIntegrityError(f"{label} contains a symbolic-link component")


def _parse_unified_inventory(
    manifest: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    inventory = _require_mapping(manifest.get("inventory"), field="comparison inventory")
    if set(inventory) != {"algorithm", "root", "excludes", "files"}:
        raise ValueError("Unified comparison inventory has an unexpected schema")
    if (
        inventory.get("algorithm") != "sha256"
        or inventory.get("root") != "."
        or inventory.get("excludes") != ["manifest.json"]
    ):
        raise ValueError("Unified comparison inventory metadata is invalid")
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ValueError("Unified comparison inventory files must be an array")
    parsed: list[tuple[str, str]] = []
    for index, item in enumerate(files):
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise ValueError(f"Invalid unified comparison inventory entry {index}")
        relative = _safe_relative_path(
            item.get("path"), field=f"comparison inventory files[{index}].path"
        )
        if relative == "manifest.json":
            raise ValueError("Unified comparison manifest cannot checksum itself")
        if Path(relative).suffix.lower() in _EXECUTABLE_MODEL_SUFFIXES:
            raise ValueError("Executable model artifacts are forbidden in comparison packs")
        digest = _sha256_string(
            item.get("sha256"), field=f"comparison inventory {relative} sha256"
        )
        assert digest is not None
        parsed.append((relative, digest))
    if parsed != sorted(parsed, key=lambda entry: entry[0]):
        raise ValueError("Unified comparison inventory must be sorted by path")
    if len({relative for relative, _digest in parsed}) != len(parsed):
        raise ValueError("Unified comparison inventory contains duplicate paths")
    if {relative for relative, _digest in parsed} != _UNIFIED_EXPECTED_FILES:
        raise ValueError("Unified comparison inventory does not match the exact pack contract")
    return tuple(parsed)


def _frozen_unified_source_authority(
    configured: Mapping[str, Mapping[str, str]] | None,
) -> Mapping[str, Mapping[str, str]]:
    raw_authority: object = (
        AUTHORITATIVE_SOURCE_ROOTS if configured is None else configured
    )
    if not isinstance(raw_authority, Mapping):
        raise ArtifactIntegrityError(
            "Unified comparison source authority is malformed"
        )
    authority = dict(raw_authority)
    expected_roles = {
        "baseline_training",
        "baseline_vertical_slice",
        "lstm",
    }
    if set(authority) != expected_roles:
        raise ArtifactIntegrityError(
            "Unified comparison source authority is incomplete"
        )
    normalized: dict[str, Mapping[str, str]] = {}
    for role in sorted(expected_roles):
        raw_identity = authority[role]
        if not isinstance(raw_identity, Mapping):
            raise ArtifactIntegrityError(
                "Unified comparison source authority is malformed"
            )
        identity = dict(raw_identity)
        expected_keys = {"run_id", "manifest_sha256"}
        if role == "baseline_vertical_slice":
            expected_keys.add("portable_certificate_sha256")
        if set(identity) != expected_keys:
            raise ArtifactIntegrityError(
                "Unified comparison source authority is malformed"
            )
        normalized_identity: dict[str, str] = {}
        for key, value in identity.items():
            if key == "run_id":
                normalized_identity[key] = _validate_identifier(
                    value, field=f"{role} authority run ID"
                )
            else:
                digest = _sha256_string(
                    value, field=f"{role} authority {key}"
                )
                assert digest is not None
                normalized_identity[key] = digest
        normalized[role] = MappingProxyType(normalized_identity)
    return MappingProxyType(normalized)


def _frozen_unified_manifest_authority(
    configured: Mapping[str, str] | None,
) -> Mapping[str, str]:
    raw_authority: object = (
        AUTHORITATIVE_COMPARISON_MANIFESTS
        if configured is None
        else configured
    )
    if not isinstance(raw_authority, Mapping):
        raise ArtifactIntegrityError(
            "Unified comparison manifest authority is malformed"
        )
    normalized: dict[str, str] = {}
    for comparison_id, value in dict(raw_authority).items():
        safe_id = _validate_identifier(
            comparison_id, field="comparison manifest authority ID"
        )
        digest = _sha256_string(
            value, field=f"{safe_id} authorized manifest SHA-256"
        )
        assert digest is not None
        normalized[safe_id] = digest
    return MappingProxyType(dict(sorted(normalized.items())))


class ArtifactRepository:
    """Validate and project immutable result artifacts into API-safe values."""

    def __init__(self, settings: RepositorySettings):
        source_authority = _frozen_unified_source_authority(
            settings.unified_comparison_source_authority
        )
        manifest_authority = _frozen_unified_manifest_authority(
            settings.unified_comparison_manifest_authority
        )
        self.settings = replace(
            settings,
            unified_comparison_source_authority=source_authority,
            unified_comparison_manifest_authority=manifest_authority,
        )
        self._comparison_source_authority = source_authority
        self._comparison_manifest_authority = manifest_authority
        self._cache: dict[str, ValidatedRun] = {}
        self._comparison_cache: dict[str, ValidatedUnifiedComparison] = {}
        self._lock = threading.RLock()

    @property
    def default_run_id(self) -> str:
        value = self.settings.default_vertical_slice_run_id
        if value is None:
            raise DefaultRunNotConfiguredError(
                "PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID is not configured"
            )
        return _validate_identifier(value, field="default run ID")

    @property
    def default_unified_comparison_id(self) -> str:
        value = self.settings.default_unified_comparison_run_id
        if value is None:
            raise DefaultRunNotConfiguredError(
                "PAISA_DEFAULT_UNIFIED_COMPARISON_RUN_ID is not configured"
            )
        return _validate_identifier(value, field="default comparison run ID")

    @property
    def unified_comparison_root(self) -> Path:
        configured = self.settings.unified_comparison_root
        if configured is not None:
            return configured
        return self.settings.repository_root / "artifacts" / "model_comparisons"

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
        except (OSError, ValueError):
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

    def _inventoried_artifact(self, run: ValidatedRun, relative: str) -> tuple[str, str]:
        safe_relative = _safe_relative_path(relative, field="artifact path")
        inventory = dict(_parse_vertical_inventory(run.manifest))
        expected_digest = inventory.get(safe_relative)
        if expected_digest is None or safe_relative not in run.artifact_paths:
            raise ArtifactIntegrityError("Requested repository artifact is not inventoried")
        return safe_relative, expected_digest

    def _json_artifact(self, run: ValidatedRun, relative: str) -> dict[str, Any]:
        try:
            safe_relative, expected_digest = self._inventoried_artifact(run, relative)
            payload = _open_regular_bytes_beneath(
                run.directory, safe_relative, label=f"artifact {safe_relative}"
            )
            if hashlib.sha256(payload).hexdigest() != expected_digest:
                raise ValueError("JSON bytes do not match the manifest checksum")
            return _decode_json(payload, label=f"artifact {safe_relative}")
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
        expected_sha256: str | None = None,
    ) -> list[dict[str, str]]:
        try:
            safe_relative, inventory_digest = self._inventoried_artifact(run, relative)
            if expected_sha256 is not None and expected_sha256 != inventory_digest:
                raise ValueError("Expected CSV checksum disagrees with the inventory")
            payload = _open_regular_bytes_beneath(
                run.directory, safe_relative, label=f"artifact {safe_relative}"
            )
            if hashlib.sha256(payload).hexdigest() != inventory_digest:
                raise ValueError("CSV bytes do not match the manifest checksum")
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

    def _price_snapshot_artifact(
        self, run: ValidatedRun
    ) -> tuple[str, str, dict[str, Any] | None]:
        data = _require_mapping(run.manifest.get("data"), field="manifest data")
        expected_digest = _sha256_string(
            data.get("input_prices_sha256"), field="input_prices_sha256"
        )
        if expected_digest is None:  # pragma: no cover - non-null contract above
            raise ValueError("input_prices_sha256 is required")

        summary_data = _require_mapping(
            run.summary.get("data_source"), field="summary data_source"
        )
        reference_digest = _sha256_string(
            run.training_reference.get("input_prices_sha256"),
            field="training reference input_prices_sha256",
        )
        summary_digest = _sha256_string(
            summary_data.get("input_prices_sha256"),
            field="summary input_prices_sha256",
        )
        if reference_digest != expected_digest or summary_digest != expected_digest:
            raise ArtifactIntegrityError(
                "Price snapshot checksum references are inconsistent"
            )

        candidates = [
            (relative, digest)
            for relative, digest in _parse_vertical_inventory(run.manifest)
            if len(PurePosixPath(relative).parts) == 2
            and PurePosixPath(relative).parts[0] == "input_snapshot"
            and PurePosixPath(relative).suffix.lower() == ".csv"
            and digest == expected_digest
        ]
        if len(candidates) != 1:
            raise ArtifactIntegrityError(
                "Run manifest does not identify exactly one immutable price snapshot"
            )
        relative, digest = candidates[0]

        pipeline_contract: dict[str, Any] | None = None
        pipeline_relative = "input_snapshot/pipeline_metadata.json"
        is_real_data = _boolean(data.get("is_real_data"), field="is_real_data")
        if pipeline_relative in run.artifact_paths:
            metadata = self._json_artifact(run, pipeline_relative)
            _schema_version(
                metadata.get("schema_version"),
                field="pipeline metadata schema_version",
            )
            pipeline_contract = _require_mapping(
                metadata.get("prices_artifact"), field="prices_artifact"
            )
            _schema_version(
                pipeline_contract.get("schema_version"),
                field="prices_artifact schema_version",
            )
            if pipeline_contract.get("status") != "complete":
                raise ValueError("prices_artifact status must be complete")
            if pipeline_contract.get("frequency") != "daily":
                raise ValueError("prices_artifact frequency must be daily")
            declared_name = _safe_relative_path(
                pipeline_contract.get("path"), field="prices_artifact path"
            )
            if len(PurePosixPath(declared_name).parts) != 1:
                raise ValueError("prices_artifact path must be a filename")
            if f"input_snapshot/{declared_name}" != relative:
                raise ValueError("prices_artifact path does not match the run snapshot")
            if (
                _sha256_string(
                    pipeline_contract.get("sha256"), field="prices_artifact sha256"
                )
                != digest
            ):
                raise ValueError("prices_artifact checksum does not match the run snapshot")
        elif is_real_data:
            raise ArtifactIntegrityError(
                "Real-data run is missing its input snapshot metadata"
            )
        return relative, digest, pipeline_contract

    def load_stock_prices(
        self,
        run_id: str,
        ticker: str,
        *,
        limit: int,
        offset: int,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        """Return validated daily prices owned by one completed vertical-slice run."""

        safe_ticker = _validate_identifier(ticker, field="ticker").upper()
        if start_date is not None and end_date is not None and start_date > end_date:
            raise InvalidDateRangeError("start_date must be on or before end_date")

        run = self._validate_run(run_id)
        relative, expected_digest, pipeline_contract = self._price_snapshot_artifact(run)
        raw_rows = self._csv_artifact(
            run,
            relative,
            required_columns=("date", "ticker", "open", "close", "volume", "source"),
            expected_sha256=expected_digest,
        )

        observed_keys: set[tuple[str, str]] = set()
        normalized_rows: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_rows, start=2):
            raw_ticker = _require_string(raw.get("ticker"), field=f"row {index} ticker")
            normalized_ticker = raw_ticker.upper()
            if not _SAFE_ID.fullmatch(normalized_ticker):
                raise ValueError(f"row {index} ticker is invalid")
            normalized_date = _iso_date(raw.get("date"), field=f"row {index} date")
            key = (normalized_ticker, normalized_date)
            if key in observed_keys:
                raise ValueError("Price snapshot contains duplicate ticker/date rows")
            observed_keys.add(key)

            source = _require_string(raw.get("source"), field=f"row {index} source")
            if source.lower() in {
                "nan",
                "+nan",
                "-nan",
                "inf",
                "+inf",
                "-inf",
                "infinity",
                "+infinity",
                "-infinity",
                "none",
                "null",
            }:
                raise ValueError(f"row {index} source is invalid")
            volume = _integer(raw.get("volume"), field=f"row {index} volume")
            if volume < 0:
                raise ValueError(f"row {index} volume must be non-negative")

            normalized_rows.append(
                {
                    "ticker": normalized_ticker,
                    "date": normalized_date,
                    "open": _positive_decimal(
                        raw.get("open"), field=f"row {index} open"
                    ),
                    "high": _positive_decimal(
                        raw.get("high"), field=f"row {index} high", nullable=True
                    ),
                    "low": _positive_decimal(
                        raw.get("low"), field=f"row {index} low", nullable=True
                    ),
                    "close": _positive_decimal(
                        raw.get("close"), field=f"row {index} close"
                    ),
                    "volume": volume,
                    "source": source,
                }
            )

        if pipeline_contract is not None:
            if _integer(
                pipeline_contract.get("row_count"), field="prices_artifact row_count"
            ) != len(normalized_rows):
                raise ArtifactIntegrityError(
                    "Price snapshot row count does not match its input contract"
                )
            declared_symbols = _require_mapping(
                pipeline_contract.get("symbols"), field="prices_artifact symbols"
            )
            observed_symbols = sorted({row["ticker"] for row in normalized_rows})
            normalized_declared_symbols = sorted(
                str(symbol).strip().upper() for symbol in declared_symbols
            )
            if normalized_declared_symbols != observed_symbols:
                raise ArtifactIntegrityError(
                    "Price snapshot symbols do not match its input contract"
                )
            for symbol in observed_symbols:
                symbol_rows = [
                    row for row in normalized_rows if row["ticker"] == symbol
                ]
                symbol_rows.sort(key=lambda row: row["date"])
                coverage = _require_mapping(
                    declared_symbols.get(symbol), field=f"prices_artifact symbols.{symbol}"
                )
                declared_source = _require_string(
                    coverage.get("source"),
                    field=f"prices_artifact symbols.{symbol}.source",
                )
                if (
                    _integer(
                        coverage.get("row_count"),
                        field=f"prices_artifact symbols.{symbol}.row_count",
                    )
                    != len(symbol_rows)
                    or _iso_date(
                        coverage.get("date_start"),
                        field=f"prices_artifact symbols.{symbol}.date_start",
                    )
                    != symbol_rows[0]["date"]
                    or _iso_date(
                        coverage.get("date_end"),
                        field=f"prices_artifact symbols.{symbol}.date_end",
                    )
                    != symbol_rows[-1]["date"]
                    or any(row["source"] != declared_source for row in symbol_rows)
                ):
                    raise ArtifactIntegrityError(
                        "Price snapshot coverage does not match its input contract"
                    )

        data = _require_mapping(run.manifest.get("data"), field="manifest data")
        declared_symbol = _require_string(
            run.manifest.get("symbol"), field="manifest symbol"
        ).upper()
        declared_rows = [
            row for row in normalized_rows if row["ticker"] == declared_symbol
        ]
        declared_rows.sort(key=lambda row: row["date"])
        expected_range = _require_mapping(
            data.get("actual_ogdc_date_range"), field="actual_ogdc_date_range"
        )
        if (
            not declared_rows
            or len(declared_rows)
            != _integer(data.get("row_count"), field="row_count")
            or declared_rows[0]["date"]
            != _iso_date(expected_range.get("start"), field="data date start")
            or declared_rows[-1]["date"]
            != _iso_date(expected_range.get("end"), field="data date end")
        ):
            raise ArtifactIntegrityError(
                "Price snapshot coverage does not match the run manifest"
            )

        ticker_rows = [row for row in normalized_rows if row["ticker"] == safe_ticker]
        if not ticker_rows:
            raise StockNotFoundError(
                f"Ticker '{safe_ticker}' is not present in vertical-slice run '{run.run_id}'"
            )
        ticker_rows.sort(key=lambda row: row["date"])

        full_date_range = {
            "start": ticker_rows[0]["date"],
            "end": ticker_rows[-1]["date"],
        }
        filtered = ticker_rows
        if start_date is not None:
            start = start_date.isoformat()
            filtered = [row for row in filtered if row["date"] >= start]
        if end_date is not None:
            end = end_date.isoformat()
            filtered = [row for row in filtered if row["date"] <= end]

        context = self._context(run)
        public_rows = [
            {key: value for key, value in row.items() if key != "ticker"}
            for row in filtered[offset : offset + limit]
        ]
        return {
            "schema_version": context["schema_version"],
            "run_id": run.run_id,
            "ticker": safe_ticker,
            "source_mode": context["source_mode"],
            "data_date_range": full_date_range,
            "total": len(filtered),
            "limit": limit,
            "offset": offset,
            "rows": public_rows,
            "warnings": context["warnings"],
            "disclaimer": context["disclaimer"],
        }

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

    # Unified price-model comparison packs are already certified by the offline
    # comparison verifier.  API reads intentionally validate the immutable
    # manifest/checksum/schema boundary without replaying model metrics or
    # portfolio accounting during a GET request.

    def _open_comparison_root_descriptor(self) -> int:
        try:
            return _open_absolute_directory_descriptor(
                self.unified_comparison_root
            )
        except FileNotFoundError as exc:
            raise RunNotFoundError(
                "Configured unified comparison root was not found"
            ) from exc
        except (NotImplementedError, OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                "Configured unified comparison root cannot be traversed securely"
            ) from exc

    @staticmethod
    def _open_comparison_descriptor(
        root_descriptor: int, comparison_id: str
    ) -> int:
        try:
            return _open_child_directory_descriptor(
                root_descriptor, comparison_id
            )
        except FileNotFoundError as exc:
            raise RunNotFoundError(
                f"Unified comparison run '{comparison_id}' was not found"
            ) from exc
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(
                f"Unified comparison run '{comparison_id}' has an unsafe directory"
            ) from exc

    def _unified_source_authority(self) -> Mapping[str, Mapping[str, str]]:
        return self._comparison_source_authority

    @staticmethod
    def _unified_source_inventory(
        value: Any,
        *,
        field: str,
        expected_excludes: tuple[str, ...],
    ) -> dict[str, str]:
        """Parse one authoritative source inventory without touching its files."""

        inventory = _require_mapping(value, field=field)
        if set(inventory) != {"algorithm", "root", "excludes", "files"}:
            raise ValueError(
                f"{field} must contain exactly algorithm, root, excludes, and files"
            )
        if inventory.get("algorithm") != "sha256" or inventory.get("root") != ".":
            raise ValueError(f"{field} must use SHA-256 relative to the artifact root")
        if inventory.get("excludes") != list(expected_excludes):
            raise ValueError(f"{field} has an invalid self-exclusion policy")
        raw_files = inventory.get("files")
        if not isinstance(raw_files, list) or not raw_files:
            raise ValueError(f"{field}.files must be a non-empty array")
        parsed: list[tuple[str, str]] = []
        for index, raw_entry in enumerate(raw_files):
            entry = _require_mapping(
                raw_entry, field=f"{field}.files[{index}]"
            )
            if set(entry) != {"path", "sha256"}:
                raise ValueError(
                    f"{field}.files[{index}] must contain exactly path and sha256"
                )
            relative = _safe_relative_path(
                entry.get("path"), field=f"{field}.files[{index}].path"
            )
            digest = _sha256_string(
                entry.get("sha256"), field=f"{field}.files[{index}].sha256"
            )
            parsed.append((relative, digest))
        if parsed != sorted(parsed) or len({path for path, _ in parsed}) != len(
            parsed
        ):
            raise ValueError(f"{field}.files must be sorted and duplicate-free")
        return dict(parsed)

    def _cached_comparison(
        self, comparison_id: str, directory_descriptor: int
    ) -> ValidatedUnifiedComparison | None:
        cached = self._comparison_cache.get(comparison_id)
        if cached is None:
            return None
        try:
            signature = _descriptor_tree_signature(directory_descriptor)
        except (OSError, ValueError):
            return None
        return cached if signature == cached.signature else None

    @staticmethod
    def _comparison_json(
        payloads: Mapping[str, bytes],
        inventory: Mapping[str, str],
        relative: str,
    ) -> dict[str, Any]:
        expected = inventory.get(relative)
        if expected is None:
            raise ValueError(f"Required comparison artifact is not inventoried: {relative}")
        payload = payloads.get(relative)
        if payload is None:
            raise ValueError(f"Required comparison artifact is missing: {relative}")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Unified comparison checksum mismatch: {relative}")
        return _decode_json(payload, label=f"comparison artifact {relative}")

    @staticmethod
    def _count_map(value: Any, *, field: str) -> dict[str, int]:
        mapping = _require_mapping(value, field=field)
        if set(mapping) != {"0", "1"}:
            raise ValueError(f"{field} must contain exactly classes 0 and 1")
        result = {
            label: _strict_integer(count, field=f"{field}.{label}")
            for label, count in mapping.items()
        }
        if any(
            count < 0 or count > _UNIFIED_MAX_CLASSIFICATION_COUNT
            for count in result.values()
        ):
            raise ValueError(f"{field} counts are outside the supported range")
        return result

    @classmethod
    def _unified_prediction_metric(
        cls,
        value: Any,
        *,
        field: str,
        expected_scope: str,
        expected_type: str,
    ) -> dict[str, Any]:
        row = _require_mapping(value, field=field)
        competitor_id = _require_string(row.get("competitor_id"), field=f"{field}.competitor_id")
        competitor_type = _require_string(
            row.get("competitor_type"), field=f"{field}.competitor_type"
        )
        if competitor_type != expected_type:
            raise ValueError(f"{field}.competitor_type is invalid")
        scope = _require_string(row.get("evaluation_scope"), field=f"{field}.evaluation_scope")
        if scope != expected_scope:
            raise ValueError(f"{field}.evaluation_scope is invalid")

        registered_raw = row.get("registered_model_id")
        algorithm_raw = row.get("algorithm")
        label_raw = row.get("label")
        if expected_type == "model":
            registered_model_id = _require_string(
                registered_raw, field=f"{field}.registered_model_id"
            )
            algorithm = _require_string(algorithm_raw, field=f"{field}.algorithm")
            configuration_family = _require_string(label_raw, field=f"{field}.label")
        else:
            if registered_raw is not None or algorithm_raw is not None or label_raw is not None:
                raise ValueError(f"{field} reference baseline identity is invalid")
            registered_model_id = None
            algorithm = None
            configuration_family = None

        row_count = _strict_integer(row.get("row_count"), field=f"{field}.row_count")
        support = _strict_integer(row.get("support"), field=f"{field}.support")
        if (
            row_count <= 0
            or row_count > _UNIFIED_MAX_CLASSIFICATION_COUNT
            or support != row_count
        ):
            raise ValueError(f"{field} row count and support must agree and be positive")
        class_counts = cls._count_map(row.get("class_counts"), field=f"{field}.class_counts")
        predicted_counts = cls._count_map(
            row.get("predicted_class_counts"), field=f"{field}.predicted_class_counts"
        )
        if sum(class_counts.values()) != row_count or sum(predicted_counts.values()) != row_count:
            raise ValueError(f"{field} class counts do not reconcile")

        raw_matrix = row.get("confusion_matrix")
        if (
            not isinstance(raw_matrix, list)
            or len(raw_matrix) != 2
            or any(
                not isinstance(matrix_row, list) or len(matrix_row) != 2
                for matrix_row in raw_matrix
            )
        ):
            raise ValueError(f"{field}.confusion_matrix must be a 2x2 array")
        confusion_matrix = [
            [
                _strict_integer(
                    cell,
                    field=(
                        f"{field}.confusion_matrix[{row_index}]"
                        f"[{column_index}]"
                    ),
                )
                for column_index, cell in enumerate(matrix_row)
            ]
            for row_index, matrix_row in enumerate(raw_matrix)
        ]
        if any(
            cell < 0 or cell > _UNIFIED_MAX_CLASSIFICATION_COUNT
            for matrix_row in confusion_matrix
            for cell in matrix_row
        ):
            raise ValueError(
                f"{field}.confusion_matrix cells are outside the supported range"
            )
        if sum(cell for matrix_row in confusion_matrix for cell in matrix_row) != row_count:
            raise ValueError(f"{field}.confusion_matrix does not reconcile")
        if [sum(matrix_row) for matrix_row in confusion_matrix] != [
            class_counts["0"],
            class_counts["1"],
        ]:
            raise ValueError(f"{field}.confusion_matrix actual-class counts disagree")
        if [
            confusion_matrix[0][column] + confusion_matrix[1][column]
            for column in range(2)
        ] != [predicted_counts["0"], predicted_counts["1"]]:
            raise ValueError(f"{field}.confusion_matrix predicted-class counts disagree")
        if (
            _strict_integer(row.get("positive_label"), field=f"{field}.positive_label") != 1
            or _require_string(row.get("average"), field=f"{field}.average") != "binary"
            or _strict_integer(row.get("zero_division"), field=f"{field}.zero_division") != 0
        ):
            raise ValueError(f"{field} classification conventions are invalid")

        warnings = _string_array(row.get("metric_warnings"), field=f"{field}.metric_warnings")
        selection_eligible = _boolean(
            row.get("selection_eligible"), field=f"{field}.selection_eligible"
        )
        complete_class_coverage = all(class_counts[label] > 0 for label in ("0", "1"))
        if (
            _boolean(
                row.get("complete_class_coverage"),
                field=f"{field}.complete_class_coverage",
            )
            is not complete_class_coverage
            or row.get("observed_classes")
            != [int(label) for label in ("0", "1") if class_counts[label] > 0]
            or selection_eligible is not complete_class_coverage
            or ("single_class_actuals" in warnings)
            != (not complete_class_coverage)
        ):
            raise ValueError(f"{field} class-coverage semantics are inconsistent")
        normalized = {
            "competitor_id": competitor_id,
            "competitor_type": competitor_type,
            "registered_model_id": registered_model_id,
            "algorithm": algorithm,
            "configuration_family": configuration_family,
            "evaluation_scope": (
                _UNIFIED_HOLDOUT_TERM
                if expected_scope == "shared_project_holdout"
                else "validation"
            ),
            "row_count": row_count,
            "support": support,
            "accuracy": _bounded_float(
                row.get("accuracy"), field=f"{field}.accuracy", minimum=0.0, maximum=1.0
            ),
            "balanced_accuracy": _bounded_float(
                row.get("balanced_accuracy"),
                field=f"{field}.balanced_accuracy",
                minimum=0.0,
                maximum=1.0,
                nullable=True,
            ),
            "precision": _bounded_float(
                row.get("precision"), field=f"{field}.precision", minimum=0.0, maximum=1.0
            ),
            "recall": _bounded_float(
                row.get("recall"), field=f"{field}.recall", minimum=0.0, maximum=1.0
            ),
            "f1": _bounded_float(
                row.get("f1"), field=f"{field}.f1", minimum=0.0, maximum=1.0
            ),
            "matthews_corrcoef": _bounded_float(
                row.get("matthews_corrcoef"),
                field=f"{field}.matthews_corrcoef",
                minimum=-1.0,
                maximum=1.0,
                nullable=True,
            ),
            "roc_auc": _bounded_float(
                row.get("roc_auc"),
                field=f"{field}.roc_auc",
                minimum=0.0,
                maximum=1.0,
                nullable=True,
            ),
            "brier_score": _bounded_float(
                row.get("brier_score"),
                field=f"{field}.brier_score",
                minimum=0.0,
                maximum=1.0,
            ),
            "log_loss": _bounded_float(
                row.get("log_loss"),
                field=f"{field}.log_loss",
                minimum=0.0,
                nullable=True,
            ),
            "expected_calibration_error": _bounded_float(
                row.get("expected_calibration_error"),
                field=f"{field}.expected_calibration_error",
                minimum=0.0,
                maximum=1.0,
            ),
            "selection_eligible": selection_eligible,
            "metric_warnings": warnings,
            "class_counts": class_counts,
            "predicted_class_counts": predicted_counts,
            "confusion_matrix": confusion_matrix,
        }
        validate_confusion_derived_metrics(
            normalized,
            confusion_matrix,
            field=field,
        )
        if complete_class_coverage:
            if normalized["roc_auc"] is None or normalized["log_loss"] is None:
                raise ValueError(
                    f"{field} ROC AUC and log loss must be defined with both actual classes"
                )
        elif normalized["roc_auc"] is not None or normalized["log_loss"] is not None:
            raise ValueError(
                f"{field} ROC AUC and log loss must be null for single-class actuals"
            )
        return normalized

    @staticmethod
    def _unified_model_portfolio(value: Any, *, field: str) -> dict[str, Any]:
        row = _require_mapping(value, field=field)
        if row.get("competitor_type") != "model":
            raise ValueError(f"{field}.competitor_type must equal model")
        return {
            "variant_id": _require_string(row.get("competitor_id"), field=f"{field}.competitor_id"),
            "registered_model_id": _require_string(
                row.get("registered_model_id"), field=f"{field}.registered_model_id"
            ),
            "experiment_id": _require_string(row.get("experiment_id"), field=f"{field}.experiment_id"),
            "portfolio_id": _require_string(row.get("portfolio_id"), field=f"{field}.portfolio_id"),
            "initial_equity": _money(row.get("initial_equity"), field=f"{field}.initial_equity"),
            "ending_equity": _money(row.get("ending_equity"), field=f"{field}.ending_equity"),
            "total_return": _strict_float(row.get("total_return"), field=f"{field}.total_return"),
            "maximum_drawdown": _bounded_float(
                row.get("maximum_drawdown"),
                field=f"{field}.maximum_drawdown",
                minimum=0.0,
                maximum=1.0,
            ),
            "volatility": _bounded_float(
                row.get("volatility"),
                field=f"{field}.volatility",
                minimum=0.0,
                nullable=True,
            ),
            "sharpe_ratio": _strict_float(
                row.get("sharpe_ratio"), field=f"{field}.sharpe_ratio", nullable=True
            ),
            "number_of_trades": _nonnegative_integer(
                row.get("number_of_trades"), field=f"{field}.number_of_trades"
            ),
            "win_rate": _bounded_float(
                row.get("win_rate"),
                field=f"{field}.win_rate",
                minimum=0.0,
                maximum=1.0,
                nullable=True,
            ),
            "average_gain": _money(row.get("average_gain"), field=f"{field}.average_gain", nullable=True),
            "average_loss": _money(row.get("average_loss"), field=f"{field}.average_loss", nullable=True),
            "turnover": _bounded_float(
                row.get("turnover"), field=f"{field}.turnover", minimum=0.0
            ),
            "fees_paid": _nonnegative_money(
                row.get("fees_paid"), field=f"{field}.fees_paid"
            ),
            "time_in_market": _bounded_float(
                row.get("time_in_market"),
                field=f"{field}.time_in_market",
                minimum=0.0,
                maximum=1.0,
            ),
            "excess_return_vs_cash_only": _strict_float(
                row.get("excess_return_vs_cash_only"),
                field=f"{field}.excess_return_vs_cash_only",
            ),
            "excess_return_vs_buy_and_hold": _strict_float(
                row.get("excess_return_vs_buy_and_hold"),
                field=f"{field}.excess_return_vs_buy_and_hold",
            ),
        }

    @staticmethod
    def _unified_benchmark(
        value: Any,
        *,
        field: str,
        model_ids: Mapping[str, str],
        experiment_ids: Mapping[str, str],
    ) -> dict[str, Any]:
        row = _require_mapping(value, field=field)
        if row.get("competitor_type") != "portfolio_benchmark":
            raise ValueError(f"{field}.competitor_type must equal portfolio_benchmark")
        provenance = row.get("ledger_provenance")
        if not isinstance(provenance, list) or len(provenance) != len(_UNIFIED_MODEL_VARIANTS):
            raise ValueError(f"{field}.ledger_provenance must contain six ledgers")
        variants: list[str] = []
        for index, entry_raw in enumerate(provenance):
            entry = _require_mapping(entry_raw, field=f"{field}.ledger_provenance[{index}]")
            if set(entry) != {
                "variant_id",
                "registered_model_id",
                "experiment_id",
                "portfolio_id",
                "config_path",
                "snapshots_path",
                "trades_path",
            }:
                raise ValueError(
                    f"{field}.ledger_provenance[{index}] has an unexpected schema"
                )
            variant = _require_string(
                entry.get("variant_id"), field=f"{field}.ledger_provenance[{index}].variant_id"
            )
            variants.append(variant)
            registered_model_id = _require_string(
                entry.get("registered_model_id"),
                field=f"{field}.ledger_provenance[{index}].registered_model_id",
            )
            if registered_model_id != model_ids.get(variant):
                raise ValueError(
                    f"{field}.ledger_provenance registered model identity mismatch"
                )
            experiment_id = _require_string(
                entry.get("experiment_id"),
                field=f"{field}.ledger_provenance[{index}].experiment_id",
            )
            if experiment_id != experiment_ids.get(variant):
                raise ValueError(
                    f"{field}.ledger_provenance experiment identity mismatch"
                )
            _require_string(
                entry.get("portfolio_id"),
                field=f"{field}.ledger_provenance[{index}].portfolio_id",
            )
            expected_prefix = f"evidence/portfolios/{variant}"
            config_path = _safe_relative_path(
                entry.get("config_path"),
                field=f"{field}.ledger_provenance[{index}].config_path",
            )
            if config_path != f"{expected_prefix}/config.json":
                raise ValueError(f"{field}.ledger_provenance config path is invalid")
            expected_snapshot = (
                "buy_and_hold_snapshots.csv"
                if row.get("benchmark_id") == "buy-and-hold"
                else "cash_only_snapshots.csv"
            )
            snapshots_path = _safe_relative_path(
                entry.get("snapshots_path"),
                field=f"{field}.ledger_provenance[{index}].snapshots_path",
            )
            if snapshots_path != f"{expected_prefix}/benchmarks/{expected_snapshot}":
                raise ValueError(f"{field}.ledger_provenance snapshot path is invalid")
            trades_path = entry.get("trades_path")
            if row.get("benchmark_id") == "buy-and-hold":
                normalized_trades_path = _safe_relative_path(
                    trades_path,
                    field=f"{field}.ledger_provenance[{index}].trades_path",
                )
                if normalized_trades_path != (
                    f"{expected_prefix}/benchmarks/buy_and_hold_trades.csv"
                ):
                    raise ValueError(f"{field}.ledger_provenance trades path is invalid")
            elif trades_path is not None:
                raise ValueError(f"{field}.cash-only ledger must not reference trades")
        if tuple(variants) != _UNIFIED_MODEL_VARIANTS:
            raise ValueError(f"{field}.ledger_provenance model order is invalid")
        return {
            "benchmark_id": _require_string(row.get("benchmark_id"), field=f"{field}.benchmark_id"),
            "initial_equity": _money(row.get("initial_equity"), field=f"{field}.initial_equity"),
            "ending_equity": _money(row.get("ending_equity"), field=f"{field}.ending_equity"),
            "total_return": _strict_float(row.get("total_return"), field=f"{field}.total_return"),
            "maximum_drawdown": _bounded_float(
                row.get("maximum_drawdown"),
                field=f"{field}.maximum_drawdown",
                minimum=0.0,
                maximum=1.0,
            ),
            "volatility": _bounded_float(
                row.get("volatility"),
                field=f"{field}.volatility",
                minimum=0.0,
                nullable=True,
            ),
            "sharpe_ratio": _strict_float(
                row.get("sharpe_ratio"), field=f"{field}.sharpe_ratio", nullable=True
            ),
            "number_of_trades": _nonnegative_integer(
                row.get("number_of_trades"), field=f"{field}.number_of_trades"
            ),
            "win_rate": _bounded_float(
                row.get("win_rate"),
                field=f"{field}.win_rate",
                minimum=0.0,
                maximum=1.0,
                nullable=True,
            ),
            "average_gain": _money(row.get("average_gain"), field=f"{field}.average_gain", nullable=True),
            "average_loss": _money(row.get("average_loss"), field=f"{field}.average_loss", nullable=True),
            "turnover": _bounded_float(
                row.get("turnover"), field=f"{field}.turnover", minimum=0.0
            ),
            "fees_paid": _nonnegative_money(
                row.get("fees_paid"), field=f"{field}.fees_paid"
            ),
            "time_in_market": _bounded_float(
                row.get("time_in_market"),
                field=f"{field}.time_in_market",
                minimum=0.0,
                maximum=1.0,
            ),
            "ledger_count": len(provenance),
        }

    @staticmethod
    def _unified_date_scope(
        value: Any, *, field: str, expected_term: str
    ) -> dict[str, Any]:
        scope = _require_mapping(value, field=field)
        terminology = _require_string(scope.get("terminology"), field=f"{field}.terminology")
        if terminology != expected_term:
            raise ValueError(f"{field}.terminology must equal {expected_term}")
        common_count = _strict_integer(scope.get("common_count"), field=f"{field}.common_count")
        if common_count <= 0:
            raise ValueError(f"{field}.common_count must be positive")
        original_counts = _require_mapping(scope.get("original_counts"), field=f"{field}.original_counts")
        common_counts = _require_mapping(scope.get("common_counts"), field=f"{field}.common_counts")
        dropped = _require_mapping(
            scope.get("dropped_target_dates"), field=f"{field}.dropped_target_dates"
        )
        for mapping_name, mapping in (
            ("original_counts", original_counts),
            ("common_counts", common_counts),
            ("dropped_target_dates", dropped),
        ):
            if set(mapping) != set(_UNIFIED_MODEL_VARIANTS):
                raise ValueError(f"{field}.{mapping_name} must identify all six models")
        normalized_original: dict[str, int] = {}
        normalized_common: dict[str, int] = {}
        normalized_dropped: dict[str, list[str]] = {}
        for variant in _UNIFIED_MODEL_VARIANTS:
            normalized_original[variant] = _strict_integer(
                original_counts[variant], field=f"{field}.original_counts.{variant}"
            )
            normalized_common[variant] = _strict_integer(
                common_counts[variant], field=f"{field}.common_counts.{variant}"
            )
            normalized_dropped[variant] = [
                _iso_date(item, field=f"{field}.dropped_target_dates.{variant}")
                for item in _string_array(
                    dropped[variant], field=f"{field}.dropped_target_dates.{variant}"
                )
            ]
            if normalized_common[variant] != common_count:
                raise ValueError(f"{field}.common_counts must equal common_count")
            if normalized_original[variant] != (
                normalized_common[variant] + len(normalized_dropped[variant])
            ):
                raise ValueError(
                    f"{field}.original_counts do not reconcile with dropped dates"
                )
            if len(set(normalized_dropped[variant])) != len(
                normalized_dropped[variant]
            ):
                raise ValueError(f"{field}.dropped_target_dates contains duplicates")

        keys = scope.get("common_keys")
        if not isinstance(keys, list) or len(keys) != common_count:
            raise ValueError(f"{field}.common_keys must contain common_count rows")
        normalized_keys: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for index, raw_key in enumerate(keys):
            key = _require_mapping(raw_key, field=f"{field}.common_keys[{index}]")
            ticker = _require_string(key.get("ticker"), field=f"{field}.common_keys[{index}].ticker")
            feature_date = _iso_date(
                key.get("feature_date"), field=f"{field}.common_keys[{index}].feature_date"
            )
            target_date = _iso_date(
                key.get("target_date"), field=f"{field}.common_keys[{index}].target_date"
            )
            if target_date <= feature_date:
                raise ValueError(f"{field}.common_keys target_date must follow feature_date")
            actual_class = _strict_integer(
                key.get("actual_class"), field=f"{field}.common_keys[{index}].actual_class"
            )
            if actual_class not in {0, 1}:
                raise ValueError(f"{field}.common_keys actual_class must be 0 or 1")
            identity = (ticker, feature_date, target_date)
            if identity in seen:
                raise ValueError(f"{field}.common_keys contains duplicates")
            seen.add(identity)
            normalized_keys.append(
                {
                    "ticker": ticker,
                    "feature_date": feature_date,
                    "target_date": target_date,
                    "actual_class": actual_class,
                }
            )
        if normalized_keys != sorted(normalized_keys, key=lambda row: (row["target_date"], row["ticker"])):
            raise ValueError(f"{field}.common_keys must be chronological")

        feature_start = _iso_date(
            scope.get("feature_start"), field=f"{field}.feature_start"
        )
        feature_end = _iso_date(
            scope.get("feature_end"), field=f"{field}.feature_end"
        )
        target_start = _iso_date(
            scope.get("target_start"), field=f"{field}.target_start"
        )
        target_end = _iso_date(scope.get("target_end"), field=f"{field}.target_end")
        if (
            feature_start != min(row["feature_date"] for row in normalized_keys)
            or feature_end != max(row["feature_date"] for row in normalized_keys)
            or target_start != min(row["target_date"] for row in normalized_keys)
            or target_end != max(row["target_date"] for row in normalized_keys)
        ):
            raise ValueError(f"{field} date boundaries disagree with common_keys")

        result: dict[str, Any] = {
            "terminology": terminology,
            "common_count": common_count,
            "feature_start": feature_start,
            "feature_end": feature_end,
            "target_start": target_start,
            "target_end": target_end,
            "original_counts": normalized_original,
            "common_counts": normalized_common,
            "dropped_target_dates": normalized_dropped,
            "common_keys": normalized_keys,
        }
        exposed = scope.get("previously_exposed_through")
        if expected_term == _UNIFIED_HOLDOUT_TERM:
            exposed_models = _string_array(
                exposed, field=f"{field}.previously_exposed_through"
            )
            if exposed_models != list(_UNIFIED_MODEL_VARIANTS[:4]):
                raise ValueError(
                    f"{field}.previously_exposed_through must identify M-001 through M-004"
                )
            result["previously_exposed_through"] = exposed_models
        elif exposed is not None:
            raise ValueError(f"{field}.previously_exposed_through is unexpected")
        return result

    @staticmethod
    def _prediction_ranking_rows(
        value: Any,
        *,
        field: str,
        expected_competitors: tuple[str, ...],
        model_ids: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) != len(expected_competitors):
            raise ValueError(f"{field} has an invalid competitor count")
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(value):
            row = _require_mapping(raw, field=f"{field}[{index}]")
            if set(row) != {
                "rank",
                "competitor_id",
                "registered_model_id",
                "accuracy",
                "balanced_accuracy",
                "f1",
            }:
                raise ValueError(f"{field}[{index}] has an unexpected schema")
            competitor = _require_string(
                row.get("competitor_id"), field=f"{field}[{index}].competitor_id"
            )
            registered = _require_string(
                row.get("registered_model_id"),
                field=f"{field}[{index}].registered_model_id",
            )
            rank = _strict_integer(row.get("rank"), field=f"{field}[{index}].rank")
            if registered != model_ids.get(competitor):
                raise ValueError(f"{field} registered model identity mismatch")
            rows.append(
                {
                    "rank": rank,
                    "competitor_id": competitor,
                    "registered_model_id": registered,
                    "accuracy": _bounded_float(
                        row.get("accuracy"),
                        field=f"{field}[{index}].accuracy",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                    "balanced_accuracy": _bounded_float(
                        row.get("balanced_accuracy"),
                        field=f"{field}[{index}].balanced_accuracy",
                        minimum=0.0,
                        maximum=1.0,
                        nullable=True,
                    ),
                    "f1": _bounded_float(
                        row.get("f1"),
                        field=f"{field}[{index}].f1",
                        minimum=0.0,
                        maximum=1.0,
                    ),
                }
            )
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"{field} ranks must be consecutive")
        if set(row["competitor_id"] for row in rows) != set(expected_competitors):
            raise ValueError(f"{field} competitors are incomplete")
        return rows

    @staticmethod
    def _portfolio_ranking_rows(
        value: Any,
        *,
        field: str,
        model_ids: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) != len(_UNIFIED_MODEL_VARIANTS):
            raise ValueError(f"{field} must contain six model rows")
        rows: list[dict[str, Any]] = []
        if field.endswith("simulated_model_portfolio_return"):
            expected_keys = {
                "rank",
                "competitor_id",
                "registered_model_id",
                "total_return",
                "sharpe_ratio",
                "maximum_drawdown",
            }
        elif field.endswith("simulated_model_portfolio_drawdown"):
            expected_keys = {
                "rank",
                "competitor_id",
                "registered_model_id",
                "maximum_drawdown",
                "total_return",
            }
        elif field.endswith("simulated_model_portfolio_fees"):
            expected_keys = {
                "rank",
                "competitor_id",
                "registered_model_id",
                "fees_paid",
                "number_of_trades",
            }
        else:
            raise ValueError(f"{field} is not a supported portfolio ranking")
        for index, raw in enumerate(value):
            row = _require_mapping(raw, field=f"{field}[{index}]")
            if set(row) != expected_keys:
                raise ValueError(f"{field}[{index}] has an unexpected schema")
            competitor = _require_string(
                row.get("competitor_id"), field=f"{field}[{index}].competitor_id"
            )
            registered = _require_string(
                row.get("registered_model_id"),
                field=f"{field}[{index}].registered_model_id",
            )
            if registered != model_ids.get(competitor):
                raise ValueError(f"{field} registered model identity mismatch")
            result: dict[str, Any] = {
                "rank": _strict_integer(row.get("rank"), field=f"{field}[{index}].rank"),
                "competitor_id": competitor,
                "registered_model_id": registered,
                "total_return": None,
                "maximum_drawdown": None,
                "sharpe_ratio": None,
                "fees_paid": None,
                "number_of_trades": None,
            }
            for metric in ("total_return", "maximum_drawdown", "sharpe_ratio"):
                if metric in row:
                    if metric == "maximum_drawdown":
                        result[metric] = _bounded_float(
                            row.get(metric),
                            field=f"{field}[{index}].{metric}",
                            minimum=0.0,
                            maximum=1.0,
                            nullable=True,
                        )
                    else:
                        result[metric] = _strict_float(
                            row.get(metric),
                            field=f"{field}[{index}].{metric}",
                            nullable=True,
                        )
            if "fees_paid" in row:
                result["fees_paid"] = _nonnegative_money(
                    row.get("fees_paid"), field=f"{field}[{index}].fees_paid"
                )
            if "number_of_trades" in row:
                result["number_of_trades"] = _nonnegative_integer(
                    row.get("number_of_trades"),
                    field=f"{field}[{index}].number_of_trades",
                )
            rows.append(result)
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"{field} ranks must be consecutive")
        if set(row["competitor_id"] for row in rows) != set(_UNIFIED_MODEL_VARIANTS):
            raise ValueError(f"{field} competitors are incomplete")
        return rows

    @classmethod
    def _unified_metric_collections(
        cls, metrics: Mapping[str, Any]
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, str],
    ]:
        expected_keys = {
            "schema_version",
            "comparison_id",
            "validation_model_metrics",
            "validation_reference_baseline_metrics",
            "shared_project_holdout_prediction_metrics",
            "model_portfolio_metrics",
            "benchmark_portfolio_metrics",
            "reference_baseline_training",
        }
        if set(metrics) != expected_keys:
            raise ValueError("metrics.json has an unexpected schema")

        def metric_rows(
            key: str,
            *,
            scope: str,
            expected_type: str,
            expected_count: int,
        ) -> list[dict[str, Any]]:
            raw = metrics.get(key)
            if not isinstance(raw, list) or len(raw) != expected_count:
                raise ValueError(f"metrics.{key} has an invalid row count")
            return [
                cls._unified_prediction_metric(
                    row,
                    field=f"metrics.{key}[{index}]",
                    expected_scope=scope,
                    expected_type=expected_type,
                )
                for index, row in enumerate(raw)
            ]

        validation_models = metric_rows(
            "validation_model_metrics",
            scope="validation",
            expected_type="model",
            expected_count=6,
        )
        validation_baselines = metric_rows(
            "validation_reference_baseline_metrics",
            scope="validation",
            expected_type="reference_baseline",
            expected_count=2,
        )
        holdout_raw = metrics.get("shared_project_holdout_prediction_metrics")
        if not isinstance(holdout_raw, list) or len(holdout_raw) != 8:
            raise ValueError("Shared project holdout metrics must contain eight rows")
        holdout_models: list[dict[str, Any]] = []
        holdout_baselines: list[dict[str, Any]] = []
        for index, row_raw in enumerate(holdout_raw):
            row = _require_mapping(
                row_raw, field=f"metrics.shared_project_holdout_prediction_metrics[{index}]"
            )
            competitor_type = row.get("competitor_type")
            if competitor_type == "model":
                holdout_models.append(
                    cls._unified_prediction_metric(
                        row,
                        field=f"metrics.shared_project_holdout_prediction_metrics[{index}]",
                        expected_scope="shared_project_holdout",
                        expected_type="model",
                    )
                )
            elif competitor_type == "reference_baseline":
                holdout_baselines.append(
                    cls._unified_prediction_metric(
                        row,
                        field=f"metrics.shared_project_holdout_prediction_metrics[{index}]",
                        expected_scope="shared_project_holdout",
                        expected_type="reference_baseline",
                    )
                )
            else:
                raise ValueError("Unknown shared project holdout competitor type")

        expected_models = set(_UNIFIED_MODEL_VARIANTS)
        for label, rows, expected in (
            ("validation model", validation_models, _UNIFIED_MODEL_VARIANTS),
            ("holdout model", holdout_models, _UNIFIED_MODEL_VARIANTS),
            (
                "validation baseline",
                validation_baselines,
                _UNIFIED_REFERENCE_BASELINES,
            ),
            ("holdout baseline", holdout_baselines, _UNIFIED_REFERENCE_BASELINES),
        ):
            observed = tuple(row["competitor_id"] for row in rows)
            if observed != expected:
                raise ValueError(
                    f"{label} metrics identities are incomplete, duplicated, or out of order"
                )

        model_ids = {
            row["competitor_id"]: str(row["registered_model_id"])
            for row in validation_models
        }
        if len(set(model_ids.values())) != len(_UNIFIED_MODEL_VARIANTS):
            raise ValueError("Each model variant must have a unique registered model ID")
        for row in holdout_models:
            variant = row["competitor_id"]
            if row["registered_model_id"] != model_ids[variant]:
                raise ValueError("Validation and holdout registered model IDs disagree")
            validation_row = next(
                candidate for candidate in validation_models if candidate["competitor_id"] == variant
            )
            if (
                row["algorithm"] != validation_row["algorithm"]
                or row["configuration_family"] != validation_row["configuration_family"]
            ):
                raise ValueError("Validation and holdout model descriptions disagree")

        raw_portfolios = metrics.get("model_portfolio_metrics")
        if not isinstance(raw_portfolios, list) or len(raw_portfolios) != 6:
            raise ValueError("Model portfolio metrics must contain six rows")
        model_portfolios = [
            cls._unified_model_portfolio(
                row, field=f"metrics.model_portfolio_metrics[{index}]"
            )
            for index, row in enumerate(raw_portfolios)
        ]
        if tuple(row["variant_id"] for row in model_portfolios) != _UNIFIED_MODEL_VARIANTS:
            raise ValueError(
                "Model portfolio identities must contain M-001 through M-006 in order"
            )
        if len({row["portfolio_id"] for row in model_portfolios}) != 6:
            raise ValueError("Model portfolio IDs must be unique")
        for row in model_portfolios:
            if row["registered_model_id"] != model_ids[row["variant_id"]]:
                raise ValueError("Portfolio registered model identity mismatch")
        experiment_ids = {
            row["variant_id"]: row["experiment_id"] for row in model_portfolios
        }

        raw_benchmarks = metrics.get("benchmark_portfolio_metrics")
        if not isinstance(raw_benchmarks, list) or len(raw_benchmarks) != 2:
            raise ValueError("Benchmark portfolio metrics must contain two rows")
        benchmarks = [
            cls._unified_benchmark(
                row,
                field=f"metrics.benchmark_portfolio_metrics[{index}]",
                model_ids=model_ids,
                experiment_ids=experiment_ids,
            )
            for index, row in enumerate(raw_benchmarks)
        ]
        if tuple(row["benchmark_id"] for row in benchmarks) != _UNIFIED_BENCHMARKS:
            raise ValueError("Benchmark rows must be cash-only then buy-and-hold")

        training = _require_mapping(
            metrics.get("reference_baseline_training"),
            field="metrics.reference_baseline_training",
        )
        if set(training) != {"training_row_count", "class_counts", "majority_class"}:
            raise ValueError("Reference baseline training evidence is malformed")
        training_count = _strict_integer(
            training.get("training_row_count"), field="reference_baseline_training.training_row_count"
        )
        training_classes = cls._count_map(
            training.get("class_counts"), field="reference_baseline_training.class_counts"
        )
        majority_class = _strict_integer(
            training.get("majority_class"), field="reference_baseline_training.majority_class"
        )
        if training_count <= 0 or sum(training_classes.values()) != training_count:
            raise ValueError("Reference baseline training counts do not reconcile")
        if majority_class not in {0, 1}:
            raise ValueError("Reference baseline majority class is invalid")

        return (
            validation_models,
            validation_baselines,
            holdout_models,
            holdout_baselines,
            model_portfolios,
            benchmarks,
            model_ids,
        )

    def _validate_unified_comparison(
        self,
        comparison_id: str,
        *,
        root_descriptor: int | None = None,
    ) -> ValidatedUnifiedComparison:
        safe_id = _validate_identifier(comparison_id, field="comparison_id")
        owned_root: int | None = None
        if root_descriptor is None:
            owned_root = self._open_comparison_root_descriptor()
            root_descriptor = owned_root
        try:
            directory_descriptor = self._open_comparison_descriptor(
                root_descriptor, safe_id
            )
        except BaseException:
            if owned_root is not None:
                os.close(owned_root)
            raise
        try:
            return self._validate_unified_comparison_descriptor(
                safe_id, directory_descriptor
            )
        finally:
            os.close(directory_descriptor)
            if owned_root is not None:
                os.close(owned_root)

    def _validate_unified_comparison_descriptor(
        self, safe_id: str, directory_descriptor: int
    ) -> ValidatedUnifiedComparison:
        with self._lock:
            cached = self._cached_comparison(safe_id, directory_descriptor)
            if cached is not None:
                return cached
            captured: _PinnedArtifactTree | None = None
            try:
                captured = _capture_pinned_artifact_tree(directory_descriptor)
                signature_before = captured.signature
                if "manifest.json" not in captured.descriptors:
                    raise RunNotCompletedError(
                        f"Unified comparison run '{safe_id}' has no completed manifest"
                    )
                manifest_payload = captured.read(
                    "manifest.json",
                    label="unified comparison manifest",
                    max_bytes=_UNIFIED_MAX_PAYLOAD_BYTES,
                )
                expected_manifest_digest = (
                    self._comparison_manifest_authority.get(safe_id)
                )
                observed_manifest_digest = hashlib.sha256(
                    manifest_payload
                ).hexdigest()
                if (
                    expected_manifest_digest is None
                    or observed_manifest_digest != expected_manifest_digest
                ):
                    raise ValueError(
                        "Unified comparison manifest is not authorized"
                    )
                manifest = _decode_json(
                    manifest_payload, label="unified comparison manifest"
                )
                if set(manifest) != {
                    "schema_version",
                    "artifact_type",
                    "comparison_id",
                    "status",
                    "inventory",
                    "source_roots",
                    "final_summary",
                    "disclaimer",
                }:
                    raise ValueError("Unified comparison manifest has an unexpected schema")
                if manifest.get("status") != "PASS":
                    raise RunNotCompletedError(
                        f"Unified comparison run '{safe_id}' is not a completed PASS run"
                    )
                if (
                    manifest.get("schema_version") != SCHEMA_VERSION
                    or manifest.get("artifact_type") != _UNIFIED_ARTIFACT_TYPE
                    or manifest.get("comparison_id") != safe_id
                    or manifest.get("source_roots") != "source_roots.json"
                    or manifest.get("final_summary") != "final_summary.json"
                    or manifest.get("disclaimer") != ACADEMIC_DISCLAIMER
                ):
                    raise ValueError("Unified comparison manifest identity is invalid")

                parsed_inventory = _parse_unified_inventory(manifest)
                inventory = dict(parsed_inventory)
                actual = set(captured.descriptors) - {"manifest.json"}
                if actual != set(inventory):
                    raise ValueError("Unified comparison artifact set does not match its manifest")
                payload_bytes = len(manifest_payload)
                payloads: dict[str, bytes] = {}
                for relative, expected_digest in parsed_inventory:
                    remaining_bytes = _UNIFIED_MAX_PAYLOAD_BYTES - payload_bytes
                    if remaining_bytes < 0:
                        raise ValueError("Unified comparison pack exceeds the size limit")
                    payload = captured.read(
                        relative,
                        label=f"comparison artifact {relative}",
                        max_bytes=remaining_bytes,
                    )
                    payloads[relative] = payload
                    payload_bytes += len(payload)
                    if payload_bytes > _UNIFIED_MAX_PAYLOAD_BYTES:
                        raise ValueError("Unified comparison pack exceeds the size limit")
                    if hashlib.sha256(payload).hexdigest() != expected_digest:
                        raise ValueError(f"Unified comparison checksum mismatch: {relative}")
                    try:
                        payload.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise ValueError(
                            f"Unified comparison artifact is not UTF-8 text: {relative}"
                        ) from exc

                summary = self._comparison_json(payloads, inventory, "final_summary.json")
                contract = self._comparison_json(payloads, inventory, "comparison_contract.json")
                sources = self._comparison_json(payloads, inventory, "source_roots.json")
                date_alignment = self._comparison_json(payloads, inventory, "date_alignment.json")
                metrics = self._comparison_json(payloads, inventory, "metrics.json")
                rankings = self._comparison_json(payloads, inventory, "rankings.json")

                for name, payload in (
                    ("final_summary.json", summary),
                    ("comparison_contract.json", contract),
                    ("source_roots.json", sources),
                    ("date_alignment.json", date_alignment),
                    ("metrics.json", metrics),
                    ("rankings.json", rankings),
                ):
                    if payload.get("schema_version") != SCHEMA_VERSION:
                        raise ValueError(f"{name} schema_version is unsupported")
                    if payload.get("comparison_id") != safe_id:
                        raise ValueError(f"{name} comparison identity mismatch")

                if set(contract) != {
                    "schema_version",
                    "comparison_id",
                    "symbol",
                    "source_frequency",
                    "models",
                    "reference_baselines",
                    "portfolio_benchmarks",
                    "target_definition",
                    "alignment_key",
                    "evaluation_scopes",
                    "selection_policy",
                    "reference_baseline_policy",
                    "portfolio_assumptions",
                    "metric_recomputation",
                    "presentation",
                    "disclaimer",
                }:
                    raise ValueError("comparison_contract.json has an unexpected schema")
                if contract.get("models") != list(_UNIFIED_MODEL_VARIANTS):
                    raise ValueError("Comparison contract must identify M-001 through M-006 once")
                if contract.get("reference_baselines") != list(_UNIFIED_REFERENCE_BASELINES):
                    raise ValueError("Comparison reference baseline contract is invalid")
                if contract.get("portfolio_benchmarks") != list(_UNIFIED_BENCHMARKS):
                    raise ValueError("Comparison portfolio benchmark contract is invalid")
                if (
                    contract.get("symbol") != "OGDC"
                    or contract.get("source_frequency") != "daily"
                    or contract.get("alignment_key")
                    != ["ticker", "feature_date", "target_date"]
                    or contract.get("disclaimer") != ACADEMIC_DISCLAIMER
                ):
                    raise ValueError("Comparison contract disclaimer is invalid")
                target_definition = _require_mapping(
                    contract.get("target_definition"), field="target_definition"
                )
                if target_definition != {
                    "name": "target_next_day_up",
                    "positive_class_condition": (
                        "close[target_date] > close[feature_date]"
                    ),
                    "prediction_horizon": {
                        "value": 1,
                        "unit": "trading_session",
                    },
                }:
                    raise ValueError("Comparison target definition is invalid")
                presentation = _require_mapping(contract.get("presentation"), field="presentation")
                if presentation != {"format": "tabular_report_only", "plots_included": False}:
                    raise ValueError("Comparison presentation contract is invalid")
                selection = _require_mapping(contract.get("selection_policy"), field="selection_policy")
                if selection != {
                    "primary": "balanced_accuracy",
                    "tie_breakers": ["f1", "accuracy", "model_id_ascending"],
                    "test_metrics_used_for_selection": False,
                }:
                    raise ValueError("Shared project holdout metrics cannot select a model")

                if set(date_alignment) != {
                    "schema_version",
                    "comparison_id",
                    "alignment_key",
                    "validation",
                    "shared_project_holdout",
                }:
                    raise ValueError("date_alignment.json has an unexpected schema")
                if date_alignment.get("alignment_key") != ["ticker", "feature_date", "target_date"]:
                    raise ValueError("Date alignment key is invalid")
                validation_scope = self._unified_date_scope(
                    date_alignment.get("validation"),
                    field="date_alignment.validation",
                    expected_term="validation",
                )
                holdout_scope = self._unified_date_scope(
                    date_alignment.get("shared_project_holdout"),
                    field="date_alignment.shared_project_holdout",
                    expected_term=_UNIFIED_HOLDOUT_TERM,
                )
                validation_target_dates = {
                    row["target_date"] for row in validation_scope["common_keys"]
                }
                holdout_target_dates = {
                    row["target_date"] for row in holdout_scope["common_keys"]
                }
                if (
                    validation_target_dates & holdout_target_dates
                    or validation_scope["feature_end"]
                    >= holdout_scope["feature_start"]
                    or validation_scope["target_end"]
                    >= holdout_scope["target_start"]
                ):
                    raise ValueError(
                        "Validation must end before the disjoint shared project holdout"
                    )
                if any(
                    key["ticker"] != contract["symbol"]
                    for scope in (validation_scope, holdout_scope)
                    for key in scope["common_keys"]
                ):
                    raise ValueError("Date alignment contains an unexpected symbol")

                evaluation_scopes = _require_mapping(
                    contract.get("evaluation_scopes"), field="evaluation_scopes"
                )
                if set(evaluation_scopes) != {
                    "validation",
                    "shared_project_holdout",
                }:
                    raise ValueError("Comparison evaluation scopes are invalid")
                validation_contract = _require_mapping(
                    evaluation_scopes.get("validation"),
                    field="evaluation_scopes.validation",
                )
                holdout_contract = _require_mapping(
                    evaluation_scopes.get("shared_project_holdout"),
                    field="evaluation_scopes.shared_project_holdout",
                )
                if validation_contract != {
                    "target_dates": [
                        row["target_date"] for row in validation_scope["common_keys"]
                    ]
                }:
                    raise ValueError("Validation scope dates disagree with date alignment")
                if holdout_contract != {
                    "term": _UNIFIED_HOLDOUT_TERM,
                    "target_dates": [
                        row["target_date"] for row in holdout_scope["common_keys"]
                    ],
                    "previously_exposed_through": list(
                        _UNIFIED_MODEL_VARIANTS[:4]
                    ),
                }:
                    raise ValueError(
                        "Shared project holdout scope disagrees with date alignment"
                    )

                portfolio_assumptions = _require_mapping(
                    contract.get("portfolio_assumptions"),
                    field="portfolio_assumptions",
                )
                evaluation_dates = _require_mapping(
                    portfolio_assumptions.get("evaluation_dates"),
                    field="portfolio_assumptions.evaluation_dates",
                )
                if (
                    portfolio_assumptions.get("symbol") != contract["symbol"]
                    or evaluation_dates
                    != {
                        "feature_start": holdout_scope["feature_start"],
                        "feature_end": holdout_scope["feature_end"],
                        "target_start": holdout_scope["target_start"],
                        "target_end": holdout_scope["target_end"],
                    }
                ):
                    raise ValueError(
                        "Portfolio evaluation dates disagree with the shared project holdout"
                    )
                metric_recomputation = _require_mapping(
                    contract.get("metric_recomputation"),
                    field="metric_recomputation",
                )
                if (
                    metric_recomputation.get("models_retrained") is not False
                    or metric_recomputation.get("backtests_rerun") is not False
                ):
                    raise ValueError(
                        "Unified comparison API requires persisted, non-retrained results"
                    )

                (
                    validation_models,
                    validation_baselines,
                    holdout_models,
                    holdout_baselines,
                    model_portfolios,
                    _benchmarks,
                    model_ids,
                ) = self._unified_metric_collections(metrics)
                if metrics.get("comparison_id") != safe_id:
                    raise ValueError("metrics.json comparison identity mismatch")
                if any(row["row_count"] != validation_scope["common_count"] for row in validation_models):
                    raise ValueError("Validation metric counts disagree with date alignment")
                if any(row["row_count"] != holdout_scope["common_count"] for row in holdout_models):
                    raise ValueError("Holdout metric counts disagree with date alignment")
                for scope_name, metric_rows, date_scope in (
                    (
                        "validation",
                        [*validation_models, *validation_baselines],
                        validation_scope,
                    ),
                    (
                        "shared project holdout",
                        [*holdout_models, *holdout_baselines],
                        holdout_scope,
                    ),
                ):
                    actual_counts = {
                        "0": sum(
                            row["actual_class"] == 0
                            for row in date_scope["common_keys"]
                        ),
                        "1": sum(
                            row["actual_class"] == 1
                            for row in date_scope["common_keys"]
                        ),
                    }
                    if any(
                        row["row_count"] != date_scope["common_count"]
                        or row["class_counts"] != actual_counts
                        for row in metric_rows
                    ):
                        raise ValueError(
                            f"{scope_name} metric class counts disagree with date alignment"
                        )

                if set(sources) != {
                    "schema_version",
                    "comparison_id",
                    "sources",
                    "relationships",
                    "evidence_files",
                    "source_revalidation",
                }:
                    raise ValueError("source_roots.json has an unexpected schema")
                source_rows = sources.get("sources")
                if not isinstance(source_rows, list) or len(source_rows) != 3:
                    raise ValueError("source_roots.json must contain three sources")
                expected_roles = {"baseline_training", "baseline_vertical_slice", "lstm"}
                expected_source_types = {
                    "baseline_training": "model_training_run",
                    "baseline_vertical_slice": "vertical_slice",
                    "lstm": "lstm_price_only_experiment",
                }
                expected_source_derivations = {
                    "baseline_training": "byte_copy",
                    "baseline_vertical_slice": "deterministic_redacted_projection",
                    "lstm": "byte_copy",
                }
                expected_integrity_root_names = {
                    "baseline_training": "run_manifest.json",
                    "baseline_vertical_slice": "manifest.json",
                    "lstm": "run_manifest.json",
                }
                expected_integrity_pack_paths = {
                    "baseline_training": (
                        "evidence/source_manifests/baseline_training_run_manifest.json"
                    ),
                    "baseline_vertical_slice": (
                        "evidence/source_manifests/"
                        "baseline_vertical_manifest_certificate.json"
                    ),
                    "lstm": "evidence/source_manifests/lstm_run_manifest.json",
                }
                source_authority = self._unified_source_authority()
                observed_roles: list[str] = []
                sources_by_role: dict[str, dict[str, Any]] = {}
                copied_manifests: dict[str, dict[str, Any]] = {}
                source_inventory_maps: dict[str, dict[str, str]] = {}
                source_payload_inventories: dict[str, dict[str, str]] = {}
                for index, raw_source in enumerate(source_rows):
                    source = _require_mapping(raw_source, field=f"sources[{index}]")
                    if set(source) != {
                        "role",
                        "artifact_type",
                        "run_id",
                        "integrity_root_name",
                        "integrity_root_sha256",
                        "payload_file_count",
                        "integrity_root_pack_path",
                        "integrity_root_pack_sha256",
                        "integrity_root_derivation",
                    }:
                        raise ValueError(f"sources[{index}] has an unexpected schema")
                    role = _require_string(source.get("role"), field=f"sources[{index}].role")
                    observed_roles.append(role)
                    sources_by_role[role] = source
                    artifact_type = _require_string(
                        source.get("artifact_type"),
                        field=f"sources[{index}].artifact_type",
                    )
                    if artifact_type != expected_source_types.get(role):
                        raise ValueError("Source role and artifact type disagree")
                    source_run_id = _validate_identifier(
                        _require_string(source.get("run_id"), field=f"sources[{index}].run_id"),
                        field=f"sources[{index}].run_id",
                    )
                    integrity_root_name = _require_string(
                        source.get("integrity_root_name"),
                        field=f"sources[{index}].integrity_root_name",
                    )
                    if integrity_root_name != expected_integrity_root_names.get(role):
                        raise ValueError("Source integrity-root name is invalid")
                    root_digest = _sha256_string(
                        source.get("integrity_root_sha256"),
                        field=f"sources[{index}].integrity_root_sha256",
                    )
                    packed_path = _safe_relative_path(
                        source.get("integrity_root_pack_path"),
                        field=f"sources[{index}].integrity_root_pack_path",
                    )
                    if packed_path != expected_integrity_pack_paths.get(role):
                        raise ValueError("Source integrity-root pack path is invalid")
                    packed_digest = _sha256_string(
                        source.get("integrity_root_pack_sha256"),
                        field=f"sources[{index}].integrity_root_pack_sha256",
                    )
                    if inventory.get(packed_path) != packed_digest:
                        raise ValueError("Source integrity root is not checksum-bound by the pack")
                    expected_identity = source_authority.get(role)
                    if (
                        expected_identity is None
                        or source_run_id != expected_identity["run_id"]
                        or root_digest != expected_identity["manifest_sha256"]
                        or (
                            role == "baseline_vertical_slice"
                            and packed_digest
                            != expected_identity["portable_certificate_sha256"]
                        )
                    ):
                        raise ValueError(
                            f"Authoritative source binding mismatch for {role}"
                        )
                    payload_file_count = _strict_integer(
                        source.get("payload_file_count"),
                        field=f"sources[{index}].payload_file_count",
                    )
                    if payload_file_count <= 0:
                        raise ValueError("Source payload_file_count must be positive")
                    derivation = _require_string(
                        source.get("integrity_root_derivation"),
                        field=f"sources[{index}].integrity_root_derivation",
                    )
                    if derivation != expected_source_derivations.get(role):
                        raise ValueError("Source integrity-root derivation is invalid")
                    if derivation == "byte_copy" and root_digest != packed_digest:
                        raise ValueError(
                            "Byte-copied source integrity root digest is inconsistent"
                        )
                    integrity_evidence: dict[str, Any]
                    if derivation == "byte_copy":
                        copied_manifest = self._comparison_json(
                            payloads, inventory, packed_path
                        )
                        if (
                            copied_manifest.get("schema_version") != SCHEMA_VERSION
                            or copied_manifest.get("run_id") != source.get("run_id")
                        ):
                            raise ValueError(
                                "Byte-copied source manifest identity is inconsistent"
                            )
                        copied_manifests[role] = copied_manifest
                        integrity_evidence = copied_manifest
                        if role == "baseline_training":
                            raw_models = copied_manifest.get("models")
                            if not isinstance(raw_models, list) or len(raw_models) != 4:
                                raise ValueError(
                                    "Baseline source manifest model identities are incomplete"
                                )
                            source_model_ids = {
                                _require_string(
                                    _require_mapping(
                                        row,
                                        field="baseline source manifest model",
                                    ).get("variant_id"),
                                    field="baseline source manifest model.variant_id",
                                ): _require_string(
                                    _require_mapping(
                                        row,
                                        field="baseline source manifest model",
                                    ).get("model_id"),
                                    field="baseline source manifest model.model_id",
                                )
                                for row in raw_models
                            }
                            if source_model_ids != {
                                variant: model_ids[variant]
                                for variant in _UNIFIED_MODEL_VARIANTS[:4]
                            }:
                                raise ValueError(
                                    "Baseline source manifest model identities disagree"
                                )
                        elif role == "lstm":
                            if copied_manifest.get("model_ids") != {
                                variant: model_ids[variant]
                                for variant in _UNIFIED_MODEL_VARIANTS[4:]
                            }:
                                raise ValueError(
                                    "LSTM source manifest model identities disagree"
                                )
                    if derivation == "deterministic_redacted_projection":
                        certificate = self._comparison_json(
                            payloads, inventory, packed_path
                        )
                        certificate_source = _require_mapping(
                            certificate.get("source_manifest"),
                            field="source integrity certificate.source_manifest",
                        )
                        if (
                            certificate.get("artifact_type")
                            != "paisa_source_integrity_certificate"
                            or certificate.get("run_id") != source.get("run_id")
                            or certificate.get("source_role") != role
                            or certificate.get("raw_manifest_included") is not False
                            or certificate_source
                            != {
                                "name": integrity_root_name,
                                "sha256": root_digest,
                            }
                        ):
                            raise ValueError(
                                "Redacted source integrity certificate is invalid"
                            )
                        if (
                            set(certificate)
                            != {
                                "schema_version",
                                "artifact_type",
                                "source_role",
                                "run_id",
                                "source_manifest",
                                "projection_policy",
                                "raw_manifest_included",
                                "source_assertions",
                                "artifact_inventory",
                                "disclaimer",
                            }
                            or certificate.get("schema_version") != SCHEMA_VERSION
                            or certificate.get("projection_policy")
                            != "allowlisted_fields_v1"
                            or certificate.get("disclaimer") != ACADEMIC_DISCLAIMER
                            or certificate.get("source_assertions")
                            != {
                                "schema_version": SCHEMA_VERSION,
                                "status": "PASS",
                                "symbol": "OGDC",
                                "model_training_run_id": source_authority[
                                    "baseline_training"
                                ]["run_id"],
                                "registered_model_ids": {
                                    variant: model_ids[variant]
                                    for variant in _UNIFIED_MODEL_VARIANTS[:4]
                                },
                                "disclaimer": ACADEMIC_DISCLAIMER,
                            }
                        ):
                            raise ValueError(
                                "Redacted source integrity certificate assertions are invalid"
                            )
                        integrity_evidence = certificate
                    source_inventory = self._unified_source_inventory(
                        integrity_evidence.get("artifact_inventory"),
                        field=f"sources[{index}] integrity artifact_inventory",
                        expected_excludes=(
                            ("manifest.json",)
                            if role == "baseline_vertical_slice"
                            else ("run_manifest.json",)
                        ),
                    )
                    if len(source_inventory) != payload_file_count:
                        raise ValueError(
                            "Source payload_file_count disagrees with integrity evidence"
                        )
                    source_payload_inventories[role] = source_inventory
                    source_inventory_maps[role] = {
                        integrity_root_name: root_digest,
                        **source_inventory,
                    }
                if observed_roles != [
                    "baseline_training",
                    "baseline_vertical_slice",
                    "lstm",
                ]:
                    raise ValueError("Source roles are incomplete or duplicated")
                lstm_manifest = copied_manifests.get("lstm")
                if lstm_manifest is None:
                    raise ValueError("LSTM source manifest evidence is missing")
                lstm_artifact_manifest_path = (
                    "evidence/source_manifests/lstm_artifact_manifest.json"
                )
                lstm_artifact_manifest = self._comparison_json(
                    payloads, inventory, lstm_artifact_manifest_path
                )
                lstm_nested_inventory = self._unified_source_inventory(
                    lstm_artifact_manifest,
                    field="Copied LSTM artifact inventory",
                    expected_excludes=("artifact_manifest.json", "run_manifest.json"),
                )
                lstm_root_inventory = source_payload_inventories.get("lstm")
                if lstm_root_inventory is None:
                    raise ValueError("LSTM source inventory evidence is missing")
                if (
                    lstm_root_inventory.get("artifact_manifest.json")
                    != inventory.get(lstm_artifact_manifest_path)
                    or lstm_nested_inventory
                    != {
                        relative: digest
                        for relative, digest in lstm_root_inventory.items()
                        if relative != "artifact_manifest.json"
                    }
                ):
                    raise ValueError(
                        "Copied LSTM nested and root inventories disagree"
                    )
                for reference_key, expected_role in (
                    ("baseline_training_run", "baseline_training"),
                    ("baseline_vertical_slice", "baseline_vertical_slice"),
                ):
                    reference = _require_mapping(
                        lstm_manifest.get(reference_key),
                        field=f"LSTM source manifest.{reference_key}",
                    )
                    expected_source = sources_by_role[expected_role]
                    if (
                        reference.get("run_id") != expected_source.get("run_id")
                        or reference.get("manifest_sha256")
                        != expected_source.get("integrity_root_sha256")
                    ):
                        raise ValueError(
                            "LSTM source manifest relationships are inconsistent"
                        )
                if sources.get("relationships") != {
                    "vertical_slice_uses_training_run": True,
                    "lstm_uses_training_run": True,
                    "lstm_uses_vertical_slice": True,
                }:
                    raise ValueError("Source relationships are invalid")
                if sources.get("source_revalidation") != {
                    "validated_before_generation": True,
                    "revalidated_before_publication": True,
                }:
                    raise ValueError("Source revalidation evidence is invalid")
                evidence_rows = sources.get("evidence_files")
                if not isinstance(evidence_rows, list):
                    raise ValueError("Source evidence_files must be an array")
                expected_bindings = expected_evidence_bindings(
                    model_ids,
                    portfolio_files=_UNIFIED_PORTFOLIO_FILES,
                    source_authority=source_authority,
                )
                observed_bindings: dict[str, tuple[str, str, str]] = {}
                previous_pack_path = ""
                for index, raw_evidence in enumerate(evidence_rows):
                    evidence = _require_mapping(raw_evidence, field=f"evidence_files[{index}]")
                    if set(evidence) != {
                        "source_role",
                        "source_path",
                        "pack_path",
                        "sha256",
                        "purpose",
                        "derivation",
                    }:
                        raise ValueError(
                            f"evidence_files[{index}] has an unexpected schema"
                        )
                    pack_path = _safe_relative_path(
                        evidence.get("pack_path"), field=f"evidence_files[{index}].pack_path"
                    )
                    if (
                        not pack_path.startswith("evidence/")
                        or pack_path <= previous_pack_path
                    ):
                        raise ValueError(
                            "Evidence pack paths must be sorted, unique, and under evidence/"
                        )
                    previous_pack_path = pack_path
                    source_path = _safe_relative_path(
                        evidence.get("source_path"), field=f"evidence_files[{index}].source_path"
                    )
                    if Path(source_path).suffix.lower() in _EXECUTABLE_MODEL_SUFFIXES:
                        raise ValueError("Executable model source references are forbidden")
                    digest = _sha256_string(
                        evidence.get("sha256"), field=f"evidence_files[{index}].sha256"
                    )
                    source_role = evidence.get("source_role")
                    if source_role not in expected_roles:
                        raise ValueError("Source evidence role is invalid")
                    purpose = _require_string(
                        evidence.get("purpose"),
                        field=f"evidence_files[{index}].purpose",
                    )
                    if (
                        evidence.get("derivation") != "byte_copy"
                        or inventory.get(pack_path) != digest
                        or source_inventory_maps[source_role].get(source_path)
                        != digest
                    ):
                        raise ValueError(
                            f"Source evidence checksum or derivation mismatch: {pack_path}"
                        )
                    observed_bindings[pack_path] = (
                        source_role,
                        source_path,
                        purpose,
                    )
                if observed_bindings != expected_bindings:
                    raise ValueError(
                        "Source evidence bindings are incomplete or inconsistent"
                    )
                actual_evidence = {
                    relative
                    for relative in inventory
                    if relative.startswith("evidence/")
                }
                if actual_evidence != {
                    *expected_bindings,
                    "evidence/source_manifests/"
                    "baseline_vertical_manifest_certificate.json",
                }:
                    raise ValueError(
                        "Pack evidence files do not match source evidence bindings"
                    )

                expected_ranking_keys = {
                    "schema_version",
                    "comparison_id",
                    "validation_model_selection",
                    "lstm_validation_preference",
                    "shared_project_holdout_balanced_accuracy",
                    "shared_project_holdout_f1",
                    "simulated_model_portfolio_return",
                    "simulated_model_portfolio_drawdown",
                    "simulated_model_portfolio_fees",
                }
                if set(rankings) != expected_ranking_keys:
                    raise ValueError("rankings.json has an unexpected schema")
                validation_ranking = self._prediction_ranking_rows(
                    rankings.get("validation_model_selection"),
                    field="rankings.validation_model_selection",
                    expected_competitors=_UNIFIED_MODEL_VARIANTS,
                    model_ids=model_ids,
                )
                lstm_ranking = self._prediction_ranking_rows(
                    rankings.get("lstm_validation_preference"),
                    field="rankings.lstm_validation_preference",
                    expected_competitors=("M-005", "M-006"),
                    model_ids=model_ids,
                )
                holdout_balanced_ranking = self._prediction_ranking_rows(
                    rankings.get("shared_project_holdout_balanced_accuracy"),
                    field="rankings.shared_project_holdout_balanced_accuracy",
                    expected_competitors=_UNIFIED_MODEL_VARIANTS,
                    model_ids=model_ids,
                )
                holdout_f1_ranking = self._prediction_ranking_rows(
                    rankings.get("shared_project_holdout_f1"),
                    field="rankings.shared_project_holdout_f1",
                    expected_competitors=_UNIFIED_MODEL_VARIANTS,
                    model_ids=model_ids,
                )
                portfolio_rankings: dict[str, list[dict[str, Any]]] = {}
                for ranking_name in (
                    "simulated_model_portfolio_return",
                    "simulated_model_portfolio_drawdown",
                    "simulated_model_portfolio_fees",
                ):
                    portfolio_rankings[ranking_name] = self._portfolio_ranking_rows(
                        rankings.get(ranking_name),
                        field=f"rankings.{ranking_name}",
                        model_ids=model_ids,
                    )

                validation_by_id = {
                    row["competitor_id"]: row for row in validation_models
                }
                holdout_by_id = {row["competitor_id"]: row for row in holdout_models}
                portfolio_by_id = {
                    row["variant_id"]: row for row in model_portfolios
                }

                def validate_prediction_ranking(
                    ranking_rows: list[dict[str, Any]],
                    metric_rows: Mapping[str, dict[str, Any]],
                    *,
                    primary: str,
                    field: str,
                ) -> None:
                    tie_breakers = (
                        ("balanced_accuracy", "f1", "accuracy")
                        if primary == "balanced_accuracy"
                        else ("f1", "balanced_accuracy", "accuracy")
                    )
                    for row in ranking_rows:
                        metric = metric_rows[row["competitor_id"]]
                        if any(row[key] != metric[key] for key in tie_breakers):
                            raise ValueError(
                                f"{field} values disagree with persisted metrics"
                            )

                    def key(row: dict[str, Any]) -> tuple[Any, ...]:
                        metric = metric_rows[row["competitor_id"]]
                        numeric = tuple(
                            math.inf if metric[name] is None else -metric[name]
                            for name in tie_breakers
                        )
                        return (*numeric, metric["registered_model_id"])

                    if ranking_rows != sorted(ranking_rows, key=key):
                        raise ValueError(f"{field} ordering contradicts its metrics")

                validate_prediction_ranking(
                    validation_ranking,
                    validation_by_id,
                    primary="balanced_accuracy",
                    field="rankings.validation_model_selection",
                )
                validate_prediction_ranking(
                    lstm_ranking,
                    validation_by_id,
                    primary="balanced_accuracy",
                    field="rankings.lstm_validation_preference",
                )
                validate_prediction_ranking(
                    holdout_balanced_ranking,
                    holdout_by_id,
                    primary="balanced_accuracy",
                    field="rankings.shared_project_holdout_balanced_accuracy",
                )
                validate_prediction_ranking(
                    holdout_f1_ranking,
                    holdout_by_id,
                    primary="f1",
                    field="rankings.shared_project_holdout_f1",
                )

                portfolio_ranking_fields = {
                    "simulated_model_portfolio_return": (
                        "total_return",
                        "sharpe_ratio",
                        "maximum_drawdown",
                    ),
                    "simulated_model_portfolio_drawdown": (
                        "maximum_drawdown",
                        "total_return",
                    ),
                    "simulated_model_portfolio_fees": (
                        "fees_paid",
                        "number_of_trades",
                    ),
                }
                for ranking_name, persisted_fields in portfolio_ranking_fields.items():
                    ranking_rows = portfolio_rankings[ranking_name]
                    for row in ranking_rows:
                        metric = portfolio_by_id[row["competitor_id"]]
                        if any(
                            row[field_name] != metric[field_name]
                            for field_name in persisted_fields
                        ):
                            raise ValueError(
                                f"rankings.{ranking_name} values disagree with persisted metrics"
                            )

                expected_portfolio_orders = {
                    "simulated_model_portfolio_return": sorted(
                        _UNIFIED_MODEL_VARIANTS,
                        key=lambda variant: (
                            -portfolio_by_id[variant]["total_return"],
                            math.inf
                            if portfolio_by_id[variant]["sharpe_ratio"] is None
                            else -portfolio_by_id[variant]["sharpe_ratio"],
                            portfolio_by_id[variant]["maximum_drawdown"],
                            variant,
                        ),
                    ),
                    "simulated_model_portfolio_drawdown": sorted(
                        _UNIFIED_MODEL_VARIANTS,
                        key=lambda variant: (
                            portfolio_by_id[variant]["maximum_drawdown"],
                            -portfolio_by_id[variant]["total_return"],
                            variant,
                        ),
                    ),
                    "simulated_model_portfolio_fees": sorted(
                        _UNIFIED_MODEL_VARIANTS,
                        key=lambda variant: (
                            Decimal(portfolio_by_id[variant]["fees_paid"]),
                            portfolio_by_id[variant]["number_of_trades"],
                            variant,
                        ),
                    ),
                }
                for ranking_name, expected_order in expected_portfolio_orders.items():
                    if [
                        row["competitor_id"]
                        for row in portfolio_rankings[ranking_name]
                    ] != expected_order:
                        raise ValueError(
                            f"rankings.{ranking_name} ordering contradicts its metrics"
                        )

                if set(summary) != {
                    "schema_version",
                    "artifact_type",
                    "comparison_id",
                    "status",
                    "symbol",
                    "path_base",
                    "holdout_context",
                    "outcomes_are_not_combined",
                    "reported_outcomes",
                    "artifact_paths",
                    "limitations",
                    "disclaimer",
                }:
                    raise ValueError("final_summary.json has an unexpected schema")
                if (
                    summary.get("artifact_type") != _UNIFIED_ARTIFACT_TYPE
                    or summary.get("status") != "PASS"
                    or summary.get("outcomes_are_not_combined") is not True
                    or summary.get("disclaimer") != ACADEMIC_DISCLAIMER
                    or summary.get("symbol") != contract.get("symbol")
                    or summary.get("path_base") != "comparison_root"
                ):
                    raise ValueError("Final summary completion or identity contract is invalid")
                expected_artifact_paths = {
                    "source_roots": "source_roots.json",
                    "comparison_contract": "comparison_contract.json",
                    "date_alignment": "date_alignment.json",
                    "metrics": "metrics.json",
                    "rankings": "rankings.json",
                    "reference_baseline_predictions": (
                        "derived/reference_baseline_predictions.csv"
                    ),
                    "human_report": "report.md",
                }
                if summary.get("artifact_paths") != expected_artifact_paths:
                    raise ValueError("Final summary artifact paths are invalid")
                holdout_context = _require_mapping(
                    summary.get("holdout_context"), field="summary.holdout_context"
                )
                if (
                    holdout_context.get("term") != _UNIFIED_HOLDOUT_TERM
                    or holdout_context.get("common_count") != holdout_scope["common_count"]
                    or holdout_context.get("target_start") != holdout_scope["target_start"]
                    or holdout_context.get("target_end") != holdout_scope["target_end"]
                    or holdout_context.get("previously_exposed_through")
                    != holdout_scope["previously_exposed_through"]
                ):
                    raise ValueError("Summary shared project holdout context is invalid")
                limitations = _string_array(summary.get("limitations"), field="summary.limitations")
                if not limitations:
                    raise ValueError("Summary limitations must not be empty")
                outcomes = _require_mapping(
                    summary.get("reported_outcomes"), field="summary.reported_outcomes"
                )
                if set(outcomes) != {
                    "validation_selected_model",
                    "lstm_validation_preferred_model",
                    "best_shared_holdout_balanced_accuracy",
                    "best_shared_holdout_f1",
                    "best_simulated_model_portfolio",
                    "lowest_drawdown_model_portfolio",
                    "lowest_fee_model_portfolio",
                    "cash_only_benchmark",
                    "buy_and_hold_result",
                    "any_model_beat_buy_and_hold",
                }:
                    raise ValueError("Summary reported outcomes are incomplete")
                for outcome_name in (
                    "validation_selected_model",
                    "lstm_validation_preferred_model",
                    "best_shared_holdout_balanced_accuracy",
                    "best_shared_holdout_f1",
                    "best_simulated_model_portfolio",
                    "lowest_drawdown_model_portfolio",
                    "lowest_fee_model_portfolio",
                ):
                    outcome = _require_mapping(outcomes[outcome_name], field=f"outcomes.{outcome_name}")
                    variant = _require_string(
                        outcome.get("variant_id"), field=f"outcomes.{outcome_name}.variant_id"
                    )
                    if variant not in model_ids or outcome.get("registered_model_id") != model_ids[variant]:
                        raise ValueError("Summary outcome model identity mismatch")
                prediction_expectations = {
                    "validation_selected_model": (
                        validation_ranking[0]["competitor_id"],
                        validation_by_id,
                        "validation balanced accuracy, then F1, accuracy, and model ID",
                        "all_six_models",
                    ),
                    "lstm_validation_preferred_model": (
                        lstm_ranking[0]["competitor_id"],
                        validation_by_id,
                        "validation balanced accuracy, then F1, accuracy, and model ID",
                        "M-005_and_M-006_only",
                    ),
                    "best_shared_holdout_balanced_accuracy": (
                        holdout_balanced_ranking[0]["competitor_id"],
                        holdout_by_id,
                        "highest shared project holdout balanced accuracy",
                        _UNIFIED_HOLDOUT_TERM,
                    ),
                    "best_shared_holdout_f1": (
                        holdout_f1_ranking[0]["competitor_id"],
                        holdout_by_id,
                        "highest shared project holdout F1",
                        _UNIFIED_HOLDOUT_TERM,
                    ),
                }
                for outcome_name, (
                    expected_variant,
                    metric_rows,
                    expected_criterion,
                    expected_scope,
                ) in prediction_expectations.items():
                    outcome = _require_mapping(
                        outcomes[outcome_name], field=f"outcomes.{outcome_name}"
                    )
                    if set(outcome) != {
                        "variant_id",
                        "registered_model_id",
                        "criterion",
                        "scope",
                        "accuracy",
                        "balanced_accuracy",
                        "f1",
                    }:
                        raise ValueError(
                            f"Summary prediction outcome {outcome_name} has an unexpected schema"
                        )
                    expected_metric = metric_rows[expected_variant]
                    if (
                        outcome.get("variant_id") != expected_variant
                        or outcome.get("registered_model_id")
                        != expected_metric["registered_model_id"]
                        or outcome.get("accuracy") != expected_metric["accuracy"]
                        or outcome.get("balanced_accuracy")
                        != expected_metric["balanced_accuracy"]
                        or outcome.get("f1") != expected_metric["f1"]
                        or outcome.get("criterion") != expected_criterion
                        or outcome.get("scope") != expected_scope
                    ):
                        raise ValueError(
                            f"Summary prediction outcome {outcome_name} disagrees with persisted rankings or metrics"
                        )
                portfolio_expectations = {
                    "best_simulated_model_portfolio": (
                        portfolio_rankings["simulated_model_portfolio_return"][0][
                            "competitor_id"
                        ],
                        "highest simulated total return",
                    ),
                    "lowest_drawdown_model_portfolio": (
                        portfolio_rankings[
                            "simulated_model_portfolio_drawdown"
                        ][0]["competitor_id"],
                        "lowest simulated maximum drawdown",
                    ),
                    "lowest_fee_model_portfolio": (
                        portfolio_rankings["simulated_model_portfolio_fees"][0][
                            "competitor_id"
                        ],
                        "lowest simulated fees paid",
                    ),
                }
                for outcome_name, (
                    expected_variant,
                    expected_criterion,
                ) in portfolio_expectations.items():
                    outcome = _require_mapping(
                        outcomes[outcome_name], field=f"outcomes.{outcome_name}"
                    )
                    if set(outcome) != {
                        "variant_id",
                        "registered_model_id",
                        "portfolio_id",
                        "criterion",
                        "ending_equity",
                        "total_return",
                        "maximum_drawdown",
                        "fees_paid",
                    }:
                        raise ValueError(
                            f"Summary portfolio outcome {outcome_name} has an unexpected schema"
                        )
                    expected_portfolio = portfolio_by_id[expected_variant]
                    for key in (
                        "registered_model_id",
                        "portfolio_id",
                        "ending_equity",
                        "total_return",
                        "maximum_drawdown",
                        "fees_paid",
                    ):
                        if outcome.get(key) != expected_portfolio[key]:
                            raise ValueError(
                                f"Summary portfolio outcome {outcome_name} disagrees with persisted rankings or metrics"
                            )
                    if outcome.get("criterion") != expected_criterion:
                        raise ValueError(
                            f"Summary portfolio outcome {outcome_name} has an invalid criterion"
                        )

                benchmark_by_id = {
                    row["benchmark_id"]: row for row in _benchmarks
                }
                for benchmark_name, benchmark_id in (
                    ("cash_only_benchmark", "cash-only"),
                    ("buy_and_hold_result", "buy-and-hold"),
                ):
                    outcome = _require_mapping(
                        outcomes[benchmark_name], field=f"outcomes.{benchmark_name}"
                    )
                    expected_benchmark = benchmark_by_id[benchmark_id]
                    expected_keys = {
                        "initial_equity",
                        "ending_equity",
                        "total_return",
                        "maximum_drawdown",
                        "volatility",
                        "sharpe_ratio",
                        "number_of_trades",
                        "fees_paid",
                    }
                    if set(outcome) != expected_keys or any(
                        outcome.get(key) != expected_benchmark[key]
                        for key in expected_keys
                    ):
                        raise ValueError(
                            f"Summary benchmark outcome {benchmark_name} disagrees with persisted metrics"
                        )
                beat = _require_mapping(
                    outcomes["any_model_beat_buy_and_hold"],
                    field="outcomes.any_model_beat_buy_and_hold",
                )
                _boolean(beat.get("value"), field="outcomes.any_model_beat_buy_and_hold.value")
                best_model_variant = portfolio_rankings[
                    "simulated_model_portfolio_return"
                ][0]["competitor_id"]
                best_model_excess_return = portfolio_by_id[best_model_variant][
                    "excess_return_vs_buy_and_hold"
                ]
                if (
                    set(beat)
                    != {
                        "value",
                        "comparison_metric",
                        "best_model_variant_id",
                        "best_model_minus_buy_and_hold",
                    }
                    or beat.get("comparison_metric") != "simulated total return"
                    or beat.get("best_model_variant_id") != best_model_variant
                    or not math.isclose(
                        _strict_float(
                            beat.get("best_model_minus_buy_and_hold"),
                            field=(
                                "outcomes.any_model_beat_buy_and_hold."
                                "best_model_minus_buy_and_hold"
                            ),
                        ),
                        best_model_excess_return,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or beat.get("value") != (best_model_excess_return > 0.0)
                ):
                    raise ValueError("Buy-and-hold comparison metric is invalid")

                public_artifacts = {
                    "summary": summary,
                    "contract": contract,
                    "sources": sources,
                    "metrics": metrics,
                    "rankings": rankings,
                }
                _validate_public_text_tree(
                    public_artifacts, field="unified comparison public text"
                )
                prohibited_text = json.dumps(
                    public_artifacts,
                    sort_keys=True,
                ).lower()
                if "universal winner" in prohibited_text or "universal_winner" in prohibited_text:
                    raise ValueError("Unified comparison must not claim a universal winner")

                signature_after = _descriptor_tree_signature(
                    directory_descriptor
                )
                if signature_before != signature_after:
                    raise ValueError("Unified comparison run changed during validation")
                validated = ValidatedUnifiedComparison(
                    comparison_id=safe_id,
                    manifest=manifest,
                    summary=summary,
                    contract=contract,
                    sources=sources,
                    date_alignment=date_alignment,
                    metrics=metrics,
                    rankings=rankings,
                    artifact_paths=frozenset(inventory),
                    signature=signature_after,
                )
                self._comparison_cache[safe_id] = validated
                return validated
            except (RunNotCompletedError, RunNotFoundError):
                raise
            except InvalidIdentifierError as exc:
                raise ArtifactIntegrityError(
                    f"Unified comparison run '{safe_id}' contains an unsafe artifact identity"
                ) from exc
            except ArtifactRepositoryError:
                raise
            except Exception as exc:
                raise ArtifactIntegrityError(
                    f"Unified comparison run '{safe_id}' failed integrity validation"
                ) from exc
            finally:
                if captured is not None:
                    captured.close()

    def _unified_context(
        self, comparison: ValidatedUnifiedComparison
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "comparison_id": comparison.comparison_id,
            "status": "PASS",
            "symbol": _require_string(
                comparison.summary.get("symbol"), field="summary.symbol"
            ),
            "holdout_term": _UNIFIED_HOLDOUT_TERM,
            "outcomes_are_not_combined": True,
            "limitations": _string_array(
                comparison.summary.get("limitations"), field="summary.limitations"
            ),
            "disclaimer": ACADEMIC_DISCLAIMER,
        }

    def configured_default_unified_comparison_id(self) -> str | None:
        value = self.settings.default_unified_comparison_run_id
        if value is None:
            return None
        return _validate_identifier(value, field="default comparison run ID")

    def list_completed_unified_comparisons(self) -> list[dict[str, Any]]:
        try:
            root_descriptor = self._open_comparison_root_descriptor()
        except RunNotFoundError:
            return []
        try:
            try:
                with os.scandir(root_descriptor) as entries:
                    names = sorted(entry.name for entry in entries)
            except OSError as exc:
                raise ArtifactIntegrityError(
                    "Configured unified comparison root cannot be enumerated securely"
                ) from exc
            default_id = self.configured_default_unified_comparison_id()
            rows: list[dict[str, Any]] = []
            for name in names:
                try:
                    _validate_identifier(name, field="comparison_id")
                    metadata = os.stat(
                        name,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(metadata.st_mode):
                        continue
                    comparison = self._validate_unified_comparison(
                        name, root_descriptor=root_descriptor
                    )
                    context = self._unified_context(comparison)
                    alignment = self._unified_common_date_contract_from(
                        comparison
                    )
                except (ArtifactRepositoryError, OSError, TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "comparison_id": comparison.comparison_id,
                        "status": "PASS",
                        "symbol": context["symbol"],
                        "model_count": len(_UNIFIED_MODEL_VARIANTS),
                        "validation_count": alignment["validation"]["common_count"],
                        "shared_holdout_count": alignment["shared_project_holdout"]["common_count"],
                        "shared_holdout_target_start": alignment["shared_project_holdout"]["target_start"],
                        "shared_holdout_target_end": alignment["shared_project_holdout"]["target_end"],
                        "holdout_term": _UNIFIED_HOLDOUT_TERM,
                        "is_default": comparison.comparison_id == default_id,
                    }
                )
            return rows
        finally:
            os.close(root_descriptor)

    def load_unified_comparison_detail(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        (
            validation_models,
            _validation_baselines,
            _holdout_models,
            _holdout_baselines,
            _portfolios,
            _benchmarks,
            _model_ids,
        ) = self._unified_metric_collections(comparison.metrics)
        models_by_id = {row["competitor_id"]: row for row in validation_models}
        date_contract = self._unified_common_date_contract_from(comparison)
        selection = _require_mapping(
            comparison.contract.get("selection_policy"), field="selection_policy"
        )
        presentation = _require_mapping(
            comparison.contract.get("presentation"), field="presentation"
        )
        return {
            **self._unified_context(comparison),
            "models": [
                {
                    "variant_id": variant,
                    "registered_model_id": models_by_id[variant]["registered_model_id"],
                    "algorithm": models_by_id[variant]["algorithm"],
                    "configuration_family": models_by_id[variant]["configuration_family"],
                }
                for variant in _UNIFIED_MODEL_VARIANTS
            ],
            "source_frequency": _require_string(
                comparison.contract.get("source_frequency"), field="source_frequency"
            ),
            "validation_count": date_contract["validation"]["common_count"],
            "shared_holdout_count": date_contract["shared_project_holdout"]["common_count"],
            "selection_policy": {
                "primary": _require_string(selection.get("primary"), field="selection_policy.primary"),
                "tie_breakers": _string_array(
                    selection.get("tie_breakers"), field="selection_policy.tie_breakers"
                ),
                "test_metrics_used_for_selection": _boolean(
                    selection.get("test_metrics_used_for_selection"),
                    field="selection_policy.test_metrics_used_for_selection",
                ),
            },
            "presentation": {
                "format": _require_string(presentation.get("format"), field="presentation.format"),
                "plots_included": _boolean(
                    presentation.get("plots_included"), field="presentation.plots_included"
                ),
                "data_sources": [
                    "prediction-metrics",
                    "validation-metrics",
                    "portfolio-metrics",
                    "prediction-rankings",
                    "portfolio-rankings",
                    "common-date-contract",
                    "confusion-matrices",
                ],
            },
        }

    @staticmethod
    def _normalized_summary_outcomes(
        comparison: ValidatedUnifiedComparison,
    ) -> dict[str, Any]:
        outcomes = _require_mapping(
            comparison.summary.get("reported_outcomes"), field="reported_outcomes"
        )

        def prediction(name: str) -> dict[str, Any]:
            raw = _require_mapping(outcomes.get(name), field=name)
            return {
                "variant_id": _require_string(raw.get("variant_id"), field=f"{name}.variant_id"),
                "registered_model_id": _require_string(
                    raw.get("registered_model_id"), field=f"{name}.registered_model_id"
                ),
                "criterion": _require_string(raw.get("criterion"), field=f"{name}.criterion"),
                "scope": _require_string(raw.get("scope"), field=f"{name}.scope"),
                "accuracy": _bounded_float(
                    raw.get("accuracy"),
                    field=f"{name}.accuracy",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "balanced_accuracy": _bounded_float(
                    raw.get("balanced_accuracy"),
                    field=f"{name}.balanced_accuracy",
                    minimum=0.0,
                    maximum=1.0,
                    nullable=True,
                ),
                "f1": _bounded_float(
                    raw.get("f1"),
                    field=f"{name}.f1",
                    minimum=0.0,
                    maximum=1.0,
                ),
            }

        def portfolio(name: str) -> dict[str, Any]:
            raw = _require_mapping(outcomes.get(name), field=name)
            return {
                "variant_id": _require_string(raw.get("variant_id"), field=f"{name}.variant_id"),
                "registered_model_id": _require_string(
                    raw.get("registered_model_id"), field=f"{name}.registered_model_id"
                ),
                "portfolio_id": _require_string(raw.get("portfolio_id"), field=f"{name}.portfolio_id"),
                "criterion": _require_string(raw.get("criterion"), field=f"{name}.criterion"),
                "ending_equity": _money(raw.get("ending_equity"), field=f"{name}.ending_equity"),
                "total_return": _strict_float(raw.get("total_return"), field=f"{name}.total_return"),
                "maximum_drawdown": _bounded_float(
                    raw.get("maximum_drawdown"),
                    field=f"{name}.maximum_drawdown",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "fees_paid": _nonnegative_money(
                    raw.get("fees_paid"), field=f"{name}.fees_paid"
                ),
            }

        def benchmark(name: str) -> dict[str, Any]:
            raw = _require_mapping(outcomes.get(name), field=name)
            return {
                "initial_equity": _money(raw.get("initial_equity"), field=f"{name}.initial_equity"),
                "ending_equity": _money(raw.get("ending_equity"), field=f"{name}.ending_equity"),
                "total_return": _strict_float(raw.get("total_return"), field=f"{name}.total_return"),
                "maximum_drawdown": _bounded_float(
                    raw.get("maximum_drawdown"),
                    field=f"{name}.maximum_drawdown",
                    minimum=0.0,
                    maximum=1.0,
                ),
                "volatility": _bounded_float(
                    raw.get("volatility"),
                    field=f"{name}.volatility",
                    minimum=0.0,
                    nullable=True,
                ),
                "sharpe_ratio": _strict_float(
                    raw.get("sharpe_ratio"), field=f"{name}.sharpe_ratio", nullable=True
                ),
                "number_of_trades": _nonnegative_integer(
                    raw.get("number_of_trades"), field=f"{name}.number_of_trades"
                ),
                "fees_paid": _nonnegative_money(
                    raw.get("fees_paid"), field=f"{name}.fees_paid"
                ),
            }

        beat_raw = _require_mapping(
            outcomes.get("any_model_beat_buy_and_hold"),
            field="any_model_beat_buy_and_hold",
        )
        beat = {
            "value": _boolean(beat_raw.get("value"), field="any_model_beat_buy_and_hold.value"),
            "best_model_variant_id": _require_string(
                beat_raw.get("best_model_variant_id"),
                field="any_model_beat_buy_and_hold.best_model_variant_id",
            ),
            "best_model_minus_buy_and_hold": _strict_float(
                beat_raw.get("best_model_minus_buy_and_hold"),
                field="any_model_beat_buy_and_hold.best_model_minus_buy_and_hold",
            ),
            "comparison_metric": _require_string(
                beat_raw.get("comparison_metric"),
                field="any_model_beat_buy_and_hold.comparison_metric",
            ),
        }
        return {
            "validation_selected_model": prediction("validation_selected_model"),
            "lstm_validation_preferred_model": prediction("lstm_validation_preferred_model"),
            "best_shared_holdout_balanced_accuracy": prediction(
                "best_shared_holdout_balanced_accuracy"
            ),
            "best_shared_holdout_f1": prediction("best_shared_holdout_f1"),
            "best_simulated_model_portfolio": portfolio("best_simulated_model_portfolio"),
            "lowest_drawdown_model_portfolio": portfolio(
                "lowest_drawdown_model_portfolio"
            ),
            "lowest_fee_model_portfolio": portfolio("lowest_fee_model_portfolio"),
            "cash_only_benchmark": benchmark("cash_only_benchmark"),
            "buy_and_hold_result": benchmark("buy_and_hold_result"),
            "any_model_beat_buy_and_hold": beat,
        }

    def load_unified_summary(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        holdout = _require_mapping(
            comparison.summary.get("holdout_context"), field="holdout_context"
        )
        return {
            **self._unified_context(comparison),
            "holdout_context": {
                "term": _require_string(holdout.get("term"), field="holdout_context.term"),
                "common_count": _strict_integer(
                    holdout.get("common_count"), field="holdout_context.common_count"
                ),
                "target_start": _iso_date(
                    holdout.get("target_start"), field="holdout_context.target_start"
                ),
                "target_end": _iso_date(
                    holdout.get("target_end"), field="holdout_context.target_end"
                ),
                "previously_exposed_through": _string_array(
                    holdout.get("previously_exposed_through"),
                    field="holdout_context.previously_exposed_through",
                ),
            },
            "reported_outcomes": self._normalized_summary_outcomes(comparison),
        }

    def load_unified_sources(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        raw_sources = comparison.sources.get("sources")
        assert isinstance(raw_sources, list)
        sources: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_sources):
            source = _require_mapping(raw, field=f"sources[{index}]")
            sources.append(
                {
                    "role": _require_string(source.get("role"), field=f"sources[{index}].role"),
                    "artifact_type": _require_string(
                        source.get("artifact_type"), field=f"sources[{index}].artifact_type"
                    ),
                    "run_id": _require_string(source.get("run_id"), field=f"sources[{index}].run_id"),
                    "integrity_root_sha256": _sha256_string(
                        source.get("integrity_root_sha256"),
                        field=f"sources[{index}].integrity_root_sha256",
                    ),
                    "payload_file_count": _strict_integer(
                        source.get("payload_file_count"),
                        field=f"sources[{index}].payload_file_count",
                    ),
                    "integrity_root_derivation": _require_string(
                        source.get("integrity_root_derivation"),
                        field=f"sources[{index}].integrity_root_derivation",
                    ),
                }
            )
        return {
            **self._unified_context(comparison),
            "sources": sources,
            "relationships": {
                key: _boolean(value, field=f"relationships.{key}")
                for key, value in _require_mapping(
                    comparison.sources.get("relationships"), field="relationships"
                ).items()
            },
        }

    def load_unified_prediction_metrics(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        (
            _validation_models,
            _validation_baselines,
            holdout_models,
            holdout_baselines,
            _portfolios,
            _benchmarks,
            _model_ids,
        ) = self._unified_metric_collections(comparison.metrics)
        return {
            **self._unified_context(comparison),
            "models": holdout_models,
            "reference_baselines": holdout_baselines,
        }

    def load_unified_validation_metrics(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        (
            validation_models,
            validation_baselines,
            _holdout_models,
            _holdout_baselines,
            _portfolios,
            _benchmarks,
            _model_ids,
        ) = self._unified_metric_collections(comparison.metrics)
        return {
            **self._unified_context(comparison),
            "evaluation_scope": "validation",
            "models": validation_models,
            "reference_baselines": validation_baselines,
        }

    def load_unified_portfolio_metrics(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        (
            _validation_models,
            _validation_baselines,
            _holdout_models,
            _holdout_baselines,
            portfolios,
            benchmarks,
            _model_ids,
        ) = self._unified_metric_collections(comparison.metrics)
        return {
            **self._unified_context(comparison),
            "models": portfolios,
            "benchmarks": benchmarks,
        }

    def _unified_model_ids(self, comparison: ValidatedUnifiedComparison) -> dict[str, str]:
        return self._unified_metric_collections(comparison.metrics)[-1]

    def load_unified_prediction_rankings(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        model_ids = self._unified_model_ids(comparison)
        return {
            **self._unified_context(comparison),
            "validation_model_selection": self._prediction_ranking_rows(
                comparison.rankings.get("validation_model_selection"),
                field="rankings.validation_model_selection",
                expected_competitors=_UNIFIED_MODEL_VARIANTS,
                model_ids=model_ids,
            ),
            "lstm_validation_preference": self._prediction_ranking_rows(
                comparison.rankings.get("lstm_validation_preference"),
                field="rankings.lstm_validation_preference",
                expected_competitors=("M-005", "M-006"),
                model_ids=model_ids,
            ),
            "shared_project_holdout_balanced_accuracy": self._prediction_ranking_rows(
                comparison.rankings.get("shared_project_holdout_balanced_accuracy"),
                field="rankings.shared_project_holdout_balanced_accuracy",
                expected_competitors=_UNIFIED_MODEL_VARIANTS,
                model_ids=model_ids,
            ),
            "shared_project_holdout_f1": self._prediction_ranking_rows(
                comparison.rankings.get("shared_project_holdout_f1"),
                field="rankings.shared_project_holdout_f1",
                expected_competitors=_UNIFIED_MODEL_VARIANTS,
                model_ids=model_ids,
            ),
        }

    def load_unified_portfolio_rankings(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        model_ids = self._unified_model_ids(comparison)
        return {
            **self._unified_context(comparison),
            "simulated_model_portfolio_return": self._portfolio_ranking_rows(
                comparison.rankings.get("simulated_model_portfolio_return"),
                field="rankings.simulated_model_portfolio_return",
                model_ids=model_ids,
            ),
            "simulated_model_portfolio_drawdown": self._portfolio_ranking_rows(
                comparison.rankings.get("simulated_model_portfolio_drawdown"),
                field="rankings.simulated_model_portfolio_drawdown",
                model_ids=model_ids,
            ),
            "simulated_model_portfolio_fees": self._portfolio_ranking_rows(
                comparison.rankings.get("simulated_model_portfolio_fees"),
                field="rankings.simulated_model_portfolio_fees",
                model_ids=model_ids,
            ),
        }

    def load_unified_common_date_contract(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        return self._unified_common_date_contract_from(comparison)

    def _unified_common_date_contract_from(
        self, comparison: ValidatedUnifiedComparison
    ) -> dict[str, Any]:
        validation = self._unified_date_scope(
            comparison.date_alignment.get("validation"),
            field="date_alignment.validation",
            expected_term="validation",
        )
        holdout = self._unified_date_scope(
            comparison.date_alignment.get("shared_project_holdout"),
            field="date_alignment.shared_project_holdout",
            expected_term=_UNIFIED_HOLDOUT_TERM,
        )
        return {
            **self._unified_context(comparison),
            "alignment_key": ["ticker", "feature_date", "target_date"],
            "validation": validation,
            "shared_project_holdout": holdout,
        }

    def load_unified_confusion_matrices(self, comparison_id: str) -> dict[str, Any]:
        comparison = self._validate_unified_comparison(comparison_id)
        (
            validation_models,
            validation_baselines,
            holdout_models,
            holdout_baselines,
            _portfolios,
            _benchmarks,
            _model_ids,
        ) = self._unified_metric_collections(comparison.metrics)

        def matrices(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    "competitor_id": row["competitor_id"],
                    "competitor_type": row["competitor_type"],
                    "registered_model_id": row["registered_model_id"],
                    "evaluation_scope": row["evaluation_scope"],
                    "class_labels": [0, 1],
                    "confusion_matrix": row["confusion_matrix"],
                    "support": row["support"],
                }
                for row in rows
            ]

        return {
            **self._unified_context(comparison),
            "validation_models": matrices(validation_models),
            "validation_reference_baselines": matrices(validation_baselines),
            "shared_project_holdout_models": matrices(holdout_models),
            "shared_project_holdout_reference_baselines": matrices(holdout_baselines),
        }


@lru_cache(maxsize=16)
def _repository_for_settings(
    repository_root: str,
    vertical_slice_root: str,
    model_training_root: str,
    default_run_id: str | None,
    unified_comparison_root: str,
    default_unified_comparison_run_id: str | None,
) -> ArtifactRepository:
    return ArtifactRepository(
        RepositorySettings(
            repository_root=Path(repository_root),
            vertical_slice_root=Path(vertical_slice_root),
            model_training_root=Path(model_training_root),
            default_vertical_slice_run_id=default_run_id,
            unified_comparison_root=Path(unified_comparison_root),
            default_unified_comparison_run_id=default_unified_comparison_run_id,
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
        str(settings.unified_comparison_root),
        settings.default_unified_comparison_run_id,
    )


def clear_artifact_repository_cache() -> None:
    """Clear repository instances for tests or explicit process reconfiguration."""

    _repository_for_settings.cache_clear()
