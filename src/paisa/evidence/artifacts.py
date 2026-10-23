"""Immutable evidence-pack publication and verification."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from paisa.evidence.artifact_security import (
    make_pack_private,
    prepare_output_root,
    reject_symlink_components,
    remove_staging_tree,
    verify_pack_modes,
)
from paisa.evidence.models import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceFetchResult,
    EvidenceRecord,
    parse_utc_timestamp,
    utc_iso,
)
from paisa.evidence.policy import (
    EvidencePermissionPolicy,
    load_evidence_permission_policy,
)
from paisa.sentiment.artifacts import (
    SentimentRunCollisionError,
    artifact_inventory,
    publish_staged_run,
    regular_file_inventory,
)


EVIDENCE_PACK_SCHEMA_VERSION = "1.0"
_RUN_ID = re.compile(r"^EVID-[0-9]{8}-[0-9]{6}-[A-Za-z0-9]{8}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEAL_FILES = frozenset({"run_manifest.json", "completion.json"})
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|credential|password|secret|token)([_-]|$)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_UNC = re.compile(r"^\\\\[^\\/\s]+[\\/][^\\/\s]+")
_EMBEDDED_POSIX_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9:/])/(?!/)[^\s\"'<>]+"
)
_EMBEDDED_POSIX_NETWORK_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9:/])//[^/\s\"'<>]+/[^\s\"'<>]+"
)
_EMBEDDED_WINDOWS_ABSOLUTE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>]+"
)
_EMBEDDED_WINDOWS_UNC = re.compile(
    r"(?<![A-Za-z0-9])\\\\[^\\/\s]+[\\/][^\s\"'<>]+"
)
_FILE_URI = re.compile(
    r"(?i)(?<![A-Za-z0-9])file:(?://[^/\s\"'<>]*)?/[^\s\"'<>]*"
)


class EvidenceRunCollisionError(FileExistsError):
    """An immutable evidence run already exists."""


class EvidenceVerificationError(ValueError):
    """An evidence pack failed checksum or semantic verification."""


@dataclass(frozen=True, slots=True)
class EvidenceRunVerification:
    run_id: str
    status: str
    record_count: int
    capture_count: int
    provider_id: str
    manifest_sha256: str
    payload_inventory_sha256: str
    scanned_record_count: int | None
    timestamp_valid_record_count: int | None
    rejected_record_count: int | None
    unmatched_record_count: int | None
    matched_record_count: int | None
    duplicate_record_count: int | None
    verification_basis: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "record_count": self.record_count,
            "capture_count": self.capture_count,
            "provider_id": self.provider_id,
            "manifest_sha256": self.manifest_sha256,
            "payload_inventory_sha256": self.payload_inventory_sha256,
            "scanned_record_count": self.scanned_record_count,
            "timestamp_valid_record_count": self.timestamp_valid_record_count,
            "rejected_record_count": self.rejected_record_count,
            "unmatched_record_count": self.unmatched_record_count,
            "matched_record_count": self.matched_record_count,
            "duplicate_record_count": self.duplicate_record_count,
            "verification": "PASS",
            "verification_basis": self.verification_basis,
        }


@dataclass(frozen=True, slots=True)
class LoadedEvidenceRun:
    records: tuple[EvidenceRecord, ...]
    reference: Mapping[str, Any]
    verification: EvidenceRunVerification


def make_evidence_run_id(now: datetime | None = None) -> str:
    selected = now or datetime.now(timezone.utc)
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("Evidence run timestamps must be timezone-aware")
    selected = selected.astimezone(timezone.utc)
    return f"EVID-{selected:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


def validate_evidence_run_id(value: str) -> str:
    text = str(value)
    if not _RUN_ID.fullmatch(text):
        raise ValueError("run_id must match EVID-YYYYMMDD-HHMMSS-xxxxxxxx")
    return text


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_private(path: Path, content: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _write_json(path: Path, value: Any) -> None:
    _write_private(path, _json_bytes(value))


def _sanitize_text(value: str) -> str:
    text = _JWT.sub("[REDACTED_JWT]", value)
    text = _FILE_URI.sub("[REDACTED_FILE_URI]", text)
    parsed = urlsplit(text)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        host = parsed.hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        query = urlencode(
            [
                (
                    key,
                    "[REDACTED]" if _SENSITIVE_KEY.search(key) else item_value,
                )
                for key, item_value in parse_qsl(
                    parsed.query, keep_blank_values=True
                )
            ],
            doseq=True,
        )
        return urlunsplit((parsed.scheme, host, parsed.path, query, parsed.fragment))
    text = _EMBEDDED_POSIX_NETWORK_ABSOLUTE.sub(
        "[REDACTED_ABSOLUTE_PATH]", text
    )
    text = _EMBEDDED_POSIX_ABSOLUTE.sub("[REDACTED_ABSOLUTE_PATH]", text)
    text = _EMBEDDED_WINDOWS_ABSOLUTE.sub("[REDACTED_ABSOLUTE_PATH]", text)
    text = _EMBEDDED_WINDOWS_UNC.sub("[REDACTED_ABSOLUTE_PATH]", text)
    if (
        text.startswith("/")
        or _WINDOWS_ABSOLUTE.match(text)
        or _WINDOWS_UNC.match(text)
    ):
        return "[REDACTED_ABSOLUTE_PATH]"
    return text


def _sanitize(value: Any, *, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _safe_capture_name(index: int, filename: str) -> str:
    if PurePosixPath(filename).name != filename:
        raise ValueError("Capture filename must be a basename")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not safe:
        safe = "capture.bin"
    return f"capture_{index:04d}_{safe[:180]}"


def _inventory_digest(inventory: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes(list(inventory)))


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    def run(*args: str, preserve_empty: bool = False) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = result.stdout.strip()
        return output if output or preserve_empty else None

    revision = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain", preserve_empty=True)
    return {
        "code_version": revision,
        "dirty_tree": None if dirty is None else bool(dirty),
    }


_CSV_FIELDS = (
    "schema_version",
    "evidence_id",
    "evidence_type",
    "provider_id",
    "provider_mode",
    "data_class",
    "source_name",
    "source_url",
    "source_record_id",
    "ticker",
    "company_name",
    "title",
    "source_published_at",
    "source_published_date",
    "available_at",
    "fetched_at",
    "imported_at",
    "historical_availability_basis",
    "time_precision",
    "contemporaneous_source_id",
    "contemporaneous_batch_at",
    "permission_policy_id",
    "raw_evidence_reference",
    "content_sha256",
    "language",
    "event_category",
    "announcement_type",
    "pdf_reference",
    "pdf_sha256",
    "normalization_warnings",
    "entity_match_evidence",
    "source_metadata",
)


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    text = str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@")) else text


def _write_records_csv(path: Path, records: Sequence[EvidenceRecord]) -> None:
    _write_private(path, _records_csv_bytes(records))


def _records_csv_bytes(records: Sequence[EvidenceRecord]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for record in records:
        payload = record.to_dict()
        writer.writerow(
            {field: _csv_scalar(payload.get(field)) for field in _CSV_FIELDS}
        )
    return handle.getvalue().encode("utf-8")


def _date_range(records: Sequence[EvidenceRecord], field: str) -> dict[str, str | None]:
    values = sorted(
        value
        for value in (getattr(record, field) for record in records)
        if value is not None
    )
    return {
        "start": utc_iso(values[0]) if values else None,
        "end": utc_iso(values[-1]) if values else None,
    }


def _status(result: EvidenceFetchResult) -> str:
    if not result.captures and not result.records:
        return "FAIL"
    if result.errors:
        return "PARTIAL"
    return "PASS"


def _raw_replay_count_claims(
    result: EvidenceFetchResult,
) -> dict[str, Any] | None:
    if (result.provider_id, result.provider_mode) == (
        "psx-announcement",
        "manual_import",
    ):
        return {
            "scanned_record_count": result.counters.get("input_rows"),
            "timestamp_valid_record_count": result.counters.get(
                "accepted_records"
            ),
            "rejected_record_count": 0,
            "unmatched_record_count": 0,
            "matched_record_count": len(result.records),
            "duplicate_record_count": 0,
        }
    if (result.provider_id, result.provider_mode) != (
        "gdelt-gkg-2.1",
        "raw_archive",
    ):
        return None
    return {
        "scanned_record_count": result.counters.get("records_scanned"),
        "timestamp_valid_record_count": result.timestamp_validation.get(
            "validated_record_count"
        ),
        "rejected_record_count": result.counters.get("records_rejected"),
        "unmatched_record_count": result.counters.get("records_unmatched"),
        "matched_record_count": len(result.records),
        "duplicate_record_count": result.counters.get(
            "duplicate_record_count"
        ),
    }


def publish_evidence_run(
    result: EvidenceFetchResult,
    *,
    output_root: str | Path,
    permission_policy: EvidencePermissionPolicy,
    repository_root: str | Path,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Write and atomically publish one immutable evidence pack."""

    selected_id = validate_evidence_run_id(run_id or make_evidence_run_id(created_at))
    if permission_policy.policy_id != result.permission_policy_id:
        raise ValueError("Result permission policy ID differs from supplied policy")
    permission_policy.require(
        result.provider_id,
        result.provider_mode,
        automated=result.data_class == "real",
    )
    repository = Path(repository_root)
    root = prepare_output_root(
        output_root,
        repository_root=repository,
        error_type=ValueError,
        label="Evidence output root",
    )
    repository = repository.resolve(strict=True)
    destination = root / selected_id
    if os.path.lexists(destination):
        raise EvidenceRunCollisionError(
            f"Evidence artifact destination already exists: {selected_id}"
        )
    staging = root / f".{selected_id}.staging-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    published = False
    try:
        capture_rows: list[dict[str, Any]] = []
        used_capture_names: set[str] = set()
        for index, capture in enumerate(result.captures, start=1):
            relative = _safe_capture_name(index, capture.filename)
            if relative in used_capture_names:
                raise ValueError("Capture filenames collide after sanitization")
            used_capture_names.add(relative)
            if capture.sha256 is not None and capture.sha256 != _sha256_bytes(capture.content):
                raise ValueError("Capture checksum differs from captured bytes")
            _write_private(staging / relative, capture.content)
            metadata = _sanitize(capture.metadata_dict())
            metadata["artifact_reference"] = relative
            metadata["sha256"] = sha256_file(staging / relative)
            metadata["size_bytes"] = len(capture.content)
            capture_rows.append(metadata)

        for record in result.records:
            raw_path = record.raw_evidence_reference.partition("#")[0]
            if raw_path not in used_capture_names:
                raise ValueError(
                    "Every raw_evidence_reference must resolve to a retained capture"
                )

        record_payloads = [record.to_dict() for record in result.records]
        jsonl = b"".join(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
            for payload in record_payloads
        )
        _write_private(staging / "normalized_evidence.jsonl", jsonl)
        _write_records_csv(staging / "normalized_evidence.csv", result.records)
        _write_json(staging / "capture_metadata.json", capture_rows)
        _write_json(
            staging / "provider_config.json",
            {
                "provider_id": result.provider_id,
                "provider_mode": result.provider_mode,
                "configured_sources": _sanitize(result.configured_sources),
                "effective_config": _sanitize(result.effective_config),
            },
        )
        _write_json(
            staging / "partition_plan.json",
            {"partitions": _sanitize(result.partition_plan)},
        )
        _write_json(
            staging / "permission_policy_reference.json",
            permission_policy.reference(),
        )
        _write_json(
            staging / "timestamp_validation_report.json",
            _sanitize(result.timestamp_validation),
        )
        _write_json(
            staging / "provider_errors.json",
            _sanitize([error.to_dict() for error in result.errors]),
        )
        status = _status(result)
        raw_replay_counts = _raw_replay_count_claims(result)
        quality = {
            "schema_version": "1.0",
            "status": (
                "FAIL" if status == "FAIL" else "WARN" if result.errors or not result.records else "PASS"
            ),
            "record_count": len(result.records),
            "capture_count": len(result.captures),
            "provider_error_count": len(result.errors),
            "warnings": _sanitize(list(result.warnings)),
            "counters": dict(result.counters),
        }
        _write_json(staging / "quality_report.json", quality)
        summary = {
            "schema_version": "1.0",
            "run_id": selected_id,
            "status": status,
            "quality_status": quality["status"],
            "provider_id": result.provider_id,
            "provider_mode": result.provider_mode,
            "data_class": result.data_class,
            "permission_policy_id": result.permission_policy_id,
            "record_count": len(result.records),
            "capture_count": len(result.captures),
            "provider_error_count": len(result.errors),
            "evidence_type_counts": {
                evidence_type: sum(
                    record.evidence_type == evidence_type for record in result.records
                )
                for evidence_type in sorted(
                    {record.evidence_type for record in result.records}
                )
            },
            "source_published_range": _date_range(
                result.records, "source_published_at"
            ),
            "available_range": _date_range(result.records, "available_at"),
            "fetched_at": utc_iso(result.fetched_at),
            "limitations": _sanitize(list(result.warnings)),
            "academic_disclaimer": "Academic simulation; not financial advice.",
        }
        if raw_replay_counts is not None:
            summary["raw_replay_counts"] = raw_replay_counts
        _write_json(staging / "final_summary.json", summary)

        payload_files = sorted(
            item.name for item in staging.iterdir() if item.name not in _SEAL_FILES
        )
        payload_inventory = artifact_inventory(
            staging, expected_files=payload_files
        )
        payload_inventory_sha = _inventory_digest(payload_inventory)
        generated_at = created_at or result.fetched_at
        if generated_at.tzinfo is None or generated_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        manifest = {
            "schema_version": EVIDENCE_PACK_SCHEMA_VERSION,
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "run_id": selected_id,
            "status": status,
            "provider_id": result.provider_id,
            "provider_mode": result.provider_mode,
            "data_class": result.data_class,
            "created_at": utc_iso(generated_at),
            "fetched_at": utc_iso(result.fetched_at),
            "permission_policy_reference": permission_policy.reference(),
            "record_count": len(result.records),
            "capture_count": len(result.captures),
            "provider_error_count": len(result.errors),
            "normalized_evidence_sha256": sha256_file(
                staging / "normalized_evidence.jsonl"
            ),
            "provider_config_sha256": sha256_file(
                staging / "provider_config.json"
            ),
            "payload_artifact_inventory": payload_inventory,
            "payload_artifact_inventory_sha256": payload_inventory_sha,
            "inventory_excludes": sorted(_SEAL_FILES),
            "completion_marker": "completion.json",
            "implementation_provenance": _git_provenance(Path(repository_root)),
        }
        if raw_replay_counts is not None:
            manifest["raw_replay_counts"] = raw_replay_counts
        _write_json(staging / "run_manifest.json", manifest)
        completion = {
            "schema_version": "1.0",
            "run_id": selected_id,
            "status": status,
            "finalization_state": "COMMITTED",
            "completed_at": utc_iso(generated_at),
            "manifest_sha256": sha256_file(staging / "run_manifest.json"),
            "manifest_size_bytes": (staging / "run_manifest.json").stat().st_size,
            "summary_sha256": sha256_file(staging / "final_summary.json"),
            "summary_size_bytes": (staging / "final_summary.json").stat().st_size,
            "payload_artifact_inventory_sha256": payload_inventory_sha,
            "record_count": len(result.records),
            "capture_count": len(result.captures),
        }
        _write_json(staging / "completion.json", completion)
        all_files = sorted(item.name for item in staging.iterdir())
        make_pack_private(
            staging,
            error_type=EvidenceVerificationError,
            label="Evidence",
        )
        reloaded_policy = load_evidence_permission_policy(
            repository / permission_policy.policy_path,
            repository_root=repository,
        )
        if reloaded_policy.reference() != permission_policy.reference():
            raise EvidenceVerificationError(
                "Permission policy changed during evidence publication"
            )
        reloaded_decision = reloaded_policy.require(
            result.provider_id,
            result.provider_mode,
            automated=result.data_class == "real",
        )
        _verify_evidence_root(
            staging,
            expected_run_id=selected_id,
            expected_permission_reference=reloaded_policy.reference(),
            expected_permission_decision=reloaded_decision,
        )
        publication_inventory = artifact_inventory(
            staging, expected_files=all_files
        )
        try:
            publish_staged_run(
                staging,
                destination,
                expected_inventory=publication_inventory,
            )
        except SentimentRunCollisionError as exc:
            raise EvidenceRunCollisionError(
                f"Evidence artifact destination already exists: {selected_id}"
            ) from exc
        published = True
        return destination
    finally:
        if not published:
            remove_staging_tree(staging, output_root=root)


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceVerificationError(f"Missing regular artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError(f"Invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, Mapping):
        raise EvidenceVerificationError(f"JSON artifact is not an object: {path.name}")
    if _sanitize(value) != value:
        raise EvidenceVerificationError(
            f"JSON artifact contains unsafe private data: {path.name}"
        )
    return value


def _read_json_array(path: Path) -> list[Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceVerificationError(f"Missing regular artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError(f"Invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, list):
        raise EvidenceVerificationError(f"JSON artifact is not an array: {path.name}")
    if _sanitize(value) != value:
        raise EvidenceVerificationError(
            f"JSON artifact contains unsafe private data: {path.name}"
        )
    return value


def _normalized_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EvidenceVerificationError("Payload inventory is missing")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise EvidenceVerificationError("Payload inventory entry is invalid")
        path = str(item.get("path") or "")
        digest = str(item.get("sha256") or "")
        size = item.get("size_bytes")
        if (
            PurePosixPath(path).name != path
            or path in seen
            or not _SHA256.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise EvidenceVerificationError("Payload inventory entry is invalid")
        seen.add(path)
        records.append({"path": path, "sha256": digest, "size_bytes": size})
    return sorted(records, key=lambda item: item["path"])


def _nonnegative_count(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceVerificationError(
            f"Evidence count is invalid: {field_name}"
        )
    return value


def _required_timestamp(value: Any, *, field_name: str) -> datetime:
    try:
        parsed = parse_utc_timestamp(value, field_name=field_name)
    except (TypeError, ValueError) as exc:
        raise EvidenceVerificationError(
            f"Evidence timestamp is invalid: {field_name}"
        ) from exc
    if parsed is None:
        raise EvidenceVerificationError(
            f"Evidence timestamp is missing: {field_name}"
        )
    return parsed


def _validate_permission_reference(value: Mapping[str, Any]) -> None:
    expected_keys = {
        "policy_id",
        "schema_version",
        "path",
        "sha256",
        "assessment_references",
    }
    policy_id = value.get("policy_id")
    policy_path = str(value.get("path") or "")
    assessments = value.get("assessment_references")
    if (
        set(value) != expected_keys
        or not isinstance(policy_id, str)
        or not policy_id
        or value.get("schema_version") != "1.0"
        or not _SHA256.fullmatch(str(value.get("sha256") or ""))
        or PurePosixPath(policy_path).is_absolute()
        or ".." in PurePosixPath(policy_path).parts
        or "\\" in policy_path
        or not policy_path
        or not isinstance(assessments, list)
        or not assessments
    ):
        raise EvidenceVerificationError("Permission-policy reference is invalid")
    seen_paths: set[str] = set()
    for item in assessments:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise EvidenceVerificationError("Permission assessment reference is invalid")
        path = str(item.get("path") or "")
        if (
            not path
            or path in seen_paths
            or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts
            or "\\" in path
            or not _SHA256.fullmatch(str(item.get("sha256") or ""))
        ):
            raise EvidenceVerificationError(
                "Permission assessment reference is invalid"
            )
        seen_paths.add(path)


def _require_count(
    counters: Mapping[str, Any], key: str, expected: int
) -> None:
    if _nonnegative_count(counters.get(key), field_name=key) != expected:
        raise EvidenceVerificationError(
            f"Evidence raw replay counter differs: {key}"
        )


@dataclass(frozen=True, slots=True)
class _EvidenceReplayMetrics:
    scanned_record_count: int | None
    timestamp_valid_record_count: int | None
    rejected_record_count: int | None
    unmatched_record_count: int | None
    matched_record_count: int | None
    duplicate_record_count: int | None
    verification_basis: str

    def count_claims(self) -> dict[str, int | None]:
        return {
            "scanned_record_count": self.scanned_record_count,
            "timestamp_valid_record_count": self.timestamp_valid_record_count,
            "rejected_record_count": self.rejected_record_count,
            "unmatched_record_count": self.unmatched_record_count,
            "matched_record_count": self.matched_record_count,
            "duplicate_record_count": self.duplicate_record_count,
        }


def _verify_gdelt_raw_replay(
    root: Path,
    *,
    normalized_records: Sequence[EvidenceRecord],
    capture_rows: Sequence[Mapping[str, Any]],
    provider_config: Mapping[str, Any],
    partition_document: Mapping[str, Any],
    permission_reference: Mapping[str, Any],
    quality: Mapping[str, Any],
    timestamp_report: Mapping[str, Any],
    provider_errors: Sequence[Mapping[str, Any]],
    expected_permission_decision: Mapping[str, Any] | None = None,
) -> _EvidenceReplayMetrics:
    """Reparse sealed GKG ZIP bytes and reconcile every derived claim."""

    # Importing the provider here keeps ordinary PSX pack verification free of
    # requests/session initialization.  The replay helper itself performs no I/O
    # other than reading the already sealed bytes supplied below.
    from paisa.evidence.gdelt_gkg import (
        GDELT_GKG_PROVIDER_ID,
        GDELT_GKG_PROVIDER_MODE,
        GdeltGkgPartition,
        GdeltGkgReplayCapture,
        replay_gdelt_gkg_captures,
    )

    expected_provider_config_keys = {
        "provider_id",
        "provider_mode",
        "configured_sources",
        "effective_config",
    }
    expected_partition_keys = {
        "sequence",
        "partition_id",
        "archive_url",
        "archive_filename",
        "batch_at",
        "expected_size_bytes",
        "expected_md5",
        "expected_sha256",
        "requested_start",
        "requested_end",
    }
    expected_source_keys = {
        "configured_source",
        "partition_id",
        "archive_url",
        "archive_filename",
        "batch_at",
        "expected_size_bytes",
        "expected_md5",
        "expected_sha256",
        "provider_mode",
        "source_collection_policy",
        "permission_policy_id",
        "permission_policy_sha256",
        "permission_decision_id",
        "permission_status",
    }
    expected_effective_keys = {
        "provider_id",
        "provider_mode",
        "permission_policy_id",
        "permission_policy_sha256",
        "permission_decision_id",
        "permission_status",
        "archive_host",
        "requested_start",
        "requested_end",
        "max_files",
        "max_compressed_bytes_per_file",
        "max_compressed_bytes",
        "max_decompressed_bytes_per_file",
        "max_decompressed_bytes",
        "max_records",
        "max_row_bytes",
        "max_field_characters",
        "maximum_errors",
        "source_collection_policy",
        "dns_resolution_policy",
        "connected_peer_policy",
        "tls_hostname_verification",
        "redirect_policy",
        "dns_timeout_seconds",
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "total_request_timeout_seconds",
        "environment_authentication",
        "publisher_page_fetching",
        "checkpointing",
    }

    if (
        set(provider_config) != expected_provider_config_keys
        or provider_config.get("provider_id") != GDELT_GKG_PROVIDER_ID
        or provider_config.get("provider_mode") != GDELT_GKG_PROVIDER_MODE
    ):
        raise EvidenceVerificationError("GDELT provider configuration differs")
    configured_sources = provider_config.get("configured_sources")
    effective_config = provider_config.get("effective_config")
    partition_rows = partition_document.get("partitions")
    if (
        not isinstance(configured_sources, list)
        or any(not isinstance(item, Mapping) for item in configured_sources)
        or not isinstance(effective_config, Mapping)
        or set(partition_document) != {"partitions"}
        or not isinstance(partition_rows, list)
        or not partition_rows
        or any(not isinstance(item, Mapping) for item in partition_rows)
    ):
        raise EvidenceVerificationError("GDELT replay configuration is invalid")
    if (
        set(effective_config) != expected_effective_keys
        or effective_config.get("permission_policy_sha256")
        != permission_reference.get("sha256")
        or not isinstance(effective_config.get("permission_decision_id"), str)
        or not effective_config.get("permission_decision_id")
        or effective_config.get("permission_status") != "PERMITTED"
        or effective_config.get("provider_id") != GDELT_GKG_PROVIDER_ID
        or effective_config.get("provider_mode") != GDELT_GKG_PROVIDER_MODE
        or effective_config.get("permission_policy_id")
        != permission_reference.get("policy_id")
        or effective_config.get("source_collection_policy")
        != "web_only_identifier_1"
        or effective_config.get("publisher_page_fetching") != "disabled"
        or effective_config.get("archive_host") != "data.gdeltproject.org"
        or effective_config.get("dns_resolution_policy")
        != "bounded_global_addresses_only"
        or effective_config.get("connected_peer_policy")
        != "global_and_in_approved_resolution_set"
        or effective_config.get("tls_hostname_verification") != "required"
        or effective_config.get("redirect_policy") != "reject"
        or effective_config.get("environment_authentication") != "disabled"
        or effective_config.get("checkpointing")
        not in {"disabled", "validated_archive_cache"}
    ):
        raise EvidenceVerificationError("GDELT effective configuration differs")
    for timeout_name in (
        "dns_timeout_seconds",
        "connect_timeout_seconds",
        "read_timeout_seconds",
        "total_request_timeout_seconds",
    ):
        timeout_value = effective_config.get(timeout_name)
        maximum_timeout = (
            600.0 if timeout_name == "total_request_timeout_seconds" else 300.0
        )
        if (
            isinstance(timeout_value, bool)
            or not isinstance(timeout_value, (int, float))
            or not math.isfinite(float(timeout_value))
            or float(timeout_value) <= 0
            or float(timeout_value) > maximum_timeout
        ):
            raise EvidenceVerificationError(
                "GDELT network timeout configuration is invalid"
            )
    if expected_permission_decision is not None and (
        expected_permission_decision.get("provider_id") != GDELT_GKG_PROVIDER_ID
        or expected_permission_decision.get("provider_mode")
        != GDELT_GKG_PROVIDER_MODE
        or expected_permission_decision.get("status") != "PERMITTED"
        or expected_permission_decision.get("automated_access") is not True
        or expected_permission_decision.get("publisher_page_fetching") is not False
        or effective_config.get("permission_decision_id")
        != expected_permission_decision.get("decision_id")
    ):
        raise EvidenceVerificationError(
            "GDELT permission decision differs from the verified policy"
        )

    requested_start = _required_timestamp(
        effective_config.get("requested_start"), field_name="requested_start"
    )
    requested_end = _required_timestamp(
        effective_config.get("requested_end"), field_name="requested_end"
    )
    if requested_start > requested_end:
        raise EvidenceVerificationError("GDELT requested interval is invalid")

    partitions: dict[str, GdeltGkgPartition] = {}
    partition_order: dict[str, int] = {}
    try:
        for ordinal, row in enumerate(partition_rows, start=1):
            if set(row) != expected_partition_keys:
                raise EvidenceVerificationError(
                    "GDELT partition-plan schema differs"
                )
            partition = GdeltGkgPartition.from_mapping(
                {
                    "partition_id": row.get("partition_id"),
                    "archive_url": row.get("archive_url"),
                    "expected_size_bytes": row.get("expected_size_bytes"),
                    "expected_md5": row.get("expected_md5"),
                    "expected_sha256": row.get("expected_sha256"),
                }
            )
            if (
                partition.partition_id in partitions
                or row.get("sequence") != ordinal
                or row.get("archive_filename") != partition.filename
                or row.get("batch_at") != utc_iso(partition.batch_at)
                or row.get("requested_start") != utc_iso(requested_start)
                or row.get("requested_end") != utc_iso(requested_end)
            ):
                raise EvidenceVerificationError(
                    "GDELT partition-plan semantics differ"
                )
            partitions[partition.partition_id] = partition
            partition_order[partition.partition_id] = ordinal
    except EvidenceVerificationError:
        raise
    except (TypeError, ValueError) as exc:
        raise EvidenceVerificationError("GDELT partition plan is invalid") from exc

    sources: dict[str, Mapping[str, Any]] = {}
    for source in configured_sources:
        partition_id = str(source.get("partition_id") or "")
        partition = partitions.get(partition_id)
        if partition is None or partition_id in sources:
            raise EvidenceVerificationError("GDELT configured-source plan differs")
        expected_source = f"gdelt-gkg:{partition_id}"
        if (
            set(source) != expected_source_keys
            or source.get("provider_mode") != GDELT_GKG_PROVIDER_MODE
            or source.get("source_collection_policy")
            != "web_only_identifier_1"
            or source.get("configured_source") != expected_source
            or source.get("archive_url") != partition.archive_url
            or source.get("archive_filename") != partition.filename
            or source.get("batch_at") != utc_iso(partition.batch_at)
            or source.get("expected_size_bytes") != partition.expected_size_bytes
            or source.get("expected_md5") != partition.expected_md5
            or source.get("expected_sha256") != partition.expected_sha256
            or source.get("permission_policy_id")
            != permission_reference.get("policy_id")
            or source.get("permission_policy_sha256")
            != permission_reference.get("sha256")
            or source.get("permission_decision_id")
            != effective_config.get("permission_decision_id")
            or source.get("permission_status") != "PERMITTED"
        ):
            raise EvidenceVerificationError("GDELT configured-source semantics differ")
        sources[partition_id] = source
    if set(sources) != set(partitions):
        raise EvidenceVerificationError("GDELT configured sources are incomplete")

    replay_captures: list[GdeltGkgReplayCapture] = []
    captured_partition_ids: set[str] = set()
    previous_partition_ordinal = 0
    try:
        for capture_ordinal, row in enumerate(capture_rows, start=1):
            configured_source = str(row.get("configured_source") or "")
            prefix = "gdelt-gkg:"
            if not configured_source.startswith(prefix):
                raise EvidenceVerificationError(
                    "GDELT capture configured source is invalid"
                )
            partition_id = configured_source.removeprefix(prefix)
            partition = partitions.get(partition_id)
            artifact_reference = str(row.get("artifact_reference") or "")
            if (
                partition is None
                or partition_id in captured_partition_ids
                or partition_order[partition_id] <= previous_partition_ordinal
                or row.get("capture_id") != f"gkg-{partition_id}"
                or row.get("filename") != partition.filename
                or row.get("source_url") != partition.archive_url
                or row.get("content_type") != "application/zip"
                or artifact_reference
                != _safe_capture_name(capture_ordinal, partition.filename)
            ):
                raise EvidenceVerificationError("GDELT capture provenance differs")
            fetched_at = _required_timestamp(
                row.get("fetched_at"), field_name="capture fetched_at"
            )
            replay_captures.append(
                GdeltGkgReplayCapture(
                    partition=partition,
                    content=(root / artifact_reference).read_bytes(),
                    fetched_at=fetched_at,
                    capture_reference=artifact_reference,
                )
            )
            captured_partition_ids.add(partition_id)
            previous_partition_ordinal = partition_order[partition_id]
    except EvidenceVerificationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise EvidenceVerificationError("GDELT replay captures are invalid") from exc

    replay_options: dict[str, Any] = {}
    for name in (
        "max_files",
        "max_compressed_bytes_per_file",
        "max_compressed_bytes",
        "max_decompressed_bytes_per_file",
        "max_decompressed_bytes",
        "max_records",
        "max_row_bytes",
        "max_field_characters",
        "maximum_errors",
    ):
        if name in effective_config:
            replay_options[name] = effective_config[name]
    try:
        replay = replay_gdelt_gkg_captures(
            replay_captures,
            requested_start=requested_start,
            requested_end=requested_end,
            permission_policy_id=str(permission_reference.get("policy_id") or ""),
            **replay_options,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise EvidenceVerificationError(
            "GDELT raw archive semantic replay failed"
        ) from exc

    if tuple(normalized_records) != replay.records:
        raise EvidenceVerificationError(
            "Normalized evidence differs from GDELT raw replay"
        )
    counters = quality.get("counters")
    if not isinstance(counters, Mapping):
        raise EvidenceVerificationError("Evidence quality counters are invalid")
    _require_count(counters, "partitions_configured", len(partitions))
    _require_count(counters, "partitions_completed", len(replay_captures))
    downloaded = _nonnegative_count(
        counters.get("partitions_downloaded"), field_name="partitions_downloaded"
    )
    resumed = _nonnegative_count(
        counters.get("partitions_resumed"), field_name="partitions_resumed"
    )
    if downloaded + resumed != len(replay_captures):
        raise EvidenceVerificationError(
            "Evidence completed-partition counters disagree"
        )
    _require_count(counters, "compressed_bytes", replay.compressed_bytes)
    _require_count(counters, "decompressed_bytes", replay.decompressed_bytes)
    _require_count(counters, "records_scanned", replay.scanned_record_count)
    _require_count(counters, "records_unmatched", replay.unmatched_record_count)
    _require_count(counters, "records_rejected", replay.rejected_record_count)
    _require_count(counters, "records_retained", replay.matched_record_count)
    _require_count(
        counters, "duplicate_record_count", replay.duplicate_record_count
    )
    if counters.get("rejected_reason_counts") != replay.rejected_reason_counts:
        raise EvidenceVerificationError(
            "Evidence rejected-reason counters differ from raw replay"
        )

    expected_timestamp_report = {
        "future_publication_policy": "reject",
        "publication_delay": dict(replay.publication_delay),
        "batch_delay": dict(replay.batch_delay),
        "negative_publication_delay_count": 0,
        "validated_record_count": replay.timestamp_valid_record_count,
        "rejected_record_count": replay.rejected_record_count,
        "rejected_reason_counts": dict(replay.rejected_reason_counts),
    }
    if timestamp_report != expected_timestamp_report:
        raise EvidenceVerificationError(
            "Evidence timestamp-validation report differs from raw replay"
        )

    replay_errors = [error.to_dict() for error in replay.errors]
    stored_row_errors: list[Mapping[str, Any]] = []
    stored_nonrow_errors: list[Mapping[str, Any]] = []
    failed_partition_ids: set[str] = set()
    for error in provider_errors:
        details = error.get("details")
        if not isinstance(details, Mapping):
            raise EvidenceVerificationError("Evidence provider error is invalid")
        if "row_number" in details:
            stored_row_errors.append(error)
        else:
            stored_nonrow_errors.append(error)
            error_partition = error.get("partition_id")
            configured_source = error.get("configured_source")
            if error_partition in captured_partition_ids:
                raise EvidenceVerificationError(
                    "Captured GDELT partition has an unexplained provider error"
                )
            if (
                error_partition not in partitions
                or configured_source != f"gdelt-gkg:{error_partition}"
            ):
                raise EvidenceVerificationError(
                    "GDELT partition failure provenance is invalid"
                )
            failed_partition_ids.add(str(error_partition))
    if captured_partition_ids | failed_partition_ids != set(partitions):
        raise EvidenceVerificationError(
            "Every configured GDELT partition requires a capture or failure"
        )
    maximum_errors = _nonnegative_count(
        effective_config.get("maximum_errors", 1_000),
        field_name="maximum_errors",
    )
    expected_row_error_count = min(
        len(replay_errors), max(maximum_errors - len(stored_nonrow_errors), 0)
    )
    if (
        len(stored_row_errors) != expected_row_error_count
        or list(stored_row_errors) != replay_errors[:expected_row_error_count]
    ):
        raise EvidenceVerificationError(
            "Evidence provider row errors differ from raw replay"
        )

    return _EvidenceReplayMetrics(
        scanned_record_count=replay.scanned_record_count,
        timestamp_valid_record_count=replay.timestamp_valid_record_count,
        rejected_record_count=replay.rejected_record_count,
        unmatched_record_count=replay.unmatched_record_count,
        matched_record_count=replay.matched_record_count,
        duplicate_record_count=replay.duplicate_record_count,
        verification_basis="RAW_GDELT_GKG_ZIP_SEMANTIC_REPLAY",
    )


def _verify_psx_raw_replay(
    root: Path,
    *,
    normalized_records: Sequence[EvidenceRecord],
    capture_rows: Sequence[Mapping[str, Any]],
    provider_config: Mapping[str, Any],
    partition_document: Mapping[str, Any],
    permission_reference: Mapping[str, Any],
    quality: Mapping[str, Any],
    timestamp_report: Mapping[str, Any],
    provider_errors: Sequence[Mapping[str, Any]],
    expected_permission_decision: Mapping[str, Any] | None = None,
) -> _EvidenceReplayMetrics:
    """Reparse the sealed authorized PSX CSV and reconcile every record."""

    from paisa.evidence.psx_announcements import (
        CAPTURE_FILENAME,
        PROVIDER_ID,
        PROVIDER_MODE,
        replay_psx_announcement_capture,
    )

    if set(provider_config) != {
        "provider_id",
        "provider_mode",
        "configured_sources",
        "effective_config",
    }:
        raise EvidenceVerificationError("PSX provider configuration schema differs")
    configured_sources = provider_config.get("configured_sources")
    effective_config = provider_config.get("effective_config")
    partitions = partition_document.get("partitions")
    if (
        provider_config.get("provider_id") != PROVIDER_ID
        or provider_config.get("provider_mode") != PROVIDER_MODE
        or not isinstance(configured_sources, list)
        or len(configured_sources) != 1
        or not isinstance(configured_sources[0], Mapping)
        or not isinstance(effective_config, Mapping)
        or not isinstance(partitions, list)
        or len(partitions) != 1
        or not isinstance(partitions[0], Mapping)
        or len(capture_rows) != 1
    ):
        raise EvidenceVerificationError("PSX replay inputs are incomplete")

    source = configured_sources[0]
    partition = partitions[0]
    capture = capture_rows[0]
    expected_source_keys = {
        "configured_source",
        "source_name",
        "evidence_type",
        "provider_mode",
        "automated_access",
        "permission_policy_id",
        "permission_decision_id",
        "permission_status",
        "manual_permission_attestation_sha256",
        "manual_permission_reference_sha256",
        "input_sha256",
        "input_size_bytes",
        "pdf_handling",
    }
    expected_effective_keys = {
        "automated_access",
        "max_bytes",
        "max_records",
        "pdf_retrieval",
        "source_url_retrieval",
        "manual_permission_attestation_sha256",
        "manual_permission_reference_sha256",
    }
    expected_partition_keys = {
        "partition_id",
        "acquisition_mode",
        "automated_access",
        "input_sha256",
        "maximum_bytes",
        "maximum_records",
    }
    input_sha256 = str(source.get("input_sha256") or "")
    attestation_sha256 = str(
        source.get("manual_permission_attestation_sha256") or ""
    )
    reference_sha256 = str(
        source.get("manual_permission_reference_sha256") or ""
    )
    configured_source = str(source.get("configured_source") or "")
    permission_status = source.get("permission_status")
    if (
        set(source) != expected_source_keys
        or set(effective_config) != expected_effective_keys
        or set(partition) != expected_partition_keys
        or source.get("source_name") != "Pakistan Stock Exchange"
        or source.get("evidence_type") != "exchange_announcement"
        or source.get("provider_mode") != PROVIDER_MODE
        or source.get("automated_access") is not False
        or source.get("permission_policy_id")
        != permission_reference.get("policy_id")
        or not isinstance(source.get("permission_decision_id"), str)
        or not source.get("permission_decision_id")
        or permission_status not in {"MANUAL_ONLY", "PERMITTED"}
        or not _SHA256.fullmatch(input_sha256)
        or not _SHA256.fullmatch(attestation_sha256)
        or not _SHA256.fullmatch(reference_sha256)
        or not configured_source.startswith("manual-csv:")
        or configured_source == "manual-csv:"
        or source.get("pdf_handling")
        != "reference_and_supplied_checksum_only_no_retrieval"
    ):
        raise EvidenceVerificationError("PSX configured-source semantics differ")

    max_bytes = effective_config.get("max_bytes")
    max_records = effective_config.get("max_records")
    if (
        effective_config.get("automated_access") is not False
        or effective_config.get("pdf_retrieval") != "disabled"
        or effective_config.get("source_url_retrieval") != "disabled"
        or effective_config.get("manual_permission_attestation_sha256")
        != attestation_sha256
        or effective_config.get("manual_permission_reference_sha256")
        != reference_sha256
        or isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes <= 0
        or isinstance(max_records, bool)
        or not isinstance(max_records, int)
        or max_records <= 0
    ):
        raise EvidenceVerificationError("PSX effective configuration differs")

    if (
        partition.get("partition_id") != "manual-csv-input"
        or partition.get("acquisition_mode") != PROVIDER_MODE
        or partition.get("automated_access") is not False
        or partition.get("input_sha256") != input_sha256
        or partition.get("maximum_bytes") != max_bytes
        or partition.get("maximum_records") != max_records
    ):
        raise EvidenceVerificationError("PSX partition-plan semantics differ")

    if (
        capture.get("capture_id") != "psx-announcements-input"
        or capture.get("provider_id") != PROVIDER_ID
        or capture.get("configured_source") != configured_source
        or capture.get("filename") != CAPTURE_FILENAME
        or capture.get("source_url") is not None
        or capture.get("content_type") != "text/csv"
        or capture.get("sha256") != input_sha256
        or capture.get("size_bytes") != source.get("input_size_bytes")
        or capture.get("artifact_reference")
        != f"capture_0001_{CAPTURE_FILENAME}"
        or capture.get("warnings")
        != ["manual_import_only_no_network_or_pdf_retrieval"]
    ):
        raise EvidenceVerificationError("PSX capture provenance differs")

    if expected_permission_decision is not None and (
        expected_permission_decision.get("provider_id") != PROVIDER_ID
        or expected_permission_decision.get("provider_mode") != PROVIDER_MODE
        or expected_permission_decision.get("status") != permission_status
        or expected_permission_decision.get("status")
        not in {"MANUAL_ONLY", "PERMITTED"}
        or expected_permission_decision.get("automated_access") is not False
        or expected_permission_decision.get("decision_id")
        != source.get("permission_decision_id")
        or expected_permission_decision.get("authorization_attestation_sha256")
        != attestation_sha256
        or expected_permission_decision.get("authorization_reference_sha256")
        != reference_sha256
    ):
        raise EvidenceVerificationError(
            "PSX permission decision differs from the verified policy"
        )

    try:
        imported_at = _required_timestamp(
            capture.get("fetched_at"), field_name="PSX capture fetched_at"
        )
        capture_bytes = (
            root / str(capture.get("artifact_reference") or "")
        ).read_bytes()
        replay_records = replay_psx_announcement_capture(
            capture_bytes,
            imported_at=imported_at,
            permission_policy_id=str(permission_reference.get("policy_id") or ""),
            permission_decision_id=str(source.get("permission_decision_id") or ""),
            authorization_attestation_sha256=attestation_sha256,
            authorization_reference_sha256=reference_sha256,
            max_bytes=max_bytes,
            max_records=max_records,
        )
    except EvidenceVerificationError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise EvidenceVerificationError("PSX raw CSV semantic replay failed") from exc

    if tuple(normalized_records) != replay_records:
        raise EvidenceVerificationError(
            "Normalized evidence differs from PSX raw replay"
        )
    if provider_errors:
        raise EvidenceVerificationError(
            "PSX manual import cannot retain provider errors after acceptance"
        )

    date_only_count = sum(
        record.time_precision == "date_only" for record in replay_records
    )
    timed_count = len(replay_records) - date_only_count
    expected_counters = {
        "input_rows": len(replay_records),
        "accepted_records": len(replay_records),
        "ogdc_records": len(replay_records),
        "timed_records": timed_count,
        "date_only_records": date_only_count,
        "pdf_reference_records": sum(
            record.pdf_reference is not None for record in replay_records
        ),
    }
    if quality.get("counters") != expected_counters:
        raise EvidenceVerificationError(
            "PSX quality counters differ from raw replay"
        )
    expected_timestamp_report = {
        "policy": (
            "official timestamp maps by exact available_at; date-only "
            "evidence has no invented timestamp and maps to the first "
            "observed session strictly after its source date"
        ),
        "timezone": "Asia/Karachi",
        "timed_records": timed_count,
        "date_only_records": date_only_count,
        "future_timestamp_rejections": 0,
    }
    if timestamp_report != expected_timestamp_report:
        raise EvidenceVerificationError(
            "PSX timestamp-validation report differs from raw replay"
        )

    return _EvidenceReplayMetrics(
        scanned_record_count=len(replay_records),
        timestamp_valid_record_count=len(replay_records),
        rejected_record_count=0,
        unmatched_record_count=0,
        matched_record_count=len(replay_records),
        duplicate_record_count=0,
        verification_basis="RAW_PSX_CSV_SEMANTIC_REPLAY",
    )


def _verify_evidence_root(
    root: Path,
    *,
    expected_run_id: str,
    expected_permission_reference: Mapping[str, Any] | None = None,
    expected_permission_decision: Mapping[str, Any] | None = None,
) -> EvidenceRunVerification:
    """Verify a completed pack at either its private staging or final path."""

    verify_pack_modes(
        root,
        error_type=EvidenceVerificationError,
        label="Evidence",
    )
    run_id = validate_evidence_run_id(expected_run_id)
    manifest = _read_json(root / "run_manifest.json")
    completion = _read_json(root / "completion.json")
    summary = _read_json(root / "final_summary.json")
    if manifest.get("schema_version") != EVIDENCE_PACK_SCHEMA_VERSION:
        raise EvidenceVerificationError("Evidence pack schema is unsupported")
    if manifest.get("evidence_schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise EvidenceVerificationError("Evidence record schema is unsupported")
    if any(value.get("run_id") != run_id for value in (manifest, completion, summary)):
        raise EvidenceVerificationError("Evidence pack identities disagree")
    if completion.get("finalization_state") != "COMMITTED":
        raise EvidenceVerificationError("Evidence pack is not committed")
    status = str(manifest.get("status") or "")
    if status not in {"PASS", "PARTIAL", "FAIL"} or any(
        value.get("status") != status for value in (completion, summary)
    ):
        raise EvidenceVerificationError("Evidence statuses disagree")
    manifest_path = root / "run_manifest.json"
    summary_path = root / "final_summary.json"
    if (
        completion.get("manifest_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_size_bytes") != manifest_path.stat().st_size
        or completion.get("summary_sha256") != sha256_file(summary_path)
        or completion.get("summary_size_bytes") != summary_path.stat().st_size
    ):
        raise EvidenceVerificationError("Evidence completion seal is invalid")
    inventory = _normalized_inventory(manifest.get("payload_artifact_inventory"))
    inventory_digest = _inventory_digest(inventory)
    if (
        manifest.get("payload_artifact_inventory_sha256") != inventory_digest
        or completion.get("payload_artifact_inventory_sha256") != inventory_digest
    ):
        raise EvidenceVerificationError("Evidence inventory digest is invalid")
    observed_names = sorted(item.name for item in root.iterdir())
    expected_names = sorted(
        [item["path"] for item in inventory] + list(_SEAL_FILES)
    )
    if observed_names != expected_names:
        raise EvidenceVerificationError("Evidence artifact set differs")
    observed_payload = regular_file_inventory(
        root, files=[item["path"] for item in inventory]
    )
    if observed_payload != inventory:
        raise EvidenceVerificationError("Evidence payload inventory differs")

    normalized_path = root / "normalized_evidence.jsonl"
    record_count = 0
    evidence_ids: set[str] = set()
    normalized_records: list[EvidenceRecord] = []
    try:
        with normalized_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n"):
                    raise EvidenceVerificationError(
                        "Normalized evidence JSONL must end each row with newline"
                    )
                payload = json.loads(line)
                try:
                    record = EvidenceRecord.from_dict(payload)
                except (TypeError, ValueError) as exc:
                    raise EvidenceVerificationError(
                        f"Evidence row {line_number} is invalid: {exc}"
                    ) from exc
                expected_line = (
                    json.dumps(
                        record.to_dict(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
                if line != expected_line:
                    raise EvidenceVerificationError(
                        f"Evidence row {line_number} is not canonical"
                    )
                evidence_id = record.evidence_id
                if not evidence_id or evidence_id in evidence_ids:
                    raise EvidenceVerificationError(
                        "Normalized evidence IDs must be complete and unique"
                    )
                evidence_ids.add(evidence_id)
                normalized_records.append(record)
                record_count += 1
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError("Normalized evidence JSONL is invalid") from exc
    try:
        observed_csv = (root / "normalized_evidence.csv").read_bytes()
    except OSError as exc:
        raise EvidenceVerificationError("Normalized evidence CSV is unreadable") from exc
    if observed_csv != _records_csv_bytes(normalized_records):
        raise EvidenceVerificationError(
            "Normalized evidence CSV differs from normalized JSONL"
        )

    capture_rows = _read_json_array(root / "capture_metadata.json")
    capture_references: set[str] = set()
    expected_capture_fields = {
        "capture_id",
        "provider_id",
        "configured_source",
        "filename",
        "fetched_at",
        "source_url",
        "content_type",
        "sha256",
        "warnings",
        "artifact_reference",
        "size_bytes",
    }
    for capture_ordinal, row in enumerate(capture_rows, start=1):
        if not isinstance(row, Mapping):
            raise EvidenceVerificationError("Capture metadata row is invalid")
        reference = str(row.get("artifact_reference") or "")
        filename = str(row.get("filename") or "")
        digest = str(row.get("sha256") or "")
        size = row.get("size_bytes")
        if (
            PurePosixPath(reference).name != reference
            or PurePosixPath(filename).name != filename
            or reference != _safe_capture_name(capture_ordinal, filename)
            or reference in capture_references
            or not _SHA256.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or set(row) != expected_capture_fields
            or not isinstance(row.get("warnings"), list)
            or any(not isinstance(item, str) for item in row.get("warnings", ()))
        ):
            raise EvidenceVerificationError("Capture metadata row is invalid")
        _required_timestamp(row.get("fetched_at"), field_name="capture fetched_at")
        capture_references.add(reference)
        captured_path = root / reference
        if (
            not captured_path.is_file()
            or captured_path.is_symlink()
            or captured_path.stat().st_size != size
            or sha256_file(captured_path) != digest
        ):
            raise EvidenceVerificationError("Captured evidence bytes differ")

    fixed_payload_files = {
        "capture_metadata.json",
        "final_summary.json",
        "normalized_evidence.csv",
        "normalized_evidence.jsonl",
        "partition_plan.json",
        "permission_policy_reference.json",
        "provider_config.json",
        "provider_errors.json",
        "quality_report.json",
        "timestamp_validation_report.json",
    }
    if {item["path"] for item in inventory} != (
        fixed_payload_files | capture_references
    ):
        raise EvidenceVerificationError("Evidence payload artifact set is invalid")

    try:
        with normalized_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                reference = str(payload.get("raw_evidence_reference") or "")
                if reference.partition("#")[0] not in capture_references:
                    raise EvidenceVerificationError(
                        "Normalized evidence references a missing capture"
                    )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError("Normalized evidence JSONL is invalid") from exc

    permission_reference = _read_json(root / "permission_policy_reference.json")
    _validate_permission_reference(permission_reference)
    manifest_permission = manifest.get("permission_policy_reference")
    if permission_reference != manifest_permission:
        raise EvidenceVerificationError("Permission-policy references disagree")
    if (
        expected_permission_reference is not None
        and permission_reference != expected_permission_reference
    ):
        raise EvidenceVerificationError(
            "Permission-policy reference differs from the verified policy"
        )
    if summary.get("permission_policy_id") != permission_reference.get("policy_id"):
        raise EvidenceVerificationError("Permission-policy identity differs")
    provider_config = _read_json(root / "provider_config.json")
    if (
        provider_config.get("provider_id") != manifest.get("provider_id")
        or provider_config.get("provider_mode") != manifest.get("provider_mode")
        or not isinstance(provider_config.get("configured_sources"), list)
        or not isinstance(provider_config.get("effective_config"), Mapping)
        or manifest.get("provider_config_sha256")
        != sha256_file(root / "provider_config.json")
    ):
        raise EvidenceVerificationError("Provider configuration differs")
    if any(
        record.provider_id != manifest.get("provider_id")
        or record.provider_mode != manifest.get("provider_mode")
        or record.data_class != manifest.get("data_class")
        or record.permission_policy_id != permission_reference.get("policy_id")
        for record in normalized_records
    ):
        raise EvidenceVerificationError(
            "Normalized record semantics differ from the owning run"
        )
    if any(
        row.get("provider_id") != manifest.get("provider_id")
        for row in capture_rows
    ):
        raise EvidenceVerificationError("Capture provider differs from the owning run")
    if (
        summary.get("provider_id") != manifest.get("provider_id")
        or summary.get("provider_mode") != manifest.get("provider_mode")
        or summary.get("data_class") != manifest.get("data_class")
    ):
        raise EvidenceVerificationError("Evidence summary semantics differ")

    partition_document = _read_json(root / "partition_plan.json")
    partitions = partition_document.get("partitions")
    if not isinstance(partitions, list) or any(
        not isinstance(item, Mapping) for item in partitions
    ):
        raise EvidenceVerificationError("Evidence partition plan is invalid")
    timestamp_report = _read_json(root / "timestamp_validation_report.json")
    quality = _read_json(root / "quality_report.json")
    raw_provider_errors = _read_json_array(root / "provider_errors.json")
    if any(not isinstance(item, Mapping) for item in raw_provider_errors):
        raise EvidenceVerificationError("Evidence provider error is invalid")
    provider_errors: list[Mapping[str, Any]] = list(raw_provider_errors)
    for error in provider_errors:
        if (
            error.get("provider_id") != manifest.get("provider_id")
            or not isinstance(error.get("configured_source"), str)
            or not error.get("configured_source")
            or not isinstance(error.get("error_type"), str)
            or not error.get("error_type")
            or not isinstance(error.get("message"), str)
            or not error.get("message")
            or not isinstance(error.get("details"), Mapping)
        ):
            raise EvidenceVerificationError("Evidence provider error is invalid")
        _required_timestamp(
            error.get("occurred_at"), field_name="provider error occurred_at"
        )

    record_claim = manifest.get("record_count")
    capture_claim = manifest.get("capture_count")
    provider_error_count = len(provider_errors)
    count_claims = (
        (record_claim, record_count, "manifest record_count"),
        (completion.get("record_count"), record_count, "completion record_count"),
        (summary.get("record_count"), record_count, "summary record_count"),
        (capture_claim, len(capture_rows), "manifest capture_count"),
        (
            completion.get("capture_count"),
            len(capture_rows),
            "completion capture_count",
        ),
        (summary.get("capture_count"), len(capture_rows), "summary capture_count"),
        (
            manifest.get("provider_error_count"),
            provider_error_count,
            "manifest provider_error_count",
        ),
        (
            summary.get("provider_error_count"),
            provider_error_count,
            "summary provider_error_count",
        ),
    )
    if any(
        _nonnegative_count(claim, field_name=name) != expected
        for claim, expected, name in count_claims
    ):
        raise EvidenceVerificationError("Evidence pack counts disagree")
    if manifest.get("normalized_evidence_sha256") != sha256_file(normalized_path):
        raise EvidenceVerificationError("Normalized evidence checksum differs")

    warnings = quality.get("warnings")
    counters = quality.get("counters")
    expected_quality_status = (
        "FAIL"
        if status == "FAIL"
        else "WARN"
        if provider_errors or not normalized_records
        else "PASS"
    )
    if (
        quality.get("schema_version") != "1.0"
        or quality.get("status") != expected_quality_status
        or summary.get("quality_status") != expected_quality_status
        or _nonnegative_count(
            quality.get("record_count"), field_name="quality record_count"
        )
        != record_count
        or _nonnegative_count(
            quality.get("capture_count"), field_name="quality capture_count"
        )
        != len(capture_rows)
        or _nonnegative_count(
            quality.get("provider_error_count"),
            field_name="quality provider_error_count",
        )
        != provider_error_count
        or not isinstance(warnings, list)
        or any(not isinstance(item, str) for item in warnings)
        or summary.get("limitations") != warnings
        or not isinstance(counters, Mapping)
    ):
        raise EvidenceVerificationError("Evidence quality report differs")
    expected_status = (
        "FAIL"
        if not capture_rows and not normalized_records
        else "PARTIAL"
        if provider_errors
        else "PASS"
    )
    if status != expected_status:
        raise EvidenceVerificationError("Evidence status contradicts provider errors")
    expected_type_counts = {
        evidence_type: sum(
            record.evidence_type == evidence_type for record in normalized_records
        )
        for evidence_type in sorted(
            {record.evidence_type for record in normalized_records}
        )
    }
    if (
        summary.get("schema_version") != "1.0"
        or summary.get("evidence_type_counts") != expected_type_counts
        or summary.get("source_published_range")
        != _date_range(normalized_records, "source_published_at")
        or summary.get("available_range")
        != _date_range(normalized_records, "available_at")
        or summary.get("fetched_at") != manifest.get("fetched_at")
        or summary.get("academic_disclaimer")
        != "Academic simulation; not financial advice."
    ):
        raise EvidenceVerificationError("Evidence final summary differs")
    _required_timestamp(manifest.get("fetched_at"), field_name="manifest fetched_at")

    replay_metrics = _EvidenceReplayMetrics(
        scanned_record_count=None,
        timestamp_valid_record_count=None,
        rejected_record_count=None,
        unmatched_record_count=None,
        matched_record_count=None,
        duplicate_record_count=None,
        verification_basis="PACK_STRUCTURAL_AND_NORMALIZED_RECORD_VALIDATION",
    )
    if (
        manifest.get("provider_id"),
        manifest.get("provider_mode"),
    ) == ("gdelt-gkg-2.1", "raw_archive"):
        replay_metrics = _verify_gdelt_raw_replay(
            root,
            normalized_records=normalized_records,
            capture_rows=capture_rows,
            provider_config=provider_config,
            partition_document=partition_document,
            permission_reference=permission_reference,
            quality=quality,
            timestamp_report=timestamp_report,
            provider_errors=provider_errors,
            expected_permission_decision=expected_permission_decision,
        )
    elif (
        manifest.get("provider_id"),
        manifest.get("provider_mode"),
    ) == ("psx-announcement", "manual_import"):
        replay_metrics = _verify_psx_raw_replay(
            root,
            normalized_records=normalized_records,
            capture_rows=capture_rows,
            provider_config=provider_config,
            partition_document=partition_document,
            permission_reference=permission_reference,
            quality=quality,
            timestamp_report=timestamp_report,
            provider_errors=provider_errors,
            expected_permission_decision=expected_permission_decision,
        )
    if replay_metrics.verification_basis.startswith("RAW_"):
        expected_raw_counts = replay_metrics.count_claims()
        if (
            manifest.get("raw_replay_counts") != expected_raw_counts
            or summary.get("raw_replay_counts") != expected_raw_counts
        ):
            raise EvidenceVerificationError(
                "Evidence raw replay count claims differ"
            )

    return EvidenceRunVerification(
        run_id=run_id,
        status=status,
        record_count=record_count,
        capture_count=len(capture_rows),
        provider_id=str(manifest.get("provider_id") or ""),
        manifest_sha256=sha256_file(manifest_path),
        payload_inventory_sha256=inventory_digest,
        scanned_record_count=replay_metrics.scanned_record_count,
        timestamp_valid_record_count=replay_metrics.timestamp_valid_record_count,
        rejected_record_count=replay_metrics.rejected_record_count,
        unmatched_record_count=replay_metrics.unmatched_record_count,
        matched_record_count=replay_metrics.matched_record_count,
        duplicate_record_count=replay_metrics.duplicate_record_count,
        verification_basis=replay_metrics.verification_basis,
    )


def verify_evidence_run(run_directory: str | Path) -> EvidenceRunVerification:
    candidate = reject_symlink_components(
        run_directory,
        error_type=EvidenceVerificationError,
        label="Evidence run path",
    )
    try:
        candidate.lstat()
    except OSError as exc:
        raise EvidenceVerificationError("Evidence run root is missing") from exc
    if candidate.is_symlink() or not candidate.is_dir():
        raise EvidenceVerificationError(
            "Evidence run root must be a real directory"
        )
    try:
        root = candidate.resolve(strict=True)
    except OSError as exc:
        raise EvidenceVerificationError("Evidence run root is unavailable") from exc
    return _verify_evidence_root(root, expected_run_id=root.name)


def load_evidence_run(run_directory: str | Path) -> LoadedEvidenceRun:
    """Verify and load one immutable evidence pack without network access."""

    candidate = Path(run_directory)
    verification = verify_evidence_run(candidate)
    root = candidate.resolve(strict=True)
    manifest = _read_json(root / "run_manifest.json")
    summary = _read_json(root / "final_summary.json")
    policy_reference = _read_json(root / "permission_policy_reference.json")
    records: list[EvidenceRecord] = []
    try:
        with (root / "normalized_evidence.jsonl").open(
            "r", encoding="utf-8"
        ) as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    records.append(EvidenceRecord.from_dict(json.loads(line)))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise EvidenceVerificationError(
                        f"Evidence row {line_number} is invalid"
                    ) from exc
    except (OSError, UnicodeError) as exc:
        raise EvidenceVerificationError(
            "Normalized evidence JSONL is unreadable"
        ) from exc
    reference = {
        "run_id": verification.run_id,
        "provider_id": verification.provider_id,
        "provider_mode": manifest.get("provider_mode"),
        "data_class": manifest.get("data_class"),
        "status": verification.status,
        "record_count": verification.record_count,
        "capture_count": verification.capture_count,
        "scanned_record_count": verification.scanned_record_count,
        "timestamp_valid_record_count": verification.timestamp_valid_record_count,
        "rejected_record_count": verification.rejected_record_count,
        "unmatched_record_count": verification.unmatched_record_count,
        "matched_record_count": verification.matched_record_count,
        "duplicate_record_count": verification.duplicate_record_count,
        "verification_basis": verification.verification_basis,
        "manifest_sha256": verification.manifest_sha256,
        "payload_inventory_sha256": verification.payload_inventory_sha256,
        "permission_policy_reference": policy_reference,
        "normalized_evidence_sha256": manifest.get(
            "normalized_evidence_sha256"
        ),
        "evidence_type_counts": summary.get("evidence_type_counts", {}),
        "artifact_reference": f"evidence/{verification.run_id}",
    }
    return LoadedEvidenceRun(
        records=tuple(records),
        reference=reference,
        verification=verification,
    )


__all__ = [
    "EVIDENCE_PACK_SCHEMA_VERSION",
    "EvidenceRunCollisionError",
    "EvidenceRunVerification",
    "EvidenceVerificationError",
    "LoadedEvidenceRun",
    "load_evidence_run",
    "make_evidence_run_id",
    "publish_evidence_run",
    "sha256_file",
    "validate_evidence_run_id",
    "verify_evidence_run",
]
