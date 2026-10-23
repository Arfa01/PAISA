"""Immutable evidence-pack publication and verification."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from paisa.evidence.models import (
    EVIDENCE_SCHEMA_VERSION,
    EvidenceFetchResult,
    EvidenceRecord,
    utc_iso,
)
from paisa.evidence.policy import EvidencePermissionPolicy
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "record_count": self.record_count,
            "capture_count": self.capture_count,
            "provider_id": self.provider_id,
            "manifest_sha256": self.manifest_sha256,
            "payload_inventory_sha256": self.payload_inventory_sha256,
            "verification": "PASS",
            "verification_basis": "ARTIFACT_BOUND_SEMANTIC_REPLAY",
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
    if text.startswith(("/Users/", "/home/", "C:\\Users\\")):
        return "[REDACTED_ABSOLUTE_PATH]"
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
    def run(*args: str) -> str | None:
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
        return result.stdout.strip() or None

    revision = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
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
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for record in records:
            payload = record.to_dict()
            writer.writerow(
                {field: _csv_scalar(payload.get(field)) for field in _CSV_FIELDS}
            )


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
    if result.errors and not result.captures and not result.records:
        return "FAIL"
    if result.errors:
        return "PARTIAL"
    return "PASS"


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
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / selected_id
    if destination.exists() or destination.is_symlink():
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
            metadata = capture.metadata_dict()
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
            [error.to_dict() for error in result.errors],
        )
        status = _status(result)
        quality = {
            "schema_version": "1.0",
            "status": (
                "FAIL" if status == "FAIL" else "WARN" if result.errors or not result.records else "PASS"
            ),
            "record_count": len(result.records),
            "capture_count": len(result.captures),
            "provider_error_count": len(result.errors),
            "warnings": list(result.warnings),
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
            "limitations": list(result.warnings),
            "academic_disclaimer": "Academic simulation; not financial advice.",
        }
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
        os.chmod(destination, 0o700)
        for item in destination.iterdir():
            os.chmod(item, 0o600)
        verify_evidence_run(destination)
        return destination
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceVerificationError(f"Missing regular artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError(f"Invalid JSON artifact: {path.name}") from exc
    if not isinstance(value, Mapping):
        raise EvidenceVerificationError(f"JSON artifact is not an object: {path.name}")
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


def verify_evidence_run(run_directory: str | Path) -> EvidenceRunVerification:
    candidate = Path(run_directory)
    if candidate.is_symlink():
        raise EvidenceVerificationError("Evidence run root cannot be a symlink")
    root = candidate.resolve()
    if not root.is_dir():
        raise EvidenceVerificationError("Evidence run root is missing")
    run_id = validate_evidence_run_id(root.name)
    for item in root.iterdir():
        mode = item.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise EvidenceVerificationError("Evidence packs may contain only files")
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
        capture_rows = json.loads((root / "capture_metadata.json").read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceVerificationError("Capture metadata is invalid") from exc
    if not isinstance(capture_rows, list):
        raise EvidenceVerificationError("Capture metadata must be an array")
    capture_references: set[str] = set()
    for row in capture_rows:
        if not isinstance(row, Mapping):
            raise EvidenceVerificationError("Capture metadata row is invalid")
        reference = str(row.get("artifact_reference") or "")
        digest = str(row.get("sha256") or "")
        size = row.get("size_bytes")
        if (
            PurePosixPath(reference).name != reference
            or reference in capture_references
            or not _SHA256.fullmatch(digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise EvidenceVerificationError("Capture metadata row is invalid")
        capture_references.add(reference)
        captured_path = root / reference
        if (
            not captured_path.is_file()
            or captured_path.is_symlink()
            or captured_path.stat().st_size != size
            or sha256_file(captured_path) != digest
        ):
            raise EvidenceVerificationError("Captured evidence bytes differ")

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
    manifest_permission = manifest.get("permission_policy_reference")
    if permission_reference != manifest_permission:
        raise EvidenceVerificationError("Permission-policy references disagree")
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
    record_claim = manifest.get("record_count")
    capture_claim = manifest.get("capture_count")
    if (
        isinstance(record_claim, bool)
        or not isinstance(record_claim, int)
        or record_claim != record_count
        or completion.get("record_count") != record_count
        or summary.get("record_count") != record_count
        or isinstance(capture_claim, bool)
        or not isinstance(capture_claim, int)
        or capture_claim != len(capture_rows)
        or completion.get("capture_count") != len(capture_rows)
        or summary.get("capture_count") != len(capture_rows)
    ):
        raise EvidenceVerificationError("Evidence pack counts disagree")
    if manifest.get("normalized_evidence_sha256") != sha256_file(normalized_path):
        raise EvidenceVerificationError("Normalized evidence checksum differs")
    return EvidenceRunVerification(
        run_id=run_id,
        status=status,
        record_count=record_count,
        capture_count=len(capture_rows),
        provider_id=str(manifest.get("provider_id") or ""),
        manifest_sha256=sha256_file(manifest_path),
        payload_inventory_sha256=inventory_digest,
    )


def load_evidence_run(run_directory: str | Path) -> LoadedEvidenceRun:
    """Verify and load one immutable evidence pack without network access."""

    root = Path(run_directory).resolve()
    verification = verify_evidence_run(root)
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
