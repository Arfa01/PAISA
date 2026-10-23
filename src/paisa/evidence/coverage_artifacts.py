"""Immutable artifacts for contemporaneous-evidence coverage audits.

Each pack seals canonical normalized evidence and the exact stock calendar.
The portable verifier rebuilds provider-derived profiles and session mappings
from those inputs instead of trusting persisted labels.  This is still not a
replay of the original provider archives.  Coverage readiness records
caller-defined gate satisfaction for an academic simulation, not financial
advice.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from paisa.evidence.artifact_security import (
    make_pack_private,
    prepare_output_root,
    reject_symlink_components,
    remove_staging_tree,
    verify_pack_modes,
)
from paisa.evidence.artifacts import sha256_file
from paisa.evidence.coverage import (
    AUDIT_PERIODS,
    EVIDENCE_COVERAGE_ALGORITHM_VERSION,
    EVIDENCE_COVERAGE_PROFILES,
    EVIDENCE_COVERAGE_SCHEMA_VERSION,
    MODEL_SPLITS,
    EvidenceCoverageComputation,
    EvidenceCoverageRow,
    EvidenceReadinessCriteria,
    calculate_evidence_readiness,
    compute_evidence_coverage,
)
from paisa.evidence.models import (
    EVIDENCE_TYPES,
    TIME_PRECISIONS,
    EvidenceRecord,
    utc_iso,
)
from paisa.news.coverage import CoverageConfig, StockEvidence, parse_cutoff
from paisa.sentiment.artifacts import (
    SentimentRunCollisionError,
    artifact_inventory,
    publish_staged_run,
    regular_file_inventory,
)


EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION = "1.1"
ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
_RUN_ID = re.compile(
    r"^ECOV-[0-9]{8}-[0-9]{6}-[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_RUN_ID = re.compile(
    r"^EVID-[0-9]{8}-[0-9]{6}-[A-Za-z0-9][A-Za-z0-9._-]{0,63}$"
)
_NEWS_RUN_ID = re.compile(
    r"^NEWS-[0-9]{8}-[0-9]{6}-[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
)
_JWT = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{8,}\."
    r"[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}(?![A-Za-z0-9_-])"
)
_SENSITIVE_KEY = re.compile(
    r"(^|[_-])(?:api[_-]?key|access[_-]?(?:key|token)|authorization|auth|"
    r"credential|cookie|jwt|password|secret|signature|signed|token)"
    r"([_-]|$)",
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
_SEAL_FILES = frozenset({"run_manifest.json", "completion.json"})
_PAYLOAD_FILES = frozenset(
    {
        "input_evidence_reference.json",
        "normalized_evidence.jsonl",
        "stock_calendar.json",
        "stock_artifact_reference.json",
        "evidence_session_mappings.csv",
        "split_coverage.csv",
        "monthly_coverage.csv",
        "missing_intervals.csv",
        "quality_report.json",
        "final_summary.json",
    }
)
_ALL_FILES = _PAYLOAD_FILES | _SEAL_FILES

_MAPPING_FIELDS = (
    "evidence_id",
    "evidence_type",
    "provider_id",
    "ticker",
    "profiles",
    "time_precision",
    "source_published_at",
    "source_published_date",
    "available_at",
    "fetched_at",
    "assigned_session",
    "assigned_split",
    "decision_cutoff_utc",
    "arrival_class",
    "mapping_status",
    "cutoff_eligible",
)
_SPLIT_FIELDS = (
    "profile",
    "split",
    "trading_sessions",
    "sessions_with_records",
    "coverage_percent",
    "record_count",
    "zero_record_sessions",
    "longest_missing_gap_sessions",
    "before_cutoff_records",
    "after_cutoff_records",
    "non_trading_day_records",
    "date_only_records",
    "ogdc_specific_records",
    "evidence_type_counts",
    "provider_counts",
)
_MONTHLY_FIELDS = (
    "profile",
    "split",
    "month",
    "trading_sessions",
    "sessions_with_records",
    "coverage_percent",
    "record_count",
    "date_only_records",
    "evidence_type_counts",
)
_MISSING_FIELDS = (
    "profile",
    "split",
    "start_date",
    "end_date",
    "missing_trading_sessions",
)


class EvidenceCoverageRunCollisionError(FileExistsError):
    """An immutable evidence-coverage run already exists."""


class EvidenceCoverageVerificationError(ValueError):
    """An evidence-coverage pack failed integrity or semantic verification."""


@dataclass(frozen=True, slots=True)
class EvidenceCoverageRunVerification:
    run_id: str
    status: str
    readiness_status_by_profile: Mapping[str, str]
    mapping_count: int
    artifact_count: int
    manifest_sha256: str
    payload_inventory_sha256: str
    verification_basis: str = "PORTABLE_RECORD_CALENDAR_RECOMPUTATION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "readiness_status_by_profile": dict(
                self.readiness_status_by_profile
            ),
            "mapping_count": self.mapping_count,
            "artifact_count": self.artifact_count,
            "manifest_sha256": self.manifest_sha256,
            "payload_inventory_sha256": self.payload_inventory_sha256,
            "verification": "PASS",
            "verification_basis": self.verification_basis,
            "source_evidence_bound": False,
            "completion_acceptance": (
                "PACK_RECORD_CALENDAR_RECOMPUTATION_NOT_SOURCE_ARCHIVE_BOUND"
            ),
        }


def make_evidence_coverage_run_id(now: datetime | None = None) -> str:
    selected = now or datetime.now(timezone.utc)
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("Coverage run timestamps must be timezone-aware")
    selected = selected.astimezone(timezone.utc)
    return f"ECOV-{selected:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


def validate_evidence_coverage_run_id(value: str) -> str:
    text = str(value)
    if not _RUN_ID.fullmatch(text):
        raise ValueError(
            "run_id must match ECOV-YYYYMMDD-HHMMSS-safe-identifier"
        )
    return text


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _inventory_digest(inventory: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes(list(inventory)))


def _write_private(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
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


def _write_evidence_jsonl(
    path: Path, records: Sequence[EvidenceRecord]
) -> None:
    payload = b"".join(
        _canonical_bytes(record.to_dict()) + b"\n" for record in records
    )
    _write_private(path, payload)


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Coverage CSV values must be finite")
        return str(value)
    if isinstance(value, (dict, list, tuple)):
        return _canonical_bytes(value).decode("utf-8")
    text = str(value)
    return f"'{text}" if text.startswith(("=", "+", "-", "@")) else text


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_scalar(row.get(field)) for field in fields}
                )
    except BaseException:
        try:
            path.unlink()
        except OSError:
            pass
        raise


def _stock_calendar_document(
    computation: EvidenceCoverageComputation,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "symbol": computation.stock_symbol,
        "market_timezone": computation.market_timezone,
        "decision_cutoff": computation.decision_cutoff,
        "trading_dates": [value.isoformat() for value in computation.trading_dates],
        "split_dates": {
            split: [value.isoformat() for value in computation.split_dates[split]]
            for split in MODEL_SPLITS
        },
    }


def _sorted_computation_tables(
    computation: EvidenceCoverageComputation,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    profile_order = {
        name: index for index, name in enumerate(EVIDENCE_COVERAGE_PROFILES)
    }
    period_order = {name: index for index, name in enumerate(AUDIT_PERIODS)}
    mappings = sorted(
        (item.to_dict() for item in computation.mapping_rows),
        key=lambda item: str(item["evidence_id"]),
    )
    split_rows = sorted(
        (item.to_dict() for item in computation.split_rows),
        key=lambda item: (
            profile_order.get(str(item["profile"]), 999),
            period_order.get(str(item["split"]), 999),
        ),
    )
    monthly_rows = sorted(
        (item.to_dict() for item in computation.monthly_rows),
        key=lambda item: (
            profile_order.get(str(item["profile"]), 999),
            period_order.get(str(item["split"]), 999),
            str(item["month"]),
        ),
    )
    missing_rows = sorted(
        (item.to_dict() for item in computation.missing_intervals),
        key=lambda item: (
            profile_order.get(str(item["profile"]), 999),
            period_order.get(str(item["split"]), 999),
            str(item["start_date"]),
        ),
    )
    return mappings, split_rows, monthly_rows, missing_rows


def _csv_formatted_rows(
    rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> list[dict[str, str]]:
    return [
        {field: _csv_scalar(row.get(field)) for field in fields}
        for row in rows
    ]


def _sanitize_text(value: str) -> str:
    text = _JWT.sub("[REDACTED_JWT]", value)
    text = _FILE_URI.sub("[REDACTED_FILE_URI]", text)
    parsed = urlsplit(text)
    if parsed.scheme in {"http", "https"} and parsed.hostname:
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            host = f"{host}:{port}"
        query = urlencode(
            [
                (
                    key,
                    "[REDACTED]" if _SENSITIVE_KEY.search(key) else item,
                )
                for key, item in parse_qsl(
                    parsed.query, keep_blank_values=True
                )
            ],
            doseq=True,
        )
        return urlunsplit(
            (parsed.scheme, host, parsed.path, query, parsed.fragment)
        )
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
    if "/" in text and ".." in PurePosixPath(text).parts:
        return "[REDACTED_UNSAFE_PATH]"
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


def _assert_hygiene(value: Any, *, key: str = "") -> None:
    if _SENSITIVE_KEY.search(key) and value != "[REDACTED]":
        raise EvidenceCoverageVerificationError(
            "Coverage references contain an unredacted credential field"
        )
    if isinstance(value, Mapping):
        for item_key, item_value in value.items():
            _assert_hygiene(item_value, key=str(item_key))
        return
    if isinstance(value, list):
        for item in value:
            _assert_hygiene(item)
        return
    if not isinstance(value, str):
        return
    if _JWT.search(value):
        raise EvidenceCoverageVerificationError(
            "Coverage references contain an unredacted token"
        )
    if _FILE_URI.search(value):
        raise EvidenceCoverageVerificationError(
            "Coverage references contain an absolute file URI"
        )
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https"}:
        if parsed.username is not None or parsed.password is not None:
            raise EvidenceCoverageVerificationError(
                "Coverage references contain URL credentials"
            )
        for query_key, query_value in parse_qsl(
            parsed.query, keep_blank_values=True
        ):
            if (
                _SENSITIVE_KEY.search(query_key)
                and query_value != "[REDACTED]"
            ):
                raise EvidenceCoverageVerificationError(
                    "Coverage references contain a credential query value"
                )
        return
    if (
        _EMBEDDED_POSIX_NETWORK_ABSOLUTE.search(value)
        or _EMBEDDED_POSIX_ABSOLUTE.search(value)
        or _EMBEDDED_WINDOWS_ABSOLUTE.search(value)
        or _EMBEDDED_WINDOWS_UNC.search(value)
    ):
        raise EvidenceCoverageVerificationError(
            "Coverage references contain an embedded absolute host path"
        )
    if (
        value.startswith("/")
        or _WINDOWS_ABSOLUTE.match(value)
        or _WINDOWS_UNC.match(value)
        or _EMBEDDED_WINDOWS_UNC.search(value)
    ):
        raise EvidenceCoverageVerificationError(
            "Coverage references contain an absolute host path"
        )
    if "/" in value and ".." in PurePosixPath(value).parts:
        raise EvidenceCoverageVerificationError(
            "Coverage references contain path traversal"
        )


def _reference_error(message: str) -> EvidenceCoverageVerificationError:
    return EvidenceCoverageVerificationError(
        f"Coverage source reference is invalid: {message}"
    )


def _require_sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise _reference_error(f"{field} must be a SHA-256 digest")
    return value


def _require_count(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _reference_error(f"{field} must be a non-negative integer")
    return value


def _require_reference(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _reference_error(f"{field} must be a relative POSIX reference")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value == ".":
        raise _reference_error(f"{field} must be a safe relative reference")
    return value


def _validate_permission_reference(value: Any, *, field: str) -> None:
    if not isinstance(value, Mapping) or value.get("schema_version") != "1.0":
        raise _reference_error(f"{field} is missing or unsupported")
    if "sha256" in value:
        _require_sha(value.get("sha256"), field=f"{field}.sha256")
        _require_reference(value.get("path"), field=f"{field}.path")
        if not isinstance(value.get("policy_id"), str) or not value["policy_id"]:
            raise _reference_error(f"{field}.policy_id is missing")
        assessments = value.get("assessment_references")
        if not isinstance(assessments, list) or not assessments:
            raise _reference_error(f"{field}.assessment_references is missing")
        for index, assessment in enumerate(assessments):
            if not isinstance(assessment, Mapping):
                raise _reference_error(f"{field}.assessment_references is malformed")
            _require_reference(
                assessment.get("path"),
                field=f"{field}.assessment_references[{index}].path",
            )
            _require_sha(
                assessment.get("sha256"),
                field=f"{field}.assessment_references[{index}].sha256",
            )
        return
    _require_sha(value.get("policy_sha256"), field=f"{field}.policy_sha256")
    _require_sha(
        value.get("assessment_sha256"), field=f"{field}.assessment_sha256"
    )
    _require_reference(
        value.get("policy_reference"), field=f"{field}.policy_reference"
    )
    _require_reference(
        value.get("assessment_document"), field=f"{field}.assessment_document"
    )


def _validate_gdelt_reference(value: Mapping[str, Any]) -> None:
    if (
        value.get("provider_id") != "gdelt-gkg-2.1"
        or value.get("provider_mode") != "raw_archive"
        or value.get("data_class") != "real"
        or value.get("status") not in {"PASS", "PARTIAL"}
        or not isinstance(value.get("run_id"), str)
        or not _EVIDENCE_RUN_ID.fullmatch(value["run_id"])
    ):
        raise _reference_error("GDELT run identity or semantics differ")
    for field in (
        "manifest_sha256",
        "payload_inventory_sha256",
        "normalized_evidence_sha256",
    ):
        _require_sha(value.get(field), field=f"gdelt_gkg.{field}")
    for field in ("record_count", "capture_count"):
        _require_count(value.get(field), field=f"gdelt_gkg.{field}")
    _require_reference(
        value.get("artifact_reference"), field="gdelt_gkg.artifact_reference"
    )
    _validate_permission_reference(
        value.get("permission_policy_reference"),
        field="gdelt_gkg.permission_policy_reference",
    )


def _validate_psx_reference(value: Mapping[str, Any]) -> None:
    if (
        value.get("provider_id") != "psx-announcement"
        or value.get("provider_mode") != "manual_import"
    ):
        raise _reference_error("PSX provider identity differs")
    _validate_permission_reference(
        value.get("permission_policy_reference"),
        field="psx_announcement_evidence.permission_policy_reference",
    )
    count = _require_count(
        value.get("record_count"), field="psx_announcement_evidence.record_count"
    )
    if value.get("status") == "BLOCKED_PERMISSION_REVIEW":
        if (
            count != 0
            or value.get("run_reference") is not None
            or value.get("permission_status") != "REVIEW_REQUIRED"
            or value.get("automated_access") is not False
            or not isinstance(value.get("decision_id"), str)
            or not value["decision_id"]
        ):
            raise _reference_error("blocked PSX reference is contradictory")
        return
    if value.get("status") != "LOADED":
        raise _reference_error("PSX status is unsupported")
    run = value.get("run_reference")
    if not isinstance(run, Mapping):
        raise _reference_error("loaded PSX run reference is missing")
    _validate_gdelt_like_evidence_run(
        run,
        provider_id="psx-announcement",
        provider_mode="manual_import",
        field="psx_announcement_evidence.run_reference",
    )
    if run.get("record_count") != count:
        raise _reference_error("PSX record counts disagree")


def _validate_gdelt_like_evidence_run(
    value: Mapping[str, Any],
    *,
    provider_id: str,
    provider_mode: str,
    field: str,
) -> None:
    if (
        value.get("provider_id") != provider_id
        or value.get("provider_mode") != provider_mode
        or value.get("status") not in {"PASS", "PARTIAL"}
        or not isinstance(value.get("run_id"), str)
        or not _EVIDENCE_RUN_ID.fullmatch(value["run_id"])
    ):
        raise _reference_error(f"{field} identity or semantics differ")
    for name in (
        "manifest_sha256",
        "payload_inventory_sha256",
        "normalized_evidence_sha256",
    ):
        _require_sha(value.get(name), field=f"{field}.{name}")
    for name in ("record_count", "capture_count"):
        _require_count(value.get(name), field=f"{field}.{name}")
    _require_reference(value.get("artifact_reference"), field=f"{field}.artifact_reference")


def _validate_publisher_reference(value: Mapping[str, Any]) -> None:
    count = _require_count(value.get("record_count"), field="publisher_news.record_count")
    status = value.get("status")
    if status == "NOT_SUPPLIED":
        if count != 0:
            raise _reference_error("omitted publisher-news count must be zero")
        return
    if status != "LOADED":
        raise _reference_error("publisher-news status is unsupported")
    runs = value.get("run_references")
    if not isinstance(runs, list) or not runs:
        raise _reference_error("publisher-news run references are missing")
    total_occurrences = 0
    seen: set[str] = set()
    for index, run in enumerate(runs):
        if not isinstance(run, Mapping):
            raise _reference_error("publisher-news run reference is malformed")
        run_id = run.get("run_id")
        if (
            not isinstance(run_id, str)
            or not _NEWS_RUN_ID.fullmatch(run_id)
            or run_id in seen
            or run.get("status") not in {"PASS", "PARTIAL"}
        ):
            raise _reference_error("publisher-news run identity differs")
        seen.add(run_id)
        for name in (
            "run_manifest_sha256",
            "artifact_inventory_sha256",
            "normalized_articles_sha256",
        ):
            _require_sha(run.get(name), field=f"publisher_news.runs[{index}].{name}")
        total_occurrences += _require_count(
            run.get("occurrence_count"),
            field=f"publisher_news.runs[{index}].occurrence_count",
        )
        _require_reference(
            run.get("artifact_reference"),
            field=f"publisher_news.runs[{index}].artifact_reference",
        )
        decision = run.get("source_permission_decision")
        if (
            not isinstance(decision, Mapping)
            or decision.get("passed") is not True
            or decision.get("status") != "PERMITTED"
            or not isinstance(decision.get("decision_id"), str)
            or not decision["decision_id"]
        ):
            raise _reference_error("publisher-news permission decision did not pass")
    if count > total_occurrences:
        raise _reference_error("publisher-news canonical count exceeds occurrences")
    _validate_permission_reference(
        value.get("permission_policy_reference"),
        field="publisher_news.permission_policy_reference",
    )
    fingerprints = value.get("database_fingerprints")
    if not isinstance(fingerprints, Mapping):
        raise _reference_error("publisher-news database fingerprint is missing")
    if fingerprints.get("provided") is True:
        files = fingerprints.get("files")
        if not isinstance(files, list) or not files:
            raise _reference_error("publisher-news database files are missing")
        for index, item in enumerate(files):
            if not isinstance(item, Mapping):
                raise _reference_error("publisher-news database file is malformed")
            _require_sha(
                item.get("sha256"),
                field=f"publisher_news.database_files[{index}].sha256",
            )
            _require_count(
                item.get("size_bytes"),
                field=f"publisher_news.database_files[{index}].size_bytes",
            )


def _validate_input_reference_document(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION:
        raise _reference_error("input-reference schema is unsupported")
    inputs = value.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise _reference_error("input-reference list is missing")
    counts: dict[str, int] = {}
    for item in inputs:
        if not isinstance(item, Mapping):
            raise _reference_error("input reference is not an object")
        kind = item.get("input_kind")
        if not isinstance(kind, str):
            raise _reference_error("input_kind is missing")
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "gdelt_gkg":
            _validate_gdelt_reference(item)
        elif kind == "psx_announcement_evidence":
            _validate_psx_reference(item)
        elif kind == "publisher_news":
            _validate_publisher_reference(item)
        elif kind == "readiness_configuration":
            if item.get("schema_version") != "1.0":
                raise _reference_error("readiness configuration schema differs")
            _require_sha(item.get("sha256"), field="readiness_configuration.sha256")
            _require_reference(
                item.get("artifact_reference"),
                field="readiness_configuration.artifact_reference",
            )
        else:
            raise _reference_error(f"unsupported input_kind {kind!r}")
    if (
        counts.get("gdelt_gkg", 0) < 1
        or counts.get("psx_announcement_evidence") != 1
        or counts.get("publisher_news") != 1
        or counts.get("readiness_configuration") != 1
        or set(counts)
        != {
            "gdelt_gkg",
            "psx_announcement_evidence",
            "publisher_news",
            "readiness_configuration",
        }
    ):
        raise _reference_error("required source populations are missing or duplicated")


def _referenced_record_count(value: Mapping[str, Any]) -> int:
    inputs = value.get("inputs")
    if not isinstance(inputs, list):
        raise _reference_error("input-reference list is missing")
    return sum(
        int(item["record_count"])
        for item in inputs
        if isinstance(item, Mapping)
        and item.get("input_kind")
        in {"gdelt_gkg", "psx_announcement_evidence", "publisher_news"}
    )


def _referenced_population_counts(value: Mapping[str, Any]) -> dict[str, int]:
    counts = {name: 0 for name in EVIDENCE_TYPES}
    for item in value.get("inputs", []):
        if not isinstance(item, Mapping):
            continue
        kind = item.get("input_kind")
        if kind == "gdelt_gkg":
            counts["gdelt_metadata"] += int(item["record_count"])
        elif kind == "psx_announcement_evidence":
            counts["exchange_announcement"] += int(item["record_count"])
        elif kind == "publisher_news":
            counts["publisher_news"] += int(item["record_count"])
    return counts


def _permission_ids_by_evidence_type(
    value: Mapping[str, Any],
) -> dict[str, set[str]]:
    result = {name: set() for name in EVIDENCE_TYPES}
    for item in value.get("inputs", []):
        if not isinstance(item, Mapping):
            continue
        kind = item.get("input_kind")
        if kind in {"gdelt_gkg", "psx_announcement_evidence"}:
            permission = item.get("permission_policy_reference")
            if isinstance(permission, Mapping):
                policy_id = permission.get("policy_id")
                if isinstance(policy_id, str) and policy_id:
                    target = (
                        "gdelt_metadata"
                        if kind == "gdelt_gkg"
                        else "exchange_announcement"
                    )
                    result[target].add(policy_id)
        elif kind == "publisher_news":
            runs = item.get("run_references")
            if isinstance(runs, list):
                for run in runs:
                    if not isinstance(run, Mapping):
                        continue
                    decision = run.get("source_permission_decision")
                    if isinstance(decision, Mapping):
                        decision_id = decision.get("decision_id")
                        if isinstance(decision_id, str) and decision_id:
                            result["publisher_news"].add(decision_id)
    return result


def _verify_record_sources(
    records: Sequence[EvidenceRecord], inputs: Mapping[str, Any]
) -> None:
    expected_counts = _referenced_population_counts(inputs)
    observed_counts = {
        evidence_type: sum(
            record.evidence_type == evidence_type for record in records
        )
        for evidence_type in EVIDENCE_TYPES
    }
    if observed_counts != expected_counts:
        raise EvidenceCoverageVerificationError(
            "Coverage normalized evidence populations disagree with source references"
        )
    allowed_permissions = _permission_ids_by_evidence_type(inputs)
    for record in records:
        permitted = allowed_permissions[record.evidence_type]
        if not permitted or record.permission_policy_id not in permitted:
            raise EvidenceCoverageVerificationError(
                "Coverage normalized evidence permission identity is unbound"
            )


def _validate_stock_reference(value: Mapping[str, Any]) -> None:
    if value.get("symbol") != "OGDC" or value.get("is_real_data") is not True:
        raise _reference_error("stock reference must identify real OGDC evidence")
    for field in (
        "vertical_manifest_sha256",
        "price_artifact_sha256",
        "training_manifest_sha256",
        "training_data_manifest_sha256",
        "training_input_snapshot_sha256",
        "split_manifest_sha256",
    ):
        _require_sha(value.get(field), field=f"stock.{field}")
    if value.get("training_input_snapshot_sha256") != value.get(
        "price_artifact_sha256"
    ):
        raise _reference_error("stock and training snapshots disagree")
    _require_reference(
        value.get("price_artifact_reference"), field="stock.price_artifact_reference"
    )
    if not isinstance(value.get("vertical_slice_run_id"), str) or not value[
        "vertical_slice_run_id"
    ]:
        raise _reference_error("stock vertical-slice identity is missing")
    if not isinstance(value.get("model_training_run_id"), str) or not value[
        "model_training_run_id"
    ]:
        raise _reference_error("stock training identity is missing")
    price_count = _require_count(
        value.get("price_trading_day_count"), field="stock.price_trading_day_count"
    )
    date_range = value.get("price_date_range")
    split_counts = value.get("split_counts")
    split_ranges = value.get("split_ranges")
    if (
        not isinstance(date_range, Mapping)
        or not isinstance(split_counts, Mapping)
        or not isinstance(split_ranges, Mapping)
        or value.get("split_method")
        != "chronological_target_date_70_15_15_no_shuffle"
    ):
        raise _reference_error("stock split contract is incomplete")
    try:
        price_start = date.fromisoformat(str(date_range.get("start")))
        price_end = date.fromisoformat(str(date_range.get("end")))
    except ValueError as exc:
        raise _reference_error("stock date range is invalid") from exc
    if price_start > price_end or price_count <= 0:
        raise _reference_error("stock date range or count is invalid")
    previous_end: date | None = None
    total = 0
    for split in MODEL_SPLITS:
        total += _require_count(split_counts.get(split), field=f"stock.{split}_count")
        raw_range = split_ranges.get(split)
        if not isinstance(raw_range, Mapping):
            raise _reference_error(f"stock {split} range is missing")
        try:
            start = date.fromisoformat(str(raw_range.get("start")))
            end = date.fromisoformat(str(raw_range.get("end")))
        except ValueError as exc:
            raise _reference_error(f"stock {split} range is invalid") from exc
        if start > end or start < price_start or end > price_end:
            raise _reference_error(f"stock {split} range is outside prices")
        if previous_end is not None and start <= previous_end:
            raise _reference_error("stock split ranges are not chronological")
        previous_end = end
    if total > price_count:
        raise _reference_error("stock split counts exceed price sessions")


def _verify_stock_calendar_reference(
    stock: StockEvidence, reference: Mapping[str, Any]
) -> None:
    if stock.symbol != reference.get("symbol"):
        raise EvidenceCoverageVerificationError(
            "Coverage stock calendar symbol disagrees with its reference"
        )
    observed_range = {
        "start": stock.trading_dates[0].isoformat(),
        "end": stock.trading_dates[-1].isoformat(),
    }
    if (
        reference.get("price_trading_day_count") != len(stock.trading_dates)
        or reference.get("price_date_range") != observed_range
        or reference.get("split_counts")
        != {split: len(stock.split_dates[split]) for split in MODEL_SPLITS}
        or reference.get("split_ranges")
        != {
            split: {
                "start": stock.split_dates[split][0].isoformat(),
                "end": stock.split_dates[split][-1].isoformat(),
            }
            for split in MODEL_SPLITS
        }
    ):
        raise EvidenceCoverageVerificationError(
            "Coverage stock calendar disagrees with its immutable stock reference"
        )


def _normalize_input_references(
    value: Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(value, Mapping):
        selected: Sequence[Mapping[str, Any]] = (value,)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        selected = value
    else:
        raise TypeError("input_references must be a mapping or mapping sequence")
    if not selected or any(not isinstance(item, Mapping) for item in selected):
        raise ValueError("input_references must contain at least one mapping")
    result = {
        "schema_version": EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION,
        "inputs": [_sanitize(dict(item)) for item in selected],
    }
    _assert_hygiene(result)
    _validate_input_reference_document(result)
    return result


def _git_provenance(repository_root: Path) -> dict[str, Any]:
    def run(*args: str, preserve_empty: bool = False) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        output = completed.stdout.strip()
        return output if output or preserve_empty else None

    revision = run("rev-parse", "HEAD")
    dirty = run(
        "status", "--porcelain", "--untracked-files=all", preserve_empty=True
    )
    return {
        "producer": "paisa.evidence.coverage_artifacts",
        "code_version": revision,
        "dirty_tree": None if dirty is None else bool(dirty),
    }


def _require_ignored_if_inside_repository(
    output_root: Path, repository_root: Path
) -> None:
    repository = repository_root.resolve()
    if not (repository / ".git").exists():
        return
    try:
        relative = output_root.resolve().relative_to(repository)
    except ValueError:
        return
    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", relative.as_posix()],
            cwd=repository,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(
            "Could not verify repository artifact-ignore policy"
        ) from exc
    if result.returncode != 0:
        raise ValueError(
            "Coverage output inside the repository must be git-ignored"
        )


def _read_json(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise EvidenceCoverageVerificationError(
            f"Missing regular coverage artifact: {path.name}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceCoverageVerificationError(
            f"Invalid JSON coverage artifact: {path.name}"
        ) from exc
    if not isinstance(value, Mapping):
        raise EvidenceCoverageVerificationError(
            f"Coverage JSON artifact is not an object: {path.name}"
        )
    return value


def _read_csv(
    root: Path, filename: str, fields: Sequence[str]
) -> list[dict[str, str]]:
    path = root / filename
    if path.is_symlink() or not path.is_file():
        raise EvidenceCoverageVerificationError(
            f"Missing regular coverage artifact: {filename}"
        )
    try:
        payload = path.read_bytes()
        if not payload.endswith(b"\n") or b"\0" in payload:
            raise EvidenceCoverageVerificationError(
                f"Coverage CSV framing is invalid: {filename}"
            )
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise EvidenceCoverageVerificationError(
                    f"Coverage CSV schema differs: {filename}"
                )
            rows = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise EvidenceCoverageVerificationError(
                        f"Coverage CSV row width differs: {filename}"
                    )
                for field, value in row.items():
                    if value.startswith(("=", "+", "-", "@")):
                        raise EvidenceCoverageVerificationError(
                            f"Coverage CSV formula is not neutralized: {filename}"
                        )
                    _assert_hygiene(value, key=field)
                rows.append(dict(row))
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise EvidenceCoverageVerificationError(
            f"Coverage CSV is invalid: {filename}"
        ) from exc


def _read_evidence_records(root: Path) -> tuple[EvidenceRecord, ...]:
    path = root / "normalized_evidence.jsonl"
    if path.is_symlink() or not path.is_file():
        raise EvidenceCoverageVerificationError(
            "Missing regular coverage artifact: normalized_evidence.jsonl"
        )
    try:
        payload = path.read_bytes()
        if payload and (not payload.endswith(b"\n") or b"\0" in payload):
            raise EvidenceCoverageVerificationError(
                "Coverage normalized evidence framing is invalid"
            )
        records: list[EvidenceRecord] = []
        for line_number, raw_line in enumerate(payload.splitlines(), start=1):
            try:
                decoded = raw_line.decode("utf-8")
                value = json.loads(decoded)
            except (UnicodeError, json.JSONDecodeError) as exc:
                raise EvidenceCoverageVerificationError(
                    "Coverage normalized evidence is invalid"
                ) from exc
            if not isinstance(value, Mapping) or _canonical_bytes(value) != raw_line:
                raise EvidenceCoverageVerificationError(
                    "Coverage normalized evidence is not canonical"
                )
            try:
                records.append(EvidenceRecord.from_dict(value))
            except (TypeError, ValueError) as exc:
                raise EvidenceCoverageVerificationError(
                    f"Coverage normalized evidence contract failed at line {line_number}"
                ) from exc
    except OSError as exc:
        raise EvidenceCoverageVerificationError(
            "Coverage normalized evidence could not be read"
        ) from exc
    identifiers = [record.evidence_id for record in records]
    if identifiers != sorted(identifiers) or len(identifiers) != len(set(identifiers)):
        raise EvidenceCoverageVerificationError(
            "Coverage normalized evidence identities are not canonical and unique"
        )
    return tuple(records)


def _read_stock_calendar(
    root: Path,
) -> tuple[StockEvidence, CoverageConfig]:
    value = _read_json(root / "stock_calendar.json")
    expected_keys = {
        "schema_version",
        "symbol",
        "market_timezone",
        "decision_cutoff",
        "trading_dates",
        "split_dates",
    }
    if set(value) != expected_keys or value.get("schema_version") != "1.0":
        raise EvidenceCoverageVerificationError(
            "Coverage stock calendar schema is invalid"
        )
    raw_dates = value.get("trading_dates")
    raw_splits = value.get("split_dates")
    if (
        not isinstance(raw_dates, list)
        or not raw_dates
        or not isinstance(raw_splits, Mapping)
        or set(raw_splits) != set(MODEL_SPLITS)
        or any(not isinstance(item, str) for item in raw_dates)
        or any(
            not isinstance(raw_splits[split], list)
            or not raw_splits[split]
            or any(not isinstance(item, str) for item in raw_splits[split])
            for split in MODEL_SPLITS
        )
    ):
        raise EvidenceCoverageVerificationError(
            "Coverage stock calendar dates are invalid"
        )
    try:
        trading_dates = tuple(
            _iso_date(item, field="stock trading date") for item in raw_dates
        )
        split_dates = {
            split: tuple(
                _iso_date(item, field=f"stock {split} date")
                for item in raw_splits[split]
            )
            for split in MODEL_SPLITS
        }
        config = CoverageConfig(
            market_timezone=str(value.get("market_timezone") or ""),
            decision_cutoff=parse_cutoff(str(value.get("decision_cutoff") or "")),
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceCoverageVerificationError(
            "Coverage stock calendar configuration is invalid"
        ) from exc
    symbol = value.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise EvidenceCoverageVerificationError(
            "Coverage stock calendar symbol is invalid"
        )
    return (
        StockEvidence(
            symbol=symbol,
            trading_dates=trading_dates,
            split_dates=split_dates,
            reference={},
        ),
        config,
    )


def _normalize_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise EvidenceCoverageVerificationError(
            "Coverage payload inventory is invalid"
        )
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, Mapping):
            raise EvidenceCoverageVerificationError(
                "Coverage payload inventory entry is invalid"
            )
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
            raise EvidenceCoverageVerificationError(
                "Coverage payload inventory entry is invalid"
            )
        seen.add(path)
        records.append({"path": path, "sha256": digest, "size_bytes": size})
    return sorted(records, key=lambda item: item["path"])


def _plain_csv(value: str) -> str:
    if len(value) >= 2 and value[0] == "'" and value[1] in "=+-@":
        return value[1:]
    return value


def _integer(value: str, *, field: str) -> int:
    if not re.fullmatch(r"0|[1-9][0-9]*", value):
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is not a canonical non-negative integer"
        )
    return int(value)


def _floating(value: str, *, field: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is not numeric"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is invalid"
        )
    return parsed


def _json_cell(value: str, *, field: str, expected: type) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is not valid JSON"
        ) from exc
    if not isinstance(parsed, expected):
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} has the wrong JSON type"
        )
    if _canonical_bytes(parsed).decode("utf-8") != value:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is not canonically serialized"
        )
    return parsed


def _iso_date(value: str, *, field: str) -> date:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is not an ISO date"
        ) from exc
    if parsed.isoformat() != value:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is not a canonical ISO date"
        )
    return parsed


def _timestamp(
    value: str, *, field: str, optional: bool = True
) -> datetime | None:
    if value == "" and optional:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} is not an ISO timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceCoverageVerificationError(
            f"Coverage {field} must be timezone-aware"
        )
    return parsed.astimezone(timezone.utc)


def _typed_mappings(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    typed: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for row in rows:
        evidence_id = _plain_csv(row["evidence_id"])
        provider_id = _plain_csv(row["provider_id"])
        ticker = _plain_csv(row["ticker"])
        profiles = _json_cell(
            row["profiles"], field="mapping profiles", expected=list
        )
        if (
            not evidence_id
            or evidence_id in identifiers
            or not profiles
            or len(profiles) != len(set(profiles))
            or any(item not in EVIDENCE_COVERAGE_PROFILES for item in profiles)
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage mapping identities or profiles are invalid"
            )
        identifiers.add(evidence_id)
        if row["evidence_type"] not in EVIDENCE_TYPES:
            raise EvidenceCoverageVerificationError(
                "Coverage mapping evidence type is invalid"
            )
        if row["time_precision"] not in TIME_PRECISIONS:
            raise EvidenceCoverageVerificationError(
                "Coverage mapping time precision is invalid"
            )
        _timestamp(
            row["source_published_at"], field="source_published_at"
        )
        if row["source_published_date"]:
            _iso_date(
                row["source_published_date"], field="source_published_date"
            )
        available_at = _timestamp(row["available_at"], field="available_at")
        _timestamp(row["fetched_at"], field="fetched_at", optional=False)
        decision_cutoff = _timestamp(
            row["decision_cutoff_utc"], field="decision_cutoff_utc"
        )
        if row["cutoff_eligible"] not in {"true", "false"}:
            raise EvidenceCoverageVerificationError(
                "Coverage mapping cutoff eligibility is invalid"
            )
        eligible = row["cutoff_eligible"] == "true"
        assigned = row["assigned_session"]
        assigned_split = row["assigned_split"]
        if assigned:
            _iso_date(assigned, field="assigned_session")
        if assigned_split and assigned_split not in MODEL_SPLITS:
            raise EvidenceCoverageVerificationError(
                "Coverage mapping split is invalid"
            )
        if eligible != bool(assigned) or (assigned_split and not eligible):
            raise EvidenceCoverageVerificationError(
                "Coverage mapping assignment contradicts cutoff eligibility"
            )
        if eligible != bool(row["decision_cutoff_utc"]):
            raise EvidenceCoverageVerificationError(
                "Coverage mapping cutoff timestamp is contradictory"
            )
        if row["time_precision"] == "date_only" and (
            not row["source_published_date"] or row["available_at"]
        ):
            raise EvidenceCoverageVerificationError(
                "Date-only coverage mapping invents or omits time evidence"
            )
        if row["time_precision"] == "minute" and (
            available_at is None
            or available_at.second != 0
            or available_at.microsecond != 0
        ):
            raise EvidenceCoverageVerificationError(
                "Minute-precision coverage mapping invents seconds"
            )
        if (
            eligible
            and available_at is not None
            and decision_cutoff is not None
            and available_at > decision_cutoff
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage occurrence is later than its claimed cutoff"
            )
        typed.append(
            {
                **dict(row),
                "evidence_id": evidence_id,
                "provider_id": provider_id,
                "ticker": ticker,
                "profiles": tuple(profiles),
                "cutoff_eligible": eligible,
            }
        )
    if [item["evidence_id"] for item in typed] != sorted(identifiers):
        raise EvidenceCoverageVerificationError(
            "Coverage mappings are not in canonical evidence-ID order"
        )
    return typed


def _count_mapping_rows(
    mappings: Sequence[Mapping[str, Any]], profile: str, split: str
) -> list[Mapping[str, Any]]:
    selected = [
        row
        for row in mappings
        if profile in row["profiles"] and row["cutoff_eligible"]
    ]
    if split != "full_price_period":
        selected = [row for row in selected if row["assigned_split"] == split]
    return selected


def _count_map(values: Sequence[str]) -> dict[str, int]:
    return {
        name: sum(value == name for value in values)
        for name in sorted(set(values))
    }


def _typed_split_rows(
    rows: Sequence[Mapping[str, str]],
    mappings: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], EvidenceCoverageRow]:
    expected_keys = {
        (profile, split)
        for profile in EVIDENCE_COVERAGE_PROFILES
        for split in AUDIT_PERIODS
    }
    observed: dict[tuple[str, str], EvidenceCoverageRow] = {}
    for row in rows:
        profile = row["profile"]
        split = row["split"]
        key = (profile, split)
        if key not in expected_keys or key in observed:
            raise EvidenceCoverageVerificationError(
                "Coverage split table keys are invalid"
            )
        integers = {
            field: _integer(row[field], field=field)
            for field in _SPLIT_FIELDS
            if field
            not in {
                "profile",
                "split",
                "coverage_percent",
                "evidence_type_counts",
                "provider_counts",
            }
        }
        coverage = _floating(
            row["coverage_percent"], field="coverage_percent"
        )
        type_counts = _json_cell(
            row["evidence_type_counts"],
            field="evidence_type_counts",
            expected=dict,
        )
        provider_counts = _json_cell(
            row["provider_counts"],
            field="provider_counts",
            expected=dict,
        )
        if set(type_counts) != set(EVIDENCE_TYPES):
            raise EvidenceCoverageVerificationError(
                "Coverage evidence-type counters are incomplete"
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in (*type_counts.values(), *provider_counts.values())
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage split counters are invalid"
            )
        trading = integers["trading_sessions"]
        covered = integers["sessions_with_records"]
        record_count = integers["record_count"]
        if (
            trading <= 0
            or covered > trading
            or integers["zero_record_sessions"] != trading - covered
            or integers["longest_missing_gap_sessions"]
            > integers["zero_record_sessions"]
            or not math.isclose(
                coverage,
                round(100.0 * covered / trading, 6),
                abs_tol=1e-9,
            )
            or sum(type_counts.values()) != record_count
            or sum(provider_counts.values()) != record_count
            or any(
                integers[field] > record_count
                for field in (
                    "before_cutoff_records",
                    "after_cutoff_records",
                    "non_trading_day_records",
                    "date_only_records",
                    "ogdc_specific_records",
                )
            )
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage split arithmetic is contradictory"
            )
        selected = _count_mapping_rows(mappings, profile, split)
        expected_types = {
            name: sum(item["evidence_type"] == name for item in selected)
            for name in sorted(EVIDENCE_TYPES)
        }
        expected_providers = _count_map(
            [str(item["provider_id"]) for item in selected]
        )
        expected_values = {
            "record_count": len(selected),
            "sessions_with_records": len(
                {item["assigned_session"] for item in selected}
            ),
            "before_cutoff_records": sum(
                item["arrival_class"] == "before_or_at_decision_cutoff"
                for item in selected
            ),
            "after_cutoff_records": sum(
                item["arrival_class"] == "after_decision_cutoff"
                for item in selected
            ),
            "non_trading_day_records": sum(
                item["arrival_class"] == "non_trading_day"
                for item in selected
            ),
            "date_only_records": sum(
                item["time_precision"] == "date_only" for item in selected
            ),
            "ogdc_specific_records": sum(
                str(item["ticker"]).upper() == "OGDC" for item in selected
            ),
        }
        if any(integers[name] != value for name, value in expected_values.items()):
            raise EvidenceCoverageVerificationError(
                "Coverage split counts disagree with session mappings"
            )
        if type_counts != expected_types or provider_counts != expected_providers:
            raise EvidenceCoverageVerificationError(
                "Coverage split populations disagree with session mappings"
            )
        observed[key] = EvidenceCoverageRow(
            profile=profile,
            split=split,
            trading_sessions=trading,
            sessions_with_records=covered,
            coverage_percent=coverage,
            record_count=record_count,
            zero_record_sessions=integers["zero_record_sessions"],
            longest_missing_gap_sessions=integers[
                "longest_missing_gap_sessions"
            ],
            before_cutoff_records=integers["before_cutoff_records"],
            after_cutoff_records=integers["after_cutoff_records"],
            non_trading_day_records=integers["non_trading_day_records"],
            date_only_records=integers["date_only_records"],
            ogdc_specific_records=integers["ogdc_specific_records"],
            evidence_type_counts=type_counts,
            provider_counts=provider_counts,
        )
    if set(observed) != expected_keys:
        raise EvidenceCoverageVerificationError(
            "Coverage split table is incomplete"
        )
    return observed


def _verify_monthly_rows(
    rows: Sequence[Mapping[str, str]],
    mappings: Sequence[Mapping[str, Any]],
    splits: Mapping[tuple[str, str], EvidenceCoverageRow],
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    keys: set[tuple[str, str, str]] = set()
    for row in rows:
        profile, split, month = row["profile"], row["split"], row["month"]
        key = (profile, split, month)
        if (
            profile not in EVIDENCE_COVERAGE_PROFILES
            or split not in AUDIT_PERIODS
            or not re.fullmatch(r"[0-9]{4}-(?:0[1-9]|1[0-2])", month)
            or key in keys
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage monthly table keys are invalid"
            )
        keys.add(key)
        trading = _integer(row["trading_sessions"], field="trading_sessions")
        covered = _integer(
            row["sessions_with_records"], field="sessions_with_records"
        )
        record_count = _integer(row["record_count"], field="record_count")
        date_only = _integer(
            row["date_only_records"], field="date_only_records"
        )
        coverage = _floating(
            row["coverage_percent"], field="coverage_percent"
        )
        type_counts = _json_cell(
            row["evidence_type_counts"],
            field="monthly evidence_type_counts",
            expected=dict,
        )
        if (
            trading <= 0
            or covered > trading
            or date_only > record_count
            or set(type_counts) != set(EVIDENCE_TYPES)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in type_counts.values()
            )
            or sum(type_counts.values()) != record_count
            or not math.isclose(
                coverage,
                round(100.0 * covered / trading, 6),
                abs_tol=1e-9,
            )
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage monthly arithmetic is contradictory"
            )
        selected = [
            item
            for item in _count_mapping_rows(mappings, profile, split)
            if str(item["assigned_session"]).startswith(f"{month}-")
        ]
        expected_types = {
            name: sum(item["evidence_type"] == name for item in selected)
            for name in sorted(EVIDENCE_TYPES)
        }
        if (
            record_count != len(selected)
            or covered
            != len({str(item["assigned_session"]) for item in selected})
            or date_only
            != sum(item["time_precision"] == "date_only" for item in selected)
            or type_counts != expected_types
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage monthly rows disagree with session mappings"
            )
        groups.setdefault((profile, split), []).append(
            {
                "trading_sessions": trading,
                "sessions_with_records": covered,
                "record_count": record_count,
                "date_only_records": date_only,
                "evidence_type_counts": type_counts,
            }
        )
    for profile in EVIDENCE_COVERAGE_PROFILES:
        for split in AUDIT_PERIODS:
            selected = groups.get((profile, split), [])
            if not selected:
                raise EvidenceCoverageVerificationError(
                    "Coverage monthly table is incomplete"
                )
            aggregate = splits[(profile, split)]
            if (
                sum(item["trading_sessions"] for item in selected)
                != aggregate.trading_sessions
                or sum(item["sessions_with_records"] for item in selected)
                != aggregate.sessions_with_records
                or sum(item["record_count"] for item in selected)
                != aggregate.record_count
                or sum(item["date_only_records"] for item in selected)
                != aggregate.date_only_records
                or {
                    name: sum(
                        item["evidence_type_counts"][name]
                        for item in selected
                    )
                    for name in sorted(EVIDENCE_TYPES)
                }
                != dict(aggregate.evidence_type_counts)
            ):
                raise EvidenceCoverageVerificationError(
                    "Coverage monthly totals disagree with split coverage"
                )


def _verify_missing_rows(
    rows: Sequence[Mapping[str, str]],
    splits: Mapping[tuple[str, str], EvidenceCoverageRow],
) -> None:
    groups: dict[tuple[str, str], list[tuple[date, date, int]]] = {}
    for row in rows:
        profile, split = row["profile"], row["split"]
        if profile not in EVIDENCE_COVERAGE_PROFILES or split not in AUDIT_PERIODS:
            raise EvidenceCoverageVerificationError(
                "Coverage missing-interval key is invalid"
            )
        start = _iso_date(row["start_date"], field="missing start_date")
        end = _iso_date(row["end_date"], field="missing end_date")
        count = _integer(
            row["missing_trading_sessions"],
            field="missing_trading_sessions",
        )
        if start > end or count <= 0 or count > (end - start).days + 1:
            raise EvidenceCoverageVerificationError(
                "Coverage missing interval is contradictory"
            )
        groups.setdefault((profile, split), []).append((start, end, count))
    for key, split_row in splits.items():
        selected = groups.get(key, [])
        ordered = sorted(selected, key=lambda item: item[0])
        if selected != ordered or any(
            previous[1] >= current[0]
            for previous, current in zip(ordered, ordered[1:])
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage missing intervals overlap or are not canonical"
            )
        if (
            sum(item[2] for item in selected)
            != split_row.zero_record_sessions
            or max((item[2] for item in selected), default=0)
            != split_row.longest_missing_gap_sessions
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage missing intervals disagree with split coverage"
            )


def _criteria_from_summary(summary: Mapping[str, Any]) -> EvidenceReadinessCriteria:
    value = summary.get("readiness_criteria")
    if not isinstance(value, Mapping):
        raise EvidenceCoverageVerificationError(
            "Coverage readiness criteria are missing"
        )
    try:
        criteria = EvidenceReadinessCriteria(**dict(value))
    except (TypeError, ValueError) as exc:
        raise EvidenceCoverageVerificationError(
            "Coverage readiness criteria are invalid"
        ) from exc
    if criteria.to_dict() != dict(value):
        raise EvidenceCoverageVerificationError(
            "Coverage readiness criteria are not canonical"
        )
    return criteria


def _verify_portable_recomputation(
    root: Path,
    *,
    summary: Mapping[str, Any],
    inputs: Mapping[str, Any],
    stock_reference: Mapping[str, Any],
    mapping_rows: Sequence[Mapping[str, str]],
    split_rows: Sequence[Mapping[str, str]],
    monthly_rows: Sequence[Mapping[str, str]],
    missing_rows: Sequence[Mapping[str, str]],
) -> EvidenceCoverageComputation:
    records = _read_evidence_records(root)
    stock, config = _read_stock_calendar(root)
    _verify_record_sources(records, inputs)
    _verify_stock_calendar_reference(stock, stock_reference)
    criteria = _criteria_from_summary(summary)
    try:
        recomputed = compute_evidence_coverage(
            records,
            stock,
            config,
            criteria=criteria,
        )
    except (TypeError, ValueError) as exc:
        raise EvidenceCoverageVerificationError(
            "Coverage records and calendar cannot be recomputed safely"
        ) from exc
    expected = _sorted_computation_tables(recomputed)
    contracts = (
        (mapping_rows, expected[0], _MAPPING_FIELDS, "session mappings"),
        (split_rows, expected[1], _SPLIT_FIELDS, "split coverage"),
        (monthly_rows, expected[2], _MONTHLY_FIELDS, "monthly coverage"),
        (missing_rows, expected[3], _MISSING_FIELDS, "missing intervals"),
    )
    for observed, expected_rows, fields, label in contracts:
        if list(observed) != _csv_formatted_rows(expected_rows, fields):
            raise EvidenceCoverageVerificationError(
                f"Coverage {label} differ from portable record/calendar recomputation"
            )
    for field, expected_value in recomputed.summary.items():
        if summary.get(field) != expected_value:
            raise EvidenceCoverageVerificationError(
                "Coverage final summary differs from portable recomputation"
            )
    return recomputed


def _verify_semantics(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    completion: Mapping[str, Any],
    summary: Mapping[str, Any],
    quality: Mapping[str, Any],
    inputs: Mapping[str, Any],
    stock_reference: Mapping[str, Any],
) -> tuple[int, dict[str, str]]:
    mapping_rows = _read_csv(root, "evidence_session_mappings.csv", _MAPPING_FIELDS)
    split_rows = _read_csv(root, "split_coverage.csv", _SPLIT_FIELDS)
    monthly_rows = _read_csv(root, "monthly_coverage.csv", _MONTHLY_FIELDS)
    missing_rows = _read_csv(root, "missing_intervals.csv", _MISSING_FIELDS)
    _verify_portable_recomputation(
        root,
        summary=summary,
        inputs=inputs,
        stock_reference=stock_reference,
        mapping_rows=mapping_rows,
        split_rows=split_rows,
        monthly_rows=monthly_rows,
        missing_rows=missing_rows,
    )
    mappings = _typed_mappings(mapping_rows)
    splits = _typed_split_rows(split_rows, mappings)
    _verify_monthly_rows(monthly_rows, mappings, splits)
    _verify_missing_rows(missing_rows, splits)

    criteria = _criteria_from_summary(summary)
    recalculated = {
        profile: calculate_evidence_readiness(
            profile,
            tuple(splits[(profile, split)] for split in MODEL_SPLITS),
            criteria=criteria,
        ).to_dict()
        for profile in EVIDENCE_COVERAGE_PROFILES
    }
    statuses = {
        profile: recalculated[profile]["status"]
        for profile in EVIDENCE_COVERAGE_PROFILES
    }
    profile_counts = {
        profile: sum(profile in item["profiles"] for item in mappings)
        for profile in EVIDENCE_COVERAGE_PROFILES
    }
    table_counts = {
        "mapping_rows": len(mapping_rows),
        "split_rows": len(split_rows),
        "monthly_rows": len(monthly_rows),
        "missing_interval_rows": len(missing_rows),
    }
    expected_quality_checks = {
        "separate_evidence_populations": True,
        "conservative_date_only_mapping": True,
        "precision_aware_interval_mapping": True,
        "portable_record_calendar_recomputation": True,
        "split_specific_coverage": True,
        "announcement_only_ready_guard": True,
    }
    if (
        summary.get("schema_version") != EVIDENCE_COVERAGE_SCHEMA_VERSION
        or summary.get("algorithm_version")
        != EVIDENCE_COVERAGE_ALGORITHM_VERSION
        or summary.get("academic_use_notice") != ACADEMIC_DISCLAIMER
        or summary.get("date_only_policy")
        != "strictly_next_eligible_trading_session"
        or summary.get("record_count_profiled") != len(mappings)
        or not isinstance(summary.get("record_count_input"), int)
        or isinstance(summary.get("record_count_input"), bool)
        or int(summary["record_count_input"]) < len(mappings)
        or summary.get("profile_input_counts") != profile_counts
        or summary.get("readiness_status_by_profile") != statuses
        or summary.get("readiness") != recalculated
        or summary.get("table_counts") != table_counts
    ):
        raise EvidenceCoverageVerificationError(
            "Coverage final summary contradicts the sealed tables"
        )
    if (
        quality.get("schema_version")
        != EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION
        or quality.get("status") != "PASS"
        or quality.get("mapping_count") != len(mappings)
        or quality.get("table_counts") != table_counts
        or quality.get("profile_input_counts") != profile_counts
        or quality.get("readiness") != recalculated
        or quality.get("checks") != expected_quality_checks
        or quality.get("academic_disclaimer") != ACADEMIC_DISCLAIMER
    ):
        raise EvidenceCoverageVerificationError(
            "Coverage quality report contradicts the sealed tables"
        )
    for value in (manifest, completion):
        if (
            value.get("mapping_count") != len(mappings)
            or value.get("table_counts") != table_counts
            or value.get("readiness_status_by_profile") != statuses
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage seal metadata contradicts the sealed tables"
            )
    return len(mappings), statuses


def _verify_root(root: Path, *, expected_run_id: str) -> EvidenceCoverageRunVerification:
    verify_pack_modes(
        root,
        error_type=EvidenceCoverageVerificationError,
        label="Evidence-coverage",
    )
    names: list[str] = []
    for item in root.iterdir():
        names.append(item.name)
    if set(names) != set(_ALL_FILES) or len(names) != len(_ALL_FILES):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage artifact set differs from the flat pack contract"
        )

    run_id = validate_evidence_coverage_run_id(expected_run_id)
    manifest = _read_json(root / "run_manifest.json")
    completion = _read_json(root / "completion.json")
    summary = _read_json(root / "final_summary.json")
    quality = _read_json(root / "quality_report.json")
    inputs = _read_json(root / "input_evidence_reference.json")
    stock = _read_json(root / "stock_artifact_reference.json")
    for value in (manifest, completion, summary, quality, inputs, stock):
        _assert_hygiene(value)
    if (
        inputs.get("schema_version")
        != EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION
        or not isinstance(inputs.get("inputs"), list)
        or not inputs["inputs"]
        or any(not isinstance(item, Mapping) for item in inputs["inputs"])
        or not stock
    ):
        raise EvidenceCoverageVerificationError(
            "Coverage input or stock references are invalid"
        )
    _validate_input_reference_document(inputs)
    _validate_stock_reference(stock)
    if _referenced_record_count(inputs) != summary.get("record_count_input"):
        raise EvidenceCoverageVerificationError(
            "Coverage source-reference counts contradict the final summary"
        )
    if any(value.get("run_id") != run_id for value in (manifest, completion, summary)):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage pack identities disagree"
        )
    if (
        manifest.get("schema_version")
        != EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION
        or manifest.get("coverage_schema_version")
        != EVIDENCE_COVERAGE_SCHEMA_VERSION
        or manifest.get("algorithm_version")
        != EVIDENCE_COVERAGE_ALGORITHM_VERSION
        or completion.get("schema_version")
        != EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION
        or completion.get("coverage_schema_version")
        != EVIDENCE_COVERAGE_SCHEMA_VERSION
        or completion.get("algorithm_version")
        != EVIDENCE_COVERAGE_ALGORITHM_VERSION
    ):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage schema or algorithm version differs"
        )
    if any(value.get("status") != "PASS" for value in (manifest, completion, summary)):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage statuses disagree"
        )
    if any(
        value.get("finalization_state") != "COMMITTED"
        for value in (manifest, completion, summary)
    ):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage pack is not committed"
        )
    if (
        manifest.get("completion_marker") != "completion.json"
        or completion.get("completed_at") != manifest.get("completed_at")
        or summary.get("created_at") != manifest.get("created_at")
    ):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage completion provenance differs"
        )
    _timestamp(str(manifest.get("created_at") or ""), field="created_at", optional=False)
    _timestamp(
        str(manifest.get("completed_at") or ""),
        field="completed_at",
        optional=False,
    )

    manifest_path = root / "run_manifest.json"
    summary_path = root / "final_summary.json"
    if (
        completion.get("manifest_sha256") != sha256_file(manifest_path)
        or completion.get("manifest_size_bytes") != manifest_path.stat().st_size
        or completion.get("summary_sha256") != sha256_file(summary_path)
        or completion.get("summary_size_bytes") != summary_path.stat().st_size
    ):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage completion seal is invalid"
        )
    inventory = _normalize_inventory(
        manifest.get("payload_artifact_inventory")
    )
    if {item["path"] for item in inventory} != set(_PAYLOAD_FILES):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage payload inventory is incomplete"
        )
    digest = _inventory_digest(inventory)
    if (
        manifest.get("payload_artifact_inventory_sha256") != digest
        or completion.get("payload_artifact_inventory_sha256") != digest
        or completion.get("artifact_count") != len(inventory)
    ):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage inventory seal differs"
        )
    try:
        observed = regular_file_inventory(
            root, files=[item["path"] for item in inventory]
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage payload inventory could not be verified"
        ) from exc
    if observed != inventory:
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage payload inventory differs"
        )
    input_sha = sha256_file(root / "input_evidence_reference.json")
    stock_sha = sha256_file(root / "stock_artifact_reference.json")
    records_sha = sha256_file(root / "normalized_evidence.jsonl")
    calendar_sha = sha256_file(root / "stock_calendar.json")
    fingerprint = _sha256_bytes(
        _canonical_bytes({"input_references": inputs, "stock_reference": stock})
    )
    if (
        manifest.get("input_evidence_reference_sha256") != input_sha
        or manifest.get("stock_artifact_reference_sha256") != stock_sha
        or manifest.get("normalized_evidence_sha256") != records_sha
        or manifest.get("stock_calendar_sha256") != calendar_sha
        or manifest.get("input_fingerprint_sha256") != fingerprint
        or summary.get("input_evidence_reference_sha256") != input_sha
        or summary.get("stock_artifact_reference_sha256") != stock_sha
        or summary.get("normalized_evidence_sha256") != records_sha
        or summary.get("stock_calendar_sha256") != calendar_sha
    ):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage input references differ from their seals"
        )
    mapping_count, statuses = _verify_semantics(
        root,
        manifest=manifest,
        completion=completion,
        summary=summary,
        quality=quality,
        inputs=inputs,
        stock_reference=stock,
    )
    return EvidenceCoverageRunVerification(
        run_id=run_id,
        status="PASS",
        readiness_status_by_profile=statuses,
        mapping_count=mapping_count,
        artifact_count=len(inventory),
        manifest_sha256=sha256_file(manifest_path),
        payload_inventory_sha256=digest,
    )


def verify_evidence_coverage_pack(
    run_directory: str | Path,
) -> EvidenceCoverageRunVerification:
    candidate = reject_symlink_components(
        run_directory,
        error_type=EvidenceCoverageVerificationError,
        label="Evidence-coverage run path",
    )
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage run root is missing"
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise EvidenceCoverageVerificationError(
            "Evidence-coverage run root must be a real directory"
        )
    root = candidate.resolve(strict=True)
    return _verify_root(root, expected_run_id=root.name)


def publish_evidence_coverage_run(
    computation: EvidenceCoverageComputation,
    *,
    output_root: str | Path,
    input_references: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    stock_reference: Mapping[str, Any],
    repository_root: str | Path,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> Path:
    """Seal and atomically publish one immutable evidence-coverage pack."""

    if not isinstance(computation, EvidenceCoverageComputation):
        raise TypeError("computation must be EvidenceCoverageComputation")
    if not isinstance(stock_reference, Mapping) or not stock_reference:
        raise ValueError("stock_reference must be a non-empty mapping")
    instant = created_at or datetime.now(timezone.utc)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    instant = instant.astimezone(timezone.utc)
    selected_id = validate_evidence_coverage_run_id(
        run_id or make_evidence_coverage_run_id(instant)
    )
    root = prepare_output_root(
        output_root,
        repository_root=repository_root,
        error_type=ValueError,
        label="Coverage output root",
    )
    destination = root / selected_id
    if os.path.lexists(destination):
        raise EvidenceCoverageRunCollisionError(
            f"Evidence-coverage run already exists: {selected_id}"
        )
    staging = root / f".{selected_id}.staging-{uuid4().hex}"
    staging.mkdir(mode=0o700)
    published = False
    try:
        normalized_inputs = _normalize_input_references(input_references)
        normalized_stock = _sanitize(dict(stock_reference))
        _assert_hygiene(normalized_stock)
        _validate_stock_reference(normalized_stock)
        if _referenced_record_count(normalized_inputs) != computation.summary.get(
            "record_count_input"
        ):
            raise EvidenceCoverageVerificationError(
                "Coverage source-reference counts contradict the computation"
            )
        _verify_record_sources(computation.source_records, normalized_inputs)
        record_payloads = [record.to_dict() for record in computation.source_records]
        _assert_hygiene(record_payloads)
        _write_json(staging / "input_evidence_reference.json", normalized_inputs)
        _write_json(staging / "stock_artifact_reference.json", normalized_stock)
        _write_evidence_jsonl(
            staging / "normalized_evidence.jsonl",
            computation.source_records,
        )
        calendar = _stock_calendar_document(computation)
        _assert_hygiene(calendar)
        _write_json(staging / "stock_calendar.json", calendar)

        mappings, split_rows, monthly_rows, missing_rows = (
            _sorted_computation_tables(computation)
        )
        for value in (mappings, split_rows, monthly_rows, missing_rows):
            _assert_hygiene(value)
        _write_csv(
            staging / "evidence_session_mappings.csv",
            mappings,
            _MAPPING_FIELDS,
        )
        _write_csv(staging / "split_coverage.csv", split_rows, _SPLIT_FIELDS)
        _write_csv(
            staging / "monthly_coverage.csv", monthly_rows, _MONTHLY_FIELDS
        )
        _write_csv(
            staging / "missing_intervals.csv", missing_rows, _MISSING_FIELDS
        )

        readiness = {
            profile: computation.readiness[profile].to_dict()
            for profile in EVIDENCE_COVERAGE_PROFILES
        }
        statuses = {
            profile: readiness[profile]["status"]
            for profile in EVIDENCE_COVERAGE_PROFILES
        }
        profile_counts = {
            profile: sum(profile in item["profiles"] for item in mappings)
            for profile in EVIDENCE_COVERAGE_PROFILES
        }
        table_counts = {
            "mapping_rows": len(mappings),
            "split_rows": len(split_rows),
            "monthly_rows": len(monthly_rows),
            "missing_interval_rows": len(missing_rows),
        }
        quality = {
            "schema_version": EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION,
            "status": "PASS",
            "mapping_count": len(mappings),
            "table_counts": table_counts,
            "profile_input_counts": profile_counts,
            "readiness": readiness,
            "checks": {
                "separate_evidence_populations": True,
                "conservative_date_only_mapping": True,
                "precision_aware_interval_mapping": True,
                "portable_record_calendar_recomputation": True,
                "split_specific_coverage": True,
                "announcement_only_ready_guard": True,
            },
            "academic_disclaimer": ACADEMIC_DISCLAIMER,
        }
        _write_json(staging / "quality_report.json", quality)

        input_sha = sha256_file(staging / "input_evidence_reference.json")
        stock_sha = sha256_file(staging / "stock_artifact_reference.json")
        records_sha = sha256_file(staging / "normalized_evidence.jsonl")
        calendar_sha = sha256_file(staging / "stock_calendar.json")
        final_summary = dict(_sanitize(dict(computation.summary)))
        final_summary.update(
            {
                "run_id": selected_id,
                "status": "PASS",
                "finalization_state": "COMMITTED",
                "created_at": utc_iso(instant),
                "academic_use_notice": ACADEMIC_DISCLAIMER,
                "profile_input_counts": profile_counts,
                "readiness_status_by_profile": statuses,
                "readiness": readiness,
                "table_counts": table_counts,
                "input_evidence_reference_sha256": input_sha,
                "stock_artifact_reference_sha256": stock_sha,
                "normalized_evidence_sha256": records_sha,
                "stock_calendar_sha256": calendar_sha,
            }
        )
        _assert_hygiene(final_summary)
        _write_json(staging / "final_summary.json", final_summary)

        payload_inventory = artifact_inventory(
            staging, expected_files=sorted(_PAYLOAD_FILES)
        )
        inventory_sha = _inventory_digest(payload_inventory)
        manifest = {
            "schema_version": EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION,
            "coverage_schema_version": EVIDENCE_COVERAGE_SCHEMA_VERSION,
            "algorithm_version": EVIDENCE_COVERAGE_ALGORITHM_VERSION,
            "run_id": selected_id,
            "status": "PASS",
            "finalization_state": "COMMITTED",
            "created_at": utc_iso(instant),
            "completed_at": utc_iso(instant),
            "mapping_count": len(mappings),
            "table_counts": table_counts,
            "readiness_status_by_profile": statuses,
            "input_evidence_reference_sha256": input_sha,
            "stock_artifact_reference_sha256": stock_sha,
            "normalized_evidence_sha256": records_sha,
            "stock_calendar_sha256": calendar_sha,
            "input_fingerprint_sha256": _sha256_bytes(
                _canonical_bytes(
                    {
                        "input_references": normalized_inputs,
                        "stock_reference": normalized_stock,
                    }
                )
            ),
            "payload_artifact_inventory": payload_inventory,
            "payload_artifact_inventory_sha256": inventory_sha,
            "inventory_excludes": sorted(_SEAL_FILES),
            "completion_marker": "completion.json",
            "implementation_provenance": _git_provenance(
                Path(repository_root)
            ),
            "academic_disclaimer": ACADEMIC_DISCLAIMER,
        }
        _assert_hygiene(manifest)
        _write_json(staging / "run_manifest.json", manifest)
        completion = {
            "schema_version": EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION,
            "coverage_schema_version": EVIDENCE_COVERAGE_SCHEMA_VERSION,
            "algorithm_version": EVIDENCE_COVERAGE_ALGORITHM_VERSION,
            "run_id": selected_id,
            "status": "PASS",
            "finalization_state": "COMMITTED",
            "completed_at": utc_iso(instant),
            "manifest_sha256": sha256_file(staging / "run_manifest.json"),
            "manifest_size_bytes": (staging / "run_manifest.json").stat().st_size,
            "summary_sha256": sha256_file(staging / "final_summary.json"),
            "summary_size_bytes": (staging / "final_summary.json").stat().st_size,
            "payload_artifact_inventory_sha256": inventory_sha,
            "artifact_count": len(payload_inventory),
            "mapping_count": len(mappings),
            "table_counts": table_counts,
            "readiness_status_by_profile": statuses,
        }
        # This non-circular completion marker is intentionally written last.
        _write_json(staging / "completion.json", completion)
        make_pack_private(
            staging,
            error_type=EvidenceCoverageVerificationError,
            label="Evidence-coverage",
        )
        _verify_root(staging, expected_run_id=selected_id)
        publication_inventory = artifact_inventory(
            staging, expected_files=sorted(_ALL_FILES)
        )
        try:
            publish_staged_run(
                staging,
                destination,
                expected_inventory=publication_inventory,
            )
        except SentimentRunCollisionError as exc:
            raise EvidenceCoverageRunCollisionError(
                f"Evidence-coverage run already exists: {selected_id}"
            ) from exc
        published = True
        return destination
    finally:
        if not published:
            remove_staging_tree(staging, output_root=root)


__all__ = [
    "ACADEMIC_DISCLAIMER",
    "EVIDENCE_COVERAGE_PACK_SCHEMA_VERSION",
    "EvidenceCoverageRunCollisionError",
    "EvidenceCoverageRunVerification",
    "EvidenceCoverageVerificationError",
    "make_evidence_coverage_run_id",
    "publish_evidence_coverage_run",
    "validate_evidence_coverage_run_id",
    "verify_evidence_coverage_pack",
]
