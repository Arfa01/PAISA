"""Deterministic historical-news versus stock-session coverage auditing.

This module deliberately stops at evidence mapping and coverage measurement.  It
does not aggregate sentiment, train a model, create a signal, or change a
portfolio.  All readiness labels are engineering/data-coverage gates, not
claims of statistical sufficiency or financial advice.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
import csv
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paisa.news.verification import NewsRunVerificationError, verify_news_run

from paisa.news.providers.manual_csv import LICENSING_REVIEW_REQUIRED


COVERAGE_SCHEMA_VERSION = "1.0"
COVERAGE_ALGORITHM_VERSION = "news-price-coverage-v2"
ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
_COVERAGE_MANIFEST = "run_manifest.json"
_COVERAGE_COMPLETION = "completion.json"
_COVERAGE_SEAL_FILES = frozenset({_COVERAGE_MANIFEST, _COVERAGE_COMPLETION})
READINESS_STATUSES = {
    "READY_FOR_AGGREGATION",
    "PARTIALLY_COVERED",
    "INSUFFICIENT_OVERLAP",
    "NO_OVERLAP",
}
POPULATIONS = ("all_financial_news", "ogdc_linked_news", "energy_sector_news")
SPLITS = ("train", "validation", "test")
_SAFE_RUN_ID = re.compile(
    r"^COVERAGE-\d{8}-\d{6}-[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_QUERY_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?(?:key|token)|token|secret|password|"
    r"credential|authorization|auth|signature|signed|session|cookie|jwt)(?:$|[_-])",
    re.IGNORECASE,
)
_TRACKING_QUERY_KEYS = {
    "dclid",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}


class CoverageAuditError(ValueError):
    """The requested audit evidence is missing, unsafe, or contradictory."""


class CoverageRunCollisionError(FileExistsError):
    """The immutable coverage run destination already exists."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CoverageAuditError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _aware_utc(value, field_name="timestamp").isoformat().replace(
        "+00:00", "Z"
    )


def parse_timestamp(value: Any, *, field_name: str) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise CoverageAuditError(f"{field_name} is not a valid ISO timestamp") from exc
    return _aware_utc(parsed, field_name=field_name)


def _parse_date(value: Any, *, field_name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise CoverageAuditError(f"{field_name} must be YYYY-MM-DD") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _safe_relative(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CoverageAuditError(f"{field_name} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise CoverageAuditError(f"{field_name} must be a normalized relative path")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise CoverageAuditError(f"{field_name} contains an unsafe path component")
    return value


def _regular_file(path: Path, *, root: Path | None = None, label: str) -> Path:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise CoverageAuditError(f"Missing {label}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise CoverageAuditError(f"{label} must be a regular file")
    if root is not None:
        try:
            path.resolve(strict=True).relative_to(root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise CoverageAuditError(f"{label} escapes its evidence root") from exc
    return path


def _read_json(path: Path, *, root: Path | None = None, label: str) -> dict[str, Any]:
    source = _regular_file(path, root=root, label=label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CoverageAuditError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise CoverageAuditError(f"{label} must contain a JSON object")
    return value


def _json_list(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None or str(value).strip() == "":
        return ()
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise CoverageAuditError(f"{field_name} must contain a JSON array") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise CoverageAuditError(f"{field_name} must contain a JSON string array")
    return tuple(str(item) for item in parsed)


def load_source_permission_policy(
    policy_path: str | Path,
    source_assessment_path: str | Path,
) -> SourcePermissionPolicy:
    """Load decisions whose policy and reviewed Markdown bytes are both bound."""

    policy_candidate = Path(policy_path)
    assessment_candidate = Path(source_assessment_path)
    if policy_candidate.is_symlink() or assessment_candidate.is_symlink():
        raise CoverageAuditError("Source permission evidence must not use symlinks")
    policy_source = policy_candidate.resolve()
    assessment_source = assessment_candidate.resolve()
    policy = _read_json(policy_source, label="source permission policy")
    assessment = _regular_file(
        assessment_source, label="historical-news source assessment"
    )
    if policy.get("schema_version") != "1.0":
        raise CoverageAuditError("Source permission policy schema is unsupported")
    assessment_document = _safe_relative(
        policy.get("assessment_document"),
        field_name="source permission assessment reference",
    )
    expected_assessment_sha = policy.get("assessment_sha256")
    if (
        not isinstance(expected_assessment_sha, str)
        or not _SHA256.fullmatch(expected_assessment_sha)
        or sha256_file(assessment) != expected_assessment_sha
    ):
        raise CoverageAuditError(
            "Source permission policy does not match the reviewed assessment bytes"
        )
    if Path(assessment_document).name != assessment.name:
        raise CoverageAuditError(
            "Source permission assessment label does not match the supplied document"
        )
    raw_decisions = policy.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise CoverageAuditError("Source permission policy has no decisions")
    decisions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in raw_decisions:
        if not isinstance(item, Mapping):
            raise CoverageAuditError("Source permission decision must be an object")
        decision_id = str(item.get("decision_id") or "").strip()
        provider = str(item.get("provider") or "").strip()
        provider_mode = str(item.get("provider_mode") or "").strip().casefold()
        status = str(item.get("status") or "").strip().upper()
        scope = str(item.get("scope") or "").strip()
        if not decision_id or not provider or not scope:
            raise CoverageAuditError("Source permission decision is incomplete")
        if provider_mode not in {"real", "imported"}:
            raise CoverageAuditError("Source permission provider mode is invalid")
        if status not in {"PERMITTED", "REVIEW_REQUIRED"}:
            raise CoverageAuditError("Source permission decision status is invalid")
        note_sha = item.get("licensing_usage_note_sha256")
        if note_sha is not None and (
            not isinstance(note_sha, str) or not _SHA256.fullmatch(note_sha)
        ):
            raise CoverageAuditError("Source permission note checksum is invalid")
        required_options = item.get("required_provider_options", {})
        if not isinstance(required_options, Mapping):
            raise CoverageAuditError(
                "Source permission required provider options are invalid"
            )
        allowed_run_ids = item.get("allowed_run_ids")
        if allowed_run_ids is not None and (
            not isinstance(allowed_run_ids, list)
            or not allowed_run_ids
            or not all(isinstance(value, str) and value for value in allowed_run_ids)
        ):
            raise CoverageAuditError("Source permission run allowlist is invalid")
        key = (provider, provider_mode)
        if key in decisions:
            raise CoverageAuditError("Source permission decision is duplicated")
        decisions[key] = dict(item)
    return SourcePermissionPolicy(
        decisions=decisions,
        reference={
            "schema_version": "1.0",
            "policy_reference": policy_source.name,
            "policy_sha256": sha256_file(policy_source),
            "assessment_document": assessment_document,
            "assessment_sha256": expected_assessment_sha,
        },
    )


def _source_permission_decision(
    *,
    policy: SourcePermissionPolicy | None,
    run_id: str,
    provider: str,
    provider_mode: str,
    provider_options: Mapping[str, Any],
    licensing_usage_note: str | None,
) -> dict[str, Any]:
    if policy is None:
        return {
            "passed": False,
            "status": "UNVERIFIED_POLICY",
            "reason": "No checksum-bound source permission policy was supplied.",
        }
    configured = policy.decisions.get((provider, provider_mode))
    if configured is None:
        return {
            "passed": False,
            "status": "UNASSESSED",
            "reason": "The selected provider/mode has no reviewed policy decision.",
        }
    status = str(configured["status"]).upper()
    result = {
        "passed": False,
        "status": status,
        "decision_id": configured["decision_id"],
        "scope": configured["scope"],
    }
    if status != "PERMITTED":
        result["reason"] = "The reviewed policy requires additional permission review."
        return result
    allowed_run_ids = configured.get("allowed_run_ids")
    if isinstance(allowed_run_ids, list) and run_id not in allowed_run_ids:
        result["status"] = "RUN_NOT_PERMITTED"
        result["reason"] = "The ingestion run is outside the reviewed allowlist."
        return result
    expected_note_sha = configured.get("licensing_usage_note_sha256")
    actual_note_sha = (
        _sha256_bytes(licensing_usage_note.encode("utf-8"))
        if licensing_usage_note is not None
        else None
    )
    if expected_note_sha is not None and actual_note_sha != expected_note_sha:
        result["status"] = "LICENSING_NOTE_MISMATCH"
        result["reason"] = "The retained licensing note differs from the reviewed policy."
        return result
    required_options = configured.get("required_provider_options", {})
    mismatched_options = sorted(
        key
        for key, expected in required_options.items()
        if provider_options.get(key) != expected
    )
    if mismatched_options:
        result["status"] = "PROVIDER_CONFIGURATION_MISMATCH"
        result["reason"] = (
            "Reviewed provider controls differ: " + ", ".join(mismatched_options)
        )
        return result
    result["passed"] = True
    result["reason"] = "The checksum-bound policy permits this provider configuration."
    return result


def _public_url(value: str | None) -> str | None:
    """Retain a traceable URL without copying credentials into reports."""

    if not value or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    query = urlencode(
        [
            (key, "[REDACTED]" if _SENSITIVE_QUERY_KEY.search(key) else item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in _TRACKING_QUERY_KEYS
        ],
        doseq=True,
    )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def make_coverage_run_id(now: datetime | None = None) -> str:
    instant = _aware_utc(now or _utc_now(), field_name="run ID time")
    return f"COVERAGE-{instant.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"


def validate_coverage_run_id(value: str) -> str:
    run_id = value.strip()
    if run_id != value or not _SAFE_RUN_ID.fullmatch(run_id):
        raise CoverageAuditError(
            "Coverage run IDs must use COVERAGE-YYYYMMDD-HHMMSS-<unique-suffix>"
        )
    return run_id


@dataclass(frozen=True, slots=True)
class ReadinessThresholds:
    """Configurable engineering gates; these do not prove statistical power."""

    ready_minimum_all_news_coverage_percent_each_split: float = 80.0
    ready_minimum_ogdc_coverage_percent_each_split: float = 20.0
    ready_maximum_all_news_missing_gap_sessions: int = 10
    ready_minimum_distinct_sources: int = 2
    ready_minimum_publication_time_completeness_percent: float = 95.0
    ready_minimum_available_at_completeness_percent: float = 100.0
    partial_minimum_all_news_coverage_percent_each_split: float = 5.0
    partial_minimum_ogdc_coverage_percent_each_split: float = 1.0

    def __post_init__(self) -> None:
        percentage_fields = (
            "ready_minimum_all_news_coverage_percent_each_split",
            "ready_minimum_ogdc_coverage_percent_each_split",
            "ready_minimum_publication_time_completeness_percent",
            "ready_minimum_available_at_completeness_percent",
            "partial_minimum_all_news_coverage_percent_each_split",
            "partial_minimum_ogdc_coverage_percent_each_split",
        )
        for name in percentage_fields:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 100.0:
                raise CoverageAuditError(f"{name} must be between 0 and 100")
        if self.ready_maximum_all_news_missing_gap_sessions < 0:
            raise CoverageAuditError(
                "ready_maximum_all_news_missing_gap_sessions cannot be negative"
            )
        if self.ready_minimum_distinct_sources < 1:
            raise CoverageAuditError("ready_minimum_distinct_sources must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReadinessThresholds":
        allowed = set(cls.__dataclass_fields__)
        unexpected = sorted(set(value) - allowed - {"schema_version", "purpose"})
        if unexpected:
            raise CoverageAuditError(
                "Readiness configuration has unsupported fields: "
                + ", ".join(unexpected)
            )
        if value.get("schema_version", COVERAGE_SCHEMA_VERSION) != COVERAGE_SCHEMA_VERSION:
            raise CoverageAuditError("Unsupported readiness configuration schema")
        kwargs = {name: value[name] for name in allowed if name in value}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CoverageConfig:
    market_timezone: str = "Asia/Karachi"
    decision_cutoff: time = time(15, 30)
    thresholds: ReadinessThresholds = field(default_factory=ReadinessThresholds)

    def __post_init__(self) -> None:
        try:
            ZoneInfo(self.market_timezone)
        except ZoneInfoNotFoundError as exc:
            raise CoverageAuditError("Unknown market timezone") from exc
        if self.decision_cutoff.tzinfo is not None:
            raise CoverageAuditError("decision_cutoff must be a local wall-clock time")

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_timezone": self.market_timezone,
            "decision_cutoff": self.decision_cutoff.isoformat(timespec="seconds"),
            "readiness_thresholds": self.thresholds.to_dict(),
            "threshold_interpretation": (
                "Engineering/data-coverage gates only; not proof of statistical "
                "sufficiency or predictive value."
            ),
        }


def parse_cutoff(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise CoverageAuditError("Decision cutoff must be HH:MM or HH:MM:SS") from exc
    if parsed.tzinfo is not None:
        raise CoverageAuditError("Decision cutoff must not include a timezone offset")
    return parsed


@dataclass(frozen=True, slots=True)
class NewsOccurrence:
    occurrence_key: str
    ingestion_run_id: str
    article_id: str
    canonical_article_id: str
    provider: str
    provider_mode: str
    source_name: str
    source_url: str | None
    title: str | None
    published_at: datetime | None
    available_at: datetime | None
    fetched_at: datetime | None
    published_timezone: str | None
    language: str | None
    tickers: tuple[str, ...]
    companies: tuple[str, ...]
    sectors: tuple[str, ...]
    content_sha256: str
    raw_artifact_reference: str
    data_class: str
    duplicate_of: str | None
    entity_links: tuple[Mapping[str, str], ...]

    @property
    def is_ogdc(self) -> bool:
        return any(
            str(link.get("ticker") or "").upper() == "OGDC"
            for link in self.entity_links
        )

    @property
    def is_energy(self) -> bool:
        return any(
            sector.casefold() in {"energy", "oil & gas", "oil and gas"}
            for sector in self.sectors
        )


@dataclass(frozen=True, slots=True)
class NewsEvidence:
    occurrences: tuple[NewsOccurrence, ...]
    run_references: tuple[Mapping[str, Any], ...]
    database_fingerprints: Mapping[str, Any]
    permission_review_required: bool = False
    permission_notes: tuple[str, ...] = ()
    permission_policy_reference: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourcePermissionPolicy:
    """Checksum-bound, machine-readable decisions from the source assessment."""

    decisions: Mapping[tuple[str, str], Mapping[str, Any]]
    reference: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StockEvidence:
    symbol: str
    trading_dates: tuple[date, ...]
    split_dates: Mapping[str, tuple[date, ...]]
    reference: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SessionAssignment:
    session_date: date | None
    arrival_class: str
    mapping_status: str


@dataclass(frozen=True, slots=True)
class CoverageComputation:
    trading_day_rows: tuple[Mapping[str, Any], ...]
    monthly_rows: tuple[Mapping[str, Any], ...]
    split_rows: tuple[Mapping[str, Any], ...]
    missing_interval_rows: tuple[Mapping[str, Any], ...]
    article_session_rows: tuple[Mapping[str, Any], ...]
    ogdc_article_rows: tuple[Mapping[str, Any], ...]
    quality_report: Mapping[str, Any]
    final_summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CoverageRunVerification:
    """Offline integrity result for one completed coverage evidence pack."""

    run_id: str
    status: str
    readiness_status: str
    artifact_count: int
    manifest_sha256: str
    artifact_inventory_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "readiness_status": self.readiness_status,
            "artifact_count": self.artifact_count,
            "manifest_sha256": self.manifest_sha256,
            "artifact_inventory_sha256": self.artifact_inventory_sha256,
            "completion_acceptance": "ARTIFACT_AND_MANIFEST_BOUND",
            "verification": "PASS",
        }


def discover_news_run_directories(root: str | Path) -> list[Path]:
    candidate = Path(root)
    if candidate.is_symlink():
        raise CoverageAuditError("News run root must be a regular directory")
    base = candidate.resolve()
    if not base.is_dir():
        raise CoverageAuditError("News run root must be a regular directory")
    selected: list[Path] = []
    for child in sorted(base.iterdir(), key=lambda item: item.name):
        if not child.is_dir() or child.is_symlink():
            continue
        manifest_path = child / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _read_json(manifest_path, root=child, label="news run manifest")
        if manifest.get("status") in {"PASS", "PARTIAL"}:
            selected.append(child)
    if not selected:
        raise CoverageAuditError("No completed news runs were found")
    return selected


def _manifest_inventory(manifest: Mapping[str, Any]) -> dict[str, tuple[str, int | None]]:
    raw = manifest.get("artifact_inventory")
    if not isinstance(raw, list):
        # Historical schema-1 packs used the same key; fail closed for anything
        # without a checksum inventory.
        raise CoverageAuditError("News manifest has no artifact checksum inventory")
    inventory: dict[str, tuple[str, int | None]] = {}
    previous = ""
    for item in raw:
        if not isinstance(item, Mapping):
            raise CoverageAuditError("News artifact inventory entries must be objects")
        relative = _safe_relative(item.get("path"), field_name="news artifact path")
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CoverageAuditError("News artifact inventory has an invalid SHA-256")
        if size is not None and (
            not isinstance(size, int) or isinstance(size, bool) or size < 0
        ):
            raise CoverageAuditError("News artifact inventory has an invalid size")
        if relative in inventory or relative < previous:
            raise CoverageAuditError(
                "News artifact inventory must be unique and deterministically ordered"
            )
        previous = relative
        inventory[relative] = (digest, size)
    return inventory


def _validate_inventoried_file(
    root: Path,
    relative: str,
    inventory: Mapping[str, tuple[str, int | None]],
) -> Path:
    if relative not in inventory:
        raise CoverageAuditError(f"Required news artifact is not inventoried: {relative}")
    path = _regular_file(root / relative, root=root, label=f"news artifact {relative}")
    expected_digest, expected_size = inventory[relative]
    metadata = path.stat()
    if expected_size is not None and metadata.st_size != expected_size:
        raise CoverageAuditError(f"News artifact size mismatch: {relative}")
    if sha256_file(path) != expected_digest:
        raise CoverageAuditError(f"News artifact checksum mismatch: {relative}")
    return path


def _load_entity_link_evidence(
    root: Path,
    inventory: Mapping[str, tuple[str, int | None]],
) -> dict[str, tuple[Mapping[str, str], ...]]:
    relative = "article_entity_links.csv"
    if relative not in inventory:
        return {}
    path = _validate_inventoried_file(root, relative, inventory)
    required = {
        "link_id",
        "article_id",
        "ticker",
        "company_name",
        "matched_alias",
        "match_method",
        "match_location",
        "confidence",
    }
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    seen_link_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise CoverageAuditError(
                    "News entity-link CSV is missing required evidence fields"
                )
            for index, row in enumerate(reader, start=2):
                evidence = {
                    field: str(row.get(field) or "").strip() for field in required
                }
                if any(not value for value in evidence.values()):
                    raise CoverageAuditError(
                        f"News entity-link evidence is incomplete at row {index}"
                    )
                link_id = evidence["link_id"]
                if link_id in seen_link_ids:
                    raise CoverageAuditError("News entity-link identity is duplicated")
                seen_link_ids.add(link_id)
                grouped[evidence["article_id"]].append(evidence)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CoverageAuditError("News entity-link CSV could not be read") from exc
    return {
        article_id: tuple(sorted(values, key=lambda item: item["link_id"]))
        for article_id, values in grouped.items()
    }


def _load_one_news_run(
    run_dir: Path,
    *,
    real_mode: bool,
    allow_legacy_unsealed: bool,
    database_path: str | Path | None,
    source_permission_policy: SourcePermissionPolicy | None,
) -> tuple[list[NewsOccurrence], dict[str, Any]]:
    root = run_dir.resolve()
    if run_dir.is_symlink() or not root.is_dir():
        raise CoverageAuditError("News run directory must not be a symlink")
    manifest_path = _regular_file(
        root / "run_manifest.json", root=root, label="news run manifest"
    )
    manifest = _read_json(manifest_path, root=root, label="news run manifest")
    run_id = str(manifest.get("run_id") or "")
    if run_id != root.name:
        raise CoverageAuditError("News run directory and manifest identity differ")
    if manifest.get("status") not in {"PASS", "PARTIAL"}:
        raise CoverageAuditError("Only completed PASS/PARTIAL news runs may be audited")
    provider_mode = str(
        manifest.get("provider_mode") or manifest.get("data_kind") or ""
    ).casefold()
    if provider_mode not in {"real", "imported", "fixture"}:
        raise CoverageAuditError("News run provider mode is unsupported")
    if real_mode and provider_mode == "fixture":
        raise CoverageAuditError("Fixture/mock/synthetic/test news is forbidden in real mode")

    completion_path = root / "completion.json"
    completion_marker_present = completion_path.is_file()
    if os.path.lexists(completion_path) and not completion_marker_present:
        raise CoverageAuditError("News completion marker must be a regular file")
    if completion_marker_present:
        try:
            verified = verify_news_run(root, database_path=database_path)
        except NewsRunVerificationError as exc:
            raise CoverageAuditError(
                "Current-contract news completion/replay verification failed"
            ) from exc
        if verified.run_id != run_id or verified.status != manifest.get("status"):
            raise CoverageAuditError("Verified news completion identity differs")
        verification_basis = (
            "CURRENT_COMPLETION_REPLAY_AND_DATABASE_SEAL"
            if database_path is not None
            else "CURRENT_COMPLETION_REPLAY"
        )
    else:
        if not allow_legacy_unsealed:
            raise CoverageAuditError(
                "Unsealed legacy news evidence requires explicit compatibility opt-in"
            )
        verification_basis = "LEGACY_MANIFEST_CHECKSUM_INVENTORY"

    inventory = _manifest_inventory(manifest)
    # Verify the complete declared evidence pack, not only the table consumed
    # for metrics. This binds raw captures, row evidence, quality reports, and
    # provider configuration before any readiness claim is calculated.
    for relative in inventory:
        _validate_inventoried_file(root, relative, inventory)
    csv_path = _validate_inventoried_file(root, "normalized_articles.csv", inventory)
    entity_links = _load_entity_link_evidence(root, inventory)
    licensing_usage_note: str | None = None
    provider_options: Mapping[str, Any] = {}
    if "provider_config.json" in inventory:
        provider_config_path = _validate_inventoried_file(
            root, "provider_config.json", inventory
        )
        provider_config = _read_json(
            provider_config_path, root=root, label="news provider configuration"
        )
        options = provider_config.get("provider_options")
        if isinstance(options, Mapping):
            provider_options = options
            licensing_usage_note = (
                str(options.get("licensing_usage_note") or "").strip() or None
            )
    licensing_review_required = (
        licensing_usage_note is None
        or licensing_usage_note.casefold() == LICENSING_REVIEW_REQUIRED.casefold()
    )
    if real_mode and provider_mode == "imported" and licensing_review_required:
        raise CoverageAuditError(
            "A real historical import requires a retained licensing/usage note"
        )
    permission_decision = _source_permission_decision(
        policy=source_permission_policy,
        run_id=run_id,
        provider=str(manifest.get("provider") or ""),
        provider_mode=provider_mode,
        provider_options=provider_options,
        licensing_usage_note=licensing_usage_note,
    )
    occurrences: list[NewsOccurrence] = []
    seen_article_ids: set[str] = set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "article_id",
                "provider",
                "source_name",
                "published_at",
                "available_at",
                "fetched_at",
                "mentioned_tickers",
                "mentioned_companies",
                "mentioned_sectors",
                "content_sha256",
                "duplicate_of",
                "ingestion_run_id",
                "raw_artifact_reference",
                "data_class",
            }
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise CoverageAuditError(
                    "Normalized news CSV is missing required coverage fields"
                )
            for index, row in enumerate(reader, start=2):
                article_id = str(row.get("article_id") or "").strip()
                row_run_id = str(row.get("ingestion_run_id") or "").strip()
                data_class = str(row.get("data_class") or "").casefold()
                if not article_id or row_run_id != run_id:
                    raise CoverageAuditError(
                        f"Normalized news identity mismatch at row {index}"
                    )
                if article_id in seen_article_ids:
                    raise CoverageAuditError("Normalized news article identity is duplicated")
                seen_article_ids.add(article_id)
                if data_class not in {"real", "imported", "fixture"}:
                    raise CoverageAuditError(f"Unsupported news data class at row {index}")
                if real_mode and data_class == "fixture":
                    raise CoverageAuditError(
                        "Fixture/mock/synthetic/test news is forbidden in real mode"
                    )
                if data_class != provider_mode:
                    raise CoverageAuditError(
                        f"News provider mode and row data class differ at row {index}"
                    )
                row_provider = str(row.get("provider") or "").strip()
                manifest_provider = str(manifest.get("provider") or "").strip()
                if not row_provider or row_provider != manifest_provider:
                    raise CoverageAuditError(
                        f"News provider identity differs at row {index}"
                    )
                raw_reference = _safe_relative(
                    row.get("raw_artifact_reference"),
                    field_name="raw news evidence reference",
                )
                if raw_reference not in inventory:
                    raise CoverageAuditError(
                        f"Raw news evidence is not checksum-inventoried at row {index}"
                    )
                duplicate_of = str(row.get("duplicate_of") or "").strip() or None
                content_digest = str(row.get("content_sha256") or "").strip()
                if not _SHA256.fullmatch(content_digest):
                    raise CoverageAuditError(
                        f"Invalid news content checksum at row {index}"
                    )
                published_at = parse_timestamp(
                    row.get("published_at"),
                    field_name=f"published_at row {index}",
                )
                available_at = parse_timestamp(
                    row.get("available_at"),
                    field_name=f"available_at row {index}",
                )
                fetched_at = parse_timestamp(
                    row.get("fetched_at"),
                    field_name=f"fetched_at row {index}",
                )
                if available_at is None or fetched_at is None:
                    raise CoverageAuditError(
                        f"News availability/fetch time is missing at row {index}"
                    )
                if available_at < fetched_at or (
                    published_at is not None and available_at < published_at
                ):
                    raise CoverageAuditError(
                        f"News availability precedes its evidence at row {index}"
                    )
                tickers = _json_list(
                    row.get("mentioned_tickers"),
                    field_name=f"mentioned_tickers row {index}",
                )
                companies = _json_list(
                    row.get("mentioned_companies"),
                    field_name=f"mentioned_companies row {index}",
                )
                row_links = entity_links.get(article_id, ())
                linked_tickers = {
                    str(link["ticker"]).upper() for link in row_links
                }
                linked_companies = {
                    str(link["company_name"]) for link in row_links
                }
                if {ticker.upper() for ticker in tickers} != linked_tickers:
                    raise CoverageAuditError(
                        f"News ticker links disagree with checksum-bound evidence at row {index}"
                    )
                if set(companies) != linked_companies:
                    raise CoverageAuditError(
                        f"News company links disagree with checksum-bound evidence at row {index}"
                    )
                occurrences.append(
                    NewsOccurrence(
                        occurrence_key=f"{run_id}:{article_id}",
                        ingestion_run_id=run_id,
                        article_id=article_id,
                        canonical_article_id=duplicate_of or article_id,
                        provider=row_provider,
                        provider_mode=provider_mode,
                        source_name=str(row.get("source_name") or "unknown").strip()
                        or "unknown",
                        source_url=_public_url(
                            str(
                                row.get("canonical_url")
                                or row.get("source_url")
                                or ""
                            )
                        ),
                        title=str(row.get("title") or "").strip() or None,
                        published_at=published_at,
                        available_at=available_at,
                        fetched_at=fetched_at,
                        published_timezone=(
                            str(row.get("published_timezone") or "").strip() or None
                        ),
                        language=str(row.get("language") or "").strip().casefold()
                        or None,
                        tickers=tickers,
                        companies=companies,
                        sectors=_json_list(
                            row.get("mentioned_sectors"),
                            field_name=f"mentioned_sectors row {index}",
                        ),
                        content_sha256=content_digest,
                        raw_artifact_reference=raw_reference,
                        data_class=data_class,
                        duplicate_of=duplicate_of,
                        entity_links=row_links,
                    )
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CoverageAuditError("Normalized news CSV could not be read") from exc
    unknown_link_articles = sorted(set(entity_links) - seen_article_ids)
    if unknown_link_articles:
        raise CoverageAuditError("News entity-link evidence references an unknown article")

    manifest_digest = sha256_file(manifest_path)
    normalized_digest = sha256_file(csv_path)
    record_counts = manifest.get("record_counts")
    provider_error_count = 0
    rejected_record_count = 0
    if record_counts is not None and not isinstance(record_counts, Mapping):
        raise CoverageAuditError("News manifest record counts are invalid")
    if isinstance(record_counts, Mapping):
        for field_name, output_name in (
            ("provider_errors", "provider errors"),
            ("rejected", "rejected records"),
        ):
            value = record_counts.get(field_name, 0)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CoverageAuditError(
                    f"News manifest {output_name} count is invalid"
                )
            if field_name == "provider_errors":
                provider_error_count = value
            else:
                rejected_record_count = value
    if "provider_errors.json" in inventory:
        provider_errors = _read_json(
            _validate_inventoried_file(root, "provider_errors.json", inventory),
            root=root,
            label="news provider errors",
        )
        errors = provider_errors.get("errors")
        if provider_errors.get("run_id") != run_id or not isinstance(errors, list):
            raise CoverageAuditError("News provider-error evidence is inconsistent")
        if isinstance(record_counts, Mapping) and len(errors) != provider_error_count:
            raise CoverageAuditError("News provider-error count is inconsistent")
        provider_error_count = len(errors)
    reference = {
        "run_id": run_id,
        "status": manifest.get("status"),
        "provider": manifest.get("provider"),
        "provider_mode": provider_mode,
        "quality_status": manifest.get("quality_status"),
        "occurrence_count": len(occurrences),
        "run_manifest_sha256": manifest_digest,
        "normalized_articles_sha256": normalized_digest,
        "artifact_inventory_sha256": _sha256_bytes(
            _canonical_json_bytes(
                [
                    {"path": path, "sha256": digest, "size_bytes": size}
                    for path, (digest, size) in inventory.items()
                ]
            )
        ),
        "completion_marker_present": completion_marker_present,
        "verification_basis": verification_basis,
        "provider_error_count": provider_error_count,
        "rejected_record_count": rejected_record_count,
        "ingestion_complete": (
            manifest.get("status") == "PASS" and provider_error_count == 0
        ),
        "artifact_reference": f"news_ingestion/{run_id}",
        "licensing_usage_note": (
            licensing_usage_note
            or (
                "LEGACY_EVIDENCE_LICENSING_REVIEW_REQUIRED: see the "
                "checksum-bound historical-news source assessment."
            )
        ),
        "source_permission_decision": permission_decision,
    }
    return occurrences, reference


@contextmanager
def _immutable_database_snapshot(
    path: str | Path | None,
) -> Iterator[tuple[Path | None, Mapping[str, Any]]]:
    """Yield one immutable SQLite image for both seal checks and fingerprinting.

    Hashing a live main database and WAL after separately verifying run seals can
    bind the report to a different logical database state.  SQLite's online
    backup API folds the source connection's consistent view (including visible
    WAL pages) into one private database image.  Every selected run is verified
    against that exact image and the report fingerprints those same bytes.
    """

    if path is None:
        yield None, {"provided": False, "files": []}
        return

    candidate = Path(path)
    if candidate.is_symlink():
        raise CoverageAuditError("News database must not be a symlink")
    database = candidate.resolve()
    _regular_file(database, label="news database")

    try:
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="paisa-news-coverage-db-"
        )
    except OSError as exc:
        raise CoverageAuditError(
            "Immutable news database snapshot could not be prepared"
        ) from exc
    with temporary_directory as temporary:
        try:
            snapshot = Path(temporary) / "news-database-snapshot.sqlite3"
            source: sqlite3.Connection | None = None
            destination: sqlite3.Connection | None = None
            try:
                source = sqlite3.connect(
                    f"file:{quote(database.as_posix(), safe='/:')}?mode=ro",
                    uri=True,
                    timeout=5,
                )
                source.execute("PRAGMA query_only = ON")
                destination = sqlite3.connect(snapshot, timeout=5)
                source.backup(destination)
                destination.commit()
            except sqlite3.Error as exc:
                raise CoverageAuditError(
                    "News database could not be snapshotted safely"
                ) from exc
            finally:
                if destination is not None:
                    destination.close()
                if source is not None:
                    source.close()

            _regular_file(snapshot, label="immutable news database snapshot")
            snapshot.chmod(0o400)
            metadata = snapshot.stat()
            fingerprint = {
                "provided": True,
                "engine": "sqlite",
                "snapshot_method": "sqlite_online_backup_read_only_source",
                "wal_state_folded_into_snapshot": True,
                "files": [
                    {
                        "label": "database",
                        "role": "immutable_verification_snapshot",
                        "sha256": sha256_file(snapshot),
                        "size_bytes": metadata.st_size,
                    }
                ],
                "note": (
                    "The fingerprint identifies the immutable SQLite backup used "
                    "for every completion-seal verification in this audit; normalized "
                    "run packs remain the immutable row evidence."
                ),
            }
        except OSError as exc:
            raise CoverageAuditError(
                "Immutable news database snapshot could not be prepared"
            ) from exc
        yield snapshot, fingerprint


def load_news_evidence(
    run_directories: Iterable[str | Path],
    *,
    real_mode: bool = True,
    database_path: str | Path | None = None,
    allow_legacy_unsealed: bool = False,
    source_permission_policy: SourcePermissionPolicy | None = None,
) -> NewsEvidence:
    resolved_roots: set[Path] = set()
    for path in run_directories:
        candidate = Path(path)
        if candidate.is_symlink():
            raise CoverageAuditError("News run directory must not be a symlink")
        resolved_roots.add(candidate.resolve())
    roots = sorted(resolved_roots, key=str)
    if not roots:
        raise CoverageAuditError("At least one completed news run is required")
    occurrences: list[NewsOccurrence] = []
    references: list[Mapping[str, Any]] = []
    seen_runs: set[str] = set()
    seen_occurrences: set[str] = set()
    with _immutable_database_snapshot(database_path) as (
        verification_database,
        database_fingerprints,
    ):
        for root in roots:
            loaded, reference = _load_one_news_run(
                root,
                real_mode=real_mode,
                allow_legacy_unsealed=allow_legacy_unsealed,
                database_path=verification_database,
                source_permission_policy=source_permission_policy,
            )
            run_id = str(reference["run_id"])
            if run_id in seen_runs:
                raise CoverageAuditError("A news run was supplied more than once")
            seen_runs.add(run_id)
            for occurrence in loaded:
                if occurrence.occurrence_key in seen_occurrences:
                    raise CoverageAuditError("Duplicate news occurrence identity")
                seen_occurrences.add(occurrence.occurrence_key)
                occurrences.append(occurrence)
            references.append(reference)
    permission_notes = tuple(
        sorted(
            f"{reference['run_id']}: "
            f"{reference['source_permission_decision']['status']}: "
            f"{reference['source_permission_decision']['reason']}"
            for reference in references
            if not reference["source_permission_decision"]["passed"]
        )
    )
    return NewsEvidence(
        occurrences=tuple(sorted(occurrences, key=lambda item: item.occurrence_key)),
        run_references=tuple(sorted(references, key=lambda item: str(item["run_id"]))),
        database_fingerprints=database_fingerprints,
        permission_review_required=bool(permission_notes),
        permission_notes=permission_notes,
        permission_policy_reference=(
            source_permission_policy.reference
            if source_permission_policy is not None
            else {}
        ),
    )


def _vertical_inventory(manifest: Mapping[str, Any]) -> dict[str, str]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise CoverageAuditError("Vertical-slice manifest has no artifact inventory")
    result: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise CoverageAuditError("Vertical-slice inventory entry is invalid")
        relative = _safe_relative(
            item.get("path"), field_name="vertical-slice artifact path"
        )
        digest = item.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CoverageAuditError("Vertical-slice inventory checksum is invalid")
        if relative in result:
            raise CoverageAuditError("Vertical-slice inventory path is duplicated")
        result[relative] = digest
    return result


def _training_inventory(manifest: Mapping[str, Any]) -> dict[str, str]:
    inventory = manifest.get("artifact_inventory")
    if not isinstance(inventory, Mapping) or inventory.get("algorithm") != "sha256":
        raise CoverageAuditError("Training manifest has no SHA-256 artifact inventory")
    raw = inventory.get("files")
    if not isinstance(raw, list):
        raise CoverageAuditError("Training artifact inventory files are missing")
    result: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise CoverageAuditError("Training inventory entry is invalid")
        relative = _safe_relative(
            item.get("path"), field_name="training artifact path"
        )
        digest = item.get("sha256")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CoverageAuditError("Training inventory checksum is invalid")
        if relative in result:
            raise CoverageAuditError("Training inventory path is duplicated")
        result[relative] = digest
    return result


def _inventoried_path(
    root: Path,
    relative: str,
    inventory: Mapping[str, str],
    *,
    label: str,
) -> Path:
    if relative not in inventory:
        raise CoverageAuditError(f"{label} is not checksum-inventoried")
    path = _regular_file(root / relative, root=root, label=label)
    if sha256_file(path) != inventory[relative]:
        raise CoverageAuditError(f"{label} checksum mismatch")
    return path


def load_stock_evidence(
    vertical_slice_directory: str | Path,
    model_training_directory: str | Path,
    *,
    symbol: str = "OGDC",
) -> StockEvidence:
    vertical_candidate = Path(vertical_slice_directory)
    training_candidate = Path(model_training_directory)
    if vertical_candidate.is_symlink():
        raise CoverageAuditError("Vertical-slice directory is missing or unsafe")
    if training_candidate.is_symlink():
        raise CoverageAuditError("Model-training directory is missing or unsafe")
    vertical_root = vertical_candidate.resolve()
    training_root = training_candidate.resolve()
    if not vertical_root.is_dir():
        raise CoverageAuditError("Vertical-slice directory is missing or unsafe")
    if not training_root.is_dir():
        raise CoverageAuditError("Model-training directory is missing or unsafe")

    vertical_manifest_path = _regular_file(
        vertical_root / "manifest.json",
        root=vertical_root,
        label="vertical-slice manifest",
    )
    vertical = _read_json(
        vertical_manifest_path, root=vertical_root, label="vertical-slice manifest"
    )
    if vertical.get("status") != "PASS":
        raise CoverageAuditError("Vertical-slice input must be a completed PASS")
    if str(vertical.get("symbol") or "").upper() != symbol.upper():
        raise CoverageAuditError("Vertical-slice symbol does not match the audit symbol")
    if vertical.get("data", {}).get("is_real_data") is not True:
        raise CoverageAuditError("Coverage audit requires real stock evidence")
    vertical_run_id = str(vertical.get("vertical_slice_run_id") or "")
    if vertical_run_id != vertical_root.name:
        raise CoverageAuditError("Vertical-slice directory and manifest identity differ")
    training_run_id = str(vertical.get("model_training_run_id") or "")
    if training_run_id != training_root.name:
        raise CoverageAuditError("Training run does not match the vertical slice")

    vertical_inventory = _vertical_inventory(vertical)
    price_candidates = sorted(
        path
        for path in vertical_inventory
        if path.startswith("input_snapshot/") and path.endswith(".csv")
    )
    if len(price_candidates) != 1:
        raise CoverageAuditError("Vertical slice must inventory one input price CSV")
    price_relative = price_candidates[0]
    price_path = _inventoried_path(
        vertical_root,
        price_relative,
        vertical_inventory,
        label="vertical-slice input price artifact",
    )
    expected_price_digest = vertical.get("data", {}).get("input_prices_sha256")
    if expected_price_digest != vertical_inventory[price_relative]:
        raise CoverageAuditError("Vertical price fingerprint disagrees with its manifest")

    trading_dates: list[date] = []
    try:
        with price_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"date", "ticker", "source"}.issubset(
                reader.fieldnames
            ):
                raise CoverageAuditError("Price artifact is missing required columns")
            for index, row in enumerate(reader, start=2):
                if str(row.get("ticker") or "").upper() != symbol.upper():
                    continue
                source = str(row.get("source") or "").casefold()
                if source in {"", "sample", "fixture", "synthetic", "mock", "test"}:
                    raise CoverageAuditError(
                        f"Non-real stock row is forbidden at price row {index}"
                    )
                trading_dates.append(
                    _parse_date(row.get("date"), field_name=f"price date row {index}")
                )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CoverageAuditError("Price artifact could not be read") from exc
    if not trading_dates or trading_dates != sorted(set(trading_dates)):
        raise CoverageAuditError("Stock trading dates must be nonempty, unique, and sorted")

    claimed_range = vertical.get("data", {}).get("actual_ogdc_date_range")
    if not isinstance(claimed_range, Mapping) or claimed_range != {
        "start": trading_dates[0].isoformat(),
        "end": trading_dates[-1].isoformat(),
    }:
        raise CoverageAuditError("Stock date range disagrees with the vertical manifest")

    training_manifest_path = _regular_file(
        training_root / "run_manifest.json",
        root=training_root,
        label="training run manifest",
    )
    training = _read_json(
        training_manifest_path, root=training_root, label="training run manifest"
    )
    if training.get("run_id") != training_run_id:
        raise CoverageAuditError("Training manifest identity mismatch")
    if str(training.get("symbol") or "").upper() != symbol.upper():
        raise CoverageAuditError("Training symbol does not match the audit symbol")
    training_inventory = _training_inventory(training)
    data_manifest_path = _inventoried_path(
        training_root,
        "data_manifest.json",
        training_inventory,
        label="training data manifest",
    )
    data_manifest = _read_json(
        data_manifest_path, root=training_root, label="training data manifest"
    )
    if str(data_manifest.get("symbol") or "").upper() != symbol.upper():
        raise CoverageAuditError("Training data symbol does not match the audit symbol")
    acquisition = data_manifest.get("acquisition")
    if not isinstance(acquisition, Mapping):
        raise CoverageAuditError("Training data acquisition provenance is missing")
    if acquisition.get("input_snapshot_sha256") != expected_price_digest:
        raise CoverageAuditError(
            "Training data and vertical slice do not share the frozen price evidence"
        )
    if data_manifest.get("price_rows") != len(trading_dates):
        raise CoverageAuditError("Training data price-row count is inconsistent")
    split_path = _inventoried_path(
        training_root,
        "split_manifest.json",
        training_inventory,
        label="training split manifest",
    )
    split_manifest = _read_json(
        split_path, root=training_root, label="training split manifest"
    )
    if split_manifest.get("method") != "chronological_target_date_70_15_15_no_shuffle":
        raise CoverageAuditError("Training split is not the expected chronological split")
    raw_target_dates = split_manifest.get("target_dates")
    row_counts = split_manifest.get("row_counts")
    if not isinstance(raw_target_dates, Mapping) or not isinstance(row_counts, Mapping):
        raise CoverageAuditError("Training split manifest is incomplete")
    split_dates: dict[str, tuple[date, ...]] = {}
    all_targets: list[date] = []
    observed = set(trading_dates)
    previous_end: date | None = None
    for split in SPLITS:
        raw = raw_target_dates.get(split)
        if not isinstance(raw, list):
            raise CoverageAuditError(f"Training split {split} dates are missing")
        dates = tuple(
            _parse_date(value, field_name=f"{split} target date") for value in raw
        )
        if not dates or list(dates) != sorted(set(dates)):
            raise CoverageAuditError(f"Training split {split} must be sorted and unique")
        if row_counts.get(split) != len(dates):
            raise CoverageAuditError(f"Training split {split} row count is inconsistent")
        if not set(dates).issubset(observed):
            raise CoverageAuditError(f"Training split {split} contains a non-price date")
        if previous_end is not None and dates[0] <= previous_end:
            raise CoverageAuditError("Training splits overlap or are not chronological")
        previous_end = dates[-1]
        split_dates[split] = dates
        all_targets.extend(dates)
    if len(all_targets) != len(set(all_targets)):
        raise CoverageAuditError("Training split target dates overlap")

    reference = {
        "symbol": symbol.upper(),
        "vertical_slice_run_id": vertical_run_id,
        "vertical_manifest_sha256": sha256_file(vertical_manifest_path),
        "price_artifact_reference": f"vertical_slice/{vertical_run_id}/{price_relative}",
        "price_artifact_sha256": vertical_inventory[price_relative],
        "price_date_range": {
            "start": trading_dates[0].isoformat(),
            "end": trading_dates[-1].isoformat(),
        },
        "price_trading_day_count": len(trading_dates),
        "model_training_run_id": training_run_id,
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "training_data_manifest_sha256": training_inventory["data_manifest.json"],
        "training_input_snapshot_sha256": acquisition[
            "input_snapshot_sha256"
        ],
        "split_manifest_sha256": training_inventory["split_manifest.json"],
        "split_method": split_manifest.get("method"),
        "split_counts": {split: len(split_dates[split]) for split in SPLITS},
        "split_ranges": {
            split: {
                "start": split_dates[split][0].isoformat(),
                "end": split_dates[split][-1].isoformat(),
            }
            for split in SPLITS
        },
        "acquisition_mode": vertical.get("data", {}).get("acquisition_mode"),
        "provider": vertical.get("data", {}).get("source"),
        "is_real_data": True,
    }
    return StockEvidence(
        symbol=symbol.upper(),
        trading_dates=tuple(trading_dates),
        split_dates=split_dates,
        reference=reference,
    )


def _representative_canonicals(
    occurrences: Sequence[NewsOccurrence],
) -> tuple[dict[str, NewsOccurrence], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[NewsOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence.canonical_article_id].append(occurrence)
    representatives: dict[str, NewsOccurrence] = {}
    attributes: dict[str, dict[str, Any]] = {}
    far_future = datetime.max.replace(tzinfo=timezone.utc)
    for canonical_id, values in grouped.items():
        ordered = sorted(
            values,
            key=lambda item: (
                item.available_at or far_future,
                item.fetched_at or far_future,
                item.occurrence_key,
            ),
        )
        representatives[canonical_id] = ordered[0]
        ogdc_values = [item for item in ordered if item.is_ogdc]
        energy_values = [item for item in ordered if item.is_energy]
        ogdc_link_evidence = sorted(
            (
                {
                    "ingestion_run_id": item.ingestion_run_id,
                    **dict(link),
                }
                for item in ogdc_values
                for link in item.entity_links
                if str(link.get("ticker") or "").upper() == "OGDC"
            ),
            key=lambda item: (
                item["ingestion_run_id"],
                item["article_id"],
                item["link_id"],
            ),
        )
        attributes[canonical_id] = {
            "is_ogdc": bool(ogdc_values),
            "is_energy": bool(energy_values),
            "occurrence_count": len(values),
            "sources": sorted({item.source_name for item in values}),
            "ogdc_occurrence_count": len(ogdc_values),
            "ogdc_sources": sorted({item.source_name for item in ogdc_values}),
            "ogdc_link_evidence": ogdc_link_evidence,
            "population_representatives": {
                "all_financial_news": ordered[0],
                "ogdc_linked_news": ogdc_values[0] if ogdc_values else None,
                "energy_sector_news": energy_values[0] if energy_values else None,
            },
        }
    return representatives, attributes


def _session_cutoffs(
    dates: Sequence[date], config: CoverageConfig
) -> tuple[list[datetime], ZoneInfo]:
    zone = ZoneInfo(config.market_timezone)
    values = [
        datetime.combine(session_date, config.decision_cutoff, tzinfo=zone).astimezone(
            timezone.utc
        )
        for session_date in dates
    ]
    return values, zone


def assign_to_session(
    available_at: datetime | None,
    stock: StockEvidence,
    config: CoverageConfig,
) -> SessionAssignment:
    if available_at is None:
        return SessionAssignment(None, "missing_available_at", "UNMAPPED_MISSING_TIME")
    instant = _aware_utc(available_at, field_name="available_at")
    cutoffs, zone = _session_cutoffs(stock.trading_dates, config)
    local = instant.astimezone(zone)
    local_date = local.date()
    date_to_cutoff = dict(zip(stock.trading_dates, cutoffs, strict=True))
    if local_date in date_to_cutoff:
        arrival_class = (
            "before_or_at_decision_cutoff"
            if instant <= date_to_cutoff[local_date]
            else "after_decision_cutoff"
        )
    else:
        arrival_class = "non_trading_day"

    first_local_midnight = datetime.combine(
        stock.trading_dates[0], time.min, tzinfo=zone
    ).astimezone(timezone.utc)
    if instant < first_local_midnight:
        return SessionAssignment(None, arrival_class, "BEFORE_PRICE_PERIOD")
    index = bisect_left(cutoffs, instant)
    if index >= len(cutoffs):
        return SessionAssignment(None, arrival_class, "AFTER_PRICE_PERIOD")
    assigned = stock.trading_dates[index]
    if assigned == local_date:
        status = "SAME_SESSION_BEFORE_CUTOFF"
    elif local_date in date_to_cutoff:
        status = "AFTER_CUTOFF_CARRIED_TO_NEXT_SESSION"
    else:
        status = "NON_TRADING_DAY_CARRIED_TO_NEXT_SESSION"
    return SessionAssignment(assigned, arrival_class, status)


def _date_range(values: Iterable[datetime | None]) -> dict[str, str | None]:
    present = sorted(value for value in values if value is not None)
    return {
        "start": utc_iso(present[0]) if present else None,
        "end": utc_iso(present[-1]) if present else None,
    }


def _calendar_overlap(
    values: Iterable[datetime | None],
    stock: StockEvidence,
    zone: ZoneInfo,
) -> dict[str, str | None]:
    dates = sorted(
        value.astimezone(zone).date() for value in values if value is not None
    )
    if not dates:
        return {"start": None, "end": None}
    overlapping = [
        value
        for value in dates
        if stock.trading_dates[0] <= value <= stock.trading_dates[-1]
    ]
    if not overlapping:
        return {"start": None, "end": None}
    return {
        "start": overlapping[0].isoformat(),
        "end": overlapping[-1].isoformat(),
    }


def _occurrence_population_match(
    population: str, occurrence: NewsOccurrence
) -> bool:
    if population == "all_financial_news":
        return True
    if population == "ogdc_linked_news":
        return occurrence.is_ogdc
    if population == "energy_sector_news":
        return occurrence.is_energy
    raise CoverageAuditError(f"Unknown coverage population: {population}")


def _longest_gap(dates: Sequence[date], covered: set[date]) -> int:
    longest = current = 0
    for value in dates:
        if value in covered:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _missing_intervals(
    dates: Sequence[date],
    covered: set[date],
    *,
    population: str,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    start: date | None = None
    previous: date | None = None
    count = 0
    for current in dates:
        if current in covered:
            if start is not None and previous is not None:
                rows.append(
                    {
                        "population": population,
                        "split": split,
                        "start_date": start.isoformat(),
                        "end_date": previous.isoformat(),
                        "missing_trading_sessions": count,
                    }
                )
            start = previous = None
            count = 0
            continue
        if start is None:
            start = current
        previous = current
        count += 1
    if start is not None and previous is not None:
        rows.append(
            {
                "population": population,
                "split": split,
                "start_date": start.isoformat(),
                "end_date": previous.isoformat(),
                "missing_trading_sessions": count,
            }
        )
    return rows


def calculate_readiness(
    split_rows: Sequence[Mapping[str, Any]],
    *,
    thresholds: ReadinessThresholds,
    publication_completeness_percent: float,
    available_at_completeness_percent: float,
    distinct_sources: int,
    publication_overlap_exists: bool = False,
) -> tuple[str, dict[str, Any]]:
    lookup = {
        (str(row["population"]), str(row["split"])): row for row in split_rows
    }
    full = lookup[("all_financial_news", "full_price_period")]
    if int(full["trading_days_with_articles"]) == 0:
        status = "INSUFFICIENT_OVERLAP" if publication_overlap_exists else "NO_OVERLAP"
        return status, {
            "passed": False,
            "reason": (
                "Publication dates intersect the price period, but no canonical "
                "article is cutoff-eligible under available_at."
                if publication_overlap_exists
                else "Neither publication dates nor cutoff-eligible availability "
                "overlap the price period."
            ),
            "checks": {},
        }

    all_split_values = [
        float(lookup[("all_financial_news", split)]["coverage_percent"])
        for split in SPLITS
    ]
    ogdc_split_values = [
        float(lookup[("ogdc_linked_news", split)]["coverage_percent"])
        for split in SPLITS
    ]
    maximum_gap = max(
        int(lookup[("all_financial_news", split)]["longest_missing_gap_sessions"])
        for split in SPLITS
    )
    checks = {
        "all_news_each_split": {
            "actual_minimum_percent": min(all_split_values),
            "required_minimum_percent": (
                thresholds.ready_minimum_all_news_coverage_percent_each_split
            ),
            "passed": min(all_split_values)
            >= thresholds.ready_minimum_all_news_coverage_percent_each_split,
        },
        "ogdc_news_each_split": {
            "actual_minimum_percent": min(ogdc_split_values),
            "required_minimum_percent": (
                thresholds.ready_minimum_ogdc_coverage_percent_each_split
            ),
            "passed": min(ogdc_split_values)
            >= thresholds.ready_minimum_ogdc_coverage_percent_each_split,
        },
        "maximum_missing_gap": {
            "actual_sessions": maximum_gap,
            "maximum_sessions": thresholds.ready_maximum_all_news_missing_gap_sessions,
            "passed": maximum_gap
            <= thresholds.ready_maximum_all_news_missing_gap_sessions,
        },
        "distinct_sources": {
            "actual": distinct_sources,
            "minimum": thresholds.ready_minimum_distinct_sources,
            "passed": distinct_sources >= thresholds.ready_minimum_distinct_sources,
        },
        "publication_time_completeness": {
            "actual_percent": publication_completeness_percent,
            "minimum_percent": (
                thresholds.ready_minimum_publication_time_completeness_percent
            ),
            "passed": publication_completeness_percent
            >= thresholds.ready_minimum_publication_time_completeness_percent,
        },
        "available_at_completeness": {
            "actual_percent": available_at_completeness_percent,
            "minimum_percent": (
                thresholds.ready_minimum_available_at_completeness_percent
            ),
            "passed": available_at_completeness_percent
            >= thresholds.ready_minimum_available_at_completeness_percent,
        },
    }
    if all(bool(value["passed"]) for value in checks.values()):
        return "READY_FOR_AGGREGATION", {"passed": True, "checks": checks}

    partial = (
        min(all_split_values)
        >= thresholds.partial_minimum_all_news_coverage_percent_each_split
        and min(ogdc_split_values)
        >= thresholds.partial_minimum_ogdc_coverage_percent_each_split
    )
    return (
        "PARTIALLY_COVERED" if partial else "INSUFFICIENT_OVERLAP",
        {
            "passed": False,
            "checks": checks,
            "partial_gate": {
                "all_news_actual_minimum_percent": min(all_split_values),
                "all_news_required_minimum_percent": (
                    thresholds.partial_minimum_all_news_coverage_percent_each_split
                ),
                "ogdc_actual_minimum_percent": min(ogdc_split_values),
                "ogdc_required_minimum_percent": (
                    thresholds.partial_minimum_ogdc_coverage_percent_each_split
                ),
                "passed": partial,
            },
        },
    )


def compute_coverage(
    news: NewsEvidence,
    stock: StockEvidence,
    config: CoverageConfig,
) -> CoverageComputation:
    occurrences = list(news.occurrences)
    energy_evidence_present = any(item.sectors for item in occurrences)
    population_supported = {
        "all_financial_news": True,
        "ogdc_linked_news": True,
        "energy_sector_news": energy_evidence_present,
    }
    representatives, attributes = _representative_canonicals(occurrences)
    canonical_assignments: dict[str, dict[str, SessionAssignment]] = {
        population: {} for population in POPULATIONS
    }
    for canonical_id, values in attributes.items():
        for population in POPULATIONS:
            representative = values["population_representatives"][population]
            if representative is not None:
                canonical_assignments[population][canonical_id] = assign_to_session(
                    representative.available_at, stock, config
                )
    occurrence_assignments = {
        item.occurrence_key: assign_to_session(item.available_at, stock, config)
        for item in occurrences
    }
    article_session_rows: list[dict[str, Any]] = []
    for item in sorted(occurrences, key=lambda value: value.occurrence_key):
        occurrence_assignment = occurrence_assignments[item.occurrence_key]
        canonical_attributes = attributes[item.canonical_article_id]
        for population in POPULATIONS:
            canonical_representative = canonical_attributes[
                "population_representatives"
            ][population]
            canonical_assignment = canonical_assignments[population].get(
                item.canonical_article_id
            )
            article_session_rows.append(
                {
                    "occurrence_key": item.occurrence_key,
                    "ingestion_run_id": item.ingestion_run_id,
                    "article_id": item.article_id,
                    "canonical_article_id": item.canonical_article_id,
                    "provider": item.provider,
                    "source_name": item.source_name,
                    "data_class": item.data_class,
                    "population": population,
                    "population_supported": population_supported[population],
                    "occurrence_in_population": _occurrence_population_match(
                        population, item
                    ),
                    "is_duplicate_occurrence": item.duplicate_of is not None,
                    "published_at": utc_iso(item.published_at),
                    "available_at": utc_iso(item.available_at),
                    "fetched_at": utc_iso(item.fetched_at),
                    "occurrence_assigned_trading_date": (
                        occurrence_assignment.session_date.isoformat()
                        if occurrence_assignment.session_date is not None
                        else None
                    ),
                    "occurrence_mapping_status": occurrence_assignment.mapping_status,
                    "occurrence_arrival_class": occurrence_assignment.arrival_class,
                    "canonical_representative_article_id": (
                        canonical_representative.article_id
                        if canonical_representative is not None
                        else None
                    ),
                    "canonical_representative_ingestion_run_id": (
                        canonical_representative.ingestion_run_id
                        if canonical_representative is not None
                        else None
                    ),
                    "is_population_canonical_representative": (
                        canonical_representative is not None
                        and canonical_representative.occurrence_key
                        == item.occurrence_key
                    ),
                    "canonical_available_at": (
                        utc_iso(canonical_representative.available_at)
                        if canonical_representative is not None
                        else None
                    ),
                    "canonical_assigned_trading_date": (
                        canonical_assignment.session_date.isoformat()
                        if canonical_assignment is not None
                        and canonical_assignment.session_date is not None
                        else None
                    ),
                    "canonical_mapping_status": (
                        canonical_assignment.mapping_status
                        if canonical_assignment is not None
                        else None
                    ),
                    "canonical_arrival_class": (
                        canonical_assignment.arrival_class
                        if canonical_assignment is not None
                        else None
                    ),
                    "content_sha256": item.content_sha256,
                    "raw_artifact_reference": item.raw_artifact_reference,
                }
            )
    split_for_date: dict[date, str] = {}
    for split, values in stock.split_dates.items():
        for value in values:
            split_for_date[value] = split
    cutoffs, zone = _session_cutoffs(stock.trading_dates, config)

    canonical_ids_by_date: dict[str, dict[date, set[str]]] = {
        population: defaultdict(set) for population in POPULATIONS
    }
    occurrence_keys_by_date: dict[date, set[str]] = defaultdict(set)
    for population, assignments in canonical_assignments.items():
        for canonical_id, assignment in assignments.items():
            if assignment.session_date is not None:
                canonical_ids_by_date[population][assignment.session_date].add(
                    canonical_id
                )
    for item in occurrences:
        assigned = occurrence_assignments[item.occurrence_key].session_date
        if assigned is not None:
            occurrence_keys_by_date[assigned].add(item.occurrence_key)

    def canonical_count(session_date: date, population: str) -> int:
        return len(canonical_ids_by_date[population].get(session_date, set()))

    occurrences_by_key = {item.occurrence_key: item for item in occurrences}

    def occurrence_count(session_date: date, population: str) -> int:
        count = 0
        for key in occurrence_keys_by_date.get(session_date, set()):
            item = occurrences_by_key[key]
            count += int(_occurrence_population_match(population, item))
        return count

    trading_day_rows: list[dict[str, Any]] = []
    for session_date, cutoff in zip(stock.trading_dates, cutoffs, strict=True):
        row: dict[str, Any] = {
            "trading_date": session_date.isoformat(),
            "split": split_for_date.get(session_date, "outside_model_splits"),
            "decision_cutoff_utc": utc_iso(cutoff),
        }
        for population in POPULATIONS:
            prefix = population.removesuffix("_news")
            row[f"{prefix}_occurrence_count"] = occurrence_count(
                session_date, population
            )
            row[f"{prefix}_canonical_article_count"] = canonical_count(
                session_date, population
            )
        trading_day_rows.append(row)

    split_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    date_groups: dict[str, tuple[date, ...]] = {
        "full_price_period": stock.trading_dates,
        **{split: stock.split_dates[split] for split in SPLITS},
    }
    for population in POPULATIONS:
        for split, dates in date_groups.items():
            covered = {
                session_date
                for session_date in dates
                if canonical_count(session_date, population) > 0
            }
            occurrence_total = sum(
                occurrence_count(session_date, population) for session_date in dates
            )
            canonical_total = sum(
                canonical_count(session_date, population) for session_date in dates
            )
            trading_day_count = len(dates)
            days_with = len(covered)
            split_rows.append(
                {
                    "population": population,
                    "split": split,
                    "trading_days": trading_day_count,
                    "trading_days_with_articles": days_with,
                    "coverage_percent": round(
                        100.0 * days_with / trading_day_count, 6
                    ),
                    "article_occurrences": occurrence_total,
                    "canonical_articles": canonical_total,
                    "articles_per_trading_day": round(
                        canonical_total / trading_day_count, 8
                    ),
                    "zero_article_trading_days": trading_day_count - days_with,
                    "longest_missing_gap_sessions": _longest_gap(dates, covered),
                }
            )
            missing_rows.extend(
                _missing_intervals(
                    dates, covered, population=population, split=split
                )
            )

    month_dates: dict[str, list[date]] = defaultdict(list)
    for session_date in stock.trading_dates:
        month_dates[session_date.strftime("%Y-%m")].append(session_date)
    monthly_rows: list[dict[str, Any]] = []
    for month, dates in sorted(month_dates.items()):
        for population in POPULATIONS:
            covered = {
                session_date
                for session_date in dates
                if canonical_count(session_date, population) > 0
            }
            canonical_total = sum(
                canonical_count(session_date, population) for session_date in dates
            )
            monthly_rows.append(
                {
                    "month": month,
                    "population": population,
                    "trading_days": len(dates),
                    "trading_days_with_articles": len(covered),
                    "coverage_percent": round(100.0 * len(covered) / len(dates), 6),
                    "article_occurrences": sum(
                        occurrence_count(session_date, population)
                        for session_date in dates
                    ),
                    "canonical_articles": canonical_total,
                    "articles_per_trading_day": round(canonical_total / len(dates), 8),
                    "zero_article_trading_days": len(dates) - len(covered),
                }
            )

    ogdc_rows: list[dict[str, Any]] = []
    for canonical_id in sorted(representatives):
        if not attributes[canonical_id]["is_ogdc"]:
            continue
        representative = attributes[canonical_id]["population_representatives"][
            "ogdc_linked_news"
        ]
        if representative is None:  # defensive invariant
            raise CoverageAuditError("OGDC canonical has no linked evidence")
        assignment = canonical_assignments["ogdc_linked_news"][canonical_id]
        ogdc_rows.append(
            {
                "canonical_article_id": canonical_id,
                "representative_article_id": representative.article_id,
                "source_name": representative.source_name,
                "source_url": representative.source_url,
                "title": representative.title,
                "published_at": utc_iso(representative.published_at),
                "available_at": utc_iso(representative.available_at),
                "assigned_trading_date": (
                    assignment.session_date.isoformat()
                    if assignment.session_date is not None
                    else None
                ),
                "mapping_status": assignment.mapping_status,
                "arrival_class": assignment.arrival_class,
                "occurrence_count": attributes[canonical_id][
                    "ogdc_occurrence_count"
                ],
                "sources": attributes[canonical_id]["ogdc_sources"],
                "link_basis": "checksum_bound_article_entity_links",
                "link_evidence": attributes[canonical_id]["ogdc_link_evidence"],
                "content_sha256": representative.content_sha256,
            }
        )

    total_occurrences = len(occurrences)
    unique_canonicals = len(representatives)
    missing_publication = sum(item.published_at is None for item in occurrences)
    missing_available = sum(item.available_at is None for item in occurrences)
    publication_completeness = round(
        100.0 * (total_occurrences - missing_publication) / total_occurrences, 6
    ) if total_occurrences else 0.0
    availability_completeness = round(
        100.0 * (total_occurrences - missing_available) / total_occurrences, 6
    ) if total_occurrences else 0.0
    sources = Counter(item.source_name for item in occurrences)
    source_canonicals: dict[str, set[str]] = defaultdict(set)
    for item in occurrences:
        source_canonicals[item.source_name].add(item.canonical_article_id)
    languages = Counter(item.language or "unknown" for item in occurrences)
    language_canonicals: dict[str, set[str]] = defaultdict(set)
    for item in occurrences:
        language_canonicals[item.language or "unknown"].add(
            item.canonical_article_id
        )
    providers = Counter(item.provider for item in occurrences)
    provider_canonicals: dict[str, set[str]] = defaultdict(set)
    for item in occurrences:
        provider_canonicals[item.provider].add(item.canonical_article_id)
    arrival_counts = Counter(
        assignment.arrival_class for assignment in occurrence_assignments.values()
    )
    mapping_counts = Counter(
        assignment.mapping_status for assignment in occurrence_assignments.values()
    )
    publication_overlap = _calendar_overlap(
        (item.published_at for item in occurrences), stock, zone
    )
    availability_overlap = _calendar_overlap(
        (item.available_at for item in occurrences), stock, zone
    )
    readiness, readiness_evaluation = calculate_readiness(
        split_rows,
        thresholds=config.thresholds,
        publication_completeness_percent=publication_completeness,
        available_at_completeness_percent=availability_completeness,
        distinct_sources=len(sources),
        publication_overlap_exists=publication_overlap["start"] is not None,
    )
    readiness_evaluation = dict(readiness_evaluation)
    readiness_evaluation["source_permission_evidence"] = {
        "passed": not news.permission_review_required,
        "review_required": news.permission_review_required,
        "notes": list(news.permission_notes),
    }
    incomplete_runs = sorted(
        str(reference["run_id"])
        for reference in news.run_references
        if not bool(reference.get("ingestion_complete", False))
    )
    provider_error_count = sum(
        int(reference.get("provider_error_count", 0))
        for reference in news.run_references
    )
    rejected_record_count = sum(
        int(reference.get("rejected_record_count", 0))
        for reference in news.run_references
    )
    readiness_evaluation["ingestion_completeness"] = {
        "passed": not incomplete_runs,
        "incomplete_run_ids": incomplete_runs,
        "provider_error_count": provider_error_count,
        "rejected_record_count": rejected_record_count,
    }
    database_files = news.database_fingerprints.get("files")
    database_fingerprinted = bool(
        news.database_fingerprints.get("provided")
        and isinstance(database_files, list)
        and any(
            isinstance(item, Mapping) and item.get("label") == "database"
            for item in database_files
        )
    )
    unsealed_run_ids = sorted(
        str(reference["run_id"])
        for reference in news.run_references
        if reference.get("verification_basis")
        != "CURRENT_COMPLETION_REPLAY_AND_DATABASE_SEAL"
    )
    database_bound = database_fingerprinted and not unsealed_run_ids
    readiness_evaluation["database_provenance"] = {
        "passed": database_bound,
        "authoritative_completion_seal_available": database_bound,
        "database_fingerprinted": database_fingerprinted,
        "unsealed_run_ids": unsealed_run_ids,
        "reason": (
            "Every selected news pack replayed against the fingerprinted SQLite seal."
            if database_bound
            else (
                "The SQLite database was fingerprinted, but one or more legacy packs "
                "have no current database-bound completion seal."
                if database_fingerprinted
                else "No SQLite database fingerprint was supplied; this is artifact-only replay."
            )
        ),
    }
    readiness_blockers: list[str] = []
    if news.permission_review_required:
        readiness_blockers.append("source licensing/usage review")
    if incomplete_runs:
        readiness_blockers.append("incomplete or partial ingestion evidence")
    if not database_bound:
        readiness_blockers.append("missing SQLite database provenance")
    if readiness_blockers and readiness == "READY_FOR_AGGREGATION":
        readiness = "PARTIALLY_COVERED"
        readiness_evaluation["passed"] = False
        readiness_evaluation["reason"] = (
            "Coverage thresholds passed, but readiness is blocked by: "
            + "; ".join(readiness_blockers)
            + "."
        )
    if readiness not in READINESS_STATUSES:  # defensive invariant
        raise CoverageAuditError("Unexpected readiness status")

    quality_report: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "algorithm_version": COVERAGE_ALGORITHM_VERSION,
        "total_news_occurrences": total_occurrences,
        "unique_canonical_articles": unique_canonicals,
        "ogdc_linked_occurrences": sum(item.is_ogdc for item in occurrences),
        "ogdc_linked_canonical_articles": sum(
            bool(value["is_ogdc"]) for value in attributes.values()
        ),
        "energy_linked_occurrences": sum(item.is_energy for item in occurrences),
        "energy_linked_canonical_articles": sum(
            bool(value["is_energy"]) for value in attributes.values()
        ),
        "energy_sector_linking": {
            "supported": energy_evidence_present,
            "reason": (
                "Deterministic mentioned_sectors evidence was present."
                if energy_evidence_present
                else "The current news contract does not infer sectors; general news "
                "was not relabelled as energy-sector news."
            ),
        },
        "duplicated_occurrences": sum(item.duplicate_of is not None for item in occurrences),
        "canonical_article_count": unique_canonicals,
        "publication_time": {
            "complete_count": total_occurrences - missing_publication,
            "missing_count": missing_publication,
            "completeness_percent": publication_completeness,
            "range": _date_range(item.published_at for item in occurrences),
            "calendar_overlap_with_prices": publication_overlap,
        },
        "available_at": {
            "complete_count": total_occurrences - missing_available,
            "missing_count": missing_available,
            "completeness_percent": availability_completeness,
            "range": _date_range(item.available_at for item in occurrences),
            "calendar_overlap_with_prices": availability_overlap,
        },
        "arrival_counts": dict(sorted(arrival_counts.items())),
        "session_mapping_counts": dict(sorted(mapping_counts.items())),
        "source_coverage": {
            key: {
                "occurrences": sources[key],
                "canonical_articles": len(source_canonicals[key]),
            }
            for key in sorted(sources)
        },
        "provider_coverage": {
            key: {
                "occurrences": providers[key],
                "canonical_articles": len(provider_canonicals[key]),
            }
            for key in sorted(providers)
        },
        "language_coverage": {
            key: {
                "occurrences": languages[key],
                "canonical_articles": len(language_canonicals[key]),
            }
            for key in sorted(languages)
        },
        "readiness_status": readiness,
        "readiness_evaluation": readiness_evaluation,
        "readiness_thresholds": config.thresholds.to_dict(),
        "readiness_interpretation": (
            "Engineering/data-coverage thresholds only; this result is not proof "
            "of statistical sufficiency, predictive improvement, or financial advice."
        ),
        "source_permission_evidence": readiness_evaluation[
            "source_permission_evidence"
        ],
        "ingestion_completeness": readiness_evaluation[
            "ingestion_completeness"
        ],
        "database_provenance": readiness_evaluation["database_provenance"],
        "limitations": [
            "Coverage uses available_at for session eligibility and never backdates "
            "a later import to its historical publication date.",
            "This audit performs no sentiment aggregation or model training.",
            ACADEMIC_DISCLAIMER,
        ],
    }
    full_rows = {
        row["population"]: row
        for row in split_rows
        if row["split"] == "full_price_period"
    }
    final_summary: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "status": "PASS",
        "readiness_status": readiness,
        "symbol": stock.symbol,
        "total_news_occurrences": total_occurrences,
        "unique_canonical_articles": unique_canonicals,
        "ogdc_linked_canonical_articles": quality_report[
            "ogdc_linked_canonical_articles"
        ],
        "ogdc_linked_occurrences": quality_report["ogdc_linked_occurrences"],
        "duplicated_occurrences": quality_report["duplicated_occurrences"],
        "publication_date_range": quality_report["publication_time"]["range"],
        "available_at_range": quality_report["available_at"]["range"],
        "publication_calendar_overlap_with_prices": quality_report[
            "publication_time"
        ]["calendar_overlap_with_prices"],
        "availability_calendar_overlap_with_prices": quality_report["available_at"][
            "calendar_overlap_with_prices"
        ],
        "stock_date_range": stock.reference["price_date_range"],
        "full_price_coverage": {
            population: {
                "trading_days_with_articles": full_rows[population][
                    "trading_days_with_articles"
                ],
                "coverage_percent": full_rows[population]["coverage_percent"],
            }
            for population in POPULATIONS
        },
        "split_coverage": {
            split: {
                population: next(
                    row["coverage_percent"]
                    for row in split_rows
                    if row["split"] == split and row["population"] == population
                )
                for population in POPULATIONS
            }
            for split in SPLITS
        },
        "source_count": len(sources),
        "population_support": {
            "all_financial_news": {
                "supported": population_supported["all_financial_news"]
            },
            "ogdc_linked_news": {
                "supported": population_supported["ogdc_linked_news"]
            },
            "energy_sector_news": quality_report["energy_sector_linking"],
        },
        "readiness_thresholds": config.thresholds.to_dict(),
        "readiness_evaluation": readiness_evaluation,
        "source_permission_evidence": readiness_evaluation[
            "source_permission_evidence"
        ],
        "ingestion_completeness": readiness_evaluation[
            "ingestion_completeness"
        ],
        "database_provenance": readiness_evaluation["database_provenance"],
        "limitations": list(quality_report["limitations"]),
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    return CoverageComputation(
        trading_day_rows=tuple(trading_day_rows),
        monthly_rows=tuple(monthly_rows),
        split_rows=tuple(split_rows),
        missing_interval_rows=tuple(missing_rows),
        article_session_rows=tuple(article_session_rows),
        ogdc_article_rows=tuple(ogdc_rows),
        quality_report=quality_report,
        final_summary=final_summary,
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (list, tuple, dict))
                    else value
                    for field, value in ((field, row.get(field)) for field in fields)
                }
            )


def _artifact_inventory(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name in exclude:
            continue
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise CoverageAuditError("Coverage packs may contain only regular files")
        records.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": metadata.st_size,
            }
        )
    return records


def _normalize_coverage_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CoverageAuditError("Coverage manifest artifact inventory is missing")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    previous = ""
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "sha256",
            "size_bytes",
        }:
            raise CoverageAuditError("Coverage artifact inventory entry is invalid")
        relative = _safe_relative(
            item.get("path"), field_name="coverage artifact inventory path"
        )
        digest = item.get("sha256")
        size = item.get("size_bytes")
        if relative in _COVERAGE_SEAL_FILES or relative in seen:
            raise CoverageAuditError(
                "Coverage artifact inventory paths must be unique payload files"
            )
        if relative < previous:
            raise CoverageAuditError(
                "Coverage artifact inventory must be deterministically ordered"
            )
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise CoverageAuditError("Coverage artifact inventory SHA-256 is invalid")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise CoverageAuditError("Coverage artifact inventory size is invalid")
        seen.add(relative)
        previous = relative
        normalized.append(
            {"path": relative, "sha256": digest, "size_bytes": size}
        )
    return normalized


def verify_coverage_run(run_directory: str | Path) -> CoverageRunVerification:
    """Verify a completed coverage pack without contacting a provider or database."""

    supplied = Path(run_directory)
    try:
        metadata = supplied.lstat()
    except OSError as exc:
        raise CoverageAuditError("Coverage run directory does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CoverageAuditError("Coverage run root must be a real directory")
    root = supplied.resolve(strict=True)
    return _verify_coverage_root(root, expected_run_id=root.name)


def _verify_coverage_root(
    root: Path, *, expected_run_id: str
) -> CoverageRunVerification:
    """Verify a resolved pack root, including a not-yet-renamed staging root."""

    manifest_path = _regular_file(
        root / _COVERAGE_MANIFEST, root=root, label="coverage run manifest"
    )
    summary_path = _regular_file(
        root / "final_summary.json", root=root, label="coverage final summary"
    )
    completion_path = _regular_file(
        root / _COVERAGE_COMPLETION, root=root, label="coverage completion marker"
    )
    manifest = _read_json(
        manifest_path, root=root, label="coverage run manifest"
    )
    summary = _read_json(summary_path, root=root, label="coverage final summary")
    completion = _read_json(
        completion_path, root=root, label="coverage completion marker"
    )

    expected_completion_keys = {
        "schema_version",
        "coverage_schema_version",
        "algorithm_version",
        "run_id",
        "status",
        "readiness_status",
        "finalization_state",
        "completed_at",
        "manifest_sha256",
        "manifest_size_bytes",
        "summary_sha256",
        "artifact_inventory_sha256",
        "artifact_count",
    }
    if set(completion) != expected_completion_keys:
        raise CoverageAuditError("Coverage completion marker schema is invalid")
    if completion.get("schema_version") != "1.0":
        raise CoverageAuditError("Coverage completion marker version is unsupported")

    manifest_digest = sha256_file(manifest_path)
    summary_digest = sha256_file(summary_path)
    if completion.get("manifest_sha256") != manifest_digest:
        raise CoverageAuditError(
            "Coverage completion marker does not bind this manifest"
        )
    if completion.get("manifest_size_bytes") != manifest_path.stat().st_size:
        raise CoverageAuditError("Coverage completion marker manifest size differs")
    if completion.get("summary_sha256") != summary_digest:
        raise CoverageAuditError(
            "Coverage completion marker does not bind this summary"
        )

    run_id = completion.get("run_id")
    status = completion.get("status")
    readiness = completion.get("readiness_status")
    if (
        not isinstance(run_id, str)
        or validate_coverage_run_id(run_id) != run_id
        or expected_run_id != run_id
        or manifest.get("run_id") != run_id
        or summary.get("run_id") != run_id
    ):
        raise CoverageAuditError(
            "Coverage completion, manifest, summary, and directory IDs differ"
        )
    if (
        status != "PASS"
        or manifest.get("status") != status
        or summary.get("status") != status
    ):
        raise CoverageAuditError("Coverage completion evidence has an invalid status")
    if (
        readiness not in READINESS_STATUSES
        or manifest.get("readiness_status") != readiness
        or summary.get("readiness_status") != readiness
    ):
        raise CoverageAuditError("Coverage completion readiness evidence differs")
    if any(
        value.get("finalization_state") != "COMMITTED"
        for value in (completion, manifest, summary)
    ):
        raise CoverageAuditError("Coverage completion evidence is not committed")
    if (
        completion.get("coverage_schema_version") != COVERAGE_SCHEMA_VERSION
        or manifest.get("schema_version") != COVERAGE_SCHEMA_VERSION
        or summary.get("schema_version") != COVERAGE_SCHEMA_VERSION
    ):
        raise CoverageAuditError("Coverage artifact schema versions differ")
    if (
        completion.get("algorithm_version") != COVERAGE_ALGORITHM_VERSION
        or manifest.get("algorithm_version") != COVERAGE_ALGORITHM_VERSION
        or manifest.get("completion_marker") != _COVERAGE_COMPLETION
        or completion.get("completed_at") != manifest.get("completed_at")
    ):
        raise CoverageAuditError("Coverage completion and manifest provenance differ")
    if (
        manifest.get("config") != summary.get("config")
        or manifest.get("effective_readiness_policy_sha256")
        != summary.get("effective_readiness_policy_sha256")
    ):
        raise CoverageAuditError("Coverage manifest and summary configuration differ")

    inventory = _normalize_coverage_inventory(manifest.get("artifact_inventory"))
    inventory_digest = _sha256_bytes(_canonical_json_bytes(inventory))
    if (
        manifest.get("artifact_inventory_sha256") != inventory_digest
        or completion.get("artifact_inventory_sha256") != inventory_digest
        or completion.get("artifact_count") != len(inventory)
    ):
        raise CoverageAuditError("Coverage artifact inventory seal differs")
    if "final_summary.json" not in {item["path"] for item in inventory}:
        raise CoverageAuditError("Coverage final summary is not payload-inventoried")
    observed = _artifact_inventory(root, exclude=set(_COVERAGE_SEAL_FILES))
    if observed != inventory:
        raise CoverageAuditError(
            "Coverage artifacts are missing, extra, changed, or unsafe"
        )

    return CoverageRunVerification(
        run_id=run_id,
        status=status,
        readiness_status=readiness,
        artifact_count=len(inventory),
        manifest_sha256=manifest_digest,
        artifact_inventory_sha256=inventory_digest,
    )


def _portable_source_assessment_reference(path: Path) -> dict[str, Any]:
    source = _regular_file(path, label="historical-news source assessment")
    return {
        "document_reference": source.name,
        "sha256": sha256_file(source),
        "selection": (
            "The assessment records each source's permission status. Evidence that "
            "still requires review may support descriptive audit only and cannot "
            "support READY_FOR_AGGREGATION."
        ),
    }


def _implementation_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parents[3]
    selected = [Path(__file__).resolve()]
    cli = repository_root / "scripts" / "audit_news_price_overlap.py"
    if cli.is_file():
        selected.append(cli.resolve())
    digest = hashlib.sha256()
    for path in sorted(selected, key=lambda item: item.as_posix()):
        relative = path.relative_to(repository_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    try:
        code_version = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty_tree = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repository_root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        code_version = "unknown"
        dirty_tree = True
    return {
        "producer": "paisa.news.coverage.audit_news_price_overlap",
        "code_version": code_version or "unknown",
        "dirty_tree": dirty_tree,
        "implementation_fingerprint_sha256": digest.hexdigest(),
        "implementation_files": [
            path.relative_to(repository_root).as_posix()
            for path in sorted(selected, key=lambda item: item.as_posix())
        ],
    }


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory while refusing an existing destination."""

    if os.name == "nt":
        try:
            os.rename(source, destination)
        except OSError as exc:
            if isinstance(exc, FileExistsError) or exc.errno in {
                errno.EEXIST,
                errno.ENOTEMPTY,
            }:
                raise CoverageRunCollisionError(
                    "Coverage run already exists; refusing to overwrite: "
                    f"{destination.name}"
                ) from exc
            raise CoverageAuditError(
                "Atomic no-replace coverage publication failed"
            ) from exc
        return

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    operation: Any
    arguments: tuple[Any, ...]
    if sys.platform == "darwin":
        operation = getattr(libc, "renamex_np", None)
        if operation is None:
            raise CoverageAuditError(
                "Atomic no-replace directory publication is unsupported"
            )
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        operation.restype = ctypes.c_int
        arguments = (encoded_source, encoded_destination, 0x00000004)
    elif sys.platform.startswith("linux"):
        operation = getattr(libc, "renameat2", None)
        if operation is None:
            raise CoverageAuditError(
                "Atomic no-replace directory publication is unsupported"
            )
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        arguments = (-100, encoded_source, -100, encoded_destination, 0x00000001)
    else:
        raise CoverageAuditError(
            "Atomic no-replace directory publication is unsupported"
        )

    if operation(*arguments) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise CoverageRunCollisionError(
            "Coverage run already exists; refusing to overwrite: "
            f"{destination.name}"
        ) from OSError(error_number, os.strerror(error_number))
    raise CoverageAuditError(
        "Atomic no-replace coverage publication failed"
    ) from OSError(error_number, os.strerror(error_number))


def _discard_coverage_staging(staging: Path) -> None:
    """Best-effort removal of a self-owned staging path without following links."""

    try:
        metadata = staging.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        shutil.rmtree(staging)
    else:
        staging.unlink(missing_ok=True)


def publish_coverage_run(
    computation: CoverageComputation,
    *,
    news: NewsEvidence,
    stock: StockEvidence,
    config: CoverageConfig,
    output_root: str | Path,
    source_assessment_path: str | Path,
    run_id: str | None = None,
    created_at: datetime | None = None,
) -> Path:
    selected_id = validate_coverage_run_id(run_id or make_coverage_run_id(created_at))
    root_candidate = Path(output_root)
    if root_candidate.is_symlink():
        raise CoverageAuditError("Coverage output root must not be a symlink")
    root_candidate.mkdir(parents=True, exist_ok=True)
    root = root_candidate.resolve()
    destination = root / selected_id
    staging = root / f".{selected_id}.staging"
    reservation = root / f".{selected_id}.publish.lock"
    reservation_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    reservation_flags |= getattr(os, "O_CLOEXEC", 0)
    reservation_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        reservation_fd = os.open(reservation, reservation_flags, 0o600)
    except FileExistsError as exc:
        raise CoverageRunCollisionError(
            f"Coverage run already exists; refusing to overwrite: {selected_id}"
        ) from exc
    os.close(reservation_fd)
    staging_created = False
    try:
        if os.path.lexists(destination) or os.path.lexists(staging):
            raise CoverageRunCollisionError(
                f"Coverage run already exists; refusing to overwrite: {selected_id}"
            )
        staging.mkdir(mode=0o700)
        staging_created = True
        assessment_candidate = Path(source_assessment_path)
        if assessment_candidate.is_symlink():
            raise CoverageAuditError(
                "Historical-news source assessment must not be a symlink"
            )
        assessment_reference = _portable_source_assessment_reference(
            assessment_candidate.resolve()
        )
        assessment_reference["input_source_evidence"] = [
            {
                "run_id": reference["run_id"],
                "provider": reference.get("provider"),
                "provider_mode": reference.get("provider_mode"),
                "licensing_usage_note": reference.get("licensing_usage_note"),
                "source_permission_decision": reference.get(
                    "source_permission_decision"
                ),
            }
            for reference in news.run_references
        ]
        assessment_reference["permission_policy"] = dict(
            news.permission_policy_reference
        )
        _write_json(staging / "source_assessment_reference.json", assessment_reference)
        news_reference = {
            "schema_version": COVERAGE_SCHEMA_VERSION,
            "news_runs": list(news.run_references),
            "database_fingerprints": news.database_fingerprints,
            "occurrence_count": len(news.occurrences),
            "canonical_count": len(
                {item.canonical_article_id for item in news.occurrences}
            ),
            "provider_modes": sorted({item.provider_mode for item in news.occurrences}),
            "data_classes": sorted({item.data_class for item in news.occurrences}),
        }
        _write_json(staging / "imported_or_acquired_news_reference.json", news_reference)
        _write_json(staging / "stock_artifact_reference.json", stock.reference)
        _write_csv(
            staging / "overlap_by_trading_day.csv",
            computation.trading_day_rows,
            (
                "trading_date",
                "split",
                "decision_cutoff_utc",
                "all_financial_occurrence_count",
                "all_financial_canonical_article_count",
                "ogdc_linked_occurrence_count",
                "ogdc_linked_canonical_article_count",
                "energy_sector_occurrence_count",
                "energy_sector_canonical_article_count",
            ),
        )
        _write_csv(
            staging / "overlap_by_month.csv",
            computation.monthly_rows,
            (
                "month",
                "population",
                "trading_days",
                "trading_days_with_articles",
                "coverage_percent",
                "article_occurrences",
                "canonical_articles",
                "articles_per_trading_day",
                "zero_article_trading_days",
            ),
        )
        _write_csv(
            staging / "split_coverage.csv",
            computation.split_rows,
            (
                "population",
                "split",
                "trading_days",
                "trading_days_with_articles",
                "coverage_percent",
                "article_occurrences",
                "canonical_articles",
                "articles_per_trading_day",
                "zero_article_trading_days",
                "longest_missing_gap_sessions",
            ),
        )
        _write_csv(
            staging / "missing_intervals.csv",
            computation.missing_interval_rows,
            (
                "population",
                "split",
                "start_date",
                "end_date",
                "missing_trading_sessions",
            ),
        )
        _write_csv(
            staging / "article_session_mappings.csv",
            computation.article_session_rows,
            (
                "occurrence_key",
                "ingestion_run_id",
                "article_id",
                "canonical_article_id",
                "provider",
                "source_name",
                "data_class",
                "population",
                "population_supported",
                "occurrence_in_population",
                "is_duplicate_occurrence",
                "published_at",
                "available_at",
                "fetched_at",
                "occurrence_assigned_trading_date",
                "occurrence_mapping_status",
                "occurrence_arrival_class",
                "canonical_representative_article_id",
                "canonical_representative_ingestion_run_id",
                "is_population_canonical_representative",
                "canonical_available_at",
                "canonical_assigned_trading_date",
                "canonical_mapping_status",
                "canonical_arrival_class",
                "content_sha256",
                "raw_artifact_reference",
            ),
        )
        _write_csv(
            staging / "ogdc_linked_articles.csv",
            computation.ogdc_article_rows,
            (
                "canonical_article_id",
                "representative_article_id",
                "source_name",
                "source_url",
                "title",
                "published_at",
                "available_at",
                "assigned_trading_date",
                "mapping_status",
                "arrival_class",
                "occurrence_count",
                "sources",
                "link_basis",
                "link_evidence",
                "content_sha256",
            ),
        )
        _write_json(staging / "quality_report.json", computation.quality_report)
        effective_readiness_policy_sha256 = _sha256_bytes(
            _canonical_json_bytes(config.to_dict())
        )
        final_summary = dict(computation.final_summary)
        final_summary.update(
            {
                "run_id": selected_id,
                "artifact_directory": f"news_coverage/{selected_id}",
                "finalization_state": "COMMITTED",
                "config": config.to_dict(),
                "effective_readiness_policy_sha256": (
                    effective_readiness_policy_sha256
                ),
            }
        )
        _write_json(staging / "final_summary.json", final_summary)

        inventory = _artifact_inventory(staging, exclude=set(_COVERAGE_SEAL_FILES))
        instant = _aware_utc(created_at or _utc_now(), field_name="created_at")
        input_fingerprint = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "news_runs": list(news.run_references),
                    "database": news.database_fingerprints,
                    "stock": stock.reference,
                    "source_assessment": assessment_reference,
                    "config": config.to_dict(),
                }
            )
        )
        manifest = {
            "schema_version": COVERAGE_SCHEMA_VERSION,
            "algorithm_version": COVERAGE_ALGORITHM_VERSION,
            "run_id": selected_id,
            "status": "PASS",
            "finalization_state": "COMMITTED",
            "created_at": utc_iso(instant),
            "completed_at": utc_iso(instant),
            "readiness_status": computation.final_summary["readiness_status"],
            "input_fingerprint_sha256": input_fingerprint,
            **_implementation_provenance(),
            "config": config.to_dict(),
            "effective_readiness_policy_sha256": (
                effective_readiness_policy_sha256
            ),
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": _sha256_bytes(
                _canonical_json_bytes(inventory)
            ),
            "completion_marker": _COVERAGE_COMPLETION,
            "disclaimer": ACADEMIC_DISCLAIMER,
        }
        manifest_path = staging / _COVERAGE_MANIFEST
        summary_path = staging / "final_summary.json"
        _write_json(manifest_path, manifest)
        completion = {
            "schema_version": "1.0",
            "coverage_schema_version": COVERAGE_SCHEMA_VERSION,
            "algorithm_version": COVERAGE_ALGORITHM_VERSION,
            "run_id": selected_id,
            "status": "PASS",
            "readiness_status": computation.final_summary["readiness_status"],
            "finalization_state": "COMMITTED",
            "completed_at": utc_iso(instant),
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "summary_sha256": sha256_file(summary_path),
            "artifact_inventory_sha256": manifest["artifact_inventory_sha256"],
            "artifact_count": len(inventory),
        }
        # This is the non-circular commit point: it is written only after the
        # final manifest and every payload file have their stable identities.
        _write_json(staging / _COVERAGE_COMPLETION, completion)
        _verify_coverage_root(
            staging.resolve(strict=True), expected_run_id=selected_id
        )
        # The O_EXCL reservation serializes cooperating publishers. Recheck at
        # the last possible point so a non-cooperating late path is never
        # knowingly replaced by POSIX directory-rename semantics.
        if os.path.lexists(destination):
            raise CoverageRunCollisionError(
                f"Coverage run already exists; refusing to overwrite: {selected_id}"
            )
        _rename_directory_noreplace(staging, destination)
    except Exception:
        try:
            if staging_created:
                _discard_coverage_staging(staging)
        except Exception:
            pass
        try:
            reservation.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    reservation.unlink(missing_ok=True)
    return destination


def audit_news_price_overlap(
    *,
    news_run_directories: Iterable[str | Path],
    vertical_slice_directory: str | Path,
    model_training_directory: str | Path,
    output_root: str | Path,
    source_assessment_path: str | Path,
    config: CoverageConfig | None = None,
    run_id: str | None = None,
    real_mode: bool = True,
    database_path: str | Path | None = None,
    allow_legacy_unsealed: bool = False,
    source_permission_policy_path: str | Path | None = None,
    created_at: datetime | None = None,
) -> tuple[CoverageComputation, Path]:
    selected_config = config or CoverageConfig()
    permission_policy = (
        load_source_permission_policy(
            source_permission_policy_path, source_assessment_path
        )
        if source_permission_policy_path is not None
        else None
    )
    news = load_news_evidence(
        news_run_directories,
        real_mode=real_mode,
        database_path=database_path,
        allow_legacy_unsealed=allow_legacy_unsealed,
        source_permission_policy=permission_policy,
    )
    stock = load_stock_evidence(
        vertical_slice_directory, model_training_directory, symbol="OGDC"
    )
    computation = compute_coverage(news, stock, selected_config)
    destination = publish_coverage_run(
        computation,
        news=news,
        stock=stock,
        config=selected_config,
        output_root=output_root,
        source_assessment_path=source_assessment_path,
        run_id=run_id,
        created_at=created_at,
    )
    return computation, destination


__all__ = [
    "ACADEMIC_DISCLAIMER",
    "COVERAGE_ALGORITHM_VERSION",
    "COVERAGE_SCHEMA_VERSION",
    "CoverageAuditError",
    "CoverageComputation",
    "CoverageConfig",
    "CoverageRunCollisionError",
    "CoverageRunVerification",
    "NewsEvidence",
    "NewsOccurrence",
    "ReadinessThresholds",
    "SessionAssignment",
    "SourcePermissionPolicy",
    "StockEvidence",
    "assign_to_session",
    "audit_news_price_overlap",
    "calculate_readiness",
    "compute_coverage",
    "discover_news_run_directories",
    "load_news_evidence",
    "load_source_permission_policy",
    "load_stock_evidence",
    "make_coverage_run_id",
    "parse_cutoff",
    "publish_coverage_run",
    "verify_coverage_run",
    "validate_coverage_run_id",
]
