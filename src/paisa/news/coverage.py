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
from dataclasses import dataclass, field, replace
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
import warnings
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paisa.news.models import NEWS_AVAILABILITY_POLICY_VERSION
from paisa.news.verification import NewsRunVerificationError, verify_news_run

from paisa.news.providers.manual_csv import LICENSING_REVIEW_REQUIRED


COVERAGE_SCHEMA_VERSION = "1.1"
COVERAGE_ALGORITHM_VERSION = "news-price-coverage-v3"
READINESS_CONFIG_SCHEMA_VERSION = "1.0"
ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
_COVERAGE_MANIFEST = "run_manifest.json"
_COVERAGE_COMPLETION = "completion.json"
_COVERAGE_SEAL_FILES = frozenset({_COVERAGE_MANIFEST, _COVERAGE_COMPLETION})
_COVERAGE_PAYLOAD_FILES = frozenset(
    {
        "article_session_mappings.csv",
        "final_summary.json",
        "imported_or_acquired_news_reference.json",
        "missing_intervals.csv",
        "ogdc_linked_articles.csv",
        "overlap_by_month.csv",
        "overlap_by_trading_day.csv",
        "quality_report.json",
        "source_assessment_reference.json",
        "split_coverage.csv",
        "stock_artifact_reference.json",
    }
)
READINESS_STATUSES = {
    "READY_FOR_AGGREGATION",
    "PARTIALLY_COVERED",
    "INSUFFICIENT_OVERLAP",
    "NO_OVERLAP",
}
POPULATIONS = ("all_financial_news", "ogdc_linked_news", "energy_sector_news")
SPLITS = ("train", "validation", "test")
ENERGY_SECTOR_LINK_CONTRACT_SCHEMA_VERSION = "1.0"
ENERGY_SECTOR_LINK_METHOD = "reviewed_deterministic_article_sector_links"
_MINIMUM_READY_ALL_NEWS_COVERAGE_PERCENT = 5.0
_MINIMUM_READY_OGDC_COVERAGE_PERCENT = 1.0
_MINIMUM_READY_COMPLETENESS_PERCENT = 1.0
_MINIMUM_READY_DISTINCT_SOURCES = 2
_MINIMUM_READY_COVERED_SESSIONS_EACH_SPLIT = 2
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
            raw_value = getattr(self, name)
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise CoverageAuditError(f"{name} must be a finite number")
            value = float(raw_value)
            if not math.isfinite(value) or not 0.0 < value <= 100.0:
                raise CoverageAuditError(
                    f"{name} must be strictly positive and no greater than 100"
                )
        integer_fields = (
            "ready_maximum_all_news_missing_gap_sessions",
            "ready_minimum_distinct_sources",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CoverageAuditError(f"{name} must be an integer")
        if self.ready_maximum_all_news_missing_gap_sessions < 0:
            raise CoverageAuditError(
                "ready_maximum_all_news_missing_gap_sessions cannot be negative"
            )
        if self.ready_minimum_distinct_sources < _MINIMUM_READY_DISTINCT_SOURCES:
            raise CoverageAuditError(
                "ready_minimum_distinct_sources must be at least "
                f"{_MINIMUM_READY_DISTINCT_SOURCES}"
            )
        if (
            self.ready_minimum_all_news_coverage_percent_each_split
            < _MINIMUM_READY_ALL_NEWS_COVERAGE_PERCENT
        ):
            raise CoverageAuditError(
                "ready_minimum_all_news_coverage_percent_each_split must be at "
                f"least {_MINIMUM_READY_ALL_NEWS_COVERAGE_PERCENT}"
            )
        if (
            self.ready_minimum_ogdc_coverage_percent_each_split
            < _MINIMUM_READY_OGDC_COVERAGE_PERCENT
        ):
            raise CoverageAuditError(
                "ready_minimum_ogdc_coverage_percent_each_split must be at least "
                f"{_MINIMUM_READY_OGDC_COVERAGE_PERCENT}"
            )
        for name in (
            "ready_minimum_publication_time_completeness_percent",
            "ready_minimum_available_at_completeness_percent",
        ):
            if float(getattr(self, name)) < _MINIMUM_READY_COMPLETENESS_PERCENT:
                raise CoverageAuditError(
                    f"{name} must be at least {_MINIMUM_READY_COMPLETENESS_PERCENT}"
                )
        if (
            self.ready_minimum_all_news_coverage_percent_each_split
            < self.partial_minimum_all_news_coverage_percent_each_split
        ):
            raise CoverageAuditError(
                "ready all-news coverage threshold cannot be weaker than its "
                "partial threshold"
            )
        if (
            self.ready_minimum_ogdc_coverage_percent_each_split
            < self.partial_minimum_ogdc_coverage_percent_each_split
        ):
            raise CoverageAuditError(
                "ready OGDC coverage threshold cannot be weaker than its partial "
                "threshold"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReadinessThresholds":
        allowed = set(cls.__dataclass_fields__)
        unexpected = sorted(set(value) - allowed - {"schema_version", "purpose"})
        if unexpected:
            raise CoverageAuditError(
                "Readiness configuration has unsupported fields: "
                + ", ".join(unexpected)
            )
        if (
            value.get("schema_version", READINESS_CONFIG_SCHEMA_VERSION)
            != READINESS_CONFIG_SCHEMA_VERSION
        ):
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
        if self.decision_cutoff.microsecond:
            raise CoverageAuditError(
                "decision_cutoff must use whole-second precision"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "market_timezone": self.market_timezone,
            "decision_cutoff": self.decision_cutoff.isoformat(timespec="seconds"),
            "readiness_thresholds": self.thresholds.to_dict(),
            "threshold_interpretation": (
                "Caller-defined gate satisfaction only; not proof of statistical "
                "sufficiency, predictive value, or financial merit."
            ),
        }


def parse_cutoff(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise CoverageAuditError("Decision cutoff must be HH:MM or HH:MM:SS") from exc
    if parsed.tzinfo is not None:
        raise CoverageAuditError("Decision cutoff must not include a timezone offset")
    if parsed.microsecond:
        raise CoverageAuditError("Decision cutoff must use whole-second precision")
    return parsed


@dataclass(frozen=True, slots=True)
class EnergySectorLinkContract:
    """Explicit reviewed authority for interpreting sector-link fields."""

    contract_id: str
    evidence_sha256: str
    schema_version: str = ENERGY_SECTOR_LINK_CONTRACT_SCHEMA_VERSION
    link_method: str = ENERGY_SECTOR_LINK_METHOD
    evidence_field: str = "mentioned_sectors"

    def __post_init__(self) -> None:
        if (
            not self.contract_id
            or self.contract_id.strip() != self.contract_id
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.contract_id)
        ):
            raise CoverageAuditError("Energy-sector link contract ID is invalid")
        if self.schema_version != ENERGY_SECTOR_LINK_CONTRACT_SCHEMA_VERSION:
            raise CoverageAuditError("Energy-sector link contract schema is unsupported")
        if self.link_method != ENERGY_SECTOR_LINK_METHOD:
            raise CoverageAuditError("Energy-sector link method is unsupported")
        if self.evidence_field != "mentioned_sectors":
            raise CoverageAuditError("Energy-sector evidence field is unsupported")
        if not _SHA256.fullmatch(self.evidence_sha256):
            raise CoverageAuditError(
                "Energy-sector link evidence SHA-256 is invalid"
            )

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_id": self.contract_id,
            "schema_version": self.schema_version,
            "link_method": self.link_method,
            "evidence_field": self.evidence_field,
            "evidence_sha256": self.evidence_sha256,
        }


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
    source_observed_at: datetime | None = None

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


def _energy_sector_link_evidence_sha256(
    occurrences: Sequence[NewsOccurrence],
) -> str:
    payload = [
        {
            "occurrence_key": item.occurrence_key,
            "article_id": item.article_id,
            "canonical_article_id": item.canonical_article_id,
            "content_sha256": item.content_sha256,
            "mentioned_sectors": sorted(set(item.sectors)),
        }
        for item in sorted(occurrences, key=lambda value: value.occurrence_key)
    ]
    return _sha256_bytes(_canonical_json_bytes(payload))


def _publisher_document_key(value: str | None) -> str | None:
    """Return conservative IDs only for reviewed publisher URL structures."""

    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    path = re.sub(r"/{2,}", "/", parsed.path).casefold()
    patterns = {
        "dawn.com": ("news", r"^/news/([0-9]+)(?:/.*)?$"),
        "brecorder.com": ("news", r"^/news/([0-9]+)(?:/.*)?$"),
        "thenews.com.pk": ("print", r"^/print/([0-9]+)(?:[-/].*)?$"),
    }
    configured = patterns.get(host)
    if configured is not None:
        namespace, pattern = configured
        match = re.fullmatch(pattern, path)
        if match is not None:
            return f"{host}:{namespace}:{match.group(1)}"
    if host == "iranoilgas.com" and path == "/news/details.aspx":
        identifiers = [
            item
            for key, item in parse_qsl(parsed.query, keep_blank_values=False)
            if key.casefold() == "id" and item.isdecimal()
        ]
        if len(identifiers) == 1:
            return f"{host}:news:{identifiers[0]}"
    return None


def _publisher_document_duplicate_candidates(
    occurrences: Sequence[NewsOccurrence],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"canonical_article_ids": set(), "source_urls": set()}
    )
    for item in occurrences:
        key = _publisher_document_key(item.source_url)
        if key is None or item.source_url is None:
            continue
        grouped[key]["canonical_article_ids"].add(item.canonical_article_id)
        grouped[key]["source_urls"].add(item.source_url)
    return [
        {
            "publisher_document_key": key,
            "canonical_article_ids": sorted(values["canonical_article_ids"]),
            "source_urls": sorted(values["source_urls"]),
        }
        for key, values in sorted(grouped.items())
        if len(values["canonical_article_ids"]) > 1
    ]


@dataclass(frozen=True, slots=True)
class NewsEvidence:
    occurrences: tuple[NewsOccurrence, ...]
    run_references: tuple[Mapping[str, Any], ...]
    database_fingerprints: Mapping[str, Any]
    permission_review_required: bool = False
    permission_notes: tuple[str, ...] = ()
    permission_policy_reference: Mapping[str, Any] = field(default_factory=dict)
    energy_sector_link_contract: EnergySectorLinkContract | None = None


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
    """Verification result with an explicit evidence-authority boundary."""

    run_id: str
    status: str
    readiness_status: str
    artifact_count: int
    manifest_sha256: str
    artifact_inventory_sha256: str
    verification_basis: str = "PACK_SEMANTIC"

    def to_dict(self) -> dict[str, Any]:
        source_bound = self.verification_basis == "SOURCE_EVIDENCE_BOUND"
        return {
            "run_id": self.run_id,
            "status": self.status,
            "readiness_status": self.readiness_status,
            "artifact_count": self.artifact_count,
            "manifest_sha256": self.manifest_sha256,
            "artifact_inventory_sha256": self.artifact_inventory_sha256,
            "verification_basis": self.verification_basis,
            "source_evidence_bound": source_bound,
            "completion_acceptance": (
                "SOURCE_EVIDENCE_BOUND_SEMANTIC_REPLAY"
                if source_bound
                else "PACK_SEMANTIC_RECONCILIATION_NOT_SOURCE_EVIDENCE_BOUND"
            ),
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
    news_schema_version = str(manifest.get("schema_version") or "")
    availability_policy_version = manifest.get("availability_policy_version")
    if news_schema_version == "1.1":
        if availability_policy_version != NEWS_AVAILABILITY_POLICY_VERSION:
            raise CoverageAuditError(
                "News run availability policy version is missing or unsupported"
            )
    elif news_schema_version == "1.0":
        if availability_policy_version is not None:
            raise CoverageAuditError(
                "Legacy news schema 1.0 cannot declare a later availability policy"
            )
    else:
        raise CoverageAuditError("News run record schema is unsupported")
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
                source_observed_at = parse_timestamp(
                    row.get("source_observed_at"),
                    field_name=f"source_observed_at row {index}",
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
                        source_observed_at=source_observed_at,
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
        "availability_policy_version": availability_policy_version,
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
    all_split_covered_sessions = [
        int(lookup[("all_financial_news", split)]["trading_days_with_articles"])
        for split in SPLITS
    ]
    ogdc_split_covered_sessions = [
        int(lookup[("ogdc_linked_news", split)]["trading_days_with_articles"])
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
        "non_configurable_minimum_evidence": {
            "actual_all_news_minimum": min(all_split_covered_sessions),
            "actual_ogdc_minimum": min(ogdc_split_covered_sessions),
            "minimum_covered_sessions_each_split": (
                _MINIMUM_READY_COVERED_SESSIONS_EACH_SPLIT
            ),
            "passed": (
                min(all_split_covered_sessions)
                >= _MINIMUM_READY_COVERED_SESSIONS_EACH_SPLIT
                and min(ogdc_split_covered_sessions)
                >= _MINIMUM_READY_COVERED_SESSIONS_EACH_SPLIT
            ),
            "interpretation": (
                "Non-configurable engineering sanity floor; not a statistical-"
                "sufficiency claim."
            ),
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
    energy_link_contract = news.energy_sector_link_contract
    if (
        energy_link_contract is not None
        and energy_link_contract.evidence_sha256
        != _energy_sector_link_evidence_sha256(occurrences)
    ):
        raise CoverageAuditError(
            "Energy-sector link contract does not match the supplied occurrence evidence"
        )
    energy_evidence_present = energy_link_contract is not None
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
            if not population_supported[population]:
                continue
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
            population_is_supported = population_supported[population]
            canonical_representative = (
                canonical_attributes["population_representatives"][population]
                if population_is_supported
                else None
            )
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
                    "source_url": item.source_url,
                    "publisher_document_key": _publisher_document_key(
                        item.source_url
                    ),
                    "mentioned_sectors": list(item.sectors),
                    "data_class": item.data_class,
                    "population": population,
                    "population_supported": population_is_supported,
                    "occurrence_in_population": (
                        _occurrence_population_match(population, item)
                        if population_is_supported
                        else None
                    ),
                    "is_duplicate_occurrence": item.duplicate_of is not None,
                    "published_at": utc_iso(item.published_at),
                    "source_observed_at": utc_iso(item.source_observed_at),
                    "available_at": utc_iso(item.available_at),
                    "fetched_at": utc_iso(item.fetched_at),
                    "occurrence_assigned_trading_date": (
                        occurrence_assignment.session_date.isoformat()
                        if population_is_supported
                        and occurrence_assignment.session_date is not None
                        else None
                    ),
                    "occurrence_mapping_status": (
                        occurrence_assignment.mapping_status
                        if population_is_supported
                        else None
                    ),
                    "occurrence_arrival_class": (
                        occurrence_assignment.arrival_class
                        if population_is_supported
                        else None
                    ),
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
                        (
                            canonical_representative is not None
                            and canonical_representative.occurrence_key
                            == item.occurrence_key
                        )
                        if population_is_supported
                        else None
                    ),
                    "canonical_available_at": (
                        utc_iso(canonical_representative.available_at)
                        if canonical_representative is not None
                        else None
                    ),
                    "canonical_source_observed_at": (
                        utc_iso(canonical_representative.source_observed_at)
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
            row[f"{prefix}_occurrence_count"] = (
                occurrence_count(session_date, population)
                if population_supported[population]
                else None
            )
            row[f"{prefix}_canonical_article_count"] = (
                canonical_count(session_date, population)
                if population_supported[population]
                else None
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
            if not population_supported[population]:
                split_rows.append(
                    {
                        "population": population,
                        "population_supported": False,
                        "split": split,
                        "trading_days": len(dates),
                        "trading_days_with_articles": None,
                        "coverage_percent": None,
                        "article_occurrences": None,
                        "canonical_articles": None,
                        "articles_per_trading_day": None,
                        "zero_article_trading_days": None,
                        "longest_missing_gap_sessions": None,
                    }
                )
                continue
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
                    "population_supported": True,
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
            if not population_supported[population]:
                monthly_rows.append(
                    {
                        "month": month,
                        "population": population,
                        "population_supported": False,
                        "trading_days": len(dates),
                        "trading_days_with_articles": None,
                        "coverage_percent": None,
                        "article_occurrences": None,
                        "canonical_articles": None,
                        "articles_per_trading_day": None,
                        "zero_article_trading_days": None,
                    }
                )
                continue
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
                    "population_supported": True,
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
                "source_observed_at": utc_iso(representative.source_observed_at),
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
    missing_source_observed = sum(
        item.source_observed_at is None for item in occurrences
    )
    missing_available = sum(item.available_at is None for item in occurrences)
    publication_completeness = round(
        100.0 * (total_occurrences - missing_publication) / total_occurrences, 6
    ) if total_occurrences else 0.0
    availability_completeness = round(
        100.0 * (total_occurrences - missing_available) / total_occurrences, 6
    ) if total_occurrences else 0.0
    source_observed_completeness = round(
        100.0
        * (total_occurrences - missing_source_observed)
        / total_occurrences,
        6,
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
    publisher_duplicate_candidates = _publisher_document_duplicate_candidates(
        occurrences
    )
    publisher_candidate_duplicate_rows = sum(
        len(item["canonical_article_ids"]) - 1
        for item in publisher_duplicate_candidates
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
        "availability_policy_version": NEWS_AVAILABILITY_POLICY_VERSION,
        "total_news_occurrences": total_occurrences,
        "unique_canonical_articles": unique_canonicals,
        "ogdc_linked_occurrences": sum(item.is_ogdc for item in occurrences),
        "ogdc_linked_canonical_articles": sum(
            bool(value["is_ogdc"]) for value in attributes.values()
        ),
        "energy_linked_occurrences": (
            sum(item.is_energy for item in occurrences)
            if energy_evidence_present
            else None
        ),
        "energy_linked_canonical_articles": (
            sum(bool(value["is_energy"]) for value in attributes.values())
            if energy_evidence_present
            else None
        ),
        "energy_sector_linking": (
            {
                "supported": True,
                "reason": (
                    "A checksum-bound, explicit versioned energy-sector link "
                    "contract authorizes mentioned_sectors interpretation."
                ),
                "contract": energy_link_contract.to_dict(),
            }
            if energy_link_contract is not None
            else {
                "supported": False,
                "reason": (
                    "No explicit versioned energy-sector link contract was supplied; "
                    "mentioned_sectors values are not interpreted as coverage evidence."
                ),
            }
        ),
        "duplicated_occurrences": sum(item.duplicate_of is not None for item in occurrences),
        "canonical_article_count": unique_canonicals,
        "canonical_identity": {
            "uniqueness_scope": (
                "current_exact_canonical_url_content_hash_or_title_source_time_rules"
            ),
            "unique_under_current_rules": unique_canonicals,
            "publisher_document_candidate_detection_policy": (
                "reviewed_publisher_document_url_patterns_v1"
            ),
            "publisher_document_candidate_group_count": len(
                publisher_duplicate_candidates
            ),
            "publisher_document_candidate_duplicate_rows": (
                publisher_candidate_duplicate_rows
            ),
            "publisher_document_candidate_groups": publisher_duplicate_candidates,
        },
        "publication_time": {
            "complete_count": total_occurrences - missing_publication,
            "missing_count": missing_publication,
            "completeness_percent": publication_completeness,
            "range": _date_range(item.published_at for item in occurrences),
            "calendar_overlap_with_prices": publication_overlap,
        },
        "source_observed_at": {
            "complete_count": total_occurrences - missing_source_observed,
            "missing_count": missing_source_observed,
            "completeness_percent": source_observed_completeness,
            "range": _date_range(
                item.source_observed_at for item in occurrences
            ),
            "eligibility_semantics": (
                "audit provenance only; never used for cutoff eligibility "
                "without a new versioned policy"
            ),
            "eligibility_policy_version": NEWS_AVAILABILITY_POLICY_VERSION,
            "eligible_for_feature_cutoff": False,
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
            "source_observed_at is audit-only provider provenance and is never "
            "used for eligibility under this policy version.",
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
        "availability_policy_version": NEWS_AVAILABILITY_POLICY_VERSION,
        "status": "PASS",
        "readiness_status": readiness,
        "symbol": stock.symbol,
        "total_news_occurrences": total_occurrences,
        "unique_canonical_articles": unique_canonicals,
        "canonical_uniqueness_scope": quality_report["canonical_identity"][
            "uniqueness_scope"
        ],
        "publisher_document_candidate_duplicate_groups": quality_report[
            "canonical_identity"
        ]["publisher_document_candidate_group_count"],
        "publisher_document_candidate_duplicate_rows": quality_report[
            "canonical_identity"
        ]["publisher_document_candidate_duplicate_rows"],
        "ogdc_linked_canonical_articles": quality_report[
            "ogdc_linked_canonical_articles"
        ],
        "ogdc_linked_occurrences": quality_report["ogdc_linked_occurrences"],
        "duplicated_occurrences": quality_report["duplicated_occurrences"],
        "publication_date_range": quality_report["publication_time"]["range"],
        "source_observed_at_range": quality_report["source_observed_at"]["range"],
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
                "supported": population_supported[population],
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


_SPREADSHEET_FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def _spreadsheet_safe_text(value: str) -> str:
    """Encode a CSV scalar reversibly so spreadsheet imports cannot execute it."""

    if value.startswith("'"):
        return "'" + value
    if value.startswith(_SPREADSHEET_FORMULA_PREFIXES):
        return "'" + value
    return value


def _spreadsheet_restore_text(value: str) -> str:
    if value.startswith("''"):
        return value[1:]
    if value.startswith("'") and value[1:].startswith(
        _SPREADSHEET_FORMULA_PREFIXES
    ):
        return value[1:]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _spreadsheet_safe_text(
                        json.dumps(value, ensure_ascii=False, sort_keys=True)
                        if isinstance(value, (list, tuple, dict))
                        else ""
                        if value is None
                        else str(value)
                    )
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


def _coverage_csv_rows(
    root: Path,
    filename: str,
    *,
    required_fields: set[str],
) -> list[dict[str, str]]:
    """Read one sealed CSV with a non-ambiguous, minimally complete schema."""

    path = _regular_file(
        root / filename, root=root, label=f"coverage payload {filename}"
    )
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if fields is None or len(fields) != len(set(fields)):
                raise CoverageAuditError(
                    f"Coverage semantic verification found an invalid {filename} header"
                )
            missing = sorted(required_fields - set(fields))
            if missing:
                raise CoverageAuditError(
                    f"Coverage semantic verification found missing {filename} fields: "
                    + ", ".join(missing)
                )
            rows: list[dict[str, str]] = []
            for row in reader:
                restored: dict[str, str] = {}
                for field, value in row.items():
                    if value is None:
                        raise CoverageAuditError(
                            f"Coverage semantic verification found an invalid "
                            f"{filename} row"
                        )
                    raw = _spreadsheet_restore_text(value)
                    if _spreadsheet_safe_text(raw) != value:
                        raise CoverageAuditError(
                            f"Coverage semantic verification found unsafe CSV text "
                            f"in {filename}"
                        )
                    restored[field] = raw
                rows.append(restored)
            return rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise CoverageAuditError(
            f"Coverage semantic verification could not parse {filename}"
        ) from exc


def _coverage_int(value: Any, *, field_name: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise CoverageAuditError(
            f"Coverage semantic verification found invalid {field_name}"
        ) from exc
    if parsed < 0 or str(parsed) != str(value):
        raise CoverageAuditError(
            f"Coverage semantic verification found invalid {field_name}"
        )
    return parsed


def _coverage_float(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as exc:
        raise CoverageAuditError(
            f"Coverage semantic verification found invalid {field_name}"
        ) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise CoverageAuditError(
            f"Coverage semantic verification found invalid {field_name}"
        )
    return parsed


def _coverage_bool(value: Any, *, field_name: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise CoverageAuditError(
        f"Coverage semantic verification found invalid {field_name}"
    )


def _verify_sha256_fields(value: Any, *, field_name: str) -> None:
    """Validate every explicitly named SHA-256 in a portable reference tree."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            label = f"{field_name}.{key}"
            if str(key).endswith("sha256"):
                if not isinstance(item, str) or not _SHA256.fullmatch(item):
                    raise CoverageAuditError(
                        f"Coverage semantic verification found invalid {label}"
                    )
            else:
                _verify_sha256_fields(item, field_name=label)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _verify_sha256_fields(item, field_name=f"{field_name}[{index}]")


def _verify_coverage_semantics(
    root: Path,
    *,
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    """Reconcile sealed metrics with mappings and portable input references."""

    quality = _read_json(
        root / "quality_report.json", root=root, label="coverage quality report"
    )
    news_reference = _read_json(
        root / "imported_or_acquired_news_reference.json",
        root=root,
        label="coverage news reference",
    )
    stock_reference = _read_json(
        root / "stock_artifact_reference.json",
        root=root,
        label="coverage stock reference",
    )
    assessment_reference = _read_json(
        root / "source_assessment_reference.json",
        root=root,
        label="coverage source-assessment reference",
    )
    _verify_sha256_fields(news_reference, field_name="news_reference")
    _verify_sha256_fields(stock_reference, field_name="stock_reference")
    _verify_sha256_fields(assessment_reference, field_name="source_assessment")

    config = manifest.get("config")
    if not isinstance(config, Mapping):
        raise CoverageAuditError(
            "Coverage semantic verification found an invalid configuration"
        )
    try:
        serialized_thresholds = config["readiness_thresholds"]
        if not isinstance(serialized_thresholds, Mapping):
            raise CoverageAuditError(
                "Coverage semantic verification found invalid readiness thresholds"
            )
        reconstructed_config = CoverageConfig(
            market_timezone=str(config["market_timezone"]),
            decision_cutoff=parse_cutoff(str(config["decision_cutoff"])),
            thresholds=ReadinessThresholds.from_mapping(serialized_thresholds),
        )
    except (KeyError, TypeError) as exc:
        raise CoverageAuditError(
            "Coverage semantic verification found invalid serialized configuration"
        ) from exc
    if reconstructed_config.to_dict() != dict(config):
        raise CoverageAuditError(
            "Coverage semantic verification found a non-canonical configuration"
        )
    policy_digest = _sha256_bytes(_canonical_json_bytes(config))
    if manifest.get("effective_readiness_policy_sha256") != policy_digest:
        raise CoverageAuditError(
            "Coverage semantic verification found a readiness-policy fingerprint mismatch"
        )
    expected_input_fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "news_runs": news_reference.get("news_runs"),
                "database": news_reference.get("database_fingerprints"),
                "energy_sector_link_contract": news_reference.get(
                    "energy_sector_link_contract"
                ),
                "stock": stock_reference,
                "source_assessment": assessment_reference,
                "config": config,
            }
        )
    )
    if manifest.get("input_fingerprint_sha256") != expected_input_fingerprint:
        raise CoverageAuditError(
            "Coverage semantic verification found an input fingerprint mismatch"
        )
    if news_reference.get("schema_version") != COVERAGE_SCHEMA_VERSION:
        raise CoverageAuditError(
            "Coverage semantic verification found an invalid news-reference schema"
        )
    if (
        quality.get("schema_version") != COVERAGE_SCHEMA_VERSION
        or quality.get("algorithm_version") != COVERAGE_ALGORITHM_VERSION
        or quality.get("availability_policy_version")
        != NEWS_AVAILABILITY_POLICY_VERSION
        or summary.get("availability_policy_version")
        != NEWS_AVAILABILITY_POLICY_VERSION
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found invalid quality-report provenance"
        )
    source_observation_policy = quality.get("source_observed_at")
    if (
        not isinstance(source_observation_policy, Mapping)
        or source_observation_policy.get("eligibility_policy_version")
        != NEWS_AVAILABILITY_POLICY_VERSION
        or source_observation_policy.get("eligible_for_feature_cutoff") is not False
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found invalid availability-policy evidence"
        )
    referenced_news_runs = news_reference.get("news_runs")
    if not isinstance(referenced_news_runs, list) or any(
        not isinstance(reference, Mapping)
        or reference.get("availability_policy_version")
        not in {None, NEWS_AVAILABILITY_POLICY_VERSION}
        for reference in referenced_news_runs
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found an unsupported input availability policy"
        )
    serialized_energy_contract = news_reference.get("energy_sector_link_contract")
    if serialized_energy_contract is None:
        energy_supported = False
        expected_energy_contract: dict[str, str] | None = None
    elif isinstance(serialized_energy_contract, Mapping):
        if set(serialized_energy_contract) != {
            "contract_id",
            "schema_version",
            "link_method",
            "evidence_field",
            "evidence_sha256",
        }:
            raise CoverageAuditError(
                "Coverage semantic verification found an invalid energy link contract"
            )
        try:
            parsed_energy_contract = EnergySectorLinkContract(
                **dict(serialized_energy_contract)
            )
        except TypeError as exc:
            raise CoverageAuditError(
                "Coverage semantic verification found an invalid energy link contract"
            ) from exc
        expected_energy_contract = parsed_energy_contract.to_dict()
        energy_supported = True
    else:
        raise CoverageAuditError(
            "Coverage semantic verification found an invalid energy link contract"
        )
    energy_quality = quality.get("energy_sector_linking")
    if not isinstance(energy_quality, Mapping) or energy_quality.get(
        "supported"
    ) is not energy_supported:
        raise CoverageAuditError(
            "Coverage semantic verification found energy-support contradictions"
        )
    if energy_supported:
        if energy_quality.get("contract") != expected_energy_contract:
            raise CoverageAuditError(
                "Coverage semantic verification found an energy-contract contradiction"
            )
    elif "contract" in energy_quality:
        raise CoverageAuditError(
            "Coverage semantic verification found a contract for unsupported energy data"
        )

    mappings = _coverage_csv_rows(
        root,
        "article_session_mappings.csv",
        required_fields={
            "occurrence_key",
            "ingestion_run_id",
            "article_id",
            "canonical_article_id",
            "provider",
            "source_name",
            "source_url",
            "publisher_document_key",
            "mentioned_sectors",
            "data_class",
            "population",
            "population_supported",
            "occurrence_in_population",
            "is_duplicate_occurrence",
            "published_at",
            "source_observed_at",
            "available_at",
            "fetched_at",
            "occurrence_assigned_trading_date",
            "occurrence_mapping_status",
            "occurrence_arrival_class",
            "canonical_representative_article_id",
            "canonical_representative_ingestion_run_id",
            "is_population_canonical_representative",
            "canonical_available_at",
            "canonical_source_observed_at",
            "canonical_assigned_trading_date",
            "canonical_mapping_status",
            "canonical_arrival_class",
            "content_sha256",
            "raw_artifact_reference",
        },
    )
    trading_rows = _coverage_csv_rows(
        root,
        "overlap_by_trading_day.csv",
        required_fields={
            "trading_date",
            "split",
            *{
                f"{population.removesuffix('_news')}_{kind}_count"
                for population in POPULATIONS
                for kind in ("occurrence", "canonical_article")
            },
        },
    )
    split_rows = _coverage_csv_rows(
        root,
        "split_coverage.csv",
        required_fields={
            "population",
            "population_supported",
            "split",
            "trading_days",
            "trading_days_with_articles",
            "coverage_percent",
            "article_occurrences",
            "canonical_articles",
            "articles_per_trading_day",
            "zero_article_trading_days",
            "longest_missing_gap_sessions",
        },
    )
    monthly_rows = _coverage_csv_rows(
        root,
        "overlap_by_month.csv",
        required_fields={
            "month",
            "population",
            "population_supported",
            "trading_days",
            "trading_days_with_articles",
            "coverage_percent",
            "article_occurrences",
            "canonical_articles",
            "articles_per_trading_day",
            "zero_article_trading_days",
        },
    )
    missing_rows = _coverage_csv_rows(
        root,
        "missing_intervals.csv",
        required_fields={
            "population",
            "split",
            "start_date",
            "end_date",
            "missing_trading_sessions",
        },
    )
    ogdc_rows = _coverage_csv_rows(
        root,
        "ogdc_linked_articles.csv",
        required_fields={
            "canonical_article_id",
            "representative_article_id",
            "source_name",
            "source_url",
            "published_at",
            "source_observed_at",
            "available_at",
            "assigned_trading_date",
            "mapping_status",
            "arrival_class",
            "occurrence_count",
            "sources",
            "link_basis",
            "link_evidence",
            "content_sha256",
        },
    )

    mapping_keys = [(row["occurrence_key"], row["population"]) for row in mappings]
    if len(mapping_keys) != len(set(mapping_keys)):
        raise CoverageAuditError(
            "Coverage semantic verification found duplicate mapping rows"
        )
    occurrence_keys = {key for key, _ in mapping_keys}
    canonical_ids = {row["canonical_article_id"] for row in mappings}
    expected_mapping_keys = {
        (occurrence_key, population)
        for occurrence_key in occurrence_keys
        for population in POPULATIONS
    }
    if set(mapping_keys) != expected_mapping_keys:
        raise CoverageAuditError(
            "Coverage semantic verification found incomplete population mappings"
        )
    invariant_fields = (
        "ingestion_run_id",
        "article_id",
        "canonical_article_id",
        "provider",
        "source_name",
        "source_url",
        "publisher_document_key",
        "mentioned_sectors",
        "data_class",
        "is_duplicate_occurrence",
        "published_at",
        "source_observed_at",
        "available_at",
        "fetched_at",
        "content_sha256",
        "raw_artifact_reference",
    )
    mappings_by_occurrence: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in mappings:
        mappings_by_occurrence[row["occurrence_key"]].append(row)
    for occurrence_key, rows in mappings_by_occurrence.items():
        if len(rows) != len(POPULATIONS) or any(
            len({row[field] for row in rows}) != 1 for field in invariant_fields
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found population-varying occurrence "
                f"identity for {occurrence_key}"
            )
        if not _SHA256.fullmatch(rows[0]["content_sha256"]):
            raise CoverageAuditError(
                "Coverage semantic verification found an invalid content identity"
            )
        _safe_relative(
            rows[0]["raw_artifact_reference"],
            field_name="mapping raw artifact reference",
        )
    population_support: dict[str, bool] = {}
    summary_population_support = summary.get("population_support")
    if not isinstance(summary_population_support, Mapping):
        raise CoverageAuditError(
            "Coverage semantic verification found missing population support"
        )
    for population in POPULATIONS:
        flags = {
            _coverage_bool(
                row["population_supported"], field_name="population_supported"
            )
            for row in mappings
            if row["population"] == population
        }
        declared_support = summary_population_support.get(population)
        if not isinstance(declared_support, Mapping) or not isinstance(
            declared_support.get("supported"), bool
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found invalid population support"
            )
        if len(flags) > 1 or (
            flags and flags != {declared_support["supported"]}
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found inconsistent population support"
            )
        population_support[population] = bool(declared_support["supported"])
    if not population_support["all_financial_news"] or not population_support[
        "ogdc_linked_news"
    ]:
        raise CoverageAuditError(
            "Coverage semantic verification found unsupported core populations"
        )
    if population_support["energy_sector_news"] is not energy_supported:
        raise CoverageAuditError(
            "Coverage semantic verification found energy-contract/support contradictions"
        )
    canonical_invariant_fields = (
        "canonical_representative_article_id",
        "canonical_representative_ingestion_run_id",
        "canonical_available_at",
        "canonical_source_observed_at",
        "canonical_assigned_trading_date",
        "canonical_mapping_status",
        "canonical_arrival_class",
    )
    mappings_by_population_canonical: dict[
        tuple[str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in mappings:
        mappings_by_population_canonical[
            (row["population"], row["canonical_article_id"])
        ].append(row)
    for (population, _canonical_id), rows in mappings_by_population_canonical.items():
        if any(
            len({row[field] for row in rows}) != 1
            for field in canonical_invariant_fields
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found inconsistent canonical "
                "representative evidence"
            )
        representative_flags = [
            _coverage_bool(
                row["is_population_canonical_representative"],
                field_name="is_population_canonical_representative",
            )
            for row in rows
            if row["is_population_canonical_representative"] != ""
        ]
        membership_values = [
            row["occurrence_in_population"]
            for row in rows
            if row["occurrence_in_population"] != ""
        ]
        matched_population = any(
            _coverage_bool(
                value,
                field_name="occurrence_in_population",
            )
            for value in membership_values
        )
        representative_id = rows[0]["canonical_representative_article_id"]
        if not population_support[population]:
            if representative_flags or representative_id:
                raise CoverageAuditError(
                    "Coverage semantic verification found canonical evidence for an "
                    "unsupported population"
                )
            continue
        if len(representative_flags) != len(rows) or len(membership_values) != len(
            rows
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found missing supported-population flags"
            )
        if matched_population:
            representative_rows = [
                row for row, selected in zip(rows, representative_flags, strict=True)
                if selected
            ]
            if (
                len(representative_rows) != 1
                or not representative_id
                or representative_rows[0]["article_id"] != representative_id
                or representative_rows[0]["ingestion_run_id"]
                != rows[0]["canonical_representative_ingestion_run_id"]
            ):
                raise CoverageAuditError(
                    "Coverage semantic verification found an invalid canonical "
                    "representative identity"
                )
        elif any(representative_flags) or any(
            rows[0][field] for field in canonical_invariant_fields
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found a representative outside its "
                "population"
            )
    occurrence_count = _coverage_int(
        news_reference.get("occurrence_count"), field_name="news occurrence count"
    )
    canonical_count = _coverage_int(
        news_reference.get("canonical_count"), field_name="news canonical count"
    )
    if occurrence_count != len(occurrence_keys) or canonical_count != len(canonical_ids):
        raise CoverageAuditError(
            "Coverage semantic verification found news-reference count contradictions"
        )
    for payload, label in ((summary, "summary"), (quality, "quality report")):
        if payload.get("total_news_occurrences") != occurrence_count:
            raise CoverageAuditError(
                f"Coverage semantic verification found a {label} occurrence contradiction"
            )
        unique_value = payload.get(
            "unique_canonical_articles", payload.get("canonical_article_count")
        )
        if unique_value != canonical_count:
            raise CoverageAuditError(
                f"Coverage semantic verification found a {label} canonical contradiction"
            )

    first_by_occurrence = {
        row["occurrence_key"]: row
        for row in mappings
        if row["population"] == "all_financial_news"
    }
    energy_evidence_payload: list[dict[str, Any]] = []
    for occurrence_key, row in sorted(first_by_occurrence.items()):
        sectors = _json_list(
            row["mentioned_sectors"], field_name="mapping mentioned_sectors"
        )
        energy_evidence_payload.append(
            {
                "occurrence_key": occurrence_key,
                "article_id": row["article_id"],
                "canonical_article_id": row["canonical_article_id"],
                "content_sha256": row["content_sha256"],
                "mentioned_sectors": sorted(set(sectors)),
            }
        )
    observed_energy_evidence_sha256 = _sha256_bytes(
        _canonical_json_bytes(energy_evidence_payload)
    )
    if energy_supported and (
        expected_energy_contract is None
        or expected_energy_contract["evidence_sha256"]
        != observed_energy_evidence_sha256
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found an energy evidence-digest "
            "contradiction"
        )
    candidate_groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"canonical_article_ids": set(), "source_urls": set()}
    )
    for row in first_by_occurrence.values():
        source_url = row["source_url"] or None
        expected_document_key = _publisher_document_key(source_url)
        if row["publisher_document_key"] != (expected_document_key or ""):
            raise CoverageAuditError(
                "Coverage semantic verification found an invalid publisher document key"
            )
        if expected_document_key is not None and source_url is not None:
            candidate_groups[expected_document_key]["canonical_article_ids"].add(
                row["canonical_article_id"]
            )
            candidate_groups[expected_document_key]["source_urls"].add(source_url)
    expected_candidate_groups = [
        {
            "publisher_document_key": key,
            "canonical_article_ids": sorted(values["canonical_article_ids"]),
            "source_urls": sorted(values["source_urls"]),
        }
        for key, values in sorted(candidate_groups.items())
        if len(values["canonical_article_ids"]) > 1
    ]
    expected_candidate_duplicate_rows = sum(
        len(item["canonical_article_ids"]) - 1
        for item in expected_candidate_groups
    )
    canonical_identity = quality.get("canonical_identity")
    if not isinstance(canonical_identity, Mapping) or canonical_identity != {
        "uniqueness_scope": (
            "current_exact_canonical_url_content_hash_or_title_source_time_rules"
        ),
        "unique_under_current_rules": canonical_count,
        "publisher_document_candidate_detection_policy": (
            "reviewed_publisher_document_url_patterns_v1"
        ),
        "publisher_document_candidate_group_count": len(expected_candidate_groups),
        "publisher_document_candidate_duplicate_rows": (
            expected_candidate_duplicate_rows
        ),
        "publisher_document_candidate_groups": expected_candidate_groups,
    }:
        raise CoverageAuditError(
            "Coverage semantic verification found publisher-document candidate "
            "contradictions"
        )
    if (
        summary.get("canonical_uniqueness_scope")
        != canonical_identity["uniqueness_scope"]
        or summary.get("publisher_document_candidate_duplicate_groups")
        != len(expected_candidate_groups)
        or summary.get("publisher_document_candidate_duplicate_rows")
        != expected_candidate_duplicate_rows
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found summary candidate contradictions"
        )
    duplicate_count = sum(
        _coverage_bool(
            row["is_duplicate_occurrence"],
            field_name="is_duplicate_occurrence",
        )
        for row in first_by_occurrence.values()
    )
    if summary.get("duplicated_occurrences") != duplicate_count or quality.get(
        "duplicated_occurrences"
    ) != duplicate_count:
        raise CoverageAuditError(
            "Coverage semantic verification found a duplicate-count contradiction"
        )
    published_values = [
        parse_timestamp(
            row.get("published_at"), field_name="mapping published_at"
        )
        for row in first_by_occurrence.values()
    ]
    available_values = [
        parse_timestamp(
            row.get("available_at"), field_name="mapping available_at"
        )
        for row in first_by_occurrence.values()
    ]
    source_observed_values = [
        parse_timestamp(
            row.get("source_observed_at"), field_name="mapping source_observed_at"
        )
        for row in first_by_occurrence.values()
    ]
    expected_publication_complete = sum(value is not None for value in published_values)
    expected_available_complete = sum(value is not None for value in available_values)
    publication_quality = quality.get("publication_time")
    availability_quality = quality.get("available_at")
    source_observed_quality = quality.get("source_observed_at")
    if not isinstance(publication_quality, Mapping) or not isinstance(
        availability_quality, Mapping
    ) or not isinstance(source_observed_quality, Mapping):
        raise CoverageAuditError(
            "Coverage semantic verification found incomplete timestamp quality evidence"
        )
    expected_publication_completeness = (
        round(100.0 * expected_publication_complete / occurrence_count, 6)
        if occurrence_count
        else 0.0
    )
    expected_availability_completeness = (
        round(100.0 * expected_available_complete / occurrence_count, 6)
        if occurrence_count
        else 0.0
    )
    expected_source_observed_complete = sum(
        value is not None for value in source_observed_values
    )
    expected_source_observed_completeness = (
        round(100.0 * expected_source_observed_complete / occurrence_count, 6)
        if occurrence_count
        else 0.0
    )
    if (
        publication_quality.get("complete_count") != expected_publication_complete
        or publication_quality.get("missing_count")
        != occurrence_count - expected_publication_complete
        or publication_quality.get("completeness_percent")
        != expected_publication_completeness
        or publication_quality.get("range") != _date_range(published_values)
        or availability_quality.get("complete_count") != expected_available_complete
        or availability_quality.get("missing_count")
        != occurrence_count - expected_available_complete
        or availability_quality.get("completeness_percent")
        != expected_availability_completeness
        or availability_quality.get("range") != _date_range(available_values)
        or source_observed_quality.get("complete_count")
        != expected_source_observed_complete
        or source_observed_quality.get("missing_count")
        != occurrence_count - expected_source_observed_complete
        or source_observed_quality.get("completeness_percent")
        != expected_source_observed_completeness
        or source_observed_quality.get("range")
        != _date_range(source_observed_values)
        or summary.get("publication_date_range") != _date_range(published_values)
        or summary.get("available_at_range") != _date_range(available_values)
        or summary.get("source_observed_at_range")
        != _date_range(source_observed_values)
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found timestamp metric contradictions"
        )

    ogdc_mappings = [
        row
        for row in mappings
        if row["population"] == "ogdc_linked_news"
        and _coverage_bool(
            row["occurrence_in_population"], field_name="occurrence_in_population"
        )
    ]
    ogdc_canonical_ids = {row["canonical_article_id"] for row in ogdc_mappings}
    if (
        summary.get("ogdc_linked_occurrences") != len(ogdc_mappings)
        or summary.get("ogdc_linked_canonical_articles") != len(ogdc_canonical_ids)
        or quality.get("ogdc_linked_occurrences") != len(ogdc_mappings)
        or quality.get("ogdc_linked_canonical_articles")
        != len(ogdc_canonical_ids)
        or {row["canonical_article_id"] for row in ogdc_rows}
        != ogdc_canonical_ids
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found an OGDC-count contradiction"
        )
    ogdc_occurrences_by_canonical = Counter(
        row["canonical_article_id"] for row in ogdc_mappings
    )
    for row in ogdc_rows:
        try:
            link_evidence = json.loads(row["link_evidence"])
        except json.JSONDecodeError as exc:
            raise CoverageAuditError(
                "Coverage semantic verification found invalid OGDC link evidence"
            ) from exc
        if (
            row["link_basis"] != "checksum_bound_article_entity_links"
            or not isinstance(link_evidence, list)
            or not link_evidence
            or _coverage_int(
                row["occurrence_count"], field_name="OGDC occurrence count"
            )
            != ogdc_occurrences_by_canonical[row["canonical_article_id"]]
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found invalid OGDC article evidence"
            )
        canonical_mapping_rows = mappings_by_population_canonical[
            ("ogdc_linked_news", row["canonical_article_id"])
        ]
        representative_rows = [
            mapping
            for mapping in canonical_mapping_rows
            if _coverage_bool(
                mapping["is_population_canonical_representative"],
                field_name="OGDC canonical representative",
            )
        ]
        if len(representative_rows) != 1:
            raise CoverageAuditError(
                "Coverage semantic verification found invalid OGDC representative evidence"
            )
        representative = representative_rows[0]
        try:
            sources = json.loads(row["sources"])
        except json.JSONDecodeError as exc:
            raise CoverageAuditError(
                "Coverage semantic verification found invalid OGDC source evidence"
            ) from exc
        expected_sources = sorted(
            {
                mapping["source_name"]
                for mapping in canonical_mapping_rows
                if _coverage_bool(
                    mapping["occurrence_in_population"],
                    field_name="OGDC population membership",
                )
            }
        )
        if (
            sources != expected_sources
            or row["representative_article_id"] != representative["article_id"]
            or row["source_name"] != representative["source_name"]
            or row["source_url"] != representative["source_url"]
            or row["published_at"] != representative["published_at"]
            or row["source_observed_at"] != representative["source_observed_at"]
            or row["available_at"] != representative["available_at"]
            or row["assigned_trading_date"]
            != representative["canonical_assigned_trading_date"]
            or row["mapping_status"] != representative["canonical_mapping_status"]
            or row["arrival_class"] != representative["canonical_arrival_class"]
            or row["content_sha256"] != representative["content_sha256"]
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found OGDC/mapping contradictions"
            )

    trading_dates: list[date] = []
    trading_by_date: dict[str, dict[str, str]] = {}
    for row in trading_rows:
        parsed_date = _parse_date(
            row["trading_date"], field_name="coverage trading date"
        )
        if row["trading_date"] in trading_by_date:
            raise CoverageAuditError(
                "Coverage semantic verification found duplicate trading dates"
            )
        trading_dates.append(parsed_date)
        trading_by_date[row["trading_date"]] = row
    if not trading_dates or trading_dates != sorted(trading_dates):
        raise CoverageAuditError(
            "Coverage semantic verification found unordered trading dates"
        )
    claimed_range = stock_reference.get("price_date_range")
    observed_range = {
        "start": trading_dates[0].isoformat(),
        "end": trading_dates[-1].isoformat(),
    }
    if claimed_range != observed_range or summary.get("stock_date_range") != observed_range:
        raise CoverageAuditError(
            "Coverage semantic verification found a stock date-range contradiction"
        )
    if "price_trading_day_count" in stock_reference and stock_reference.get(
        "price_trading_day_count"
    ) != len(trading_rows):
        raise CoverageAuditError(
            "Coverage semantic verification found a stock row-count contradiction"
        )
    symbol = stock_reference.get("symbol")
    if not isinstance(symbol, str) or not symbol or summary.get("symbol") != symbol:
        raise CoverageAuditError(
            "Coverage semantic verification found a stock symbol contradiction"
        )
    if (
        "training_input_snapshot_sha256" in stock_reference
        and stock_reference.get("training_input_snapshot_sha256")
        != stock_reference.get("price_artifact_sha256")
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found mismatched stock/training evidence"
        )
    split_counts = stock_reference.get("split_counts")
    split_ranges = stock_reference.get("split_ranges")
    if split_counts is not None or split_ranges is not None:
        if not isinstance(split_counts, Mapping) or not isinstance(
            split_ranges, Mapping
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found incomplete split references"
            )
        for split in SPLITS:
            selected_dates = [
                row["trading_date"] for row in trading_rows if row["split"] == split
            ]
            if (
                split_counts.get(split) != len(selected_dates)
                or not selected_dates
                or split_ranges.get(split)
                != {"start": selected_dates[0], "end": selected_dates[-1]}
            ):
                raise CoverageAuditError(
                    "Coverage semantic verification found split-reference contradictions"
                )

    reconstructed_split_dates = {
        split: tuple(
            _parse_date(row["trading_date"], field_name=f"{split} trading date")
            for row in trading_rows
            if row["split"] == split
        )
        for split in SPLITS
    }
    reconstructed_stock = StockEvidence(
        symbol=symbol,
        trading_dates=tuple(trading_dates),
        split_dates=reconstructed_split_dates,
        reference=stock_reference,
    )
    cutoffs, _ = _session_cutoffs(
        reconstructed_stock.trading_dates, reconstructed_config
    )
    for row, cutoff in zip(trading_rows, cutoffs, strict=True):
        if row.get("decision_cutoff_utc") != utc_iso(cutoff):
            raise CoverageAuditError(
                "Coverage semantic verification found a decision-cutoff contradiction"
            )

    for row in mappings:
        population = row["population"]
        available_at = parse_timestamp(
            row["available_at"], field_name="mapping available_at"
        )
        fetched_at = parse_timestamp(
            row["fetched_at"], field_name="mapping fetched_at"
        )
        published_at = parse_timestamp(
            row["published_at"], field_name="mapping published_at"
        )
        source_observed_at = parse_timestamp(
            row["source_observed_at"], field_name="mapping source_observed_at"
        )
        if available_at is None or fetched_at is None or available_at < fetched_at:
            raise CoverageAuditError(
                "Coverage semantic verification found invalid availability evidence"
            )
        if published_at is not None and available_at < published_at:
            raise CoverageAuditError(
                "Coverage semantic verification found backdated publication eligibility"
            )
        if source_observed_at is not None and available_at < source_observed_at:
            raise CoverageAuditError(
                "Coverage semantic verification found backdated source-observation eligibility"
            )
        if not population_support[population]:
            continue
        occurrence_assignment = assign_to_session(
            available_at, reconstructed_stock, reconstructed_config
        )
        expected_occurrence_date = (
            occurrence_assignment.session_date.isoformat()
            if occurrence_assignment.session_date is not None
            else ""
        )
        if (
            row["occurrence_assigned_trading_date"] != expected_occurrence_date
            or row["occurrence_mapping_status"]
            != occurrence_assignment.mapping_status
            or row["occurrence_arrival_class"]
            != occurrence_assignment.arrival_class
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found a backdated or invalid "
                "occurrence assignment"
            )
        canonical_available_at = parse_timestamp(
            row["canonical_available_at"],
            field_name="mapping canonical_available_at",
        )
        if canonical_available_at is None:
            if any(
                row[field]
                for field in (
                    "canonical_representative_article_id",
                    "canonical_representative_ingestion_run_id",
                    "canonical_source_observed_at",
                    "canonical_assigned_trading_date",
                    "canonical_mapping_status",
                    "canonical_arrival_class",
                )
            ):
                raise CoverageAuditError(
                    "Coverage semantic verification found incomplete canonical evidence"
                )
            continue
        canonical_assignment = assign_to_session(
            canonical_available_at, reconstructed_stock, reconstructed_config
        )
        expected_canonical_date = (
            canonical_assignment.session_date.isoformat()
            if canonical_assignment.session_date is not None
            else ""
        )
        if (
            row["canonical_assigned_trading_date"] != expected_canonical_date
            or row["canonical_mapping_status"] != canonical_assignment.mapping_status
            or row["canonical_arrival_class"] != canonical_assignment.arrival_class
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found a backdated or invalid "
                "canonical assignment"
            )

    news_runs = news_reference.get("news_runs")
    database_fingerprints = news_reference.get("database_fingerprints")
    if not isinstance(news_runs, list) or not all(
        isinstance(item, Mapping) and isinstance(item.get("run_id"), str)
        for item in news_runs
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found invalid news-run references"
        )
    referenced_run_ids = {str(item["run_id"]) for item in news_runs}
    mapped_run_ids = {row["ingestion_run_id"] for row in mappings}
    if not mapped_run_ids.issubset(referenced_run_ids):
        raise CoverageAuditError(
            "Coverage semantic verification found unreferenced news mappings"
        )
    if all("occurrence_count" in reference for reference in news_runs):
        mapped_occurrences_by_run = Counter(
            row["ingestion_run_id"] for row in first_by_occurrence.values()
        )
        if any(
            _coverage_int(
                reference["occurrence_count"],
                field_name="news-run occurrence count",
            )
            != mapped_occurrences_by_run[str(reference["run_id"])]
            for reference in news_runs
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found news-run count contradictions"
            )
    expected_input_source_evidence = [
        {
            "run_id": reference["run_id"],
            "provider": reference.get("provider"),
            "provider_mode": reference.get("provider_mode"),
            "licensing_usage_note": reference.get("licensing_usage_note"),
            "source_permission_decision": reference.get(
                "source_permission_decision"
            ),
        }
        for reference in news_runs
    ]
    if assessment_reference.get(
        "input_source_evidence"
    ) != expected_input_source_evidence:
        raise CoverageAuditError(
            "Coverage semantic verification found source-reference contradictions"
        )
    if not isinstance(database_fingerprints, Mapping):
        raise CoverageAuditError(
            "Coverage semantic verification found invalid database fingerprints"
        )
    database_files = database_fingerprints.get("files")
    if database_fingerprints.get("provided") is True and (
        not isinstance(database_files, list)
        or not any(
            isinstance(item, Mapping)
            and item.get("label") == "database"
            and isinstance(item.get("size_bytes"), int)
            and not isinstance(item.get("size_bytes"), bool)
            and item["size_bytes"] >= 0
            for item in database_files
        )
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found incomplete database evidence"
        )
    incomplete_run_ids = sorted(
        str(reference["run_id"])
        for reference in news_runs
        if not bool(reference.get("ingestion_complete", False))
    )
    expected_ingestion_evidence = {
        "passed": not incomplete_run_ids,
        "incomplete_run_ids": incomplete_run_ids,
        "provider_error_count": sum(
            _coverage_int(
                reference.get("provider_error_count", 0),
                field_name="provider error count",
            )
            for reference in news_runs
        ),
        "rejected_record_count": sum(
            _coverage_int(
                reference.get("rejected_record_count", 0),
                field_name="rejected record count",
            )
            for reference in news_runs
        ),
    }
    if quality.get("ingestion_completeness") != expected_ingestion_evidence:
        raise CoverageAuditError(
            "Coverage semantic verification found ingestion-evidence contradictions"
        )
    database_fingerprinted = bool(
        database_fingerprints.get("provided")
        and isinstance(database_files, list)
        and any(
            isinstance(item, Mapping) and item.get("label") == "database"
            for item in database_files
        )
    )
    unsealed_run_ids = sorted(
        str(reference["run_id"])
        for reference in news_runs
        if reference.get("verification_basis")
        != "CURRENT_COMPLETION_REPLAY_AND_DATABASE_SEAL"
    )
    database_bound = database_fingerprinted and not unsealed_run_ids
    expected_database_evidence = {
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
    if quality.get("database_provenance") != expected_database_evidence:
        raise CoverageAuditError(
            "Coverage semantic verification found database-evidence contradictions"
        )
    permission_decisions = [
        reference.get("source_permission_decision") for reference in news_runs
    ]
    if all(isinstance(decision, Mapping) for decision in permission_decisions):
        if not all(
            isinstance(decision.get("passed"), bool)
            and isinstance(decision.get("status"), str)
            and isinstance(decision.get("reason"), str)
            for decision in permission_decisions
            if isinstance(decision, Mapping)
        ):
            raise CoverageAuditError(
                "Coverage semantic verification found invalid permission decisions"
            )
        expected_permission_notes = sorted(
            f"{reference['run_id']}: "
            f"{reference['source_permission_decision']['status']}: "
            f"{reference['source_permission_decision']['reason']}"
            for reference in news_runs
            if not reference["source_permission_decision"].get("passed", False)
        )
        expected_permission_evidence = {
            "passed": not expected_permission_notes,
            "review_required": bool(expected_permission_notes),
            "notes": expected_permission_notes,
        }
        if quality.get("source_permission_evidence") != expected_permission_evidence:
            raise CoverageAuditError(
                "Coverage semantic verification found permission-evidence contradictions"
            )

    for population in POPULATIONS:
        prefix = population.removesuffix("_news")
        if not population_support[population]:
            unsupported_mapping_fields = (
                "occurrence_in_population",
                "occurrence_assigned_trading_date",
                "occurrence_mapping_status",
                "occurrence_arrival_class",
                "canonical_representative_article_id",
                "canonical_representative_ingestion_run_id",
                "is_population_canonical_representative",
                "canonical_available_at",
                "canonical_source_observed_at",
                "canonical_assigned_trading_date",
                "canonical_mapping_status",
                "canonical_arrival_class",
            )
            if any(
                row[field] != ""
                for row in mappings
                if row["population"] == population
                for field in unsupported_mapping_fields
            ):
                raise CoverageAuditError(
                    "Coverage semantic verification found measured values for an "
                    "unsupported population"
                )
            for row in trading_rows:
                if (
                    row[f"{prefix}_occurrence_count"] != ""
                    or row[f"{prefix}_canonical_article_count"] != ""
                ):
                    raise CoverageAuditError(
                        "Coverage semantic verification found trading metrics for an "
                        "unsupported population"
                    )
            continue
        occurrence_by_date = Counter(
            row["occurrence_assigned_trading_date"]
            for row in mappings
            if row["population"] == population
            and _coverage_bool(
                row["occurrence_in_population"],
                field_name="occurrence_in_population",
            )
            and row["occurrence_assigned_trading_date"]
        )
        canonical_by_date = Counter(
            row["canonical_assigned_trading_date"]
            for row in mappings
            if row["population"] == population
            and _coverage_bool(
                row["is_population_canonical_representative"],
                field_name="is_population_canonical_representative",
            )
            and row["canonical_assigned_trading_date"]
        )
        for trading_date, row in trading_by_date.items():
            observed_occurrences = _coverage_int(
                row[f"{prefix}_occurrence_count"],
                field_name=f"{population} trading occurrence count",
            )
            observed_canonicals = _coverage_int(
                row[f"{prefix}_canonical_article_count"],
                field_name=f"{population} trading canonical count",
            )
            if (
                observed_occurrences != occurrence_by_date[trading_date]
                or observed_canonicals != canonical_by_date[trading_date]
            ):
                raise CoverageAuditError(
                    "Coverage semantic verification found trading-day/mapping contradictions"
                )

    split_lookup: dict[tuple[str, str], dict[str, str]] = {}
    for row in split_rows:
        key = (row["population"], row["split"])
        if key in split_lookup:
            raise CoverageAuditError(
                "Coverage semantic verification found duplicate split rows"
            )
        split_lookup[key] = row
    expected_split_keys = {
        (population, split)
        for population in POPULATIONS
        for split in (*SPLITS, "full_price_period")
    }
    if set(split_lookup) != expected_split_keys:
        raise CoverageAuditError(
            "Coverage semantic verification found incomplete split coverage"
        )

    def rows_for_split(split: str) -> list[dict[str, str]]:
        if split == "full_price_period":
            return trading_rows
        return [row for row in trading_rows if row["split"] == split]

    for population, split in sorted(expected_split_keys):
        rows = rows_for_split(split)
        prefix = population.removesuffix("_news")
        expected = split_lookup[(population, split)]
        supported = population_support[population]
        if _coverage_bool(
            expected["population_supported"],
            field_name="split population_supported",
        ) is not supported:
            raise CoverageAuditError(
                "Coverage semantic verification found split support contradictions"
            )
        if not supported:
            if _coverage_int(
                expected["trading_days"], field_name="unsupported split trading days"
            ) != len(rows) or any(
                expected[field] != ""
                for field in (
                    "trading_days_with_articles",
                    "coverage_percent",
                    "article_occurrences",
                    "canonical_articles",
                    "articles_per_trading_day",
                    "zero_article_trading_days",
                    "longest_missing_gap_sessions",
                )
            ):
                raise CoverageAuditError(
                    "Coverage semantic verification found measured split metrics for "
                    "an unsupported population"
                )
            if split == "full_price_period":
                full = summary.get("full_price_coverage", {}).get(population, {})
                if full != {
                    "supported": False,
                    "trading_days_with_articles": None,
                    "coverage_percent": None,
                }:
                    raise CoverageAuditError(
                        "Coverage semantic verification found unsupported full-period "
                        "summary metrics"
                    )
            elif summary.get("split_coverage", {}).get(split, {}).get(
                population
            ) is not None:
                raise CoverageAuditError(
                    "Coverage semantic verification found unsupported split summary metrics"
                )
            continue
        canonical_values = [
            _coverage_int(
                row[f"{prefix}_canonical_article_count"],
                field_name=f"{population} canonical count",
            )
            for row in rows
        ]
        occurrence_values = [
            _coverage_int(
                row[f"{prefix}_occurrence_count"],
                field_name=f"{population} occurrence count",
            )
            for row in rows
        ]
        days = len(rows)
        days_with = sum(value > 0 for value in canonical_values)
        expected_coverage = round(100.0 * days_with / days, 6) if days else 0.0
        expected_per_day = round(sum(canonical_values) / days, 8) if days else 0.0
        observed = (
            _coverage_int(expected["trading_days"], field_name="split trading days"),
            _coverage_int(
                expected["trading_days_with_articles"],
                field_name="split covered days",
            ),
            _coverage_float(expected["coverage_percent"], field_name="split coverage"),
            _coverage_int(
                expected["article_occurrences"],
                field_name="split occurrences",
            ),
            _coverage_int(
                expected["canonical_articles"], field_name="split canonicals"
            ),
            _coverage_float(
                expected["articles_per_trading_day"],
                field_name="split articles per day",
            ),
            _coverage_int(
                expected["zero_article_trading_days"],
                field_name="split zero days",
            ),
            _coverage_int(
                expected["longest_missing_gap_sessions"],
                field_name="split longest missing gap",
            ),
        )
        selected_dates = tuple(
            _parse_date(row["trading_date"], field_name="split trading date")
            for row in rows
        )
        covered_dates = {
            selected_dates[index]
            for index, value in enumerate(canonical_values)
            if value > 0
        }
        calculated = (
            days,
            days_with,
            expected_coverage,
            sum(occurrence_values),
            sum(canonical_values),
            expected_per_day,
            days - days_with,
            _longest_gap(selected_dates, covered_dates),
        )
        if observed != calculated:
            raise CoverageAuditError(
                "Coverage semantic verification found split/trading-day contradictions"
            )
        if split == "full_price_period":
            full = summary.get("full_price_coverage", {}).get(population, {})
            if full not in (
                {
                    "trading_days_with_articles": days_with,
                    "coverage_percent": expected_coverage,
                },
                {
                    "supported": True,
                    "trading_days_with_articles": days_with,
                    "coverage_percent": expected_coverage,
                },
            ):
                raise CoverageAuditError(
                    "Coverage semantic verification found a summary/full-period contradiction"
                )
        else:
            if summary.get("split_coverage", {}).get(split, {}).get(
                population
            ) != expected_coverage:
                raise CoverageAuditError(
                    "Coverage semantic verification found a summary/split contradiction"
                )

    expected_month_keys = {
        (value.strftime("%Y-%m"), population)
        for value in trading_dates
        for population in POPULATIONS
    }
    month_keys = [(row["month"], row["population"]) for row in monthly_rows]
    if len(month_keys) != len(set(month_keys)) or set(month_keys) != expected_month_keys:
        raise CoverageAuditError(
            "Coverage semantic verification found incomplete monthly coverage"
        )

    for row in monthly_rows:
        population = row["population"]
        supported = population_support[population]
        if _coverage_bool(
            row["population_supported"],
            field_name="monthly population_supported",
        ) is not supported:
            raise CoverageAuditError(
                "Coverage semantic verification found monthly support contradictions"
            )
        selected_rows = [
            trading
            for trading in trading_rows
            if trading["trading_date"].startswith(f"{row['month']}-")
        ]
        if _coverage_int(
            row["trading_days"], field_name="monthly trading days"
        ) != len(selected_rows):
            raise CoverageAuditError(
                "Coverage semantic verification found monthly trading-day contradictions"
            )
        measured_fields = (
            "trading_days_with_articles",
            "coverage_percent",
            "article_occurrences",
            "canonical_articles",
            "articles_per_trading_day",
            "zero_article_trading_days",
        )
        if not supported:
            if any(row[field] != "" for field in measured_fields):
                raise CoverageAuditError(
                    "Coverage semantic verification found monthly metrics for an "
                    "unsupported population"
                )
            continue
        prefix = population.removesuffix("_news")
        canonical_values = [
            _coverage_int(
                trading[f"{prefix}_canonical_article_count"],
                field_name="monthly canonical count",
            )
            for trading in selected_rows
        ]
        occurrence_values = [
            _coverage_int(
                trading[f"{prefix}_occurrence_count"],
                field_name="monthly occurrence count",
            )
            for trading in selected_rows
        ]
        days = len(selected_rows)
        days_with = sum(value > 0 for value in canonical_values)
        observed_month = (
            _coverage_int(
                row["trading_days_with_articles"],
                field_name="monthly covered days",
            ),
            _coverage_float(
                row["coverage_percent"], field_name="monthly coverage"
            ),
            _coverage_int(
                row["article_occurrences"], field_name="monthly occurrences"
            ),
            _coverage_int(
                row["canonical_articles"], field_name="monthly canonicals"
            ),
            _coverage_float(
                row["articles_per_trading_day"],
                field_name="monthly articles per day",
            ),
            _coverage_int(
                row["zero_article_trading_days"],
                field_name="monthly zero days",
            ),
        )
        calculated_month = (
            days_with,
            round(100.0 * days_with / days, 6),
            sum(occurrence_values),
            sum(canonical_values),
            round(sum(canonical_values) / days, 8),
            days - days_with,
        )
        if observed_month != calculated_month:
            raise CoverageAuditError(
                "Coverage semantic verification found monthly/trading-day contradictions"
            )

    normalized_missing_rows: list[dict[str, Any]] = []
    for row in missing_rows:
        if row["population"] not in POPULATIONS or row["split"] not in {
            *SPLITS,
            "full_price_period",
        }:
            raise CoverageAuditError(
                "Coverage semantic verification found an invalid missing interval"
            )
        start = _parse_date(row["start_date"], field_name="missing interval start")
        end = _parse_date(row["end_date"], field_name="missing interval end")
        interval_size = _coverage_int(
            row["missing_trading_sessions"],
            field_name="missing interval sessions",
        )
        if start > end or interval_size < 1:
            raise CoverageAuditError(
                "Coverage semantic verification found an invalid missing interval"
            )
        if not population_support[row["population"]]:
            raise CoverageAuditError(
                "Coverage semantic verification found missing intervals for an "
                "unsupported population"
            )
        normalized_missing_rows.append(
            {
                "population": row["population"],
                "split": row["split"],
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "missing_trading_sessions": interval_size,
            }
        )
    expected_missing_rows: list[dict[str, Any]] = []
    for population in POPULATIONS:
        if not population_support[population]:
            continue
        prefix = population.removesuffix("_news")
        for split in (*SPLITS, "full_price_period"):
            selected = rows_for_split(split)
            selected_dates = tuple(
                _parse_date(row["trading_date"], field_name="missing trading date")
                for row in selected
            )
            covered = {
                selected_dates[index]
                for index, row in enumerate(selected)
                if _coverage_int(
                    row[f"{prefix}_canonical_article_count"],
                    field_name="missing canonical count",
                )
                > 0
            }
            expected_missing_rows.extend(
                _missing_intervals(
                    selected_dates,
                    covered,
                    population=population,
                    split=split,
                )
            )

    def missing_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["population"],
            row["split"],
            row["start_date"],
            row["end_date"],
        )
    if sorted(normalized_missing_rows, key=missing_sort_key) != sorted(
        expected_missing_rows, key=missing_sort_key
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found missing-interval contradictions"
        )

    readiness_rows: list[dict[str, Any]] = []
    for (population, split), row in split_lookup.items():
        if not population_support[population]:
            continue
        readiness_rows.append(
            {
                "population": population,
                "split": split,
                "trading_days_with_articles": _coverage_int(
                    row["trading_days_with_articles"],
                    field_name="readiness covered days",
                ),
                "coverage_percent": _coverage_float(
                    row["coverage_percent"], field_name="readiness coverage"
                ),
                "longest_missing_gap_sessions": _coverage_int(
                    row["longest_missing_gap_sessions"],
                    field_name="readiness missing gap",
                ),
            }
        )
    publication_overlap = summary.get("publication_calendar_overlap_with_prices")
    if not isinstance(publication_overlap, Mapping):
        raise CoverageAuditError(
            "Coverage semantic verification found invalid publication overlap evidence"
        )
    recalculated_readiness, _ = calculate_readiness(
        readiness_rows,
        thresholds=reconstructed_config.thresholds,
        publication_completeness_percent=expected_publication_completeness,
        available_at_completeness_percent=expected_availability_completeness,
        distinct_sources=len(
            {row["source_name"] for row in first_by_occurrence.values()}
        ),
        publication_overlap_exists=publication_overlap.get("start") is not None,
    )
    if recalculated_readiness == "READY_FOR_AGGREGATION" and any(
        not bool(quality.get(field, {}).get("passed", False))
        for field in (
            "source_permission_evidence",
            "ingestion_completeness",
            "database_provenance",
        )
    ):
        recalculated_readiness = "PARTIALLY_COVERED"
    if summary.get("readiness_status") != recalculated_readiness:
        raise CoverageAuditError(
            "Coverage semantic verification found a readiness-status contradiction"
        )

    if (
        summary.get("readiness_status") != quality.get("readiness_status")
        or summary.get("readiness_evaluation") != quality.get("readiness_evaluation")
        or summary.get("readiness_thresholds") != quality.get("readiness_thresholds")
        or summary.get("source_count")
        != len({row["source_name"] for row in first_by_occurrence.values()})
    ):
        raise CoverageAuditError(
            "Coverage semantic verification found summary/quality contradictions"
        )


def verify_coverage_pack(run_directory: str | Path) -> CoverageRunVerification:
    """Reconcile one sealed pack without claiming its source files were reopened.

    This mode proves byte integrity and internal semantic consistency.  Call
    :func:`verify_coverage_run` when the original news, SQLite, price, split,
    assessment, and permission evidence is available for a source-bound replay.
    """

    supplied = Path(run_directory)
    try:
        metadata = supplied.lstat()
    except OSError as exc:
        raise CoverageAuditError("Coverage run directory does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise CoverageAuditError("Coverage run root must be a real directory")
    root = supplied.resolve(strict=True)
    return _verify_coverage_root(root, expected_run_id=root.name)


def _config_from_coverage_manifest(manifest: Mapping[str, Any]) -> CoverageConfig:
    serialized = manifest.get("config")
    if not isinstance(serialized, Mapping):
        raise CoverageAuditError("Coverage manifest configuration is invalid")
    thresholds = serialized.get("readiness_thresholds")
    if not isinstance(thresholds, Mapping):
        raise CoverageAuditError("Coverage manifest readiness configuration is invalid")
    try:
        config = CoverageConfig(
            market_timezone=str(serialized["market_timezone"]),
            decision_cutoff=parse_cutoff(str(serialized["decision_cutoff"])),
            thresholds=ReadinessThresholds.from_mapping(thresholds),
        )
    except (KeyError, TypeError) as exc:
        raise CoverageAuditError("Coverage manifest configuration is invalid") from exc
    if config.to_dict() != dict(serialized):
        raise CoverageAuditError("Coverage manifest configuration is not canonical")
    return config


def _coverage_csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _assert_source_bound_csv(
    root: Path,
    filename: str,
    expected_rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    observed = _coverage_csv_rows(
        root,
        filename,
        required_fields=set(fields),
    )
    if observed and tuple(observed[0]) != tuple(fields):
        raise CoverageAuditError(
            f"Source-bound verification found a non-canonical {filename} schema"
        )
    expected = [
        {field: _coverage_csv_scalar(row.get(field)) for field in fields}
        for row in expected_rows
    ]
    if observed != expected:
        raise CoverageAuditError(
            f"Source-bound verification found {filename} differs from source replay"
        )


def verify_coverage_run(
    run_directory: str | Path,
    *,
    news_run_directories: Iterable[str | Path],
    vertical_slice_directory: str | Path,
    model_training_directory: str | Path,
    source_assessment_path: str | Path,
    database_path: str | Path | None = None,
    source_permission_policy_path: str | Path | None = None,
    energy_sector_link_contract_path: str | Path | None = None,
    real_mode: bool = True,
    allow_legacy_unsealed: bool = False,
) -> CoverageRunVerification:
    """Verify a pack and replay it from the exact referenced source evidence."""

    pack_result = verify_coverage_pack(run_directory)
    root = Path(run_directory).resolve(strict=True)
    manifest = _read_json(
        root / _COVERAGE_MANIFEST,
        root=root,
        label="coverage run manifest",
    )
    config = _config_from_coverage_manifest(manifest)
    actual_news_reference = _read_json(
        root / "imported_or_acquired_news_reference.json",
        root=root,
        label="coverage news reference",
    )
    actual_assessment_reference = _read_json(
        root / "source_assessment_reference.json",
        root=root,
        label="coverage source-assessment reference",
    )
    sealed_database = actual_news_reference.get("database_fingerprints")
    if (
        isinstance(sealed_database, Mapping)
        and sealed_database.get("provided") is True
        and database_path is None
    ):
        raise CoverageAuditError(
            "Source-bound verification requires the referenced SQLite database"
        )
    sealed_permission_policy = actual_assessment_reference.get(
        "permission_policy"
    )
    if sealed_permission_policy and source_permission_policy_path is None:
        raise CoverageAuditError(
            "Source-bound verification requires the referenced permission policy"
        )
    serialized_energy_contract = actual_news_reference.get(
        "energy_sector_link_contract"
    )
    if serialized_energy_contract is not None and energy_sector_link_contract_path is None:
        raise CoverageAuditError(
            "Source-bound verification requires the external energy-sector link contract"
        )
    if serialized_energy_contract is None and energy_sector_link_contract_path is not None:
        raise CoverageAuditError(
            "An energy-sector link contract was supplied for an unsupported sealed population"
        )
    assessment_candidate = Path(source_assessment_path)
    if assessment_candidate.is_symlink():
        raise CoverageAuditError(
            "Historical-news source assessment must not be a symlink"
        )
    assessment_path = assessment_candidate.resolve()
    permission_policy = (
        load_source_permission_policy(
            source_permission_policy_path,
            assessment_path,
        )
        if source_permission_policy_path is not None
        else None
    )
    run_directories = tuple(news_run_directories)
    news = load_news_evidence(
        run_directories,
        real_mode=real_mode,
        database_path=database_path,
        allow_legacy_unsealed=allow_legacy_unsealed,
        source_permission_policy=permission_policy,
    )
    stock = load_stock_evidence(
        vertical_slice_directory,
        model_training_directory,
        symbol="OGDC",
    )

    if serialized_energy_contract is not None:
        contract_candidate = Path(str(energy_sector_link_contract_path))
        if contract_candidate.is_symlink():
            raise CoverageAuditError(
                "Energy-sector link contract must not be a symlink"
            )
        external_energy_contract = _read_json(
            contract_candidate.resolve(),
            label="energy-sector link contract",
        )
        if (
            not isinstance(serialized_energy_contract, Mapping)
            or external_energy_contract != serialized_energy_contract
        ):
            raise CoverageAuditError("Coverage energy-sector contract is invalid")
        try:
            energy_contract = EnergySectorLinkContract(
                **dict(external_energy_contract)
            )
        except TypeError as exc:
            raise CoverageAuditError(
                "Coverage energy-sector contract is invalid"
            ) from exc
        news = replace(news, energy_sector_link_contract=energy_contract)

    expected_news_reference = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "news_runs": list(news.run_references),
        "database_fingerprints": news.database_fingerprints,
        "energy_sector_link_contract": (
            news.energy_sector_link_contract.to_dict()
            if news.energy_sector_link_contract is not None
            else None
        ),
        "occurrence_count": len(news.occurrences),
        "canonical_count": len(
            {item.canonical_article_id for item in news.occurrences}
        ),
        "provider_modes": sorted(
            {item.provider_mode for item in news.occurrences}
        ),
        "data_classes": sorted({item.data_class for item in news.occurrences}),
    }
    if actual_news_reference != expected_news_reference:
        raise CoverageAuditError(
            "Source-bound verification found the news/SQLite reference differs"
        )

    actual_stock_reference = _read_json(
        root / "stock_artifact_reference.json",
        root=root,
        label="coverage stock reference",
    )
    if actual_stock_reference != stock.reference:
        raise CoverageAuditError(
            "Source-bound verification found the price/split reference differs"
        )

    expected_assessment_reference = _portable_source_assessment_reference(
        assessment_path
    )
    expected_assessment_reference["input_source_evidence"] = [
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
    expected_assessment_reference["permission_policy"] = dict(
        news.permission_policy_reference
    )
    if actual_assessment_reference != expected_assessment_reference:
        raise CoverageAuditError(
            "Source-bound verification found the assessment/policy reference differs"
        )

    computation = compute_coverage(news, stock, config)
    actual_quality = _read_json(
        root / "quality_report.json", root=root, label="coverage quality report"
    )
    if actual_quality != computation.quality_report:
        raise CoverageAuditError(
            "Source-bound verification found quality metrics differ from source replay"
        )
    effective_policy_sha256 = _sha256_bytes(
        _canonical_json_bytes(config.to_dict())
    )
    expected_summary = {
        **dict(computation.final_summary),
        "run_id": pack_result.run_id,
        "artifact_directory": f"news_coverage/{pack_result.run_id}",
        "finalization_state": "COMMITTED",
        "config": config.to_dict(),
        "effective_readiness_policy_sha256": effective_policy_sha256,
    }
    actual_summary = _read_json(
        root / "final_summary.json", root=root, label="coverage final summary"
    )
    if actual_summary != expected_summary:
        raise CoverageAuditError(
            "Source-bound verification found final summary differs from source replay"
        )

    csv_contracts: tuple[
        tuple[str, Sequence[Mapping[str, Any]], tuple[str, ...]], ...
    ] = (
        (
            "overlap_by_trading_day.csv",
            computation.trading_day_rows,
            (
                "trading_date", "split", "decision_cutoff_utc",
                "all_financial_occurrence_count",
                "all_financial_canonical_article_count",
                "ogdc_linked_occurrence_count",
                "ogdc_linked_canonical_article_count",
                "energy_sector_occurrence_count",
                "energy_sector_canonical_article_count",
            ),
        ),
        (
            "overlap_by_month.csv",
            computation.monthly_rows,
            (
                "month", "population", "population_supported", "trading_days",
                "trading_days_with_articles", "coverage_percent",
                "article_occurrences", "canonical_articles",
                "articles_per_trading_day", "zero_article_trading_days",
            ),
        ),
        (
            "split_coverage.csv",
            computation.split_rows,
            (
                "population", "population_supported", "split", "trading_days",
                "trading_days_with_articles", "coverage_percent",
                "article_occurrences", "canonical_articles",
                "articles_per_trading_day", "zero_article_trading_days",
                "longest_missing_gap_sessions",
            ),
        ),
        (
            "missing_intervals.csv",
            computation.missing_interval_rows,
            (
                "population", "split", "start_date", "end_date",
                "missing_trading_sessions",
            ),
        ),
        (
            "article_session_mappings.csv",
            computation.article_session_rows,
            (
                "occurrence_key", "ingestion_run_id", "article_id",
                "canonical_article_id", "provider", "source_name", "source_url",
                "publisher_document_key", "mentioned_sectors", "data_class",
                "population", "population_supported", "occurrence_in_population",
                "is_duplicate_occurrence", "published_at", "source_observed_at",
                "available_at", "fetched_at", "occurrence_assigned_trading_date",
                "occurrence_mapping_status", "occurrence_arrival_class",
                "canonical_representative_article_id",
                "canonical_representative_ingestion_run_id",
                "is_population_canonical_representative", "canonical_available_at",
                "canonical_source_observed_at", "canonical_assigned_trading_date",
                "canonical_mapping_status", "canonical_arrival_class",
                "content_sha256", "raw_artifact_reference",
            ),
        ),
        (
            "ogdc_linked_articles.csv",
            computation.ogdc_article_rows,
            (
                "canonical_article_id", "representative_article_id", "source_name",
                "source_url", "title", "published_at", "source_observed_at",
                "available_at", "assigned_trading_date", "mapping_status",
                "arrival_class", "occurrence_count", "sources", "link_basis",
                "link_evidence", "content_sha256",
            ),
        ),
    )
    for filename, rows, fields in csv_contracts:
        _assert_source_bound_csv(root, filename, rows, fields)

    return replace(pack_result, verification_basis="SOURCE_EVIDENCE_BOUND")


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

    declared_schema_versions = {
        manifest.get("schema_version"),
        summary.get("schema_version"),
        completion.get("coverage_schema_version"),
    }
    if declared_schema_versions != {COVERAGE_SCHEMA_VERSION}:
        if declared_schema_versions == {"1.0"}:
            raise CoverageAuditError(
                "Coverage schema 1.0 is historical and lacks the current semantic "
                "verification contract; regenerate the coverage pack as schema 1.1"
            )
        raise CoverageAuditError(
            "Coverage artifact schema versions are unsupported or inconsistent"
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
    if {item["path"] for item in inventory} != _COVERAGE_PAYLOAD_FILES:
        raise CoverageAuditError(
            "Coverage artifact inventory does not contain the complete payload contract"
        )
    observed = _artifact_inventory(root, exclude=set(_COVERAGE_SEAL_FILES))
    if observed != inventory:
        raise CoverageAuditError(
            "Coverage artifacts are missing, extra, changed, or unsafe"
        )
    _verify_coverage_semantics(root, manifest=manifest, summary=summary)

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
    recomputed = compute_coverage(news, stock, config)
    if computation != recomputed:
        raise CoverageAuditError(
            "Coverage computation does not match recomputed input evidence"
        )
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
            "energy_sector_link_contract": (
                news.energy_sector_link_contract.to_dict()
                if news.energy_sector_link_contract is not None
                else None
            ),
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
                "population_supported",
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
                "population_supported",
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
                "source_url",
                "publisher_document_key",
                "mentioned_sectors",
                "data_class",
                "population",
                "population_supported",
                "occurrence_in_population",
                "is_duplicate_occurrence",
                "published_at",
                "source_observed_at",
                "available_at",
                "fetched_at",
                "occurrence_assigned_trading_date",
                "occurrence_mapping_status",
                "occurrence_arrival_class",
                "canonical_representative_article_id",
                "canonical_representative_ingestion_run_id",
                "is_population_canonical_representative",
                "canonical_available_at",
                "canonical_source_observed_at",
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
                "source_observed_at",
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
                    "energy_sector_link_contract": (
                        news.energy_sector_link_contract.to_dict()
                        if news.energy_sector_link_contract is not None
                        else None
                    ),
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
    try:
        reservation.unlink(missing_ok=True)
    except OSError:
        # Publication has crossed its atomic commit point. Warning policy (for
        # example ``PYTHONWARNINGS=error``) must not turn housekeeping telemetry
        # into a false publication failure after the immutable run is visible.
        try:
            warnings.warn(
                "Coverage run is already committed; reservation cleanup failed and "
                "requires a later housekeeping retry.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:
            pass
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
    "EnergySectorLinkContract",
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
    "verify_coverage_pack",
    "verify_coverage_run",
    "validate_coverage_run_id",
]
