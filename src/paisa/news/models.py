from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any, Literal, Mapping


NEWS_SCHEMA_VERSION = "1.1"
SUPPORTED_NEWS_SCHEMA_VERSIONS = frozenset({"1.0", NEWS_SCHEMA_VERSION})
NEWS_AVAILABILITY_POLICY_VERSION = "news-availability-v1"
DataClass = Literal["real", "imported", "fixture"]
RunStatus = Literal["PASS", "PARTIAL", "FAIL"]
QualityStatus = Literal["PASS", "WARN", "FAIL"]
DuplicateScope = Literal["within_run", "historical"]


def utc_iso(value: datetime | None) -> str | None:
    """Serialize an aware timestamp as a UTC RFC 3339 value."""

    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("News timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat(timespec="auto").replace(
        "+00:00", "Z"
    )


def _safe_relative_reference(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Artifact references must be safe relative POSIX paths.")
    return path.as_posix()


@dataclass(frozen=True)
class ProviderError:
    provider_id: str
    configured_source: str
    error_type: str
    message: str
    occurred_at: datetime
    http_status: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        utc_iso(self.occurred_at)
        if not isinstance(self.details, Mapping):
            raise TypeError("Provider error details must be a mapping.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "configured_source": self.configured_source,
            "error_type": self.error_type,
            "message": self.message,
            "occurred_at": utc_iso(self.occurred_at),
            "http_status": self.http_status,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class RawCapture:
    """One byte-preserved provider response or imported source file."""

    capture_id: str
    provider_id: str
    configured_source: str
    source_name: str
    fetched_at: datetime
    data_class: DataClass
    content: bytes = field(repr=False)
    filename: str = "capture.bin"
    http_status: int | None = None
    content_type: str | None = None
    response_sha256: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        utc_iso(self.fetched_at)
        if self.data_class not in {"real", "imported", "fixture"}:
            raise ValueError(f"Unsupported data_class: {self.data_class}")
        if PurePosixPath(self.filename).name != self.filename:
            raise ValueError("Raw capture filename must be a basename.")

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "provider_id": self.provider_id,
            "configured_source": self.configured_source,
            "source_name": self.source_name,
            "fetched_at": utc_iso(self.fetched_at),
            "data_class": self.data_class,
            "filename": self.filename,
            "http_status": self.http_status,
            "content_type": self.content_type,
            "response_sha256": self.response_sha256,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class RawNewsRecord:
    """Provider-neutral record retaining the provider's unnormalized evidence."""

    raw_record_id: str
    provider_id: str
    configured_source: str
    source_name: str
    source_type: str
    fetched_at: datetime
    data_class: DataClass
    source_url: str | None = None
    title: str | None = None
    summary: str | None = None
    body: str | None = None
    published_at_raw: str | None = None
    published_timezone: str | None = None
    available_at_raw: str | None = None
    language: str | None = None
    provider_record_id: str | None = None
    capture_id: str | None = None
    http_status: int | None = None
    response_content_type: str | None = None
    response_sha256: str | None = None
    raw_payload: Mapping[str, Any] = field(default_factory=dict, repr=False)
    provider_warnings: tuple[str, ...] = ()
    # Append new optional fields so callers using the original positional
    # constructor retain the v0 argument order.
    source_observed_at_raw: str | None = None

    def __post_init__(self) -> None:
        utc_iso(self.fetched_at)
        if self.data_class not in {"real", "imported", "fixture"}:
            raise ValueError(f"Unsupported data_class: {self.data_class}")

    def metadata_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fetched_at"] = utc_iso(self.fetched_at)
        data["raw_payload"] = dict(self.raw_payload)
        data["provider_warnings"] = list(self.provider_warnings)
        return data


@dataclass(frozen=True)
class ProviderFetchResult:
    provider_id: str
    data_class: DataClass
    fetched_at: datetime
    configured_sources: tuple[Mapping[str, Any], ...]
    records: tuple[RawNewsRecord, ...] = ()
    captures: tuple[RawCapture, ...] = ()
    errors: tuple[ProviderError, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        utc_iso(self.fetched_at)
        if self.data_class not in {"real", "imported", "fixture"}:
            raise ValueError(f"Unsupported data_class: {self.data_class}")


@dataclass(frozen=True)
class NormalizedNewsArticle:
    schema_version: str
    article_id: str
    raw_record_id: str
    provider: str
    source_name: str
    source_type: str
    source_url: str | None
    canonical_url: str | None
    title: str | None
    summary: str | None
    body: str | None
    published_at: datetime | None
    published_timezone: str | None
    available_at: datetime
    fetched_at: datetime
    language: str | None
    mentioned_tickers: tuple[str, ...]
    mentioned_companies: tuple[str, ...]
    mentioned_sectors: tuple[str, ...]
    content_sha256: str
    duplicate_of: str | None
    ingestion_run_id: str
    raw_artifact_reference: str
    normalization_warnings: tuple[str, ...]
    data_class: DataClass
    source_observed_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_NEWS_SCHEMA_VERSIONS:
            raise ValueError(f"Unsupported news schema: {self.schema_version}")
        if self.schema_version == "1.0" and self.source_observed_at is not None:
            raise ValueError(
                "source_observed_at requires news record schema 1.1"
            )
        utc_iso(self.published_at)
        utc_iso(self.source_observed_at)
        utc_iso(self.available_at)
        utc_iso(self.fetched_at)
        if self.available_at < self.fetched_at:
            raise ValueError("available_at cannot precede fetched_at.")
        if self.published_at is not None and self.available_at < self.published_at:
            raise ValueError("available_at cannot precede published_at.")
        _safe_relative_reference(self.raw_artifact_reference)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "article_id": self.article_id,
            "raw_record_id": self.raw_record_id,
            "provider": self.provider,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "canonical_url": self.canonical_url,
            "title": self.title,
            "summary": self.summary,
            "body": self.body,
            "published_at": utc_iso(self.published_at),
            "published_timezone": self.published_timezone,
            "available_at": utc_iso(self.available_at),
            "fetched_at": utc_iso(self.fetched_at),
            "language": self.language,
            "mentioned_tickers": list(self.mentioned_tickers),
            "mentioned_companies": list(self.mentioned_companies),
            "mentioned_sectors": list(self.mentioned_sectors),
            "content_sha256": self.content_sha256,
            "duplicate_of": self.duplicate_of,
            "ingestion_run_id": self.ingestion_run_id,
            "raw_artifact_reference": self.raw_artifact_reference,
            "normalization_warnings": list(self.normalization_warnings),
            "data_class": self.data_class,
        }
        if self.schema_version != "1.0":
            payload["source_observed_at"] = utc_iso(self.source_observed_at)
        return payload


@dataclass(frozen=True)
class ArticleEntityLink:
    link_id: str
    article_id: str
    ticker: str
    company_name: str
    matched_alias: str
    match_method: Literal["exact", "normalized_exact"]
    match_location: Literal["title", "summary", "body"]
    confidence: Literal["exact", "normalized_exact"]

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DuplicateRelationship:
    article_id: str
    canonical_article_id: str
    rule: Literal["canonical_url", "content_sha256", "title_source_time"]
    duplicate_scope: DuplicateScope | None = None
    canonical_owner_run_id: str | None = None

    def __post_init__(self) -> None:
        if self.duplicate_scope not in {None, "within_run", "historical"}:
            raise ValueError(f"Unsupported duplicate scope: {self.duplicate_scope}")
        if (self.duplicate_scope is None) != (self.canonical_owner_run_id is None):
            raise ValueError(
                "duplicate_scope and canonical_owner_run_id must be supplied together"
            )
        if (
            self.canonical_owner_run_id is not None
            and not self.canonical_owner_run_id.strip()
        ):
            raise ValueError("canonical_owner_run_id cannot be blank")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "article_id": self.article_id,
            "duplicate_of": self.canonical_article_id,
            "deduplication_rule": self.rule,
            "duplicate_scope": self.duplicate_scope,
            "canonical_owner_run_id": self.canonical_owner_run_id,
        }


@dataclass(frozen=True)
class PipelineSummary:
    schema_version: str
    run_id: str
    run_status: RunStatus
    quality_status: QualityStatus
    provider: str
    provider_mode: DataClass
    source_count: int
    raw_record_count: int
    normalized_record_count: int
    duplicate_count: int
    ogdc_linked_article_count: int
    missing_publication_time_count: int
    date_range: Mapping[str, str | None]
    artifact_directory: str
    database_path: str
    limitations: tuple[str, ...] = ()
    normalized_occurrence_count: int | None = None
    new_canonical_article_count: int | None = None
    matched_existing_canonical_count: int | None = None
    within_run_duplicate_count: int | None = None
    historical_duplicate_count: int | None = None
    represented_canonical_count: int | None = None
    deduplication_rule_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        named_counts = {
            "normalized_occurrence_count": self.normalized_occurrence_count,
            "new_canonical_article_count": self.new_canonical_article_count,
            "matched_existing_canonical_count": self.matched_existing_canonical_count,
            "within_run_duplicate_count": self.within_run_duplicate_count,
            "historical_duplicate_count": self.historical_duplicate_count,
            "represented_canonical_count": self.represented_canonical_count,
        }
        for name, value in named_counts.items():
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")

        complete_counts = all(value is not None for value in named_counts.values())
        if complete_counts:
            assert self.normalized_occurrence_count is not None
            assert self.new_canonical_article_count is not None
            assert self.matched_existing_canonical_count is not None
            assert self.within_run_duplicate_count is not None
            assert self.historical_duplicate_count is not None
            assert self.represented_canonical_count is not None
            if self.normalized_occurrence_count != self.normalized_record_count:
                raise ValueError(
                    "normalized occurrence count must equal normalized record count"
                )
            if (
                self.within_run_duplicate_count
                + self.historical_duplicate_count
                != self.duplicate_count
            ):
                raise ValueError(
                    "within-run and historical duplicate counts must sum to "
                    "duplicate_count"
                )
            if self.matched_existing_canonical_count != self.historical_duplicate_count:
                raise ValueError(
                    "matched existing canonical count must equal historical "
                    "duplicate count"
                )
            if (
                self.new_canonical_article_count + self.duplicate_count
                != self.normalized_occurrence_count
            ):
                raise ValueError(
                    "new canonical and duplicate counts must cover all normalized "
                    "occurrences"
                )
            if not (
                self.new_canonical_article_count
                <= self.represented_canonical_count
                <= self.normalized_occurrence_count
            ):
                raise ValueError(
                    "represented canonical count is outside the possible range"
                )
        if self.deduplication_rule_counts and (
            sum(self.deduplication_rule_counts.values()) != self.duplicate_count
        ):
            raise ValueError(
                "deduplication rule counts must sum to duplicate_count"
            )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["limitations"] = list(self.limitations)
        return data
