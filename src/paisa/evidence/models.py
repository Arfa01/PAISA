"""Provider-neutral contemporaneous evidence models.

The evidence contract intentionally permits a defensible historical
``available_at`` to precede PAISA's later ``fetched_at``.  That exception is
available only to reviewed contemporaneous sources such as a GDELT batch or an
official exchange posting; it is not applied to ordinary publisher news.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Any, Literal, Mapping
from urllib.parse import parse_qsl, urlsplit


EVIDENCE_SCHEMA_VERSION = "1.0"
EvidenceType = Literal[
    "publisher_news",
    "gdelt_metadata",
    "exchange_announcement",
]
DataClass = Literal["real", "imported", "fixture"]
TimePrecision = Literal["second", "minute", "date_only", "unknown"]
RunStatus = Literal["PASS", "PARTIAL", "FAIL"]

EVENT_CATEGORIES = frozenset(
    {
        "financial_results",
        "board_meeting",
        "material_information",
        "production_update",
        "dividend",
        "annual_report",
        "AGM_EOGM",
        "corporate_action",
        "other",
    }
)
EVIDENCE_TYPES = frozenset(
    {"publisher_news", "gdelt_metadata", "exchange_announcement"}
)
PROVIDER_EVIDENCE_TYPES: Mapping[tuple[str, str], EvidenceType] = {
    ("gdelt-gkg-2.1", "raw_archive"): "gdelt_metadata",
    ("psx-announcement", "manual_import"): "exchange_announcement",
}
TIME_PRECISIONS = frozenset({"second", "minute", "date_only", "unknown"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_SENSITIVE_QUERY_KEY = re.compile(
    r"(^|[_-])(api[_-]?key|authorization|credential|password|secret|token)([_-]|$)",
    re.IGNORECASE,
)


def utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Evidence timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="auto").replace(
        "+00:00", "Z"
    )


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _safe_relative(value: str, *, field_name: str) -> str:
    text = str(value)
    path_text, separator, fragment = text.partition("#")
    path = PurePosixPath(path_text)
    if (
        not path_text
        or path_text == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in path_text
    ):
        raise ValueError(f"{field_name} must be a safe relative POSIX reference")
    if separator and (not fragment or "#" in fragment or len(fragment) > 512):
        raise ValueError(f"{field_name} fragment is invalid")
    return text


def _validate_public_url(value: str | None, *, field_name: str) -> None:
    if value is None:
        return
    text = _single_line(value, field_name=field_name, maximum=8_000)
    parsed = urlsplit(text)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field_name} cannot contain embedded credentials")
    if any(_SENSITIVE_QUERY_KEY.search(key) for key, _ in parse_qsl(parsed.query)):
        raise ValueError(f"{field_name} cannot contain credential query values")


def parse_utc_timestamp(value: Any, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an ISO-8601 string or null")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} is not a valid ISO-8601 timestamp") from exc
    return _aware_utc(parsed, field_name=field_name)


def _single_line(value: str, *, field_name: str, maximum: int = 2_000) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name} cannot be blank")
    if "\r" in text or "\n" in text:
        raise ValueError(f"{field_name} must be a single-line value")
    if len(text) > maximum:
        raise ValueError(f"{field_name} exceeds its maximum length")
    return text


@dataclass(frozen=True, slots=True)
class EvidenceError:
    provider_id: str
    configured_source: str
    error_type: str
    message: str
    occurred_at: datetime
    partition_id: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _aware_utc(self.occurred_at, field_name="occurred_at")
        _single_line(self.provider_id, field_name="provider_id", maximum=128)
        _single_line(
            self.configured_source, field_name="configured_source", maximum=2_000
        )
        _single_line(self.error_type, field_name="error_type", maximum=128)
        _single_line(self.message, field_name="message", maximum=4_000)
        if not isinstance(self.details, Mapping):
            raise TypeError("Evidence error details must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["occurred_at"] = utc_iso(self.occurred_at)
        payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceCapture:
    """One exact downloaded archive or manually supplied input file."""

    capture_id: str
    provider_id: str
    configured_source: str
    filename: str
    fetched_at: datetime
    content: bytes = field(repr=False)
    source_url: str | None = None
    content_type: str | None = None
    sha256: str | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _aware_utc(self.fetched_at, field_name="fetched_at")
        if PurePosixPath(self.filename).name != self.filename:
            raise ValueError("Evidence capture filename must be a basename")
        if self.sha256 is not None and not _SHA256.fullmatch(self.sha256):
            raise ValueError("Evidence capture SHA-256 is invalid")

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "provider_id": self.provider_id,
            "configured_source": self.configured_source,
            "filename": self.filename,
            "fetched_at": utc_iso(self.fetched_at),
            "source_url": self.source_url,
            "content_type": self.content_type,
            "sha256": self.sha256,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """One normalized, typed contemporaneous evidence occurrence."""

    evidence_id: str
    evidence_type: EvidenceType
    provider_id: str
    provider_mode: str
    data_class: DataClass
    source_name: str
    source_url: str | None
    source_record_id: str
    ticker: str
    company_name: str
    title: str | None
    source_published_at: datetime | None
    source_published_date: date | None
    available_at: datetime | None
    fetched_at: datetime
    imported_at: datetime
    historical_availability_basis: str
    time_precision: TimePrecision
    contemporaneous_source_id: str
    contemporaneous_batch_at: datetime | None
    permission_policy_id: str
    raw_evidence_reference: str
    content_sha256: str
    language: str | None = None
    event_category: str | None = None
    announcement_type: str | None = None
    pdf_reference: str | None = None
    pdf_sha256: str | None = None
    normalization_warnings: tuple[str, ...] = ()
    entity_match_evidence: tuple[Mapping[str, str], ...] = ()
    source_metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.evidence_id):
            raise ValueError("evidence_id is invalid")
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"Unsupported evidence_type: {self.evidence_type}")
        if self.data_class not in {"real", "imported", "fixture"}:
            raise ValueError(f"Unsupported data_class: {self.data_class}")
        if self.time_precision not in TIME_PRECISIONS:
            raise ValueError(f"Unsupported time_precision: {self.time_precision}")
        expected_evidence_type = PROVIDER_EVIDENCE_TYPES.get(
            (self.provider_id, self.provider_mode)
        )
        if (
            expected_evidence_type is not None
            and self.evidence_type != expected_evidence_type
        ):
            raise ValueError(
                "Provider mode requires evidence_type "
                f"{expected_evidence_type!r}"
            )
        for name in (
            "provider_id",
            "provider_mode",
            "source_name",
            "source_record_id",
            "ticker",
            "company_name",
            "historical_availability_basis",
            "contemporaneous_source_id",
            "permission_policy_id",
        ):
            _single_line(str(getattr(self, name)), field_name=name)
        _validate_public_url(self.source_url, field_name="source_url")
        if self.pdf_reference is not None and self.pdf_reference.startswith(
            ("http://", "https://")
        ):
            _validate_public_url(self.pdf_reference, field_name="pdf_reference")
        elif self.pdf_reference is not None:
            _safe_relative(self.pdf_reference, field_name="pdf_reference")
        _aware_utc(self.fetched_at, field_name="fetched_at")
        _aware_utc(self.imported_at, field_name="imported_at")
        if self.source_published_at is not None:
            _aware_utc(
                self.source_published_at,
                field_name="source_published_at",
            )
        if self.available_at is not None:
            _aware_utc(self.available_at, field_name="available_at")
        if self.contemporaneous_batch_at is not None:
            _aware_utc(
                self.contemporaneous_batch_at,
                field_name="contemporaneous_batch_at",
            )
        _safe_relative(
            self.raw_evidence_reference, field_name="raw_evidence_reference"
        )
        if not _SHA256.fullmatch(self.content_sha256):
            raise ValueError("content_sha256 is invalid")
        if self.pdf_sha256 is not None and not _SHA256.fullmatch(self.pdf_sha256):
            raise ValueError("pdf_sha256 is invalid")
        if (
            self.event_category is not None
            and self.event_category not in EVENT_CATEGORIES
        ):
            raise ValueError(f"Unsupported event_category: {self.event_category}")
        if not isinstance(self.source_metadata, Mapping):
            raise TypeError("source_metadata must be a mapping")
        if any(
            not isinstance(item, Mapping) for item in self.entity_match_evidence
        ):
            raise TypeError("entity_match_evidence entries must be mappings")

        if self.time_precision == "date_only":
            if self.source_published_date is None:
                raise ValueError("date-only evidence requires source_published_date")
            if self.source_published_at is not None or self.available_at is not None:
                raise ValueError(
                    "date-only evidence cannot invent source_published_at or available_at"
                )
        elif self.source_published_date is not None:
            raise ValueError(
                "source_published_date is reserved for date-only evidence"
            )

        if self.evidence_type == "gdelt_metadata":
            if self.provider_mode != "raw_archive":
                raise ValueError("GDELT metadata evidence requires raw_archive mode")
            if self.data_class != "real":
                raise ValueError("GDELT archive evidence must be labelled real")
            if self.time_precision != "second":
                raise ValueError("GDELT archive timestamps require second precision")
            if self.available_at is None or self.contemporaneous_batch_at is None:
                raise ValueError("GDELT archive evidence requires a batch timestamp")
            if self.available_at != self.contemporaneous_batch_at:
                raise ValueError("GDELT available_at must equal its batch timestamp")
            if self.source_published_at is None:
                raise ValueError("GDELT web evidence requires V2.1DATE")
            if self.historical_availability_basis != "gdelt_gkg_batch_timestamp":
                raise ValueError("GDELT availability basis is invalid")
        elif self.evidence_type == "exchange_announcement":
            if self.ticker.upper() != "OGDC":
                raise ValueError("The initial PSX announcement slice requires OGDC")
            if self.event_category is None or self.announcement_type is None:
                raise ValueError(
                    "Exchange announcements require type and event category"
                )
            if self.time_precision == "date_only":
                if (
                    self.historical_availability_basis
                    != "official_exchange_date_conservative_next_session"
                ):
                    raise ValueError("Date-only exchange availability basis is invalid")
            else:
                if self.available_at is None or self.source_published_at is None:
                    raise ValueError(
                        "Timed exchange announcements require posting and availability"
                    )
                if self.available_at != self.source_published_at:
                    raise ValueError(
                        "Timed exchange available_at must equal exchange posting time"
                    )
                if (
                    self.historical_availability_basis
                    != "official_exchange_posted_timestamp"
                ):
                    raise ValueError("Timed exchange availability basis is invalid")
        elif self.available_at is not None and self.available_at < self.fetched_at:
            raise ValueError(
                "Publisher news cannot claim availability before PAISA fetched it"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "provider_id": self.provider_id,
            "provider_mode": self.provider_mode,
            "data_class": self.data_class,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "source_record_id": self.source_record_id,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "title": self.title,
            "source_published_at": utc_iso(self.source_published_at),
            "source_published_date": (
                self.source_published_date.isoformat()
                if self.source_published_date is not None
                else None
            ),
            "available_at": utc_iso(self.available_at),
            "fetched_at": utc_iso(self.fetched_at),
            "imported_at": utc_iso(self.imported_at),
            "historical_availability_basis": self.historical_availability_basis,
            "time_precision": self.time_precision,
            "contemporaneous_source_id": self.contemporaneous_source_id,
            "contemporaneous_batch_at": utc_iso(self.contemporaneous_batch_at),
            "permission_policy_id": self.permission_policy_id,
            "raw_evidence_reference": self.raw_evidence_reference,
            "content_sha256": self.content_sha256,
            "language": self.language,
            "event_category": self.event_category,
            "announcement_type": self.announcement_type,
            "pdf_reference": self.pdf_reference,
            "pdf_sha256": self.pdf_sha256,
            "normalization_warnings": list(self.normalization_warnings),
            "entity_match_evidence": [
                dict(item) for item in self.entity_match_evidence
            ],
            "source_metadata": dict(self.source_metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceRecord":
        """Reconstruct and validate a normalized JSON evidence row."""

        if not isinstance(value, Mapping):
            raise TypeError("Evidence row must be a mapping")
        if value.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError("Evidence row schema is unsupported")
        published_date_raw = value.get("source_published_date")
        published_date: date | None
        if published_date_raw is None:
            published_date = None
        elif isinstance(published_date_raw, str):
            try:
                published_date = date.fromisoformat(published_date_raw)
            except ValueError as exc:
                raise ValueError("source_published_date is invalid") from exc
        else:
            raise ValueError("source_published_date must be an ISO date or null")
        warnings = value.get("normalization_warnings")
        matches = value.get("entity_match_evidence")
        metadata = value.get("source_metadata")
        if not isinstance(warnings, list) or any(
            not isinstance(item, str) for item in warnings
        ):
            raise ValueError("normalization_warnings must be a string list")
        if not isinstance(matches, list) or any(
            not isinstance(item, Mapping) for item in matches
        ):
            raise ValueError("entity_match_evidence must be an object list")
        if not isinstance(metadata, Mapping):
            raise ValueError("source_metadata must be an object")
        return cls(
            evidence_id=str(value.get("evidence_id") or ""),
            evidence_type=value.get("evidence_type"),
            provider_id=str(value.get("provider_id") or ""),
            provider_mode=str(value.get("provider_mode") or ""),
            data_class=value.get("data_class"),
            source_name=str(value.get("source_name") or ""),
            source_url=value.get("source_url"),
            source_record_id=str(value.get("source_record_id") or ""),
            ticker=str(value.get("ticker") or ""),
            company_name=str(value.get("company_name") or ""),
            title=value.get("title"),
            source_published_at=parse_utc_timestamp(
                value.get("source_published_at"),
                field_name="source_published_at",
            ),
            source_published_date=published_date,
            available_at=parse_utc_timestamp(
                value.get("available_at"), field_name="available_at"
            ),
            fetched_at=parse_utc_timestamp(
                value.get("fetched_at"), field_name="fetched_at"
            ),
            imported_at=parse_utc_timestamp(
                value.get("imported_at"), field_name="imported_at"
            ),
            historical_availability_basis=str(
                value.get("historical_availability_basis") or ""
            ),
            time_precision=value.get("time_precision"),
            contemporaneous_source_id=str(
                value.get("contemporaneous_source_id") or ""
            ),
            contemporaneous_batch_at=parse_utc_timestamp(
                value.get("contemporaneous_batch_at"),
                field_name="contemporaneous_batch_at",
            ),
            permission_policy_id=str(value.get("permission_policy_id") or ""),
            raw_evidence_reference=str(value.get("raw_evidence_reference") or ""),
            content_sha256=str(value.get("content_sha256") or ""),
            language=value.get("language"),
            event_category=value.get("event_category"),
            announcement_type=value.get("announcement_type"),
            pdf_reference=value.get("pdf_reference"),
            pdf_sha256=value.get("pdf_sha256"),
            normalization_warnings=tuple(warnings),
            entity_match_evidence=tuple(dict(item) for item in matches),
            source_metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class EvidenceFetchResult:
    provider_id: str
    provider_mode: str
    data_class: DataClass
    fetched_at: datetime
    permission_policy_id: str
    configured_sources: tuple[Mapping[str, Any], ...]
    partition_plan: tuple[Mapping[str, Any], ...]
    records: tuple[EvidenceRecord, ...] = ()
    captures: tuple[EvidenceCapture, ...] = ()
    errors: tuple[EvidenceError, ...] = ()
    warnings: tuple[str, ...] = ()
    counters: Mapping[str, int] = field(default_factory=dict)
    timestamp_validation: Mapping[str, Any] = field(default_factory=dict)
    effective_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _aware_utc(self.fetched_at, field_name="fetched_at")
        if self.data_class not in {"real", "imported", "fixture"}:
            raise ValueError(f"Unsupported data_class: {self.data_class}")
        _single_line(self.provider_id, field_name="provider_id", maximum=128)
        _single_line(self.provider_mode, field_name="provider_mode", maximum=128)
        _single_line(
            self.permission_policy_id,
            field_name="permission_policy_id",
            maximum=128,
        )
        if not isinstance(self.effective_config, Mapping):
            raise TypeError("effective_config must be a mapping")
        if any(not isinstance(item, Mapping) for item in self.configured_sources):
            raise TypeError("configured_sources entries must be mappings")
        if any(not isinstance(item, Mapping) for item in self.partition_plan):
            raise TypeError("partition_plan entries must be mappings")
        if any(
            item.permission_policy_id != self.permission_policy_id
            for item in self.records
        ):
            raise ValueError("Record permission policy differs from its run")
        if any(
            item.provider_id != self.provider_id
            or item.provider_mode != self.provider_mode
            or item.data_class != self.data_class
            for item in self.records
        ):
            raise ValueError("Record provider semantics differ from its run")
        if any(item.provider_id != self.provider_id for item in self.captures):
            raise ValueError("Capture provider differs from its run")
        if any(item.provider_id != self.provider_id for item in self.errors):
            raise ValueError("Provider error differs from its run")


__all__ = [
    "DataClass",
    "EVIDENCE_SCHEMA_VERSION",
    "EVIDENCE_TYPES",
    "EVENT_CATEGORIES",
    "PROVIDER_EVIDENCE_TYPES",
    "EvidenceCapture",
    "EvidenceError",
    "EvidenceFetchResult",
    "EvidenceRecord",
    "EvidenceType",
    "RunStatus",
    "TIME_PRECISIONS",
    "TimePrecision",
    "parse_utc_timestamp",
    "utc_iso",
]
