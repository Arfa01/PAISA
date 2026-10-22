"""Manual import adapter for user-supplied, permitted news CSV datasets."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import re
from typing import Any

from paisa.news.models import (
    ProviderError,
    ProviderFetchResult,
    RawCapture,
    RawNewsRecord,
)
from paisa.news.providers.base import NewsProvider, sha256_bytes, utc_datetime


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "provider_record_id": ("article_id", "record_id", "id", "guid"),
    "source_url": ("source_url", "article_url", "url", "link"),
    "title": ("title", "headline"),
    "summary": ("summary", "description", "abstract"),
    "body": ("body", "content", "article_text", "text"),
    "published_at": (
        "published_at",
        "published",
        "publication_time",
        "published_date",
        "pub_date",
        "date",
    ),
    "available_at": ("available_at", "availability_time"),
    "published_timezone": (
        "published_timezone",
        "original_timezone",
        "timezone",
    ),
    "language": ("language", "lang"),
    "source_name": ("source_name", "publisher", "publication", "source"),
    "source_type": ("source_type",),
}


def _field(row: Mapping[str, str | None], field: str) -> str | None:
    for alias in _COLUMN_ALIASES[field]:
        value = _optional_text(row.get(alias))
        if value is not None:
            return value
    return None


class ManualCsvNewsProvider(NewsProvider):
    """Capture one CSV exactly and expose each row as imported evidence."""

    provider_id = "manual-csv"
    data_class = "imported"

    def __init__(
        self,
        csv_path: str | Path,
        *,
        source_name: str | None = None,
        default_language: str | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.source_name = source_name or self.csv_path.stem
        self.default_language = _optional_text(default_language)
        self.clock = clock
        self.configured_sources = (
            {"path": str(self.csv_path), "source_name": self.source_name},
        )
        self.effective_config = {
            "source_name": self.source_name,
            "default_language": self.default_language,
        }

    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult:
        configured_source = str(self.csv_path)
        capture_id = "manual-csv-input"
        try:
            payload = self.csv_path.read_bytes()
        except OSError as exc:
            acquisition_time = utc_datetime(fetched_at or self.clock())
            return ProviderFetchResult(
                provider_id=self.provider_id,
                data_class="imported",
                fetched_at=acquisition_time,
                configured_sources=self.configured_sources,
                errors=(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type="file_read_error",
                        message=str(exc),
                        occurred_at=acquisition_time,
                    ),
                ),
            )
        # File evidence becomes available only once the bytes have been read.
        # ``fetched_at`` exists solely for deterministic callers and tests.
        acquisition_time = utc_datetime(fetched_at or self.clock())

        checksum = sha256_bytes(payload)
        evidence_warnings: list[str] = []
        records: list[RawNewsRecord] = []
        errors: list[ProviderError] = []
        try:
            text = payload.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if reader.fieldnames is None:
                raise ValueError("Manual CSV is missing a header row")
            normalized_headers = [_column_key(name) for name in reader.fieldnames]
            if any(not name for name in normalized_headers):
                raise ValueError("Manual CSV contains a blank column name")
            if len(normalized_headers) != len(set(normalized_headers)):
                raise ValueError("Manual CSV column names collide after normalization")

            for row_index, source_row in enumerate(reader):
                row = {
                    normalized_headers[index]: _optional_text(source_row.get(original))
                    for index, original in enumerate(reader.fieldnames)
                }
                if not any(row.values()):
                    evidence_warnings.append(f"skipped_blank_row:{row_index + 2}")
                    continue
                title = _field(row, "title")
                source_url = _field(row, "source_url")
                published = _field(row, "published_at")
                warnings: list[str] = []
                if title is None:
                    warnings.append("missing_title")
                if source_url is None:
                    warnings.append("missing_source_url")
                if published is None:
                    warnings.append("missing_publication_time")
                records.append(
                    RawNewsRecord(
                        raw_record_id=f"{capture_id}-{row_index:06d}",
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        source_name=_field(row, "source_name") or self.source_name,
                        source_type=_field(row, "source_type") or "manual_csv",
                        fetched_at=acquisition_time,
                        data_class="imported",
                        source_url=source_url,
                        title=title,
                        summary=_field(row, "summary"),
                        body=_field(row, "body"),
                        published_at_raw=published,
                        published_timezone=_field(row, "published_timezone"),
                        available_at_raw=_field(row, "available_at"),
                        language=_field(row, "language") or self.default_language,
                        provider_record_id=_field(row, "provider_record_id"),
                        capture_id=capture_id,
                        http_status=None,
                        response_content_type="text/csv",
                        response_sha256=checksum,
                        raw_payload=dict(row),
                        provider_warnings=tuple(warnings),
                    )
                )
            if not records:
                evidence_warnings.append("csv_contains_no_data_rows")
        except (UnicodeDecodeError, csv.Error, ValueError) as exc:
            errors.append(
                ProviderError(
                    provider_id=self.provider_id,
                    configured_source=configured_source,
                    error_type="csv_parse_error",
                    message=str(exc),
                    occurred_at=acquisition_time,
                )
            )

        capture = RawCapture(
            capture_id=capture_id,
            provider_id=self.provider_id,
            configured_source=configured_source,
            source_name=self.source_name,
            fetched_at=acquisition_time,
            data_class="imported",
            content=payload,
            filename="manual_input.csv",
            content_type="text/csv",
            response_sha256=checksum,
            warnings=tuple(evidence_warnings),
        )
        return ProviderFetchResult(
            provider_id=self.provider_id,
            data_class="imported",
            fetched_at=acquisition_time,
            configured_sources=self.configured_sources,
            records=tuple(records),
            captures=(capture,),
            errors=tuple(errors),
        )


__all__ = ["ManualCsvNewsProvider"]
