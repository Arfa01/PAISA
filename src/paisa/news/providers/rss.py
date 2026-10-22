"""Configurable RSS 2.0 and Atom acquisition without article-page scraping."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlsplit
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from paisa.news.models import (
    ProviderError,
    ProviderFetchResult,
    RawCapture,
    RawNewsRecord,
)
from paisa.news.normalization import infer_timezone_evidence
from paisa.news.providers.base import NewsProvider, sha256_bytes, utc_datetime


_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_UTC_OFFSET = re.compile(r"^[+-](?:[01]\d|2[0-3]):?[0-5]\d$")
_SENSITIVE_QUERY_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|token|secret|password|credential|authorization)",
    re.IGNORECASE,
)
_DEFAULT_USER_AGENT = (
    "PAISA-FYP/0.1 academic news research; contact via repository owner"
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1].lower()


def _element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return _optional_text("".join(element.itertext()))


def _direct_child(element: ET.Element, *names: str) -> ET.Element | None:
    wanted = {name.lower() for name in names}
    for child in element:
        if _local_name(child.tag) in wanted:
            return child
    return None


def _direct_text(element: ET.Element, *names: str) -> str | None:
    return _element_text(_direct_child(element, *names))


def _atom_link(entry: ET.Element) -> str | None:
    fallback: str | None = None
    for child in entry:
        if _local_name(child.tag) != "link":
            continue
        href = _optional_text(child.attrib.get("href")) or _element_text(child)
        if href is None:
            continue
        if fallback is None:
            fallback = href
        if child.attrib.get("rel", "alternate").lower() == "alternate":
            return href
    return fallback


@dataclass(frozen=True)
class RssSourceConfig:
    """One explicitly permitted RSS/Atom endpoint."""

    source_id: str
    source_name: str
    url: str
    language: str | None = None
    default_timezone: str | None = None
    source_type: str | None = None

    def __post_init__(self) -> None:
        if not _SOURCE_ID.fullmatch(self.source_id):
            raise ValueError(
                "RSS source_id must start with a letter or digit and contain only "
                "letters, digits, dots, underscores, and hyphens"
            )
        if not self.source_name.strip():
            raise ValueError("RSS source_name cannot be blank")
        try:
            parsed = urlsplit(self.url)
            parsed.port
        except ValueError as exc:
            raise ValueError("RSS source URL is invalid") from exc
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("RSS source URL must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("RSS source URL must not contain credentials")
        if any(_SENSITIVE_QUERY_KEY.search(key) for key, _ in parse_qsl(parsed.query)):
            raise ValueError(
                "RSS source URL must not contain credentials in query parameters"
            )
        if self.source_type not in {None, "rss", "atom"}:
            raise ValueError("RSS source_type must be 'rss' or 'atom' when provided")
        if self.default_timezone:
            timezone_name = self.default_timezone.strip()
            if timezone_name.casefold() not in {"utc", "gmt", "z"} and not _UTC_OFFSET.fullmatch(
                timezone_name
            ):
                try:
                    ZoneInfo(timezone_name)
                except ZoneInfoNotFoundError as exc:
                    raise ValueError(
                        f"Unknown RSS default_timezone: {self.default_timezone}"
                    ) from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RssSourceConfig":
        return cls(
            source_id=str(value.get("source_id") or value.get("id") or "").strip(),
            source_name=str(value.get("source_name") or value.get("name") or "").strip(),
            url=str(value.get("url") or "").strip(),
            language=_optional_text(value.get("language")),
            default_timezone=_optional_text(value.get("default_timezone")),
            source_type=_optional_text(value.get("source_type")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "url": self.url,
            "language": self.language,
            "default_timezone": self.default_timezone,
            "source_type": self.source_type,
        }


def _load_rss_document(
    path: str | Path,
) -> tuple[Mapping[str, Any], tuple[RssSourceConfig, ...]]:
    config_path = Path(path)
    value = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("RSS configuration must be a JSON object")
    if value.get("schema_version", "1.0") != "1.0":
        raise ValueError("RSS configuration schema_version must be '1.0'")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("RSS configuration must contain a non-empty sources array")
    if any(not isinstance(item, Mapping) for item in sources):
        raise ValueError("Every RSS source must be a JSON object")
    enabled_sources: list[Mapping[str, Any]] = []
    for item in sources:
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("RSS source enabled must be true or false")
        if enabled:
            enabled_sources.append(item)
    if not enabled_sources:
        raise ValueError("RSS configuration must contain at least one enabled source")
    parsed = tuple(RssSourceConfig.from_mapping(item) for item in enabled_sources)
    ids = [source.source_id for source in parsed]
    if len(ids) != len(set(ids)):
        raise ValueError("RSS source_id values must be unique")
    return value, parsed


def load_rss_config(path: str | Path) -> tuple[RssSourceConfig, ...]:
    """Load enabled source entries from a versioned RSS configuration."""

    _, sources = _load_rss_document(path)
    return sources


def _parse_feed(
    payload: bytes,
    *,
    source: RssSourceConfig,
    capture_id: str,
    fetched_at: datetime,
    http_status: int,
    response_content_type: str | None,
    response_sha256: str,
) -> tuple[list[RawNewsRecord], list[str]]:
    root = ET.fromstring(payload)
    root_name = _local_name(root.tag)
    if root_name == "feed":
        source_type = "atom"
        entries = [child for child in root if _local_name(child.tag) == "entry"]
        feed_language = _optional_text(
            root.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
        )
    elif root_name in {"rss", "rdf"}:
        source_type = "rss"
        entries = [
            element for element in root.iter() if _local_name(element.tag) == "item"
        ]
        channel = next(
            (
                element
                for element in root.iter()
                if _local_name(element.tag) == "channel"
            ),
            root,
        )
        feed_language = _direct_text(channel, "language")
    else:
        raise ValueError(f"Unsupported feed root element: {root_name!r}")

    feed_warnings: list[str] = []
    if not entries:
        feed_warnings.append("feed_contains_no_entries")

    records: list[RawNewsRecord] = []
    for index, entry in enumerate(entries):
        warnings: list[str] = []
        title = _direct_text(entry, "title")
        if source_type == "atom":
            raw_link = _atom_link(entry)
            summary = _direct_text(entry, "summary")
            body = _direct_text(entry, "content")
            published = _direct_text(entry, "published")
            available = _direct_text(entry, "updated")
            if published is None and available is not None:
                warnings.append(
                    "atom_published_missing_used_updated_as_availability"
                )
            provider_record_id = _direct_text(entry, "id")
            language = _optional_text(
                entry.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            )
        else:
            raw_link = _direct_text(entry, "link")
            summary = _direct_text(entry, "description", "summary")
            body = _direct_text(entry, "encoded", "content")
            published = _direct_text(entry, "pubdate", "published", "date")
            available = None
            provider_record_id = _direct_text(entry, "guid", "id")
            language = _direct_text(entry, "language")

        source_url = urljoin(source.url, raw_link) if raw_link else None
        if title is None:
            warnings.append("missing_title")
        if source_url is None:
            warnings.append("missing_source_url")
        if published is None:
            warnings.append("missing_publication_time")

        raw_payload = {
            "provider_record_id": provider_record_id,
            "source_url": raw_link,
            "title": title,
            "summary": summary,
            "body": body,
            "published_at": published,
            "available_at": available,
            "language": language or feed_language or source.language,
        }
        records.append(
            RawNewsRecord(
                raw_record_id=f"{capture_id}-{index:06d}",
                provider_id="rss",
                configured_source=source.url,
                source_name=source.source_name,
                source_type=source_type,
                fetched_at=fetched_at,
                data_class="real",
                source_url=source_url,
                title=title,
                summary=summary,
                body=body,
                published_at_raw=published,
                published_timezone=(
                    infer_timezone_evidence(published or available)
                    or source.default_timezone
                ),
                available_at_raw=available,
                language=language or feed_language or source.language,
                provider_record_id=provider_record_id,
                capture_id=capture_id,
                http_status=http_status,
                response_content_type=response_content_type,
                response_sha256=response_sha256,
                raw_payload=raw_payload,
                provider_warnings=tuple(warnings),
            )
        )
    return records, feed_warnings


class RssNewsProvider(NewsProvider):
    """Fetch configured feeds only; this adapter never follows article links."""

    provider_id = "rss"
    data_class = "real"

    def __init__(
        self,
        sources: Sequence[RssSourceConfig],
        *,
        session: Any | None = None,
        timeout_seconds: float = 30.0,
        user_agent: str = _DEFAULT_USER_AGENT,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not sources:
            raise ValueError("At least one RSS source is required")
        ids = [source.source_id for source in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("RSS source_id values must be unique")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not user_agent.strip() or "\r" in user_agent or "\n" in user_agent:
            raise ValueError("user_agent must be a non-empty single-line value")
        self.sources = tuple(sources)
        self.configured_sources = tuple(source.to_dict() for source in self.sources)
        self.session = session or requests.Session()
        self.timeout_seconds = float(timeout_seconds)
        self.user_agent = user_agent.strip()
        self.clock = clock
        self.effective_config = {
            "timeout_seconds": self.timeout_seconds,
            "user_agent": self.user_agent,
            "redirect_policy": "reject",
        }

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> "RssNewsProvider":
        document, sources = _load_rss_document(path)
        options = dict(kwargs)
        if "timeout_seconds" not in options and "timeout_seconds" in document:
            options["timeout_seconds"] = float(document["timeout_seconds"])
        if "user_agent" not in options and "user_agent" in document:
            options["user_agent"] = str(document["user_agent"])
        return cls(sources, **options)

    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult:
        captures: list[RawCapture] = []
        records: list[RawNewsRecord] = []
        errors: list[ProviderError] = []
        observed_times: list[datetime] = []

        for source in self.sources:
            capture_id = f"rss-{source.source_id}"
            try:
                response = self.session.get(
                    source.url,
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": (
                            "application/atom+xml,application/rss+xml,"
                            "application/xml,text/xml;q=0.9,*/*;q=0.1"
                        ),
                    },
                )
                # Timestamp the evidence only after the response bytes have
                # actually been acquired.  ``fetched_at`` is an explicit test
                # hook; production callers leave it unset.
                payload = bytes(response.content)
                acquisition_time = utc_datetime(fetched_at or self.clock())
                observed_times.append(acquisition_time)
            except Exception as exc:
                acquisition_time = utc_datetime(fetched_at or self.clock())
                observed_times.append(acquisition_time)
                errors.append(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=source.url,
                        error_type="network_error",
                        message=str(exc),
                        occurred_at=acquisition_time,
                    )
                )
                continue

            checksum = sha256_bytes(payload)
            status = int(response.status_code)
            content_type = _optional_text(response.headers.get("Content-Type"))
            source_warnings: list[str] = []
            if content_type and not any(
                marker in content_type.lower() for marker in ("xml", "rss", "atom")
            ):
                source_warnings.append("unexpected_response_content_type")

            if not 200 <= status < 300:
                captures.append(
                    RawCapture(
                        capture_id=capture_id,
                        provider_id=self.provider_id,
                        configured_source=source.url,
                        source_name=source.source_name,
                        fetched_at=acquisition_time,
                        data_class="real",
                        content=payload,
                        filename=f"{source.source_id}.xml",
                        http_status=status,
                        content_type=content_type,
                        response_sha256=checksum,
                        warnings=tuple(source_warnings),
                    )
                )
                errors.append(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=source.url,
                        error_type="http_error",
                        message=f"RSS source returned HTTP {status}",
                        occurred_at=acquisition_time,
                        http_status=status,
                    )
                )
                continue

            try:
                parsed_records, parse_warnings = _parse_feed(
                    payload,
                    source=source,
                    capture_id=capture_id,
                    fetched_at=acquisition_time,
                    http_status=status,
                    response_content_type=content_type,
                    response_sha256=checksum,
                )
                source_warnings.extend(parse_warnings)
                actual_source_types = {record.source_type for record in parsed_records}
                if source.source_type and actual_source_types != {source.source_type}:
                    source_warnings.append("configured_source_type_mismatch")
                records.extend(parsed_records)
            except (ET.ParseError, ValueError) as exc:
                errors.append(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=source.url,
                        error_type="feed_parse_error",
                        message=str(exc),
                        occurred_at=acquisition_time,
                        http_status=status,
                    )
                )

            captures.append(
                RawCapture(
                    capture_id=capture_id,
                    provider_id=self.provider_id,
                    configured_source=source.url,
                    source_name=source.source_name,
                    fetched_at=acquisition_time,
                    data_class="real",
                    content=payload,
                    filename=f"{source.source_id}.xml",
                    http_status=status,
                    content_type=content_type,
                    response_sha256=checksum,
                    warnings=tuple(source_warnings),
                )
            )

        completed_at = max(observed_times) if observed_times else utc_datetime(
            fetched_at or self.clock()
        )
        return ProviderFetchResult(
            provider_id=self.provider_id,
            data_class="real",
            fetched_at=completed_at,
            configured_sources=self.configured_sources,
            records=tuple(records),
            captures=tuple(captures),
            errors=tuple(errors),
        )


__all__ = ["RssNewsProvider", "RssSourceConfig", "load_rss_config"]
