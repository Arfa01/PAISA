"""Configurable RSS 2.0 and Atom acquisition without article-page scraping."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import socket
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import zlib

import requests
from requests.adapters import HTTPAdapter
from defusedxml import DefusedXmlException
from defusedxml import ElementTree as SafeElementTree

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
    r"(?:api[_-]?key|key|access[_-]?token|token|secret|password|credential|"
    r"authorization|auth|signature|sig|signed)",
    re.IGNORECASE,
)
_DEFAULT_USER_AGENT = (
    "PAISA-FYP/0.1 academic news research; contact via repository owner"
)
_DEFAULT_MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
_DEFAULT_MAX_DECODED_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_ENTRIES = 500
_DEFAULT_MAX_XML_DEPTH = 48
_DEFAULT_MAX_XML_ELEMENTS = 20_000
_DEFAULT_MAX_FIELD_CHARACTERS = 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


class _RssRejected(ValueError):
    """A deterministic, safe-to-persist provider rejection."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.safe_message = message


class _PinnedAddressAdapter(HTTPAdapter):
    """Keep the request hostname while connecting only to one validated IP."""

    def __init__(self, pinned_address: str) -> None:
        super().__init__(max_retries=0)
        self.pinned_address = str(_normalized_ip(pinned_address))

    def _pin_pool(self, pool: Any) -> Any:
        if getattr(pool, "_paisa_pinned_address", None) == self.pinned_address:
            return pool
        connection_class = pool.ConnectionCls
        pinned_address = self.pinned_address

        class PinnedConnection(connection_class):
            def _new_conn(self) -> Any:
                # Only the socket connection is pinned. Restore the original
                # hostname before urllib3 performs TLS SNI/certificate checks
                # or emits the HTTP Host header.
                original_dns_host = self._dns_host
                self._dns_host = pinned_address
                try:
                    return super()._new_conn()
                finally:
                    self._dns_host = original_dns_host

        pool.ConnectionCls = PinnedConnection
        pool._paisa_pinned_address = self.pinned_address
        return pool

    def get_connection_with_tls_context(
        self,
        request: Any,
        verify: Any,
        proxies: Mapping[str, str] | None = None,
        cert: Any = None,
    ) -> Any:
        pool = super().get_connection_with_tls_context(
            request,
            verify,
            proxies={},
            cert=cert,
        )
        return self._pin_pool(pool)

    def get_connection(
        self,
        url: str,
        proxies: Mapping[str, str] | None = None,
    ) -> Any:
        # Requests before 2.32 calls this compatibility path.
        pool = super().get_connection(url, proxies={})
        return self._pin_pool(pool)


def _safe_url_identity(value: str) -> str:
    """Return a useful URL identity without retaining any query value."""

    parsed = urlsplit(value)
    redacted_query = urlencode(
        [(key, "REDACTED") for key, _ in parse_qsl(parsed.query, keep_blank_values=True)]
    )
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, redacted_query, "")
    )


def _safe_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    if parsed.scheme.casefold() in {"http", "https"} and parsed.netloc:
        return _safe_url_identity(value)
    return value


def _normalized_ip(value: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    parsed = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(parsed, ipaddress.IPv6Address) and parsed.ipv4_mapped is not None:
        return parsed.ipv4_mapped
    return parsed


def _resolve_addresses(host: str, port: int) -> tuple[str, ...]:
    addresses = {
        str(_normalized_ip(sockaddr[0]))
        for _family, _type, _proto, _canonname, sockaddr in socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    }
    return tuple(sorted(addresses))


def _response_peer_address(response: Any) -> str:
    raw = getattr(response, "raw", None)
    for connection in (
        getattr(raw, "_connection", None),
        getattr(raw, "connection", None),
    ):
        sock = getattr(connection, "sock", None)
        if sock is not None:
            peer = sock.getpeername()
            if peer:
                return str(peer[0])
    raise _RssRejected(
        "unsafe_destination",
        "RSS connected peer could not be verified",
    )


def _positive_number(value: int | float, *, field: str) -> None:
    if value <= 0:
        raise ValueError(f"{field} must be positive")


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


def _bounded_text(
    value: str | None,
    *,
    field: str,
    max_characters: int,
) -> str | None:
    if value is not None and len(value) > max_characters:
        raise _RssRejected(
            "feed_limit_exceeded",
            f"RSS field exceeds configured character limit: {field}",
        )
    return value


def _validate_xml_tree(
    root: ET.Element,
    *,
    max_depth: int,
    max_elements: int,
) -> None:
    seen = 0
    pending = [(root, 1)]
    while pending:
        element, depth = pending.pop()
        seen += 1
        if seen > max_elements:
            raise _RssRejected(
                "feed_limit_exceeded",
                "RSS document exceeds configured element limit",
            )
        if depth > max_depth:
            raise _RssRejected(
                "feed_limit_exceeded",
                "RSS document exceeds configured XML depth limit",
            )
        pending.extend((child, depth + 1) for child in element)


@dataclass(frozen=True)
class RssSourceConfig:
    """One explicitly permitted RSS/Atom endpoint."""

    source_id: str
    source_name: str
    url: str
    language: str | None = None
    default_timezone: str | None = None
    source_type: str | None = None
    allow_private_network: bool = False

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
        if not isinstance(self.allow_private_network, bool):
            raise ValueError("RSS allow_private_network must be true or false")
        try:
            literal_address = _normalized_ip(parsed.hostname or "")
        except ValueError:
            literal_address = None
        if (
            literal_address is not None
            and not literal_address.is_global
            and not self.allow_private_network
        ):
            raise ValueError(
                "RSS source URL must use a public destination unless private "
                "network access is explicitly enabled"
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
            allow_private_network=value.get("allow_private_network", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "url": _safe_url_identity(self.url),
            "url_sha256": hashlib.sha256(self.url.encode("utf-8")).hexdigest(),
            "language": self.language,
            "default_timezone": self.default_timezone,
            "source_type": self.source_type,
            "allow_private_network": self.allow_private_network,
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
    max_entries: int = _DEFAULT_MAX_ENTRIES,
    max_xml_depth: int = _DEFAULT_MAX_XML_DEPTH,
    max_xml_elements: int = _DEFAULT_MAX_XML_ELEMENTS,
    max_field_characters: int = _DEFAULT_MAX_FIELD_CHARACTERS,
) -> tuple[list[RawNewsRecord], list[str]]:
    try:
        root = SafeElementTree.fromstring(payload, forbid_dtd=True)
    except DefusedXmlException as exc:
        raise _RssRejected(
            "unsafe_xml",
            "RSS document contains a prohibited DTD or entity declaration",
        ) from exc
    _validate_xml_tree(
        root,
        max_depth=max_xml_depth,
        max_elements=max_xml_elements,
    )
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

    if len(entries) > max_entries:
        raise _RssRejected(
            "feed_limit_exceeded",
            "RSS document exceeds configured entry limit",
        )

    feed_warnings: list[str] = []
    if not entries:
        feed_warnings.append("feed_contains_no_entries")

    records: list[RawNewsRecord] = []
    for index, entry in enumerate(entries):
        warnings: list[str] = []
        title = _bounded_text(
            _direct_text(entry, "title"),
            field="title",
            max_characters=max_field_characters,
        )
        if source_type == "atom":
            raw_link = _bounded_text(
                _atom_link(entry),
                field="source_url",
                max_characters=max_field_characters,
            )
            summary = _bounded_text(
                _direct_text(entry, "summary"),
                field="summary",
                max_characters=max_field_characters,
            )
            body = _bounded_text(
                _direct_text(entry, "content"),
                field="body",
                max_characters=max_field_characters,
            )
            published = _bounded_text(
                _direct_text(entry, "published"),
                field="published_at",
                max_characters=max_field_characters,
            )
            available = _bounded_text(
                _direct_text(entry, "updated"),
                field="available_at",
                max_characters=max_field_characters,
            )
            if published is None and available is not None:
                warnings.append(
                    "atom_published_missing_used_updated_as_availability"
                )
            provider_record_id = _safe_identifier(_bounded_text(
                _direct_text(entry, "id"),
                field="provider_record_id",
                max_characters=max_field_characters,
            ))
            language = _optional_text(
                entry.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            )
        else:
            raw_link = _bounded_text(
                _direct_text(entry, "link"),
                field="source_url",
                max_characters=max_field_characters,
            )
            summary = _bounded_text(
                _direct_text(entry, "description", "summary"),
                field="summary",
                max_characters=max_field_characters,
            )
            body = _bounded_text(
                _direct_text(entry, "encoded", "content"),
                field="body",
                max_characters=max_field_characters,
            )
            published = _bounded_text(
                _direct_text(entry, "pubdate", "published", "date"),
                field="published_at",
                max_characters=max_field_characters,
            )
            available = None
            provider_record_id = _safe_identifier(_bounded_text(
                _direct_text(entry, "guid", "id"),
                field="provider_record_id",
                max_characters=max_field_characters,
            ))
            language = _direct_text(entry, "language")

        source_url = (
            _safe_url_identity(urljoin(source.url, raw_link)) if raw_link else None
        )
        if title is None:
            warnings.append("missing_title")
        if source_url is None:
            warnings.append("missing_source_url")
        if published is None:
            warnings.append("missing_publication_time")

        raw_payload = {
            "provider_record_id": provider_record_id,
            "source_url": _safe_url_identity(raw_link) if raw_link else None,
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
                configured_source=_safe_url_identity(source.url),
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


def _iter_wire_chunks(response: Any) -> Any:
    raw = getattr(response, "raw", None)
    stream = getattr(raw, "stream", None)
    if callable(stream):
        yield from stream(_READ_CHUNK_BYTES, decode_content=False)
        return
    yield bytes(response.content)


def _append_limited(
    destination: list[bytes],
    chunk: bytes,
    *,
    current_size: int,
    limit: int,
) -> int:
    updated = current_size + len(chunk)
    if updated > limit:
        raise _RssRejected(
            "response_limit_exceeded",
            "RSS response exceeds configured decoded-byte limit",
        )
    destination.append(chunk)
    return updated


def _read_bounded_payload(
    response: Any,
    *,
    max_compressed_bytes: int,
    max_decoded_bytes: int,
    deadline: float,
    monotonic_clock: Callable[[], float],
) -> bytes:
    content_length = _optional_text(response.headers.get("Content-Length"))
    if content_length is not None:
        try:
            declared_size = int(content_length)
        except ValueError as exc:
            raise _RssRejected(
                "invalid_response_metadata",
                "RSS response Content-Length is invalid",
            ) from exc
        if declared_size < 0 or declared_size > max_compressed_bytes:
            raise _RssRejected(
                "response_limit_exceeded",
                "RSS response exceeds configured compressed-byte limit",
            )

    encoding = _optional_text(response.headers.get("Content-Encoding"))
    normalized_encoding = (encoding or "identity").casefold().strip()
    if "," in normalized_encoding or normalized_encoding not in {
        "identity",
        "gzip",
        "x-gzip",
        "deflate",
    }:
        raise _RssRejected(
            "unsupported_content_encoding",
            "RSS response uses an unsupported Content-Encoding",
        )

    decompressor: zlib.Decompress | None
    if normalized_encoding in {"gzip", "x-gzip"}:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif normalized_encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        decompressor = None

    decoded_chunks: list[bytes] = []
    compressed_size = 0
    decoded_size = 0
    try:
        for raw_chunk in _iter_wire_chunks(response):
            if monotonic_clock() > deadline:
                raise _RssRejected(
                    "total_timeout",
                    "RSS acquisition exceeded the configured total deadline",
                )
            chunk = bytes(raw_chunk)
            compressed_size += len(chunk)
            if compressed_size > max_compressed_bytes:
                raise _RssRejected(
                    "response_limit_exceeded",
                    "RSS response exceeds configured compressed-byte limit",
                )
            if decompressor is None:
                decoded = chunk
            else:
                remaining = max_decoded_bytes - decoded_size
                decoded = decompressor.decompress(chunk, max(remaining + 1, 1))
            decoded_size = _append_limited(
                decoded_chunks,
                decoded,
                current_size=decoded_size,
                limit=max_decoded_bytes,
            )
            if decompressor is not None and decompressor.unconsumed_tail:
                raise _RssRejected(
                    "response_limit_exceeded",
                    "RSS response exceeds configured decoded-byte limit",
                )
        if decompressor is not None:
            remaining = max_decoded_bytes - decoded_size
            final = decompressor.flush(max(remaining + 1, 1))
            decoded_size = _append_limited(
                decoded_chunks,
                final,
                current_size=decoded_size,
                limit=max_decoded_bytes,
            )
            if not decompressor.eof or decompressor.unused_data:
                raise _RssRejected(
                    "invalid_content_encoding",
                    "RSS response compression stream is incomplete or ambiguous",
                )
    except zlib.error as exc:
        raise _RssRejected(
            "invalid_content_encoding",
            "RSS response compression stream is invalid",
        ) from exc

    if monotonic_clock() > deadline:
        raise _RssRejected(
            "total_timeout",
            "RSS acquisition exceeded the configured total deadline",
        )
    return b"".join(decoded_chunks)


def _validated_destination_addresses(
    source: RssSourceConfig,
    *,
    resolver: Callable[[str, int], Sequence[str]],
) -> tuple[str, ...]:
    parsed = urlsplit(source.url)
    host = parsed.hostname
    if host is None:
        raise _RssRejected("unsafe_destination", "RSS source host is missing")
    port = parsed.port or (443 if parsed.scheme.casefold() == "https" else 80)
    try:
        addresses = tuple(
            sorted({str(_normalized_ip(value)) for value in resolver(host, port)})
        )
    except (OSError, ValueError) as exc:
        raise _RssRejected(
            "unsafe_destination",
            "RSS source hostname could not be resolved safely",
        ) from exc
    if not addresses:
        raise _RssRejected(
            "unsafe_destination",
            "RSS source hostname returned no usable address",
        )
    if not source.allow_private_network and any(
        not _normalized_ip(address).is_global for address in addresses
    ):
        raise _RssRejected(
            "unsafe_destination",
            "RSS source resolved to a non-public destination",
        )
    return addresses


def _validate_connected_peer(
    response: Any,
    *,
    allowed_addresses: Sequence[str],
    allow_private_network: bool,
    peer_address_getter: Callable[[Any], str],
) -> None:
    try:
        peer = _normalized_ip(peer_address_getter(response))
    except (OSError, ValueError, _RssRejected) as exc:
        if isinstance(exc, _RssRejected):
            raise
        raise _RssRejected(
            "unsafe_destination",
            "RSS connected peer could not be verified",
        ) from exc
    allowed = {_normalized_ip(value) for value in allowed_addresses}
    if peer not in allowed:
        raise _RssRejected(
            "unsafe_destination",
            "RSS connected peer does not match the validated DNS result",
        )
    if not allow_private_network and not peer.is_global:
        raise _RssRejected(
            "unsafe_destination",
            "RSS connected peer is not a public destination",
        )


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
        resolver: Callable[[str, int], Sequence[str]] | None = None,
        peer_address_getter: Callable[[Any], str] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        connect_timeout_seconds: float | None = None,
        read_timeout_seconds: float | None = None,
        total_timeout_seconds: float | None = None,
        max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES,
        max_decoded_bytes: int = _DEFAULT_MAX_DECODED_BYTES,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_xml_depth: int = _DEFAULT_MAX_XML_DEPTH,
        max_xml_elements: int = _DEFAULT_MAX_XML_ELEMENTS,
        max_field_characters: int = _DEFAULT_MAX_FIELD_CHARACTERS,
    ) -> None:
        if not sources:
            raise ValueError("At least one RSS source is required")
        ids = [source.source_id for source in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("RSS source_id values must be unique")
        _positive_number(timeout_seconds, field="timeout_seconds")
        if not user_agent.strip() or "\r" in user_agent or "\n" in user_agent:
            raise ValueError("user_agent must be a non-empty single-line value")
        selected_total_timeout = float(
            timeout_seconds
            if total_timeout_seconds is None
            else total_timeout_seconds
        )
        selected_connect_timeout = float(
            min(timeout_seconds, 10.0)
            if connect_timeout_seconds is None
            else connect_timeout_seconds
        )
        selected_read_timeout = float(
            min(timeout_seconds, 15.0)
            if read_timeout_seconds is None
            else read_timeout_seconds
        )
        for value, field in (
            (selected_connect_timeout, "connect_timeout_seconds"),
            (selected_read_timeout, "read_timeout_seconds"),
            (selected_total_timeout, "total_timeout_seconds"),
            (max_compressed_bytes, "max_compressed_bytes"),
            (max_decoded_bytes, "max_decoded_bytes"),
            (max_entries, "max_entries"),
            (max_xml_depth, "max_xml_depth"),
            (max_xml_elements, "max_xml_elements"),
            (max_field_characters, "max_field_characters"),
        ):
            _positive_number(value, field=field)
        self.sources = tuple(sources)
        self.configured_sources = tuple(source.to_dict() for source in self.sources)
        self._owns_session = session is None
        self.session = session or requests.Session()
        if self._owns_session:
            self.session.trust_env = False
            self.session.auth = None
            self.session.headers.pop("Authorization", None)
            self.session.headers.pop("Cookie", None)
            self.session.cookies.clear()
        self.timeout_seconds = float(timeout_seconds)
        self.connect_timeout_seconds = selected_connect_timeout
        self.read_timeout_seconds = selected_read_timeout
        self.total_timeout_seconds = selected_total_timeout
        self.max_compressed_bytes = int(max_compressed_bytes)
        self.max_decoded_bytes = int(max_decoded_bytes)
        self.max_entries = int(max_entries)
        self.max_xml_depth = int(max_xml_depth)
        self.max_xml_elements = int(max_xml_elements)
        self.max_field_characters = int(max_field_characters)
        self.user_agent = user_agent.strip()
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.resolver = (
            resolver
            or getattr(self.session, "resolve_addresses", None)
            or _resolve_addresses
        )
        self.peer_address_getter = (
            peer_address_getter
            or getattr(self.session, "peer_address", None)
            or _response_peer_address
        )
        self.effective_config = {
            "timeout_seconds": self.timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "total_timeout_seconds": self.total_timeout_seconds,
            "max_compressed_bytes": self.max_compressed_bytes,
            "max_decoded_bytes": self.max_decoded_bytes,
            "max_entries": self.max_entries,
            "max_xml_depth": self.max_xml_depth,
            "max_xml_elements": self.max_xml_elements,
            "max_field_characters": self.max_field_characters,
            "user_agent": self.user_agent,
            "redirect_policy": "reject",
            "network_policy": "public_only_unless_source_opt_in",
            "environment_authentication": "disabled",
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
        float_options = (
            "connect_timeout_seconds",
            "read_timeout_seconds",
            "total_timeout_seconds",
        )
        integer_options = (
            "max_compressed_bytes",
            "max_decoded_bytes",
            "max_entries",
            "max_xml_depth",
            "max_xml_elements",
            "max_field_characters",
        )
        for name in float_options:
            if name not in options and name in document:
                options[name] = float(document[name])
        for name in integer_options:
            if name not in options and name in document:
                options[name] = int(document[name])
        return cls(sources, **options)

    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult:
        captures: list[RawCapture] = []
        records: list[RawNewsRecord] = []
        errors: list[ProviderError] = []
        observed_times: list[datetime] = []

        for source in self.sources:
            capture_id = f"rss-{source.source_id}"
            configured_source = _safe_url_identity(source.url)
            response: Any | None = None
            pinned_adapter: _PinnedAddressAdapter | None = None
            deadline = self.monotonic_clock() + self.total_timeout_seconds
            try:
                allowed_addresses = _validated_destination_addresses(
                    source,
                    resolver=self.resolver,
                )
                if self.monotonic_clock() > deadline:
                    raise _RssRejected(
                        "total_timeout",
                        "RSS acquisition exceeded the configured total deadline",
                    )
                if self._owns_session:
                    self.session.cookies.clear()
                    scheme = urlsplit(source.url).scheme.casefold()
                    pinned_adapter = _PinnedAddressAdapter(allowed_addresses[0])
                    self.session.mount(f"{scheme}://", pinned_adapter)
                response = self.session.get(
                    source.url,
                    timeout=(
                        self.connect_timeout_seconds,
                        self.read_timeout_seconds,
                    ),
                    allow_redirects=False,
                    stream=True,
                    proxies={},
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": (
                            "application/atom+xml,application/rss+xml,"
                            "application/xml,text/xml;q=0.9,*/*;q=0.1"
                        ),
                    },
                )
                if self.monotonic_clock() > deadline:
                    raise _RssRejected(
                        "total_timeout",
                        "RSS acquisition exceeded the configured total deadline",
                    )
                _validate_connected_peer(
                    response,
                    allowed_addresses=allowed_addresses,
                    allow_private_network=source.allow_private_network,
                    peer_address_getter=self.peer_address_getter,
                )
                payload = _read_bounded_payload(
                    response,
                    max_compressed_bytes=self.max_compressed_bytes,
                    max_decoded_bytes=self.max_decoded_bytes,
                    deadline=deadline,
                    monotonic_clock=self.monotonic_clock,
                )
                # Timestamp the evidence only after the response bytes have
                # actually been acquired.  ``fetched_at`` is an explicit test
                # hook; production callers leave it unset.
                acquisition_time = utc_datetime(fetched_at or self.clock())
                observed_times.append(acquisition_time)
            except _RssRejected as exc:
                acquisition_time = utc_datetime(fetched_at or self.clock())
                observed_times.append(acquisition_time)
                errors.append(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type=exc.error_type,
                        message=exc.safe_message,
                        occurred_at=acquisition_time,
                    )
                )
                continue
            except Exception as exc:
                acquisition_time = utc_datetime(fetched_at or self.clock())
                observed_times.append(acquisition_time)
                errors.append(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type="network_error",
                        message=(
                            "RSS network request failed "
                            f"({type(exc).__name__})"
                        ),
                        occurred_at=acquisition_time,
                    )
                )
                continue
            finally:
                if response is not None:
                    close = getattr(response, "close", None)
                    if callable(close):
                        close()
                if self._owns_session:
                    self.session.cookies.clear()
                if pinned_adapter is not None:
                    pinned_adapter.close()

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
                        configured_source=configured_source,
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
                        configured_source=configured_source,
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
                    max_entries=self.max_entries,
                    max_xml_depth=self.max_xml_depth,
                    max_xml_elements=self.max_xml_elements,
                    max_field_characters=self.max_field_characters,
                )
                source_warnings.extend(parse_warnings)
                actual_source_types = {record.source_type for record in parsed_records}
                if source.source_type and actual_source_types != {source.source_type}:
                    source_warnings.append("configured_source_type_mismatch")
                records.extend(parsed_records)
            except _RssRejected as exc:
                errors.append(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type=exc.error_type,
                        message=exc.safe_message,
                        occurred_at=acquisition_time,
                        http_status=status,
                    )
                )
            except (ET.ParseError, ValueError):
                errors.append(
                    ProviderError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type="feed_parse_error",
                        message="RSS feed could not be parsed safely",
                        occurred_at=acquisition_time,
                        http_status=status,
                    )
                )

            captures.append(
                RawCapture(
                    capture_id=capture_id,
                    provider_id=self.provider_id,
                    configured_source=configured_source,
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
