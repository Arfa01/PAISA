"""Bounded, metadata-only historical acquisition from GDELT DOC 2.0.

The adapter is deliberately limited to the public DOC 2.0 ``artlist`` JSON
endpoint.  Configured inclusive ranges are proactively covered by overlapping
three-month request tiles because ArticleList only considers the final three
months of a request.  GDELT's request bounds are exclusive, so PAISA expands
the outer bounds and deduplicates records repeated at tile or split boundaries.
A tile reaching the record cap is bisected until complete or failed closed.
The adapter never follows article URLs and never treats GDELT ``seendate`` as
a publisher-supplied publication timestamp.
"""

from __future__ import annotations

import calendar
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import zlib

import requests
from requests.adapters import HTTPAdapter

from paisa.news.models import (
    NEWS_AVAILABILITY_POLICY_VERSION,
    ProviderError,
    ProviderFetchResult,
    RawCapture,
    RawNewsRecord,
)
from paisa.news.providers.base import NewsProvider, sha256_bytes, utc_datetime


GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
_WINDOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?(?:key|token)|token|secret|password|"
    r"credential|authorization|auth|signature|signed|session|cookie|jwt)(?:$|[_-])",
    re.IGNORECASE,
)
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
_DEFAULT_USER_AGENT = (
    "PAISA-FYP/0.1 academic historical-news research; contact via repository owner"
)
_DEFAULT_MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
_DEFAULT_MAX_DECODED_BYTES = 8 * 1024 * 1024
_DEFAULT_MAX_FIELD_CHARACTERS = 100_000
_MAX_WINDOWS = 128
_MAX_RECORDS = 250
_MAX_QUERY_CHARACTERS = 2_048
_MAX_COMPRESSED_BYTES = 8 * 1024 * 1024
_MAX_DECODED_BYTES = 16 * 1024 * 1024
_MAX_FIELD_CHARACTERS = 1024 * 1024
_MAX_TIMEOUT_SECONDS = 120.0
_MAX_RETRIES = 5
_MAX_SPLIT_DEPTH = 24
_MAX_BACKOFF_SECONDS = 300.0
_MAX_REQUESTS_PER_RUN = 4_096
_GDELT_EARLIEST = datetime(2017, 1, 1, tzinfo=timezone.utc)
_COVERAGE_SEMANTICS = (
    "GDELT DOC ArticleList is non-exhaustive, has no pagination, caps responses "
    "at maxrecords, and considers at most the last three months of each request; "
    "PAISA treats configured bounds as inclusive, expands the API's exclusive "
    "bounds by one second, overlaps tiles and splits, deduplicates boundary "
    "records, and bisects saturated tiles."
)
_READ_CHUNK_BYTES = 64 * 1024
_TOP_LEVEL_KEYS = {
    "schema_version",
    "provider",
    "endpoint",
    "query",
    "windows",
    "source_name",
    "permission_basis",
    "licensing_usage_note",
    "attribution_text",
    "coverage_semantics",
    "max_records_per_window",
    "connect_timeout_seconds",
    "read_timeout_seconds",
    "soft_total_timeout_seconds",
    "max_compressed_bytes",
    "max_decoded_bytes",
    "max_field_characters",
    "minimum_interval_seconds",
    "minimum_window_seconds",
    "max_split_depth",
    "max_retries",
    "max_requests_per_run",
    "backoff_initial_seconds",
    "backoff_max_seconds",
    "user_agent",
}
_WINDOW_KEYS = {"window_id", "start_datetime", "end_datetime", "enabled"}


class _GdeltRejected(ValueError):
    """A deterministic provider rejection whose message is safe to persist."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.safe_message = message


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _single_line(value: Any, *, field: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} cannot be blank")
    if "\r" in text or "\n" in text:
        raise ValueError(f"{field} must be a single-line value")
    if len(text) > maximum:
        raise ValueError(f"{field} exceeds its maximum length")
    return text


def _bounded_positive_number(
    value: int | float,
    *,
    field: str,
    maximum: int | float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0 or number > maximum:
        raise ValueError(f"{field} must be positive and at most {maximum}")
    return number


def _bounded_positive_integer(value: Any, *, field: str, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field} must be an integer in [1, {maximum}]")
    return value


def _parse_utc_datetime(value: Any, *, field: str) -> datetime:
    text = _single_line(value, field=field, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include timezone evidence")
    return parsed.astimezone(timezone.utc)


def _utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _gdelt_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def _safe_url_identity(value: str) -> str | None:
    """Retain URL identity while removing credentials, tracking, and fragments."""

    try:
        parsed = urlsplit(value.strip())
        parsed.port
    except ValueError:
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not _SENSITIVE_QUERY_KEY.search(key)
            and not key.casefold().startswith("utm_")
            and key.casefold() not in _TRACKING_QUERY_KEYS
        ]
    )
    return urlunsplit((parsed.scheme.casefold(), host, parsed.path, query, ""))


def _bounded_text(value: Any, *, field: str, maximum: int) -> str | None:
    text = _optional_text(value)
    if text is not None and len(text) > maximum:
        raise _GdeltRejected(
            "field_limit_exceeded",
            f"GDELT field exceeds configured character limit: {field}",
        )
    return text


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


def _resolve_addresses_in_cancellable_subprocess(
    host: str,
    port: int,
    *,
    timeout_seconds: float,
) -> tuple[str, ...]:
    """Resolve with an OS process that can be terminated at the deadline."""

    program = (
        "import json,socket,sys;"
        "host=sys.argv[1];port=int(sys.argv[2]);"
        "values=sorted({row[4][0] for row in "
        "socket.getaddrinfo(host,port,type=socket.SOCK_STREAM)});"
        "print(json.dumps(values,separators=(',',':')))"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program, host, str(port)],
            capture_output=True,
            text=True,
            timeout=max(0.001, timeout_seconds),
            check=True,
        )
        value = json.loads(completed.stdout)
    except subprocess.TimeoutExpired as exc:
        raise _GdeltRejected(
            "soft_total_timeout",
            "GDELT DNS resolution exceeded the end-to-end request deadline",
        ) from exc
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise _GdeltRejected(
            "unsafe_destination",
            "GDELT hostname could not be resolved safely",
        ) from exc
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise _GdeltRejected(
            "unsafe_destination",
            "GDELT hostname resolver returned invalid evidence",
        )
    return tuple(value)


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
    raise _GdeltRejected(
        "unsafe_destination",
        "GDELT connected peer could not be verified",
    )


class _PinnedAddressAdapter(HTTPAdapter):
    """Keep GDELT's hostname for TLS while pinning the validated socket IP."""

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
        return self._pin_pool(super().get_connection(url, proxies={}))


def _validated_addresses(
    resolver: Callable[[str, int], Sequence[str]],
    *,
    timeout_seconds: float,
) -> tuple[str, ...]:
    host = urlsplit(GDELT_DOC_ENDPOINT).hostname
    assert host is not None
    if resolver is _resolve_addresses:
        resolved = _resolve_addresses_in_cancellable_subprocess(
            host,
            443,
            timeout_seconds=timeout_seconds,
        )
    else:
        resolved = _bounded_custom_resolution(
            resolver,
            host=host,
            port=443,
            timeout_seconds=timeout_seconds,
        )
    try:
        addresses = tuple(
            sorted({str(_normalized_ip(value)) for value in resolved})
        )
    except (OSError, ValueError) as exc:
        raise _GdeltRejected(
            "unsafe_destination",
            "GDELT hostname could not be resolved safely",
        ) from exc
    if not addresses or any(not _normalized_ip(item).is_global for item in addresses):
        raise _GdeltRejected(
            "unsafe_destination",
            "GDELT endpoint did not resolve exclusively to public addresses",
        )
    return addresses


def _bounded_custom_resolution(
    resolver: Callable[[str, int], Sequence[str]],
    *,
    host: str,
    port: int,
    timeout_seconds: float,
) -> Sequence[str]:
    """Bound injected/test resolvers without letting them block acquisition."""

    result: list[Sequence[str]] = []
    failure: list[BaseException] = []

    def resolve() -> None:
        try:
            result.append(resolver(host, port))
        except BaseException as exc:  # propagated on the acquisition thread
            failure.append(exc)

    worker = threading.Thread(
        target=resolve,
        name="paisa-gdelt-dns",
        daemon=True,
    )
    worker.start()
    worker.join(max(0.0, timeout_seconds))
    if worker.is_alive():
        raise _GdeltRejected(
            "soft_total_timeout",
            "GDELT DNS resolution exceeded the end-to-end request deadline",
        )
    if failure:
        exc = failure[0]
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise exc
        raise _GdeltRejected(
            "unsafe_destination",
            "GDELT hostname could not be resolved safely",
        ) from exc
    return result[0]


def _call_before_deadline(
    action: Callable[[], Any],
    *,
    deadline: float,
    monotonic_clock: Callable[[], float],
    timeout_cleanup: Callable[[], Any] | None = None,
) -> Any:
    """Run one potentially blocking network operation within the shared deadline.

    Requests exposes separate connect/read timeouts and streamed reads can block
    between yielded chunks.  A daemon worker lets the acquisition caller fail at
    the single end-to-end deadline; best-effort cleanup closes the live session
    so the abandoned operation does not remain usable.
    """

    remaining = deadline - monotonic_clock()
    if remaining <= 0:
        raise _GdeltRejected(
            "soft_total_timeout",
            "GDELT acquisition exceeded the configured end-to-end deadline",
        )
    result: list[Any] = []
    failure: list[BaseException] = []
    finished = threading.Event()

    def invoke() -> None:
        try:
            result.append(action())
        except BaseException as exc:  # propagated on the acquisition thread
            failure.append(exc)
        finally:
            finished.set()

    worker = threading.Thread(
        target=invoke,
        name="paisa-gdelt-request",
        daemon=True,
    )
    worker.start()
    wait_budget = max(0.0, deadline - monotonic_clock())
    if not finished.wait(wait_budget):
        if timeout_cleanup is not None:
            def cleanup() -> None:
                try:
                    timeout_cleanup()
                except Exception:
                    pass

            # Cleanup is advisory after the hard acquisition deadline.  Never
            # wait for a provider/session close implementation that can itself
            # block; the abandoned request worker is already daemonized.
            threading.Thread(
                target=cleanup,
                name="paisa-gdelt-timeout-cleanup",
                daemon=True,
            ).start()
        raise _GdeltRejected(
            "soft_total_timeout",
            "GDELT acquisition exceeded the configured end-to-end deadline",
        )
    if failure:
        raise failure[0]
    if monotonic_clock() > deadline:
        raise _GdeltRejected(
            "soft_total_timeout",
            "GDELT acquisition exceeded the configured end-to-end deadline",
        )
    return result[0]


def _validate_peer(
    response: Any,
    *,
    allowed_addresses: Sequence[str],
    peer_address_getter: Callable[[Any], str],
) -> None:
    try:
        peer = _normalized_ip(peer_address_getter(response))
    except _GdeltRejected:
        raise
    except (OSError, ValueError) as exc:
        raise _GdeltRejected(
            "unsafe_destination",
            "GDELT connected peer could not be verified",
        ) from exc
    if peer not in {_normalized_ip(value) for value in allowed_addresses} or not peer.is_global:
        raise _GdeltRejected(
            "unsafe_destination",
            "GDELT connected peer does not match validated public DNS",
        )


def _iter_wire_chunks(response: Any) -> Any:
    raw = getattr(response, "raw", None)
    stream = getattr(raw, "stream", None)
    if callable(stream):
        yield from stream(_READ_CHUNK_BYTES, decode_content=False)
        return
    yield bytes(response.content)


def _read_bounded_payload(
    response: Any,
    *,
    max_compressed_bytes: int,
    max_decoded_bytes: int,
    deadline: float,
    monotonic_clock: Callable[[], float],
) -> tuple[bytes, bytes]:
    content_length = _optional_text(response.headers.get("Content-Length"))
    if content_length is not None:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise _GdeltRejected(
                "invalid_response_metadata",
                "GDELT response Content-Length is invalid",
            ) from exc
        if declared < 0 or declared > max_compressed_bytes:
            raise _GdeltRejected(
                "response_limit_exceeded",
                "GDELT response exceeds configured compressed-byte limit",
            )

    encoding = (_optional_text(response.headers.get("Content-Encoding")) or "identity").casefold()
    if "," in encoding or encoding not in {"identity", "gzip", "x-gzip", "deflate"}:
        raise _GdeltRejected(
            "unsupported_content_encoding",
            "GDELT response uses an unsupported Content-Encoding",
        )
    if encoding in {"gzip", "x-gzip"}:
        decompressor: zlib.Decompress | None = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        decompressor = None

    wire_output: list[bytes] = []
    decoded_output: list[bytes] = []
    compressed_size = 0
    decoded_size = 0
    try:
        for wire_chunk in _iter_wire_chunks(response):
            if monotonic_clock() > deadline:
                raise _GdeltRejected(
                    "soft_total_timeout",
                    "GDELT acquisition exceeded the configured soft deadline",
                )
            chunk = bytes(wire_chunk)
            wire_output.append(chunk)
            compressed_size += len(chunk)
            if compressed_size > max_compressed_bytes:
                raise _GdeltRejected(
                    "response_limit_exceeded",
                    "GDELT response exceeds configured compressed-byte limit",
                )
            if decompressor is None:
                decoded = chunk
            else:
                decoded = decompressor.decompress(
                    chunk,
                    max(max_decoded_bytes - decoded_size + 1, 1),
                )
            decoded_size += len(decoded)
            if decoded_size > max_decoded_bytes or (
                decompressor is not None and decompressor.unconsumed_tail
            ):
                raise _GdeltRejected(
                    "response_limit_exceeded",
                    "GDELT response exceeds configured decoded-byte limit",
                )
            decoded_output.append(decoded)
        if decompressor is not None:
            final = decompressor.flush(max(max_decoded_bytes - decoded_size + 1, 1))
            decoded_size += len(final)
            if decoded_size > max_decoded_bytes or not decompressor.eof or decompressor.unused_data:
                raise _GdeltRejected(
                    "invalid_content_encoding",
                    "GDELT response compression stream is incomplete or ambiguous",
                )
            decoded_output.append(final)
    except zlib.error as exc:
        raise _GdeltRejected(
            "invalid_content_encoding",
            "GDELT response compression stream is invalid",
        ) from exc
    if monotonic_clock() > deadline:
        raise _GdeltRejected(
            "soft_total_timeout",
            "GDELT acquisition exceeded the configured soft deadline",
        )
    return b"".join(wire_output), b"".join(decoded_output)


@dataclass(frozen=True)
class GdeltQueryWindow:
    window_id: str
    start_datetime: datetime
    end_datetime: datetime

    def __post_init__(self) -> None:
        if not _WINDOW_ID.fullmatch(self.window_id):
            raise ValueError("GDELT window_id is not filesystem-safe")
        start = utc_datetime(self.start_datetime)
        end = utc_datetime(self.end_datetime)
        if start >= end:
            raise ValueError("GDELT query window start must precede end")
        if start < _GDELT_EARLIEST:
            raise ValueError("GDELT DOC query windows cannot begin before 2017-01-01")
        if start.microsecond or end.microsecond:
            raise ValueError("GDELT query windows must use whole-second bounds")
        try:
            end + timedelta(seconds=1)
        except OverflowError as exc:
            raise ValueError(
                "GDELT query window end must permit exclusive-bound expansion"
            ) from exc
        object.__setattr__(self, "start_datetime", start)
        object.__setattr__(self, "end_datetime", end)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GdeltQueryWindow":
        unexpected = set(value) - _WINDOW_KEYS
        if unexpected:
            raise ValueError(
                "Unsupported GDELT window fields: " + ", ".join(sorted(unexpected))
            )
        return cls(
            window_id=_single_line(
                value.get("window_id"), field="window_id", maximum=100
            ),
            start_datetime=_parse_utc_datetime(
                value.get("start_datetime"), field="start_datetime"
            ),
            end_datetime=_parse_utc_datetime(
                value.get("end_datetime"), field="end_datetime"
            ),
        )

    def to_dict(self, *, query: str) -> dict[str, Any]:
        return {
            "configured_source": f"gdelt-doc:{self.window_id}",
            "provider": "GDELT DOC 2.0",
            "provider_mode": "historical_api",
            "endpoint": GDELT_DOC_ENDPOINT,
            "query": query,
            "link_evidence": {
                "endpoint": GDELT_DOC_ENDPOINT,
                "query": query,
                "match_mode": "full_text_query",
            },
            "coverage_semantics": _COVERAGE_SEMANTICS,
            "maximum_request_window_months": 3,
            "configured_window_bound_semantics": "inclusive",
            "api_request_bound_semantics": "exclusive",
            "window_id": self.window_id,
            "start_datetime": _utc_iso(self.start_datetime),
            "end_datetime": _utc_iso(self.end_datetime),
        }


@dataclass(frozen=True)
class _RequestWindow:
    root: GdeltQueryWindow
    start_datetime: datetime
    end_datetime: datetime
    split_path: str = "r"
    depth: int = 0

    @property
    def configured_source(self) -> str:
        return f"gdelt-doc:{self.root.window_id}"

    @property
    def request_start_datetime(self) -> datetime:
        """Exclusive API start bound that includes the target start second."""

        return self.start_datetime - timedelta(seconds=1)

    @property
    def request_end_datetime(self) -> datetime:
        """Exclusive API end bound that includes the target end second."""

        return self.end_datetime + timedelta(seconds=1)

    def capture_id(self, attempt: int) -> str:
        start = _gdelt_datetime(self.start_datetime)
        end = _gdelt_datetime(self.end_datetime)
        return (
            f"gdelt-{self.root.window_id}-{start}-{end}-"
            f"{self.split_path}-a{attempt + 1}"
        )

    def split(
        self,
        *,
        minimum_window_seconds: int,
    ) -> tuple["_RequestWindow", "_RequestWindow"] | None:
        span_seconds = int((self.end_datetime - self.start_datetime).total_seconds())
        if span_seconds < (2 * minimum_window_seconds) + 1:
            return None
        midpoint = self.start_datetime + timedelta(seconds=span_seconds // 2)
        if (
            (midpoint - self.start_datetime).total_seconds() < minimum_window_seconds
            or (self.end_datetime - midpoint).total_seconds()
            < minimum_window_seconds
        ):
            return None
        return (
            _RequestWindow(
                root=self.root,
                start_datetime=self.start_datetime,
                end_datetime=midpoint,
                split_path=f"{self.split_path}L",
                depth=self.depth + 1,
            ),
            _RequestWindow(
                root=self.root,
                start_datetime=midpoint,
                end_datetime=self.end_datetime,
                split_path=f"{self.split_path}R",
                depth=self.depth + 1,
            ),
        )


def _add_three_months(value: datetime) -> datetime:
    month_index = (value.year * 12) + (value.month - 1) + 3
    year, zero_based_month = divmod(month_index, 12)
    if year > datetime.max.year:
        return datetime.max.replace(tzinfo=value.tzinfo, microsecond=0)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _tile_root_window(root: GdeltQueryWindow) -> tuple[_RequestWindow, ...]:
    """Cover inclusive target bounds with overlapping exclusive API windows."""

    tiles: list[_RequestWindow] = []
    start = root.start_datetime
    index = 0
    while start <= root.end_datetime:
        exclusive_start = start - timedelta(seconds=1)
        latest_exclusive_end = _add_three_months(exclusive_start)
        latest_target_end = latest_exclusive_end - timedelta(seconds=1)
        end = min(root.end_datetime, latest_target_end)
        tiles.append(
            _RequestWindow(
                root=root,
                start_datetime=start,
                end_datetime=end,
                split_path=f"t{index:03d}",
            )
        )
        if end == root.end_datetime:
            break
        # Adjacent target tiles share their boundary second.  This is deliberate:
        # GDELT's API bounds are exclusive, and overlap avoids an unqueryable gap.
        start = end
        index += 1
    return tuple(tiles)

@dataclass(frozen=True)
class _RetryDelay:
    requested_seconds: float | None
    applied_seconds: float | None
    header_valid: bool
    exceeds_wait_ceiling: bool


def _retry_delay_seconds(
    *,
    attempt: int,
    retry_after: str | None,
    occurred_at: datetime,
    initial_seconds: float,
    maximum_seconds: float,
) -> _RetryDelay:
    delay = min(initial_seconds * (2**attempt), maximum_seconds)
    if retry_after is None:
        return _RetryDelay(None, delay, False, False)
    suggested: float | None = None
    try:
        numeric = float(retry_after)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            retry_at = None
        if retry_at is not None and retry_at.tzinfo is not None:
            suggested = max(
                0.0,
                (retry_at.astimezone(timezone.utc) - occurred_at).total_seconds(),
            )
    else:
        if math.isfinite(numeric) and numeric >= 0:
            suggested = numeric
    if suggested is None:
        return _RetryDelay(None, delay, False, False)
    if suggested > maximum_seconds:
        return _RetryDelay(suggested, None, True, True)
    delay = max(delay, suggested)
    return _RetryDelay(suggested, delay, True, False)


def _load_config(path: str | Path) -> tuple[Mapping[str, Any], tuple[GdeltQueryWindow, ...]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("GDELT configuration must be a JSON object")
    unexpected = set(value) - _TOP_LEVEL_KEYS
    if unexpected:
        raise ValueError(
            "Unsupported GDELT configuration fields: " + ", ".join(sorted(unexpected))
        )
    if any(_SENSITIVE_KEY.search(str(key)) for key in value):
        raise ValueError("GDELT configuration must not contain credentials")
    if value.get("schema_version", "1.0") != "1.0":
        raise ValueError("GDELT configuration schema_version must be '1.0'")
    if value.get("provider", "gdelt-doc") != "gdelt-doc":
        raise ValueError("GDELT configuration provider must be 'gdelt-doc'")
    if value.get("endpoint", GDELT_DOC_ENDPOINT) != GDELT_DOC_ENDPOINT:
        raise ValueError("GDELT configuration endpoint must be the official DOC 2.0 endpoint")
    _single_line(
        value.get("query"),
        field="GDELT query",
        maximum=_MAX_QUERY_CHARACTERS,
    )
    _single_line(
        value.get("permission_basis"),
        field="permission_basis",
        maximum=2_000,
    )
    _single_line(
        value.get("licensing_usage_note"),
        field="licensing_usage_note",
        maximum=2_000,
    )
    _single_line(
        value.get("attribution_text"),
        field="attribution_text",
        maximum=2_000,
    )
    configured_semantics = value.get("coverage_semantics")
    if configured_semantics is not None and configured_semantics != _COVERAGE_SEMANTICS:
        raise ValueError(
            "GDELT coverage_semantics must match the provider's fixed limitations"
        )
    windows = value.get("windows")
    if not isinstance(windows, list) or not windows:
        raise ValueError("GDELT configuration requires a non-empty windows array")
    if len(windows) > _MAX_WINDOWS:
        raise ValueError(f"GDELT configuration supports at most {_MAX_WINDOWS} windows")
    enabled: list[GdeltQueryWindow] = []
    for item in windows:
        if not isinstance(item, Mapping):
            raise ValueError("Every GDELT query window must be an object")
        flag = item.get("enabled", True)
        if not isinstance(flag, bool):
            raise ValueError("GDELT window enabled must be true or false")
        if flag:
            enabled.append(GdeltQueryWindow.from_mapping(item))
    if not enabled:
        raise ValueError("GDELT configuration requires at least one enabled window")
    identifiers = [window.window_id for window in enabled]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("GDELT window_id values must be unique")
    return value, tuple(enabled)


def load_gdelt_config(path: str | Path) -> tuple[GdeltQueryWindow, ...]:
    """Load enabled, timezone-aware query windows from a versioned config."""

    _, windows = _load_config(path)
    return windows


def _normalized_seendate(value: str | None) -> str | None:
    if value is None:
        return None
    formats = ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S")
    for pattern in formats:
        try:
            parsed = datetime.strptime(value, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        return _utc_iso(parsed)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return _utc_iso(parsed)


def _provider_record_id(
    *,
    source_url: str | None,
    title: str | None,
    domain: str | None,
    seendate: str | None,
) -> str:
    identity = json.dumps(
        {
            "domain": domain or "",
            "seendate": seendate or "",
            "title": title or "",
            "url": source_url or "",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"GDELT-{hashlib.sha256(identity).hexdigest()[:24]}"


def _parse_articles(
    payload: bytes,
    *,
    target_start: datetime,
    target_end: datetime,
    request_start: datetime,
    request_end: datetime,
    query: str,
    configured_source: str,
    capture_id: str,
    fetched_at: datetime,
    http_status: int,
    content_type: str | None,
    response_sha256: str,
    max_records: int,
    max_field_characters: int,
    permission_basis: str,
    licensing_usage_note: str,
    attribution_text: str,
) -> tuple[list[RawNewsRecord], list[str]]:
    try:
        document = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _GdeltRejected(
            "response_parse_error",
            "GDELT response is not valid UTF-8 JSON",
        ) from exc
    if not isinstance(document, Mapping) or not isinstance(document.get("articles"), list):
        raise _GdeltRejected(
            "response_parse_error",
            "GDELT response does not contain an articles array",
        )
    articles = document["articles"]
    if len(articles) > max_records:
        raise _GdeltRejected(
            "record_limit_exceeded",
            "GDELT response exceeds the configured record limit",
        )
    warnings: list[str] = []
    if not articles:
        warnings.append("query_window_contains_no_articles")
    if len(articles) == max_records:
        warnings.append("result_limit_reached_split_required")

    records: list[RawNewsRecord] = []
    for index, item in enumerate(articles):
        if not isinstance(item, Mapping):
            raise _GdeltRejected(
                "response_parse_error",
                "GDELT articles must be JSON objects",
            )
        title = _bounded_text(
            item.get("title"), field="title", maximum=max_field_characters
        )
        raw_url = _bounded_text(
            item.get("url"), field="url", maximum=max_field_characters
        )
        source_url = _safe_url_identity(raw_url) if raw_url else None
        raw_mobile_url = _bounded_text(
            item.get("url_mobile"),
            field="url_mobile",
            maximum=max_field_characters,
        )
        mobile_url = _safe_url_identity(raw_mobile_url) if raw_mobile_url else None
        domain = _bounded_text(
            item.get("domain"), field="domain", maximum=max_field_characters
        )
        language = _bounded_text(
            item.get("language"), field="language", maximum=max_field_characters
        )
        source_country = _bounded_text(
            item.get("sourcecountry"),
            field="sourcecountry",
            maximum=max_field_characters,
        )
        raw_seendate = _bounded_text(
            item.get("seendate"), field="seendate", maximum=64
        )
        source_observed_at = _normalized_seendate(raw_seendate)
        record_warnings = [
            "missing_publication_time",
            "gdelt_seendate_is_source_observation_not_publication_or_eligibility",
        ]
        if title is None:
            record_warnings.append("missing_title")
        if source_url is None:
            record_warnings.append("missing_source_url")
        if raw_seendate is None or source_observed_at is None:
            raise _GdeltRejected(
                "invalid_source_observation_time",
                "GDELT historical response contains a missing or malformed "
                "seendate; response quarantined because target-window "
                "membership cannot be proven",
            )
        seen = datetime.fromisoformat(source_observed_at.replace("Z", "+00:00"))
        if not target_start <= seen <= target_end:
            raise _GdeltRejected(
                "record_outside_target_window",
                "GDELT response contained seendate outside the inclusive "
                "target window; response quarantined",
            )

        provider_record_id = _provider_record_id(
            source_url=raw_url,
            title=title,
            domain=domain,
            seendate=raw_seendate,
        )
        raw_payload = {
            "provider_record_id": provider_record_id,
            "url": source_url,
            "source_url_sha256": (
                hashlib.sha256(raw_url.encode("utf-8")).hexdigest()
                if raw_url is not None
                else None
            ),
            "title": title,
            "domain": domain,
            "language": language,
            "sourcecountry": source_country,
            "seendate": raw_seendate,
            "seendate_normalized_utc": source_observed_at,
            "original_timezone_evidence": raw_seendate,
            "request_query": query,
            "provider_match_mode": "full_text_query",
            "target_start_datetime": _utc_iso(target_start),
            "target_end_datetime": _utc_iso(target_end),
            "request_start_datetime": _utc_iso(request_start),
            "request_end_datetime": _utc_iso(request_end),
            "configured_window_bound_semantics": "inclusive",
            "api_request_bound_semantics": "exclusive",
            "provider_mode": "historical_api",
            "permission_basis": permission_basis,
            "licensing_usage_note": licensing_usage_note,
            "attribution_text": attribution_text,
            "url_mobile": mobile_url,
            "url_mobile_sha256": (
                hashlib.sha256(raw_mobile_url.encode("utf-8")).hexdigest()
                if raw_mobile_url is not None
                else None
            ),
            "socialimage_omitted_by_policy": "socialimage" in item,
        }
        records.append(
            RawNewsRecord(
                raw_record_id=f"{capture_id}-{index:06d}",
                provider_id="gdelt-doc",
                configured_source=configured_source,
                source_name=domain or "GDELT DOC 2.0",
                source_type="gdelt_doc_archive",
                fetched_at=fetched_at,
                data_class="real",
                source_url=source_url,
                title=title,
                summary=None,
                body=None,
                published_at_raw=None,
                published_timezone=None,
                available_at_raw=None,
                source_observed_at_raw=source_observed_at,
                language=language,
                provider_record_id=provider_record_id,
                capture_id=capture_id,
                http_status=http_status,
                response_content_type=content_type,
                response_sha256=response_sha256,
                raw_payload=raw_payload,
                provider_warnings=tuple(record_warnings),
            )
        )
    return records, warnings


class GdeltDocNewsProvider(NewsProvider):
    """Query bounded historical windows from GDELT without fetching articles."""

    provider_id = "gdelt-doc"
    data_class = "real"

    def __init__(
        self,
        query: str,
        windows: Sequence[GdeltQueryWindow],
        *,
        source_name: str = "GDELT DOC 2.0",
        permission_basis: str,
        licensing_usage_note: str,
        attribution_text: str,
        max_records_per_window: int = _MAX_RECORDS,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 30.0,
        soft_total_timeout_seconds: float = 45.0,
        max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES,
        max_decoded_bytes: int = _DEFAULT_MAX_DECODED_BYTES,
        max_field_characters: int = _DEFAULT_MAX_FIELD_CHARACTERS,
        minimum_interval_seconds: float = 1.0,
        minimum_window_seconds: int = 60,
        max_split_depth: int = 20,
        max_retries: int = 2,
        max_requests_per_run: int = 512,
        backoff_initial_seconds: float = 1.0,
        backoff_max_seconds: float = 30.0,
        user_agent: str = _DEFAULT_USER_AGENT,
        session: Any | None = None,
        clock: Callable[[], datetime] = _utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        resolver: Callable[[str, int], Sequence[str]] | None = None,
        peer_address_getter: Callable[[Any], str] | None = None,
    ) -> None:
        self.query = _single_line(
            query, field="GDELT query", maximum=_MAX_QUERY_CHARACTERS
        )
        if not windows or len(windows) > _MAX_WINDOWS:
            raise ValueError(f"GDELT requires between 1 and {_MAX_WINDOWS} windows")
        identifiers = [window.window_id for window in windows]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("GDELT window_id values must be unique")
        if not isinstance(max_records_per_window, int) or isinstance(
            max_records_per_window, bool
        ) or not 1 <= max_records_per_window <= _MAX_RECORDS:
            raise ValueError(f"max_records_per_window must be in [1, {_MAX_RECORDS}]")
        selected_connect_timeout = _bounded_positive_number(
            connect_timeout_seconds,
            field="connect_timeout_seconds",
            maximum=_MAX_TIMEOUT_SECONDS,
        )
        selected_read_timeout = _bounded_positive_number(
            read_timeout_seconds,
            field="read_timeout_seconds",
            maximum=_MAX_TIMEOUT_SECONDS,
        )
        selected_soft_total_timeout = _bounded_positive_number(
            soft_total_timeout_seconds,
            field="soft_total_timeout_seconds",
            maximum=_MAX_TIMEOUT_SECONDS,
        )
        if selected_connect_timeout > selected_soft_total_timeout:
            raise ValueError(
                "connect_timeout_seconds cannot exceed soft_total_timeout_seconds"
            )
        if selected_read_timeout > selected_soft_total_timeout:
            raise ValueError(
                "read_timeout_seconds cannot exceed soft_total_timeout_seconds"
            )
        for value, field, maximum in (
            (max_compressed_bytes, "max_compressed_bytes", _MAX_COMPRESSED_BYTES),
            (max_decoded_bytes, "max_decoded_bytes", _MAX_DECODED_BYTES),
            (max_field_characters, "max_field_characters", _MAX_FIELD_CHARACTERS),
        ):
            _bounded_positive_integer(value, field=field, maximum=maximum)
        if isinstance(minimum_interval_seconds, bool) or not isinstance(
            minimum_interval_seconds, (int, float)
        ):
            raise ValueError("minimum_interval_seconds must be numeric")
        selected_interval = float(minimum_interval_seconds)
        if (
            not math.isfinite(selected_interval)
            or selected_interval < 0
            or selected_interval > 60
        ):
            raise ValueError("minimum_interval_seconds must be in [0, 60]")
        if (
            not isinstance(minimum_window_seconds, int)
            or isinstance(minimum_window_seconds, bool)
            or minimum_window_seconds < 1
        ):
            raise ValueError("minimum_window_seconds must be a positive integer")
        if (
            not isinstance(max_split_depth, int)
            or isinstance(max_split_depth, bool)
            or not 0 <= max_split_depth <= _MAX_SPLIT_DEPTH
        ):
            raise ValueError(
                f"max_split_depth must be in [0, {_MAX_SPLIT_DEPTH}]"
            )
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or not 0 <= max_retries <= _MAX_RETRIES
        ):
            raise ValueError(f"max_retries must be in [0, {_MAX_RETRIES}]")
        if (
            not isinstance(max_requests_per_run, int)
            or isinstance(max_requests_per_run, bool)
            or not 1 <= max_requests_per_run <= _MAX_REQUESTS_PER_RUN
        ):
            raise ValueError(
                "max_requests_per_run must be in "
                f"[1, {_MAX_REQUESTS_PER_RUN}]"
            )
        selected_backoff_initial = _bounded_positive_number(
            backoff_initial_seconds,
            field="backoff_initial_seconds",
            maximum=_MAX_BACKOFF_SECONDS,
        )
        selected_backoff_max = _bounded_positive_number(
            backoff_max_seconds,
            field="backoff_max_seconds",
            maximum=_MAX_BACKOFF_SECONDS,
        )
        if selected_backoff_initial > selected_backoff_max:
            raise ValueError(
                "backoff_initial_seconds cannot exceed backoff_max_seconds"
            )
        self.source_name = _single_line(source_name, field="source_name", maximum=200)
        self.permission_basis = _single_line(
            permission_basis, field="permission_basis", maximum=2_000
        )
        self.licensing_usage_note = _single_line(
            licensing_usage_note, field="licensing_usage_note", maximum=2_000
        )
        self.attribution_text = _single_line(
            attribution_text, field="attribution_text", maximum=2_000
        )
        self.windows = tuple(windows)
        self.max_records_per_window = max_records_per_window
        self.connect_timeout_seconds = selected_connect_timeout
        self.read_timeout_seconds = selected_read_timeout
        self.soft_total_timeout_seconds = selected_soft_total_timeout
        self.max_compressed_bytes = int(max_compressed_bytes)
        self.max_decoded_bytes = int(max_decoded_bytes)
        self.max_field_characters = int(max_field_characters)
        self.minimum_interval_seconds = selected_interval
        self.minimum_window_seconds = minimum_window_seconds
        self.max_split_depth = max_split_depth
        self.max_retries = max_retries
        self.max_requests_per_run = max_requests_per_run
        self.backoff_initial_seconds = selected_backoff_initial
        self.backoff_max_seconds = selected_backoff_max
        self.user_agent = _single_line(user_agent, field="user_agent", maximum=500)
        self.configured_sources = tuple(
            window.to_dict(query=self.query) for window in self.windows
        )
        self.effective_config = {
            "provider_mode": "historical_api",
            "endpoint": GDELT_DOC_ENDPOINT,
            "mode": "artlist",
            "format": "json",
            "sort": "dateasc",
            "source_name": self.source_name,
            "permission_basis": self.permission_basis,
            "licensing_usage_note": self.licensing_usage_note,
            "attribution_text": self.attribution_text,
            "coverage_semantics": _COVERAGE_SEMANTICS,
            "maximum_request_window_months": 3,
            "configured_window_bound_semantics": "inclusive",
            "api_request_bound_semantics": "exclusive",
            "api_request_outer_bound_expansion_seconds": 1,
            "internal_boundary_policy": "overlap_and_deduplicate",
            "max_records_per_window": self.max_records_per_window,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "read_timeout_seconds": self.read_timeout_seconds,
            "soft_total_timeout_seconds": self.soft_total_timeout_seconds,
            "soft_total_timeout_semantics": (
                "one end-to-end request deadline covers bounded DNS resolution, "
                "connect, response headers, and bounded payload reading"
            ),
            "max_compressed_bytes": self.max_compressed_bytes,
            "max_decoded_bytes": self.max_decoded_bytes,
            "max_field_characters": self.max_field_characters,
            "minimum_interval_seconds": self.minimum_interval_seconds,
            "minimum_window_seconds": self.minimum_window_seconds,
            "max_split_depth": self.max_split_depth,
            "max_retries": self.max_retries,
            "max_requests_per_run": self.max_requests_per_run,
            "backoff_initial_seconds": self.backoff_initial_seconds,
            "backoff_max_seconds": self.backoff_max_seconds,
            "user_agent": self.user_agent,
            "redirect_policy": "reject",
            "article_page_fetching": "disabled",
            "environment_authentication": "disabled",
            "saturation_policy": "deterministic_bisection_or_fail_closed",
            "source_observed_at_eligibility_policy": (
                "audit_only_never_eligibility_without_new_versioned_policy"
            ),
            "availability_policy_version": NEWS_AVAILABILITY_POLICY_VERSION,
            "out_of_target_seendate_policy": "quarantine_response",
            "missing_or_invalid_seendate_policy": "quarantine_response",
            "retry_policy": (
                "bounded_exponential_backoff_429_and_5xx_honor_retry_after_"
                "within_wait_ceiling_else_fail_window"
            ),
        }
        self._owns_session = session is None
        self.session = session or requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        if hasattr(self.session, "auth"):
            self.session.auth = None
        if hasattr(self.session, "cert"):
            self.session.cert = None
        session_headers = getattr(self.session, "headers", None)
        if session_headers is not None:
            for key in tuple(session_headers):
                if _SENSITIVE_KEY.search(str(key)):
                    session_headers.pop(key, None)
        session_params = getattr(self.session, "params", None)
        clear_params = getattr(session_params, "clear", None)
        if callable(clear_params):
            clear_params()
        session_proxies = getattr(self.session, "proxies", None)
        clear_proxies = getattr(session_proxies, "clear", None)
        if callable(clear_proxies):
            clear_proxies()
        session_cookies = getattr(self.session, "cookies", None)
        if session_cookies is not None:
            clear = getattr(session_cookies, "clear", None)
            if callable(clear):
                clear()
        self.clock = clock
        self.monotonic_clock = monotonic_clock
        self.sleeper = sleeper
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

    @classmethod
    def from_config(cls, path: str | Path, **kwargs: Any) -> "GdeltDocNewsProvider":
        document, windows = _load_config(path)
        options = dict(kwargs)
        for name in (
            "source_name",
            "permission_basis",
            "licensing_usage_note",
            "attribution_text",
            "user_agent",
        ):
            if name not in options and name in document:
                options[name] = document[name]
        for name in (
            "max_records_per_window",
            "max_compressed_bytes",
            "max_decoded_bytes",
            "max_field_characters",
            "minimum_window_seconds",
            "max_split_depth",
            "max_retries",
            "max_requests_per_run",
        ):
            if name not in options and name in document:
                options[name] = document[name]
        for name in (
            "connect_timeout_seconds",
            "read_timeout_seconds",
            "soft_total_timeout_seconds",
            "minimum_interval_seconds",
            "backoff_initial_seconds",
            "backoff_max_seconds",
        ):
            if name not in options and name in document:
                options[name] = document[name]
        return cls(str(document.get("query") or ""), windows, **options)

    def _request_parameters(self, window: _RequestWindow) -> dict[str, str | int]:
        return {
            "query": self.query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": self.max_records_per_window,
            "startdatetime": _gdelt_datetime(window.request_start_datetime),
            "enddatetime": _gdelt_datetime(window.request_end_datetime),
            "sort": "dateasc",
        }

    def _request_context(
        self,
        window: _RequestWindow,
        attempt: int,
        **extra: Any,
    ) -> dict[str, Any]:
        parameters = self._request_parameters(window)
        encoded = json.dumps(
            parameters,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        context: dict[str, Any] = {
            "capture_id": window.capture_id(attempt),
            "split_path": window.split_path,
            "target_start_datetime": _utc_iso(window.start_datetime),
            "target_end_datetime": _utc_iso(window.end_datetime),
            "request_start_datetime": _utc_iso(window.request_start_datetime),
            "request_end_datetime": _utc_iso(window.request_end_datetime),
            "attempt_number": attempt + 1,
            "query_sha256": hashlib.sha256(self.query.encode("utf-8")).hexdigest(),
            "request_parameters_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        context.update(extra)
        return context

    def _acquire(
        self,
        window: _RequestWindow,
        *,
        fetched_at: datetime | None,
    ) -> tuple[bytes, bytes, int, Mapping[str, Any], datetime]:
        pinned_adapter: _PinnedAddressAdapter | None = None
        deadline = self.monotonic_clock() + self.soft_total_timeout_seconds
        try:
            resolver_budget = deadline - self.monotonic_clock()
            if resolver_budget <= 0:
                raise _GdeltRejected(
                    "soft_total_timeout",
                    "GDELT acquisition exceeded the configured end-to-end deadline",
                )
            allowed_addresses = _validated_addresses(
                self.resolver,
                timeout_seconds=resolver_budget,
            )
            if self.monotonic_clock() > deadline:
                raise _GdeltRejected(
                    "soft_total_timeout",
                    "GDELT acquisition exceeded the configured soft deadline",
                )
            session_cookies = getattr(self.session, "cookies", None)
            clear_cookies = getattr(session_cookies, "clear", None)
            if callable(clear_cookies):
                clear_cookies()
            mount = getattr(self.session, "mount", None)
            if callable(mount):
                pinned_adapter = _PinnedAddressAdapter(allowed_addresses[0])
                mount("https://", pinned_adapter)
            remaining_seconds = deadline - self.monotonic_clock()
            if remaining_seconds <= 0:
                raise _GdeltRejected(
                    "soft_total_timeout",
                    "GDELT acquisition exceeded the configured soft deadline",
                )
            def request_and_read() -> tuple[bytes, bytes, int, Mapping[str, Any]]:
                response: Any | None = None
                try:
                    response = self.session.get(
                        GDELT_DOC_ENDPOINT,
                        params=self._request_parameters(window),
                        timeout=(
                            min(self.connect_timeout_seconds, remaining_seconds),
                            min(self.read_timeout_seconds, remaining_seconds),
                        ),
                        allow_redirects=False,
                        stream=True,
                        proxies={},
                        headers={
                            "User-Agent": self.user_agent,
                            "Accept": "application/json",
                        },
                    )
                    _validate_peer(
                        response,
                        allowed_addresses=allowed_addresses,
                        peer_address_getter=self.peer_address_getter,
                    )
                    wire_payload, decoded_payload = _read_bounded_payload(
                        response,
                        max_compressed_bytes=self.max_compressed_bytes,
                        max_decoded_bytes=self.max_decoded_bytes,
                        deadline=deadline,
                        monotonic_clock=self.monotonic_clock,
                    )
                    return (
                        wire_payload,
                        decoded_payload,
                        int(response.status_code),
                        {
                            str(key).casefold(): value
                            for key, value in response.headers.items()
                        },
                    )
                finally:
                    if response is not None:
                        close = getattr(response, "close", None)
                        if callable(close):
                            close()

            close_session = getattr(self.session, "close", None)
            wire_payload, decoded_payload, status_code, headers = (
                _call_before_deadline(
                    request_and_read,
                    deadline=deadline,
                    monotonic_clock=self.monotonic_clock,
                    timeout_cleanup=(
                        close_session if callable(close_session) else None
                    ),
                )
            )
            acquisition_time = utc_datetime(fetched_at or self.clock())
            return (
                wire_payload,
                decoded_payload,
                status_code,
                headers,
                acquisition_time,
            )
        finally:
            session_cookies = getattr(self.session, "cookies", None)
            clear_cookies = getattr(session_cookies, "clear", None)
            if callable(clear_cookies):
                clear_cookies()
            if pinned_adapter is not None:
                pinned_adapter.close()

    def _pace_after(self, previous_request_at: float | None) -> None:
        if previous_request_at is None or not self.minimum_interval_seconds:
            return
        remaining = self.minimum_interval_seconds - (
            self.monotonic_clock() - previous_request_at
        )
        if remaining > 0:
            self.sleeper(remaining)

    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult:
        captures: list[RawCapture] = []
        records_by_provider_id: dict[str, RawNewsRecord] = {}
        errors: list[ProviderError] = []
        observed_times: list[datetime] = []
        previous_request_at: float | None = None
        request_count = 0
        request_budget_exhausted = False

        for root_window in self.windows:
            pending = list(reversed(_tile_root_window(root_window)))
            while pending:
                request_window = pending.pop()
                configured_source = request_window.configured_source
                if request_count >= self.max_requests_per_run:
                    occurred_at = utc_datetime(fetched_at or self.clock())
                    observed_times.append(occurred_at)
                    errors.append(
                        ProviderError(
                            provider_id=self.provider_id,
                            configured_source=configured_source,
                            error_type="request_limit_exceeded",
                            message=(
                                "GDELT acquisition exhausted the configured request "
                                "budget before all windows were complete"
                            ),
                            occurred_at=occurred_at,
                            details=self._request_context(request_window, 0),
                        )
                    )
                    request_budget_exhausted = True
                    break

                for attempt in range(self.max_retries + 1):
                    if request_count >= self.max_requests_per_run:
                        occurred_at = utc_datetime(fetched_at or self.clock())
                        observed_times.append(occurred_at)
                        errors.append(
                            ProviderError(
                                provider_id=self.provider_id,
                                configured_source=configured_source,
                                error_type="request_limit_exceeded",
                                message=(
                                    "GDELT acquisition exhausted the configured "
                                    "request budget during retry"
                                ),
                                occurred_at=occurred_at,
                                details=self._request_context(
                                    request_window, attempt
                                ),
                            )
                        )
                        request_budget_exhausted = True
                        break

                    self._pace_after(previous_request_at)
                    request_count += 1
                    capture_id = request_window.capture_id(attempt)
                    try:
                        (
                            wire_payload,
                            decoded_payload,
                            status,
                            headers,
                            acquisition_time,
                        ) = self._acquire(request_window, fetched_at=fetched_at)
                        observed_times.append(acquisition_time)
                    except _GdeltRejected as exc:
                        acquisition_time = utc_datetime(fetched_at or self.clock())
                        observed_times.append(acquisition_time)
                        errors.append(
                            ProviderError(
                                provider_id=self.provider_id,
                                configured_source=configured_source,
                                error_type=exc.error_type,
                                message=exc.safe_message,
                                occurred_at=acquisition_time,
                                details=self._request_context(
                                    request_window, attempt
                                ),
                            )
                        )
                        break
                    except Exception as exc:
                        acquisition_time = utc_datetime(fetched_at or self.clock())
                        observed_times.append(acquisition_time)
                        errors.append(
                            ProviderError(
                                provider_id=self.provider_id,
                                configured_source=configured_source,
                                error_type="network_error",
                                message=(
                                    "GDELT network request failed "
                                    f"({type(exc).__name__})"
                                ),
                                occurred_at=acquisition_time,
                                details=self._request_context(
                                    request_window, attempt
                                ),
                            )
                        )
                        break
                    finally:
                        previous_request_at = self.monotonic_clock()

                    checksum = sha256_bytes(wire_payload)
                    content_type = _optional_text(headers.get("content-type"))
                    content_encoding = (
                        _optional_text(headers.get("content-encoding")) or "identity"
                    ).casefold()
                    retry_after = _optional_text(headers.get("retry-after"))
                    capture_warnings = [
                        "target_start_datetime="
                        f"{_utc_iso(request_window.start_datetime)}",
                        "target_end_datetime="
                        f"{_utc_iso(request_window.end_datetime)}",
                        "configured_window_bound_semantics=inclusive",
                        "api_request_bound_semantics=exclusive",
                        "request_start_datetime="
                        f"{_utc_iso(request_window.request_start_datetime)}",
                        "request_end_datetime="
                        f"{_utc_iso(request_window.request_end_datetime)}",
                        f"request_split_path={request_window.split_path}",
                        f"request_attempt={attempt + 1}",
                        "request_mode=artlist",
                        "request_sort=dateasc",
                    ]
                    if content_type and "json" not in content_type.casefold():
                        capture_warnings.append("unexpected_response_content_type")
                    if retry_after is not None:
                        capture_warnings.append("retry_after_header_present")
                    if content_encoding != "identity":
                        capture_warnings.append(
                            f"wire_content_encoding={content_encoding}"
                        )

                    def make_capture() -> RawCapture:
                        suffix = (
                            ".json.gz"
                            if content_encoding in {"gzip", "x-gzip"}
                            else ".json.deflate"
                            if content_encoding == "deflate"
                            else ".json"
                        )
                        return RawCapture(
                            capture_id=capture_id,
                            provider_id=self.provider_id,
                            configured_source=configured_source,
                            source_name=self.source_name,
                            fetched_at=acquisition_time,
                            data_class="real",
                            content=wire_payload,
                            filename=f"{capture_id}{suffix}",
                            http_status=status,
                            content_type=content_type,
                            response_sha256=checksum,
                            warnings=tuple(capture_warnings),
                        )

                    retryable = status == 429 or 500 <= status <= 599
                    if retryable:
                        retry_delay = _retry_delay_seconds(
                            attempt=attempt,
                            retry_after=retry_after,
                            occurred_at=acquisition_time,
                            initial_seconds=self.backoff_initial_seconds,
                            maximum_seconds=self.backoff_max_seconds,
                        )
                        if retry_after is not None:
                            if retry_delay.header_valid:
                                capture_warnings.append(
                                    "retry_after_requested_seconds="
                                    f"{retry_delay.requested_seconds}"
                                )
                            else:
                                capture_warnings.append(
                                    "retry_after_invalid_ignored"
                                )
                        can_retry = (
                            attempt < self.max_retries
                            and request_count < self.max_requests_per_run
                            and not retry_delay.exceeds_wait_ceiling
                        )
                        if can_retry:
                            capture_warnings.append("bounded_retry_scheduled")
                            capture_warnings.append(
                                "retry_delay_applied_seconds="
                                f"{retry_delay.applied_seconds}"
                            )
                            if retry_delay.header_valid:
                                capture_warnings.append(
                                    "retry_after_applied_seconds="
                                    f"{retry_delay.applied_seconds}"
                                )
                        elif retry_delay.exceeds_wait_ceiling:
                            capture_warnings.append(
                                "retry_after_refused_above_wait_ceiling"
                            )
                        else:
                            capture_warnings.append("retry_limit_reached")
                        captures.append(make_capture())
                        error_type = (
                            "retry_after_exceeds_policy"
                            if retry_delay.exceeds_wait_ceiling
                            else "rate_limit_error"
                            if status == 429
                            else "server_error"
                        )
                        if can_retry:
                            assert retry_delay.applied_seconds is not None
                            self.sleeper(retry_delay.applied_seconds)
                            continue
                        retry_details = {
                            "retry_after_requested_seconds": (
                                retry_delay.requested_seconds
                            ),
                            "retry_after_applied_seconds": None,
                            "retry_wait_ceiling_seconds": (
                                self.backoff_max_seconds
                            ),
                            "retry_after_header_valid": retry_delay.header_valid,
                        }
                        errors.append(
                            ProviderError(
                                provider_id=self.provider_id,
                                configured_source=configured_source,
                                error_type=error_type,
                                message=(
                                    f"GDELT DOC returned HTTP {status}; "
                                    + (
                                        "Retry-After exceeds the approved wait "
                                        "ceiling"
                                        if retry_delay.exceeds_wait_ceiling
                                        else "retry limit reached"
                                    )
                                ),
                                occurred_at=acquisition_time,
                                http_status=status,
                                details=self._request_context(
                                    request_window,
                                    attempt,
                                    **retry_details,
                                ),
                            )
                        )
                        break
                    if not 200 <= status < 300:
                        captures.append(make_capture())
                        errors.append(
                            ProviderError(
                                provider_id=self.provider_id,
                                configured_source=configured_source,
                                error_type="http_error",
                                message=f"GDELT DOC returned HTTP {status}",
                                occurred_at=acquisition_time,
                                http_status=status,
                                details=self._request_context(
                                    request_window, attempt
                                ),
                            )
                        )
                        break

                    try:
                        parsed, parse_warnings = _parse_articles(
                            decoded_payload,
                            target_start=request_window.start_datetime,
                            target_end=request_window.end_datetime,
                            request_start=request_window.request_start_datetime,
                            request_end=request_window.request_end_datetime,
                            query=self.query,
                            configured_source=configured_source,
                            capture_id=capture_id,
                            fetched_at=acquisition_time,
                            http_status=status,
                            content_type=content_type,
                            response_sha256=checksum,
                            max_records=self.max_records_per_window,
                            max_field_characters=self.max_field_characters,
                            permission_basis=self.permission_basis,
                            licensing_usage_note=self.licensing_usage_note,
                            attribution_text=self.attribution_text,
                        )
                        capture_warnings.extend(parse_warnings)
                    except _GdeltRejected as exc:
                        captures.append(make_capture())
                        errors.append(
                            ProviderError(
                                provider_id=self.provider_id,
                                configured_source=configured_source,
                                error_type=exc.error_type,
                                message=exc.safe_message,
                                occurred_at=acquisition_time,
                                http_status=status,
                                details=self._request_context(
                                    request_window, attempt
                                ),
                            )
                        )
                        break

                    for record in parsed:
                        identity = record.provider_record_id or record.raw_record_id
                        records_by_provider_id.setdefault(identity, record)

                    if len(parsed) == self.max_records_per_window:
                        split = (
                            None
                            if request_window.depth >= self.max_split_depth
                            else request_window.split(
                                minimum_window_seconds=self.minimum_window_seconds
                            )
                        )
                        if split is None:
                            capture_warnings.append("saturated_window_failed_closed")
                            errors.append(
                                ProviderError(
                                    provider_id=self.provider_id,
                                    configured_source=configured_source,
                                    error_type="saturated_window_error",
                                    message=(
                                        "GDELT result reached maxrecords and cannot "
                                        "be split within configured safety bounds"
                                    ),
                                    occurred_at=acquisition_time,
                                    http_status=status,
                                    details=self._request_context(
                                        request_window, attempt
                                    ),
                                )
                            )
                        else:
                            left, right = split
                            capture_warnings.append("saturated_window_split")
                            pending.append(right)
                            pending.append(left)
                    captures.append(make_capture())
                    break

                if request_budget_exhausted:
                    break
            if request_budget_exhausted:
                break

        completed_at = max(observed_times) if observed_times else utc_datetime(
            fetched_at or self.clock()
        )
        return ProviderFetchResult(
            provider_id=self.provider_id,
            data_class="real",
            fetched_at=completed_at,
            configured_sources=self.configured_sources,
            records=tuple(records_by_provider_id.values()),
            captures=tuple(captures),
            errors=tuple(errors),
            warnings=(
                "gdelt_doc_articlelist_is_non_exhaustive",
                "gdelt_doc_requests_are_proactively_tiled_to_three_months",
                "gdelt_doc_saturated_tiles_are_bisected_without_pagination",
                "gdelt_doc_exclusive_boundaries_overlap_and_deduplicate",
            ),
        )


__all__ = [
    "GDELT_DOC_ENDPOINT",
    "GdeltDocNewsProvider",
    "GdeltQueryWindow",
    "load_gdelt_config",
]
