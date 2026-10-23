from __future__ import annotations

import calendar
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import time as wall_time
from urllib.parse import parse_qsl, urlsplit

import pytest

from paisa.news.normalization import normalize_record
from paisa.news.pipeline import _sanitize_provider_result
from paisa.news.providers import gdelt_doc as gdelt_module
from paisa.news.providers.gdelt_doc import (
    GDELT_DOC_ENDPOINT,
    GdeltDocNewsProvider,
    GdeltQueryWindow,
    load_gdelt_config,
)


NOW = datetime(2026, 10, 23, 10, 0, tzinfo=timezone.utc)
WINDOW = GdeltQueryWindow(
    "ogdc-december",
    datetime(2025, 12, 16, tzinfo=timezone.utc),
    datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
)
PERMISSION = "Public GDELT DOC metadata API reviewed for this academic audit."
USAGE = "Metadata and URLs only; linked publisher pages are not fetched."
ATTRIBUTION = "Data source: GDELT Project (https://www.gdeltproject.org/)."


class FakeRawStream:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def stream(self, chunk_size, *, decode_content):
        assert decode_content is False
        for start in range(0, len(self.content), 11):
            yield self.content[start : start + 11]


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        content_type: str = "application/json; charset=utf-8",
        headers: dict[str, str] | None = None,
        peer_ip: str = "93.184.216.34",
    ) -> None:
        self._content = content
        self.raw = FakeRawStream(content)
        self.status_code = status_code
        self.headers = {"Content-Type": content_type, **(headers or {})}
        self.peer_ip = peer_ip
        self.closed = False

    @property
    def content(self):
        return self._content

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @staticmethod
    def resolve_addresses(host, port):
        assert host == "api.gdeltproject.org"
        assert port == 443
        return ("93.184.216.34",)

    @staticmethod
    def peer_address(response):
        return response.peer_ip


def _provider(session, **overrides):
    values = {
        "query": '(OGDC OR OGDCL OR "Oil and Gas Development Company")',
        "windows": [WINDOW],
        "permission_basis": PERMISSION,
        "licensing_usage_note": USAGE,
        "attribution_text": ATTRIBUTION,
        "session": session,
        "clock": lambda: NOW,
        "minimum_interval_seconds": 0,
    }
    values.update(overrides)
    return GdeltDocNewsProvider(**values)


def _payload(
    *,
    url: str = "https://publisher.example/story?id=123",
    seendate: str = "20251220T103000Z",
) -> bytes:
    return json.dumps(
        {
            "articles": [
                {
                    "url": url,
                    "title": "OGDC announces a gas discovery",
                    "seendate": seendate,
                    "domain": "publisher.example",
                    "language": "English",
                    "sourcecountry": "Pakistan",
                }
            ]
        },
        separators=(",", ":"),
    ).encode("utf-8")


def test_gdelt_fetch_is_exact_bounded_metadata_only_evidence():
    payload = _payload()
    session = FakeSession([FakeResponse(payload)])

    result = _provider(session).fetch()

    assert result.provider_id == "gdelt-doc"
    assert result.data_class == "real"
    assert result.errors == ()
    assert len(session.calls) == 1
    request_url, request = session.calls[0]
    assert request_url == GDELT_DOC_ENDPOINT
    assert request["allow_redirects"] is False
    assert request["stream"] is True
    assert request["proxies"] == {}
    assert request["params"] == {
        "query": '(OGDC OR OGDCL OR "Oil and Gas Development Company")',
        "mode": "artlist",
        "format": "json",
        "maxrecords": 250,
        "startdatetime": "20251215235959",
        "enddatetime": "20260101000000",
        "sort": "dateasc",
    }

    capture = result.captures[0]
    assert capture.content == payload
    assert capture.response_sha256 == hashlib.sha256(payload).hexdigest()
    record = result.records[0]
    assert record.provider_id == "gdelt-doc"
    assert record.data_class == "real"
    assert record.source_type == "gdelt_doc_archive"
    assert record.title == "OGDC announces a gas discovery"
    assert record.source_url == "https://publisher.example/story?id=123"
    assert record.source_name == "publisher.example"
    assert record.language == "English"
    assert record.summary is None
    assert record.body is None
    assert record.published_at_raw is None
    assert record.published_timezone is None
    assert record.source_observed_at_raw == "2025-12-20T10:30:00Z"
    assert record.available_at_raw is None
    assert record.provider_record_id.startswith("GDELT-")
    assert record.raw_payload["seendate"] == "20251220T103000Z"
    assert record.raw_payload["permission_basis"] == PERMISSION
    assert record.raw_payload["licensing_usage_note"] == USAGE
    assert record.raw_payload["attribution_text"] == ATTRIBUTION
    assert record.raw_payload["provider_mode"] == "historical_api"
    assert record.raw_payload["target_start_datetime"] == "2025-12-16T00:00:00Z"
    assert record.raw_payload["target_end_datetime"] == "2025-12-31T23:59:59Z"
    assert record.raw_payload["request_start_datetime"] == "2025-12-15T23:59:59Z"
    assert record.raw_payload["request_end_datetime"] == "2026-01-01T00:00:00Z"
    assert result.configured_sources[0]["link_evidence"]["query"] == (
        '(OGDC OR OGDCL OR "Oil and Gas Development Company")'
    )
    assert record.raw_payload["source_url_sha256"] == hashlib.sha256(
        b"https://publisher.example/story?id=123"
    ).hexdigest()


def test_exclusive_api_bounds_preserve_both_inclusive_target_endpoints():
    payload = json.dumps(
        {
            "articles": [
                {
                    "url": "https://publisher.example/at-start",
                    "title": "Start boundary",
                    "seendate": "20251216T000000Z",
                },
                {
                    "url": "https://publisher.example/at-end",
                    "title": "End boundary",
                    "seendate": "20251231T235959Z",
                },
            ]
        }
    ).encode()

    result = _provider(FakeSession([FakeResponse(payload)])).fetch()

    assert result.errors == ()
    assert [record.title for record in result.records] == [
        "Start boundary",
        "End boundary",
    ]


def test_response_with_seendate_outside_inclusive_target_is_quarantined():
    payload = _payload(seendate="20251215T235959Z")

    result = _provider(FakeSession([FakeResponse(payload)])).fetch()

    assert result.records == ()
    assert result.errors[0].error_type == "record_outside_target_window"
    assert "response quarantined" in result.errors[0].message
    assert result.captures[0].content == payload


@pytest.mark.parametrize("seendate", [None, "not-a-timestamp"])
def test_historical_response_without_valid_seendate_is_quarantined(seendate):
    article = {
        "url": "https://publisher.example/unprovable",
        "title": "Historical membership cannot be proven",
    }
    if seendate is not None:
        article["seendate"] = seendate
    payload = json.dumps({"articles": [article]}).encode()

    result = _provider(FakeSession([FakeResponse(payload)])).fetch()

    assert result.records == ()
    assert result.errors[0].error_type == "invalid_source_observation_time"
    assert "response quarantined" in result.errors[0].message
    assert result.captures[0].content == payload


def test_seendate_never_becomes_publication_time_or_backdated_availability():
    result = _provider(FakeSession([FakeResponse(_payload())])).fetch()
    raw = result.records[0]

    normalized = normalize_record(
        raw,
        ingestion_run_id="NEWS-20261023-100000-gdelt001",
        raw_artifact_reference="raw/record-000001.json",
    )

    assert normalized.published_at is None
    assert normalized.schema_version == "1.1"
    assert normalized.source_observed_at == datetime(
        2025, 12, 20, 10, 30, tzinfo=timezone.utc
    )
    assert normalized.available_at == NOW
    assert "gdelt_seendate_is_source_observation_not_publication_or_eligibility" in (
        normalized.normalization_warnings
    )
    assert "published_at_missing" in normalized.normalization_warnings


def test_provider_record_identity_is_deterministic():
    first = _provider(FakeSession([FakeResponse(_payload())])).fetch().records[0]
    second = _provider(FakeSession([FakeResponse(_payload())])).fetch().records[0]

    assert first.provider_record_id == second.provider_record_id


def test_public_urls_keep_identity_values_but_remove_query_secrets_and_tracking():
    first_secret = "api-key-private"
    second_secret = "token-private"
    jwt_secret = "jwt-private"
    shared_secret = "tracking-private"
    first_fragment_secret = "fragment-private-one"
    second_fragment_secret = "fragment-private-two"
    first_raw_url = (
        "https://publisher.example/story?id=1"
        f"&api_key={first_secret}&jwt={jwt_secret}&utm_source={shared_secret}"
        "&view=full&author=desk"
        f"#access_token={first_fragment_secret}"
    )
    second_raw_url = (
        "https://publisher.example/story?id=2"
        f"&token={second_secret}&utm_source={shared_secret}&view=full&author=desk"
        f"#token={second_fragment_secret}"
    )
    first_result = _provider(
        FakeSession(
            [
                FakeResponse(
                    _payload(url=first_raw_url)
                )
            ]
        )
    ).fetch()
    second_result = _provider(
        FakeSession(
            [
                FakeResponse(
                    _payload(url=second_raw_url)
                )
            ]
        )
    ).fetch()
    first = _sanitize_provider_result(first_result).records[0]
    second = _sanitize_provider_result(second_result).records[0]

    assert first.source_url != second.source_url
    assert first.provider_record_id != second.provider_record_id
    assert [key for key, _value in parse_qsl(urlsplit(first.source_url).query)] == [
        "id",
        "view",
        "author",
    ]
    assert parse_qsl(urlsplit(first.source_url).query) == [
        ("id", "1"),
        ("view", "full"),
        ("author", "desk"),
    ]
    assert urlsplit(first.source_url).fragment == ""
    assert urlsplit(second.source_url).fragment == ""
    assert first_result.records[0].raw_payload["source_url_sha256"] == (
        hashlib.sha256(first_raw_url.encode()).hexdigest()
    )
    assert second_result.records[0].raw_payload["source_url_sha256"] == (
        hashlib.sha256(second_raw_url.encode()).hexdigest()
    )
    first_normalized = normalize_record(
        first,
        ingestion_run_id="NEWS-20261023-100000-urlhash1",
        raw_artifact_reference="raw/first.json",
    )
    second_normalized = normalize_record(
        second,
        ingestion_run_id="NEWS-20261023-100000-urlhash2",
        raw_artifact_reference="raw/second.json",
    )
    assert first_normalized.canonical_url != second_normalized.canonical_url
    serialized = json.dumps(
        [first.metadata_dict(), second.metadata_dict()],
        sort_keys=True,
    )
    assert first_secret not in serialized
    assert second_secret not in serialized
    assert jwt_secret not in serialized
    assert shared_secret not in serialized
    assert first_fragment_secret not in serialized
    assert second_fragment_secret not in serialized


def test_provider_removes_ambient_jwt_headers_and_authentication_state():
    session = FakeSession([FakeResponse(b'{"articles":[]}')])
    session.headers = {"jwt": "jwt-private", "X-Request-ID": "public-id"}
    session.params = {"jwt": "jwt-param-private"}
    session.proxies = {"https": "https://proxy.example"}
    session.cookies = {"session": "cookie-private"}
    session.auth = ("user", "password")
    session.cert = "client.pem"
    session.trust_env = True

    _provider(session)

    assert session.headers == {"X-Request-ID": "public-id"}
    assert session.params == {}
    assert session.proxies == {}
    assert session.cookies == {}
    assert session.auth is None
    assert session.cert is None
    assert session.trust_env is False


@pytest.mark.parametrize(
    ("status", "body", "expected_error"),
    [
        (429, b'{"message":"slow down"}', "rate_limit_error"),
        (503, b'{"message":"unavailable"}', "server_error"),
        (200, b"not-json", "response_parse_error"),
    ],
)
def test_rate_limit_server_and_parse_failures_retain_response_evidence(
    status, body, expected_error
):
    headers = {"Retry-After": "10"} if status == 429 else None
    response = FakeResponse(body, status_code=status, headers=headers)

    result = _provider(FakeSession([response]), max_retries=0).fetch()

    assert result.records == ()
    assert result.errors[0].error_type == expected_error
    assert result.errors[0].http_status == status
    assert result.captures[0].content == body
    assert result.captures[0].response_sha256 == hashlib.sha256(body).hexdigest()
    if status == 429:
        assert "retry_after_header_present" in result.captures[0].warnings
        assert "retry limit reached" in result.errors[0].message


def test_response_and_record_limits_fail_closed():
    oversized = FakeResponse(
        b"{}",
        headers={"Content-Length": "999"},
    )
    bounded = _provider(
        FakeSession([oversized]),
        max_compressed_bytes=16,
    ).fetch()
    assert bounded.records == ()
    assert bounded.errors[0].error_type == "response_limit_exceeded"

    too_many = json.dumps(
        {
            "articles": [
                {
                    "url": f"https://example.test/{index}",
                    "title": f"Article {index}",
                    "seendate": "20251220T103000Z",
                }
                for index in range(2)
            ]
        }
    ).encode()
    limited = _provider(
        FakeSession([FakeResponse(too_many)]),
        max_records_per_window=1,
    ).fetch()
    assert limited.records == ()
    assert limited.errors[0].error_type == "record_limit_exceeded"
    assert limited.captures[0].content == too_many


def test_config_is_windowed_credential_free_and_endpoint_locked(tmp_path: Path):
    base = {
        "schema_version": "1.0",
        "provider": "gdelt-doc",
        "endpoint": GDELT_DOC_ENDPOINT,
        "query": "OGDC",
        "permission_basis": PERMISSION,
        "licensing_usage_note": USAGE,
        "attribution_text": ATTRIBUTION,
        "windows": [
            {
                "window_id": "one",
                "start_datetime": "2025-12-16T00:00:00Z",
                "end_datetime": "2025-12-17T00:00:00Z",
            },
            {
                "window_id": "disabled",
                "start_datetime": "2025-12-17T00:00:00Z",
                "end_datetime": "2025-12-18T00:00:00Z",
                "enabled": False,
            },
        ],
    }
    path = tmp_path / "gdelt.json"
    path.write_text(json.dumps(base), encoding="utf-8")

    assert [item.window_id for item in load_gdelt_config(path)] == ["one"]
    provider = GdeltDocNewsProvider.from_config(
        path,
        session=FakeSession([FakeResponse(b'{"articles":[]}')]),
    )
    assert provider.provider_id == "gdelt-doc"

    malicious = dict(base, endpoint="https://example.test/archive")
    path.write_text(json.dumps(malicious), encoding="utf-8")
    with pytest.raises(ValueError, match="official DOC 2.0 endpoint"):
        GdeltDocNewsProvider.from_config(path)

    credentialed = dict(base, api_key="do-not-accept")
    path.write_text(json.dumps(credentialed), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported|credentials"):
        GdeltDocNewsProvider.from_config(path)


def test_config_rejects_naive_windows_and_unbounded_options(tmp_path: Path):
    document = {
        "query": "OGDC",
        "permission_basis": PERMISSION,
        "licensing_usage_note": USAGE,
        "attribution_text": ATTRIBUTION,
        "max_records_per_window": 251,
        "windows": [
            {
                "window_id": "unsafe",
                "start_datetime": "2025-12-16T00:00:00",
                "end_datetime": "2025-12-17T00:00:00Z",
            }
        ],
    }
    path = tmp_path / "gdelt.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="timezone"):
        GdeltDocNewsProvider.from_config(path)

    document["windows"][0]["start_datetime"] = "2025-12-16T00:00:00Z"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="max_records_per_window"):
        GdeltDocNewsProvider.from_config(path)


def test_soft_deadline_clips_operation_timeouts_and_rejects_larger_values():
    ticks = iter((0.0, 2.0))

    def monotonic() -> float:
        return next(ticks, 2.0)

    session = FakeSession([FakeResponse(_payload())])
    provider = _provider(
        session,
        connect_timeout_seconds=4,
        read_timeout_seconds=4,
        soft_total_timeout_seconds=5,
        monotonic_clock=monotonic,
    )

    result = provider.fetch()

    assert result.errors == ()
    assert session.calls[0][1]["timeout"] == (3.0, 3.0)
    assert "DNS resolution" in provider.effective_config[
        "soft_total_timeout_semantics"
    ]

    with pytest.raises(ValueError, match="connect_timeout_seconds"):
        _provider(
            FakeSession([]),
            connect_timeout_seconds=6,
            soft_total_timeout_seconds=5,
        )


def test_dns_resolution_is_bounded_by_the_end_to_end_request_deadline():
    def blocked_resolver(_host, _port):
        wall_time.sleep(2)
        return ("93.184.216.34",)

    started = wall_time.monotonic()
    result = _provider(
        FakeSession([]),
        resolver=blocked_resolver,
        connect_timeout_seconds=0.05,
        read_timeout_seconds=0.05,
        soft_total_timeout_seconds=0.05,
    ).fetch()
    elapsed = wall_time.monotonic() - started

    assert elapsed < 0.5
    assert result.errors[0].error_type == "soft_total_timeout"
    assert result.errors[0].details["capture_id"].endswith("-t000-a1")
    assert result.errors[0].details["attempt_number"] == 1


def test_response_header_wait_cannot_exceed_the_end_to_end_deadline():
    class SlowHeaderSession(FakeSession):
        def get(self, url, **kwargs):
            wall_time.sleep(0.3)
            return super().get(url, **kwargs)

    started = wall_time.monotonic()
    result = _provider(
        SlowHeaderSession([FakeResponse(_payload())]),
        connect_timeout_seconds=0.05,
        read_timeout_seconds=0.05,
        soft_total_timeout_seconds=0.05,
    ).fetch()
    elapsed = wall_time.monotonic() - started

    assert elapsed < 0.2
    assert result.records == ()
    assert result.errors[0].error_type == "soft_total_timeout"


def test_slow_timeout_cleanup_cannot_extend_the_end_to_end_deadline():
    class SlowHeaderAndCloseSession(FakeSession):
        def get(self, url, **kwargs):
            wall_time.sleep(0.3)
            return super().get(url, **kwargs)

        def close(self):
            wall_time.sleep(0.3)

    started = wall_time.monotonic()
    result = _provider(
        SlowHeaderAndCloseSession([FakeResponse(_payload())]),
        connect_timeout_seconds=0.05,
        read_timeout_seconds=0.05,
        soft_total_timeout_seconds=0.05,
    ).fetch()
    elapsed = wall_time.monotonic() - started

    assert elapsed < 0.2
    assert result.records == ()
    assert result.errors[0].error_type == "soft_total_timeout"


def test_stalled_stream_read_cannot_exceed_the_end_to_end_deadline():
    class SlowRawStream(FakeRawStream):
        def stream(self, chunk_size, *, decode_content):
            wall_time.sleep(0.3)
            yield from super().stream(chunk_size, decode_content=decode_content)

    response = FakeResponse(_payload())
    response.raw = SlowRawStream(response.content)
    started = wall_time.monotonic()
    result = _provider(
        FakeSession([response]),
        connect_timeout_seconds=0.05,
        read_timeout_seconds=0.05,
        soft_total_timeout_seconds=0.05,
    ).fetch()
    elapsed = wall_time.monotonic() - started

    assert elapsed < 0.2
    assert result.records == ()
    assert result.errors[0].error_type == "soft_total_timeout"


def test_default_dns_resolution_uses_a_cancellable_process(
    monkeypatch: pytest.MonkeyPatch,
):
    observed: dict[str, object] = {}

    def timeout_process(command, **kwargs):
        observed["command"] = command
        observed["timeout"] = kwargs["timeout"]
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(gdelt_module.subprocess, "run", timeout_process)

    with pytest.raises(
        gdelt_module._GdeltRejected,
        match="DNS resolution exceeded",
    ) as captured:
        gdelt_module._validated_addresses(
            gdelt_module._resolve_addresses,
            timeout_seconds=0.125,
        )

    assert captured.value.error_type == "soft_total_timeout"
    assert observed["command"][0]  # type: ignore[index]
    assert observed["timeout"] == 0.125


def test_exact_compressed_response_bytes_are_captured_but_decoded_for_parsing():
    decoded = _payload()
    wire = gzip.compress(decoded)

    result = _provider(
        FakeSession(
            [FakeResponse(wire, headers={"Content-Encoding": "gzip"})]
        )
    ).fetch()

    assert result.errors == ()
    assert result.records[0].title == "OGDC announces a gas discovery"
    assert result.captures[0].content == wire
    assert result.captures[0].response_sha256 == hashlib.sha256(wire).hexdigest()
    assert result.captures[0].filename.endswith(".json.gz")
    assert "wire_content_encoding=gzip" in result.captures[0].warnings


def test_saturated_windows_split_deterministically_and_retain_parent_records():
    empty = b'{"articles":[]}'
    session = FakeSession(
        [FakeResponse(_payload()), FakeResponse(empty), FakeResponse(empty)]
    )

    result = _provider(session, max_records_per_window=1).fetch()

    assert result.errors == ()
    assert [record.title for record in result.records] == [
        "OGDC announces a gas discovery"
    ]
    assert len(result.captures) == 3
    assert "saturated_window_split" in result.captures[0].warnings
    root_params = session.calls[0][1]["params"]
    left_params = session.calls[1][1]["params"]
    right_params = session.calls[2][1]["params"]
    assert left_params["startdatetime"] == root_params["startdatetime"]
    assert right_params["enddatetime"] == root_params["enddatetime"]
    left_end = datetime.strptime(
        left_params["enddatetime"], "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    right_start = datetime.strptime(
        right_params["startdatetime"], "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    assert right_start == left_end - timedelta(seconds=2)


def test_sparse_five_year_window_uses_complete_overlapping_three_month_tiles():
    long_window = GdeltQueryWindow(
        "full-history",
        datetime(2021, 9, 16, tzinfo=timezone.utc),
        datetime(2026, 9, 15, 23, 59, 59, tzinfo=timezone.utc),
    )
    empty = b'{"articles":[]}'
    session = FakeSession([FakeResponse(empty) for _ in range(32)])

    result = _provider(session, windows=[long_window]).fetch()

    assert result.errors == ()
    assert result.records == ()
    assert len(session.calls) == len(result.captures)
    assert len(session.calls) > 1
    requested = [call[1]["params"] for call in session.calls]
    bounds = [
        (
            datetime.strptime(item["startdatetime"], "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            ),
            datetime.strptime(item["enddatetime"], "%Y%m%d%H%M%S").replace(
                tzinfo=timezone.utc
            ),
        )
        for item in requested
    ]

    def three_month_boundary(value: datetime) -> datetime:
        month_index = value.year * 12 + value.month - 1 + 3
        year, month_index = divmod(month_index, 12)
        month = month_index + 1
        day = min(value.day, calendar.monthrange(year, month)[1])
        return value.replace(year=year, month=month, day=day)

    assert bounds[0][0] == long_window.start_datetime - timedelta(seconds=1)
    assert bounds[-1][1] == long_window.end_datetime + timedelta(seconds=1)
    assert all(end <= three_month_boundary(start) for start, end in bounds)
    assert all(
        following_start == prior_end - timedelta(seconds=2)
        for (_prior_start, prior_end), (following_start, _following_end) in zip(
            bounds,
            bounds[1:],
        )
    )
    assert (
        "gdelt_doc_requests_are_proactively_tiled_to_three_months"
        in result.warnings
    )


def test_overlapping_tile_boundary_records_are_deduplicated():
    boundary = datetime(2025, 3, 31, 23, 59, 58, tzinfo=timezone.utc)
    window = GdeltQueryWindow(
        "boundary-overlap",
        datetime(2025, 1, 1, tzinfo=timezone.utc),
        datetime(2025, 4, 2, tzinfo=timezone.utc),
    )
    payload = _payload(seendate="20250331T235958Z")
    session = FakeSession([FakeResponse(payload), FakeResponse(payload)])

    result = _provider(session, windows=[window]).fetch()

    assert result.errors == ()
    assert len(session.calls) == 2
    assert len(result.records) == 1
    first_end = datetime.strptime(
        session.calls[0][1]["params"]["enddatetime"], "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    second_start = datetime.strptime(
        session.calls[1][1]["params"]["startdatetime"], "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc)
    assert boundary == first_end - timedelta(seconds=1)
    assert boundary == second_start + timedelta(seconds=1)
    assert (
        "gdelt_doc_exclusive_boundaries_overlap_and_deduplicate"
        in result.warnings
    )


def test_saturated_minimum_window_fails_closed():
    start = datetime(2025, 12, 20, tzinfo=timezone.utc)
    short_window = GdeltQueryWindow(
        "short",
        start,
        start + timedelta(seconds=120),
    )

    result = _provider(
        FakeSession([FakeResponse(_payload(seendate="20251220T000100Z"))]),
        windows=[short_window],
        max_records_per_window=1,
        minimum_window_seconds=60,
    ).fetch()

    assert [record.title for record in result.records] == [
        "OGDC announces a gas discovery"
    ]
    assert result.errors[0].error_type == "saturated_window_error"
    assert "saturated_window_failed_closed" in result.captures[0].warnings


@pytest.mark.parametrize(
    ("status", "headers", "expected_delay", "expected_requested"),
    [
        (429, {"Retry-After": "10"}, 10.0, 10.0),
        (
            429,
            {"Retry-After": "Fri, 23 Oct 2026 10:00:12 GMT"},
            12.0,
            12.0,
        ),
        (503, None, 1.0, None),
    ],
)
def test_recovered_retryable_response_is_capture_warning_not_terminal_error(
    status, headers, expected_delay, expected_requested
):
    delays: list[float] = []
    session = FakeSession(
        [
            FakeResponse(
                b'{"message":"slow down"}',
                status_code=status,
                headers=headers,
            ),
            FakeResponse(_payload()),
        ]
    )

    result = _provider(
        session,
        max_retries=1,
        backoff_initial_seconds=1,
        backoff_max_seconds=15,
        sleeper=delays.append,
    ).fetch()

    assert len(session.calls) == 2
    assert delays == [expected_delay]
    assert len(result.captures) == 2
    assert result.records[0].title == "OGDC announces a gas discovery"
    assert result.errors == ()
    assert "bounded_retry_scheduled" in result.captures[0].warnings
    requested_marker = f"retry_after_requested_seconds={expected_requested}"
    assert (requested_marker in result.captures[0].warnings) is (
        expected_requested is not None
    )
    assert f"retry_delay_applied_seconds={expected_delay}" in (
        result.captures[0].warnings
    )
    retry_after_applied = f"retry_after_applied_seconds={expected_delay}"
    assert (retry_after_applied in result.captures[0].warnings) is (
        expected_requested is not None
    )


def test_retry_after_above_approved_wait_ceiling_fails_without_early_retry():
    session = FakeSession(
        [
            FakeResponse(
                b'{"message":"slow down"}',
                status_code=429,
                headers={"Retry-After": "600"},
            ),
            FakeResponse(_payload()),
        ]
    )
    delays: list[float] = []

    result = _provider(
        session,
        max_retries=1,
        backoff_max_seconds=30,
        sleeper=delays.append,
    ).fetch()

    assert len(session.calls) == 1
    assert delays == []
    assert result.records == ()
    assert result.errors[0].error_type == "retry_after_exceeds_policy"
    assert result.errors[0].details["retry_after_requested_seconds"] == 600.0
    assert result.errors[0].details["retry_after_applied_seconds"] is None
    assert "retry_after_requested_seconds=600.0" in result.captures[0].warnings
    assert "retry_after_refused_above_wait_ceiling" in result.captures[0].warnings


def test_network_error_carries_exact_sanitized_request_attempt_context():
    result = _provider(
        FakeSession([TimeoutError("network token=private")]),
        max_retries=0,
    ).fetch()

    assert result.records == ()
    error = result.errors[0]
    assert error.configured_source == "gdelt-doc:ogdc-december"
    assert error.details == {
        "capture_id": (
            "gdelt-ogdc-december-20251216000000-20251231235959-t000-a1"
        ),
        "split_path": "t000",
        "target_start_datetime": "2025-12-16T00:00:00Z",
        "target_end_datetime": "2025-12-31T23:59:59Z",
        "request_start_datetime": "2025-12-15T23:59:59Z",
        "request_end_datetime": "2026-01-01T00:00:00Z",
        "attempt_number": 1,
        "query_sha256": hashlib.sha256(
            '(OGDC OR OGDCL OR "Oil and Gas Development Company")'.encode()
        ).hexdigest(),
        "request_parameters_sha256": error.details[
            "request_parameters_sha256"
        ],
    }
    assert len(error.details["request_parameters_sha256"]) == 64
    assert "private" not in json.dumps(error.to_dict())


def test_committed_gdelt_example_config_loads_and_carries_usage_evidence():
    config = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "news"
        / "gdelt_ogdc_history.example.json"
    )

    windows = load_gdelt_config(config)
    provider = GdeltDocNewsProvider.from_config(
        config,
        session=FakeSession([FakeResponse(b'{"articles":[]}')]),
    )

    assert windows
    assert provider.data_class == "real"
    assert provider.effective_config["licensing_usage_note"].startswith(
        "GDELT-released metadata"
    )
    assert provider.effective_config["attribution_text"].startswith(
        "Data source: GDELT Project"
    )
    assert provider.configured_sources[0]["provider_mode"] == "historical_api"
