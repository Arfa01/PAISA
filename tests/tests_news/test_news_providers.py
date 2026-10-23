from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json

import pytest

from paisa.news.providers.fixture import FixtureNewsProvider
from paisa.news.providers.manual_csv import (
    LICENSING_REVIEW_REQUIRED,
    ManualCsvNewsProvider,
)
from paisa.news.providers.rss import (
    RssNewsProvider,
    RssSourceConfig,
    load_rss_config,
)


NOW = datetime(2026, 10, 22, 9, 30, tzinfo=timezone.utc)
MANUAL_LICENSE_NOTE = (
    "Licensed academic import; supplied title, summary, URL, and body may be stored."
)


class FakeRawStream:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.bytes_yielded = 0

    def stream(self, chunk_size, *, decode_content):
        assert decode_content is False
        for start in range(0, len(self.content), min(chunk_size, 7)):
            chunk = self.content[start : start + min(chunk_size, 7)]
            self.bytes_yielded += len(chunk)
            yield chunk


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        content_type: str = "application/rss+xml; charset=utf-8",
        url: str = "https://publisher.example/feed.xml",
        content_encoding: str | None = None,
        peer_ip: str = "93.184.216.34",
        stream_wire_bytes: bool = False,
    ) -> None:
        self._content = content
        self.raw = FakeRawStream(content) if stream_wire_bytes else None
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        if content_encoding is not None:
            self.headers["Content-Encoding"] = content_encoding
        self.url = url
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
        return ("93.184.216.34",)

    @staticmethod
    def peer_address(response):
        return response.peer_ip


def _serialized_provider_result(result):
    return json.dumps(
        {
            "configured_sources": result.configured_sources,
            "records": [record.metadata_dict() for record in result.records],
            "captures": [capture.metadata_dict() for capture in result.captures],
            "errors": [error.to_dict() for error in result.errors],
            "warnings": result.warnings,
        },
        default=str,
        sort_keys=True,
    )


def test_rss_provider_captures_exact_response_and_extracts_without_scraping():
    payload = b"""<?xml version="1.0"?>
    <rss version="2.0"><channel><language>en-PK</language><item>
      <guid>ogdc-1</guid><title>OGDC announces discovery</title>
      <link>/business/ogdc-1</link><description>Short feed summary.</description>
      <pubDate>Wed, 21 Oct 2026 14:30:00 +0500</pubDate>
    </item></channel></rss>"""
    session = FakeSession([FakeResponse(payload)])
    provider = RssNewsProvider(
        [
            RssSourceConfig(
                source_id="publisher-business",
                source_name="Publisher Business",
                url="https://publisher.example/feed.xml",
                default_timezone="Asia/Karachi",
            )
        ],
        session=session,
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert result.provider_id == "rss"
    assert result.errors == ()
    assert len(result.records) == 1
    record = result.records[0]
    assert record.data_class == "real"
    assert record.source_type == "rss"
    assert record.source_url == "https://publisher.example/business/ogdc-1"
    assert record.title == "OGDC announces discovery"
    assert record.summary == "Short feed summary."
    assert record.body is None
    assert record.published_at_raw == "Wed, 21 Oct 2026 14:30:00 +0500"
    assert record.published_timezone == "+0500"
    assert record.language == "en-PK"
    assert result.captures[0].content == payload
    assert result.captures[0].response_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.captures[0].http_status == 200
    assert len(session.calls) == 1, "the adapter must fetch only the configured feed"
    assert session.calls[0][1]["allow_redirects"] is False


@pytest.mark.parametrize(
    "url",
    [
        "https://alice:secret@publisher.example/feed.xml",
        "https://publisher.example/feed.xml?api_key=secret",
        "https://publisher.example/feed.xml?key=secret",
        "https://publisher.example/feed.xml?signature=secret",
        "https://publisher.example:bad/feed.xml",
    ],
)
def test_rss_source_rejects_credential_bearing_or_invalid_urls(url: str):
    with pytest.raises(ValueError):
        RssSourceConfig("unsafe", "Unsafe", url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/feed",
        "http://[::1]/feed",
        "http://10.0.0.5/feed",
        "http://169.254.169.254/latest/meta-data",
    ],
)
def test_rss_source_rejects_non_public_literal_destinations_by_default(url: str):
    with pytest.raises(ValueError, match="public"):
        RssSourceConfig("private", "Private", url)


def test_rss_source_persists_only_query_redacted_identity_and_digest():
    source = RssSourceConfig(
        "publisher",
        "Publisher",
        "https://publisher.example/feed.xml?edition=pk&opaque=do-not-retain",
    )

    serialized = json.dumps(source.to_dict(), sort_keys=True)

    assert "edition=pk" not in serialized
    assert "do-not-retain" not in serialized
    assert "REDACTED" in serialized
    assert source.to_dict()["url_sha256"] == hashlib.sha256(
        source.url.encode("utf-8")
    ).hexdigest()


def test_rss_provider_rejects_private_dns_results_before_request():
    session = FakeSession([])
    provider = RssNewsProvider(
        [RssSourceConfig("internal", "Internal", "https://publisher.example/feed")],
        session=session,
        resolver=lambda _host, _port: ("10.0.0.5",),
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert session.calls == []
    assert result.errors[0].error_type == "unsafe_destination"


def test_rss_provider_rejects_mixed_public_and_private_dns_results():
    session = FakeSession([])
    provider = RssNewsProvider(
        [RssSourceConfig("mixed", "Mixed", "https://publisher.example/feed")],
        session=session,
        resolver=lambda _host, _port: ("93.184.216.34", "169.254.169.254"),
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert session.calls == []
    assert result.errors[0].error_type == "unsafe_destination"


def test_rss_provider_rejects_connected_peer_outside_validated_dns_set():
    response = FakeResponse(
        b"<rss><channel/></rss>",
        peer_ip="93.184.216.35",
    )
    provider = RssNewsProvider(
        [RssSourceConfig("rebound", "Rebound", "https://publisher.example/feed")],
        session=FakeSession([response]),
        resolver=lambda _host, _port: ("93.184.216.34",),
        peer_address_getter=lambda value: value.peer_ip,
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert result.captures == ()
    assert result.errors[0].error_type == "unsafe_destination"
    assert response.closed is True


def test_rss_private_destination_requires_explicit_opt_in():
    response = FakeResponse(b"<rss><channel/></rss>", peer_ip="10.0.0.5")
    provider = RssNewsProvider(
        [
            RssSourceConfig(
                "reviewed-private",
                "Reviewed Private",
                "http://10.0.0.5/feed",
                allow_private_network=True,
            )
        ],
        session=FakeSession([response]),
        resolver=lambda _host, _port: ("10.0.0.5",),
        peer_address_getter=lambda value: value.peer_ip,
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert result.errors == ()
    assert result.captures[0].configured_source == "http://10.0.0.5/feed"


def test_rss_provider_disables_implicit_environment_authentication(monkeypatch):
    import paisa.news.providers.rss as rss_module

    session = rss_module.requests.Session()
    session.trust_env = True
    session.cookies.set("credential", "must-not-be-sent")
    monkeypatch.setattr(rss_module.requests, "Session", lambda: session)

    RssNewsProvider(
        [RssSourceConfig("public", "Public", "https://publisher.example/feed")]
    )

    assert session.trust_env is False
    assert len(session.cookies) == 0


def test_pinned_adapter_preserves_hostname_but_replaces_socket_dns_host(monkeypatch):
    import paisa.news.providers.rss as rss_module

    socket_targets = []

    class FakeConnection:
        def __init__(self, *, host, **_kwargs):
            self.host = host
            self._dns_host = host

        def _new_conn(self):
            socket_targets.append(self._dns_host)
            return object()

    class FakePool:
        ConnectionCls = FakeConnection

    pool = FakePool()
    monkeypatch.setattr(
        rss_module.HTTPAdapter,
        "get_connection_with_tls_context",
        lambda self, request, verify, proxies, cert: pool,
    )
    request = rss_module.requests.Request(
        "GET", "https://publisher.example/feed"
    ).prepare()
    adapter = rss_module._PinnedAddressAdapter("93.184.216.34")

    pinned_pool = adapter.get_connection_with_tls_context(
        request,
        True,
        proxies={"https": "http://127.0.0.1:9999"},
    )
    connection = pinned_pool.ConnectionCls(host="publisher.example")

    assert connection.host == "publisher.example"
    connection._new_conn()
    assert socket_targets == ["93.184.216.34"]
    assert connection._dns_host == "publisher.example"
    assert connection.host == "publisher.example"
    assert request.url == "https://publisher.example/feed"


def test_owned_rss_client_mounts_validated_address_without_second_resolution(
    monkeypatch,
):
    import paisa.news.providers.rss as rss_module

    class OwnedFakeSession(FakeSession):
        def __init__(self, outcomes):
            super().__init__(outcomes)
            self.trust_env = True
            self.auth = None
            self.headers = {}
            self.cookies = rss_module.requests.cookies.RequestsCookieJar()
            self.mounted = {}

        def mount(self, prefix, adapter):
            self.mounted[prefix] = adapter

    response = FakeResponse(b"<rss><channel/></rss>")
    owned_session = OwnedFakeSession([response])
    monkeypatch.setattr(rss_module.requests, "Session", lambda: owned_session)
    resolutions = []

    def resolver(_host, _port):
        resolutions.append(True)
        if len(resolutions) == 1:
            return ("93.184.216.34",)
        return ("169.254.169.254",)

    provider = RssNewsProvider(
        [RssSourceConfig("pinned", "Pinned", "https://publisher.example/feed")],
        resolver=resolver,
        peer_address_getter=lambda value: value.peer_ip,
        clock=lambda: NOW,
    )
    result = provider.fetch()

    assert result.errors == ()
    assert len(resolutions) == 1
    assert owned_session.mounted["https://"].pinned_address == "93.184.216.34"
    assert owned_session.calls[0][1]["proxies"] == {}
    assert owned_session.calls[0][1]["allow_redirects"] is False


def test_rss_acquisition_time_is_observed_after_response_bytes_arrive():
    payload = b"<rss><channel><item><title>OGDC</title></item></channel></rss>"
    session = FakeSession([FakeResponse(payload)])

    def after_response_clock():
        assert len(session.calls) == 1
        return NOW

    result = RssNewsProvider(
        [RssSourceConfig("timed", "Timed", "https://publisher.example/feed.xml")],
        session=session,
        clock=after_response_clock,
    ).fetch()

    assert result.fetched_at == NOW
    assert result.captures[0].fetched_at == NOW
    assert result.records[0].fetched_at == NOW


def test_rss_provider_rejects_dtd_and_entity_expansion():
    payload = b"""<?xml version="1.0"?>
    <!DOCTYPE rss [<!ENTITY secret "expanded">]>
    <rss><channel><item><title>&secret;</title></item></channel></rss>"""
    result = RssNewsProvider(
        [RssSourceConfig("unsafe-xml", "Unsafe XML", "https://example.test/feed")],
        session=FakeSession([FakeResponse(payload)]),
        clock=lambda: NOW,
    ).fetch()

    assert result.records == ()
    assert result.errors[0].error_type == "unsafe_xml"
    assert "expanded" not in result.errors[0].message


def test_rss_provider_rejects_dtd_without_entity_declarations():
    payload = b"<!DOCTYPE rss><rss><channel/></rss>"
    result = RssNewsProvider(
        [RssSourceConfig("unsafe-dtd", "Unsafe DTD", "https://example.test/feed")],
        session=FakeSession([FakeResponse(payload)]),
        clock=lambda: NOW,
    ).fetch()

    assert result.records == ()
    assert result.errors[0].error_type == "unsafe_xml"


def test_rss_provider_enforces_compressed_and_decoded_response_limits():
    payload = b"<rss><channel><item><title>OGDC</title></item></channel></rss>"
    too_large = RssNewsProvider(
        [RssSourceConfig("large", "Large", "https://example.test/feed")],
        session=FakeSession([FakeResponse(payload)]),
        max_compressed_bytes=16,
        clock=lambda: NOW,
    ).fetch()
    assert too_large.captures == ()
    assert too_large.errors[0].error_type == "response_limit_exceeded"

    compressed = gzip.compress(payload)
    expanded = RssNewsProvider(
        [RssSourceConfig("compressed", "Compressed", "https://example.test/feed")],
        session=FakeSession(
            [FakeResponse(compressed, content_encoding="gzip")]
        ),
        max_decoded_bytes=32,
        clock=lambda: NOW,
    ).fetch()
    assert expanded.captures == ()
    assert expanded.errors[0].error_type == "response_limit_exceeded"


def test_rss_provider_streams_wire_bytes_and_stops_at_compressed_limit():
    response = FakeResponse(b"x" * 100, stream_wire_bytes=True)
    result = RssNewsProvider(
        [RssSourceConfig("stream", "Stream", "https://example.test/feed")],
        session=FakeSession([response]),
        max_compressed_bytes=20,
        clock=lambda: NOW,
    ).fetch()

    assert result.errors[0].error_type == "response_limit_exceeded"
    assert response.raw.bytes_yielded < 100


def test_rss_provider_enforces_entry_depth_field_and_total_time_limits():
    two_entries = (
        b"<rss><channel><item><title>one</title></item>"
        b"<item><title>two</title></item></channel></rss>"
    )
    entries_result = RssNewsProvider(
        [RssSourceConfig("entries", "Entries", "https://example.test/feed")],
        session=FakeSession([FakeResponse(two_entries)]),
        max_entries=1,
        clock=lambda: NOW,
    ).fetch()
    assert entries_result.errors[0].error_type == "feed_limit_exceeded"

    deep_xml = b"<rss><channel><wrapper><item><title>x</title></item></wrapper></channel></rss>"
    depth_result = RssNewsProvider(
        [RssSourceConfig("depth", "Depth", "https://example.test/feed")],
        session=FakeSession([FakeResponse(deep_xml)]),
        max_xml_depth=3,
        clock=lambda: NOW,
    ).fetch()
    assert depth_result.errors[0].error_type == "feed_limit_exceeded"

    long_field = b"<rss><channel><item><title>long-title</title></item></channel></rss>"
    field_result = RssNewsProvider(
        [RssSourceConfig("field", "Field", "https://example.test/feed")],
        session=FakeSession([FakeResponse(long_field)]),
        max_field_characters=4,
        clock=lambda: NOW,
    ).fetch()
    assert field_result.errors[0].error_type == "feed_limit_exceeded"

    ticks = iter((0.0, 2.0, 2.0, 2.0))
    time_result = RssNewsProvider(
        [RssSourceConfig("slow", "Slow", "https://example.test/feed")],
        session=FakeSession([FakeResponse(b"<rss><channel/></rss>")]),
        total_timeout_seconds=1.0,
        monotonic_clock=lambda: next(ticks),
        clock=lambda: NOW,
    ).fetch()
    assert time_result.errors[0].error_type == "total_timeout"


def test_rss_provider_redacts_query_values_and_exception_secrets():
    source = RssSourceConfig(
        "redacted",
        "Redacted",
        "https://publisher.example/feed?edition=private-value",
    )
    result = RssNewsProvider(
        [source],
        session=FakeSession([RuntimeError("failed with private-value")]),
        clock=lambda: NOW,
    ).fetch()

    serialized = _serialized_provider_result(result)
    assert "private-value" not in serialized
    assert "REDACTED" in serialized


def test_rss_provider_redacts_article_query_values_from_structured_metadata():
    payload = b"""<rss><channel><item>
      <guid>https://publisher.example/id?opaque=guid-secret</guid>
      <title>OGDC update</title>
      <link>https://publisher.example/story?edition=link-secret</link>
    </item></channel></rss>"""
    result = RssNewsProvider(
        [RssSourceConfig("links", "Links", "https://publisher.example/feed")],
        session=FakeSession([FakeResponse(payload)]),
        clock=lambda: NOW,
    ).fetch()

    serialized = _serialized_provider_result(result)
    assert "guid-secret" not in serialized
    assert "link-secret" not in serialized
    assert serialized.count("REDACTED") >= 2


def test_atom_provider_preserves_updated_as_availability_not_publication():
    payload = b"""<feed xmlns="http://www.w3.org/2005/Atom" xml:lang="en">
      <entry><id>tag:example,2026:1</id><title>Oil &amp; Gas update</title>
      <updated>2026-10-21T09:30:00+05:00</updated>
      <link rel="alternate" href="news/1"/><summary>Summary</summary>
      <content>Feed-supplied body</content></entry></feed>"""
    provider = RssNewsProvider(
        [
            RssSourceConfig(
                "atom",
                "Atom Publisher",
                "https://example.test/base/feed",
                default_timezone="Asia/Karachi",
                source_type="atom",
            )
        ],
        session=FakeSession(
            [FakeResponse(payload, content_type="application/atom+xml")]
        ),
        clock=lambda: NOW,
    )

    result = provider.fetch()

    record = result.records[0]
    assert record.source_type == "atom"
    assert record.source_url == "https://example.test/base/news/1"
    assert record.body == "Feed-supplied body"
    assert record.language == "en"
    assert record.published_at_raw is None
    assert record.available_at_raw == "2026-10-21T09:30:00+05:00"
    assert record.published_timezone == "+05:00"
    assert (
        "atom_published_missing_used_updated_as_availability"
        in record.provider_warnings
    )


def test_rss_provider_records_partial_errors_and_never_falls_back_to_fixture():
    good_payload = b"<rss><channel><item><title>OGDC</title></item></channel></rss>"
    session = FakeSession(
        [RuntimeError("network unavailable"), FakeResponse(good_payload)]
    )
    provider = RssNewsProvider(
        [
            RssSourceConfig("broken", "Broken", "https://broken.example/feed"),
            RssSourceConfig("working", "Working", "https://working.example/feed"),
        ],
        session=session,
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert len(result.errors) == 1
    assert result.errors[0].error_type == "network_error"
    assert len(result.records) == 1
    assert {record.data_class for record in result.records} == {"real"}
    assert all(record.provider_id != "fixture" for record in result.records)


def test_rss_http_failure_retains_response_evidence_and_status():
    payload = b"rate limited"
    provider = RssNewsProvider(
        [RssSourceConfig("limited", "Limited", "https://example.test/feed")],
        session=FakeSession(
            [FakeResponse(payload, status_code=429, content_type="text/plain")]
        ),
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert result.records == ()
    assert result.errors[0].error_type == "http_error"
    assert result.errors[0].http_status == 429
    assert result.captures[0].content == payload
    assert result.captures[0].http_status == 429
    assert "unexpected_response_content_type" in result.captures[0].warnings


def test_rss_configuration_filters_disabled_sources_and_honors_provider_options(tmp_path):
    config = tmp_path / "rss.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "timeout_seconds": 12,
                "user_agent": "PAISA test agent/1.0",
                "sources": [
                    {
                        "id": "business",
                        "name": "Business",
                        "url": "https://example.test/rss",
                        "language": "en",
                        "default_timezone": "Asia/Karachi",
                        "source_type": "rss",
                        "enabled": True,
                    },
                    {
                        "id": "disabled",
                        "name": "Disabled",
                        "url": "https://disabled.example/rss",
                        "enabled": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    sources = load_rss_config(config)
    assert sources == (
        RssSourceConfig(
            "business",
            "Business",
            "https://example.test/rss",
            "en",
            "Asia/Karachi",
            "rss",
        ),
    )

    session = FakeSession(
        [FakeResponse(b"<rss><channel/></rss>", url="https://example.test/rss")]
    )
    provider = RssNewsProvider.from_config(config, session=session, clock=lambda: NOW)
    provider.fetch()
    assert provider.timeout_seconds == 12
    assert provider.user_agent == "PAISA test agent/1.0"
    assert session.calls[0][1]["headers"]["User-Agent"] == "PAISA test agent/1.0"


def test_manual_csv_preserves_bytes_and_flexible_columns_as_imported(tmp_path):
    path = tmp_path / "permitted-news.csv"
    path.write_text(
        "ID,Headline,Article URL,Description,Article Text,Publication Time,"
        "Original Timezone,Publisher,Lang\n"
        "a-1,OGDCL result,https://example.test/a-1,Summary,Permitted body,"
        "2026-10-21 10:00:00,+05:00,Example Publisher,en\n",
        encoding="utf-8",
    )
    provider = ManualCsvNewsProvider(
        path,
        licensing_usage_note=MANUAL_LICENSE_NOTE,
        clock=lambda: NOW,
    )

    result = provider.fetch()

    assert result.errors == ()
    record = result.records[0]
    assert record.data_class == "imported"
    assert record.provider_record_id == "a-1"
    assert record.title == "OGDCL result"
    assert record.summary == "Summary"
    assert record.body == "Permitted body"
    assert record.published_timezone == "+05:00"
    assert record.source_name == "Example Publisher"
    assert record.raw_payload["provider_mode"] == "manual_import"
    assert record.raw_payload["licensing_usage_note"] == MANUAL_LICENSE_NOTE
    assert provider.effective_config["licensing_usage_note"] == MANUAL_LICENSE_NOTE
    payload = path.read_bytes()
    assert result.captures[0].content == payload
    assert result.captures[0].response_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.captures[0].data_class == "imported"


def test_manual_csv_parse_error_keeps_raw_evidence(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"\xff\xfe-not-utf8")

    result = ManualCsvNewsProvider(
        path,
        licensing_usage_note=MANUAL_LICENSE_NOTE,
        clock=lambda: NOW,
    ).fetch()

    assert result.records == ()
    assert result.errors[0].error_type == "csv_parse_error"
    assert result.captures[0].content == path.read_bytes()
    assert result.captures[0].data_class == "imported"


def test_manual_csv_provider_never_persists_absolute_input_path(tmp_path):
    path = tmp_path / "private-input.csv"
    path.write_text("title\nOGDC update\n", encoding="utf-8")

    result = ManualCsvNewsProvider(
        path,
        licensing_usage_note=MANUAL_LICENSE_NOTE,
        clock=lambda: NOW,
    ).fetch()
    serialized = _serialized_provider_result(result)

    assert str(tmp_path) not in serialized
    assert "manual-csv:private-input.csv" in serialized


def test_manual_csv_requires_nonblank_licensing_usage_note(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text("title\nOGDC update\n", encoding="utf-8")

    with pytest.raises(ValueError, match="licensing_usage_note cannot be blank"):
        ManualCsvNewsProvider(path, licensing_usage_note="   ")


def test_manual_csv_programmatic_default_marks_licensing_review_required(tmp_path):
    path = tmp_path / "news.csv"
    path.write_text("title\nOGDC update\n", encoding="utf-8")

    provider = ManualCsvNewsProvider(path, clock=lambda: NOW)
    result = provider.fetch()

    assert provider.effective_config["licensing_usage_note"] == (
        LICENSING_REVIEW_REQUIRED
    )
    assert result.records[0].raw_payload["licensing_usage_note"] == (
        LICENSING_REVIEW_REQUIRED
    )


def test_fixture_provider_is_deterministic_and_unambiguously_marked_fixture():
    rows = [
        {
            "title": "Fixture OGDC headline",
            "source_url": "https://fixture.example/article",
            "published_at": "2026-10-20T10:00:00Z",
        }
    ]

    first = FixtureNewsProvider(rows, fetched_at=NOW).fetch()
    second = FixtureNewsProvider(rows, fetched_at=NOW).fetch()

    assert first == second
    assert first.records[0].data_class == "fixture"
    assert first.captures[0].data_class == "fixture"
    assert first.captures[0].response_sha256 == second.captures[0].response_sha256
