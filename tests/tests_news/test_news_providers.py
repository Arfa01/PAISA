from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json

import pytest

from paisa.news.providers.fixture import FixtureNewsProvider
from paisa.news.providers.manual_csv import ManualCsvNewsProvider
from paisa.news.providers.rss import (
    RssNewsProvider,
    RssSourceConfig,
    load_rss_config,
)


NOW = datetime(2026, 10, 22, 9, 30, tzinfo=timezone.utc)


class FakeResponse:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        content_type: str = "application/rss+xml; charset=utf-8",
        url: str = "https://publisher.example/feed.xml",
    ) -> None:
        self.content = content
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.url = url


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
        "https://publisher.example:bad/feed.xml",
    ],
)
def test_rss_source_rejects_credential_bearing_or_invalid_urls(url: str):
    with pytest.raises(ValueError):
        RssSourceConfig("unsafe", "Unsafe", url)


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
    provider = ManualCsvNewsProvider(path, clock=lambda: NOW)

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
    payload = path.read_bytes()
    assert result.captures[0].content == payload
    assert result.captures[0].response_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.captures[0].data_class == "imported"


def test_manual_csv_parse_error_keeps_raw_evidence(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"\xff\xfe-not-utf8")

    result = ManualCsvNewsProvider(path, clock=lambda: NOW).fetch()

    assert result.records == ()
    assert result.errors[0].error_type == "csv_parse_error"
    assert result.captures[0].content == path.read_bytes()
    assert result.captures[0].data_class == "imported"


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
