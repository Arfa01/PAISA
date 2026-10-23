from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from paisa.news.models import RawNewsRecord
from paisa.news.normalization import (
    available_by,
    canonicalize_url,
    content_sha256,
    normalize_record,
    parse_timestamp,
)


FETCHED_AT = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


def raw_record(**overrides) -> RawNewsRecord:
    values = {
        "raw_record_id": "rss-source-001",
        "provider_id": "rss",
        "configured_source": "https://example.com/feed.xml",
        "source_name": "Example Business",
        "source_type": "rss",
        "fetched_at": FETCHED_AT,
        "data_class": "real",
        "source_url": "HTTPS://Example.COM:443/news/ogdc/?utm_source=rss&b=2&a=1#top",
        "title": "OGDC announces discovery",
        "summary": "New discovery details.",
        "body": None,
        "published_at_raw": "Thu, 22 Oct 2026 16:30:00 +0500",
        "language": "EN_us",
    }
    values.update(overrides)
    return RawNewsRecord(**values)


def test_raw_record_preserves_the_original_positional_argument_order():
    record = RawNewsRecord(
        "record-1",
        "rss",
        "https://example.com/feed.xml",
        "Example Business",
        "rss",
        FETCHED_AT,
        "real",
        "https://example.com/article",
        "OGDC update",
        "Summary",
        None,
        "2026-10-22T10:00:00Z",
        "UTC",
        "2026-10-22T12:00:00Z",
        "en",
        "provider-record-1",
        "capture-1",
        200,
        "application/rss+xml",
        "a" * 64,
        {"provider": "payload"},
        ("body_missing",),
    )

    assert record.language == "en"
    assert record.provider_record_id == "provider-record-1"
    assert record.raw_payload == {"provider": "payload"}
    assert record.provider_warnings == ("body_missing",)
    assert record.source_observed_at_raw is None


def test_timestamp_parsing_supports_rfc_and_iso_timezone_conversion():
    rfc, warnings = parse_timestamp("Thu, 22 Oct 2026 16:30:00 +0500")
    iso, iso_warnings = parse_timestamp("2026-10-22T16:30:00+05:00")

    assert warnings == ()
    assert iso_warnings == ()
    assert rfc == iso == datetime(2026, 10, 22, 11, 30, tzinfo=timezone.utc)


def test_naive_and_date_only_publication_times_are_not_invented():
    naive, naive_warnings = parse_timestamp("2026-10-22 16:30:00")
    date_only, date_warnings = parse_timestamp(
        "2026-10-22", timezone_evidence="Asia/Karachi"
    )

    assert naive is None
    assert naive_warnings == ("timestamp_missing_timezone",)
    assert date_only is None
    assert date_warnings == ("timestamp_date_only",)


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-22Z",
        "2026-10-22+05:00",
        "20261022",
        "Thu, 22 Oct 2026",
    ],
)
def test_date_only_variants_never_gain_an_invented_midnight(value: str):
    parsed, warnings = parse_timestamp(value, timezone_evidence="Asia/Karachi")

    assert parsed is None
    assert warnings == ("timestamp_date_only",)


def test_configured_timezone_can_make_naive_clock_time_defensible():
    parsed, warnings = parse_timestamp(
        "2026-10-22 16:30:00", timezone_evidence="Asia/Karachi"
    )
    assert warnings == ()
    assert parsed == datetime(2026, 10, 22, 11, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("value", "expected_warning"),
    [
        ("2024-11-03 01:30:00", "timestamp_ambiguous_local_time"),
        ("2024-03-10 02:30:00", "timestamp_nonexistent_local_time"),
    ],
)
def test_iana_timezone_rejects_ambiguous_or_nonexistent_wall_times(
    value: str, expected_warning: str
):
    parsed, warnings = parse_timestamp(
        value,
        timezone_evidence="America/New_York",
    )

    assert parsed is None
    assert warnings == (expected_warning,)


def test_aware_timestamp_rejects_conflicting_explicit_timezone_evidence():
    parsed, warnings = parse_timestamp(
        "2026-10-22T16:30:00+05:00",
        timezone_evidence="UTC",
        field_name="published_at",
    )

    assert parsed is None
    assert warnings == ("published_at_timezone_conflict",)

    matching, matching_warnings = parse_timestamp(
        "2026-10-22T16:30:00+05:00",
        timezone_evidence="Asia/Karachi",
        field_name="published_at",
    )
    assert matching == datetime(2026, 10, 22, 11, 30, tzinfo=timezone.utc)
    assert matching_warnings == ()


@pytest.mark.parametrize(
    ("published_at_raw", "published_timezone", "expected_warning"),
    [
        (
            "2024-11-03 01:30:00",
            "America/New_York",
            "published_at_ambiguous_local_time",
        ),
        (
            "2024-03-10 02:30:00",
            "America/New_York",
            "published_at_nonexistent_local_time",
        ),
        (
            "2026-10-22T16:30:00+05:00",
            "UTC",
            "published_at_timezone_conflict",
        ),
    ],
)
def test_normalization_fails_closed_for_unsafe_publication_time_evidence(
    published_at_raw: str,
    published_timezone: str,
    expected_warning: str,
):
    article = normalize_record(
        raw_record(
            published_at_raw=published_at_raw,
            published_timezone=published_timezone,
        ),
        ingestion_run_id="NEWS-20261022-120000-abcd1234",
        raw_artifact_reference="raw/rss-source-001.json",
    )

    assert article.published_at is None
    assert article.available_at == FETCHED_AT
    assert expected_warning in article.normalization_warnings


def test_normalization_preserves_missing_publication_and_uses_fetch_availability():
    article = normalize_record(
        raw_record(published_at_raw=None),
        ingestion_run_id="NEWS-20261022-120000-abcd1234",
        raw_artifact_reference="raw/rss-source-001.json",
    )

    assert article.published_at is None
    assert article.available_at == FETCHED_AT
    assert "published_at_missing" in article.normalization_warnings
    assert article.published_timezone is None


def test_future_publication_cannot_enter_an_earlier_feature_cutoff():
    article = normalize_record(
        raw_record(published_at_raw="2026-10-23T09:00:00Z"),
        ingestion_run_id="NEWS-20261022-120000-abcd1234",
        raw_artifact_reference="raw/rss-source-001.json",
    )

    assert article.available_at == datetime(2026, 10, 23, 9, tzinfo=timezone.utc)
    assert not available_by(article, datetime(2026, 10, 23, 8, 59, tzinfo=timezone.utc))
    assert available_by(article, datetime(2026, 10, 23, 9, 0, tzinfo=timezone.utc))


def test_available_at_never_precedes_fetch_or_source_availability():
    article = normalize_record(
        raw_record(
            published_at_raw="2026-10-22T10:00:00Z",
            available_at_raw="2026-10-22T13:00:00Z",
        ),
        ingestion_run_id="NEWS-20261022-120000-abcd1234",
        raw_artifact_reference="raw/rss-source-001.json",
    )
    assert article.available_at == datetime(2026, 10, 22, 13, tzinfo=timezone.utc)


def test_canonical_url_normalization_is_conservative_and_stable():
    assert canonicalize_url(
        "HTTPS://Example.COM:443/news//ogdc/?utm_source=rss&b=2&a=1#top"
    ) == "https://example.com/news/ogdc?a=1&b=2"
    assert canonicalize_url("javascript:alert(1)") is None
    assert canonicalize_url("https://user:secret@example.com/story") is None


def test_content_hash_uses_normalized_text_and_is_deterministic():
    first = content_sha256(
        title=" OGDC   update ", summary="<p>Production &amp; output</p>", body=None
    )
    second = content_sha256(
        title="OGDC update", summary="Production & output", body=None
    )
    assert first == second
    assert len(first) == 64


def test_title_only_hash_is_scoped_by_source_and_publication_context():
    first = normalize_record(
        raw_record(
            configured_source="https://publisher-one.example/feed",
            source_name="Publisher One",
            title="Market update",
            summary=None,
            body=None,
            published_at_raw="2025-01-02T08:00:00Z",
        ),
        ingestion_run_id="NEWS-20261022-120000-sparse0001",
        raw_artifact_reference="raw/one.json",
    )
    second = normalize_record(
        raw_record(
            raw_record_id="rss-source-002",
            configured_source="https://publisher-two.example/feed",
            source_name="Publisher Two",
            title="Market update",
            summary=None,
            body=None,
            published_at_raw="2026-01-02T08:00:00Z",
        ),
        ingestion_run_id="NEWS-20261022-120000-sparse0001",
        raw_artifact_reference="raw/two.json",
    )

    assert first.content_sha256 != second.content_sha256
    assert "content_hash_scoped_to_source_context" in first.normalization_warnings
    assert "content_hash_scoped_to_source_context" in second.normalization_warnings


def test_title_only_hash_is_stable_for_the_same_source_and_publication_context():
    first = normalize_record(
        raw_record(title="Market update", summary=None, body=None),
        ingestion_run_id="NEWS-20261022-120000-sparse0002",
        raw_artifact_reference="raw/one.json",
    )
    second = normalize_record(
        raw_record(
            raw_record_id="rss-source-002",
            title="  Market   update ",
            summary=None,
            body=None,
        ),
        ingestion_run_id="NEWS-20261022-120000-sparse0002",
        raw_artifact_reference="raw/two.json",
    )

    assert first.content_sha256 == second.content_sha256


def test_title_only_hash_rejects_missing_scope_context():
    with pytest.raises(ValueError, match="stable source"):
        content_sha256(title="Market update", summary=None, body=None)


def test_same_captured_record_normalizes_deterministically():
    kwargs = {
        "ingestion_run_id": "NEWS-20261022-120000-abcd1234",
        "raw_artifact_reference": "raw/rss-source-001.json",
    }
    first = normalize_record(raw_record(), **kwargs)
    second = normalize_record(raw_record(), **kwargs)

    assert first == second
    assert first.canonical_url == "https://example.com/news/ogdc?a=1&b=2"
    assert first.language == "en-us"
    assert first.data_class == "real"


def test_cutoff_must_be_timezone_aware():
    article = normalize_record(
        raw_record(),
        ingestion_run_id="NEWS-20261022-120000-abcd1234",
        raw_artifact_reference="raw/rss-source-001.json",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        available_by(article, datetime(2026, 10, 22, 12, 0))


def test_invalid_url_is_retained_as_evidence_and_warned():
    article = normalize_record(
        replace(raw_record(), source_url="not a URL"),
        ingestion_run_id="NEWS-20261022-120000-abcd1234",
        raw_artifact_reference="raw/rss-source-001.json",
    )
    assert article.source_url == "not a URL"
    assert article.canonical_url is None
    assert "source_url_invalid" in article.normalization_warnings
