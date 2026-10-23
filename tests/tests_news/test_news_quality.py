from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from paisa.news.models import (
    NEWS_SCHEMA_VERSION,
    ArticleEntityLink,
    DuplicateRelationship,
    NormalizedNewsArticle,
    ProviderError,
    RawNewsRecord,
)
from paisa.news.quality import calculate_quality, calculate_quality_report


UTC = timezone.utc


def _raw(raw_record_id: str = "raw-1", provider_id: str = "manual-csv") -> RawNewsRecord:
    return RawNewsRecord(
        raw_record_id=raw_record_id,
        provider_id=provider_id,
        configured_source="input.csv",
        source_name="Example News",
        source_type="manual_csv",
        fetched_at=datetime(2026, 1, 2, 9, tzinfo=UTC),
        data_class="imported",
    )


def _article(
    article_id: str = "article-1",
    *,
    title: str | None = "OGDC announces discovery",
    body: str | None = "Oil and Gas Development Company Limited announced a discovery.",
    published_at: datetime | None = datetime(2026, 1, 2, 8, tzinfo=UTC),
    available_at: datetime = datetime(2026, 1, 2, 9, tzinfo=UTC),
    source_url: str | None = "https://news.example/story?id=1",
    canonical_url: str | None = "https://news.example/story?id=1",
    language: str | None = "EN",
    provider: str = "manual-csv",
) -> NormalizedNewsArticle:
    return NormalizedNewsArticle(
        schema_version=NEWS_SCHEMA_VERSION,
        article_id=article_id,
        raw_record_id=f"raw-{article_id}",
        provider=provider,
        source_name="Example News",
        source_type="manual_csv",
        source_url=source_url,
        canonical_url=canonical_url,
        title=title,
        summary="Summary",
        body=body,
        published_at=published_at,
        published_timezone="UTC" if published_at else None,
        available_at=available_at,
        fetched_at=available_at,
        language=language,
        mentioned_tickers=("OGDC",),
        mentioned_companies=("Oil and Gas Development Company Limited",),
        mentioned_sectors=("Energy",),
        content_sha256="a" * 64,
        duplicate_of=None,
        ingestion_run_id="NEWS-20260102-090000-abc123",
        raw_artifact_reference=f"raw/{article_id}.json",
        normalization_warnings=(),
        data_class="imported",
    )


def _link(article_id: str = "article-1", ticker: str = "OGDC") -> ArticleEntityLink:
    return ArticleEntityLink(
        link_id=f"link-{article_id}-{ticker}",
        article_id=article_id,
        ticker=ticker,
        company_name="Oil and Gas Development Company Limited",
        matched_alias="OGDC",
        match_method="exact",
        match_location="title",
        confidence="exact",
    )


def test_news_schema_1_0_cannot_claim_schema_1_1_observation_provenance() -> None:
    legacy = replace(_article(), schema_version="1.0")

    assert "source_observed_at" not in legacy.to_dict()
    with pytest.raises(ValueError, match="schema 1.1"):
        replace(
            legacy,
            source_observed_at=datetime(2026, 1, 2, 8, 30, tzinfo=UTC),
        )


def test_quality_report_calculates_all_metrics_deterministically() -> None:
    assert calculate_quality_report is calculate_quality
    articles = [
        _article(),
        replace(
            _article(
                "article-2",
                title=" ",
                body=None,
                published_at=None,
                available_at=datetime(2026, 1, 3, 9, tzinfo=UTC),
                source_url="not a URL",
                canonical_url=None,
                language=None,
                provider="rss",
            ),
            duplicate_of="article-1",
        ),
    ]
    duplicates = [
        DuplicateRelationship(
            article_id="article-2",
            canonical_article_id="article-1",
            rule="content_sha256",
            duplicate_scope="within_run",
            canonical_owner_run_id="NEWS-20260102-090000-abc123",
        )
    ]
    errors = [
        ProviderError(
            provider_id="rss",
            configured_source="https://feed.example/rss",
            error_type="HTTPError",
            message="503",
            occurred_at=datetime(2026, 1, 3, 9, tzinfo=UTC),
            http_status=503,
        )
    ]

    report = calculate_quality(
        raw_records=[_raw("raw-2", "rss"), _raw("raw-1")],
        normalized_articles=articles,
        duplicate_relationships=duplicates,
        entity_links=[_link(), _link(), _link("article-2", "HBL")],
        rejected_records=[
            {"raw_record_id": "raw-4", "reason": "invalid row"},
            {"raw_record_id": "raw-3", "reason": "missing source"},
            {"raw_record_id": "raw-5", "reason": "invalid row"},
        ],
        provider_errors=errors,
    )

    assert report["schema_version"] == "2.0"
    assert report["quality_status"] == report["minimum_quality_status"] == "WARN"
    assert report["total_raw_records"] == 2
    assert report["total_normalized_records"] == 2
    assert report["duplicate_count"] == 1
    assert report["duplicate_relationship_count"] == 1
    assert report["normalized_occurrence_count"] == 2
    assert report["new_canonical_article_count"] == 1
    assert report["matched_existing_canonical_count"] == 0
    assert report["within_run_duplicate_count"] == 1
    assert report["historical_duplicate_count"] == 0
    assert report["represented_canonical_count"] == 1
    assert report["deduplication_rule_counts"] == {
        "canonical_url": 0,
        "content_sha256": 1,
        "title_source_time": 0,
    }
    assert report["missing_title_count"] == 1
    assert report["missing_publication_time_count"] == 1
    assert report["missing_body_count"] == 1
    assert report["invalid_url_count"] == 1
    assert report["language_counts"] == {"en": 1, "unknown": 1}
    assert report["provider_counts"] == {"manual-csv": 1, "rss": 1}
    assert report["raw_provider_counts"] == {"manual-csv": 1, "rss": 1}
    assert report["daily_article_counts"] == {"2026-01-02": 1}
    assert report["ogdc_linked_article_count"] == 1
    assert report["earliest_publication_time"] == "2026-01-02T08:00:00Z"
    assert report["latest_publication_time"] == "2026-01-02T08:00:00Z"
    assert report["earliest_available_time"] == "2026-01-02T09:00:00Z"
    assert report["latest_available_time"] == "2026-01-03T09:00:00Z"
    assert report["rejected_record_count"] == 3
    assert report["rejection_reason_counts"] == {"invalid row": 2, "missing source": 1}
    assert report["provider_error_count"] == 1
    assert report["provider_error_counts"] == {"rss": 1}
    assert list(report["language_counts"]) == ["en", "unknown"]
    assert json.loads(json.dumps(report)) == report


def test_quality_report_distinguishes_within_run_and_historical_duplicates() -> None:
    run_id = "NEWS-20260102-090000-abc123"
    canonical = _article("article-1")
    within_run = replace(_article("article-2"), duplicate_of="article-1")
    historical = replace(_article("article-3"), duplicate_of="historical-1")
    relationships = [
        DuplicateRelationship(
            article_id=within_run.article_id,
            canonical_article_id="article-1",
            rule="canonical_url",
            duplicate_scope="within_run",
            canonical_owner_run_id=run_id,
        ),
        DuplicateRelationship(
            article_id=historical.article_id,
            canonical_article_id="historical-1",
            rule="title_source_time",
            duplicate_scope="historical",
            canonical_owner_run_id="NEWS-20260101-090000-owner001",
        ),
    ]

    report = calculate_quality(
        raw_records=[_raw("raw-1"), _raw("raw-2"), _raw("raw-3")],
        normalized_articles=[canonical, within_run, historical],
        duplicate_relationships=relationships,
        entity_links=[],
        rejected_records=[],
        provider_errors=[],
    )

    assert report["normalized_occurrence_count"] == 3
    assert report["new_canonical_article_count"] == 1
    assert report["matched_existing_canonical_count"] == 1
    assert report["within_run_duplicate_count"] == 1
    assert report["historical_duplicate_count"] == 1
    assert report["represented_canonical_count"] == 2
    assert report["duplicate_relationship_count"] == 2
    assert report["duplicate_count"] == 2
    assert report["deduplication_rule_counts"] == {
        "canonical_url": 1,
        "content_sha256": 0,
        "title_source_time": 1,
    }


@pytest.mark.parametrize("raw_count,article_count", [(0, 1), (1, 0), (0, 0)])
def test_quality_fails_when_raw_or_normalized_records_are_empty(
    raw_count: int, article_count: int
) -> None:
    report = calculate_quality(
        raw_records=[_raw()] * raw_count,
        normalized_articles=[_article()] * article_count,
        duplicate_relationships=[],
        entity_links=[],
        rejected_records=[],
        provider_errors=[],
    )

    assert report["quality_status"] == "FAIL"


def test_missing_body_alone_is_informational() -> None:
    report = calculate_quality(
        raw_records=[_raw()],
        normalized_articles=[_article(body="  ")],
        duplicate_relationships=[],
        entity_links=[],
        rejected_records=[],
        provider_errors=[],
    )

    assert report["missing_body_count"] == 1
    assert report["quality_status"] == "PASS"


@pytest.mark.parametrize(
    "override",
    [
        {"title": None},
        {"published_at": None},
        {"source_url": "ftp://example.test/story"},
    ],
)
def test_warn_conditions(override: dict[str, object]) -> None:
    report = calculate_quality(
        raw_records=[_raw()],
        normalized_articles=[_article(**override)],
        duplicate_relationships=[],
        entity_links=[],
        rejected_records=[],
        provider_errors=[],
    )

    assert report["quality_status"] == "WARN"


@pytest.mark.parametrize(
    "source_url",
    [
        "https://user:secret@example.test/story",
        "https://example.test:bad/story",
    ],
)
def test_quality_uses_the_same_strict_url_contract_as_normalization(
    source_url: str,
) -> None:
    report = calculate_quality(
        raw_records=[_raw()],
        normalized_articles=[_article(source_url=source_url, canonical_url=None)],
        duplicate_relationships=[],
        entity_links=[],
        rejected_records=[],
        provider_errors=[],
    )

    assert report["invalid_url_count"] == 1
    assert report["quality_status"] == "WARN"


def test_rejected_record_requires_an_exact_reason() -> None:
    with pytest.raises(ValueError, match="exact reason"):
        calculate_quality(
            raw_records=[_raw()],
            normalized_articles=[_article()],
            duplicate_relationships=[],
            entity_links=[],
            rejected_records=[{"raw_record_id": "raw-bad"}],
            provider_errors=[],
        )
