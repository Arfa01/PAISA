from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from paisa.news.deduplication import deduplicate_articles
from paisa.news.entity_linking import OGDC_ALIASES, link_ogdc
from paisa.news.models import NormalizedNewsArticle
from paisa.news.normalization import content_sha256


NOW = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)


def article(
    article_id: str,
    *,
    title: str = "Energy update",
    source_name: str = "Example News",
    source_url: str | None = None,
    canonical_url: str | None = None,
    summary: str | None = "Summary",
    body: str | None = None,
    published_at: datetime | None = NOW,
) -> NormalizedNewsArticle:
    return NormalizedNewsArticle(
        schema_version="1.0",
        article_id=article_id,
        raw_record_id=f"raw-{article_id}",
        provider="fixture",
        source_name=source_name,
        source_type="fixture",
        source_url=source_url,
        canonical_url=canonical_url,
        title=title,
        summary=summary,
        body=body,
        published_at=published_at,
        published_timezone="UTC" if published_at else None,
        available_at=NOW,
        fetched_at=NOW,
        language="en",
        mentioned_tickers=(),
        mentioned_companies=(),
        mentioned_sectors=(),
        content_sha256=content_sha256(title=title, summary=summary, body=body),
        duplicate_of=None,
        ingestion_run_id="NEWS-20261022-120000-test0001",
        raw_artifact_reference=f"raw/{article_id}.json",
        normalization_warnings=(),
        data_class="fixture",
    )


@pytest.mark.parametrize("alias", OGDC_ALIASES)
def test_every_controlled_ogdc_alias_links_exactly(alias: str):
    linked, links = link_ogdc(article("a1", title=f"{alias} reports results"))

    assert linked.mentioned_tickers == ("OGDC",)
    assert linked.mentioned_companies == ("Oil and Gas Development Company Limited",)
    assert len(links) == 1
    assert links[0].matched_alias == alias
    assert links[0].match_location == "title"
    assert links[0].confidence in {"exact", "normalized_exact"}


@pytest.mark.parametrize(
    "text",
    [
        "BOGDCAST is an unrelated token",
        "The ogdcx index moved",
        "ordinary gas development remains slow",
        "oil and gas development companies reported results",
    ],
)
def test_ogdc_linker_avoids_substring_and_near_match_false_positives(text: str):
    linked, links = link_ogdc(article("a1", title=text))
    assert links == []
    assert linked.mentioned_tickers == ()


def test_normalized_ampersand_alias_match_is_recorded():
    linked, links = link_ogdc(
        article("a1", title="Oil   &   Gas Development Company reports results")
    )
    assert linked.mentioned_tickers == ("OGDC",)
    assert links[0].match_method == "normalized_exact"


def test_deduplication_rule_priority_is_canonical_url_then_hash():
    first = article(
        "a1",
        title="First",
        summary="First content",
        canonical_url="https://example.com/story",
    )
    second = article(
        "a2",
        title="Different",
        summary="Different content",
        canonical_url="https://example.com/story",
    )
    third = article(
        "a3",
        title="First",
        summary="First content",
        canonical_url="https://mirror.example/story",
    )

    resolved, relationships = deduplicate_articles([first, second, third])

    assert resolved[0].duplicate_of is None
    assert resolved[1].duplicate_of == "a1"
    assert relationships[0].rule == "canonical_url"
    assert resolved[2].duplicate_of == "a1"
    assert relationships[1].rule == "content_sha256"


def test_title_source_publication_minute_is_third_duplicate_rule():
    first = article(
        "a1", title="OGDC Result", summary="one", canonical_url="https://one.test/a"
    )
    second = article(
        "a2", title="  ogdc  result ", summary="two", canonical_url="https://two.test/b"
    )

    resolved, relationships = deduplicate_articles([first, second])
    assert resolved[1].duplicate_of == "a1"
    assert relationships[0].rule == "title_source_time"


def test_title_time_rule_does_not_run_without_publication_time():
    first = article(
        "a1", title="Same title", summary="one", canonical_url="https://one.test/a",
        published_at=None,
    )
    second = article(
        "a2", title="Same title", summary="two", canonical_url="https://two.test/b",
        published_at=None,
    )
    resolved, relationships = deduplicate_articles([first, second])
    assert all(item.duplicate_of is None for item in resolved)
    assert relationships == []


def test_existing_canonical_articles_participate_in_duplicate_detection():
    existing = article("existing", canonical_url="https://example.com/story")
    incoming = article("incoming", canonical_url="https://example.com/story")

    resolved, relationships = deduplicate_articles(
        [incoming], existing_articles=[existing]
    )
    assert resolved[0].duplicate_of == "existing"
    assert relationships[0].canonical_article_id == "existing"


def test_duplicate_evidence_keeps_distinct_occurrence_identity():
    first = article("a1")
    second = replace(article("a2"), content_sha256=first.content_sha256)
    resolved, relationships = deduplicate_articles([first, second])

    assert [item.article_id for item in resolved] == ["a1", "a2"]
    assert resolved[1].raw_record_id == "raw-a2"
    assert relationships[0].article_id == "a2"
