from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import unicodedata
from typing import Iterable

from paisa.news.models import DuplicateRelationship, NormalizedNewsArticle


def normalized_dedup_key(value: str | None) -> str | None:
    """Return the shared Unicode/case/whitespace key used by dedup rules."""

    if value is None:
        return None
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split()) or None


def title_source_time_key(
    *,
    title: str | None,
    source_name: str | None,
    published_at: datetime | None,
) -> tuple[str, str, str] | None:
    """Build the exact third-rule key used both within and across runs."""

    title_key = normalized_dedup_key(title)
    source_key = normalized_dedup_key(source_name)
    if not title_key or not source_key or published_at is None:
        return None
    published_minute = (
        published_at.astimezone(timezone.utc)
        .replace(second=0, microsecond=0)
        .isoformat()
    )
    return title_key, source_key, published_minute


def _article_title_source_time_key(
    article: NormalizedNewsArticle,
) -> tuple[str, str, str] | None:
    return title_source_time_key(
        title=article.title,
        source_name=article.source_name,
        published_at=article.published_at,
    )


def deduplicate_articles(
    articles: Iterable[NormalizedNewsArticle],
    *,
    existing_articles: Iterable[NormalizedNewsArticle] = (),
) -> tuple[list[NormalizedNewsArticle], list[DuplicateRelationship]]:
    """Apply ordered deterministic rules while retaining every occurrence."""

    by_url: dict[str, str] = {}
    by_hash: dict[str, str] = {}
    by_title_time: dict[tuple[str, str, str], str] = {}

    def index(article: NormalizedNewsArticle) -> None:
        canonical_id = article.duplicate_of or article.article_id
        if article.canonical_url:
            by_url.setdefault(article.canonical_url, canonical_id)
        by_hash.setdefault(article.content_sha256, canonical_id)
        title_key = _article_title_source_time_key(article)
        if title_key:
            by_title_time.setdefault(title_key, canonical_id)

    for existing in existing_articles:
        if existing.duplicate_of is None:
            index(existing)

    resolved: list[NormalizedNewsArticle] = []
    relationships: list[DuplicateRelationship] = []
    for article in articles:
        duplicate_of: str | None = None
        rule: str | None = None
        if article.canonical_url and article.canonical_url in by_url:
            duplicate_of = by_url[article.canonical_url]
            rule = "canonical_url"
        elif article.content_sha256 in by_hash:
            duplicate_of = by_hash[article.content_sha256]
            rule = "content_sha256"
        else:
            title_key = _article_title_source_time_key(article)
            if title_key and title_key in by_title_time:
                duplicate_of = by_title_time[title_key]
                rule = "title_source_time"

        if duplicate_of is not None and duplicate_of != article.article_id:
            duplicate = replace(article, duplicate_of=duplicate_of)
            resolved.append(duplicate)
            relationships.append(
                DuplicateRelationship(
                    article_id=article.article_id,
                    canonical_article_id=duplicate_of,
                    rule=rule,  # type: ignore[arg-type]
                )
            )
        else:
            resolved.append(article)
            index(article)
    return resolved, relationships
