"""Deterministic data-quality reporting for news-ingestion runs.

The report intentionally describes missing or invalid evidence instead of
repairing it.  In particular, missing publication timestamps remain missing;
future feature builders must use the article's independently-derived
``available_at`` cutoff.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any
from paisa.news.models import (
    NEWS_SCHEMA_VERSION,
    ArticleEntityLink,
    DuplicateRelationship,
    NormalizedNewsArticle,
    ProviderError,
    RawNewsRecord,
    utc_iso,
)
from paisa.news.normalization import canonicalize_url


QUALITY_REPORT_SCHEMA_VERSION = "1.0"

QUALITY_CRITERIA = {
    "FAIL": [
        "total_raw_records is zero",
        "total_normalized_records is zero",
    ],
    "WARN": [
        "one or more provider errors were recorded",
        "one or more records were rejected",
        "one or more normalized articles have a missing title",
        "one or more normalized articles have a missing publication time",
        "one or more normalized articles have an invalid URL",
    ],
    "PASS": [
        "neither FAIL nor WARN criteria apply",
        "missing article bodies alone are informational and do not lower status",
    ],
}


def _missing_text(value: str | None) -> bool:
    return value is None or not value.strip()


def _article_has_invalid_url(article: NormalizedNewsArticle) -> bool:
    """Count articles containing a present but invalid source/canonical URL."""

    if "source_url_invalid" in article.normalization_warnings:
        return True
    urls = (article.source_url, article.canonical_url)
    return any(
        value is not None
        and bool(value.strip())
        and canonicalize_url(value) is None
        for value in urls
    )


def _ordered_counts(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def _utc_date(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Quality-report timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).date().isoformat()


def _time_bounds(values: Sequence[datetime]) -> tuple[str | None, str | None]:
    if not values:
        return None, None
    return utc_iso(min(values)), utc_iso(max(values))


def _rejected_record_details(rejected_records: Sequence[Any]) -> list[dict[str, str | None]]:
    """Reduce rejection inputs to stable IDs and exact, user-supplied reasons."""

    details: list[dict[str, str | None]] = []
    for item in rejected_records:
        if is_dataclass(item):
            item = asdict(item)

        record_id: Any = None
        reason: Any = None
        if isinstance(item, Mapping):
            record_id = item.get("raw_record_id", item.get("record_id"))
            reason = item.get("reason", item.get("rejection_reason"))
        elif isinstance(item, tuple) and len(item) == 2:
            record_id, reason = item
        elif isinstance(item, str):
            reason = item
        else:
            record_id = getattr(item, "raw_record_id", getattr(item, "record_id", None))
            reason = getattr(item, "reason", getattr(item, "rejection_reason", None))

        if reason is None or not str(reason).strip():
            raise ValueError("Every rejected record must include an exact reason.")
        details.append(
            {
                "raw_record_id": None if record_id is None else str(record_id),
                "reason": str(reason),
            }
        )

    return sorted(details, key=lambda item: (item["raw_record_id"] or "", item["reason"] or ""))


def _provider_error_counts(provider_errors: Sequence[ProviderError | Mapping[str, Any]]) -> dict[str, int]:
    provider_ids: list[str] = []
    for error in provider_errors:
        if isinstance(error, Mapping):
            provider_id = error.get("provider_id")
        else:
            provider_id = error.provider_id
        provider_ids.append(str(provider_id or "unknown"))
    return _ordered_counts(provider_ids)


def calculate_quality(
    raw_records: Sequence[RawNewsRecord],
    normalized_articles: Sequence[NormalizedNewsArticle],
    duplicate_relationships: Sequence[DuplicateRelationship],
    entity_links: Sequence[ArticleEntityLink],
    rejected_records: Sequence[Any],
    provider_errors: Sequence[ProviderError | Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a deterministic, JSON-serializable ingestion quality report.

    ``FAIL`` takes precedence over ``WARN``.  Empty raw or normalized inputs
    fail the minimum-quality gate.  Provider errors, rejected records, missing
    titles/publication times, and invalid URLs warn.  Missing bodies are
    reported but are informational because many permitted feeds expose only a
    title and summary.
    """

    articles = list(normalized_articles)
    raw = list(raw_records)
    duplicates = list(duplicate_relationships)
    links = list(entity_links)
    errors = list(provider_errors)
    rejected_details = _rejected_record_details(list(rejected_records))

    missing_title_count = sum(_missing_text(article.title) for article in articles)
    missing_publication_time_count = sum(article.published_at is None for article in articles)
    missing_body_count = sum(_missing_text(article.body) for article in articles)
    invalid_url_count = sum(_article_has_invalid_url(article) for article in articles)

    language_counts = _ordered_counts(
        [(article.language or "unknown").strip().lower() or "unknown" for article in articles]
    )
    provider_counts = _ordered_counts([article.provider for article in articles])
    raw_provider_counts = _ordered_counts([record.provider_id for record in raw])
    daily_article_counts = _ordered_counts(
        [_utc_date(article.published_at) for article in articles if article.published_at is not None]
    )

    linked_ogdc_article_ids = {
        link.article_id for link in links if link.ticker.strip().upper() == "OGDC"
    }
    publication_times = [
        article.published_at for article in articles if article.published_at is not None
    ]
    available_times = [article.available_at for article in articles]
    earliest_publication_time, latest_publication_time = _time_bounds(publication_times)
    earliest_available_time, latest_available_time = _time_bounds(available_times)

    rejection_reason_counts = _ordered_counts(
        [detail["reason"] or "unknown" for detail in rejected_details]
    )
    provider_error_count = len(errors)

    if not raw or not articles:
        quality_status = "FAIL"
    elif (
        provider_error_count
        or rejected_details
        or missing_title_count
        or missing_publication_time_count
        or invalid_url_count
    ):
        quality_status = "WARN"
    else:
        quality_status = "PASS"

    return {
        "schema_version": QUALITY_REPORT_SCHEMA_VERSION,
        "news_record_schema_version": NEWS_SCHEMA_VERSION,
        "quality_status": quality_status,
        "minimum_quality_status": quality_status,
        "quality_criteria": QUALITY_CRITERIA,
        "total_raw_records": len(raw),
        "total_normalized_records": len(articles),
        "duplicate_count": len(duplicates),
        "missing_title_count": missing_title_count,
        "missing_publication_time_count": missing_publication_time_count,
        "missing_body_count": missing_body_count,
        "invalid_url_count": invalid_url_count,
        "language_counts": language_counts,
        "provider_counts": provider_counts,
        "raw_provider_counts": raw_provider_counts,
        "daily_article_counts": daily_article_counts,
        "ogdc_linked_article_count": len(linked_ogdc_article_ids),
        "earliest_publication_time": earliest_publication_time,
        "latest_publication_time": latest_publication_time,
        "earliest_available_time": earliest_available_time,
        "latest_available_time": latest_available_time,
        "rejected_record_count": len(rejected_details),
        "rejected_records": rejected_details,
        "rejection_reason_counts": rejection_reason_counts,
        "provider_error_count": provider_error_count,
        "provider_error_counts": _provider_error_counts(errors),
    }


# Descriptive public name used by the orchestration layer.  Keep the shorter
# spelling as a compatibility convenience for direct analytical callers.
calculate_quality_report = calculate_quality
