from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paisa.sentiment.time_alignment import (
    SentimentTimingEvidenceError,
    filter_sentiment_by_decision_cutoff,
)


UTC = timezone.utc


def test_later_news_is_excluded_from_an_earlier_stock_decision():
    cutoff = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    early = {"article_id": "early", "available_at": cutoff}
    late = {
        "article_id": "late",
        "available_at": cutoff + timedelta(seconds=1),
    }

    assert filter_sentiment_by_decision_cutoff([early, late], cutoff) == (early,)


def test_future_correction_cannot_retroactively_enter_an_old_decision():
    cutoff = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    original = {
        "article_id": "article-v1",
        "available_at": cutoff - timedelta(minutes=5),
    }
    later_correction = {
        "article_id": "article-v2",
        "available_at": cutoff + timedelta(hours=1),
    }

    assert filter_sentiment_by_decision_cutoff(
        [original, later_correction], cutoff
    ) == (original,)


def test_inference_timestamp_never_replaces_source_availability():
    cutoff = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
    inferred_later = {
        "article_id": "known-before-cutoff",
        "available_at": cutoff - timedelta(minutes=1),
        "inference_timestamp": cutoff + timedelta(days=10),
    }

    assert filter_sentiment_by_decision_cutoff([inferred_later], cutoff) == (
        inferred_later,
    )


@pytest.mark.parametrize(
    "observation",
    [
        {"article_id": "missing"},
        {"article_id": "none", "available_at": None},
        {
            "article_id": "naive",
            "available_at": datetime(2026, 10, 22, 12, 0),
        },
        {"article_id": "string", "available_at": "2026-10-22T12:00:00Z"},
    ],
)
def test_missing_or_unsupported_timing_evidence_is_never_invented(observation):
    cutoff = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)

    with pytest.raises(SentimentTimingEvidenceError):
        filter_sentiment_by_decision_cutoff([observation], cutoff)


def test_naive_decision_cutoff_is_rejected():
    with pytest.raises(SentimentTimingEvidenceError):
        filter_sentiment_by_decision_cutoff(
            [], datetime(2026, 10, 22, 12, 0)
        )
