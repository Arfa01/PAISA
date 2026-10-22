from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import pytest

from paisa.sentiment.config import (
    FINBERT_MODEL_REPOSITORY,
    FINBERT_MODEL_REVISION,
    FINBERT_TOKENIZER_REVISION,
)
from paisa.sentiment.models import BackendPrediction


NOW = datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ExampleArticle:
    article_id: str = "article-1"
    ingestion_run_id: str = "NEWS-20261022-100000-source01"
    title: str | None = "OGDC reports higher production"
    summary: str | None = "The company reported an increase in output."
    body: str | None = "A permitted body excerpt with additional details."
    language: str | None = "en"
    published_at: datetime | None = NOW - timedelta(hours=2)
    available_at: datetime = NOW - timedelta(hours=1)
    fetched_at: datetime = NOW - timedelta(hours=1)


class MockBackend:
    model_repository = FINBERT_MODEL_REPOSITORY
    model_revision = FINBERT_MODEL_REVISION
    tokenizer_revision = FINBERT_TOKENIZER_REVISION
    device = "cpu"

    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    @property
    def library_versions(self) -> Mapping[str, str]:
        return {"python": "test", "torch": "mock", "transformers": "mock"}

    def score_batch(
        self,
        texts: Sequence[str],
        *,
        maximum_sequence_length: int,
    ) -> Sequence[BackendPrediction]:
        self.batches.append(tuple(texts))
        predictions = []
        for text in texts:
            token_count = len(text.split()) + 2
            predictions.append(
                BackendPrediction(
                    negative_probability=0.1,
                    neutral_probability=0.2,
                    positive_probability=0.7,
                    predicted_label="positive",
                    predicted_probability=0.7,
                    tokenizer_token_count=token_count,
                    token_truncation_applied=(
                        token_count > maximum_sequence_length
                    ),
                )
            )
        return tuple(predictions)


@pytest.fixture
def article() -> ExampleArticle:
    return ExampleArticle()


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()
