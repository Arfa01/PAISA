from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

import pytest

from paisa.sentiment.config import (
    FINBERT_LICENSE,
    FINBERT_LICENSE_URL,
    FINBERT_MODEL_REPOSITORY,
    FINBERT_MODEL_REVISION,
    FINBERT_REVIEWED_SNAPSHOT_SHA256,
    FINBERT_TOKENIZER_REVISION,
)
from paisa.sentiment.backend import SentimentBackendIdentity
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
    def identity(self) -> SentimentBackendIdentity:
        return SentimentBackendIdentity(
            model_repository=self.model_repository,
            model_revision=self.model_revision,
            tokenizer_repository=self.model_repository,
            tokenizer_revision=self.tokenizer_revision,
            original_labels_by_index=("positive", "negative", "neutral"),
            canonical_probability_index_mapping={
                "negative": 1,
                "neutral": 2,
                "positive": 0,
            },
            license_identifier=FINBERT_LICENSE,
            license_url=FINBERT_LICENSE_URL,
            snapshot_files_sha256=FINBERT_REVIEWED_SNAPSHOT_SHA256,
        )

    @property
    def snapshot_inventory(self):
        return tuple(
            {
                "path": path,
                "sha256": digest,
                "size_bytes": 1,
            }
            for path, digest in sorted(FINBERT_REVIEWED_SNAPSHOT_SHA256.items())
        )

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
            final_token_count = min(token_count, maximum_sequence_length)
            predictions.append(
                BackendPrediction(
                    negative_probability=0.1,
                    neutral_probability=0.2,
                    positive_probability=0.7,
                    predicted_label="positive",
                    predicted_probability=0.7,
                    tokenizer_token_count=token_count,
                    final_token_count=final_token_count,
                    truncation_side="right",
                    token_truncation_applied=(token_count > final_token_count),
                )
            )
        return tuple(predictions)


@pytest.fixture
def article() -> ExampleArticle:
    return ExampleArticle()


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()
