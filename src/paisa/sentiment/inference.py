"""Memory-safe article sentiment inference orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import islice
from time import monotonic
from typing import Iterable, Iterator, Protocol, Union

from paisa.sentiment.backend import SentimentInferenceBackend
from paisa.sentiment.config import (
    CANONICAL_SENTIMENT_LABELS,
    DEFAULT_INPUT_POLICY_ID,
    DEFAULT_MAXIMUM_INPUT_CHARACTERS,
    DEFAULT_MAXIMUM_TOKENIZER_LENGTH,
)
from paisa.sentiment.models import (
    ArticleSentiment,
    ArticleSentimentError,
    BackendPrediction,
    TextSelection,
    utc_iso,
)
from paisa.sentiment.text import MissingArticleTextError, select_article_text


class InferenceArticle(Protocol):
    article_id: str
    ingestion_run_id: str
    title: str | None
    summary: str | None
    body: str | None
    language: str | None
    published_at: datetime | None
    available_at: datetime
    fetched_at: datetime


class SentimentInferenceTimeoutError(TimeoutError):
    """Raised between batches when the configured watchdog has expired."""


SentimentOutcome = Union[ArticleSentiment, ArticleSentimentError]


@dataclass(frozen=True)
class SentimentInferenceConfig:
    batch_size: int = 8
    maximum_input_characters: int = DEFAULT_MAXIMUM_INPUT_CHARACTERS
    maximum_sequence_length: int = DEFAULT_MAXIMUM_TOKENIZER_LENGTH
    maximum_articles: int | None = 1_000
    timeout_seconds: float | None = None
    input_policy_id: str = DEFAULT_INPUT_POLICY_ID

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.maximum_input_characters <= 0:
            raise ValueError("maximum_input_characters must be positive.")
        if not 1 <= self.maximum_sequence_length <= 512:
            raise ValueError("maximum_sequence_length must be between 1 and 512.")
        if self.maximum_articles is not None and self.maximum_articles <= 0:
            raise ValueError("maximum_articles must be positive when supplied.")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive when supplied.")
        if not self.input_policy_id.strip():
            raise ValueError("input_policy_id cannot be blank.")

    @property
    def effective_input_policy_id(self) -> str:
        """Identity of every setting that can change the selected model input."""

        return (
            f"{self.input_policy_id}-chars{self.maximum_input_characters}"
            f"-tokens{self.maximum_sequence_length}"
        )


def _language_is_supported(language: str | None) -> bool:
    if language is None:
        return False
    normalized = language.strip().lower().replace("_", "-")
    return normalized == "en" or normalized.startswith("en-")


def _validate_source_timing(article: InferenceArticle) -> None:
    utc_iso(article.published_at)
    utc_iso(article.available_at)
    utc_iso(article.fetched_at)


def _safe_error(
    article: InferenceArticle,
    backend: SentimentInferenceBackend,
    config: SentimentInferenceConfig,
    inference_timestamp: datetime,
    *,
    error_code: str,
    error_message: str,
    selection: TextSelection | None = None,
) -> ArticleSentimentError:
    return ArticleSentimentError(
        article_id=article.article_id,
        source_ingestion_run_id=article.ingestion_run_id,
        input_policy_id=config.effective_input_policy_id,
        selected_fields=selection.selected_fields if selection else (),
        input_text_sha256=selection.input_text_sha256 if selection else None,
        original_language=article.language,
        model_repository=backend.model_repository,
        model_revision=backend.model_revision,
        tokenizer_revision=backend.tokenizer_revision,
        published_at=article.published_at,
        available_at=article.available_at,
        fetched_at=article.fetched_at,
        inference_timestamp=inference_timestamp,
        error_code=error_code,
        error_message=error_message,
    )

def _success(
    article: InferenceArticle,
    selection: TextSelection,
    prediction: BackendPrediction,
    backend: SentimentInferenceBackend,
    config: SentimentInferenceConfig,
    inference_timestamp: datetime,
) -> ArticleSentiment:
    warning_codes = (
        ("input_character_truncated",)
        if selection.character_truncation_applied
        else ()
    )
    if prediction.token_truncation_applied:
        warning_codes += ("input_token_truncated",)
    return ArticleSentiment(
        article_id=article.article_id,
        source_ingestion_run_id=article.ingestion_run_id,
        input_policy_id=selection.policy_id,
        selected_fields=selection.selected_fields,
        input_text_sha256=selection.input_text_sha256,
        character_count=selection.character_count,
        original_character_count=selection.original_character_count,
        tokenizer_token_count=prediction.tokenizer_token_count,
        final_token_count=prediction.final_token_count,
        truncation_side=prediction.truncation_side,
        truncation_applied=(
            selection.character_truncation_applied
            or prediction.token_truncation_applied
        ),
        character_truncation_applied=selection.character_truncation_applied,
        token_truncation_applied=prediction.token_truncation_applied,
        maximum_input_characters=config.maximum_input_characters,
        maximum_sequence_length=config.maximum_sequence_length,
        original_language=article.language,
        model_repository=backend.model_repository,
        model_revision=backend.model_revision,
        tokenizer_revision=backend.tokenizer_revision,
        class_labels=CANONICAL_SENTIMENT_LABELS,
        negative_probability=prediction.negative_probability,
        neutral_probability=prediction.neutral_probability,
        positive_probability=prediction.positive_probability,
        predicted_label=prediction.predicted_label,
        predicted_probability=prediction.predicted_probability,
        published_at=article.published_at,
        available_at=article.available_at,
        fetched_at=article.fetched_at,
        inference_timestamp=inference_timestamp,
        device=backend.device,
        batch_size=config.batch_size,
        library_versions=backend.library_versions,
        warning_codes=warning_codes,
    )


def iter_article_sentiment(
    articles: Iterable[InferenceArticle],
    *,
    backend: SentimentInferenceBackend,
    config: SentimentInferenceConfig | None = None,
    inference_timestamp: datetime | None = None,
) -> Iterator[SentimentOutcome]:
    """Score bounded batches, yielding successes and safe per-article errors."""

    resolved_config = config or SentimentInferenceConfig()
    timestamp = inference_timestamp or datetime.now(timezone.utc)
    utc_iso(timestamp)
    started = monotonic()
    source = iter(articles)
    if resolved_config.maximum_articles is not None:
        source = iter(islice(source, resolved_config.maximum_articles))

    pending: list[tuple[InferenceArticle, TextSelection]] = []

    def check_watchdog() -> None:
        if (
            resolved_config.timeout_seconds is not None
            and monotonic() - started > resolved_config.timeout_seconds
        ):
            raise SentimentInferenceTimeoutError(
                "Sentiment inference watchdog expired between batches."
            )

    def flush() -> list[SentimentOutcome]:
        if not pending:
            return []
        check_watchdog()
        current = tuple(pending)
        pending.clear()
        try:
            predictions = tuple(
                backend.score_batch(
                    [selection.input_text for _, selection in current],
                    maximum_sequence_length=resolved_config.maximum_sequence_length,
                )
            )
            if len(predictions) != len(current):
                raise ValueError("Backend returned an unexpected prediction count.")
            outcomes: list[SentimentOutcome] = [
                _success(
                    article,
                    selection,
                    prediction,
                    backend,
                    resolved_config,
                    timestamp,
                )
                for (article, selection), prediction in zip(current, predictions)
            ]
        except Exception:
            outcomes = [
                _safe_error(
                    article,
                    backend,
                    resolved_config,
                    timestamp,
                    error_code="backend_inference_failed",
                    error_message="Pinned sentiment backend inference failed.",
                    selection=selection,
                )
                for article, selection in current
            ]
        check_watchdog()
        return outcomes

    for article in source:
        try:
            _validate_source_timing(article)
        except (TypeError, ValueError):
            # The error schema itself requires defensible source timing, so
            # unsupported timing evidence must be rejected rather than invented.
            raise ValueError(
                f"Article {article.article_id!r} lacks timezone-aware timing evidence."
            ) from None

        if not _language_is_supported(article.language):
            yield from flush()
            yield _safe_error(
                article,
                backend,
                resolved_config,
                timestamp,
                error_code="unsupported_language",
                error_message="Pinned FinBERT supports English-language text only.",
            )
            continue
        try:
            selection = select_article_text(
                article,
                maximum_input_characters=resolved_config.maximum_input_characters,
                policy_id=resolved_config.effective_input_policy_id,
            )
        except MissingArticleTextError:
            yield from flush()
            yield _safe_error(
                article,
                backend,
                resolved_config,
                timestamp,
                error_code="missing_input_text",
                error_message="Article has no usable text for the input policy.",
            )
            continue
        pending.append((article, selection))
        if len(pending) >= resolved_config.batch_size:
            yield from flush()

    yield from flush()


def score_articles(
    articles: Iterable[InferenceArticle],
    *,
    backend: SentimentInferenceBackend,
    config: SentimentInferenceConfig | None = None,
    inference_timestamp: datetime | None = None,
) -> tuple[SentimentOutcome, ...]:
    """Convenience wrapper for callers that already enforce a bounded input."""

    return tuple(
        iter_article_sentiment(
            articles,
            backend=backend,
            config=config,
            inference_timestamp=inference_timestamp,
        )
    )
