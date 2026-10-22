from __future__ import annotations

from dataclasses import replace

from paisa.sentiment.inference import (
    SentimentInferenceConfig,
    iter_article_sentiment,
    score_articles,
)
from paisa.sentiment.models import ArticleSentiment, ArticleSentimentError


def test_cpu_inference_with_mock_backend_emits_stable_schema(
    article, mock_backend
):
    result = score_articles(
        [article],
        backend=mock_backend,
        inference_timestamp=article.available_at,
    )[0]

    assert isinstance(result, ArticleSentiment)
    assert result.device == "cpu"
    assert sum(result.probabilities.values()) == 1.0
    assert result.predicted_label == "positive"
    assert result.available_at == article.available_at
    assert set(result.to_dict()) == {
        "schema_version",
        "article_id",
        "source_ingestion_run_id",
        "input_policy_id",
        "selected_fields",
        "input_text_sha256",
        "character_count",
        "original_character_count",
        "tokenizer_token_count",
        "truncation_applied",
        "character_truncation_applied",
        "token_truncation_applied",
        "maximum_input_characters",
        "maximum_sequence_length",
        "original_language",
        "model_repository",
        "model_revision",
        "tokenizer_revision",
        "class_labels",
        "negative_probability",
        "neutral_probability",
        "positive_probability",
        "predicted_label",
        "predicted_probability",
        "published_at",
        "available_at",
        "fetched_at",
        "inference_timestamp",
        "device",
        "batch_size",
        "library_versions",
        "warning_codes",
        "error_code",
        "error_message",
    }


def test_inference_respects_batch_boundaries(article, mock_backend):
    articles = [replace(article, article_id=f"article-{index}") for index in range(5)]

    results = tuple(
        iter_article_sentiment(
            articles,
            backend=mock_backend,
            config=SentimentInferenceConfig(batch_size=2),
        )
    )

    assert len(results) == 5
    assert [len(batch) for batch in mock_backend.batches] == [2, 2, 1]


def test_maximum_articles_bounds_iteration(article, mock_backend):
    articles = [replace(article, article_id=f"article-{index}") for index in range(5)]

    results = score_articles(
        articles,
        backend=mock_backend,
        config=SentimentInferenceConfig(maximum_articles=3, batch_size=2),
    )

    assert [result.article_id for result in results] == [
        "article-0",
        "article-1",
        "article-2",
    ]


def test_token_and_character_truncation_metadata_are_combined(
    article, mock_backend
):
    result = score_articles(
        [replace(article, title="many words in this title")],
        backend=mock_backend,
        config=SentimentInferenceConfig(
            maximum_input_characters=25,
            maximum_sequence_length=3,
        ),
    )[0]

    assert isinstance(result, ArticleSentiment)
    assert result.truncation_applied is True
    assert result.character_truncation_applied is True
    assert result.token_truncation_applied is True
    assert "input_character_truncated" in result.warning_codes
    assert "input_token_truncated" in result.warning_codes


def test_effective_policy_identity_includes_both_input_limits(article, mock_backend):
    first_config = SentimentInferenceConfig(
        maximum_input_characters=100,
        maximum_sequence_length=64,
    )
    second_config = SentimentInferenceConfig(
        maximum_input_characters=101,
        maximum_sequence_length=64,
    )
    third_config = SentimentInferenceConfig(
        maximum_input_characters=100,
        maximum_sequence_length=65,
    )

    assert first_config.effective_input_policy_id.endswith("-chars100-tokens64")
    assert len(
        {
            first_config.effective_input_policy_id,
            second_config.effective_input_policy_id,
            third_config.effective_input_policy_id,
        }
    ) == 3
    result = score_articles(
        [article], backend=mock_backend, config=first_config
    )[0]
    assert result.input_policy_id == first_config.effective_input_policy_id


def test_unsupported_language_is_a_structured_error_and_is_not_scored(
    article, mock_backend
):
    result = score_articles(
        [replace(article, language="ur")],
        backend=mock_backend,
    )[0]

    assert isinstance(result, ArticleSentimentError)
    assert result.error_code == "unsupported_language"
    assert mock_backend.batches == []


def test_missing_language_is_not_silently_assumed_to_be_english(
    article, mock_backend
):
    result = score_articles(
        [replace(article, language=None)],
        backend=mock_backend,
    )[0]

    assert isinstance(result, ArticleSentimentError)
    assert result.error_code == "unsupported_language"


def test_missing_text_is_a_structured_error_without_article_body(
    article, mock_backend, caplog
):
    result = score_articles(
        [replace(article, title=None, summary="SECRET", body="PRIVATE BODY")],
        backend=mock_backend,
    )[0]

    assert isinstance(result, ArticleSentimentError)
    assert result.error_code == "missing_input_text"
    assert "SECRET" not in str(result.to_dict())
    assert "PRIVATE BODY" not in str(result.to_dict())
    assert not caplog.records
