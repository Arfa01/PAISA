"""Stable records shared by sentiment inference, persistence, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping

from paisa.sentiment.config import (
    CANONICAL_SENTIMENT_LABELS,
    PROBABILITY_SUM_TOLERANCE,
    SENTIMENT_SCHEMA_VERSION,
)


def utc_iso(value: datetime | None) -> str | None:
    """Return an aware timestamp in the repository's UTC RFC 3339 form."""

    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Sentiment timestamps must be timezone-aware.")
    return value.astimezone(timezone.utc).isoformat(timespec="auto").replace(
        "+00:00", "Z"
    )


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability in [0, 1].")


@dataclass(frozen=True)
class TextSelection:
    """Selected model input plus non-sensitive reproducibility metadata."""

    policy_id: str
    selected_fields: tuple[str, ...]
    input_text: str = field(repr=False)
    input_text_sha256: str
    character_count: int
    original_character_count: int
    character_truncation_applied: bool
    maximum_input_characters: int

    def __post_init__(self) -> None:
        if not self.input_text:
            raise ValueError("Selected sentiment input cannot be empty.")
        if self.character_count != len(self.input_text):
            raise ValueError("character_count does not match selected input text.")
        if self.original_character_count < self.character_count:
            raise ValueError("Original character count cannot be smaller than input.")
        if self.maximum_input_characters <= 0:
            raise ValueError("maximum_input_characters must be positive.")
        if self.character_count > self.maximum_input_characters:
            raise ValueError("Selected input exceeds maximum_input_characters.")

    def metadata_dict(self) -> dict[str, Any]:
        """Serialize metadata without duplicating article text."""

        return {
            "input_policy_id": self.policy_id,
            "selected_fields": list(self.selected_fields),
            "input_text_sha256": self.input_text_sha256,
            "character_count": self.character_count,
            "original_character_count": self.original_character_count,
            "character_truncation_applied": self.character_truncation_applied,
            "maximum_input_characters": self.maximum_input_characters,
        }

@dataclass(frozen=True)
class BackendPrediction:
    """One backend prediction already mapped to canonical label names."""

    negative_probability: float
    neutral_probability: float
    positive_probability: float
    predicted_label: str
    predicted_probability: float
    tokenizer_token_count: int
    final_token_count: int
    truncation_side: str
    token_truncation_applied: bool

    def __post_init__(self) -> None:
        probabilities = self.probabilities
        for label, probability in probabilities.items():
            _validate_probability(probability, name=f"{label}_probability")
        if not math.isclose(
            sum(probabilities.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_TOLERANCE,
        ):
            raise ValueError("Sentiment probabilities must sum to one.")
        if self.predicted_label not in CANONICAL_SENTIMENT_LABELS:
            raise ValueError(f"Unsupported predicted label: {self.predicted_label}")
        if not math.isclose(
            self.predicted_probability,
            probabilities[self.predicted_label],
            abs_tol=1e-7,
        ):
            raise ValueError("Predicted probability does not match predicted label.")
        if self.tokenizer_token_count <= 0:
            raise ValueError("tokenizer_token_count must be positive.")
        if not 0 < self.final_token_count <= self.tokenizer_token_count:
            raise ValueError(
                "final_token_count must be positive and no greater than the "
                "pre-truncation tokenizer_token_count."
            )
        if self.truncation_side != "right":
            raise ValueError("Only explicitly audited right truncation is supported.")
        if self.token_truncation_applied != (
            self.final_token_count < self.tokenizer_token_count
        ):
            raise ValueError(
                "token_truncation_applied must match the before/after token counts."
            )

    @property
    def probabilities(self) -> dict[str, float]:
        return {
            "negative": self.negative_probability,
            "neutral": self.neutral_probability,
            "positive": self.positive_probability,
        }


@dataclass(frozen=True)
class ArticleSentiment:
    """Stable successful article-level sentiment inference schema."""

    article_id: str
    source_ingestion_run_id: str
    input_policy_id: str
    selected_fields: tuple[str, ...]
    input_text_sha256: str
    character_count: int
    original_character_count: int
    tokenizer_token_count: int
    final_token_count: int
    truncation_side: str
    truncation_applied: bool
    character_truncation_applied: bool
    token_truncation_applied: bool
    maximum_input_characters: int
    maximum_sequence_length: int
    original_language: str | None
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    class_labels: tuple[str, ...]
    negative_probability: float
    neutral_probability: float
    positive_probability: float
    predicted_label: str
    predicted_probability: float
    published_at: datetime | None
    available_at: datetime
    fetched_at: datetime
    inference_timestamp: datetime
    device: str
    batch_size: int
    library_versions: Mapping[str, str]
    warning_codes: tuple[str, ...] = ()
    error_code: None = None
    error_message: None = None
    schema_version: str = SENTIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SENTIMENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported sentiment schema version: {self.schema_version}"
            )
        if self.class_labels != CANONICAL_SENTIMENT_LABELS:
            raise ValueError("class_labels must use PAISA's canonical order.")
        for value in (self.available_at, self.fetched_at, self.inference_timestamp):
            utc_iso(value)
        utc_iso(self.published_at)
        if self.maximum_input_characters <= 0:
            raise ValueError("maximum_input_characters must be positive.")
        if self.maximum_sequence_length <= 0:
            raise ValueError("maximum_sequence_length must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        prediction = BackendPrediction(
            negative_probability=self.negative_probability,
            neutral_probability=self.neutral_probability,
            positive_probability=self.positive_probability,
            predicted_label=self.predicted_label,
            predicted_probability=self.predicted_probability,
            tokenizer_token_count=self.tokenizer_token_count,
            final_token_count=self.final_token_count,
            truncation_side=self.truncation_side,
            token_truncation_applied=self.token_truncation_applied,
        )
        del prediction
        expected_final_token_count = min(
            self.tokenizer_token_count, self.maximum_sequence_length
        )
        if self.final_token_count != expected_final_token_count:
            raise ValueError(
                "final_token_count must match the post-limit token count."
            )
        if self.truncation_applied != (
            self.character_truncation_applied or self.token_truncation_applied
        ):
            raise ValueError(
                "truncation_applied must combine character and token truncation."
            )

    @property
    def probabilities(self) -> dict[str, float]:
        return {
            "negative": self.negative_probability,
            "neutral": self.neutral_probability,
            "positive": self.positive_probability,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "article_id": self.article_id,
            "source_ingestion_run_id": self.source_ingestion_run_id,
            "input_policy_id": self.input_policy_id,
            "selected_fields": list(self.selected_fields),
            "input_text_sha256": self.input_text_sha256,
            "character_count": self.character_count,
            "original_character_count": self.original_character_count,
            "tokenizer_token_count": self.tokenizer_token_count,
            "final_token_count": self.final_token_count,
            "truncation_side": self.truncation_side,
            "truncation_applied": self.truncation_applied,
            "character_truncation_applied": self.character_truncation_applied,
            "token_truncation_applied": self.token_truncation_applied,
            "maximum_input_characters": self.maximum_input_characters,
            "maximum_sequence_length": self.maximum_sequence_length,
            "original_language": self.original_language,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "class_labels": list(self.class_labels),
            "negative_probability": self.negative_probability,
            "neutral_probability": self.neutral_probability,
            "positive_probability": self.positive_probability,
            "predicted_label": self.predicted_label,
            "predicted_probability": self.predicted_probability,
            "published_at": utc_iso(self.published_at),
            "available_at": utc_iso(self.available_at),
            "fetched_at": utc_iso(self.fetched_at),
            "inference_timestamp": utc_iso(self.inference_timestamp),
            "device": self.device,
            "batch_size": self.batch_size,
            "library_versions": dict(sorted(self.library_versions.items())),
            "warning_codes": list(self.warning_codes),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

@dataclass(frozen=True)
class ArticleSentimentError:
    """Stable structured error that never contains full source article text."""

    article_id: str
    source_ingestion_run_id: str
    input_policy_id: str
    original_language: str | None
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    published_at: datetime | None
    available_at: datetime
    fetched_at: datetime
    inference_timestamp: datetime
    error_code: str
    error_message: str
    selected_fields: tuple[str, ...] = ()
    input_text_sha256: str | None = None
    warning_codes: tuple[str, ...] = ()
    schema_version: str = SENTIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SENTIMENT_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported sentiment schema version: {self.schema_version}"
            )
        for value in (self.available_at, self.fetched_at, self.inference_timestamp):
            utc_iso(value)
        utc_iso(self.published_at)
        if not self.error_code or not self.error_message:
            raise ValueError("Sentiment errors require a code and safe message.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "article_id": self.article_id,
            "source_ingestion_run_id": self.source_ingestion_run_id,
            "input_policy_id": self.input_policy_id,
            "selected_fields": list(self.selected_fields),
            "input_text_sha256": self.input_text_sha256,
            "original_language": self.original_language,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "published_at": utc_iso(self.published_at),
            "available_at": utc_iso(self.available_at),
            "fetched_at": utc_iso(self.fetched_at),
            "inference_timestamp": utc_iso(self.inference_timestamp),
            "warning_codes": list(self.warning_codes),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }
