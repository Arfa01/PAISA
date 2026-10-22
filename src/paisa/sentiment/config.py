"""Immutable model and schema configuration for article sentiment inference."""

from __future__ import annotations

from typing import Final


SENTIMENT_SCHEMA_VERSION: Final = "1.0"

FINBERT_MODEL_REPOSITORY: Final = "ProsusAI/finbert"
FINBERT_MODEL_REVISION: Final = (
    "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
)
FINBERT_TOKENIZER_REVISION: Final = FINBERT_MODEL_REVISION
FINBERT_LICENSE: Final = "CC-BY-NC-SA-4.0"
FINBERT_LICENSE_URL: Final = (
    "https://creativecommons.org/licenses/by-nc-sa/4.0/"
)

# This is the ordering declared by the pinned model configuration.  PAISA's
# stable output order is intentionally different, so callers must map labels
# by name and must never treat an array position as a canonical sentiment.
FINBERT_ORIGINAL_LABELS_BY_INDEX: Final = (
    "positive",
    "negative",
    "neutral",
)
CANONICAL_SENTIMENT_LABELS: Final = (
    "negative",
    "neutral",
    "positive",
)

DEFAULT_INPUT_POLICY_ID: Final = "title-summary-body-fallback-v1"
DEFAULT_MAXIMUM_INPUT_CHARACTERS: Final = 4_000
DEFAULT_MAXIMUM_TOKENIZER_LENGTH: Final = 512
PROBABILITY_SUM_TOLERANCE: Final = 1e-6
