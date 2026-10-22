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

# SHA-256 values for every file used by the reviewed model/tokenizer snapshot.
# These values are part of the model identity, not merely download checks.  A
# cache whose bytes differ from this inventory must never be loaded under the
# pinned revision name.
FINBERT_REVIEWED_SNAPSHOT_SHA256: Final = {
    "config.json": "f6449ddda85eb726207a40be59c0cd3bd4b142ccb27298d5e45f9ae3396b1abe",
    "model.safetensors": "e5897858ff819aad7629b96ce521ae5477952d03634ef5dd30ed2d76357a9f00",
    "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
    "tokenizer_config.json": "ffc7913f6084b138dad68de5a0c2f0ee25ac983709260901e01f53be5796b9d9",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}

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
