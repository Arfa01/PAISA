"""Reproducible article-level financial sentiment for PAISA.

The package intentionally stops at article inference. It performs no daily
aggregation, stock prediction, trading decision, or portfolio operation.
"""

from paisa.sentiment.backend import (
    DeviceUnavailableError,
    ModelLabelMappingError,
    SentimentBackendError,
    SentimentInferenceBackend,
    TransformersFinBERTBackend,
    map_probabilities_by_label,
    validate_finbert_label_mapping,
)
from paisa.sentiment.config import (
    CANONICAL_SENTIMENT_LABELS,
    DEFAULT_INPUT_POLICY_ID,
    FINBERT_LICENSE,
    FINBERT_LICENSE_URL,
    FINBERT_MODEL_REPOSITORY,
    FINBERT_MODEL_REVISION,
    FINBERT_ORIGINAL_LABELS_BY_INDEX,
    FINBERT_TOKENIZER_REVISION,
    PROBABILITY_SUM_TOLERANCE,
    SENTIMENT_SCHEMA_VERSION,
)
from paisa.sentiment.inference import (
    SentimentInferenceConfig,
    SentimentInferenceTimeoutError,
    SentimentOutcome,
    iter_article_sentiment,
    score_articles,
)
from paisa.sentiment.models import (
    ArticleSentiment,
    ArticleSentimentError,
    BackendPrediction,
    TextSelection,
)
from paisa.sentiment.pipeline import SentimentRunSummary, run_sentiment_inference
from paisa.sentiment.text import MissingArticleTextError, select_article_text
from paisa.sentiment.time_alignment import (
    SentimentTimingEvidenceError,
    filter_sentiment_by_decision_cutoff,
)

__all__ = [
    "ArticleSentiment",
    "ArticleSentimentError",
    "BackendPrediction",
    "CANONICAL_SENTIMENT_LABELS",
    "DEFAULT_INPUT_POLICY_ID",
    "DeviceUnavailableError",
    "FINBERT_LICENSE",
    "FINBERT_LICENSE_URL",
    "FINBERT_MODEL_REPOSITORY",
    "FINBERT_MODEL_REVISION",
    "FINBERT_ORIGINAL_LABELS_BY_INDEX",
    "FINBERT_TOKENIZER_REVISION",
    "MissingArticleTextError",
    "ModelLabelMappingError",
    "PROBABILITY_SUM_TOLERANCE",
    "SENTIMENT_SCHEMA_VERSION",
    "SentimentBackendError",
    "SentimentInferenceBackend",
    "SentimentInferenceConfig",
    "SentimentInferenceTimeoutError",
    "SentimentRunSummary",
    "SentimentOutcome",
    "SentimentTimingEvidenceError",
    "TextSelection",
    "TransformersFinBERTBackend",
    "filter_sentiment_by_decision_cutoff",
    "iter_article_sentiment",
    "map_probabilities_by_label",
    "score_articles",
    "select_article_text",
    "run_sentiment_inference",
    "validate_finbert_label_mapping",
]
