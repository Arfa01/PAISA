"""Reproducible financial-news ingestion for PAISA.

This package stops at acquisition, normalization, deterministic deduplication,
and first-stage entity linking.  Sentiment and model features deliberately live
outside this boundary.
"""

from paisa.news.models import (
    NEWS_AVAILABILITY_POLICY_VERSION,
    NEWS_SCHEMA_VERSION,
    ArticleEntityLink,
    DuplicateRelationship,
    NormalizedNewsArticle,
    ProviderError,
    ProviderFetchResult,
    RawCapture,
    RawNewsRecord,
)

__all__ = [
    "NEWS_AVAILABILITY_POLICY_VERSION",
    "NEWS_SCHEMA_VERSION",
    "ArticleEntityLink",
    "DuplicateRelationship",
    "NormalizedNewsArticle",
    "ProviderError",
    "ProviderFetchResult",
    "RawCapture",
    "RawNewsRecord",
]
