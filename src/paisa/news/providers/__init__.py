"""News acquisition adapters, intentionally separate from price providers."""

from paisa.news.models import (
    ProviderError,
    ProviderFetchResult,
    RawCapture,
    RawNewsRecord,
)
from paisa.news.providers.base import NewsProvider
from paisa.news.providers.manual_csv import ManualCsvNewsProvider
from paisa.news.providers.rss import RssNewsProvider, RssSourceConfig, load_rss_config


__all__ = [
    "ManualCsvNewsProvider",
    "NewsProvider",
    "ProviderError",
    "ProviderFetchResult",
    "RawCapture",
    "RawNewsRecord",
    "RssNewsProvider",
    "RssSourceConfig",
    "load_rss_config",
]
