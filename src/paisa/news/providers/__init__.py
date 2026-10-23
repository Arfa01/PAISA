"""News acquisition adapters, intentionally separate from price providers."""

from paisa.news.models import (
    ProviderError,
    ProviderFetchResult,
    RawCapture,
    RawNewsRecord,
)
from paisa.news.providers.base import NewsProvider
from paisa.news.providers.gdelt_doc import (
    GDELT_DOC_ENDPOINT,
    GdeltDocNewsProvider,
    GdeltQueryWindow,
    load_gdelt_config,
)
from paisa.news.providers.manual_csv import (
    LICENSING_REVIEW_REQUIRED,
    ManualCsvNewsProvider,
)
from paisa.news.providers.rss import RssNewsProvider, RssSourceConfig, load_rss_config


__all__ = [
    "GDELT_DOC_ENDPOINT",
    "GdeltDocNewsProvider",
    "GdeltQueryWindow",
    "LICENSING_REVIEW_REQUIRED",
    "ManualCsvNewsProvider",
    "NewsProvider",
    "ProviderError",
    "ProviderFetchResult",
    "RawCapture",
    "RawNewsRecord",
    "RssNewsProvider",
    "RssSourceConfig",
    "load_gdelt_config",
    "load_rss_config",
]
