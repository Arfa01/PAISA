"""Acquisition-only provider interface for the PAISA news pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import hashlib

from paisa.news.models import ProviderFetchResult


def utc_datetime(value: datetime) -> datetime:
    """Return an aware UTC datetime and reject ambiguous naive values."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Provider clocks must return timezone-aware datetimes")
    return value.astimezone(timezone.utc)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class NewsProvider(ABC):
    """Acquisition-only interface implemented by every PAISA news adapter."""

    provider_id: str

    @abstractmethod
    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult:
        """Capture configured sources and return extracted raw records."""


__all__ = ["NewsProvider", "sha256_bytes", "utc_datetime"]
