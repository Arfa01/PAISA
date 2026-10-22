"""Deterministic fixture provider for automated tests only.

Production CLI code must never select this adapter or use it as a fallback.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import json
from typing import Any

from paisa.news.models import ProviderFetchResult, RawCapture, RawNewsRecord
from paisa.news.providers.base import NewsProvider, sha256_bytes, utc_datetime


class FixtureNewsProvider(NewsProvider):
    """Return caller-supplied deterministic records explicitly marked fixture."""

    provider_id = "fixture"
    data_class = "fixture"

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        fetched_at: datetime = datetime(2000, 1, 1, tzinfo=timezone.utc),
    ) -> None:
        self.rows = tuple(dict(row) for row in rows)
        self.default_fetched_at = utc_datetime(fetched_at)
        self.configured_sources = (
            {"source": "fixture://deterministic-news", "test_only": True},
        )

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult:
        acquisition_time = utc_datetime(fetched_at or self.default_fetched_at)
        payload = json.dumps(
            self.rows,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        checksum = sha256_bytes(payload)
        capture_id = "fixture-deterministic"
        capture = RawCapture(
            capture_id=capture_id,
            provider_id=self.provider_id,
            configured_source="fixture://deterministic-news",
            source_name="Deterministic test fixture",
            fetched_at=acquisition_time,
            data_class="fixture",
            content=payload,
            filename="fixture.json",
            content_type="application/json",
            response_sha256=checksum,
        )
        records = tuple(
            RawNewsRecord(
                raw_record_id=f"{capture_id}-{index:06d}",
                provider_id=self.provider_id,
                configured_source="fixture://deterministic-news",
                source_name=self._text(row.get("source_name")) or "Fixture News",
                source_type="fixture",
                fetched_at=acquisition_time,
                data_class="fixture",
                source_url=self._text(row.get("source_url")),
                title=self._text(row.get("title")),
                summary=self._text(row.get("summary")),
                body=self._text(row.get("body")),
                published_at_raw=self._text(row.get("published_at")),
                published_timezone=self._text(row.get("published_timezone")),
                available_at_raw=self._text(row.get("available_at")),
                language=self._text(row.get("language")),
                provider_record_id=self._text(row.get("provider_record_id")),
                capture_id=capture_id,
                response_content_type="application/json",
                response_sha256=checksum,
                raw_payload=dict(row),
            )
            for index, row in enumerate(self.rows)
        )
        return ProviderFetchResult(
            provider_id=self.provider_id,
            data_class="fixture",
            fetched_at=acquisition_time,
            configured_sources=self.configured_sources,
            records=records,
            captures=(capture,),
        )


__all__ = ["FixtureNewsProvider"]
