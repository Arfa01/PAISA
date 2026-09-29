from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
import requests

from paisa.exceptions import DataSourceError
from paisa.providers.base import StockDataProvider


class PsxDpsProvider(StockDataProvider):
    """Fetch historical EOD data from PSX Data Portal's public time-series endpoint.

    Endpoint used by several public PSX tools:
        https://dps.psx.com.pk/timeseries/eod/{SYMBOL}

    Observed row shape:
        [unix_timestamp, close_price, volume, open_price]

    Important v0 limitation:
        This endpoint does not consistently provide high/low. PAISA leaves those
        as missing rather than inventing them. The ML v0 can still train on open,
        close, volume, returns, moving averages, volatility, and lag features.
    """

    source_name = "psx_dps"
    endpoint = "https://dps.psx.com.pk/timeseries/eod/{symbol}"

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _headers() -> dict[str, str]:
        """Return browser-like headers for the undocumented DPS JSON endpoint.

        DPS is not a contracted public API. In practice it can return 403/404 to
        plain script-looking requests even when the same endpoint works from a
        browser-like client. These headers keep the request server-side and avoid
        browser CORS issues while looking like a normal page-initiated JSON fetch.
        """
        return {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/150.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://dps.psx.com.pk/",
            "Origin": "https://dps.psx.com.pk",
            "X-Requested-With": "XMLHttpRequest",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        symbol = symbol.upper().strip()
        base_url = self.endpoint.format(symbol=symbol)
        candidate_urls = [base_url, f"{base_url}/"]
        last_error: str | None = None
        payload: dict[str, Any] | None = None

        with requests.Session() as session:
            session.headers.update(self._headers())
            for url in candidate_urls:
                try:
                    response = session.get(url, timeout=self.timeout_seconds)
                    response.raise_for_status()
                    payload = response.json()
                    break
                except Exception as exc:  # requests and JSON parsing errors
                    last_error = f"{type(exc).__name__}: {exc}"

        if payload is None:
            raise DataSourceError(
                f"Could not fetch PSX DPS data for {symbol}. Last error: {last_error}. "
                "The DPS endpoint is undocumented and may temporarily reject requests. "
                "If this persists, run the same training command with `--provider pypsx` "
                "to use the alternate real PSX provider instead of sample/synthetic data."
            )

        if payload.get("status") != 1 or not payload.get("data"):
            raise DataSourceError(f"PSX DPS returned no historical data for {symbol}. Payload keys: {list(payload.keys())}")

        rows: list[dict[str, Any]] = []
        for entry in payload["data"]:
            if not isinstance(entry, list) or len(entry) < 4:
                continue
            ts, close_price, volume, open_price = entry[:4]
            try:
                trade_date = datetime.fromtimestamp(int(ts), tz=timezone.utc).date()
                rows.append(
                    {
                        "date": trade_date,
                        "open": open_price,
                        "high": None,
                        "low": None,
                        "close": close_price,
                        "volume": volume,
                    }
                )
            except Exception:
                continue

        frame = pd.DataFrame(rows)
        if frame.empty:
            raise DataSourceError(f"PSX DPS payload for {symbol} did not contain parseable rows.")

        frame = frame.sort_values("date")
        if start:
            frame = frame[frame["date"] >= start]
        if end:
            frame = frame[frame["date"] <= end]
        return frame.reset_index(drop=True)
