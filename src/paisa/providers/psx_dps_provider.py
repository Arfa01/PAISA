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

    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        symbol = symbol.upper().strip()
        url = self.endpoint.format(symbol=symbol)
        try:
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={
                    "User-Agent": "PAISA-FYP/0.1 academic project; contact via repository owner",
                    "Accept": "application/json,text/plain,*/*",
                },
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except Exception as exc:  # requests and JSON parsing errors
            raise DataSourceError(f"Could not fetch PSX DPS data for {symbol}: {exc}") from exc

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
