from __future__ import annotations

from datetime import date

import pandas as pd

from paisa.exceptions import DataSourceError
from paisa.providers.base import StockDataProvider


class PyPsxToolkitProvider(StockDataProvider):
    """Fetch full PSX OHLCV data through pypsx-toolkit, if installed.

    This provider is useful when you want high/low columns in addition to the
    PSX DPS open/close/volume fields. The project still keeps psx-dps as the
    no-key PSX-first provider, but DPS is undocumented and may reject requests.
    """

    source_name = "pypsx_toolkit"

    def __init__(self, period: str = "10y") -> None:
        self.period = period

    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        try:
            import pypsx_toolkit as pt  # type: ignore
        except Exception as exc:
            raise DataSourceError(
                "pypsx-toolkit is not installed or failed to import. "
                f"Actual import error: {type(exc).__name__}: {exc}. "
                "Run `python -m pip install -r requirements.txt` from the active virtualenv, "
                "then verify with `python -c \"import pypsx_toolkit as pt; print(pt.__file__)\"`."
            ) from exc

        symbol = symbol.upper().strip()
        try:
            # pypsx-toolkit supports the yfinance-like download(symbol, period, interval) API.
            frame = pt.download(symbol, period=self.period, interval="1d")
        except TypeError:
            # Older compatible builds may not accept interval.
            try:
                frame = pt.download(symbol, period=self.period)
            except Exception as exc:
                raise DataSourceError(f"pypsx-toolkit could not download {symbol}: {type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            raise DataSourceError(f"pypsx-toolkit could not download {symbol}: {type(exc).__name__}: {exc}") from exc

        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise DataSourceError(f"pypsx-toolkit returned no rows for {symbol}.")

        frame = frame.copy()
        if "date" not in {str(c).lower() for c in frame.columns}:
            frame = frame.reset_index()

        if start or end:
            # Let the common preprocessor do robust date parsing too; this is just a rough early filter.
            possible_date_cols = [c for c in frame.columns if str(c).lower() in {"date", "datetime", "time", "index"}]
            if possible_date_cols:
                date_col = possible_date_cols[0]
                parsed = pd.to_datetime(frame[date_col], errors="coerce").dt.date
                if start:
                    frame = frame[parsed >= start]
                if end:
                    frame = frame[parsed <= end]

        return frame.reset_index(drop=True)
