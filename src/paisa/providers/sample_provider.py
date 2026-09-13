from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from paisa.providers.base import StockDataProvider


class SampleProvider(StockDataProvider):
    """Deterministic fake data source used only for tests and offline API demos."""

    source_name = "sample"

    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        start = start or date(2021, 1, 1)
        end = end or date.today()
        dates = pd.bdate_range(start=start, end=end)
        if len(dates) < 60:
            raise ValueError("SampleProvider needs at least ~60 business days for rolling features.")

        seed = abs(hash(symbol)) % (2**32)
        rng = np.random.default_rng(seed)
        base_price = 80 + (seed % 70)
        drift = 0.00015 + ((seed % 7) * 0.00005)
        noise = rng.normal(loc=drift, scale=0.018, size=len(dates))
        close = base_price * np.exp(np.cumsum(noise))
        open_ = close * (1 + rng.normal(0, 0.006, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.015, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.015, len(dates)))
        volume = rng.integers(80_000, 3_000_000, len(dates))

        return pd.DataFrame(
            {
                "date": dates.date,
                "open": open_.round(2),
                "high": high.round(2),
                "low": low.round(2),
                "close": close.round(2),
                "volume": volume,
            }
        )
