from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class StockDataProvider(ABC):
    """Contract for every stock data source used by PAISA."""

    source_name: str

    @abstractmethod
    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        """Return a DataFrame for one symbol with at least date/open/close/volume where available."""
