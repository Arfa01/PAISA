from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from paisa.exceptions import DataSourceError
from paisa.providers.base import StockDataProvider


class ManualCsvProvider(StockDataProvider):
    """Read manually downloaded PSX CSVs from a folder.

    Expected file names:
        data/external/manual_psx/OGDC.csv
        data/external/manual_psx/HBL.csv

    The preprocessor accepts common column spellings, so exact headers can be
    date/open/high/low/close/volume or common variants.
    """

    source_name = "manual_csv"

    def __init__(self, csv_dir: str | Path) -> None:
        self.csv_dir = Path(csv_dir)

    def fetch_history(self, symbol: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
        path = self.csv_dir / f"{symbol.upper().strip()}.csv"
        if not path.exists():
            raise DataSourceError(f"Manual CSV not found: {path}")
        frame = pd.read_csv(path)
        # Filtering is handled by the common preprocessor.
        return frame
