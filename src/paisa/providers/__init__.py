from __future__ import annotations

from pathlib import Path

from paisa.providers.base import StockDataProvider
from paisa.providers.manual_csv_provider import ManualCsvProvider
from paisa.providers.psx_dps_provider import PsxDpsProvider
from paisa.providers.pypsx_provider import PyPsxToolkitProvider
from paisa.providers.sample_provider import SampleProvider


def get_provider(name: str, manual_csv_dir: str | Path | None = None) -> StockDataProvider:
    normalized = name.lower().replace("_", "-")
    if normalized == "sample":
        return SampleProvider()
    if normalized in {"psx-dps", "dps", "psx"}:
        return PsxDpsProvider()
    if normalized in {"pypsx", "pypsx-toolkit"}:
        return PyPsxToolkitProvider()
    if normalized in {"manual-csv", "csv"}:
        if not manual_csv_dir:
            raise ValueError("manual-csv provider requires --manual-csv-dir")
        return ManualCsvProvider(manual_csv_dir)
    raise ValueError(f"Unknown provider: {name}. Use sample, psx-dps, pypsx, or manual-csv.")


__all__ = ["get_provider", "StockDataProvider"]
