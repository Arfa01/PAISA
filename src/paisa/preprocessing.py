from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

REQUIRED_OUTPUT_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume", "source"]

COLUMN_ALIASES: dict[str, set[str]] = {
    "date": {"date", "datetime", "time", "timestamp", "trade_date", "scraped_at", "index"},
    "open": {"open", "open_price", "opening", "op", "o"},
    "high": {"high", "high_price", "h"},
    "low": {"low", "low_price", "l"},
    "close": {"close", "close_price", "closing", "price", "last", "c"},
    "volume": {"volume", "vol", "turnover", "shares", "v"},
}


def _normalize_col_name(value: object) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def _find_column(columns: Iterable[object], target: str) -> object | None:
    aliases = COLUMN_ALIASES[target]
    normalized = {_normalize_col_name(col): col for col in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.replace("--", "", regex=False).str.strip(),
        errors="coerce",
    )


def standardize_price_frame(
    frame: pd.DataFrame,
    ticker: str,
    source: str,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Convert a provider-specific frame into PAISA's standard price schema."""
    if frame.empty:
        raise ValueError(f"No rows received for {ticker} from {source}.")

    working = frame.copy()
    if not isinstance(working.index, pd.RangeIndex):
        index_name = working.index.name or "index"
        if index_name not in working.columns:
            working = working.reset_index(names=index_name)
        else:
            working = working.reset_index()

    date_col = _find_column(working.columns, "date")
    open_col = _find_column(working.columns, "open")
    high_col = _find_column(working.columns, "high")
    low_col = _find_column(working.columns, "low")
    close_col = _find_column(working.columns, "close")
    volume_col = _find_column(working.columns, "volume")

    missing = [name for name, col in {"date": date_col, "open": open_col, "close": close_col, "volume": volume_col}.items() if col is None]
    if missing:
        raise ValueError(f"Missing required columns for {ticker} from {source}: {missing}. Got: {list(working.columns)}")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(working[date_col], errors="coerce", utc=False).dt.date
    out["ticker"] = ticker.upper().strip()
    out["open"] = _to_numeric(working[open_col])
    out["high"] = _to_numeric(working[high_col]) if high_col is not None else pd.NA
    out["low"] = _to_numeric(working[low_col]) if low_col is not None else pd.NA
    out["close"] = _to_numeric(working[close_col])
    out["volume"] = _to_numeric(working[volume_col]).round().astype("Int64")
    out["source"] = source

    out = out.dropna(subset=["date", "open", "close", "volume"])
    out = out[(out["open"] > 0) & (out["close"] > 0) & (out["volume"] >= 0)]
    if start:
        out = out[out["date"] >= start]
    if end:
        out = out[out["date"] <= end]

    out = out.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return out[REQUIRED_OUTPUT_COLUMNS].reset_index(drop=True)


def combine_price_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No price frames to combine.")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    return combined.reset_index(drop=True)
