from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "open",
    "close",
    "volume",
    "daily_return",
    "log_return",
    "close_lag_1",
    "close_lag_5",
    "close_lag_10",
    "volume_lag_1",
    "ma_5",
    "ma_10",
    "ma_20",
    "return_volatility_5",
    "return_volatility_10",
    "volume_change",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
]

TARGET_COLUMNS = ["target_next_day_return", "target_next_day_up"]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """Create technical features and the next-trading-day classification label."""
    required = {"date", "ticker", "open", "close", "volume", "source"}
    missing = required - set(price_data.columns)
    if missing:
        raise ValueError(f"price_data missing required columns: {sorted(missing)}")

    frames: list[pd.DataFrame] = []
    for ticker, group in price_data.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        g = group.copy().reset_index(drop=True)
        close = pd.to_numeric(g["close"], errors="coerce")
        volume = pd.to_numeric(g["volume"], errors="coerce")

        g["daily_return"] = close.pct_change()
        g["log_return"] = np.log(close).diff()
        g["close_lag_1"] = close.shift(1)
        g["close_lag_5"] = close.shift(5)
        g["close_lag_10"] = close.shift(10)
        g["volume_lag_1"] = volume.shift(1)
        g["ma_5"] = close.rolling(5, min_periods=5).mean()
        g["ma_10"] = close.rolling(10, min_periods=10).mean()
        g["ma_20"] = close.rolling(20, min_periods=20).mean()
        g["return_volatility_5"] = g["daily_return"].rolling(5, min_periods=5).std()
        g["return_volatility_10"] = g["daily_return"].rolling(10, min_periods=10).std()
        g["volume_change"] = volume.pct_change()
        g["rsi_14"] = _rsi(close, 14)

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        g["macd_line"] = ema_12 - ema_26
        g["macd_signal"] = g["macd_line"].ewm(span=9, adjust=False).mean()
        g["macd_hist"] = g["macd_line"] - g["macd_signal"]

        next_close = close.shift(-1)
        g["target_next_day_return"] = (next_close - close) / close
        g["target_next_day_up"] = pd.Series(np.where(next_close.notna(), (next_close > close).astype(int), np.nan), index=g.index)
        frames.append(g)

    return pd.concat(frames, ignore_index=True)


def build_ml_dataset(features: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are safe for supervised training: complete features + known label."""
    cols = ["date", "ticker", "source", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    missing_cols = [col for col in cols if col not in features.columns]
    if missing_cols:
        raise ValueError(f"features missing expected columns: {missing_cols}")
    dataset = features[cols].copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.dropna(subset=[*FEATURE_COLUMNS, "target_next_day_up"])
    dataset["target_next_day_up"] = dataset["target_next_day_up"].astype(int)
    return dataset.sort_values(["date", "ticker"]).reset_index(drop=True)
