from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from paisa.config import get_paths
from paisa.features import build_features, build_ml_dataset
from paisa.models.baseline import train_baseline_random_forest
from paisa.preprocessing import combine_price_frames, standardize_price_frame
from paisa.providers import get_provider
from paisa.storage import ensure_dirs, save_csv, save_json


def run_v0_pipeline(
    symbols: list[str],
    provider_name: str = "psx-dps",
    start: date | None = None,
    end: date | None = None,
    train: bool = True,
    data_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    manual_csv_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Fetch, clean, feature-engineer, label, and optionally train PAISA v0."""
    if not symbols:
        raise ValueError("At least one symbol is required.")

    paths = get_paths(data_dir=data_dir, model_dir=model_dir)
    ensure_dirs(paths.raw_dir, paths.processed_dir, paths.model_dir)
    provider = get_provider(provider_name, manual_csv_dir=manual_csv_dir)

    raw_row_counts: dict[str, int] = {}
    clean_frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    for raw_symbol in symbols:
        symbol = raw_symbol.upper().strip()
        if not symbol:
            continue
        try:
            raw = provider.fetch_history(symbol, start=start, end=end)
            raw_row_counts[symbol] = int(len(raw))
            save_csv(raw, paths.raw_dir / f"{symbol}_{provider.source_name}_raw.csv")
            cleaned = standardize_price_frame(raw, ticker=symbol, source=provider.source_name, start=start, end=end)
            clean_frames.append(cleaned)
        except Exception as exc:
            errors[symbol] = str(exc)

    if not clean_frames:
        raise RuntimeError(f"No symbols were successfully fetched/cleaned. Errors: {errors}")

    prices = combine_price_frames(clean_frames)
    features = build_features(prices)
    dataset = build_ml_dataset(features)

    save_csv(prices, paths.processed_dir / "psx_prices.csv")
    save_csv(features, paths.processed_dir / "psx_features.csv")
    save_csv(dataset, paths.processed_dir / "ml_dataset.csv")

    metrics: dict[str, Any] | None = None
    if train:
        metrics = train_baseline_random_forest(dataset, paths.model_dir)

    metadata: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider.source_name,
        "symbols_requested": [s.upper().strip() for s in symbols],
        "raw_row_counts": raw_row_counts,
        "errors": errors,
        "prices_rows": int(len(prices)),
        "features_rows": int(len(features)),
        "ml_dataset_rows": int(len(dataset)),
        "date_range": {
            "start": str(pd.to_datetime(prices["date"]).min().date()),
            "end": str(pd.to_datetime(prices["date"]).max().date()),
        },
        "files": {
            "prices": str(paths.processed_dir / "psx_prices.csv"),
            "features": str(paths.processed_dir / "psx_features.csv"),
            "dataset": str(paths.processed_dir / "ml_dataset.csv"),
            "model_dir": str(paths.model_dir),
        },
        "model_metrics": metrics,
    }
    save_json(metadata, paths.processed_dir / "pipeline_metadata.json")
    return metadata
