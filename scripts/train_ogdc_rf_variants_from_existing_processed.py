from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Allow `python scripts/train_ogdc_rf_variants_from_existing_processed.py` before editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from paisa.experiments.experiment_config import DEFAULT_START, DEFAULT_SYMBOL, RANDOM_SEED
import paisa.experiments.trainer as trainer


def _read_existing_real_processed_data(repo: Path, symbol: str, start: str, provider: str) -> dict[str, Any]:
    """Use already-generated real PSX processed files instead of hitting a live provider.

    This is intentionally not a synthetic/sample fallback. It exists for cases where
    the live DPS endpoint later blocks scripts, but the repository already contains
    `data/processed/psx_prices.csv` and `data/processed/psx_features.csv` generated
    from a previous successful real PSX acquisition.
    """
    symbol = symbol.upper().strip()
    prices_path = repo / "data" / "processed" / "psx_prices.csv"
    features_path = repo / "data" / "processed" / "psx_features.csv"

    missing = [str(path) for path in [prices_path, features_path] if not path.exists()]
    if missing:
        raise RuntimeError(
            "Cannot use existing processed data because required files are missing: "
            f"{missing}. Run a real provider first, or provide manual real PSX CSVs."
        )

    prices = pd.read_csv(prices_path)
    features = pd.read_csv(features_path)
    if "ticker" not in prices.columns or "ticker" not in features.columns:
        raise RuntimeError("Existing processed files must contain a ticker column.")

    price_rows = prices[prices["ticker"].astype(str).str.upper() == symbol].copy()
    feature_rows = features[features["ticker"].astype(str).str.upper() == symbol].copy()
    if price_rows.empty or feature_rows.empty:
        raise RuntimeError(
            f"Existing processed files do not contain usable {symbol} rows. "
            f"prices_rows={len(price_rows)}, features_rows={len(feature_rows)}"
        )

    sources = sorted({str(value) for value in price_rows.get("source", pd.Series(dtype=str)).dropna().unique()})
    if sources and all("sample" in source.lower() for source in sources):
        raise RuntimeError(
            "Refusing to train from sample/synthetic processed data. "
            f"Detected sources: {sources}. Run a real PSX provider or use manual real PSX data."
        )

    price_rows["date"] = pd.to_datetime(price_rows["date"], errors="raise")
    feature_rows["date"] = pd.to_datetime(feature_rows["date"], errors="raise")
    if start:
        requested_start = pd.to_datetime(start).date()
        price_rows = price_rows[price_rows["date"].dt.date >= requested_start]
        feature_rows = feature_rows[feature_rows["date"].dt.date >= requested_start]

    if price_rows.empty or feature_rows.empty:
        raise RuntimeError(f"No {symbol} rows remain after applying start={start}.")

    print(
        "Using existing processed real PSX data instead of live provider fetch: "
        f"{prices_path} and {features_path}"
    )
    print(f"Detected source values for {symbol}: {sources or ['unknown']}")

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "symbols_requested": [symbol],
        "raw_row_counts": {symbol: int(len(price_rows))},
        "errors": {},
        "prices_rows": int(len(price_rows)),
        "features_rows": int(len(feature_rows)),
        "ml_dataset_rows": None,
        "date_range": {
            "start": str(price_rows["date"].min().date()),
            "end": str(price_rows["date"].max().date()),
        },
        "files": {
            "prices": str(prices_path),
            "features": str(features_path),
            "dataset": str(repo / "data" / "processed" / "ml_dataset.csv"),
            "model_dir": str(repo / "models"),
        },
        "model_metrics": None,
        "note": (
            "Training used existing processed PSX files already present locally. "
            "This is intended for live-provider outage/blocking situations and is not sample data."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train frozen Random Forest variants using existing real PSX processed files. "
            "Use this only when data/processed files were generated from a previous real PSX run."
        )
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Ticker symbol. Default: OGDC")
    parser.add_argument("--start", default=DEFAULT_START, help="Requested start date YYYY-MM-DD.")
    parser.add_argument("--output-dir", default="artifacts/model_training", help="Output root for immutable run artifacts.")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED, help="Deterministic random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Monkey-patch only this process so the reusable trainer can remain unchanged
    # for normal psx-dps / pypsx live-provider runs.
    trainer._run_real_psx_acquisition = _read_existing_real_processed_data  # type: ignore[attr-defined]
    trainer.run_training_stage(
        symbol=args.symbol,
        start=args.start,
        provider="existing-processed-psx",
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
