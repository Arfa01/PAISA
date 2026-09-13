from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Allow `python scripts/run_v0_pipeline.py` before the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paisa.config import get_paths
from paisa.data_quality import write_data_quality_report
from paisa.pipeline import run_v0_pipeline


def _load_symbols_from_csv(path: str | Path, max_priority: int | None = None) -> list[str]:
    frame = pd.read_csv(path)
    if "ticker" not in frame.columns:
        raise ValueError(f"{path} must contain a 'ticker' column.")
    if max_priority is not None and "priority" in frame.columns:
        frame = frame[pd.to_numeric(frame["priority"], errors="coerce") <= max_priority]
    symbols = frame["ticker"].dropna().astype(str).str.upper().str.strip().tolist()
    # Keep order but remove duplicates.
    return list(dict.fromkeys(symbol for symbol in symbols if symbol))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAISA v0 PSX data pipeline.")
    parser.add_argument("--provider", default="psx-dps", choices=["psx-dps", "pypsx", "sample", "manual-csv"], help="Data source adapter.")
    parser.add_argument("--symbols", nargs="+", default=None, help="PSX tickers, e.g. OGDC HBL MCB")
    parser.add_argument("--symbols-from-csv", default=None, help="Read ticker symbols from a CSV with a 'ticker' column.")
    parser.add_argument("--max-priority", type=int, default=None, help="When using --symbols-from-csv, include rows with priority <= this value.")
    parser.add_argument("--start", default="2021-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD; omit for today/provider max")
    parser.add_argument("--manual-csv-dir", default="data/external/manual_psx", help="Folder used by provider=manual-csv")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--model-dir", default=None, help="Override model directory")
    parser.add_argument("--train", action="store_true", help="Train baseline RandomForest after dataset generation")
    parser.add_argument("--quality-report", action="store_true", help="Also generate data-quality CSV/JSON and docs/data_quality_report_v0.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.symbols_from_csv:
        symbols = _load_symbols_from_csv(args.symbols_from_csv, max_priority=args.max_priority)
    else:
        symbols = args.symbols or ["OGDC", "HBL", "MCB"]

    metadata = run_v0_pipeline(
        symbols=symbols,
        provider_name=args.provider,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        train=args.train,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        manual_csv_dir=args.manual_csv_dir,
    )

    if args.quality_report:
        paths = get_paths(data_dir=args.data_dir, model_dir=args.model_dir)
        quality = write_data_quality_report(
            prices_path=paths.processed_dir / "psx_prices.csv",
            features_path=paths.processed_dir / "psx_features.csv",
            dataset_path=paths.processed_dir / "ml_dataset.csv",
            output_dir=paths.processed_dir,
            docs_dir=Path("docs"),
        )
        metadata["data_quality"] = {
            "tickers_count": quality["tickers_count"],
            "issues_count": len(quality["issues"]),
            "report_files": {
                "json": str(paths.processed_dir / "data_quality_report.json"),
                "csv": str(paths.processed_dir / "data_quality_report.csv"),
                "markdown": "docs/data_quality_report_v0.md",
            },
        }

    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
