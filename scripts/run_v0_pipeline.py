from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

# Allow `python scripts/run_v0_pipeline.py` before the package is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paisa.pipeline import run_v0_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAISA v0 PSX data pipeline.")
    parser.add_argument("--provider", default="psx-dps", choices=["psx-dps", "pypsx", "sample", "manual-csv"], help="Data source adapter.")
    parser.add_argument("--symbols", nargs="+", default=["OGDC", "HBL", "MCB"], help="PSX tickers, e.g. OGDC HBL MCB")
    parser.add_argument("--start", default="2021-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD; omit for today/provider max")
    parser.add_argument("--manual-csv-dir", default="data/external/manual_psx", help="Folder used by provider=manual-csv")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    parser.add_argument("--model-dir", default=None, help="Override model directory")
    parser.add_argument("--train", action="store_true", help="Train baseline RandomForest after dataset generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_v0_pipeline(
        symbols=args.symbols,
        provider_name=args.provider,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        train=args.train,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        manual_csv_dir=args.manual_csv_dir,
    )
    print(json.dumps(metadata, indent=2, default=str))


if __name__ == "__main__":
    main()
