from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python scripts/generate_data_quality_report.py` before package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paisa.config import get_paths
from paisa.data_quality import write_data_quality_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate PAISA v0 data-quality reports from processed CSVs.")
    parser.add_argument("--data-dir", default=None, help="Override data directory. Defaults to PAISA_DATA_DIR or ./data.")
    parser.add_argument("--model-dir", default=None, help="Override model directory. Only used to resolve project paths.")
    parser.add_argument("--docs-dir", default="docs", help="Where to write data_quality_report_v0.md. Use empty string to skip docs output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths(data_dir=args.data_dir, model_dir=args.model_dir)
    docs_dir = Path(args.docs_dir) if args.docs_dir else None

    report = write_data_quality_report(
        prices_path=paths.processed_dir / "psx_prices.csv",
        features_path=paths.processed_dir / "psx_features.csv",
        dataset_path=paths.processed_dir / "ml_dataset.csv",
        output_dir=paths.processed_dir,
        docs_dir=docs_dir,
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
