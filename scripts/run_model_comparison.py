from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Allow `python scripts/run_model_comparison.py` before package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paisa.config import get_paths
from paisa.models.experiments import run_price_only_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PAISA v0.1 price-only model comparison.")
    parser.add_argument("--data-dir", default=None, help="Override data directory. Defaults to PAISA_DATA_DIR or ./data.")
    parser.add_argument("--model-dir", default=None, help="Override model directory. Defaults to PAISA_MODEL_DIR or ./models.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = get_paths(data_dir=args.data_dir, model_dir=args.model_dir)
    dataset_path = paths.processed_dir / "ml_dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError("ml_dataset.csv not found. Run scripts/run_v0_pipeline.py first.")
    dataset = pd.read_csv(dataset_path)
    result = run_price_only_model_comparison(dataset=dataset, model_dir=paths.model_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
