from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/train_ogdc_rf_variants.py` before editable install.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from paisa.experiments.experiment_config import DEFAULT_PROVIDER, DEFAULT_START, DEFAULT_SYMBOL, RANDOM_SEED
from paisa.experiments.trainer import run_training_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen Random Forest variants for the PAISA OGDC experiment stage."
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help="Ticker symbol. Default: OGDC")
    parser.add_argument("--start", default=DEFAULT_START, help="Requested start date YYYY-MM-DD.")
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Data provider. Default: psx-dps.")
    parser.add_argument("--output-dir", default="artifacts/model_training", help="Output root for immutable run artifacts.")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED, help="Deterministic random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_training_stage(
        symbol=args.symbol,
        start=args.start,
        provider=args.provider,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
