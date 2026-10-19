"""Run one immutable PAISA backtesting experiment from contract files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support the documented command before an editable package install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paisa.experiments.runner import run_experiment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a long-only daily backtest from standardized predictions. "
            "Outputs are academic simulations, not financial advice."
        )
    )
    parser.add_argument("--predictions", required=True, help="Prediction-record CSV path")
    parser.add_argument("--prices", required=True, help="Daily market-price CSV path")
    parser.add_argument("--config", required=True, help="Experiment configuration JSON path")
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Immutable experiment output root (default: runs)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    completed = run_experiment(
        predictions_path=args.predictions,
        prices_path=args.prices,
        config_path=args.config,
        runs_root=args.runs_dir,
    )
    print(
        json.dumps(
            {
                "experiment_id": completed.config.experiment_id,
                "artifact_path": str(completed.artifact_path),
                "ending_equity": completed.metrics["ending_equity"],
                "total_return": completed.metrics["total_return"],
                "disclaimer": "Academic simulation; not financial advice.",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
