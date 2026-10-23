"""Run the immutable PAISA OGDC price-only LSTM comparison."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.experiments.lstm_runner import (  # noqa: E402
    ACADEMIC_DISCLAIMER,
    run_lstm_price_only_experiments,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train immutable M-005 and M-006 price-only LSTMs, freeze selection "
            "on validation, evaluate test once, and compare all six PAISA models "
            "through the same backtester. This is an academic simulation, not "
            "financial advice. No network access is used."
        )
    )
    parser.add_argument(
        "--prices",
        required=True,
        help="Checksum-bound canonical OGDC daily prices CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/lstm_experiments",
        help="Immutable LSTM experiment output root.",
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="New filesystem-safe run ID; existing destinations are refused.",
    )
    parser.add_argument(
        "--baseline-training-run",
        required=True,
        help="Checksum-sealed M-001 through M-004 training-run directory.",
    )
    parser.add_argument(
        "--baseline-vertical-slice",
        required=True,
        help="Checksum-inventoried retained vertical-slice directory.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu", "metal"),
        default="auto",
        help=(
            "Requested TensorFlow device policy (default: auto); metal is "
            "macOS-only and requires a TensorFlow-visible GPU."
        ),
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Do not create the optional ZIP adjacent to the immutable run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_lstm_price_only_experiments(
            prices_path=args.prices,
            output_dir=args.output_dir,
            run_id=args.run_id,
            baseline_training_run=args.baseline_training_run,
            baseline_vertical_slice=args.baseline_vertical_slice,
            create_zip=not args.no_zip,
            device=args.device,
        )
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "disclaimer": ACADEMIC_DISCLAIMER,
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
