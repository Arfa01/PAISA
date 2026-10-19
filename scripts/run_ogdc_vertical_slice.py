"""Run the complete real-data PAISA OGDC vertical slice."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Support the documented command before an editable package install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.experiments.experiment_config import DEFAULT_START, RANDOM_SEED  # noqa: E402
from paisa.vertical_slice import (  # noqa: E402
    ACADEMIC_DISCLAIMER,
    run_ogdc_vertical_slice,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch real PSX OGDC data, train M-001 through M-004, run four "
            "independent portfolio simulations and two benchmarks, and publish "
            "one immutable comparison report. Outputs are academic simulations, "
            "not financial advice."
        )
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START,
        help=f"Earliest daily PSX session to use (default: {DEFAULT_START}).",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/vertical_slice",
        help="Immutable vertical-slice output root.",
    )
    parser.add_argument(
        "--model-output-dir",
        default="artifacts/model_training",
        help="Immutable model-training output root.",
    )
    parser.add_argument(
        "--cached-prices",
        default="data/processed/psx_prices.csv",
        help=(
            "Previously cleaned real PSX file considered only when the live "
            "psx-dps request fails. Sample/synthetic/fixture sources are rejected."
        ),
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique, filesystem-safe vertical-slice run ID.",
    )
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = run_ogdc_vertical_slice(
            output_dir=args.output_dir,
            model_output_dir=args.model_output_dir,
            cached_prices_path=args.cached_prices,
            start=args.start,
            vertical_slice_run_id=args.run_id,
            random_state=args.random_state,
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
