"""Build the immutable PAISA M-001 through M-006 comparison pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.experiments.unified_comparison import (  # noqa: E402
    ACADEMIC_DISCLAIMER,
    build_unified_price_model_comparison,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recompute the unified price-only comparison from checksum-verified "
            "persisted predictions and portfolio ledgers. Models are not retrained "
            "and backtests are not rerun. Academic simulation; not financial advice."
        )
    )
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--vertical-slice", required=True)
    parser.add_argument("--lstm-run", required=True)
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Publish only the immutable directory, without a ZIP and checksum.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_unified_price_model_comparison(
            comparison_id=args.comparison_id,
            output_dir=args.output_dir,
            training_run=args.training_run,
            vertical_slice=args.vertical_slice,
            lstm_run=args.lstm_run,
            create_zip=not args.no_zip,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "disclaimer": ACADEMIC_DISCLAIMER,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
