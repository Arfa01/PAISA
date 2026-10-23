"""Offline verifier for a PAISA unified price-model comparison pack."""

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
    verify_unified_comparison_pack,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify comparison checksums, source bindings, exact inventory, and "
            "replayed metrics without network access or model loading."
        )
    )
    parser.add_argument("pack", help="Comparison directory or canonical ZIP.")
    parser.add_argument(
        "--checksum",
        help=(
            "ZIP SHA-256 sidecar; required for ZIP input and auto-detected at "
            "<pack>.sha256 when omitted."
        ),
    )
    parser.add_argument("--training-run", help="Optional authoritative training root.")
    parser.add_argument("--vertical-slice", help="Optional authoritative vertical root.")
    parser.add_argument("--lstm-run", help="Optional authoritative LSTM root.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = verify_unified_comparison_pack(
            args.pack,
            checksum=args.checksum,
            training_run=args.training_run,
            vertical_slice=args.vertical_slice,
            lstm_run=args.lstm_run,
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
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
