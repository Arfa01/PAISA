"""Build the immutable PAISA M-001 through M-006 comparison pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.evidence.diagnostics import (  # noqa: E402
    ArgumentParsingError,
    RedactingArgumentParser,
    redact_diagnostic,
)
from paisa.experiments.unified_comparison import (  # noqa: E402
    ACADEMIC_DISCLAIMER,
    build_unified_price_model_comparison,
)


def _failure_payload(exc: Exception, *, debug: bool) -> dict[str, str]:
    payload = {
        "status": "FAIL",
        "error_type": type(exc).__name__,
        "error": redact_diagnostic(exc),
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    if debug:
        payload["traceback"] = redact_diagnostic(traceback.format_exc())
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RedactingArgumentParser(
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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include a sanitized traceback in failure JSON.",
    )
    return parser.parse_args(argv)


def _debug_requested(argv: list[str]) -> bool:
    for value in argv:
        if value == "--":
            return False
        if value == "--debug":
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    debug = _debug_requested(command_line)
    try:
        args = parse_args(command_line)
        summary = build_unified_price_model_comparison(
            comparison_id=args.comparison_id,
            output_dir=args.output_dir,
            training_run=args.training_run,
            vertical_slice=args.vertical_slice,
            lstm_run=args.lstm_run,
            create_zip=not args.no_zip,
        )
    except ArgumentParsingError as exc:
        print(
            json.dumps(_failure_payload(exc, debug=debug), indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                _failure_payload(exc, debug=debug), indent=2, sort_keys=True
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
