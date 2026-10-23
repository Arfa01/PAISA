"""Verify one completed PAISA news-coverage evidence pack offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.news.coverage import verify_coverage_run  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the completion seal, manifest, and complete payload inventory "
            "of one immutable COVERAGE-* evidence pack without network access."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed immutable COVERAGE-* directory to verify.",
    )
    return parser


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_coverage_run(args.run_dir)
    except Exception as exc:
        _print_json(
            {
                "status": "FAIL",
                "verification": "FAIL",
                "error_type": type(exc).__name__,
                "error": (
                    "Coverage verification failed; inspect the immutable pack "
                    "and its completion evidence."
                ),
            }
        )
        return 1
    _print_json(result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
