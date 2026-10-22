"""Verify and replay an immutable PAISA news-ingestion artifact pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.news.verification import (  # noqa: E402
    NewsRunVerificationError,
    verify_news_run,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Offline integrity verification and deterministic replay for one "
            "completed PAISA news-ingestion artifact directory. No network or "
            "database writes are performed."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed immutable news run directory containing completion.json.",
    )
    parser.add_argument(
        "--database",
        help=(
            "Optional SQLite database to open read-only and verify the committed "
            "run row plus its manifest, inventory, and completion-marker seals."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify_news_run(args.run_dir, database_path=args.database)
    except (NewsRunVerificationError, OSError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "verification": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
