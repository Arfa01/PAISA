#!/usr/bin/env python3
"""Verify one immutable contemporaneous-evidence coverage pack offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paisa.evidence.coverage_artifacts import (  # noqa: E402
    verify_evidence_coverage_pack,
)


ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify checksums and internal table semantics for one immutable "
            "ECOV-* pack without network access. This is pack-semantic "
            "reconciliation, not a replay of the source evidence."
        )
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    try:
        verification = verify_evidence_coverage_pack(args.run_dir)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "verification": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": (
                        "Evidence coverage verification failed; inspect the "
                        "immutable pack and its completion evidence."
                    ),
                    "academic_disclaimer": ACADEMIC_DISCLAIMER,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                **verification.to_dict(),
                "academic_disclaimer": ACADEMIC_DISCLAIMER,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
