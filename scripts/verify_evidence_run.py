#!/usr/bin/env python3
"""Verify one immutable contemporaneous-evidence pack offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paisa.evidence.artifacts import verify_evidence_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify checksums, completion seal, and normalized evidence semantics."
    )
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            verify_evidence_run(args.run_dir).to_dict(),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
