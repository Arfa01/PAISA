#!/usr/bin/env python3
"""Restore the checksum-bound canonical EVID/ECOV bundle safely."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paisa.evidence.transfer import restore_evidence_bundle  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the exact four-pack PAISA evidence transfer in private "
            "staging and publish it without replacing any existing pack."
        )
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination-root", default=".")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        restored = restore_evidence_bundle(
            args.archive,
            destination_root=args.destination_root,
        )
    except Exception:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": "evidence_bundle_restore_failed",
                    "message": (
                        "Evidence bundle was not restored; its allowlist, seals, "
                        "staging verification, or no-replace boundary failed."
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "restored_roots": [
                    path.relative_to(Path(args.destination_root).resolve()).as_posix()
                    for path in restored
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
