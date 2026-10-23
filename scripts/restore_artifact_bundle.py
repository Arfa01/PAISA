#!/usr/bin/env python3
"""Restore an explicitly allowlisted PAISA transfer through private staging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paisa.evidence.transfer import (  # noqa: E402
    restore_allowlisted_artifact_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reject unsafe tar members, extract only explicitly allowed PAISA "
            "roots into private staging, and publish without replacement."
        )
    )
    parser.add_argument("--archive", required=True)
    parser.add_argument("--destination-root", default=".")
    parser.add_argument("--allow-directory", action="append", default=[])
    parser.add_argument("--allow-file", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        restored = restore_allowlisted_artifact_bundle(
            args.archive,
            destination_root=args.destination_root,
            allowed_directories=args.allow_directory,
            allowed_files=args.allow_file,
        )
    except Exception:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": "artifact_bundle_restore_failed",
                    "message": (
                        "Artifact bundle was not restored; its member allowlist, "
                        "private staging, or no-replace boundary failed."
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    destination = Path(args.destination_root).resolve()
    print(
        json.dumps(
            {
                "status": "PASS",
                "restored_roots": [
                    path.relative_to(destination).as_posix() for path in restored
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
