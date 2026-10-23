#!/usr/bin/env python3
"""Acquire bounded official GDELT GKG 2.1 raw-archive evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paisa.evidence.artifacts import (  # noqa: E402
    publish_evidence_run,
    verify_evidence_run,
)
from paisa.evidence.gdelt_gkg import GdeltGkgArchiveProvider  # noqa: E402
from paisa.evidence.policy import load_evidence_permission_policy  # noqa: E402
from paisa.sentiment.artifacts import logical_storage_label  # noqa: E402


def _artifact_directory_label(destination: Path) -> str:
    parent = logical_storage_label(
        destination.parent,
        storage_root=REPOSITORY_ROOT,
        kind="evidence",
    )
    return f"{parent}/{destination.name}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download only explicit official GKG 2.1 partitions under byte, "
            "row, and archive limits; publisher pages are never fetched."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/evidence/gdelt_gkg_test_subset.json",
        help="Bounded raw-archive configuration.",
    )
    parser.add_argument(
        "--permission-policy",
        default="configs/evidence/source_permissions.json",
    )
    parser.add_argument(
        "--checkpoint-dir",
        help=(
            "Private resume directory. Defaults below data/evidence using the "
            "configuration filename."
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable validated archive checkpoint caching for this run.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/contemporaneous_evidence",
    )
    parser.add_argument("--run-id", help="Optional fresh EVID-... identity.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config_path = Path(args.config)
    try:
        config_document = json.loads(config_path.read_text("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SystemExit("GDELT GKG configuration is unreadable") from exc
    policy = load_evidence_permission_policy(
        args.permission_policy,
        repository_root=REPOSITORY_ROOT,
    )
    policy.require("gdelt-gkg-2.1", "raw_archive", automated=True)
    if config_document.get("permission_policy_id") != policy.policy_id:
        raise SystemExit(
            "GDELT configuration permission_policy_id differs from the supplied policy"
        )
    checkpoint_directory: Path | None
    if args.no_resume:
        checkpoint_directory = None
    elif args.checkpoint_dir:
        checkpoint_directory = Path(args.checkpoint_dir)
    else:
        checkpoint_directory = (
            REPOSITORY_ROOT
            / "data"
            / "evidence"
            / "gdelt_gkg_checkpoints"
            / config_path.stem
        )
    provider = GdeltGkgArchiveProvider.from_config(
        config_path,
        permission_policy=policy,
        checkpoint_directory=checkpoint_directory,
    )
    result = provider.fetch()
    destination = publish_evidence_run(
        result,
        output_root=args.output_dir,
        permission_policy=policy,
        repository_root=REPOSITORY_ROOT,
        run_id=args.run_id,
    )
    verification = verify_evidence_run(destination)
    print(
        json.dumps(
            {
                **verification.to_dict(),
                "artifact_directory": _artifact_directory_label(destination),
                "partitions_configured": result.counters.get(
                    "partitions_configured", 0
                ),
                "partitions_completed": result.counters.get(
                    "partitions_completed", 0
                ),
                "records_scanned": result.counters.get("records_scanned", 0),
                "ogdc_record_count": len(result.records),
                "academic_disclaimer": (
                    "Academic simulation; not financial advice."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 2 if verification.status == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
