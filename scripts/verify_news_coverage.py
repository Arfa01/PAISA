"""Verify one completed PAISA news-coverage evidence pack offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.news.coverage import (  # noqa: E402
    CoverageAuditError,
    discover_news_run_directories,
    verify_coverage_pack,
    verify_coverage_run,
)


_GENERIC_VERIFICATION_ERROR = (
    "Coverage verification failed; inspect the immutable pack and its "
    "completion evidence."
)
_SAFE_EVIDENCE_REQUIREMENT_ERRORS = frozenset(
    {
        "Source-bound verification requires the referenced SQLite database",
        "Source-bound verification requires the referenced permission policy",
        (
            "Source-bound verification requires the external energy-sector "
            "link contract"
        ),
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify one immutable COVERAGE-* pack without network access. The "
            "default source-bound mode reopens and replays the exact news, SQLite, "
            "price, split, assessment, and permission evidence. Use --pack-only "
            "only to acknowledge the weaker sealed-pack semantic check."
        )
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed immutable COVERAGE-* directory to verify.",
    )
    news = parser.add_mutually_exclusive_group()
    news.add_argument(
        "--news-run-dir",
        action="append",
        help="Exact news-ingestion run directory; repeat for every bound run.",
    )
    news.add_argument(
        "--news-root",
        help="Discover completed news-ingestion runs below this directory.",
    )
    parser.add_argument("--news-database", help="Exact bound SQLite news database.")
    parser.add_argument(
        "--vertical-slice-dir",
        help="Exact immutable OGDC vertical-slice directory.",
    )
    parser.add_argument(
        "--model-training-dir",
        help="Exact model-training directory containing the split manifest.",
    )
    parser.add_argument(
        "--source-assessment",
        help="Exact reviewed historical-news source assessment.",
    )
    parser.add_argument(
        "--source-permission-policy",
        help="Exact checksum-bound source-permission policy, when used.",
    )
    parser.add_argument(
        "--energy-sector-link-contract",
        help=(
            "Exact external versioned energy-sector link contract when the "
            "sealed pack measures that optional population."
        ),
    )
    parser.add_argument(
        "--allow-legacy-unsealed",
        action="store_true",
        help="Replay explicitly accepted legacy news packs without current seals.",
    )
    parser.add_argument(
        "--allow-fixture",
        action="store_true",
        help="Test-only: allow fixture news evidence during source-bound replay.",
    )
    parser.add_argument(
        "--pack-only",
        action="store_true",
        help=(
            "Acknowledge a weaker pack-semantic check that does not reopen the "
            "referenced source evidence."
        ),
    )
    return parser


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _safe_verification_error(exc: Exception) -> str:
    """Expose only fixed evidence requirements, never arbitrary exception text."""

    if isinstance(exc, CoverageAuditError):
        message = str(exc)
        if message in _SAFE_EVIDENCE_REQUIREMENT_ERRORS:
            return message
    return _GENERIC_VERIFICATION_ERROR


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.pack_only:
            source_options = (
                args.news_run_dir,
                args.news_root,
                args.news_database,
                args.vertical_slice_dir,
                args.model_training_dir,
                args.source_assessment,
                args.source_permission_policy,
                args.energy_sector_link_contract,
            )
            if any(value for value in source_options):
                parser.error(
                    "--pack-only cannot be combined with source-evidence options"
                )
            result = verify_coverage_pack(args.run_dir)
        else:
            missing = [
                option
                for option, value in (
                    ("--news-run-dir/--news-root", args.news_run_dir or args.news_root),
                    ("--vertical-slice-dir", args.vertical_slice_dir),
                    ("--model-training-dir", args.model_training_dir),
                    ("--source-assessment", args.source_assessment),
                )
                if not value
            ]
            if missing:
                parser.error(
                    "source-bound verification requires " + ", ".join(missing)
                )
            run_directories = (
                discover_news_run_directories(args.news_root)
                if args.news_root
                else [Path(value) for value in args.news_run_dir]
            )
            result = verify_coverage_run(
                args.run_dir,
                news_run_directories=run_directories,
                vertical_slice_directory=args.vertical_slice_dir,
                model_training_directory=args.model_training_dir,
                source_assessment_path=args.source_assessment,
                database_path=args.news_database or None,
                source_permission_policy_path=args.source_permission_policy,
                energy_sector_link_contract_path=(
                    args.energy_sector_link_contract
                ),
                real_mode=not args.allow_fixture,
                allow_legacy_unsealed=args.allow_legacy_unsealed,
            )
    except Exception as exc:
        _print_json(
            {
                "status": "FAIL",
                "verification": "FAIL",
                "error_type": type(exc).__name__,
                "error": _safe_verification_error(exc),
            }
        )
        return 1
    _print_json(result.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
