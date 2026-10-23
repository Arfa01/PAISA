"""Publish one immutable PAISA historical-news coverage audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.news.coverage import (  # noqa: E402
    CoverageConfig,
    ReadinessThresholds,
    audit_news_price_overlap,
    discover_news_run_directories,
    make_coverage_run_id,
    parse_cutoff,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit immutable financial-news evidence against the OGDC price and "
            "chronological split artifacts. This command maps availability to "
            "decision cutoffs; it does not aggregate sentiment or provide financial advice."
        )
    )
    news = parser.add_mutually_exclusive_group(required=True)
    news.add_argument(
        "--news-run-dir",
        action="append",
        help=(
            "Completed immutable news-ingestion run directory. Repeat for every "
            "run that belongs in the evidence set."
        ),
    )
    news.add_argument(
        "--news-root",
        help="Discover every completed PASS/PARTIAL child below this news-run root.",
    )
    parser.add_argument(
        "--vertical-slice-dir",
        default="artifacts/vertical_slice/VSL-20261020-OGDC-main-02",
        help="Completed immutable OGDC vertical-slice directory.",
    )
    parser.add_argument(
        "--model-training-dir",
        default="artifacts/model_training/TRN-20261020-191815-624662-nogit",
        help="Immutable model-training directory containing the exact split manifest.",
    )
    parser.add_argument(
        "--news-database",
        default="data/news/paisa_news.sqlite3",
        help=(
            "Optional current SQLite database to fingerprint. Normalized run packs, "
            "not mutable queries, remain the row evidence. Pass an empty value to omit."
        ),
    )
    parser.add_argument(
        "--source-assessment",
        default="docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md",
        help="Reviewed source/permission assessment whose checksum is bound to the run.",
    )
    parser.add_argument(
        "--source-permission-policy",
        default="configs/news/historical_source_permissions.json",
        help=(
            "Machine-readable permission decisions bound to the reviewed source "
            "assessment; unresolved decisions cannot support READY."
        ),
    )
    parser.add_argument(
        "--readiness-config",
        default="configs/news_coverage/default_readiness.json",
        help="Versioned engineering/data-coverage threshold configuration.",
    )
    parser.add_argument(
        "--market-timezone",
        default="Asia/Karachi",
        help="IANA timezone used to construct each observed session cutoff.",
    )
    parser.add_argument(
        "--decision-cutoff",
        default="15:30:00",
        help="Local decision cutoff as HH:MM or HH:MM:SS.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--real-mode",
        action="store_true",
        default=True,
        help="Reject fixture/mock/synthetic/test evidence (default).",
    )
    mode.add_argument(
        "--allow-fixture",
        action="store_true",
        help="Test-only mode; never use its result as a real historical audit.",
    )
    parser.add_argument(
        "--allow-legacy-unsealed",
        action="store_true",
        help=(
            "Explicitly accept pre-completion-seal news packs after validating "
            "their complete manifest checksum inventories. This weaker legacy "
            "boundary is recorded and does not upgrade those packs."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/news_coverage",
        help="Root for immutable COVERAGE-* run directories.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique COVERAGE-prefixed run ID; collisions are rejected.",
    )
    return parser


def _read_thresholds(path: str | Path) -> ReadinessThresholds:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Readiness configuration is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Readiness configuration must contain a JSON object")
    return ReadinessThresholds.from_mapping(value)


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected_run_id = args.run_id or make_coverage_run_id()
    try:
        run_directories = (
            discover_news_run_directories(args.news_root)
            if args.news_root
            else [Path(value) for value in args.news_run_dir]
        )
        config = CoverageConfig(
            market_timezone=args.market_timezone,
            decision_cutoff=parse_cutoff(args.decision_cutoff),
            thresholds=_read_thresholds(args.readiness_config),
        )
        computation, destination = audit_news_price_overlap(
            news_run_directories=run_directories,
            vertical_slice_directory=args.vertical_slice_dir,
            model_training_directory=args.model_training_dir,
            output_root=args.output_dir,
            source_assessment_path=args.source_assessment,
            config=config,
            run_id=selected_run_id,
            real_mode=not args.allow_fixture,
            database_path=args.news_database or None,
            allow_legacy_unsealed=args.allow_legacy_unsealed,
            source_permission_policy_path=args.source_permission_policy,
        )
    except Exception as exc:
        # Arbitrary provider/config/database messages can contain host paths or
        # credentials. Keep CLI failure output body-free and path-free.
        _print_json(
            {
                "status": "FAIL",
                "run_id": selected_run_id,
                "error_type": type(exc).__name__,
                "error": (
                    "Coverage audit failed; review the command arguments and "
                    "checksum-bound input evidence."
                ),
            }
        )
        return 1

    payload = dict(computation.final_summary)
    payload.update(
        {
            "run_id": selected_run_id,
            "artifact_directory": f"news_coverage/{destination.name}",
        }
    )
    _print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
