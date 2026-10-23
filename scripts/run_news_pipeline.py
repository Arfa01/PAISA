"""Run one immutable PAISA financial-news ingestion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# Support the documented command before an editable package install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.news.pipeline import make_news_run_id, run_news_pipeline  # noqa: E402
from paisa.news.providers.gdelt_doc import GdeltDocNewsProvider  # noqa: E402
from paisa.news.providers.manual_csv import ManualCsvNewsProvider  # noqa: E402
from paisa.news.providers.rss import RssNewsProvider  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire permitted RSS/Atom, GDELT DOC archive metadata, or manually "
            "imported financial-news "
            "records and publish one immutable, normalized evidence pack. "
            "This command does not run sentiment analysis and is not financial advice."
        )
    )
    parser.add_argument(
        "--provider",
        required=True,
        choices=("rss", "gdelt-doc", "manual-csv"),
        help="News acquisition adapter. Fixture data is never a CLI option.",
    )
    parser.add_argument(
        "--config",
        help=(
            "Versioned source configuration; required for --provider rss or "
            "--provider gdelt-doc."
        ),
    )
    parser.add_argument(
        "--input",
        help="Permitted news CSV path; required for --provider manual-csv.",
    )
    parser.add_argument(
        "--source-name",
        help="Default publisher/source name for manual CSV rows that omit it.",
    )
    parser.add_argument(
        "--default-language",
        help="Default language tag for manual CSV rows that omit it, for example en.",
    )
    parser.add_argument(
        "--licensing-usage-note",
        help=(
            "Required permission/licensing note for a manual CSV import; records "
            "the allowed academic storage and use of the supplied fields."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/news_ingestion",
        help="Immutable ingestion-run output root.",
    )
    parser.add_argument(
        "--database",
        default="data/news/paisa_news.sqlite3",
        help="Local SQLite news database path.",
    )
    parser.add_argument(
        "--storage-root",
        default=".",
        help=(
            "Trusted root containing artifact and database paths. Paths outside "
            "this root are rejected unless explicitly allowed."
        ),
    )
    parser.add_argument(
        "--allow-external-storage",
        action="store_true",
        help=(
            "Explicitly permit artifact/database paths outside --storage-root. "
            "Persisted summaries still use non-host-identifying logical labels."
        ),
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return exit status 0 for a retained PARTIAL run; default is non-zero.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional unique NEWS-prefixed run ID; existing destinations are rejected.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.provider in {"rss", "gdelt-doc"}:
        if not args.config:
            parser.error(f"--provider {args.provider} requires --config")
        if args.input:
            parser.error("--input is only valid with --provider manual-csv")
        if args.source_name or args.default_language or args.licensing_usage_note:
            parser.error(
                "--source-name, --default-language, and --licensing-usage-note "
                "are only valid with --provider manual-csv"
            )
    else:
        if not args.input:
            parser.error("--provider manual-csv requires --input")
        if not args.licensing_usage_note or not args.licensing_usage_note.strip():
            parser.error("--provider manual-csv requires --licensing-usage-note")
        if args.config:
            parser.error(
                "--config is only valid with --provider rss or --provider gdelt-doc"
            )
    return args


def _build_provider(args: argparse.Namespace) -> Any:
    if args.provider == "rss":
        return RssNewsProvider.from_config(args.config)
    if args.provider == "gdelt-doc":
        return GdeltDocNewsProvider.from_config(args.config)
    return ManualCsvNewsProvider(
        args.input,
        licensing_usage_note=args.licensing_usage_note,
        source_name=args.source_name,
        default_language=args.default_language,
    )


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_run_id = args.run_id or make_news_run_id()
    try:
        provider = _build_provider(args)
        summary = run_news_pipeline(
            provider,
            output_dir=args.output_dir,
            database_path=args.database,
            storage_root=args.storage_root,
            allow_external_storage=args.allow_external_storage,
            run_id=selected_run_id,
        )
    except Exception as exc:
        # Do not echo arbitrary provider exceptions: request URLs or third-party
        # messages can contain credentials. Detailed provider failures, when a
        # run was published, live in its redacted immutable evidence instead.
        _print_json(
            {
                "status": "FAIL",
                "run_id": selected_run_id,
                "error_type": type(exc).__name__,
                "error": (
                    "News ingestion failed; review the command arguments and "
                    "the immutable run evidence if it was published."
                ),
            }
        )
        return 1

    payload = summary.to_dict()
    _print_json(payload)
    if summary.run_status == "PASS":
        return 0
    if summary.run_status == "PARTIAL" and args.allow_partial:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
