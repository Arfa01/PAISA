"""Run one immutable PAISA article-level FinBERT inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.sentiment.artifacts import make_sentiment_run_id  # noqa: E402
from paisa.sentiment.config import (  # noqa: E402
    DEFAULT_MAXIMUM_INPUT_CHARACTERS,
    DEFAULT_MAXIMUM_TOKENIZER_LENGTH,
)
from paisa.sentiment.inference import SentimentInferenceConfig  # noqa: E402
from paisa.sentiment.pipeline import run_sentiment_inference  # noqa: E402


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _positive_number(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the pinned ProsusAI FinBERT revision to normalized canonical "
            "articles from one completed news-ingestion run. This produces an "
            "academic article-level simulation and is not financial advice."
        )
    )
    parser.add_argument(
        "--source-run-id",
        required=True,
        help="Completed NEWS-... ingestion run whose canonical articles are scored.",
    )
    parser.add_argument(
        "--database",
        default="data/news/paisa_news.sqlite3",
        help="SQLite database containing completed normalized news evidence.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/sentiment_inference",
        help="Immutable sentiment-run artifact root.",
    )
    parser.add_argument(
        "--model-cache",
        default="data/sentiment_model_cache",
        help="Configurable local cache for the exact pinned model snapshot.",
    )
    parser.add_argument(
        "--storage-root",
        default=".",
        help="Trusted root containing database, artifacts, and model cache.",
    )
    parser.add_argument(
        "--allow-external-storage",
        action="store_true",
        help=(
            "Permit reviewed paths outside --storage-root; public artifacts still "
            "record only portable labels."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "mps"),
        default="cpu",
        help="Inference device. CPU is the reproducible default.",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=8,
        help="Maximum articles passed to the model in one batch.",
    )
    parser.add_argument(
        "--max-input-characters",
        type=_positive_integer,
        default=DEFAULT_MAXIMUM_INPUT_CHARACTERS,
        help="Deterministic character limit applied before tokenization.",
    )
    parser.add_argument(
        "--max-token-length",
        type=_positive_integer,
        default=DEFAULT_MAXIMUM_TOKENIZER_LENGTH,
        help="Tokenizer sequence limit, from 1 through 512.",
    )
    parser.add_argument(
        "--max-articles",
        type=_positive_integer,
        default=1_000,
        help="Hard bound on canonical articles considered by this invocation.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=_positive_number,
        help="Optional watchdog checked between inference batches.",
    )
    parser.add_argument(
        "--audit-sample-size",
        type=_positive_integer,
        default=10,
        help="Deterministic maximum number of rows in the audit sample.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional new SENT-YYYYMMDD-HHMMSS-xxxxxxxx identity.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return zero for a retained PARTIAL run; default is non-zero.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_token_length > 512:
        parser.error("--max-token-length must be no greater than 512")
    return args


def _print_json(value: dict[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    selected_run_id = args.run_id or make_sentiment_run_id()
    config = SentimentInferenceConfig(
        batch_size=args.batch_size,
        maximum_input_characters=args.max_input_characters,
        maximum_sequence_length=args.max_token_length,
        maximum_articles=args.max_articles,
        timeout_seconds=args.timeout_seconds,
    )
    try:
        summary = run_sentiment_inference(
            source_ingestion_run_id=args.source_run_id,
            database_path=args.database,
            output_dir=args.output_dir,
            model_cache=args.model_cache,
            storage_root=args.storage_root,
            allow_external_storage=args.allow_external_storage,
            run_id=selected_run_id,
            device=args.device,
            config=config,
            audit_sample_size=args.audit_sample_size,
        )
    except Exception as exc:
        # Arbitrary exception text can contain remote URLs, cache paths, or
        # credentials. Detailed controlled failures belong in the run pack.
        _print_json(
            {
                "schema_version": "1.0",
                "run_id": selected_run_id,
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": (
                    "Sentiment inference failed; review arguments and any "
                    "structured immutable run evidence."
                ),
                "academic_use_only": True,
                "financial_advice": False,
            }
        )
        return 1

    _print_json(summary.to_dict())
    if summary.status == "PASS":
        return 0
    if summary.status == "PARTIAL" and args.allow_partial:
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
