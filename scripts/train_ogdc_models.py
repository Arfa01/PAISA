"""Train immutable PAISA OGDC model variants from PSX or an existing CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support the documented command before an editable package install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.experiments.experiment_config import (  # noqa: E402
    DEFAULT_PROVIDER,
    DEFAULT_START,
    DEFAULT_SYMBOL,
    MODEL_VARIANTS,
    RANDOM_SEED,
)
from paisa.experiments.trainer import run_training_stage  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train versioned Random Forest and XGBoost variants for OGDC. "
            "Outputs are research artifacts, not financial advice."
        )
    )
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument(
        "--prices",
        help=(
            "Optional existing standardized daily price CSV. When omitted, "
            "the existing PSX provider pipeline fetches data."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/model_training",
        help="Immutable training-run output root.",
    )
    parser.add_argument("--run-id", help="Optional unique, filesystem-safe run ID.")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--model-id",
        action="append",
        choices=[variant.model_id for variant in MODEL_VARIANTS],
        help="Train only this variant; repeat to select several. Default: all.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip the adjacent ZIP archive (useful for local tests).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = run_training_stage(
        symbol=args.symbol,
        start=args.start,
        provider=args.provider,
        prices_path=args.prices,
        output_dir=args.output_dir,
        run_id=args.run_id,
        random_state=args.random_state,
        create_zip=not args.no_zip,
        model_ids=args.model_id,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
