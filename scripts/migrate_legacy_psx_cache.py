"""Migrate a genuine pre-schema PAISA PSX cache into a verified bundle."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

# Support the documented command before an editable package install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paisa.legacy_cache import migrate_legacy_psx_cache  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and migrate a genuine pre-schema PAISA PSX cache into an "
            "immutable schema-1.0 verified-legacy bundle. This command performs "
            "no live acquisition and never modifies the source files."
        )
    )
    parser.add_argument(
        "--legacy-prices",
        required=True,
        help="Path to the original legacy processed psx_prices.csv.",
    )
    parser.add_argument(
        "--legacy-metadata",
        required=True,
        help="Path to the original pre-schema pipeline_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New bundle directory; the command refuses to overwrite it.",
    )
    parser.add_argument(
        "--legacy-raw-dir",
        help=(
            "Legacy raw-data root. When omitted, the command inspects the data/raw "
            "sibling inferred from --legacy-prices. Complete exact "
            "<SYMBOL>_psx_dps_raw.csv captures are mandatory for every requested "
            "symbol; processed-only caches are rejected."
        ),
    )
    parser.add_argument("--provider", default="psx_dps")
    parser.add_argument("--required-symbol", default="OGDC")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = migrate_legacy_psx_cache(
            args.legacy_prices,
            args.legacy_metadata,
            args.output_dir,
            legacy_raw_dir=args.legacy_raw_dir,
            expected_provider=args.provider,
            required_symbol=args.required_symbol,
        )
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "live_acquisition_performed": False,
        }
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
