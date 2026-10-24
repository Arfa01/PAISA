"""Build an immutable PAISA Tree SHAP explainability pack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."


class ArgumentParsingError(ValueError):
    """Argument failure rendered through the same guarded JSON surface."""


class RedactingArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentParsingError(message)


def _redact_failure(exc: Exception) -> str:
    try:
        from paisa.evidence.diagnostics import redact_diagnostic
    except Exception:
        return "Runtime initialization failed; rerun with --debug for diagnostics."
    return redact_diagnostic(exc)


def _failure_payload(exc: Exception) -> dict[str, str]:
    return {
        "status": "FAIL",
        "error_type": type(exc).__name__,
        "error": _redact_failure(exc),
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = RedactingArgumentParser(
        description=(
            "Build a checksum-verified SHAP explanation pack for PAISA tree models "
            "M-001 through M-004 over the shared OGDC holdout. Models are not "
            "retrained. Academic simulation; not financial advice."
        )
    )
    parser.add_argument(
        "--model-training-run",
        required=True,
        help="Authoritative checksum-sealed M-001 through M-004 training run.",
    )
    parser.add_argument(
        "--comparison-run",
        required=True,
        help="Authoritative unified six-model comparison pack.",
    )
    parser.add_argument(
        "--vertical-slice-run",
        required=True,
        help="Authoritative retained OGDC vertical-slice run.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Git-ignored root for immutable explainability runs.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Optional new XAI run ID; when omitted, generate a UTC timestamped "
            "filesystem-safe ID. Existing destinations are refused."
        ),
    )
    parser.add_argument(
        "--trusted-model-training-root",
        help=(
            "Optional trusted training-root boundary. Defaults to "
            "PAISA_MODEL_TRAINING_ROOT or artifacts/model_training."
        ),
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Publish only the immutable directory, without a ZIP and checksum.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Re-raise failures for developer diagnostics.",
    )
    return parser.parse_args(argv)


def _debug_requested(argv: list[str]) -> bool:
    for value in argv:
        if value == "--":
            return False
        if value == "--debug":
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    command_line = list(sys.argv[1:] if argv is None else argv)
    debug = _debug_requested(command_line)
    try:
        args = parse_args(command_line)
        from paisa.explainability.tree_shap import build_tree_shap_explainability

        result = build_tree_shap_explainability(
            model_training_run=args.model_training_run,
            comparison_run=args.comparison_run,
            vertical_slice_run=args.vertical_slice_run,
            output_dir=args.output_dir,
            run_id=args.run_id,
            trusted_model_training_root=args.trusted_model_training_root,
            create_zip=not args.no_zip,
        )
    except ArgumentParsingError as exc:
        if debug:
            raise
        print(
            json.dumps(_failure_payload(exc), indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        if debug:
            raise
        print(
            json.dumps(_failure_payload(exc), indent=2, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
