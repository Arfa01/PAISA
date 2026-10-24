"""Offline verifier for a PAISA Tree SHAP explainability pack."""

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
            "Verify an explainability directory or canonical ZIP entirely offline, "
            "without loading executable model artifacts."
        )
    )
    parser.add_argument("pack", help="Explainability run directory or canonical ZIP.")
    parser.add_argument(
        "--checksum",
        help=(
            "ZIP SHA-256 transport sidecar; required for ZIP input and "
            "auto-detected at <pack>.sha256 when omitted. A sidecar detects "
            "corruption but does not authenticate provenance."
        ),
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        help=(
            "Independently obtained SHA-256 of artifact_manifest.json. Official "
            "allowlisted runs use their committed digest automatically."
        ),
    )
    parser.add_argument(
        "--require-authenticated",
        action="store_true",
        help=(
            "Fail unless the pack matches a committed official digest or the "
            "independent digest supplied with --expected-manifest-sha256."
        ),
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
        from paisa.explainability.tree_shap import verify_explainability_pack

        result = verify_explainability_pack(
            args.pack,
            checksum=args.checksum,
            expected_manifest_sha256=args.expected_manifest_sha256,
            require_authenticated=args.require_authenticated,
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
