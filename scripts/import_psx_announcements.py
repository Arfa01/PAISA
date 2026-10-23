#!/usr/bin/env python3
"""Import one authorized, operator-supplied PSX OGDC announcement CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paisa.evidence.artifacts import publish_evidence_run  # noqa: E402
from paisa.evidence.policy import load_evidence_permission_policy  # noqa: E402
from paisa.evidence.psx_announcements import (  # noqa: E402
    PsxAnnouncementManualImporter,
)
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
            "Import authorized PSX OGDC announcement metadata without network "
            "access and publish one immutable evidence pack."
        )
    )
    parser.add_argument("--input", required=True, help="Operator-supplied CSV path.")
    parser.add_argument(
        "--permission-policy",
        default="configs/evidence/source_permissions.json",
        help="Checksum-bound evidence permission policy.",
    )
    attestation = parser.add_mutually_exclusive_group(required=True)
    attestation.add_argument(
        "--authorization-attestation",
        help="Explicit statement that the supplied data and intended use are authorized.",
    )
    attestation.add_argument(
        "--authorization-attestation-file",
        help="UTF-8 file containing the explicit authorization attestation.",
    )
    authorization_reference = parser.add_mutually_exclusive_group(required=True)
    authorization_reference.add_argument(
        "--authorization-reference",
        help=(
            "Exact reviewed authorization reference bound by SHA-256 in the "
            "permission policy."
        ),
    )
    authorization_reference.add_argument(
        "--authorization-reference-file",
        help="UTF-8 file containing the reviewed authorization reference.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/contemporaneous_evidence",
        help="Immutable evidence-pack root.",
    )
    parser.add_argument("--run-id", help="Optional fresh EVID-... identity.")
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 * 1024)
    parser.add_argument("--max-records", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def read_authorization_value(
        *, inline_value: str | None, file_value: str | None, label: str
    ) -> str:
        if not file_value:
            return str(inline_value or "").strip()
        authorization_path = Path(file_value)
        if authorization_path.is_symlink() or not authorization_path.is_file():
            raise SystemExit(f"{label} file must be a regular file")
        try:
            return authorization_path.read_text("utf-8").strip()
        except (OSError, UnicodeError) as exc:
            raise SystemExit(f"{label} file is unreadable") from exc

    authorization_attestation = read_authorization_value(
        inline_value=args.authorization_attestation,
        file_value=args.authorization_attestation_file,
        label="Authorization attestation",
    )
    authorization_reference_value = read_authorization_value(
        inline_value=args.authorization_reference,
        file_value=args.authorization_reference_file,
        label="Authorization reference",
    )

    policy = load_evidence_permission_policy(
        args.permission_policy,
        repository_root=REPOSITORY_ROOT,
    )
    result = PsxAnnouncementManualImporter(
        args.input,
        permission_policy=policy,
        authorization_attestation=authorization_attestation,
        authorization_reference=authorization_reference_value,
        max_bytes=args.max_bytes,
        max_records=args.max_records,
    ).fetch()
    destination = publish_evidence_run(
        result,
        output_root=args.output_dir,
        permission_policy=policy,
        repository_root=REPOSITORY_ROOT,
        run_id=args.run_id,
    )
    print(
        json.dumps(
            {
                "run_id": destination.name,
                "provider_id": result.provider_id,
                "provider_mode": result.provider_mode,
                "record_count": len(result.records),
                "artifact_directory": _artifact_directory_label(destination),
                "academic_disclaimer": (
                    "Academic simulation; not financial advice."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
