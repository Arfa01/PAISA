"""Safe, deterministic helpers for immutable sentiment-run artifacts."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4


_RUN_ID = re.compile(r"^SENT-[0-9]{8}-[0-9]{6}-[A-Za-z0-9]{8}$")


class SentimentRunCollisionError(FileExistsError):
    """Raised when an immutable run identity or destination already exists."""


def utc_now_z() -> str:
    """Return a timezone-aware UTC timestamp using the repository convention."""

    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def make_sentiment_run_id(now: datetime | None = None) -> str:
    """Create a filesystem-safe, sortable sentiment inference run identity."""

    selected = now or datetime.now(timezone.utc)
    if selected.tzinfo is None or selected.utcoffset() is None:
        raise ValueError("sentiment run timestamps must be timezone-aware")
    selected = selected.astimezone(timezone.utc)
    return f"SENT-{selected:%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


def validate_sentiment_run_id(value: str) -> str:
    """Validate the strict public run-ID format before using it as a path."""

    if not _RUN_ID.fullmatch(str(value)):
        raise ValueError(
            "sentiment run_id must match SENT-YYYYMMDD-HHMMSS-xxxxxxxx"
        )
    return str(value)


def json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically and reject non-finite floats."""

    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    """Write a JSON artifact inside a newly-created staging directory."""

    path.write_bytes(json_bytes(value))


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    return value


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    """Write stable UTF-8 CSV without ever logging row content."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_inventory(
    root: Path,
    *,
    expected_files: Sequence[str],
) -> list[dict[str, Any]]:
    """Return an exact, sorted checksum inventory for regular files.

    ``run_manifest.json`` is intentionally not passed here because a file cannot
    contain its own stable checksum.  The contract documents that single
    self-reference exception.
    """

    expected = sorted(str(PurePosixPath(item)) for item in expected_files)
    if any(
        value in {"", "."}
        or PurePosixPath(value).is_absolute()
        or ".." in PurePosixPath(value).parts
        for value in expected
    ):
        raise ValueError("artifact inventory paths must be safe relative paths")
    if len(expected) != len(set(expected)):
        raise ValueError("artifact inventory paths must be unique")

    observed: list[str] = []
    for candidate in sorted(root.iterdir(), key=lambda item: item.name):
        mode = candidate.lstat().st_mode
        if not stat.S_ISREG(mode):
            raise ValueError("sentiment artifacts must be regular files")
        observed.append(candidate.name)
    if observed != expected:
        raise ValueError(
            f"sentiment artifact set mismatch: expected={expected!r}, "
            f"observed={observed!r}"
        )

    return [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in expected
    ]


def publish_staged_run(staging: Path, destination: Path) -> None:
    """Atomically publish a completed staging directory without replacement."""

    if destination.exists() or destination.is_symlink():
        raise SentimentRunCollisionError(
            f"sentiment artifact destination already exists: {destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(staging, destination)
    except FileExistsError as exc:
        raise SentimentRunCollisionError(
            f"sentiment artifact destination already exists: {destination.name}"
        ) from exc


def logical_storage_label(
    path: Path,
    *,
    storage_root: Path,
    kind: str,
) -> str:
    """Return a portable label and never expose an absolute host path."""

    resolved = path.resolve(strict=False)
    trusted = storage_root.resolve(strict=False)
    try:
        return resolved.relative_to(trusted).as_posix()
    except ValueError:
        return f"external-{kind}"


__all__ = [
    "SentimentRunCollisionError",
    "artifact_inventory",
    "json_bytes",
    "logical_storage_label",
    "make_sentiment_run_id",
    "publish_staged_run",
    "sha256_file",
    "utc_now_z",
    "validate_sentiment_run_id",
    "write_csv",
    "write_json",
]
