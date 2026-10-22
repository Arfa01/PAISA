"""Safe, deterministic helpers for immutable sentiment-run artifacts."""

from __future__ import annotations

import csv
import ctypes
from datetime import datetime, timezone
import errno
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


def _regular_file_record(root_fd: int, relative: str) -> dict[str, Any]:
    """Hash one unchanged regular file without following symbolic links.

    The directory descriptor anchors the lookup, ``O_NOFOLLOW`` protects the
    final component, and the pre/open/post identity checks reject replacement
    or mutation while the seal is being calculated.
    """

    path = PurePosixPath(relative)
    if (
        relative in {"", "."}
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 1
    ):
        raise ValueError("artifact inventory paths must be safe file names")
    before = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("sentiment artifacts must be regular files")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_fd = os.open(relative, flags, dir_fd=root_fd)
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError("sentiment artifacts must not be symbolic links") from exc
        raise
    try:
        opened = os.fstat(file_fd)
        identity = (before.st_dev, before.st_ino, before.st_mode)
        if identity != (opened.st_dev, opened.st_ino, opened.st_mode):
            raise RuntimeError("sentiment artifact changed before it could be sealed")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after_fd = os.fstat(file_fd)
    finally:
        os.close(file_fd)
    after_path = os.stat(relative, dir_fd=root_fd, follow_symlinks=False)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(opened, field) != getattr(after_fd, field) for field in stable_fields) or any(
        getattr(after_fd, field) != getattr(after_path, field) for field in stable_fields
    ):
        raise RuntimeError("sentiment artifact changed while it was being sealed")
    return {
        "path": relative,
        "sha256": digest.hexdigest(),
        "size_bytes": after_fd.st_size,
    }


def regular_file_inventory(
    root: Path,
    *,
    files: Sequence[str],
) -> list[dict[str, Any]]:
    """Seal named regular files relative to an unchanged real directory."""

    root_before = root.lstat()
    if not stat.S_ISDIR(root_before.st_mode):
        raise ValueError("sentiment artifact root must be a real directory")
    root_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    root_flags |= getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(root, root_flags)
    try:
        root_opened = os.fstat(root_fd)
        if (root_before.st_dev, root_before.st_ino) != (
            root_opened.st_dev,
            root_opened.st_ino,
        ):
            raise RuntimeError("sentiment artifact root changed before sealing")
        records = [_regular_file_record(root_fd, relative) for relative in sorted(files)]
        root_after = root.lstat()
        if (root_opened.st_dev, root_opened.st_ino) != (
            root_after.st_dev,
            root_after.st_ino,
        ):
            raise RuntimeError("sentiment artifact root changed while sealing")
        return records
    finally:
        os.close(root_fd)


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

    return regular_file_inventory(root, files=expected)


def _normalize_publication_inventory(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the complete checksum set expected at the publication boundary."""

    if isinstance(inventory, (str, bytes, bytearray)):
        raise TypeError("expected_inventory must be a sequence of mappings")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in inventory:
        if not isinstance(raw_record, Mapping):
            raise TypeError("every expected inventory entry must be a mapping")
        relative = str(raw_record.get("path", ""))
        path = PurePosixPath(relative)
        if (
            relative in {"", "."}
            or path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 1
        ):
            raise ValueError("publication inventory paths must be safe file names")
        if relative in seen:
            raise ValueError("publication inventory paths must be unique")
        seen.add(relative)
        digest = str(raw_record.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("publication inventory hashes must be lowercase SHA-256")
        size = raw_record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                "publication inventory size_bytes must be a non-negative integer"
            )
        records.append({"path": relative, "sha256": digest, "size_bytes": size})
    if not records:
        raise ValueError("expected_inventory cannot be empty")
    return sorted(records, key=lambda item: item["path"])


def _directory_identity(status: os.stat_result) -> tuple[int, int, int]:
    return status.st_dev, status.st_ino, status.st_mode


def _verified_publication_source(
    source_parent_fd: int,
    source_name: str,
    *,
    expected_source_identity: tuple[int, int, int],
    expected_inventory: Sequence[Mapping[str, Any]],
) -> int:
    """Open and re-seal the exact staging tree immediately before publication."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source_name, flags, dir_fd=source_parent_fd)
    except OSError as exc:
        raise RuntimeError(
            "sentiment staging root changed before publication"
        ) from exc
    try:
        opened = os.fstat(source_fd)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _directory_identity(opened) != expected_source_identity
        ):
            raise RuntimeError("sentiment staging root changed before publication")
        expected = _normalize_publication_inventory(expected_inventory)
        observed_names = sorted(os.listdir(source_fd))
        expected_names = [record["path"] for record in expected]
        if observed_names != expected_names:
            raise RuntimeError("sentiment artifact set changed before publication")
        observed = [
            _regular_file_record(source_fd, relative) for relative in expected_names
        ]
        if observed != expected:
            raise RuntimeError("sentiment artifact bytes changed before publication")
        after = os.fstat(source_fd)
        if _directory_identity(after) != expected_source_identity:
            raise RuntimeError("sentiment staging root changed before publication")
        return source_fd
    except BaseException:
        os.close(source_fd)
        raise


def _rename_noreplace(
    source: Path,
    destination: Path,
    *,
    expected_source_identity: tuple[int, int, int],
    expected_inventory: Sequence[Mapping[str, Any]],
) -> None:
    """Revalidate and atomically rename a directory only when destination is absent."""

    if source.parent != destination.parent:
        raise ValueError("staging and destination must share one publication directory")
    if source.name in {"", ".", ".."} or destination.name in {"", ".", ".."}:
        raise ValueError("publication paths must be direct children")

    if os.name == "nt":
        # Windows does not expose the dir_fd operations used below. Re-seal the
        # source by path immediately before os.rename, whose Windows semantics
        # already refuse an existing destination.
        source_status = source.lstat()
        if (
            not stat.S_ISDIR(source_status.st_mode)
            or _directory_identity(source_status) != expected_source_identity
        ):
            raise RuntimeError("sentiment staging root changed before publication")
        expected = _normalize_publication_inventory(expected_inventory)
        observed = artifact_inventory(
            source,
            expected_files=tuple(record["path"] for record in expected),
        )
        if observed != expected:
            raise RuntimeError("sentiment artifact bytes changed before publication")
        source_status = source.lstat()
        if _directory_identity(source_status) != expected_source_identity:
            raise RuntimeError("sentiment staging root changed before publication")
        os.rename(source, destination)
        published = destination.lstat()
        if (
            not stat.S_ISDIR(published.st_mode)
            or _directory_identity(published) != expected_source_identity
        ):
            raise RuntimeError("published sentiment root does not match sealed staging")
        return

    parent_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    parent_fd = os.open(source.parent, parent_flags)
    source_fd: int | None = None
    try:
        source_fd = _verified_publication_source(
            parent_fd,
            source.name,
            expected_source_identity=expected_source_identity,
            expected_inventory=expected_inventory,
        )
        libc = ctypes.CDLL(None, use_errno=True)
        source_bytes = os.fsencode(source.name)
        destination_bytes = os.fsencode(destination.name)
        result: int
        if hasattr(libc, "renameat2"):
            renameat2 = libc.renameat2
            renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameat2.restype = ctypes.c_int
            result = renameat2(
                parent_fd,
                source_bytes,
                parent_fd,
                destination_bytes,
                1,  # RENAME_NOREPLACE
            )
        elif hasattr(libc, "renameatx_np"):
            renameatx_np = libc.renameatx_np
            renameatx_np.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            renameatx_np.restype = ctypes.c_int
            result = renameatx_np(
                parent_fd,
                source_bytes,
                parent_fd,
                destination_bytes,
                0x00000004,  # RENAME_EXCL
            )
        else:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace publication is unavailable on this platform",
            )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(
                    error_number, os.strerror(error_number), destination
                )
            raise OSError(error_number, os.strerror(error_number), destination)

        published = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(published.st_mode)
            or _directory_identity(published) != expected_source_identity
            or _directory_identity(os.fstat(source_fd)) != expected_source_identity
        ):
            raise RuntimeError("published sentiment root does not match sealed staging")
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)


def publish_staged_run(
    staging: Path,
    destination: Path,
    *,
    expected_inventory: Sequence[Mapping[str, Any]],
) -> None:
    """Re-seal and atomically publish a completed directory without replacement."""

    if destination.exists() or destination.is_symlink():
        raise SentimentRunCollisionError(
            f"sentiment artifact destination already exists: {destination.name}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging_status = staging.lstat()
    except OSError as exc:
        raise RuntimeError("sentiment staging root is unavailable") from exc
    if not stat.S_ISDIR(staging_status.st_mode):
        raise ValueError("sentiment staging root must be a real directory")
    normalized_inventory = _normalize_publication_inventory(expected_inventory)
    try:
        _rename_noreplace(
            staging,
            destination,
            expected_source_identity=_directory_identity(staging_status),
            expected_inventory=normalized_inventory,
        )
    except (FileExistsError, IsADirectoryError) as exc:
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
    "regular_file_inventory",
    "sha256_file",
    "utc_now_z",
    "validate_sentiment_run_id",
    "write_csv",
    "write_json",
]
