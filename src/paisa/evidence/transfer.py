"""Safe restoration of checksum-bound immutable evidence bundles."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
from typing import Any, BinaryIO, Mapping, Sequence
from uuid import uuid4

from paisa.evidence.artifact_security import (
    prepare_output_root,
    reject_symlink_components,
    verify_pack_modes,
)
from paisa.sentiment.artifacts import (
    SentimentRunCollisionError,
    artifact_inventory,
    publish_staged_run,
    regular_file_inventory,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEALS = frozenset({"run_manifest.json", "completion.json"})
_MAX_SEAL_BYTES = 2 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
_MAX_BUNDLE_BYTES = 1024 * 1024 * 1024
_MAX_BUNDLE_MEMBERS = 4096


class EvidenceBundleRestoreError(ValueError):
    """A transferred evidence bundle is unsafe or does not match its seal."""


@dataclass(frozen=True, slots=True)
class EvidencePackRestoreSpec:
    """Trusted identities required before one immutable pack may be restored."""

    relative_root: str
    manifest_sha256: str
    inventory_sha256: str
    completion_sha256: str
    verifier: str | None = None

    def __post_init__(self) -> None:
        root = PurePosixPath(self.relative_root)
        if (
            root.is_absolute()
            or ".." in root.parts
            or len(root.parts) < 2
            or root.as_posix() != self.relative_root
        ):
            raise ValueError("Evidence restore roots must be safe relative paths")
        for value in (
            self.manifest_sha256,
            self.inventory_sha256,
            self.completion_sha256,
        ):
            if not _SHA256.fullmatch(value):
                raise ValueError("Evidence restore identities must be SHA-256 values")
        if self.verifier not in {None, "evidence", "coverage"}:
            raise ValueError("Unknown evidence restore verifier")


CANONICAL_EVIDENCE_RESTORE_SPECS = (
    EvidencePackRestoreSpec(
        "artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685",
        "15ae19a7d186bd1417545914e3898b23ba80e1f1adb31bb7f8e8018f01f633a4",
        "636b096f9bedaed73756c7175b35676555049251c6a3b14f8ed609e960c50888",
        "7c7c58a7d97e278dbf7ec3411818301ad49509b8ebb48fbdda17c1c769982515",
        "evidence",
    ),
    EvidencePackRestoreSpec(
        "artifacts/evidence_coverage/ECOV-20261023-175904-be50a499",
        "6b35a6c6c7a9a83b20bc93099fbfaff05743e49150ea0dc2cfa3ceb1e6667250",
        "baf14a5d30f60fd7c5fad2d1daa75322007b43e2257e0a1a1ecf498efc23b826",
        "d3a98c57d33c36c1d0fa1a41bedfef13b3c5fa32dcfcdae2bb25bccf9991e265",
        "coverage",
    ),
    EvidencePackRestoreSpec(
        "artifacts/contemporaneous_evidence/EVID-20261023-163905-gkgfinal",
        "a335bcf657ca855e52aebc2a0bcd4721e85434716aa8d6fa13a405cc0d42991e",
        "1675a8902f4c0175972392d04a207f09157d034ec1b118eb4c449e2ac9d77e7e",
        "ba95e027fde05175d189e243268360487295ae01d8d0e3309084ae2feed1d742",
    ),
    EvidencePackRestoreSpec(
        "artifacts/evidence_coverage/ECOV-20261023-164000-final001",
        "d4c31758d39e1a97ee7629e487fe04a801b82ce291923e47392b552f4f99335c",
        "e01f33b278bc780d02e74fa5dc88abbba803f12ffc8e90b95562ffb272010e4d",
        "7a7bcdafe3887b712dd468abd0868c7b1836d2ea92bdc2352fd30e4488f98641",
    ),
)


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_member_name(raw_name: str) -> str:
    if not isinstance(raw_name, str) or not raw_name or "\\" in raw_name:
        raise EvidenceBundleRestoreError("Evidence bundle contains an unsafe member")
    normalized = raw_name[:-1] if raw_name.endswith("/") else raw_name
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != normalized
        or any(part in {"", "."} for part in path.parts)
    ):
        raise EvidenceBundleRestoreError("Evidence bundle contains an unsafe member")
    return normalized


def _member_bytes(archive: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if not member.isreg() or member.size < 0:
        raise EvidenceBundleRestoreError(
            "Evidence bundle expected a regular file member"
        )
    if member.size > _MAX_BUNDLE_BYTES:
        raise EvidenceBundleRestoreError("Evidence bundle member is too large")
    extracted = archive.extractfile(member)
    if extracted is None:
        raise EvidenceBundleRestoreError("Evidence bundle member is unreadable")
    payload = extracted.read(member.size + 1)
    if len(payload) != member.size:
        raise EvidenceBundleRestoreError("Evidence bundle member size differs")
    return payload


def _trusted_file_claims(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    spec: EvidencePackRestoreSpec,
) -> dict[str, tuple[int, str]]:
    manifest_name = f"{spec.relative_root}/run_manifest.json"
    completion_name = f"{spec.relative_root}/completion.json"
    if manifest_name not in members or completion_name not in members:
        raise EvidenceBundleRestoreError("Evidence bundle is missing required seals")
    if members[manifest_name].size > _MAX_SEAL_BYTES:
        raise EvidenceBundleRestoreError("Evidence bundle manifest is too large")
    raw_manifest = _member_bytes(archive, members[manifest_name])
    raw_completion = _member_bytes(archive, members[completion_name])
    if hashlib.sha256(raw_manifest).hexdigest() != spec.manifest_sha256:
        raise EvidenceBundleRestoreError("Evidence bundle manifest identity differs")
    if hashlib.sha256(raw_completion).hexdigest() != spec.completion_sha256:
        raise EvidenceBundleRestoreError("Evidence bundle completion identity differs")
    try:
        manifest = json.loads(raw_manifest)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceBundleRestoreError("Evidence bundle manifest is invalid") from exc
    if not isinstance(manifest, Mapping):
        raise EvidenceBundleRestoreError("Evidence bundle manifest is invalid")
    inventory = manifest.get("payload_artifact_inventory")
    if not isinstance(inventory, list) or manifest.get("inventory_excludes") != [
        "completion.json",
        "run_manifest.json",
    ]:
        raise EvidenceBundleRestoreError("Evidence bundle inventory contract differs")
    if _canonical_json_sha256(inventory) != spec.inventory_sha256:
        raise EvidenceBundleRestoreError("Evidence bundle inventory identity differs")
    if manifest.get("payload_artifact_inventory_sha256") != spec.inventory_sha256:
        raise EvidenceBundleRestoreError("Evidence bundle inventory seal differs")

    claims: dict[str, tuple[int, str]] = {
        "run_manifest.json": (len(raw_manifest), spec.manifest_sha256),
        "completion.json": (len(raw_completion), spec.completion_sha256),
    }
    for item in inventory:
        if not isinstance(item, Mapping):
            raise EvidenceBundleRestoreError("Evidence bundle inventory is invalid")
        name = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if (
            not isinstance(name, str)
            or PurePosixPath(name).name != name
            or not name
            or name in claims
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
        ):
            raise EvidenceBundleRestoreError("Evidence bundle inventory is invalid")
        claims[name] = (size, digest)
    return claims


def _open_archive(path: Path) -> tuple[tarfile.TarFile, BinaryIO]:
    lexical = reject_symlink_components(
        path,
        error_type=EvidenceBundleRestoreError,
        label="Evidence transfer archive",
    )
    try:
        before = lexical.lstat()
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lexical, flags)
        handle = os.fdopen(descriptor, "rb")
        opened = os.fstat(handle.fileno())
    except OSError as exc:
        raise EvidenceBundleRestoreError(
            "Evidence transfer archive is unavailable"
        ) from exc
    if not stat.S_ISREG(before.st_mode) or (
        before.st_dev,
        before.st_ino,
    ) != (opened.st_dev, opened.st_ino):
        handle.close()
        raise EvidenceBundleRestoreError(
            "Evidence transfer archive must be an unchanged regular file"
        )
    if opened.st_size > _MAX_ARCHIVE_BYTES:
        handle.close()
        raise EvidenceBundleRestoreError("Evidence transfer archive is too large")
    try:
        archive = tarfile.open(fileobj=handle, mode="r:gz")
    except (OSError, tarfile.TarError) as exc:
        handle.close()
        raise EvidenceBundleRestoreError(
            "Evidence transfer archive is invalid"
        ) from exc
    return archive, handle


def _inspect_archive(
    archive: tarfile.TarFile,
    specs: Sequence[EvidencePackRestoreSpec],
) -> tuple[dict[str, tarfile.TarInfo], dict[str, dict[str, tuple[int, str]]]]:
    members: dict[str, tarfile.TarInfo] = {}
    total_size = 0
    try:
        archive_members = archive.getmembers()
    except (OSError, tarfile.TarError) as exc:
        raise EvidenceBundleRestoreError(
            "Evidence transfer archive is invalid"
        ) from exc
    if len(archive_members) > _MAX_BUNDLE_MEMBERS:
        raise EvidenceBundleRestoreError("Evidence bundle has too many members")
    for member in archive_members:
        name = _safe_member_name(member.name)
        if name in members:
            raise EvidenceBundleRestoreError(
                "Evidence bundle contains a duplicate member"
            )
        if not (member.isdir() or member.isreg()):
            raise EvidenceBundleRestoreError(
                "Evidence bundle contains a link or special member"
            )
        if member.isreg():
            total_size += member.size
            if total_size > _MAX_BUNDLE_BYTES:
                raise EvidenceBundleRestoreError("Evidence bundle is too large")
        members[name] = member

    claims_by_root = {
        spec.relative_root: _trusted_file_claims(archive, members, spec)
        for spec in specs
    }
    expected = {spec.relative_root for spec in specs}
    for root, claims in claims_by_root.items():
        expected.update(f"{root}/{name}" for name in claims)
    if set(members) != expected:
        raise EvidenceBundleRestoreError(
            "Evidence bundle member allowlist differs"
        )
    for spec in specs:
        if not members[spec.relative_root].isdir():
            raise EvidenceBundleRestoreError(
                "Evidence bundle pack root must be a directory"
            )
        for name, (size, _digest) in claims_by_root[spec.relative_root].items():
            member = members[f"{spec.relative_root}/{name}"]
            if not member.isreg() or member.size != size:
                raise EvidenceBundleRestoreError(
                    "Evidence bundle member size differs from its trusted inventory"
                )
    return members, claims_by_root


def _write_private_file(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def _extract_to_private_staging(
    archive: tarfile.TarFile,
    members: Mapping[str, tarfile.TarInfo],
    claims_by_root: Mapping[str, Mapping[str, tuple[int, str]]],
    stage: Path,
) -> None:
    for relative_root, claims in claims_by_root.items():
        root = stage / relative_root
        root.mkdir(parents=True, mode=0o700)
        for parent in (stage / "artifacts", root.parent, root):
            os.chmod(parent, 0o700)
        for name, (_size, digest) in claims.items():
            payload = _member_bytes(archive, members[f"{relative_root}/{name}"])
            if hashlib.sha256(payload).hexdigest() != digest:
                raise EvidenceBundleRestoreError(
                    "Evidence bundle payload identity differs"
                )
            _write_private_file(root / name, payload)


def _verify_staged_pack(root: Path, spec: EvidencePackRestoreSpec) -> None:
    verify_pack_modes(
        root,
        error_type=EvidenceBundleRestoreError,
        label="Restored evidence",
    )
    raw_manifest = (root / "run_manifest.json").read_bytes()
    raw_completion = (root / "completion.json").read_bytes()
    if hashlib.sha256(raw_manifest).hexdigest() != spec.manifest_sha256:
        raise EvidenceBundleRestoreError("Staged evidence manifest identity differs")
    if hashlib.sha256(raw_completion).hexdigest() != spec.completion_sha256:
        raise EvidenceBundleRestoreError("Staged evidence completion identity differs")
    manifest = json.loads(raw_manifest)
    inventory = manifest["payload_artifact_inventory"]
    expected_files = tuple(item["path"] for item in inventory)
    observed = regular_file_inventory(root, files=expected_files)
    if (
        observed != inventory
        or _canonical_json_sha256(observed) != spec.inventory_sha256
    ):
        raise EvidenceBundleRestoreError("Staged evidence inventory differs")
    observed_names = {item.name for item in root.iterdir()}
    if observed_names != set(expected_files) | _SEALS:
        raise EvidenceBundleRestoreError("Staged evidence file allowlist differs")
    if spec.verifier == "evidence":
        from paisa.evidence.artifacts import verify_evidence_run

        if verify_evidence_run(root).status != "PASS":
            raise EvidenceBundleRestoreError("Staged evidence verification failed")
    elif spec.verifier == "coverage":
        from paisa.evidence.coverage_artifacts import verify_evidence_coverage_pack

        if verify_evidence_coverage_pack(root).status != "PASS":
            raise EvidenceBundleRestoreError(
                "Staged evidence-coverage verification failed"
            )


def _safe_restore_entry(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or len(path.parts) < 2
    ):
        raise ValueError(f"{label} must be a safe repository-relative path")
    return value


def _rename_entry_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one staged file or directory without replacement."""

    if source.parent != destination.parent:
        raise ValueError("Restore staging and destination must share a parent")
    if os.name == "nt":
        os.rename(source, destination)
        return
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(source.parent, flags)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        source_bytes = os.fsencode(source.name)
        destination_bytes = os.fsencode(destination.name)
        if hasattr(libc, "renameat2"):
            operation = libc.renameat2
            operation.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            operation.restype = ctypes.c_int
            result = operation(
                parent_fd,
                source_bytes,
                parent_fd,
                destination_bytes,
                1,  # RENAME_NOREPLACE
            )
        elif hasattr(libc, "renameatx_np"):
            operation = libc.renameatx_np
            operation.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            operation.restype = ctypes.c_int
            result = operation(
                parent_fd,
                source_bytes,
                parent_fd,
                destination_bytes,
                0x00000004,  # RENAME_EXCL
            )
        else:
            raise OSError(
                errno.ENOTSUP,
                "atomic no-replace restore is unavailable on this platform",
            )
        if result != 0:
            error_number = ctypes.get_errno()
            if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(
                    error_number,
                    os.strerror(error_number),
                    destination,
                )
            raise OSError(error_number, os.strerror(error_number), destination)
    finally:
        os.close(parent_fd)


def _remove_restore_entry(path: Path) -> None:
    if not os.path.lexists(path):
        return
    mode = path.lstat().st_mode
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        shutil.rmtree(path)
    else:
        path.unlink()


def _verify_restore_target_git_hygiene(
    destination: Path,
    *,
    repository: Path,
) -> None:
    """Require an in-repository restore target to be ignored and untracked.

    The policy intentionally applies to the exact target rather than its
    parent. Repository-owned parents may contain tracked placeholders (for
    example ``data/news/.gitkeep``) while the restored database itself remains
    excluded from version control.
    """

    if not (repository / ".git").exists():
        return
    try:
        relative = destination.relative_to(repository).as_posix()
        tracked = subprocess.run(
            ["git", "ls-files", "--", relative],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", "--", relative],
            cwd=repository,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise EvidenceBundleRestoreError(
            "Could not verify artifact restore target Git hygiene"
        ) from exc
    if tracked.returncode != 0:
        raise EvidenceBundleRestoreError(
            "Could not verify artifact restore target tracked-file state"
        )
    if tracked.stdout.strip():
        raise EvidenceBundleRestoreError(
            "Artifact restore target cannot be Git-tracked"
        )
    if ignored.returncode not in {0, 1}:
        raise EvidenceBundleRestoreError(
            "Could not verify artifact restore target ignored-file state"
        )
    if ignored.returncode == 1:
        raise EvidenceBundleRestoreError(
            "Artifact restore target inside the repository must be Git-ignored"
        )


def _prepare_restore_parent(repository: Path, relative_parent: str) -> Path:
    """Create a destination parent without following repository-relative links."""

    relative = PurePosixPath(relative_parent)
    if (
        relative.is_absolute()
        or relative.as_posix() != relative_parent
        or ".." in relative.parts
        or any(part in {"", "."} for part in relative.parts)
    ):
        raise EvidenceBundleRestoreError("Artifact restore parent is unsafe")

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        current_fd = os.open(repository, directory_flags)
    except OSError as exc:
        raise EvidenceBundleRestoreError(
            "Artifact restore destination is unavailable"
        ) from exc
    final_identity: tuple[int, int] | None = None
    try:
        for part in relative.parts:
            created = False
            next_fd: int | None = None
            try:
                try:
                    next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                except FileNotFoundError:
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=current_fd)
                        created = True
                    except FileExistsError:
                        # A concurrent creator is acceptable only if the
                        # no-follow descriptor open below proves it made a real
                        # directory.
                        pass
                    next_fd = os.open(part, directory_flags, dir_fd=current_fd)
                if created:
                    os.fchmod(next_fd, 0o700)
                metadata = os.fstat(next_fd)
                if not stat.S_ISDIR(metadata.st_mode):
                    raise EvidenceBundleRestoreError(
                        "Artifact restore parent must be a real directory"
                    )
            except Exception:
                if next_fd is not None:
                    os.close(next_fd)
                raise
            os.close(current_fd)
            current_fd = next_fd
        metadata = os.fstat(current_fd)
        final_identity = (metadata.st_dev, metadata.st_ino)
    except EvidenceBundleRestoreError:
        raise
    except OSError as exc:
        raise EvidenceBundleRestoreError(
            "Artifact restore parent could not be created safely"
        ) from exc
    finally:
        os.close(current_fd)

    parent = repository / relative_parent
    try:
        lexical = reject_symlink_components(
            parent,
            error_type=EvidenceBundleRestoreError,
            label="Artifact restore parent",
        )
        metadata = lexical.lstat()
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(repository)
    except (OSError, ValueError) as exc:
        raise EvidenceBundleRestoreError(
            "Artifact restore parent could not be validated safely"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or (metadata.st_dev, metadata.st_ino) != final_identity
    ):
        raise EvidenceBundleRestoreError(
            "Artifact restore parent identity changed"
        )
    return lexical


def restore_allowlisted_artifact_bundle(
    archive_path: str | Path,
    *,
    destination_root: str | Path,
    allowed_directories: Sequence[str] = (),
    allowed_files: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Restore only explicitly allowed roots through private staging.

    This generic transfer boundary protects dynamically checksummed supervisor
    bundles whose complete per-file inventory is not known at code-release
    time. It rejects archive members outside the named roots, links, special
    files, traversal, duplicates, and replacement of any existing target.
    Artifact-specific semantic verification remains a required follow-up.
    """

    directories = tuple(
        _safe_restore_entry(value, label="Allowed restore directory")
        for value in allowed_directories
    )
    files = tuple(
        _safe_restore_entry(value, label="Allowed restore file")
        for value in allowed_files
    )
    roots = directories + files
    if not roots or len(roots) != len(set(roots)):
        raise ValueError("Allowed restore entries must be non-empty and unique")
    if any(
        first != second
        and (
            first.startswith(f"{second}/")
            or second.startswith(f"{first}/")
        )
        for first in roots
        for second in roots
    ):
        raise ValueError("Allowed restore entries must not overlap")

    repository = reject_symlink_components(
        destination_root,
        error_type=EvidenceBundleRestoreError,
        label="Artifact restore destination",
    )
    if not repository.is_dir():
        raise EvidenceBundleRestoreError(
            "Artifact restore destination must be a real directory"
        )
    destinations = tuple(repository / root for root in roots)
    if any(os.path.lexists(path) for path in destinations):
        raise EvidenceBundleRestoreError(
            "Artifact restore refuses to replace an existing target"
        )
    for destination in destinations:
        _verify_restore_target_git_hygiene(
            destination,
            repository=repository,
        )

    archive, archive_handle = _open_archive(Path(archive_path))
    stage = Path(tempfile.mkdtemp(prefix=".paisa-artifact-restore-", dir=repository))
    os.chmod(stage, 0o700)
    publication_stages: list[Path] = []
    published: list[Path] = []
    try:
        try:
            archive_members = archive.getmembers()
        except (OSError, tarfile.TarError) as exc:
            raise EvidenceBundleRestoreError(
                "Artifact transfer archive is invalid"
            ) from exc
        if len(archive_members) > _MAX_BUNDLE_MEMBERS:
            raise EvidenceBundleRestoreError("Artifact bundle has too many members")
        members: dict[str, tarfile.TarInfo] = {}
        total_size = 0
        for member in archive_members:
            name = _safe_member_name(member.name)
            if name in members:
                raise EvidenceBundleRestoreError(
                    "Artifact bundle contains a duplicate member"
                )
            if not (member.isdir() or member.isreg()):
                raise EvidenceBundleRestoreError(
                    "Artifact bundle contains a link or special member"
                )
            allowed = name in roots or any(
                name.startswith(f"{root}/") for root in directories
            )
            if not allowed:
                raise EvidenceBundleRestoreError(
                    "Artifact bundle member allowlist differs"
                )
            if member.isreg():
                total_size += member.size
                if total_size > _MAX_BUNDLE_BYTES:
                    raise EvidenceBundleRestoreError("Artifact bundle is too large")
            members[name] = member
        if any(
            root not in members or not members[root].isdir()
            for root in directories
        ) or any(
            root not in members or not members[root].isreg()
            for root in files
        ):
            raise EvidenceBundleRestoreError(
                "Artifact bundle is missing an allowed root"
            )

        for name, member in sorted(
            members.items(),
            key=lambda item: (len(PurePosixPath(item[0]).parts), item[0]),
        ):
            destination = stage / name
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=False, mode=0o700)
                os.chmod(destination, 0o700)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                for parent in destination.parents:
                    if parent == stage.parent:
                        break
                    os.chmod(parent, 0o700)
                _write_private_file(destination, _member_bytes(archive, member))

        parent_paths = {
            PurePosixPath(root).parent.as_posix() for root in roots
        }
        for relative in parent_paths:
            _prepare_restore_parent(repository, relative)
        if any(os.path.lexists(path) for path in destinations):
            raise EvidenceBundleRestoreError(
                "Artifact restore refuses to replace an existing target"
            )
        for root, destination in zip(roots, destinations, strict=True):
            parent_relative = PurePosixPath(root).parent.as_posix()
            parent = _prepare_restore_parent(repository, parent_relative)
            _verify_restore_target_git_hygiene(
                destination,
                repository=repository,
            )
            publication_stage = parent / f".{destination.name}.restore-{uuid4().hex}"
            os.rename(stage / root, publication_stage)
            publication_stages.append(publication_stage)
            try:
                _rename_entry_noreplace(publication_stage, destination)
            except (FileExistsError, IsADirectoryError) as exc:
                raise EvidenceBundleRestoreError(
                    "Artifact restore refuses to replace an existing target"
                ) from exc
            published.append(destination)
        return tuple(published)
    finally:
        committed = len(published) == len(roots)
        try:
            archive.close()
        except Exception:
            if not committed:
                raise
        try:
            archive_handle.close()
        except Exception:
            if not committed:
                raise
        for publication_stage in publication_stages:
            if os.path.lexists(publication_stage):
                try:
                    _remove_restore_entry(publication_stage)
                except OSError:
                    if not committed:
                        raise
        if os.path.lexists(stage):
            try:
                shutil.rmtree(stage)
            except OSError:
                if not committed:
                    raise


def restore_evidence_bundle(
    archive_path: str | Path,
    *,
    destination_root: str | Path,
    specs: Sequence[EvidencePackRestoreSpec] = CANONICAL_EVIDENCE_RESTORE_SPECS,
) -> tuple[Path, ...]:
    """Validate in private staging, then atomically publish without replacement."""

    selected = tuple(specs)
    if not selected or len({spec.relative_root for spec in selected}) != len(selected):
        raise ValueError("Evidence restore specs must be non-empty and unique")
    repository = reject_symlink_components(
        destination_root,
        error_type=EvidenceBundleRestoreError,
        label="Evidence restore destination",
    )
    if not repository.is_dir():
        raise EvidenceBundleRestoreError(
            "Evidence restore destination must be a real directory"
        )
    destinations = tuple(repository / spec.relative_root for spec in selected)
    if any(os.path.lexists(destination) for destination in destinations):
        raise EvidenceBundleRestoreError(
            "Evidence restore refuses to replace an existing pack"
        )

    archive, archive_handle = _open_archive(Path(archive_path))
    stage = Path(
        tempfile.mkdtemp(prefix=".paisa-evidence-restore-", dir=repository)
    )
    os.chmod(stage, 0o700)
    publication_stages: list[Path] = []
    published: list[Path] = []
    try:
        members, claims_by_root = _inspect_archive(archive, selected)
        _extract_to_private_staging(
            archive,
            members,
            claims_by_root,
            stage,
        )
        for spec in selected:
            _verify_staged_pack(stage / spec.relative_root, spec)

        parent_by_relative = {
            PurePosixPath(spec.relative_root).parent.as_posix()
            for spec in selected
        }
        parents = {
            relative: prepare_output_root(
                repository / relative,
                repository_root=repository,
                error_type=EvidenceBundleRestoreError,
                label="Evidence restore parent",
            )
            for relative in parent_by_relative
        }
        if any(os.path.lexists(destination) for destination in destinations):
            raise EvidenceBundleRestoreError(
                "Evidence restore refuses to replace an existing pack"
            )
        for spec, destination in zip(selected, destinations, strict=True):
            if os.path.lexists(destination):
                raise EvidenceBundleRestoreError(
                    "Evidence restore refuses to replace an existing pack"
                )
            source = stage / spec.relative_root
            parent_relative = PurePosixPath(spec.relative_root).parent.as_posix()
            parent = parents[parent_relative]
            publication_stage = parent / f".{destination.name}.restore-{uuid4().hex}"
            os.rename(source, publication_stage)
            publication_stages.append(publication_stage)
            expected_inventory = artifact_inventory(
                publication_stage,
                expected_files=tuple(
                    sorted(item.name for item in publication_stage.iterdir())
                ),
            )
            try:
                publish_staged_run(
                    publication_stage,
                    destination,
                    expected_inventory=expected_inventory,
                )
            except SentimentRunCollisionError as exc:
                raise EvidenceBundleRestoreError(
                    "Evidence restore refuses to replace an existing pack"
                ) from exc
            published.append(destination)
        return tuple(published)
    finally:
        # A successfully published immutable pack is never removed on a later
        # cleanup failure; unpublished private staging is safe to discard.
        committed = len(published) == len(selected)
        try:
            archive.close()
        except Exception:
            if not committed:
                raise
        try:
            archive_handle.close()
        except Exception:
            if not committed:
                raise
        for publication_stage in publication_stages:
            if os.path.lexists(publication_stage):
                try:
                    shutil.rmtree(publication_stage)
                except OSError:
                    if not committed:
                        raise
        if os.path.lexists(stage):
            try:
                shutil.rmtree(stage)
            except OSError:
                if not committed:
                    raise


__all__ = [
    "CANONICAL_EVIDENCE_RESTORE_SPECS",
    "EvidenceBundleRestoreError",
    "EvidencePackRestoreSpec",
    "restore_allowlisted_artifact_bundle",
    "restore_evidence_bundle",
]
