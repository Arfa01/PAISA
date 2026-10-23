"""Filesystem boundary helpers for immutable evidence artifacts.

Publication code uses these helpers before the atomic no-replace rename.  They
keep the staging tree private, reject repository output locations that could be
committed accidentally, and avoid exposing host paths in validation errors.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
from typing import NoReturn, TypeVar


_Error = TypeVar("_Error", bound=Exception)


def _raise(error_type: type[_Error], message: str) -> NoReturn:
    raise error_type(message)


def _absolute_without_symlink_resolution(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _reject_symlink_components(
    path: Path,
    *,
    boundary: Path,
    error_type: type[_Error],
    label: str,
) -> None:
    """Reject symlinks between a trusted lexical boundary and ``path``."""

    relative = path.relative_to(boundary)
    current = boundary
    for part in relative.parts:
        current = current / part
        if os.path.lexists(current) and current.is_symlink():
            _raise(error_type, f"{label} cannot contain symlink components")


def reject_symlink_components(
    path: str | Path,
    *,
    error_type: type[_Error] = ValueError,
    label: str,
) -> Path:
    """Return an absolute lexical path after rejecting every symlink component."""

    lexical = _absolute_without_symlink_resolution(Path(path))
    anchor = Path(lexical.anchor)
    _reject_symlink_components(
        lexical,
        boundary=anchor,
        error_type=error_type,
        label=label,
    )
    return lexical


def _git_output_policy(
    output_root: Path,
    *,
    repository_root: Path,
    error_type: type[_Error],
    label: str,
) -> None:
    if not (repository_root / ".git").exists():
        return
    relative = output_root.relative_to(repository_root).as_posix()
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "--", relative],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        ignored = subprocess.run(
            ["git", "check-ignore", "--quiet", "--", f"{relative}/"],
            cwd=repository_root,
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        _raise(error_type, f"Could not verify {label.lower()} Git hygiene")
    if tracked.returncode != 0:
        _raise(error_type, f"Could not verify {label.lower()} tracked-file state")
    if tracked.stdout.strip():
        _raise(error_type, f"{label} cannot be a Git-tracked location")
    if ignored.returncode != 0:
        _raise(error_type, f"{label} inside the repository must be Git-ignored")


def prepare_output_root(
    output_root: str | Path,
    *,
    repository_root: str | Path,
    error_type: type[_Error] = ValueError,
    label: str,
) -> Path:
    """Create and validate a real output directory without following root links.

    External output roots remain supported for tests and controlled exports. If
    the lexical output path is inside the configured repository, every existing
    component below that boundary must be link-free, the resolved path must stay
    inside the repository, and Git must report it ignored and untracked.
    """

    candidate = Path(output_root)
    lexical = reject_symlink_components(
        candidate,
        error_type=error_type,
        label=label,
    )
    repository_lexical = _absolute_without_symlink_resolution(Path(repository_root))
    try:
        repository = Path(repository_root).resolve(strict=True)
    except OSError:
        _raise(error_type, "Configured repository root is unavailable")
    if not repository.is_dir():
        _raise(error_type, "Configured repository root must be a directory")

    if os.path.lexists(lexical) and lexical.is_symlink():
        _raise(error_type, f"{label} cannot be a symlink")

    lexically_inside = _is_relative_to(lexical, repository_lexical)
    if lexically_inside:
        unresolved_target = lexical.resolve(strict=False)
        if not _is_relative_to(unresolved_target, repository):
            _raise(error_type, f"{label} resolves outside the configured repository")
        _git_output_policy(
            unresolved_target,
            repository_root=repository,
            error_type=error_type,
            label=label,
        )

    try:
        lexical.mkdir(parents=True, exist_ok=True, mode=0o700)
        metadata = lexical.lstat()
    except OSError:
        _raise(error_type, f"{label} could not be created safely")
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        _raise(error_type, f"{label} must be a real directory")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError:
        _raise(error_type, f"{label} could not be resolved safely")

    if lexically_inside and not _is_relative_to(resolved, repository):
        _raise(error_type, f"{label} resolves outside the configured repository")
    if _is_relative_to(resolved, repository):
        _git_output_policy(
            resolved,
            repository_root=repository,
            error_type=error_type,
            label=label,
        )
    return resolved


def make_pack_private(
    root: Path,
    *,
    error_type: type[_Error],
    label: str,
) -> None:
    """Set exact 0700/0600 modes using no-follow file descriptors."""

    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, directory_flags)
    except OSError:
        _raise(error_type, f"{label} staging root is unavailable")
    try:
        root_metadata = os.fstat(root_fd)
        if not stat.S_ISDIR(root_metadata.st_mode):
            _raise(error_type, f"{label} staging root must be a real directory")
        os.fchmod(root_fd, 0o700)
        for item in root.iterdir():
            metadata = item.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                _raise(error_type, f"{label} packs may contain only regular files")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(item.name, flags, dir_fd=root_fd)
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or (
                    opened.st_dev,
                    opened.st_ino,
                ) != (metadata.st_dev, metadata.st_ino):
                    _raise(error_type, f"{label} artifact identity changed")
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
    except OSError:
        _raise(error_type, f"{label} private permissions could not be applied")
    finally:
        os.close(root_fd)


def verify_pack_modes(
    root: Path,
    *,
    error_type: type[_Error],
    label: str,
) -> None:
    """Require exact private modes for a completed or staged flat pack."""

    try:
        root_mode = root.lstat().st_mode
    except OSError:
        _raise(error_type, f"{label} run root is missing")
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        _raise(error_type, f"{label} run root must be a real directory")
    if stat.S_IMODE(root_mode) != 0o700:
        _raise(error_type, f"{label} run root is not private; mode must be 0700")
    try:
        children = tuple(root.iterdir())
    except OSError:
        _raise(error_type, f"{label} run root is unreadable")
    for item in children:
        try:
            mode = item.lstat().st_mode
        except OSError:
            _raise(error_type, f"{label} artifact is unavailable")
        if not stat.S_ISREG(mode):
            _raise(error_type, f"{label} packs may contain only regular files")
        if stat.S_IMODE(mode) != 0o600:
            _raise(error_type, f"{label} artifact is not private; mode must be 0600")


def remove_staging_tree(staging: Path, *, output_root: Path) -> None:
    """Remove only the selected direct-child staging entry, without following it."""

    if (
        staging.parent != output_root
        or not staging.name.startswith(".")
        or ".staging-" not in staging.name
    ):
        raise RuntimeError("Refusing to remove an unsafe staging target")
    try:
        mode = staging.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        shutil.rmtree(staging)
    else:
        staging.unlink()


__all__ = [
    "make_pack_private",
    "prepare_output_root",
    "reject_symlink_components",
    "remove_staging_tree",
    "verify_pack_modes",
]
