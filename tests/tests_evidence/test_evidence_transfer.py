from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile

import pytest

from paisa.evidence import transfer as transfer_module
from paisa.evidence.transfer import (
    EvidenceBundleRestoreError,
    EvidencePackRestoreSpec,
    restore_allowlisted_artifact_bundle,
    restore_evidence_bundle,
)


ROOT = "artifacts/contemporaneous_evidence/EVID-20261023-120000-test0001"


def _fixture(
    root: str = ROOT,
    *,
    payload: bytes = b"bounded evidence\n",
) -> tuple[EvidencePackRestoreSpec, dict[str, bytes]]:
    inventory = [
        {
            "path": "payload.bin",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    ]
    inventory_sha256 = hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest = (
        json.dumps(
            {
                "inventory_excludes": ["completion.json", "run_manifest.json"],
                "payload_artifact_inventory": inventory,
                "payload_artifact_inventory_sha256": inventory_sha256,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")
    completion = b'{"status":"PASS"}\n'
    spec = EvidencePackRestoreSpec(
        root,
        hashlib.sha256(manifest).hexdigest(),
        inventory_sha256,
        hashlib.sha256(completion).hexdigest(),
    )
    return spec, {
        f"{root}/payload.bin": payload,
        f"{root}/run_manifest.json": manifest,
        f"{root}/completion.json": completion,
    }


def _write_archive(
    path: Path,
    files: dict[str, bytes],
    *,
    additions: tuple[tarfile.TarInfo, ...] = (),
    duplicate: str | None = None,
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo(ROOT)
        root.type = tarfile.DIRTYPE
        root.mode = 0o777
        archive.addfile(root)
        for name, payload in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(payload))
            if name == duplicate:
                archive.addfile(member, io.BytesIO(payload))
        for member in additions:
            payload = b"unexpected" if member.isreg() else None
            if payload is not None:
                member.size = len(payload)
            archive.addfile(
                member,
                io.BytesIO(payload) if payload is not None else None,
            )


def _write_multi_archive(
    path: Path,
    fixtures: tuple[tuple[EvidencePackRestoreSpec, dict[str, bytes]], ...],
) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for spec, files in fixtures:
            root = tarfile.TarInfo(spec.relative_root)
            root.type = tarfile.DIRTYPE
            root.mode = 0o777
            archive.addfile(root)
            for name, payload in files.items():
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(payload))


def test_restore_validates_private_staging_and_never_replaces_a_pack(
    tmp_path: Path,
) -> None:
    spec, files = _fixture()
    archive = tmp_path / "evidence.tar.gz"
    destination = tmp_path / "checkout"
    destination.mkdir()
    _write_archive(archive, files)

    restored = restore_evidence_bundle(
        archive,
        destination_root=destination,
        specs=(spec,),
    )

    root = destination / ROOT
    assert restored == (root,)
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert {item.name for item in root.iterdir()} == {
        "payload.bin",
        "run_manifest.json",
        "completion.json",
    }
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o600 for item in root.iterdir())
    before = (root / "payload.bin").read_bytes()
    with pytest.raises(EvidenceBundleRestoreError, match="refuses to replace"):
        restore_evidence_bundle(
            archive,
            destination_root=destination,
            specs=(spec,),
        )
    assert (root / "payload.bin").read_bytes() == before


def test_restore_successfully_validates_and_publishes_four_independent_packs(
    tmp_path: Path,
) -> None:
    roots = (
        "artifacts/contemporaneous_evidence/EVID-20261023-120000-pack0001",
        "artifacts/evidence_coverage/ECOV-20261023-120001-pack0002",
        "artifacts/contemporaneous_evidence/EVID-20261023-120002-pack0003",
        "artifacts/evidence_coverage/ECOV-20261023-120003-pack0004",
    )
    fixtures = tuple(
        _fixture(root, payload=f"payload-{index}\n".encode("ascii"))
        for index, root in enumerate(roots, start=1)
    )
    archive = tmp_path / "four-pack.tar.gz"
    destination = tmp_path / "checkout"
    destination.mkdir()
    _write_multi_archive(archive, fixtures)

    restored = restore_evidence_bundle(
        archive,
        destination_root=destination,
        specs=tuple(spec for spec, _files in fixtures),
    )

    assert restored == tuple(destination / root for root in roots)
    for root in restored:
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(item.stat().st_mode) == 0o600 for item in root.iterdir()
        )


def test_concurrent_later_collision_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    roots = (
        "artifacts/contemporaneous_evidence/EVID-20261023-121000-race0001",
        "artifacts/evidence_coverage/ECOV-20261023-121001-race0002",
        "artifacts/contemporaneous_evidence/EVID-20261023-121002-race0003",
        "artifacts/evidence_coverage/ECOV-20261023-121003-race0004",
    )
    fixtures = tuple(
        _fixture(root, payload=f"payload-{index}\n".encode("ascii"))
        for index, root in enumerate(roots, start=1)
    )
    archive = tmp_path / "racing-four-pack.tar.gz"
    destination = tmp_path / "checkout"
    destination.mkdir()
    _write_multi_archive(archive, fixtures)
    original_publish = transfer_module.publish_staged_run
    calls = 0

    def publish_with_collision(source, target, *, expected_inventory):
        nonlocal calls
        calls += 1
        if calls == 2:
            target.mkdir(parents=True)
            (target / "owner.txt").write_text("existing\n", encoding="utf-8")
        return original_publish(
            source,
            target,
            expected_inventory=expected_inventory,
        )

    monkeypatch.setattr(
        transfer_module,
        "publish_staged_run",
        publish_with_collision,
    )

    with pytest.raises(EvidenceBundleRestoreError, match="refuses to replace"):
        restore_evidence_bundle(
            archive,
            destination_root=destination,
            specs=tuple(spec for spec, _files in fixtures),
        )

    assert (destination / roots[0] / "payload.bin").is_file()
    assert (destination / roots[1] / "owner.txt").read_text(encoding="utf-8") == (
        "existing\n"
    )
    assert not (destination / roots[1] / "payload.bin").exists()
    assert not os.path.lexists(destination / roots[2])
    assert not os.path.lexists(destination / roots[3])


def _extra_member(kind: str) -> tarfile.TarInfo:
    names = {
        "unexpected": "README.md",
        "traversal": "../outside.txt",
        "absolute": "/private/tmp/paisa-outside.txt",
        "symlink": f"{ROOT}/unsafe-link",
        "hardlink": f"{ROOT}/unsafe-hardlink",
        "fifo": f"{ROOT}/unsafe-fifo",
    }
    member = tarfile.TarInfo(names[kind])
    if kind == "symlink":
        member.type = tarfile.SYMTYPE
        member.linkname = "payload.bin"
    elif kind == "hardlink":
        member.type = tarfile.LNKTYPE
        member.linkname = f"{ROOT}/payload.bin"
    elif kind == "fifo":
        member.type = tarfile.FIFOTYPE
    return member


@pytest.mark.parametrize(
    "kind",
    ("unexpected", "traversal", "absolute", "symlink", "hardlink", "fifo"),
)
def test_restore_rejects_unexpected_traversal_link_and_special_members(
    tmp_path: Path,
    kind: str,
) -> None:
    spec, files = _fixture()
    archive = tmp_path / f"{kind}.tar.gz"
    destination = tmp_path / "checkout"
    destination.mkdir()
    _write_archive(archive, files, additions=(_extra_member(kind),))

    with pytest.raises(EvidenceBundleRestoreError):
        restore_evidence_bundle(
            archive,
            destination_root=destination,
            specs=(spec,),
        )

    assert not os.path.lexists(destination / ROOT)
    assert not (tmp_path / "outside.txt").exists()
    assert not any(
        item.name.startswith(".paisa-evidence-restore-")
        for item in destination.iterdir()
    )


def test_restore_rejects_duplicate_archive_members(tmp_path: Path) -> None:
    spec, files = _fixture()
    archive = tmp_path / "duplicate.tar.gz"
    destination = tmp_path / "checkout"
    destination.mkdir()
    _write_archive(archive, files, duplicate=f"{ROOT}/payload.bin")

    with pytest.raises(EvidenceBundleRestoreError, match="duplicate"):
        restore_evidence_bundle(
            archive,
            destination_root=destination,
            specs=(spec,),
        )

    assert not os.path.lexists(destination / ROOT)


def test_restore_rejects_tamper_before_publication_and_cleans_staging(
    tmp_path: Path,
) -> None:
    spec, files = _fixture()
    files[f"{ROOT}/payload.bin"] = b"tampered evidence"
    archive = tmp_path / "tampered.tar.gz"
    destination = tmp_path / "checkout"
    destination.mkdir()
    _write_archive(archive, files)

    with pytest.raises(EvidenceBundleRestoreError):
        restore_evidence_bundle(
            archive,
            destination_root=destination,
            specs=(spec,),
        )

    assert not os.path.lexists(destination / ROOT)
    assert not any(
        item.name.startswith(".paisa-evidence-restore-")
        for item in destination.iterdir()
    )


def test_allowlisted_restore_uses_private_staging_and_rejects_extras(
    tmp_path: Path,
) -> None:
    spec, files = _fixture()
    archive = tmp_path / "allowlisted.tar.gz"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _write_archive(archive, files)

    restored = restore_allowlisted_artifact_bundle(
        archive,
        destination_root=checkout,
        allowed_directories=(spec.relative_root,),
    )

    target = checkout / spec.relative_root
    assert restored == (target,)
    assert (target / "payload.bin").read_bytes() == b"bounded evidence\n"
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in target.iterdir()
    )

    malicious = tmp_path / "allowlisted-extra.tar.gz"
    _write_archive(
        malicious,
        files,
        additions=(_extra_member("unexpected"),),
    )
    clean_checkout = tmp_path / "clean-checkout"
    clean_checkout.mkdir()
    with pytest.raises(EvidenceBundleRestoreError, match="allowlist"):
        restore_allowlisted_artifact_bundle(
            malicious,
            destination_root=clean_checkout,
            allowed_directories=(spec.relative_root,),
        )
    assert not os.path.lexists(clean_checkout / spec.relative_root)


def test_allowlisted_restore_supports_an_exact_regular_file(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "database.tar.gz"
    relative = "data/news/paisa_news.sqlite3"
    payload = b"SQLite format 3\x00bounded fixture"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(relative)
        member.size = len(payload)
        member.mode = 0o644
        bundle.addfile(member, io.BytesIO(payload))
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    restored = restore_allowlisted_artifact_bundle(
        archive,
        destination_root=checkout,
        allowed_files=(relative,),
    )

    target = checkout / relative
    assert restored == (target,)
    assert target.read_bytes() == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_allowlisted_restore_allows_tracked_parent_with_ignored_file_target(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "database.tar.gz"
    relative = "data/news/paisa_news.sqlite3"
    payload = b"SQLite format 3\x00bounded fixture"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(relative)
        member.size = len(payload)
        member.mode = 0o644
        bundle.addfile(member, io.BytesIO(payload))

    checkout = tmp_path / "checkout"
    news = checkout / "data" / "news"
    news.mkdir(parents=True)
    (news / ".gitkeep").write_text("", encoding="utf-8")
    (checkout / ".gitignore").write_text(
        "/data/news/*.sqlite3\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", ".gitignore", "data/news/.gitkeep"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )

    restored = restore_allowlisted_artifact_bundle(
        archive,
        destination_root=checkout,
        allowed_files=(relative,),
    )

    target = checkout / relative
    assert restored == (target,)
    assert target.read_bytes() == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    tracked_parent = subprocess.run(
        ["git", "ls-files", "--", "data/news/.gitkeep"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    assert tracked_parent.stdout.strip() == "data/news/.gitkeep"
    assert subprocess.run(
        ["git", "check-ignore", "--quiet", "--", relative],
        cwd=checkout,
        check=False,
    ).returncode == 0


def test_allowlisted_restore_rejects_unignored_repository_target(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "database.tar.gz"
    relative = "data/news/paisa_news.sqlite3"
    payload = b"SQLite format 3\x00bounded fixture"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(relative)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )

    with pytest.raises(EvidenceBundleRestoreError, match="Git-ignored"):
        restore_allowlisted_artifact_bundle(
            archive,
            destination_root=checkout,
            allowed_files=(relative,),
        )

    assert not os.path.lexists(checkout / relative)


def test_allowlisted_restore_rejects_deleted_but_tracked_target(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "database.tar.gz"
    relative = "data/news/paisa_news.sqlite3"
    payload = b"SQLite format 3\x00bounded fixture"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(relative)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    checkout = tmp_path / "checkout"
    target = checkout / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(b"tracked placeholder")
    (checkout / ".gitignore").write_text(
        "/data/news/*.sqlite3\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "init", "--quiet"],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "-f", relative],
        cwd=checkout,
        check=True,
        capture_output=True,
    )
    target.unlink()

    with pytest.raises(EvidenceBundleRestoreError, match="Git-tracked"):
        restore_allowlisted_artifact_bundle(
            archive,
            destination_root=checkout,
            allowed_files=(relative,),
        )

    assert not os.path.lexists(target)


def test_allowlisted_restore_rejects_an_intermediate_parent_symlink(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "database.tar.gz"
    relative = "data/news/paisa_news.sqlite3"
    payload = b"SQLite format 3\x00bounded fixture"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo(relative)
        member.size = len(payload)
        bundle.addfile(member, io.BytesIO(payload))
    checkout = tmp_path / "checkout"
    data = checkout / "data"
    outside = tmp_path / "outside"
    data.mkdir(parents=True)
    outside.mkdir()
    (data / "news").symlink_to(outside, target_is_directory=True)

    with pytest.raises(EvidenceBundleRestoreError):
        restore_allowlisted_artifact_bundle(
            archive,
            destination_root=checkout,
            allowed_files=(relative,),
        )

    assert not (outside / "paisa_news.sqlite3").exists()
