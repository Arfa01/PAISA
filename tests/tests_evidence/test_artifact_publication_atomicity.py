from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from paisa.evidence.artifact_security import prepare_output_root
from paisa.evidence.artifacts import (
    EvidenceRunCollisionError,
    EvidenceVerificationError,
    _records_csv_bytes,
    publish_evidence_run,
    verify_evidence_run,
)
from paisa.evidence.models import EvidenceFetchResult, EvidenceRecord
from paisa.evidence.policy import load_evidence_permission_policy
from paisa.evidence.psx_announcements import PsxAnnouncementManualImporter


UTC = timezone.utc
POLICY_ID = "test-manual-artifact-policy"
RUN_ID = "EVID-20261023-170000-a70mic01"
ATTESTATION = "Authorized metadata-only test import."
AUTHORIZATION_REFERENCE = "Test permission record PSX-ATOMIC-001."


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _manual_policy(root: Path):
    assessment = root / "assessment.md"
    assessment.write_text("reviewed manual test source\n", encoding="utf-8")
    policy = root / "policy.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "policy_id": POLICY_ID,
                "assessment_references": [
                    {
                        "path": "assessment.md",
                        "sha256": _digest(assessment.read_bytes()),
                    }
                ],
                "decisions": [
                    {
                        "provider_id": "psx-announcement",
                        "provider_mode": "manual_import",
                        "status": "MANUAL_ONLY",
                        "automated_access": False,
                        "decision_id": "test-manual-permission",
                        "authorization_attestation_sha256": _digest(
                            ATTESTATION.encode("utf-8")
                        ),
                        "authorization_reference_sha256": _digest(
                            AUTHORIZATION_REFERENCE.encode("utf-8")
                        ),
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return load_evidence_permission_policy(policy, repository_root=root)


def _manual_result(
    root: Path,
    *,
    permission_policy,
    duplicate: bool = False,
) -> EvidenceFetchResult:
    source = root / "authorized-psx-input.csv"
    source.write_text(
        "announcement_id,ticker,company_name,announcement_type,title,"
        "exchange_posted_at,official_source_url,pdf_url,pdf_sha256,language\n"
        "PSX-ATOMIC-001,OGDC,Oil and Gas Development Company Limited,"
        "Board Meeting,Board meeting,2026-09-14T14:45:00+05:00,"
        "https://dps.psx.com.pk/company/OGDC,,,en\n",
        encoding="utf-8",
    )
    imported = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    result = PsxAnnouncementManualImporter(
        source,
        permission_policy=permission_policy,
        authorization_attestation=ATTESTATION,
        authorization_reference=AUTHORIZATION_REFERENCE,
    ).fetch(fetched_at=imported)
    result = replace(
        result,
        warnings=(
            *result.warnings,
            "operator input /Users/example/private/input.csv was retained",
            "failed file:///Users/example/private/secret.csv",
            "failed file:/Users/example/private/single-slash.csv",
            "failed file://server/private/network.csv",
            "failed //server/share/private.csv",
            r"failed \\server\share\private.csv",
            (
                "https://www.investegate.co.uk/announcement/rns/"
                "orcadian-energy--orca/-500-000-equity-financing-tvr-/7947964"
            ),
        ),
    )
    if duplicate:
        return replace(
            result,
            records=(result.records[0], result.records[0]),
            counters={**result.counters, "accepted_records": 2},
        )
    return result


def _publish_manual(root: Path, *, duplicate: bool = False) -> Path:
    permission_policy = _manual_policy(root)
    return publish_evidence_run(
        _manual_result(
            root,
            permission_policy=permission_policy,
            duplicate=duplicate,
        ),
        output_root=root / "evidence",
        permission_policy=permission_policy,
        repository_root=root,
        run_id=RUN_ID,
        created_at=datetime(2026, 10, 23, 17, 0, tzinfo=UTC),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _reseal_pack(pack: Path) -> None:
    manifest_path = pack / "run_manifest.json"
    completion_path = pack / "completion.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    inventory = manifest["payload_artifact_inventory"]
    for item in inventory:
        payload_path = pack / item["path"]
        item["sha256"] = _digest(payload_path.read_bytes())
        item["size_bytes"] = payload_path.stat().st_size
    inventory.sort(key=lambda item: item["path"])
    inventory_sha = _digest(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )
    manifest["payload_artifact_inventory_sha256"] = inventory_sha
    manifest["provider_config_sha256"] = _digest(
        (pack / "provider_config.json").read_bytes()
    )
    manifest["normalized_evidence_sha256"] = _digest(
        (pack / "normalized_evidence.jsonl").read_bytes()
    )
    _write_json(manifest_path, manifest)
    completion = json.loads(completion_path.read_text("utf-8"))
    completion["manifest_sha256"] = _digest(manifest_path.read_bytes())
    completion["manifest_size_bytes"] = manifest_path.stat().st_size
    completion["summary_sha256"] = _digest(
        (pack / "final_summary.json").read_bytes()
    )
    completion["summary_size_bytes"] = (
        pack / "final_summary.json"
    ).stat().st_size
    completion["payload_artifact_inventory_sha256"] = inventory_sha
    _write_json(completion_path, completion)


def test_duplicate_id_failure_is_unpublished_and_run_id_is_reusable(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "evidence" / RUN_ID

    with pytest.raises(EvidenceVerificationError, match="unique"):
        _publish_manual(tmp_path, duplicate=True)

    assert not os.path.lexists(destination)
    assert not tuple((tmp_path / "evidence").glob(f".{RUN_ID}.staging-*"))

    published = _publish_manual(tmp_path)
    assert published == destination
    assert verify_evidence_run(published).record_count == 1
    retained = b"".join(item.read_bytes() for item in published.iterdir())
    assert b"/Users/example/private" not in retained
    assert b"file:" not in retained
    assert b"//server/share" not in retained
    assert b"\\\\server\\share" not in retained
    assert b"equity-financing-tvr-/7947964" in retained


def test_copied_evidence_pack_with_public_modes_is_rejected(tmp_path: Path) -> None:
    published = _publish_manual(tmp_path)
    copied = tmp_path / "copied" / RUN_ID
    copied.parent.mkdir()
    shutil.copytree(published, copied)

    os.chmod(copied, 0o755)
    with pytest.raises(EvidenceVerificationError, match="0700"):
        verify_evidence_run(copied)

    os.chmod(copied, 0o700)
    os.chmod(copied / "quality_report.json", 0o644)
    with pytest.raises(EvidenceVerificationError, match="0600"):
        verify_evidence_run(copied)


def test_verifier_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    published = _publish_manual(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(published.parent, target_is_directory=True)

    with pytest.raises(EvidenceVerificationError, match="symlink components"):
        verify_evidence_run(alias / published.name)


def test_existing_final_pack_is_never_altered(tmp_path: Path) -> None:
    published = _publish_manual(tmp_path)
    before = {
        item.name: (item.read_bytes(), item.stat().st_mode)
        for item in published.iterdir()
    }

    with pytest.raises(EvidenceRunCollisionError):
        _publish_manual(tmp_path, duplicate=True)

    after = {
        item.name: (item.read_bytes(), item.stat().st_mode)
        for item in published.iterdir()
    }
    assert after == before
    assert verify_evidence_run(published).record_count == 1


def test_staging_rejects_coherently_forged_psx_permission_claims(
    tmp_path: Path,
) -> None:
    permission_policy = _manual_policy(tmp_path)
    result = _manual_result(tmp_path, permission_policy=permission_policy)
    forged_attestation = "a" * 64
    forged_reference = "b" * 64
    source = dict(result.configured_sources[0])
    source.update(
        {
            "permission_decision_id": "forged-decision",
            "manual_permission_attestation_sha256": forged_attestation,
            "manual_permission_reference_sha256": forged_reference,
        }
    )
    metadata = dict(result.records[0].source_metadata)
    metadata.update(
        {
            "permission_decision_id": "forged-decision",
            "manual_permission_attestation_sha256": forged_attestation,
            "manual_permission_reference_sha256": forged_reference,
        }
    )
    forged = replace(
        result,
        configured_sources=(source,),
        records=(replace(result.records[0], source_metadata=metadata),),
        effective_config={
            **result.effective_config,
            "manual_permission_attestation_sha256": forged_attestation,
            "manual_permission_reference_sha256": forged_reference,
        },
    )

    with pytest.raises(
        EvidenceVerificationError,
        match="permission decision differs from the verified policy",
    ):
        publish_evidence_run(
            forged,
            output_root=tmp_path / "evidence",
            permission_policy=permission_policy,
            repository_root=tmp_path,
            run_id="EVID-20261023-170001-f0a6ed01",
        )

    assert not (tmp_path / "evidence" / "EVID-20261023-170001-f0a6ed01").exists()


def test_portable_verifier_rejects_resealed_psx_precision_rewrite(
    tmp_path: Path,
) -> None:
    published = _publish_manual(tmp_path)
    jsonl_path = published / "normalized_evidence.jsonl"
    payload = json.loads(jsonl_path.read_text("utf-8"))
    payload["time_precision"] = "minute"
    payload["source_metadata"]["exchange_posted_at_raw"] = (
        "2026-09-14T14:45+05:00"
    )
    record = EvidenceRecord.from_dict(payload)
    jsonl_path.write_text(
        json.dumps(
            record.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (published / "normalized_evidence.csv").write_bytes(
        _records_csv_bytes((record,))
    )
    _reseal_pack(published)

    with pytest.raises(
        EvidenceVerificationError,
        match="differs from PSX raw replay",
    ):
        verify_evidence_run(published)


def test_output_root_rejects_existing_and_dangling_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_output_root(
            linked,
            repository_root=tmp_path,
            label="Evidence output root",
        )

    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        prepare_output_root(
            dangling,
            repository_root=tmp_path,
            label="Evidence output root",
        )

    external_repository = tmp_path / "repository"
    external_repository.mkdir()
    external_real = tmp_path / "external-real"
    external_real.mkdir()
    external_alias = tmp_path / "external-alias"
    external_alias.symlink_to(external_real, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink components"):
        prepare_output_root(
            external_alias / "evidence",
            repository_root=external_repository,
            label="Evidence output root",
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_output_root_rejects_windows_junction_components(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    target = tmp_path / "junction-target"
    target.mkdir()
    junction = tmp_path / "junction"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        pytest.skip("This Windows environment cannot create directory junctions")
    try:
        with pytest.raises(ValueError, match="symlink components"):
            prepare_output_root(
                junction / "evidence",
                repository_root=repository,
                label="Evidence output root",
            )
    finally:
        os.rmdir(junction)


def test_repository_output_case_alias_cannot_bypass_git_hygiene(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "case-sensitive-name"
    repository.mkdir()
    alias = repository.with_name(repository.name.upper())
    if alias == repository or not alias.exists() or not alias.samefile(repository):
        pytest.skip("The temporary filesystem is case-sensitive")

    subprocess.run(
        ["git", "init", "--quiet"], cwd=repository, check=True, timeout=10
    )
    tracked = repository / "tracked"
    tracked.mkdir()
    (tracked / ".keep").write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked/.keep"], cwd=repository, check=True, timeout=10
    )

    with pytest.raises(ValueError, match="Git-tracked"):
        prepare_output_root(
            alias / "tracked",
            repository_root=repository,
            label="Evidence output root",
        )

    with pytest.raises(ValueError, match="Git-ignored"):
        prepare_output_root(
            alias / "unignored",
            repository_root=repository,
            label="Evidence output root",
        )
    assert not (repository / "unignored").exists()


def test_repository_output_must_be_ignored_untracked_and_link_free(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet"], cwd=repository, check=True, timeout=10
    )

    with pytest.raises(ValueError, match="Git-ignored"):
        prepare_output_root(
            repository / "unignored",
            repository_root=repository,
            label="Evidence output root",
        )
    assert not (repository / "unignored").exists()

    # A rule crafted to match only the old synthetic probe must not make an
    # otherwise unignored artifact root appear safe.
    (repository / ".gitignore").write_text(
        "/probe-only/.paisa-artifact-ignore-probe\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Git-ignored"):
        prepare_output_root(
            repository / "probe-only",
            repository_root=repository,
            label="Evidence output root",
        )
    assert not (repository / "probe-only").exists()

    (repository / ".gitignore").write_text("/safe/\n/tracked/\n", encoding="utf-8")
    safe = prepare_output_root(
        repository / "safe",
        repository_root=repository,
        label="Evidence output root",
    )
    assert safe == (repository / "safe").resolve()

    tracked = repository / "tracked"
    tracked.mkdir()
    (tracked / ".keep").write_text("tracked\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", "tracked/.keep"],
        cwd=repository,
        check=True,
        timeout=10,
    )
    with pytest.raises(ValueError, match="Git-tracked"):
        prepare_output_root(
            tracked,
            repository_root=repository,
            label="Evidence output root",
        )

    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink components"):
        prepare_output_root(
            repository / "escape" / "evidence",
            repository_root=repository,
            label="Evidence output root",
        )
