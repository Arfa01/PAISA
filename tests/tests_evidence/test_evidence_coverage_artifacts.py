from __future__ import annotations

import csv
from datetime import date, datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import pytest

from paisa.evidence.artifacts import sha256_file
from paisa.evidence.coverage import compute_evidence_coverage
from paisa.evidence.coverage_artifacts import (
    EvidenceCoverageRunCollisionError,
    EvidenceCoverageVerificationError,
    publish_evidence_coverage_run,
    verify_evidence_coverage_pack,
)
from paisa.evidence.models import EvidenceRecord
from paisa.news.coverage import CoverageConfig, StockEvidence
from paisa.sentiment.artifacts import regular_file_inventory


UTC = timezone.utc
RUN_ID = "ECOV-20261023-120000-deadbeef"
PAYLOAD_FILES = (
    "evidence_session_mappings.csv",
    "final_summary.json",
    "input_evidence_reference.json",
    "missing_intervals.csv",
    "monthly_coverage.csv",
    "quality_report.json",
    "split_coverage.csv",
    "stock_artifact_reference.json",
)
ALL_FILES = set(PAYLOAD_FILES) | {"run_manifest.json", "completion.json"}


def _at(day: str, clock: str = "09:00:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+00:00")


def _stock() -> StockEvidence:
    dates = tuple(
        date.fromisoformat(value)
        for value in (
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        )
    )
    return StockEvidence(
        symbol="OGDC",
        trading_dates=dates,
        split_dates={
            "train": dates[:2],
            "validation": dates[2:4],
            "test": dates[4:],
        },
        reference={
            "symbol": "OGDC",
            "vertical_slice_run_id": "VSL-20261020-OGDC-test-01",
            "vertical_manifest_sha256": "1" * 64,
            "price_artifact_reference": (
                "vertical_slice/VSL-20261020-OGDC-test-01/"
                "input_snapshot/psx_prices.csv"
            ),
            "price_artifact_sha256": "2" * 64,
            "price_date_range": {
                "start": "2024-01-01",
                "end": "2024-01-08",
            },
            "price_trading_day_count": 6,
            "model_training_run_id": "TRN-20261020-test0001",
            "training_manifest_sha256": "3" * 64,
            "training_data_manifest_sha256": "4" * 64,
            "training_input_snapshot_sha256": "2" * 64,
            "split_manifest_sha256": "5" * 64,
            "split_method": "chronological_target_date_70_15_15_no_shuffle",
            "split_counts": {"train": 2, "validation": 2, "test": 2},
            "split_ranges": {
                "train": {"start": "2024-01-01", "end": "2024-01-02"},
                "validation": {
                    "start": "2024-01-03",
                    "end": "2024-01-04",
                },
                "test": {"start": "2024-01-05", "end": "2024-01-08"},
            },
            "acquisition_mode": "test_verified_input",
            "provider": "test-provider",
            "is_real_data": True,
        },
    )


def _permission_reference() -> dict[str, object]:
    return {
        "policy_id": "test-evidence-policy",
        "schema_version": "1.0",
        "path": "configs/evidence/test-policy.json",
        "sha256": "6" * 64,
        "assessment_references": [
            {"path": "docs/test-assessment.md", "sha256": "7" * 64}
        ],
    }


def _input_references(
    *, include_sensitive_extras: bool = False, gdelt_record_count: int = 3
):
    gdelt = {
        "input_kind": "gdelt_gkg",
        "run_id": "EVID-20261023-110000-feedface",
        "provider_id": "gdelt-gkg-2.1",
        "provider_mode": "raw_archive",
        "data_class": "real",
        "status": "PASS",
        "record_count": gdelt_record_count,
        "capture_count": 1,
        "manifest_sha256": "8" * 64,
        "payload_inventory_sha256": "9" * 64,
        "normalized_evidence_sha256": "a" * 64,
        "artifact_reference": "evidence/EVID-20261023-110000-feedface",
        "permission_policy_reference": _permission_reference(),
    }
    if include_sensitive_extras:
        gdelt.update(
            {
                "path": "/Users/example/private/evidence",
                "source_url": (
                    "https://example.com/archive?token=secret-value&part=1"
                ),
                "api_key": "do-not-retain",
            }
        )
    return (
        gdelt,
        {
            "input_kind": "psx_announcement_evidence",
            "provider_id": "psx-announcement",
            "provider_mode": "manual_import",
            "status": "BLOCKED_PERMISSION_REVIEW",
            "record_count": 0,
            "run_reference": None,
            "decision_id": "psx-review-v1",
            "permission_status": "REVIEW_REQUIRED",
            "automated_access": False,
            "permission_policy_reference": _permission_reference(),
        },
        {
            "input_kind": "publisher_news",
            "status": "NOT_SUPPLIED",
            "record_count": 0,
            "reason": "No publisher-news input supplied for this test.",
        },
        {
            "input_kind": "readiness_configuration",
            "schema_version": "1.0",
            "artifact_reference": "configs/evidence/default_readiness.json",
            "sha256": "b" * 64,
        },
    )


def _gkg(identifier: str, available_at: datetime) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=identifier,
        evidence_type="gdelt_metadata",
        provider_id="=archive-provider",
        provider_mode="raw_archive",
        data_class="real",
        source_name="example.com",
        source_url="https://example.com/story",
        source_record_id=f"{available_at:%Y%m%d%H%M%S}-1",
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        title=None,
        source_published_at=available_at,
        source_published_date=None,
        available_at=available_at,
        fetched_at=_at("2026-10-23"),
        imported_at=_at("2026-10-23"),
        historical_availability_basis="gdelt_gkg_batch_timestamp",
        time_precision="second",
        contemporaneous_source_id=f"batch-{available_at:%Y%m%d%H%M%S}",
        contemporaneous_batch_at=available_at,
        permission_policy_id="permission-policy",
        raw_evidence_reference=f"capture.zip#{identifier}",
        content_sha256="a" * 64,
        entity_match_evidence=(
            {"field": "V2ORGANIZATIONS", "matched_alias": "OGDC"},
        ),
    )


def _computation():
    records = (
        _gkg("gkg-train", _at("2024-01-01")),
        _gkg("gkg-validation", _at("2024-01-03")),
        _gkg("gkg-test", _at("2024-01-05")),
    )
    return compute_evidence_coverage(records, _stock(), CoverageConfig())


def _publish(tmp_path: Path, *, run_id: str = RUN_ID) -> Path:
    return publish_evidence_coverage_run(
        _computation(),
        output_root=tmp_path / "coverage",
        input_references=_input_references(include_sensitive_extras=True),
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id=run_id,
        created_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
    )


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite_json(path: Path, value) -> None:
    path.write_bytes(_json_bytes(value))
    os.chmod(path, 0o600)


def _reseal_payload(pack: Path) -> None:
    inventory = regular_file_inventory(pack, files=PAYLOAD_FILES)
    inventory_digest = hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = pack / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_artifact_inventory"] = inventory
    manifest["payload_artifact_inventory_sha256"] = inventory_digest
    _rewrite_json(manifest_path, manifest)

    completion_path = pack / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_sha256"] = sha256_file(manifest_path)
    completion["manifest_size_bytes"] = manifest_path.stat().st_size
    completion["payload_artifact_inventory_sha256"] = inventory_digest
    _rewrite_json(completion_path, completion)


def test_publish_verify_private_flat_pack_and_redact_references(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path)

    verification = verify_evidence_coverage_pack(pack)

    assert verification.run_id == RUN_ID
    assert verification.mapping_count == 3
    assert verification.artifact_count == len(PAYLOAD_FILES)
    assert not verification.to_dict()["source_evidence_bound"]
    assert {item.name for item in pack.iterdir()} == ALL_FILES
    assert pack.stat().st_mode & 0o077 == 0
    assert all(item.stat().st_mode & 0o077 == 0 for item in pack.iterdir())

    retained = b"\n".join(item.read_bytes() for item in pack.iterdir())
    assert b"secret-value" not in retained
    assert b"do-not-retain" not in retained
    assert b"/Users/example/private" not in retained
    references = json.loads(
        (pack / "input_evidence_reference.json").read_text(encoding="utf-8")
    )
    assert references["inputs"][0]["path"] == "[REDACTED_ABSOLUTE_PATH]"
    assert references["inputs"][0]["api_key"] == "[REDACTED]"

    with (pack / "evidence_session_mappings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["provider_id"] == "'=archive-provider"


def test_publication_refuses_collision_and_dangling_destination(
    tmp_path: Path,
) -> None:
    _publish(tmp_path)
    with pytest.raises(EvidenceCoverageRunCollisionError):
        _publish(tmp_path)

    dangling_id = "ECOV-20261023-120001-cafebabe"
    dangling = tmp_path / "coverage" / dangling_id
    dangling.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(EvidenceCoverageRunCollisionError):
        _publish(tmp_path, run_id=dangling_id)


def test_checksum_tamper_and_extra_artifact_are_rejected(tmp_path: Path) -> None:
    pack = _publish(tmp_path)
    with (pack / "quality_report.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(EvidenceCoverageVerificationError, match="inventory"):
        verify_evidence_coverage_pack(pack)

    second = _publish(
        tmp_path,
        run_id="ECOV-20261023-120002-00ddf00d",
    )
    extra = second / "unexpected.txt"
    extra.write_text("unexpected\n", encoding="utf-8")
    os.chmod(extra, 0o600)
    with pytest.raises(EvidenceCoverageVerificationError, match="artifact set"):
        verify_evidence_coverage_pack(second)


def test_resealed_split_contradiction_is_rejected_semantically(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path)
    split_path = pack / "split_coverage.csv"
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    target = next(
        row
        for row in rows
        if row["profile"] == "gdelt_gkg_ogdc" and row["split"] == "train"
    )
    target["record_count"] = str(int(target["record_count"]) + 1)
    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(split_path, 0o600)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="contradictory|disagree",
    ):
        verify_evidence_coverage_pack(pack)


def test_resealed_source_reference_without_checksum_is_rejected(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path)
    reference_path = pack / "input_evidence_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["inputs"][0].pop("manifest_sha256")
    _rewrite_json(reference_path, reference)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="manifest_sha256",
    ):
        verify_evidence_coverage_pack(pack)


def test_symlink_artifact_and_non_private_mode_are_rejected(tmp_path: Path) -> None:
    pack = _publish(tmp_path)
    target = pack / "quality_report.json"
    saved = tmp_path / "saved-quality.json"
    saved.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(saved)
    with pytest.raises(
        EvidenceCoverageVerificationError, match="only regular files"
    ):
        verify_evidence_coverage_pack(pack)

    second = _publish(
        tmp_path,
        run_id="ECOV-20261023-120003-abcd1234",
    )
    os.chmod(second / "quality_report.json", 0o644)
    with pytest.raises(EvidenceCoverageVerificationError, match="not private"):
        verify_evidence_coverage_pack(second)


def test_zero_match_evidence_audit_still_produces_a_valid_pack(
    tmp_path: Path,
) -> None:
    computation = compute_evidence_coverage((), _stock(), CoverageConfig())
    pack = publish_evidence_coverage_run(
        computation,
        output_root=tmp_path / "coverage",
        input_references=_input_references(gdelt_record_count=0),
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id="ECOV-20261023-120004-zero0000",
        created_at=datetime(2026, 10, 23, 12, 4, tzinfo=UTC),
    )

    verification = verify_evidence_coverage_pack(pack)

    assert verification.mapping_count == 0
    assert set(verification.readiness_status_by_profile.values()) == {
        "NO_OVERLAP"
    }


def test_publication_rejects_untyped_or_checksum_free_source_references(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="source reference is invalid",
    ):
        publish_evidence_coverage_run(
            _computation(),
            output_root=tmp_path / "coverage",
            input_references={"run_id": "fabricated"},
            stock_reference=_stock().reference,
            repository_root=tmp_path,
            run_id="ECOV-20261023-120005-badref00",
        )

    invalid_stock = dict(_stock().reference)
    invalid_stock.pop("split_manifest_sha256")
    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="split_manifest_sha256",
    ):
        publish_evidence_coverage_run(
            _computation(),
            output_root=tmp_path / "coverage",
            input_references=_input_references(),
            stock_reference=invalid_stock,
            repository_root=tmp_path,
            run_id="ECOV-20261023-120006-badstock",
        )
