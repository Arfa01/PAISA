from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from paisa.evidence.artifacts import (
    EvidenceRunCollisionError,
    EvidenceVerificationError,
    _git_provenance,
    load_evidence_run,
    publish_evidence_run,
    verify_evidence_run,
)
from paisa.evidence.models import EvidenceCapture, EvidenceFetchResult, EvidenceRecord
from paisa.evidence.gdelt_gkg import (
    GKG_V21_COLUMNS,
    GdeltGkgPartition,
    GdeltGkgReplayCapture,
    replay_gdelt_gkg_captures,
)
from paisa.evidence.policy import (
    EvidencePermissionError,
    load_evidence_permission_policy,
)


UTC = timezone.utc
POLICY_ID = "test-evidence-policy"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _policy_bytes(*, status: str = "PERMITTED") -> bytes:
    return (
        json.dumps(
            {
                "schema_version": "1.0",
                "policy_id": POLICY_ID,
                "assessment_references": [
                    {
                        "path": "assessment.md",
                        "sha256": _digest(b"reviewed test assessment\n"),
                    }
                ],
                "decisions": [
                    {
                        "decision_id": "gdelt-gkg-test-decision-v1",
                        "provider_id": "gdelt-gkg-2.1",
                        "provider_mode": "raw_archive",
                        "status": status,
                        "automated_access": status == "PERMITTED",
                        "publisher_page_fetching": False,
                    }
                ],
            }
        )
        + "\n"
    ).encode("utf-8")


def _policy(root: Path, *, status: str = "PERMITTED"):
    assessment = root / "assessment.md"
    assessment.write_text("reviewed test assessment\n", encoding="utf-8")
    config = root / "policy.json"
    config.write_bytes(_policy_bytes(status=status))
    return load_evidence_permission_policy(config, repository_root=root)


def _gkg_record() -> EvidenceRecord:
    published = datetime(2026, 9, 4, 8, 58, tzinfo=UTC)
    batch = datetime(2026, 9, 4, 9, 0, tzinfo=UTC)
    fetched = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    return EvidenceRecord(
        evidence_id="gkg:20260904090000:1",
        evidence_type="gdelt_metadata",
        provider_id="gdelt-gkg-2.1",
        provider_mode="raw_archive",
        data_class="real",
        source_name="example.com",
        source_url="https://example.com/ogdc",
        source_record_id="20260904090000-1",
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        title=None,
        source_published_at=published,
        source_published_date=None,
        available_at=batch,
        fetched_at=fetched,
        imported_at=fetched,
        historical_availability_basis="gdelt_gkg_batch_timestamp",
        time_precision="second",
        contemporaneous_source_id="20260904090000-1",
        contemporaneous_batch_at=batch,
        permission_policy_id=POLICY_ID,
        raw_evidence_reference="capture_0001_batch.zip#20260904090000-1",
        content_sha256=_digest(b"record"),
        language="en",
        entity_match_evidence=(
            {
                "field": "V2ENHANCEDORGANIZATIONS",
                "alias": "OGDC",
                "matched_value": "OGDC",
                "match_method": "exact",
            },
        ),
        source_metadata={
            "GKGRECORDID": "20260904090000-1",
            "V2.1DATE": "20260904085800",
            "V2SOURCECOLLECTIONIDENTIFIER": "1",
            "V2SOURCECOMMONNAME": "example.com",
            "V2DOCUMENTIDENTIFIER": "https://example.com/ogdc",
            "V2ENHANCEDORGANIZATIONS": "OGDC,1",
            "archive_filename": "20260904090000.gkg.csv.zip",
        },
    )


def _result(*, with_rejections: bool = False) -> EvidenceFetchResult:
    fetched = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    batch = "20260904090000"
    archive_filename = f"{batch}.gkg.csv.zip"
    columns = {name: index for index, name in enumerate(GKG_V21_COLUMNS)}
    values = [""] * len(GKG_V21_COLUMNS)
    values[columns["GKGRECORDID"]] = f"{batch}-1"
    values[columns["V2.1DATE"]] = "20260904085800"
    values[columns["V2SOURCECOLLECTIONIDENTIFIER"]] = "1"
    values[columns["V2SOURCECOMMONNAME"]] = "example.com"
    values[columns["V2DOCUMENTIDENTIFIER"]] = "https://example.com/ogdc"
    values[columns["V1ORGANIZATIONS"]] = "OGDC,1"
    values[columns["V2ENHANCEDORGANIZATIONS"]] = "OGDC,1"
    rows = ["\t".join(values).encode("utf-8")]
    if with_rejections:
        unmatched = list(values)
        unmatched[columns["GKGRECORDID"]] = f"{batch}-2"
        unmatched[columns["V2DOCUMENTIDENTIFIER"]] = (
            "https://example.com/other"
        )
        unmatched[columns["V1ORGANIZATIONS"]] = "Other Company,1"
        unmatched[columns["V2ENHANCEDORGANIZATIONS"]] = "Other Company,1"
        future = list(values)
        future[columns["GKGRECORDID"]] = f"{batch}-3"
        future[columns["V2.1DATE"]] = "20260904090001"
        rows.extend(
            (
                "\t".join(unmatched).encode("utf-8"),
                "\t".join(future).encode("utf-8"),
                rows[0],
            )
        )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        member = zipfile.ZipInfo(f"{batch}.gkg.csv", (1980, 1, 1, 0, 0, 0))
        member.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(member, b"\n".join(rows) + b"\n")
    payload = output.getvalue()
    partition = GdeltGkgPartition(
        partition_id=batch,
        archive_url=(
            f"https://data.gdeltproject.org/gdeltv2/{archive_filename}"
        ),
        expected_size_bytes=len(payload),
        expected_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        expected_sha256=_digest(payload),
    )
    requested_start = datetime(2026, 9, 4, 0, 0, tzinfo=UTC)
    requested_end = datetime(2026, 9, 4, 23, 59, 59, tzinfo=UTC)
    capture_reference = f"capture_0001_{archive_filename}"
    replay = replay_gdelt_gkg_captures(
        (
            GdeltGkgReplayCapture(
                partition=partition,
                content=payload,
                fetched_at=fetched,
                capture_reference=capture_reference,
            ),
        ),
        requested_start=requested_start,
        requested_end=requested_end,
        permission_policy_id=POLICY_ID,
        max_files=1,
        maximum_errors=10,
    )
    source = {
        "configured_source": f"gdelt-gkg:{batch}",
        **partition.to_dict(),
        "provider_mode": "raw_archive",
        "source_collection_policy": "web_only_identifier_1",
        "permission_policy_id": POLICY_ID,
        "permission_policy_sha256": _digest(_policy_bytes()),
        "permission_decision_id": "gdelt-gkg-test-decision-v1",
        "permission_status": "PERMITTED",
    }
    plan = {
        "sequence": 1,
        **partition.to_dict(),
        "requested_start": requested_start.isoformat().replace("+00:00", "Z"),
        "requested_end": requested_end.isoformat().replace("+00:00", "Z"),
    }
    return EvidenceFetchResult(
        provider_id="gdelt-gkg-2.1",
        provider_mode="raw_archive",
        data_class="real",
        fetched_at=fetched,
        permission_policy_id=POLICY_ID,
        configured_sources=(source,),
        partition_plan=(plan,),
        records=replay.records,
        captures=(
            EvidenceCapture(
                capture_id=f"gkg-{batch}",
                provider_id="gdelt-gkg-2.1",
                configured_source=f"gdelt-gkg:{batch}",
                filename=archive_filename,
                fetched_at=fetched,
                content=payload,
                source_url=partition.archive_url,
                content_type="application/zip",
                sha256=_digest(payload),
            ),
        ),
        errors=replay.errors,
        counters={
            "partitions_configured": 1,
            "partitions_downloaded": 1,
            "partitions_resumed": 0,
            "partitions_completed": 1,
            "compressed_bytes": replay.compressed_bytes,
            "decompressed_bytes": replay.decompressed_bytes,
            "records_scanned": replay.scanned_record_count,
            "records_unmatched": replay.unmatched_record_count,
            "records_rejected": replay.rejected_record_count,
            "duplicate_record_count": replay.duplicate_record_count,
            "rejected_reason_counts": dict(replay.rejected_reason_counts),
            "records_retained": replay.matched_record_count,
        },
        timestamp_validation={
            "future_publication_policy": "reject",
            "publication_delay": dict(replay.publication_delay),
            "batch_delay": dict(replay.batch_delay),
            "negative_publication_delay_count": 0,
            "validated_record_count": replay.timestamp_valid_record_count,
            "rejected_record_count": replay.rejected_record_count,
            "rejected_reason_counts": dict(replay.rejected_reason_counts),
        },
        effective_config={
            "provider_id": "gdelt-gkg-2.1",
            "provider_mode": "raw_archive",
            "permission_policy_id": POLICY_ID,
            "permission_policy_sha256": _digest(_policy_bytes()),
            "permission_decision_id": "gdelt-gkg-test-decision-v1",
            "permission_status": "PERMITTED",
            "archive_host": "data.gdeltproject.org",
            "requested_start": requested_start.isoformat().replace("+00:00", "Z"),
            "requested_end": requested_end.isoformat().replace("+00:00", "Z"),
            "max_files": 1,
            "max_compressed_bytes_per_file": 64 * 1024 * 1024,
            "max_compressed_bytes": 256 * 1024 * 1024,
            "max_decompressed_bytes_per_file": 256 * 1024 * 1024,
            "max_decompressed_bytes": 1024 * 1024 * 1024,
            "max_records": 1_000_000,
            "max_row_bytes": 2 * 1024 * 1024,
            "max_field_characters": 1_000_000,
            "maximum_errors": 10,
            "source_collection_policy": "web_only_identifier_1",
            "dns_resolution_policy": "bounded_global_addresses_only",
            "connected_peer_policy": "global_and_in_approved_resolution_set",
            "tls_hostname_verification": "required",
            "redirect_policy": "reject",
            "dns_timeout_seconds": 5.0,
            "connect_timeout_seconds": 10.0,
            "read_timeout_seconds": 60.0,
            "total_request_timeout_seconds": 120.0,
            "environment_authentication": "disabled",
            "publisher_page_fetching": "disabled",
            "checkpointing": "disabled",
        },
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
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
            allow_nan=False,
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


def test_historical_batch_can_precede_later_acquisition() -> None:
    record = _gkg_record()
    assert record.available_at < record.fetched_at
    assert EvidenceRecord.from_dict(record.to_dict()) == record


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"provider_id": "totally-fake-provider"},
            "approved raw provider",
        ),
        (
            {"contemporaneous_source_id": "20260904090000-2"},
            "record identities disagree",
        ),
        (
            {"contemporaneous_batch_at": datetime(2026, 9, 4, 9, 1, tzinfo=UTC)},
            "available_at must equal",
        ),
        (
            {"source_published_at": datetime(2026, 9, 4, 9, 0, 1, tzinfo=UTC)},
            "publication timestamp cannot follow",
        ),
        (
            {"fetched_at": datetime(2026, 9, 4, 8, 59, tzinfo=UTC)},
            "cannot follow fetched_at",
        ),
        (
            {"imported_at": datetime(2026, 9, 4, 8, 59, tzinfo=UTC)},
            "cannot follow imported_at",
        ),
    ],
)
def test_gkg_constructor_rejects_incoherent_top_level_provenance(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_gkg_record(), **changes)


@pytest.mark.parametrize(
    ("metadata_changes", "message"),
    [
        ({"GKGRECORDID": "20260904091500-1"}, "record identities disagree"),
        (
            {"archive_filename": "20260904091500.gkg.csv.zip"},
            "archive batch differs",
        ),
        ({"V2.1DATE": "20260904085759"}, "V2.1DATE differs"),
        ({"V2SOURCECOLLECTIONIDENTIFIER": "2"}, "collection is unsupported"),
        ({"V2ORGANIZATIONS": "OGDC"}, "V2ENHANCEDORGANIZATIONS"),
        ({"V2SOURCECOMMONNAME": "forged.example"}, "common name provenance"),
        ({"V2DOCUMENTIDENTIFIER": "https://example.com/forged"}, "identifier provenance"),
        ({"V2ENHANCEDORGANIZATIONS": "Other Company,1"}, "source field does not"),
    ],
)
def test_gkg_constructor_rejects_forged_source_metadata(
    metadata_changes: dict[str, object], message: str
) -> None:
    record = _gkg_record()
    metadata = {**record.source_metadata, **metadata_changes}
    with pytest.raises(ValueError, match=message):
        replace(record, source_metadata=metadata)


@pytest.mark.parametrize(
    ("match", "message"),
    [
        (
            {
                "field": "totally_fake",
                "alias": "OGDC",
                "matched_value": "OGDC",
                "match_method": "exact",
            },
            "field is unsupported",
        ),
        (
            {
                "field": "V2ENHANCEDORGANIZATIONS",
                "alias": "Fake Oil Company",
                "matched_value": "Fake Oil Company",
                "match_method": "exact",
            },
            "alias is not controlled",
        ),
        (
            {
                "field": "V2ENHANCEDORGANIZATIONS",
                "alias": "OGDC",
                "matched_value": "Other Company",
                "match_method": "exact",
            },
            "matched value does not contain",
        ),
        (
            {
                "field": "V2ENHANCEDORGANIZATIONS",
                "alias": "OGDC",
                "matched_value": "OGDC",
                "match_method": "substring",
            },
            "method is unsupported",
        ),
    ],
)
def test_gkg_constructor_rejects_forged_match_evidence(
    match: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_gkg_record(), entity_match_evidence=(match,))


def test_publisher_news_cannot_backdate_availability() -> None:
    record = _gkg_record()
    with pytest.raises(ValueError, match="Publisher news cannot claim"):
        replace(
            record,
            evidence_type="publisher_news",
            provider_id="gdelt-doc",
            provider_mode="real",
            data_class="real",
            historical_availability_basis="publisher_news_availability",
            contemporaneous_batch_at=None,
        )


def test_psx_date_only_evidence_does_not_invent_an_instant() -> None:
    fetched = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    record = EvidenceRecord(
        evidence_id="psx:OGDC:123",
        evidence_type="exchange_announcement",
        provider_id="psx-announcement",
        provider_mode="manual_import",
        data_class="imported",
        source_name="Pakistan Stock Exchange",
        source_url="https://dps.psx.com.pk/company/OGDC",
        source_record_id="123",
        ticker="OGDC",
        company_name="Oil & Gas Development Company Limited",
        title="Financial results",
        source_published_at=None,
        source_published_date=date(2026, 9, 4),
        available_at=None,
        fetched_at=fetched,
        imported_at=fetched,
        historical_availability_basis=(
            "official_exchange_date_conservative_next_session"
        ),
        time_precision="date_only",
        contemporaneous_source_id="123",
        contemporaneous_batch_at=None,
        permission_policy_id=POLICY_ID,
        raw_evidence_reference="capture_0001_input.csv#123",
        content_sha256=_digest(b"announcement"),
        event_category="financial_results",
        announcement_type="Financial Results",
        entity_match_evidence=(
            {
                "field": "ticker",
                "matched_value": "OGDC",
                "match_method": "official_exact_ticker",
            },
        ),
        source_metadata={
            "source_mode": "manual_import",
            "automated_access": False,
            "exchange_posted_at_raw": "2026-09-04",
            "permission_decision_id": "psx-test-decision",
            "manual_permission_attestation_sha256": "1" * 64,
            "manual_permission_reference_sha256": "2" * 64,
        },
    )
    assert record.available_at is None
    with pytest.raises(ValueError, match="cannot invent"):
        replace(record, available_at=fetched)


def test_permission_policy_is_checksum_bound(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    assert policy.require("gdelt-gkg-2.1", "raw_archive", automated=True)
    (tmp_path / "assessment.md").write_text("changed\n", encoding="utf-8")
    with pytest.raises(EvidencePermissionError, match="checksum"):
        load_evidence_permission_policy(
            tmp_path / "policy.json", repository_root=tmp_path
        )


def test_git_provenance_distinguishes_clean_tree_from_unavailable(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["git", "init", "--quiet"], cwd=tmp_path, check=True, timeout=10
    )

    assert _git_provenance(tmp_path)["dirty_tree"] is False

    (tmp_path / "untracked.txt").write_text("change\n", encoding="utf-8")
    assert _git_provenance(tmp_path)["dirty_tree"] is True


def test_record_rejects_mixed_provider_semantics_before_result_construction() -> None:
    with pytest.raises(ValueError, match="approved raw provider"):
        replace(_gkg_record(), provider_id="unreviewed-provider")


def test_known_provider_mode_cannot_claim_another_evidence_type() -> None:
    fetched = _gkg_record().fetched_at
    with pytest.raises(ValueError, match="requires evidence_type"):
        replace(
            _gkg_record(),
            evidence_type="publisher_news",
            available_at=fetched,
            historical_availability_basis="publisher_news_availability",
            contemporaneous_batch_at=None,
        )


def test_from_dict_rejects_missing_required_timestamp_cleanly() -> None:
    payload = _gkg_record().to_dict()
    payload["fetched_at"] = None
    with pytest.raises(ValueError, match="fetched_at must be a datetime"):
        EvidenceRecord.from_dict(payload)


def test_publish_verify_collision_and_tamper(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    run_id = "EVID-20261023-120000-deadbeef"
    output = tmp_path / "evidence"
    result = _result()
    published = publish_evidence_run(
        result,
        output_root=output,
        permission_policy=policy,
        repository_root=tmp_path,
        run_id=run_id,
    )
    verification = verify_evidence_run(published)
    assert verification.status == "PASS"
    assert verification.record_count == 1
    assert verification.scanned_record_count == 1
    assert verification.timestamp_valid_record_count == 1
    assert verification.rejected_record_count == 0
    assert verification.unmatched_record_count == 0
    assert verification.matched_record_count == 1
    assert verification.duplicate_record_count == 0
    assert (
        verification.verification_basis
        == "RAW_GDELT_GKG_ZIP_SEMANTIC_REPLAY"
    )
    loaded = load_evidence_run(published)
    assert loaded.records == result.records
    assert loaded.reference["artifact_reference"] == f"evidence/{run_id}"
    assert loaded.reference["scanned_record_count"] == 1
    provider_config = json.loads(
        (published / "provider_config.json").read_text("utf-8")
    )
    assert provider_config["effective_config"]["max_files"] == 1
    with pytest.raises(EvidenceRunCollisionError):
        publish_evidence_run(
            _result(),
            output_root=output,
            permission_policy=policy,
            repository_root=tmp_path,
            run_id=run_id,
        )
    with (published / "normalized_evidence.jsonl").open("ab") as handle:
        handle.write(b"{}\n")
    with pytest.raises(EvidenceVerificationError, match="inventory differs"):
        verify_evidence_run(published)


@pytest.mark.parametrize(
    ("artifact_name", "field", "message"),
    [
        (
            "quality_report.json",
            ("counters", "records_scanned"),
            "raw replay counter differs",
        ),
        (
            "quality_report.json",
            ("counters", "records_rejected"),
            "raw replay counter differs",
        ),
        (
            "quality_report.json",
            ("counters", "records_retained"),
            "raw replay counter differs",
        ),
        (
            "timestamp_validation_report.json",
            ("validated_record_count",),
            "timestamp-validation report differs",
        ),
        (
            "run_manifest.json",
            ("raw_replay_counts", "matched_record_count"),
            "raw replay count claims differ",
        ),
        (
            "final_summary.json",
            ("raw_replay_counts", "scanned_record_count"),
            "raw replay count claims differ",
        ),
    ],
)
def test_resealed_raw_replay_counter_claims_are_rejected(
    tmp_path: Path,
    artifact_name: str,
    field: tuple[str, ...],
    message: str,
) -> None:
    published = publish_evidence_run(
        _result(),
        output_root=tmp_path / "evidence",
        permission_policy=_policy(tmp_path),
        repository_root=tmp_path,
        run_id="EVID-20261023-120003-0badc0de",
    )
    artifact_path = published / artifact_name
    payload = json.loads(artifact_path.read_text("utf-8"))
    target = payload
    for key in field[:-1]:
        target = target[key]
    target[field[-1]] += 1
    _write_json(artifact_path, payload)
    _reseal_pack(published)

    with pytest.raises(EvidenceVerificationError, match=message):
        verify_evidence_run(published)


def test_resealed_normalized_csv_claim_is_rejected(tmp_path: Path) -> None:
    published = publish_evidence_run(
        _result(),
        output_root=tmp_path / "evidence",
        permission_policy=_policy(tmp_path),
        repository_root=tmp_path,
        run_id="EVID-20261023-120004-f00dbabe",
    )
    csv_path = published / "normalized_evidence.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"\n")
    _reseal_pack(published)

    with pytest.raises(EvidenceVerificationError, match="CSV differs"):
        verify_evidence_run(published)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        (("effective_config", "tls_hostname_verification"), "disabled"),
        (("effective_config", "total_request_timeout_seconds"), 1e100),
        (("configured_sources", 0, "permission_status"), "PROHIBITED"),
    ],
)
def test_resealed_gkg_security_or_permission_claim_is_rejected(
    tmp_path: Path,
    target: tuple[object, ...],
    value: object,
) -> None:
    published = publish_evidence_run(
        _result(),
        output_root=tmp_path / "evidence",
        permission_policy=_policy(tmp_path),
        repository_root=tmp_path,
        run_id="EVID-20261023-120008-badcab1e",
    )
    path = published / "provider_config.json"
    payload = json.loads(path.read_text("utf-8"))
    selected = payload
    for part in target[:-1]:
        selected = selected[part]
    selected[target[-1]] = value
    _write_json(path, payload)
    _reseal_pack(published)

    with pytest.raises(EvidenceVerificationError, match="GDELT"):
        verify_evidence_run(published)


def test_resealed_fake_gkg_match_field_is_rejected_by_portable_verifier(
    tmp_path: Path,
) -> None:
    published = publish_evidence_run(
        _result(),
        output_root=tmp_path / "evidence",
        permission_policy=_policy(tmp_path),
        repository_root=tmp_path,
        run_id="EVID-20261023-120009-fa1e0001",
    )
    normalized_path = published / "normalized_evidence.jsonl"
    payload = json.loads(normalized_path.read_text("utf-8"))
    payload["entity_match_evidence"][0]["field"] = "totally_fake"
    normalized_path.write_text(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _reseal_pack(published)

    with pytest.raises(
        EvidenceVerificationError,
        match="Evidence row 1 is invalid: GDELT entity match field is unsupported",
    ):
        verify_evidence_run(published)


def test_resealed_embedded_network_path_is_rejected(
    tmp_path: Path,
) -> None:
    published = publish_evidence_run(
        _result(),
        output_root=tmp_path / "evidence",
        permission_policy=_policy(tmp_path),
        repository_root=tmp_path,
        run_id="EVID-20261023-120011-pathsafe",
    )
    summary_path = published / "final_summary.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["extra_note"] = "stored at //server/share/private.csv"
    _write_json(summary_path, summary)
    _reseal_pack(published)

    with pytest.raises(EvidenceVerificationError, match="unsafe private data"):
        verify_evidence_run(published)


def test_resealed_raw_capture_still_must_match_partition_checksums(
    tmp_path: Path,
) -> None:
    published = publish_evidence_run(
        _result(),
        output_root=tmp_path / "evidence",
        permission_policy=_policy(tmp_path),
        repository_root=tmp_path,
        run_id="EVID-20261023-120006-facefeed",
    )
    metadata_path = published / "capture_metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))
    capture_path = published / metadata[0]["artifact_reference"]
    content = capture_path.read_bytes()
    capture_path.write_bytes(content[:-1] + bytes((content[-1] ^ 1,)))
    metadata[0]["sha256"] = _digest(capture_path.read_bytes())
    metadata[0]["size_bytes"] = capture_path.stat().st_size
    _write_json(metadata_path, metadata)
    _reseal_pack(published)

    with pytest.raises(EvidenceVerificationError, match="semantic replay failed"):
        verify_evidence_run(published)


def test_raw_replay_reports_all_record_classes_and_rejection_reasons(
    tmp_path: Path,
) -> None:
    result = _result(with_rejections=True)
    published = publish_evidence_run(
        result,
        output_root=tmp_path / "evidence",
        permission_policy=_policy(tmp_path),
        repository_root=tmp_path,
        run_id="EVID-20261023-120005-decafbad",
    )

    verification = verify_evidence_run(published)
    assert verification.status == "PARTIAL"
    assert verification.scanned_record_count == 4
    assert verification.timestamp_valid_record_count == 3
    assert verification.rejected_record_count == 2
    assert verification.unmatched_record_count == 1
    assert verification.matched_record_count == 1
    assert verification.duplicate_record_count == 1
    assert json.loads(
        (published / "quality_report.json").read_text("utf-8")
    )["counters"]["rejected_reason_counts"] == {
        "duplicate_source_record": 1,
        "publication_after_batch": 1,
    }


def test_planned_gkg_partition_cannot_false_pass_without_capture_or_error(
    tmp_path: Path,
) -> None:
    result = replace(
        _result(),
        records=(),
        captures=(),
        errors=(),
        counters={
            "partitions_configured": 1,
            "partitions_downloaded": 0,
            "partitions_resumed": 0,
            "partitions_completed": 0,
            "compressed_bytes": 0,
            "decompressed_bytes": 0,
            "records_scanned": 0,
            "records_unmatched": 0,
            "records_rejected": 0,
            "duplicate_record_count": 0,
            "rejected_reason_counts": {},
            "records_retained": 0,
        },
        timestamp_validation={
            "future_publication_policy": "reject",
            "publication_delay": {
                "minimum_seconds": None,
                "maximum_seconds": None,
            },
            "batch_delay": {
                "minimum_seconds": None,
                "maximum_seconds": None,
            },
            "negative_publication_delay_count": 0,
            "validated_record_count": 0,
            "rejected_record_count": 0,
            "rejected_reason_counts": {},
        },
    )
    output = tmp_path / "evidence"
    run_id = "EVID-20261023-120007-0faded00"

    with pytest.raises(EvidenceVerificationError, match="capture or failure"):
        publish_evidence_run(
            result,
            output_root=output,
            permission_policy=_policy(tmp_path),
            repository_root=tmp_path,
            run_id=run_id,
        )

    assert not (output / run_id).exists()


def test_publication_enforces_permission_decision(tmp_path: Path) -> None:
    policy = _policy(tmp_path, status="PROHIBITED")
    with pytest.raises(EvidencePermissionError, match="PROHIBITED"):
        publish_evidence_run(
            _result(),
            output_root=tmp_path / "evidence",
            permission_policy=policy,
            repository_root=tmp_path,
            run_id="EVID-20261023-120001-feedface",
        )


def test_raw_reference_must_resolve_to_retained_capture(tmp_path: Path) -> None:
    policy = _policy(tmp_path)
    result = replace(
        _result(),
        records=(
            replace(
                _gkg_record(),
                raw_evidence_reference="capture_9999_missing.zip#record",
            ),
        ),
    )
    with pytest.raises(ValueError, match="resolve to a retained capture"):
        publish_evidence_run(
            result,
            output_root=tmp_path / "evidence",
            permission_policy=policy,
            repository_root=tmp_path,
            run_id="EVID-20261023-120002-cafebabe",
        )
