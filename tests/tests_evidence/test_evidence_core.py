from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from paisa.evidence.artifacts import (
    EvidenceRunCollisionError,
    EvidenceVerificationError,
    load_evidence_run,
    publish_evidence_run,
    verify_evidence_run,
)
from paisa.evidence.models import EvidenceCapture, EvidenceFetchResult, EvidenceRecord
from paisa.evidence.policy import (
    EvidencePermissionError,
    load_evidence_permission_policy,
)


UTC = timezone.utc
POLICY_ID = "test-evidence-policy"


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _policy(root: Path, *, status: str = "PERMITTED"):
    assessment = root / "assessment.md"
    assessment.write_text("reviewed test assessment\n", encoding="utf-8")
    config = root / "policy.json"
    config.write_text(
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
                        "provider_id": "gdelt-gkg-2.1",
                        "provider_mode": "raw_archive",
                        "status": status,
                        "automated_access": status == "PERMITTED",
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
                "field": "V2ORGANIZATIONS",
                "matched_alias": "OGDC",
                "source_value": "OGDC",
            },
        ),
        source_metadata={
            "GKGRECORDID": "20260904090000-1",
            "V2.1DATE": "20260904085800",
        },
    )


def _result() -> EvidenceFetchResult:
    fetched = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    payload = b"bounded raw archive"
    return EvidenceFetchResult(
        provider_id="gdelt-gkg-2.1",
        provider_mode="raw_archive",
        data_class="real",
        fetched_at=fetched,
        permission_policy_id=POLICY_ID,
        configured_sources=({"source": "official-test"},),
        partition_plan=({"partition_id": "20260904090000"},),
        records=(_gkg_record(),),
        captures=(
            EvidenceCapture(
                capture_id="20260904090000",
                provider_id="gdelt-gkg-2.1",
                configured_source="official-test",
                filename="batch.zip",
                fetched_at=fetched,
                content=payload,
                sha256=_digest(payload),
            ),
        ),
        counters={"archives_completed": 1},
        timestamp_validation={"accepted": 1, "quarantined": 0},
        effective_config={"max_files": 1, "publisher_page_fetching": "disabled"},
    )


def test_historical_batch_can_precede_later_acquisition() -> None:
    record = _gkg_record()
    assert record.available_at < record.fetched_at
    assert EvidenceRecord.from_dict(record.to_dict()) == record


def test_publisher_news_cannot_backdate_availability() -> None:
    record = _gkg_record()
    with pytest.raises(ValueError, match="Publisher news cannot claim"):
        replace(
            record,
            evidence_type="publisher_news",
            provider_id="publisher",
            provider_mode="feed",
            data_class="imported",
            historical_availability_basis="publisher_news_availability",
            contemporaneous_batch_at=None,
        )


def test_psx_date_only_evidence_does_not_invent_an_instant() -> None:
    fetched = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
    record = EvidenceRecord(
        evidence_id="psx:OGDC:123",
        evidence_type="exchange_announcement",
        provider_id="psx-announcements",
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


def test_fetch_result_rejects_mixed_provider_semantics() -> None:
    with pytest.raises(ValueError, match="Record provider semantics"):
        replace(
            _result(),
            records=(replace(_gkg_record(), provider_id="unreviewed-provider"),),
        )


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
    published = publish_evidence_run(
        _result(),
        output_root=output,
        permission_policy=policy,
        repository_root=tmp_path,
        run_id=run_id,
    )
    verification = verify_evidence_run(published)
    assert verification.status == "PASS"
    assert verification.record_count == 1
    loaded = load_evidence_run(published)
    assert loaded.records == (_gkg_record(),)
    assert loaded.reference["artifact_reference"] == f"evidence/{run_id}"
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
