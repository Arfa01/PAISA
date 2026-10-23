from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import socket
import urllib.request

import pytest

from paisa.evidence.policy import EvidencePermissionError, EvidencePermissionPolicy
from paisa.evidence.psx_announcements import (
    PsxAnnouncementImportError,
    PsxAnnouncementManualImporter,
    categorize_announcement,
)


IMPORTED_AT = datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc)
ATTESTATION = (
    "Written metadata-only academic authorization retained under reference PSX-FYP-001."
)
AUTHORIZATION_REFERENCE = "PSX written-permission record PSX-FYP-001 dated 2026-10-20."


def _policy(
    *,
    status: str = "MANUAL_ONLY",
    automated_access: bool | None = False,
    expected_attestation: str | None = ATTESTATION,
    expected_reference: str | None = AUTHORIZATION_REFERENCE,
) -> EvidencePermissionPolicy:
    decision = {
        "decision_id": "psx-ogdc-manual-metadata-v1",
        "provider_id": "psx-announcement",
        "provider_mode": "manual_import",
        "status": status,
    }
    if automated_access is not None:
        decision["automated_access"] = automated_access
    if expected_attestation is not None:
        decision["authorization_attestation_sha256"] = hashlib.sha256(
            expected_attestation.encode("utf-8")
        ).hexdigest()
    if expected_reference is not None:
        decision["authorization_reference_sha256"] = hashlib.sha256(
            expected_reference.encode("utf-8")
        ).hexdigest()
    return EvidencePermissionPolicy(
        policy_id="contemporaneous-evidence-v1",
        schema_version="1.0",
        policy_path="configs/evidence/contemporaneous_sources.json",
        policy_sha256="a" * 64,
        assessment_references=(
            {
                "path": "docs/PSX_ANNOUNCEMENT_SOURCE_ASSESSMENT.md",
                "sha256": "b" * 64,
            },
        ),
        decisions={
            ("psx-announcement", "manual_import"): decision,
        },
    )


def _write_csv(tmp_path: Path, rows: list[str], *, header: str | None = None) -> Path:
    selected_header = header or (
        "announcement_id,ticker,company_name,announcement_type,title,"
        "exchange_posted_at,official_source_url,pdf_url,pdf_sha256,language"
    )
    path = tmp_path / "operator-export.csv"
    path.write_text(selected_header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _importer(path: Path, **kwargs) -> PsxAnnouncementManualImporter:
    return PsxAnnouncementManualImporter(
        path,
        permission_policy=kwargs.pop("permission_policy", _policy()),
        authorization_attestation=kwargs.pop(
            "authorization_attestation", ATTESTATION
        ),
        authorization_reference=kwargs.pop(
            "authorization_reference", AUTHORIZATION_REFERENCE
        ),
        clock=lambda: IMPORTED_AT,
        **kwargs,
    )


def test_manual_import_captures_exact_input_and_preserves_time_semantics(tmp_path: Path):
    pdf_digest = "c" * 64
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,ogdc,Oil and Gas Development Company Limited,"
                "Board Meeting,Meeting of the Board,"
                "2026-09-14T14:45:00+05:00,"
                "https://dps.psx.com.pk/company/OGDC,"
                "https://dps.psx.com.pk/download/document/123,"
                f"{pdf_digest},en"
            ),
            (
                "PSX-002,OGDC,Oil and Gas Development Company Limited,"
                "Annual Report,Annual report for 2026,2026-09-15,"
                "http://www.psx.com.pk/psx/announcement/PSX-002,,,en"
            ),
        ],
    )
    expected_bytes = path.read_bytes()

    result = _importer(path).fetch()

    assert result.provider_id == "psx-announcement"
    assert result.provider_mode == "manual_import"
    assert result.data_class == "imported"
    assert result.fetched_at == IMPORTED_AT
    assert len(result.captures) == 1
    assert result.captures[0].content == expected_bytes
    assert result.captures[0].sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert result.counters == {
        "input_rows": 2,
        "accepted_records": 2,
        "ogdc_records": 2,
        "timed_records": 1,
        "date_only_records": 1,
        "pdf_reference_records": 1,
    }

    timed, date_only = result.records
    assert timed.ticker == "OGDC"
    assert timed.evidence_type == "exchange_announcement"
    assert timed.available_at == datetime(
        2026, 9, 14, 9, 45, tzinfo=timezone.utc
    )
    assert timed.source_published_at == timed.available_at
    assert timed.source_published_date is None
    assert timed.time_precision == "second"
    assert timed.historical_availability_basis == "official_exchange_posted_timestamp"
    assert timed.imported_at == IMPORTED_AT
    assert timed.fetched_at == IMPORTED_AT
    assert timed.event_category == "board_meeting"
    assert timed.pdf_reference == "https://dps.psx.com.pk/download/document/123"
    assert timed.pdf_sha256 == pdf_digest
    assert timed.raw_evidence_reference == "capture_0001_psx_announcements.csv#row=2"
    assert timed.entity_match_evidence == (
        {
            "field": "ticker",
            "matched_value": "OGDC",
            "match_method": "official_exact_ticker",
        },
    )

    assert date_only.source_published_at is None
    assert date_only.available_at is None
    assert date_only.source_published_date.isoformat() == "2026-09-15"
    assert date_only.time_precision == "date_only"
    assert (
        date_only.historical_availability_basis
        == "official_exchange_date_conservative_next_session"
    )
    assert "date_only_exchange_posting_conservative_next_session" in (
        date_only.normalization_warnings
    )
    assert date_only.raw_evidence_reference == "capture_0001_psx_announcements.csv#row=3"


def test_minute_timestamp_preserves_declared_precision_without_warning(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,"
                "Board Meeting,Meeting of the Board,"
                "2026-09-14T14:45+05:00,"
                "https://dps.psx.com.pk/company/OGDC,,,en"
            ),
        ],
    )

    record = _importer(path).fetch().records[0]

    assert record.source_published_at == datetime(
        2026, 9, 14, 9, 45, tzinfo=timezone.utc
    )
    assert record.available_at == record.source_published_at
    assert record.time_precision == "minute"
    assert record.normalization_warnings == ()


@pytest.mark.parametrize(
    "timestamp",
    [
        "2026-09-14T14+05:00",
        "2026-09-14T144500+0500",
        "2026-W38-1T14:45+05:00",
        "2026-09-14 14:45+05:00",
        "2026-09-14T14:45+0500",
        "2026-09-14T14:45:00.1+05:00",
        "2026-09-14T14:45:00z",
    ],
)
def test_timestamp_requires_exact_calendar_grammar(
    tmp_path: Path,
    timestamp: str,
) -> None:
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,"
                "Board Meeting,Meeting of the Board,"
                f"{timestamp},https://dps.psx.com.pk/company/OGDC,,,en"
            ),
        ],
    )

    with pytest.raises(PsxAnnouncementImportError, match="exact YYYY-MM-DD"):
        _importer(path).fetch()


@pytest.mark.parametrize(
    ("timestamp", "expected_precision"),
    [
        ("2026-09-14T14:45Z", "minute"),
        ("2026-09-14T14:45:00Z", "second"),
        ("2026-09-14T14:45-03:30", "minute"),
        ("2026-09-14T14:45:00-03:30", "second"),
    ],
)
def test_exact_timestamp_grammar_preserves_minute_or_second_precision(
    tmp_path: Path,
    timestamp: str,
    expected_precision: str,
) -> None:
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,"
                f"Notice,{timestamp},https://dps.psx.com.pk/company/OGDC,,,en"
            )
        ],
    )

    assert _importer(path).fetch().records[0].time_precision == expected_precision


@pytest.mark.parametrize("ticker", ["OGDCL", "OGDCX", "OGDC-L", "XOGDC", "OG DC"])
def test_ticker_match_is_exact_and_rejects_substring_false_positives(
    tmp_path: Path,
    ticker: str,
):
    path = _write_csv(
        tmp_path,
        [
            (
                f"PSX-001,{ticker},Oil and Gas Development Company Limited,"
                "Other,Notice,2026-09-15,"
                "https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )

    with pytest.raises(PsxAnnouncementImportError, match="exactly equal OGDC"):
        _importer(path).fetch()


def test_duplicate_announcement_ids_fail_the_whole_import(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,"
                "First,2026-09-14,https://dps.psx.com.pk/company/OGDC,,,"
            ),
            (
                "psx-001,OGDC,Oil and Gas Development Company Limited,Other,"
                "Second,2026-09-15,https://dps.psx.com.pk/company/OGDC,,,"
            ),
        ],
    )

    with pytest.raises(PsxAnnouncementImportError, match="duplicated"):
        _importer(path).fetch()


@pytest.mark.parametrize(
    ("header", "row", "message"),
    [
        (
            "announcement_id,ticker,company_name,announcement_type,title,"
            "exchange_posted_at",
            "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,2026-09-15",
            "missing required columns",
        ),
        (
            "announcement_id,ticker,company_name,announcement_type,title,"
            "exchange_posted_at,official_source_url,body",
            "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
            "2026-09-15,https://dps.psx.com.pk/company/OGDC,forbidden text",
            "unsupported columns",
        ),
        (
            "announcement_id,ticker,company_name,announcement_type,title,"
            "exchange_posted_at,official_source_url",
            "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,,"
            "2026-09-15,https://dps.psx.com.pk/company/OGDC",
            "title row 2 cannot be blank",
        ),
    ],
)
def test_missing_or_body_fields_are_rejected(
    tmp_path: Path,
    header: str,
    row: str,
    message: str,
):
    path = _write_csv(tmp_path, [row], header=header)

    with pytest.raises(PsxAnnouncementImportError, match=message):
        _importer(path).fetch()


def test_permission_decision_and_bound_authorization_are_required(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15,https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )

    with pytest.raises(EvidencePermissionError, match="REVIEW_REQUIRED"):
        _importer(path, permission_policy=_policy(status="REVIEW_REQUIRED"))
    with pytest.raises(EvidencePermissionError, match="explicitly deny"):
        _importer(path, permission_policy=_policy(automated_access=None))
    with pytest.raises(EvidencePermissionError, match="pending-review"):
        _importer(path, authorization_attestation="REVIEW_REQUIRED")
    with pytest.raises(
        EvidencePermissionError,
        match="explicit authorization attestation",
    ):
        _importer(path, authorization_attestation="")
    with pytest.raises(
        EvidencePermissionError,
        match="explicit authorization reference",
    ):
        _importer(path, authorization_reference="")
    with pytest.raises(
        EvidencePermissionError,
        match="requires a valid authorization_attestation_sha256",
    ):
        _importer(path, permission_policy=_policy(expected_attestation=None))
    with pytest.raises(
        EvidencePermissionError,
        match="requires a valid authorization_reference_sha256",
    ):
        _importer(path, permission_policy=_policy(expected_reference=None))
    with pytest.raises(EvidencePermissionError, match="does not match"):
        _importer(
            path,
            permission_policy=_policy(expected_attestation="different attestation"),
        )
    with pytest.raises(EvidencePermissionError, match="does not match"):
        _importer(
            path,
            permission_policy=_policy(expected_reference="different reference"),
        )


@pytest.mark.parametrize("status", ["MANUAL_ONLY", "PERMITTED"])
def test_bound_authorization_allows_reviewed_manual_statuses(
    tmp_path: Path,
    status: str,
):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15,https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )

    result = _importer(path, permission_policy=_policy(status=status)).fetch()

    assert result.configured_sources[0]["permission_status"] == status


def test_authorization_values_are_never_retained_in_result_metadata(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15,https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )
    importer = _importer(path)

    result = importer.fetch()

    attestation_sha256 = hashlib.sha256(ATTESTATION.encode("utf-8")).hexdigest()
    reference_sha256 = hashlib.sha256(
        AUTHORIZATION_REFERENCE.encode("utf-8")
    ).hexdigest()
    configured_source = result.configured_sources[0]
    assert configured_source["manual_permission_attestation_sha256"] == (
        attestation_sha256
    )
    assert configured_source["manual_permission_reference_sha256"] == (
        reference_sha256
    )
    assert result.records[0].source_metadata[
        "manual_permission_attestation_sha256"
    ] == attestation_sha256
    assert result.records[0].source_metadata[
        "manual_permission_reference_sha256"
    ] == reference_sha256
    retained_state = repr(vars(importer)) + repr(result)
    assert ATTESTATION not in retained_state
    assert AUTHORIZATION_REFERENCE not in retained_state


def test_extra_csv_values_are_rejected_instead_of_becoming_hidden_content(tmp_path: Path):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15,https://dps.psx.com.pk/company/OGDC,,,,"
                "unexpected hidden value"
            )
        ],
    )

    with pytest.raises(PsxAnnouncementImportError, match="extra values"):
        _importer(path).fetch()


@pytest.mark.parametrize(
    "url",
    [
        "https://evilpsx.com.pk/announcement/1",
        "https://psx.com.pk.evil.example/announcement/1",
        "ftp://dps.psx.com.pk/announcement/1",
        "https://user:secret@dps.psx.com.pk/announcement/1",
        "https://dps.psx.com.pk:8443/announcement/1",
        "https://dps.psx.com.pk/announcement/1?token=secret",
    ],
)
def test_only_credential_free_official_psx_urls_are_accepted(tmp_path: Path, url: str):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                f"2026-09-15,{url},,,"
            )
        ],
    )

    with pytest.raises(PsxAnnouncementImportError):
        _importer(path).fetch()


@pytest.mark.parametrize(
    ("announcement_type", "title", "expected"),
    [
        ("Financial Results", "Quarter ended September", "financial_results"),
        ("Other", "Meeting of the Board of Directors", "board_meeting"),
        ("Material Information", "Generic disclosure", "material_information"),
        ("Other", "Operational update for Q3", "production_update"),
        ("Other", "Interim dividend declaration", "dividend"),
        ("Annual Report", "Report for the year", "annual_report"),
        ("Other", "Notice of EOGM", "AGM_EOGM"),
        ("Other", "Book closure notice", "corporate_action"),
        ("Other", "Change in contact details", "other"),
    ],
)
def test_event_category_mapping_is_deterministic(
    announcement_type: str,
    title: str,
    expected: str,
):
    assert categorize_announcement(announcement_type, title) == expected
    assert categorize_announcement(announcement_type, title) == expected


def test_date_only_and_future_time_rules_never_invent_or_backdate_time(tmp_path: Path):
    future = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-10-24,https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )
    with pytest.raises(PsxAnnouncementImportError, match="later than the import date"):
        _importer(future).fetch()

    naive = _write_csv(
        tmp_path,
        [
            (
                "PSX-002,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15T14:00:00,https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )
    with pytest.raises(PsxAnnouncementImportError, match="explicit UTC offset"):
        _importer(naive).fetch()


def test_pdf_reference_is_metadata_only_and_no_network_or_pdf_bytes_are_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15,https://dps.psx.com.pk/company/OGDC,"
                f"https://dps.psx.com.pk/unreachable.pdf,{'d' * 64},en"
            )
        ],
    )
    def forbidden_network(*_args, **_kwargs):
        raise AssertionError("PSX manual import must never access the network")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden_network)

    result = _importer(path).fetch()

    assert result.records[0].pdf_reference.endswith("unreachable.pdf")
    assert result.records[0].pdf_sha256 == "d" * 64
    assert result.records[0].source_metadata["pdf_handling"] == (
        "reference_and_supplied_checksum_only_no_retrieval"
    )
    assert result.records[0].source_metadata["pdf_checksum_verification"] == (
        "operator_supplied_unverified_claim"
    )
    assert "pdf_sha256_operator_supplied_unverified_claim" in (
        result.records[0].normalization_warnings
    )


def test_manual_input_must_be_a_regular_nonsymlink_file(tmp_path: Path) -> None:
    regular = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15,https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )
    linked = tmp_path / "linked.csv"
    linked.symlink_to(regular)
    with pytest.raises(PsxAnnouncementImportError, match="regular non-symlink"):
        _importer(linked).fetch()

    directory = tmp_path / "directory.csv"
    directory.mkdir()
    with pytest.raises(PsxAnnouncementImportError, match="regular non-symlink"):
        _importer(directory).fetch()

    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.csv"
        os.mkfifo(fifo)
        with pytest.raises(PsxAnnouncementImportError, match="regular non-symlink"):
            _importer(fifo).fetch()


def test_manual_input_replacement_during_read_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_csv(
        tmp_path,
        [
            (
                "PSX-001,OGDC,Oil and Gas Development Company Limited,Other,Notice,"
                "2026-09-15,https://dps.psx.com.pk/company/OGDC,,,"
            )
        ],
    )
    original_read = os.read
    replaced = False

    def replacing_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            replacement = tmp_path / "replacement.csv"
            replacement.write_bytes(path.read_bytes())
            os.replace(replacement, path)
            replaced = True
        return chunk

    monkeypatch.setattr(os, "read", replacing_read)

    with pytest.raises(PsxAnnouncementImportError, match="changed while"):
        _importer(path).fetch()
