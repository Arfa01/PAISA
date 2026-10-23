from __future__ import annotations

from datetime import date, datetime, time, timezone

import pytest

from paisa.evidence.coverage import (
    EVIDENCE_COVERAGE_PROFILES,
    EvidenceCoverageError,
    EvidenceReadinessCriteria,
    assign_date_only_to_session,
    assign_evidence_to_session,
    compute_evidence_coverage,
)
from paisa.evidence.models import EvidenceRecord
from paisa.news.coverage import CoverageConfig, StockEvidence


UTC = timezone.utc
SHA256 = "a" * 64


def _at(day: str, clock: str = "09:00:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+00:00")


def _stock(
    values: tuple[date, ...] | None = None,
) -> StockEvidence:
    dates = values or tuple(
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
    if len(dates) == 6:
        split_dates = {
            "train": dates[:2],
            "validation": dates[2:4],
            "test": dates[4:],
        }
    else:
        split_dates = {
            "train": dates[:1],
            "validation": dates[1:2],
            "test": dates[2:],
        }
    return StockEvidence(
        symbol="OGDC",
        trading_dates=dates,
        split_dates=split_dates,
        reference={"fixture": True},
    )


def _gkg(identifier: str, available_at: datetime) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=identifier,
        evidence_type="gdelt_metadata",
        provider_id="gdelt-gkg-2.1",
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
        permission_policy_id="gdelt-policy",
        raw_evidence_reference=f"capture.zip#{identifier}",
        content_sha256=SHA256,
        entity_match_evidence=(
            {"field": "V2ORGANIZATIONS", "matched_alias": "OGDC"},
        ),
    )


def _psx_date(identifier: str, source_date: date) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=identifier,
        evidence_type="exchange_announcement",
        provider_id="psx-ogdc-announcements",
        provider_mode="manual_import",
        data_class="imported",
        source_name="Pakistan Stock Exchange",
        source_url="https://dps.psx.com.pk/company/OGDC",
        source_record_id=identifier,
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        title="Official announcement",
        source_published_at=None,
        source_published_date=source_date,
        available_at=None,
        fetched_at=_at("2026-10-23"),
        imported_at=_at("2026-10-23"),
        historical_availability_basis=(
            "official_exchange_date_conservative_next_session"
        ),
        time_precision="date_only",
        contemporaneous_source_id=identifier,
        contemporaneous_batch_at=None,
        permission_policy_id="psx-manual-policy",
        raw_evidence_reference=f"announcements.csv#{identifier}",
        content_sha256=SHA256,
        event_category="material_information",
        announcement_type="Material information",
    )


def _psx_timed(identifier: str, posted_at: datetime) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=identifier,
        evidence_type="exchange_announcement",
        provider_id="psx-ogdc-announcements",
        provider_mode="manual_import",
        data_class="imported",
        source_name="Pakistan Stock Exchange",
        source_url="https://dps.psx.com.pk/company/OGDC",
        source_record_id=identifier,
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        title="Official announcement",
        source_published_at=posted_at,
        source_published_date=None,
        available_at=posted_at,
        fetched_at=_at("2026-10-23"),
        imported_at=_at("2026-10-23"),
        historical_availability_basis="official_exchange_posted_timestamp",
        time_precision="second",
        contemporaneous_source_id=identifier,
        contemporaneous_batch_at=None,
        permission_policy_id="psx-manual-policy",
        raw_evidence_reference=f"announcements.csv#{identifier}",
        content_sha256=SHA256,
        event_category="financial_results",
        announcement_type="Financial results",
    )


def _publisher(
    identifier: str,
    available_at: datetime,
    *,
    ticker: str = "MEBL",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=identifier,
        evidence_type="publisher_news",
        provider_id="publisher-rss",
        provider_mode="rss",
        data_class="real",
        source_name="Example News",
        source_url="https://example.com/news",
        source_record_id=identifier,
        ticker=ticker,
        company_name=(
            "Oil and Gas Development Company Limited"
            if ticker == "OGDC"
            else "Meezan Bank Limited"
        ),
        title="Financial news",
        source_published_at=available_at,
        source_published_date=None,
        available_at=available_at,
        fetched_at=available_at,
        imported_at=available_at,
        historical_availability_basis="paisa_first_observed_at",
        time_precision="second",
        contemporaneous_source_id=identifier,
        contemporaneous_batch_at=None,
        permission_policy_id="rss-policy",
        raw_evidence_reference=f"feed.xml#{identifier}",
        content_sha256=SHA256,
    )


def _row(computation, profile: str, split: str):
    return next(
        row
        for row in computation.split_rows
        if row.profile == profile and row.split == split
    )


def test_date_only_mapping_is_strictly_next_session_for_trading_days_and_holidays():
    dates = tuple(
        date.fromisoformat(value)
        for value in ("2024-01-05", "2024-01-08", "2024-01-10")
    )
    stock = _stock(dates)
    config = CoverageConfig()

    friday = assign_date_only_to_session(date(2024, 1, 5), stock, config)
    weekend = assign_date_only_to_session(date(2024, 1, 6), stock, config)
    holiday_eve = assign_date_only_to_session(date(2024, 1, 8), stock, config)
    final_day = assign_date_only_to_session(date(2024, 1, 10), stock, config)
    before_period = assign_date_only_to_session(date(2023, 12, 31), stock, config)

    assert friday.session_date == date(2024, 1, 8)
    assert weekend.session_date == date(2024, 1, 8)
    assert holiday_eve.session_date == date(2024, 1, 10)
    assert friday.mapping_status == "DATE_ONLY_CONSERVATIVE_NEXT_SESSION"
    assert final_day.session_date is None
    assert final_day.mapping_status == "AFTER_PRICE_PERIOD"
    assert before_period.session_date is None
    assert before_period.mapping_status == "BEFORE_PRICE_PERIOD"


def test_timed_exchange_posting_obeys_exact_cutoff_and_next_session_policy():
    stock = _stock()
    config = CoverageConfig(
        market_timezone="Asia/Karachi", decision_cutoff=time(15, 30)
    )
    exact = assign_evidence_to_session(
        _psx_timed("psx-exact", _at("2024-01-05", "10:30:00")),
        stock,
        config,
    )
    late = assign_evidence_to_session(
        _psx_timed("psx-late", _at("2024-01-05", "10:30:01")),
        stock,
        config,
    )

    assert exact.session_date == date(2024, 1, 5)
    assert exact.mapping_status == "SAME_SESSION_BEFORE_CUTOFF"
    assert late.session_date == date(2024, 1, 8)
    assert late.mapping_status == "AFTER_CUTOFF_CARRIED_TO_NEXT_SESSION"


def test_historical_batch_time_not_later_fetch_time_drives_session_mapping():
    stock = _stock()
    record = _gkg("gkg-historical", _at("2024-01-03", "09:00:00"))

    assignment = assign_evidence_to_session(record, stock, CoverageConfig())

    assert record.fetched_at.year == 2026
    assert assignment.session_date == date(2024, 1, 3)
    assert assignment.mapping_status == "SAME_SESSION_BEFORE_CUTOFF"


def test_future_availability_is_never_backdated_to_publication_session():
    stock = _stock()
    # The defensible cutoff is the GKG batch time.  This fixture's batch is
    # after the stock period, so coverage must not backdate it into the period.
    record = _gkg("gkg-future-batch", _at("2024-01-09"))

    assignment = assign_evidence_to_session(record, stock, CoverageConfig())

    assert assignment.session_date is None
    assert assignment.mapping_status == "AFTER_PRICE_PERIOD"


def test_profiles_stay_separate_and_report_split_specific_counts_and_readiness():
    stock = _stock()
    records = (
        _gkg("gkg-train", _at("2024-01-01")),
        _gkg("gkg-validation", _at("2024-01-03")),
        _gkg("gkg-test", _at("2024-01-05")),
        _psx_date("psx-train", date(2024, 1, 1)),
        _psx_date("psx-validation", date(2024, 1, 2)),
        _psx_date("psx-test", date(2024, 1, 4)),
        _publisher("publisher-train", _at("2024-01-02")),
    )
    criteria = EvidenceReadinessCriteria(
        ready_minimum_coverage_percent_each_split=1,
        ready_minimum_covered_sessions_each_split=1,
        ready_maximum_missing_gap_sessions_each_split=10,
        partial_minimum_coverage_percent_each_split=1,
        partial_minimum_covered_sessions_each_split=1,
    )

    result = compute_evidence_coverage(
        records,
        stock,
        CoverageConfig(),
        criteria=criteria,
    )

    assert tuple(result.readiness) == EVIDENCE_COVERAGE_PROFILES
    assert _row(result, "gdelt_gkg_ogdc", "train").record_count == 1
    assert _row(result, "psx_ogdc_announcements", "validation").record_count == 1
    combined_test = _row(result, "combined_ogdc_evidence", "test")
    assert combined_test.sessions_with_records == 1
    assert combined_test.record_count == 2
    assert combined_test.date_only_records == 1
    assert combined_test.ogdc_specific_records == 2
    assert combined_test.evidence_type_counts == {
        "exchange_announcement": 1,
        "gdelt_metadata": 1,
        "publisher_news": 0,
    }
    assert _row(result, "general_financial_news", "train").record_count == 1
    assert _row(result, "general_financial_news", "validation").record_count == 0
    assert result.readiness["gdelt_gkg_ogdc"].status == "READY_FOR_AGGREGATION"
    assert (
        result.readiness["psx_ogdc_announcements"].status
        == "PARTIALLY_COVERED"
    )
    assert not result.readiness["psx_ogdc_announcements"].passed
    assert (
        result.readiness["combined_ogdc_evidence"].status
        == "READY_FOR_AGGREGATION"
    )
    assert (
        result.readiness["general_financial_news"].status
        == "INSUFFICIENT_OVERLAP"
    )

    gkg_mapping = next(
        row for row in result.mapping_rows if row.evidence_id == "gkg-train"
    )
    assert gkg_mapping.profiles == (
        "gdelt_gkg_ogdc",
        "combined_ogdc_evidence",
    )
    publisher_mapping = next(
        row for row in result.mapping_rows if row.evidence_id == "publisher-train"
    )
    assert publisher_mapping.profiles == ("general_financial_news",)


def test_cutoff_counters_and_monthly_coverage_are_json_serializable():
    stock = _stock()
    records = (
        _psx_timed("psx-before", _at("2024-01-05", "10:30:00")),
        _psx_timed("psx-after", _at("2024-01-05", "10:30:01")),
        _psx_date("psx-date", date(2024, 1, 4)),
    )

    result = compute_evidence_coverage(records, stock, CoverageConfig())
    test_row = _row(result, "psx_ogdc_announcements", "test")

    assert test_row.before_cutoff_records == 1
    assert test_row.after_cutoff_records == 1
    assert test_row.date_only_records == 1
    assert test_row.sessions_with_records == 2
    monthly = [
        row
        for row in result.monthly_rows
        if row.profile == "psx_ogdc_announcements" and row.split == "test"
    ]
    assert len(monthly) == 1
    assert monthly[0].record_count == 3
    payload = result.to_dict()
    assert payload["summary"]["date_only_policy"] == (
        "strictly_next_eligible_trading_session"
    )
    assert payload["mapping_rows"][0]["cutoff_eligible"] is True


def test_duplicate_evidence_identity_is_rejected():
    duplicate = _gkg("same-id", _at("2024-01-01"))
    with pytest.raises(EvidenceCoverageError, match="identities must be unique"):
        compute_evidence_coverage(
            (duplicate, duplicate),
            _stock(),
            CoverageConfig(),
        )


def test_stock_splits_must_be_chronological_and_cover_a_contiguous_target_tail():
    stock = _stock()
    reordered = StockEvidence(
        symbol=stock.symbol,
        trading_dates=stock.trading_dates,
        split_dates={
            "train": stock.split_dates["validation"],
            "validation": stock.split_dates["train"],
            "test": stock.split_dates["test"],
        },
        reference=stock.reference,
    )
    with pytest.raises(EvidenceCoverageError, match="chronologically ordered"):
        compute_evidence_coverage((), reordered, CoverageConfig())

    dates = stock.trading_dates
    gapped = StockEvidence(
        symbol=stock.symbol,
        trading_dates=dates,
        split_dates={
            "train": dates[:2],
            "validation": dates[3:4],
            "test": dates[4:],
        },
        reference=stock.reference,
    )
    with pytest.raises(EvidenceCoverageError, match="contiguous trailing"):
        compute_evidence_coverage((), gapped, CoverageConfig())


def test_non_announcement_ready_gate_is_required_in_every_split():
    stock = _stock()
    records = (
        _publisher("publisher-train", _at("2024-01-01"), ticker="OGDC"),
        _psx_timed("psx-validation", _at("2024-01-03")),
        _psx_timed("psx-test", _at("2024-01-05")),
    )
    criteria = EvidenceReadinessCriteria(
        ready_minimum_coverage_percent_each_split=1,
        ready_minimum_covered_sessions_each_split=1,
        ready_maximum_missing_gap_sessions_each_split=10,
        partial_minimum_coverage_percent_each_split=1,
        partial_minimum_covered_sessions_each_split=1,
    )

    result = compute_evidence_coverage(
        records,
        stock,
        CoverageConfig(),
        criteria=criteria,
    )

    readiness = result.readiness["combined_ogdc_evidence"]
    assert readiness.status == "PARTIALLY_COVERED"
    assert readiness.checks["non_announcement_evidence"]["passed"] is False
    assert readiness.checks["non_announcement_evidence"][
        "record_counts_by_split"
    ] == {"train": 1, "validation": 0, "test": 0}


def test_ogdc_publisher_news_is_general_news_and_combined_ogdc_evidence():
    result = compute_evidence_coverage(
        (_publisher("publisher-ogdc", _at("2024-01-05"), ticker="OGDC"),),
        _stock(),
        CoverageConfig(),
    )

    mapping = result.mapping_rows[0]
    assert mapping.profiles == (
        "general_financial_news",
        "combined_ogdc_evidence",
    )
    assert _row(result, "general_financial_news", "test").record_count == 1
    combined = _row(result, "combined_ogdc_evidence", "test")
    assert combined.record_count == 1
    assert combined.evidence_type_counts["publisher_news"] == 1
