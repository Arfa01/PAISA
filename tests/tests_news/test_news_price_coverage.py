from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import warnings
from zoneinfo import ZoneInfo

import pytest

import paisa.news.coverage as coverage_module
from paisa.news.models import NEWS_AVAILABILITY_POLICY_VERSION
from paisa.news.coverage import (
    CoverageAuditError,
    CoverageConfig,
    CoverageRunCollisionError,
    EnergySectorLinkContract,
    NewsEvidence,
    NewsOccurrence,
    ReadinessThresholds,
    StockEvidence,
    assign_to_session,
    calculate_readiness,
    compute_coverage,
    load_news_evidence,
    load_source_permission_policy,
    load_stock_evidence,
    parse_cutoff,
    publish_coverage_run,
    sha256_file,
    verify_coverage_pack,
    verify_coverage_run,
)
from paisa.news.pipeline import run_news_pipeline
from paisa.news.providers.fixture import FixtureNewsProvider
from paisa.news.providers.manual_csv import LICENSING_REVIEW_REQUIRED


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "audit_news_price_overlap.py"
VERIFY_SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "verify_news_coverage.py"
UTC = timezone.utc
KARACHI = ZoneInfo("Asia/Karachi")


def _utc(day: str, clock: str = "10:00:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+00:00")


def _local(day: str, clock: str = "15:00:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}").replace(
        tzinfo=KARACHI
    ).astimezone(UTC)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _occurrence(
    article_id: str,
    *,
    available_at: datetime | None,
    published_at: datetime | None | object = ...,
    duplicate_of: str | None = None,
    tickers: tuple[str, ...] = (),
    sectors: tuple[str, ...] = (),
    source: str = "Reviewed Wire",
    run_id: str = "NEWS-20240110-120000-tests",
    data_class: str = "real",
    source_url: str | None = None,
    source_observed_at: datetime | None = None,
) -> NewsOccurrence:
    if published_at is ...:
        published_at = available_at
    assert published_at is None or isinstance(published_at, datetime)
    return NewsOccurrence(
        occurrence_key=f"{run_id}:{article_id}",
        ingestion_run_id=run_id,
        article_id=article_id,
        canonical_article_id=duplicate_of or article_id,
        provider="reviewed-provider",
        provider_mode="real" if data_class == "real" else data_class,
        source_name=source,
        source_url=source_url or f"https://example.test/{article_id}",
        title=f"Article {article_id}",
        published_at=published_at,
        available_at=available_at,
        fetched_at=available_at,
        published_timezone="Asia/Karachi",
        language="en",
        tickers=tickers,
        companies=("Oil and Gas Development Company Limited",) if tickers else (),
        sectors=sectors,
        content_sha256=_digest(article_id),
        raw_artifact_reference=f"raw/{article_id}.json",
        data_class=data_class,
        duplicate_of=duplicate_of,
        entity_links=tuple(
            {
                "link_id": f"link-{article_id}-{ticker}",
                "article_id": article_id,
                "ticker": ticker,
                "company_name": "Oil and Gas Development Company Limited",
                "matched_alias": ticker,
                "match_method": "exact",
                "match_location": "title",
                "confidence": "exact",
            }
            for ticker in tickers
        ),
        source_observed_at=source_observed_at,
    )


def _news(
    *occurrences: NewsOccurrence,
    permission_review_required: bool = False,
    run_status: str = "PASS",
    provider_error_count: int = 0,
    database_provided: bool = True,
    verification_basis: str | None = None,
    energy_sector_link_contract: EnergySectorLinkContract | None = None,
) -> NewsEvidence:
    run_ids = sorted({item.ingestion_run_id for item in occurrences})
    return NewsEvidence(
        occurrences=tuple(occurrences),
        run_references=tuple(
            {
                "run_id": run_id,
                "status": run_status,
                "artifact_reference": f"news_ingestion/{run_id}",
                "normalized_articles_sha256": _digest(run_id),
                "provider_error_count": provider_error_count,
                "rejected_record_count": 0,
                "ingestion_complete": (
                    run_status == "PASS" and provider_error_count == 0
                ),
                "verification_basis": (
                    verification_basis
                    or (
                        "CURRENT_COMPLETION_REPLAY_AND_DATABASE_SEAL"
                        if database_provided
                        else "CURRENT_COMPLETION_REPLAY"
                    )
                ),
            }
            for run_id in run_ids
        ),
        database_fingerprints={
            "provided": database_provided,
            "files": (
                [
                    {
                        "label": "database",
                        "sha256": _digest("news-database"),
                        "size_bytes": 123,
                    }
                ]
                if database_provided
                else []
            ),
        },
        permission_review_required=permission_review_required,
        permission_notes=(
            ("LICENSING_REVIEW_REQUIRED",)
            if permission_review_required
            else ()
        ),
        energy_sector_link_contract=energy_sector_link_contract,
    )


def _energy_contract(
    contract_id: str,
    *occurrences: NewsOccurrence,
) -> EnergySectorLinkContract:
    return EnergySectorLinkContract(
        contract_id=contract_id,
        evidence_sha256=coverage_module._energy_sector_link_evidence_sha256(
            occurrences
        ),
    )


def _stock(
    days: tuple[str, ...],
    *,
    split_sizes: tuple[int, int, int] | None = None,
) -> StockEvidence:
    dates = tuple(date.fromisoformat(value) for value in days)
    if split_sizes is None:
        assert len(dates) >= 3
        first = len(dates) // 3
        second = len(dates) // 3
        split_sizes = (first, second, len(dates) - first - second)
    train_count, validation_count, test_count = split_sizes
    assert train_count + validation_count + test_count == len(dates)
    assert min(split_sizes) > 0
    train_end = train_count
    validation_end = train_count + validation_count
    return StockEvidence(
        symbol="OGDC",
        trading_dates=dates,
        split_dates={
            "train": dates[:train_end],
            "validation": dates[train_end:validation_end],
            "test": dates[validation_end:],
        },
        reference={
            "symbol": "OGDC",
            "price_date_range": {
                "start": dates[0].isoformat(),
                "end": dates[-1].isoformat(),
            },
            "price_artifact_reference": "vertical_slice/VSL-test/input_snapshot/prices.csv",
            "price_artifact_sha256": _digest("prices"),
            "split_manifest_sha256": _digest("splits"),
        },
    )


def _split_row(
    population: str,
    split: str,
    coverage: float,
    *,
    gap: int = 0,
    days_with: int = 1,
) -> dict[str, object]:
    return {
        "population": population,
        "split": split,
        "trading_days_with_articles": days_with,
        "coverage_percent": coverage,
        "longest_missing_gap_sessions": gap,
    }


def _readiness_rows(
    *,
    all_coverage: float,
    ogdc_coverage: float,
    full_days_with: int = 1,
    gap: int = 0,
    split_days_with: int = 2,
) -> list[dict[str, object]]:
    rows = [
        _split_row(
            "all_financial_news",
            "full_price_period",
            all_coverage,
            days_with=full_days_with,
            gap=gap,
        )
    ]
    for split in ("train", "validation", "test"):
        rows.append(
            _split_row(
                "all_financial_news",
                split,
                all_coverage,
                gap=gap,
                days_with=split_days_with,
            )
        )
        rows.append(
            _split_row(
                "ogdc_linked_news",
                split,
                ogdc_coverage,
                days_with=split_days_with,
            )
        )
    return rows


def _all_news_split(
    computation, split: str = "full_price_period"
) -> dict[str, object]:
    return next(
        row
        for row in computation.split_rows
        if row["population"] == "all_financial_news" and row["split"] == split
    )


def _ogdc_split(computation, split: str) -> dict[str, object]:
    return next(
        row
        for row in computation.split_rows
        if row["population"] == "ogdc_linked_news" and row["split"] == split
    )


def test_zero_overlap_is_detected_without_backdating_publication() -> None:
    stock = _stock(
        (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        )
    )
    article = _occurrence(
        "later",
        published_at=_utc("2024-02-01"),
        available_at=_utc("2024-02-02"),
        tickers=("OGDC",),
    )

    result = compute_coverage(_news(article), stock, CoverageConfig())

    assert result.final_summary["readiness_status"] == "NO_OVERLAP"
    assert _all_news_split(result)["trading_days_with_articles"] == 0
    assert result.final_summary["publication_calendar_overlap_with_prices"] == {
        "start": None,
        "end": None,
    }
    assert result.ogdc_article_rows[0]["mapping_status"] == "AFTER_PRICE_PERIOD"


def test_source_observation_is_audited_but_never_used_for_eligibility() -> None:
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    source_seen = _utc("2024-01-02", "08:00:00")
    article = _occurrence(
        "source-observed-only",
        published_at=None,
        source_observed_at=source_seen,
        available_at=_utc("2024-02-02"),
        tickers=("OGDC",),
    )

    result = compute_coverage(_news(article), stock, CoverageConfig())

    assert result.final_summary["readiness_status"] == "NO_OVERLAP"
    assert result.final_summary["source_observed_at_range"] == {
        "start": "2024-01-02T08:00:00Z",
        "end": "2024-01-02T08:00:00Z",
    }
    provenance = result.quality_report["source_observed_at"]
    assert provenance["complete_count"] == 1
    assert "never used for cutoff eligibility" in provenance[
        "eligibility_semantics"
    ]
    assert (
        provenance["eligibility_policy_version"]
        == NEWS_AVAILABILITY_POLICY_VERSION
    )
    assert provenance["eligible_for_feature_cutoff"] is False
    assert (
        result.final_summary["availability_policy_version"]
        == NEWS_AVAILABILITY_POLICY_VERSION
    )
    mapping = result.article_session_rows[0]
    assert mapping["source_observed_at"] == "2024-01-02T08:00:00Z"
    assert mapping["occurrence_mapping_status"] == "AFTER_PRICE_PERIOD"


def test_partial_overlap_and_exact_train_validation_test_denominators() -> None:
    stock = _stock(
        (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        ),
        split_sizes=(2, 2, 2),
    )
    result = compute_coverage(
        _news(
            _occurrence("train", available_at=_local("2024-01-02")),
            _occurrence(
                "test", available_at=_local("2024-01-08"), tickers=("OGDC",)
            ),
        ),
        stock,
        CoverageConfig(),
    )

    assert _all_news_split(result, "train")["coverage_percent"] == 50.0
    assert _all_news_split(result, "validation")["coverage_percent"] == 0.0
    assert _all_news_split(result, "test")["coverage_percent"] == 50.0
    assert _ogdc_split(result, "train")["coverage_percent"] == 0.0
    assert _ogdc_split(result, "test")["coverage_percent"] == 50.0
    assert result.final_summary["readiness_status"] == "INSUFFICIENT_OVERLAP"


def test_complete_overlap_is_one_hundred_percent_for_every_split() -> None:
    days = (
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    )
    thresholds = ReadinessThresholds(
        ready_minimum_all_news_coverage_percent_each_split=100,
        ready_minimum_ogdc_coverage_percent_each_split=100,
        ready_maximum_all_news_missing_gap_sessions=0,
        ready_minimum_distinct_sources=2,
        ready_minimum_publication_time_completeness_percent=100,
        ready_minimum_available_at_completeness_percent=100,
        partial_minimum_all_news_coverage_percent_each_split=1,
        partial_minimum_ogdc_coverage_percent_each_split=1,
    )
    occurrences = tuple(
        _occurrence(
            f"ogdc-{index}",
            available_at=_local(day),
            tickers=("OGDC",),
            source=f"Reviewed Wire {index % 2}",
        )
        for index, day in enumerate(days)
    )
    result = compute_coverage(
        _news(*occurrences),
        _stock(days, split_sizes=(2, 2, 2)),
        CoverageConfig(thresholds=thresholds),
    )

    for split in ("train", "validation", "test"):
        assert _all_news_split(result, split)["coverage_percent"] == 100.0
        assert _ogdc_split(result, split)["coverage_percent"] == 100.0
    assert _all_news_split(result)["canonical_articles"] == len(days)
    assert result.final_summary["readiness_status"] == "READY_FOR_AGGREGATION"

    artifact_only = compute_coverage(
        _news(*occurrences, database_provided=False),
        _stock(days, split_sizes=(2, 2, 2)),
        CoverageConfig(thresholds=thresholds),
    )
    assert artifact_only.final_summary["readiness_status"] == "PARTIALLY_COVERED"
    assert artifact_only.final_summary["database_provenance"]["passed"] is False

    legacy_unsealed = compute_coverage(
        _news(
            *occurrences,
            verification_basis="LEGACY_MANIFEST_CHECKSUM_INVENTORY",
        ),
        _stock(days, split_sizes=(2, 2, 2)),
        CoverageConfig(thresholds=thresholds),
    )
    assert legacy_unsealed.final_summary["readiness_status"] == "PARTIALLY_COVERED"
    assert legacy_unsealed.final_summary["database_provenance"][
        "database_fingerprinted"
    ] is True
    assert legacy_unsealed.final_summary["database_provenance"]["unsealed_run_ids"]


def test_unresolved_source_permission_prevents_ready_status() -> None:
    days = (
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    )
    thresholds = ReadinessThresholds(
        ready_minimum_all_news_coverage_percent_each_split=100,
        ready_minimum_ogdc_coverage_percent_each_split=100,
        ready_maximum_all_news_missing_gap_sessions=0,
        ready_minimum_distinct_sources=2,
        ready_minimum_publication_time_completeness_percent=100,
        ready_minimum_available_at_completeness_percent=100,
        partial_minimum_all_news_coverage_percent_each_split=1,
        partial_minimum_ogdc_coverage_percent_each_split=1,
    )
    evidence = _news(
        *(
            _occurrence(
                f"permission-{index}",
                available_at=_local(day),
                tickers=("OGDC",),
                source=f"Reviewed Wire {index % 2}",
            )
            for index, day in enumerate(days)
        ),
        permission_review_required=True,
    )

    result = compute_coverage(
        evidence,
        _stock(days, split_sizes=(2, 2, 2)),
        CoverageConfig(thresholds=thresholds),
    )

    assert result.final_summary["readiness_status"] == "PARTIALLY_COVERED"
    assert result.final_summary["source_permission_evidence"] == {
        "passed": False,
        "review_required": True,
        "notes": ["LICENSING_REVIEW_REQUIRED"],
    }


def test_partial_ingestion_evidence_prevents_ready_status() -> None:
    days = (
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    )
    thresholds = ReadinessThresholds(
        ready_minimum_all_news_coverage_percent_each_split=100,
        ready_minimum_ogdc_coverage_percent_each_split=100,
        ready_maximum_all_news_missing_gap_sessions=0,
        ready_minimum_distinct_sources=2,
        ready_minimum_publication_time_completeness_percent=100,
        ready_minimum_available_at_completeness_percent=100,
        partial_minimum_all_news_coverage_percent_each_split=1,
        partial_minimum_ogdc_coverage_percent_each_split=1,
    )
    evidence = _news(
        *(
            _occurrence(
                f"partial-{index}",
                available_at=_local(day),
                tickers=("OGDC",),
                source=f"Reviewed Wire {index % 2}",
            )
            for index, day in enumerate(days)
        ),
        run_status="PARTIAL",
        provider_error_count=1,
    )

    result = compute_coverage(
        evidence,
        _stock(days, split_sizes=(2, 2, 2)),
        CoverageConfig(thresholds=thresholds),
    )

    assert result.final_summary["readiness_status"] == "PARTIALLY_COVERED"
    completeness = result.final_summary["ingestion_completeness"]
    assert completeness["passed"] is False
    assert completeness["provider_error_count"] == len(
        evidence.run_references
    )
    assert completeness["incomplete_run_ids"]


def test_weekend_after_cutoff_and_timezone_mapping_prevent_future_leakage() -> None:
    stock = _stock(("2024-01-05", "2024-01-08", "2024-01-09"))
    config = CoverageConfig(
        market_timezone="Asia/Karachi", decision_cutoff=time(15, 30)
    )

    exact_cutoff = assign_to_session(_utc("2024-01-05", "10:30:00"), stock, config)
    after_cutoff = assign_to_session(_utc("2024-01-05", "10:30:01"), stock, config)
    weekend = assign_to_session(_local("2024-01-06", "12:00:00"), stock, config)

    assert exact_cutoff.session_date == date(2024, 1, 5)
    assert exact_cutoff.mapping_status == "SAME_SESSION_BEFORE_CUTOFF"
    assert after_cutoff.session_date == date(2024, 1, 8)
    assert after_cutoff.arrival_class == "after_decision_cutoff"
    assert after_cutoff.mapping_status == "AFTER_CUTOFF_CARRIED_TO_NEXT_SESSION"
    assert weekend.session_date == date(2024, 1, 8)
    assert weekend.mapping_status == "NON_TRADING_DAY_CARRIED_TO_NEXT_SESSION"

    leaked_if_publication_were_used = _occurrence(
        "no-leak",
        published_at=_local("2024-01-05", "10:00:00"),
        available_at=_local("2024-01-05", "16:00:00"),
    )
    result = compute_coverage(_news(leaked_if_publication_were_used), stock, config)
    friday, monday, _ = result.trading_day_rows
    assert friday["all_financial_canonical_article_count"] == 0
    assert monday["all_financial_canonical_article_count"] == 1


@pytest.mark.parametrize(
    "value",
    ("15:30:00.000001", "15:30:00.999999"),
)
def test_fractional_decision_cutoffs_are_rejected(value: str) -> None:
    with pytest.raises(CoverageAuditError, match="whole-second precision"):
        parse_cutoff(value)

    with pytest.raises(CoverageAuditError, match="whole-second precision"):
        CoverageConfig(decision_cutoff=time(15, 30, 0, 1))


def test_general_news_is_not_counted_as_ogdc_or_energy_news() -> None:
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    result = compute_coverage(
        _news(
            _occurrence("general", available_at=_local("2024-01-02")),
            _occurrence(
                "ogdc", available_at=_local("2024-01-02"), tickers=("OGDC",)
            ),
        ),
        stock,
        CoverageConfig(),
    )

    first = result.trading_day_rows[0]
    assert first["all_financial_canonical_article_count"] == 2
    assert first["ogdc_linked_canonical_article_count"] == 1
    assert first["energy_sector_canonical_article_count"] is None
    assert result.quality_report["energy_sector_linking"] == {
        "supported": False,
        "reason": (
            "No explicit versioned energy-sector link contract was supplied; "
            "mentioned_sectors values are not interpreted as coverage evidence."
        ),
    }
    assert result.final_summary["population_support"]["energy_sector_news"][
        "supported"
    ] is False
    assert result.final_summary["readiness_thresholds"]
    assert result.final_summary["readiness_evaluation"]


def test_arbitrary_sector_tags_do_not_enable_energy_coverage_without_contract() -> None:
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    result = compute_coverage(
        _news(
            _occurrence(
                "uncontracted-energy",
                available_at=_local("2024-01-02"),
                sectors=("energy",),
            )
        ),
        stock,
        CoverageConfig(),
    )

    assert result.final_summary["population_support"]["energy_sector_news"] == {
        "supported": False,
        "reason": (
            "No explicit versioned energy-sector link contract was supplied; "
            "mentioned_sectors values are not interpreted as coverage evidence."
        ),
    }
    assert result.trading_day_rows[0][
        "energy_sector_canonical_article_count"
    ] is None
    energy_splits = [
        row for row in result.split_rows if row["population"] == "energy_sector_news"
    ]
    assert energy_splits
    assert all(row["population_supported"] is False for row in energy_splits)
    assert all(row["coverage_percent"] is None for row in energy_splits)
    assert result.final_summary["full_price_coverage"]["energy_sector_news"] == {
        "supported": False,
        "trading_days_with_articles": None,
        "coverage_percent": None,
    }
    assert all(
        row["energy_sector_news"] is None
        for row in result.final_summary["split_coverage"].values()
    )
    assert not any(
        row["population"] == "energy_sector_news"
        for row in result.missing_interval_rows
    )


def test_explicit_versioned_energy_contract_enables_energy_measurement() -> None:
    occurrence = _occurrence(
        "contracted-energy",
        available_at=_local("2024-01-02"),
        sectors=("energy",),
    )
    contract = _energy_contract("ENERGY-LINKS-review-001", occurrence)
    result = compute_coverage(
        _news(
            occurrence,
            energy_sector_link_contract=contract,
        ),
        _stock(("2024-01-02", "2024-01-03", "2024-01-04")),
        CoverageConfig(),
    )

    support = result.final_summary["population_support"]["energy_sector_news"]
    assert support["supported"] is True
    assert support["contract"] == contract.to_dict()
    energy_full = next(
        row
        for row in result.split_rows
        if row["population"] == "energy_sector_news"
        and row["split"] == "full_price_period"
    )
    assert energy_full["population_supported"] is True
    assert energy_full["trading_days_with_articles"] == 1


def test_energy_contract_checksum_must_match_exact_occurrence_evidence() -> None:
    occurrence = _occurrence(
        "energy-evidence",
        available_at=_local("2024-01-02"),
        sectors=("energy",),
    )
    mismatched = EnergySectorLinkContract(
        contract_id="ENERGY-LINKS-mismatch-001",
        evidence_sha256=_digest("different rows"),
    )

    with pytest.raises(CoverageAuditError, match="does not match"):
        compute_coverage(
            _news(
                occurrence,
                energy_sector_link_contract=mismatched,
            ),
            _stock(("2024-01-02", "2024-01-03", "2024-01-04")),
            CoverageConfig(),
        )


def test_energy_contract_is_bound_into_published_input_and_offline_verification(
    tmp_path: Path,
) -> None:
    occurrence = _occurrence(
        "published-energy",
        available_at=_local("2024-01-02"),
        sectors=("energy",),
    )
    contract = _energy_contract("ENERGY-LINKS-published-001", occurrence)
    news = _news(
        occurrence,
        energy_sector_link_contract=contract,
    )
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    config = CoverageConfig()
    assessment = tmp_path / "assessment.md"
    assessment.write_text("# Reviewed energy links\n", encoding="utf-8")

    destination = publish_coverage_run(
        compute_coverage(news, stock, config),
        news=news,
        stock=stock,
        config=config,
        output_root=tmp_path / "coverage",
        source_assessment_path=assessment,
        run_id="COVERAGE-20240110-120000-energy",
        created_at=_utc("2024-01-10", "12:00:00"),
    )

    reference = json.loads(
        (destination / "imported_or_acquired_news_reference.json").read_text(
            encoding="utf-8"
        )
    )
    assert reference["energy_sector_link_contract"] == contract.to_dict()
    assert verify_coverage_pack(destination).status == "PASS"
    with pytest.raises(
        CoverageAuditError,
        match="requires the external energy-sector link contract",
    ):
        verify_coverage_run(
            destination,
            news_run_directories=[tmp_path / "unused-news"],
            vertical_slice_directory=tmp_path / "unused-vertical",
            model_training_directory=tmp_path / "unused-training",
            source_assessment_path=assessment,
            database_path=tmp_path / "unused-news.sqlite3",
        )

    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT_PATH),
            "--run-dir",
            str(destination),
            "--news-run-dir",
            str(tmp_path / "unused-news"),
            "--news-database",
            str(tmp_path / "unused-news.sqlite3"),
            "--vertical-slice-dir",
            str(tmp_path / "unused-vertical"),
            "--model-training-dir",
            str(tmp_path / "unused-training"),
            "--source-assessment",
            str(assessment),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["error"] == (
        "Source-bound verification requires the external energy-sector link contract"
    )
    assert str(tmp_path) not in completed.stdout


def test_missing_publication_time_remains_missing_but_availability_can_map() -> None:
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    result = compute_coverage(
        _news(
            _occurrence(
                "missing-published",
                published_at=None,
                available_at=_local("2024-01-02"),
            )
        ),
        stock,
        CoverageConfig(),
    )

    publication = result.quality_report["publication_time"]
    assert publication["missing_count"] == 1
    assert publication["range"] == {"start": None, "end": None}
    assert _all_news_split(result)["trading_days_with_articles"] == 1
    assert result.quality_report["available_at"]["complete_count"] == 1


def test_duplicate_occurrences_do_not_inflate_canonical_coverage() -> None:
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    canonical = _occurrence("canonical", available_at=_local("2024-01-02"))
    duplicate = _occurrence(
        "duplicate",
        available_at=_local("2024-01-02", "15:10:00"),
        duplicate_of="canonical",
        tickers=("OGDC",),
        run_id="NEWS-20240110-120100-tests",
    )

    result = compute_coverage(_news(canonical, duplicate), stock, CoverageConfig())

    first = result.trading_day_rows[0]
    assert result.quality_report["total_news_occurrences"] == 2
    assert result.quality_report["unique_canonical_articles"] == 1
    assert result.quality_report["duplicated_occurrences"] == 1
    assert first["all_financial_occurrence_count"] == 2
    assert first["all_financial_canonical_article_count"] == 1
    assert first["ogdc_linked_canonical_article_count"] == 1
    assert result.ogdc_article_rows[0]["occurrence_count"] == 1


def test_publisher_document_variants_are_reported_as_duplicate_candidates() -> None:
    result = compute_coverage(
        _news(
            _occurrence(
                "dawn-short",
                available_at=_local("2024-01-02"),
                source="Dawn",
                source_url="https://www.dawn.com/news/1767623",
            ),
            _occurrence(
                "dawn-slug",
                available_at=_local("2024-01-03"),
                source="Dawn",
                source_url="https://www.dawn.com/news/1767623/company-news",
            ),
        ),
        _stock(("2024-01-02", "2024-01-03", "2024-01-04")),
        CoverageConfig(),
    )

    identity = result.quality_report["canonical_identity"]
    assert identity["uniqueness_scope"] == (
        "current_exact_canonical_url_content_hash_or_title_source_time_rules"
    )
    assert identity["unique_under_current_rules"] == 2
    assert identity["publisher_document_candidate_group_count"] == 1
    assert identity["publisher_document_candidate_duplicate_rows"] == 1
    assert identity["publisher_document_candidate_groups"] == [
        {
            "publisher_document_key": "dawn.com:news:1767623",
            "canonical_article_ids": ["dawn-short", "dawn-slug"],
            "source_urls": [
                "https://www.dawn.com/news/1767623",
                "https://www.dawn.com/news/1767623/company-news",
            ],
        }
    ]
    assert result.final_summary["canonical_uniqueness_scope"] == identity[
        "uniqueness_scope"
    ]
    assert result.final_summary[
        "publisher_document_candidate_duplicate_groups"
    ] == 1
    mapped_documents = {
        (row["source_url"], row["publisher_document_key"])
        for row in result.article_session_rows
        if row["population"] == "all_financial_news"
    }
    assert mapped_documents == {
        (
            "https://www.dawn.com/news/1767623",
            "dawn.com:news:1767623",
        ),
        (
            "https://www.dawn.com/news/1767623/company-news",
            "dawn.com:news:1767623",
        ),
    }


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://www.dawn.com/news/1814669/changed-slug",
            "dawn.com:news:1814669",
        ),
        (
            "https://www.brecorder.com/news/40297919/changed-slug",
            "brecorder.com:news:40297919",
        ),
        (
            "https://www.thenews.com.pk/print/1185864-changed-slug",
            "thenews.com.pk:print:1185864",
        ),
        (
            "https://www.iranoilgas.com/news/details.aspx?id=25514&title=changed",
            "iranoilgas.com:news:25514",
        ),
    ],
)
def test_candidate_detection_uses_only_reviewed_publisher_document_patterns(
    url: str,
    expected: str,
) -> None:
    assert coverage_module._publisher_document_key(url) == expected


def test_later_duplicate_link_evidence_cannot_backdate_ogdc_coverage() -> None:
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    early_general = _occurrence(
        "canonical",
        available_at=_local("2024-01-02"),
    )
    later_linked = _occurrence(
        "later-linked",
        duplicate_of="canonical",
        available_at=_local("2024-01-03"),
        tickers=("OGDC",),
    )

    result = compute_coverage(
        _news(early_general, later_linked), stock, CoverageConfig()
    )

    first, second, _ = result.trading_day_rows
    assert first["all_financial_canonical_article_count"] == 1
    assert first["ogdc_linked_canonical_article_count"] == 0
    assert second["ogdc_linked_canonical_article_count"] == 1
    assert second["ogdc_linked_occurrence_count"] == 1
    assert result.ogdc_article_rows[0]["representative_article_id"] == "later-linked"
    assert result.ogdc_article_rows[0]["occurrence_count"] == 1


def test_article_session_mapping_is_exhaustive_and_population_specific() -> None:
    stock = _stock(
        ("2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"),
        split_sizes=(2, 1, 1),
    )
    general = _occurrence(
        "canonical",
        available_at=_local("2024-01-02", "14:00:00"),
    )
    later_ogdc_duplicate = _occurrence(
        "linked-duplicate",
        available_at=_local("2024-01-03", "14:00:00"),
        duplicate_of="canonical",
        tickers=("OGDC",),
    )
    energy = _occurrence(
        "energy",
        available_at=_local("2024-01-04", "16:00:00"),
        sectors=("energy",),
    )

    computation = compute_coverage(
        _news(
            general,
            later_ogdc_duplicate,
            energy,
            energy_sector_link_contract=_energy_contract(
                "ENERGY-LINKS-mapping-001",
                general,
                later_ogdc_duplicate,
                energy,
            ),
        ),
        stock,
        CoverageConfig(),
    )

    rows = computation.article_session_rows
    assert len(rows) == 3 * 3
    assert {
        (row["occurrence_key"], row["population"]) for row in rows
    } == {
        (item.occurrence_key, population)
        for item in (general, later_ogdc_duplicate, energy)
        for population in (
            "all_financial_news",
            "ogdc_linked_news",
            "energy_sector_news",
        )
    }
    lookup = {(row["article_id"], row["population"]): row for row in rows}
    general_ogdc = lookup[("canonical", "ogdc_linked_news")]
    assert general_ogdc["occurrence_in_population"] is False
    assert general_ogdc["canonical_representative_article_id"] == "linked-duplicate"
    assert general_ogdc["canonical_assigned_trading_date"] == "2024-01-03"
    assert general_ogdc["is_population_canonical_representative"] is False
    linked_ogdc = lookup[("linked-duplicate", "ogdc_linked_news")]
    assert linked_ogdc["occurrence_in_population"] is True
    assert linked_ogdc["is_duplicate_occurrence"] is True
    assert linked_ogdc["is_population_canonical_representative"] is True
    energy_row = lookup[("energy", "energy_sector_news")]
    assert energy_row["population_supported"] is True
    assert energy_row["occurrence_mapping_status"] == (
        "AFTER_CUTOFF_CARRIED_TO_NEXT_SESSION"
    )
    assert energy_row["occurrence_assigned_trading_date"] == "2024-01-05"


def test_long_missing_interval_counts_observed_sessions_not_calendar_days() -> None:
    days = (
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
    )
    result = compute_coverage(
        _news(
            _occurrence("first", available_at=_local(days[0])),
            _occurrence("last", available_at=_local(days[-1])),
        ),
        _stock(days, split_sizes=(2, 2, 3)),
        CoverageConfig(),
    )

    full = _all_news_split(result)
    assert full["longest_missing_gap_sessions"] == 5
    interval = next(
        row
        for row in result.missing_interval_rows
        if row["population"] == "all_financial_news"
        and row["split"] == "full_price_period"
    )
    assert interval == {
        "population": "all_financial_news",
        "split": "full_price_period",
        "start_date": "2024-01-03",
        "end_date": "2024-01-09",
        "missing_trading_sessions": 5,
    }


@pytest.mark.parametrize(
    (
        "all_coverage",
        "ogdc_coverage",
        "full_days_with",
        "gap",
        "sources",
        "publication",
        "availability",
        "expected",
    ),
    [
        (0, 0, 0, 99, 1, 100, 100, "NO_OVERLAP"),
        (4, 0, 1, 20, 1, 100, 100, "INSUFFICIENT_OVERLAP"),
        (10, 5, 1, 20, 1, 100, 100, "PARTIALLY_COVERED"),
        (90, 30, 1, 5, 2, 100, 100, "READY_FOR_AGGREGATION"),
    ],
)
def test_all_readiness_statuses_are_calculated_from_explicit_gates(
    all_coverage: float,
    ogdc_coverage: float,
    full_days_with: int,
    gap: int,
    sources: int,
    publication: float,
    availability: float,
    expected: str,
) -> None:
    status, evaluation = calculate_readiness(
        _readiness_rows(
            all_coverage=all_coverage,
            ogdc_coverage=ogdc_coverage,
            full_days_with=full_days_with,
            gap=gap,
        ),
        thresholds=ReadinessThresholds(),
        publication_completeness_percent=publication,
        available_at_completeness_percent=availability,
        distinct_sources=sources,
    )

    assert status == expected
    assert evaluation["passed"] is (expected == "READY_FOR_AGGREGATION")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "ready_minimum_all_news_coverage_percent_each_split",
            4.999,
            "at least 5.0",
        ),
        (
            "ready_minimum_ogdc_coverage_percent_each_split",
            0.999,
            "at least 1.0",
        ),
        (
            "ready_minimum_publication_time_completeness_percent",
            0,
            "strictly positive",
        ),
        (
            "ready_minimum_available_at_completeness_percent",
            0,
            "strictly positive",
        ),
        (
            "ready_minimum_publication_time_completeness_percent",
            True,
            "must be a finite number",
        ),
        (
            "partial_minimum_all_news_coverage_percent_each_split",
            0,
            "strictly positive",
        ),
        ("ready_minimum_distinct_sources", 1, "at least 2"),
        ("ready_minimum_distinct_sources", True, "must be an integer"),
        ("ready_maximum_all_news_missing_gap_sessions", 1.5, "must be an integer"),
    ],
)
def test_readiness_thresholds_enforce_types_and_positive_safety_floors(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(CoverageAuditError, match=message):
        ReadinessThresholds(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("ready_field", "partial_field"),
    [
        (
            "ready_minimum_all_news_coverage_percent_each_split",
            "partial_minimum_all_news_coverage_percent_each_split",
        ),
        (
            "ready_minimum_ogdc_coverage_percent_each_split",
            "partial_minimum_ogdc_coverage_percent_each_split",
        ),
    ],
)
def test_ready_coverage_threshold_cannot_be_weaker_than_partial_threshold(
    ready_field: str,
    partial_field: str,
) -> None:
    values = {
        ready_field: 10.0,
        partial_field: 20.0,
    }
    with pytest.raises(CoverageAuditError, match="cannot be weaker"):
        ReadinessThresholds(**values)  # type: ignore[arg-type]


def test_non_configurable_minimum_evidence_guard_blocks_sparse_ready_result() -> None:
    rows = _readiness_rows(
        all_coverage=100.0,
        ogdc_coverage=100.0,
        full_days_with=3,
        gap=0,
        split_days_with=1,
    )
    # The configurable percentage gates pass, but every model split has only one
    # covered session. That is descriptive overlap, not readiness.
    status, evaluation = calculate_readiness(
        rows,
        thresholds=ReadinessThresholds(
            ready_minimum_all_news_coverage_percent_each_split=5.0,
            ready_minimum_ogdc_coverage_percent_each_split=1.0,
            ready_maximum_all_news_missing_gap_sessions=0,
            ready_minimum_distinct_sources=2,
            ready_minimum_publication_time_completeness_percent=1.0,
            ready_minimum_available_at_completeness_percent=1.0,
            partial_minimum_all_news_coverage_percent_each_split=1.0,
            partial_minimum_ogdc_coverage_percent_each_split=1.0,
        ),
        publication_completeness_percent=100.0,
        available_at_completeness_percent=100.0,
        distinct_sources=2,
    )

    assert status == "PARTIALLY_COVERED"
    guard = evaluation["checks"]["non_configurable_minimum_evidence"]
    assert guard["passed"] is False
    assert guard["minimum_covered_sessions_each_split"] == 2
    assert guard["actual_all_news_minimum"] == 1
    assert guard["actual_ogdc_minimum"] == 1


NEWS_FIELDS = (
    "article_id",
    "provider",
    "source_name",
    "source_url",
    "canonical_url",
    "title",
    "published_at",
    "published_timezone",
    "available_at",
    "fetched_at",
    "language",
    "mentioned_tickers",
    "mentioned_companies",
    "mentioned_sectors",
    "content_sha256",
    "duplicate_of",
    "ingestion_run_id",
    "raw_artifact_reference",
    "data_class",
)


def _write_news_pack(
    root: Path,
    *,
    run_id: str = "NEWS-20240110-120000-loader",
    provider: str = "reviewed-provider",
    provider_mode: str = "real",
    data_class: str = "real",
    url: str = "https://example.test/article",
    licensing_usage_note: str | None = None,
    provider_options: dict[str, object] | None = None,
) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    raw_dir = run_dir / "raw"
    raw_dir.mkdir()
    raw_record = raw_dir / "article-1.json"
    raw_record.write_text(
        json.dumps({"article_id": "article-1"}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = run_dir / "normalized_articles.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEWS_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "article_id": "article-1",
                "provider": provider,
                "source_name": "Reviewed Wire",
                "source_url": url,
                "canonical_url": url,
                "title": "OGDC historical evidence",
                "published_at": "2024-01-02T09:00:00Z",
                "published_timezone": "UTC",
                "available_at": "2024-01-02T09:05:00Z",
                "fetched_at": "2024-01-02T09:05:00Z",
                "language": "en",
                "mentioned_tickers": '["OGDC"]',
                "mentioned_companies": (
                    '["Oil and Gas Development Company Limited"]'
                ),
                "mentioned_sectors": "[]",
                "content_sha256": _digest("article-1"),
                "duplicate_of": "",
                "ingestion_run_id": run_id,
                "raw_artifact_reference": "raw/article-1.json",
                "data_class": data_class,
            }
        )
    links_path = run_dir / "article_entity_links.csv"
    with links_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "link_id",
                "article_id",
                "ticker",
                "company_name",
                "matched_alias",
                "match_method",
                "match_location",
                "confidence",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "link_id": "link-article-1-ogdc",
                "article_id": "article-1",
                "ticker": "OGDC",
                "company_name": "Oil and Gas Development Company Limited",
                "matched_alias": "OGDC",
                "match_method": "exact",
                "match_location": "title",
                "confidence": "exact",
            }
        )
    inventory = [
        {
            "path": "article_entity_links.csv",
            "sha256": sha256_file(links_path),
            "size_bytes": links_path.stat().st_size,
        },
        {
            "path": "normalized_articles.csv",
            "sha256": sha256_file(csv_path),
            "size_bytes": csv_path.stat().st_size,
        }
    ]
    if licensing_usage_note is not None or provider_options is not None:
        effective_options = dict(provider_options or {})
        if licensing_usage_note is not None:
            effective_options["licensing_usage_note"] = licensing_usage_note
        provider_config_path = run_dir / "provider_config.json"
        provider_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "provider_id": provider,
                    "provider_mode": provider_mode,
                    "provider_options": effective_options,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        inventory.append(
            {
                "path": "provider_config.json",
                "sha256": sha256_file(provider_config_path),
                "size_bytes": provider_config_path.stat().st_size,
            }
        )
    inventory.append(
        {
            "path": "raw/article-1.json",
            "sha256": sha256_file(raw_record),
            "size_bytes": raw_record.stat().st_size,
        }
    )
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "PASS",
        "provider": provider,
        "provider_mode": provider_mode,
        "quality_status": "PASS",
        "artifact_inventory": inventory,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def _rewrite_normalized_row(run_dir: Path, **updates: str) -> None:
    csv_path = run_dir / "normalized_articles.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(NEWS_FIELDS)
    assert len(rows) == 1
    rows[0].update(updates)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEWS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["artifact_inventory"]
        if item["path"] == "normalized_articles.csv"
    )
    entry["sha256"] = sha256_file(csv_path)
    entry["size_bytes"] = csv_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_stock_packs(root: Path) -> tuple[Path, Path]:
    vertical_id = "VSL-20240110-OGDC-loader"
    training_id = "TRN-20240110-120000-loader"
    vertical = root / vertical_id
    training = root / training_id
    (vertical / "input_snapshot").mkdir(parents=True)
    training.mkdir(parents=True)

    dates = ("2024-01-02", "2024-01-03", "2024-01-04")
    prices = vertical / "input_snapshot" / "prices.csv"
    with prices.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("date", "ticker", "source"))
        writer.writeheader()
        for value in dates:
            writer.writerow({"date": value, "ticker": "OGDC", "source": "psx_dps"})

    split_manifest = {
        "schema_version": "1.0",
        "method": "chronological_target_date_70_15_15_no_shuffle",
        "target_dates": {
            "train": [dates[0]],
            "validation": [dates[1]],
            "test": [dates[2]],
        },
        "row_counts": {"train": 1, "validation": 1, "test": 1},
    }
    split_path = training / "split_manifest.json"
    split_path.write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    price_sha = sha256_file(prices)
    data_manifest = {
        "schema_version": "1.0",
        "symbol": "OGDC",
        "provider": "psx_dps",
        "price_rows": len(dates),
        "acquisition": {
            "input_snapshot_sha256": price_sha,
            "mode": "reviewed_test_evidence",
            "provider": "psx_dps",
        },
    }
    data_manifest_path = training / "data_manifest.json"
    data_manifest_path.write_text(
        json.dumps(data_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    training_manifest = {
        "schema_version": "1.0",
        "run_id": training_id,
        "symbol": "OGDC",
        "artifact_inventory": {
            "algorithm": "sha256",
            "files": [
                {
                    "path": "data_manifest.json",
                    "sha256": sha256_file(data_manifest_path),
                },
                {
                    "path": "split_manifest.json",
                    "sha256": sha256_file(split_path),
                }
            ],
        },
    }
    (training / "run_manifest.json").write_text(
        json.dumps(training_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    vertical_manifest = {
        "schema_version": "1.0",
        "vertical_slice_run_id": vertical_id,
        "status": "PASS",
        "symbol": "OGDC",
        "model_training_run_id": training_id,
        "data": {
            "is_real_data": True,
            "input_prices_sha256": price_sha,
            "actual_ogdc_date_range": {"start": dates[0], "end": dates[-1]},
            "acquisition_mode": "reviewed_test_evidence",
            "source": "psx_dps",
        },
        "artifacts": [
            {
                "path": "input_snapshot/prices.csv",
                "sha256": price_sha,
            }
        ],
    }
    (vertical / "manifest.json").write_text(
        json.dumps(vertical_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return vertical, training


def test_real_mode_rejects_fixture_pack_and_test_mode_keeps_its_label(
    tmp_path: Path,
) -> None:
    run_dir = _write_news_pack(
        tmp_path, provider_mode="fixture", data_class="fixture"
    )

    with pytest.raises(CoverageAuditError, match="forbidden in real mode"):
        load_news_evidence([run_dir], real_mode=True)

    loaded = load_news_evidence(
        [run_dir], real_mode=False, allow_legacy_unsealed=True
    )
    assert loaded.occurrences[0].data_class == "fixture"
    assert loaded.occurrences[0].provider_mode == "fixture"


def test_unsealed_legacy_pack_requires_explicit_compatibility_opt_in(
    tmp_path: Path,
) -> None:
    run_dir = _write_news_pack(tmp_path)

    with pytest.raises(CoverageAuditError, match="compatibility opt-in"):
        load_news_evidence([run_dir])


def test_database_seals_and_fingerprint_use_one_immutable_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "news-runs"
    database = tmp_path / "news.sqlite3"
    started = _utc("2024-01-02", "09:05:00")
    provider_rows = [
        {
            "source_name": "Fixture Business",
            "source_url": "https://example.test/ogdc",
            "title": "OGDC reports an update",
            "summary": "Oil and Gas Development Company Limited reports an update.",
            "published_at": "2024-01-02T09:00:00Z",
            "published_timezone": "UTC",
            "available_at": "2024-01-02T09:00:00Z",
            "language": "en",
            "provider_record_id": "fixture-ogdc-1",
        }
    ]
    run_ids = (
        "NEWS-20240102-090500-snapshot-one",
        "NEWS-20240102-090501-snapshot-two",
    )
    for run_id in run_ids:
        summary = run_news_pipeline(
            FixtureNewsProvider(provider_rows, fetched_at=started),
            output_dir=artifacts,
            database_path=database,
            storage_root=tmp_path,
            run_id=run_id,
            started_at=started,
        )
        assert summary.run_status == "PASS"

    original_verify = coverage_module.verify_news_run
    verified_paths: list[Path] = []
    snapshot_digests: list[str] = []
    concurrent_marker_absent: list[bool] = []

    def verify_and_mutate_source(run_dir, *, database_path=None):
        assert database_path is not None
        snapshot = Path(database_path)
        verified_paths.append(snapshot)
        result = original_verify(run_dir, database_path=snapshot)
        if len(verified_paths) == 1:
            with sqlite3.connect(database) as live:
                live.execute("CREATE TABLE concurrent_change (value TEXT)")
                live.commit()
        with sqlite3.connect(f"file:{snapshot.as_posix()}?mode=ro", uri=True) as frozen:
            concurrent_marker_absent.append(
                frozen.execute(
                    "SELECT 1 FROM sqlite_master "
                    "WHERE type = 'table' AND name = 'concurrent_change'"
                ).fetchone()
                is None
            )
        snapshot_digests.append(sha256_file(snapshot))
        return result

    monkeypatch.setattr(coverage_module, "verify_news_run", verify_and_mutate_source)
    evidence = load_news_evidence(
        [artifacts / run_id for run_id in run_ids],
        real_mode=False,
        database_path=database,
    )

    assert len(verified_paths) == 2
    assert len(set(verified_paths)) == 1
    assert verified_paths[0] != database.resolve()
    assert concurrent_marker_absent == [True, True]
    assert len(set(snapshot_digests)) == 1
    fingerprint = evidence.database_fingerprints
    assert fingerprint["snapshot_method"] == (
        "sqlite_online_backup_read_only_source"
    )
    assert fingerprint["wal_state_folded_into_snapshot"] is True
    assert fingerprint["files"] == [
        {
            "label": "database",
            "role": "immutable_verification_snapshot",
            "sha256": snapshot_digests[0],
            "size_bytes": fingerprint["files"][0]["size_bytes"],
        }
    ]
    assert all(
        reference["verification_basis"]
        == "CURRENT_COMPLETION_REPLAY_AND_DATABASE_SEAL"
        for reference in evidence.run_references
    )
    assert all(
        reference["availability_policy_version"]
        == NEWS_AVAILABILITY_POLICY_VERSION
        for reference in evidence.run_references
    )
    assert not verified_paths[0].exists()
    with sqlite3.connect(database) as live:
        assert live.execute(
            "SELECT 1 FROM sqlite_master "
            "WHERE type = 'table' AND name = 'concurrent_change'"
        ).fetchone() == (1,)


def test_source_permission_policy_binds_reviewed_assessment_bytes(
    tmp_path: Path,
) -> None:
    assessment = tmp_path / "HISTORICAL_NEWS_SOURCE_ASSESSMENT.md"
    assessment.write_text("# Reviewed\n", encoding="utf-8")
    policy_path = tmp_path / "permissions.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "assessment_document": assessment.name,
                "assessment_sha256": sha256_file(assessment),
                "decisions": [
                    {
                        "decision_id": "reviewed-provider-v1",
                        "provider": "reviewed-provider",
                        "provider_mode": "real",
                        "status": "PERMITTED",
                        "scope": "Metadata evidence only.",
                        "required_provider_options": {},
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    policy = load_source_permission_policy(policy_path, assessment)
    run_dir = _write_news_pack(tmp_path / "news")
    evidence = load_news_evidence(
        [run_dir],
        allow_legacy_unsealed=True,
        source_permission_policy=policy,
    )
    assert evidence.permission_review_required is False
    assert evidence.run_references[0]["source_permission_decision"]["passed"] is True

    assessment.write_text("# Changed after review\n", encoding="utf-8")
    with pytest.raises(CoverageAuditError, match="assessment bytes"):
        load_source_permission_policy(policy_path, assessment)


def test_committed_gdelt_policy_enforces_exact_attribution_text(
    tmp_path: Path,
) -> None:
    assessment = REPOSITORY_ROOT / "docs" / "HISTORICAL_NEWS_SOURCE_ASSESSMENT.md"
    policy = load_source_permission_policy(
        REPOSITORY_ROOT / "configs" / "news" / "historical_source_permissions.json",
        assessment,
    )
    attribution = "Data source: GDELT Project (https://www.gdeltproject.org/)."
    licensing_note = (
        "GDELT-released metadata is available for unrestricted use with GDELT "
        "citation; linked publisher bodies and images are not fetched or stored."
    )
    options = {
        "article_page_fetching": "disabled",
        "attribution_text": attribution,
        "endpoint": "https://api.gdeltproject.org/api/v2/doc/doc",
        "environment_authentication": "disabled",
        "provider_mode": "historical_api",
    }
    allowed_run_id = "NEWS-20261023-060000-gdelt-ogdc2"
    permitted_run = _write_news_pack(
        tmp_path / "permitted",
        run_id=allowed_run_id,
        provider="gdelt-doc",
        licensing_usage_note=licensing_note,
        provider_options=options,
    )

    permitted = load_news_evidence(
        [permitted_run],
        allow_legacy_unsealed=True,
        source_permission_policy=policy,
    )

    assert permitted.permission_review_required is False
    assert permitted.run_references[0]["source_permission_decision"]["passed"] is True

    unreviewed_run = _write_news_pack(
        tmp_path / "not-allowlisted",
        run_id="NEWS-20261023-060001-unreviewed",
        provider="gdelt-doc",
        licensing_usage_note=licensing_note,
        provider_options=options,
    )
    unreviewed = load_news_evidence(
        [unreviewed_run],
        allow_legacy_unsealed=True,
        source_permission_policy=policy,
    )
    unreviewed_decision = unreviewed.run_references[0][
        "source_permission_decision"
    ]
    assert unreviewed.permission_review_required is True
    assert unreviewed_decision["status"] == "RUN_NOT_PERMITTED"

    mismatched_run = _write_news_pack(
        tmp_path / "mismatched",
        run_id=allowed_run_id,
        provider="gdelt-doc",
        licensing_usage_note=licensing_note,
        provider_options={**options, "attribution_text": "GDELT"},
    )
    mismatched = load_news_evidence(
        [mismatched_run],
        allow_legacy_unsealed=True,
        source_permission_policy=policy,
    )
    decision = mismatched.run_references[0]["source_permission_decision"]
    assert mismatched.permission_review_required is True
    assert decision["status"] == "PROVIDER_CONFIGURATION_MISMATCH"
    assert "attribution_text" in decision["reason"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"data_class": "imported"}, "provider mode and row data class"),
        ({"provider": "different-provider"}, "provider identity"),
        ({"mentioned_tickers": "[]"}, "ticker links disagree"),
        ({"mentioned_companies": "[]"}, "company links disagree"),
        (
            {"available_at": "2024-01-02T09:04:59Z"},
            "availability precedes its evidence",
        ),
    ],
)
def test_news_loader_rejects_contradictory_row_evidence(
    tmp_path: Path,
    updates: dict[str, str],
    message: str,
) -> None:
    run_dir = _write_news_pack(tmp_path)
    _rewrite_normalized_row(run_dir, **updates)

    with pytest.raises(CoverageAuditError, match=message):
        load_news_evidence([run_dir], allow_legacy_unsealed=True)


@pytest.mark.parametrize(
    "licensing_usage_note",
    [None, LICENSING_REVIEW_REQUIRED],
)
def test_real_mode_rejects_import_without_reviewed_licensing_evidence(
    tmp_path: Path,
    licensing_usage_note: str | None,
) -> None:
    run_dir = _write_news_pack(
        tmp_path,
        provider_mode="imported",
        data_class="imported",
        licensing_usage_note=licensing_usage_note,
    )

    with pytest.raises(CoverageAuditError, match="licensing/usage note"):
        load_news_evidence(
            [run_dir], real_mode=True, allow_legacy_unsealed=True
        )


def test_free_form_import_note_does_not_self_authorize_readiness(
    tmp_path: Path,
) -> None:
    run_dir = _write_news_pack(
        tmp_path,
        provider_mode="imported",
        data_class="imported",
        licensing_usage_note="Operator claims this dataset is licensed.",
    )

    evidence = load_news_evidence(
        [run_dir], real_mode=True, allow_legacy_unsealed=True
    )

    assert evidence.permission_review_required is True
    decision = evidence.run_references[0]["source_permission_decision"]
    assert decision["status"] == "UNVERIFIED_POLICY"
    assert decision["passed"] is False


def test_news_and_stock_loaders_reject_checksum_tampering(tmp_path: Path) -> None:
    news_dir = _write_news_pack(tmp_path / "news")
    csv_path = news_dir / "normalized_articles.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    with pytest.raises(CoverageAuditError, match="size mismatch|checksum mismatch"):
        load_news_evidence([news_dir], allow_legacy_unsealed=True)

    raw_news_dir = _write_news_pack(
        tmp_path / "raw-news",
        run_id="NEWS-20240110-120001-loader",
    )
    raw_path = raw_news_dir / "raw" / "article-1.json"
    raw_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(CoverageAuditError, match="size mismatch|checksum mismatch"):
        load_news_evidence([raw_news_dir], allow_legacy_unsealed=True)

    vertical, training = _write_stock_packs(tmp_path / "stock")
    loaded = load_stock_evidence(vertical, training)
    assert loaded.split_dates == {
        "train": (date(2024, 1, 2),),
        "validation": (date(2024, 1, 3),),
        "test": (date(2024, 1, 4),),
    }
    price_path = vertical / "input_snapshot" / "prices.csv"
    price_path.write_text(
        price_path.read_text(encoding="utf-8") + "2024-01-05,OGDC,psx_dps\n",
        encoding="utf-8",
    )
    with pytest.raises(CoverageAuditError, match="checksum mismatch"):
        load_stock_evidence(vertical, training)


def test_stock_loader_rejects_mismatched_training_price_provenance(
    tmp_path: Path,
) -> None:
    vertical, training = _write_stock_packs(tmp_path)
    data_manifest_path = training / "data_manifest.json"
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    data_manifest["acquisition"]["input_snapshot_sha256"] = "0" * 64
    data_manifest_path.write_text(
        json.dumps(data_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_manifest_path = training / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    data_entry = next(
        item
        for item in run_manifest["artifact_inventory"]["files"]
        if item["path"] == "data_manifest.json"
    )
    data_entry["sha256"] = sha256_file(data_manifest_path)
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CoverageAuditError, match="frozen price evidence"):
        load_stock_evidence(vertical, training)


def test_published_pack_is_portable_credential_free_immutable_and_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    secrets = {
        "api_key": "API-KEY-DO-NOT-PERSIST",
        "auth": "AUTH-DO-NOT-PERSIST",
        "session": "SESSION-DO-NOT-PERSIST",
        "signature": "SIGNATURE-DO-NOT-PERSIST",
        "signed": "SIGNED-DO-NOT-PERSIST",
        "jwt": "JWT-DO-NOT-PERSIST",
        "fragment": "FRAGMENT-DO-NOT-PERSIST",
        "tracking": "TRACKING-DO-NOT-PERSIST",
    }
    source_url = (
        "https://example.test/article?id=123&author=desk"
        f"&api_key={secrets['api_key']}"
        f"&auth={secrets['auth']}"
        f"&session={secrets['session']}"
        f"&signature={secrets['signature']}"
        f"&signed={secrets['signed']}"
        f"&jwt={secrets['jwt']}"
        f"&utm_source={secrets['tracking']}"
        f"#access_token={secrets['fragment']}"
    )
    news_dir = _write_news_pack(
        tmp_path / "captured-news",
        url=source_url,
    )
    vertical, training = _write_stock_packs(tmp_path / "captured-stock")
    news = load_news_evidence(
        [news_dir], database_path=None, allow_legacy_unsealed=True
    )
    stock = load_stock_evidence(vertical, training)
    config = CoverageConfig()
    first_computation = compute_coverage(news, stock, config)
    second_computation = compute_coverage(news, stock, config)
    assert first_computation == second_computation

    assessment = tmp_path / "private" / "source-assessment.md"
    assessment.parent.mkdir()
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")
    output = tmp_path / "published"
    created_at = _utc("2024-01-10", "12:00:00")
    first_id = "COVERAGE-20240110-120000-first"
    second_id = "COVERAGE-20240110-120000-second"
    first = publish_coverage_run(
        first_computation,
        news=news,
        stock=stock,
        config=config,
        output_root=output,
        source_assessment_path=assessment,
        run_id=first_id,
        created_at=created_at,
    )
    second = publish_coverage_run(
        second_computation,
        news=news,
        stock=stock,
        config=config,
        output_root=output,
        source_assessment_path=assessment,
        run_id=second_id,
        created_at=created_at,
    )

    first_verification = verify_coverage_pack(first)
    second_verification = verify_coverage_pack(second)
    assert first_verification.to_dict()["verification"] == "PASS"
    assert first_verification.run_id == first_id
    assert second_verification.run_id == second_id

    first_completion = json.loads((first / "completion.json").read_text())
    assert first_completion["manifest_sha256"] == sha256_file(
        first / "run_manifest.json"
    )
    assert first_completion["summary_sha256"] == sha256_file(
        first / "final_summary.json"
    )

    for filename in (
        "overlap_by_trading_day.csv",
        "overlap_by_month.csv",
        "split_coverage.csv",
        "missing_intervals.csv",
        "article_session_mappings.csv",
        "ogdc_linked_articles.csv",
        "quality_report.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    first_manifest = json.loads((first / "run_manifest.json").read_text())
    second_manifest = json.loads((second / "run_manifest.json").read_text())
    inventoried_paths = [
        item["path"] for item in first_manifest["artifact_inventory"]
    ]
    assert inventoried_paths == sorted(
        path.name
        for path in first.iterdir()
        if path.name not in {"run_manifest.json", "completion.json"}
    )
    assert first_completion["artifact_count"] == len(inventoried_paths)
    assert first_completion["artifact_inventory_sha256"] == first_manifest[
        "artifact_inventory_sha256"
    ]
    assert first_manifest["input_fingerprint_sha256"] == second_manifest[
        "input_fingerprint_sha256"
    ]
    first_summary = json.loads((first / "final_summary.json").read_text())
    assert first_manifest["effective_readiness_policy_sha256"] == first_summary[
        "effective_readiness_policy_sha256"
    ]
    with (first / "ogdc_linked_articles.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        published_ogdc = list(csv.DictReader(handle))
    assert published_ogdc[0]["link_basis"] == (
        "checksum_bound_article_entity_links"
    )
    published_links = json.loads(published_ogdc[0]["link_evidence"])
    assert published_links[0]["matched_alias"] == "OGDC"
    assert published_links[0]["match_method"] == "exact"
    with (first / "article_session_mappings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        published_mappings = list(csv.DictReader(handle))
    assert len(published_mappings) == 3
    assert {row["population"] for row in published_mappings} == {
        "all_financial_news",
        "ogdc_linked_news",
        "energy_sector_news",
    }
    all_news_mapping = next(
        row
        for row in published_mappings
        if row["population"] == "all_financial_news"
    )
    assert all_news_mapping["occurrence_in_population"] == "True"
    assert all_news_mapping["canonical_assigned_trading_date"] == "2024-01-02"

    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(first.iterdir())
        if path.suffix in {".json", ".csv"}
    )
    assert all(secret not in serialized for secret in secrets.values())
    assert str(tmp_path) not in serialized
    for key in ("api_key", "auth", "session", "signature", "signed", "jwt"):
        assert f"{key}=%5BREDACTED%5D" in serialized
    assert "id=123" in serialized
    assert "author=desk" in serialized
    assert "utm_source" not in serialized
    assert "access_token" not in serialized

    completed = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT_PATH),
            "--run-dir",
            str(first),
            "--pack-only",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    verification_payload = json.loads(completed.stdout)
    assert verification_payload["verification"] == "PASS"
    assert verification_payload["verification_basis"] == "PACK_SEMANTIC"
    assert verification_payload["source_evidence_bound"] is False
    assert str(tmp_path) not in completed.stdout

    with pytest.raises(CoverageRunCollisionError):
        publish_coverage_run(
            first_computation,
            news=news,
            stock=stock,
            config=config,
            output_root=output,
            source_assessment_path=assessment,
            run_id=first_id,
            created_at=created_at,
        )

    second_manifest["status"] = "PARTIAL"
    (second / "run_manifest.json").write_text(
        json.dumps(second_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        CoverageAuditError, match="completion marker does not bind this manifest"
    ):
        verify_coverage_pack(second)
    failed_verification = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT_PATH),
            "--run-dir",
            str(second),
                "--pack-only",
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed_verification.returncode == 1
    assert json.loads(failed_verification.stdout)["verification"] == "FAIL"
    assert str(tmp_path) not in failed_verification.stdout

    reserved_id = "COVERAGE-20240110-120000-reserved"
    reservation = output / f".{reserved_id}.publish.lock"
    reservation.write_bytes(b"")
    with pytest.raises(CoverageRunCollisionError):
        publish_coverage_run(
            first_computation,
            news=news,
            stock=stock,
            config=config,
            output_root=output,
            source_assessment_path=assessment,
            run_id=reserved_id,
            created_at=created_at,
        )
    assert not (output / reserved_id).exists()
    assert not (output / f".{reserved_id}.staging").exists()
    reservation.unlink()

    late_id = "COVERAGE-20240110-120000-late"
    original_rename = coverage_module._rename_directory_noreplace

    def inject_late_collision(
        source: Path, destination: Path
    ) -> None:
        destination.mkdir()
        original_rename(source, destination)

    monkeypatch.setattr(
        coverage_module, "_rename_directory_noreplace", inject_late_collision
    )
    with pytest.raises(CoverageRunCollisionError):
        publish_coverage_run(
            first_computation,
            news=news,
            stock=stock,
            config=config,
            output_root=output,
            source_assessment_path=assessment,
            run_id=late_id,
            created_at=created_at,
        )
    assert (output / late_id).is_dir()
    assert list((output / late_id).iterdir()) == []
    assert not (output / f".{late_id}.staging").exists()
    assert not (output / f".{late_id}.publish.lock").exists()

    monkeypatch.setattr(
        coverage_module, "_rename_directory_noreplace", original_rename
    )
    unsafe_id = "COVERAGE-20240110-120000-unsafe"
    original_verify = coverage_module._verify_coverage_root
    cleanup_sentinel = tmp_path / "cleanup-sentinel"
    cleanup_sentinel.mkdir()
    sentinel_file = cleanup_sentinel / "keep.txt"
    sentinel_file.write_text("keep\n", encoding="utf-8")

    def inject_unsafe_staging_entry(
        root: Path, *, expected_run_id: str
    ) -> coverage_module.CoverageRunVerification:
        nested = root / "unexpected-directory"
        nested.mkdir()
        (nested / "payload.txt").write_text("unsafe\n", encoding="utf-8")
        (root / "unexpected-link").symlink_to(
            cleanup_sentinel, target_is_directory=True
        )
        return original_verify(root, expected_run_id=expected_run_id)

    monkeypatch.setattr(
        coverage_module, "_verify_coverage_root", inject_unsafe_staging_entry
    )
    with pytest.raises(CoverageAuditError, match="only regular files"):
        publish_coverage_run(
            first_computation,
            news=news,
            stock=stock,
            config=config,
            output_root=output,
            source_assessment_path=assessment,
            run_id=unsafe_id,
            created_at=created_at,
        )
    assert not (output / unsafe_id).exists()
    assert not (output / f".{unsafe_id}.staging").exists()
    assert not (output / f".{unsafe_id}.publish.lock").exists()
    assert sentinel_file.read_text(encoding="utf-8") == "keep\n"

    monkeypatch.setattr(
        coverage_module, "_verify_coverage_root", original_verify
    )

    unexpected_directory = first / "unexpected"
    unexpected_directory.mkdir()
    with pytest.raises(CoverageAuditError, match="only regular files"):
        verify_coverage_pack(first)
    unexpected_directory.rmdir()

    unexpected_symlink = first / "unexpected-link"
    unexpected_symlink.symlink_to("final_summary.json")
    with pytest.raises(CoverageAuditError, match="only regular files"):
        verify_coverage_pack(first)


def _publish_semantic_verification_fixture(tmp_path: Path) -> Path:
    news = _news(
        _occurrence(
            "semantic-article",
            available_at=_local("2024-01-02", "14:00:00"),
            tickers=("OGDC",),
        )
    )
    stock = _stock(
        (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        )
    )
    config = CoverageConfig()
    assessment = tmp_path / "source-assessment.md"
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")
    return publish_coverage_run(
        compute_coverage(news, stock, config),
        news=news,
        stock=stock,
        config=config,
        output_root=tmp_path / "published",
        source_assessment_path=assessment,
        run_id="COVERAGE-20240110-120000-semantic",
        created_at=_utc("2024-01-10", "12:00:00"),
    )


def _reseal_coverage_pack(run_directory: Path) -> None:
    manifest_path = run_directory / "run_manifest.json"
    completion_path = run_directory / "completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inventory = coverage_module._artifact_inventory(
        run_directory,
        exclude={"run_manifest.json", "completion.json"},
    )
    manifest["artifact_inventory"] = inventory
    inventory_digest = coverage_module._sha256_bytes(
        coverage_module._canonical_json_bytes(inventory)
    )
    manifest["artifact_inventory_sha256"] = inventory_digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion.update(
        {
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_size_bytes": manifest_path.stat().st_size,
            "summary_sha256": sha256_file(run_directory / "final_summary.json"),
            "artifact_inventory_sha256": inventory_digest,
            "artifact_count": len(inventory),
        }
    )
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_source_bound_verifier_requires_sealed_sqlite_evidence(
    tmp_path: Path,
) -> None:
    destination = _publish_semantic_verification_fixture(tmp_path)

    with pytest.raises(
        CoverageAuditError,
        match="requires the referenced SQLite database",
    ):
        verify_coverage_run(
            destination,
            news_run_directories=[tmp_path / "unused-news"],
            vertical_slice_directory=tmp_path / "unused-vertical",
            model_training_directory=tmp_path / "unused-training",
            source_assessment_path=tmp_path / "source-assessment.md",
        )


def _run_source_bound_cli_without_optional_evidence(
    run_directory: Path,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT_PATH),
            "--run-dir",
            str(run_directory),
            "--news-run-dir",
            str(tmp_path / "unused-news"),
            "--vertical-slice-dir",
            str(tmp_path / "unused-vertical"),
            "--model-training-dir",
            str(tmp_path / "unused-training"),
            "--source-assessment",
            str(tmp_path / "source-assessment.md"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_verification_cli_reports_required_sqlite_without_path_leakage(
    tmp_path: Path,
) -> None:
    destination = _publish_semantic_verification_fixture(tmp_path)

    completed = _run_source_bound_cli_without_optional_evidence(
        destination,
        tmp_path,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["error"] == (
        "Source-bound verification requires the referenced SQLite database"
    )
    assert str(tmp_path) not in completed.stdout


def test_verification_cli_reports_required_permission_policy_without_path_leakage(
    tmp_path: Path,
) -> None:
    occurrence = _occurrence(
        "permission-bound",
        available_at=_local("2024-01-02", "14:00:00"),
    )
    news = replace(
        _news(occurrence, database_provided=False),
        permission_policy_reference={
            "schema_version": "1.0",
            "policy_reference": "source-permission-policy.json",
            "policy_sha256": _digest("source-permission-policy"),
            "assessment_document": "source-assessment.md",
            "assessment_sha256": _digest("# Reviewed source assessment\n"),
        },
    )
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    config = CoverageConfig()
    assessment = tmp_path / "source-assessment.md"
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")
    destination = publish_coverage_run(
        compute_coverage(news, stock, config),
        news=news,
        stock=stock,
        config=config,
        output_root=tmp_path / "published",
        source_assessment_path=assessment,
        run_id="COVERAGE-20240110-120000-permission-bound",
        created_at=_utc("2024-01-10", "12:00:00"),
    )

    completed = _run_source_bound_cli_without_optional_evidence(
        destination,
        tmp_path,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["error"] == (
        "Source-bound verification requires the referenced permission policy"
    )
    assert str(tmp_path) not in completed.stdout


def test_source_bound_verifier_reopens_exact_news_price_and_split_evidence(
    tmp_path: Path,
) -> None:
    news_dir = _write_news_pack(tmp_path / "news")
    vertical, training = _write_stock_packs(tmp_path / "stock")
    news = load_news_evidence(
        [news_dir],
        database_path=None,
        allow_legacy_unsealed=True,
    )
    stock = load_stock_evidence(vertical, training)
    config = CoverageConfig()
    assessment = tmp_path / "source-assessment.md"
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")
    destination = publish_coverage_run(
        compute_coverage(news, stock, config),
        news=news,
        stock=stock,
        config=config,
        output_root=tmp_path / "coverage",
        source_assessment_path=assessment,
        run_id="COVERAGE-20240110-120000-source-bound",
        created_at=_utc("2024-01-10", "12:00:00"),
    )

    pack_only = verify_coverage_pack(destination)
    assert pack_only.to_dict()["verification_basis"] == "PACK_SEMANTIC"
    assert pack_only.to_dict()["source_evidence_bound"] is False

    source_bound = verify_coverage_run(
        destination,
        news_run_directories=[news_dir],
        vertical_slice_directory=vertical,
        model_training_directory=training,
        source_assessment_path=assessment,
        database_path=None,
        allow_legacy_unsealed=True,
    )
    assert source_bound.to_dict()["verification_basis"] == (
        "SOURCE_EVIDENCE_BOUND"
    )
    assert source_bound.to_dict()["source_evidence_bound"] is True


def test_source_bound_verifier_rejects_self_consistent_false_source_references(
    tmp_path: Path,
) -> None:
    news_dir = _write_news_pack(tmp_path / "news")
    vertical, training = _write_stock_packs(tmp_path / "stock")
    news = load_news_evidence(
        [news_dir],
        database_path=None,
        allow_legacy_unsealed=True,
    )
    stock = load_stock_evidence(vertical, training)
    config = CoverageConfig()
    assessment = tmp_path / "source-assessment.md"
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")
    destination = publish_coverage_run(
        compute_coverage(news, stock, config),
        news=news,
        stock=stock,
        config=config,
        output_root=tmp_path / "coverage",
        source_assessment_path=assessment,
        run_id="COVERAGE-20240110-120000-false-reference",
        created_at=_utc("2024-01-10", "12:00:00"),
    )

    stock_reference_path = destination / "stock_artifact_reference.json"
    stock_reference = json.loads(stock_reference_path.read_text(encoding="utf-8"))
    stock_reference["price_artifact_sha256"] = "0" * 64
    stock_reference["training_input_snapshot_sha256"] = "0" * 64
    stock_reference_path.write_text(
        json.dumps(stock_reference, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path = destination / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    news_reference = json.loads(
        (destination / "imported_or_acquired_news_reference.json").read_text(
            encoding="utf-8"
        )
    )
    assessment_reference = json.loads(
        (destination / "source_assessment_reference.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["input_fingerprint_sha256"] = coverage_module._sha256_bytes(
        coverage_module._canonical_json_bytes(
            {
                "news_runs": news_reference["news_runs"],
                "database": news_reference["database_fingerprints"],
                "energy_sector_link_contract": news_reference[
                    "energy_sector_link_contract"
                ],
                "stock": stock_reference,
                "source_assessment": assessment_reference,
                "config": manifest["config"],
            }
        )
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _reseal_coverage_pack(destination)

    # The pack is internally self-consistent and its forged hashes are shaped
    # correctly, so only an exact source replay can disprove the claim.
    assert verify_coverage_pack(destination).status == "PASS"
    with pytest.raises(CoverageAuditError, match="price/split reference differs"):
        verify_coverage_run(
            destination,
            news_run_directories=[news_dir],
            vertical_slice_directory=vertical,
            model_training_directory=training,
            source_assessment_path=assessment,
            database_path=None,
            allow_legacy_unsealed=True,
        )


def test_publish_rejects_a_supplied_computation_that_does_not_match_inputs(
    tmp_path: Path,
) -> None:
    news = _news(
        _occurrence("real-article", available_at=_local("2024-01-02", "14:00:00"))
    )
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    config = CoverageConfig()
    computation = compute_coverage(news, stock, config)
    false_summary = dict(computation.final_summary)
    false_summary["total_news_occurrences"] = 999_999
    contradictory = replace(
        computation,
        article_session_rows=(),
        final_summary=false_summary,
    )
    assessment = tmp_path / "source-assessment.md"
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")

    with pytest.raises(CoverageAuditError, match="recomputed input evidence"):
        publish_coverage_run(
            contradictory,
            news=news,
            stock=stock,
            config=config,
            output_root=tmp_path / "published",
            source_assessment_path=assessment,
            run_id="COVERAGE-20240110-120000-contradictory",
            created_at=_utc("2024-01-10", "12:00:00"),
        )


def test_offline_verifier_rejects_semantically_false_resealed_metrics(
    tmp_path: Path,
) -> None:
    run_directory = _publish_semantic_verification_fixture(tmp_path)
    summary_path = run_directory / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["total_news_occurrences"] = 999_999
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _reseal_coverage_pack(run_directory)

    with pytest.raises(CoverageAuditError, match="semantic"):
        verify_coverage_pack(run_directory)


def test_offline_verifier_recomputes_session_assignments_from_availability(
    tmp_path: Path,
) -> None:
    run_directory = _publish_semantic_verification_fixture(tmp_path)
    mappings_path = run_directory / "article_session_mappings.csv"
    with mappings_path.open("r", encoding="utf-8", newline="") as handle:
        mapping_reader = csv.DictReader(handle)
        mapping_fields = list(mapping_reader.fieldnames or ())
        mappings = list(mapping_reader)
    for row in mappings:
        if row["population_supported"] == "True":
            row["occurrence_assigned_trading_date"] = "2024-01-03"
            if row["canonical_assigned_trading_date"]:
                row["canonical_assigned_trading_date"] = "2024-01-03"
    with mappings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mapping_fields)
        writer.writeheader()
        writer.writerows(mappings)

    trading_path = run_directory / "overlap_by_trading_day.csv"
    with trading_path.open("r", encoding="utf-8", newline="") as handle:
        trading_reader = csv.DictReader(handle)
        trading_fields = list(trading_reader.fieldnames or ())
        trading_rows = list(trading_reader)
    for row in trading_rows:
        if row["trading_date"] not in {"2024-01-02", "2024-01-03"}:
            continue
        replacement = "0" if row["trading_date"] == "2024-01-02" else "1"
        for prefix in ("all_financial", "ogdc_linked"):
            row[f"{prefix}_occurrence_count"] = replacement
            row[f"{prefix}_canonical_article_count"] = replacement
    with trading_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trading_fields)
        writer.writeheader()
        writer.writerows(trading_rows)

    missing_rows: list[dict[str, object]] = []
    for population, prefix in (
        ("all_financial_news", "all_financial"),
        ("ogdc_linked_news", "ogdc_linked"),
    ):
        for split in ("full_price_period", "train", "validation", "test"):
            selected = (
                trading_rows
                if split == "full_price_period"
                else [row for row in trading_rows if row["split"] == split]
            )
            dates = tuple(date.fromisoformat(row["trading_date"]) for row in selected)
            covered = {
                date.fromisoformat(row["trading_date"])
                for row in selected
                if int(row[f"{prefix}_canonical_article_count"]) > 0
            }
            missing_rows.extend(
                coverage_module._missing_intervals(
                    dates,
                    covered,
                    population=population,
                    split=split,
                )
            )
    coverage_module._write_csv(
        run_directory / "missing_intervals.csv",
        missing_rows,
        (
            "population",
            "split",
            "start_date",
            "end_date",
            "missing_trading_sessions",
        ),
    )

    ogdc_path = run_directory / "ogdc_linked_articles.csv"
    with ogdc_path.open("r", encoding="utf-8", newline="") as handle:
        ogdc_reader = csv.DictReader(handle)
        ogdc_fields = list(ogdc_reader.fieldnames or ())
        ogdc_rows = list(ogdc_reader)
    for row in ogdc_rows:
        row["assigned_trading_date"] = "2024-01-03"
    with ogdc_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ogdc_fields)
        writer.writeheader()
        writer.writerows(ogdc_rows)

    _reseal_coverage_pack(run_directory)

    with pytest.raises(CoverageAuditError, match="invalid occurrence assignment"):
        verify_coverage_pack(run_directory)


def test_offline_verifier_rejects_population_varying_occurrence_identity(
    tmp_path: Path,
) -> None:
    run_directory = _publish_semantic_verification_fixture(tmp_path)
    mappings_path = run_directory / "article_session_mappings.csv"
    with mappings_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    energy_row = next(
        row for row in rows if row["population"] == "energy_sector_news"
    )
    energy_row["source_name"] = "Contradictory Source"
    with mappings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _reseal_coverage_pack(run_directory)

    with pytest.raises(CoverageAuditError, match="population-varying occurrence"):
        verify_coverage_pack(run_directory)


def test_offline_verifier_requires_regeneration_of_schema_1_0_pack(
    tmp_path: Path,
) -> None:
    run_directory = _publish_semantic_verification_fixture(tmp_path)
    manifest_path = run_directory / "run_manifest.json"
    summary_path = run_directory / "final_summary.json"
    completion_path = run_directory / "completion.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.0"
    manifest["algorithm_version"] = "news-price-coverage-v2"
    summary["schema_version"] = "1.0"
    completion["coverage_schema_version"] = "1.0"
    completion["algorithm_version"] = "news-price-coverage-v2"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    completion_path.write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _reseal_coverage_pack(run_directory)

    with pytest.raises(CoverageAuditError, match="regenerate.*schema 1.1"):
        verify_coverage_pack(run_directory)


def test_post_commit_reservation_cleanup_failure_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    news = _news(
        _occurrence("cleanup-article", available_at=_local("2024-01-02", "14:00:00"))
    )
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    config = CoverageConfig()
    assessment = tmp_path / "source-assessment.md"
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")
    output = tmp_path / "published"
    run_id = "COVERAGE-20240110-120000-cleanup"
    destination = output / run_id
    original_unlink = Path.unlink

    def fail_post_commit_cleanup(
        path: Path, missing_ok: bool = False,
    ) -> None:
        if path.name == f".{run_id}.publish.lock" and destination.is_dir():
            raise OSError("simulated post-commit reservation cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_post_commit_cleanup)
    with pytest.warns(RuntimeWarning, match="already committed"):
        published = publish_coverage_run(
            compute_coverage(news, stock, config),
            news=news,
            stock=stock,
            config=config,
            output_root=output,
            source_assessment_path=assessment,
            run_id=run_id,
            created_at=_utc("2024-01-10", "12:00:00"),
        )

    assert published == destination
    assert verify_coverage_pack(destination).status == "PASS"


def test_post_commit_cleanup_warning_cannot_overturn_success_when_warnings_are_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    news = _news(
        _occurrence(
            "cleanup-warning-error",
            available_at=_local("2024-01-02", "14:00:00"),
        )
    )
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    config = CoverageConfig()
    assessment = tmp_path / "source-assessment.md"
    assessment.write_text("# Reviewed source assessment\n", encoding="utf-8")
    output = tmp_path / "published"
    run_id = "COVERAGE-20240110-120000-cleanup-warning-error"
    destination = output / run_id
    original_unlink = Path.unlink

    def fail_post_commit_cleanup(
        path: Path, missing_ok: bool = False,
    ) -> None:
        if path.name == f".{run_id}.publish.lock" and destination.is_dir():
            raise OSError("simulated post-commit reservation cleanup failure")
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_post_commit_cleanup)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        published = publish_coverage_run(
            compute_coverage(news, stock, config),
            news=news,
            stock=stock,
            config=config,
            output_root=output,
            source_assessment_path=assessment,
            run_id=run_id,
            created_at=_utc("2024-01-10", "12:00:00"),
        )

    assert published == destination
    assert verify_coverage_pack(destination).status == "PASS"


def test_coverage_csv_neutralizes_formula_leading_provider_text(
    tmp_path: Path,
) -> None:
    occurrence = replace(
        _occurrence(
            "formula-safe",
            available_at=_local("2024-01-02", "14:00:00"),
            tickers=("OGDC",),
            source="+SUM(1,1)",
        ),
        title="=HYPERLINK(\"https://example.test\")",
    )
    news = _news(occurrence)
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
    config = CoverageConfig()
    assessment = tmp_path / "assessment.md"
    assessment.write_text("# Reviewed\n", encoding="utf-8")
    destination = publish_coverage_run(
        compute_coverage(news, stock, config),
        news=news,
        stock=stock,
        config=config,
        output_root=tmp_path / "coverage",
        source_assessment_path=assessment,
        run_id="COVERAGE-20240110-120000-formula-safe",
        created_at=_utc("2024-01-10", "12:00:00"),
    )

    with (destination / "article_session_mappings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        mapping = next(csv.DictReader(handle))
    with (destination / "ogdc_linked_articles.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        linked = next(csv.DictReader(handle))
    assert mapping["source_name"] == "'+SUM(1,1)"
    assert linked["title"] == "'=HYPERLINK(\"https://example.test\")"
    assert verify_coverage_pack(destination).status == "PASS"


def test_coverage_cli_help_exposes_reproducibility_and_safety_options() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    for option in (
        "--news-run-dir",
        "--vertical-slice-dir",
        "--model-training-dir",
        "--source-assessment",
        "--source-permission-policy",
        "--readiness-config",
        "--market-timezone",
        "--decision-cutoff",
        "--real-mode",
        "--allow-legacy-unsealed",
        "--run-id",
    ):
        assert option in completed.stdout
    assert "financial advice" in completed.stdout


def test_coverage_verification_cli_help_exposes_offline_run_check() -> None:
    completed = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT_PATH), "--help"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    for option in (
        "--run-dir",
        "--news-run-dir",
        "--news-database",
        "--vertical-slice-dir",
        "--model-training-dir",
        "--source-assessment",
        "--source-permission-policy",
        "--energy-sector-link-contract",
        "--pack-only",
    ):
        assert option in completed.stdout
    assert "source-bound" in completed.stdout
    assert "without network access" in completed.stdout
