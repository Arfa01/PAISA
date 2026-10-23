from __future__ import annotations

import csv
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from zoneinfo import ZoneInfo

import pytest

import paisa.news.coverage as coverage_module
from paisa.news.coverage import (
    CoverageAuditError,
    CoverageConfig,
    CoverageRunCollisionError,
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
    publish_coverage_run,
    sha256_file,
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
    )


def _news(
    *occurrences: NewsOccurrence,
    permission_review_required: bool = False,
    run_status: str = "PASS",
    provider_error_count: int = 0,
    database_provided: bool = True,
    verification_basis: str | None = None,
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
                "all_financial_news", split, all_coverage, gap=gap
            )
        )
        rows.append(_split_row("ogdc_linked_news", split, ogdc_coverage))
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
    stock = _stock(("2024-01-02", "2024-01-03", "2024-01-04"))
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
        ready_minimum_distinct_sources=1,
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
        ready_minimum_distinct_sources=1,
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
        ready_minimum_distinct_sources=1,
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
    assert first["energy_sector_canonical_article_count"] == 0
    assert result.quality_report["energy_sector_linking"] == {
        "supported": False,
        "reason": (
            "The current news contract does not infer sectors; general news "
            "was not relabelled as energy-sector news."
        ),
    }
    assert result.final_summary["population_support"]["energy_sector_news"][
        "supported"
    ] is False
    assert result.final_summary["readiness_thresholds"]
    assert result.final_summary["readiness_evaluation"]


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
        _news(general, later_ogdc_duplicate, energy),
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

    first_verification = verify_coverage_run(first)
    second_verification = verify_coverage_run(second)
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
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["verification"] == "PASS"
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
        verify_coverage_run(second)
    failed_verification = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT_PATH),
            "--run-dir",
            str(second),
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
        verify_coverage_run(first)
    unexpected_directory.rmdir()

    unexpected_symlink = first / "unexpected-link"
    unexpected_symlink.symlink_to("final_summary.json")
    with pytest.raises(CoverageAuditError, match="only regular files"):
        verify_coverage_run(first)


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
    assert "--run-dir" in completed.stdout
    assert "without network access" in completed.stdout
