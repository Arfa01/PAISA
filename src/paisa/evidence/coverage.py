"""Leakage-safe coverage auditing for contemporaneous evidence.

This module measures evidence availability against the immutable stock-session
calendar.  It deliberately keeps GDELT metadata, exchange announcements, and
publisher news as separate populations.  Coverage and readiness are engineering
gates only; they are not evidence of predictive value and are not financial
advice.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import math
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from paisa.evidence.models import EVIDENCE_TYPES, EvidenceRecord, utc_iso
from paisa.news.coverage import (
    CoverageConfig,
    SessionAssignment,
    StockEvidence,
    assign_to_session,
)


EVIDENCE_COVERAGE_SCHEMA_VERSION = "1.0"
EVIDENCE_COVERAGE_ALGORITHM_VERSION = "contemporaneous-evidence-coverage-v1"
EVIDENCE_COVERAGE_PROFILES = (
    "gdelt_gkg_ogdc",
    "psx_ogdc_announcements",
    "combined_ogdc_evidence",
    "general_financial_news",
)
MODEL_SPLITS = ("train", "validation", "test")
AUDIT_PERIODS = ("full_price_period", *MODEL_SPLITS)
READINESS_STATUSES = frozenset(
    {
        "READY_FOR_AGGREGATION",
        "PARTIALLY_COVERED",
        "INSUFFICIENT_OVERLAP",
        "NO_OVERLAP",
    }
)


class EvidenceCoverageError(ValueError):
    """Coverage input is contradictory, incomplete, or leakage-unsafe."""


@dataclass(frozen=True, slots=True)
class EvidenceReadinessCriteria:
    """Explicit engineering gates, not statistical-sufficiency claims."""

    ready_minimum_coverage_percent_each_split: float = 20.0
    ready_minimum_covered_sessions_each_split: int = 2
    ready_maximum_missing_gap_sessions_each_split: int = 20
    partial_minimum_coverage_percent_each_split: float = 1.0
    partial_minimum_covered_sessions_each_split: int = 1

    def __post_init__(self) -> None:
        for name in (
            "ready_minimum_coverage_percent_each_split",
            "partial_minimum_coverage_percent_each_split",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 < float(value) <= 100.0
            ):
                raise EvidenceCoverageError(
                    f"{name} must be a finite percentage in (0, 100]"
                )
        for name in (
            "ready_minimum_covered_sessions_each_split",
            "ready_maximum_missing_gap_sessions_each_split",
            "partial_minimum_covered_sessions_each_split",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise EvidenceCoverageError(
                    f"{name} must be a non-negative integer"
                )
        if (
            self.ready_minimum_coverage_percent_each_split
            < self.partial_minimum_coverage_percent_each_split
        ):
            raise EvidenceCoverageError(
                "The ready coverage gate cannot be weaker than the partial gate"
            )
        if (
            self.ready_minimum_covered_sessions_each_split
            < self.partial_minimum_covered_sessions_each_split
        ):
            raise EvidenceCoverageError(
                "The ready session gate cannot be weaker than the partial gate"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceSessionMapping:
    evidence_id: str
    evidence_type: str
    provider_id: str
    ticker: str
    profiles: tuple[str, ...]
    time_precision: str
    source_published_at: str | None
    source_published_date: str | None
    available_at: str | None
    fetched_at: str
    assigned_session: str | None
    assigned_split: str | None
    decision_cutoff_utc: str | None
    arrival_class: str
    mapping_status: str
    cutoff_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["profiles"] = list(self.profiles)
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceCoverageRow:
    profile: str
    split: str
    trading_sessions: int
    sessions_with_records: int
    coverage_percent: float
    record_count: int
    zero_record_sessions: int
    longest_missing_gap_sessions: int
    before_cutoff_records: int
    after_cutoff_records: int
    non_trading_day_records: int
    date_only_records: int
    ogdc_specific_records: int
    evidence_type_counts: Mapping[str, int]
    provider_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_type_counts"] = dict(self.evidence_type_counts)
        payload["provider_counts"] = dict(self.provider_counts)
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceMonthlyCoverageRow:
    profile: str
    split: str
    month: str
    trading_sessions: int
    sessions_with_records: int
    coverage_percent: float
    record_count: int
    date_only_records: int
    evidence_type_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evidence_type_counts"] = dict(self.evidence_type_counts)
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceMissingInterval:
    profile: str
    split: str
    start_date: str
    end_date: str
    missing_trading_sessions: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvidenceProfileReadiness:
    profile: str
    status: str
    passed: bool
    reason: str
    checks: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in READINESS_STATUSES:
            raise EvidenceCoverageError(f"Unsupported readiness status: {self.status}")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checks"] = dict(self.checks)
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceCoverageComputation:
    mapping_rows: tuple[EvidenceSessionMapping, ...]
    split_rows: tuple[EvidenceCoverageRow, ...]
    monthly_rows: tuple[EvidenceMonthlyCoverageRow, ...]
    missing_intervals: tuple[EvidenceMissingInterval, ...]
    readiness: Mapping[str, EvidenceProfileReadiness]
    summary: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVIDENCE_COVERAGE_SCHEMA_VERSION,
            "algorithm_version": EVIDENCE_COVERAGE_ALGORITHM_VERSION,
            "mapping_rows": [item.to_dict() for item in self.mapping_rows],
            "split_rows": [item.to_dict() for item in self.split_rows],
            "monthly_rows": [item.to_dict() for item in self.monthly_rows],
            "missing_intervals": [
                item.to_dict() for item in self.missing_intervals
            ],
            "readiness": {
                profile: self.readiness[profile].to_dict()
                for profile in EVIDENCE_COVERAGE_PROFILES
            },
            "summary": dict(self.summary),
        }


def _validate_stock(stock: StockEvidence) -> None:
    dates = stock.trading_dates
    if not dates:
        raise EvidenceCoverageError("Stock evidence has no trading sessions")
    if tuple(sorted(set(dates))) != dates:
        raise EvidenceCoverageError(
            "Stock trading dates must be unique and strictly chronological"
        )
    stock_dates = set(dates)
    assigned: set[date] = set()
    ordered_targets: list[date] = []
    previous_end: date | None = None
    for split in MODEL_SPLITS:
        values = stock.split_dates.get(split)
        if values is None or not values:
            raise EvidenceCoverageError(f"Stock evidence has no {split} sessions")
        if tuple(sorted(set(values))) != tuple(values):
            raise EvidenceCoverageError(
                f"Stock {split} dates must be unique and chronological"
            )
        if not set(values).issubset(stock_dates):
            raise EvidenceCoverageError(
                f"Stock {split} dates are outside the trading calendar"
            )
        if assigned.intersection(values):
            raise EvidenceCoverageError("Stock model splits overlap")
        if previous_end is not None and values[0] <= previous_end:
            raise EvidenceCoverageError(
                "Stock model splits must be chronologically ordered"
            )
        previous_end = values[-1]
        assigned.update(values)
        ordered_targets.extend(values)
    expected_targets = tuple(
        session_date
        for session_date in dates
        if session_date >= ordered_targets[0]
    )
    if tuple(ordered_targets) != expected_targets:
        raise EvidenceCoverageError(
            "Stock model splits must cover one contiguous trailing target calendar"
        )


def _profiles_for(record: EvidenceRecord) -> tuple[str, ...]:
    ticker = record.ticker.upper()
    profiles: list[str] = []
    if record.evidence_type == "gdelt_metadata" and ticker == "OGDC":
        profiles.extend(("gdelt_gkg_ogdc", "combined_ogdc_evidence"))
    if record.evidence_type == "exchange_announcement" and ticker == "OGDC":
        profiles.extend(("psx_ogdc_announcements", "combined_ogdc_evidence"))
    if record.evidence_type == "publisher_news":
        profiles.append("general_financial_news")
        if ticker == "OGDC":
            profiles.append("combined_ogdc_evidence")
    return tuple(profiles)


def _split_lookup(stock: StockEvidence) -> dict[date, str]:
    result: dict[date, str] = {}
    for split in MODEL_SPLITS:
        for session_date in stock.split_dates[split]:
            result[session_date] = split
    return result


def _session_cutoff(
    session_date: date, config: CoverageConfig
) -> datetime:
    return datetime.combine(
        session_date,
        config.decision_cutoff,
        tzinfo=ZoneInfo(config.market_timezone),
    ).astimezone(timezone.utc)


def assign_date_only_to_session(
    source_date: date,
    stock: StockEvidence,
    config: CoverageConfig,
) -> SessionAssignment:
    """Map date-only evidence strictly after its source calendar date.

    A date without a posting time is never considered known during the same
    session.  ``bisect_right`` is intentional: a Friday-dated announcement maps
    to Monday (or the next actual session), and a trading-day announcement also
    maps to the following session.
    """

    del config  # Retained for a uniform assignment API and future policy IDs.
    _validate_stock(stock)
    if source_date < stock.trading_dates[0]:
        return SessionAssignment(
            None,
            "date_only_conservative",
            "BEFORE_PRICE_PERIOD",
        )
    index = bisect_right(stock.trading_dates, source_date)
    if index >= len(stock.trading_dates):
        return SessionAssignment(
            None,
            "date_only_conservative",
            "AFTER_PRICE_PERIOD",
        )
    return SessionAssignment(
        stock.trading_dates[index],
        "date_only_conservative",
        "DATE_ONLY_CONSERVATIVE_NEXT_SESSION",
    )


def assign_evidence_to_session(
    record: EvidenceRecord,
    stock: StockEvidence,
    config: CoverageConfig,
) -> SessionAssignment:
    """Assign one record using only its defensible historical availability."""

    if record.time_precision == "date_only":
        if record.source_published_date is None:
            raise EvidenceCoverageError(
                f"Date-only evidence {record.evidence_id} has no source date"
            )
        return assign_date_only_to_session(
            record.source_published_date, stock, config
        )
    if record.available_at is None:
        return SessionAssignment(
            None,
            "missing_available_at",
            "UNMAPPED_MISSING_TIME",
        )
    assignment = assign_to_session(record.available_at, stock, config)
    if assignment.session_date is not None:
        cutoff = _session_cutoff(assignment.session_date, config)
        instant = record.available_at.astimezone(timezone.utc)
        if instant > cutoff:
            raise EvidenceCoverageError(
                f"Evidence {record.evidence_id} was assigned before availability"
            )
    return assignment


def _longest_gap(dates: Sequence[date], covered: set[date]) -> int:
    longest = current = 0
    for session_date in dates:
        if session_date in covered:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def _missing_intervals(
    dates: Sequence[date],
    covered: set[date],
    *,
    profile: str,
    split: str,
) -> list[EvidenceMissingInterval]:
    rows: list[EvidenceMissingInterval] = []
    start: date | None = None
    previous: date | None = None
    count = 0
    for current in dates:
        if current in covered:
            if start is not None and previous is not None:
                rows.append(
                    EvidenceMissingInterval(
                        profile,
                        split,
                        start.isoformat(),
                        previous.isoformat(),
                        count,
                    )
                )
            start = previous = None
            count = 0
            continue
        start = start or current
        previous = current
        count += 1
    if start is not None and previous is not None:
        rows.append(
            EvidenceMissingInterval(
                profile,
                split,
                start.isoformat(),
                previous.isoformat(),
                count,
            )
        )
    return rows


def _ordered_type_counts(records: Iterable[EvidenceRecord]) -> dict[str, int]:
    counts = Counter(item.evidence_type for item in records)
    return {name: counts.get(name, 0) for name in sorted(EVIDENCE_TYPES)}


def _ordered_provider_counts(records: Iterable[EvidenceRecord]) -> dict[str, int]:
    counts = Counter(item.provider_id for item in records)
    return {name: counts[name] for name in sorted(counts)}


def _coverage_row(
    *,
    profile: str,
    split: str,
    dates: tuple[date, ...],
    records: Sequence[EvidenceRecord],
    assignments: Mapping[str, SessionAssignment],
) -> EvidenceCoverageRow:
    date_set = set(dates)
    selected = [
        item
        for item in records
        if assignments[item.evidence_id].session_date in date_set
    ]
    covered = {
        assignments[item.evidence_id].session_date
        for item in selected
        if assignments[item.evidence_id].session_date is not None
    }
    total = len(dates)
    sessions_with = len(covered)
    return EvidenceCoverageRow(
        profile=profile,
        split=split,
        trading_sessions=total,
        sessions_with_records=sessions_with,
        coverage_percent=round(100.0 * sessions_with / total, 6),
        record_count=len(selected),
        zero_record_sessions=total - sessions_with,
        longest_missing_gap_sessions=_longest_gap(dates, covered),
        before_cutoff_records=sum(
            assignments[item.evidence_id].arrival_class
            == "before_or_at_decision_cutoff"
            for item in selected
        ),
        after_cutoff_records=sum(
            assignments[item.evidence_id].arrival_class == "after_decision_cutoff"
            for item in selected
        ),
        non_trading_day_records=sum(
            assignments[item.evidence_id].arrival_class == "non_trading_day"
            for item in selected
        ),
        date_only_records=sum(item.time_precision == "date_only" for item in selected),
        ogdc_specific_records=sum(item.ticker.upper() == "OGDC" for item in selected),
        evidence_type_counts=_ordered_type_counts(selected),
        provider_counts=_ordered_provider_counts(selected),
    )


def _monthly_rows(
    *,
    profile: str,
    split: str,
    dates: tuple[date, ...],
    records: Sequence[EvidenceRecord],
    assignments: Mapping[str, SessionAssignment],
) -> list[EvidenceMonthlyCoverageRow]:
    sessions_by_month: dict[str, list[date]] = defaultdict(list)
    for session_date in dates:
        sessions_by_month[session_date.strftime("%Y-%m")].append(session_date)
    result: list[EvidenceMonthlyCoverageRow] = []
    for month in sorted(sessions_by_month):
        month_dates = set(sessions_by_month[month])
        selected = [
            item
            for item in records
            if assignments[item.evidence_id].session_date in month_dates
        ]
        covered = {
            assignments[item.evidence_id].session_date
            for item in selected
            if assignments[item.evidence_id].session_date is not None
        }
        session_count = len(month_dates)
        result.append(
            EvidenceMonthlyCoverageRow(
                profile=profile,
                split=split,
                month=month,
                trading_sessions=session_count,
                sessions_with_records=len(covered),
                coverage_percent=round(
                    100.0 * len(covered) / session_count, 6
                ),
                record_count=len(selected),
                date_only_records=sum(
                    item.time_precision == "date_only" for item in selected
                ),
                evidence_type_counts=_ordered_type_counts(selected),
            )
        )
    return result


def calculate_evidence_readiness(
    profile: str,
    split_rows: Sequence[EvidenceCoverageRow],
    *,
    criteria: EvidenceReadinessCriteria,
) -> EvidenceProfileReadiness:
    """Apply explicit split gates and the announcement-only safety guard."""

    if profile not in EVIDENCE_COVERAGE_PROFILES:
        raise EvidenceCoverageError(f"Unknown evidence profile: {profile}")
    rows = {
        row.split: row
        for row in split_rows
        if row.profile == profile and row.split in MODEL_SPLITS
    }
    if set(rows) != set(MODEL_SPLITS):
        raise EvidenceCoverageError(
            f"Readiness for {profile} requires train/validation/test rows"
        )
    total_records = sum(row.record_count for row in rows.values())
    if total_records == 0:
        return EvidenceProfileReadiness(
            profile,
            "NO_OVERLAP",
            False,
            "No cutoff-eligible evidence maps to any model split.",
            {},
        )

    coverage_values = [rows[split].coverage_percent for split in MODEL_SPLITS]
    session_values = [rows[split].sessions_with_records for split in MODEL_SPLITS]
    gap_values = [rows[split].longest_missing_gap_sessions for split in MODEL_SPLITS]
    type_totals = {
        evidence_type: sum(
            rows[split].evidence_type_counts[evidence_type]
            for split in MODEL_SPLITS
        )
        for evidence_type in sorted(EVIDENCE_TYPES)
    }
    non_announcement_by_split = {
        split: sum(
            count
            for evidence_type, count in rows[split].evidence_type_counts.items()
            if evidence_type != "exchange_announcement"
        )
        for split in MODEL_SPLITS
    }
    has_non_announcement_each_split = all(
        count > 0 for count in non_announcement_by_split.values()
    )
    checks: dict[str, Any] = {
        "coverage_each_split": {
            "actual_minimum_percent": min(coverage_values),
            "required_minimum_percent": (
                criteria.ready_minimum_coverage_percent_each_split
            ),
            "passed": min(coverage_values)
            >= criteria.ready_minimum_coverage_percent_each_split,
        },
        "covered_sessions_each_split": {
            "actual_minimum": min(session_values),
            "required_minimum": (
                criteria.ready_minimum_covered_sessions_each_split
            ),
            "passed": min(session_values)
            >= criteria.ready_minimum_covered_sessions_each_split,
        },
        "missing_gap_each_split": {
            "actual_maximum_sessions": max(gap_values),
            "allowed_maximum_sessions": (
                criteria.ready_maximum_missing_gap_sessions_each_split
            ),
            "passed": max(gap_values)
            <= criteria.ready_maximum_missing_gap_sessions_each_split,
        },
        "non_announcement_evidence": {
            "evidence_type_counts": type_totals,
            "record_counts_by_split": non_announcement_by_split,
            "passed": has_non_announcement_each_split,
            "interpretation": (
                "Every split requires at least one non-announcement record; "
                "official announcements alone cannot make an evidence profile "
                "READY_FOR_AGGREGATION."
            ),
        },
    }
    if all(bool(item["passed"]) for item in checks.values()):
        return EvidenceProfileReadiness(
            profile,
            "READY_FOR_AGGREGATION",
            True,
            "All explicit engineering coverage gates passed.",
            checks,
        )

    partial_coverage = (
        min(coverage_values)
        >= criteria.partial_minimum_coverage_percent_each_split
    )
    partial_sessions = (
        min(session_values)
        >= criteria.partial_minimum_covered_sessions_each_split
    )
    partial = partial_coverage and partial_sessions
    checks["partial_gate"] = {
        "coverage_actual_minimum_percent": min(coverage_values),
        "coverage_required_minimum_percent": (
            criteria.partial_minimum_coverage_percent_each_split
        ),
        "sessions_actual_minimum": min(session_values),
        "sessions_required_minimum": (
            criteria.partial_minimum_covered_sessions_each_split
        ),
        "passed": partial,
    }
    if partial:
        reason = (
            "Ready gates passed except that the mapped evidence is announcement-only."
            if not has_non_announcement_each_split
            and all(
                bool(item["passed"])
                for name, item in checks.items()
                if name not in {"non_announcement_evidence", "partial_gate"}
            )
            else "Some evidence overlaps every split, but ready gates did not all pass."
        )
        return EvidenceProfileReadiness(
            profile,
            "PARTIALLY_COVERED",
            False,
            reason,
            checks,
        )
    return EvidenceProfileReadiness(
        profile,
        "INSUFFICIENT_OVERLAP",
        False,
        "Evidence overlaps the price period but does not cover every model split.",
        checks,
    )


def compute_evidence_coverage(
    records: Sequence[EvidenceRecord],
    stock: StockEvidence,
    config: CoverageConfig,
    *,
    criteria: EvidenceReadinessCriteria | None = None,
) -> EvidenceCoverageComputation:
    """Compute deterministic coverage for all four evidence profiles."""

    _validate_stock(stock)
    criteria = criteria or EvidenceReadinessCriteria()
    ordered_records = sorted(records, key=lambda item: item.evidence_id)
    identifiers = [item.evidence_id for item in ordered_records]
    if len(set(identifiers)) != len(identifiers):
        raise EvidenceCoverageError("Evidence identities must be unique")

    profiles_by_id = {
        item.evidence_id: _profiles_for(item) for item in ordered_records
    }
    relevant_records = [
        item for item in ordered_records if profiles_by_id[item.evidence_id]
    ]
    assignments = {
        item.evidence_id: assign_evidence_to_session(item, stock, config)
        for item in relevant_records
    }
    split_lookup = _split_lookup(stock)
    mapping_rows: list[EvidenceSessionMapping] = []
    for item in relevant_records:
        assignment = assignments[item.evidence_id]
        assigned = assignment.session_date
        cutoff = _session_cutoff(assigned, config) if assigned is not None else None
        mapping_rows.append(
            EvidenceSessionMapping(
                evidence_id=item.evidence_id,
                evidence_type=item.evidence_type,
                provider_id=item.provider_id,
                ticker=item.ticker,
                profiles=profiles_by_id[item.evidence_id],
                time_precision=item.time_precision,
                source_published_at=utc_iso(item.source_published_at),
                source_published_date=(
                    item.source_published_date.isoformat()
                    if item.source_published_date is not None
                    else None
                ),
                available_at=utc_iso(item.available_at),
                fetched_at=utc_iso(item.fetched_at) or "",
                assigned_session=assigned.isoformat() if assigned else None,
                assigned_split=split_lookup.get(assigned) if assigned else None,
                decision_cutoff_utc=utc_iso(cutoff),
                arrival_class=assignment.arrival_class,
                mapping_status=assignment.mapping_status,
                cutoff_eligible=assigned is not None,
            )
        )

    periods: dict[str, tuple[date, ...]] = {
        "full_price_period": stock.trading_dates,
        **{
            split: tuple(stock.split_dates[split])
            for split in MODEL_SPLITS
        },
    }
    split_rows: list[EvidenceCoverageRow] = []
    monthly_rows: list[EvidenceMonthlyCoverageRow] = []
    missing_intervals: list[EvidenceMissingInterval] = []
    profile_records: dict[str, list[EvidenceRecord]] = {}
    for profile in EVIDENCE_COVERAGE_PROFILES:
        selected = [
            item
            for item in relevant_records
            if profile in profiles_by_id[item.evidence_id]
        ]
        profile_records[profile] = selected
        for split in AUDIT_PERIODS:
            dates = periods[split]
            row = _coverage_row(
                profile=profile,
                split=split,
                dates=dates,
                records=selected,
                assignments=assignments,
            )
            split_rows.append(row)
            monthly_rows.extend(
                _monthly_rows(
                    profile=profile,
                    split=split,
                    dates=dates,
                    records=selected,
                    assignments=assignments,
                )
            )
            covered = {
                assignments[item.evidence_id].session_date
                for item in selected
                if assignments[item.evidence_id].session_date in set(dates)
            }
            missing_intervals.extend(
                _missing_intervals(
                    dates,
                    covered,
                    profile=profile,
                    split=split,
                )
            )

    readiness = {
        profile: calculate_evidence_readiness(
            profile,
            split_rows,
            criteria=criteria,
        )
        for profile in EVIDENCE_COVERAGE_PROFILES
    }
    summary = {
        "schema_version": EVIDENCE_COVERAGE_SCHEMA_VERSION,
        "algorithm_version": EVIDENCE_COVERAGE_ALGORITHM_VERSION,
        "academic_use_notice": "Academic simulation; not financial advice.",
        "market_timezone": config.market_timezone,
        "decision_cutoff": config.decision_cutoff.isoformat(timespec="seconds"),
        "date_only_policy": "strictly_next_eligible_trading_session",
        "record_count_input": len(records),
        "record_count_profiled": len(relevant_records),
        "profile_input_counts": {
            profile: len(profile_records[profile])
            for profile in EVIDENCE_COVERAGE_PROFILES
        },
        "readiness_status_by_profile": {
            profile: readiness[profile].status
            for profile in EVIDENCE_COVERAGE_PROFILES
        },
        "readiness_criteria": criteria.to_dict(),
        "readiness_interpretation": (
            "Engineering/data-coverage gates only; not proof of statistical "
            "sufficiency or predictive value."
        ),
    }
    return EvidenceCoverageComputation(
        mapping_rows=tuple(mapping_rows),
        split_rows=tuple(split_rows),
        monthly_rows=tuple(monthly_rows),
        missing_intervals=tuple(missing_intervals),
        readiness=readiness,
        summary=summary,
    )


__all__ = [
    "AUDIT_PERIODS",
    "EVIDENCE_COVERAGE_ALGORITHM_VERSION",
    "EVIDENCE_COVERAGE_PROFILES",
    "EVIDENCE_COVERAGE_SCHEMA_VERSION",
    "EvidenceCoverageComputation",
    "EvidenceCoverageError",
    "EvidenceCoverageRow",
    "EvidenceMissingInterval",
    "EvidenceMonthlyCoverageRow",
    "EvidenceProfileReadiness",
    "EvidenceReadinessCriteria",
    "EvidenceSessionMapping",
    "MODEL_SPLITS",
    "READINESS_STATUSES",
    "assign_date_only_to_session",
    "assign_evidence_to_session",
    "calculate_evidence_readiness",
    "compute_evidence_coverage",
]
