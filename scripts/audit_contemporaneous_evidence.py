#!/usr/bin/env python3
"""Publish an offline, immutable contemporaneous-evidence coverage audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from paisa.evidence.artifacts import load_evidence_run, sha256_file  # noqa: E402
from paisa.evidence.coverage import (  # noqa: E402
    EvidenceReadinessCriteria,
    compute_evidence_coverage,
)
from paisa.evidence.coverage_artifacts import (  # noqa: E402
    make_evidence_coverage_run_id,
    publish_evidence_coverage_run,
    verify_evidence_coverage_pack,
)
from paisa.evidence.models import EvidenceRecord  # noqa: E402
from paisa.evidence.policy import load_evidence_permission_policy  # noqa: E402
from paisa.news.coverage import (  # noqa: E402
    CoverageConfig,
    NewsEvidence,
    NewsOccurrence,
    load_news_evidence,
    load_source_permission_policy,
    load_stock_evidence,
    parse_cutoff,
)


ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
_GENERIC_FAILURE = (
    "Evidence coverage audit failed; review the command arguments and "
    "checksum-bound local evidence."
)
_READINESS_FIELDS = (
    "ready_minimum_coverage_percent_each_split",
    "ready_minimum_covered_sessions_each_split",
    "ready_maximum_missing_gap_sessions_each_split",
    "partial_minimum_coverage_percent_each_split",
    "partial_minimum_covered_sessions_each_split",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit local GDELT GKG metadata, the explicitly blocked PSX "
            "announcement path, and optional publisher-news evidence against "
            "the immutable OGDC calendar. No network access is performed. "
            + ACADEMIC_DISCLAIMER
        )
    )
    parser.add_argument(
        "--gdelt-run-dir",
        action="append",
        required=True,
        help="Immutable EVID-* GDELT GKG run directory; repeat if needed.",
    )
    parser.add_argument(
        "--publisher-news-run-dir",
        action="append",
        help=(
            "Optional immutable NEWS-* run directory. When omitted, general "
            "publisher-news coverage is measured as an empty population."
        ),
    )
    parser.add_argument(
        "--news-database",
        default="data/news/paisa_news.sqlite3",
        help=(
            "SQLite database used only for local news-run seal replay. It is "
            "ignored when no publisher-news run is supplied."
        ),
    )
    parser.add_argument(
        "--news-source-assessment",
        default="docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md",
    )
    parser.add_argument(
        "--news-permission-policy",
        default="configs/news/historical_source_permissions.json",
    )
    parser.add_argument(
        "--evidence-permission-policy",
        default="configs/evidence/source_permissions.json",
        help="Policy that records the current PSX manual-import block.",
    )
    parser.add_argument(
        "--vertical-slice-dir",
        default="artifacts/vertical_slice/VSL-20261020-OGDC-main-02",
    )
    parser.add_argument(
        "--model-training-dir",
        default="artifacts/model_training/TRN-20261020-191815-624662-nogit",
    )
    parser.add_argument(
        "--readiness-config",
        default="configs/evidence/default_readiness.json",
    )
    parser.add_argument("--market-timezone", default="Asia/Karachi")
    parser.add_argument("--decision-cutoff", default="15:30:00")
    parser.add_argument(
        "--output-dir",
        default="artifacts/evidence_coverage",
        help="Root for immutable ECOV-* run directories.",
    )
    parser.add_argument("--run-id", help="Optional fresh ECOV-* identity.")
    return parser


def _json_print(value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _load_readiness_criteria(
    path: str | Path,
) -> tuple[EvidenceReadinessCriteria, dict[str, Any]]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Readiness configuration is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != "1.0":
        raise ValueError("Readiness configuration schema is unsupported")
    missing = [field for field in _READINESS_FIELDS if field not in value]
    if missing:
        raise ValueError("Readiness configuration is incomplete")
    criteria = EvidenceReadinessCriteria(
        **{field: value[field] for field in _READINESS_FIELDS}
    )
    reference = {
        "input_kind": "readiness_configuration",
        "schema_version": "1.0",
        "artifact_reference": f"configs/evidence/{source.name}",
        "sha256": sha256_file(source),
    }
    return criteria, reference


def _publisher_permission_ids(news: NewsEvidence) -> dict[str, str]:
    result: dict[str, str] = {}
    for reference in news.run_references:
        run_id = str(reference.get("run_id") or "")
        decision = reference.get("source_permission_decision")
        if not isinstance(decision, Mapping) or decision.get("passed") is not True:
            raise ValueError("Publisher-news evidence is outside its permission policy")
        decision_id = str(decision.get("decision_id") or "")
        if not run_id or not decision_id:
            raise ValueError("Publisher-news permission reference is incomplete")
        result[run_id] = decision_id
    return result


def adapt_publisher_news(news: NewsEvidence) -> tuple[EvidenceRecord, ...]:
    """Convert one representative per canonical article without backdating it."""

    permission_ids = _publisher_permission_ids(news)
    grouped: dict[str, list[NewsOccurrence]] = defaultdict(list)
    for occurrence in news.occurrences:
        grouped[occurrence.canonical_article_id].append(occurrence)

    far_future = datetime.max.replace(tzinfo=timezone.utc)
    records: list[EvidenceRecord] = []
    for canonical_id in sorted(grouped):
        occurrences = sorted(
            grouped[canonical_id],
            key=lambda item: (
                item.available_at or far_future,
                item.fetched_at or far_future,
                item.occurrence_key,
            ),
        )
        representative = occurrences[0]
        if representative.fetched_at is None:
            raise ValueError("Publisher-news evidence has no acquisition timestamp")
        if (
            representative.available_at is not None
            and representative.available_at < representative.fetched_at
        ):
            raise ValueError("Publisher-news evidence attempts historical backdating")
        if representative.ingestion_run_id not in permission_ids:
            raise ValueError("Publisher-news run has no permission decision")

        ogdc_occurrences = [item for item in occurrences if item.is_ogdc]
        entity_links = tuple(
            dict(link)
            for item in ogdc_occurrences
            for link in item.entity_links
            if str(link.get("ticker") or "").upper() == "OGDC"
        )
        identity = hashlib.sha256(
            (canonical_id + "\0" + representative.occurrence_key).encode("utf-8")
        ).hexdigest()[:24]
        records.append(
            EvidenceRecord(
                evidence_id=f"EVID-NEWS-{identity}",
                evidence_type="publisher_news",
                provider_id=representative.provider,
                provider_mode=representative.provider_mode,
                data_class=representative.data_class,
                source_name=representative.source_name,
                source_url=representative.source_url,
                source_record_id=canonical_id,
                ticker="OGDC" if ogdc_occurrences else "GENERAL",
                company_name=(
                    "Oil & Gas Development Company Limited"
                    if ogdc_occurrences
                    else "General financial news"
                ),
                title=representative.title,
                source_published_at=representative.published_at,
                source_published_date=None,
                available_at=representative.available_at,
                fetched_at=representative.fetched_at,
                imported_at=representative.fetched_at,
                historical_availability_basis=(
                    "news-availability-v1-conservative-acquisition"
                ),
                time_precision=(
                    "second" if representative.published_at is not None else "unknown"
                ),
                contemporaneous_source_id=canonical_id,
                contemporaneous_batch_at=None,
                permission_policy_id=permission_ids[
                    representative.ingestion_run_id
                ],
                raw_evidence_reference=(
                    "news_ingestion/"
                    f"{representative.ingestion_run_id}/"
                    f"{representative.raw_artifact_reference}"
                ),
                content_sha256=representative.content_sha256,
                language=representative.language,
                entity_match_evidence=entity_links,
                source_metadata={
                    "canonical_article_id": canonical_id,
                    "representative_occurrence_key": representative.occurrence_key,
                    "occurrence_count": len(occurrences),
                    "ogdc_occurrence_count": len(ogdc_occurrences),
                    "source_observed_at": (
                        representative.source_observed_at.isoformat()
                        if representative.source_observed_at is not None
                        else None
                    ),
                },
            )
        )
    return tuple(records)


def _publisher_reference(news: NewsEvidence, count: int) -> dict[str, Any]:
    return {
        "input_kind": "publisher_news",
        "status": "LOADED" if news.run_references else "NOT_SUPPLIED",
        "record_count": count,
        "run_references": [dict(item) for item in news.run_references],
        "database_fingerprints": dict(news.database_fingerprints),
        "permission_policy_reference": dict(news.permission_policy_reference),
    }


def _no_publisher_reference() -> dict[str, Any]:
    return {
        "input_kind": "publisher_news",
        "status": "NOT_SUPPLIED",
        "record_count": 0,
        "reason": "No optional publisher-news run was supplied.",
    }


def _psx_blocked_reference(policy_path: str | Path) -> dict[str, Any]:
    policy = load_evidence_permission_policy(
        policy_path,
        repository_root=REPOSITORY_ROOT,
    )
    decision = policy.decision("psx-announcement", "manual_import")
    if decision.get("status") != "REVIEW_REQUIRED":
        raise ValueError(
            "PSX policy is no longer REVIEW_REQUIRED; integrate an explicitly "
            "authorized immutable input before auditing it"
        )
    return {
        "input_kind": "psx_announcement_evidence",
        "provider_id": "psx-announcement",
        "provider_mode": "manual_import",
        "status": "BLOCKED_PERMISSION_REVIEW",
        "record_count": 0,
        "run_reference": None,
        "decision_id": decision.get("decision_id"),
        "permission_status": decision.get("status"),
        "automated_access": decision.get("automated_access"),
        "permission_policy_reference": policy.reference(),
        "reason": (
            "No authorized PSX input was supplied; public-page automation and "
            "unattested manual import remain outside the approved boundary."
        ),
    }


def _load_gdelt_runs(
    directories: Iterable[str | Path],
) -> tuple[list[EvidenceRecord], list[dict[str, Any]]]:
    records: list[EvidenceRecord] = []
    references: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for directory in directories:
        loaded = load_evidence_run(directory)
        if loaded.verification.provider_id != "gdelt-gkg-2.1":
            raise ValueError("A supplied GDELT input uses a different provider")
        if any(item.evidence_type != "gdelt_metadata" for item in loaded.records):
            raise ValueError("A supplied GDELT input contains another evidence type")
        overlap = seen_ids.intersection(item.evidence_id for item in loaded.records)
        if overlap:
            raise ValueError("GDELT evidence identity is duplicated across runs")
        seen_ids.update(item.evidence_id for item in loaded.records)
        records.extend(loaded.records)
        references.append({"input_kind": "gdelt_gkg", **dict(loaded.reference)})
    return records, references


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = args.run_id or make_evidence_coverage_run_id()
    try:
        records, input_references = _load_gdelt_runs(args.gdelt_run_dir)
        input_references.append(
            _psx_blocked_reference(args.evidence_permission_policy)
        )

        if args.publisher_news_run_dir:
            news_policy = load_source_permission_policy(
                args.news_permission_policy,
                args.news_source_assessment,
            )
            news = load_news_evidence(
                args.publisher_news_run_dir,
                real_mode=True,
                database_path=args.news_database or None,
                allow_legacy_unsealed=False,
                source_permission_policy=news_policy,
            )
            publisher_records = adapt_publisher_news(news)
            records.extend(publisher_records)
            input_references.append(
                _publisher_reference(news, len(publisher_records))
            )
        else:
            input_references.append(_no_publisher_reference())

        criteria, readiness_reference = _load_readiness_criteria(
            args.readiness_config
        )
        input_references.append(readiness_reference)
        stock = load_stock_evidence(
            args.vertical_slice_dir,
            args.model_training_dir,
            symbol="OGDC",
        )
        config = CoverageConfig(
            market_timezone=args.market_timezone,
            decision_cutoff=parse_cutoff(args.decision_cutoff),
        )
        computation = compute_evidence_coverage(
            records,
            stock,
            config,
            criteria=criteria,
        )
        destination = publish_evidence_coverage_run(
            computation,
            output_root=args.output_dir,
            input_references=input_references,
            stock_reference=stock.reference,
            repository_root=REPOSITORY_ROOT,
            run_id=run_id,
        )
        verification = verify_evidence_coverage_pack(destination)
    except Exception as exc:
        _json_print(
            {
                "status": "FAIL",
                "run_id": run_id,
                "error_type": type(exc).__name__,
                "error": _GENERIC_FAILURE,
                "academic_disclaimer": ACADEMIC_DISCLAIMER,
            }
        )
        return 1

    _json_print(
        {
            **verification.to_dict(),
            "artifact_directory": f"evidence_coverage/{destination.name}",
            "profile_input_counts": dict(
                computation.summary["profile_input_counts"]
            ),
            "psx_input_status": "BLOCKED_PERMISSION_REVIEW",
            "academic_disclaimer": ACADEMIC_DISCLAIMER,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
