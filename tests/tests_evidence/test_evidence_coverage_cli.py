from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paisa.news.coverage import NewsEvidence, NewsOccurrence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUDIT_SCRIPT = REPOSITORY_ROOT / "scripts" / "audit_contemporaneous_evidence.py"
VERIFY_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_evidence_coverage.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_contemporaneous_evidence", AUDIT_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _occurrence(
    *,
    key: str,
    canonical: str,
    fetched_at: datetime,
    available_at: datetime,
    ogdc: bool,
    run_id: str = "NEWS-20260101-000000-test0001",
) -> NewsOccurrence:
    article_id = f"ARTICLE-{key}"
    link_id = "NEWSLINK-" + hashlib.sha256(
        "\0".join((article_id, "OGDC", "OGDC", "title", "exact")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return NewsOccurrence(
        occurrence_key=f"{run_id}:{article_id}",
        ingestion_run_id=run_id,
        article_id=article_id,
        canonical_article_id=canonical,
        provider="gdelt-doc",
        provider_mode="real",
        source_name="example.com",
        source_url=f"https://example.com/{key}",
        title=f"Title {key}",
        published_at=None,
        available_at=available_at,
        fetched_at=fetched_at,
        published_timezone=None,
        language="english",
        tickers=("OGDC",) if ogdc else (),
        companies=("Oil & Gas Development Company Limited",) if ogdc else (),
        sectors=(),
        content_sha256=("a" if key == "first" else "b") * 64,
        raw_artifact_reference=f"raw/{key}.json",
        data_class="real",
        duplicate_of=None,
        entity_links=(
            (
                {
                    "link_id": link_id,
                    "article_id": article_id,
                    "ticker": "OGDC",
                    "company_name": "Oil and Gas Development Company Limited",
                    "matched_alias": "OGDC",
                    "match_method": "exact",
                    "match_location": "title",
                    "confidence": "exact",
                },
            )
            if ogdc
            else ()
        ),
    )


def _news(*occurrences: NewsOccurrence) -> NewsEvidence:
    run_ids = sorted({item.ingestion_run_id for item in occurrences})
    return NewsEvidence(
        occurrences=occurrences,
        run_references=tuple(
            {
                "run_id": run_id,
                "source_permission_decision": {
                    "passed": True,
                    "decision_id": "fixture-permission-v1",
                },
            }
            for run_id in run_ids
        ),
        database_fingerprints={"provided": False, "files": []},
    )


def test_publisher_adapter_deduplicates_and_preserves_ogdc_link() -> None:
    module = _load_audit_module()
    instant = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    news = _news(
        _occurrence(
            key="first",
            canonical="CANONICAL-1",
            fetched_at=instant,
            available_at=instant,
            ogdc=False,
        ),
        _occurrence(
            key="second",
            canonical="CANONICAL-1",
            fetched_at=instant + timedelta(minutes=1),
            available_at=instant + timedelta(minutes=1),
            ogdc=True,
        ),
    )

    records = module.adapt_publisher_news(news)

    assert len(records) == 1
    assert records[0].ticker == "OGDC"
    assert records[0].source_record_id == "CANONICAL-1"
    assert records[0].source_metadata["occurrence_count"] == 2
    assert records[0].source_metadata["ogdc_occurrence_count"] == 1
    assert records[0].entity_match_evidence[0]["ticker"] == "OGDC"
    assert records[0].entity_match_evidence[0]["ingestion_run_id"] == (
        "NEWS-20260101-000000-test0001"
    )
    assert records[0].available_at == instant + timedelta(minutes=1)
    assert records[0].fetched_at == instant + timedelta(minutes=1)
    assert records[0].content_sha256 == "b" * 64
    assert records[0].raw_evidence_reference.endswith("raw/second.json")
    assert records[0].source_metadata["representative_occurrence_key"].endswith(
        ":ARTICLE-second"
    )


def test_publisher_adapter_preserves_cross_run_occurrence_link_identity() -> None:
    module = _load_audit_module()
    instant = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    first_run = "NEWS-20260101-000000-test0001"
    linked_run = "NEWS-20260102-000000-test0002"
    news = _news(
        _occurrence(
            key="first",
            canonical="CANONICAL-CROSS-RUN",
            fetched_at=instant,
            available_at=instant,
            ogdc=False,
            run_id=first_run,
        ),
        _occurrence(
            key="linked",
            canonical="CANONICAL-CROSS-RUN",
            fetched_at=instant + timedelta(minutes=1),
            available_at=instant + timedelta(minutes=1),
            ogdc=True,
            run_id=linked_run,
        ),
    )
    news = NewsEvidence(
        occurrences=news.occurrences,
        run_references=(
            {
                "run_id": first_run,
                "source_permission_decision": {
                    "passed": True,
                    "decision_id": "general-run-permission",
                },
            },
            {
                "run_id": linked_run,
                "source_permission_decision": {
                    "passed": True,
                    "decision_id": "linked-run-permission",
                },
            },
        ),
        database_fingerprints=news.database_fingerprints,
    )

    record = module.adapt_publisher_news(news)[0]

    assert record.source_metadata["representative_occurrence_key"].startswith(
        f"{linked_run}:"
    )
    assert record.available_at == instant + timedelta(minutes=1)
    assert record.raw_evidence_reference.startswith(f"news_ingestion/{linked_run}/")
    assert record.permission_policy_id == "linked-run-permission"
    assert record.entity_match_evidence[0]["ingestion_run_id"] == linked_run
    assert {
        item["ingestion_run_id"]
        for item in record.source_metadata["occurrence_identities"]
    } == {first_run, linked_run}


def test_publisher_adapter_rejects_historical_backdating() -> None:
    module = _load_audit_module()
    instant = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)
    news = _news(
        _occurrence(
            key="first",
            canonical="CANONICAL-1",
            fetched_at=instant,
            available_at=instant - timedelta(seconds=1),
            ogdc=False,
        )
    )

    with pytest.raises(ValueError, match="backdating"):
        module.adapt_publisher_news(news)


def test_verification_cli_does_not_leak_invalid_host_path(tmp_path: Path) -> None:
    missing = tmp_path / "private" / "ECOV-missing"
    completed = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--run-dir", str(missing)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "FAIL"
    assert payload["verification"] == "FAIL"
    assert str(tmp_path) not in completed.stdout
    assert payload["academic_disclaimer"] == (
        "Academic simulation; not financial advice."
    )
