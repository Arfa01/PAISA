from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
) -> NewsOccurrence:
    return NewsOccurrence(
        occurrence_key=key,
        ingestion_run_id="NEWS-20260101-000000-test0001",
        article_id=f"ARTICLE-{key}",
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
                    "ticker": "OGDC",
                    "article_id": f"ARTICLE-{key}",
                    "link_id": f"LINK-{key}",
                },
            )
            if ogdc
            else ()
        ),
    )


def _news(*occurrences: NewsOccurrence) -> NewsEvidence:
    return NewsEvidence(
        occurrences=occurrences,
        run_references=(
            {
                "run_id": "NEWS-20260101-000000-test0001",
                "source_permission_decision": {
                    "passed": True,
                    "decision_id": "fixture-permission-v1",
                },
            },
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
