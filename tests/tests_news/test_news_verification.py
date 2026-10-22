from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paisa.news.models import ProviderFetchResult, RawCapture, RawNewsRecord
from paisa.news.pipeline import run_news_pipeline
from paisa.news.verification import NewsRunVerificationError, verify_news_run


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERIFY_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_news_run.py"
STARTED = datetime(2026, 10, 22, 16, 0, tzinfo=timezone.utc)
CAPTURE = b"retained feed response for offline replay\n"


class VerificationProvider:
    provider_id = "verification-provider"
    data_class = "real"
    configured_sources = (
        {
            "source_id": "verification-feed",
            "url": "https://news.example.test/feed.xml",
        },
    )

    def fetch(self, *, fetched_at=None) -> ProviderFetchResult:
        acquired = fetched_at or STARTED
        digest = hashlib.sha256(CAPTURE).hexdigest()
        capture = RawCapture(
            capture_id="verification-capture",
            provider_id=self.provider_id,
            configured_source="https://news.example.test/feed.xml",
            source_name="Verification News",
            fetched_at=acquired,
            data_class="real",
            content=CAPTURE,
            filename="feed.xml",
            http_status=200,
            content_type="application/rss+xml",
            response_sha256=digest,
        )
        record = RawNewsRecord(
            raw_record_id="provider-item-1",
            provider_id=self.provider_id,
            configured_source="https://news.example.test/feed.xml",
            source_name="Verification News",
            source_type="rss",
            fetched_at=acquired,
            data_class="real",
            source_url="https://news.example.test/ogdc-result",
            title="OGDC reports a result",
            summary="Oil and Gas Development Company Limited released an update.",
            published_at_raw="2026-10-22T15:00:00Z",
            published_timezone="UTC",
            language="en",
            provider_record_id="publisher-guid-1",
            capture_id=capture.capture_id,
            http_status=200,
            response_content_type=capture.content_type,
            response_sha256=digest,
            raw_payload={"guid": "publisher-guid-1"},
        )
        return ProviderFetchResult(
            provider_id=self.provider_id,
            data_class="real",
            fetched_at=acquired,
            configured_sources=self.configured_sources,
            records=(record,),
            captures=(capture,),
        )


def _completed_run(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-160000-offline-verify"
    summary = run_news_pipeline(
        VerificationProvider(),
        output_dir=artifacts,
        database_path=database,
        storage_root=tmp_path,
        run_id=run_id,
        started_at=STARTED,
    )
    assert summary.run_status == "PASS"
    return artifacts / run_id, database


def test_completed_run_replays_without_database_or_network(tmp_path):
    run_dir, database = _completed_run(tmp_path)
    database.unlink()

    verified = verify_news_run(run_dir)

    assert verified.run_id == run_dir.name
    assert verified.status == "PASS"
    assert verified.capture_count == 1
    assert verified.raw_record_count == 1
    assert verified.normalized_record_count == 1
    assert verified.duplicate_count == 0
    assert verified.replayed is True
    assert verified.database_verified is False


def test_completed_run_requires_matching_database_seal_when_requested(tmp_path):
    run_dir, database = _completed_run(tmp_path)

    verified = verify_news_run(run_dir, database_path=database)

    assert verified.database_verified is True
    assert verified.to_dict()["completion_acceptance"] == (
        "DATABASE_AND_ARTIFACT_BOUND"
    )


def test_historical_dedup_replay_uses_frozen_index_not_live_database(tmp_path):
    artifacts = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    first_id = "NEWS-20261022-160000-offline-owner"
    second_id = "NEWS-20261022-160001-offline-dependent"
    for run_id in (first_id, second_id):
        run_news_pipeline(
            VerificationProvider(),
            output_dir=artifacts,
            database_path=database,
            storage_root=tmp_path,
            run_id=run_id,
            started_at=STARTED,
        )

    second_dir = artifacts / second_id
    database.unlink()
    verified = verify_news_run(second_dir)

    assert verified.duplicate_count == 1
    context = json.loads(
        (second_dir / "deduplication_context.json").read_text(encoding="utf-8")
    )
    assert context["canonical_index_count"] == 1
    owner_manifest_hash = hashlib.sha256(
        (artifacts / first_id / "run_manifest.json").read_bytes()
    ).hexdigest()
    assert context["dependency_runs"] == [
        {
            "run_id": first_id,
            "manifest_sha256": owner_manifest_hash,
        }
    ]


@pytest.mark.parametrize(
    "relative_path",
    [
        "normalized_articles.csv",
        "raw/capture-0001-feed.xml",
        "deduplication_context.json",
        "final_summary.json",
        "run_manifest.json",
    ],
)
def test_verifier_rejects_changed_sealed_or_completion_bound_files(
    tmp_path, relative_path
):
    run_dir, _ = _completed_run(tmp_path)
    target = run_dir / relative_path
    target.write_bytes(target.read_bytes() + b"tampered\n")

    with pytest.raises(NewsRunVerificationError):
        verify_news_run(run_dir)


def test_verifier_rejects_symlinked_artifact(tmp_path):
    run_dir, _ = _completed_run(tmp_path)
    target = run_dir / "normalized_articles.csv"
    external = tmp_path / "outside.csv"
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)

    with pytest.raises(NewsRunVerificationError, match="symlink"):
        verify_news_run(run_dir)


def test_verifier_rejects_symlinked_operational_lock(tmp_path):
    run_dir, _ = _completed_run(tmp_path)
    external = tmp_path / "outside.lock"
    external.write_bytes(b"")
    (run_dir / ".run.lock").symlink_to(external)

    with pytest.raises(NewsRunVerificationError, match="lock"):
        verify_news_run(run_dir)


def test_verifier_rejects_frozen_context_even_if_manifest_is_rehashed(tmp_path):
    run_dir, _ = _completed_run(tmp_path)
    context_path = run_dir / "deduplication_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["ordered_rules"] = [
        "content_sha256",
        "canonical_url",
        "title_source_time",
    ]
    context_path.write_text(json.dumps(context), encoding="utf-8")

    with pytest.raises(NewsRunVerificationError):
        verify_news_run(run_dir)


def test_verify_news_run_cli_reports_pass_and_nonzero_on_tamper(tmp_path):
    run_dir, _ = _completed_run(tmp_path)

    passed = subprocess.run(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--run-dir",
            str(run_dir),
            "--database",
            str(tmp_path / "news.sqlite3"),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert passed.returncode == 0, passed.stdout + passed.stderr
    passed_payload = json.loads(passed.stdout)
    assert passed_payload["verification"] == "PASS"
    assert passed_payload["database_verified"] is True

    (run_dir / "quality_report.json").write_text("{}", encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--run-dir", str(run_dir)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["verification"] == "FAIL"
