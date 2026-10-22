from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from paisa.news.pipeline import run_news_pipeline
from paisa.news.providers.fixture import FixtureNewsProvider
from paisa.news.storage import NewsStorage
import paisa.sentiment.pipeline as sentiment_pipeline
from paisa.sentiment.artifacts import SentimentRunCollisionError
from paisa.sentiment.config import FINBERT_MODEL_REVISION
from paisa.sentiment.inference import SentimentInferenceConfig
from paisa.sentiment.pipeline import run_sentiment_inference


SOURCE_RUN_ID = "NEWS-20261022-100000-feedbeef"
SENTIMENT_RUN_ID = "SENT-20261022-110000-deadbeef"
NOW = datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc)


def _create_source_run(tmp_path: Path, *, second_language: str = "en-us") -> Path:
    database = tmp_path / "data" / "news.sqlite3"
    provider = FixtureNewsProvider(
        [
            {
                "provider_record_id": "one",
                "source_url": "https://publisher.example/one",
                "title": "OGDC reports higher production",
                "summary": "DO-NOT-DUPLICATE-SUMMARY one",
                "body": "DO-NOT-DUPLICATE-BODY one",
                "published_at": "2026-10-22T09:00:00Z",
                "language": "en-us",
            },
            {
                "provider_record_id": "two",
                "source_url": "https://publisher.example/two",
                "title": "A second market headline",
                "summary": "DO-NOT-DUPLICATE-SUMMARY two",
                "body": "DO-NOT-DUPLICATE-BODY two",
                "published_at": "2026-10-22T09:30:00Z",
                "language": second_language,
            },
        ],
        fetched_at=NOW,
    )
    summary = run_news_pipeline(
        provider,
        output_dir=tmp_path / "artifacts" / "news_ingestion",
        database_path=database,
        storage_root=tmp_path,
        run_id=SOURCE_RUN_ID,
        started_at=NOW,
    )
    assert summary.run_status == "PASS"
    return database


def _run(tmp_path: Path, mock_backend, *, run_id: str = SENTIMENT_RUN_ID):
    return run_sentiment_inference(
        source_ingestion_run_id=SOURCE_RUN_ID,
        database_path=tmp_path / "data" / "news.sqlite3",
        output_dir=tmp_path / "artifacts" / "sentiment_inference",
        model_cache=tmp_path / "private-cache" / "credential-sentinel" / "huggingface",
        storage_root=tmp_path,
        run_id=run_id,
        backend=mock_backend,
        config=SentimentInferenceConfig(batch_size=1, maximum_articles=10),
        audit_sample_size=1,
    )


def _artifact_root(tmp_path: Path, run_id: str = SENTIMENT_RUN_ID) -> Path:
    return tmp_path / "artifacts" / "sentiment_inference" / run_id


def test_pipeline_publishes_checksum_bound_body_free_run(
    tmp_path, mock_backend, capsys
):
    database = _create_source_run(tmp_path)

    summary = _run(tmp_path, mock_backend)

    assert summary.status == "PASS"
    assert summary.articles_requested == 2
    assert summary.articles_successfully_scored == 2
    assert summary.ogdc_linked_articles_scored == 1
    assert summary.label_counts == {"positive": 2}
    assert [len(batch) for batch in mock_backend.batches] == [1, 1]

    root = _artifact_root(tmp_path)
    assert {path.name for path in root.iterdir()} == {
        "run_manifest.json",
        "model_manifest.json",
        "input_manifest.json",
        "article_sentiment.csv",
        "inference_errors.json",
        "audit_sample.csv",
        "quality_report.json",
        "final_summary.json",
    }
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["status"] == "PASS"
    assert manifest["artifact_inventory_self_exclusion"] == "run_manifest.json"
    for item in manifest["artifact_inventory"]:
        payload = (root / item["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == item["sha256"]
        assert len(payload) == item["size_bytes"]

    public_bytes = b"".join(path.read_bytes() for path in root.iterdir())
    assert b"DO-NOT-DUPLICATE-SUMMARY" not in public_bytes
    assert b"DO-NOT-DUPLICATE-BODY" not in public_bytes
    assert str(tmp_path / "private-cache").encode() not in public_bytes
    assert b"private-cache/huggingface" not in public_bytes
    assert b"credential-sentinel" not in public_bytes
    captured = capsys.readouterr()
    assert "DO-NOT-DUPLICATE-SUMMARY" not in captured.out + captured.err
    assert "DO-NOT-DUPLICATE-BODY" not in captured.out + captured.err

    with (root / "article_sentiment.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["available_at"] == "2026-10-22T10:00:00Z" for row in rows)
    assert all(row["inference_timestamp"] != row["available_at"] for row in rows)

    storage = NewsStorage(database).initialize()
    assert storage.get_run(SOURCE_RUN_ID)["status"] == "PASS"
    assert storage.get_sentiment_run(SENTIMENT_RUN_ID)["status"] == "PASS"


def test_exact_rerun_is_idempotently_skipped_without_model_call(tmp_path, mock_backend):
    _create_source_run(tmp_path)
    first = _run(tmp_path, mock_backend)
    assert first.articles_successfully_scored == 2

    second_backend = type(mock_backend)()
    second = _run(
        tmp_path,
        second_backend,
        run_id="SENT-20261022-110100-cafebabe",
    )

    assert second.status == "PASS"
    assert second.articles_requested == 2
    assert second.articles_successfully_scored == 0
    assert second.articles_skipped == 2
    assert second_backend.batches == []


def test_different_model_revision_creates_distinct_outputs(tmp_path, mock_backend):
    database = _create_source_run(tmp_path)
    _run(tmp_path, mock_backend)

    changed_backend = type(mock_backend)()
    changed_backend.model_revision = "1" * 40
    changed_backend.tokenizer_revision = "1" * 40
    second = _run(
        tmp_path,
        changed_backend,
        run_id="SENT-20261022-110200-facefeed",
    )

    assert second.status == "PASS"
    assert second.articles_successfully_scored == 2
    storage = NewsStorage(database).initialize()
    with storage.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM article_sentiment_outputs"
        ).fetchone()[0] == 4
        assert connection.execute(
            "SELECT count(*) FROM sentiment_model_revisions"
        ).fetchone()[0] == 2


def test_model_load_failure_publishes_fail_without_corrupting_source(
    tmp_path, mock_backend
):
    database = _create_source_run(tmp_path)

    class FailingBackend(type(mock_backend)):
        @property
        def library_versions(self):
            raise RuntimeError("SECRET-DOWNLOAD-URL")

    summary = _run(tmp_path, FailingBackend())

    assert summary.status == "FAIL"
    root = _artifact_root(tmp_path)
    assert json.loads((root / "run_manifest.json").read_text())["status"] == "FAIL"
    error_text = (root / "inference_errors.json").read_text()
    assert "Pinned sentiment runtime failed" in error_text
    assert "SECRET-DOWNLOAD-URL" not in error_text
    storage = NewsStorage(database).initialize()
    assert storage.get_run(SOURCE_RUN_ID)["status"] == "PASS"
    assert storage.get_sentiment_run(SENTIMENT_RUN_ID)["status"] == "FAIL"
    assert storage.list_article_sentiments_for_run(SENTIMENT_RUN_ID) == []


def test_unsupported_language_is_retained_as_partial_skip(tmp_path, mock_backend):
    _create_source_run(tmp_path, second_language="ur")

    summary = _run(tmp_path, mock_backend)

    assert summary.status == "PARTIAL"
    assert summary.articles_successfully_scored == 1
    assert summary.articles_skipped == 1
    assert summary.articles_failed == 0
    errors = json.loads(
        (_artifact_root(tmp_path) / "inference_errors.json").read_text()
    )
    assert errors["errors"][0]["error_code"] == "unsupported_language"


def test_existing_run_collision_is_rejected_without_overwrite(tmp_path, mock_backend):
    _create_source_run(tmp_path)
    first = _run(tmp_path, mock_backend)
    manifest_before = (_artifact_root(tmp_path) / "run_manifest.json").read_bytes()

    with pytest.raises(SentimentRunCollisionError):
        _run(tmp_path, mock_backend)

    assert (_artifact_root(tmp_path) / "run_manifest.json").read_bytes() == manifest_before
    assert first.model_revision == FINBERT_MODEL_REVISION


def test_watchdog_bounds_the_entire_multi_batch_invocation(
    tmp_path, mock_backend, monkeypatch
):
    database = _create_source_run(tmp_path)
    ticks = iter((0.0, 0.25, 1.25, 1.5))
    monkeypatch.setattr(sentiment_pipeline, "monotonic", lambda: next(ticks))

    summary = run_sentiment_inference(
        source_ingestion_run_id=SOURCE_RUN_ID,
        database_path=database,
        output_dir=tmp_path / "artifacts" / "sentiment_inference",
        model_cache=tmp_path / "private-cache" / "huggingface",
        storage_root=tmp_path,
        run_id=SENTIMENT_RUN_ID,
        backend=mock_backend,
        config=SentimentInferenceConfig(
            batch_size=1,
            maximum_articles=10,
            timeout_seconds=1.0,
        ),
        audit_sample_size=1,
    )

    assert summary.status == "FAIL"
    assert summary.articles_successfully_scored == 1
    assert len(mock_backend.batches) == 1
    errors = json.loads(
        (_artifact_root(tmp_path) / "inference_errors.json").read_text()
    )
    assert errors["errors"][-1]["error_code"] == "SentimentInferenceTimeoutError"


def test_artifact_write_failure_finalizes_database_as_fail(
    tmp_path, mock_backend, monkeypatch
):
    database = _create_source_run(tmp_path)

    def fail_write(*_args, **_kwargs):
        raise OSError("private filesystem detail")

    monkeypatch.setattr(sentiment_pipeline, "write_json", fail_write)

    with pytest.raises(OSError, match="private filesystem detail"):
        _run(tmp_path, mock_backend)

    storage = NewsStorage(database).initialize()
    assert storage.get_run(SOURCE_RUN_ID)["status"] == "PASS"
    assert storage.get_sentiment_run(SENTIMENT_RUN_ID)["status"] == "FAIL"
    errors = storage.list_sentiment_errors(SENTIMENT_RUN_ID)
    assert errors[-1]["error_type"] == "artifact_publication_failed"
    assert not _artifact_root(tmp_path).exists()


def test_sqlite_finalization_failure_publishes_only_truthful_fail(
    tmp_path, mock_backend, monkeypatch
):
    database = _create_source_run(tmp_path)
    original = NewsStorage.finalize_sentiment_run
    calls = 0

    def fail_once(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient finalization failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NewsStorage, "finalize_sentiment_run", fail_once)

    summary = _run(tmp_path, mock_backend)

    assert summary.status == "FAIL"
    root = _artifact_root(tmp_path)
    assert json.loads((root / "run_manifest.json").read_text())["status"] == "FAIL"
    assert json.loads((root / "final_summary.json").read_text())["status"] == "FAIL"
    storage = NewsStorage(database).initialize()
    assert storage.get_sentiment_run(SENTIMENT_RUN_ID)["status"] == "FAIL"


def test_driver_error_after_exact_commit_publishes_matching_pack(
    tmp_path, mock_backend, monkeypatch
):
    database = _create_source_run(tmp_path)
    original = NewsStorage.finalize_sentiment_run

    def commit_then_raise(self, *args, **kwargs):
        original(self, *args, **kwargs)
        raise RuntimeError("simulated driver error after commit")

    monkeypatch.setattr(NewsStorage, "finalize_sentiment_run", commit_then_raise)

    summary = _run(tmp_path, mock_backend)

    assert summary.status == "PASS"
    root = _artifact_root(tmp_path)
    artifact_summary = json.loads((root / "final_summary.json").read_text())
    storage = NewsStorage(database).initialize()
    persisted = storage.get_sentiment_run(SENTIMENT_RUN_ID)
    assert persisted is not None
    assert persisted["status"] == artifact_summary["status"]
    assert persisted["completed_at"] == artifact_summary["completed_at"]
    assert json.loads(persisted["summary_json"]) == artifact_summary


def test_driver_error_after_mismatched_commit_refuses_publication(
    tmp_path, mock_backend, monkeypatch
):
    database = _create_source_run(tmp_path)
    original = NewsStorage.finalize_sentiment_run

    def commit_different_summary_then_raise(self, run_id, **kwargs):
        altered = dict(kwargs["summary"])
        altered["articles_requested"] = 999
        original(self, run_id, **{**kwargs, "summary": altered})
        raise RuntimeError("simulated mismatched commit")

    monkeypatch.setattr(
        NewsStorage,
        "finalize_sentiment_run",
        commit_different_summary_then_raise,
    )

    with pytest.raises(RuntimeError, match="does not match the staged run"):
        _run(tmp_path, mock_backend)

    assert not _artifact_root(tmp_path).exists()
    storage = NewsStorage(database).initialize()
    persisted = storage.get_sentiment_run(SENTIMENT_RUN_ID)
    assert persisted is not None
    assert persisted["status"] == "PASS"
    assert json.loads(persisted["summary_json"])["articles_requested"] == 999


def test_interruption_before_database_finalization_never_exposes_pass(
    tmp_path, mock_backend, monkeypatch
):
    database = _create_source_run(tmp_path)

    def interrupt_finalization(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        NewsStorage,
        "finalize_sentiment_run",
        interrupt_finalization,
    )

    with pytest.raises(KeyboardInterrupt):
        _run(tmp_path, mock_backend)

    assert not _artifact_root(tmp_path).exists()
    storage = NewsStorage(database).initialize()
    assert storage.get_run(SOURCE_RUN_ID)["status"] == "PASS"
    assert storage.get_sentiment_run(SENTIMENT_RUN_ID)["status"] == "RUNNING"


def test_publication_failure_after_database_commit_is_recoverable(
    tmp_path, mock_backend, monkeypatch
):
    database = _create_source_run(tmp_path)
    original = sentiment_pipeline.publish_staged_run
    calls = 0

    def fail_publication_once(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("publication rename failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        sentiment_pipeline,
        "publish_staged_run",
        fail_publication_once,
    )

    with pytest.raises(OSError, match="publication rename failed"):
        _run(tmp_path, mock_backend)

    assert not _artifact_root(tmp_path).exists()
    staging = (
        tmp_path
        / "artifacts"
        / "sentiment_inference"
        / f".{SENTIMENT_RUN_ID}.staging"
    )
    assert staging.is_dir()
    storage = NewsStorage(database).initialize()
    assert storage.get_sentiment_run(SENTIMENT_RUN_ID)["status"] == "PASS"

    recovered = _run(tmp_path, mock_backend)

    assert recovered.status == "PASS"
    assert _artifact_root(tmp_path).is_dir()
    assert not staging.exists()
    assert calls == 2


def test_recovery_refuses_tampered_manifest_contract(
    tmp_path, mock_backend, monkeypatch
):
    _create_source_run(tmp_path)
    original = sentiment_pipeline.publish_staged_run

    def fail_publication(*_args, **_kwargs):
        raise OSError("publication rename failed")

    monkeypatch.setattr(sentiment_pipeline, "publish_staged_run", fail_publication)
    with pytest.raises(OSError, match="publication rename failed"):
        _run(tmp_path, mock_backend)

    staging = (
        tmp_path
        / "artifacts"
        / "sentiment_inference"
        / f".{SENTIMENT_RUN_ID}.staging"
    )
    manifest_path = staging / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["device"] = "cuda"
    manifest["financial_advice"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sentiment_pipeline, "publish_staged_run", original)

    with pytest.raises(RuntimeError, match="evidence seal is invalid"):
        _run(tmp_path, mock_backend)

    assert not _artifact_root(tmp_path).exists()


def test_recovery_refuses_payload_and_manifest_checksum_tampering(
    tmp_path, mock_backend, monkeypatch
):
    _create_source_run(tmp_path)
    original = sentiment_pipeline.publish_staged_run

    def fail_publication(*_args, **_kwargs):
        raise OSError("publication rename failed")

    monkeypatch.setattr(sentiment_pipeline, "publish_staged_run", fail_publication)
    with pytest.raises(OSError, match="publication rename failed"):
        _run(tmp_path, mock_backend)

    staging = (
        tmp_path
        / "artifacts"
        / "sentiment_inference"
        / f".{SENTIMENT_RUN_ID}.staging"
    )
    csv_path = staging / "article_sentiment.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"tampered-row\n")
    manifest_path = staging / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    inventory_entry = next(
        item
        for item in manifest["artifact_inventory"]
        if item["path"] == "article_sentiment.csv"
    )
    inventory_entry["sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    inventory_entry["size_bytes"] = csv_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sentiment_pipeline, "publish_staged_run", original)

    with pytest.raises(RuntimeError, match="evidence seal is invalid"):
        _run(tmp_path, mock_backend)

    assert not _artifact_root(tmp_path).exists()
