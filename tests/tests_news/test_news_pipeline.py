from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3

import pytest

import paisa.news.pipeline as news_pipeline
from paisa.news.models import ProviderFetchResult, RawCapture, RawNewsRecord
from paisa.news.pipeline import (
    NewsRunCollisionError,
    NewsRunFinalizationError,
    make_news_run_id,
    run_news_pipeline,
    validate_news_run_id,
)
from paisa.news.storage import DATABASE_SCHEMA_VERSION, NewsStorage


STARTED = datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc)
CAPTURE_BYTES = b"permitted captured feed bytes\n"


class StaticRealProvider:
    provider_id = "test-rss"
    data_class = "real"

    def __init__(
        self,
        *,
        with_error: bool = False,
        source_name: str = "Publisher Business",
        source_url: str = "https://publisher.example/ogdc?a=1&utm_source=feed",
        title: str = "OGDC announces discovery",
        summary: str = "Oil and Gas Development Company Limited reports an update.",
        published_at: str = "2026-10-22T09:00:00Z",
        acquired_at: datetime = STARTED,
    ) -> None:
        self.with_error = with_error
        self.source_name = source_name
        self.source_url = source_url
        self.title = title
        self.summary = summary
        self.published_at = published_at
        self.acquired_at = acquired_at
        self.calls = 0
        configured_sources = [
            {
                "source_id": "permitted-business",
                "url": "https://publisher.example/feed.xml",
                "api_token": "must-not-be-persisted",
            },
        ]
        if with_error:
            configured_sources.append(
                {
                    "source_id": "secondary",
                    "url": "https://secondary.example/feed.xml",
                }
            )
        self.configured_sources = tuple(configured_sources)

    def fetch(self, *, fetched_at=None) -> ProviderFetchResult:
        self.calls += 1
        acquired = fetched_at or self.acquired_at
        checksum = hashlib.sha256(CAPTURE_BYTES).hexdigest()
        capture = RawCapture(
            capture_id="capture-1",
            provider_id=self.provider_id,
            configured_source="https://publisher.example/feed.xml",
            source_name=self.source_name,
            fetched_at=acquired,
            data_class="real",
            content=CAPTURE_BYTES,
            filename="feed.xml",
            http_status=200,
            content_type="application/rss+xml",
            response_sha256=checksum,
        )
        record = RawNewsRecord(
            raw_record_id="provider-record-1",
            provider_id=self.provider_id,
            configured_source="https://publisher.example/feed.xml",
            source_name=self.source_name,
            source_type="rss",
            fetched_at=acquired,
            data_class="real",
            source_url=self.source_url,
            title=self.title,
            summary=self.summary,
            body=None,
            published_at_raw=self.published_at,
            published_timezone="UTC",
            language="en",
            provider_record_id="publisher-guid-1",
            capture_id=capture.capture_id,
            http_status=200,
            response_content_type="application/rss+xml",
            response_sha256=checksum,
            raw_payload={"guid": "publisher-guid-1", "title": self.title},
        )
        errors = ()
        if self.with_error:
            from paisa.news.models import ProviderError

            errors = (
                ProviderError(
                    provider_id=self.provider_id,
                    configured_source="https://secondary.example/feed.xml",
                    error_type="network_error",
                    message="secondary source unavailable",
                    occurred_at=acquired,
                ),
            )
        return ProviderFetchResult(
            provider_id=self.provider_id,
            data_class="real",
            fetched_at=acquired,
            configured_sources=self.configured_sources,
            records=(record,),
            captures=(capture,),
            errors=errors,
        )


class ExplodingRealProvider:
    provider_id = "exploding-rss"
    data_class = "real"
    configured_sources = (
        {"source_id": "broken", "url": "https://broken.example/feed.xml"},
    )

    def __init__(self) -> None:
        self.calls = 0

    def fetch(self, *, fetched_at=None):
        self.calls += 1
        raise RuntimeError("controlled provider failure")


class SecretExplodingProvider(ExplodingRealProvider):
    configured_sources = (
        {
            "url": "https://alice:SECRET@publisher.example/feed?api_key=TOPSECRET",
            "authorization": "Basic PRIVATE",
            "client_secret": "CLIENTSECRET",
            "X-Api-Key": "HEADERSECRET",
            "NEWS_API_KEY": "news-prefixed-value",
            "AWS_SECRET_ACCESS_KEY": "aws-prefixed-value",
            "DB_PASSWORD": "db-prefixed-value",
            "MY_ACCESS_TOKEN": "token-prefixed-value",
        },
    )

    def fetch(self, *, fetched_at=None):
        self.calls += 1
        raise RuntimeError(
            "request failed for https://alice:SECRET@publisher.example/feed?"
            "api_key=TOPSECRET; client_secret=CLIENTSECRET; "
            "password: TOP SECRET; Authorization: Basic PRIVATE; "
            "X-Api-Key: HEADERSECRET; NEWS_API_KEY=news-error-value; "
            "AWS_SECRET_ACCESS_KEY=aws-error-value; "
            "DB_PASSWORD=db-error-value; MY_ACCESS_TOKEN=token-error-value"
        )


class BadChecksumProvider(StaticRealProvider):
    def fetch(self, *, fetched_at=None) -> ProviderFetchResult:
        result = super().fetch(fetched_at=fetched_at)
        capture = result.captures[0]
        return ProviderFetchResult(
            provider_id=result.provider_id,
            data_class=result.data_class,
            fetched_at=result.fetched_at,
            configured_sources=result.configured_sources,
            records=result.records,
            captures=(
                RawCapture(
                    capture_id=capture.capture_id,
                    provider_id=capture.provider_id,
                    configured_source=capture.configured_source,
                    source_name=capture.source_name,
                    fetched_at=capture.fetched_at,
                    data_class=capture.data_class,
                    content=capture.content,
                    filename=capture.filename,
                    http_status=capture.http_status,
                    content_type=capture.content_type,
                    response_sha256="0" * 64,
                ),
            ),
        )


class UndeclaredSourceProvider(StaticRealProvider):
    def fetch(self, *, fetched_at=None) -> ProviderFetchResult:
        result = super().fetch(fetched_at=fetched_at)
        undeclared = "https://undeclared.example/feed.xml"
        return replace(
            result,
            captures=(
                replace(result.captures[0], configured_source=undeclared),
            ),
            records=(
                replace(result.records[0], configured_source=undeclared),
            ),
        )


class SecretMetadataProvider(StaticRealProvider):
    effective_config = {
        "NEWS_API_KEY": "news-config-value",
        "AWS_SECRET_ACCESS_KEY": "aws-config-value",
        "nested": {
            "DB_PASSWORD": "db-config-value",
            "MY_ACCESS_TOKEN": "token-config-value",
        },
    }

    def fetch(self, *, fetched_at=None) -> ProviderFetchResult:
        result = super().fetch(fetched_at=fetched_at)
        secret_source_name = (
            "Publisher X-Api-Key: SOURCESECRET; "
            "DB_PASSWORD=db-metadata-value; "
            "MY_ACCESS_TOKEN=token-metadata-value"
        )
        return replace(
            result,
            captures=(
                replace(result.captures[0], source_name=secret_source_name),
            ),
            records=(
                replace(
                    result.records[0],
                    source_name=secret_source_name,
                    provider_record_id=(
                        'client_secret="RECORDSECRET"; '
                        "NEWS_API_KEY=news-record-value"
                    ),
                    raw_payload={
                        "password": "PAYLOADSECRET",
                        "AWS_SECRET_ACCESS_KEY": "aws-payload-value",
                    },
                ),
            ),
        )


def _load_cli_module():
    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "run_news_pipeline.py"
    spec = importlib.util.spec_from_file_location("paisa_news_pipeline_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_ids_have_timestamp_format_and_are_unique():
    values = {make_news_run_id(STARTED) for _ in range(32)}

    assert len(values) == 32
    assert all(
        re.fullmatch(r"NEWS-20261022-100000-[0-9a-f]{8}", value)
        for value in values
    )
    assert all(validate_news_run_id(value) == value for value in values)
    for invalid in (
        "NEWS-loose",
        "NEWS-20261022-1000-short",
        "NEWS-20261022-100000-../escape",
        "VSL-20261022-100000-wrong-prefix",
    ):
        with pytest.raises(ValueError):
            validate_news_run_id(invalid)


def test_complete_artifact_tree_inventory_and_raw_preservation(tmp_path):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-artifacts"

    summary = run_news_pipeline(
        StaticRealProvider(),
        output_dir=output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )

    run_dir = output / run_id
    required = {
        "run_manifest.json",
        "provider_config.json",
        "raw/index.json",
        "raw/record-000001.json",
        "normalized_articles.csv",
        "article_entity_links.csv",
        "duplicates.csv",
        "provider_errors.json",
        "quality_report.json",
        "final_summary.json",
    }
    actual = {
        path.relative_to(run_dir).as_posix()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert summary.run_status == "PASS"
    assert required <= actual

    captures = list((run_dir / "raw").glob("capture-*-feed.xml"))
    assert len(captures) == 1
    assert captures[0].read_bytes() == CAPTURE_BYTES
    raw_record = _read_json(run_dir / "raw" / "record-000001.json")
    assert raw_record["raw_record_id"] == "provider-record-1"
    assert raw_record["raw_payload"]["guid"] == "publisher-guid-1"

    provider_config = _read_json(run_dir / "provider_config.json")
    assert provider_config["configured_sources"][0]["api_token"] == "[REDACTED]"
    manifest = _read_json(run_dir / "run_manifest.json")
    assert manifest["status"] == "PASS"
    assert manifest["finalization_state"] == "COMMITTED"
    assert manifest["database"]["schema_version"] == DATABASE_SCHEMA_VERSION
    inventory = manifest["artifact_inventory"]
    assert [entry["path"] for entry in inventory] == sorted(actual - {"run_manifest.json"})
    for entry in inventory:
        path = run_dir / entry["path"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["size_bytes"] == path.stat().st_size


def test_destination_and_database_collisions_refuse_without_overwrite(tmp_path):
    database = tmp_path / "news.sqlite3"
    first_output = tmp_path / "first"
    run_id = "NEWS-20261022-100000-collision"
    provider = StaticRealProvider()
    run_news_pipeline(
        provider,
        output_dir=first_output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )
    run_dir = first_output / run_id
    before = {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    with pytest.raises(NewsRunCollisionError):
        run_news_pipeline(
            provider,
            output_dir=first_output,
            database_path=database,
            run_id=run_id,
            started_at=STARTED,
        )
    after = {
        path.relative_to(run_dir).as_posix(): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert provider.calls == 1

    second_output = tmp_path / "second"
    with pytest.raises(NewsRunCollisionError):
        run_news_pipeline(
            provider,
            output_dir=second_output,
            database_path=database,
            run_id=run_id,
            started_at=STARTED,
        )
    assert provider.calls == 1
    assert not (second_output / run_id).exists()
    assert not list(second_output.glob(f".{run_id}.tmp-*"))


def test_dangling_symlink_destination_is_a_collision(tmp_path):
    output = tmp_path / "artifacts"
    output.mkdir()
    run_id = "NEWS-20261022-100000-dangling"
    destination = output / run_id
    destination.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    provider = StaticRealProvider()

    with pytest.raises(NewsRunCollisionError):
        run_news_pipeline(
            provider,
            output_dir=output,
            database_path=tmp_path / "news.sqlite3",
            run_id=run_id,
            started_at=STARTED,
        )

    assert provider.calls == 0
    assert destination.is_symlink()


def test_explicit_naive_run_time_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        run_news_pipeline(
            StaticRealProvider(),
            output_dir=tmp_path / "artifacts",
            database_path=tmp_path / "news.sqlite3",
            run_id="NEWS-20261022-100000-naive",
            started_at=datetime(2026, 10, 22, 10, 0),
        )


def test_provider_exception_publishes_truthful_fail_and_no_fixture_fallback(tmp_path):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-provider-fail"
    provider = ExplodingRealProvider()

    summary = run_news_pipeline(
        provider,
        output_dir=output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )

    run_dir = output / run_id
    assert provider.calls == 1
    assert summary.run_status == "FAIL"
    assert summary.quality_status == "FAIL"
    assert summary.provider == "exploding-rss"
    assert summary.provider_mode == "real"
    assert summary.raw_record_count == 0
    assert summary.normalized_record_count == 0
    assert _read_json(run_dir / "run_manifest.json")["status"] == "FAIL"
    assert _read_json(run_dir / "final_summary.json")["run_status"] == "FAIL"
    errors = _read_json(run_dir / "provider_errors.json")["errors"]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "RuntimeError"
    assert _read_json(run_dir / "quality_report.json")["provider_error_count"] == 1
    assert _read_json(run_dir / "raw" / "index.json")["records"] == []

    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        run = connection.execute(
            "SELECT * FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        error_count = connection.execute(
            "SELECT count(*) FROM provider_errors WHERE ingestion_run_id = ?",
            (run_id,),
        ).fetchone()[0]
        fixture_count = connection.execute(
            "SELECT count(*) FROM raw_source_records WHERE data_kind = 'fixture'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert run["status"] == "FAIL"
    assert run["data_kind"] == "real"
    assert run["provider_mode"] == "real"
    assert error_count == 1
    assert fixture_count == 0


def test_provider_evidence_checksum_mismatch_becomes_truthful_failure(tmp_path):
    output = tmp_path / "artifacts"
    run_id = "NEWS-20261022-100000-bad-checksum"

    summary = run_news_pipeline(
        BadChecksumProvider(),
        output_dir=output,
        database_path=tmp_path / "news.sqlite3",
        run_id=run_id,
        started_at=STARTED,
    )

    assert summary.run_status == "FAIL"
    assert summary.raw_record_count == 0
    errors = _read_json(output / run_id / "provider_errors.json")["errors"]
    assert errors[0]["error_type"] == "ValueError"
    assert "checksum" in errors[0]["message"].lower()
    assert list((output / run_id / "raw").glob("capture-*")) == []


def test_credentials_are_redacted_from_failed_run_artifacts_and_database(tmp_path):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-redaction"

    summary = run_news_pipeline(
        SecretExplodingProvider(),
        output_dir=output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )

    assert summary.run_status == "FAIL"
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (output / run_id).rglob("*")
        if path.is_file()
    )
    with sqlite3.connect(database) as connection:
        database_text = "\n".join(connection.iterdump())
    combined = artifact_text + database_text
    for secret in (
        "alice:SECRET",
        "TOPSECRET",
        "PRIVATE",
        "CLIENTSECRET",
        "HEADERSECRET",
        "TOP SECRET",
        "news-prefixed-value",
        "aws-prefixed-value",
        "db-prefixed-value",
        "token-prefixed-value",
        "news-error-value",
        "aws-error-value",
        "db-error-value",
        "token-error-value",
    ):
        assert secret not in combined
    assert "[REDACTED]" in combined


def test_credentials_in_success_metadata_are_redacted_everywhere(tmp_path):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-metadata-redaction"

    summary = run_news_pipeline(
        SecretMetadataProvider(),
        output_dir=output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )

    assert summary.run_status == "PASS"
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (output / run_id).rglob("*")
        if path.is_file()
    )
    with sqlite3.connect(database) as connection:
        database_text = "\n".join(connection.iterdump())
    combined = artifact_text + database_text
    for secret in (
        "SOURCESECRET",
        "RECORDSECRET",
        "PAYLOADSECRET",
        "news-config-value",
        "aws-config-value",
        "db-config-value",
        "token-config-value",
        "db-metadata-value",
        "token-metadata-value",
        "news-record-value",
        "aws-payload-value",
    ):
        assert secret not in combined
    assert "[REDACTED]" in combined


@pytest.mark.parametrize(
    "provider, expected_message",
    [
        (UndeclaredSourceProvider(), "not declared"),
        (
            StaticRealProvider(acquired_at=STARTED - timedelta(microseconds=1)),
            "precede run start",
        ),
    ],
)
def test_undeclared_or_backdated_provider_evidence_fails_closed(
    tmp_path, provider, expected_message
):
    run_id = f"NEWS-20261022-100000-invalid-{provider.__class__.__name__.lower()}"
    output = tmp_path / provider.__class__.__name__

    summary = run_news_pipeline(
        provider,
        output_dir=output,
        database_path=tmp_path / f"{provider.__class__.__name__}.sqlite3",
        run_id=run_id,
        started_at=STARTED,
    )

    assert summary.run_status == "FAIL"
    errors = _read_json(output / run_id / "provider_errors.json")["errors"]
    assert expected_message in errors[0]["message"]
    assert _read_json(output / run_id / "raw" / "index.json")["records"] == []


def test_cli_reports_controlled_provider_failure_as_nonzero(tmp_path, monkeypatch, capsys):
    module = _load_cli_module()
    provider = ExplodingRealProvider()
    monkeypatch.setattr(module, "_build_provider", lambda args: provider)
    run_id = "NEWS-20261022-100000-cli-provider-fail"

    exit_code = module.main(
        [
            "--provider",
            "manual-csv",
            "--input",
            "unused.csv",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--database",
            str(tmp_path / "news.sqlite3"),
            "--run-id",
            run_id,
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["run_id"] == run_id
    assert payload["run_status"] == "FAIL"
    assert payload["quality_status"] == "FAIL"
    assert (tmp_path / "artifacts" / run_id / "run_manifest.json").is_file()


def test_cross_run_dedup_preserves_both_evidence_occurrences(tmp_path):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    first_id = "NEWS-20261022-100000-cross-run-a"
    second_id = "NEWS-20261022-110000-cross-run-b"
    first = run_news_pipeline(
        StaticRealProvider(),
        output_dir=output,
        database_path=database,
        run_id=first_id,
        started_at=STARTED,
    )
    second = run_news_pipeline(
        StaticRealProvider(acquired_at=STARTED + timedelta(hours=1)),
        output_dir=output,
        database_path=database,
        run_id=second_id,
        started_at=STARTED + timedelta(hours=1),
    )

    assert first.duplicate_count == 0
    assert second.duplicate_count == 1
    first_rows = (output / first_id / "normalized_articles.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    second_rows = (output / second_id / "normalized_articles.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    first_article_id = first_rows[1].split(",", maxsplit=2)[1]
    assert first_article_id in second_rows[1]
    duplicate_report = (output / second_id / "duplicates.csv").read_text(
        encoding="utf-8"
    )
    assert "canonical_url" in duplicate_report
    assert first_article_id in duplicate_report

    connection = sqlite3.connect(database)
    try:
        counts = {
            table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in (
                "ingestion_runs",
                "raw_source_records",
                "normalized_articles",
                "article_ingestions",
                "deduplication_relationships",
            )
        }
    finally:
        connection.close()
    assert counts == {
        "ingestion_runs": 2,
        "raw_source_records": 2,
        "normalized_articles": 1,
        "article_ingestions": 2,
        "deduplication_relationships": 1,
    }


def test_cross_run_title_source_minute_rule_matches_with_case_normalization(tmp_path):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    first_id = "NEWS-20261022-100000-title-a"
    second_id = "NEWS-20261022-110000-title-b"
    first_provider = StaticRealProvider(
        source_name="Publisher Business",
        source_url="https://publisher.example/first",
        title="OGDC Result",
        summary="first wording",
        published_at="2026-10-22T09:00:01Z",
    )
    second_provider = StaticRealProvider(
        source_name="publisher business",
        source_url="https://publisher.example/second",
        title="  ogdc   result ",
        summary="second wording",
        published_at="2026-10-22T09:00:59Z",
        acquired_at=STARTED + timedelta(hours=1),
    )

    first = run_news_pipeline(
        first_provider,
        output_dir=output,
        database_path=database,
        run_id=first_id,
        started_at=STARTED,
    )
    second = run_news_pipeline(
        second_provider,
        output_dir=output,
        database_path=database,
        run_id=second_id,
        started_at=STARTED + timedelta(hours=1),
    )

    assert first.duplicate_count == 0
    assert second.duplicate_count == 1
    duplicates = (output / second_id / "duplicates.csv").read_text(encoding="utf-8")
    assert "title_source_time" in duplicates


def test_same_captured_evidence_normalizes_deterministically(tmp_path):
    run_id = "NEWS-20261022-100000-deterministic"
    outputs = []
    for name in ("one", "two"):
        output = tmp_path / name / "artifacts"
        run_news_pipeline(
            StaticRealProvider(),
            output_dir=output,
            database_path=tmp_path / name / "news.sqlite3",
            run_id=run_id,
            started_at=STARTED,
        )
        run_dir = output / run_id
        outputs.append(
            {
                filename: (run_dir / filename).read_bytes()
                for filename in (
                    "normalized_articles.csv",
                    "article_entity_links.csv",
                    "duplicates.csv",
                    "raw/record-000001.json",
                    "raw/index.json",
                )
            }
        )

    assert outputs[0] == outputs[1]


def test_provider_error_with_usable_records_is_partial_and_recorded(tmp_path):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-partial"

    summary = run_news_pipeline(
        StaticRealProvider(with_error=True),
        output_dir=output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )

    assert summary.run_status == "PARTIAL"
    assert summary.quality_status == "WARN"
    assert _read_json(output / run_id / "run_manifest.json")["status"] == "PARTIAL"
    connection = sqlite3.connect(database)
    try:
        status = connection.execute(
            "SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        error_count = connection.execute(
            "SELECT count(*) FROM provider_errors WHERE ingestion_run_id = ?",
            (run_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    assert status == "PARTIAL"
    assert error_count == 1


def test_database_finalization_failure_can_never_leave_pass_artifacts(
    tmp_path, monkeypatch
):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-finalize-fail"

    def fail_finalization(self, *args, **kwargs):
        raise OSError("simulated durable finalization failure")

    monkeypatch.setattr(NewsStorage, "finalize_run", fail_finalization)
    with pytest.raises(NewsRunFinalizationError):
        run_news_pipeline(
            StaticRealProvider(),
            output_dir=output,
            database_path=database,
            run_id=run_id,
            started_at=STARTED,
        )

    run_dir = output / run_id
    summary = _read_json(run_dir / "final_summary.json")
    manifest = _read_json(run_dir / "run_manifest.json")
    assert summary["run_status"] == "FAIL"
    assert summary["finalization_state"] == "DATABASE_FINALIZATION_FAILED"
    assert manifest["status"] == "FAIL"
    assert manifest["finalization_state"] == "DATABASE_FINALIZATION_FAILED"
    assert "simulated durable finalization failure" in manifest["finalization_error"]
    with sqlite3.connect(database) as connection:
        status = connection.execute(
            "SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert status == "RUNNING"


def test_committed_manifest_write_failure_is_pending_and_idempotently_recovered(
    tmp_path, monkeypatch
):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-manifest-recovery"
    provider = StaticRealProvider()
    original_atomic_write = news_pipeline._atomic_write_json
    injected = False

    def fail_first_committed_manifest(path, value):
        nonlocal injected
        if (
            not injected
            and path.name == "run_manifest.json"
            and isinstance(value, dict)
            and value.get("finalization_state") == "COMMITTED"
        ):
            injected = True
            raise OSError("simulated committed manifest write failure")
        return original_atomic_write(path, value)

    monkeypatch.setattr(
        news_pipeline,
        "_atomic_write_json",
        fail_first_committed_manifest,
    )
    with pytest.raises(NewsRunFinalizationError, match="can be retried safely"):
        run_news_pipeline(
            provider,
            output_dir=output,
            database_path=database,
            run_id=run_id,
            started_at=STARTED,
        )

    run_dir = output / run_id
    manifest = _read_json(run_dir / "run_manifest.json")
    pending_summary = _read_json(run_dir / "final_summary.json")
    assert provider.calls == 1
    assert manifest["status"] == "PENDING"
    assert manifest["finalization_state"] == "ARTIFACT_COMMIT_RETRY_REQUIRED"
    assert pending_summary["run_status"] == "PENDING"
    assert (run_dir / news_pipeline._FINALIZATION_JOURNAL).is_file()
    with sqlite3.connect(database) as connection:
        database_status = connection.execute(
            "SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert database_status == "RUNNING"

    monkeypatch.setattr(
        news_pipeline,
        "_atomic_write_json",
        original_atomic_write,
    )
    recovered = run_news_pipeline(
        provider,
        output_dir=output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )

    assert recovered.run_status == "PASS"
    assert provider.calls == 1
    assert _read_json(run_dir / "run_manifest.json")["status"] == "PASS"
    assert (
        _read_json(run_dir / "run_manifest.json")["finalization_state"]
        == "COMMITTED"
    )
    assert _read_json(run_dir / "final_summary.json")["run_status"] == "PASS"
    assert not (run_dir / news_pipeline._FINALIZATION_JOURNAL).exists()
    with sqlite3.connect(database) as connection:
        recovered_database_status = connection.execute(
            "SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert recovered_database_status == "PASS"

    with pytest.raises(NewsRunCollisionError):
        run_news_pipeline(
            provider,
            output_dir=output,
            database_path=database,
            run_id=run_id,
            started_at=STARTED,
        )
    assert provider.calls == 1


def test_database_and_failure_marker_errors_retain_recovery_journal(
    tmp_path, monkeypatch
):
    output = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-100000-compensation-recovery"
    provider = StaticRealProvider()
    original_finalize = NewsStorage.finalize_run
    original_atomic_write = news_pipeline._atomic_write_json
    finalize_injected = False
    marker_injected = False

    def fail_first_database_finalization(self, *args, **kwargs):
        nonlocal finalize_injected
        if not finalize_injected:
            finalize_injected = True
            raise OSError("simulated database finalization failure")
        return original_finalize(self, *args, **kwargs)

    def fail_first_database_failure_manifest(path, value):
        nonlocal marker_injected
        if (
            not marker_injected
            and path.name == "run_manifest.json"
            and isinstance(value, dict)
            and value.get("finalization_state") == "DATABASE_FINALIZATION_FAILED"
        ):
            marker_injected = True
            raise OSError("simulated failure-marker write failure")
        return original_atomic_write(path, value)

    monkeypatch.setattr(NewsStorage, "finalize_run", fail_first_database_finalization)
    monkeypatch.setattr(
        news_pipeline,
        "_atomic_write_json",
        fail_first_database_failure_manifest,
    )
    with pytest.raises(NewsRunFinalizationError, match="journal was retained"):
        run_news_pipeline(
            provider,
            output_dir=output,
            database_path=database,
            run_id=run_id,
            started_at=STARTED,
        )

    run_dir = output / run_id
    assert provider.calls == 1
    assert (run_dir / news_pipeline._FINALIZATION_JOURNAL).is_file()
    assert (
        _read_json(run_dir / "run_manifest.json")["finalization_state"]
        == "ARTIFACT_COMMIT_RETRY_REQUIRED"
    )
    assert _read_json(run_dir / "final_summary.json")["run_status"] == "PENDING"
    with sqlite3.connect(database) as connection:
        pending_database_status = connection.execute(
            "SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert pending_database_status == "RUNNING"

    monkeypatch.setattr(NewsStorage, "finalize_run", original_finalize)
    monkeypatch.setattr(
        news_pipeline,
        "_atomic_write_json",
        original_atomic_write,
    )
    recovered = run_news_pipeline(
        provider,
        output_dir=output,
        database_path=database,
        run_id=run_id,
        started_at=STARTED,
    )

    assert recovered.run_status == "PASS"
    assert provider.calls == 1
    assert not (run_dir / news_pipeline._FINALIZATION_JOURNAL).exists()
    assert _read_json(run_dir / "run_manifest.json")["status"] == "PASS"
    with sqlite3.connect(database) as connection:
        final_database_status = connection.execute(
            "SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
    assert final_database_status == "PASS"
