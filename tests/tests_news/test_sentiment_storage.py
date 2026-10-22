from __future__ import annotations

from datetime import datetime, timezone
import sqlite3

import pytest

from paisa.news import storage as storage_module
from paisa.news.pipeline import run_news_pipeline
from paisa.news.providers.fixture import FixtureNewsProvider
from paisa.news.storage import DATABASE_SCHEMA_VERSION, NewsStorage


SOURCE_RUN_ID = "NEWS-20261022-140000-sentiment-storage"


@pytest.fixture
def completed_news_storage(tmp_path):
    database = tmp_path / "data" / "news.sqlite3"
    provider = FixtureNewsProvider(
        [
            {
                "title": "OGDC reports higher profit",
                "summary": "Oil and Gas Development Company beat expectations.",
                "published_at": "2026-10-22T08:30:00Z",
                "language": "en",
                "source_url": "https://example.test/ogdc-profit",
            },
            {
                "title": "Market closes unchanged",
                "summary": "The benchmark index ended near its opening level.",
                "published_at": "2026-10-22T09:00:00Z",
                "language": "en-us",
                "source_url": "https://example.test/market-close",
            },
        ],
        fetched_at=datetime(2026, 10, 22, 10, 0, tzinfo=timezone.utc),
    )
    summary = run_news_pipeline(
        provider,
        output_dir="artifacts/news_ingestion",
        database_path="data/news.sqlite3",
        storage_root=tmp_path,
        run_id=SOURCE_RUN_ID,
        started_at=datetime(2026, 10, 22, 9, 59, tzinfo=timezone.utc),
    )
    assert summary.run_status == "PASS"
    return NewsStorage(database).initialize()


def _register_model(
    storage: NewsStorage,
    suffix: str,
    *,
    revision: str | None = None,
) -> str:
    return storage.register_sentiment_model_revision(
        model_revision_id=f"FINBERT-{suffix}",
        model_repository="example/financial-classifier",
        model_revision=revision or suffix * 40,
        tokenizer_repository="example/financial-classifier",
        tokenizer_revision=revision or suffix * 40,
        original_labels={"0": "positive", "1": "negative", "2": "neutral"},
        label_mapping={
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        },
        license_identifier="Apache-2.0",
        library_versions={"transformers": "test", "torch": "test"},
        created_at="2026-10-22T10:01:00Z",
    )


def _begin_sentiment_run(
    storage: NewsStorage,
    run_id: str,
    model_revision_id: str,
    *,
    policy: str = "title-summary-body-v1",
) -> None:
    storage.begin_sentiment_run(
        run_id=run_id,
        source_ingestion_run_id=SOURCE_RUN_ID,
        model_revision_id=model_revision_id,
        input_policy_id=policy,
        started_at="2026-10-22T10:02:00Z",
        device="cpu",
        batch_size=2,
        maximum_input_characters=10_000,
        maximum_sequence_length=512,
        maximum_articles=10,
        timeout_seconds=30,
        artifact_directory=f"artifacts/sentiment_inference/{run_id}",
        configuration={"offline": True},
    )


def _save_output(
    storage: NewsStorage,
    run_id: str,
    article: dict[str, object],
    *,
    input_digest: str = "a" * 64,
    **overrides: object,
) -> tuple[str, bool]:
    values = {
        "inference_run_id": run_id,
        "article_id": article["article_id"],
        "selected_fields": ["title", "summary"],
        "input_text_sha256": input_digest,
        "character_count": 80,
        "original_character_count": 120,
        "character_truncation_applied": True,
        "tokenizer_token_count": 24,
        "token_truncation_applied": False,
        "maximum_input_characters": 10_000,
        "truncation_applied": True,
        "original_language": article["language"],
        "class_labels": ["negative", "neutral", "positive"],
        "negative_probability": 0.1,
        "neutral_probability": 0.2,
        "positive_probability": 0.7,
        "predicted_label": "positive",
        "predicted_probability": 0.7,
        "inference_timestamp": "2026-10-22T10:03:00Z",
        "library_versions": {"transformers": "test", "torch": "test"},
        "warnings": [],
    }
    values.update(overrides)
    return storage.save_article_sentiment(**values)


def test_v5_schema_and_bounded_completed_run_input_iteration(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    with storage.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "sentiment_model_revisions",
        "sentiment_inference_runs",
        "article_sentiment_outputs",
        "sentiment_inference_errors",
    } <= tables

    batches = list(
        storage.iter_sentiment_input_articles(
            SOURCE_RUN_ID, batch_size=1, maximum_articles=2
        )
    )
    assert [len(batch) for batch in batches] == [1, 1]
    articles = [article for batch in batches for article in batch]
    assert [article["article_id"] for article in articles] == sorted(
        article["article_id"] for article in articles
    )
    assert all(article["source_ingestion_run_id"] == SOURCE_RUN_ID for article in articles)
    assert all(article["canonical_owner_run_id"] == SOURCE_RUN_ID for article in articles)
    assert all(article["title"] and article["summary"] for article in articles)
    assert [article["ogdc_linked"] for article in articles].count(True) == 1
    assert all(isinstance(article["mentioned_tickers"], list) for article in articles)


def test_model_revision_and_article_output_are_immutable_and_idempotent(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    article = next(storage.iter_sentiment_input_articles(SOURCE_RUN_ID, batch_size=1))[0]
    model_v1 = _register_model(storage, "a")
    assert _register_model(storage, "a") == model_v1
    with pytest.raises(ValueError, match="different metadata"):
        storage.register_sentiment_model_revision(
            model_revision_id=model_v1,
            model_repository="example/financial-classifier",
            model_revision="a" * 40,
            original_labels={"0": "negative"},
            label_mapping={"negative": "negative"},
            license_identifier="different",
            library_versions={},
        )

    first_run = "SENT-20261022-100200-first"
    _begin_sentiment_run(storage, first_run, model_v1)
    output_id, inserted = _save_output(storage, first_run, article)
    assert inserted is True
    stored = storage.get_article_sentiment(
        str(article["article_id"]), model_v1, "title-summary-body-v1"
    )
    assert stored is not None
    assert stored["sentiment_output_id"] == output_id
    assert stored["published_at"] == article["published_at"]
    assert stored["available_at"] == article["available_at"]
    assert stored["fetched_at"] == article["fetched_at"]
    assert stored["character_count"] == 80
    assert stored["original_character_count"] == 120
    assert stored["character_truncation_applied"] == 1
    assert stored["token_truncation_applied"] == 0
    assert stored["maximum_input_characters"] == 10_000
    assert stored["truncation_applied"] == 1
    storage.finalize_sentiment_run(first_run, status="PASS", summary={"scored": 1})

    repeat_run = "SENT-20261022-100300-repeat"
    _begin_sentiment_run(storage, repeat_run, model_v1)
    repeated_id, repeated_insert = _save_output(storage, repeat_run, article)
    assert (repeated_id, repeated_insert) == (output_id, False)
    with pytest.raises(RuntimeError, match="disagrees"):
        _save_output(storage, repeat_run, article, input_digest="b" * 64)
    storage.finalize_sentiment_run(repeat_run, status="PASS", summary={"skipped": 1})

    model_v2 = _register_model(storage, "b")
    second_revision_run = "SENT-20261022-100400-new-revision"
    _begin_sentiment_run(storage, second_revision_run, model_v2)
    second_id, second_inserted = _save_output(
        storage, second_revision_run, article
    )
    assert second_inserted is True
    assert second_id != output_id
    storage.finalize_sentiment_run(
        second_revision_run, status="PASS", summary={"scored": 1}
    )

    with storage.connection() as connection:
        count = connection.execute(
            """
            SELECT count(*) FROM article_sentiment_outputs
            WHERE article_id = ?
            """,
            (article["article_id"],),
        ).fetchone()[0]
        assert count == 2
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                """
                UPDATE sentiment_model_revisions SET license_identifier = 'changed'
                WHERE model_revision_id = ?
                """,
                (model_v1,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                """
                UPDATE article_sentiment_outputs SET predicted_label = 'negative'
                WHERE sentiment_output_id = ?
                """,
                (output_id,),
            )


def test_timing_context_errors_and_final_status_are_database_guarded(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    article = next(storage.iter_sentiment_input_articles(SOURCE_RUN_ID, batch_size=1))[0]
    model = _register_model(storage, "c")
    run_id = "SENT-20261022-100500-errors"
    _begin_sentiment_run(storage, run_id, model)

    with pytest.raises(sqlite3.IntegrityError, match="article evidence"):
        storage.save_article_sentiment(
            inference_run_id=run_id,
            article_id=article["article_id"],
            selected_fields=["title"],
            input_text_sha256="c" * 64,
            character_count=20,
            tokenizer_token_count=8,
            truncation_applied=False,
            original_language=article["language"],
            published_at=article["published_at"],
            available_at="2026-10-23T10:00:00Z",
            fetched_at=article["fetched_at"],
            class_labels=["negative", "neutral", "positive"],
            negative_probability=0.7,
            neutral_probability=0.2,
            positive_probability=0.1,
            predicted_label="negative",
            predicted_probability=0.7,
            library_versions={},
        )

    error_id = storage.save_sentiment_error(
        inference_run_id=run_id,
        article_id=article["article_id"],
        stage="language_validation",
        error_type="UnsupportedLanguageError",
        message="The declared language is unsupported.",
        original_language=article["language"],
        warning_codes=["unsupported_language"],
        retryable=False,
        details={"language": article["language"]},
    )
    assert storage.list_sentiment_errors(run_id)[0]["error_id"] == error_id
    with pytest.raises(RuntimeError, match="cannot contain errors"):
        storage.finalize_sentiment_run(run_id, status="PASS")
    storage.finalize_sentiment_run(run_id, status="PARTIAL", summary={"failed": 1})

    with pytest.raises(RuntimeError, match="finalized"):
        storage.save_sentiment_error(
            inference_run_id=run_id,
            stage="inference",
            error_type="LateWrite",
            message="must fail",
        )
    with storage.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                """
                UPDATE sentiment_inference_runs SET summary_json = '{}'
                WHERE run_id = ?
                """,
                (run_id,),
            )


def test_output_rejects_inconsistent_label_and_truncation_provenance(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    article = next(
        storage.iter_sentiment_input_articles(SOURCE_RUN_ID, batch_size=1)
    )[0]
    model = _register_model(storage, "d")
    run_id = "SENT-20261022-100600-invalid-provenance"
    _begin_sentiment_run(storage, run_id, model)

    with pytest.raises(ValueError, match="class_labels"):
        _save_output(
            storage,
            run_id,
            article,
            class_labels=["positive", "negative", "neutral"],
        )
    with pytest.raises(ValueError, match="character_truncation_applied"):
        _save_output(
            storage,
            run_id,
            article,
            original_character_count=80,
        )
    with pytest.raises(ValueError, match="token_truncation_applied"):
        _save_output(
            storage,
            run_id,
            article,
            tokenizer_token_count=513,
        )

    _save_output(storage, run_id, article)
    storage.finalize_sentiment_run(run_id, status="PASS", summary={"scored": 1})


def _create_versioned_database(path, version: int) -> None:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        for stage in range(1, version + 1):
            storage_module._apply_schema_stage(connection, stage)
            connection.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, '2026-10-22T10:00:00Z')
                """,
                (stage, storage_module._MIGRATION_NAMES[stage]),
            )
            connection.execute(f"PRAGMA user_version = {stage}")
        connection.commit()


def _rewrite_as_recognized_legacy_v3(path) -> None:
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        secondary = [
            (row["type"], row["name"], row["tbl_name"], row["sql"])
            for row in connection.execute(
                """
                SELECT type, name, tbl_name, sql FROM sqlite_master
                WHERE type IN ('index', 'trigger')
                  AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
                """
            )
        ]
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        for object_type, name, _table, _sql in secondary:
            if object_type == "trigger":
                connection.execute(f"DROP TRIGGER {name}")
        targets = tuple(storage_module._RECOGNIZED_LEGACY_V3_TABLE_SQL)
        for table in targets:
            legacy = f"{table}__legacy"
            sql = storage_module._RECOGNIZED_LEGACY_V3_TABLE_SQL[table].replace(
                f"CREATE TABLE {table}", f"CREATE TABLE {legacy}", 1
            )
            connection.execute(sql)
            columns = [
                row["name"]
                for row in connection.execute(f"PRAGMA table_info({table})")
            ]
            column_sql = ", ".join(columns)
            connection.execute(
                f"INSERT INTO {legacy} ({column_sql}) "
                f"SELECT {column_sql} FROM {table}"
            )
        for table in targets:
            connection.execute(f"DROP TABLE {table}")
            connection.execute(f"ALTER TABLE {table}__legacy RENAME TO {table}")
        for object_type, _name, table, sql in secondary:
            if object_type == "trigger" or (
                object_type == "index" and table in targets
            ):
                connection.execute(sql)
        connection.commit()


def test_exact_retained_legacy_v3_signature_repairs_then_migrates_to_v5(tmp_path):
    database = tmp_path / "legacy-v3.sqlite3"
    _create_versioned_database(database, 3)
    legacy = NewsStorage(database)
    legacy.begin_run(
        run_id="NEWS-20261022-100000-legacy",
        provider_id="fixture",
        provider_mode="fixture",
        configured_source="fixture://legacy",
        data_kind="fixture",
        started_at="2026-10-22T10:00:00Z",
        artifact_directory="artifacts/news_ingestion/legacy",
    )
    raw_id = legacy.save_raw(
        ingestion_run_id="NEWS-20261022-100000-legacy",
        raw_record_id="legacy-1",
        provider_id="fixture",
        fetched_at="2026-10-22T10:00:00Z",
        response_checksum="d" * 64,
        data_kind="fixture",
        raw_artifact_reference="raw/legacy.json",
        payload={"title": "Legacy valid row"},
    )
    legacy.save_article(
        article_id="legacy-article",
        schema_version="1.0",
        provider="fixture",
        source_name="Fixture",
        source_type="fixture",
        title="Legacy valid row",
        published_at="2026-10-22T09:00:00Z",
        available_at="2026-10-22T10:00:00Z",
        fetched_at="2026-10-22T10:00:00Z",
        content_sha256="e" * 64,
        ingestion_run_id="NEWS-20261022-100000-legacy",
        raw_evidence_id=raw_id,
        raw_artifact_reference="raw/legacy.json",
    )
    _rewrite_as_recognized_legacy_v3(database)

    migrated = NewsStorage(database).initialize()

    with migrated.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 5
        assert connection.execute(
            "SELECT count(*) FROM normalized_articles"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT count(*) FROM article_ingestions"
        ).fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        latest = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version DESC"
        ).fetchone()
    assert tuple(latest) == (DATABASE_SCHEMA_VERSION, "financial_sentiment_foundation")


def test_unrecognized_v3_drift_is_not_repaired(tmp_path):
    database = tmp_path / "tampered-v3.sqlite3"
    _create_versioned_database(database, 3)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE normalized_articles ADD COLUMN surprise TEXT")

    with pytest.raises(RuntimeError, match="schema verification failed"):
        NewsStorage(database).initialize()

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
