from __future__ import annotations

from datetime import datetime, timezone
import json
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
    digest = (suffix if suffix in "abcdef0123456789" else "a") * 64
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
        backend_identity_sha256=digest,
        snapshot_inventory=[
            {"path": "config.json", "sha256": digest, "size_bytes": 1},
            {"path": "model.safetensors", "sha256": digest, "size_bytes": 1},
            {"path": "special_tokens_map.json", "sha256": digest, "size_bytes": 1},
            {"path": "tokenizer_config.json", "sha256": digest, "size_bytes": 1},
            {"path": "vocab.txt", "sha256": digest, "size_bytes": 1},
        ],
        model_config_sha256=digest,
        tokenizer_config_sha256=digest,
        model_weights_sha256=digest,
        tokenizer_vocabulary_sha256=digest,
        tokenizer_special_tokens_sha256=digest,
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
        "final_token_count": 24,
        "truncation_side": "right",
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


def _register_input(
    storage: NewsStorage,
    run_id: str,
    article: dict[str, object],
    *,
    ordinal: int = 0,
) -> None:
    storage.register_sentiment_run_input(
        run_id,
        str(article["article_id"]),
        input_ordinal=ordinal,
    )


def _complete_output(
    storage: NewsStorage,
    run_id: str,
    article: dict[str, object],
    output_id: str,
    *,
    inserted: bool,
) -> None:
    storage.complete_sentiment_run_input(
        run_id,
        str(article["article_id"]),
        outcome="GENERATED" if inserted else "REUSED",
        sentiment_output_id=output_id,
    )


def _finalize_run(storage: NewsStorage, run_id: str, *, status: str) -> None:
    storage.finalize_sentiment_run(
        run_id,
        status=status,
        completed_at="2026-10-22T10:04:00Z",
        summary={
            "sealed_payload_inventory_sha256": "e" * 64,
            "run_manifest_contract_sha256": "f" * 64,
        },
    )


def test_v6_schema_and_bounded_completed_run_input_iteration(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    with storage.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
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
        "sentiment_run_articles",
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
            backend_identity_sha256="a" * 64,
            snapshot_inventory=[
                {"path": "config.json", "sha256": "a" * 64, "size_bytes": 1}
            ],
            model_config_sha256="a" * 64,
            tokenizer_config_sha256="a" * 64,
            model_weights_sha256="a" * 64,
            tokenizer_vocabulary_sha256="a" * 64,
            tokenizer_special_tokens_sha256="a" * 64,
        )

    first_run = "SENT-20261022-100200-first"
    _begin_sentiment_run(storage, first_run, model_v1)
    _register_input(storage, first_run, article)
    output_id, inserted = _save_output(storage, first_run, article)
    assert inserted is True
    _complete_output(
        storage, first_run, article, output_id, inserted=inserted
    )
    stored = storage.get_article_sentiment(
        str(article["article_id"]), model_v1, "title-summary-body-v1"
    )
    assert stored is None
    _finalize_run(storage, first_run, status="PASS")
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
    repeat_run = "SENT-20261022-100300-repeat"
    _begin_sentiment_run(storage, repeat_run, model_v1)
    _register_input(storage, repeat_run, article)
    repeated_id, repeated_insert = _save_output(storage, repeat_run, article)
    assert (repeated_id, repeated_insert) == (output_id, False)
    with pytest.raises(RuntimeError, match="disagrees"):
        _save_output(storage, repeat_run, article, input_digest="b" * 64)
    _complete_output(
        storage,
        repeat_run,
        article,
        repeated_id,
        inserted=repeated_insert,
    )
    _finalize_run(storage, repeat_run, status="PASS")

    model_v2 = _register_model(storage, "b")
    second_revision_run = "SENT-20261022-100400-new-revision"
    _begin_sentiment_run(storage, second_revision_run, model_v2)
    _register_input(storage, second_revision_run, article)
    second_id, second_inserted = _save_output(
        storage, second_revision_run, article
    )
    assert second_inserted is True
    assert second_id != output_id
    _complete_output(
        storage,
        second_revision_run,
        article,
        second_id,
        inserted=second_inserted,
    )
    _finalize_run(storage, second_revision_run, status="PASS")

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


def test_failed_run_cannot_finalize_with_reusable_output(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    article = next(
        storage.iter_sentiment_input_articles(SOURCE_RUN_ID, batch_size=1)
    )[0]
    model = _register_model(storage, "f")
    run_id = "SENT-20261022-100450-failed-output"
    _begin_sentiment_run(storage, run_id, model)
    _register_input(storage, run_id, article)
    output_id, inserted = _save_output(storage, run_id, article)
    _complete_output(storage, run_id, article, output_id, inserted=inserted)

    with pytest.raises(sqlite3.IntegrityError, match="certified evidence"):
        _finalize_run(storage, run_id, status="FAIL")

    assert storage.get_sentiment_run(run_id)["status"] == "RUNNING"
    storage.reset_unpublished_sentiment_outputs(run_id)
    _finalize_run(storage, run_id, status="FAIL")
    assert storage.get_article_sentiment(
        str(article["article_id"]), model, "title-summary-body-v1"
    ) is None


def test_timing_context_errors_and_final_status_are_database_guarded(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    article = next(storage.iter_sentiment_input_articles(SOURCE_RUN_ID, batch_size=1))[0]
    model = _register_model(storage, "c")
    run_id = "SENT-20261022-100500-errors"
    _begin_sentiment_run(storage, run_id, model)
    _register_input(storage, run_id, article)

    with pytest.raises(sqlite3.IntegrityError, match="article evidence"):
        storage.save_article_sentiment(
            inference_run_id=run_id,
            article_id=article["article_id"],
            selected_fields=["title"],
            input_text_sha256="c" * 64,
            character_count=20,
            tokenizer_token_count=8,
            final_token_count=8,
            truncation_side="right",
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
    storage.complete_sentiment_run_input(
        run_id,
        str(article["article_id"]),
        outcome="FAILED",
        error_id=error_id,
    )
    _finalize_run(storage, run_id, status="PARTIAL")

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


def test_running_output_is_not_globally_reused_and_can_be_discarded_atomically(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    article = next(
        storage.iter_sentiment_input_articles(SOURCE_RUN_ID, batch_size=1)
    )[0]
    model = _register_model(storage, "e")
    interrupted = "SENT-20261022-100510-stale000"
    replacement = "SENT-20261022-100511-replace0"
    _begin_sentiment_run(storage, interrupted, model)
    _register_input(storage, interrupted, article)
    stale_output_id, inserted = _save_output(storage, interrupted, article)
    assert inserted is True

    assert storage.get_article_sentiment(
        str(article["article_id"]), model, "title-summary-body-v1"
    ) is None

    _begin_sentiment_run(storage, replacement, model)
    _register_input(storage, replacement, article)
    with pytest.raises(RuntimeError, match="unfinished run"):
        _save_output(storage, replacement, article)

    storage.discard_running_sentiment_run(interrupted)
    assert storage.get_sentiment_run(interrupted) is None
    assert storage.get_article_sentiment_by_id(stale_output_id) is None

    replacement_output_id, replacement_inserted = _save_output(
        storage, replacement, article
    )
    assert replacement_inserted is True
    _complete_output(
        storage,
        replacement,
        article,
        replacement_output_id,
        inserted=True,
    )
    _finalize_run(storage, replacement, status="PASS")


def test_sqlite_rejects_fabricated_zero_output_pass(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    model = _register_model(storage, "f")
    run_id = "SENT-20261022-100520-fabricat"
    _begin_sentiment_run(storage, run_id, model)
    completed_at = "2026-10-22T10:04:00Z"
    fabricated = {
        "summary_schema_version": "2.0",
        "schema_version": "1.0",
        "run_id": run_id,
        "status": "PASS",
        "source_ingestion_run_id": SOURCE_RUN_ID,
        "completed_at": completed_at,
        "articles_requested": 999,
        "unique_canonical_articles": 999,
        "articles_successfully_scored": 999,
        "articles_skipped": 0,
        "articles_failed": 0,
        "truncated_input_count": 0,
        "label_counts": {"negative": 0, "neutral": 0, "positive": 999},
        "sealed_payload_inventory_sha256": "e" * 64,
        "run_manifest_contract_sha256": "f" * 64,
    }
    with storage.connection() as connection:
        with pytest.raises(sqlite3.IntegrityError, match="certified evidence"):
            connection.execute(
                """
                UPDATE sentiment_inference_runs
                SET completed_at = ?, status = 'PASS', summary_json = ?,
                    sealed_payload_inventory_sha256 = ?,
                    run_manifest_contract_sha256 = ?
                WHERE run_id = ?
                """,
                (
                    completed_at,
                    json.dumps(fabricated, sort_keys=True),
                    "e" * 64,
                    "f" * 64,
                    run_id,
                ),
            )
    with pytest.raises(RuntimeError, match="persisted input evidence"):
        storage.finalize_sentiment_run(
            run_id,
            status="PASS",
            completed_at=completed_at,
            summary=fabricated,
        )


def test_pass_requires_terminal_output_association_and_both_recovery_seals(
    completed_news_storage: NewsStorage,
):
    storage = completed_news_storage
    article = next(
        storage.iter_sentiment_input_articles(SOURCE_RUN_ID, batch_size=1)
    )[0]
    model = _register_model(storage, "a", revision="9" * 40)
    run_id = "SENT-20261022-100530-assoc001"
    _begin_sentiment_run(storage, run_id, model)
    _register_input(storage, run_id, article)
    output_id, inserted = _save_output(storage, run_id, article)
    assert inserted is True

    with pytest.raises(RuntimeError, match="pending inputs"):
        storage.finalize_sentiment_run(
            run_id,
            status="PASS",
            summary={
                "sealed_payload_inventory_sha256": "e" * 64,
                "run_manifest_contract_sha256": "f" * 64,
            },
        )

    _complete_output(storage, run_id, article, output_id, inserted=True)
    with pytest.raises(ValueError, match="run_manifest_contract_sha256"):
        storage.finalize_sentiment_run(
            run_id,
            status="PASS",
            summary={"sealed_payload_inventory_sha256": "e" * 64},
        )

    _finalize_run(storage, run_id, status="PASS")
    persisted = storage.get_sentiment_run(run_id)
    assert persisted is not None
    assert persisted["sealed_payload_inventory_sha256"] == "e" * 64
    assert persisted["run_manifest_contract_sha256"] == "f" * 64


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
    _register_input(storage, run_id, article)

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
    with pytest.raises(ValueError, match="final_token_count is required"):
        _save_output(
            storage,
            run_id,
            article,
            final_token_count=None,
        )
    with pytest.raises(ValueError, match="truncation_side is required"):
        _save_output(
            storage,
            run_id,
            article,
            truncation_side=None,
        )
    with pytest.raises(ValueError, match="token_truncation_applied"):
        _save_output(
            storage,
            run_id,
            article,
            tokenizer_token_count=513,
            final_token_count=512,
        )

    output_id, inserted = _save_output(storage, run_id, article)
    _complete_output(storage, run_id, article, output_id, inserted=inserted)
    _finalize_run(storage, run_id, status="PASS")


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


def test_v5_model_identity_remains_legacy_while_hashed_rerun_can_register(
    tmp_path,
):
    database = tmp_path / "populated-v5.sqlite3"
    _create_versioned_database(database, 5)
    repository = "ProsusAI/finbert"
    revision = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
    legacy_id = "SMR-c1858cea79620827dd9ab42c"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            INSERT INTO sentiment_model_revisions(
                model_revision_id, schema_version, model_repository,
                model_revision, tokenizer_repository, tokenizer_revision,
                original_labels_json, label_mapping_json, license_identifier,
                library_versions_json, model_config_sha256,
                tokenizer_config_sha256, created_at
            ) VALUES (?, '1.0', ?, ?, ?, ?, ?, ?, 'CC-BY-NC-SA-4.0',
                      '{}', NULL, NULL, '2026-10-22T10:01:00Z')
            """,
            (
                legacy_id,
                repository,
                revision,
                repository,
                revision,
                json.dumps(["positive", "negative", "neutral"]),
                json.dumps({"negative": 1, "neutral": 2, "positive": 0}),
            ),
        )
        # Keep actual version-5 child references, including output owned by an
        # interrupted RUNNING sentiment run. The narrowly bypassed ingestion
        # evidence triggers are restored before initialize() verifies the
        # exact v5 schema.
        bypassed_triggers = (
            "trg_ingestion_runs_insert_running",
            "trg_raw_source_records_insert_running",
            "trg_normalized_articles_insert_running",
            "trg_article_ingestions_insert_running",
        )
        trigger_sql = {
            name: connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
                (name,),
            ).fetchone()[0]
            for name in bypassed_triggers
        }
        for name in bypassed_triggers:
            connection.execute(f"DROP TRIGGER {name}")
        connection.execute(
            """
            INSERT INTO ingestion_runs(
                run_id, schema_version, provider_id, provider_mode,
                configured_source, data_kind, started_at, completed_at,
                status, artifact_directory, summary_json
            ) VALUES (
                'NEWS-20261022-100000-legacyv5', '1.0', 'fixture', 'fixture',
                'fixture://legacy-v5', 'fixture', '2026-10-22T10:00:00Z',
                '2026-10-22T10:01:00Z', 'PASS',
                'artifacts/news_ingestion/NEWS-20261022-100000-legacyv5', '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO raw_source_records(
                raw_evidence_id, ingestion_run_id, raw_record_id, provider_id,
                provider_record_id, configured_source, fetched_at, data_kind,
                raw_artifact_reference, payload_json
            ) VALUES (
                'raw-legacy-running', 'NEWS-20261022-100000-legacyv5',
                'raw-legacy-running', 'fixture', 'item-1', 'fixture://legacy-v5',
                '2026-10-22T10:00:00Z', 'fixture', 'raw/item-1.json', '{}'
            )
            """
        )
        connection.execute(
            """
            INSERT INTO normalized_articles(
                article_id, schema_version, provider, source_name, source_type,
                source_url, canonical_url, title, normalized_title, summary,
                published_at, published_timezone, available_at, fetched_at,
                language, content_sha256, first_ingestion_run_id,
                raw_artifact_reference, created_at
            ) VALUES (
                'article-legacy-running', '1.0', 'fixture', 'Fixture', 'fixture',
                'https://example.test/item-1', 'https://example.test/item-1',
                'OGDC legacy item', 'ogdc legacy item', 'Summary',
                '2026-10-22T09:00:00Z', 'UTC', '2026-10-22T10:00:00Z',
                '2026-10-22T10:00:00Z', 'en-us', ?,
                'NEWS-20261022-100000-legacyv5', 'raw/item-1.json',
                '2026-10-22T10:00:00Z'
            )
            """,
            ("b" * 64,),
        )
        connection.execute(
            """
            INSERT INTO article_ingestions(
                occurrence_id, ingestion_run_id, raw_evidence_id,
                candidate_article_id, canonical_article_id, available_at,
                fetched_at, raw_artifact_reference, normalized_payload_json,
                created_at
            ) VALUES (
                'occurrence-legacy-running', 'NEWS-20261022-100000-legacyv5',
                'raw-legacy-running', 'article-legacy-running',
                'article-legacy-running', '2026-10-22T10:00:00Z',
                '2026-10-22T10:00:00Z', 'raw/item-1.json', '{}',
                '2026-10-22T10:00:00Z'
            )
            """
        )
        for name in bypassed_triggers:
            connection.execute(trigger_sql[name])
        connection.execute(
            """
            INSERT INTO sentiment_inference_runs(
                run_id, schema_version, source_ingestion_run_id,
                model_revision_id, input_policy_id, started_at, status,
                device, batch_size, maximum_input_characters,
                maximum_sequence_length, maximum_articles, timeout_seconds,
                artifact_directory, configuration_json, summary_json
            ) VALUES (
                'SENT-20261022-100200-legacyv5', '1.0',
                'NEWS-20261022-100000-legacyv5', ?,
                'title-summary-body-v1', '2026-10-22T10:02:00Z', 'RUNNING',
                'cpu', 2, 4000, 512, 10, 30,
                'artifacts/sentiment_inference/SENT-20261022-100200-legacyv5',
                '{}', '{}'
            )
            """,
            (legacy_id,),
        )
        connection.execute(
            """
            INSERT INTO article_sentiment_outputs(
                sentiment_output_id, schema_version, inference_run_id,
                article_id, source_ingestion_run_id, model_revision_id,
                input_policy_id, selected_fields_json, input_text_sha256,
                character_count, original_character_count,
                character_truncation_applied, token_truncation_applied,
                maximum_input_characters, tokenizer_token_count,
                truncation_applied, maximum_sequence_length, original_language,
                published_at, available_at, fetched_at, class_labels_json,
                negative_probability, neutral_probability,
                positive_probability, predicted_label, predicted_probability,
                inference_timestamp, device, batch_size,
                library_versions_json, warning_codes_json
            ) VALUES (
                'SENTOUT-legacy-running', '1.0',
                'SENT-20261022-100200-legacyv5', 'article-legacy-running',
                'NEWS-20261022-100000-legacyv5', ?,
                'title-summary-body-v1', '["title"]', ?, 10, 10, 0, 0,
                4000, 4, 0, 512, 'en-us', '2026-10-22T09:00:00Z',
                '2026-10-22T10:00:00Z', '2026-10-22T10:00:00Z',
                '["negative","neutral","positive"]', 0.1, 0.2, 0.7,
                'positive', 0.7, '2026-10-22T10:03:00Z', 'cpu', 2, '{}', '[]'
            )
            """,
            (legacy_id, "c" * 64),
        )
        connection.execute(
            """
            INSERT INTO sentiment_inference_runs(
                run_id, schema_version, source_ingestion_run_id,
                model_revision_id, input_policy_id, started_at, status,
                device, batch_size, maximum_input_characters,
                maximum_sequence_length, maximum_articles, timeout_seconds,
                artifact_directory, configuration_json, summary_json
            ) VALUES (
                'SENT-20261022-100300-legacy-pass', '1.0',
                'NEWS-20261022-100000-legacyv5', ?,
                'title-only-v1', '2026-10-22T10:02:00Z', 'RUNNING',
                'cpu', 2, 4000, 512, 10, 30,
                'artifacts/sentiment_inference/SENT-20261022-100300-legacy-pass',
                '{}', '{}'
            )
            """,
            (legacy_id,),
        )
        connection.execute(
            """
            INSERT INTO article_sentiment_outputs(
                sentiment_output_id, schema_version, inference_run_id,
                article_id, source_ingestion_run_id, model_revision_id,
                input_policy_id, selected_fields_json, input_text_sha256,
                character_count, original_character_count,
                character_truncation_applied, token_truncation_applied,
                maximum_input_characters, tokenizer_token_count,
                truncation_applied, maximum_sequence_length, original_language,
                published_at, available_at, fetched_at, class_labels_json,
                negative_probability, neutral_probability,
                positive_probability, predicted_label, predicted_probability,
                inference_timestamp, device, batch_size,
                library_versions_json, warning_codes_json
            )
            SELECT
                'SENTOUT-legacy-pass', schema_version,
                'SENT-20261022-100300-legacy-pass', article_id,
                source_ingestion_run_id, model_revision_id, 'title-only-v1',
                selected_fields_json, input_text_sha256, character_count,
                original_character_count, character_truncation_applied,
                token_truncation_applied, maximum_input_characters,
                tokenizer_token_count, truncation_applied,
                maximum_sequence_length, original_language, published_at,
                available_at, fetched_at, class_labels_json,
                negative_probability, neutral_probability,
                positive_probability, predicted_label, predicted_probability,
                inference_timestamp, device, batch_size,
                library_versions_json, warning_codes_json
            FROM article_sentiment_outputs
            WHERE sentiment_output_id = 'SENTOUT-legacy-running'
            """
        )
        connection.execute(
            """
            UPDATE sentiment_inference_runs
            SET status = 'PASS', completed_at = '2026-10-22T10:04:00Z'
            WHERE run_id = 'SENT-20261022-100300-legacy-pass'
            """
        )

    storage = NewsStorage(database).initialize()
    digest = "a" * 64
    hashed_id = storage.register_sentiment_model_revision(
        model_revision_id="SMR-hashbound-v6",
        schema_version="1.0",
        model_repository=repository,
        model_revision=revision,
        tokenizer_repository=repository,
        tokenizer_revision=revision,
        original_labels=["positive", "negative", "neutral"],
        label_mapping={"negative": 1, "neutral": 2, "positive": 0},
        license_identifier="CC-BY-NC-SA-4.0",
        library_versions={},
        backend_identity_sha256=digest,
        snapshot_inventory=[
            {"path": "config.json", "sha256": digest, "size_bytes": 1}
        ],
        model_config_sha256=digest,
        tokenizer_config_sha256=digest,
        model_weights_sha256=digest,
        tokenizer_vocabulary_sha256=digest,
        tokenizer_special_tokens_sha256=digest,
        created_at="2026-10-23T10:00:00Z",
    )

    assert hashed_id == "SMR-hashbound-v6"
    # A later caller may propose a different local alias, but the persisted
    # hash-bound identity remains canonical and must be returned to the run.
    assert storage.register_sentiment_model_revision(
        model_revision_id="SMR-another-alias",
        schema_version="1.0",
        model_repository=repository,
        model_revision=revision,
        tokenizer_repository=repository,
        tokenizer_revision=revision,
        original_labels=["positive", "negative", "neutral"],
        label_mapping={"negative": 1, "neutral": 2, "positive": 0},
        license_identifier="CC-BY-NC-SA-4.0",
        library_versions={},
        backend_identity_sha256=digest,
        snapshot_inventory=[
            {"path": "config.json", "sha256": digest, "size_bytes": 1}
        ],
        model_config_sha256=digest,
        tokenizer_config_sha256=digest,
        model_weights_sha256=digest,
        tokenizer_vocabulary_sha256=digest,
        tokenizer_special_tokens_sha256=digest,
        created_at="2026-10-23T10:01:00Z",
    ) == hashed_id

    with storage.connection() as connection:
        rows = connection.execute(
            """
            SELECT model_revision_id, backend_identity_sha256,
                   model_weights_sha256
            FROM sentiment_model_revisions
            ORDER BY model_revision_id
            """
        ).fetchall()
    assert [row["model_revision_id"] for row in rows] == [
        legacy_id,
        hashed_id,
    ]
    assert rows[0]["backend_identity_sha256"] is None
    assert rows[0]["model_weights_sha256"] is None
    assert rows[1]["backend_identity_sha256"] == digest
    legacy_run = storage.get_sentiment_run("SENT-20261022-100200-legacyv5")
    assert legacy_run is not None
    assert legacy_run["model_revision_id"] == legacy_id
    assert storage.get_article_sentiment(
        "article-legacy-running", legacy_id, "title-only-v1"
    ) is None
    with storage.connection() as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute(
            """
            SELECT count(*) FROM sentiment_run_articles
            WHERE inference_run_id = 'SENT-20261022-100200-legacyv5'
            """
        ).fetchone()[0] == 0
        assert connection.execute(
            """
            SELECT count(*) FROM sentiment_run_articles
            WHERE inference_run_id = 'SENT-20261022-100300-legacy-pass'
              AND outcome = 'GENERATED'
            """
        ).fetchone()[0] == 1

    storage.discard_running_sentiment_run("SENT-20261022-100200-legacyv5")
    assert storage.get_sentiment_run("SENT-20261022-100200-legacyv5") is None
    with storage.connection() as connection:
        assert connection.execute(
            """
            SELECT count(*) FROM article_sentiment_outputs
            WHERE sentiment_output_id = 'SENTOUT-legacy-running'
            """
        ).fetchone()[0] == 0


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


def test_exact_retained_legacy_v3_signature_repairs_then_migrates_to_v6(tmp_path):
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
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 6
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
    assert tuple(latest) == (DATABASE_SCHEMA_VERSION, "sentiment_run_certification")


def test_unrecognized_v3_drift_is_not_repaired(tmp_path):
    database = tmp_path / "tampered-v3.sqlite3"
    _create_versioned_database(database, 3)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE normalized_articles ADD COLUMN surprise TEXT")

    with pytest.raises(RuntimeError, match="schema verification failed"):
        NewsStorage(database).initialize()

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
