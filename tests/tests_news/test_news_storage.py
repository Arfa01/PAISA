from __future__ import annotations

import json
import sqlite3
import threading

import pytest

from paisa.news import storage as storage_module
from paisa.news.storage import DATABASE_SCHEMA_VERSION, NewsStorage


NOW = "2026-10-22T10:00:00Z"


@pytest.fixture
def storage(tmp_path):
    return NewsStorage(tmp_path / "nested" / "paisa_news.sqlite3").initialize()


def _begin(
    storage: NewsStorage,
    run_id: str,
    *,
    data_kind: str = "real",
    schema_version: str = "1.0",
) -> None:
    storage.begin_run(
        run_id=run_id,
        schema_version=schema_version,
        provider_id="permitted-rss",
        provider_mode="rss",
        configured_source="https://news.example.test/feed.xml",
        data_kind=data_kind,
        started_at=NOW,
        artifact_directory=f"artifacts/news_ingestion/{run_id}",
    )


def _raw(storage: NewsStorage, run_id: str, raw_record_id: str = "item/1") -> str:
    return storage.save_raw(
        ingestion_run_id=run_id,
        raw_record_id=raw_record_id,
        provider_id="permitted-rss",
        provider_record_id="https://news.example.test/item/1",
        configured_source="https://news.example.test/feed.xml",
        fetched_at=NOW,
        http_status=200,
        response_content_type="application/rss+xml",
        response_checksum="b" * 64,
        data_kind="real",
        raw_artifact_reference=f"raw/{raw_record_id.replace('/', '-')}.json",
        payload={"title": "OGDC announces discovery"},
        provider_warnings=[],
        provider_errors=[],
    )


def _article(
    run_id: str,
    raw_evidence_id: str,
    *,
    article_id: str = "article-1",
    canonical_url: str = "https://news.example.test/ogdc-discovery",
    title: str | None = "OGDC announces discovery",
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "article_id": article_id,
        "provider": "permitted-rss",
        "source_name": "Example Financial News",
        "source_type": "rss",
        "source_url": canonical_url + "?utm_source=feed",
        "canonical_url": canonical_url,
        "title": title,
        "summary": "Oil and Gas Development Company reported a discovery.",
        "body": None,
        "published_at": "2026-10-22T09:00:00+05:00",
        "published_timezone": "+05:00",
        "available_at": NOW,
        "fetched_at": NOW,
        "language": "en",
        "mentioned_tickers": ["OGDC"],
        "mentioned_companies": ["Oil and Gas Development Company Limited"],
        "mentioned_sectors": ["energy"],
        "content_sha256": "a" * 64,
        "duplicate_of": None,
        "ingestion_run_id": run_id,
        "raw_evidence_id": raw_evidence_id,
        "raw_artifact_reference": "raw/item-1.json",
        "normalization_warnings": ["body_unavailable"],
    }


def _quality_payload(
    storage: NewsStorage,
    run_id: str,
    *,
    quality_status: str | None = None,
) -> dict[str, object]:
    snapshot = storage.get_run_evidence_snapshot(run_id)
    return {
        "schema_version": "2.0",
        "quality_status": quality_status or snapshot["expected_quality_status"],
        "total_raw_records": snapshot["raw_record_count"],
        "total_normalized_records": snapshot["normalized_occurrence_count"],
        "normalized_occurrence_count": snapshot["normalized_occurrence_count"],
        "new_canonical_article_count": snapshot["new_canonical_article_count"],
        "matched_existing_canonical_count": snapshot[
            "matched_existing_canonical_count"
        ],
        "within_run_duplicate_count": snapshot["within_run_duplicate_count"],
        "historical_duplicate_count": snapshot["historical_duplicate_count"],
        "represented_canonical_count": snapshot["represented_canonical_count"],
        "duplicate_relationship_count": snapshot[
            "duplicate_relationship_count"
        ],
        "duplicate_count": snapshot["duplicate_relationship_count"],
        "deduplication_rule_counts": snapshot["deduplication_rule_counts"],
        "provider_error_count": snapshot["provider_error_count"],
        "rejected_record_count": snapshot["rejected_record_count"],
        "missing_title_count": snapshot["missing_title_count"],
        "missing_publication_time_count": snapshot[
            "missing_publication_time_count"
        ],
        "missing_body_count": snapshot["missing_body_count"],
        "invalid_url_count": snapshot["invalid_url_count"],
        "ogdc_linked_article_count": snapshot["ogdc_linked_article_count"],
    }


def _seal(
    storage: NewsStorage,
    run_id: str,
    *,
    manifest_sha256: str = "2" * 64,
    completion_marker_sha256: str = "3" * 64,
) -> dict[str, object]:
    return storage.seal_run(
        run_id,
        artifact_inventory=[
            {
                "path": "normalized_articles.csv",
                "sha256": "1" * 64,
                "size_bytes": 1,
            }
        ],
        manifest_sha256=manifest_sha256,
        completion_marker_sha256=completion_marker_sha256,
        expected_counts=storage.get_run_evidence_snapshot(run_id),
    )


def _save_quality_and_seal(
    storage: NewsStorage,
    run_id: str,
    *,
    quality_status: str | None = None,
) -> dict[str, object]:
    report = _quality_payload(storage, run_id, quality_status=quality_status)
    storage.save_quality(
        ingestion_run_id=run_id,
        quality_status=report["quality_status"],
        report=report,
    )
    return _seal(storage, run_id)


def _finalize_valid_run(
    storage: NewsStorage,
    run_id: str,
    *,
    status: str = "PASS",
) -> None:
    quality_status = "PASS" if status == "PASS" else "WARN"
    binding = _save_quality_and_seal(
        storage, run_id, quality_status=quality_status
    )
    storage.finalize_run(
        run_id,
        status=status,
        expected_manifest_sha256=binding["manifest_sha256"],
        expected_inventory_sha256=binding["artifact_inventory_sha256"],
        expected_completion_sha256=binding["completion_marker_sha256"],
    )


def _create_versioned_database(path, version: int) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        for stage in range(1, version + 1):
            storage_module._apply_schema_stage(connection, stage)
            connection.execute(
                """
                INSERT INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (stage, storage_module._MIGRATION_NAMES[stage], NOW),
            )
            connection.execute(f"PRAGMA user_version = {stage}")
        connection.commit()
    finally:
        connection.close()


def test_initialize_creates_versioned_schema_and_enables_wal(storage: NewsStorage):
    expected_tables = {
        "schema_migrations",
        "ingestion_runs",
        "raw_source_records",
        "normalized_articles",
        "article_ingestions",
        "article_company_links",
        "deduplication_relationships",
        "provider_errors",
        "quality_reports",
        "sentiment_model_revisions",
        "sentiment_inference_runs",
        "article_sentiment_outputs",
        "sentiment_inference_errors",
        "sentiment_run_articles",
    }
    with storage.connection() as connection:
        tables = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        migrations = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchone()
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert (
            connection.execute("PRAGMA user_version").fetchone()[0]
            == DATABASE_SCHEMA_VERSION
        )

    assert expected_tables <= tables
    assert migrations["version"] == 1
    with storage.connection() as connection:
        latest = connection.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version DESC"
        ).fetchone()
    assert latest["version"] == DATABASE_SCHEMA_VERSION
    assert latest["name"] == "source_observation_provenance"


def test_source_observation_is_first_class_and_never_replaces_availability(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-100000-source-observed"
    _begin(storage, run_id, schema_version="1.1")
    raw_id = _raw(storage, run_id)
    article = _article(run_id, raw_id)
    article["schema_version"] = "1.1"
    article["source_observed_at"] = "2025-12-20T10:30:00Z"

    storage.save_article(article)

    with storage.connection() as connection:
        canonical = connection.execute(
            """
            SELECT source_observed_at, available_at
            FROM normalized_articles WHERE article_id = ?
            """,
            (article["article_id"],),
        ).fetchone()
        occurrence = connection.execute(
            """
            SELECT source_observed_at, available_at
            FROM article_ingestions WHERE ingestion_run_id = ?
            """,
            (run_id,),
        ).fetchone()

    assert canonical["source_observed_at"] == "2025-12-20T10:30:00Z"
    assert occurrence["source_observed_at"] == "2025-12-20T10:30:00Z"
    assert canonical["available_at"] == NOW
    assert occurrence["available_at"] == NOW


def test_schema_1_0_storage_rejects_source_observation_provenance(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-100000-invalid-source-observed-version"
    _begin(storage, run_id)
    raw_id = _raw(storage, run_id)
    article = _article(run_id, raw_id)
    article["source_observed_at"] = "2025-12-20T10:30:00Z"

    with pytest.raises(ValueError, match="schema 1.1"):
        storage.save_article(article)

    with storage.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM normalized_articles"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT count(*) FROM article_ingestions"
        ).fetchone()[0] == 0


def test_schema_1_0_storage_omits_explicit_null_source_observation_payload(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-100000-legacy-source-observed-null"
    _begin(storage, run_id, schema_version="1.0")
    raw_id = _raw(storage, run_id)
    article = _article(run_id, raw_id)
    article["source_observed_at"] = None

    storage.save_article(article)

    with storage.connection() as connection:
        canonical = connection.execute(
            "SELECT source_observed_at FROM normalized_articles"
        ).fetchone()
        occurrence = connection.execute(
            """
            SELECT source_observed_at, normalized_payload_json
            FROM article_ingestions WHERE ingestion_run_id = ?
            """,
            (run_id,),
        ).fetchone()

    assert canonical["source_observed_at"] is None
    assert occurrence["source_observed_at"] is None
    serialized = json.loads(occurrence["normalized_payload_json"])
    assert "source_observed_at" not in serialized
    assert serialized["schema_version"] == "1.0"


def test_article_schema_must_match_its_ingestion_run(storage: NewsStorage):
    run_id = "NEWS-20261022-100000-record-schema-mismatch"
    _begin(storage, run_id, schema_version="1.1")
    raw_id = _raw(storage, run_id)

    with pytest.raises(ValueError, match="match its ingestion run"):
        storage.save_article(_article(run_id, raw_id))


@pytest.mark.parametrize("prior_version", [1, 2, 3, 4, 5, 6])
def test_initialize_migrates_each_verified_prior_schema_to_v7(
    tmp_path, prior_version
):
    database = tmp_path / f"prior-v{prior_version}.sqlite3"
    _create_versioned_database(database, prior_version)

    storage = NewsStorage(database).initialize()

    with storage.connection() as connection:
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(ingestion_runs)")
        }
        versions = [
            (row["version"], row["name"])
            for row in connection.execute(
                "SELECT version, name FROM schema_migrations ORDER BY version"
            )
        ]
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
    assert "sealed_manifest_sha256" in columns
    assert versions[-1] == (7, "source_observation_provenance")


def test_v6_migration_preserves_legacy_rows_with_unknown_source_observation(
    tmp_path,
):
    database = tmp_path / "legacy-v6-source-observed.sqlite3"
    _create_versioned_database(database, 6)
    legacy = NewsStorage(database)
    run_id = "NEWS-20261022-100000-legacy-v6"
    _begin(legacy, run_id)
    raw_id = _raw(legacy, run_id)
    legacy.save_article(_article(run_id, raw_id))

    migrated = NewsStorage(database).initialize()

    with migrated.connection() as connection:
        canonical = connection.execute(
            "SELECT source_observed_at FROM normalized_articles"
        ).fetchone()
        occurrence = connection.execute(
            "SELECT source_observed_at FROM article_ingestions"
        ).fetchone()
    assert canonical["source_observed_at"] is None
    assert occurrence["source_observed_at"] is None


def test_initialize_rejects_malformed_stamped_schema_without_advancing(tmp_path):
    database = tmp_path / "malformed-v1.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE ingestion_runs(run_id TEXT PRIMARY KEY)")
        connection.execute("PRAGMA user_version = 1")

    with pytest.raises(RuntimeError, match="schema verification failed"):
        NewsStorage(database).initialize()

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ingestion_runs)")
        }
    assert columns == {"run_id"}


def test_round_trip_run_evidence_and_duplicate_candidate_lookup(storage: NewsStorage):
    run_id = "NEWS-20261022-100000-roundtrip"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    assert storage.save_article(_article(run_id, raw_evidence_id)) == "article-1"

    occurrence = storage.list_article_ingestions(run_id)[0]
    link_id = storage.save_link(
        occurrence_id=occurrence["occurrence_id"],
        canonical_article_id="article-1",
        ticker="ogdc",
        company_name="Oil and Gas Development Company Limited",
        matched_alias="OGDC",
        match_method="exact",
        match_location="title",
        confidence="exact",
    )
    assert link_id.startswith("link-")

    assert storage.find_duplicate_candidates(
        canonical_url="https://news.example.test/ogdc-discovery"
    ) == []

    binding = _save_quality_and_seal(storage, run_id)
    storage.finalize_run(
        run_id,
        status="PASS",
        completed_at="2026-10-22T10:01:00Z",
        summary={"normalized_record_count": 1},
        expected_manifest_sha256=binding["manifest_sha256"],
        expected_inventory_sha256=binding["artifact_inventory_sha256"],
        expected_completion_sha256=binding["completion_marker_sha256"],
    )

    candidates = storage.find_duplicate_candidates(
        canonical_url="https://news.example.test/ogdc-discovery"
    )
    assert [row["article_id"] for row in candidates] == ["article-1"]
    assert candidates[0]["first_ingestion_run_id"] == run_id
    assert candidates[0]["owner_manifest_sha256"] == "2" * 64
    assert storage.find_duplicate_candidates(content_sha256="a" * 64)
    assert storage.find_duplicate_candidates(
        title="  OGDC   ANNOUNCES DISCOVERY ",
        source_name="example financial news",
        published_at="2026-10-22T04:00:59.999999Z",
        time_tolerance_seconds=0,
    )
    assert storage.find_duplicate_candidates(
        title="OGDC announces discovery",
        source_name="Example Financial News",
        published_at="2026-10-22T04:01:00Z",
        time_tolerance_seconds=86_400,
    ) == []
    with pytest.raises(ValueError, match="non-negative"):
        storage.find_duplicate_candidates(
            title="OGDC announces discovery",
            source_name="Example Financial News",
            published_at="2026-10-22T04:00:00Z",
            time_tolerance_seconds=-1,
        )

    persisted_run = storage.get_run(run_id)
    assert persisted_run is not None
    assert persisted_run["status"] == "PASS"
    assert json.loads(persisted_run["summary_json"])["normalized_record_count"] == 1
    article = storage.get_article("article-1")
    assert article is not None
    assert article["published_at"] == "2026-10-22T04:00:00Z"
    assert article["normalized_title"] == "ogdc announces discovery"
    canonical = storage.list_canonical_articles()[0]
    assert canonical["content_sha256"] == "a" * 64
    assert canonical["first_ingestion_run_id"] == run_id
    assert canonical["owner_manifest_sha256"] == "2" * 64
    report = storage.get_quality_report(run_id)
    assert report is not None
    assert json.loads(report["report_json"])["total_raw_records"] == 1


def test_repeat_ingestion_preserves_raw_and_occurrence_evidence(storage: NewsStorage):
    first_run = "NEWS-20261022-100000-first"
    second_run = "NEWS-20261022-110000-second"
    _begin(storage, first_run)
    first_raw = _raw(storage, first_run, raw_record_id="shared/source-id")
    storage.save_article(_article(first_run, first_raw))
    _finalize_valid_run(storage, first_run)

    _begin(storage, second_run)
    second_raw = _raw(storage, second_run, raw_record_id="shared/source-id")
    second_occurrence = storage.save_article_ingestion(
        ingestion_run_id=second_run,
        raw_evidence_id=second_raw,
        candidate_article_id="article-1",
        canonical_article_id="article-1",
        available_at="2026-10-22T11:00:00Z",
        fetched_at="2026-10-22T11:00:00Z",
        raw_artifact_reference="raw/shared-source-id.json",
        normalized_payload={
            "schema_version": "1.0",
            "article_id": "article-1",
            "source_observed_at": None,
        },
    )

    assert first_raw != second_raw
    with storage.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM raw_source_records"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM normalized_articles"
        ).fetchone()[0] == 1
        assert connection.execute(
            "SELECT count(*) FROM article_ingestions"
        ).fetchone()[0] == 2
        row = connection.execute(
            """
            SELECT duplicate_of, source_observed_at, normalized_payload_json
            FROM article_ingestions WHERE occurrence_id = ?
            """,
            (second_occurrence,),
        ).fetchone()
    assert row["duplicate_of"] is None
    assert row["source_observed_at"] is None
    assert "source_observed_at" not in json.loads(row["normalized_payload_json"])


def test_cross_run_evidence_associations_are_rejected_by_api_and_triggers(
    storage: NewsStorage,
):
    first_run = "NEWS-20261022-100000-owner-one"
    second_run = "NEWS-20261022-110000-owner-two"
    _begin(storage, first_run)
    first_raw = _raw(storage, first_run, raw_record_id="owner-one")
    storage.save_article(_article(first_run, first_raw))
    _finalize_valid_run(storage, first_run)

    _begin(storage, second_run)
    second_raw = _raw(storage, second_run, raw_record_id="owner-two")
    second_article = _article(
        second_run,
        second_raw,
        article_id="article-2",
        canonical_url="https://news.example.test/second-article",
        title="Second OGDC article",
    )
    second_article["content_sha256"] = "c" * 64
    storage.save_article(second_article)
    second_occurrence = next(
        row
        for row in storage.list_article_ingestions(second_run)
        if row["candidate_article_id"] == "article-2"
    )["occurrence_id"]

    cross_run_article = _article(
        second_run,
        first_raw,
        article_id="cross-run-article",
        canonical_url="https://news.example.test/cross-run",
    )
    with pytest.raises(ValueError, match="same ingestion run"):
        storage.save_article(cross_run_article)
    assert storage.get_article("cross-run-article") is None

    with pytest.raises(ValueError, match="same ingestion run"):
        storage.save_article_ingestion(
            ingestion_run_id=second_run,
            raw_evidence_id=first_raw,
            candidate_article_id="cross-run-occurrence",
            canonical_article_id="article-1",
            duplicate_of="article-1",
            available_at=NOW,
            fetched_at=NOW,
            raw_artifact_reference="raw/cross-run.json",
            normalized_payload={},
        )

    with pytest.raises(ValueError, match="occurrence ingestion run"):
        storage.save_dedup(
            ingestion_run_id=first_run,
            occurrence_id=second_occurrence,
            duplicate_candidate_article_id="article-2",
            canonical_article_id="article-2",
            deduplication_rule="canonical_url",
        )

    with pytest.raises(ValueError, match="does not match"):
        storage.save_link(
            occurrence_id=second_occurrence,
            canonical_article_id="article-1",
            ticker="OGDC",
            company_name="Oil and Gas Development Company Limited",
            matched_alias="OGDC",
            match_method="exact",
            match_location="title",
            confidence="exact",
        )

    with pytest.raises(sqlite3.IntegrityError, match="same ingestion run"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO article_ingestions(
                    occurrence_id, ingestion_run_id, raw_evidence_id,
                    candidate_article_id, canonical_article_id, duplicate_of,
                    deduplication_rule, available_at, fetched_at,
                    raw_artifact_reference, normalized_payload_json,
                    normalization_warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '[]', ?)
                """,
                (
                    "occ-cross-run-sql",
                    second_run,
                    first_raw,
                    "candidate-cross-run-sql",
                    "article-1",
                    "article-1",
                    "content_sha256",
                    NOW,
                    NOW,
                    "raw/cross-run-sql.json",
                    NOW,
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="sealed"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO deduplication_relationships(
                    relationship_id, ingestion_run_id, occurrence_id,
                    duplicate_candidate_article_id, canonical_article_id,
                    duplicate_of, deduplication_rule, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "dedup-cross-run-sql",
                    first_run,
                    second_occurrence,
                    "article-2",
                    "article-2",
                    "article-2",
                    "canonical_url",
                    NOW,
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="ingestion occurrence"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO article_company_links(
                    link_id, occurrence_id, canonical_article_id, ticker,
                    company_name, matched_alias, match_method, match_location,
                    confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "link-cross-run-sql",
                    second_occurrence,
                    "article-1",
                    "OGDC",
                    "Oil and Gas Development Company Limited",
                    "OGDC",
                    "exact",
                    "title",
                    "exact",
                    NOW,
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="same ingestion run"):
        with storage.transaction() as connection:
            connection.execute(
                """
                UPDATE article_ingestions SET raw_evidence_id = ?
                WHERE occurrence_id = ?
                """,
                (first_raw, second_occurrence),
            )


def test_duplicate_occurrence_and_link_retain_noncanonical_evidence(storage: NewsStorage):
    run_id = "NEWS-20261022-100000-duplicates"
    _begin(storage, run_id)
    canonical_raw = _raw(storage, run_id, raw_record_id="canonical")
    storage.save_article(_article(run_id, canonical_raw))
    duplicate_raw = _raw(storage, run_id, raw_record_id="duplicate")

    duplicate = _article(
        run_id,
        duplicate_raw,
        article_id="article-candidate-2",
        canonical_url="https://news.example.test/ogdc-discovery?duplicate=1",
    )
    duplicate.update(
        {
            "duplicate_of": "article-1",
            "deduplication_rule": "content_sha256",
            "raw_artifact_reference": "raw/duplicate.json",
        }
    )
    assert storage.save_article(duplicate) == "article-1"
    duplicate_occurrence = next(
        row
        for row in storage.list_article_ingestions(run_id)
        if row["candidate_article_id"] == "article-candidate-2"
    )
    relationship_id = storage.save_dedup(
        ingestion_run_id=run_id,
        occurrence_id=duplicate_occurrence["occurrence_id"],
        duplicate_candidate_article_id="article-candidate-2",
        canonical_article_id="article-1",
        deduplication_rule="content_sha256",
    )
    link_id = storage.save_link(
        occurrence_id=duplicate_occurrence["occurrence_id"],
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        matched_alias="OGDC",
        match_method="exact",
        match_location="title",
        confidence="exact",
    )

    with storage.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM normalized_articles"
        ).fetchone()[0] == 1
        relation = connection.execute(
            "SELECT * FROM deduplication_relationships WHERE relationship_id = ?",
            (relationship_id,),
        ).fetchone()
        link = connection.execute(
            "SELECT * FROM article_company_links WHERE link_id = ?", (link_id,)
        ).fetchone()
    assert relation["duplicate_of"] == "article-1"
    assert link["occurrence_id"] == duplicate_occurrence["occurrence_id"]
    assert link["canonical_article_id"] == "article-1"


def test_dedup_relationship_must_exactly_match_a_duplicate_occurrence(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-100000-dedup-identity"
    _begin(storage, run_id)
    canonical_raw = _raw(storage, run_id, raw_record_id="canonical-identity")
    storage.save_article(_article(run_id, canonical_raw))
    canonical_occurrence = next(
        row
        for row in storage.list_article_ingestions(run_id)
        if row["candidate_article_id"] == "article-1"
    )["occurrence_id"]

    duplicate_raw = _raw(storage, run_id, raw_record_id="duplicate-identity")
    duplicate = _article(
        run_id,
        duplicate_raw,
        article_id="article-candidate-identity",
        canonical_url="https://news.example.test/duplicate-identity",
    )
    duplicate.update(
        {
            "duplicate_of": "article-1",
            "deduplication_rule": "content_sha256",
            "raw_artifact_reference": "raw/duplicate-identity.json",
        }
    )
    storage.save_article(duplicate)
    duplicate_occurrence = next(
        row
        for row in storage.list_article_ingestions(run_id)
        if row["candidate_article_id"] == "article-candidate-identity"
    )["occurrence_id"]

    unsupported_raw = _raw(storage, run_id, raw_record_id="unsupported-rule")
    unsupported_article = _article(
        run_id,
        unsupported_raw,
        article_id="article-unsupported-rule",
        canonical_url="https://news.example.test/unsupported-rule",
    )
    unsupported_article.update(
        {
            "duplicate_of": "article-1",
            "deduplication_rule": "url_hash",
            "raw_artifact_reference": "raw/unsupported-rule.json",
        }
    )
    with pytest.raises(ValueError, match="deduplication_rule must be one of"):
        storage.save_article(unsupported_article)
    with pytest.raises(ValueError, match="deduplication_rule must be one of"):
        storage.save_article_ingestion(
            ingestion_run_id=run_id,
            raw_evidence_id=unsupported_raw,
            candidate_article_id="article-unsupported-ingestion",
            canonical_article_id="article-1",
            duplicate_of="article-1",
            deduplication_rule="url_hash",
            available_at=NOW,
            fetched_at=NOW,
            raw_artifact_reference="raw/unsupported-ingestion.json",
            normalized_payload={},
        )
    with pytest.raises(ValueError, match="deduplication_rule must be one of"):
        storage.save_dedup(
            ingestion_run_id=run_id,
            occurrence_id=duplicate_occurrence,
            duplicate_candidate_article_id="article-candidate-identity",
            canonical_article_id="article-1",
            deduplication_rule="url_hash",
        )

    with pytest.raises(ValueError, match="marked as a duplicate"):
        storage.save_dedup(
            ingestion_run_id=run_id,
            occurrence_id=canonical_occurrence,
            duplicate_candidate_article_id="article-1",
            canonical_article_id="article-1",
            deduplication_rule="content_sha256",
        )
    with pytest.raises(ValueError, match="duplicate_candidate_article_id"):
        storage.save_dedup(
            ingestion_run_id=run_id,
            occurrence_id=duplicate_occurrence,
            duplicate_candidate_article_id="wrong-candidate",
            canonical_article_id="article-1",
            deduplication_rule="content_sha256",
        )
    with pytest.raises(ValueError, match="canonical_article_id/duplicate_of"):
        storage.save_dedup(
            ingestion_run_id=run_id,
            occurrence_id=duplicate_occurrence,
            duplicate_candidate_article_id="article-candidate-identity",
            canonical_article_id="wrong-canonical",
            deduplication_rule="content_sha256",
        )
    with pytest.raises(ValueError, match="deduplication_rule does not match"):
        storage.save_dedup(
            ingestion_run_id=run_id,
            occurrence_id=duplicate_occurrence,
            duplicate_candidate_article_id="article-candidate-identity",
            canonical_article_id="article-1",
            deduplication_rule="canonical_url",
        )

    with pytest.raises(sqlite3.IntegrityError, match="unsupported"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO article_ingestions(
                    occurrence_id, ingestion_run_id, raw_evidence_id,
                    candidate_article_id, canonical_article_id, duplicate_of,
                    deduplication_rule, available_at, fetched_at,
                    raw_artifact_reference, normalized_payload_json,
                    normalization_warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', '[]', ?)
                """,
                (
                    "occ-unsupported-rule",
                    run_id,
                    unsupported_raw,
                    "article-unsupported-sql",
                    "article-1",
                    "article-1",
                    "url_hash",
                    NOW,
                    NOW,
                    "raw/unsupported-rule-sql.json",
                    NOW,
                ),
            )

    with pytest.raises(sqlite3.IntegrityError, match="exactly match"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO deduplication_relationships(
                    relationship_id, ingestion_run_id, occurrence_id,
                    duplicate_candidate_article_id, canonical_article_id,
                    duplicate_of, deduplication_rule, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "dedup-canonical-occurrence",
                    run_id,
                    canonical_occurrence,
                    "article-1",
                    "article-1",
                    "article-1",
                    "content_sha256",
                    NOW,
                ),
            )
    with pytest.raises(sqlite3.IntegrityError, match="exactly match"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO deduplication_relationships(
                    relationship_id, ingestion_run_id, occurrence_id,
                    duplicate_candidate_article_id, canonical_article_id,
                    duplicate_of, deduplication_rule, detected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "dedup-wrong-candidate",
                    run_id,
                    duplicate_occurrence,
                    "wrong-candidate",
                    "article-1",
                    "article-1",
                    "content_sha256",
                    NOW,
                ),
            )

    for relationship_id, rule in (
        ("dedup-wrong-supported-rule", "canonical_url"),
        ("dedup-unsupported-rule", "url_hash"),
    ):
        with pytest.raises(
            sqlite3.IntegrityError,
            match="supported rule and exactly match",
        ):
            with storage.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO deduplication_relationships(
                        relationship_id, ingestion_run_id, occurrence_id,
                        duplicate_candidate_article_id, canonical_article_id,
                        duplicate_of, deduplication_rule, detected_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        relationship_id,
                        run_id,
                        duplicate_occurrence,
                        "article-candidate-identity",
                        "article-1",
                        "article-1",
                        rule,
                        NOW,
                    ),
                )

    relationship_id = storage.save_dedup(
        ingestion_run_id=run_id,
        occurrence_id=duplicate_occurrence,
        duplicate_candidate_article_id="article-candidate-identity",
        canonical_article_id="article-1",
        deduplication_rule="content_sha256",
    )
    with pytest.raises(sqlite3.IntegrityError, match="exactly match"):
        with storage.transaction() as connection:
            connection.execute(
                """
                UPDATE deduplication_relationships
                SET duplicate_candidate_article_id = 'tampered-candidate'
                WHERE relationship_id = ?
                """,
                (relationship_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="exactly match"):
        with storage.transaction() as connection:
            connection.execute(
                """
                UPDATE deduplication_relationships SET occurrence_id = ?
                WHERE relationship_id = ?
                """,
                (canonical_occurrence, relationship_id),
            )
    for rule in ("canonical_url", "url_hash"):
        with pytest.raises(
            sqlite3.IntegrityError,
            match="supported rule and exactly match",
        ):
            with storage.transaction() as connection:
                connection.execute(
                    """
                    UPDATE deduplication_relationships
                    SET deduplication_rule = ?
                    WHERE relationship_id = ?
                    """,
                    (rule, relationship_id),
                )
    with pytest.raises(sqlite3.IntegrityError, match="remain consistent"):
        with storage.transaction() as connection:
            connection.execute(
                """
                UPDATE article_ingestions
                SET deduplication_rule = 'canonical_url'
                WHERE occurrence_id = ?
                """,
                (duplicate_occurrence,),
            )


def test_constraints_and_transaction_rollback(storage: NewsStorage):
    run_id = "NEWS-20261022-100000-constraints"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))

    with pytest.raises(sqlite3.IntegrityError):
        storage.save_article(
            _article(
                run_id,
                raw_evidence_id,
                article_id="article-2",
                canonical_url="https://news.example.test/ogdc-discovery",
            )
        )
    assert storage.get_article("article-2") is None

    invalid = _article(
        run_id,
        "raw-evidence-does-not-exist",
        article_id="article-foreign-key",
        canonical_url="https://news.example.test/missing-raw",
    )
    with pytest.raises(KeyError, match="No raw evidence"):
        storage.save_article(invalid)
    assert storage.get_article("article-foreign-key") is None

    with pytest.raises(RuntimeError, match="sealed"):
        storage.finalize_run(run_id, status="PASS")
    assert storage.get_run(run_id)["status"] == "RUNNING"

    with pytest.raises(RuntimeError, match="force rollback"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO provider_errors(
                    error_id, ingestion_run_id, provider_id, occurred_at,
                    error_type, message, retryable, details_json
                ) VALUES ('rollback-error', ?, 'manual', ?,
                          'test', 'must roll back', 0, '{}')
                """,
                (run_id, NOW),
            )
            raise RuntimeError("force rollback")
    with storage.connection() as connection:
        assert connection.execute(
            "SELECT count(*) FROM provider_errors WHERE error_id = 'rollback-error'"
        ).fetchone()[0] == 0

    with pytest.raises(KeyboardInterrupt):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE raw_source_records SET payload_json = '{}' "
                "WHERE raw_evidence_id = ?",
                (raw_evidence_id,),
            )
            raise KeyboardInterrupt
    with storage.connection() as connection:
        payload = connection.execute(
            "SELECT payload_json FROM raw_source_records WHERE raw_evidence_id = ?",
            (raw_evidence_id,),
        ).fetchone()[0]
    assert payload != "{}"


def test_missing_title_is_stored_for_quality_reporting(storage: NewsStorage):
    run_id = "NEWS-20261022-100000-missing-title"
    _begin(storage, run_id, data_kind="imported")
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(
        _article(
            run_id,
            raw_evidence_id,
            article_id="article-without-title",
            canonical_url="https://news.example.test/no-title",
            title=None,
        )
    )
    article = storage.get_article("article-without-title")
    assert article["title"] is None
    assert article["normalized_title"] == ""


def test_provider_error_and_partial_run_round_trip(storage: NewsStorage):
    run_id = "NEWS-20261022-100000-provider-error"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    error_id = storage.save_error(
        ingestion_run_id=run_id,
        provider_id="permitted-rss",
        configured_source="https://news.example.test/feed.xml",
        error_type="HTTPError",
        message="upstream returned an error",
        http_status=503,
        retryable=True,
        details={"credential_present": False},
        occurred_at=NOW,
    )
    binding = _save_quality_and_seal(storage, run_id, quality_status="WARN")
    storage.finalize_run(
        run_id,
        status="PARTIAL",
        error_message="One configured source failed",
        expected_manifest_sha256=binding["manifest_sha256"],
        expected_inventory_sha256=binding["artifact_inventory_sha256"],
        expected_completion_sha256=binding["completion_marker_sha256"],
    )
    with storage.connection() as connection:
        error = connection.execute(
            "SELECT * FROM provider_errors WHERE error_id = ?", (error_id,)
        ).fetchone()
    assert error["retryable"] == 1
    assert storage.get_run(run_id)["status"] == "PARTIAL"


def test_finalized_run_rejects_all_evidence_writes_updates_and_deletes(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-120000-finalized"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    occurrence_id = storage.list_article_ingestions(run_id)[0]["occurrence_id"]
    storage.save_link(
        occurrence_id=occurrence_id,
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        matched_alias="OGDC",
        match_method="exact",
        match_location="title",
        confidence="exact",
    )
    duplicate_raw = _raw(storage, run_id, raw_record_id="finalized-duplicate")
    duplicate = _article(
        run_id,
        duplicate_raw,
        article_id="finalized-duplicate-candidate",
        canonical_url="https://news.example.test/finalized-duplicate",
    )
    duplicate.update(
        {
            "duplicate_of": "article-1",
            "deduplication_rule": "canonical_url",
            "raw_artifact_reference": "raw/finalized-duplicate.json",
        }
    )
    storage.save_article(duplicate)
    duplicate_occurrence = next(
        row
        for row in storage.list_article_ingestions(run_id)
        if row["candidate_article_id"] == "finalized-duplicate-candidate"
    )["occurrence_id"]
    storage.save_dedup(
        ingestion_run_id=run_id,
        occurrence_id=duplicate_occurrence,
        duplicate_candidate_article_id="finalized-duplicate-candidate",
        canonical_article_id="article-1",
        deduplication_rule="canonical_url",
    )
    _finalize_valid_run(storage, run_id)

    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        storage.save_raw(
            ingestion_run_id=run_id,
            raw_record_id="late-raw",
            provider_id="permitted-rss",
            fetched_at=NOW,
            data_kind="real",
            raw_artifact_reference="raw/late.json",
            payload={},
        )
    late_article = _article(
        run_id,
        raw_evidence_id,
        article_id="late-article",
        canonical_url="https://news.example.test/late",
    )
    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        storage.save_article(late_article)
    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        storage.save_article_ingestion(
            ingestion_run_id=run_id,
            raw_evidence_id=raw_evidence_id,
            candidate_article_id="late-candidate",
            canonical_article_id="article-1",
            duplicate_of="article-1",
            deduplication_rule="content_sha256",
            available_at=NOW,
            fetched_at=NOW,
            raw_artifact_reference="raw/item-1.json",
            normalized_payload={"title": "late"},
        )
    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        storage.save_link(
            occurrence_id=occurrence_id,
            ticker="OGDC",
            company_name="Oil and Gas Development Company Limited",
            matched_alias="OGDCL",
            match_method="exact",
            match_location="summary",
            confidence="exact",
        )
    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        storage.save_dedup(
            ingestion_run_id=run_id,
            occurrence_id=duplicate_occurrence,
            duplicate_candidate_article_id="finalized-duplicate-candidate",
            canonical_article_id="article-1",
            deduplication_rule="canonical_url",
        )
    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        storage.save_error(
            ingestion_run_id=run_id,
            provider_id="permitted-rss",
            error_type="late_error",
            message="must not be appended",
        )
    with pytest.raises(RuntimeError, match="finalized"):
        storage.save_quality(
            ingestion_run_id=run_id,
            quality_status="PASS",
            report={},
        )

    with storage.connection() as connection:
        triggers = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger'"
            )
        }
    for table in (
        "raw_source_records",
        "normalized_articles",
        "article_ingestions",
        "article_company_links",
        "deduplication_relationships",
        "provider_errors",
        "quality_reports",
    ):
        assert f"trg_{table}_insert_running" in triggers
        assert f"trg_{table}_update_running" in triggers
        assert f"trg_{table}_delete_running" in triggers

    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE raw_source_records SET payload_json = '{}' "
                "WHERE raw_evidence_id = ?",
                (raw_evidence_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="sealed|finalized"):
        with storage.transaction() as connection:
            connection.execute(
                "DELETE FROM article_company_links WHERE occurrence_id = ?",
                (occurrence_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_runs SET summary_json = '{}' WHERE run_id = ?",
                (run_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        with storage.transaction() as connection:
            connection.execute(
                "DELETE FROM ingestion_runs WHERE run_id = ?", (run_id,)
            )


def test_final_status_must_agree_with_quality_and_provider_errors(
    storage: NewsStorage,
):
    failed_quality = "NEWS-20261022-120000-failed-quality"
    _begin(storage, failed_quality)
    _save_quality_and_seal(storage, failed_quality, quality_status="FAIL")
    with pytest.raises(RuntimeError, match="PASS quality"):
        storage.finalize_run(failed_quality, status="PASS")
    assert storage.get_run(failed_quality)["status"] == "RUNNING"
    with pytest.raises(sqlite3.IntegrityError, match="PASS requires"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_runs SET status = 'PASS' WHERE run_id = ?",
                (failed_quality,),
            )
    storage.finalize_run(failed_quality, status="FAIL")

    provider_error_run = "NEWS-20261022-120000-provider-error-pass"
    _begin(storage, provider_error_run)
    raw_evidence_id = _raw(storage, provider_error_run)
    storage.save_article(_article(provider_error_run, raw_evidence_id))
    storage.save_error(
        ingestion_run_id=provider_error_run,
        provider_id="permitted-rss",
        error_type="network_error",
        message="one source failed",
    )
    with pytest.raises(ValueError, match="more optimistic"):
        storage.save_quality(
            ingestion_run_id=provider_error_run,
            quality_status="PASS",
            report=_quality_payload(
                storage, provider_error_run, quality_status="PASS"
            ),
        )
    storage.finalize_run(provider_error_run, status="FAIL")

    partial_run = "NEWS-20261022-120000-partial-quality"
    _begin(storage, partial_run)
    raw_evidence_id = _raw(storage, partial_run)
    article = _article(
        partial_run,
        raw_evidence_id,
        article_id="partial-quality-article",
        canonical_url="https://news.example.test/partial-quality",
        title="Partial quality article",
    )
    article["content_sha256"] = "c" * 64
    storage.save_article(article)
    _save_quality_and_seal(storage, partial_run, quality_status="PASS")
    with pytest.raises(RuntimeError, match="WARN quality"):
        storage.finalize_run(partial_run, status="PARTIAL")
    storage.finalize_run(partial_run, status="FAIL")

    invalid_quality = "NEWS-20261022-120000-invalid-quality"
    _begin(storage, invalid_quality)
    with pytest.raises(ValueError, match="quality_status"):
        storage.save_quality(
            ingestion_run_id=invalid_quality,
            quality_status="MAYBE",
            report={},
        )


def test_rejects_unsafe_references_impossible_timelines_and_missing_raw(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-120000-validation"
    _begin(storage, run_id)
    with pytest.raises(ValueError, match="relative POSIX"):
        storage.save_raw(
            ingestion_run_id=run_id,
            raw_record_id="unsafe",
            provider_id="permitted-rss",
            fetched_at=NOW,
            data_kind="real",
            raw_artifact_reference="../outside.json",
            payload={},
        )
    raw_evidence_id = _raw(storage, run_id)

    unsafe_article = _article(
        run_id,
        raw_evidence_id,
        article_id="unsafe-article",
        canonical_url="https://news.example.test/unsafe",
    )
    unsafe_article["raw_artifact_reference"] = "/absolute/raw.json"
    with pytest.raises(ValueError, match="relative POSIX"):
        storage.save_article(unsafe_article)

    before_fetch = _article(
        run_id,
        raw_evidence_id,
        article_id="before-fetch",
        canonical_url="https://news.example.test/before-fetch",
    )
    before_fetch["available_at"] = "2026-10-22T09:59:59.999999Z"
    with pytest.raises(ValueError, match="precede fetched_at"):
        storage.save_article(before_fetch)

    before_publication = _article(
        run_id,
        raw_evidence_id,
        article_id="before-publication",
        canonical_url="https://news.example.test/before-publication",
    )
    before_publication["published_at"] = "2026-10-22T10:00:00.000001Z"
    with pytest.raises(ValueError, match="precede published_at"):
        storage.save_article(before_publication)

    missing_raw = _article(
        run_id,
        raw_evidence_id,
        article_id="missing-raw",
        canonical_url="https://news.example.test/missing-raw",
    )
    missing_raw.pop("raw_evidence_id")
    with pytest.raises(ValueError, match="requires raw evidence"):
        storage.save_article(missing_raw)
    with pytest.raises(ValueError, match="requires raw evidence"):
        storage.save_article_ingestion(
            ingestion_run_id=run_id,
            candidate_article_id="missing-raw-occurrence",
            canonical_article_id="does-not-matter",
            available_at=NOW,
            fetched_at=NOW,
            raw_artifact_reference="raw/item.json",
            normalized_payload={},
        )


def test_subsecond_timestamps_are_preserved_exactly(storage: NewsStorage):
    run_id = "NEWS-20261022-120000-subseconds"
    started = "2026-10-22T10:00:00.123456Z"
    storage.begin_run(
        run_id=run_id,
        schema_version="1.0",
        provider_id="permitted-rss",
        provider_mode="rss",
        data_kind="real",
        started_at=started,
    )
    raw_evidence_id = storage.save_raw(
        ingestion_run_id=run_id,
        raw_record_id="subsecond-raw",
        provider_id="permitted-rss",
        fetched_at="2026-10-22T10:00:00.654320Z",
        data_kind="real",
        raw_artifact_reference="raw/subsecond.json",
        payload={},
    )
    article = _article(
        run_id,
        raw_evidence_id,
        article_id="subsecond-article",
        canonical_url="https://news.example.test/subseconds",
    )
    article.update(
        {
            "published_at": "2026-10-22T09:00:00.111111Z",
            "fetched_at": "2026-10-22T10:00:00.654320Z",
            "available_at": "2026-10-22T10:00:00.654321Z",
        }
    )
    storage.save_article(article)

    with storage.connection() as connection:
        run = connection.execute(
            "SELECT started_at FROM ingestion_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        raw = connection.execute(
            "SELECT fetched_at FROM raw_source_records WHERE raw_evidence_id = ?",
            (raw_evidence_id,),
        ).fetchone()
        saved = connection.execute(
            """
            SELECT published_at, available_at, fetched_at
            FROM normalized_articles WHERE article_id = 'subsecond-article'
            """
        ).fetchone()
    assert run["started_at"] == started
    assert raw["fetched_at"] == "2026-10-22T10:00:00.654320Z"
    assert saved["published_at"] == "2026-10-22T09:00:00.111111Z"
    assert saved["available_at"] == "2026-10-22T10:00:00.654321Z"
    assert saved["fetched_at"] == "2026-10-22T10:00:00.654320Z"


def test_direct_sql_cannot_insert_completed_or_bless_empty_run(
    storage: NewsStorage,
):
    with pytest.raises(sqlite3.IntegrityError, match="inserted as RUNNING"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs(
                    run_id, schema_version, provider_id, provider_mode,
                    data_kind, started_at, status, summary_json
                ) VALUES ('direct-pass', '1.0', 'manual', 'manual-csv',
                          'fixture', ?, 'PASS', '{}')
                """,
                (NOW,),
            )

    run_id = "NEWS-20261022-130000-empty-pass"
    _begin(storage, run_id)
    with pytest.raises(ValueError, match="more optimistic"):
        storage.save_quality(
            ingestion_run_id=run_id,
            quality_status="PASS",
            report=_quality_payload(storage, run_id, quality_status="PASS"),
        )

    with storage.transaction() as connection:
        connection.execute(
            """
            INSERT INTO quality_reports(
                ingestion_run_id, schema_version, quality_status,
                created_at, report_json
            ) VALUES (?, '1.0', 'PASS', ?, ?)
            """,
            (
                run_id,
                NOW,
                json.dumps(
                    {
                        "quality_status": "PASS",
                        "total_raw_records": 0,
                        "total_normalized_records": 0,
                        "duplicate_count": 0,
                        "provider_error_count": 0,
                    }
                ),
            ),
        )
        connection.execute(
            """
            UPDATE ingestion_runs
            SET prepared_at = ?, sealed_artifact_inventory_json = '[]',
                sealed_artifact_inventory_sha256 = ?,
                sealed_manifest_sha256 = ?, sealed_completion_sha256 = ?
            WHERE run_id = ?
            """,
            (NOW, "1" * 64, "2" * 64, "3" * 64, run_id),
        )
    with pytest.raises(sqlite3.IntegrityError, match="nonempty"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_runs SET status = 'PASS' WHERE run_id = ?",
                (run_id,),
            )
    assert storage.get_run(run_id)["status"] == "RUNNING"


def test_public_quality_report_requires_every_database_verifiable_v2_count(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-130000-quality-v2-required"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    report = _quality_payload(storage, run_id)
    report.pop("within_run_duplicate_count")

    with pytest.raises(ValueError, match="within_run_duplicate_count"):
        storage.save_quality(
            ingestion_run_id=run_id,
            quality_status="PASS",
            report=report,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("within_run_duplicate_count", None),
        ("new_canonical_article_count", 99),
        ("missing_body_count", 99),
        ("deduplication_rule_counts", {"canonical_url": 99}),
    ),
)
def test_direct_sql_cannot_finalize_with_incomplete_or_false_v2_quality(
    storage: NewsStorage,
    field: str,
    replacement: object,
):
    run_id = f"NEWS-20261022-130000-direct-quality-{field.replace('_', '-')}"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    report = _quality_payload(storage, run_id)
    if replacement is None:
        report.pop(field)
    else:
        report[field] = replacement

    with storage.transaction() as connection:
        connection.execute(
            """
            INSERT INTO quality_reports(
                ingestion_run_id, schema_version, quality_status,
                created_at, report_json
            ) VALUES (?, '2.0', 'PASS', ?, ?)
            """,
            (run_id, NOW, json.dumps(report, sort_keys=True)),
        )
        connection.execute(
            """
            UPDATE ingestion_runs
            SET prepared_at = ?, sealed_artifact_inventory_json = '[]',
                sealed_artifact_inventory_sha256 = ?,
                sealed_manifest_sha256 = ?, sealed_completion_sha256 = ?
            WHERE run_id = ?
            """,
            (NOW, "1" * 64, "2" * 64, "3" * 64, run_id),
        )

    with pytest.raises(sqlite3.IntegrityError, match="internally consistent"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_runs SET status = 'PASS' WHERE run_id = ?",
                (run_id,),
            )
    assert storage.get_run(run_id)["status"] == "RUNNING"


@pytest.mark.parametrize(
    ("column_schema_version", "report_schema_version"),
    (("1.0", "2.0"), ("2.0", "1.0")),
)
def test_direct_sql_completion_requires_current_quality_schema_identity(
    storage: NewsStorage,
    column_schema_version: str,
    report_schema_version: str,
):
    run_id = "NEWS-20261022-130000-direct-quality-schema"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    report = _quality_payload(storage, run_id)
    report["schema_version"] = report_schema_version

    with storage.transaction() as connection:
        connection.execute(
            """
            INSERT INTO quality_reports(
                ingestion_run_id, schema_version, quality_status,
                created_at, report_json
            ) VALUES (?, ?, 'PASS', ?, ?)
            """,
            (
                run_id,
                column_schema_version,
                NOW,
                json.dumps(report, sort_keys=True),
            ),
        )
        connection.execute(
            """
            UPDATE ingestion_runs
            SET prepared_at = ?, sealed_artifact_inventory_json = '[]',
                sealed_artifact_inventory_sha256 = ?,
                sealed_manifest_sha256 = ?, sealed_completion_sha256 = ?
            WHERE run_id = ?
            """,
            (NOW, "1" * 64, "2" * 64, "3" * 64, run_id),
        )

    with pytest.raises(sqlite3.IntegrityError, match="internally consistent"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_runs SET status = 'PASS' WHERE run_id = ?",
                (run_id,),
            )


def test_direct_sql_partial_rejects_more_occurrences_than_raw_records(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-130000-negative-rejected-count"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    duplicate = _article(
        run_id,
        raw_evidence_id,
        article_id="same-raw-second-occurrence",
        canonical_url="https://news.example.test/same-raw-second-occurrence",
    )
    duplicate.update(
        {
            "duplicate_of": "article-1",
            "deduplication_rule": "canonical_url",
            "raw_artifact_reference": "raw/same-record-again.json",
        }
    )
    storage.save_article(duplicate)
    duplicate_occurrence = next(
        row
        for row in storage.list_article_ingestions(run_id)
        if row["candidate_article_id"] == "same-raw-second-occurrence"
    )
    storage.save_dedup(
        ingestion_run_id=run_id,
        occurrence_id=duplicate_occurrence["occurrence_id"],
        duplicate_candidate_article_id="same-raw-second-occurrence",
        canonical_article_id="article-1",
        deduplication_rule="canonical_url",
    )
    report = {
        "schema_version": "2.0",
        "quality_status": "WARN",
        "total_raw_records": 1,
        "total_normalized_records": 2,
        "normalized_occurrence_count": 2,
        "new_canonical_article_count": 1,
        "matched_existing_canonical_count": 0,
        "within_run_duplicate_count": 1,
        "historical_duplicate_count": 0,
        "represented_canonical_count": 1,
        "duplicate_relationship_count": 1,
        "duplicate_count": 1,
        "deduplication_rule_counts": {
            "canonical_url": 1,
            "content_sha256": 0,
            "title_source_time": 0,
        },
        "provider_error_count": 0,
        "rejected_record_count": -1,
        "missing_title_count": 0,
        "missing_publication_time_count": 0,
        "missing_body_count": 2,
        "invalid_url_count": 0,
        "ogdc_linked_article_count": 0,
    }
    with storage.transaction() as connection:
        connection.execute(
            """
            INSERT INTO quality_reports(
                ingestion_run_id, schema_version, quality_status,
                created_at, report_json
            ) VALUES (?, '2.0', 'WARN', ?, ?)
            """,
            (run_id, NOW, json.dumps(report, sort_keys=True)),
        )
        connection.execute(
            """
            UPDATE ingestion_runs
            SET prepared_at = ?, sealed_artifact_inventory_json = '[]',
                sealed_artifact_inventory_sha256 = ?,
                sealed_manifest_sha256 = ?, sealed_completion_sha256 = ?
            WHERE run_id = ?
            """,
            (NOW, "1" * 64, "2" * 64, "3" * 64, run_id),
        )

    with pytest.raises(sqlite3.IntegrityError, match="internally consistent"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_runs SET status = 'PARTIAL' WHERE run_id = ?",
                (run_id,),
            )


def test_duplicate_occurrence_requires_exactly_one_relationship_before_seal(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-130000-bijection"
    _begin(storage, run_id)
    canonical_raw = _raw(storage, run_id, raw_record_id="canonical")
    storage.save_article(_article(run_id, canonical_raw))
    duplicate_raw = _raw(storage, run_id, raw_record_id="duplicate")
    duplicate = _article(
        run_id,
        duplicate_raw,
        article_id="duplicate-without-relationship",
        canonical_url="https://news.example.test/duplicate-without-relationship",
    )
    duplicate.update(
        {
            "duplicate_of": "article-1",
            "deduplication_rule": "content_sha256",
            "raw_artifact_reference": "raw/duplicate.json",
        }
    )
    storage.save_article(duplicate)
    report = _quality_payload(storage, run_id)
    storage.save_quality(
        ingestion_run_id=run_id,
        quality_status=report["quality_status"],
        report=report,
    )

    snapshot = storage.get_run_evidence_snapshot(run_id)
    assert snapshot["duplicate_occurrence_count"] == 1
    assert snapshot["duplicate_relationship_count"] == 0
    assert snapshot["relationship_bijection_valid"] is False
    with pytest.raises(RuntimeError, match="incomplete duplicate relationships"):
        _seal(storage, run_id)


def test_normalized_articles_cannot_store_duplicate_chains_or_cycles(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-130000-canonical-only"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))

    with pytest.raises(sqlite3.IntegrityError, match="canonical rows only"):
        with storage.transaction() as connection:
            connection.execute(
                """
                UPDATE normalized_articles SET duplicate_of = article_id
                WHERE article_id = 'article-1'
                """
            )
    assert storage.get_article("article-1")["duplicate_of"] is None


def test_seal_binds_inventory_manifest_and_completion_identity(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-130000-sealed"
    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    report = _quality_payload(storage, run_id)
    assert report["deduplication_rule_counts"] == {
        "canonical_url": 0,
        "content_sha256": 0,
        "title_source_time": 0,
    }
    storage.save_quality(
        ingestion_run_id=run_id,
        quality_status="PASS",
        report=report,
    )
    binding = storage.seal_run(
        run_id,
        artifact_inventory=[
            {"path": "z.json", "sha256": "f" * 64, "size_bytes": 9},
            {"path": "a.csv", "sha256": "e" * 64, "size_bytes": 3},
        ],
        manifest_sha256="d" * 64,
        completion_marker_sha256="c" * 64,
        expected_counts=storage.get_run_evidence_snapshot(run_id),
    )

    assert [item["path"] for item in binding["artifact_inventory"]] == [
        "a.csv",
        "z.json",
    ]
    assert storage.get_run_finalization_binding(run_id) == binding
    persisted = storage.get_run(run_id)
    assert persisted["sealed_manifest_sha256"] == "d" * 64
    with pytest.raises(sqlite3.IntegrityError, match="seal is immutable"):
        with storage.transaction() as connection:
            connection.execute(
                """
                UPDATE ingestion_runs SET sealed_manifest_sha256 = ?
                WHERE run_id = ?
                """,
                ("b" * 64, run_id),
            )
    with pytest.raises(sqlite3.IntegrityError, match="sealed"):
        _raw(storage, run_id, raw_record_id="late")
    with pytest.raises(RuntimeError, match="manifest digest"):
        storage.finalize_run(
            run_id,
            status="PASS",
            expected_manifest_sha256="b" * 64,
        )
    storage.finalize_run(
        run_id,
        status="PASS",
        expected_manifest_sha256=binding["manifest_sha256"],
        expected_inventory_sha256=binding["artifact_inventory_sha256"],
        expected_completion_sha256=binding["completion_marker_sha256"],
    )


def test_only_one_running_ingestion_can_be_created_concurrently(storage: NewsStorage):
    barrier = threading.Barrier(3)
    results: list[tuple[str, str]] = []
    result_lock = threading.Lock()

    def begin(run_id: str) -> None:
        barrier.wait()
        try:
            _begin(storage, run_id)
        except Exception as exc:  # the losing writer must fail deterministically
            result = ("error", str(exc))
        else:
            result = ("ok", run_id)
        with result_lock:
            results.append(result)

    threads = [
        threading.Thread(target=begin, args=(f"NEWS-concurrent-{index}",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sum(kind == "ok" for kind, _ in results) == 1
    errors = [message for kind, message in results if kind == "error"]
    assert len(errors) == 1
    assert "already active" in errors[0]
    active_run = next(value for kind, value in results if kind == "ok")
    storage.discard_running_run(active_run)


def test_discard_running_run_removes_partial_evidence_and_allows_retry(
    storage: NewsStorage,
):
    run_id = "NEWS-20261022-130000-interrupted"
    _begin(storage, run_id)
    canonical_raw = _raw(storage, run_id, raw_record_id="canonical")
    storage.save_article(_article(run_id, canonical_raw))
    canonical_occurrence = storage.list_article_ingestions(run_id)[0]
    storage.save_link(
        occurrence_id=canonical_occurrence["occurrence_id"],
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        matched_alias="OGDC",
        match_method="exact",
        match_location="title",
        confidence="exact",
    )
    duplicate_raw = _raw(storage, run_id, raw_record_id="duplicate")
    duplicate = _article(
        run_id,
        duplicate_raw,
        article_id="interrupted-duplicate",
        canonical_url="https://news.example.test/interrupted-duplicate",
    )
    duplicate.update(
        {
            "duplicate_of": "article-1",
            "deduplication_rule": "canonical_url",
            "raw_artifact_reference": "raw/duplicate.json",
        }
    )
    storage.save_article(duplicate)
    duplicate_occurrence = next(
        row
        for row in storage.list_article_ingestions(run_id)
        if row["candidate_article_id"] == "interrupted-duplicate"
    )
    storage.save_dedup(
        ingestion_run_id=run_id,
        occurrence_id=duplicate_occurrence["occurrence_id"],
        duplicate_candidate_article_id="interrupted-duplicate",
        canonical_article_id="article-1",
        deduplication_rule="canonical_url",
    )
    storage.save_error(
        ingestion_run_id=run_id,
        provider_id="permitted-rss",
        error_type="interrupted",
        message="partial evidence",
    )
    report = _quality_payload(storage, run_id, quality_status="WARN")
    storage.save_quality(
        ingestion_run_id=run_id,
        quality_status="WARN",
        report=report,
    )

    storage.discard_running_run(run_id)

    assert storage.get_run(run_id) is None
    with storage.connection() as connection:
        for table in (
            "raw_source_records",
            "normalized_articles",
            "article_ingestions",
            "article_company_links",
            "deduplication_relationships",
            "provider_errors",
            "quality_reports",
        ):
            assert connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0

    _begin(storage, run_id)
    raw_evidence_id = _raw(storage, run_id)
    storage.save_article(_article(run_id, raw_evidence_id))
    _save_quality_and_seal(storage, run_id)
    with pytest.raises(RuntimeError, match="sealed"):
        storage.discard_running_run(run_id)


def test_schema_migration_rolls_back_on_base_exception(tmp_path, monkeypatch):
    database = tmp_path / "migration-interrupted.sqlite3"
    _create_versioned_database(database, 3)
    original_verify = storage_module._verify_schema

    def interrupt_v4(connection, version):
        if version == 4:
            raise KeyboardInterrupt
        return original_verify(connection, version)

    monkeypatch.setattr(storage_module, "_verify_schema", interrupt_v4)
    with pytest.raises(KeyboardInterrupt):
        NewsStorage(database).initialize()

    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(ingestion_runs)")
        }
        migrations = connection.execute(
            "SELECT count(*) FROM schema_migrations"
        ).fetchone()[0]
    assert "sealed_manifest_sha256" not in columns
    assert migrations == 3


def test_schema_migration_locks_before_validating_data_preconditions(
    tmp_path, monkeypatch
):
    database = tmp_path / "migration-validation-lock.sqlite3"
    _create_versioned_database(database, 3)
    legacy = NewsStorage(database)
    run_id = "NEWS-20261022-100000-migration-race"
    _begin(legacy, run_id)

    first_raw = _raw(legacy, run_id, "item/1")
    legacy.save_article(_article(run_id, first_raw))
    second_raw = _raw(legacy, run_id, "item/2")
    second = _article(
        run_id,
        second_raw,
        article_id="article-2",
        canonical_url="https://news.example.test/ogdc-second",
        title="OGDC announces second discovery",
    )
    second.update(
        {
            "content_sha256": "c" * 64,
            "raw_artifact_reference": "raw/item-2.json",
        }
    )
    legacy.save_article(second)

    validation_reached = threading.Event()
    writer_finished = threading.Event()
    writer_result: list[str] = []

    def concurrent_writer() -> None:
        assert validation_reached.wait(timeout=5)
        connection = sqlite3.connect(
            database,
            timeout=0.1,
            isolation_level=None,
        )
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute(
                """
                UPDATE normalized_articles
                SET duplicate_of = 'article-2'
                WHERE article_id = 'article-1'
                """
            )
        except sqlite3.OperationalError as exc:
            assert "locked" in str(exc).lower()
            writer_result.append("locked")
        else:
            writer_result.append("committed")
        finally:
            connection.close()
            writer_finished.set()

    original_validate = storage_module._validate_v4_migration_preconditions

    def pause_after_validation(connection):
        original_validate(connection)
        validation_reached.set()
        assert writer_finished.wait(timeout=5)

    monkeypatch.setattr(
        storage_module,
        "_validate_v4_migration_preconditions",
        pause_after_validation,
    )
    writer = threading.Thread(target=concurrent_writer)
    writer.start()
    migrated = NewsStorage(database).initialize()
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert writer_result == ["locked"]
    with migrated.connection() as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 7
        duplicate_of = connection.execute(
            """
            SELECT duplicate_of FROM normalized_articles
            WHERE article_id = 'article-1'
            """
        ).fetchone()[0]
    assert duplicate_of is None


def test_initialize_rejects_tampered_current_schema(tmp_path):
    database = tmp_path / "tampered-v4.sqlite3"
    storage = NewsStorage(database).initialize()
    with storage.connection() as connection:
        connection.execute("DROP TRIGGER trg_ingestion_runs_insert_running")

    with pytest.raises(RuntimeError, match="schema verification failed"):
        storage.initialize()
