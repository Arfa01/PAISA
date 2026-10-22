from __future__ import annotations

import json
import sqlite3

import pytest

from paisa.news.storage import DATABASE_SCHEMA_VERSION, NewsStorage


NOW = "2026-10-22T10:00:00Z"


@pytest.fixture
def storage(tmp_path):
    return NewsStorage(tmp_path / "nested" / "paisa_news.sqlite3").initialize()


def _begin(storage: NewsStorage, run_id: str, *, data_kind: str = "real") -> None:
    storage.begin_run(
        run_id=run_id,
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
    assert latest["name"] == "news_storage_cross_run_integrity"


def test_initialize_refreshes_prior_schema_v3_dedup_identity_triggers(tmp_path):
    storage = NewsStorage(tmp_path / "prior-v3.sqlite3").initialize()
    with storage.connection() as connection:
        connection.executescript(
            """
            DROP TRIGGER trg_deduplication_relationships_occurrence_identity_insert;
            DROP TRIGGER trg_deduplication_relationships_occurrence_identity_update;

            CREATE TRIGGER
            trg_deduplication_relationships_occurrence_identity_insert
            BEFORE INSERT ON deduplication_relationships
            WHEN NOT EXISTS (
                SELECT 1 FROM article_ingestions
                WHERE occurrence_id = NEW.occurrence_id
                  AND ingestion_run_id = NEW.ingestion_run_id
                  AND duplicate_of IS NOT NULL
                  AND candidate_article_id =
                      NEW.duplicate_candidate_article_id
                  AND canonical_article_id = NEW.canonical_article_id
                  AND duplicate_of = NEW.duplicate_of
            )
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'deduplication relationship must exactly match a duplicate article occurrence'
                );
            END;

            CREATE TRIGGER
            trg_deduplication_relationships_occurrence_identity_update
            BEFORE UPDATE OF ingestion_run_id, occurrence_id,
                             duplicate_candidate_article_id,
                             canonical_article_id, duplicate_of
            ON deduplication_relationships
            WHEN NOT EXISTS (
                SELECT 1 FROM article_ingestions
                WHERE occurrence_id = NEW.occurrence_id
                  AND ingestion_run_id = NEW.ingestion_run_id
                  AND duplicate_of IS NOT NULL
                  AND candidate_article_id =
                      NEW.duplicate_candidate_article_id
                  AND canonical_article_id = NEW.canonical_article_id
                  AND duplicate_of = NEW.duplicate_of
            )
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'deduplication relationship must exactly match a duplicate article occurrence'
                );
            END;
            """
        )
        prior_insert_sql = connection.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'trigger'
              AND name =
                  'trg_deduplication_relationships_occurrence_identity_insert'
            """
        ).fetchone()["sql"]
    assert "deduplication_rule = NEW.deduplication_rule" not in prior_insert_sql

    storage.initialize()

    with storage.connection() as connection:
        triggers = {
            row["name"]: " ".join(row["sql"].split())
            for row in connection.execute(
                """
                SELECT name, sql FROM sqlite_master
                WHERE type = 'trigger'
                  AND name LIKE '%dedup%identity%'
                """
            )
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 3
    assert (
        "deduplication_rule = NEW.deduplication_rule"
        in triggers[
            "trg_deduplication_relationships_occurrence_identity_insert"
        ]
    )
    assert (
        "duplicate_of, deduplication_rule"
        in triggers[
            "trg_deduplication_relationships_occurrence_identity_update"
        ]
    )


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

    candidates = storage.find_duplicate_candidates(
        canonical_url="https://news.example.test/ogdc-discovery"
    )
    assert [row["article_id"] for row in candidates] == ["article-1"]
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

    storage.save_quality(
        ingestion_run_id=run_id,
        quality_status="PASS",
        report={"total_raw_records": 1, "total_normalized_records": 1},
    )
    storage.finalize_run(
        run_id,
        status="PASS",
        completed_at="2026-10-22T10:01:00Z",
        summary={"normalized_record_count": 1},
    )

    persisted_run = storage.get_run(run_id)
    assert persisted_run is not None
    assert persisted_run["status"] == "PASS"
    assert json.loads(persisted_run["summary_json"])["normalized_record_count"] == 1
    article = storage.get_article("article-1")
    assert article is not None
    assert article["published_at"] == "2026-10-22T04:00:00Z"
    assert article["normalized_title"] == "ogdc announces discovery"
    assert storage.list_canonical_articles()[0]["content_sha256"] == "a" * 64
    report = storage.get_quality_report(run_id)
    assert report is not None
    assert json.loads(report["report_json"])["total_raw_records"] == 1


def test_repeat_ingestion_preserves_raw_and_occurrence_evidence(storage: NewsStorage):
    first_run = "NEWS-20261022-100000-first"
    second_run = "NEWS-20261022-110000-second"
    _begin(storage, first_run)
    first_raw = _raw(storage, first_run, raw_record_id="shared/source-id")
    storage.save_article(_article(first_run, first_raw))

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
        normalized_payload={"article_id": "article-1"},
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
            "SELECT duplicate_of FROM article_ingestions WHERE occurrence_id = ?",
            (second_occurrence,),
        ).fetchone()
    assert row["duplicate_of"] is None


def test_cross_run_evidence_associations_are_rejected_by_api_and_triggers(
    storage: NewsStorage,
):
    first_run = "NEWS-20261022-100000-owner-one"
    second_run = "NEWS-20261022-110000-owner-two"
    _begin(storage, first_run)
    first_raw = _raw(storage, first_run, raw_record_id="owner-one")
    storage.save_article(_article(first_run, first_raw))

    _begin(storage, second_run)
    second_raw = _raw(storage, second_run, raw_record_id="owner-two")
    second_article = _article(
        second_run,
        second_raw,
        article_id="article-2",
        canonical_url="https://news.example.test/second-article",
    )
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

    with pytest.raises(sqlite3.IntegrityError, match="deduplication relationship"):
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

    with pytest.raises(RuntimeError, match="quality report"):
        storage.finalize_run(run_id, status="PASS")
    assert storage.get_run(run_id)["status"] == "RUNNING"

    rolled_back = "NEWS-20261022-100000-rolled-back"
    with pytest.raises(RuntimeError, match="force rollback"):
        with storage.transaction() as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs(
                    run_id, schema_version, provider_id, provider_mode,
                    configured_source, data_kind, started_at, status,
                    summary_json
                ) VALUES (?, '1.0', 'manual', 'manual-csv', NULL,
                          'imported', ?, 'RUNNING', '{}')
                """,
                (rolled_back, NOW),
            )
            raise RuntimeError("force rollback")
    assert storage.get_run(rolled_back) is None


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
    storage.save_quality(
        ingestion_run_id=run_id,
        quality_status="WARN",
        report={"provider_error_count": 1},
    )
    storage.finalize_run(
        run_id,
        status="PARTIAL",
        error_message="One configured source failed",
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
    storage.save_quality(
        ingestion_run_id=run_id,
        quality_status="PASS",
        report={"quality_status": "PASS"},
    )
    storage.finalize_run(run_id, status="PASS")

    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
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
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
        storage.save_article(late_article)
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
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
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
        storage.save_link(
            occurrence_id=occurrence_id,
            ticker="OGDC",
            company_name="Oil and Gas Development Company Limited",
            matched_alias="OGDCL",
            match_method="exact",
            match_location="summary",
            confidence="exact",
        )
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
        storage.save_dedup(
            ingestion_run_id=run_id,
            occurrence_id=duplicate_occurrence,
            duplicate_candidate_article_id="finalized-duplicate-candidate",
            canonical_article_id="article-1",
            deduplication_rule="canonical_url",
        )
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
        storage.save_error(
            ingestion_run_id=run_id,
            provider_id="permitted-rss",
            error_type="late_error",
            message="must not be appended",
        )
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
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

    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE raw_source_records SET payload_json = '{}' "
                "WHERE raw_evidence_id = ?",
                (raw_evidence_id,),
            )
    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
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
    storage.save_quality(
        ingestion_run_id=failed_quality,
        quality_status="FAIL",
        report={"quality_status": "FAIL"},
    )
    with pytest.raises(RuntimeError, match="PASS quality"):
        storage.finalize_run(failed_quality, status="PASS")
    assert storage.get_run(failed_quality)["status"] == "RUNNING"
    with pytest.raises(sqlite3.IntegrityError, match="PASS requires"):
        with storage.transaction() as connection:
            connection.execute(
                "UPDATE ingestion_runs SET status = 'PASS' WHERE run_id = ?",
                (failed_quality,),
            )

    provider_error_run = "NEWS-20261022-120000-provider-error-pass"
    _begin(storage, provider_error_run)
    storage.save_error(
        ingestion_run_id=provider_error_run,
        provider_id="permitted-rss",
        error_type="network_error",
        message="one source failed",
    )
    storage.save_quality(
        ingestion_run_id=provider_error_run,
        quality_status="PASS",
        report={"quality_status": "PASS"},
    )
    with pytest.raises(RuntimeError, match="provider errors"):
        storage.finalize_run(provider_error_run, status="PASS")

    partial_run = "NEWS-20261022-120000-partial-quality"
    _begin(storage, partial_run)
    storage.save_quality(
        ingestion_run_id=partial_run,
        quality_status="PASS",
        report={"quality_status": "PASS"},
    )
    with pytest.raises(RuntimeError, match="WARN quality"):
        storage.finalize_run(partial_run, status="PARTIAL")

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
