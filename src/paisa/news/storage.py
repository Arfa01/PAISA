"""SQLite persistence for immutable PAISA news-ingestion evidence.

The repository deliberately accepts mappings (and mapping-like dataclasses)
instead of importing the news domain models.  That keeps persistence at the
edge of the package and avoids coupling the database schema to Python object
construction.

``normalized_articles`` stores one canonical article.  Every normalized
occurrence, including a repeated acquisition or a duplicate candidate, is
retained in ``article_ingestions`` and remains connected to its raw evidence.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any


DATABASE_SCHEMA_VERSION = 7
NEWS_RECORD_SCHEMA_VERSION = "1.1"
_SUPPORTED_NEWS_RECORD_SCHEMA_VERSIONS = frozenset(
    {"1.0", NEWS_RECORD_SCHEMA_VERSION}
)
QUALITY_REPORT_SCHEMA_VERSION = "2.0"
SENTIMENT_RUN_SUMMARY_SCHEMA_VERSION = "2.0"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SUPPORTED_DEDUPLICATION_RULES = frozenset(
    {"canonical_url", "content_sha256", "title_source_time"}
)


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    applied_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    provider_mode TEXT NOT NULL,
    configured_source TEXT,
    data_kind TEXT NOT NULL CHECK (data_kind IN ('real', 'imported', 'fixture')),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('RUNNING', 'PASS', 'PARTIAL', 'FAIL')),
    artifact_directory TEXT,
    summary_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS raw_source_records (
    raw_evidence_id TEXT PRIMARY KEY,
    ingestion_run_id TEXT NOT NULL,
    raw_record_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    provider_record_id TEXT,
    configured_source TEXT,
    fetched_at TEXT NOT NULL,
    http_status INTEGER CHECK (
        http_status IS NULL OR (http_status >= 100 AND http_status <= 599)
    ),
    response_content_type TEXT,
    response_checksum TEXT CHECK (
        response_checksum IS NULL OR length(response_checksum) = 64
    ),
    data_kind TEXT NOT NULL CHECK (data_kind IN ('real', 'imported', 'fixture')),
    raw_artifact_reference TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    provider_warnings_json TEXT NOT NULL DEFAULT '[]',
    provider_errors_json TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_raw_record_per_run
ON raw_source_records(ingestion_run_id, raw_record_id);

CREATE INDEX IF NOT EXISTS ix_raw_provider_record
ON raw_source_records(provider_id, provider_record_id)
WHERE provider_record_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS normalized_articles (
    article_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_url TEXT,
    canonical_url TEXT,
    title TEXT,
    normalized_title TEXT NOT NULL DEFAULT '',
    summary TEXT,
    body TEXT,
    published_at TEXT,
    published_timezone TEXT,
    available_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    language TEXT,
    mentioned_tickers_json TEXT NOT NULL DEFAULT '[]',
    mentioned_companies_json TEXT NOT NULL DEFAULT '[]',
    mentioned_sectors_json TEXT NOT NULL DEFAULT '[]',
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    duplicate_of TEXT,
    first_ingestion_run_id TEXT NOT NULL,
    raw_artifact_reference TEXT NOT NULL,
    normalization_warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    CHECK (duplicate_of IS NULL OR duplicate_of <> article_id),
    CHECK (julianday(available_at) >= julianday(fetched_at)),
    CHECK (
        published_at IS NULL
        OR julianday(available_at) >= julianday(published_at)
    ),
    FOREIGN KEY (duplicate_of) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (first_ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_canonical_article_url
ON normalized_articles(canonical_url)
WHERE canonical_url IS NOT NULL AND duplicate_of IS NULL;

CREATE INDEX IF NOT EXISTS ix_normalized_articles_content_sha256
ON normalized_articles(content_sha256);

CREATE INDEX IF NOT EXISTS ix_normalized_articles_title_source_time
ON normalized_articles(normalized_title, source_name, published_at);

CREATE TABLE IF NOT EXISTS article_ingestions (
    occurrence_id TEXT PRIMARY KEY,
    ingestion_run_id TEXT NOT NULL,
    raw_evidence_id TEXT NOT NULL,
    candidate_article_id TEXT NOT NULL,
    canonical_article_id TEXT NOT NULL,
    duplicate_of TEXT,
    deduplication_rule TEXT,
    available_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    raw_artifact_reference TEXT NOT NULL,
    normalized_payload_json TEXT NOT NULL,
    normalization_warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    CHECK (
        (duplicate_of IS NULL AND candidate_article_id = canonical_article_id)
        OR duplicate_of = canonical_article_id
    ),
    CHECK (julianday(available_at) >= julianday(fetched_at)),
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (raw_evidence_id) REFERENCES raw_source_records(raw_evidence_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (canonical_article_id) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (duplicate_of) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    UNIQUE (ingestion_run_id, candidate_article_id)
);

CREATE INDEX IF NOT EXISTS ix_article_ingestions_canonical
ON article_ingestions(canonical_article_id);

CREATE TABLE IF NOT EXISTS article_company_links (
    link_id TEXT PRIMARY KEY,
    occurrence_id TEXT NOT NULL,
    canonical_article_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company_name TEXT NOT NULL,
    matched_alias TEXT NOT NULL,
    match_method TEXT NOT NULL,
    match_location TEXT NOT NULL,
    confidence TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (occurrence_id) REFERENCES article_ingestions(occurrence_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (canonical_article_id) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    UNIQUE (
        occurrence_id, ticker, matched_alias, match_method, match_location
    )
);

CREATE TABLE IF NOT EXISTS deduplication_relationships (
    relationship_id TEXT PRIMARY KEY,
    ingestion_run_id TEXT NOT NULL,
    occurrence_id TEXT NOT NULL UNIQUE,
    duplicate_candidate_article_id TEXT NOT NULL,
    canonical_article_id TEXT NOT NULL,
    duplicate_of TEXT NOT NULL,
    deduplication_rule TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    CHECK (canonical_article_id = duplicate_of),
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (occurrence_id) REFERENCES article_ingestions(occurrence_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (canonical_article_id) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (duplicate_of) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS provider_errors (
    error_id TEXT PRIMARY KEY,
    ingestion_run_id TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    configured_source TEXT,
    occurred_at TEXT NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    http_status INTEGER CHECK (
        http_status IS NULL OR (http_status >= 100 AND http_status <= 599)
    ),
    retryable INTEGER NOT NULL CHECK (retryable IN (0, 1)),
    details_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS quality_reports (
    ingestion_run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    quality_status TEXT NOT NULL CHECK (quality_status IN ('PASS', 'WARN', 'FAIL')),
    created_at TEXT NOT NULL,
    report_json TEXT NOT NULL,
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);
"""


def _evidence_guard_triggers(
    table: str, *, new_run_expression: str, old_run_expression: str
) -> str:
    """Build insert/update/delete guards for a table owned by an ingestion run."""

    return f"""
CREATE TRIGGER IF NOT EXISTS trg_{table}_insert_running
BEFORE INSERT ON {table}
WHEN COALESCE((SELECT status FROM ingestion_runs WHERE run_id = {new_run_expression}), '') <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'cannot write evidence for a finalized or missing ingestion run');
END;

CREATE TRIGGER IF NOT EXISTS trg_{table}_update_running
BEFORE UPDATE ON {table}
WHEN COALESCE((SELECT status FROM ingestion_runs WHERE run_id = {old_run_expression}), '') <> 'RUNNING'
  OR COALESCE((SELECT status FROM ingestion_runs WHERE run_id = {new_run_expression}), '') <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'cannot update evidence for a finalized or missing ingestion run');
END;

CREATE TRIGGER IF NOT EXISTS trg_{table}_delete_running
BEFORE DELETE ON {table}
WHEN COALESCE((SELECT status FROM ingestion_runs WHERE run_id = {old_run_expression}), '') <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'cannot delete evidence for a finalized ingestion run');
END;
"""


_IMMUTABILITY_SQL = "".join(
    (
        _evidence_guard_triggers(
            "raw_source_records",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _evidence_guard_triggers(
            "normalized_articles",
            new_run_expression="NEW.first_ingestion_run_id",
            old_run_expression="OLD.first_ingestion_run_id",
        ),
        _evidence_guard_triggers(
            "article_ingestions",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _evidence_guard_triggers(
            "article_company_links",
            new_run_expression=(
                "(SELECT ingestion_run_id FROM article_ingestions "
                "WHERE occurrence_id = NEW.occurrence_id)"
            ),
            old_run_expression=(
                "(SELECT ingestion_run_id FROM article_ingestions "
                "WHERE occurrence_id = OLD.occurrence_id)"
            ),
        ),
        _evidence_guard_triggers(
            "deduplication_relationships",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _evidence_guard_triggers(
            "provider_errors",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _evidence_guard_triggers(
            "quality_reports",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
    )
) + """
CREATE TRIGGER IF NOT EXISTS trg_ingestion_runs_protect_finalized_update
BEFORE UPDATE ON ingestion_runs
WHEN OLD.status <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'finalized ingestion runs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_ingestion_runs_protect_finalized_delete
BEFORE DELETE ON ingestion_runs
WHEN OLD.status <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'finalized ingestion runs are immutable');
END;

CREATE TRIGGER IF NOT EXISTS trg_ingestion_runs_truthful_pass
BEFORE UPDATE OF status ON ingestion_runs
WHEN NEW.status = 'PASS' AND (
    NOT EXISTS (
        SELECT 1 FROM quality_reports
        WHERE ingestion_run_id = OLD.run_id AND quality_status = 'PASS'
    )
    OR EXISTS (
        SELECT 1 FROM provider_errors WHERE ingestion_run_id = OLD.run_id
    )
)
BEGIN
    SELECT RAISE(ABORT, 'PASS requires PASS quality and no provider errors');
END;

CREATE TRIGGER IF NOT EXISTS trg_ingestion_runs_truthful_partial
BEFORE UPDATE OF status ON ingestion_runs
WHEN NEW.status = 'PARTIAL' AND NOT EXISTS (
    SELECT 1 FROM quality_reports
    WHERE ingestion_run_id = OLD.run_id AND quality_status = 'WARN'
)
BEGIN
    SELECT RAISE(ABORT, 'PARTIAL requires WARN quality');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_ingestions_require_raw_insert
BEFORE INSERT ON article_ingestions
WHEN NEW.raw_evidence_id IS NULL
BEGIN
    SELECT RAISE(ABORT, 'article ingestion requires raw evidence');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_ingestions_require_raw_update
BEFORE UPDATE OF raw_evidence_id ON article_ingestions
WHEN NEW.raw_evidence_id IS NULL
BEGIN
    SELECT RAISE(ABORT, 'article ingestion requires raw evidence');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_ingestions_raw_run_insert
BEFORE INSERT ON article_ingestions
WHEN NOT EXISTS (
    SELECT 1 FROM raw_source_records
    WHERE raw_evidence_id = NEW.raw_evidence_id
      AND ingestion_run_id = NEW.ingestion_run_id
)
BEGIN
    SELECT RAISE(ABORT, 'article ingestion raw evidence must belong to the same ingestion run');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_ingestions_raw_run_update
BEFORE UPDATE OF ingestion_run_id, raw_evidence_id ON article_ingestions
WHEN NOT EXISTS (
    SELECT 1 FROM raw_source_records
    WHERE raw_evidence_id = NEW.raw_evidence_id
      AND ingestion_run_id = NEW.ingestion_run_id
)
BEGIN
    SELECT RAISE(ABORT, 'article ingestion raw evidence must belong to the same ingestion run');
END;

CREATE TRIGGER IF NOT EXISTS trg_deduplication_relationships_occurrence_run_insert
BEFORE INSERT ON deduplication_relationships
WHEN NOT EXISTS (
    SELECT 1 FROM article_ingestions
    WHERE occurrence_id = NEW.occurrence_id
      AND ingestion_run_id = NEW.ingestion_run_id
)
BEGIN
    SELECT RAISE(ABORT, 'deduplication relationship must belong to its occurrence ingestion run');
END;

CREATE TRIGGER IF NOT EXISTS trg_deduplication_relationships_occurrence_run_update
BEFORE UPDATE OF ingestion_run_id, occurrence_id ON deduplication_relationships
WHEN NOT EXISTS (
    SELECT 1 FROM article_ingestions
    WHERE occurrence_id = NEW.occurrence_id
      AND ingestion_run_id = NEW.ingestion_run_id
)
BEGIN
    SELECT RAISE(ABORT, 'deduplication relationship must belong to its occurrence ingestion run');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_company_links_occurrence_owner_insert
BEFORE INSERT ON article_company_links
WHEN NOT EXISTS (
    SELECT 1 FROM article_ingestions
    WHERE occurrence_id = NEW.occurrence_id
      AND canonical_article_id = NEW.canonical_article_id
)
BEGIN
    SELECT RAISE(ABORT, 'article-company link must belong to its ingestion occurrence');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_company_links_occurrence_owner_update
BEFORE UPDATE OF occurrence_id, canonical_article_id ON article_company_links
WHEN NOT EXISTS (
    SELECT 1 FROM article_ingestions
    WHERE occurrence_id = NEW.occurrence_id
      AND canonical_article_id = NEW.canonical_article_id
)
BEGIN
    SELECT RAISE(ABORT, 'article-company link must belong to its ingestion occurrence');
END;

CREATE TRIGGER IF NOT EXISTS trg_quality_reports_status_insert
BEFORE INSERT ON quality_reports
WHEN NEW.quality_status NOT IN ('PASS', 'WARN', 'FAIL')
BEGIN
    SELECT RAISE(ABORT, 'invalid quality status');
END;

CREATE TRIGGER IF NOT EXISTS trg_quality_reports_status_update
BEFORE UPDATE OF quality_status ON quality_reports
WHEN NEW.quality_status NOT IN ('PASS', 'WARN', 'FAIL')
BEGIN
    SELECT RAISE(ABORT, 'invalid quality status');
END;

CREATE TRIGGER IF NOT EXISTS trg_normalized_articles_time_insert
BEFORE INSERT ON normalized_articles
WHEN julianday(NEW.available_at) IS NULL
  OR julianday(NEW.fetched_at) IS NULL
  OR julianday(NEW.available_at) < julianday(NEW.fetched_at)
  OR (
      NEW.published_at IS NOT NULL AND (
          julianday(NEW.published_at) IS NULL
          OR julianday(NEW.available_at) < julianday(NEW.published_at)
      )
  )
BEGIN
    SELECT RAISE(ABORT, 'invalid article availability timeline');
END;

CREATE TRIGGER IF NOT EXISTS trg_normalized_articles_time_update
BEFORE UPDATE OF published_at, available_at, fetched_at ON normalized_articles
WHEN julianday(NEW.available_at) IS NULL
  OR julianday(NEW.fetched_at) IS NULL
  OR julianday(NEW.available_at) < julianday(NEW.fetched_at)
  OR (
      NEW.published_at IS NOT NULL AND (
          julianday(NEW.published_at) IS NULL
          OR julianday(NEW.available_at) < julianday(NEW.published_at)
      )
  )
BEGIN
    SELECT RAISE(ABORT, 'invalid article availability timeline');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_ingestions_time_insert
BEFORE INSERT ON article_ingestions
WHEN julianday(NEW.available_at) IS NULL
  OR julianday(NEW.fetched_at) IS NULL
  OR julianday(NEW.available_at) < julianday(NEW.fetched_at)
BEGIN
    SELECT RAISE(ABORT, 'invalid ingestion availability timeline');
END;

CREATE TRIGGER IF NOT EXISTS trg_article_ingestions_time_update
BEFORE UPDATE OF available_at, fetched_at ON article_ingestions
WHEN julianday(NEW.available_at) IS NULL
  OR julianday(NEW.fetched_at) IS NULL
  OR julianday(NEW.available_at) < julianday(NEW.fetched_at)
BEGIN
    SELECT RAISE(ABORT, 'invalid ingestion availability timeline');
END;
"""


# Schema v3 adds cross-table deduplication invariants.  It is installed exactly
# once by the migration runner and verified on every later open.
_DEDUPLICATION_INTEGRITY_SQL = """
DROP TRIGGER IF EXISTS trg_deduplication_relationships_occurrence_identity_insert;
DROP TRIGGER IF EXISTS trg_deduplication_relationships_occurrence_identity_update;
DROP TRIGGER IF EXISTS trg_article_ingestions_deduplication_rule_insert;
DROP TRIGGER IF EXISTS trg_article_ingestions_deduplication_rule_update;
DROP TRIGGER IF EXISTS trg_article_ingestions_dedup_relationship_update;

CREATE TRIGGER trg_article_ingestions_deduplication_rule_insert
BEFORE INSERT ON article_ingestions
WHEN NEW.deduplication_rule IS NOT NULL
 AND NEW.deduplication_rule NOT IN (
     'canonical_url', 'content_sha256', 'title_source_time'
 )
BEGIN
    SELECT RAISE(ABORT, 'article ingestion uses an unsupported deduplication rule');
END;

CREATE TRIGGER trg_article_ingestions_deduplication_rule_update
BEFORE UPDATE OF deduplication_rule ON article_ingestions
WHEN NEW.deduplication_rule IS NOT NULL
 AND NEW.deduplication_rule NOT IN (
     'canonical_url', 'content_sha256', 'title_source_time'
 )
BEGIN
    SELECT RAISE(ABORT, 'article ingestion uses an unsupported deduplication rule');
END;

CREATE TRIGGER trg_deduplication_relationships_occurrence_identity_insert
BEFORE INSERT ON deduplication_relationships
WHEN NEW.deduplication_rule NOT IN (
        'canonical_url', 'content_sha256', 'title_source_time'
     )
  OR NOT EXISTS (
    SELECT 1 FROM article_ingestions
    WHERE occurrence_id = NEW.occurrence_id
      AND ingestion_run_id = NEW.ingestion_run_id
      AND duplicate_of IS NOT NULL
      AND candidate_article_id = NEW.duplicate_candidate_article_id
      AND canonical_article_id = NEW.canonical_article_id
      AND duplicate_of = NEW.duplicate_of
      AND deduplication_rule = NEW.deduplication_rule
)
BEGIN
    SELECT RAISE(ABORT, 'deduplication relationship must use a supported rule and exactly match a duplicate article occurrence');
END;

CREATE TRIGGER trg_deduplication_relationships_occurrence_identity_update
BEFORE UPDATE OF ingestion_run_id, occurrence_id, duplicate_candidate_article_id,
                 canonical_article_id, duplicate_of, deduplication_rule
ON deduplication_relationships
WHEN NEW.deduplication_rule NOT IN (
        'canonical_url', 'content_sha256', 'title_source_time'
     )
  OR NOT EXISTS (
    SELECT 1 FROM article_ingestions
    WHERE occurrence_id = NEW.occurrence_id
      AND ingestion_run_id = NEW.ingestion_run_id
      AND duplicate_of IS NOT NULL
      AND candidate_article_id = NEW.duplicate_candidate_article_id
      AND canonical_article_id = NEW.canonical_article_id
      AND duplicate_of = NEW.duplicate_of
      AND deduplication_rule = NEW.deduplication_rule
)
BEGIN
    SELECT RAISE(ABORT, 'deduplication relationship must use a supported rule and exactly match a duplicate article occurrence');
END;

CREATE TRIGGER trg_article_ingestions_dedup_relationship_update
BEFORE UPDATE OF occurrence_id, ingestion_run_id, candidate_article_id,
                 canonical_article_id, duplicate_of, deduplication_rule
ON article_ingestions
WHEN EXISTS (
    SELECT 1 FROM deduplication_relationships
    WHERE occurrence_id = OLD.occurrence_id
)
 AND NOT EXISTS (
    SELECT 1 FROM deduplication_relationships
    WHERE occurrence_id = OLD.occurrence_id
      AND occurrence_id = NEW.occurrence_id
      AND ingestion_run_id = NEW.ingestion_run_id
      AND duplicate_candidate_article_id = NEW.candidate_article_id
      AND canonical_article_id = NEW.canonical_article_id
      AND duplicate_of = NEW.duplicate_of
      AND deduplication_rule = NEW.deduplication_rule
)
BEGIN
    SELECT RAISE(ABORT, 'article ingestion must remain consistent with its deduplication relationship');
END;
"""


def _sealed_evidence_guard_triggers(
    table: str, *, new_run_expression: str, old_run_expression: str
) -> str:
    """Prevent any evidence mutation after a run has been cryptographically sealed."""

    return f"""
CREATE TRIGGER trg_{table}_insert_unsealed
BEFORE INSERT ON {table}
WHEN (SELECT sealed_manifest_sha256 FROM ingestion_runs
      WHERE run_id = {new_run_expression}) IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'cannot write evidence for a sealed ingestion run');
END;

CREATE TRIGGER trg_{table}_update_unsealed
BEFORE UPDATE ON {table}
WHEN (SELECT sealed_manifest_sha256 FROM ingestion_runs
      WHERE run_id = {old_run_expression}) IS NOT NULL
  OR (SELECT sealed_manifest_sha256 FROM ingestion_runs
      WHERE run_id = {new_run_expression}) IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'cannot update evidence for a sealed ingestion run');
END;

CREATE TRIGGER trg_{table}_delete_unsealed
BEFORE DELETE ON {table}
WHEN (SELECT sealed_manifest_sha256 FROM ingestion_runs
      WHERE run_id = {old_run_expression}) IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'cannot delete evidence for a sealed ingestion run');
END;
"""


_SEALED_EVIDENCE_SQL = "".join(
    (
        _sealed_evidence_guard_triggers(
            "raw_source_records",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _sealed_evidence_guard_triggers(
            "normalized_articles",
            new_run_expression="NEW.first_ingestion_run_id",
            old_run_expression="OLD.first_ingestion_run_id",
        ),
        _sealed_evidence_guard_triggers(
            "article_ingestions",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _sealed_evidence_guard_triggers(
            "article_company_links",
            new_run_expression=(
                "(SELECT ingestion_run_id FROM article_ingestions "
                "WHERE occurrence_id = NEW.occurrence_id)"
            ),
            old_run_expression=(
                "(SELECT ingestion_run_id FROM article_ingestions "
                "WHERE occurrence_id = OLD.occurrence_id)"
            ),
        ),
        _sealed_evidence_guard_triggers(
            "deduplication_relationships",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _sealed_evidence_guard_triggers(
            "provider_errors",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
        _sealed_evidence_guard_triggers(
            "quality_reports",
            new_run_expression="NEW.ingestion_run_id",
            old_run_expression="OLD.ingestion_run_id",
        ),
    )
)


_SCHEMA_V4_SQL = """
ALTER TABLE ingestion_runs ADD COLUMN prepared_at TEXT;
ALTER TABLE ingestion_runs ADD COLUMN sealed_artifact_inventory_json TEXT;
ALTER TABLE ingestion_runs ADD COLUMN sealed_artifact_inventory_sha256 TEXT;
ALTER TABLE ingestion_runs ADD COLUMN sealed_manifest_sha256 TEXT;
ALTER TABLE ingestion_runs ADD COLUMN sealed_completion_sha256 TEXT;

CREATE UNIQUE INDEX uq_ingestion_runs_single_running
ON ingestion_runs((1)) WHERE status = 'RUNNING';

CREATE UNIQUE INDEX uq_canonical_article_content_sha256
ON normalized_articles(content_sha256)
WHERE duplicate_of IS NULL;

CREATE UNIQUE INDEX uq_canonical_article_title_source_minute
ON normalized_articles(
    normalized_title,
    source_name,
    strftime('%Y-%m-%dT%H:%M', published_at)
)
WHERE duplicate_of IS NULL
  AND normalized_title <> ''
  AND published_at IS NOT NULL;

CREATE TRIGGER trg_ingestion_runs_insert_running
BEFORE INSERT ON ingestion_runs
WHEN NEW.status <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'ingestion runs must be inserted as RUNNING');
END;

CREATE TRIGGER trg_ingestion_runs_seal_shape_update
BEFORE UPDATE OF prepared_at, sealed_artifact_inventory_json,
                 sealed_artifact_inventory_sha256, sealed_manifest_sha256,
                 sealed_completion_sha256
ON ingestion_runs
WHEN (
    (NEW.sealed_manifest_sha256 IS NULL) <>
    (NEW.sealed_artifact_inventory_json IS NULL)
    OR (NEW.sealed_manifest_sha256 IS NULL) <>
       (NEW.sealed_artifact_inventory_sha256 IS NULL)
    OR (NEW.sealed_manifest_sha256 IS NULL) <>
       (NEW.sealed_completion_sha256 IS NULL)
    OR (NEW.sealed_manifest_sha256 IS NULL) <> (NEW.prepared_at IS NULL)
    OR (
        NEW.sealed_manifest_sha256 IS NOT NULL AND (
            length(NEW.sealed_manifest_sha256) <> 64
            OR length(NEW.sealed_artifact_inventory_sha256) <> 64
            OR length(NEW.sealed_completion_sha256) <> 64
            OR json_valid(NEW.sealed_artifact_inventory_json) <> 1
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'ingestion run seal fields are incomplete or invalid');
END;

CREATE TRIGGER trg_ingestion_runs_seal_immutable_update
BEFORE UPDATE OF prepared_at, sealed_artifact_inventory_json,
                 sealed_artifact_inventory_sha256, sealed_manifest_sha256,
                 sealed_completion_sha256
ON ingestion_runs
WHEN OLD.sealed_manifest_sha256 IS NOT NULL AND (
    NEW.prepared_at IS NOT OLD.prepared_at
    OR NEW.sealed_artifact_inventory_json IS NOT OLD.sealed_artifact_inventory_json
    OR NEW.sealed_artifact_inventory_sha256 IS NOT OLD.sealed_artifact_inventory_sha256
    OR NEW.sealed_manifest_sha256 IS NOT OLD.sealed_manifest_sha256
    OR NEW.sealed_completion_sha256 IS NOT OLD.sealed_completion_sha256
)
BEGIN
    SELECT RAISE(ABORT, 'ingestion run finalization seal is immutable');
END;

CREATE TRIGGER trg_normalized_articles_canonical_only_insert
BEFORE INSERT ON normalized_articles
WHEN NEW.duplicate_of IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'normalized_articles stores canonical rows only');
END;

CREATE TRIGGER trg_normalized_articles_canonical_only_update
BEFORE UPDATE OF duplicate_of ON normalized_articles
WHEN NEW.duplicate_of IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'normalized_articles stores canonical rows only');
END;

CREATE TRIGGER trg_article_ingestions_finalized_target_insert
BEFORE INSERT ON article_ingestions
WHEN NOT EXISTS (
    SELECT 1
    FROM normalized_articles AS canonical
    JOIN ingestion_runs AS owner
      ON owner.run_id = canonical.first_ingestion_run_id
    WHERE canonical.article_id = NEW.canonical_article_id
      AND canonical.duplicate_of IS NULL
      AND (
          owner.run_id = NEW.ingestion_run_id
          OR owner.status IN ('PASS', 'PARTIAL')
      )
)
BEGIN
    SELECT RAISE(ABORT, 'article ingestion must target a same-run or finalized canonical');
END;

CREATE TRIGGER trg_article_ingestions_finalized_target_update
BEFORE UPDATE OF ingestion_run_id, canonical_article_id ON article_ingestions
WHEN NOT EXISTS (
    SELECT 1
    FROM normalized_articles AS canonical
    JOIN ingestion_runs AS owner
      ON owner.run_id = canonical.first_ingestion_run_id
    WHERE canonical.article_id = NEW.canonical_article_id
      AND canonical.duplicate_of IS NULL
      AND (
          owner.run_id = NEW.ingestion_run_id
          OR owner.status IN ('PASS', 'PARTIAL')
      )
)
BEGIN
    SELECT RAISE(ABORT, 'article ingestion must target a same-run or finalized canonical');
END;

CREATE TRIGGER trg_normalized_articles_referenced_update
BEFORE UPDATE ON normalized_articles
WHEN EXISTS (
    SELECT 1
    FROM article_ingestions AS occurrence
    JOIN ingestion_runs AS consumer
      ON consumer.run_id = occurrence.ingestion_run_id
    WHERE occurrence.canonical_article_id = OLD.article_id
      AND occurrence.ingestion_run_id <> OLD.first_ingestion_run_id
      AND consumer.status <> 'RUNNING'
)
BEGIN
    SELECT RAISE(ABORT, 'referenced canonical articles are append-only');
END;

CREATE TRIGGER trg_normalized_articles_referenced_delete
BEFORE DELETE ON normalized_articles
WHEN EXISTS (
    SELECT 1
    FROM article_ingestions AS occurrence
    JOIN ingestion_runs AS consumer
      ON consumer.run_id = occurrence.ingestion_run_id
    WHERE occurrence.canonical_article_id = OLD.article_id
      AND occurrence.ingestion_run_id <> OLD.first_ingestion_run_id
      AND consumer.status <> 'RUNNING'
)
BEGIN
    SELECT RAISE(ABORT, 'referenced canonical articles are append-only');
END;

DROP TRIGGER trg_ingestion_runs_truthful_pass;
DROP TRIGGER trg_ingestion_runs_truthful_partial;

CREATE TRIGGER trg_ingestion_runs_truthful_pass
BEFORE UPDATE OF status ON ingestion_runs
WHEN NEW.status = 'PASS' AND (
    NEW.sealed_manifest_sha256 IS NULL
    OR NEW.sealed_artifact_inventory_sha256 IS NULL
    OR NEW.sealed_completion_sha256 IS NULL
    OR NOT EXISTS (
        SELECT 1 FROM quality_reports AS quality
        WHERE quality.ingestion_run_id = OLD.run_id
          AND quality.schema_version = '2.0'
          AND quality.quality_status = 'PASS'
          AND json_valid(quality.report_json) = 1
          AND json_extract(quality.report_json, '$.schema_version') = '2.0'
          AND json_extract(quality.report_json, '$.quality_status') = 'PASS'
          AND json_extract(quality.report_json, '$.total_raw_records') =
              (SELECT count(*) FROM raw_source_records
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.total_normalized_records') =
              (SELECT count(*) FROM article_ingestions
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.normalized_occurrence_count') =
              (SELECT count(*) FROM article_ingestions
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.new_canonical_article_count') =
              (SELECT count(*) FROM normalized_articles
               WHERE first_ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.matched_existing_canonical_count') =
              (SELECT count(*)
               FROM article_ingestions AS occurrence
               JOIN normalized_articles AS canonical
                 ON canonical.article_id = occurrence.canonical_article_id
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND occurrence.duplicate_of IS NOT NULL
                 AND canonical.first_ingestion_run_id <> occurrence.ingestion_run_id)
          AND json_extract(quality.report_json, '$.historical_duplicate_count') =
              (SELECT count(*)
               FROM article_ingestions AS occurrence
               JOIN normalized_articles AS canonical
                 ON canonical.article_id = occurrence.canonical_article_id
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND occurrence.duplicate_of IS NOT NULL
                 AND canonical.first_ingestion_run_id <> occurrence.ingestion_run_id)
          AND json_extract(quality.report_json, '$.within_run_duplicate_count') =
              ((SELECT count(*) FROM article_ingestions
                WHERE ingestion_run_id = OLD.run_id AND duplicate_of IS NOT NULL)
               -
               (SELECT count(*)
                FROM article_ingestions AS occurrence
                JOIN normalized_articles AS canonical
                  ON canonical.article_id = occurrence.canonical_article_id
                WHERE occurrence.ingestion_run_id = OLD.run_id
                  AND occurrence.duplicate_of IS NOT NULL
                  AND canonical.first_ingestion_run_id <> occurrence.ingestion_run_id))
          AND json_extract(quality.report_json, '$.represented_canonical_count') =
              (SELECT count(DISTINCT canonical_article_id)
               FROM article_ingestions WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.duplicate_relationship_count') =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.duplicate_count') =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id)
          AND json_type(quality.report_json, '$.deduplication_rule_counts') = 'object'
          AND (SELECT count(*)
               FROM json_each(quality.report_json, '$.deduplication_rule_counts')) = 3
          AND json_extract(
                  quality.report_json,
                  '$.deduplication_rule_counts.canonical_url'
              ) =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id
                 AND deduplication_rule = 'canonical_url')
          AND json_extract(
                  quality.report_json,
                  '$.deduplication_rule_counts.content_sha256'
              ) =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id
                 AND deduplication_rule = 'content_sha256')
          AND json_extract(
                  quality.report_json,
                  '$.deduplication_rule_counts.title_source_time'
              ) =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id
                 AND deduplication_rule = 'title_source_time')
          AND json_extract(quality.report_json, '$.provider_error_count') = 0
          AND json_extract(quality.report_json, '$.rejected_record_count') = 0
          AND json_extract(quality.report_json, '$.missing_title_count') = 0
          AND json_extract(
                  quality.report_json, '$.missing_publication_time_count'
              ) = 0
          AND json_extract(quality.report_json, '$.missing_body_count') =
              (SELECT count(*) FROM article_ingestions AS occurrence
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND (
                     json_extract(occurrence.normalized_payload_json, '$.body') IS NULL
                     OR trim(json_extract(
                         occurrence.normalized_payload_json, '$.body'
                     )) = ''
                 ))
          AND json_extract(quality.report_json, '$.invalid_url_count') = 0
          AND json_extract(quality.report_json, '$.ogdc_linked_article_count') =
              (SELECT count(DISTINCT link.occurrence_id)
               FROM article_company_links AS link
               JOIN article_ingestions AS occurrence
                 ON occurrence.occurrence_id = link.occurrence_id
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND upper(link.ticker) = 'OGDC')
          AND (SELECT count(*) FROM raw_source_records
               WHERE ingestion_run_id = OLD.run_id) =
              (SELECT count(*) FROM article_ingestions
               WHERE ingestion_run_id = OLD.run_id)
          AND NOT EXISTS (
              SELECT 1 FROM article_ingestions AS occurrence
              WHERE occurrence.ingestion_run_id = OLD.run_id
                AND (
                    json_extract(occurrence.normalized_payload_json, '$.title') IS NULL
                    OR trim(json_extract(
                        occurrence.normalized_payload_json, '$.title'
                    )) = ''
                    OR json_extract(
                        occurrence.normalized_payload_json, '$.published_at'
                    ) IS NULL
                    OR json_extract(
                        occurrence.normalized_payload_json, '$.published_at'
                    ) = ''
                    OR EXISTS (
                        SELECT 1
                        FROM json_each(occurrence.normalization_warnings_json)
                        WHERE value = 'source_url_invalid'
                    )
                )
          )
    )
    OR EXISTS (
        SELECT 1 FROM provider_errors WHERE ingestion_run_id = OLD.run_id
    )
    OR (SELECT count(*) FROM raw_source_records
        WHERE ingestion_run_id = OLD.run_id) = 0
    OR (SELECT count(*) FROM article_ingestions
        WHERE ingestion_run_id = OLD.run_id) = 0
    OR EXISTS (
        SELECT 1 FROM article_ingestions AS occurrence
        LEFT JOIN deduplication_relationships AS relationship
          ON relationship.occurrence_id = occurrence.occurrence_id
        WHERE occurrence.ingestion_run_id = OLD.run_id
          AND (
              (occurrence.duplicate_of IS NULL AND relationship.occurrence_id IS NOT NULL)
              OR (occurrence.duplicate_of IS NOT NULL AND relationship.occurrence_id IS NULL)
          )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'PASS requires sealed, nonempty, internally consistent evidence');
END;

CREATE TRIGGER trg_ingestion_runs_truthful_partial
BEFORE UPDATE OF status ON ingestion_runs
WHEN NEW.status = 'PARTIAL' AND (
    NEW.sealed_manifest_sha256 IS NULL
    OR NEW.sealed_artifact_inventory_sha256 IS NULL
    OR NEW.sealed_completion_sha256 IS NULL
    OR NOT EXISTS (
        SELECT 1 FROM quality_reports AS quality
        WHERE quality.ingestion_run_id = OLD.run_id
          AND quality.schema_version = '2.0'
          AND quality.quality_status = 'WARN'
          AND json_valid(quality.report_json) = 1
          AND json_extract(quality.report_json, '$.schema_version') = '2.0'
          AND json_extract(quality.report_json, '$.quality_status') = 'WARN'
          AND json_extract(quality.report_json, '$.total_raw_records') =
              (SELECT count(*) FROM raw_source_records
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.total_normalized_records') =
              (SELECT count(*) FROM article_ingestions
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.normalized_occurrence_count') =
              (SELECT count(*) FROM article_ingestions
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.new_canonical_article_count') =
              (SELECT count(*) FROM normalized_articles
               WHERE first_ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.matched_existing_canonical_count') =
              (SELECT count(*)
               FROM article_ingestions AS occurrence
               JOIN normalized_articles AS canonical
                 ON canonical.article_id = occurrence.canonical_article_id
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND occurrence.duplicate_of IS NOT NULL
                 AND canonical.first_ingestion_run_id <> occurrence.ingestion_run_id)
          AND json_extract(quality.report_json, '$.historical_duplicate_count') =
              (SELECT count(*)
               FROM article_ingestions AS occurrence
               JOIN normalized_articles AS canonical
                 ON canonical.article_id = occurrence.canonical_article_id
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND occurrence.duplicate_of IS NOT NULL
                 AND canonical.first_ingestion_run_id <> occurrence.ingestion_run_id)
          AND json_extract(quality.report_json, '$.within_run_duplicate_count') =
              ((SELECT count(*) FROM article_ingestions
                WHERE ingestion_run_id = OLD.run_id AND duplicate_of IS NOT NULL)
               -
               (SELECT count(*)
                FROM article_ingestions AS occurrence
                JOIN normalized_articles AS canonical
                  ON canonical.article_id = occurrence.canonical_article_id
                WHERE occurrence.ingestion_run_id = OLD.run_id
                  AND occurrence.duplicate_of IS NOT NULL
                  AND canonical.first_ingestion_run_id <> occurrence.ingestion_run_id))
          AND json_extract(quality.report_json, '$.represented_canonical_count') =
              (SELECT count(DISTINCT canonical_article_id)
               FROM article_ingestions WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.duplicate_relationship_count') =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.duplicate_count') =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id)
          AND json_type(quality.report_json, '$.deduplication_rule_counts') = 'object'
          AND (SELECT count(*)
               FROM json_each(quality.report_json, '$.deduplication_rule_counts')) = 3
          AND json_extract(
                  quality.report_json,
                  '$.deduplication_rule_counts.canonical_url'
              ) =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id
                 AND deduplication_rule = 'canonical_url')
          AND json_extract(
                  quality.report_json,
                  '$.deduplication_rule_counts.content_sha256'
              ) =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id
                 AND deduplication_rule = 'content_sha256')
          AND json_extract(
                  quality.report_json,
                  '$.deduplication_rule_counts.title_source_time'
              ) =
              (SELECT count(*) FROM deduplication_relationships
               WHERE ingestion_run_id = OLD.run_id
                 AND deduplication_rule = 'title_source_time')
          AND json_extract(quality.report_json, '$.provider_error_count') =
              (SELECT count(*) FROM provider_errors
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.rejected_record_count') =
              ((SELECT count(*) FROM raw_source_records
                WHERE ingestion_run_id = OLD.run_id)
               -
               (SELECT count(*) FROM article_ingestions
                WHERE ingestion_run_id = OLD.run_id))
          AND (SELECT count(*) FROM raw_source_records
               WHERE ingestion_run_id = OLD.run_id) >=
              (SELECT count(*) FROM article_ingestions
               WHERE ingestion_run_id = OLD.run_id)
          AND json_extract(quality.report_json, '$.missing_title_count') =
              (SELECT count(*) FROM article_ingestions AS occurrence
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND (
                     json_extract(occurrence.normalized_payload_json, '$.title') IS NULL
                     OR trim(json_extract(
                         occurrence.normalized_payload_json, '$.title'
                     )) = ''
                 ))
          AND json_extract(
                  quality.report_json, '$.missing_publication_time_count'
              ) =
              (SELECT count(*) FROM article_ingestions AS occurrence
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND (
                     json_extract(
                         occurrence.normalized_payload_json, '$.published_at'
                     ) IS NULL
                     OR json_extract(
                         occurrence.normalized_payload_json, '$.published_at'
                     ) = ''
                 ))
          AND json_extract(quality.report_json, '$.missing_body_count') =
              (SELECT count(*) FROM article_ingestions AS occurrence
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND (
                     json_extract(occurrence.normalized_payload_json, '$.body') IS NULL
                     OR trim(json_extract(
                         occurrence.normalized_payload_json, '$.body'
                     )) = ''
                 ))
          AND json_extract(quality.report_json, '$.invalid_url_count') =
              (SELECT count(*) FROM article_ingestions AS occurrence
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND EXISTS (
                     SELECT 1
                     FROM json_each(occurrence.normalization_warnings_json)
                     WHERE value = 'source_url_invalid'
                 ))
          AND json_extract(quality.report_json, '$.ogdc_linked_article_count') =
              (SELECT count(DISTINCT link.occurrence_id)
               FROM article_company_links AS link
               JOIN article_ingestions AS occurrence
                 ON occurrence.occurrence_id = link.occurrence_id
               WHERE occurrence.ingestion_run_id = OLD.run_id
                 AND upper(link.ticker) = 'OGDC')
    )
    OR (SELECT count(*) FROM raw_source_records
        WHERE ingestion_run_id = OLD.run_id) = 0
    OR (SELECT count(*) FROM article_ingestions
        WHERE ingestion_run_id = OLD.run_id) = 0
    OR EXISTS (
        SELECT 1 FROM article_ingestions AS occurrence
        LEFT JOIN deduplication_relationships AS relationship
          ON relationship.occurrence_id = occurrence.occurrence_id
        WHERE occurrence.ingestion_run_id = OLD.run_id
          AND (
              (occurrence.duplicate_of IS NULL AND relationship.occurrence_id IS NOT NULL)
              OR (occurrence.duplicate_of IS NOT NULL AND relationship.occurrence_id IS NULL)
          )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'PARTIAL requires sealed, nonempty, internally consistent evidence');
END;
""" + _SEALED_EVIDENCE_SQL


_SCHEMA_V5_SQL = """
CREATE TABLE sentiment_model_revisions (
    model_revision_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    model_repository TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    tokenizer_repository TEXT NOT NULL,
    tokenizer_revision TEXT NOT NULL,
    original_labels_json TEXT NOT NULL,
    label_mapping_json TEXT NOT NULL,
    license_identifier TEXT NOT NULL,
    library_versions_json TEXT NOT NULL,
    model_config_sha256 TEXT,
    tokenizer_config_sha256 TEXT,
    created_at TEXT NOT NULL,
    CHECK (json_valid(original_labels_json) = 1),
    CHECK (json_type(original_labels_json) IN ('array', 'object')),
    CHECK (json_valid(label_mapping_json) = 1),
    CHECK (json_type(label_mapping_json) = 'object'),
    CHECK (json_valid(library_versions_json) = 1),
    CHECK (json_type(library_versions_json) = 'object'),
    CHECK (
        model_config_sha256 IS NULL OR length(model_config_sha256) = 64
    ),
    CHECK (
        tokenizer_config_sha256 IS NULL OR length(tokenizer_config_sha256) = 64
    ),
    UNIQUE (
        model_repository,
        model_revision,
        tokenizer_repository,
        tokenizer_revision
    )
);

CREATE TABLE sentiment_inference_runs (
    run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    source_ingestion_run_id TEXT NOT NULL,
    model_revision_id TEXT NOT NULL,
    input_policy_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('RUNNING', 'PASS', 'PARTIAL', 'FAIL')
    ),
    device TEXT NOT NULL,
    batch_size INTEGER NOT NULL CHECK (batch_size > 0),
    maximum_input_characters INTEGER NOT NULL CHECK (
        maximum_input_characters > 0
    ),
    maximum_sequence_length INTEGER NOT NULL CHECK (
        maximum_sequence_length > 0
    ),
    maximum_articles INTEGER CHECK (
        maximum_articles IS NULL OR maximum_articles > 0
    ),
    timeout_seconds REAL CHECK (
        timeout_seconds IS NULL OR timeout_seconds > 0
    ),
    artifact_directory TEXT NOT NULL,
    configuration_json TEXT NOT NULL DEFAULT '{}',
    summary_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    CHECK (json_valid(configuration_json) = 1),
    CHECK (json_type(configuration_json) = 'object'),
    CHECK (json_valid(summary_json) = 1),
    CHECK (json_type(summary_json) = 'object'),
    FOREIGN KEY (source_ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (model_revision_id)
        REFERENCES sentiment_model_revisions(model_revision_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX ix_sentiment_runs_source
ON sentiment_inference_runs(source_ingestion_run_id, run_id);

CREATE TABLE article_sentiment_outputs (
    sentiment_output_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    inference_run_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    source_ingestion_run_id TEXT NOT NULL,
    model_revision_id TEXT NOT NULL,
    input_policy_id TEXT NOT NULL,
    selected_fields_json TEXT NOT NULL,
    input_text_sha256 TEXT NOT NULL CHECK (length(input_text_sha256) = 64),
    character_count INTEGER NOT NULL CHECK (character_count > 0),
    original_character_count INTEGER NOT NULL CHECK (
        original_character_count >= character_count
    ),
    character_truncation_applied INTEGER NOT NULL CHECK (
        character_truncation_applied IN (0, 1)
    ),
    token_truncation_applied INTEGER NOT NULL CHECK (
        token_truncation_applied IN (0, 1)
    ),
    maximum_input_characters INTEGER NOT NULL CHECK (
        maximum_input_characters > 0
    ),
    tokenizer_token_count INTEGER NOT NULL CHECK (tokenizer_token_count > 0),
    truncation_applied INTEGER NOT NULL CHECK (truncation_applied IN (0, 1)),
    maximum_sequence_length INTEGER NOT NULL CHECK (maximum_sequence_length > 0),
    original_language TEXT,
    published_at TEXT,
    available_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    class_labels_json TEXT NOT NULL,
    negative_probability REAL NOT NULL CHECK (
        negative_probability >= 0.0 AND negative_probability <= 1.0
    ),
    neutral_probability REAL NOT NULL CHECK (
        neutral_probability >= 0.0 AND neutral_probability <= 1.0
    ),
    positive_probability REAL NOT NULL CHECK (
        positive_probability >= 0.0 AND positive_probability <= 1.0
    ),
    predicted_label TEXT NOT NULL CHECK (
        predicted_label IN ('negative', 'neutral', 'positive')
    ),
    predicted_probability REAL NOT NULL CHECK (
        predicted_probability >= 0.0 AND predicted_probability <= 1.0
    ),
    inference_timestamp TEXT NOT NULL,
    device TEXT NOT NULL,
    batch_size INTEGER NOT NULL CHECK (batch_size > 0),
    library_versions_json TEXT NOT NULL,
    warning_codes_json TEXT NOT NULL DEFAULT '[]',
    error_code TEXT,
    error_message TEXT,
    CHECK (json_valid(selected_fields_json) = 1),
    CHECK (json_type(selected_fields_json) = 'array'),
    CHECK (json_array_length(selected_fields_json) > 0),
    CHECK (json_valid(class_labels_json) = 1),
    CHECK (json_type(class_labels_json) = 'array'),
    CHECK (json_array_length(class_labels_json) = 3),
    CHECK (json_extract(class_labels_json, '$[0]') = 'negative'),
    CHECK (json_extract(class_labels_json, '$[1]') = 'neutral'),
    CHECK (json_extract(class_labels_json, '$[2]') = 'positive'),
    CHECK (json_valid(library_versions_json) = 1),
    CHECK (json_type(library_versions_json) = 'object'),
    CHECK (json_valid(warning_codes_json) = 1),
    CHECK (json_type(warning_codes_json) = 'array'),
    CHECK (character_count <= maximum_input_characters),
    CHECK (
        character_truncation_applied = (
            original_character_count > character_count
        )
    ),
    CHECK (
        token_truncation_applied = (
            tokenizer_token_count > maximum_sequence_length
        )
    ),
    CHECK (
        truncation_applied = (
            character_truncation_applied OR token_truncation_applied
        )
    ),
    CHECK (
        abs(
            negative_probability + neutral_probability + positive_probability
            - 1.0
        ) <= 0.000001
    ),
    CHECK (
        (predicted_label = 'negative'
         AND abs(predicted_probability - negative_probability) <= 0.000001)
        OR (predicted_label = 'neutral'
            AND abs(predicted_probability - neutral_probability) <= 0.000001)
        OR (predicted_label = 'positive'
            AND abs(predicted_probability - positive_probability) <= 0.000001)
    ),
    CHECK (
        predicted_probability + 0.000001 >= negative_probability
        AND predicted_probability + 0.000001 >= neutral_probability
        AND predicted_probability + 0.000001 >= positive_probability
    ),
    CHECK (
        (error_code IS NULL AND error_message IS NULL)
        OR (error_code IS NOT NULL AND error_message IS NOT NULL)
    ),
    UNIQUE (article_id, model_revision_id, input_policy_id),
    FOREIGN KEY (inference_run_id) REFERENCES sentiment_inference_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (article_id) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (source_ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (model_revision_id)
        REFERENCES sentiment_model_revisions(model_revision_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX ix_article_sentiment_outputs_run
ON article_sentiment_outputs(inference_run_id, article_id);

CREATE INDEX ix_article_sentiment_outputs_available
ON article_sentiment_outputs(available_at, article_id);

CREATE TABLE sentiment_inference_errors (
    error_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    inference_run_id TEXT NOT NULL,
    article_id TEXT,
    source_ingestion_run_id TEXT NOT NULL,
    model_revision_id TEXT NOT NULL,
    input_policy_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    input_text_sha256 TEXT CHECK (
        input_text_sha256 IS NULL OR length(input_text_sha256) = 64
    ),
    selected_fields_json TEXT NOT NULL DEFAULT '[]',
    original_language TEXT,
    warning_codes_json TEXT NOT NULL DEFAULT '[]',
    retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
    details_json TEXT NOT NULL DEFAULT '{}',
    CHECK (json_valid(selected_fields_json) = 1),
    CHECK (json_type(selected_fields_json) = 'array'),
    CHECK (json_valid(warning_codes_json) = 1),
    CHECK (json_type(warning_codes_json) = 'array'),
    CHECK (json_valid(details_json) = 1),
    CHECK (json_type(details_json) = 'object'),
    FOREIGN KEY (inference_run_id) REFERENCES sentiment_inference_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (article_id) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (source_ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (model_revision_id)
        REFERENCES sentiment_model_revisions(model_revision_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX ix_sentiment_inference_errors_run
ON sentiment_inference_errors(inference_run_id, error_id);

CREATE TRIGGER trg_sentiment_model_revisions_immutable_update
BEFORE UPDATE ON sentiment_model_revisions
BEGIN
    SELECT RAISE(ABORT, 'sentiment model revisions are immutable');
END;

CREATE TRIGGER trg_sentiment_model_revisions_immutable_delete
BEFORE DELETE ON sentiment_model_revisions
BEGIN
    SELECT RAISE(ABORT, 'sentiment model revisions are immutable');
END;

CREATE TRIGGER trg_sentiment_runs_insert_running
BEFORE INSERT ON sentiment_inference_runs
WHEN NEW.status <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'sentiment inference runs must start as RUNNING');
END;

CREATE TRIGGER trg_sentiment_runs_completed_source_insert
BEFORE INSERT ON sentiment_inference_runs
WHEN NOT EXISTS (
    SELECT 1 FROM ingestion_runs
    WHERE run_id = NEW.source_ingestion_run_id
      AND status IN ('PASS', 'PARTIAL')
)
BEGIN
    SELECT RAISE(ABORT, 'sentiment input requires a completed ingestion run');
END;

CREATE TRIGGER trg_sentiment_runs_protect_finalized_update
BEFORE UPDATE ON sentiment_inference_runs
WHEN OLD.status <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'finalized sentiment inference runs are immutable');
END;

CREATE TRIGGER trg_sentiment_runs_protect_finalized_delete
BEFORE DELETE ON sentiment_inference_runs
WHEN OLD.status <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'finalized sentiment inference runs are immutable');
END;

CREATE TRIGGER trg_sentiment_runs_configuration_immutable_update
BEFORE UPDATE OF schema_version, source_ingestion_run_id, model_revision_id,
                 input_policy_id, started_at, device, batch_size,
                 maximum_input_characters, maximum_sequence_length,
                 maximum_articles, timeout_seconds, artifact_directory,
                 configuration_json
ON sentiment_inference_runs
BEGIN
    SELECT RAISE(ABORT, 'sentiment inference run configuration is immutable');
END;

CREATE TRIGGER trg_sentiment_runs_final_status_shape
BEFORE UPDATE OF status ON sentiment_inference_runs
WHEN NEW.status <> 'RUNNING' AND (
    NEW.completed_at IS NULL
    OR (NEW.status = 'PASS' AND (
        EXISTS (
            SELECT 1 FROM sentiment_inference_errors
            WHERE inference_run_id = OLD.run_id
        )
        OR EXISTS (
            SELECT 1 FROM article_sentiment_outputs
            WHERE inference_run_id = OLD.run_id AND error_code IS NOT NULL
        )
    ))
    OR (NEW.status = 'PARTIAL' AND NOT (
        EXISTS (
            SELECT 1 FROM sentiment_inference_errors
            WHERE inference_run_id = OLD.run_id
        )
        OR EXISTS (
            SELECT 1 FROM article_sentiment_outputs
            WHERE inference_run_id = OLD.run_id AND error_code IS NOT NULL
        )
    ))
)
BEGIN
    SELECT RAISE(ABORT, 'sentiment final status contradicts persisted evidence');
END;

CREATE TRIGGER trg_article_sentiment_outputs_context_insert
BEFORE INSERT ON article_sentiment_outputs
WHEN NOT EXISTS (
    SELECT 1
    FROM sentiment_inference_runs AS run
    JOIN article_ingestions AS occurrence
      ON occurrence.ingestion_run_id = run.source_ingestion_run_id
     AND occurrence.canonical_article_id = NEW.article_id
    JOIN normalized_articles AS article
      ON article.article_id = occurrence.canonical_article_id
    WHERE run.run_id = NEW.inference_run_id
      AND run.status = 'RUNNING'
      AND run.source_ingestion_run_id = NEW.source_ingestion_run_id
      AND run.model_revision_id = NEW.model_revision_id
      AND run.input_policy_id = NEW.input_policy_id
      AND article.published_at IS NEW.published_at
      AND article.available_at = NEW.available_at
      AND article.fetched_at = NEW.fetched_at
      AND article.language IS NEW.original_language
      AND run.maximum_input_characters = NEW.maximum_input_characters
      AND run.maximum_sequence_length = NEW.maximum_sequence_length
      AND run.device = NEW.device
      AND run.batch_size = NEW.batch_size
)
BEGIN
    SELECT RAISE(ABORT, 'sentiment output does not match its run and article evidence');
END;

CREATE TRIGGER trg_article_sentiment_outputs_immutable_update
BEFORE UPDATE ON article_sentiment_outputs
BEGIN
    SELECT RAISE(ABORT, 'article sentiment outputs are immutable');
END;

CREATE TRIGGER trg_article_sentiment_outputs_immutable_delete
BEFORE DELETE ON article_sentiment_outputs
BEGIN
    SELECT RAISE(ABORT, 'article sentiment outputs are immutable');
END;

CREATE TRIGGER trg_sentiment_inference_errors_context_insert
BEFORE INSERT ON sentiment_inference_errors
WHEN NOT EXISTS (
    SELECT 1 FROM sentiment_inference_runs AS run
    WHERE run.run_id = NEW.inference_run_id
      AND run.status = 'RUNNING'
      AND run.source_ingestion_run_id = NEW.source_ingestion_run_id
      AND run.model_revision_id = NEW.model_revision_id
      AND run.input_policy_id = NEW.input_policy_id
      AND (
          NEW.article_id IS NULL OR EXISTS (
              SELECT 1 FROM article_ingestions AS occurrence
              WHERE occurrence.ingestion_run_id = run.source_ingestion_run_id
                AND occurrence.canonical_article_id = NEW.article_id
          )
      )
)
BEGIN
    SELECT RAISE(ABORT, 'sentiment error does not match its running context');
END;

CREATE TRIGGER trg_sentiment_inference_errors_immutable_update
BEFORE UPDATE ON sentiment_inference_errors
BEGIN
    SELECT RAISE(ABORT, 'sentiment inference errors are immutable');
END;

CREATE TRIGGER trg_sentiment_inference_errors_immutable_delete
BEFORE DELETE ON sentiment_inference_errors
BEGIN
    SELECT RAISE(ABORT, 'sentiment inference errors are immutable');
END;
"""


_SCHEMA_V6_SQL = """
-- Version 5 treated the declared repository/revision tuple as globally unique.
-- Version 6 makes the reviewed file inventory part of model identity, so an
-- unverified legacy row and a new hash-bound row must be able to coexist for
-- the same upstream revisions. The migration driver temporarily disables
-- enforcement while holding its write lock, rebuilds this parent under a
-- temporary name, validates every foreign key, and re-enables enforcement.
-- Existing run/output references therefore continue to identify the untouched
-- legacy row instead of retroactively certifying it with hashes it never had.

CREATE TABLE sentiment_model_revisions_v6 (
    model_revision_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    model_repository TEXT NOT NULL,
    model_revision TEXT NOT NULL,
    tokenizer_repository TEXT NOT NULL,
    tokenizer_revision TEXT NOT NULL,
    original_labels_json TEXT NOT NULL,
    label_mapping_json TEXT NOT NULL,
    license_identifier TEXT NOT NULL,
    library_versions_json TEXT NOT NULL,
    model_config_sha256 TEXT,
    tokenizer_config_sha256 TEXT,
    created_at TEXT NOT NULL,
    backend_identity_sha256 TEXT UNIQUE,
    snapshot_inventory_json TEXT,
    model_weights_sha256 TEXT,
    tokenizer_vocabulary_sha256 TEXT,
    tokenizer_special_tokens_sha256 TEXT,
    CHECK (json_valid(original_labels_json) = 1),
    CHECK (json_type(original_labels_json) IN ('array', 'object')),
    CHECK (json_valid(label_mapping_json) = 1),
    CHECK (json_type(label_mapping_json) = 'object'),
    CHECK (json_valid(library_versions_json) = 1),
    CHECK (json_type(library_versions_json) = 'object'),
    CHECK (
        model_config_sha256 IS NULL OR length(model_config_sha256) = 64
    ),
    CHECK (
        tokenizer_config_sha256 IS NULL OR length(tokenizer_config_sha256) = 64
    ),
    CHECK (
        backend_identity_sha256 IS NULL
        OR length(backend_identity_sha256) = 64
    ),
    CHECK (
        snapshot_inventory_json IS NULL OR (
            json_valid(snapshot_inventory_json) = 1
            AND json_type(snapshot_inventory_json) = 'array'
        )
    ),
    CHECK (
        model_weights_sha256 IS NULL OR length(model_weights_sha256) = 64
    ),
    CHECK (
        tokenizer_vocabulary_sha256 IS NULL
        OR length(tokenizer_vocabulary_sha256) = 64
    ),
    CHECK (
        tokenizer_special_tokens_sha256 IS NULL
        OR length(tokenizer_special_tokens_sha256) = 64
    )
);

INSERT INTO sentiment_model_revisions_v6(
    model_revision_id, schema_version, model_repository, model_revision,
    tokenizer_repository, tokenizer_revision, original_labels_json,
    label_mapping_json, license_identifier, library_versions_json,
    model_config_sha256, tokenizer_config_sha256, created_at
)
SELECT model_revision_id, schema_version, model_repository, model_revision,
       tokenizer_repository, tokenizer_revision, original_labels_json,
       label_mapping_json, license_identifier, library_versions_json,
       model_config_sha256, tokenizer_config_sha256, created_at
FROM sentiment_model_revisions;

DROP TABLE sentiment_model_revisions;
ALTER TABLE sentiment_model_revisions_v6 RENAME TO sentiment_model_revisions;

CREATE TRIGGER trg_sentiment_model_revisions_immutable_update
BEFORE UPDATE ON sentiment_model_revisions
BEGIN
    SELECT RAISE(ABORT, 'sentiment model revisions are immutable');
END;

CREATE TRIGGER trg_sentiment_model_revisions_immutable_delete
BEFORE DELETE ON sentiment_model_revisions
BEGIN
    SELECT RAISE(ABORT, 'sentiment model revisions are immutable');
END;

CREATE TRIGGER trg_sentiment_model_revisions_hashed_identity_insert
BEFORE INSERT ON sentiment_model_revisions
WHEN NEW.backend_identity_sha256 IS NULL
  OR NEW.snapshot_inventory_json IS NULL
  OR json_valid(NEW.snapshot_inventory_json) <> 1
  OR json_type(NEW.snapshot_inventory_json) <> 'array'
  OR json_array_length(NEW.snapshot_inventory_json) = 0
  OR NEW.model_config_sha256 IS NULL
  OR NEW.tokenizer_config_sha256 IS NULL
  OR NEW.model_weights_sha256 IS NULL
  OR NEW.tokenizer_vocabulary_sha256 IS NULL
  OR NEW.tokenizer_special_tokens_sha256 IS NULL
BEGIN
    SELECT RAISE(ABORT, 'sentiment model revision requires hashed snapshot identity');
END;

ALTER TABLE sentiment_inference_runs
ADD COLUMN sealed_payload_inventory_sha256 TEXT CHECK (
    sealed_payload_inventory_sha256 IS NULL
    OR length(sealed_payload_inventory_sha256) = 64
);

ALTER TABLE sentiment_inference_runs
ADD COLUMN run_manifest_contract_sha256 TEXT CHECK (
    run_manifest_contract_sha256 IS NULL
    OR length(run_manifest_contract_sha256) = 64
);

ALTER TABLE article_sentiment_outputs
ADD COLUMN final_token_count INTEGER CHECK (
    final_token_count IS NULL OR final_token_count > 0
);

ALTER TABLE article_sentiment_outputs
ADD COLUMN truncation_side TEXT CHECK (
    truncation_side IS NULL OR truncation_side = 'right'
);

CREATE TRIGGER trg_article_sentiment_outputs_truncation_insert
BEFORE INSERT ON article_sentiment_outputs
WHEN NEW.final_token_count IS NULL
  OR NEW.truncation_side <> 'right'
  OR NEW.final_token_count <> min(
      NEW.tokenizer_token_count, NEW.maximum_sequence_length
  )
  OR NEW.token_truncation_applied <> (
      NEW.final_token_count < NEW.tokenizer_token_count
  )
BEGIN
    SELECT RAISE(ABORT, 'sentiment output truncation evidence is invalid');
END;

CREATE TABLE sentiment_run_articles (
    inference_run_id TEXT NOT NULL,
    article_id TEXT NOT NULL,
    input_ordinal INTEGER NOT NULL CHECK (input_ordinal >= 0),
    outcome TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        outcome IN ('PENDING', 'GENERATED', 'REUSED', 'SKIPPED', 'FAILED')
    ),
    sentiment_output_id TEXT,
    error_id TEXT,
    PRIMARY KEY (inference_run_id, article_id),
    UNIQUE (inference_run_id, input_ordinal),
    CHECK (
        (outcome = 'PENDING' AND sentiment_output_id IS NULL AND error_id IS NULL)
        OR (
            outcome IN ('GENERATED', 'REUSED')
            AND sentiment_output_id IS NOT NULL AND error_id IS NULL
        )
        OR (
            outcome IN ('SKIPPED', 'FAILED')
            AND sentiment_output_id IS NULL AND error_id IS NOT NULL
        )
    ),
    FOREIGN KEY (inference_run_id) REFERENCES sentiment_inference_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (article_id) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (sentiment_output_id)
        REFERENCES article_sentiment_outputs(sentiment_output_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (error_id) REFERENCES sentiment_inference_errors(error_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
);

CREATE INDEX ix_sentiment_run_articles_output
ON sentiment_run_articles(sentiment_output_id)
WHERE sentiment_output_id IS NOT NULL;

-- Backfill the ownership relationship for finalized version-5 output.  The
-- old schema did not retain explicit reused/input associations, so migration
-- does not invent those missing facts.
INSERT INTO sentiment_run_articles(
    inference_run_id, article_id, input_ordinal, outcome, sentiment_output_id
)
SELECT output.inference_run_id, output.article_id,
       row_number() OVER (
           PARTITION BY output.inference_run_id ORDER BY output.article_id
       ) - 1,
       'GENERATED', output.sentiment_output_id
FROM article_sentiment_outputs AS output
JOIN sentiment_inference_runs AS owner
  ON owner.run_id = output.inference_run_id
WHERE owner.status IN ('PASS', 'PARTIAL')
  AND owner.completed_at IS NOT NULL;

CREATE TRIGGER trg_sentiment_run_articles_context_insert
BEFORE INSERT ON sentiment_run_articles
WHEN NEW.outcome <> 'PENDING'
  OR NOT EXISTS (
      SELECT 1
      FROM sentiment_inference_runs AS run
      JOIN article_ingestions AS occurrence
        ON occurrence.ingestion_run_id = run.source_ingestion_run_id
       AND occurrence.canonical_article_id = NEW.article_id
      WHERE run.run_id = NEW.inference_run_id
        AND run.status = 'RUNNING'
  )
BEGIN
    SELECT RAISE(ABORT, 'sentiment run input does not match running source evidence');
END;

CREATE TRIGGER trg_sentiment_run_articles_context_update
BEFORE UPDATE ON sentiment_run_articles
WHEN OLD.outcome <> 'PENDING'
  OR NEW.inference_run_id <> OLD.inference_run_id
  OR NEW.article_id <> OLD.article_id
  OR NEW.input_ordinal <> OLD.input_ordinal
  OR NOT EXISTS (
      SELECT 1 FROM sentiment_inference_runs AS run
      WHERE run.run_id = OLD.inference_run_id AND run.status = 'RUNNING'
  )
  OR (
      NEW.outcome IN ('GENERATED', 'REUSED')
      AND NOT EXISTS (
          SELECT 1
          FROM article_sentiment_outputs AS output
          JOIN sentiment_inference_runs AS consumer
            ON consumer.run_id = NEW.inference_run_id
          JOIN sentiment_inference_runs AS owner
            ON owner.run_id = output.inference_run_id
          WHERE output.sentiment_output_id = NEW.sentiment_output_id
            AND output.article_id = NEW.article_id
            AND output.source_ingestion_run_id = consumer.source_ingestion_run_id
            AND output.model_revision_id = consumer.model_revision_id
            AND output.input_policy_id = consumer.input_policy_id
            AND (
                (NEW.outcome = 'GENERATED'
                 AND output.inference_run_id = NEW.inference_run_id)
                OR
                (NEW.outcome = 'REUSED'
                 AND output.inference_run_id <> NEW.inference_run_id
                 AND owner.status IN ('PASS', 'PARTIAL')
                 AND owner.completed_at IS NOT NULL
                 AND owner.sealed_payload_inventory_sha256 IS NOT NULL
                 AND owner.run_manifest_contract_sha256 IS NOT NULL)
            )
      )
  )
  OR (
      NEW.outcome IN ('SKIPPED', 'FAILED')
      AND NOT EXISTS (
          SELECT 1 FROM sentiment_inference_errors AS error
          WHERE error.error_id = NEW.error_id
            AND error.inference_run_id = NEW.inference_run_id
            AND error.article_id = NEW.article_id
      )
  )
BEGIN
    SELECT RAISE(ABORT, 'sentiment run outcome contradicts persisted evidence');
END;

CREATE TRIGGER trg_sentiment_run_articles_immutable_delete
BEFORE DELETE ON sentiment_run_articles
WHEN COALESCE((
    SELECT status FROM sentiment_inference_runs
    WHERE run_id = OLD.inference_run_id
), '') <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'finalized sentiment run associations are immutable');
END;

DROP TRIGGER trg_article_sentiment_outputs_immutable_delete;
CREATE TRIGGER trg_article_sentiment_outputs_immutable_delete
BEFORE DELETE ON article_sentiment_outputs
WHEN COALESCE((
    SELECT status FROM sentiment_inference_runs
    WHERE run_id = OLD.inference_run_id
), '') <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'finalized article sentiment outputs are immutable');
END;

DROP TRIGGER trg_sentiment_inference_errors_immutable_delete;
CREATE TRIGGER trg_sentiment_inference_errors_immutable_delete
BEFORE DELETE ON sentiment_inference_errors
WHEN COALESCE((
    SELECT status FROM sentiment_inference_runs
    WHERE run_id = OLD.inference_run_id
), '') <> 'RUNNING'
BEGIN
    SELECT RAISE(ABORT, 'finalized sentiment inference errors are immutable');
END;

DROP TRIGGER trg_sentiment_runs_final_status_shape;
CREATE TRIGGER trg_sentiment_runs_final_status_shape
BEFORE UPDATE OF status ON sentiment_inference_runs
WHEN NEW.status <> 'RUNNING' AND (
    NEW.completed_at IS NULL
    OR json_valid(NEW.summary_json) <> 1
    OR json_type(NEW.summary_json) <> 'object'
    OR json_extract(NEW.summary_json, '$.summary_schema_version') IS NOT '2.0'
    OR json_extract(NEW.summary_json, '$.schema_version') IS NOT OLD.schema_version
    OR json_extract(NEW.summary_json, '$.run_id') IS NOT OLD.run_id
    OR json_extract(NEW.summary_json, '$.status') IS NOT NEW.status
    OR json_extract(NEW.summary_json, '$.source_ingestion_run_id')
       IS NOT OLD.source_ingestion_run_id
    OR json_extract(NEW.summary_json, '$.completed_at') IS NOT NEW.completed_at
    OR json_extract(NEW.summary_json, '$.model_repository') IS NOT (
        SELECT model_repository FROM sentiment_model_revisions
        WHERE model_revision_id = OLD.model_revision_id
    )
    OR json_extract(NEW.summary_json, '$.model_revision') IS NOT (
        SELECT model_revision FROM sentiment_model_revisions
        WHERE model_revision_id = OLD.model_revision_id
    )
    OR json_extract(NEW.summary_json, '$.tokenizer_revision') IS NOT (
        SELECT tokenizer_revision FROM sentiment_model_revisions
        WHERE model_revision_id = OLD.model_revision_id
    )
    OR json_extract(NEW.summary_json, '$.device') IS NOT OLD.device
    OR json_extract(NEW.summary_json, '$.artifact_directory')
       IS NOT OLD.artifact_directory
    OR json_extract(NEW.summary_json, '$.academic_use_only') IS NOT 1
    OR json_extract(NEW.summary_json, '$.financial_advice') IS NOT 0
    OR json_extract(NEW.summary_json, '$.any_articles_failed') IS NOT (
        EXISTS (
            SELECT 1 FROM sentiment_run_articles
            WHERE inference_run_id = OLD.run_id AND outcome = 'FAILED'
        )
        OR EXISTS (
            SELECT 1 FROM sentiment_inference_errors
            WHERE inference_run_id = OLD.run_id AND article_id IS NULL
        )
        OR EXISTS (
            SELECT 1 FROM article_sentiment_outputs
            WHERE inference_run_id = OLD.run_id AND error_code IS NOT NULL
        )
    )
    OR (
        NEW.status = 'FAIL'
        AND EXISTS (
            SELECT 1 FROM article_sentiment_outputs
            WHERE inference_run_id = OLD.run_id
        )
    )
    OR (
        NEW.status IN ('PASS', 'PARTIAL')
        AND (
            NEW.sealed_payload_inventory_sha256 IS NULL
            OR NEW.run_manifest_contract_sha256 IS NULL
            OR json_extract(
                NEW.summary_json, '$.sealed_payload_inventory_sha256'
            ) IS NOT NEW.sealed_payload_inventory_sha256
            OR json_extract(
                NEW.summary_json, '$.run_manifest_contract_sha256'
            ) IS NOT NEW.run_manifest_contract_sha256
            OR (SELECT count(*) FROM sentiment_run_articles
                WHERE inference_run_id = OLD.run_id) = 0
            OR EXISTS (
                SELECT 1 FROM sentiment_run_articles
                WHERE inference_run_id = OLD.run_id AND outcome = 'PENDING'
            )
            OR json_extract(NEW.summary_json, '$.articles_requested') IS NOT
               (SELECT count(*) FROM sentiment_run_articles
                WHERE inference_run_id = OLD.run_id)
            OR json_extract(NEW.summary_json, '$.unique_canonical_articles') IS NOT
               (SELECT count(*) FROM sentiment_run_articles
                WHERE inference_run_id = OLD.run_id)
            OR json_extract(
                NEW.summary_json, '$.articles_successfully_scored'
            ) IS NOT (SELECT count(*) FROM sentiment_run_articles
                  WHERE inference_run_id = OLD.run_id
                    AND outcome = 'GENERATED')
            OR json_extract(NEW.summary_json, '$.articles_skipped') IS NOT
               (SELECT count(*) FROM sentiment_run_articles
                WHERE inference_run_id = OLD.run_id
                  AND outcome IN ('REUSED', 'SKIPPED'))
            OR json_extract(NEW.summary_json, '$.articles_failed') IS NOT
               (SELECT count(*) FROM sentiment_run_articles
                WHERE inference_run_id = OLD.run_id AND outcome = 'FAILED')
            OR json_extract(NEW.summary_json, '$.truncated_input_count') IS NOT
               (SELECT count(*)
                FROM sentiment_run_articles AS input
                JOIN article_sentiment_outputs AS output
                  ON output.sentiment_output_id = input.sentiment_output_id
                WHERE input.inference_run_id = OLD.run_id
                  AND input.outcome = 'GENERATED'
                  AND output.truncation_applied = 1)
            OR json_extract(
                NEW.summary_json, '$.ogdc_linked_articles_scored'
            ) IS NOT (
                SELECT count(*)
                FROM sentiment_run_articles AS input
                WHERE input.inference_run_id = OLD.run_id
                  AND input.outcome = 'GENERATED'
                  AND EXISTS (
                      SELECT 1
                      FROM article_ingestions AS occurrence
                      JOIN article_company_links AS link
                        ON link.occurrence_id = occurrence.occurrence_id
                      WHERE occurrence.ingestion_run_id = OLD.source_ingestion_run_id
                        AND occurrence.canonical_article_id = input.article_id
                        AND upper(link.ticker) = 'OGDC'
                  )
            )
            OR json_extract(NEW.summary_json, '$.label_counts.negative') IS NOT
               (SELECT count(*)
                FROM sentiment_run_articles AS input
                JOIN article_sentiment_outputs AS output
                  ON output.sentiment_output_id = input.sentiment_output_id
                WHERE input.inference_run_id = OLD.run_id
                  AND input.outcome = 'GENERATED'
                  AND output.predicted_label = 'negative')
            OR json_extract(NEW.summary_json, '$.label_counts.neutral') IS NOT
               (SELECT count(*)
                FROM sentiment_run_articles AS input
                JOIN article_sentiment_outputs AS output
                  ON output.sentiment_output_id = input.sentiment_output_id
                WHERE input.inference_run_id = OLD.run_id
                  AND input.outcome = 'GENERATED'
                  AND output.predicted_label = 'neutral')
            OR json_extract(NEW.summary_json, '$.label_counts.positive') IS NOT
               (SELECT count(*)
                FROM sentiment_run_articles AS input
                JOIN article_sentiment_outputs AS output
                  ON output.sentiment_output_id = input.sentiment_output_id
                WHERE input.inference_run_id = OLD.run_id
                  AND input.outcome = 'GENERATED'
                  AND output.predicted_label = 'positive')
            OR (
                NEW.status = 'PASS'
                AND (
                    EXISTS (
                        SELECT 1 FROM sentiment_inference_errors
                        WHERE inference_run_id = OLD.run_id
                    )
                    OR EXISTS (
                        SELECT 1 FROM sentiment_run_articles
                        WHERE inference_run_id = OLD.run_id
                          AND outcome IN ('SKIPPED', 'FAILED')
                    )
                )
            )
            OR (
                NEW.status = 'PARTIAL'
                AND NOT EXISTS (
                    SELECT 1 FROM sentiment_inference_errors
                    WHERE inference_run_id = OLD.run_id
                )
            )
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'sentiment final status contradicts certified evidence');
END;
"""


_SCHEMA_V7_SQL = """
-- Provider observation time is audit provenance, not cutoff eligibility.
-- Legacy rows remain explicitly unknown (NULL); no historical value is
-- inferred from available_at or publication time during migration.
ALTER TABLE normalized_articles
ADD COLUMN source_observed_at TEXT CHECK (
    source_observed_at IS NULL OR julianday(source_observed_at) IS NOT NULL
);

ALTER TABLE article_ingestions
ADD COLUMN source_observed_at TEXT CHECK (
    source_observed_at IS NULL OR julianday(source_observed_at) IS NOT NULL
);
"""


# The retained PR #12 database was produced by a short-lived version-3 schema
# immediately before the final constraint hardening landed.  Its object set,
# indexes, and triggers match version 3 exactly; only these three table
# definitions lack constraints that the already-valid retained rows satisfy.
# Keep the signatures literal so initialization cannot bless arbitrary drift.
_RECOGNIZED_LEGACY_V3_TABLE_SQL = {
    "normalized_articles": """
CREATE TABLE normalized_articles (
    article_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    provider TEXT NOT NULL,
    source_name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_url TEXT,
    canonical_url TEXT,
    title TEXT,
    normalized_title TEXT NOT NULL DEFAULT '',
    summary TEXT,
    body TEXT,
    published_at TEXT,
    published_timezone TEXT,
    available_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    language TEXT,
    mentioned_tickers_json TEXT NOT NULL DEFAULT '[]',
    mentioned_companies_json TEXT NOT NULL DEFAULT '[]',
    mentioned_sectors_json TEXT NOT NULL DEFAULT '[]',
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    duplicate_of TEXT,
    first_ingestion_run_id TEXT NOT NULL,
    raw_artifact_reference TEXT NOT NULL,
    normalization_warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    CHECK (duplicate_of IS NULL OR duplicate_of <> article_id),
    FOREIGN KEY (duplicate_of) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (first_ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)
""",
    "article_ingestions": """
CREATE TABLE article_ingestions (
    occurrence_id TEXT PRIMARY KEY,
    ingestion_run_id TEXT NOT NULL,
    raw_evidence_id TEXT,
    candidate_article_id TEXT NOT NULL,
    canonical_article_id TEXT NOT NULL,
    duplicate_of TEXT,
    deduplication_rule TEXT,
    available_at TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    raw_artifact_reference TEXT NOT NULL,
    normalized_payload_json TEXT NOT NULL,
    normalization_warnings_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    CHECK (
        (duplicate_of IS NULL AND candidate_article_id = canonical_article_id)
        OR duplicate_of = canonical_article_id
    ),
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (raw_evidence_id) REFERENCES raw_source_records(raw_evidence_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (canonical_article_id) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    FOREIGN KEY (duplicate_of) REFERENCES normalized_articles(article_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT,
    UNIQUE (ingestion_run_id, candidate_article_id)
)
""",
    "quality_reports": """
CREATE TABLE quality_reports (
    ingestion_run_id TEXT PRIMARY KEY,
    schema_version TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    report_json TEXT NOT NULL,
    FOREIGN KEY (ingestion_run_id) REFERENCES ingestion_runs(run_id)
        ON UPDATE RESTRICT ON DELETE RESTRICT
)
""",
}


def utc_now_z() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="auto").replace(
        "+00:00", "Z"
    )


def _record_mapping(record: Any | None, values: Mapping[str, Any]) -> dict[str, Any]:
    if record is None:
        result: dict[str, Any] = {}
    elif isinstance(record, Mapping):
        result = dict(record)
    elif is_dataclass(record) and not isinstance(record, type):
        result = asdict(record)
    elif hasattr(record, "to_dict"):
        converted = record.to_dict()
        if not isinstance(converted, Mapping):
            raise TypeError("to_dict() must return a mapping")
        result = dict(converted)
    elif hasattr(record, "__dict__"):
        result = vars(record).copy()
    else:
        raise TypeError("record must be a mapping or mapping-like dataclass")
    result.update(values)
    return result


def _value(data: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in data:
            return data[name]
    return default


def _required_text(data: Mapping[str, Any], *names: str) -> str:
    value = _value(data, *names)
    text = "" if value is None else str(value).strip()
    if not text:
        raise ValueError(f"Required field is blank: {names[0]}")
    return text


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _news_record_schema(
    value: Any, *, source_observed_at: str | None = None
) -> str:
    schema_version = str(value or "").strip()
    if schema_version not in _SUPPORTED_NEWS_RECORD_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported news record schema: {schema_version}")
    if schema_version == "1.0" and source_observed_at is not None:
        raise ValueError("source_observed_at requires news record schema 1.1")
    return schema_version


def _normalized_payload_for_schema(value: Any, *, schema_version: str) -> Any:
    """Return serialized normalized evidence in its declared record schema."""

    if schema_version != "1.0" or not isinstance(value, Mapping):
        return value
    payload = dict(value)
    if payload.get("source_observed_at") is not None:
        raise ValueError("source_observed_at requires news record schema 1.1")
    payload.pop("source_observed_at", None)
    return payload


def _positive_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if number <= 0 or (isinstance(value, float) and not value.is_integer()):
        raise ValueError(f"{field} must be a positive integer")
    return number


def _optional_positive_integer(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, field=field)


def _optional_positive_number(value: Any, *, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive number") from exc
    if number <= 0:
        raise ValueError(f"{field} must be a positive number")
    return number


def _boolean_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if value in {0, 1}:
        return int(value)
    raise ValueError(f"{field} must be boolean")


def _probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a probability")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a probability") from exc
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be between zero and one")
    return number


def _deduplication_rule(value: Any, *, nullable: bool = False) -> str | None:
    rule = _optional_text(value)
    if rule is None:
        if nullable:
            return None
        raise ValueError("deduplication_rule is required")
    if rule not in _SUPPORTED_DEDUPLICATION_RULES:
        supported = ", ".join(sorted(_SUPPORTED_DEDUPLICATION_RULES))
        raise ValueError(f"deduplication_rule must be one of: {supported}")
    return rule


def _safe_id(value: Any, *, field: str) -> str:
    identifier = "" if value is None else str(value).strip()
    if not _SAFE_ID.fullmatch(identifier):
        raise ValueError(
            f"{field} must start with a letter or digit and contain only letters, "
            "digits, '.', '_', ':', and '-'"
        )
    return identifier


def _sha256(value: Any, *, field: str) -> str:
    digest = "" if value is None else str(value).strip().lower()
    if not _SHA256.fullmatch(digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _timestamp(value: Any, *, field: str, nullable: bool = False) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        if nullable:
            return None
        raise ValueError(f"{field} is required")
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include timezone evidence")
    return parsed.astimezone(timezone.utc).isoformat(timespec="auto").replace(
        "+00:00", "Z"
    )


def _parsed_utc_timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO 8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include timezone evidence")
    return parsed.astimezone(timezone.utc)


def _validate_availability(
    *,
    available_at: str,
    fetched_at: str,
    published_at: str | None = None,
) -> None:
    available = _parsed_utc_timestamp(available_at, field="available_at")
    fetched = _parsed_utc_timestamp(fetched_at, field="fetched_at")
    if available < fetched:
        raise ValueError("available_at cannot precede fetched_at")
    if published_at is not None:
        published = _parsed_utc_timestamp(published_at, field="published_at")
        if available < published:
            raise ValueError("available_at cannot precede published_at")


def _artifact_reference(value: Any, *, field: str) -> str:
    raw = "" if value is None else str(value).strip()
    if not raw:
        raise ValueError(f"{field} is required")
    if "\\" in raw or "://" in raw:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw
        or path.parts[0].endswith(":")
    ):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return path.as_posix()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return _timestamp(value, field="JSON timestamp")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _json_text(value: Any, *, default: Any) -> str:
    payload = default if value is None else value
    return json.dumps(
        _jsonable(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_sha256(value: Any, *, default: Any) -> tuple[str, str]:
    payload = _json_text(value, default=default)
    return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sealed_inventory(value: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("artifact_inventory must be a sequence of mappings")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in value:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("every artifact inventory entry must be a mapping")
        path = _artifact_reference(raw_entry.get("path"), field="artifact path")
        if path in seen:
            raise ValueError(f"artifact_inventory contains a duplicate path: {path}")
        seen.add(path)
        size = raw_entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("artifact inventory size_bytes must be a non-negative integer")
        entries.append(
            {
                "path": path,
                "sha256": _sha256(
                    raw_entry.get("sha256"), field="artifact inventory sha256"
                ),
                "size_bytes": size,
            }
        )
    if not entries:
        raise ValueError("artifact_inventory cannot be empty")
    return sorted(entries, key=lambda item: item["path"])


def _model_snapshot_inventory(
    value: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize the reviewed model file identity without requiring a download."""

    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError("snapshot_inventory must be a sequence of mappings")
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_entry in value:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("every snapshot inventory entry must be a mapping")
        path = _artifact_reference(raw_entry.get("path"), field="snapshot path")
        if len(PurePosixPath(path).parts) != 1:
            raise ValueError("snapshot inventory entries must be file names")
        if path in seen:
            raise ValueError(f"snapshot_inventory contains a duplicate path: {path}")
        seen.add(path)
        entry: dict[str, Any] = {
            "path": path,
            "sha256": _sha256(
                raw_entry.get("sha256"), field="snapshot inventory sha256"
            ),
        }
        size = raw_entry.get("size_bytes")
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise ValueError(
                    "snapshot inventory size_bytes must be a non-negative integer"
                )
            entry["size_bytes"] = size
        entries.append(entry)
    if not entries:
        raise ValueError("snapshot_inventory cannot be empty")
    return sorted(entries, key=lambda item: item["path"])


def _normalized_title(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _deterministic_id(prefix: str, *parts: Any) -> str:
    payload = "\0".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _validate_existing_deduplication_integrity(
    connection: sqlite3.Connection,
) -> None:
    supported_rules = tuple(sorted(_SUPPORTED_DEDUPLICATION_RULES))
    placeholders = ", ".join("?" for _ in supported_rules)
    unsupported_occurrence = connection.execute(
        f"""
        SELECT occurrence_id FROM article_ingestions
        WHERE deduplication_rule IS NOT NULL
          AND deduplication_rule NOT IN ({placeholders})
        LIMIT 1
        """,
        supported_rules,
    ).fetchone()
    if unsupported_occurrence is not None:
        raise RuntimeError(
            "Cannot harden news storage while article ingestions contain an "
            "unsupported deduplication rule: "
            f"{unsupported_occurrence['occurrence_id']}"
        )

    invalid_relationship = connection.execute(
        f"""
        SELECT relationship.relationship_id
        FROM deduplication_relationships AS relationship
        LEFT JOIN article_ingestions AS occurrence
          ON occurrence.occurrence_id = relationship.occurrence_id
        WHERE relationship.deduplication_rule NOT IN ({placeholders})
           OR occurrence.occurrence_id IS NULL
           OR occurrence.ingestion_run_id <> relationship.ingestion_run_id
           OR occurrence.duplicate_of IS NULL
           OR occurrence.candidate_article_id <>
              relationship.duplicate_candidate_article_id
           OR occurrence.canonical_article_id <>
              relationship.canonical_article_id
           OR occurrence.duplicate_of <> relationship.duplicate_of
           OR occurrence.deduplication_rule IS NOT
              relationship.deduplication_rule
        LIMIT 1
        """,
        supported_rules,
    ).fetchone()
    if invalid_relationship is not None:
        raise RuntimeError(
            "Cannot harden news storage while a deduplication relationship "
            "does not exactly match its occurrence: "
            f"{invalid_relationship['relationship_id']}"
        )


_MIGRATION_NAMES = {
    1: "news_ingestion_foundation",
    2: "news_storage_immutability",
    3: "news_storage_cross_run_integrity",
    4: "news_storage_sealed_finalization",
    5: "financial_sentiment_foundation",
    6: "sentiment_run_certification",
    7: "source_observation_provenance",
}


def _normalized_schema_sql(value: str | None) -> str:
    normalized = " ".join((value or "").split()).casefold()
    # SQLite quotes the destination table name after ALTER TABLE ... RENAME.
    # Treat that cosmetic rewrite as equivalent while continuing to compare
    # every column, constraint, index, and trigger token exactly.
    return re.sub(
        r'^(create table(?: if not exists)?)\s+"([a-z0-9_]+)"',
        r"\1 \2",
        normalized,
    )


def _schema_stage_sql(version: int) -> str:
    if version == 1:
        return _SCHEMA_SQL
    elif version == 2:
        return _IMMUTABILITY_SQL
    elif version == 3:
        return _DEDUPLICATION_INTEGRITY_SQL
    elif version == 4:
        return _SCHEMA_V4_SQL
    elif version == 5:
        return _SCHEMA_V5_SQL
    elif version == 6:
        return _SCHEMA_V6_SQL
    elif version == 7:
        return _SCHEMA_V7_SQL
    raise ValueError(f"Unsupported news schema stage: {version}")


def _apply_schema_stage(
    connection: sqlite3.Connection, version: int, *, begin: bool = False
) -> None:
    # ``Connection.executescript`` may implicitly commit an already-open
    # transaction before executing its script.  That behaviour would release a
    # migration's write lock between validation and the schema mutation.  Feed
    # complete SQLite statements to ``execute`` instead so a caller-acquired
    # ``BEGIN IMMEDIATE`` remains in force for the whole stage.  In particular,
    # ``sqlite3.complete_statement`` keeps trigger bodies together despite the
    # semicolons inside ``BEGIN ... END``.
    if begin:
        connection.execute("BEGIN IMMEDIATE")
    pending: list[str] = []
    for line in _schema_stage_sql(version).splitlines(keepends=True):
        pending.append(line)
        statement = "".join(pending)
        if sqlite3.complete_statement(statement):
            if statement.strip():
                connection.execute(statement)
            pending.clear()
    if "".join(pending).strip():
        raise RuntimeError(
            f"News schema stage {version} ended with an incomplete SQL statement"
        )


def _expected_schema_objects(version: int) -> dict[tuple[str, str], str]:
    expected = sqlite3.connect(":memory:")
    try:
        expected.execute("PRAGMA foreign_keys = ON")
        for stage in range(1, version + 1):
            _apply_schema_stage(expected, stage)
        return {
            (str(row[0]), str(row[1])): _normalized_schema_sql(row[2])
            for row in expected.execute(
                """
                SELECT type, name, sql
                FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger')
                  AND name NOT LIKE 'sqlite_%'
                """
            )
        }
    finally:
        expected.close()


def _expected_schema_definitions(
    version: int,
) -> list[tuple[str, str, str, str]]:
    """Return raw expected schema definitions for a verified migration stage."""

    expected = sqlite3.connect(":memory:")
    try:
        expected.execute("PRAGMA foreign_keys = ON")
        for stage in range(1, version + 1):
            _apply_schema_stage(expected, stage)
        return [
            (str(row[0]), str(row[1]), str(row[2]), str(row[3]))
            for row in expected.execute(
                """
                SELECT type, name, tbl_name, sql
                FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger')
                  AND name NOT LIKE 'sqlite_%'
                  AND sql IS NOT NULL
                ORDER BY type, name
                """
            )
        ]
    finally:
        expected.close()


def _is_recognized_legacy_v3_schema(connection: sqlite3.Connection) -> bool:
    expected = _expected_schema_objects(3)
    actual = {
        (str(row[0]), str(row[1])): _normalized_schema_sql(row[2])
        for row in connection.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE type IN ('table', 'index', 'trigger')
              AND name NOT LIKE 'sqlite_%'
            """
        )
    }
    if set(actual) != set(expected):
        return False
    changed = {
        key for key in expected if expected[key] != actual[key]
    }
    recognized = {
        ("table", "normalized_articles"),
        ("table", "article_ingestions"),
        ("table", "quality_reports"),
    }
    if changed != recognized:
        return False
    return all(
        actual[("table", table)] == _normalized_schema_sql(sql)
        for table, sql in _RECOGNIZED_LEGACY_V3_TABLE_SQL.items()
    )


def _repair_recognized_legacy_v3_schema(connection: sqlite3.Connection) -> None:
    """Rebuild only the exact retained pre-hardening v3 table signatures."""

    targets = tuple(_RECOGNIZED_LEGACY_V3_TABLE_SQL)
    definitions = _expected_schema_definitions(3)
    expected_tables = {
        name: sql
        for object_type, name, _table, sql in definitions
        if object_type == "table" and name in targets
    }
    secondary = [
        sql
        for object_type, _name, table, sql in definitions
        if object_type == "trigger" or (object_type == "index" and table in targets)
    ]
    trigger_names = [
        name
        for object_type, name, _table, _sql in definitions
        if object_type == "trigger"
    ]

    connection.execute("PRAGMA foreign_keys = OFF")
    try:
        connection.execute("BEGIN IMMEDIATE")
        # Both the structural signature and every data precondition must be
        # evaluated after the write lock is held.  Otherwise a concurrent
        # writer can invalidate the checked state before the rebuilt tables are
        # installed and the migration is stamped successful.
        if not _is_recognized_legacy_v3_schema(connection):
            raise RuntimeError("Refusing to repair an unrecognized news schema")

        invalid_article = connection.execute(
            """
            SELECT article_id FROM normalized_articles
            WHERE julianday(available_at) IS NULL
               OR julianday(fetched_at) IS NULL
               OR julianday(available_at) < julianday(fetched_at)
               OR (
                   published_at IS NOT NULL AND (
                       julianday(published_at) IS NULL
                       OR julianday(available_at) < julianday(published_at)
                   )
               )
            LIMIT 1
            """
        ).fetchone()
        invalid_occurrence = connection.execute(
            """
            SELECT occurrence_id FROM article_ingestions
            WHERE raw_evidence_id IS NULL
               OR julianday(available_at) IS NULL
               OR julianday(fetched_at) IS NULL
               OR julianday(available_at) < julianday(fetched_at)
            LIMIT 1
            """
        ).fetchone()
        invalid_quality = connection.execute(
            """
            SELECT ingestion_run_id FROM quality_reports
            WHERE quality_status NOT IN ('PASS', 'WARN', 'FAIL')
            LIMIT 1
            """
        ).fetchone()
        if (
            invalid_article is not None
            or invalid_occurrence is not None
            or invalid_quality is not None
        ):
            raise RuntimeError(
                "Recognized legacy news schema contains rows that violate "
                "the hardened contract"
            )

        for trigger in trigger_names:
            connection.execute(f"DROP TRIGGER {trigger}")
        for table in targets:
            repaired = f"{table}__v3_repaired"
            create_sql = expected_tables[table].replace(
                f"CREATE TABLE {table}", f"CREATE TABLE {repaired}", 1
            )
            connection.execute(create_sql)
            columns = [
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})")
            ]
            column_sql = ", ".join(columns)
            connection.execute(
                f"INSERT INTO {repaired} ({column_sql}) "
                f"SELECT {column_sql} FROM {table}"
            )

        # Restore each public name immediately. SQLite reparses surviving
        # cross-table triggers during ALTER TABLE and therefore must never see
        # all three public names absent at the same time.
        for table in ("normalized_articles", "article_ingestions", "quality_reports"):
            connection.execute(f"DROP TABLE {table}")
            connection.execute(
                f"ALTER TABLE {table}__v3_repaired RENAME TO {table}"
            )
        for sql in secondary:
            connection.execute(sql)

        foreign_key_error = connection.execute("PRAGMA foreign_key_check").fetchone()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if foreign_key_error is not None or integrity is None or integrity[0] != "ok":
            raise RuntimeError("Legacy news schema repair failed integrity validation")
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.execute("PRAGMA foreign_keys = ON")


def _verify_migration_ledger(
    connection: sqlite3.Connection, version: int
) -> None:
    rows = connection.execute(
        "SELECT version, name FROM schema_migrations ORDER BY version"
    ).fetchall()
    actual = [(int(row[0]), str(row[1])) for row in rows]
    expected = [
        (item, _MIGRATION_NAMES[item]) for item in range(1, version + 1)
    ]
    if actual != expected:
        raise RuntimeError(
            "News database migration ledger does not match its schema version"
        )


def _verify_schema(connection: sqlite3.Connection, version: int) -> None:
    actual_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if actual_version != version:
        raise RuntimeError(
            "News database user_version does not match the expected migration stage"
        )
    expected_objects = _expected_schema_objects(version)
    actual_objects = {
        (str(row[0]), str(row[1])): _normalized_schema_sql(row[2])
        for row in connection.execute(
            """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE type IN ('table', 'index', 'trigger')
              AND name NOT LIKE 'sqlite_%'
            """
        )
    }
    missing = sorted(set(expected_objects) - set(actual_objects))
    unexpected = sorted(set(actual_objects) - set(expected_objects))
    changed = sorted(
        key
        for key in set(expected_objects) & set(actual_objects)
        if expected_objects[key] != actual_objects[key]
    )
    if missing or unexpected or changed:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if changed:
            details.append(f"changed={changed}")
        raise RuntimeError(
            "News database schema verification failed: " + "; ".join(details)
        )
    _verify_migration_ledger(connection, version)


def _validate_v4_migration_preconditions(
    connection: sqlite3.Connection,
) -> None:
    if connection.execute(
        "SELECT count(*) FROM ingestion_runs WHERE status = 'RUNNING'"
    ).fetchone()[0] > 1:
        raise RuntimeError(
            "Cannot serialize news ingestion while multiple RUNNING rows exist"
        )
    noncanonical = connection.execute(
        """
        SELECT article_id FROM normalized_articles
        WHERE duplicate_of IS NOT NULL LIMIT 1
        """
    ).fetchone()
    if noncanonical is not None:
        raise RuntimeError(
            "Cannot migrate news storage while normalized_articles contains "
            f"a noncanonical row: {noncanonical['article_id']}"
        )
    duplicate_hash = connection.execute(
        """
        SELECT content_sha256 FROM normalized_articles
        GROUP BY content_sha256 HAVING count(*) > 1 LIMIT 1
        """
    ).fetchone()
    if duplicate_hash is not None:
        raise RuntimeError(
            "Cannot migrate news storage with duplicate canonical content hashes"
        )
    duplicate_title = connection.execute(
        """
        SELECT normalized_title, source_name,
               strftime('%Y-%m-%dT%H:%M', published_at) AS published_minute
        FROM normalized_articles
        WHERE normalized_title <> '' AND published_at IS NOT NULL
        GROUP BY normalized_title, source_name, published_minute
        HAVING count(*) > 1 LIMIT 1
        """
    ).fetchone()
    if duplicate_title is not None:
        raise RuntimeError(
            "Cannot migrate news storage with duplicate canonical title identities"
        )
    unsafe_target = connection.execute(
        """
        SELECT occurrence.occurrence_id
        FROM article_ingestions AS occurrence
        JOIN normalized_articles AS canonical
          ON canonical.article_id = occurrence.canonical_article_id
        JOIN ingestion_runs AS owner
          ON owner.run_id = canonical.first_ingestion_run_id
        WHERE owner.run_id <> occurrence.ingestion_run_id
          AND owner.status NOT IN ('PASS', 'PARTIAL')
        LIMIT 1
        """
    ).fetchone()
    if unsafe_target is not None:
        raise RuntimeError(
            "Cannot migrate news storage while an occurrence references an "
            f"unfinalized canonical: {unsafe_target['occurrence_id']}"
        )


def _validate_v6_migration_preconditions(
    connection: sqlite3.Connection,
) -> None:
    """Reject legacy failed output that cannot remain safely idempotent.

    Version 5 used a global article/model/policy uniqueness key.  A finalized
    failed owner cannot be discarded, and version 6 deliberately refuses to
    treat failed output as reusable evidence.  Preserve that database for
    audit, but fail closed instead of carrying a permanently poisoned key into
    the certified schema.
    """

    failed_output = connection.execute(
        """
        SELECT output.sentiment_output_id
        FROM article_sentiment_outputs AS output
        JOIN sentiment_inference_runs AS owner
          ON owner.run_id = output.inference_run_id
        WHERE owner.status = 'FAIL'
        LIMIT 1
        """
    ).fetchone()
    if failed_output is not None:
        raise RuntimeError(
            "Cannot migrate version-5 sentiment output owned by a failed run: "
            f"{failed_output['sentiment_output_id']}"
        )


def _migrate_schema(
    connection: sqlite3.Connection, from_version: int, to_version: int
) -> None:
    if to_version != from_version + 1:
        raise ValueError("News schema migrations must advance one version at a time")
    rebuilds_model_parent = to_version == 6
    if rebuilds_model_parent:
        # SQLite cannot drop/recreate a referenced parent while enforcement is
        # enabled. Disable it only before the transaction, then hold the write
        # lock across source verification, the rebuild, and an explicit full
        # foreign-key check. No other writer can enter the unchecked window.
        connection.execute("PRAGMA foreign_keys = OFF")
    try:
        # Lock before reading any schema or data precondition.  The verified
        # source state, DDL, migration ledger, and user_version now form one
        # serializable transaction.
        connection.execute("BEGIN IMMEDIATE")
        _verify_schema(connection, from_version)
        if to_version == 3:
            _validate_existing_deduplication_integrity(connection)
        if to_version == 4:
            _validate_existing_deduplication_integrity(connection)
            _validate_v4_migration_preconditions(connection)
        if to_version == 6:
            _validate_v6_migration_preconditions(connection)
        _apply_schema_stage(connection, to_version)
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, applied_at)
            VALUES (?, ?, ?)
            """,
            (to_version, _MIGRATION_NAMES[to_version], utc_now_z()),
        )
        connection.execute(f"PRAGMA user_version = {to_version}")
        _verify_schema(connection, to_version)
        if rebuilds_model_parent:
            foreign_key_error = connection.execute(
                "PRAGMA foreign_key_check"
            ).fetchone()
            if foreign_key_error is not None:
                raise RuntimeError(
                    "News schema migration produced invalid foreign-key evidence"
                )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    finally:
        if rebuilds_model_parent:
            connection.execute("PRAGMA foreign_keys = ON")


def _bootstrap_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.execute("BEGIN IMMEDIATE")
        if int(connection.execute("PRAGMA user_version").fetchone()[0]) != 0:
            raise RuntimeError(
                "Refusing to bootstrap a concurrently versioned news database"
            )
        existing = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            """
        ).fetchall()
        if existing:
            raise RuntimeError(
                "Refusing to stamp an unversioned, nonempty news database"
            )
        _apply_schema_stage(connection, 1)
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, applied_at)
            VALUES (1, ?, ?)
            """,
            (_MIGRATION_NAMES[1], utc_now_z()),
        )
        connection.execute("PRAGMA user_version = 1")
        _verify_schema(connection, 1)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _run_evidence_snapshot(
    connection: sqlite3.Connection, run_id: str
) -> dict[str, Any]:
    def count(query: str, parameters: tuple[Any, ...] = (run_id,)) -> int:
        return int(connection.execute(query, parameters).fetchone()[0])

    raw_count = count(
        "SELECT count(*) FROM raw_source_records WHERE ingestion_run_id = ?"
    )
    occurrence_count = count(
        "SELECT count(*) FROM article_ingestions WHERE ingestion_run_id = ?"
    )
    new_canonical_count = count(
        "SELECT count(*) FROM normalized_articles WHERE first_ingestion_run_id = ?"
    )
    relationship_count = count(
        """
        SELECT count(*) FROM deduplication_relationships
        WHERE ingestion_run_id = ?
        """
    )
    duplicate_occurrence_count = count(
        """
        SELECT count(*) FROM article_ingestions
        WHERE ingestion_run_id = ? AND duplicate_of IS NOT NULL
        """
    )
    represented_canonical_count = count(
        """
        SELECT count(DISTINCT canonical_article_id) FROM article_ingestions
        WHERE ingestion_run_id = ?
        """
    )
    historical_duplicate_count = count(
        """
        SELECT count(*)
        FROM article_ingestions AS occurrence
        JOIN normalized_articles AS canonical
          ON canonical.article_id = occurrence.canonical_article_id
        WHERE occurrence.ingestion_run_id = ?
          AND occurrence.duplicate_of IS NOT NULL
          AND canonical.first_ingestion_run_id <> occurrence.ingestion_run_id
        """
    )
    within_run_duplicate_count = duplicate_occurrence_count - historical_duplicate_count
    provider_error_count = count(
        "SELECT count(*) FROM provider_errors WHERE ingestion_run_id = ?"
    )
    ogdc_linked_count = count(
        """
        SELECT count(DISTINCT link.occurrence_id)
        FROM article_company_links AS link
        JOIN article_ingestions AS occurrence
          ON occurrence.occurrence_id = link.occurrence_id
        WHERE occurrence.ingestion_run_id = ? AND upper(link.ticker) = 'OGDC'
        """
    )
    mismatch_count = count(
        """
        SELECT count(*)
        FROM article_ingestions AS occurrence
        LEFT JOIN deduplication_relationships AS relationship
          ON relationship.occurrence_id = occurrence.occurrence_id
        WHERE occurrence.ingestion_run_id = ?
          AND (
              (occurrence.duplicate_of IS NULL AND relationship.occurrence_id IS NOT NULL)
              OR (occurrence.duplicate_of IS NOT NULL AND relationship.occurrence_id IS NULL)
          )
        """
    )
    relationship_rule_counts = {
        rule: 0 for rule in sorted(_SUPPORTED_DEDUPLICATION_RULES)
    }
    relationship_rule_counts.update(
        {
            str(row[0]): int(row[1])
            for row in connection.execute(
                """
                SELECT deduplication_rule, count(*)
                FROM deduplication_relationships
                WHERE ingestion_run_id = ?
                GROUP BY deduplication_rule ORDER BY deduplication_rule
                """,
                (run_id,),
            )
        }
    )

    missing_title_count = 0
    missing_publication_time_count = 0
    missing_body_count = 0
    invalid_url_count = 0
    for row in connection.execute(
        """
        SELECT normalized_payload_json, normalization_warnings_json
        FROM article_ingestions WHERE ingestion_run_id = ?
        """,
        (run_id,),
    ):
        try:
            payload = json.loads(str(row[0]))
            warnings = json.loads(str(row[1]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "Stored normalized occurrence JSON is unreadable"
            ) from exc
        if not isinstance(payload, Mapping) or not isinstance(warnings, list):
            raise RuntimeError("Stored normalized occurrence JSON has invalid shape")
        title = payload.get("title")
        body = payload.get("body")
        if title is None or not str(title).strip():
            missing_title_count += 1
        if payload.get("published_at") in (None, ""):
            missing_publication_time_count += 1
        if body is None or not str(body).strip():
            missing_body_count += 1
        if any(str(item) == "source_url_invalid" for item in warnings):
            invalid_url_count += 1

    rejected_record_count = raw_count - occurrence_count
    if rejected_record_count < 0:
        raise RuntimeError(
            "News evidence contains more normalized occurrences than raw records"
        )
    if raw_count == 0 or occurrence_count == 0:
        expected_quality_status = "FAIL"
    elif (
        provider_error_count
        or rejected_record_count
        or missing_title_count
        or missing_publication_time_count
        or invalid_url_count
    ):
        expected_quality_status = "WARN"
    else:
        expected_quality_status = "PASS"

    return {
        "raw_record_count": raw_count,
        "normalized_occurrence_count": occurrence_count,
        "normalized_record_count": occurrence_count,
        "new_canonical_article_count": new_canonical_count,
        "matched_existing_canonical_count": historical_duplicate_count,
        "within_run_duplicate_count": within_run_duplicate_count,
        "historical_duplicate_count": historical_duplicate_count,
        "represented_canonical_count": represented_canonical_count,
        "duplicate_relationship_count": relationship_count,
        "duplicate_count": relationship_count,
        "duplicate_occurrence_count": duplicate_occurrence_count,
        "deduplication_rule_counts": relationship_rule_counts,
        "provider_error_count": provider_error_count,
        "rejected_record_count": rejected_record_count,
        "missing_title_count": missing_title_count,
        "missing_publication_time_count": missing_publication_time_count,
        "missing_body_count": missing_body_count,
        "invalid_url_count": invalid_url_count,
        "ogdc_linked_article_count": ogdc_linked_count,
        "relationship_bijection_valid": mismatch_count == 0,
        "expected_quality_status": expected_quality_status,
    }


_COUNT_FIELD_MAP = {
    "total_raw_records": "raw_record_count",
    "raw_record_count": "raw_record_count",
    "total_normalized_records": "normalized_occurrence_count",
    "normalized_record_count": "normalized_occurrence_count",
    "normalized_occurrence_count": "normalized_occurrence_count",
    "new_canonical_article_count": "new_canonical_article_count",
    "matched_existing_canonical_count": "matched_existing_canonical_count",
    "within_run_duplicate_count": "within_run_duplicate_count",
    "historical_duplicate_count": "historical_duplicate_count",
    "represented_canonical_count": "represented_canonical_count",
    "duplicate_count": "duplicate_relationship_count",
    "duplicate_relationship_count": "duplicate_relationship_count",
    "provider_error_count": "provider_error_count",
    "rejected_record_count": "rejected_record_count",
    "missing_title_count": "missing_title_count",
    "missing_publication_time_count": "missing_publication_time_count",
    "missing_body_count": "missing_body_count",
    "invalid_url_count": "invalid_url_count",
    "ogdc_linked_article_count": "ogdc_linked_article_count",
}


def _validate_count_payload(
    payload: Mapping[str, Any], snapshot: Mapping[str, Any], *, label: str
) -> None:
    for supplied_field, snapshot_field in _COUNT_FIELD_MAP.items():
        if supplied_field not in payload:
            continue
        supplied = payload[supplied_field]
        if isinstance(supplied, bool):
            raise ValueError(f"{label} {supplied_field} must be an integer")
        try:
            supplied_count = int(supplied)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{label} {supplied_field} must be an integer"
            ) from exc
        if supplied_count != int(snapshot[snapshot_field]):
            raise ValueError(
                f"{label} {supplied_field} contradicts persisted evidence: "
                f"{supplied_count} != {snapshot[snapshot_field]}"
            )
    for field in ("deduplication_rule_counts", "duplicate_rule_counts"):
        if field in payload:
            supplied_rules = payload[field]
            if not isinstance(supplied_rules, Mapping):
                raise ValueError(f"{label} {field} must be a mapping")
            normalized = {
                str(key): int(value) for key, value in supplied_rules.items()
            }
            if normalized != snapshot["deduplication_rule_counts"]:
                raise ValueError(
                    f"{label} {field} contradicts persisted evidence"
                )


def _validate_quality_payload(
    quality_status: str,
    payload: Mapping[str, Any],
    snapshot: Mapping[str, Any],
) -> None:
    required = {
        "schema_version",
        "quality_status",
        "total_raw_records",
        "total_normalized_records",
        "normalized_occurrence_count",
        "new_canonical_article_count",
        "matched_existing_canonical_count",
        "within_run_duplicate_count",
        "historical_duplicate_count",
        "represented_canonical_count",
        "duplicate_relationship_count",
        "duplicate_count",
        "deduplication_rule_counts",
        "provider_error_count",
        "rejected_record_count",
        "missing_title_count",
        "missing_publication_time_count",
        "missing_body_count",
        "invalid_url_count",
        "ogdc_linked_article_count",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(
            "quality report is missing database-verifiable fields: "
            + ", ".join(missing)
        )
    if str(payload["schema_version"]) != QUALITY_REPORT_SCHEMA_VERSION:
        raise ValueError(
            "quality report schema_version must be "
            f"{QUALITY_REPORT_SCHEMA_VERSION}"
        )
    reported_status = str(payload["quality_status"]).strip().upper()
    if reported_status != quality_status:
        raise ValueError("quality report status contradicts quality_status")
    _validate_count_payload(payload, snapshot, label="quality report")
    severity = {"PASS": 0, "WARN": 1, "FAIL": 2}
    expected = str(snapshot["expected_quality_status"])
    if severity[quality_status] < severity[expected]:
        raise ValueError(
            f"quality report is more optimistic than persisted evidence: "
            f"{quality_status} < {expected}"
        )


class NewsStorage:
    """A small transactional repository around one configurable SQLite file."""

    def __init__(self, database_path: str | Path):
        self.path = Path(database_path).expanduser().resolve()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA synchronous = NORMAL")
        return connection

    def initialize(self) -> "NewsStorage":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current > DATABASE_SCHEMA_VERSION:
                raise RuntimeError(
                    "News database schema is newer than this PAISA version: "
                    f"{current} > {DATABASE_SCHEMA_VERSION}"
                )
            if current == 0:
                _bootstrap_schema(connection)
                current = 1
            else:
                if current == 3 and _is_recognized_legacy_v3_schema(connection):
                    _repair_recognized_legacy_v3_schema(connection)
                _verify_schema(connection, current)
            while current < DATABASE_SCHEMA_VERSION:
                _migrate_schema(connection, current, current + 1)
                current += 1
            _verify_schema(connection, DATABASE_SCHEMA_VERSION)
        return self

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection for read-only or caller-managed work."""

        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit all statements together, rolling back the complete unit on error."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def begin_run(self, run: Any | None = None, **values: Any) -> None:
        data = _record_mapping(run, values)
        run_id = _safe_id(_value(data, "run_id", "ingestion_run_id"), field="run_id")
        started_at = _timestamp(
            _value(data, "started_at", "fetch_timestamp", "fetched_at", default=utc_now_z()),
            field="started_at",
        )
        provider_id = _required_text(data, "provider_id", "provider")
        provider_mode = _required_text(data, "provider_mode", "provider_type")
        data_kind = _required_text(data, "data_kind", "record_kind").lower()
        schema_version = _news_record_schema(
            data.get("schema_version", NEWS_RECORD_SCHEMA_VERSION)
        )
        with self.transaction() as connection:
            active = connection.execute(
                "SELECT run_id FROM ingestion_runs WHERE status = 'RUNNING'"
            ).fetchone()
            if active is not None:
                raise RuntimeError(
                    "Another news ingestion run is already active: "
                    f"{active['run_id']}"
                )
            connection.execute(
                """
                INSERT INTO ingestion_runs(
                    run_id, schema_version, provider_id, provider_mode,
                    configured_source, data_kind, started_at, status,
                    artifact_directory, summary_json, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'RUNNING', ?, '{}', NULL)
                """,
                (
                    run_id,
                    schema_version,
                    provider_id,
                    provider_mode,
                    _optional_text(data.get("configured_source")),
                    data_kind,
                    started_at,
                    _optional_text(
                        _value(data, "artifact_directory", "artifact_dir")
                    ),
                ),
            )

    def save_raw(self, record: Any | None = None, **values: Any) -> str:
        data = _record_mapping(record, values)
        run_id = _safe_id(
            _value(data, "ingestion_run_id", "run_id"), field="ingestion_run_id"
        )
        provider_id = _required_text(data, "provider_id", "provider")
        provider_record_id = _optional_text(
            _value(data, "provider_record_id", "external_id", "record_id")
        )
        raw_reference = _artifact_reference(
            _value(data, "raw_artifact_reference", "artifact_reference"),
            field="raw_artifact_reference",
        )
        source_raw_id = _optional_text(data.get("raw_record_id")) or provider_record_id
        if source_raw_id is None:
            source_raw_id = _deterministic_id(
                "source",
                provider_id,
                raw_reference,
            )
        raw_evidence_id = _value(data, "raw_evidence_id") or _deterministic_id(
            "raw",
            run_id,
            provider_id,
            source_raw_id,
            raw_reference,
        )
        raw_evidence_id = _safe_id(raw_evidence_id, field="raw_evidence_id")
        source_raw_id = str(source_raw_id).strip()
        if not source_raw_id:
            raise ValueError("raw_record_id cannot be blank")
        checksum_value = _value(data, "response_checksum", "response_sha256")
        checksum = (
            _sha256(checksum_value, field="response_checksum")
            if checksum_value is not None
            else None
        )
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO raw_source_records(
                    raw_evidence_id, ingestion_run_id, raw_record_id, provider_id,
                    provider_record_id, configured_source, fetched_at,
                    http_status, response_content_type, response_checksum,
                    data_kind, raw_artifact_reference, payload_json,
                    provider_warnings_json, provider_errors_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    raw_evidence_id,
                    run_id,
                    source_raw_id,
                    provider_id,
                    provider_record_id,
                    _optional_text(data.get("configured_source")),
                    _timestamp(data.get("fetched_at"), field="fetched_at"),
                    data.get("http_status"),
                    _optional_text(data.get("response_content_type")),
                    checksum,
                    _required_text(data, "data_kind", "record_kind").lower(),
                    raw_reference,
                    _json_text(
                        _value(data, "payload", "raw_payload", default={}), default={}
                    ),
                    _json_text(
                        _value(data, "provider_warnings", "warnings", default=[]),
                        default=[],
                    ),
                    _json_text(
                        _value(data, "provider_errors", "errors", default=[]),
                        default=[],
                    ),
                ),
            )
        return raw_evidence_id

    save_raw_record = save_raw

    def save_article(self, record: Any | None = None, **values: Any) -> str:
        """Save a canonical article or a duplicate occurrence without data loss.

        A record with ``duplicate_of`` does not create a second canonical row;
        its complete normalized payload is retained in ``article_ingestions``.
        """

        data = _record_mapping(record, values)
        candidate_id = _safe_id(data.get("article_id"), field="article_id")
        duplicate_of = _optional_text(data.get("duplicate_of"))
        canonical_id = _safe_id(duplicate_of or candidate_id, field="canonical_article_id")
        run_id = _safe_id(
            _value(data, "ingestion_run_id", "run_id"), field="ingestion_run_id"
        )
        raw_evidence_id = _optional_text(data.get("raw_evidence_id"))
        source_raw_record_id = _optional_text(data.get("raw_record_id"))
        available_at = _timestamp(data.get("available_at"), field="available_at")
        fetched_at = _timestamp(data.get("fetched_at"), field="fetched_at")
        published_at = _timestamp(
            data.get("published_at"), field="published_at", nullable=True
        )
        source_observed_at = _timestamp(
            data.get("source_observed_at"),
            field="source_observed_at",
            nullable=True,
        )
        declared_schema = data.get("schema_version")
        schema_version = (
            None
            if declared_schema is None
            else _news_record_schema(
                declared_schema, source_observed_at=source_observed_at
            )
        )
        assert available_at is not None and fetched_at is not None
        _validate_availability(
            available_at=available_at,
            fetched_at=fetched_at,
            published_at=published_at,
        )
        raw_reference = _artifact_reference(
            data.get("raw_artifact_reference"), field="raw_artifact_reference"
        )
        created_at = _timestamp(
            data.get("created_at", utc_now_z()), field="created_at"
        )
        deduplication_rule = _deduplication_rule(
            _value(data, "deduplication_rule", "duplicate_rule"), nullable=True
        )
        occurrence_id = _safe_id(
            data.get("occurrence_id")
            or _deterministic_id("occ", run_id, candidate_id, raw_evidence_id),
            field="occurrence_id",
        )
        with self.transaction() as connection:
            if raw_evidence_id is None and source_raw_record_id is None:
                raise ValueError("article ingestion requires raw evidence")
            if raw_evidence_id is None and source_raw_record_id is not None:
                raw_row = connection.execute(
                    """
                    SELECT raw_evidence_id FROM raw_source_records
                    WHERE ingestion_run_id = ? AND raw_record_id = ?
                    """,
                    (run_id, source_raw_record_id),
                ).fetchone()
                if raw_row is None:
                    raise KeyError(
                        "No raw evidence exists for ingestion_run_id/raw_record_id"
                    )
                raw_evidence_id = str(raw_row["raw_evidence_id"])
            raw_owner = connection.execute(
                """
                SELECT ingestion_run_id FROM raw_source_records
                WHERE raw_evidence_id = ?
                """,
                (raw_evidence_id,),
            ).fetchone()
            if raw_owner is None:
                raise KeyError("No raw evidence exists for raw_evidence_id")
            if raw_owner["ingestion_run_id"] != run_id:
                raise ValueError(
                    "article ingestion raw evidence must belong to the same "
                    "ingestion run"
                )
            run_schema = connection.execute(
                "SELECT schema_version FROM ingestion_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_schema is None:
                raise ValueError("article ingestion run does not exist")
            run_schema_version = _news_record_schema(
                run_schema["schema_version"],
                source_observed_at=source_observed_at,
            )
            if schema_version is None:
                schema_version = run_schema_version
            elif run_schema_version != schema_version:
                raise ValueError(
                    "article schema_version must match its ingestion run"
                )
            if duplicate_of is None:
                title = _optional_text(data.get("title"))
                article_columns = [
                    "article_id", "schema_version", "provider", "source_name",
                    "source_type", "source_url", "canonical_url", "title",
                    "normalized_title", "summary", "body", "published_at",
                    "published_timezone", "available_at", "fetched_at", "language",
                    "mentioned_tickers_json", "mentioned_companies_json",
                    "mentioned_sectors_json", "content_sha256", "duplicate_of",
                    "first_ingestion_run_id", "raw_artifact_reference",
                    "normalization_warnings_json", "created_at",
                ]
                article_values: list[Any] = [
                    candidate_id,
                    schema_version,
                    _required_text(data, "provider"),
                    _required_text(data, "source_name"),
                    _required_text(data, "source_type"),
                    _optional_text(data.get("source_url")),
                    _optional_text(data.get("canonical_url")),
                    title,
                    _normalized_title(data.get("normalized_title", title)),
                    _optional_text(data.get("summary")),
                    _optional_text(data.get("body")),
                    published_at,
                    _optional_text(data.get("published_timezone")),
                    available_at,
                    fetched_at,
                    _optional_text(data.get("language")),
                    _json_text(data.get("mentioned_tickers"), default=[]),
                    _json_text(data.get("mentioned_companies"), default=[]),
                    _json_text(data.get("mentioned_sectors"), default=[]),
                    _sha256(data.get("content_sha256"), field="content_sha256"),
                    None,
                    run_id,
                    raw_reference,
                    _json_text(data.get("normalization_warnings"), default=[]),
                    created_at,
                ]
                supports_source_observed = any(
                    str(row[1]) == "source_observed_at"
                    for row in connection.execute(
                        "PRAGMA table_info(normalized_articles)"
                    )
                )
                if supports_source_observed:
                    article_columns.append("source_observed_at")
                    article_values.append(source_observed_at)
                elif source_observed_at is not None:
                    raise RuntimeError(
                        "source_observed_at requires news storage schema version 7"
                    )
                connection.execute(
                    f"INSERT INTO normalized_articles({', '.join(article_columns)}) "
                    f"VALUES ({', '.join('?' for _ in article_columns)})",
                    tuple(article_values),
                )

            occurrence_columns = [
                "occurrence_id", "ingestion_run_id", "raw_evidence_id",
                "candidate_article_id", "canonical_article_id", "duplicate_of",
                "deduplication_rule", "available_at", "fetched_at",
                "raw_artifact_reference", "normalized_payload_json",
                "normalization_warnings_json", "created_at",
            ]
            occurrence_values: list[Any] = [
                occurrence_id, run_id, raw_evidence_id, candidate_id, canonical_id,
                canonical_id if duplicate_of is not None else None,
                deduplication_rule, available_at, fetched_at, raw_reference,
                _json_text(
                    _normalized_payload_for_schema(
                        data, schema_version=schema_version
                    ),
                    default={},
                ),
                _json_text(data.get("normalization_warnings"), default=[]),
                created_at,
            ]
            supports_occurrence_source_observed = any(
                str(row[1]) == "source_observed_at"
                for row in connection.execute("PRAGMA table_info(article_ingestions)")
            )
            if supports_occurrence_source_observed:
                occurrence_columns.append("source_observed_at")
                occurrence_values.append(source_observed_at)
            elif source_observed_at is not None:
                raise RuntimeError(
                    "source_observed_at requires news storage schema version 7"
                )
            connection.execute(
                f"INSERT INTO article_ingestions({', '.join(occurrence_columns)}) "
                f"VALUES ({', '.join('?' for _ in occurrence_columns)})",
                tuple(occurrence_values),
            )
        return canonical_id

    def save_article_ingestion(
        self, record: Any | None = None, **values: Any
    ) -> str:
        """Store a normalized occurrence for an already saved canonical article."""

        data = _record_mapping(record, values)
        candidate_id = _safe_id(
            _value(data, "candidate_article_id", "article_id"),
            field="candidate_article_id",
        )
        canonical_id = _safe_id(
            _value(data, "canonical_article_id", "duplicate_of", "article_id"),
            field="canonical_article_id",
        )
        run_id = _safe_id(
            _value(data, "ingestion_run_id", "run_id"), field="ingestion_run_id"
        )
        raw_evidence_id = _optional_text(data.get("raw_evidence_id"))
        source_raw_record_id = _optional_text(data.get("raw_record_id"))
        occurrence_id = _safe_id(
            data.get("occurrence_id")
            or _deterministic_id("occ", run_id, candidate_id, raw_evidence_id),
            field="occurrence_id",
        )
        duplicate = candidate_id != canonical_id or bool(data.get("duplicate_of"))
        available_at = _timestamp(data.get("available_at"), field="available_at")
        fetched_at = _timestamp(data.get("fetched_at"), field="fetched_at")
        source_observed_at = _timestamp(
            data.get("source_observed_at"),
            field="source_observed_at",
            nullable=True,
        )
        declared_schema = data.get("schema_version")
        schema_version = (
            None
            if declared_schema is None
            else _news_record_schema(
                declared_schema, source_observed_at=source_observed_at
            )
        )
        assert available_at is not None and fetched_at is not None
        _validate_availability(
            available_at=available_at,
            fetched_at=fetched_at,
        )
        raw_reference = _artifact_reference(
            data.get("raw_artifact_reference"), field="raw_artifact_reference"
        )
        deduplication_rule = _deduplication_rule(
            _value(data, "deduplication_rule", "duplicate_rule"), nullable=True
        )
        with self.transaction() as connection:
            if raw_evidence_id is None and source_raw_record_id is None:
                raise ValueError("article ingestion requires raw evidence")
            if raw_evidence_id is None and source_raw_record_id is not None:
                raw_row = connection.execute(
                    """
                    SELECT raw_evidence_id FROM raw_source_records
                    WHERE ingestion_run_id = ? AND raw_record_id = ?
                    """,
                    (run_id, source_raw_record_id),
                ).fetchone()
                if raw_row is None:
                    raise KeyError(
                        "No raw evidence exists for ingestion_run_id/raw_record_id"
                    )
                raw_evidence_id = str(raw_row["raw_evidence_id"])
            raw_owner = connection.execute(
                """
                SELECT ingestion_run_id FROM raw_source_records
                WHERE raw_evidence_id = ?
                """,
                (raw_evidence_id,),
            ).fetchone()
            if raw_owner is None:
                raise KeyError("No raw evidence exists for raw_evidence_id")
            if raw_owner["ingestion_run_id"] != run_id:
                raise ValueError(
                    "article ingestion raw evidence must belong to the same "
                    "ingestion run"
                )
            run_schema = connection.execute(
                "SELECT schema_version FROM ingestion_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run_schema is None:
                raise ValueError("article ingestion run does not exist")
            run_schema_version = _news_record_schema(
                run_schema["schema_version"],
                source_observed_at=source_observed_at,
            )
            if schema_version is None:
                schema_version = run_schema_version
            elif run_schema_version != schema_version:
                raise ValueError(
                    "article schema_version must match its ingestion run"
                )
            occurrence_columns = [
                "occurrence_id", "ingestion_run_id", "raw_evidence_id",
                "candidate_article_id", "canonical_article_id", "duplicate_of",
                "deduplication_rule", "available_at", "fetched_at",
                "raw_artifact_reference", "normalized_payload_json",
                "normalization_warnings_json", "created_at",
            ]
            occurrence_values: list[Any] = [
                occurrence_id,
                run_id,
                raw_evidence_id,
                candidate_id,
                canonical_id,
                canonical_id if duplicate else None,
                deduplication_rule,
                available_at,
                fetched_at,
                raw_reference,
                _json_text(
                    _normalized_payload_for_schema(
                        _value(
                            data, "normalized_payload", "payload", default=data
                        ),
                        schema_version=schema_version,
                    ),
                    default={},
                ),
                _json_text(data.get("normalization_warnings"), default=[]),
                _timestamp(
                    data.get("created_at", utc_now_z()), field="created_at"
                ),
            ]
            supports_source_observed = any(
                str(row[1]) == "source_observed_at"
                for row in connection.execute("PRAGMA table_info(article_ingestions)")
            )
            if supports_source_observed:
                occurrence_columns.append("source_observed_at")
                occurrence_values.append(source_observed_at)
            elif source_observed_at is not None:
                raise RuntimeError(
                    "source_observed_at requires news storage schema version 7"
                )
            connection.execute(
                f"INSERT INTO article_ingestions({', '.join(occurrence_columns)}) "
                f"VALUES ({', '.join('?' for _ in occurrence_columns)})",
                tuple(occurrence_values),
            )
        return occurrence_id

    def save_link(self, link: Any | None = None, **values: Any) -> str:
        data = _record_mapping(link, values)
        ticker = _required_text(data, "ticker").upper()
        matched_alias = _required_text(data, "matched_alias")
        match_method = _required_text(data, "match_method")
        match_location = _required_text(data, "match_location")
        with self.transaction() as connection:
            occurrence_value = _optional_text(data.get("occurrence_id"))
            if occurrence_value is not None:
                occurrence_id = _safe_id(
                    occurrence_value, field="occurrence_id"
                )
                occurrence = connection.execute(
                    """
                    SELECT occurrence_id, canonical_article_id
                    FROM article_ingestions WHERE occurrence_id = ?
                    """,
                    (occurrence_id,),
                ).fetchone()
            else:
                article_id = _safe_id(
                    _value(data, "article_id", "candidate_article_id"),
                    field="article_id",
                )
                run_id = _optional_text(
                    _value(data, "ingestion_run_id", "run_id")
                )
                if run_id is None:
                    occurrences = connection.execute(
                        """
                        SELECT occurrence_id, canonical_article_id
                        FROM article_ingestions
                        WHERE candidate_article_id = ? OR canonical_article_id = ?
                        ORDER BY created_at, occurrence_id
                        """,
                        (article_id, article_id),
                    ).fetchall()
                else:
                    occurrences = connection.execute(
                        """
                        SELECT occurrence_id, canonical_article_id
                        FROM article_ingestions
                        WHERE ingestion_run_id = ? AND
                              (candidate_article_id = ? OR canonical_article_id = ?)
                        ORDER BY created_at, occurrence_id
                        """,
                        (
                            _safe_id(run_id, field="ingestion_run_id"),
                            article_id,
                            article_id,
                        ),
                    ).fetchall()
                if len(occurrences) > 1:
                    raise ValueError(
                        "article_id identifies multiple ingestion occurrences; "
                        "provide occurrence_id or ingestion_run_id"
                    )
                occurrence = occurrences[0] if occurrences else None
            if occurrence is None:
                raise KeyError("No article ingestion occurrence exists for this link")
            occurrence_id = str(occurrence["occurrence_id"])
            canonical_id = str(occurrence["canonical_article_id"])
            supplied_canonical = _optional_text(data.get("canonical_article_id"))
            if supplied_canonical is not None and supplied_canonical != canonical_id:
                raise ValueError(
                    "canonical_article_id does not match the ingestion occurrence"
                )
            link_id = _safe_id(
                data.get("link_id")
                or _deterministic_id(
                    "link",
                    occurrence_id,
                    ticker,
                    matched_alias,
                    match_method,
                    match_location,
                ),
                field="link_id",
            )
            connection.execute(
                """
                INSERT INTO article_company_links(
                    link_id, occurrence_id, canonical_article_id, ticker,
                    company_name, matched_alias, match_method, match_location,
                    confidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    link_id,
                    occurrence_id,
                    canonical_id,
                    ticker,
                    _required_text(data, "company_name"),
                    matched_alias,
                    match_method,
                    match_location,
                    _required_text(data, "confidence"),
                    _timestamp(
                        data.get("created_at", utc_now_z()), field="created_at"
                    ),
                ),
            )
        return link_id

    def save_dedup(self, relationship: Any | None = None, **values: Any) -> str:
        data = _record_mapping(relationship, values)
        run_id = _safe_id(
            _value(data, "ingestion_run_id", "run_id"), field="ingestion_run_id"
        )
        occurrence_id = _safe_id(data.get("occurrence_id"), field="occurrence_id")
        candidate_id = _safe_id(
            _value(data, "duplicate_candidate_article_id", "article_id"),
            field="duplicate_candidate_article_id",
        )
        canonical_id = _safe_id(
            _value(data, "canonical_article_id", "duplicate_of"),
            field="canonical_article_id",
        )
        rule = _deduplication_rule(
            _value(data, "deduplication_rule", "rule"), nullable=False
        )
        assert rule is not None
        relationship_id = _safe_id(
            data.get("relationship_id")
            or _deterministic_id(
                "dedup", run_id, occurrence_id, candidate_id, canonical_id, rule
            ),
            field="relationship_id",
        )
        with self.transaction() as connection:
            occurrence = connection.execute(
                """
                SELECT ingestion_run_id, candidate_article_id,
                       canonical_article_id, duplicate_of, deduplication_rule
                FROM article_ingestions
                WHERE occurrence_id = ?
                """,
                (occurrence_id,),
            ).fetchone()
            if occurrence is None:
                raise KeyError("No article ingestion occurrence exists for deduplication")
            if occurrence["ingestion_run_id"] != run_id:
                raise ValueError(
                    "deduplication relationship must belong to its occurrence "
                    "ingestion run"
                )
            if occurrence["duplicate_of"] is None:
                raise ValueError(
                    "deduplication occurrence must be marked as a duplicate"
                )
            if occurrence["candidate_article_id"] != candidate_id:
                raise ValueError(
                    "duplicate_candidate_article_id does not match the occurrence"
                )
            if (
                occurrence["canonical_article_id"] != canonical_id
                or occurrence["duplicate_of"] != canonical_id
            ):
                raise ValueError(
                    "canonical_article_id/duplicate_of does not match the occurrence"
                )
            if occurrence["deduplication_rule"] != rule:
                raise ValueError(
                    "deduplication_rule does not match the occurrence"
                )
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
                    occurrence_id,
                    candidate_id,
                    canonical_id,
                    canonical_id,
                    rule,
                    _timestamp(
                        data.get("detected_at", utc_now_z()), field="detected_at"
                    ),
                ),
            )
        return relationship_id

    save_deduplication = save_dedup

    def save_error(self, error: Any | None = None, **values: Any) -> str:
        data = _record_mapping(error, values)
        run_id = _safe_id(
            _value(data, "ingestion_run_id", "run_id"), field="ingestion_run_id"
        )
        provider_id = _required_text(data, "provider_id", "provider")
        occurred_at = _timestamp(
            data.get("occurred_at", utc_now_z()), field="occurred_at"
        )
        error_type = _required_text(data, "error_type", "type")
        message = _required_text(data, "message", "error")
        error_id = _safe_id(
            data.get("error_id")
            or _deterministic_id(
                "error", run_id, provider_id, occurred_at, error_type, message
            ),
            field="error_id",
        )
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO provider_errors(
                    error_id, ingestion_run_id, provider_id, configured_source,
                    occurred_at, error_type, message, http_status, retryable,
                    details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    error_id,
                    run_id,
                    provider_id,
                    _optional_text(data.get("configured_source")),
                    occurred_at,
                    error_type,
                    message,
                    data.get("http_status"),
                    int(bool(data.get("retryable", False))),
                    _json_text(data.get("details"), default={}),
                ),
            )
        return error_id

    save_provider_error = save_error

    def save_quality(self, report: Any | None = None, **values: Any) -> None:
        data = _record_mapping(report, values)
        run_id = _safe_id(
            _value(data, "ingestion_run_id", "run_id"), field="ingestion_run_id"
        )
        quality_status = _required_text(
            data, "quality_status", "status"
        ).upper()
        if quality_status not in {"PASS", "WARN", "FAIL"}:
            raise ValueError("quality_status must be PASS, WARN, or FAIL")
        payload = _value(data, "report", "report_payload", default=data)
        if not isinstance(payload, Mapping):
            raise TypeError("quality report payload must be a mapping")
        with self.transaction() as connection:
            run = connection.execute(
                "SELECT status FROM ingestion_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown ingestion run: {run_id}")
            if run["status"] != "RUNNING":
                raise RuntimeError(f"Ingestion run is already finalized: {run_id}")
            snapshot = _run_evidence_snapshot(connection, run_id)
            _validate_quality_payload(quality_status, payload, snapshot)
            connection.execute(
                """
                INSERT INTO quality_reports(
                    ingestion_run_id, schema_version, quality_status,
                    created_at, report_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    QUALITY_REPORT_SCHEMA_VERSION,
                    quality_status,
                    _timestamp(
                        data.get("created_at", utc_now_z()), field="created_at"
                    ),
                    _json_text(payload, default={}),
                ),
            )

    save_quality_report = save_quality

    def find_duplicate_candidates(
        self,
        *,
        canonical_url: str | None = None,
        content_sha256: str | None = None,
        title: str | None = None,
        normalized_title: str | None = None,
        source_name: str | None = None,
        published_at: str | datetime | None = None,
        time_tolerance_seconds: int = 86_400,
    ) -> list[dict[str, Any]]:
        """Return canonical candidates matching URL, hash, or title/source/minute.

        ``time_tolerance_seconds`` remains accepted for API compatibility, but
        title-based candidate retrieval intentionally uses the exact UTC minute.
        Title and source equality are evaluated by the pipeline's shared Python
        normalization rule, rather than by SQLite's case-sensitive text rules.
        """

        clauses: list[str] = []
        parameters: list[Any] = []
        url = _optional_text(canonical_url)
        if url:
            clauses.append("canonical_url = ?")
            parameters.append(url)
        if content_sha256:
            clauses.append("content_sha256 = ?")
            parameters.append(_sha256(content_sha256, field="content_sha256"))
        title_key = _normalized_title(normalized_title or title)
        if title_key and source_name and published_at is not None:
            if time_tolerance_seconds < 0:
                raise ValueError("time_tolerance_seconds must be non-negative")
            clauses.append(
                """(
                    published_at IS NOT NULL AND
                    strftime('%Y-%m-%dT%H:%M', published_at) =
                        strftime('%Y-%m-%dT%H:%M', ?)
                )"""
            )
            parameters.append(
                _timestamp(published_at, field="published_at")
            )
        if not clauses:
            return []
        query = (
            "SELECT article.*, "
            "owner.sealed_manifest_sha256 AS owner_manifest_sha256 "
            "FROM normalized_articles AS article "
            "JOIN ingestion_runs AS owner "
            "ON owner.run_id = article.first_ingestion_run_id "
            "WHERE article.duplicate_of IS NULL "
            "AND owner.status IN ('PASS', 'PARTIAL') AND ("
            + " OR ".join(clauses)
            + ") ORDER BY article.article_id"
        )
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def get_run_evidence_snapshot(self, run_id: str) -> dict[str, Any]:
        """Return counts derived from SQLite rather than caller-supplied reports."""

        identifier = _safe_id(run_id, field="run_id")
        with self.connection() as connection:
            run = connection.execute(
                "SELECT status FROM ingestion_runs WHERE run_id = ?", (identifier,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown ingestion run: {identifier}")
            snapshot = _run_evidence_snapshot(connection, identifier)
        snapshot["run_id"] = identifier
        snapshot["status"] = str(run["status"])
        return snapshot

    def validate_run_evidence(
        self,
        run_id: str,
        *,
        quality_report: Mapping[str, Any] | None = None,
        summary: Mapping[str, Any] | None = None,
        require_relationship_bijection: bool = True,
    ) -> dict[str, Any]:
        """Validate persisted counts, quality, and duplicate relationships."""

        identifier = _safe_id(run_id, field="run_id")
        with self.connection() as connection:
            run = connection.execute(
                "SELECT status FROM ingestion_runs WHERE run_id = ?", (identifier,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown ingestion run: {identifier}")
            snapshot = _run_evidence_snapshot(connection, identifier)
            if require_relationship_bijection and not snapshot[
                "relationship_bijection_valid"
            ]:
                raise RuntimeError(
                    "Every duplicate occurrence must have exactly one matching "
                    "relationship, and canonical occurrences must have none"
                )
            report_status: str | None = None
            if quality_report is None:
                quality_row = connection.execute(
                    """
                    SELECT quality_status, report_json FROM quality_reports
                    WHERE ingestion_run_id = ?
                    """,
                    (identifier,),
                ).fetchone()
                if quality_row is not None:
                    report_status = str(quality_row["quality_status"])
                    try:
                        loaded = json.loads(str(quality_row["report_json"]))
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            "Persisted quality report JSON is unreadable"
                        ) from exc
                    if not isinstance(loaded, Mapping):
                        raise RuntimeError(
                            "Persisted quality report JSON must be an object"
                        )
                    quality_report = loaded
            else:
                report_status = str(
                    quality_report.get("quality_status", "")
                ).strip().upper()
            if quality_report is not None:
                if report_status not in {"PASS", "WARN", "FAIL"}:
                    raise ValueError("quality report must declare a valid status")
                _validate_quality_payload(report_status, quality_report, snapshot)
            if summary is not None:
                _validate_count_payload(summary, snapshot, label="run summary")
        snapshot["run_id"] = identifier
        snapshot["status"] = str(run["status"])
        return snapshot

    def seal_run(
        self,
        run_id: str,
        *,
        artifact_inventory: Sequence[Mapping[str, Any]],
        manifest_sha256: str,
        completion_marker_sha256: str,
        expected_counts: Mapping[str, Any] | None = None,
        prepared_at: str | datetime | None = None,
    ) -> dict[str, Any]:
        """Bind a RUNNING row to one exact artifact set and completion identity."""

        identifier = _safe_id(run_id, field="run_id")
        normalized_inventory = _sealed_inventory(artifact_inventory)
        inventory_json, inventory_sha256 = _json_sha256(
            normalized_inventory, default=[]
        )
        manifest_digest = _sha256(manifest_sha256, field="manifest_sha256")
        completion_digest = _sha256(
            completion_marker_sha256, field="completion_marker_sha256"
        )
        prepared = _timestamp(prepared_at or utc_now_z(), field="prepared_at")
        with self.transaction() as connection:
            run = connection.execute(
                """
                SELECT status, sealed_manifest_sha256 FROM ingestion_runs
                WHERE run_id = ?
                """,
                (identifier,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown ingestion run: {identifier}")
            if run["status"] != "RUNNING":
                raise RuntimeError(f"Ingestion run is already finalized: {identifier}")
            if run["sealed_manifest_sha256"] is not None:
                raise RuntimeError(f"Ingestion run is already sealed: {identifier}")
            snapshot = _run_evidence_snapshot(connection, identifier)
            if not snapshot["relationship_bijection_valid"]:
                raise RuntimeError(
                    "Cannot seal a run with incomplete duplicate relationships"
                )
            quality_row = connection.execute(
                """
                SELECT quality_status, report_json FROM quality_reports
                WHERE ingestion_run_id = ?
                """,
                (identifier,),
            ).fetchone()
            if quality_row is None:
                raise RuntimeError("Cannot seal a run without a quality report")
            try:
                quality_payload = json.loads(str(quality_row["report_json"]))
            except json.JSONDecodeError as exc:
                raise RuntimeError("Persisted quality report JSON is unreadable") from exc
            if not isinstance(quality_payload, Mapping):
                raise RuntimeError("Persisted quality report JSON must be an object")
            _validate_quality_payload(
                str(quality_row["quality_status"]), quality_payload, snapshot
            )
            if expected_counts is not None:
                _validate_count_payload(
                    expected_counts, snapshot, label="finalization counts"
                )
            result = connection.execute(
                """
                UPDATE ingestion_runs
                SET prepared_at = ?, sealed_artifact_inventory_json = ?,
                    sealed_artifact_inventory_sha256 = ?,
                    sealed_manifest_sha256 = ?, sealed_completion_sha256 = ?
                WHERE run_id = ? AND status = 'RUNNING'
                  AND sealed_manifest_sha256 IS NULL
                """,
                (
                    prepared,
                    inventory_json,
                    inventory_sha256,
                    manifest_digest,
                    completion_digest,
                    identifier,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError(f"Could not seal ingestion run: {identifier}")
        return {
            "run_id": identifier,
            "prepared_at": prepared,
            "artifact_inventory": normalized_inventory,
            "artifact_inventory_sha256": inventory_sha256,
            "manifest_sha256": manifest_digest,
            "completion_marker_sha256": completion_digest,
        }

    def get_run_finalization_binding(
        self, run_id: str
    ) -> dict[str, Any] | None:
        identifier = _safe_id(run_id, field="run_id")
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT prepared_at, sealed_artifact_inventory_json,
                       sealed_artifact_inventory_sha256,
                       sealed_manifest_sha256, sealed_completion_sha256
                FROM ingestion_runs WHERE run_id = ?
                """,
                (identifier,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown ingestion run: {identifier}")
        if row["sealed_manifest_sha256"] is None:
            return None
        return {
            "run_id": identifier,
            "prepared_at": row["prepared_at"],
            "artifact_inventory": json.loads(
                str(row["sealed_artifact_inventory_json"])
            ),
            "artifact_inventory_sha256": row[
                "sealed_artifact_inventory_sha256"
            ],
            "manifest_sha256": row["sealed_manifest_sha256"],
            "completion_marker_sha256": row["sealed_completion_sha256"],
        }

    def discard_running_run(self, run_id: str) -> None:
        """Delete an unsealed interrupted run and only the evidence it owns."""

        identifier = _safe_id(run_id, field="run_id")
        with self.transaction() as connection:
            run = connection.execute(
                """
                SELECT status, sealed_manifest_sha256 FROM ingestion_runs
                WHERE run_id = ?
                """,
                (identifier,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown ingestion run: {identifier}")
            if run["status"] != "RUNNING":
                raise RuntimeError("Only a RUNNING ingestion run can be discarded")
            if run["sealed_manifest_sha256"] is not None:
                raise RuntimeError("A sealed ingestion run cannot be discarded")
            external_reference = connection.execute(
                """
                SELECT occurrence.occurrence_id
                FROM article_ingestions AS occurrence
                JOIN normalized_articles AS canonical
                  ON canonical.article_id = occurrence.canonical_article_id
                WHERE canonical.first_ingestion_run_id = ?
                  AND occurrence.ingestion_run_id <> ?
                LIMIT 1
                """,
                (identifier, identifier),
            ).fetchone()
            if external_reference is not None:
                raise RuntimeError(
                    "Cannot discard a RUNNING ingestion run whose canonical "
                    "evidence is referenced by another run"
                )
            connection.execute(
                """
                DELETE FROM article_company_links
                WHERE occurrence_id IN (
                    SELECT occurrence_id FROM article_ingestions
                    WHERE ingestion_run_id = ?
                )
                """,
                (identifier,),
            )
            connection.execute(
                "DELETE FROM deduplication_relationships WHERE ingestion_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM article_ingestions WHERE ingestion_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM normalized_articles WHERE first_ingestion_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM raw_source_records WHERE ingestion_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM provider_errors WHERE ingestion_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM quality_reports WHERE ingestion_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM ingestion_runs WHERE run_id = ?", (identifier,)
            )

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        completed_at: str | datetime | None = None,
        summary: Mapping[str, Any] | None = None,
        error_message: str | None = None,
        expected_manifest_sha256: str | None = None,
        expected_inventory_sha256: str | None = None,
        expected_completion_sha256: str | None = None,
    ) -> None:
        identifier = _safe_id(run_id, field="run_id")
        final_status = str(status).strip().upper()
        if final_status not in {"PASS", "PARTIAL", "FAIL"}:
            raise ValueError("Final run status must be PASS, PARTIAL, or FAIL")
        with self.transaction() as connection:
            current = connection.execute(
                """
                SELECT status, sealed_artifact_inventory_sha256,
                       sealed_manifest_sha256, sealed_completion_sha256
                FROM ingestion_runs WHERE run_id = ?
                """,
                (identifier,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown ingestion run: {identifier}")
            if current["status"] != "RUNNING":
                raise RuntimeError(f"Ingestion run is already finalized: {identifier}")
            quality = connection.execute(
                """
                SELECT quality_status, report_json FROM quality_reports
                WHERE ingestion_run_id = ?
                """,
                (identifier,),
            ).fetchone()
            if final_status in {"PASS", "PARTIAL"}:
                if (
                    current["sealed_manifest_sha256"] is None
                    or current["sealed_artifact_inventory_sha256"] is None
                    or current["sealed_completion_sha256"] is None
                ):
                    raise RuntimeError(
                        f"A {final_status} run must be sealed before finalization"
                    )
                expected_digests = (
                    (
                        expected_manifest_sha256,
                        current["sealed_manifest_sha256"],
                        "manifest",
                    ),
                    (
                        expected_inventory_sha256,
                        current["sealed_artifact_inventory_sha256"],
                        "artifact inventory",
                    ),
                    (
                        expected_completion_sha256,
                        current["sealed_completion_sha256"],
                        "completion marker",
                    ),
                )
                for expected_digest, sealed_digest, label in expected_digests:
                    if expected_digest is not None and _sha256(
                        expected_digest, field=f"expected {label} sha256"
                    ) != str(sealed_digest):
                        raise RuntimeError(
                            f"Expected {label} digest does not match the DB seal"
                        )
                snapshot = _run_evidence_snapshot(connection, identifier)
                if not snapshot["relationship_bijection_valid"]:
                    raise RuntimeError(
                        "Every duplicate occurrence must have exactly one matching "
                        "deduplication relationship"
                    )
                if quality is None:
                    raise RuntimeError(
                        f"A {final_status} run requires a quality report"
                    )
                quality_payload = json.loads(str(quality["report_json"]))
                if not isinstance(quality_payload, Mapping):
                    raise RuntimeError("Persisted quality report JSON must be an object")
                _validate_quality_payload(
                    str(quality["quality_status"]), quality_payload, snapshot
                )
                if summary is not None:
                    _validate_count_payload(summary, snapshot, label="run summary")
            if final_status == "PASS":
                if quality is None or quality["quality_status"] != "PASS":
                    raise RuntimeError(
                        "A PASS run requires a persisted PASS quality report"
                    )
                provider_errors = connection.execute(
                    """
                    SELECT count(*) FROM provider_errors
                    WHERE ingestion_run_id = ?
                    """,
                    (identifier,),
                ).fetchone()[0]
                if provider_errors:
                    raise RuntimeError(
                        "A PASS run cannot contain provider errors"
                    )
            elif final_status == "PARTIAL" and (
                quality is None or quality["quality_status"] != "WARN"
            ):
                raise RuntimeError(
                    "A PARTIAL run requires a persisted WARN quality report"
                )
            result = connection.execute(
                """
                UPDATE ingestion_runs
                SET completed_at = ?, status = ?, summary_json = ?, error_message = ?
                WHERE run_id = ? AND status = 'RUNNING'
                """,
                (
                    _timestamp(
                        completed_at or utc_now_z(), field="completed_at"
                    ),
                    final_status,
                    _json_text(summary, default={}),
                    _optional_text(error_message),
                    identifier,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError(f"Could not finalize ingestion run: {identifier}")

    def register_sentiment_model_revision(
        self, record: Any | None = None, **values: Any
    ) -> str:
        """Register one immutable model/tokenizer revision and its label contract."""

        data = _record_mapping(record, values)
        identifier = _safe_id(
            _value(data, "model_revision_id", "revision_id"),
            field="model_revision_id",
        )
        repository = _required_text(data, "model_repository", "repository")
        model_revision = _required_text(data, "model_revision", "revision")
        tokenizer_repository = _optional_text(data.get("tokenizer_repository")) or repository
        tokenizer_revision = _optional_text(data.get("tokenizer_revision")) or model_revision
        model_config_sha256 = _optional_text(data.get("model_config_sha256"))
        tokenizer_config_sha256 = _optional_text(data.get("tokenizer_config_sha256"))
        backend_identity_sha256 = _sha256(
            data.get("backend_identity_sha256"), field="backend_identity_sha256"
        )
        snapshot_inventory = _model_snapshot_inventory(
            data.get("snapshot_inventory", ())
        )
        model_config_sha256 = _sha256(
            model_config_sha256, field="model_config_sha256"
        )
        tokenizer_config_sha256 = _sha256(
            tokenizer_config_sha256, field="tokenizer_config_sha256"
        )
        model_weights_sha256 = _sha256(
            data.get("model_weights_sha256"), field="model_weights_sha256"
        )
        tokenizer_vocabulary_sha256 = _sha256(
            data.get("tokenizer_vocabulary_sha256"),
            field="tokenizer_vocabulary_sha256",
        )
        tokenizer_special_tokens_sha256 = _sha256(
            data.get("tokenizer_special_tokens_sha256"),
            field="tokenizer_special_tokens_sha256",
        )
        row = {
            "model_revision_id": identifier,
            "schema_version": str(data.get("schema_version", "1.0")),
            "model_repository": repository,
            "model_revision": model_revision,
            "tokenizer_repository": tokenizer_repository,
            "tokenizer_revision": tokenizer_revision,
            "original_labels_json": _json_text(
                _value(data, "original_labels", "class_labels"), default=[]
            ),
            "label_mapping_json": _json_text(data.get("label_mapping"), default={}),
            "license_identifier": _required_text(
                data, "license_identifier", "license"
            ),
            "library_versions_json": _json_text(
                data.get("library_versions"), default={}
            ),
            "model_config_sha256": model_config_sha256,
            "tokenizer_config_sha256": tokenizer_config_sha256,
            "backend_identity_sha256": backend_identity_sha256,
            "snapshot_inventory_json": _json_text(
                snapshot_inventory, default=[]
            ),
            "model_weights_sha256": model_weights_sha256,
            "tokenizer_vocabulary_sha256": tokenizer_vocabulary_sha256,
            "tokenizer_special_tokens_sha256": tokenizer_special_tokens_sha256,
            "created_at": _timestamp(
                data.get("created_at", utc_now_z()), field="created_at"
            ),
        }
        with self.transaction() as connection:
            existing_by_id = connection.execute(
                """
                SELECT * FROM sentiment_model_revisions
                WHERE model_revision_id = ?
                """,
                (identifier,),
            ).fetchone()
            if existing_by_id is not None:
                comparable = tuple(
                    column for column in row if column != "created_at"
                )
                if any(
                    existing_by_id[column] != row[column]
                    for column in comparable
                ):
                    raise ValueError(
                        "Sentiment model revision identity already has different metadata"
                    )
                return str(existing_by_id["model_revision_id"])

            # A caller-supplied identifier is only an alias for the immutable
            # hash-bound identity.  Return the already registered canonical ID
            # when every other field agrees.  Legacy version-5 rows have a NULL
            # backend identity and therefore never masquerade as reviewed
            # version-6 evidence merely because their repository/revision tuple
            # happens to match.
            existing_by_identity = connection.execute(
                """
                SELECT * FROM sentiment_model_revisions
                WHERE backend_identity_sha256 = ?
                """,
                (backend_identity_sha256,),
            ).fetchone()
            if existing_by_identity is not None:
                comparable = tuple(
                    column
                    for column in row
                    if column not in {"model_revision_id", "created_at"}
                )
                if any(
                    existing_by_identity[column] != row[column]
                    for column in comparable
                ):
                    raise ValueError(
                        "Sentiment model revision identity already has different metadata"
                    )
                return str(existing_by_identity["model_revision_id"])
            connection.execute(
                """
                INSERT INTO sentiment_model_revisions(
                    model_revision_id, schema_version, model_repository,
                    model_revision, tokenizer_repository, tokenizer_revision,
                    original_labels_json, label_mapping_json, license_identifier,
                    library_versions_json, model_config_sha256,
                    tokenizer_config_sha256, backend_identity_sha256,
                    snapshot_inventory_json, model_weights_sha256,
                    tokenizer_vocabulary_sha256,
                    tokenizer_special_tokens_sha256, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row.values()),
            )
        return identifier

    def begin_sentiment_run(
        self, record: Any | None = None, **values: Any
    ) -> None:
        """Begin one immutable sentiment-inference run against completed news."""

        data = _record_mapping(record, values)
        run_id = _safe_id(
            _value(data, "run_id", "inference_run_id"), field="run_id"
        )
        source_run_id = _safe_id(
            data.get("source_ingestion_run_id"), field="source_ingestion_run_id"
        )
        model_revision_id = _safe_id(
            data.get("model_revision_id"), field="model_revision_id"
        )
        input_policy_id = _safe_id(
            data.get("input_policy_id"), field="input_policy_id"
        )
        maximum_articles = _optional_positive_integer(
            data.get("maximum_articles"), field="maximum_articles"
        )
        timeout_seconds = _optional_positive_number(
            data.get("timeout_seconds"), field="timeout_seconds"
        )
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sentiment_inference_runs(
                    run_id, schema_version, source_ingestion_run_id,
                    model_revision_id, input_policy_id, started_at, completed_at,
                    status, device, batch_size, maximum_input_characters,
                    maximum_sequence_length, maximum_articles, timeout_seconds,
                    artifact_directory, configuration_json, summary_json,
                    error_message
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, 'RUNNING', ?, ?, ?, ?, ?, ?, ?, ?, '{}', NULL)
                """,
                (
                    run_id,
                    str(data.get("schema_version", "1.0")),
                    source_run_id,
                    model_revision_id,
                    input_policy_id,
                    _timestamp(
                        data.get("started_at", utc_now_z()), field="started_at"
                    ),
                    _required_text(data, "device"),
                    _positive_integer(data.get("batch_size"), field="batch_size"),
                    _positive_integer(
                        data.get("maximum_input_characters"),
                        field="maximum_input_characters",
                    ),
                    _positive_integer(
                        data.get("maximum_sequence_length"),
                        field="maximum_sequence_length",
                    ),
                    maximum_articles,
                    timeout_seconds,
                    _artifact_reference(
                        data.get("artifact_directory"), field="artifact_directory"
                    ),
                    _json_text(data.get("configuration"), default={}),
                ),
            )

    def iter_sentiment_input_articles(
        self,
        source_ingestion_run_id: str,
        *,
        batch_size: int = 100,
        maximum_articles: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Yield bounded canonical-article batches represented by a completed run."""

        source_run_id = _safe_id(
            source_ingestion_run_id, field="source_ingestion_run_id"
        )
        page_size = _positive_integer(batch_size, field="batch_size")
        maximum = _optional_positive_integer(
            maximum_articles, field="maximum_articles"
        )
        remaining = maximum
        last_article_id = ""
        with self.connection() as connection:
            source = connection.execute(
                "SELECT status FROM ingestion_runs WHERE run_id = ?",
                (source_run_id,),
            ).fetchone()
            if source is None:
                raise KeyError(f"Unknown ingestion run: {source_run_id}")
            if source["status"] not in {"PASS", "PARTIAL"}:
                raise RuntimeError(
                    "Sentiment input requires a completed PASS or PARTIAL ingestion run"
                )

            while remaining is None or remaining > 0:
                limit = page_size if remaining is None else min(page_size, remaining)
                rows = connection.execute(
                    """
                    WITH represented AS (
                        SELECT canonical_article_id
                        FROM article_ingestions
                        WHERE ingestion_run_id = ?
                        GROUP BY canonical_article_id
                    )
                    SELECT
                        article.article_id,
                        article.schema_version,
                        article.provider,
                        article.source_name,
                        article.source_type,
                        article.source_url,
                        article.canonical_url,
                        article.title,
                        article.summary,
                        article.body,
                        article.published_at,
                        article.published_timezone,
                        article.available_at,
                        article.fetched_at,
                        article.language,
                        article.mentioned_tickers_json,
                        article.mentioned_companies_json,
                        article.mentioned_sectors_json,
                        article.content_sha256,
                        article.first_ingestion_run_id AS canonical_owner_run_id,
                        article.raw_artifact_reference,
                        article.normalization_warnings_json,
                        owner.data_kind AS data_class,
                        ? AS source_ingestion_run_id,
                        EXISTS (
                            SELECT 1
                            FROM article_ingestions AS linked_occurrence
                            JOIN article_company_links AS link
                              ON link.occurrence_id = linked_occurrence.occurrence_id
                            WHERE linked_occurrence.ingestion_run_id = ?
                              AND linked_occurrence.canonical_article_id = article.article_id
                              AND upper(link.ticker) = 'OGDC'
                        ) AS ogdc_linked
                    FROM represented
                    JOIN normalized_articles AS article
                      ON article.article_id = represented.canonical_article_id
                    JOIN ingestion_runs AS owner
                      ON owner.run_id = article.first_ingestion_run_id
                    WHERE article.article_id > ?
                      AND article.duplicate_of IS NULL
                      AND owner.status IN ('PASS', 'PARTIAL')
                    ORDER BY article.article_id
                    LIMIT ?
                    """,
                    (
                        source_run_id,
                        source_run_id,
                        source_run_id,
                        last_article_id,
                        limit,
                    ),
                ).fetchall()
                if not rows:
                    return
                batch: list[dict[str, Any]] = []
                for row in rows:
                    item = dict(row)
                    for stored, public in (
                        ("mentioned_tickers_json", "mentioned_tickers"),
                        ("mentioned_companies_json", "mentioned_companies"),
                        ("mentioned_sectors_json", "mentioned_sectors"),
                        ("normalization_warnings_json", "normalization_warnings"),
                    ):
                        item[public] = json.loads(str(item.pop(stored)))
                    item["ogdc_linked"] = bool(item["ogdc_linked"])
                    batch.append(item)
                yield batch
                last_article_id = str(rows[-1]["article_id"])
                if remaining is not None:
                    remaining -= len(rows)

    def get_article_sentiment(
        self,
        article_id: str,
        model_revision_id: str,
        input_policy_id: str,
    ) -> dict[str, Any] | None:
        """Return a reusable result owned by a finalized run, if present.

        Outputs written by an interrupted ``RUNNING`` owner are intentionally
        invisible to global idempotency.  They can only be removed through the
        explicit atomic stale-run discard operation.
        """

        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT output.*
                FROM article_sentiment_outputs AS output
                JOIN sentiment_inference_runs AS owner
                  ON owner.run_id = output.inference_run_id
                WHERE output.article_id = ? AND output.model_revision_id = ?
                  AND output.input_policy_id = ?
                  AND owner.status IN ('PASS', 'PARTIAL')
                  AND owner.completed_at IS NOT NULL
                  AND owner.sealed_payload_inventory_sha256 IS NOT NULL
                  AND owner.run_manifest_contract_sha256 IS NOT NULL
                """,
                (
                    _safe_id(article_id, field="article_id"),
                    _safe_id(model_revision_id, field="model_revision_id"),
                    _safe_id(input_policy_id, field="input_policy_id"),
                ),
            ).fetchone()
        return None if row is None else dict(row)

    def register_sentiment_run_input(
        self,
        run_id: str,
        article_id: str,
        *,
        input_ordinal: int,
    ) -> None:
        """Persist one ordered input before inference or idempotent reuse."""

        identifier = _safe_id(run_id, field="run_id")
        article = _safe_id(article_id, field="article_id")
        if isinstance(input_ordinal, bool) or not isinstance(input_ordinal, int):
            raise ValueError("input_ordinal must be a non-negative integer")
        if input_ordinal < 0:
            raise ValueError("input_ordinal must be a non-negative integer")
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO sentiment_run_articles(
                    inference_run_id, article_id, input_ordinal
                ) VALUES (?, ?, ?)
                """,
                (identifier, article, input_ordinal),
            )

    def complete_sentiment_run_input(
        self,
        run_id: str,
        article_id: str,
        *,
        outcome: str,
        sentiment_output_id: str | None = None,
        error_id: str | None = None,
    ) -> None:
        """Bind one input to its generated/reused output or retained error."""

        identifier = _safe_id(run_id, field="run_id")
        article = _safe_id(article_id, field="article_id")
        normalized_outcome = str(outcome).strip().upper()
        if normalized_outcome not in {"GENERATED", "REUSED", "SKIPPED", "FAILED"}:
            raise ValueError("sentiment input outcome is invalid")
        output = (
            None
            if sentiment_output_id is None
            else _safe_id(sentiment_output_id, field="sentiment_output_id")
        )
        error = None if error_id is None else _safe_id(error_id, field="error_id")
        with self.transaction() as connection:
            result = connection.execute(
                """
                UPDATE sentiment_run_articles
                SET outcome = ?, sentiment_output_id = ?, error_id = ?
                WHERE inference_run_id = ? AND article_id = ?
                  AND outcome = 'PENDING'
                """,
                (normalized_outcome, output, error, identifier, article),
            )
            if result.rowcount != 1:
                raise RuntimeError(
                    "Sentiment run input is missing or already has a terminal outcome"
                )

    def discard_running_sentiment_run(self, run_id: str) -> None:
        """Atomically discard one explicitly selected unfinished run.

        Finalized evidence is never eligible.  Callers must name the exact run
        identity, which keeps recovery deliberate instead of guessing whether
        another process is still active.
        """

        identifier = _safe_id(run_id, field="run_id")
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM sentiment_inference_runs WHERE run_id = ?",
                (identifier,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown sentiment inference run: {identifier}")
            if current["status"] != "RUNNING":
                raise RuntimeError("Only a RUNNING sentiment run can be discarded")
            connection.execute(
                "DELETE FROM sentiment_run_articles WHERE inference_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM sentiment_inference_errors WHERE inference_run_id = ?",
                (identifier,),
            )
            connection.execute(
                "DELETE FROM article_sentiment_outputs WHERE inference_run_id = ?",
                (identifier,),
            )
            result = connection.execute(
                "DELETE FROM sentiment_inference_runs WHERE run_id = ? AND status = 'RUNNING'",
                (identifier,),
            )
            if result.rowcount != 1:
                raise RuntimeError("Could not discard the RUNNING sentiment run")

    def reset_unpublished_sentiment_outputs(self, run_id: str) -> None:
        """Remove generated outputs after artifact construction fails.

        The run and its ordered input rows remain. Generated inputs return to
        ``PENDING`` so the run can be retained as truthful ``FAIL`` evidence
        without leaving an uncertified globally unique result for later reuse.
        """

        identifier = _safe_id(run_id, field="run_id")
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM sentiment_inference_runs WHERE run_id = ?",
                (identifier,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown sentiment inference run: {identifier}")
            if current["status"] != "RUNNING":
                raise RuntimeError(
                    "Only a RUNNING sentiment run can reset unpublished outputs"
                )

            generated_inputs = connection.execute(
                """
                SELECT input.article_id, input.input_ordinal
                FROM sentiment_run_articles AS input
                JOIN article_sentiment_outputs AS output
                  ON output.sentiment_output_id = input.sentiment_output_id
                WHERE input.inference_run_id = ?
                  AND input.outcome = 'GENERATED'
                  AND output.inference_run_id = ?
                ORDER BY input.input_ordinal
                """,
                (identifier, identifier),
            ).fetchall()
            connection.execute(
                """
                DELETE FROM sentiment_run_articles
                WHERE inference_run_id = ? AND outcome = 'GENERATED'
                """,
                (identifier,),
            )
            connection.execute(
                "DELETE FROM article_sentiment_outputs WHERE inference_run_id = ?",
                (identifier,),
            )
            connection.executemany(
                """
                INSERT INTO sentiment_run_articles(
                    inference_run_id, article_id, input_ordinal
                ) VALUES (?, ?, ?)
                """,
                (
                    (identifier, str(row["article_id"]), int(row["input_ordinal"]))
                    for row in generated_inputs
                ),
            )

    def save_article_sentiment(
        self, record: Any | None = None, **values: Any
    ) -> tuple[str, bool]:
        """Persist one output, returning ``(output_id, inserted)`` idempotently."""

        data = _record_mapping(record, values)
        run_id = _safe_id(
            _value(data, "inference_run_id", "run_id"), field="inference_run_id"
        )
        article_id = _safe_id(data.get("article_id"), field="article_id")
        with self.transaction() as connection:
            run = connection.execute(
                "SELECT * FROM sentiment_inference_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown sentiment inference run: {run_id}")
            if run["status"] != "RUNNING":
                raise RuntimeError(f"Sentiment inference run is finalized: {run_id}")
            source_run_id = _safe_id(
                data.get("source_ingestion_run_id", run["source_ingestion_run_id"]),
                field="source_ingestion_run_id",
            )
            model_revision_id = _safe_id(
                data.get("model_revision_id", run["model_revision_id"]),
                field="model_revision_id",
            )
            input_policy_id = _safe_id(
                data.get("input_policy_id", run["input_policy_id"]),
                field="input_policy_id",
            )
            run_context = {
                "source_ingestion_run_id": source_run_id,
                "model_revision_id": model_revision_id,
                "input_policy_id": input_policy_id,
            }
            for field, supplied in run_context.items():
                if supplied != run[field]:
                    raise ValueError(
                        f"{field} does not match the sentiment inference run"
                    )
            article = connection.execute(
                "SELECT * FROM normalized_articles WHERE article_id = ?",
                (article_id,),
            ).fetchone()
            if article is None:
                raise KeyError(f"Unknown canonical article: {article_id}")
            represented = connection.execute(
                """
                SELECT 1 FROM article_ingestions
                WHERE ingestion_run_id = ? AND canonical_article_id = ?
                LIMIT 1
                """,
                (source_run_id, article_id),
            ).fetchone()
            if represented is None:
                raise ValueError(
                    "Article is not represented by the sentiment source ingestion run"
                )

            probabilities = {
                "negative": _probability(
                    data.get("negative_probability"), field="negative_probability"
                ),
                "neutral": _probability(
                    data.get("neutral_probability"), field="neutral_probability"
                ),
                "positive": _probability(
                    data.get("positive_probability"), field="positive_probability"
                ),
            }
            if abs(sum(probabilities.values()) - 1.0) > 0.000001:
                raise ValueError("Sentiment probabilities must sum to one")
            predicted_label = _required_text(data, "predicted_label").lower()
            if predicted_label not in probabilities:
                raise ValueError(
                    "predicted_label must be negative, neutral, or positive"
                )
            predicted_probability = _probability(
                data.get("predicted_probability"), field="predicted_probability"
            )
            if abs(predicted_probability - probabilities[predicted_label]) > 0.000001:
                raise ValueError(
                    "predicted_probability must match the predicted label probability"
                )
            if predicted_probability + 0.000001 < max(probabilities.values()):
                raise ValueError("predicted_label must identify a maximum probability")

            selected_fields_json = _json_text(
                data.get("selected_fields"), default=[]
            )
            class_labels = data.get("class_labels")
            if list(class_labels or ()) != ["negative", "neutral", "positive"]:
                raise ValueError(
                    "class_labels must be negative, neutral, positive in canonical order"
                )
            class_labels_json = _json_text(class_labels, default=[])
            library_versions_json = _json_text(
                data.get("library_versions"), default={}
            )
            warning_codes_json = _json_text(
                _value(data, "warning_codes", "warnings", default=[]), default=[]
            )
            input_digest = _sha256(
                data.get("input_text_sha256"), field="input_text_sha256"
            )
            maximum_sequence_length = _positive_integer(
                data.get(
                    "maximum_sequence_length", run["maximum_sequence_length"]
                ),
                field="maximum_sequence_length",
            )
            if maximum_sequence_length != run["maximum_sequence_length"]:
                raise ValueError(
                    "maximum_sequence_length does not match the sentiment run"
                )
            maximum_input_characters = _positive_integer(
                data.get(
                    "maximum_input_characters",
                    run["maximum_input_characters"],
                ),
                field="maximum_input_characters",
            )
            if maximum_input_characters != run["maximum_input_characters"]:
                raise ValueError(
                    "maximum_input_characters does not match the sentiment run"
                )
            character_count = _positive_integer(
                data.get("character_count"), field="character_count"
            )
            original_character_count = _positive_integer(
                data.get("original_character_count", character_count),
                field="original_character_count",
            )
            if original_character_count < character_count:
                raise ValueError(
                    "original_character_count cannot be smaller than character_count"
                )
            if character_count > maximum_input_characters:
                raise ValueError(
                    "character_count cannot exceed maximum_input_characters"
                )
            character_truncation_applied = _boolean_integer(
                data.get("character_truncation_applied", False),
                field="character_truncation_applied",
            )
            token_truncation_applied = _boolean_integer(
                data.get("token_truncation_applied", False),
                field="token_truncation_applied",
            )
            if character_truncation_applied != int(
                original_character_count > character_count
            ):
                raise ValueError(
                    "character_truncation_applied must match the character counts"
                )
            tokenizer_token_count = _positive_integer(
                data.get("tokenizer_token_count"),
                field="tokenizer_token_count",
            )
            if data.get("final_token_count") is None:
                raise ValueError(
                    "final_token_count is required as observed tokenizer output"
                )
            final_token_count = _positive_integer(
                data.get("final_token_count"),
                field="final_token_count",
            )
            if final_token_count > maximum_sequence_length:
                raise ValueError(
                    "final_token_count cannot exceed maximum_sequence_length"
                )
            if final_token_count > tokenizer_token_count:
                raise ValueError(
                    "final_token_count cannot exceed tokenizer_token_count"
                )
            if data.get("truncation_side") is None:
                raise ValueError(
                    "truncation_side is required as observed tokenizer configuration"
                )
            truncation_side = str(data.get("truncation_side")).strip().lower()
            if truncation_side != "right":
                raise ValueError("truncation_side must be right")
            if final_token_count != min(
                tokenizer_token_count, maximum_sequence_length
            ):
                raise ValueError(
                    "final_token_count must match the post-limit token count"
                )
            if token_truncation_applied != int(
                final_token_count < tokenizer_token_count
            ):
                raise ValueError(
                    "token_truncation_applied must match the tokenizer count and limit"
                )
            truncation_applied = _boolean_integer(
                data.get("truncation_applied"), field="truncation_applied"
            )
            if truncation_applied != int(
                bool(character_truncation_applied or token_truncation_applied)
            ):
                raise ValueError(
                    "truncation_applied must combine character and token truncation"
                )
            semantic_values = {
                "input_text_sha256": input_digest,
                "selected_fields_json": selected_fields_json,
                "character_count": character_count,
                "original_character_count": original_character_count,
                "character_truncation_applied": character_truncation_applied,
                "token_truncation_applied": token_truncation_applied,
                "maximum_input_characters": maximum_input_characters,
                "tokenizer_token_count": tokenizer_token_count,
                "final_token_count": final_token_count,
                "truncation_side": truncation_side,
                "truncation_applied": truncation_applied,
                "maximum_sequence_length": maximum_sequence_length,
                "class_labels_json": class_labels_json,
                "negative_probability": probabilities["negative"],
                "neutral_probability": probabilities["neutral"],
                "positive_probability": probabilities["positive"],
                "predicted_label": predicted_label,
                "predicted_probability": predicted_probability,
            }
            existing = connection.execute(
                """
                SELECT output.*, owner.status AS owner_status
                FROM article_sentiment_outputs AS output
                JOIN sentiment_inference_runs AS owner
                  ON owner.run_id = output.inference_run_id
                WHERE output.article_id = ? AND output.model_revision_id = ?
                  AND output.input_policy_id = ?
                """,
                (article_id, model_revision_id, input_policy_id),
            ).fetchone()
            if existing is not None:
                if (
                    existing["owner_status"] == "RUNNING"
                    and existing["inference_run_id"] != run_id
                ):
                    raise RuntimeError(
                        "Sentiment output is owned by an unfinished run; "
                        "discard or recover that exact run before retrying"
                    )
                for field, expected in semantic_values.items():
                    observed = existing[field]
                    if isinstance(expected, float):
                        if abs(float(observed) - expected) > 0.000001:
                            raise RuntimeError(
                                "Idempotent sentiment output disagrees with persisted evidence"
                            )
                    elif observed != expected:
                        raise RuntimeError(
                            "Idempotent sentiment output disagrees with persisted evidence"
                        )
                return str(existing["sentiment_output_id"]), False

            output_id = _safe_id(
                data.get("sentiment_output_id")
                or _deterministic_id(
                    "SENTOUT", article_id, model_revision_id, input_policy_id
                ),
                field="sentiment_output_id",
            )
            published_at = _timestamp(
                data.get("published_at", article["published_at"]),
                field="published_at",
                nullable=True,
            )
            available_at = _timestamp(
                data.get("available_at", article["available_at"]),
                field="available_at",
            )
            fetched_at = _timestamp(
                data.get("fetched_at", article["fetched_at"]), field="fetched_at"
            )
            assert available_at is not None and fetched_at is not None
            _validate_availability(
                available_at=available_at,
                fetched_at=fetched_at,
                published_at=published_at,
            )
            error_code = _optional_text(data.get("error_code"))
            error_message = _optional_text(data.get("error_message"))
            if (error_code is None) != (error_message is None):
                raise ValueError("error_code and error_message must be supplied together")
            connection.execute(
                """
                INSERT INTO article_sentiment_outputs(
                    sentiment_output_id, schema_version, inference_run_id,
                    article_id, source_ingestion_run_id, model_revision_id,
                    input_policy_id, selected_fields_json, input_text_sha256,
                    character_count, original_character_count,
                    character_truncation_applied, token_truncation_applied,
                    maximum_input_characters, tokenizer_token_count,
                    final_token_count, truncation_side, truncation_applied,
                    maximum_sequence_length,
                    original_language, published_at, available_at, fetched_at,
                    class_labels_json,
                    negative_probability, neutral_probability,
                    positive_probability, predicted_label,
                    predicted_probability, inference_timestamp, device,
                    batch_size, library_versions_json, warning_codes_json,
                    error_code, error_message
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    output_id,
                    str(data.get("schema_version", "1.0")),
                    run_id,
                    article_id,
                    source_run_id,
                    model_revision_id,
                    input_policy_id,
                    selected_fields_json,
                    input_digest,
                    semantic_values["character_count"],
                    semantic_values["original_character_count"],
                    semantic_values["character_truncation_applied"],
                    semantic_values["token_truncation_applied"],
                    semantic_values["maximum_input_characters"],
                    semantic_values["tokenizer_token_count"],
                    semantic_values["final_token_count"],
                    semantic_values["truncation_side"],
                    semantic_values["truncation_applied"],
                    maximum_sequence_length,
                    _optional_text(data.get("original_language")),
                    published_at,
                    available_at,
                    fetched_at,
                    class_labels_json,
                    probabilities["negative"],
                    probabilities["neutral"],
                    probabilities["positive"],
                    predicted_label,
                    predicted_probability,
                    _timestamp(
                        data.get("inference_timestamp", utc_now_z()),
                        field="inference_timestamp",
                    ),
                    _required_text(data, "device")
                    if "device" in data
                    else str(run["device"]),
                    _positive_integer(
                        data.get("batch_size", run["batch_size"]),
                        field="batch_size",
                    ),
                    library_versions_json,
                    warning_codes_json,
                    error_code,
                    error_message,
                ),
            )
        return output_id, True

    def save_sentiment_error(
        self, record: Any | None = None, **values: Any
    ) -> str:
        """Persist one structured, redacted inference error for a running run."""

        data = _record_mapping(record, values)
        run_id = _safe_id(
            _value(data, "inference_run_id", "run_id"), field="inference_run_id"
        )
        with self.transaction() as connection:
            run = connection.execute(
                "SELECT * FROM sentiment_inference_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown sentiment inference run: {run_id}")
            if run["status"] != "RUNNING":
                raise RuntimeError(f"Sentiment inference run is finalized: {run_id}")
            article_id = _optional_text(data.get("article_id"))
            if article_id is not None:
                article_id = _safe_id(article_id, field="article_id")
            stage = _required_text(data, "stage")
            error_type = _required_text(data, "error_type")
            message = _required_text(data, "message")
            error_id = _safe_id(
                data.get("error_id")
                or _deterministic_id(
                    "SENTERR", run_id, article_id, stage, error_type, message
                ),
                field="error_id",
            )
            input_digest = _optional_text(data.get("input_text_sha256"))
            if input_digest is not None:
                input_digest = _sha256(
                    input_digest, field="input_text_sha256"
                )
            connection.execute(
                """
                INSERT INTO sentiment_inference_errors(
                    error_id, schema_version, inference_run_id, article_id,
                    source_ingestion_run_id, model_revision_id, input_policy_id,
                    stage, error_type, message, occurred_at, input_text_sha256,
                    selected_fields_json, original_language, warning_codes_json,
                    retryable, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    error_id,
                    str(data.get("schema_version", "1.0")),
                    run_id,
                    article_id,
                    _safe_id(
                        data.get(
                            "source_ingestion_run_id",
                            run["source_ingestion_run_id"],
                        ),
                        field="source_ingestion_run_id",
                    ),
                    _safe_id(
                        data.get("model_revision_id", run["model_revision_id"]),
                        field="model_revision_id",
                    ),
                    _safe_id(
                        data.get("input_policy_id", run["input_policy_id"]),
                        field="input_policy_id",
                    ),
                    stage,
                    error_type,
                    message,
                    _timestamp(
                        data.get("occurred_at", utc_now_z()), field="occurred_at"
                    ),
                    input_digest,
                    _json_text(data.get("selected_fields"), default=[]),
                    _optional_text(data.get("original_language")),
                    _json_text(
                        _value(data, "warning_codes", "warnings", default=[]),
                        default=[],
                    ),
                    _boolean_integer(
                        data.get("retryable", False), field="retryable"
                    ),
                    _json_text(data.get("details"), default={}),
                ),
            )
        return error_id

    def finalize_sentiment_run(
        self,
        run_id: str,
        *,
        status: str,
        completed_at: str | datetime | None = None,
        summary: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        """Finalize a run from database-derived, checksum-bound evidence."""

        identifier = _safe_id(run_id, field="run_id")
        final_status = str(status).strip().upper()
        if final_status not in {"PASS", "PARTIAL", "FAIL"}:
            raise ValueError("Final sentiment status must be PASS, PARTIAL, or FAIL")
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT * FROM sentiment_inference_runs WHERE run_id = ?",
                (identifier,),
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown sentiment inference run: {identifier}")
            if current["status"] != "RUNNING":
                raise RuntimeError(
                    f"Sentiment inference run is already finalized: {identifier}"
                )
            model_revision = connection.execute(
                """
                SELECT model_repository, model_revision, tokenizer_revision
                FROM sentiment_model_revisions
                WHERE model_revision_id = ?
                """,
                (current["model_revision_id"],),
            ).fetchone()
            if model_revision is None:
                raise RuntimeError("Sentiment run model revision is missing")
            error_count = int(
                connection.execute(
                    """
                    SELECT count(*) FROM sentiment_inference_errors
                    WHERE inference_run_id = ?
                    """,
                    (identifier,),
                ).fetchone()[0]
            )
            output_error_count = int(
                connection.execute(
                    """
                    SELECT count(*) FROM article_sentiment_outputs
                    WHERE inference_run_id = ? AND error_code IS NOT NULL
                    """,
                    (identifier,),
                ).fetchone()[0]
            )
            error_count += output_error_count
            run_error_count = int(
                connection.execute(
                    """
                    SELECT count(*) FROM sentiment_inference_errors
                    WHERE inference_run_id = ? AND article_id IS NULL
                    """,
                    (identifier,),
                ).fetchone()[0]
            )
            if final_status == "PASS" and error_count:
                raise RuntimeError("A PASS sentiment run cannot contain errors")
            if final_status == "PARTIAL" and not error_count:
                raise RuntimeError("A PARTIAL sentiment run requires retained errors")

            evidence = connection.execute(
                """
                SELECT
                    count(*) AS input_count,
                    sum(outcome = 'PENDING') AS pending_count,
                    sum(outcome = 'GENERATED') AS generated_count,
                    sum(outcome = 'REUSED') AS reused_count,
                    sum(outcome = 'SKIPPED') AS skipped_count,
                    sum(outcome = 'FAILED') AS failed_count
                FROM sentiment_run_articles
                WHERE inference_run_id = ?
                """,
                (identifier,),
            ).fetchone()
            assert evidence is not None
            counts = {
                key: int(evidence[key] or 0)
                for key in (
                    "input_count",
                    "pending_count",
                    "generated_count",
                    "reused_count",
                    "skipped_count",
                    "failed_count",
                )
            }
            if final_status in {"PASS", "PARTIAL"}:
                if counts["input_count"] == 0:
                    raise RuntimeError(
                        "A completed sentiment run requires persisted input evidence"
                    )
                if counts["pending_count"]:
                    raise RuntimeError(
                        "A completed sentiment run cannot contain pending inputs"
                    )

            generated_rows = connection.execute(
                """
                SELECT output.predicted_label, output.truncation_applied,
                       output.article_id
                FROM sentiment_run_articles AS input
                JOIN article_sentiment_outputs AS output
                  ON output.sentiment_output_id = input.sentiment_output_id
                WHERE input.inference_run_id = ?
                  AND input.outcome = 'GENERATED'
                """,
                (identifier,),
            ).fetchall()
            label_counts = {label: 0 for label in ("negative", "neutral", "positive")}
            for generated in generated_rows:
                label_counts[str(generated["predicted_label"])] += 1
            ogdc_linked = int(
                connection.execute(
                    """
                    SELECT count(*)
                    FROM sentiment_run_articles AS input
                    WHERE input.inference_run_id = ?
                      AND input.outcome = 'GENERATED'
                      AND EXISTS (
                          SELECT 1
                          FROM sentiment_inference_runs AS run
                          JOIN article_ingestions AS occurrence
                            ON occurrence.ingestion_run_id = run.source_ingestion_run_id
                           AND occurrence.canonical_article_id = input.article_id
                          JOIN article_company_links AS link
                            ON link.occurrence_id = occurrence.occurrence_id
                          WHERE run.run_id = input.inference_run_id
                            AND upper(link.ticker) = 'OGDC'
                      )
                    """,
                    (identifier,),
                ).fetchone()[0]
            )
            completed = _timestamp(
                completed_at or utc_now_z(), field="completed_at"
            )
            normalized_summary = dict(summary or {})
            normalized_summary.update(
                {
                    "summary_schema_version": SENTIMENT_RUN_SUMMARY_SCHEMA_VERSION,
                    "schema_version": str(current["schema_version"]),
                    "run_id": identifier,
                    "status": final_status,
                    "source_ingestion_run_id": str(
                        current["source_ingestion_run_id"]
                    ),
                    "model_repository": str(model_revision["model_repository"]),
                    "model_revision": str(model_revision["model_revision"]),
                    "tokenizer_revision": str(
                        model_revision["tokenizer_revision"]
                    ),
                    "device": str(current["device"]),
                    "artifact_directory": str(current["artifact_directory"]),
                    "completed_at": completed,
                    "articles_requested": counts["input_count"],
                    "unique_canonical_articles": counts["input_count"],
                    "articles_successfully_scored": counts["generated_count"],
                    "articles_skipped": counts["reused_count"]
                    + counts["skipped_count"],
                    "articles_failed": counts["failed_count"],
                    "label_counts": label_counts,
                    "truncated_input_count": sum(
                        int(row["truncation_applied"]) for row in generated_rows
                    ),
                    "ogdc_linked_articles_scored": ogdc_linked,
                    "any_articles_failed": bool(
                        counts["failed_count"]
                        or run_error_count
                        or output_error_count
                    ),
                    "academic_use_only": True,
                    "financial_advice": False,
                }
            )
            payload_seal = normalized_summary.get(
                "sealed_payload_inventory_sha256"
            )
            contract_seal = normalized_summary.get("run_manifest_contract_sha256")
            if final_status in {"PASS", "PARTIAL"}:
                payload_seal = _sha256(
                    payload_seal,
                    field="sealed_payload_inventory_sha256",
                )
                contract_seal = _sha256(
                    contract_seal,
                    field="run_manifest_contract_sha256",
                )
            else:
                payload_seal = (
                    None
                    if payload_seal is None
                    else _sha256(
                        payload_seal,
                        field="sealed_payload_inventory_sha256",
                    )
                )
                contract_seal = (
                    None
                    if contract_seal is None
                    else _sha256(
                        contract_seal,
                        field="run_manifest_contract_sha256",
                    )
                )
            result = connection.execute(
                """
                UPDATE sentiment_inference_runs
                SET completed_at = ?, status = ?, summary_json = ?, error_message = ?,
                    sealed_payload_inventory_sha256 = ?,
                    run_manifest_contract_sha256 = ?
                WHERE run_id = ? AND status = 'RUNNING'
                """,
                (
                    completed,
                    final_status,
                    _json_text(normalized_summary, default={}),
                    _optional_text(error_message),
                    payload_seal,
                    contract_seal,
                    identifier,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError(
                    f"Could not finalize sentiment inference run: {identifier}"
                )

    def get_sentiment_model_revision(
        self, model_revision_id: str
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM sentiment_model_revisions
                WHERE model_revision_id = ?
                """,
                (_safe_id(model_revision_id, field="model_revision_id"),),
            ).fetchone()
        return None if row is None else dict(row)

    def get_sentiment_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM sentiment_inference_runs WHERE run_id = ?",
                (_safe_id(run_id, field="run_id"),),
            ).fetchone()
        return None if row is None else dict(row)

    def get_article_sentiment_by_id(
        self, sentiment_output_id: str
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM article_sentiment_outputs
                WHERE sentiment_output_id = ?
                """,
                (
                    _safe_id(
                        sentiment_output_id, field="sentiment_output_id"
                    ),
                ),
            ).fetchone()
        return None if row is None else dict(row)

    def list_article_sentiments_for_run(
        self, run_id: str
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM article_sentiment_outputs
                WHERE inference_run_id = ? ORDER BY article_id
                """,
                (_safe_id(run_id, field="run_id"),),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_sentiment_errors(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM sentiment_inference_errors
                WHERE inference_run_id = ? ORDER BY error_id
                """,
                (_safe_id(run_id, field="run_id"),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_runs WHERE run_id = ?",
                (_safe_id(run_id, field="run_id"),),
            ).fetchone()
        return None if row is None else dict(row)

    def get_article(self, article_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM normalized_articles WHERE article_id = ?",
                (_safe_id(article_id, field="article_id"),),
            ).fetchone()
        return None if row is None else dict(row)

    def list_canonical_articles(self) -> list[dict[str, Any]]:
        """Return deterministic canonical rows needed by deduplication."""

        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT article_id, canonical_url, content_sha256,
                       title, normalized_title, source_name, published_at,
                       available_at, fetched_at, first_ingestion_run_id,
                       owner.sealed_manifest_sha256 AS owner_manifest_sha256
                FROM normalized_articles AS article
                JOIN ingestion_runs AS owner
                  ON owner.run_id = article.first_ingestion_run_id
                WHERE article.duplicate_of IS NULL
                  AND owner.status IN ('PASS', 'PARTIAL')
                ORDER BY article.article_id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_article_ingestions(self, run_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM article_ingestions
                WHERE ingestion_run_id = ? ORDER BY occurrence_id
                """,
                (_safe_id(run_id, field="run_id"),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_quality_report(self, run_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM quality_reports WHERE ingestion_run_id = ?",
                (_safe_id(run_id, field="run_id"),),
            ).fetchone()
        return None if row is None else dict(row)


# Compatibility aliases for callers that prefer repository/database vocabulary.
NewsRepository = NewsStorage
SQLiteNewsRepository = NewsStorage


__all__ = [
    "DATABASE_SCHEMA_VERSION",
    "NEWS_RECORD_SCHEMA_VERSION",
    "SENTIMENT_RUN_SUMMARY_SCHEMA_VERSION",
    "NewsRepository",
    "NewsStorage",
    "SQLiteNewsRepository",
    "utc_now_z",
]
