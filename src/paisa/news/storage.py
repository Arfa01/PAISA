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


DATABASE_SCHEMA_VERSION = 4
NEWS_RECORD_SCHEMA_VERSION = "1.0"
QUALITY_REPORT_SCHEMA_VERSION = "2.0"
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
}


def _normalized_schema_sql(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


def _schema_stage_sql(version: int) -> str:
    if version == 1:
        return _SCHEMA_SQL
    elif version == 2:
        return _IMMUTABILITY_SQL
    elif version == 3:
        return _DEDUPLICATION_INTEGRITY_SQL
    elif version == 4:
        return _SCHEMA_V4_SQL
    raise ValueError(f"Unsupported news schema stage: {version}")


def _apply_schema_stage(
    connection: sqlite3.Connection, version: int, *, begin: bool = False
) -> None:
    prefix = "BEGIN IMMEDIATE;\n" if begin else ""
    connection.executescript(prefix + _schema_stage_sql(version))


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


def _migrate_schema(
    connection: sqlite3.Connection, from_version: int, to_version: int
) -> None:
    if to_version != from_version + 1:
        raise ValueError("News schema migrations must advance one version at a time")
    _verify_schema(connection, from_version)
    if to_version == 3:
        _validate_existing_deduplication_integrity(connection)
    if to_version == 4:
        _validate_existing_deduplication_integrity(connection)
        _validate_v4_migration_preconditions(connection)
    try:
        _apply_schema_stage(connection, to_version, begin=True)
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, applied_at)
            VALUES (?, ?, ?)
            """,
            (to_version, _MIGRATION_NAMES[to_version], utc_now_z()),
        )
        connection.execute(f"PRAGMA user_version = {to_version}")
        _verify_schema(connection, to_version)
        connection.commit()
    except BaseException:
        connection.rollback()
        raise


def _bootstrap_schema(connection: sqlite3.Connection) -> None:
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
    try:
        _apply_schema_stage(connection, 1, begin=True)
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
                    str(data.get("schema_version", NEWS_RECORD_SCHEMA_VERSION)),
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
            if duplicate_of is None:
                title = _optional_text(data.get("title"))
                connection.execute(
                    """
                    INSERT INTO normalized_articles(
                        article_id, schema_version, provider, source_name,
                        source_type, source_url, canonical_url, title,
                        normalized_title, summary, body, published_at,
                        published_timezone, available_at, fetched_at, language,
                        mentioned_tickers_json, mentioned_companies_json,
                        mentioned_sectors_json, content_sha256, duplicate_of,
                        first_ingestion_run_id, raw_artifact_reference,
                        normalization_warnings_json, created_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        NULL, ?, ?, ?, ?
                    )
                    """,
                    (
                        candidate_id,
                        str(data.get("schema_version", NEWS_RECORD_SCHEMA_VERSION)),
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
                        run_id,
                        raw_reference,
                        _json_text(data.get("normalization_warnings"), default=[]),
                        created_at,
                    ),
                )

            connection.execute(
                """
                INSERT INTO article_ingestions(
                    occurrence_id, ingestion_run_id, raw_evidence_id,
                    candidate_article_id, canonical_article_id, duplicate_of,
                    deduplication_rule, available_at, fetched_at,
                    raw_artifact_reference, normalized_payload_json,
                    normalization_warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    occurrence_id,
                    run_id,
                    raw_evidence_id,
                    candidate_id,
                    canonical_id,
                    canonical_id if duplicate_of is not None else None,
                    deduplication_rule,
                    available_at,
                    fetched_at,
                    raw_reference,
                    _json_text(data, default={}),
                    _json_text(data.get("normalization_warnings"), default=[]),
                    created_at,
                ),
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
            connection.execute(
                """
                INSERT INTO article_ingestions(
                    occurrence_id, ingestion_run_id, raw_evidence_id,
                    candidate_article_id, canonical_article_id, duplicate_of,
                    deduplication_rule, available_at, fetched_at,
                    raw_artifact_reference, normalized_payload_json,
                    normalization_warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
                        _value(data, "normalized_payload", "payload", default=data),
                        default={},
                    ),
                    _json_text(data.get("normalization_warnings"), default=[]),
                    _timestamp(
                        data.get("created_at", utc_now_z()), field="created_at"
                    ),
                ),
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
    "NewsRepository",
    "NewsStorage",
    "SQLiteNewsRepository",
    "utc_now_z",
]
