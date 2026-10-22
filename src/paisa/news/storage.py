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

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any


DATABASE_SCHEMA_VERSION = 3
NEWS_RECORD_SCHEMA_VERSION = "1.0"
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


# These triggers are deliberately refreshed on every initialization. Schema v3 was
# not released before this integrity tightening, and CREATE TRIGGER IF NOT EXISTS
# would otherwise leave an already-created v3 database with the earlier definitions.
_DEDUPLICATION_INTEGRITY_SQL = """
BEGIN IMMEDIATE;
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
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current > DATABASE_SCHEMA_VERSION:
                raise RuntimeError(
                    "News database schema is newer than this PAISA version: "
                    f"{current} > {DATABASE_SCHEMA_VERSION}"
                )
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(_SCHEMA_SQL)
            null_raw_count = connection.execute(
                """
                SELECT count(*) FROM article_ingestions
                WHERE raw_evidence_id IS NULL
                """
            ).fetchone()[0]
            if null_raw_count:
                raise RuntimeError(
                    "Cannot harden news storage while article ingestions lack "
                    "raw-evidence references"
                )
            connection.executescript(_IMMUTABILITY_SQL)
            connection.executescript(_DEDUPLICATION_INTEGRITY_SQL)
            _validate_existing_deduplication_integrity(connection)
            connection.executemany(
                """
                INSERT OR IGNORE INTO schema_migrations(version, name, applied_at)
                VALUES (?, ?, ?)
                """,
                (
                    (1, "news_ingestion_foundation", utc_now_z()),
                    (2, "news_storage_immutability", utc_now_z()),
                    (3, "news_storage_cross_run_integrity", utc_now_z()),
                ),
            )
            connection.execute(f"PRAGMA user_version = {DATABASE_SCHEMA_VERSION}")
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
        except Exception:
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
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT INTO quality_reports(
                    ingestion_run_id, schema_version, quality_status,
                    created_at, report_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    str(data.get("schema_version", NEWS_RECORD_SCHEMA_VERSION)),
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
            "SELECT * FROM normalized_articles WHERE duplicate_of IS NULL AND ("
            + " OR ".join(clauses)
            + ") ORDER BY article_id"
        )
        with self.connection() as connection:
            return [dict(row) for row in connection.execute(query, parameters)]

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str,
        completed_at: str | datetime | None = None,
        summary: Mapping[str, Any] | None = None,
        error_message: str | None = None,
    ) -> None:
        identifier = _safe_id(run_id, field="run_id")
        final_status = str(status).strip().upper()
        if final_status not in {"PASS", "PARTIAL", "FAIL"}:
            raise ValueError("Final run status must be PASS, PARTIAL, or FAIL")
        with self.transaction() as connection:
            current = connection.execute(
                "SELECT status FROM ingestion_runs WHERE run_id = ?", (identifier,)
            ).fetchone()
            if current is None:
                raise KeyError(f"Unknown ingestion run: {identifier}")
            if current["status"] != "RUNNING":
                raise RuntimeError(f"Ingestion run is already finalized: {identifier}")
            quality = connection.execute(
                """
                SELECT quality_status FROM quality_reports
                WHERE ingestion_run_id = ?
                """,
                (identifier,),
            ).fetchone()
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
                       available_at, fetched_at
                FROM normalized_articles
                WHERE duplicate_of IS NULL
                ORDER BY article_id
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
