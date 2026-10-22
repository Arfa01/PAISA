from __future__ import annotations

import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import sqlite3
import tempfile
from typing import Any, Iterable, Mapping, Protocol

from paisa.news.deduplication import deduplicate_articles, title_source_time_key
from paisa.news.entity_linking import link_ogdc_articles
from paisa.news.models import (
    NEWS_SCHEMA_VERSION,
    DuplicateRelationship,
    NormalizedNewsArticle,
    PipelineSummary,
    ProviderError,
    ProviderFetchResult,
    RawNewsRecord,
    utc_iso,
)
from paisa.news.normalization import NormalizationError, normalize_record
from paisa.news.quality import calculate_quality_report
from paisa.news.storage import DATABASE_SCHEMA_VERSION, NewsStorage


_SAFE_RUN_ID = re.compile(
    r"^NEWS-\d{8}-\d{6}-[A-Za-z0-9][A-Za-z0-9._-]*$"
)
_SENSITIVE_CONFIG_KEY = re.compile(
    r"(?:secret|token|password|credential|api[_-]?key|authorization|cookie)",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/@\s]+@")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])((?=[A-Za-z0-9_-]*"
    r"(?:api[-_]?key|access[-_]?key|secret|token|password|credential|"
    r"authorization|cookie)[A-Za-z0-9_-]*\s*[:=])[A-Za-z][A-Za-z0-9_-]*)"
    r"(\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^&,;]+)"
)
_AUTH_TOKEN = re.compile(
    r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+(?:\s+[A-Za-z0-9._~+/=-]+)*"
)
_NORMALIZED_FIELDS = (
    "schema_version",
    "article_id",
    "raw_record_id",
    "provider",
    "source_name",
    "source_type",
    "source_url",
    "canonical_url",
    "title",
    "summary",
    "body",
    "published_at",
    "published_timezone",
    "available_at",
    "fetched_at",
    "language",
    "mentioned_tickers",
    "mentioned_companies",
    "mentioned_sectors",
    "content_sha256",
    "duplicate_of",
    "ingestion_run_id",
    "raw_artifact_reference",
    "normalization_warnings",
    "data_class",
)
_LINK_FIELDS = (
    "link_id",
    "article_id",
    "ticker",
    "company_name",
    "matched_alias",
    "match_method",
    "match_location",
    "confidence",
)
_DUPLICATE_FIELDS = (
    "article_id",
    "duplicate_of",
    "deduplication_rule",
)
_FINALIZATION_JOURNAL = ".finalization-journal.json"


class NewsProvider(Protocol):
    provider_id: str
    data_class: str

    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult: ...


class NewsRunCollisionError(FileExistsError):
    """The immutable run identity already exists in artifacts or storage."""


class NewsRunFinalizationError(RuntimeError):
    """A staged artifact pack could not be durably finalized in storage."""


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def make_news_run_id(now: datetime | None = None) -> str:
    instant = _aware_utc(now or datetime.now(timezone.utc), field="run ID time")
    return f"NEWS-{instant.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"


def validate_news_run_id(run_id: str) -> str:
    value = run_id.strip()
    if not _SAFE_RUN_ID.fullmatch(value):
        raise ValueError(
            "News run IDs must use NEWS-YYYYMMDD-HHMMSS-<unique-suffix> and "
            "contain only safe letters, digits, '.', '_', and '-' in the suffix."
        )
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc_iso(value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _atomic_write_json(path: Path, value: Any) -> None:
    """Replace a JSON marker atomically within its existing directory."""

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        _write_json(temporary, value)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set, Mapping)):
        return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(value: str, fallback: str) -> str:
    name = Path(value).name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-")
    return safe or fallback


def _storage_raw_id(run_id: str, record: RawNewsRecord) -> str:
    payload = "\0".join((run_id, record.provider_id, record.raw_record_id))
    return f"NEWSRAW-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _occurrence_id(run_id: str, article_id: str) -> str:
    payload = "\0".join((run_id, article_id))
    return f"NEWSOCC-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def _redact_sensitive_text(value: str) -> str:
    redacted = _URL_USERINFO.sub(r"\1[REDACTED]@", value)
    redacted = _AUTH_TOKEN.sub(r"\1 [REDACTED]", redacted)
    redacted = _SENSITIVE_ASSIGNMENT.sub(r"\1\2[REDACTED]", redacted)
    return redacted


def _sanitize_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if _SENSITIVE_CONFIG_KEY.search(str(key)):
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = _sanitize_config(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_config(item) for item in value]
    if isinstance(value, str):
        return _redact_sensitive_text(value)
    return _jsonable(value)


def _sanitize_provider_result(result: ProviderFetchResult) -> ProviderFetchResult:
    captures = tuple(
        replace(
            capture,
            configured_source=_redact_sensitive_text(capture.configured_source),
            source_name=_redact_sensitive_text(capture.source_name),
            filename=_redact_sensitive_text(capture.filename),
            warnings=tuple(_redact_sensitive_text(item) for item in capture.warnings),
        )
        for capture in result.captures
    )
    records = tuple(
        replace(
            record,
            configured_source=_redact_sensitive_text(record.configured_source),
            source_url=(
                _redact_sensitive_text(record.source_url)
                if record.source_url is not None
                else None
            ),
            source_name=_redact_sensitive_text(record.source_name),
            provider_record_id=(
                _redact_sensitive_text(record.provider_record_id)
                if record.provider_record_id is not None
                else None
            ),
            raw_payload=_sanitize_config(record.raw_payload),
            provider_warnings=tuple(
                _redact_sensitive_text(item) for item in record.provider_warnings
            ),
        )
        for record in result.records
    )
    errors = tuple(
        replace(
            error,
            configured_source=_redact_sensitive_text(error.configured_source),
            message=_redact_sensitive_text(error.message),
        )
        for error in result.errors
    )
    return replace(
        result,
        configured_sources=tuple(
            _sanitize_config(item) for item in result.configured_sources
        ),
        records=records,
        captures=captures,
        errors=errors,
        warnings=tuple(_redact_sensitive_text(item) for item in result.warnings),
    )


def _validate_provider_result(
    provider: NewsProvider,
    result: ProviderFetchResult,
    *,
    run_started_at: datetime,
) -> None:
    """Reject internally contradictory adapter evidence before publication."""

    expected_provider = str(getattr(provider, "provider_id", "unknown"))
    expected_class = str(getattr(provider, "data_class", "real"))
    if result.provider_id != expected_provider:
        raise ValueError("Provider result ID does not match the selected adapter")
    if result.data_class != expected_class:
        raise ValueError("Provider result data class does not match the selected adapter")
    run_started = _aware_utc(run_started_at, field="run started_at")
    completed_at = _aware_utc(result.fetched_at, field="provider result fetched_at")
    if completed_at < run_started:
        raise ValueError("Provider completion cannot precede run start")

    configured = getattr(provider, "configured_sources", ())
    if callable(configured):
        configured = configured()
    adapter_sources = tuple(configured) if configured else ()
    if _jsonable(adapter_sources) != _jsonable(result.configured_sources):
        raise ValueError("Provider result sources do not match the adapter configuration")
    allowed_sources = {
        str(entry[key])
        for entry in adapter_sources
        if isinstance(entry, Mapping)
        for key in ("url", "path", "source", "configured_source")
        if key in entry and entry[key] is not None
    }

    capture_by_id: dict[str, Any] = {}
    for capture in result.captures:
        if capture.capture_id in capture_by_id:
            raise ValueError("Provider result contains duplicate capture IDs")
        if capture.provider_id != result.provider_id:
            raise ValueError("Capture provider ID does not match the provider result")
        if capture.data_class != result.data_class:
            raise ValueError("Capture data class does not match the provider result")
        capture_time = _aware_utc(capture.fetched_at, field="capture fetched_at")
        if not run_started <= capture_time <= completed_at:
            raise ValueError("Capture fetched_at is outside the ingestion run")
        if capture.configured_source not in allowed_sources:
            raise ValueError("Capture source is not declared by the selected adapter")
        observed_checksum = _sha256_bytes(capture.content)
        if capture.response_sha256 != observed_checksum:
            raise ValueError("Capture checksum does not match the captured bytes")
        capture_by_id[capture.capture_id] = capture

    raw_ids: set[str] = set()
    for record in result.records:
        if record.raw_record_id in raw_ids:
            raise ValueError("Provider result contains duplicate raw record IDs")
        raw_ids.add(record.raw_record_id)
        if record.provider_id != result.provider_id:
            raise ValueError("Raw record provider ID does not match the provider result")
        if record.data_class != result.data_class:
            raise ValueError("Raw record data class does not match the provider result")
        record_time = _aware_utc(record.fetched_at, field="record fetched_at")
        if not run_started <= record_time <= completed_at:
            raise ValueError("Raw record fetched_at is outside the ingestion run")
        if record.capture_id is None or record.capture_id not in capture_by_id:
            raise ValueError("Every raw record must reference a retained capture")
        capture = capture_by_id[record.capture_id]
        if record.fetched_at != capture.fetched_at:
            raise ValueError("Raw record and capture fetched_at values differ")
        if record.configured_source != capture.configured_source:
            raise ValueError("Raw record and capture configured sources differ")
        if record.http_status != capture.http_status:
            raise ValueError("Raw record and capture HTTP statuses differ")
        if record.response_content_type != capture.content_type:
            raise ValueError("Raw record and capture content types differ")
        if record.response_sha256 != capture.response_sha256:
            raise ValueError("Raw record and capture checksums differ")

    for error in result.errors:
        if error.provider_id != result.provider_id:
            raise ValueError("Provider error ID does not match the provider result")
        error_time = _aware_utc(error.occurred_at, field="provider error occurred_at")
        if not run_started <= error_time <= completed_at:
            raise ValueError("Provider error time is outside the ingestion run")
        if error.configured_source not in allowed_sources:
            raise ValueError("Provider error source is not declared by the adapter")


def _provider_failure_result(
    provider: NewsProvider, *, fetched_at: datetime, error: Exception
) -> ProviderFetchResult:
    provider_id = str(getattr(provider, "provider_id", "unknown"))
    data_class = str(getattr(provider, "data_class", "real"))
    if data_class not in {"real", "imported", "fixture"}:
        data_class = "real"
    configured = getattr(provider, "configured_sources", ())
    if callable(configured):
        configured = configured()
    configured_sources = tuple(configured) if configured else ()
    return ProviderFetchResult(
        provider_id=provider_id,
        data_class=data_class,  # type: ignore[arg-type]
        fetched_at=fetched_at,
        configured_sources=configured_sources,
        errors=(
            ProviderError(
                provider_id=provider_id,
                configured_source="provider",
                error_type=type(error).__name__,
                message=str(error),
                occurred_at=fetched_at,
            ),
        ),
    )


def _match_existing_duplicate(
    storage: NewsStorage, article: NormalizedNewsArticle
) -> tuple[str, str] | None:
    candidates = storage.find_duplicate_candidates(
        canonical_url=article.canonical_url,
        content_sha256=article.content_sha256,
        title=article.title,
        source_name=article.source_name,
        published_at=article.published_at,
        time_tolerance_seconds=0,
    )
    incoming_title_key = title_source_time_key(
        title=article.title,
        source_name=article.source_name,
        published_at=article.published_at,
    )
    for rule in ("canonical_url", "content_sha256", "title_source_time"):
        matches: list[str] = []
        for candidate in candidates:
            if rule == "canonical_url" and article.canonical_url:
                matched = candidate.get("canonical_url") == article.canonical_url
            elif rule == "content_sha256":
                matched = candidate.get("content_sha256") == article.content_sha256
            else:
                candidate_published = candidate.get("published_at")
                if incoming_title_key is None or not candidate_published:
                    matched = False
                else:
                    parsed = datetime.fromisoformat(
                        str(candidate_published).replace("Z", "+00:00")
                    )
                    matched = incoming_title_key == title_source_time_key(
                        title=str(candidate.get("title") or ""),
                        source_name=str(candidate.get("source_name") or ""),
                        published_at=parsed,
                    )
            if matched:
                matches.append(str(candidate["article_id"]))
        if matches:
            return sorted(matches)[0], rule
    return None


def _deduplicate_with_storage(
    storage: NewsStorage, articles: list[NormalizedNewsArticle]
) -> tuple[list[NormalizedNewsArticle], list[DuplicateRelationship]]:
    resolved: list[NormalizedNewsArticle] = []
    relationships: list[DuplicateRelationship] = []
    local_canonical: list[NormalizedNewsArticle] = []
    for article in articles:
        existing = _match_existing_duplicate(storage, article)
        if existing is not None:
            canonical_id, rule = existing
            resolved.append(replace(article, duplicate_of=canonical_id))
            relationships.append(
                DuplicateRelationship(
                    article_id=article.article_id,
                    canonical_article_id=canonical_id,
                    rule=rule,  # type: ignore[arg-type]
                )
            )
            continue
        local_result, local_relationships = deduplicate_articles(
            [article], existing_articles=local_canonical
        )
        resolved_article = local_result[0]
        resolved.append(resolved_article)
        relationships.extend(local_relationships)
        if resolved_article.duplicate_of is None:
            local_canonical.append(resolved_article)
    return resolved, relationships


def _artifact_inventory(staging_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(staging_dir.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file() and path.name not in {
            "run_manifest.json",
            _FINALIZATION_JOURNAL,
        }:
            inventory.append(
                {
                    "path": path.relative_to(staging_dir).as_posix(),
                    "sha256": _sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    return inventory


def _publish_staged_tree(staging: Path, destination: Path) -> None:
    """Move a complete staged tree into an exclusively reserved run directory."""

    for child in sorted(staging.iterdir(), key=lambda item: item.name):
        os.rename(child, destination / child.name)
    staging.rmdir()


def _summary_from_payload(value: Mapping[str, Any]) -> PipelineSummary:
    """Reconstruct a summary from a trusted, run-local recovery journal."""

    date_range = value.get("date_range")
    if not isinstance(date_range, Mapping):
        raise ValueError("Finalization journal summary has an invalid date range")
    return PipelineSummary(
        schema_version=str(value["schema_version"]),
        run_id=str(value["run_id"]),
        run_status=str(value["run_status"]),  # type: ignore[arg-type]
        quality_status=str(value["quality_status"]),  # type: ignore[arg-type]
        provider=str(value["provider"]),
        provider_mode=str(value["provider_mode"]),  # type: ignore[arg-type]
        source_count=int(value["source_count"]),
        raw_record_count=int(value["raw_record_count"]),
        normalized_record_count=int(value["normalized_record_count"]),
        duplicate_count=int(value["duplicate_count"]),
        ogdc_linked_article_count=int(value["ogdc_linked_article_count"]),
        missing_publication_time_count=int(value["missing_publication_time_count"]),
        date_range={str(key): item for key, item in date_range.items()},
        artifact_directory=str(value["artifact_directory"]),
        database_path=str(value["database_path"]),
        limitations=tuple(str(item) for item in value.get("limitations", ())),
    )


def _write_committed_markers(
    destination: Path,
    *,
    summary: PipelineSummary,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Durably stage final artifact markers before committing SQLite."""

    _atomic_write_json(destination / "final_summary.json", summary.to_dict())
    committed_manifest = dict(manifest)
    committed_manifest.update(
        {
            "status": summary.run_status,
            "finalization_state": "COMMITTED",
            "artifact_inventory": _artifact_inventory(destination),
        }
    )
    _atomic_write_json(destination / "run_manifest.json", committed_manifest)
    return committed_manifest


def _write_retryable_markers(
    destination: Path,
    *,
    summary: PipelineSummary,
    manifest: Mapping[str, Any],
    error: str,
) -> None:
    """Keep disk and SQLite pending when artifact marker publication fails."""

    pending_summary = summary.to_dict()
    pending_summary.update(
        {
            "run_status": "PENDING",
            "finalization_state": "ARTIFACT_COMMIT_RETRY_REQUIRED",
            "finalization_error": error,
        }
    )
    _atomic_write_json(destination / "final_summary.json", pending_summary)
    pending_manifest = dict(manifest)
    pending_manifest.update(
        {
            "status": "PENDING",
            "finalization_state": "ARTIFACT_COMMIT_RETRY_REQUIRED",
            "finalization_error": error,
            "recovery_journal": _FINALIZATION_JOURNAL,
            "artifact_inventory": _artifact_inventory(destination),
        }
    )
    _atomic_write_json(destination / "run_manifest.json", pending_manifest)


def _write_database_failure_markers(
    destination: Path,
    *,
    summary: PipelineSummary,
    manifest: Mapping[str, Any],
    error: str,
) -> None:
    """Publish a truthful failure when SQLite cannot be finalized."""

    journal = destination / _FINALIZATION_JOURNAL
    failed_summary = summary.to_dict()
    failed_summary.update(
        {
            "run_status": "FAIL",
            "finalization_state": "DATABASE_FINALIZATION_FAILED",
            "error": error,
        }
    )
    _atomic_write_json(destination / "final_summary.json", failed_summary)
    failed_manifest = dict(manifest)
    failed_manifest.update(
        {
            "status": "FAIL",
            "finalization_state": "DATABASE_FINALIZATION_FAILED",
            "finalization_error": error,
            "artifact_inventory": _artifact_inventory(destination),
        }
    )
    _atomic_write_json(destination / "run_manifest.json", failed_manifest)
    journal.unlink(missing_ok=True)


def _compensate_database_finalization_failure(
    destination: Path,
    *,
    summary: PipelineSummary,
    manifest: Mapping[str, Any],
    cause: Exception,
) -> None:
    """Write FAIL markers, retaining a retry journal if compensation also fails."""

    finalization_error = _redact_sensitive_text(
        f"{type(cause).__name__}: {cause}"
    )
    try:
        _write_database_failure_markers(
            destination,
            summary=summary,
            manifest=manifest,
            error=finalization_error,
        )
    except Exception as marker_exc:
        marker_error = _redact_sensitive_text(
            f"{type(marker_exc).__name__}: {marker_exc}"
        )
        try:
            _write_retryable_markers(
                destination,
                summary=summary,
                manifest=manifest,
                error=(
                    f"database finalization failed ({finalization_error}); "
                    f"failure marker write failed ({marker_error})"
                ),
            )
        except Exception:
            pass
        raise NewsRunFinalizationError(
            "Database finalization and its failure markers remain pending; "
            "the recovery journal was retained for a safe retry"
        ) from marker_exc
    raise NewsRunFinalizationError(
        "News evidence was published as FAIL because durable database "
        "finalization did not complete"
    ) from cause


def _recover_finalization(
    destination: Path,
    *,
    storage: NewsStorage,
    run_id: str,
) -> PipelineSummary:
    """Idempotently finish an interrupted marker/database commit protocol."""

    journal_path = destination / _FINALIZATION_JOURNAL
    try:
        payload = json.loads(journal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NewsRunFinalizationError(
            "The news run has an unreadable finalization recovery journal"
        ) from exc
    if not isinstance(payload, Mapping) or payload.get("run_id") != run_id:
        raise NewsRunFinalizationError(
            "The news run finalization journal does not match its run identity"
        )
    summary_payload = payload.get("summary")
    manifest_payload = payload.get("manifest")
    if not isinstance(summary_payload, Mapping) or not isinstance(
        manifest_payload, Mapping
    ):
        raise NewsRunFinalizationError(
            "The news run finalization journal is incomplete"
        )
    summary = _summary_from_payload(summary_payload)
    if summary.run_id != run_id or payload.get("desired_status") != summary.run_status:
        raise NewsRunFinalizationError(
            "The news run finalization journal has contradictory status evidence"
        )

    row = storage.get_run(run_id)
    if row is None:
        raise NewsRunFinalizationError(
            "The news run finalization journal has no matching database run"
        )
    database_status = str(row["status"])
    if database_status == "RUNNING":
        try:
            _write_committed_markers(
                destination,
                summary=summary,
                manifest=manifest_payload,
            )
        except Exception as exc:
            error = _redact_sensitive_text(f"{type(exc).__name__}: {exc}")
            try:
                _write_retryable_markers(
                    destination,
                    summary=summary,
                    manifest=manifest_payload,
                    error=error,
                )
            except Exception:
                pass
            raise NewsRunFinalizationError(
                "Artifact finalization remains pending and can be retried safely"
            ) from exc
        try:
            storage.finalize_run(
                run_id,
                status=summary.run_status,
                completed_at=payload.get("completed_at"),
                summary=summary.to_dict(),
                error_message=payload.get("database_error"),
            )
        except Exception as exc:
            current = storage.get_run(run_id)
            if current is None or current["status"] != summary.run_status:
                _compensate_database_finalization_failure(
                    destination,
                    summary=summary,
                    manifest=manifest_payload,
                    cause=exc,
                )
    elif database_status == summary.run_status:
        _write_committed_markers(
            destination,
            summary=summary,
            manifest=manifest_payload,
        )
    else:
        raise NewsRunFinalizationError(
            "The recovery journal conflicts with the finalized database status"
        )

    journal_path.unlink(missing_ok=True)
    return summary


def run_news_pipeline(
    provider: NewsProvider,
    *,
    output_dir: str | Path = "artifacts/news_ingestion",
    database_path: str | Path = "data/news/paisa_news.sqlite3",
    run_id: str | None = None,
    started_at: datetime | None = None,
) -> PipelineSummary:
    """Acquire and publish one immutable, evidence-preserving news run."""

    started = _aware_utc(
        started_at or datetime.now(timezone.utc), field="started_at"
    )
    selected_run_id = validate_news_run_id(run_id or make_news_run_id(started))
    output_root = Path(output_dir).expanduser().resolve()
    destination = output_root / selected_run_id
    if os.path.lexists(destination):
        journal_path = destination / _FINALIZATION_JOURNAL
        if (
            not destination.is_symlink()
            and destination.is_dir()
            and journal_path.is_file()
        ):
            recovery_storage = NewsStorage(database_path).initialize()
            return _recover_finalization(
                destination,
                storage=recovery_storage,
                run_id=selected_run_id,
            )
        raise NewsRunCollisionError(f"News run destination already exists: {destination}")

    storage = NewsStorage(database_path).initialize()
    if storage.get_run(selected_run_id) is not None:
        raise NewsRunCollisionError(
            f"News run ID already exists in storage: {selected_run_id}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    try:
        # The exact run directory is the exclusive reservation.  This closes
        # dangling-symlink and cooperating-writer races without ever replacing
        # an existing destination.
        destination.mkdir()
    except FileExistsError as exc:
        raise NewsRunCollisionError(
            f"News run destination already exists: {destination}"
        ) from exc
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination))
    (staging / "raw").mkdir()

    provider_id = str(getattr(provider, "provider_id", "unknown"))
    provider_mode = str(getattr(provider, "data_class", "real"))
    run_started_in_db = False
    published = False

    try:
        try:
            storage.begin_run(
                run_id=selected_run_id,
                schema_version=NEWS_SCHEMA_VERSION,
                provider_id=provider_id,
                provider_mode=provider_mode,
                configured_source="provider_config.json",
                data_kind=provider_mode,
                started_at=started,
                artifact_directory=str(destination),
            )
        except sqlite3.IntegrityError as exc:
            if storage.get_run(selected_run_id) is not None:
                raise NewsRunCollisionError(
                    f"News run ID appeared concurrently in storage: {selected_run_id}"
                ) from exc
            raise
        run_started_in_db = True

        try:
            result = provider.fetch()
            _validate_provider_result(
                provider, result, run_started_at=started
            )
        except Exception as exc:
            result = _provider_failure_result(
                provider,
                fetched_at=datetime.now(timezone.utc),
                error=exc,
            )
        result = _sanitize_provider_result(result)

        provider_id = result.provider_id
        provider_mode = result.data_class
        _write_json(
            staging / "provider_config.json",
            {
                "schema_version": NEWS_SCHEMA_VERSION,
                "provider_id": result.provider_id,
                "provider_mode": result.data_class,
                "fetched_at": result.fetched_at,
                "configured_sources": _sanitize_config(result.configured_sources),
                "provider_options": _sanitize_config(
                    getattr(provider, "effective_config", {})
                ),
                "warnings": list(result.warnings),
            },
        )

        capture_entries: list[dict[str, Any]] = []
        for index, capture in enumerate(result.captures, start=1):
            filename = _safe_filename(capture.filename, f"capture-{index}.bin")
            relative = Path("raw") / f"capture-{index:04d}-{filename}"
            (staging / relative).write_bytes(capture.content)
            metadata = capture.metadata_dict()
            metadata["artifact_reference"] = relative.as_posix()
            metadata["observed_sha256"] = _sha256_bytes(capture.content)
            capture_entries.append(metadata)

        normalized: list[NormalizedNewsArticle] = []
        rejected: list[dict[str, str]] = []
        stored_raw: list[tuple[RawNewsRecord, str]] = []
        raw_index: list[dict[str, Any]] = []
        for index, original in enumerate(result.records, start=1):
            storage_raw_id = _storage_raw_id(selected_run_id, original)
            record = replace(
                original,
                raw_record_id=storage_raw_id,
                provider_record_id=original.provider_record_id or original.raw_record_id,
            )
            relative = Path("raw") / f"record-{index:06d}.json"
            raw_payload = original.metadata_dict()
            raw_payload["storage_raw_record_id"] = storage_raw_id
            _write_json(staging / relative, raw_payload)
            raw_index.append(
                {
                    "raw_record_id": storage_raw_id,
                    "provider_record_id": record.provider_record_id,
                    "artifact_reference": relative.as_posix(),
                }
            )
            stored_raw.append((record, relative.as_posix()))
            try:
                normalized.append(
                    normalize_record(
                        record,
                        ingestion_run_id=selected_run_id,
                        raw_artifact_reference=relative.as_posix(),
                    )
                )
            except NormalizationError as exc:
                rejected.append(
                    {"raw_record_id": storage_raw_id, "reason": str(exc)}
                )

        _write_json(
            staging / "raw" / "index.json",
            {
                "schema_version": NEWS_SCHEMA_VERSION,
                "captures": capture_entries,
                "records": raw_index,
            },
        )

        deduplicated, duplicate_relationships = _deduplicate_with_storage(
            storage, normalized
        )
        linked_articles, entity_links = link_ogdc_articles(deduplicated)
        quality = calculate_quality_report(
            raw_records=[record for record, _ in stored_raw],
            normalized_articles=linked_articles,
            duplicate_relationships=duplicate_relationships,
            entity_links=entity_links,
            rejected_records=rejected,
            provider_errors=result.errors,
        )

        relationship_by_article = {
            item.article_id: item for item in duplicate_relationships
        }
        storage_error: str | None = None
        try:
            for record, reference in stored_raw:
                storage.save_raw(
                    record,
                    ingestion_run_id=selected_run_id,
                    data_kind=record.data_class,
                    raw_artifact_reference=reference,
                    payload=record.raw_payload,
                )
            for article in linked_articles:
                relationship = relationship_by_article.get(article.article_id)
                occurrence_id = _occurrence_id(selected_run_id, article.article_id)
                storage.save_article(
                    article,
                    occurrence_id=occurrence_id,
                    deduplication_rule=relationship.rule if relationship else None,
                )
            for relationship in duplicate_relationships:
                storage.save_dedup(
                    relationship,
                    ingestion_run_id=selected_run_id,
                    occurrence_id=_occurrence_id(
                        selected_run_id, relationship.article_id
                    ),
                )
            for link in entity_links:
                linked_article = next(
                    item for item in linked_articles if item.article_id == link.article_id
                )
                storage.save_link(
                    link,
                    canonical_article_id=(
                        linked_article.duplicate_of or linked_article.article_id
                    ),
                    occurrence_id=_occurrence_id(
                        selected_run_id, linked_article.article_id
                    ),
                )
            for error in result.errors:
                storage.save_error(error, ingestion_run_id=selected_run_id)
            storage.save_quality(
                ingestion_run_id=selected_run_id,
                schema_version=NEWS_SCHEMA_VERSION,
                quality_status=quality["quality_status"],
                report=quality,
            )
        except Exception as exc:
            storage_error = _redact_sensitive_text(
                f"{type(exc).__name__}: {exc}"
            )
            quality = dict(quality)
            quality["quality_status"] = "FAIL"
            quality["minimum_quality_status"] = "FAIL"
            quality.setdefault("quality_failures", []).append("database_persistence_failed")
            quality["database_error"] = storage_error

        _write_csv(
            staging / "normalized_articles.csv",
            [article.to_dict() for article in linked_articles],
            _NORMALIZED_FIELDS,
        )
        _write_csv(
            staging / "article_entity_links.csv",
            [link.to_dict() for link in entity_links],
            _LINK_FIELDS,
        )
        _write_csv(
            staging / "duplicates.csv",
            [item.to_dict() for item in duplicate_relationships],
            _DUPLICATE_FIELDS,
        )
        _write_json(
            staging / "provider_errors.json",
            {
                "schema_version": NEWS_SCHEMA_VERSION,
                "run_id": selected_run_id,
                "errors": [error.to_dict() for error in result.errors],
            },
        )
        _write_json(staging / "quality_report.json", quality)

        if quality["quality_status"] == "FAIL" or storage_error:
            run_status = "FAIL"
        elif result.errors or quality["quality_status"] == "WARN":
            run_status = "PARTIAL"
        else:
            run_status = "PASS"
        completed_at = datetime.now(timezone.utc)
        date_range = {
            "start": quality.get("earliest_publication_time"),
            "end": quality.get("latest_publication_time"),
        }
        limitations = (
            "This ingestion output contains no sentiment scores, model predictions, "
            "trading decisions, or financial advice.",
            "RSS coverage is limited to fields supplied by configured feeds; manual "
            "CSV rows are imported evidence and are not live acquisitions.",
        )
        summary = PipelineSummary(
            schema_version=NEWS_SCHEMA_VERSION,
            run_id=selected_run_id,
            run_status=run_status,  # type: ignore[arg-type]
            quality_status=quality["quality_status"],
            provider=provider_id,
            provider_mode=provider_mode,  # type: ignore[arg-type]
            source_count=len(result.configured_sources),
            raw_record_count=len(result.records),
            normalized_record_count=len(linked_articles),
            duplicate_count=len(duplicate_relationships),
            ogdc_linked_article_count=int(quality["ogdc_linked_article_count"]),
            missing_publication_time_count=int(
                quality["missing_publication_time_count"]
            ),
            date_range=date_range,
            artifact_directory=str(destination),
            database_path=str(Path(database_path).expanduser().resolve()),
            limitations=limitations,
        )
        pending_summary = summary.to_dict()
        pending_summary["run_status"] = "PENDING"
        pending_summary["finalization_state"] = "PENDING_DATABASE_FINALIZATION"
        _write_json(staging / "final_summary.json", pending_summary)
        manifest = {
            "schema_version": NEWS_SCHEMA_VERSION,
            "run_id": selected_run_id,
            "producer": "paisa.news.pipeline.run_news_pipeline",
            "status": "PENDING",
            "finalization_state": "PENDING_DATABASE_FINALIZATION",
            "quality_status": quality["quality_status"],
            "started_at": utc_iso(started),
            "completed_at": utc_iso(completed_at),
            "provider": provider_id,
            "provider_mode": provider_mode,
            "source_count": len(result.configured_sources),
            "record_counts": {
                "raw": len(result.records),
                "normalized": len(linked_articles),
                "duplicates": len(duplicate_relationships),
                "rejected": len(rejected),
                "provider_errors": len(result.errors),
            },
            "database": {
                "engine": "sqlite",
                "schema_version": DATABASE_SCHEMA_VERSION,
                "configured_path": str(database_path),
            },
            "artifact_inventory": _artifact_inventory(staging),
            "limitations": list(limitations),
        }
        _write_json(staging / "run_manifest.json", manifest)

        journal = {
            "schema_version": NEWS_SCHEMA_VERSION,
            "run_id": selected_run_id,
            "desired_status": run_status,
            "completed_at": utc_iso(completed_at),
            "database_error": storage_error,
            "summary": summary.to_dict(),
            "manifest": manifest,
        }
        _write_json(staging / _FINALIZATION_JOURNAL, journal)

        _publish_staged_tree(staging, destination)
        published = True

        try:
            _write_committed_markers(
                destination,
                summary=summary,
                manifest=manifest,
            )
        except Exception as exc:
            marker_error = _redact_sensitive_text(f"{type(exc).__name__}: {exc}")
            try:
                _write_retryable_markers(
                    destination,
                    summary=summary,
                    manifest=manifest,
                    error=marker_error,
                )
            except Exception:
                pass
            raise NewsRunFinalizationError(
                "Artifact finalization is pending and can be retried safely; "
                "the database run remains RUNNING"
            ) from exc

        if run_started_in_db:
            try:
                storage.finalize_run(
                    selected_run_id,
                    status=run_status,
                    completed_at=completed_at,
                    summary=summary.to_dict(),
                    error_message=storage_error,
                )
            except Exception as exc:
                current = storage.get_run(selected_run_id)
                if current is None or current["status"] != run_status:
                    _compensate_database_finalization_failure(
                        destination,
                        summary=summary,
                        manifest=manifest,
                        cause=exc,
                    )

        (destination / _FINALIZATION_JOURNAL).unlink(missing_ok=True)
        return summary
    except NewsRunFinalizationError:
        raise
    except Exception:
        if run_started_in_db:
            try:
                storage.finalize_run(
                    selected_run_id,
                    status="FAIL",
                    error_message="Pipeline aborted before immutable publication.",
                )
            except Exception:
                pass
        raise
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
        if not published and destination.exists():
            try:
                destination.rmdir()
            except OSError:
                # The reservation may contain partially moved, newly-created
                # files after a local I/O failure.  They are not a completed
                # run because no final manifest was committed; remove only this
                # exact run directory owned by the current invocation.
                shutil.rmtree(destination)


__all__ = [
    "NewsRunCollisionError",
    "NewsRunFinalizationError",
    "make_news_run_id",
    "run_news_pipeline",
    "validate_news_run_id",
]
