from __future__ import annotations

import csv
from contextlib import contextmanager
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
import stat
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import fcntl

from paisa.news.deduplication import title_source_time_key
from paisa.news.entity_linking import link_ogdc_articles
from paisa.news.models import (
    NEWS_AVAILABILITY_POLICY_VERSION,
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
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?(?:key|token)|token|secret|password|"
    r"credential|authorization|auth|signature|signed|session|cookie|jwt)(?:$|[_-])",
    re.IGNORECASE,
)
_URL_USERINFO = re.compile(r"(?i)(https?://)[^/@\s]+@")
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?<![A-Za-z0-9_-])((?:[A-Za-z0-9]+[-_])*"
    r"(?:api[-_]?key|access[-_]?(?:key|token)|secret|token|password|credential|"
    r"authorization|auth|signature|signed|session|cookie|jwt)"
    r"(?:[-_][A-Za-z0-9]+)*)"
    r"(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^&,;]+)"
)
_AUTH_TOKEN = re.compile(
    r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+(?:\s+[A-Za-z0-9._~+/=-]+)*"
)
_EMBEDDED_HTTP_URL = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_SENSITIVE_QUERY_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?(?:key|token)|token|secret|password|"
    r"credential|authorization|auth|signature|signed|session|cookie|jwt)(?:$|[_-])",
    re.IGNORECASE,
)
_TRACKING_QUERY_KEYS = {
    "dclid",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}
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
    "source_observed_at",
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
    "duplicate_scope",
    "canonical_owner_run_id",
)
_FINALIZATION_JOURNAL = ".finalization-journal.json"
_COMPLETION_MARKER = "completion.json"
_DEDUPLICATION_CONTEXT = "deduplication_context.json"
_RUN_STATE = ".run-state.json"
_RUN_LOCK = ".run.lock"
_MUTABLE_ROOT_ARTIFACTS = frozenset(
    {
        "run_manifest.json",
        "final_summary.json",
        _FINALIZATION_JOURNAL,
        _COMPLETION_MARKER,
        _RUN_STATE,
        _RUN_LOCK,
    }
)


class NewsProvider(Protocol):
    provider_id: str
    data_class: str

    def fetch(self, *, fetched_at: datetime | None = None) -> ProviderFetchResult: ...


class NewsRunCollisionError(FileExistsError):
    """The immutable run identity already exists in artifacts or storage."""


class NewsRunFinalizationError(RuntimeError):
    """A staged artifact pack could not be durably finalized in storage."""


@contextmanager
def _exclusive_run_lock(destination: Path):
    lock_path = destination / _RUN_LOCK
    existed = os.path.lexists(lock_path)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise NewsRunCollisionError(
            f"News run lock is unsafe or unavailable: {destination.name}"
        ) from exc
    handle = os.fdopen(descriptor, "a+b")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise NewsRunCollisionError(
                f"News run is active in another process: {destination.name}"
            ) from exc
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            if not existed:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass


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


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _jsonable(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_private_bytes(path: Path, value: bytes) -> None:
    """Write restricted raw evidence without depending on the process umask."""

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        # ``mode`` passed to ``os.open`` is still filtered through the process
        # umask.  Set the descriptor mode explicitly before writing so the
        # promised private mode is exact even under an unusually strict umask.
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(value)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_private_json(path: Path, value: Any) -> None:
    _write_private_bytes(path, _json_bytes(value))


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


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_json_bytes(value))


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _resolve_storage_path(
    value: str | Path,
    *,
    storage_root: Path,
    allow_external_storage: bool,
    field: str,
) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = storage_root / candidate
    resolved = candidate.resolve()
    if not allow_external_storage and not resolved.is_relative_to(storage_root):
        raise ValueError(f"{field} must remain inside the configured storage root")
    return resolved


def _portable_storage_label(path: Path, *, storage_root: Path, kind: str) -> str:
    try:
        return path.relative_to(storage_root).as_posix()
    except ValueError:
        identity = _sha256_bytes(path.as_posix().encode("utf-8"))[:16]
        return f"external-{kind}:sha256:{identity}/{path.name}"


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
    def redact_query_values(match: re.Match[str]) -> str:
        candidate = match.group(0)
        try:
            parsed = urlsplit(candidate)
        except ValueError:
            # The generic secret patterns below still fail closed for
            # malformed URL-shaped text.
            return candidate
        if not parsed.query and not parsed.fragment:
            return candidate
        query = urlencode(
            [
                (key, item)
                for key, item in parse_qsl(parsed.query, keep_blank_values=True)
                if not _SENSITIVE_QUERY_KEY.search(key)
                and not key.casefold().startswith("utm_")
                and key.casefold() not in _TRACKING_QUERY_KEYS
            ]
        )
        return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))

    redacted = _EMBEDDED_HTTP_URL.sub(redact_query_values, value)
    redacted = _URL_USERINFO.sub(r"\1[REDACTED]@", redacted)
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
            details=_sanitize_config(error.details),
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


def _candidate_matches_rule(
    article: NormalizedNewsArticle,
    candidate: Mapping[str, Any] | NormalizedNewsArticle,
    rule: str,
) -> bool:
    def value(name: str) -> Any:
        if isinstance(candidate, Mapping):
            return candidate.get(name)
        return getattr(candidate, name)

    incoming_title_key = title_source_time_key(
        title=article.title,
        source_name=article.source_name,
        published_at=article.published_at,
    )
    if rule == "canonical_url":
        return bool(article.canonical_url) and value("canonical_url") == article.canonical_url
    if rule == "content_sha256":
        return value("content_sha256") == article.content_sha256
    candidate_published = value("published_at")
    if incoming_title_key is None or not candidate_published:
        return False
    if isinstance(candidate_published, datetime):
        parsed = candidate_published
    else:
        parsed = datetime.fromisoformat(str(candidate_published).replace("Z", "+00:00"))
    return incoming_title_key == title_source_time_key(
        title=str(value("title") or ""),
        source_name=str(value("source_name") or ""),
        published_at=parsed,
    )


def _deduplicate_with_storage(
    storage: NewsStorage,
    articles: list[NormalizedNewsArticle],
    *,
    existing_candidates: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[list[NormalizedNewsArticle], list[DuplicateRelationship]]:
    """Apply each precedence rule globally across historical and local indexes."""

    persisted = list(
        storage.list_canonical_articles()
        if existing_candidates is None
        else existing_candidates
    )
    resolved: list[NormalizedNewsArticle] = []
    relationships: list[DuplicateRelationship] = []
    local_canonical: list[NormalizedNewsArticle] = []
    for article in articles:
        match: tuple[str, str, str, str] | None = None
        for rule in ("canonical_url", "content_sha256", "title_source_time"):
            historical_matches = sorted(
                (
                    str(candidate["article_id"]),
                    str(candidate.get("first_ingestion_run_id") or "unknown"),
                )
                for candidate in persisted
                if _candidate_matches_rule(article, candidate, rule)
            )
            if historical_matches:
                canonical_id, owner_run_id = historical_matches[0]
                match = canonical_id, rule, "historical", owner_run_id
                break
            local_matches = sorted(
                (candidate.article_id, candidate.ingestion_run_id)
                for candidate in local_canonical
                if _candidate_matches_rule(article, candidate, rule)
            )
            if local_matches:
                canonical_id, owner_run_id = local_matches[0]
                match = canonical_id, rule, "within_run", owner_run_id
                break

        if match is not None:
            canonical_id, rule, duplicate_scope, owner_run_id = match
            resolved.append(replace(article, duplicate_of=canonical_id))
            relationships.append(
                DuplicateRelationship(
                    article_id=article.article_id,
                    canonical_article_id=canonical_id,
                    rule=rule,  # type: ignore[arg-type]
                    duplicate_scope=duplicate_scope,  # type: ignore[arg-type]
                    canonical_owner_run_id=owner_run_id,
                )
            )
            continue
        resolved.append(article)
        local_canonical.append(article)
    return resolved, relationships


def _news_source_identity() -> dict[str, str | None]:
    source_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py"), key=lambda item: item.as_posix()):
        relative = path.relative_to(source_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    repository_root = source_root.parents[2]
    revision: str | None = None
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        candidate = completed.stdout.strip()
        if re.fullmatch(r"[0-9a-f]{40,64}", candidate):
            revision = candidate
    except (OSError, subprocess.SubprocessError):
        pass
    return {"news_source_sha256": digest.hexdigest(), "git_revision": revision}


def _build_deduplication_context(
    candidates: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    canonical_index: list[dict[str, Any]] = []
    dependencies: dict[str, str | None] = {}
    retained_fields = (
        "article_id",
        "canonical_url",
        "content_sha256",
        "title",
        "source_name",
        "published_at",
        "first_ingestion_run_id",
        "owner_manifest_sha256",
    )
    for candidate in candidates:
        entry = {field: candidate.get(field) for field in retained_fields}
        payload = {key: entry[key] for key in retained_fields[:-1]}
        entry["canonical_payload_sha256"] = _canonical_digest(payload)
        canonical_index.append(entry)
        owner = str(candidate.get("first_ingestion_run_id") or "unknown")
        dependencies[owner] = (
            str(candidate["owner_manifest_sha256"])
            if candidate.get("owner_manifest_sha256")
            else None
        )
    canonical_index.sort(key=lambda item: str(item["article_id"]))
    context: dict[str, Any] = {
        "schema_version": "1.0",
        "deduplication_algorithm_version": "2.0",
        "ordered_rules": [
            "canonical_url",
            "content_sha256",
            "title_source_time",
        ],
        "news_record_schema_version": NEWS_SCHEMA_VERSION,
        "canonical_index_count": len(canonical_index),
        "canonical_index": canonical_index,
        "dependency_runs": [
            {"run_id": run_id, "manifest_sha256": dependencies[run_id]}
            for run_id in sorted(dependencies)
        ],
        "code_identity": _news_source_identity(),
    }
    context["canonical_index_sha256"] = _canonical_digest(canonical_index)
    context["context_sha256"] = _canonical_digest(context)
    return context


def _artifact_inventory(artifact_root: Path) -> list[dict[str, Any]]:
    """Inventory the exact regular-file evidence set without following links."""

    root = artifact_root.resolve(strict=True)
    inventory: list[dict[str, Any]] = []
    pending = [artifact_root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda item: item.name)
        for entry in ordered:
            path = Path(entry.path)
            relative = path.relative_to(artifact_root).as_posix()
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode):
                raise NewsRunFinalizationError(
                    f"Sealed artifact trees cannot contain symlinks: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                resolved = path.resolve(strict=True)
                if not resolved.is_relative_to(root):
                    raise NewsRunFinalizationError(
                        f"Artifact directory escapes its run root: {relative}"
                    )
                pending.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise NewsRunFinalizationError(
                    f"Sealed artifact trees require regular files: {relative}"
                )
            if relative in _MUTABLE_ROOT_ARTIFACTS:
                continue
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise NewsRunFinalizationError(
                    f"Artifact file escapes its run root: {relative}"
                )
            inventory.append(
                {
                    "path": relative,
                    "sha256": _sha256_file(path),
                    "size_bytes": metadata.st_size,
                }
            )
    return sorted(inventory, key=lambda item: str(item["path"]))


def _inventory_sha256(inventory: Iterable[Mapping[str, Any]]) -> str:
    normalized = [
        {
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size_bytes"]),
        }
        for item in inventory
    ]
    return _canonical_digest(sorted(normalized, key=lambda item: item["path"]))


def _validate_artifact_inventory(
    artifact_root: Path,
    expected_inventory: Iterable[Mapping[str, Any]],
    *,
    expected_digest: str,
) -> list[dict[str, Any]]:
    expected = [
        {
            "path": str(item["path"]),
            "sha256": str(item["sha256"]),
            "size_bytes": int(item["size_bytes"]),
        }
        for item in expected_inventory
    ]
    expected.sort(key=lambda item: item["path"])
    if _inventory_sha256(expected) != expected_digest:
        raise NewsRunFinalizationError(
            "The sealed artifact inventory digest is inconsistent"
        )
    for item in expected:
        relative = Path(item["path"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() in _MUTABLE_ROOT_ARTIFACTS
        ):
            raise NewsRunFinalizationError(
                "The sealed artifact inventory contains an unsafe path"
            )
    observed = _artifact_inventory(artifact_root)
    if observed != expected:
        raise NewsRunFinalizationError(
            "The sealed artifact inventory has missing, extra, changed, or unsafe files"
        )
    return observed


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
        normalized_occurrence_count=int(
            value.get("normalized_occurrence_count", value["normalized_record_count"])
        ),
        new_canonical_article_count=int(
            value.get(
                "new_canonical_article_count",
                int(value["normalized_record_count"]) - int(value["duplicate_count"]),
            )
        ),
        matched_existing_canonical_count=int(
            value.get("matched_existing_canonical_count", value["duplicate_count"])
        ),
        within_run_duplicate_count=int(value.get("within_run_duplicate_count", 0)),
        historical_duplicate_count=int(
            value.get("historical_duplicate_count", value["duplicate_count"])
        ),
        represented_canonical_count=int(
            value.get(
                "represented_canonical_count",
                int(value["normalized_record_count"]) - int(value["duplicate_count"]),
            )
        ),
        deduplication_rule_counts={
            str(key): int(item)
            for key, item in dict(value.get("deduplication_rule_counts", {})).items()
        },
    )


def _read_regular_json(path: Path, *, label: str) -> Mapping[str, Any]:
    """Read an object marker without following a link or special file."""

    try:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise NewsRunFinalizationError(f"The {label} is not a regular file")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except NewsRunFinalizationError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise NewsRunFinalizationError(f"The {label} is unreadable") from exc
    if not isinstance(payload, Mapping):
        raise NewsRunFinalizationError(f"The {label} must be a JSON object")
    return payload


def _validate_reservation_marker(destination: Path, *, run_id: str) -> None:
    try:
        state_payload = _read_regular_json(
            destination / _RUN_STATE, label="run reservation marker"
        )
    except NewsRunFinalizationError as exc:
        raise NewsRunCollisionError(
            "Existing news run reservation cannot be authenticated"
        ) from exc
    expected_keys = {"schema_version", "run_id", "state", "started_at"}
    if (
        set(state_payload) != expected_keys
        or state_payload.get("schema_version") != NEWS_SCHEMA_VERSION
        or state_payload.get("run_id") != run_id
        or state_payload.get("state") != "RESERVED"
    ):
        raise NewsRunCollisionError(
            "Existing news run reservation marker is invalid"
        )
    try:
        started_at = datetime.fromisoformat(
            str(state_payload["started_at"]).replace("Z", "+00:00")
        )
        _aware_utc(started_at, field="reservation started_at")
    except (TypeError, ValueError) as exc:
        raise NewsRunCollisionError(
            "Existing news run reservation timestamp is invalid"
        ) from exc


def _validate_abandoned_reservation(
    destination: Path,
    *,
    run_id: str,
    before_database_begin: bool,
) -> None:
    """Prove that a stale directory is a reservation owned by this pipeline.

    A hard process termination can occur after the run directory and state
    marker are durable but before ``begin_run`` creates its SQLite row.  Only
    that narrowly shaped reservation may be reclaimed without database
    evidence.  Unexpected root entries and linked staging directories are
    treated as user-owned collisions and are never deleted.
    """

    _validate_reservation_marker(destination, run_id=run_id)

    pending_directories: list[Path] = []
    try:
        entries = list(os.scandir(destination))
    except OSError as exc:
        raise NewsRunCollisionError(
            "Existing news run reservation cannot be inspected"
        ) from exc
    for entry in entries:
        if entry.name in {_RUN_STATE, _RUN_LOCK}:
            if not entry.is_file(follow_symlinks=False):
                raise NewsRunCollisionError(
                    "Existing news run reservation contains an unsafe marker"
                )
            continue
        if entry.name.startswith(".pending-") and entry.is_dir(
            follow_symlinks=False
        ):
            pending_directories.append(Path(entry.path))
            continue
        raise NewsRunCollisionError(
            "Existing news run reservation contains unowned content"
        )
    if len(pending_directories) > 1:
        raise NewsRunCollisionError(
            "Existing news run reservation contains multiple staging directories"
        )

    if before_database_begin and pending_directories:
        # Before ``begin_run`` the newly-created staging directory is either
        # empty or contains only the empty raw directory.  Anything else could
        # not have been produced at this point in the pipeline.
        pending = pending_directories[0]
        try:
            with os.scandir(pending) as scanned_pending:
                pending_entries = list(scanned_pending)
        except OSError as exc:
            raise NewsRunCollisionError(
                "Existing pre-database reservation staging cannot be inspected"
            ) from exc
        if len(pending_entries) > 1:
            raise NewsRunCollisionError(
                "Existing pre-database reservation contains unexpected staging data"
            )
        if pending_entries:
            raw_entry = pending_entries[0]
            raw_has_entries = True
            if raw_entry.name == "raw" and raw_entry.is_dir(follow_symlinks=False):
                try:
                    with os.scandir(raw_entry.path) as scanned_raw:
                        raw_has_entries = next(scanned_raw, None) is not None
                except OSError as exc:
                    raise NewsRunCollisionError(
                        "Existing pre-database reservation raw staging is unsafe"
                    ) from exc
            if raw_entry.name != "raw" or raw_has_entries:
                raise NewsRunCollisionError(
                    "Existing pre-database reservation contains unexpected staging data"
                )


def _recover_committed_cleanup(
    destination: Path,
    *,
    storage: NewsStorage,
    run_id: str,
    database_status: str,
) -> PipelineSummary:
    """Finish removing controls after a committed run lost its journal."""

    _validate_reservation_marker(destination, run_id=run_id)
    manifest = _read_regular_json(
        destination / "run_manifest.json", label="committed run manifest"
    )
    summary_payload = _read_regular_json(
        destination / "final_summary.json", label="committed run summary"
    )
    completion = _read_regular_json(
        destination / _COMPLETION_MARKER, label="completion marker"
    )
    binding = storage.get_run_finalization_binding(run_id)
    if binding is None:
        raise NewsRunFinalizationError(
            "Completed cleanup has no immutable SQLite artifact binding"
        )
    if (
        database_status not in {"PASS", "PARTIAL", "FAIL"}
        or manifest.get("run_id") != run_id
        or summary_payload.get("run_id") != run_id
        or completion.get("run_id") != run_id
        or manifest.get("status") != database_status
        or summary_payload.get("run_status") != database_status
        or completion.get("status") != database_status
        or manifest.get("finalization_state") != "COMMITTED"
        or summary_payload.get("finalization_state") != "COMMITTED"
        or completion.get("finalization_state") != "COMMITTED"
    ):
        raise NewsRunFinalizationError(
            "Completed cleanup markers contradict the finalized database run"
        )
    manifest_sha256 = _sha256_json(manifest)
    summary_sha256 = _sha256_json(summary_payload)
    completion_sha256 = _sha256_json(completion)
    inventory = binding["artifact_inventory"]
    inventory_sha256 = str(binding["artifact_inventory_sha256"])
    if (
        manifest_sha256 != binding["manifest_sha256"]
        or completion_sha256 != binding["completion_marker_sha256"]
        or completion.get("manifest_sha256") != manifest_sha256
        or completion.get("summary_sha256") != summary_sha256
        or completion.get("artifact_inventory_sha256") != inventory_sha256
        or manifest.get("artifact_inventory") != inventory
        or manifest.get("artifact_inventory_sha256") != inventory_sha256
    ):
        raise NewsRunFinalizationError(
            "Completed cleanup markers do not match the immutable SQLite seal"
        )
    _validate_artifact_inventory(
        destination, inventory, expected_digest=inventory_sha256
    )
    _reconcile_artifacts_with_storage(destination, storage=storage, run_id=run_id)
    summary = _summary_from_payload(summary_payload)
    (destination / _RUN_STATE).unlink(missing_ok=True)
    return summary


def _finalization_payloads(
    artifact_root: Path,
    *,
    summary: PipelineSummary,
    manifest: Mapping[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    """Build identities that SQLite seals before public completion."""

    inventory = _artifact_inventory(artifact_root)
    inventory_sha256 = _inventory_sha256(inventory)
    final_summary = summary.to_dict()
    final_summary["finalization_state"] = "COMMITTED"
    final_manifest = dict(manifest)
    final_manifest.update(
        {
            "status": summary.run_status,
            "finalization_state": "COMMITTED",
            "artifact_inventory": inventory,
            "artifact_inventory_sha256": inventory_sha256,
            "completion_marker": _COMPLETION_MARKER,
        }
    )
    manifest_sha256 = _sha256_json(final_manifest)
    summary_sha256 = _sha256_json(final_summary)
    completion = {
        "schema_version": "1.0",
        "news_record_schema_version": NEWS_SCHEMA_VERSION,
        "availability_policy_version": NEWS_AVAILABILITY_POLICY_VERSION,
        "run_id": summary.run_id,
        "status": summary.run_status,
        "finalization_state": "COMMITTED",
        "completed_at": completed_at,
        "manifest_sha256": manifest_sha256,
        "summary_sha256": summary_sha256,
        "artifact_inventory_sha256": inventory_sha256,
        "database_schema_version": DATABASE_SCHEMA_VERSION,
    }
    return {
        "summary": final_summary,
        "manifest": final_manifest,
        "completion": completion,
        "manifest_sha256": manifest_sha256,
        "summary_sha256": summary_sha256,
        "completion_sha256": _sha256_json(completion),
        "artifact_inventory": inventory,
        "artifact_inventory_sha256": inventory_sha256,
    }


def _write_committed_markers(
    destination: Path, *, payloads: Mapping[str, Any]
) -> None:
    """Publish completion only after SQLite commits the bound run state."""

    _atomic_write_json(destination / "final_summary.json", payloads["summary"])
    _atomic_write_json(destination / "run_manifest.json", payloads["manifest"])
    # The final marker is the public commit point. Readers must require it and
    # verify its database-bound hashes before accepting the run.
    _atomic_write_json(destination / _COMPLETION_MARKER, payloads["completion"])


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise NewsRunFinalizationError(
            f"Cannot reconcile artifact CSV: {path.name}"
        ) from exc


def _csv_projection(payload: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, str]:
    projected: dict[str, str] = {}
    for field in fields:
        value = _csv_value(payload.get(field))
        projected[field] = "" if value is None else str(value)
    return projected


def _reconcile_artifacts_with_storage(
    destination: Path, *, storage: NewsStorage, run_id: str
) -> None:
    """Prove exported normalized, dedup, link, and quality evidence equals SQLite."""

    normalized_rows = _csv_rows(destination / "normalized_articles.csv")
    duplicate_rows = _csv_rows(destination / "duplicates.csv")
    link_rows = _csv_rows(destination / "article_entity_links.csv")
    quality_payload = _read_regular_json(
        destination / "quality_report.json", label="quality report"
    )
    with storage.connection() as connection:
        occurrences = connection.execute(
            """
            SELECT candidate_article_id, duplicate_of, deduplication_rule,
                   normalized_payload_json
            FROM article_ingestions WHERE ingestion_run_id = ?
            ORDER BY candidate_article_id
            """,
            (run_id,),
        ).fetchall()
        relationships = connection.execute(
            """
            SELECT duplicate_candidate_article_id, duplicate_of,
                   deduplication_rule
            FROM deduplication_relationships WHERE ingestion_run_id = ?
            ORDER BY duplicate_candidate_article_id
            """,
            (run_id,),
        ).fetchall()
        links = connection.execute(
            """
            SELECT link.link_id, occurrence.candidate_article_id AS article_id,
                   link.ticker, link.company_name, link.matched_alias,
                   link.match_method, link.match_location, link.confidence
            FROM article_company_links AS link
            JOIN article_ingestions AS occurrence
              ON occurrence.occurrence_id = link.occurrence_id
            WHERE occurrence.ingestion_run_id = ?
            ORDER BY link.link_id
            """,
            (run_id,),
        ).fetchall()
        quality_row = connection.execute(
            "SELECT report_json FROM quality_reports WHERE ingestion_run_id = ?",
            (run_id,),
        ).fetchone()
        canonical_owners = {
            str(row["article_id"]): str(row["first_ingestion_run_id"])
            for row in connection.execute(
                "SELECT article_id, first_ingestion_run_id FROM normalized_articles"
            ).fetchall()
        }

    expected_normalized: list[dict[str, str]] = []
    for row in occurrences:
        try:
            payload = json.loads(str(row["normalized_payload_json"]))
        except json.JSONDecodeError as exc:
            raise NewsRunFinalizationError(
                "SQLite contains unreadable normalized occurrence evidence"
            ) from exc
        if not isinstance(payload, Mapping):
            raise NewsRunFinalizationError(
                "SQLite normalized occurrence evidence is not an object"
            )
        expected_normalized.append(_csv_projection(payload, _NORMALIZED_FIELDS))
    if sorted(normalized_rows, key=lambda row: row.get("article_id", "")) != sorted(
        expected_normalized, key=lambda row: row.get("article_id", "")
    ):
        raise NewsRunFinalizationError(
            "normalized_articles.csv does not match SQLite occurrence evidence"
        )

    expected_relationships = {
        (
            str(row["duplicate_candidate_article_id"]),
            str(row["duplicate_of"]),
            str(row["deduplication_rule"]),
        )
        for row in relationships
    }
    observed_relationships: set[tuple[str, str, str]] = set()
    for row in duplicate_rows:
        relationship = (
            row.get("article_id", ""),
            row.get("duplicate_of", ""),
            row.get("deduplication_rule", ""),
        )
        observed_relationships.add(relationship)
        owner = canonical_owners.get(relationship[1])
        expected_scope = "within_run" if owner == run_id else "historical"
        if (
            row.get("duplicate_scope") != expected_scope
            or row.get("canonical_owner_run_id") != owner
        ):
            raise NewsRunFinalizationError(
                "duplicates.csv canonical ownership does not match SQLite"
            )
    if observed_relationships != expected_relationships or len(duplicate_rows) != len(
        expected_relationships
    ):
        raise NewsRunFinalizationError(
            "duplicates.csv does not match SQLite deduplication evidence"
        )

    expected_links = [
        {field: "" if row[field] is None else str(row[field]) for field in _LINK_FIELDS}
        for row in links
    ]
    if sorted(link_rows, key=lambda row: row.get("link_id", "")) != expected_links:
        raise NewsRunFinalizationError(
            "article_entity_links.csv does not match SQLite link evidence"
        )
    if quality_row is None:
        raise NewsRunFinalizationError("SQLite has no quality report for the run")
    try:
        persisted_quality = json.loads(str(quality_row["report_json"]))
    except json.JSONDecodeError as exc:
        raise NewsRunFinalizationError(
            "SQLite contains an unreadable quality report"
        ) from exc
    if quality_payload != persisted_quality:
        raise NewsRunFinalizationError(
            "quality_report.json does not match SQLite quality evidence"
        )


class _UnsealedInterruptedRun(NewsRunFinalizationError):
    """An interrupted run has no database-bound artifact identity."""


def _recover_finalization(
    destination: Path,
    *,
    storage: NewsStorage,
    run_id: str,
) -> PipelineSummary:
    """Finish only a database-bound, byte-identical interrupted commit."""

    journal_path = destination / _FINALIZATION_JOURNAL
    payload = _read_regular_json(journal_path, label="finalization recovery journal")
    if payload.get("run_id") != run_id:
        raise NewsRunFinalizationError(
            "The news run finalization journal does not match its run identity"
        )
    summary_payload = payload.get("summary")
    manifest_payload = payload.get("manifest")
    completion_payload = payload.get("completion")
    inventory = payload.get("artifact_inventory")
    if not all(
        isinstance(item, Mapping)
        for item in (summary_payload, manifest_payload, completion_payload)
    ) or not isinstance(inventory, list):
        raise NewsRunFinalizationError(
            "The news run finalization journal is incomplete"
        )
    assert isinstance(summary_payload, Mapping)
    assert isinstance(manifest_payload, Mapping)
    assert isinstance(completion_payload, Mapping)
    summary = _summary_from_payload(summary_payload)
    if summary.run_id != run_id or payload.get("desired_status") != summary.run_status:
        raise NewsRunFinalizationError(
            "The news run finalization journal has contradictory status evidence"
        )
    completed_at = payload.get("completed_at")
    if (
        not isinstance(completed_at, str)
        or completion_payload.get("completed_at") != completed_at
        or manifest_payload.get("completed_at") != completed_at
        or completion_payload.get("run_id") != run_id
        or completion_payload.get("status") != summary.run_status
        or manifest_payload.get("status") != summary.run_status
        or summary_payload.get("finalization_state") != "COMMITTED"
        or manifest_payload.get("finalization_state") != "COMMITTED"
        or completion_payload.get("finalization_state") != "COMMITTED"
        or manifest_payload.get("artifact_inventory") != inventory
    ):
        raise NewsRunFinalizationError(
            "The finalization journal contains contradictory completion evidence"
        )

    expected_hashes = {
        "manifest_sha256": _sha256_json(manifest_payload),
        "summary_sha256": _sha256_json(summary_payload),
        "completion_sha256": _sha256_json(completion_payload),
        "artifact_inventory_sha256": _inventory_sha256(inventory),
    }
    for key, observed in expected_hashes.items():
        if payload.get(key) != observed:
            raise NewsRunFinalizationError(
                f"The finalization journal has a contradictory {key}"
            )
    if (
        completion_payload.get("manifest_sha256") != expected_hashes["manifest_sha256"]
        or completion_payload.get("summary_sha256") != expected_hashes["summary_sha256"]
        or completion_payload.get("artifact_inventory_sha256")
        != expected_hashes["artifact_inventory_sha256"]
    ):
        raise NewsRunFinalizationError(
            "The completion marker identities contradict the finalization journal"
        )

    row = storage.get_run(run_id)
    if row is None:
        raise NewsRunFinalizationError(
            "The news run finalization journal has no matching database run"
        )
    binding = storage.get_run_finalization_binding(run_id)
    if binding is None:
        raise _UnsealedInterruptedRun(
            "The interrupted run has no database-bound artifact identity"
        )
    if (
        binding["artifact_inventory"] != inventory
        or binding["artifact_inventory_sha256"]
        != expected_hashes["artifact_inventory_sha256"]
        or binding["manifest_sha256"] != expected_hashes["manifest_sha256"]
        or binding["completion_marker_sha256"]
        != expected_hashes["completion_sha256"]
    ):
        raise NewsRunFinalizationError(
            "The recovery journal does not match the immutable SQLite seal"
        )
    _validate_artifact_inventory(
        destination,
        inventory,
        expected_digest=expected_hashes["artifact_inventory_sha256"],
    )
    _reconcile_artifacts_with_storage(
        destination, storage=storage, run_id=run_id
    )

    database_status = str(row["status"])
    if database_status == "RUNNING":
        storage.finalize_run(
            run_id,
            status=summary.run_status,
            completed_at=completed_at,
            summary=summary.to_dict(),
            error_message=payload.get("database_error"),
            expected_manifest_sha256=expected_hashes["manifest_sha256"],
            expected_inventory_sha256=expected_hashes[
                "artifact_inventory_sha256"
            ],
            expected_completion_sha256=expected_hashes["completion_sha256"],
        )
    elif database_status != summary.run_status:
        raise NewsRunFinalizationError(
            "The recovery journal conflicts with the finalized database status"
        )

    try:
        _write_committed_markers(destination, payloads=payload)
    except Exception as exc:
        raise NewsRunFinalizationError(
            "SQLite committed, but public completion publication is pending"
        ) from exc
    (destination / _RUN_STATE).unlink(missing_ok=True)
    # Keep the journal recoverable until the reservation marker is gone.  A
    # hard stop in this window will therefore re-enter sealed recovery rather
    # than looking like an abandoned pre-database reservation.
    journal_path.unlink(missing_ok=True)
    return summary


def run_news_pipeline(
    provider: NewsProvider,
    *,
    output_dir: str | Path = "artifacts/news_ingestion",
    database_path: str | Path = "data/news/paisa_news.sqlite3",
    storage_root: str | Path | None = None,
    allow_external_storage: bool = False,
    run_id: str | None = None,
    started_at: datetime | None = None,
) -> PipelineSummary:
    """Acquire and publish one immutable, evidence-preserving news run."""

    started = _aware_utc(
        started_at or datetime.now(timezone.utc), field="started_at"
    )
    selected_run_id = validate_news_run_id(run_id or make_news_run_id(started))
    storage_root_path = (
        Path.cwd().resolve()
        if storage_root is None
        else Path(storage_root).expanduser().resolve()
    )
    storage_root_path.mkdir(parents=True, exist_ok=True)
    output_root = _resolve_storage_path(
        output_dir,
        storage_root=storage_root_path,
        allow_external_storage=allow_external_storage,
        field="output_dir",
    )
    resolved_database = _resolve_storage_path(
        database_path,
        storage_root=storage_root_path,
        allow_external_storage=allow_external_storage,
        field="database_path",
    )
    output_label = _portable_storage_label(
        output_root, storage_root=storage_root_path, kind="artifacts"
    )
    database_label = _portable_storage_label(
        resolved_database, storage_root=storage_root_path, kind="database"
    )
    destination = output_root / selected_run_id
    storage = NewsStorage(resolved_database).initialize()
    if os.path.lexists(destination):
        if destination.is_symlink() or not destination.is_dir():
            raise NewsRunCollisionError(
                f"News run destination already exists: {destination}"
            )
        with _exclusive_run_lock(destination):
            journal_path = destination / _FINALIZATION_JOURNAL
            run_state_path = destination / _RUN_STATE
            if journal_path.is_file() and not journal_path.is_symlink():
                current = storage.get_run(selected_run_id)
                binding = (
                    storage.get_run_finalization_binding(selected_run_id)
                    if current is not None
                    else None
                )
                if (
                    current is not None
                    and current["status"] == "RUNNING"
                    and binding is None
                ):
                    # No artifact identity reached the database.  Nothing in
                    # this directory may be blessed; discard it and reacquire.
                    storage.discard_running_run(selected_run_id)
                    shutil.rmtree(destination)
                else:
                    recovered = _recover_finalization(
                        destination,
                        storage=storage,
                        run_id=selected_run_id,
                    )
                    run_state_path.unlink(missing_ok=True)
                    return recovered
            elif run_state_path.is_file() and not run_state_path.is_symlink():
                current = storage.get_run(selected_run_id)
                if current is not None and current["status"] != "RUNNING":
                    return _recover_committed_cleanup(
                        destination,
                        storage=storage,
                        run_id=selected_run_id,
                        database_status=str(current["status"]),
                    )
                _validate_abandoned_reservation(
                    destination,
                    run_id=selected_run_id,
                    before_database_begin=current is None,
                )
                try:
                    if current is not None:
                        storage.discard_running_run(selected_run_id)
                except Exception as exc:
                    raise NewsRunFinalizationError(
                        "A stale news run cannot be reconciled safely"
                    ) from exc
                shutil.rmtree(destination)
            else:
                raise NewsRunCollisionError(
                    f"News run destination already exists: {destination}"
                )

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
    lock_manager = _exclusive_run_lock(destination)
    lock_manager.__enter__()
    try:
        _atomic_write_json(
            destination / _RUN_STATE,
            {
                "schema_version": NEWS_SCHEMA_VERSION,
                "run_id": selected_run_id,
                "state": "RESERVED",
                "started_at": utc_iso(started),
            },
        )
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination))
        (staging / "raw").mkdir(mode=0o700)
    except BaseException:
        # Reservation setup happens before the main run cleanup block.  Keep
        # it failure-atomic as well so an I/O error or interruption cannot
        # strand the lock or permanently consume an otherwise unused run ID.
        try:
            lock_manager.__exit__(None, None, None)
        finally:
            if destination.exists():
                shutil.rmtree(destination)
        raise

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
                artifact_directory=(
                    f"{output_label.rstrip('/')}/{selected_run_id}"
                ),
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
                "availability_policy_version": NEWS_AVAILABILITY_POLICY_VERSION,
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
            _write_private_bytes(staging / relative, capture.content)
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
            _write_private_json(staging / relative, raw_payload)
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

        _write_private_json(
            staging / "raw" / "index.json",
            {
                "schema_version": NEWS_SCHEMA_VERSION,
                "captures": capture_entries,
                "records": raw_index,
            },
        )

        pre_run_canonical_index = storage.list_canonical_articles()
        deduplication_context = _build_deduplication_context(
            pre_run_canonical_index
        )
        _write_json(staging / _DEDUPLICATION_CONTEXT, deduplication_context)
        deduplicated, duplicate_relationships = _deduplicate_with_storage(
            storage,
            normalized,
            existing_candidates=pre_run_canonical_index,
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
                schema_version=quality["schema_version"],
                quality_status=quality["quality_status"],
                report_payload=quality,
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
            "Coverage is limited to fields supplied by the configured provider; "
            "GDELT acquisition is metadata-only and manual CSV rows are imported "
            "evidence rather than live acquisitions.",
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
            artifact_directory=(f"{output_label.rstrip('/')}/{selected_run_id}"),
            database_path=database_label,
            limitations=limitations,
            normalized_occurrence_count=int(
                quality["normalized_occurrence_count"]
            ),
            new_canonical_article_count=int(
                quality["new_canonical_article_count"]
            ),
            matched_existing_canonical_count=int(
                quality["matched_existing_canonical_count"]
            ),
            within_run_duplicate_count=int(
                quality["within_run_duplicate_count"]
            ),
            historical_duplicate_count=int(
                quality["historical_duplicate_count"]
            ),
            represented_canonical_count=int(
                quality["represented_canonical_count"]
            ),
            deduplication_rule_counts={
                str(key): int(value)
                for key, value in quality["deduplication_rule_counts"].items()
            },
        )
        pending_summary = summary.to_dict()
        pending_summary["run_status"] = "PENDING"
        pending_summary["finalization_state"] = "PENDING_DATABASE_FINALIZATION"
        manifest_base = {
            "schema_version": NEWS_SCHEMA_VERSION,
            "availability_policy_version": NEWS_AVAILABILITY_POLICY_VERSION,
            "quality_report_schema_version": quality["schema_version"],
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
                "normalized_occurrences": quality["normalized_occurrence_count"],
                "new_canonical_articles": quality["new_canonical_article_count"],
                "matched_existing_canonicals": quality[
                    "matched_existing_canonical_count"
                ],
                "within_run_duplicates": quality["within_run_duplicate_count"],
                "historical_duplicates": quality["historical_duplicate_count"],
                "represented_canonicals": quality["represented_canonical_count"],
                "rejected": len(rejected),
                "provider_errors": len(result.errors),
            },
            "deduplication_rule_counts": quality["deduplication_rule_counts"],
            "deduplication_context": {
                "path": _DEDUPLICATION_CONTEXT,
                "sha256": _sha256_file(staging / _DEDUPLICATION_CONTEXT),
                "canonical_index_sha256": deduplication_context[
                    "canonical_index_sha256"
                ],
                "context_sha256": deduplication_context["context_sha256"],
            },
            "database": {
                "engine": "sqlite",
                "schema_version": DATABASE_SCHEMA_VERSION,
                "configured_path": database_label,
            },
            "artifact_inventory": _artifact_inventory(staging),
            "limitations": list(limitations),
        }
        completed_at_text = utc_iso(completed_at)
        payloads = _finalization_payloads(
            staging,
            summary=summary,
            manifest=manifest_base,
            completed_at=completed_at_text,
        )
        pending_manifest = dict(manifest_base)
        pending_manifest.update(
            {
                "artifact_inventory": payloads["artifact_inventory"],
                "artifact_inventory_sha256": payloads[
                    "artifact_inventory_sha256"
                ],
                "recovery_journal": _FINALIZATION_JOURNAL,
                "completion_marker": _COMPLETION_MARKER,
            }
        )
        _write_json(staging / "final_summary.json", pending_summary)
        _write_json(staging / "run_manifest.json", pending_manifest)

        journal = {
            "schema_version": NEWS_SCHEMA_VERSION,
            "run_id": selected_run_id,
            "desired_status": run_status,
            "completed_at": completed_at_text,
            "database_error": storage_error,
            **payloads,
        }
        _write_json(staging / _FINALIZATION_JOURNAL, journal)

        _publish_staged_tree(staging, destination)
        published = True

        _validate_artifact_inventory(
            destination,
            payloads["artifact_inventory"],
            expected_digest=payloads["artifact_inventory_sha256"],
        )
        _reconcile_artifacts_with_storage(
            destination, storage=storage, run_id=selected_run_id
        )
        binding = storage.seal_run(
            selected_run_id,
            artifact_inventory=payloads["artifact_inventory"],
            manifest_sha256=payloads["manifest_sha256"],
            completion_marker_sha256=payloads["completion_sha256"],
            expected_counts=summary.to_dict(),
            prepared_at=completed_at,
        )
        try:
            storage.finalize_run(
                selected_run_id,
                status=run_status,
                completed_at=completed_at,
                summary=summary.to_dict(),
                error_message=storage_error,
                expected_manifest_sha256=payloads["manifest_sha256"],
                expected_inventory_sha256=binding[
                    "artifact_inventory_sha256"
                ],
                expected_completion_sha256=payloads["completion_sha256"],
            )
        except BaseException as exc:
            current = storage.get_run(selected_run_id)
            if current is None or current["status"] != run_status:
                if isinstance(exc, Exception):
                    raise NewsRunFinalizationError(
                        "SQLite finalization is pending; no completion marker was published"
                    ) from exc
                raise

        try:
            _write_committed_markers(destination, payloads=payloads)
        except Exception as exc:
            raise NewsRunFinalizationError(
                "SQLite committed, but public completion publication is pending"
            ) from exc

        (destination / _RUN_STATE).unlink(missing_ok=True)
        (destination / _FINALIZATION_JOURNAL).unlink(missing_ok=True)
        return summary
    except BaseException:
        if run_started_in_db and not published:
            try:
                storage.discard_running_run(selected_run_id)
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
        try:
            lock_manager.__exit__(None, None, None)
        finally:
            if destination.exists() and not (destination / _FINALIZATION_JOURNAL).exists():
                (destination / _RUN_LOCK).unlink(missing_ok=True)


__all__ = [
    "NewsRunCollisionError",
    "NewsRunFinalizationError",
    "make_news_run_id",
    "run_news_pipeline",
    "validate_news_run_id",
]
