"""Offline integrity verification and deterministic replay for news runs.

The artifact replay never contacts a provider and can run without SQLite.  An
optional read-only database check additionally proves that the completion
marker and sealed inventory are bound to a committed ingestion-run row.  A
completed artifact directory carries the remaining evidence needed to validate
raw captures, the frozen pre-run canonical index, normalization output, and
deduplication decisions.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import stat
from typing import Any, Iterable, Mapping
from urllib.parse import quote

from paisa.news.deduplication import title_source_time_key
from paisa.news.entity_linking import link_ogdc_articles
from paisa.news.models import (
    DuplicateRelationship,
    NormalizedNewsArticle,
    ProviderError,
    RawNewsRecord,
)
from paisa.news.normalization import NormalizationError, normalize_record
from paisa.news.quality import calculate_quality_report


_MUTABLE_ROOT_FILES = frozenset(
    {
        "run_manifest.json",
        "final_summary.json",
        "completion.json",
        ".finalization-journal.json",
        ".run-state.json",
        ".run.lock",
    }
)
_RECOVERY_FILES = frozenset(
    {".finalization-journal.json", ".run-state.json"}
)
_OPERATIONAL_LOCK = ".run.lock"
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
_DUPLICATE_FIELDS = (
    "article_id",
    "duplicate_of",
    "deduplication_rule",
    "duplicate_scope",
    "canonical_owner_run_id",
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
_CONTEXT_FIELDS = (
    "article_id",
    "canonical_url",
    "content_sha256",
    "title",
    "source_name",
    "published_at",
    "first_ingestion_run_id",
)


class NewsRunVerificationError(ValueError):
    """A completed run's retained evidence is missing or contradictory."""


@dataclass(frozen=True)
class NewsRunVerification:
    run_id: str
    status: str
    artifact_count: int
    capture_count: int
    raw_record_count: int
    normalized_record_count: int
    duplicate_count: int
    artifact_inventory_sha256: str
    replayed: bool = True
    database_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "artifact_count": self.artifact_count,
            "capture_count": self.capture_count,
            "raw_record_count": self.raw_record_count,
            "normalized_record_count": self.normalized_record_count,
            "duplicate_count": self.duplicate_count,
            "artifact_inventory_sha256": self.artifact_inventory_sha256,
            "replayed": self.replayed,
            "database_verified": self.database_verified,
            "completion_acceptance": (
                "DATABASE_AND_ARTIFACT_BOUND"
                if self.database_verified
                else "ARTIFACT_REPLAY_ONLY"
            ),
            "verification": "PASS",
        }


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise NewsRunVerificationError(f"{label} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise NewsRunVerificationError(f"{label} is not a safe POSIX path")
    return value


def _regular_file(root: Path, relative: str, *, label: str) -> Path:
    safe = _safe_relative(relative, label=label)
    path = root / safe
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise NewsRunVerificationError(f"Missing {label}: {safe}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise NewsRunVerificationError(f"{label} must be a regular file: {safe}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise NewsRunVerificationError(f"Unreadable {label}: {safe}") from exc
    if not resolved.is_relative_to(root):
        raise NewsRunVerificationError(f"{label} escapes the artifact root: {safe}")
    return path


def _read_json(root: Path, relative: str, *, label: str) -> Mapping[str, Any]:
    path = _regular_file(root, relative, label=label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise NewsRunVerificationError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise NewsRunVerificationError(f"{label} must contain a JSON object")
    return value


def _inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as exc:
            raise NewsRunVerificationError("Artifact directory is unreadable") from exc
        for entry in children:
            path = Path(entry.path)
            relative = path.relative_to(root).as_posix()
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise NewsRunVerificationError(
                    f"Artifact disappeared during verification: {relative}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise NewsRunVerificationError(
                    f"Artifact trees cannot contain symlinks: {relative}"
                )
            if stat.S_ISDIR(metadata.st_mode):
                if not path.resolve(strict=True).is_relative_to(root):
                    raise NewsRunVerificationError(
                        f"Artifact directory escapes its run root: {relative}"
                    )
                pending.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise NewsRunVerificationError(
                    f"Artifact trees require regular files: {relative}"
                )
            if relative in _MUTABLE_ROOT_FILES:
                continue
            inventory.append(
                {
                    "path": relative,
                    "sha256": _file_digest(path),
                    "size_bytes": metadata.st_size,
                }
            )
    return sorted(inventory, key=lambda item: item["path"])


def _normalize_inventory(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise NewsRunVerificationError("Manifest artifact_inventory must be a list")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in value:
        if not isinstance(entry, Mapping):
            raise NewsRunVerificationError("Artifact inventory entries must be objects")
        path = _safe_relative(entry.get("path"), label="artifact inventory path")
        if path in _MUTABLE_ROOT_FILES or path in seen:
            raise NewsRunVerificationError("Artifact inventory paths must be unique and immutable")
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise NewsRunVerificationError("Artifact inventory has an invalid SHA-256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise NewsRunVerificationError("Artifact inventory has an invalid size")
        seen.add(path)
        normalized.append({"path": path, "sha256": digest, "size_bytes": size})
    if normalized != sorted(normalized, key=lambda item: item["path"]):
        raise NewsRunVerificationError("Artifact inventory is not deterministically ordered")
    return normalized


def _parse_datetime(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise NewsRunVerificationError(f"Raw record {field} must be an ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NewsRunVerificationError(f"Raw record {field} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NewsRunVerificationError(f"Raw record {field} must be timezone-aware")
    return parsed


def _raw_record(payload: Mapping[str, Any], *, indexed_id: str) -> RawNewsRecord:
    retained_id = payload.get("storage_raw_record_id")
    if retained_id != indexed_id:
        raise NewsRunVerificationError("Raw index and record storage IDs differ")
    original_id = payload.get("raw_record_id")
    if not isinstance(original_id, str) or not original_id:
        raise NewsRunVerificationError("Retained raw record ID is missing")
    fields = dict(payload)
    fields.pop("storage_raw_record_id", None)
    fields["raw_record_id"] = indexed_id
    fields["provider_record_id"] = fields.get("provider_record_id") or original_id
    fields["fetched_at"] = _parse_datetime(fields.get("fetched_at"), field="fetched_at")
    fields["provider_warnings"] = tuple(fields.get("provider_warnings") or ())
    allowed = set(RawNewsRecord.__dataclass_fields__)
    unexpected = set(fields) - allowed
    if unexpected:
        raise NewsRunVerificationError(
            "Raw record contains unsupported fields: " + ", ".join(sorted(unexpected))
        )
    try:
        return RawNewsRecord(**fields)
    except (TypeError, ValueError) as exc:
        raise NewsRunVerificationError("Retained raw record is invalid") from exc


def _candidate_value(
    candidate: Mapping[str, Any] | NormalizedNewsArticle, name: str
) -> Any:
    if isinstance(candidate, Mapping):
        return candidate.get(name)
    return getattr(candidate, name)


def _candidate_matches(
    article: NormalizedNewsArticle,
    candidate: Mapping[str, Any] | NormalizedNewsArticle,
    rule: str,
) -> bool:
    if rule == "canonical_url":
        return bool(article.canonical_url) and (
            _candidate_value(candidate, "canonical_url") == article.canonical_url
        )
    if rule == "content_sha256":
        return _candidate_value(candidate, "content_sha256") == article.content_sha256
    published = _candidate_value(candidate, "published_at")
    if not published:
        return False
    if not isinstance(published, datetime):
        try:
            published = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
        except ValueError as exc:
            raise NewsRunVerificationError(
                "Frozen canonical index contains an invalid publication time"
            ) from exc
    incoming = title_source_time_key(
        title=article.title,
        source_name=article.source_name,
        published_at=article.published_at,
    )
    return incoming is not None and incoming == title_source_time_key(
        title=str(_candidate_value(candidate, "title") or ""),
        source_name=str(_candidate_value(candidate, "source_name") or ""),
        published_at=published,
    )


def _replay_deduplication(
    articles: Iterable[NormalizedNewsArticle],
    frozen_candidates: list[Mapping[str, Any]],
) -> tuple[list[NormalizedNewsArticle], list[DuplicateRelationship]]:
    resolved: list[NormalizedNewsArticle] = []
    relationships: list[DuplicateRelationship] = []
    local_canonicals: list[NormalizedNewsArticle] = []
    for article in articles:
        match: tuple[str, str, str, str] | None = None
        for rule in ("canonical_url", "content_sha256", "title_source_time"):
            historical = sorted(
                (
                    str(candidate["article_id"]),
                    str(candidate["first_ingestion_run_id"]),
                )
                for candidate in frozen_candidates
                if _candidate_matches(article, candidate, rule)
            )
            if historical:
                canonical_id, owner = historical[0]
                match = canonical_id, rule, "historical", owner
                break
            local = sorted(
                (candidate.article_id, candidate.ingestion_run_id)
                for candidate in local_canonicals
                if _candidate_matches(article, candidate, rule)
            )
            if local:
                canonical_id, owner = local[0]
                match = canonical_id, rule, "within_run", owner
                break
        if match is None:
            resolved.append(article)
            local_canonicals.append(article)
            continue
        canonical_id, rule, scope, owner = match
        resolved.append(replace(article, duplicate_of=canonical_id))
        relationships.append(
            DuplicateRelationship(
                article_id=article.article_id,
                canonical_article_id=canonical_id,
                rule=rule,  # type: ignore[arg-type]
                duplicate_scope=scope,  # type: ignore[arg-type]
                canonical_owner_run_id=owner,
            )
        )
    return resolved, relationships


def _csv_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, Mapping)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _expected_csv_rows(
    rows: Iterable[Mapping[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, str]]:
    return [
        {field: _csv_scalar(row.get(field)) for field in fields}
        for row in rows
    ]


def _read_csv(root: Path, relative: str, fields: tuple[str, ...]) -> list[dict[str, str]]:
    path = _regular_file(root, relative, label=relative)
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != fields:
                raise NewsRunVerificationError(f"{relative} has an unexpected schema")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as exc:
        raise NewsRunVerificationError(f"{relative} is unreadable") from exc


def _validate_context(root: Path, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = manifest.get("deduplication_context")
    if not isinstance(reference, Mapping):
        raise NewsRunVerificationError("Manifest lacks deduplication context provenance")
    context_path = _safe_relative(reference.get("path"), label="deduplication context")
    context_file = _regular_file(root, context_path, label="deduplication context")
    observed_file_hash = _file_digest(context_file)
    if observed_file_hash != reference.get("sha256"):
        raise NewsRunVerificationError("Deduplication context file digest differs")
    context = _read_json(root, context_path, label="deduplication context")
    candidates = context.get("canonical_index")
    if not isinstance(candidates, list) or any(
        not isinstance(candidate, Mapping) for candidate in candidates
    ):
        raise NewsRunVerificationError("Frozen canonical index must be a list of objects")
    if context.get("ordered_rules") != [
        "canonical_url",
        "content_sha256",
        "title_source_time",
    ]:
        raise NewsRunVerificationError("Frozen deduplication rule order is unsupported")
    if context.get("canonical_index_count") != len(candidates):
        raise NewsRunVerificationError("Frozen canonical index count differs")
    if candidates != sorted(candidates, key=lambda item: str(item.get("article_id"))):
        raise NewsRunVerificationError("Frozen canonical index is not ordered")
    for candidate in candidates:
        if any(candidate.get(field) is None for field in ("article_id", "content_sha256", "first_ingestion_run_id")):
            raise NewsRunVerificationError("Frozen canonical index entry is incomplete")
        payload = {field: candidate.get(field) for field in _CONTEXT_FIELDS}
        if candidate.get("canonical_payload_sha256") != _json_digest(payload):
            raise NewsRunVerificationError("Frozen canonical payload digest differs")
    index_hash = _json_digest(candidates)
    if index_hash != context.get("canonical_index_sha256") or index_hash != reference.get(
        "canonical_index_sha256"
    ):
        raise NewsRunVerificationError("Frozen canonical index digest differs")
    context_without_hash = dict(context)
    recorded_context_hash = context_without_hash.pop("context_sha256", None)
    if _json_digest(context_without_hash) != recorded_context_hash:
        raise NewsRunVerificationError("Deduplication context digest differs")
    if recorded_context_hash != reference.get("context_sha256"):
        raise NewsRunVerificationError("Manifest context digest differs")
    code_identity = context.get("code_identity")
    if not isinstance(code_identity, Mapping) or not isinstance(
        code_identity.get("news_source_sha256"), str
    ):
        raise NewsRunVerificationError("Deduplication code identity is missing")
    dependencies = context.get("dependency_runs")
    if not isinstance(dependencies, list):
        raise NewsRunVerificationError("Canonical dependency provenance is missing")
    dependency_map = {
        str(item.get("run_id")): item.get("manifest_sha256")
        for item in dependencies
        if isinstance(item, Mapping)
    }
    owners = {str(candidate["first_ingestion_run_id"]) for candidate in candidates}
    if set(dependency_map) != owners or any(
        dependency_map[owner] is not None
        and (
            not isinstance(dependency_map[owner], str)
            or len(dependency_map[owner]) != 64
        )
        for owner in owners
    ):
        raise NewsRunVerificationError("Canonical dependency manifests are incomplete")
    return context


def _verify_database_binding(
    database_path: str | Path,
    *,
    run_id: str,
    status: str,
    manifest_digest: str,
    completion_digest: str,
    inventory_digest: str,
    expected_inventory: list[dict[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    supplied = Path(database_path).expanduser()
    try:
        metadata = supplied.lstat()
    except OSError as exc:
        raise NewsRunVerificationError("News database does not exist") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise NewsRunVerificationError("News database must be a regular file")
    database = supplied.resolve(strict=True)
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{quote(database.as_posix(), safe='/:')}?mode=ro",
            uri=True,
            timeout=2,
        )
        connection.row_factory = sqlite3.Row
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(ingestion_runs)")
        }
        required = {
            "run_id",
            "status",
            "summary_json",
            "sealed_artifact_inventory_json",
            "sealed_artifact_inventory_sha256",
            "sealed_manifest_sha256",
            "sealed_completion_sha256",
        }
        if not required <= columns:
            raise NewsRunVerificationError(
                "News database lacks the sealed completion schema"
            )
        row = connection.execute(
            """
            SELECT status, summary_json, sealed_artifact_inventory_json,
                   sealed_artifact_inventory_sha256, sealed_manifest_sha256,
                   sealed_completion_sha256
            FROM ingestion_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
    except sqlite3.Error as exc:
        raise NewsRunVerificationError("News database could not be read safely") from exc
    finally:
        if connection is not None:
            connection.close()
    if row is None:
        raise NewsRunVerificationError("News database has no matching run")
    if row["status"] != status or row["status"] not in {"PASS", "PARTIAL", "FAIL"}:
        raise NewsRunVerificationError("Database and completion statuses differ")
    digest_pairs = (
        (row["sealed_manifest_sha256"], manifest_digest, "manifest"),
        (
            row["sealed_artifact_inventory_sha256"],
            inventory_digest,
            "artifact inventory",
        ),
        (row["sealed_completion_sha256"], completion_digest, "completion marker"),
    )
    for sealed, observed, label in digest_pairs:
        if sealed != observed:
            raise NewsRunVerificationError(
                f"Database seal does not bind this {label}"
            )
    try:
        stored_inventory = json.loads(str(row["sealed_artifact_inventory_json"]))
        stored_summary = json.loads(str(row["summary_json"]))
    except json.JSONDecodeError as exc:
        raise NewsRunVerificationError("Database completion evidence is invalid JSON") from exc
    if stored_inventory != expected_inventory:
        raise NewsRunVerificationError("Database sealed inventory differs from manifest")
    artifact_summary = dict(summary)
    artifact_summary.pop("finalization_state", None)
    if stored_summary != artifact_summary:
        raise NewsRunVerificationError("Database summary differs from final summary")


def verify_news_run(
    run_directory: str | Path, *, database_path: str | Path | None = None
) -> NewsRunVerification:
    """Verify/replay a run and optionally prove its read-only SQLite binding."""

    supplied = Path(run_directory).expanduser()
    try:
        root_metadata = supplied.lstat()
    except OSError as exc:
        raise NewsRunVerificationError("News run directory does not exist") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise NewsRunVerificationError("News run root must be a real directory")
    root = supplied.resolve(strict=True)
    lock_path = root / _OPERATIONAL_LOCK
    if os.path.lexists(lock_path):
        try:
            lock_metadata = lock_path.lstat()
        except OSError as exc:
            raise NewsRunVerificationError(
                "Completed run lock cannot be inspected"
            ) from exc
        if not stat.S_ISREG(lock_metadata.st_mode):
            raise NewsRunVerificationError(
                "Completed run lock must be a regular operational file"
            )
    for forbidden in _RECOVERY_FILES:
        if os.path.lexists(root / forbidden):
            raise NewsRunVerificationError(
                "Completed runs cannot retain recovery state"
            )

    completion = _read_json(root, "completion.json", label="completion marker")
    completion_path = _regular_file(root, "completion.json", label="completion marker")
    manifest_path = _regular_file(root, "run_manifest.json", label="run manifest")
    summary_path = _regular_file(root, "final_summary.json", label="final summary")
    manifest = _read_json(root, "run_manifest.json", label="run manifest")
    summary = _read_json(root, "final_summary.json", label="final summary")

    manifest_digest = _file_digest(manifest_path)
    summary_digest = _file_digest(summary_path)
    if completion.get("manifest_sha256") != manifest_digest:
        raise NewsRunVerificationError("Completion marker does not bind this manifest")
    if completion.get("summary_sha256") != summary_digest:
        raise NewsRunVerificationError("Completion marker does not bind this summary")
    run_id = completion.get("run_id")
    status = completion.get("status")
    if (
        not isinstance(run_id, str)
        or run_id != manifest.get("run_id")
        or run_id != summary.get("run_id")
    ):
        raise NewsRunVerificationError("Completion, manifest, and summary run IDs differ")
    if status not in {"PASS", "PARTIAL", "FAIL"}:
        raise NewsRunVerificationError("Completion marker has an invalid status")
    if status != manifest.get("status") or status != summary.get("run_status"):
        raise NewsRunVerificationError("Completion, manifest, and summary statuses differ")
    if any(
        marker.get("finalization_state") != "COMMITTED"
        for marker in (completion, manifest, summary)
    ):
        raise NewsRunVerificationError("Completion evidence is not committed")

    expected_inventory = _normalize_inventory(manifest.get("artifact_inventory"))
    inventory_digest = _json_digest(expected_inventory)
    if inventory_digest != manifest.get("artifact_inventory_sha256"):
        raise NewsRunVerificationError("Manifest inventory digest differs")
    if inventory_digest != completion.get("artifact_inventory_sha256"):
        raise NewsRunVerificationError("Completion marker inventory digest differs")
    observed_inventory = _inventory(root)
    if observed_inventory != expected_inventory:
        raise NewsRunVerificationError(
            "Sealed artifacts are missing, extra, changed, or unsafe"
        )

    context = _validate_context(root, manifest)
    raw_index = _read_json(root, "raw/index.json", label="raw index")
    captures = raw_index.get("captures")
    records = raw_index.get("records")
    if not isinstance(captures, list) or not isinstance(records, list):
        raise NewsRunVerificationError("Raw index captures and records must be lists")

    referenced_raw_paths: set[str] = {"raw/index.json"}
    for capture in captures:
        if not isinstance(capture, Mapping):
            raise NewsRunVerificationError("Raw capture index entries must be objects")
        reference = _safe_relative(capture.get("artifact_reference"), label="raw capture")
        if not reference.startswith("raw/") or reference in referenced_raw_paths:
            raise NewsRunVerificationError("Raw capture references must be unique under raw/")
        path = _regular_file(root, reference, label="raw capture")
        observed = _file_digest(path)
        if observed != capture.get("observed_sha256") or observed != capture.get(
            "response_sha256"
        ):
            raise NewsRunVerificationError("Raw capture checksum differs")
        referenced_raw_paths.add(reference)

    replay_records: list[tuple[RawNewsRecord, str]] = []
    indexed_ids: set[str] = set()
    for record_entry in records:
        if not isinstance(record_entry, Mapping):
            raise NewsRunVerificationError("Raw record index entries must be objects")
        reference = _safe_relative(record_entry.get("artifact_reference"), label="raw record")
        indexed_id = record_entry.get("raw_record_id")
        if (
            not isinstance(indexed_id, str)
            or not indexed_id
            or indexed_id in indexed_ids
            or not reference.startswith("raw/")
            or reference in referenced_raw_paths
        ):
            raise NewsRunVerificationError("Raw record index identity is invalid")
        payload = _read_json(root, reference, label="raw record")
        record = _raw_record(payload, indexed_id=indexed_id)
        if record_entry.get("provider_record_id") != record.provider_record_id:
            raise NewsRunVerificationError("Raw index provider identity differs")
        replay_records.append((record, reference))
        indexed_ids.add(indexed_id)
        referenced_raw_paths.add(reference)

    retained_raw_paths = {
        item["path"] for item in expected_inventory if str(item["path"]).startswith("raw/")
    }
    if retained_raw_paths != referenced_raw_paths:
        raise NewsRunVerificationError("Raw index does not reference the exact raw evidence set")

    normalized: list[NormalizedNewsArticle] = []
    rejected: list[dict[str, str]] = []
    for record, reference in replay_records:
        try:
            normalized.append(
                normalize_record(
                    record,
                    ingestion_run_id=run_id,
                    raw_artifact_reference=reference,
                )
            )
        except NormalizationError as exc:
            rejected.append({"raw_record_id": record.raw_record_id, "reason": str(exc)})

    frozen_candidates = context["canonical_index"]
    deduplicated, relationships = _replay_deduplication(
        normalized, frozen_candidates  # type: ignore[arg-type]
    )
    linked_articles, entity_links = link_ogdc_articles(deduplicated)
    expected_normalized = _expected_csv_rows(
        (article.to_dict() for article in linked_articles), _NORMALIZED_FIELDS
    )
    observed_normalized = _read_csv(
        root, "normalized_articles.csv", _NORMALIZED_FIELDS
    )
    if observed_normalized != expected_normalized:
        raise NewsRunVerificationError("Normalized CSV does not match raw-record replay")
    expected_duplicates = _expected_csv_rows(
        (relationship.to_dict() for relationship in relationships), _DUPLICATE_FIELDS
    )
    observed_duplicates = _read_csv(root, "duplicates.csv", _DUPLICATE_FIELDS)
    if observed_duplicates != expected_duplicates:
        raise NewsRunVerificationError("Duplicate CSV does not match frozen-index replay")

    expected_links = _expected_csv_rows(
        (link.to_dict() for link in entity_links), _LINK_FIELDS
    )
    observed_links = _read_csv(root, "article_entity_links.csv", _LINK_FIELDS)
    if observed_links != expected_links:
        raise NewsRunVerificationError("Entity-link CSV does not match replay")

    quality = _read_json(root, "quality_report.json", label="quality report")
    if quality.get("rejected_records") != sorted(
        rejected, key=lambda item: (item["raw_record_id"], item["reason"])
    ):
        raise NewsRunVerificationError("Quality rejection evidence differs from replay")
    errors_payload = _read_json(
        root, "provider_errors.json", label="provider errors"
    ).get("errors")
    if not isinstance(errors_payload, list):
        raise NewsRunVerificationError("Provider error evidence must be a list")
    provider_errors: list[ProviderError] = []
    for error in errors_payload:
        if not isinstance(error, Mapping):
            raise NewsRunVerificationError("Provider error entries must be objects")
        try:
            provider_errors.append(
                ProviderError(
                    provider_id=str(error["provider_id"]),
                    configured_source=str(error["configured_source"]),
                    error_type=str(error["error_type"]),
                    message=str(error["message"]),
                    occurred_at=_parse_datetime(
                        error.get("occurred_at"), field="provider error occurred_at"
                    ),
                    http_status=(
                        None
                        if error.get("http_status") is None
                        else int(error["http_status"])
                    ),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise NewsRunVerificationError("Provider error evidence is invalid") from exc
    replayed_quality = calculate_quality_report(
        raw_records=[record for record, _ in replay_records],
        normalized_articles=linked_articles,
        duplicate_relationships=relationships,
        entity_links=entity_links,
        rejected_records=rejected,
        provider_errors=provider_errors,
    )
    if quality != replayed_quality:
        raise NewsRunVerificationError("Quality report does not match replay")
    counts = manifest.get("record_counts")
    if not isinstance(counts, Mapping):
        raise NewsRunVerificationError("Manifest record counts are missing")
    expected_counts = {
        "raw": len(replay_records),
        "normalized": len(linked_articles),
        "duplicates": len(relationships),
        "rejected": len(rejected),
    }
    for key, expected in expected_counts.items():
        if counts.get(key) != expected:
            raise NewsRunVerificationError(f"Manifest {key} count differs from replay")

    database_verified = database_path is not None
    if database_path is not None:
        _verify_database_binding(
            database_path,
            run_id=run_id,
            status=status,
            manifest_digest=manifest_digest,
            completion_digest=_file_digest(completion_path),
            inventory_digest=inventory_digest,
            expected_inventory=expected_inventory,
            summary=summary,
        )

    return NewsRunVerification(
        run_id=run_id,
        status=status,
        artifact_count=len(expected_inventory),
        capture_count=len(captures),
        raw_record_count=len(replay_records),
        normalized_record_count=len(linked_articles),
        duplicate_count=len(relationships),
        artifact_inventory_sha256=inventory_digest,
        database_verified=database_verified,
    )


__all__ = [
    "NewsRunVerification",
    "NewsRunVerificationError",
    "verify_news_run",
]
