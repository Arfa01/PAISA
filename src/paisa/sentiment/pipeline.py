"""Immutable orchestration for bounded article-level sentiment inference."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
from time import monotonic
from typing import Any, Iterable, Mapping

from paisa.news.storage import NewsStorage, SENTIMENT_RUN_SUMMARY_SCHEMA_VERSION
from paisa.sentiment.artifacts import (
    SentimentRunCollisionError,
    json_bytes,
    logical_storage_label,
    make_sentiment_run_id,
    publish_staged_run,
    regular_file_inventory,
    utc_now_z,
    validate_sentiment_run_id,
    write_csv,
    write_json,
)
from paisa.sentiment.backend import (
    SentimentBackendIdentity,
    SentimentInferenceBackend,
    TransformersFinBERTBackend,
)
from paisa.sentiment.config import (
    CANONICAL_SENTIMENT_LABELS,
    SENTIMENT_SCHEMA_VERSION,
)
from paisa.sentiment.inference import (
    SentimentInferenceConfig,
    SentimentInferenceTimeoutError,
    iter_article_sentiment,
)
from paisa.sentiment.models import ArticleSentiment, ArticleSentimentError


_NON_MANIFEST_ARTIFACTS = (
    "article_sentiment.csv",
    "audit_sample.csv",
    "final_summary.json",
    "inference_errors.json",
    "input_manifest.json",
    "model_manifest.json",
    "quality_report.json",
)

_SEALED_PAYLOAD_ARTIFACTS = tuple(
    relative
    for relative in _NON_MANIFEST_ARTIFACTS
    if relative != "final_summary.json"
)

_RUN_MANIFEST_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "status",
        "started_at",
        "completed_at",
        "source_ingestion_run_id",
        "model_revision_id",
        "input_policy_id",
        "device",
        "batch_size",
        "timeout_seconds",
        "source_database",
        "artifact_directory",
        "model_cache",
        "artifact_inventory_self_exclusion",
        "sealed_payload_inventory_sha256",
        "academic_simulation",
        "financial_advice",
    }
)

_ARTICLE_SENTIMENT_FIELDS = (
    "schema_version",
    "sentiment_output_id",
    "sentiment_run_id",
    "article_id",
    "source_ingestion_run_id",
    "input_policy_id",
    "selected_fields",
    "input_text_sha256",
    "character_count",
    "original_character_count",
    "tokenizer_token_count",
    "final_token_count",
    "truncation_side",
    "truncation_applied",
    "character_truncation_applied",
    "token_truncation_applied",
    "maximum_input_characters",
    "maximum_sequence_length",
    "original_language",
    "model_repository",
    "model_revision",
    "tokenizer_revision",
    "class_labels",
    "negative_probability",
    "neutral_probability",
    "positive_probability",
    "predicted_label",
    "predicted_probability",
    "published_at",
    "available_at",
    "fetched_at",
    "inference_timestamp",
    "device",
    "batch_size",
    "library_versions",
    "warning_codes",
    "error_code",
    "error_message",
)

_AUDIT_FIELDS = (
    "article_id",
    "source_ingestion_run_id",
    "title",
    "ogdc_linked",
    "available_at",
    "negative_probability",
    "neutral_probability",
    "positive_probability",
    "predicted_label",
    "predicted_probability",
    "input_text_sha256",
    "selected_fields",
    "character_count",
    "original_character_count",
    "maximum_input_characters",
    "tokenizer_token_count",
    "final_token_count",
    "maximum_sequence_length",
    "truncation_side",
    "truncation_applied",
    "character_truncation_applied",
    "token_truncation_applied",
    "warning_codes",
)

_PINNED_RUNTIME_REQUIREMENTS = {
    "huggingface-hub": "1.32.0",
    "safetensors": "0.8.0",
    "tokenizers": "0.23.3",
    "torch": "2.10.0",
    "transformers": "5.20.2",
}


@dataclass(frozen=True)
class _SourceArticle:
    article_id: str
    ingestion_run_id: str
    title: str | None
    summary: str | None
    body: str | None
    language: str | None
    published_at: datetime | None
    available_at: datetime
    fetched_at: datetime


@dataclass(frozen=True)
class SentimentRunSummary:
    """Body-free public summary returned by the pipeline and CLI."""

    run_id: str
    status: str
    source_ingestion_run_id: str
    articles_requested: int
    articles_successfully_scored: int
    articles_skipped: int
    articles_failed: int
    unique_canonical_articles: int
    ogdc_linked_articles_scored: int
    label_counts: Mapping[str, int]
    truncated_input_count: int
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    device: str
    artifact_directory: str
    database: str
    any_articles_failed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_schema_version": SENTIMENT_RUN_SUMMARY_SCHEMA_VERSION,
            "schema_version": SENTIMENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "status": self.status,
            "source_ingestion_run_id": self.source_ingestion_run_id,
            "articles_requested": self.articles_requested,
            "articles_successfully_scored": self.articles_successfully_scored,
            "articles_skipped": self.articles_skipped,
            "articles_failed": self.articles_failed,
            "unique_canonical_articles": self.unique_canonical_articles,
            "ogdc_linked_articles_scored": self.ogdc_linked_articles_scored,
            "label_counts": {
                label: int(self.label_counts.get(label, 0))
                for label in CANONICAL_SENTIMENT_LABELS
            },
            "truncated_input_count": self.truncated_input_count,
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "device": self.device,
            "artifact_directory": self.artifact_directory,
            "database": self.database,
            "any_articles_failed": self.any_articles_failed,
            "academic_use_only": True,
            "financial_advice": False,
        }


def _parse_timestamp(value: Any, *, field: str, nullable: bool = False) -> datetime | None:
    if value is None or str(value).strip() == "":
        if nullable:
            return None
        raise ValueError(f"{field} is required")
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _source_article(row: Mapping[str, Any]) -> _SourceArticle:
    published = _parse_timestamp(row.get("published_at"), field="published_at", nullable=True)
    available = _parse_timestamp(row.get("available_at"), field="available_at")
    fetched = _parse_timestamp(row.get("fetched_at"), field="fetched_at")
    assert isinstance(available, datetime) and isinstance(fetched, datetime)
    return _SourceArticle(
        article_id=str(row["article_id"]),
        ingestion_run_id=str(row["source_ingestion_run_id"]),
        title=None if row.get("title") is None else str(row["title"]),
        summary=None if row.get("summary") is None else str(row["summary"]),
        body=None if row.get("body") is None else str(row["body"]),
        language=None if row.get("language") is None else str(row["language"]),
        published_at=published,
        available_at=available,
        fetched_at=fetched,
    )


def _resolved_storage_path(
    value: str | Path,
    *,
    storage_root: Path,
    allow_external_storage: bool,
    label: str,
) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = storage_root / candidate
    resolved = candidate.resolve(strict=False)
    if not allow_external_storage:
        try:
            resolved.relative_to(storage_root)
        except ValueError as exc:
            raise ValueError(
                f"{label} must remain inside --storage-root unless external storage is allowed"
            ) from exc
    return resolved


def _model_revision_id(backend: SentimentInferenceBackend) -> str:
    return "SMR-" + backend.identity.identity_sha256[:24]


def _installed_library_versions() -> dict[str, str]:
    values = {"python": platform.python_version()}
    for package in _PINNED_RUNTIME_REQUIREMENTS:
        try:
            values[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            values[package] = "not-installed"
    return dict(sorted(values.items()))


def _model_manifest(
    backend: SentimentInferenceBackend,
    *,
    model_revision_id: str,
    runtime_versions: Mapping[str, str],
    cache_label: str,
    load_status: str,
) -> dict[str, Any]:
    identity = backend.identity
    snapshot_inventory = (
        [dict(row) for row in backend.snapshot_inventory]
        if load_status == "LOADED"
        else [
            {"path": path, "sha256": digest, "size_bytes": None}
            for path, digest in sorted(identity.snapshot_files_sha256.items())
        ]
    )
    return {
        "schema_version": SENTIMENT_SCHEMA_VERSION,
        "model_revision_id": model_revision_id,
        "backend_identity_sha256": identity.identity_sha256,
        "model_repository": identity.model_repository,
        "model_revision": identity.model_revision,
        "tokenizer_repository": identity.tokenizer_repository,
        "tokenizer_revision": identity.tokenizer_revision,
        "revision_policy": "full_40_character_commit_hash",
        "intended_task": "three_class_financial_text_sentiment",
        "original_labels_by_index": {
            str(index): label
            for index, label in enumerate(identity.original_labels_by_index)
        },
        "canonical_labels": list(CANONICAL_SENTIMENT_LABELS),
        "canonical_probability_index_mapping": dict(
            sorted(identity.canonical_probability_index_mapping.items())
        ),
        "license_identifier": identity.license_identifier,
        "license_url": identity.license_url,
        "license_scope": "model_weights",
        "permitted_use": "non_commercial_academic_research",
        "commercial_use_requires_new_rights_review": True,
        "runtime_library_versions": dict(sorted(runtime_versions.items())),
        "pinned_runtime_requirements": dict(sorted(_PINNED_RUNTIME_REQUIREMENTS.items())),
        "model_cache": cache_label,
        "reviewed_snapshot_files": snapshot_inventory,
        "snapshot_bytes_verified": load_status == "LOADED",
        "model_config_sha256": identity.snapshot_files_sha256["config.json"],
        "tokenizer_config_sha256": identity.snapshot_files_sha256[
            "tokenizer_config.json"
        ],
        "model_weights_sha256": identity.snapshot_files_sha256[
            "model.safetensors"
        ],
        "tokenizer_vocabulary_sha256": identity.snapshot_files_sha256["vocab.txt"],
        "tokenizer_special_tokens_sha256": identity.snapshot_files_sha256[
            "special_tokens_map.json"
        ],
        "load_status": load_status,
        "silent_fallback_permitted": False,
    }


def _input_reference(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "article_id": str(row["article_id"]),
        "canonical_content_sha256": str(row["content_sha256"]),
        "canonical_owner_run_id": str(row["canonical_owner_run_id"]),
        "available_at": str(row["available_at"]),
        "original_language": row.get("language"),
        "ogdc_linked": bool(row.get("ogdc_linked", False)),
    }


def _input_fingerprint(references: Iterable[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(references),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _remaining_timeout_seconds(
    configured_timeout: float | None,
    *,
    started_clock: float,
) -> float | None:
    """Return the run-wide watchdog remainder, failing once it is exhausted."""

    if configured_timeout is None:
        return None
    remaining = configured_timeout - (monotonic() - started_clock)
    if remaining <= 0:
        raise SentimentInferenceTimeoutError(
            "Sentiment inference watchdog expired between batches."
        )
    return remaining


def _deterministic_audit_rows(
    output_rows: list[dict[str, Any]],
    source_metadata: Mapping[str, Mapping[str, Any]],
    *,
    sample_size: int,
    model_revision: str,
    input_policy_id: str,
) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> str:
        material = "\0".join(
            (str(row["article_id"]), model_revision, input_policy_id)
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    selected = sorted(output_rows, key=key)[:sample_size]
    result: list[dict[str, Any]] = []
    for row in selected:
        source = source_metadata[str(row["article_id"])]
        result.append(
            {
                "article_id": row["article_id"],
                "source_ingestion_run_id": row["source_ingestion_run_id"],
                "title": source.get("title"),
                "ogdc_linked": bool(source.get("ogdc_linked", False)),
                "available_at": row["available_at"],
                "negative_probability": row["negative_probability"],
                "neutral_probability": row["neutral_probability"],
                "positive_probability": row["positive_probability"],
                "predicted_label": row["predicted_label"],
                "predicted_probability": row["predicted_probability"],
                "input_text_sha256": row["input_text_sha256"],
                "selected_fields": row["selected_fields"],
                "character_count": row["character_count"],
                "original_character_count": row["original_character_count"],
                "maximum_input_characters": row["maximum_input_characters"],
                "tokenizer_token_count": row["tokenizer_token_count"],
                "final_token_count": row["final_token_count"],
                "maximum_sequence_length": row["maximum_sequence_length"],
                "truncation_side": row["truncation_side"],
                "truncation_applied": row["truncation_applied"],
                "character_truncation_applied": row[
                    "character_truncation_applied"
                ],
                "token_truncation_applied": row["token_truncation_applied"],
                "warning_codes": row["warning_codes"],
            }
        )
    return result


def _atomic_rewrite_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(json_bytes(value))
    os.replace(temporary, path)


def _inventory_rows(
    root: Path,
    relative_paths: Iterable[str],
) -> list[dict[str, Any]]:
    """Build deterministic checksums for an already-vetted artifact set."""

    return regular_file_inventory(root, files=tuple(sorted(relative_paths)))


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def _publication_inventory(run_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Bind the complete staged tree, including the self-excluded manifest."""

    manifest_bytes = json_bytes(run_manifest)
    return sorted(
        [
            *(dict(row) for row in run_manifest["artifact_inventory"]),
            {
                "path": "run_manifest.json",
                "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "size_bytes": len(manifest_bytes),
            },
        ],
        key=lambda row: str(row["path"]),
    )


def _bind_staged_pack(
    staging: Path,
    *,
    final_summary: Mapping[str, Any],
    manifest_contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind payloads and manifest provenance to the SQLite-bound summary.

    ``final_summary.json`` is bound by exact equality to immutable SQLite.  Its
    two seals then bind every other payload and every non-inventory manifest
    field without creating a recursive self-checksum.
    """

    observed_names = {candidate.name for candidate in staging.iterdir()}
    initial_names = set(_SEALED_PAYLOAD_ARTIFACTS)
    completed_names = set(_NON_MANIFEST_ARTIFACTS) | {"run_manifest.json"}
    if frozenset(observed_names) not in {
        frozenset(initial_names),
        frozenset(completed_names),
    }:
        raise RuntimeError("sentiment staging contains an unexpected artifact set")

    payload_inventory = _inventory_rows(staging, _SEALED_PAYLOAD_ARTIFACTS)
    payload_seal = _json_sha256(payload_inventory)
    bound_contract = dict(manifest_contract)
    bound_contract["sealed_payload_inventory_sha256"] = payload_seal
    if set(bound_contract) != _RUN_MANIFEST_CONTRACT_FIELDS:
        raise RuntimeError("sentiment run manifest contract fields are invalid")

    bound_summary = dict(final_summary)
    bound_summary["sealed_payload_inventory_sha256"] = payload_seal
    bound_summary["run_manifest_contract_sha256"] = _json_sha256(bound_contract)
    _atomic_rewrite_json(staging / "final_summary.json", bound_summary)

    run_manifest = {
        **bound_contract,
        "artifact_inventory": _inventory_rows(
            staging,
            _NON_MANIFEST_ARTIFACTS,
        ),
    }
    _atomic_rewrite_json(staging / "run_manifest.json", run_manifest)
    return bound_summary, run_manifest


def _persisted_run_matches_final_summary(
    persisted: Mapping[str, Any],
    *,
    status: str,
    final_summary: Mapping[str, Any],
) -> bool:
    """Recognize a driver error raised after the intended commit completed."""

    try:
        persisted_summary = json.loads(str(persisted["summary_json"]))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return (
        persisted.get("status") == status
        and persisted.get("completed_at") == final_summary.get("completed_at")
        and persisted_summary == dict(final_summary)
    )


def _summary_from_final_payload(payload: Mapping[str, Any]) -> SentimentRunSummary:
    """Rehydrate the public summary for a validated publication recovery."""

    return SentimentRunSummary(
        run_id=str(payload["run_id"]),
        status=str(payload["status"]),
        source_ingestion_run_id=str(payload["source_ingestion_run_id"]),
        articles_requested=int(payload["articles_requested"]),
        articles_successfully_scored=int(payload["articles_successfully_scored"]),
        articles_skipped=int(payload["articles_skipped"]),
        articles_failed=int(payload["articles_failed"]),
        unique_canonical_articles=int(payload["unique_canonical_articles"]),
        ogdc_linked_articles_scored=int(payload["ogdc_linked_articles_scored"]),
        label_counts={
            label: int(dict(payload["label_counts"])[label])
            for label in CANONICAL_SENTIMENT_LABELS
        },
        truncated_input_count=int(payload["truncated_input_count"]),
        model_repository=str(payload["model_repository"]),
        model_revision=str(payload["model_revision"]),
        tokenizer_revision=str(payload["tokenizer_revision"]),
        device=str(payload["device"]),
        artifact_directory=str(payload["artifact_directory"]),
        database=str(payload["database"]),
        any_articles_failed=bool(payload["any_articles_failed"]),
    )


def _recover_finalized_staging(
    *,
    storage: NewsStorage,
    staging: Path,
    destination: Path,
    run_id: str,
    source_ingestion_run_id: str,
    artifact_label: str,
) -> SentimentRunSummary:
    """Publish a checksum-valid staged pack after a prior SQLite commit.

    The deterministic hidden staging path is a small recovery journal for the
    unavoidable filesystem/SQLite transaction boundary.  Recovery never
    reruns inference or changes a finalized database record.
    """

    if staging.is_symlink() or not staging.is_dir():
        raise SentimentRunCollisionError(
            f"sentiment staging collision for immutable run: {run_id}"
        )
    persisted = storage.get_sentiment_run(run_id)
    if persisted is None or persisted.get("status") not in {
        "PASS",
        "PARTIAL",
        "FAIL",
    }:
        raise SentimentRunCollisionError(
            f"sentiment staging exists without a finalized run: {run_id}"
        )
    if (
        persisted.get("source_ingestion_run_id") != source_ingestion_run_id
        or persisted.get("artifact_directory") != artifact_label
    ):
        raise RuntimeError("staged sentiment run does not match SQLite identity")

    expected_names = set(_NON_MANIFEST_ARTIFACTS) | {"run_manifest.json"}
    observed_names = {candidate.name for candidate in staging.iterdir()}
    if observed_names != expected_names or any(
        candidate.is_symlink() or not candidate.is_file()
        for candidate in staging.iterdir()
    ):
        raise RuntimeError("staged sentiment artifact set is invalid")

    try:
        manifest = json.loads((staging / "run_manifest.json").read_text("utf-8"))
        final_summary = json.loads(
            (staging / "final_summary.json").read_text("utf-8")
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("staged sentiment metadata is invalid") from exc
    if not isinstance(manifest, dict) or not isinstance(final_summary, dict):
        raise RuntimeError("staged sentiment metadata must be JSON objects")

    if set(manifest) != _RUN_MANIFEST_CONTRACT_FIELDS | {"artifact_inventory"}:
        raise RuntimeError("staged sentiment manifest fields are invalid")
    manifest_contract = {
        field: manifest[field] for field in _RUN_MANIFEST_CONTRACT_FIELDS
    }
    payload_inventory = _inventory_rows(staging, _SEALED_PAYLOAD_ARTIFACTS)
    payload_seal = _json_sha256(payload_inventory)
    if (
        final_summary.get("sealed_payload_inventory_sha256") != payload_seal
        or manifest.get("sealed_payload_inventory_sha256") != payload_seal
        or final_summary.get("run_manifest_contract_sha256")
        != _json_sha256(manifest_contract)
    ):
        raise RuntimeError("staged sentiment evidence seal is invalid")
    if manifest.get("artifact_inventory") != _inventory_rows(
        staging,
        _NON_MANIFEST_ARTIFACTS,
    ):
        raise RuntimeError("staged sentiment checksum validation failed")

    if (
        manifest.get("schema_version") != persisted.get("schema_version")
        or manifest.get("run_id") != run_id
        or manifest.get("source_ingestion_run_id") != source_ingestion_run_id
        or manifest.get("model_revision_id")
        != persisted.get("model_revision_id")
        or manifest.get("input_policy_id") != persisted.get("input_policy_id")
        or manifest.get("started_at") != persisted.get("started_at")
        or manifest.get("device") != persisted.get("device")
        or manifest.get("batch_size") != persisted.get("batch_size")
        or manifest.get("timeout_seconds") != persisted.get("timeout_seconds")
        or manifest.get("artifact_directory") != artifact_label
        or manifest.get("source_database") != final_summary.get("database")
        or manifest.get("model_cache") != "configured-cache"
        or manifest.get("artifact_inventory_self_exclusion")
        != "run_manifest.json"
        or manifest.get("academic_simulation") is not True
        or manifest.get("financial_advice") is not False
        or final_summary.get("run_id") != run_id
        or final_summary.get("source_ingestion_run_id")
        != source_ingestion_run_id
        or final_summary.get("artifact_directory") != artifact_label
        or manifest.get("status") != final_summary.get("status")
        or manifest.get("completed_at") != final_summary.get("completed_at")
        or not _persisted_run_matches_final_summary(
            persisted,
            status=str(final_summary.get("status")),
            final_summary=final_summary,
        )
    ):
        raise RuntimeError("staged sentiment evidence does not match SQLite")

    summary = _summary_from_final_payload(final_summary)
    publish_staged_run(
        staging,
        destination,
        expected_inventory=_publication_inventory(manifest),
    )
    return summary


def _run_sentiment_inference_with_backend_for_testing(
    *,
    source_ingestion_run_id: str,
    database_path: str | Path = "data/news/paisa_news.sqlite3",
    output_dir: str | Path = "artifacts/sentiment_inference",
    model_cache: str | Path = "data/sentiment_model_cache",
    storage_root: str | Path = ".",
    allow_external_storage: bool = False,
    run_id: str | None = None,
    device: str = "cpu",
    config: SentimentInferenceConfig | None = None,
    backend: SentimentInferenceBackend | None = None,
    audit_sample_size: int = 10,
    discard_stale_running: bool = False,
) -> SentimentRunSummary:
    """Score one bounded, completed ingestion run and publish immutable evidence."""

    resolved_config = config or SentimentInferenceConfig()
    if resolved_config.maximum_articles is None:
        raise ValueError("run inference requires a finite maximum_articles bound")
    if audit_sample_size <= 0:
        raise ValueError("audit_sample_size must be positive")

    selected_run_id = validate_sentiment_run_id(
        run_id or make_sentiment_run_id()
    )
    trusted_root = Path(storage_root).expanduser().resolve(strict=True)
    database = _resolved_storage_path(
        database_path,
        storage_root=trusted_root,
        allow_external_storage=allow_external_storage,
        label="database",
    )
    output_root = _resolved_storage_path(
        output_dir,
        storage_root=trusted_root,
        allow_external_storage=allow_external_storage,
        label="output directory",
    )
    cache = _resolved_storage_path(
        model_cache,
        storage_root=trusted_root,
        allow_external_storage=allow_external_storage,
        label="model cache",
    )
    destination = output_root / selected_run_id
    staging = output_root / f".{selected_run_id}.staging"
    if destination.exists() or destination.is_symlink():
        raise SentimentRunCollisionError(
            f"sentiment artifact destination already exists: {selected_run_id}"
        )
    if not database.is_file():
        raise FileNotFoundError("sentiment inference database does not exist")

    output_root.mkdir(parents=True, exist_ok=True)
    database_label = logical_storage_label(
        database, storage_root=trusted_root, kind="database"
    )
    artifact_label = logical_storage_label(
        destination, storage_root=trusted_root, kind="artifact-directory"
    )
    # The cache location is operationally private. Public artifacts retain only
    # this opaque descriptor, never a relative or absolute filesystem path.
    cache_label = "configured-cache"
    storage = NewsStorage(database).initialize()
    persisted_run = storage.get_sentiment_run(selected_run_id)
    if staging.exists() or staging.is_symlink():
        if persisted_run is not None and persisted_run.get("status") != "RUNNING":
            return _recover_finalized_staging(
                storage=storage,
                staging=staging,
                destination=destination,
                run_id=selected_run_id,
                source_ingestion_run_id=source_ingestion_run_id,
                artifact_label=artifact_label,
            )
        if not discard_stale_running:
            raise SentimentRunCollisionError(
                f"unfinished sentiment run requires explicit discard: {selected_run_id}"
            )
        if staging.is_symlink() or not staging.is_dir():
            raise SentimentRunCollisionError(
                f"sentiment staging collision for immutable run: {selected_run_id}"
            )
        shutil.rmtree(staging)
        if persisted_run is None:
            raise SentimentRunCollisionError(
                f"staging exists without a matching RUNNING run: {selected_run_id}"
            )
        storage.discard_running_sentiment_run(selected_run_id)
        persisted_run = None
    if persisted_run is not None:
        if persisted_run.get("status") == "RUNNING" and discard_stale_running:
            storage.discard_running_sentiment_run(selected_run_id)
        else:
            raise SentimentRunCollisionError(
                f"sentiment run identity already exists without recoverable staging: "
                f"{selected_run_id}"
            )

    cache.mkdir(parents=True, exist_ok=True)
    selected_backend = backend or TransformersFinBERTBackend(
        device=device,
        cache_dir=cache,
    )
    if selected_backend.device != device.strip().lower():
        raise ValueError("backend device does not match requested device")
    identity = selected_backend.identity
    if not isinstance(identity, SentimentBackendIdentity):
        raise TypeError("sentiment backend must supply a verifiable identity")
    if (
        selected_backend.model_repository != identity.model_repository
        or selected_backend.model_revision != identity.model_revision
        or selected_backend.tokenizer_revision != identity.tokenizer_revision
    ):
        raise ValueError("backend runtime identity does not match its manifest identity")

    started_at = utc_now_z()
    inference_timestamp = _parse_timestamp(started_at, field="started_at")
    assert isinstance(inference_timestamp, datetime)
    started_clock = monotonic()

    requested_model_revision_id = _model_revision_id(selected_backend)
    model_revision_id = storage.register_sentiment_model_revision(
        model_revision_id=requested_model_revision_id,
        schema_version=SENTIMENT_SCHEMA_VERSION,
        model_repository=identity.model_repository,
        model_revision=identity.model_revision,
        tokenizer_repository=identity.tokenizer_repository,
        tokenizer_revision=identity.tokenizer_revision,
        original_labels=list(identity.original_labels_by_index),
        label_mapping=dict(identity.canonical_probability_index_mapping),
        license_identifier=identity.license_identifier,
        library_versions=_PINNED_RUNTIME_REQUIREMENTS,
        backend_identity_sha256=identity.identity_sha256,
        snapshot_inventory=[
            {"path": path, "sha256": digest}
            for path, digest in sorted(identity.snapshot_files_sha256.items())
        ],
        model_config_sha256=identity.snapshot_files_sha256["config.json"],
        tokenizer_config_sha256=identity.snapshot_files_sha256[
            "tokenizer_config.json"
        ],
        model_weights_sha256=identity.snapshot_files_sha256[
            "model.safetensors"
        ],
        tokenizer_vocabulary_sha256=identity.snapshot_files_sha256["vocab.txt"],
        tokenizer_special_tokens_sha256=identity.snapshot_files_sha256[
            "special_tokens_map.json"
        ],
        created_at=started_at,
    )
    storage.begin_sentiment_run(
        run_id=selected_run_id,
        schema_version=SENTIMENT_SCHEMA_VERSION,
        source_ingestion_run_id=source_ingestion_run_id,
        model_revision_id=model_revision_id,
        input_policy_id=resolved_config.effective_input_policy_id,
        started_at=started_at,
        device=selected_backend.device,
        batch_size=resolved_config.batch_size,
        maximum_input_characters=resolved_config.maximum_input_characters,
        maximum_sequence_length=resolved_config.maximum_sequence_length,
        maximum_articles=resolved_config.maximum_articles,
        timeout_seconds=resolved_config.timeout_seconds,
        artifact_directory=artifact_label,
        configuration={
            "audit_sample_size": audit_sample_size,
            "model_cache": cache_label,
        },
    )

    input_references: list[dict[str, Any]] = []
    source_metadata: dict[str, dict[str, Any]] = {}
    output_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    skipped_existing = 0
    skipped_article_errors = 0
    failed_article_errors = 0
    fatal_error = False
    completed_input_ids: set[str] = set()
    runtime_versions = _installed_library_versions()
    model_load_status = "FAILED"

    try:
        runtime_versions = dict(selected_backend.library_versions)
        model_load_status = "LOADED"
        for source_batch in storage.iter_sentiment_input_articles(
            source_ingestion_run_id,
            batch_size=max(resolved_config.batch_size, 1),
            maximum_articles=resolved_config.maximum_articles,
        ):
            pending: list[_SourceArticle] = []
            for source_row in source_batch:
                article_id = str(source_row["article_id"])
                storage.register_sentiment_run_input(
                    selected_run_id,
                    article_id,
                    input_ordinal=len(input_references),
                )
                input_references.append(_input_reference(source_row))
                source_metadata[article_id] = {
                    "title": source_row.get("title"),
                    "ogdc_linked": bool(source_row.get("ogdc_linked", False)),
                    "language": source_row.get("language"),
                }
                existing = storage.get_article_sentiment(
                    article_id,
                    model_revision_id,
                    resolved_config.effective_input_policy_id,
                )
                if existing is not None:
                    storage.complete_sentiment_run_input(
                        selected_run_id,
                        article_id,
                        outcome="REUSED",
                        sentiment_output_id=str(existing["sentiment_output_id"]),
                    )
                    completed_input_ids.add(article_id)
                    skipped_existing += 1
                    continue
                pending.append(_source_article(source_row))

            if not pending:
                continue
            batch_config = replace(
                resolved_config,
                maximum_articles=None,
                timeout_seconds=_remaining_timeout_seconds(
                    resolved_config.timeout_seconds,
                    started_clock=started_clock,
                ),
            )
            for outcome in iter_article_sentiment(
                pending,
                backend=selected_backend,
                config=batch_config,
                inference_timestamp=inference_timestamp,
            ):
                if isinstance(outcome, ArticleSentiment):
                    output_id, inserted = storage.save_article_sentiment(
                        outcome,
                        inference_run_id=selected_run_id,
                        model_revision_id=model_revision_id,
                    )
                    if not inserted:
                        storage.complete_sentiment_run_input(
                            selected_run_id,
                            outcome.article_id,
                            outcome="REUSED",
                            sentiment_output_id=output_id,
                        )
                        completed_input_ids.add(outcome.article_id)
                        skipped_existing += 1
                        continue
                    storage.complete_sentiment_run_input(
                        selected_run_id,
                        outcome.article_id,
                        outcome="GENERATED",
                        sentiment_output_id=output_id,
                    )
                    completed_input_ids.add(outcome.article_id)
                    row = outcome.to_dict()
                    row["sentiment_output_id"] = output_id
                    row["sentiment_run_id"] = selected_run_id
                    output_rows.append(row)
                    continue

                assert isinstance(outcome, ArticleSentimentError)
                error = outcome.to_dict()
                is_skip = outcome.error_code == "unsupported_language"
                if is_skip:
                    skipped_article_errors += 1
                else:
                    failed_article_errors += 1
                error_id = storage.save_sentiment_error(
                    inference_run_id=selected_run_id,
                    article_id=outcome.article_id,
                    stage=("input_validation" if is_skip else "inference"),
                    error_type=outcome.error_code,
                    message=outcome.error_message,
                    occurred_at=error["inference_timestamp"],
                    input_text_sha256=outcome.input_text_sha256,
                    selected_fields=list(outcome.selected_fields),
                    original_language=outcome.original_language,
                    warning_codes=list(outcome.warning_codes),
                    retryable=False,
                )
                storage.complete_sentiment_run_input(
                    selected_run_id,
                    outcome.article_id,
                    outcome=("SKIPPED" if is_skip else "FAILED"),
                    error_id=error_id,
                )
                completed_input_ids.add(outcome.article_id)
                error_rows.append(error)
    except Exception as exc:
        fatal_error = True
        safe_error_type = type(exc).__name__
        safe_message = (
            "Pinned sentiment runtime failed; no substitute model was used."
        )
        # Every registered input needs a terminal association before a
        # partially successful run can be certified. Attribute the fatal
        # interruption to each not-yet-completed input; the separate run-level
        # error below preserves the root failure for operators.
        for reference in input_references:
            article_id = str(reference["article_id"])
            if article_id in completed_input_ids:
                continue
            occurred_at = utc_now_z()
            try:
                error_id = storage.save_sentiment_error(
                    inference_run_id=selected_run_id,
                    article_id=article_id,
                    stage="run",
                    error_type=safe_error_type,
                    message=safe_message,
                    occurred_at=occurred_at,
                    original_language=reference.get("original_language"),
                    retryable=False,
                )
                storage.complete_sentiment_run_input(
                    selected_run_id,
                    article_id,
                    outcome="FAILED",
                    error_id=error_id,
                )
            except Exception:
                # The finalization trigger will reject a PARTIAL run if storage
                # could not persist the terminal association. Its existing
                # recovery path then records a non-reusable FAIL instead.
                continue
            completed_input_ids.add(article_id)
            failed_article_errors += 1
            error_rows.append(
                {
                    "schema_version": SENTIMENT_SCHEMA_VERSION,
                    "article_id": article_id,
                    "source_ingestion_run_id": source_ingestion_run_id,
                    "input_policy_id": resolved_config.effective_input_policy_id,
                    "selected_fields": [],
                    "input_text_sha256": None,
                    "original_language": reference.get("original_language"),
                    "model_repository": selected_backend.model_repository,
                    "model_revision": selected_backend.model_revision,
                    "tokenizer_revision": selected_backend.tokenizer_revision,
                    "published_at": None,
                    "available_at": reference.get("available_at"),
                    "fetched_at": None,
                    "inference_timestamp": occurred_at,
                    "warning_codes": [],
                    "error_code": safe_error_type,
                    "error_message": safe_message,
                }
            )
        try:
            storage.save_sentiment_error(
                inference_run_id=selected_run_id,
                stage="run",
                error_type=safe_error_type,
                message=safe_message,
                occurred_at=utc_now_z(),
                retryable=False,
            )
        except Exception:
            pass
        error_rows.append(
            {
                "schema_version": SENTIMENT_SCHEMA_VERSION,
                "article_id": None,
                "source_ingestion_run_id": source_ingestion_run_id,
                "input_policy_id": resolved_config.effective_input_policy_id,
                "selected_fields": [],
                "input_text_sha256": None,
                "original_language": None,
                "model_repository": selected_backend.model_repository,
                "model_revision": selected_backend.model_revision,
                "tokenizer_revision": selected_backend.tokenizer_revision,
                "published_at": None,
                "available_at": None,
                "fetched_at": None,
                "inference_timestamp": utc_now_z(),
                "warning_codes": [],
                "error_code": safe_error_type,
                "error_message": safe_message,
            }
        )

    if not input_references and not fatal_error:
        fatal_error = True
        message = "Completed source run contains no canonical sentiment inputs."
        storage.save_sentiment_error(
            inference_run_id=selected_run_id,
            stage="input",
            error_type="empty_input",
            message=message,
            occurred_at=utc_now_z(),
            retryable=False,
        )
        error_rows.append(
            {
                "schema_version": SENTIMENT_SCHEMA_VERSION,
                "article_id": None,
                "source_ingestion_run_id": source_ingestion_run_id,
                "input_policy_id": resolved_config.effective_input_policy_id,
                "selected_fields": [],
                "input_text_sha256": None,
                "original_language": None,
                "model_repository": selected_backend.model_repository,
                "model_revision": selected_backend.model_revision,
                "tokenizer_revision": selected_backend.tokenizer_revision,
                "published_at": None,
                "available_at": None,
                "fetched_at": None,
                "inference_timestamp": utc_now_z(),
                "warning_codes": [],
                "error_code": "empty_input",
                "error_message": message,
            }
        )

    if not output_rows and (fatal_error or failed_article_errors > 0):
        status = "FAIL"
    elif fatal_error or error_rows:
        status = "PARTIAL"
    else:
        status = "PASS"

    label_counts = Counter(row["predicted_label"] for row in output_rows)
    language_counts = Counter(
        str(reference.get("original_language") or "unknown")
        for reference in input_references
    )
    successful_available = [str(row["available_at"]) for row in output_rows]
    duration_seconds = monotonic() - started_clock
    truncated_count = sum(bool(row["truncation_applied"]) for row in output_rows)
    title_only_count = sum(
        tuple(row["selected_fields"]) == ("title",) for row in output_rows
    )
    ogdc_scored = sum(
        bool(source_metadata[str(row["article_id"])]["ogdc_linked"])
        for row in output_rows
    )
    quality_report = {
        "schema_version": SENTIMENT_SCHEMA_VERSION,
        "quality_status": status,
        "articles_requested": len(input_references),
        "articles_successfully_scored": len(output_rows),
        "articles_skipped": skipped_existing + skipped_article_errors,
        "articles_failed": failed_article_errors,
        "run_error_count": sum(row.get("article_id") is None for row in error_rows),
        "negative_count": int(label_counts.get("negative", 0)),
        "neutral_count": int(label_counts.get("neutral", 0)),
        "positive_count": int(label_counts.get("positive", 0)),
        "mean_probabilities": {
            label: _mean([float(row[f"{label}_probability"]) for row in output_rows])
            for label in CANONICAL_SENTIMENT_LABELS
        },
        "truncated_input_count": truncated_count,
        "title_only_count": title_only_count,
        "language_counts": dict(sorted(language_counts.items())),
        "ogdc_linked_scored_count": ogdc_scored,
        "earliest_available_at": min(successful_available) if successful_available else None,
        "latest_available_at": max(successful_available) if successful_available else None,
        "model_repository": selected_backend.model_repository,
        "model_revision": selected_backend.model_revision,
        "tokenizer_revision": selected_backend.tokenizer_revision,
        "device": selected_backend.device,
        "processing_duration_seconds": duration_seconds,
        "accuracy_evaluation": (
            "Sentiment outputs were generated, but classification accuracy was not "
            "evaluated in this run."
            if output_rows
            else "No sentiment outputs were generated; classification accuracy was "
            "not evaluated in this run."
        ),
        "limitations": [
            "Article sentiment is not a stock prediction or trading signal.",
            "Unlabelled source articles do not measure classification accuracy.",
            "One short RSS snapshot is insufficient for stock-model integration.",
        ],
    }
    summary = SentimentRunSummary(
        run_id=selected_run_id,
        status=status,
        source_ingestion_run_id=source_ingestion_run_id,
        articles_requested=len(input_references),
        articles_successfully_scored=len(output_rows),
        articles_skipped=skipped_existing + skipped_article_errors,
        articles_failed=failed_article_errors,
        unique_canonical_articles=len(input_references),
        ogdc_linked_articles_scored=ogdc_scored,
        label_counts=label_counts,
        truncated_input_count=truncated_count,
        model_repository=selected_backend.model_repository,
        model_revision=selected_backend.model_revision,
        tokenizer_revision=selected_backend.tokenizer_revision,
        device=selected_backend.device,
        artifact_directory=artifact_label,
        database=database_label,
        any_articles_failed=failed_article_errors > 0 or fatal_error,
    )
    final_summary = summary.to_dict()
    final_summary["completed_at"] = utc_now_z()
    final_summary["limitations"] = quality_report["limitations"]

    input_manifest = {
        "schema_version": SENTIMENT_SCHEMA_VERSION,
        "source_ingestion_run_id": source_ingestion_run_id,
        "source_database": database_label,
        "source_record_type": "normalized_canonical_article",
        "article_count": len(input_references),
        "articles_sha256": _input_fingerprint(input_references),
        "articles": input_references,
        "input_policy_id": resolved_config.effective_input_policy_id,
        "selected_field_order": [
            ["title", "summary"],
            ["title", "body"],
            ["title"],
        ],
        "maximum_input_characters": resolved_config.maximum_input_characters,
        "maximum_sequence_length": resolved_config.maximum_sequence_length,
        "maximum_articles": resolved_config.maximum_articles,
    }
    model_manifest = _model_manifest(
        selected_backend,
        model_revision_id=model_revision_id,
        runtime_versions=runtime_versions,
        cache_label=cache_label,
        load_status=model_load_status,
    )
    audit_rows = _deterministic_audit_rows(
        output_rows,
        source_metadata,
        sample_size=audit_sample_size,
        model_revision=selected_backend.model_revision,
        input_policy_id=resolved_config.effective_input_policy_id,
    )

    staging.mkdir(mode=0o700)
    try:
        write_json(staging / "model_manifest.json", model_manifest)
        write_json(staging / "input_manifest.json", input_manifest)
        write_csv(
            staging / "article_sentiment.csv",
            output_rows,
            _ARTICLE_SENTIMENT_FIELDS,
        )
        write_json(
            staging / "inference_errors.json",
            {
                "schema_version": SENTIMENT_SCHEMA_VERSION,
                "run_id": selected_run_id,
                "error_count": len(error_rows),
                "errors": error_rows,
            },
        )
        write_csv(staging / "audit_sample.csv", audit_rows, _AUDIT_FIELDS)
        write_json(staging / "quality_report.json", quality_report)
        manifest_contract = {
            "schema_version": SENTIMENT_SCHEMA_VERSION,
            "run_id": selected_run_id,
            "status": status,
            "started_at": started_at,
            "completed_at": final_summary["completed_at"],
            "source_ingestion_run_id": source_ingestion_run_id,
            "model_revision_id": model_revision_id,
            "input_policy_id": resolved_config.effective_input_policy_id,
            "device": selected_backend.device,
            "batch_size": resolved_config.batch_size,
            "timeout_seconds": resolved_config.timeout_seconds,
            "source_database": database_label,
            "artifact_directory": artifact_label,
            "model_cache": cache_label,
            "artifact_inventory_self_exclusion": "run_manifest.json",
            "academic_simulation": True,
            "financial_advice": False,
        }
        final_summary, run_manifest = _bind_staged_pack(
            staging,
            final_summary=final_summary,
            manifest_contract=manifest_contract,
        )
        publication_inventory = _publication_inventory(run_manifest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        failed_at = utc_now_z()
        failed_summary = dict(final_summary)
        failed_summary["status"] = "FAIL"
        failed_summary["completed_at"] = failed_at
        failed_summary["any_articles_failed"] = True
        try:
            # No complete staged pack exists, so outputs generated by this run
            # have no artifact seal. Remove them while the owner is still
            # RUNNING rather than exposing uncertified global reuse keys.
            storage.reset_unpublished_sentiment_outputs(selected_run_id)
            storage.save_sentiment_error(
                inference_run_id=selected_run_id,
                stage="artifact_publication",
                error_type="artifact_publication_failed",
                message="Sentiment artifact publication failed.",
                occurred_at=failed_at,
                retryable=False,
            )
            storage.finalize_sentiment_run(
                selected_run_id,
                status="FAIL",
                completed_at=failed_at,
                summary=failed_summary,
                error_message="Sentiment artifact publication failed.",
            )
        except Exception:
            # Preserve the original publication exception. A simultaneous
            # storage failure may prevent finalization, but no artifact can be
            # published as PASS from this branch.
            pass
        raise

    try:
        storage.finalize_sentiment_run(
            selected_run_id,
            status=status,
            completed_at=final_summary["completed_at"],
            summary=final_summary,
            error_message=(
                "Sentiment inference failed; inspect structured run evidence."
                if status == "FAIL"
                else None
            ),
        )
    except Exception as finalization_error:
        # A driver can theoretically report an error after commit. If SQLite
        # already contains the exact intended final status, the untouched
        # staged pack is safe to publish.
        try:
            persisted = storage.get_sentiment_run(selected_run_id)
        except Exception:
            persisted = None
        if persisted is not None and _persisted_run_matches_final_summary(
            persisted,
            status=status,
            final_summary=final_summary,
        ):
            publish_staged_run(
                staging,
                destination,
                expected_inventory=publication_inventory,
            )
            return summary

        # A finalized row with different completion evidence is not this
        # process's commit. Never publish or rewrite the private staging pack.
        if persisted is not None and persisted.get("status") != "RUNNING":
            raise RuntimeError(
                "Finalized SQLite evidence does not match the staged run; "
                "no public run was published."
            ) from finalization_error

        # The staging directory is still private. Convert it to truthful FAIL
        # evidence, then publish it only if SQLite also finalizes as FAIL.
        failed_at = utc_now_z()
        status = "FAIL"
        final_summary["status"] = status
        final_summary["completed_at"] = failed_at
        final_summary["any_articles_failed"] = True
        quality_report["quality_status"] = status
        quality_report["run_error_count"] = int(quality_report["run_error_count"]) + 1
        error_rows.append(
            {
                "schema_version": SENTIMENT_SCHEMA_VERSION,
                "article_id": None,
                "source_ingestion_run_id": source_ingestion_run_id,
                "input_policy_id": resolved_config.effective_input_policy_id,
                "selected_fields": [],
                "input_text_sha256": None,
                "original_language": None,
                "model_repository": selected_backend.model_repository,
                "model_revision": selected_backend.model_revision,
                "tokenizer_revision": selected_backend.tokenizer_revision,
                "published_at": None,
                "available_at": None,
                "fetched_at": None,
                "inference_timestamp": failed_at,
                "warning_codes": [],
                "error_code": "storage_finalization_failed",
                "error_message": "Sentiment run finalization failed before artifact publication.",
            }
        )
        _atomic_rewrite_json(staging / "quality_report.json", quality_report)
        _atomic_rewrite_json(
            staging / "inference_errors.json",
            {
                "schema_version": SENTIMENT_SCHEMA_VERSION,
                "run_id": selected_run_id,
                "error_count": len(error_rows),
                "errors": error_rows,
            },
        )
        manifest_contract["status"] = status
        manifest_contract["completed_at"] = failed_at
        final_summary, run_manifest = _bind_staged_pack(
            staging,
            final_summary=final_summary,
            manifest_contract=manifest_contract,
        )
        publication_inventory = _publication_inventory(run_manifest)
        try:
            # A FAIL owner is never a reusable result source. Remove outputs
            # produced by this still-RUNNING attempt before finalizing so the
            # global version-5 idempotency key cannot block a clean retry.
            storage.reset_unpublished_sentiment_outputs(selected_run_id)
            storage.save_sentiment_error(
                inference_run_id=selected_run_id,
                stage="finalization",
                error_type="storage_finalization_failed",
                message="Sentiment run finalization failed before artifact publication.",
                occurred_at=failed_at,
                retryable=False,
            )
            storage.finalize_sentiment_run(
                selected_run_id,
                status="FAIL",
                completed_at=failed_at,
                summary=final_summary,
                error_message="Sentiment run finalization failed.",
            )
        except Exception:
            raise RuntimeError(
                "Sentiment run finalization failed; no public run was published."
            ) from finalization_error
        summary = replace(summary, status="FAIL", any_articles_failed=True)

    # Publication is the final commit marker. A completed directory can never
    # become visible before its matching SQLite row reaches a final status.
    publish_staged_run(
        staging,
        destination,
        expected_inventory=publication_inventory,
    )
    return summary


def run_sentiment_inference(
    *,
    source_ingestion_run_id: str,
    database_path: str | Path = "data/news/paisa_news.sqlite3",
    output_dir: str | Path = "artifacts/sentiment_inference",
    model_cache: str | Path = "data/sentiment_model_cache",
    storage_root: str | Path = ".",
    allow_external_storage: bool = False,
    run_id: str | None = None,
    device: str = "cpu",
    config: SentimentInferenceConfig | None = None,
    audit_sample_size: int = 10,
    discard_stale_running: bool = False,
) -> SentimentRunSummary:
    """Run the exact reviewed FinBERT backend and publish immutable evidence.

    Backend injection is intentionally absent from this production boundary so
    callers cannot publish arbitrary classifier output under the reviewed
    FinBERT identity.  Unit tests exercise orchestration through the private
    helper above.
    """

    return _run_sentiment_inference_with_backend_for_testing(
        source_ingestion_run_id=source_ingestion_run_id,
        database_path=database_path,
        output_dir=output_dir,
        model_cache=model_cache,
        storage_root=storage_root,
        allow_external_storage=allow_external_storage,
        run_id=run_id,
        device=device,
        config=config,
        backend=None,
        audit_sample_size=audit_sample_size,
        discard_stale_running=discard_stale_running,
    )


__all__ = ["SentimentRunSummary", "run_sentiment_inference"]
