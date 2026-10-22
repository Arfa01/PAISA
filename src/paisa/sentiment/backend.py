"""Inference backend contract and lazy Hugging Face FinBERT implementation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from paisa.sentiment.config import (
    CANONICAL_SENTIMENT_LABELS,
    FINBERT_MODEL_REPOSITORY,
    FINBERT_MODEL_REVISION,
    FINBERT_ORIGINAL_LABELS_BY_INDEX,
    FINBERT_LICENSE,
    FINBERT_LICENSE_URL,
    FINBERT_REVIEWED_SNAPSHOT_SHA256,
    FINBERT_TOKENIZER_REVISION,
    PROBABILITY_SUM_TOLERANCE,
)
from paisa.sentiment.models import BackendPrediction


_COMMIT_REVISION = re.compile(r"^[0-9a-f]{40}$")


class SentimentBackendError(RuntimeError):
    """Base class for backend loading and inference failures."""


class ModelLabelMappingError(SentimentBackendError):
    """Raised when runtime model labels differ from the reviewed mapping."""


class DeviceUnavailableError(SentimentBackendError):
    """Raised when an explicitly requested accelerator is unavailable."""


class ModelSnapshotIntegrityError(SentimentBackendError):
    """Raised when cached model bytes do not match the reviewed snapshot."""


def validate_immutable_revision(revision: str) -> str:
    if not _COMMIT_REVISION.fullmatch(revision):
        raise ValueError("Model revisions must be full 40-character commit hashes.")
    return revision


@dataclass(frozen=True)
class SentimentBackendIdentity:
    """Verifiable model/tokenizer identity supplied by an inference backend."""

    model_repository: str
    model_revision: str
    tokenizer_repository: str
    tokenizer_revision: str
    original_labels_by_index: tuple[str, ...]
    canonical_probability_index_mapping: Mapping[str, int]
    license_identifier: str
    license_url: str
    snapshot_files_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.model_repository.strip() or not self.tokenizer_repository.strip():
            raise ValueError("Backend repositories cannot be blank.")
        validate_immutable_revision(self.model_revision)
        validate_immutable_revision(self.tokenizer_revision)
        if not self.license_identifier.strip() or not self.license_url.strip():
            raise ValueError("Backend identity requires license metadata.")
        if not self.snapshot_files_sha256:
            raise ValueError("Backend identity requires a snapshot file inventory.")
        for filename, digest in self.snapshot_files_sha256.items():
            candidate = Path(str(filename))
            if (
                candidate.is_absolute()
                or len(candidate.parts) != 1
                or candidate.name in {"", ".", ".."}
            ):
                raise ValueError("Snapshot inventory paths must be simple filenames.")
            if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
                raise ValueError("Snapshot inventory hashes must be lowercase SHA-256.")
        expected_mapping = {
            label: index
            for index, label in enumerate(self.original_labels_by_index)
        }
        if dict(self.canonical_probability_index_mapping) != {
            label: expected_mapping[label] for label in CANONICAL_SENTIMENT_LABELS
        }:
            raise ValueError("Backend label mapping is inconsistent with source labels.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "tokenizer_repository": self.tokenizer_repository,
            "tokenizer_revision": self.tokenizer_revision,
            "original_labels_by_index": list(self.original_labels_by_index),
            "canonical_probability_index_mapping": dict(
                sorted(self.canonical_probability_index_mapping.items())
            ),
            "license_identifier": self.license_identifier,
            "license_url": self.license_url,
            "snapshot_files_sha256": dict(sorted(self.snapshot_files_sha256.items())),
        }

    @property
    def identity_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def reviewed_finbert_identity() -> SentimentBackendIdentity:
    """Return the immutable identity of the only production sentiment model."""

    return SentimentBackendIdentity(
        model_repository=FINBERT_MODEL_REPOSITORY,
        model_revision=FINBERT_MODEL_REVISION,
        tokenizer_repository=FINBERT_MODEL_REPOSITORY,
        tokenizer_revision=FINBERT_TOKENIZER_REVISION,
        original_labels_by_index=FINBERT_ORIGINAL_LABELS_BY_INDEX,
        canonical_probability_index_mapping={
            "negative": 1,
            "neutral": 2,
            "positive": 0,
        },
        license_identifier=FINBERT_LICENSE,
        license_url=FINBERT_LICENSE_URL,
        snapshot_files_sha256=FINBERT_REVIEWED_SNAPSHOT_SHA256,
    )


def _hash_regular_snapshot_file(path: Path, *, trusted_root: Path | None) -> tuple[str, int]:
    """Hash one stable regular file without following a link at open time."""

    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ModelSnapshotIntegrityError(
            f"Reviewed snapshot file is unavailable: {path.name}."
        ) from exc
    if trusted_root is not None:
        try:
            resolved.relative_to(trusted_root.resolve(strict=True))
        except (OSError, ValueError) as exc:
            raise ModelSnapshotIntegrityError(
                f"Reviewed snapshot file escapes the configured cache: {path.name}."
            ) from exc
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ModelSnapshotIntegrityError(
            f"Reviewed snapshot file cannot be opened safely: {path.name}."
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ModelSnapshotIntegrityError(
                f"Reviewed snapshot entry is not a regular file: {path.name}."
            )
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ModelSnapshotIntegrityError(
                f"Reviewed snapshot file changed while hashing: {path.name}."
            )
        return digest.hexdigest(), before.st_size
    finally:
        os.close(descriptor)


def verify_reviewed_snapshot(
    files: Mapping[str, str | Path],
    *,
    expected_sha256: Mapping[str, str],
    trusted_root: Path | None,
) -> tuple[dict[str, Any], ...]:
    """Return a deterministic inventory only when every reviewed byte matches."""

    if set(files) != set(expected_sha256):
        raise ModelSnapshotIntegrityError(
            "Resolved snapshot does not contain the complete reviewed file set."
        )
    inventory: list[dict[str, Any]] = []
    for filename in sorted(expected_sha256):
        observed, size = _hash_regular_snapshot_file(
            Path(files[filename]), trusted_root=trusted_root
        )
        if observed != expected_sha256[filename]:
            raise ModelSnapshotIntegrityError(
                f"Reviewed snapshot hash mismatch: {filename}."
            )
        inventory.append(
            {"path": filename, "sha256": observed, "size_bytes": size}
        )
    return tuple(inventory)


def _materialize_reviewed_snapshot(
    files: Mapping[str, str | Path],
    *,
    destination: Path,
    trusted_root: Path | None,
) -> dict[str, Path]:
    """Copy only reviewed regular files into one new private load directory."""

    destination_status = destination.lstat()
    if not stat.S_ISDIR(destination_status.st_mode):
        raise ModelSnapshotIntegrityError(
            "Reviewed snapshot isolation root is not a real directory."
        )
    materialized: dict[str, Path] = {}
    for filename in sorted(files):
        candidate = Path(filename)
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.name in {"", ".", ".."}
        ):
            raise ModelSnapshotIntegrityError(
                "Reviewed snapshot paths must be simple filenames."
            )
        source = Path(files[filename])
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ModelSnapshotIntegrityError(
                f"Reviewed snapshot file is unavailable: {filename}."
            ) from exc
        if trusted_root is not None:
            try:
                resolved.relative_to(trusted_root.resolve(strict=True))
            except (OSError, ValueError) as exc:
                raise ModelSnapshotIntegrityError(
                    f"Reviewed snapshot file escapes the configured cache: {filename}."
                ) from exc

        source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        source_flags |= getattr(os, "O_NOFOLLOW", 0)
        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        destination_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(
            os, "O_NOFOLLOW", 0
        )
        target = destination / filename
        source_fd: int | None = None
        target_fd: int | None = None
        try:
            source_fd = os.open(resolved, source_flags)
            before = os.fstat(source_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ModelSnapshotIntegrityError(
                    f"Reviewed snapshot entry is not a regular file: {filename}."
                )
            target_fd = os.open(target, destination_flags, 0o600)
            while block := os.read(source_fd, 1024 * 1024):
                view = memoryview(block)
                while view:
                    written = os.write(target_fd, view)
                    if written <= 0:
                        raise OSError("short write while isolating reviewed snapshot")
                    view = view[written:]
            after = os.fstat(source_fd)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise ModelSnapshotIntegrityError(
                    f"Reviewed snapshot file changed while isolating: {filename}."
                )
            os.fchmod(target_fd, 0o400)
        except ModelSnapshotIntegrityError:
            raise
        except OSError as exc:
            raise ModelSnapshotIntegrityError(
                f"Reviewed snapshot file could not be isolated safely: {filename}."
            ) from exc
        finally:
            if target_fd is not None:
                os.close(target_fd)
            if source_fd is not None:
                os.close(source_fd)
        materialized[filename] = target

    if set(materialized) != set(files):
        raise ModelSnapshotIntegrityError(
            "Reviewed snapshot isolation did not copy the complete file set."
        )
    return materialized


def validate_finbert_label_mapping(
    id2label: Mapping[int | str, str],
) -> dict[str, int]:
    """Validate the pinned model's declared index mapping and invert by name."""

    try:
        normalized = {int(index): str(label).strip().lower() for index, label in id2label.items()}
    except (TypeError, ValueError) as exc:
        raise ModelLabelMappingError("Model id2label keys must be integer-like.") from exc
    expected = dict(enumerate(FINBERT_ORIGINAL_LABELS_BY_INDEX))
    if normalized != expected:
        raise ModelLabelMappingError(
            f"Pinned FinBERT label mapping mismatch: expected {expected}, got {normalized}."
        )
    return {label: index for index, label in expected.items()}


def map_probabilities_by_label(
    probabilities_by_index: Sequence[float],
    id2label: Mapping[int | str, str],
) -> dict[str, float]:
    """Map source-index probabilities into PAISA's stable canonical order."""

    label2index = validate_finbert_label_mapping(id2label)
    if len(probabilities_by_index) != len(FINBERT_ORIGINAL_LABELS_BY_INDEX):
        raise ValueError("FinBERT must emit exactly three class probabilities.")
    mapped = {
        label: float(probabilities_by_index[label2index[label]])
        for label in CANONICAL_SENTIMENT_LABELS
    }
    if any(not math.isfinite(value) or value < 0.0 for value in mapped.values()):
        raise ValueError("FinBERT emitted an invalid class probability.")
    if not math.isclose(
        sum(mapped.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError("FinBERT probabilities must sum to one.")
    return mapped


@runtime_checkable
class SentimentInferenceBackend(Protocol):
    model_repository: str
    model_revision: str
    tokenizer_revision: str
    device: str

    @property
    def identity(self) -> SentimentBackendIdentity: ...

    @property
    def snapshot_inventory(self) -> Sequence[Mapping[str, Any]]: ...

    @property
    def library_versions(self) -> Mapping[str, str]: ...

    def score_batch(
        self,
        texts: Sequence[str],
        *,
        maximum_sequence_length: int,
    ) -> Sequence[BackendPrediction]: ...


class TransformersFinBERTBackend:
    """Lazy, pinned FinBERT backend; construction performs no imports/downloads."""

    def __init__(
        self,
        *,
        model_repository: str = FINBERT_MODEL_REPOSITORY,
        model_revision: str = FINBERT_MODEL_REVISION,
        tokenizer_revision: str = FINBERT_TOKENIZER_REVISION,
        device: str = "cpu",
        cache_dir: str | Path | None = None,
    ) -> None:
        if model_repository != FINBERT_MODEL_REPOSITORY:
            raise ValueError(
                "Production FinBERT backend repository is fixed to the reviewed model."
            )
        if model_revision != FINBERT_MODEL_REVISION:
            raise ValueError(
                "Production FinBERT backend revision is fixed to the reviewed commit."
            )
        if tokenizer_revision != FINBERT_TOKENIZER_REVISION:
            raise ValueError(
                "Production tokenizer revision is fixed to the reviewed commit."
            )
        self.model_repository = model_repository
        self.model_revision = validate_immutable_revision(model_revision)
        self.tokenizer_revision = validate_immutable_revision(tokenizer_revision)
        normalized_device = device.strip().lower()
        if normalized_device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: cpu, cuda, mps.")
        self.device = normalized_device
        self._cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self._torch: Any = None
        self._tokenizer: Any = None
        self._model: Any = None
        self._id2label: dict[int | str, str] | None = None
        self._library_versions: dict[str, str] | None = None
        self._snapshot_inventory: tuple[dict[str, Any], ...] | None = None
        self._snapshot_paths: dict[str, Path] | None = None
        self._materialized_snapshot: tempfile.TemporaryDirectory[str] | None = None

    @property
    def identity(self) -> SentimentBackendIdentity:
        return reviewed_finbert_identity()

    @property
    def snapshot_inventory(self) -> Sequence[Mapping[str, Any]]:
        self._ensure_loaded()
        assert self._snapshot_inventory is not None
        return tuple(dict(row) for row in self._snapshot_inventory)

    @property
    def library_versions(self) -> Mapping[str, str]:
        self._ensure_loaded()
        assert self._library_versions is not None
        return dict(self._library_versions)

    @staticmethod
    def _installed_version(distribution: str) -> str:
        try:
            return metadata.version(distribution)
        except metadata.PackageNotFoundError:
            return "not-installed"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # Imports intentionally live here. Merely importing PAISA's sentiment
        # package cannot initialize torch, touch the model cache, or use network.
        try:
            import torch  # type: ignore
            from huggingface_hub import hf_hub_download  # type: ignore
            from transformers import (  # type: ignore
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise SentimentBackendError(
                "Sentiment runtime dependencies are not installed."
            ) from exc

        if self.device == "cuda" and not torch.cuda.is_available():
            raise DeviceUnavailableError("CUDA was requested but is unavailable.")
        if self.device == "mps":
            mps = getattr(getattr(torch, "backends", None), "mps", None)
            if mps is None or not mps.is_available():
                raise DeviceUnavailableError(
                    "Apple Metal (MPS) was requested but is unavailable."
                )

        materialized_snapshot: tempfile.TemporaryDirectory[str] | None = None
        try:
            snapshot_paths = {
                filename: Path(
                    hf_hub_download(
                        repo_id=self.model_repository,
                        filename=filename,
                        revision=self.model_revision,
                        cache_dir=str(self._cache_dir) if self._cache_dir else None,
                    )
                )
                for filename in self.identity.snapshot_files_sha256
            }
            snapshot_inventory = verify_reviewed_snapshot(
                snapshot_paths,
                expected_sha256=self.identity.snapshot_files_sha256,
                trusted_root=self._cache_dir,
            )
            materialized_snapshot = tempfile.TemporaryDirectory(
                prefix="paisa-finbert-reviewed-"
            )
            materialized_root = Path(materialized_snapshot.name)
            materialized_paths = _materialize_reviewed_snapshot(
                snapshot_paths,
                destination=materialized_root,
                trusted_root=self._cache_dir,
            )
            isolated_inventory = verify_reviewed_snapshot(
                materialized_paths,
                expected_sha256=self.identity.snapshot_files_sha256,
                trusted_root=materialized_root,
            )
            if isolated_inventory != snapshot_inventory:
                raise ModelSnapshotIntegrityError(
                    "Isolated snapshot identity differs from the reviewed cache bytes."
                )
            load_common = {
                "local_files_only": True,
                "trust_remote_code": False,
            }
            tokenizer = AutoTokenizer.from_pretrained(
                str(materialized_root),
                **load_common,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                str(materialized_root),
                use_safetensors=True,
                **load_common,
            )
            id2label = dict(getattr(model.config, "id2label", {}))
            validate_finbert_label_mapping(id2label)
            tokenizer.truncation_side = "right"
            if getattr(tokenizer, "truncation_side", None) != "right":
                raise SentimentBackendError(
                    "Pinned FinBERT tokenizer must use right-side truncation."
                )
            # Keep the private directory alive with the backend and prove that
            # model construction did not mutate any reviewed input byte.
            if verify_reviewed_snapshot(
                materialized_paths,
                expected_sha256=self.identity.snapshot_files_sha256,
                trusted_root=materialized_root,
            ) != isolated_inventory:
                raise ModelSnapshotIntegrityError(
                    "Reviewed isolated snapshot changed while the model was loading."
                )
            model.eval()
            model.to(self.device)
        except SentimentBackendError:
            if materialized_snapshot is not None:
                materialized_snapshot.cleanup()
            raise
        except Exception as exc:
            if materialized_snapshot is not None:
                materialized_snapshot.cleanup()
            raise SentimentBackendError(
                "Pinned FinBERT model or tokenizer could not be loaded."
            ) from exc
        assert materialized_snapshot is not None

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._id2label = id2label
        self._snapshot_paths = materialized_paths
        self._snapshot_inventory = isolated_inventory
        self._materialized_snapshot = materialized_snapshot
        self._library_versions = {
            "python": ".".join(map(str, sys.version_info[:3])),
            "torch": self._installed_version("torch"),
            "transformers": self._installed_version("transformers"),
            "tokenizers": self._installed_version("tokenizers"),
            "huggingface-hub": self._installed_version("huggingface-hub"),
            "safetensors": self._installed_version("safetensors"),
        }

    def score_batch(
        self,
        texts: Sequence[str],
        *,
        maximum_sequence_length: int,
    ) -> Sequence[BackendPrediction]:
        if not texts:
            return ()
        if not 1 <= maximum_sequence_length <= 512:
            raise ValueError("maximum_sequence_length must be between 1 and 512.")
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None
        assert self._id2label is not None

        token_counts: list[int] = []
        for text in texts:
            untruncated = self._tokenizer(
                text,
                add_special_tokens=True,
                truncation=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )
            token_counts.append(len(untruncated["input_ids"]))

        encoded = self._tokenizer(
            list(texts),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=maximum_sequence_length,
            return_tensors="pt",
        )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            raise SentimentBackendError(
                "Pinned FinBERT tokenizer did not return an attention mask."
            )
        try:
            final_token_counts = (
                attention_mask.sum(dim=1).detach().cpu().tolist()
            )
        except Exception as exc:
            raise SentimentBackendError(
                "Pinned FinBERT attention mask could not be audited."
            ) from exc
        if len(final_token_counts) != len(texts):
            raise SentimentBackendError(
                "Pinned FinBERT returned unexpected token-count evidence."
            )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        try:
            with self._torch.inference_mode():
                logits = self._model(**encoded).logits
                rows = self._torch.softmax(logits.float(), dim=-1).detach().cpu().tolist()
        except Exception as exc:
            raise SentimentBackendError("Pinned FinBERT inference failed.") from exc

        if len(rows) != len(texts):
            raise SentimentBackendError("FinBERT returned an unexpected batch size.")

        predictions: list[BackendPrediction] = []
        for row, token_count, final_token_count in zip(
            rows, token_counts, final_token_counts
        ):
            final_token_count = int(final_token_count)
            if not 0 < final_token_count <= min(
                token_count, maximum_sequence_length
            ):
                raise SentimentBackendError(
                    "Pinned FinBERT emitted invalid final token-count evidence."
                )
            mapped = map_probabilities_by_label(row, self._id2label)
            source_index = max(range(len(row)), key=lambda index: row[index])
            source_mapping = validate_finbert_label_mapping(self._id2label)
            index_to_label = {index: label for label, index in source_mapping.items()}
            predicted_label = index_to_label[source_index]
            predictions.append(
                BackendPrediction(
                    negative_probability=mapped["negative"],
                    neutral_probability=mapped["neutral"],
                    positive_probability=mapped["positive"],
                    predicted_label=predicted_label,
                    predicted_probability=mapped[predicted_label],
                    tokenizer_token_count=token_count,
                    final_token_count=final_token_count,
                    truncation_side="right",
                    token_truncation_applied=token_count > final_token_count,
                )
            )
        return tuple(predictions)
