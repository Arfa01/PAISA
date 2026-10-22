"""Inference backend contract and lazy Hugging Face FinBERT implementation."""

from __future__ import annotations

from importlib import metadata
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from paisa.sentiment.config import (
    CANONICAL_SENTIMENT_LABELS,
    FINBERT_MODEL_REPOSITORY,
    FINBERT_MODEL_REVISION,
    FINBERT_ORIGINAL_LABELS_BY_INDEX,
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


def validate_immutable_revision(revision: str) -> str:
    if not _COMMIT_REVISION.fullmatch(revision):
        raise ValueError("Model revisions must be full 40-character commit hashes.")
    return revision


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

        common = {
            "cache_dir": str(self._cache_dir) if self._cache_dir else None,
            "trust_remote_code": False,
        }
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_repository,
                revision=self.tokenizer_revision,
                **common,
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_repository,
                revision=self.model_revision,
                use_safetensors=True,
                **common,
            )
        except Exception as exc:
            raise SentimentBackendError(
                "Pinned FinBERT model or tokenizer could not be loaded."
            ) from exc

        id2label = dict(getattr(model.config, "id2label", {}))
        validate_finbert_label_mapping(id2label)
        model.eval()
        model.to(self.device)

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._id2label = id2label
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
        for row, token_count in zip(rows, token_counts):
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
                    token_truncation_applied=token_count > maximum_sequence_length,
                )
            )
        return tuple(predictions)
