from __future__ import annotations

import builtins
from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest

from paisa.sentiment.backend import (
    ModelLabelMappingError,
    TransformersFinBERTBackend,
    map_probabilities_by_label,
    validate_finbert_label_mapping,
    validate_immutable_revision,
)
from paisa.sentiment.config import (
    FINBERT_MODEL_REPOSITORY,
    FINBERT_MODEL_REVISION,
    FINBERT_TOKENIZER_REVISION,
)
from paisa.sentiment.models import BackendPrediction


def test_pinned_model_and_tokenizer_use_same_full_immutable_revision():
    assert FINBERT_MODEL_REPOSITORY == "ProsusAI/finbert"
    assert FINBERT_MODEL_REVISION == (
        "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"
    )
    assert FINBERT_TOKENIZER_REVISION == FINBERT_MODEL_REVISION
    assert validate_immutable_revision(FINBERT_MODEL_REVISION) == FINBERT_MODEL_REVISION
    with pytest.raises(ValueError):
        validate_immutable_revision("main")


def test_original_class_index_mapping_is_explicit_and_reordered_by_name():
    source = {0: "positive", 1: "negative", 2: "neutral"}

    assert validate_finbert_label_mapping(source) == {
        "positive": 0,
        "negative": 1,
        "neutral": 2,
    }
    assert map_probabilities_by_label((0.7, 0.1, 0.2), source) == {
        "negative": 0.1,
        "neutral": 0.2,
        "positive": 0.7,
    }


@pytest.mark.parametrize(
    "mapping",
    [
        {0: "negative", 1: "neutral", 2: "positive"},
        {0: "positive", 1: "negative"},
        {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"},
    ],
)
def test_runtime_mapping_mismatch_fails_closed(mapping):
    with pytest.raises(ModelLabelMappingError):
        validate_finbert_label_mapping(mapping)


def test_prediction_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        BackendPrediction(
            negative_probability=0.1,
            neutral_probability=0.2,
            positive_probability=0.6,
            predicted_label="positive",
            predicted_probability=0.6,
            tokenizer_token_count=4,
            token_truncation_applied=False,
        )


def test_probability_tolerance_matches_sqlite_contract():
    with pytest.raises(ValueError, match="sum to one"):
        BackendPrediction(
            negative_probability=0.1,
            neutral_probability=0.2,
            positive_probability=0.700005,
            predicted_label="positive",
            predicted_probability=0.700005,
            tokenizer_token_count=4,
            token_truncation_applied=False,
        )


def test_production_backend_is_lazy_and_does_not_import_or_download_on_creation(
    monkeypatch,
):
    imported: list[str] = []
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("transformers"):
            imported.append(name)
            raise AssertionError("runtime import would permit a model download")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    backend = TransformersFinBERTBackend()

    assert backend.device == "cpu"
    assert backend.model_revision == FINBERT_MODEL_REVISION
    assert backend.tokenizer_revision == FINBERT_TOKENIZER_REVISION
    assert imported == []


def test_production_backend_loads_pinned_safe_revision_and_scores_offline(
    tmp_path, monkeypatch
):
    calls = {}

    class FakeDeviceTensor:
        def __init__(self):
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    class FakeTensor:
        def __init__(self, rows):
            self.rows = rows

        def float(self):
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

    class FakeTokenizer:
        def __init__(self):
            self.batch_tensor = FakeDeviceTensor()

        def __call__(self, value, **kwargs):
            if isinstance(value, str):
                assert kwargs == {
                    "add_special_tokens": True,
                    "truncation": False,
                    "return_attention_mask": False,
                    "return_token_type_ids": False,
                    "verbose": False,
                }
                return {"input_ids": [101, 1001, 1002, 102]}
            calls["tokenizer_batch"] = (value, kwargs)
            return {"input_ids": self.batch_tensor}

    tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, repository, **kwargs):
            calls["tokenizer_load"] = (repository, kwargs)
            return tokenizer

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(
                id2label={0: "positive", 1: "negative", 2: "neutral"}
            )
            self.eval_called = False
            self.moved_to = None

        def eval(self):
            self.eval_called = True
            return self

        def to(self, device):
            self.moved_to = device
            return self

        def __call__(self, **encoded):
            calls["model_inputs"] = encoded
            return SimpleNamespace(logits=FakeTensor([[3.0, 1.0, 2.0]]))

    model = FakeModel()

    class FakeAutoModel:
        @classmethod
        def from_pretrained(cls, repository, **kwargs):
            calls["model_load"] = (repository, kwargs)
            return model

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False)
    )

    @contextmanager
    def inference_mode():
        calls["inference_mode"] = True
        yield

    def softmax(_logits, *, dim):
        calls["softmax_dim"] = dim
        return FakeTensor([[0.7, 0.1, 0.2]])

    fake_torch.inference_mode = inference_mode
    fake_torch.softmax = softmax
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    fake_transformers.AutoModelForSequenceClassification = FakeAutoModel
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    cache = tmp_path / "model-cache"
    backend = TransformersFinBERTBackend(cache_dir=cache)
    predictions = backend.score_batch(
        ["OGDC reports higher production"], maximum_sequence_length=8
    )

    common = {
        "cache_dir": str(cache),
        "trust_remote_code": False,
    }
    assert calls["tokenizer_load"] == (
        FINBERT_MODEL_REPOSITORY,
        {"revision": FINBERT_TOKENIZER_REVISION, **common},
    )
    assert calls["model_load"] == (
        FINBERT_MODEL_REPOSITORY,
        {
            "revision": FINBERT_MODEL_REVISION,
            "use_safetensors": True,
            **common,
        },
    )
    assert model.eval_called is True
    assert model.moved_to == "cpu"
    assert tokenizer.batch_tensor.moved_to == "cpu"
    assert calls["inference_mode"] is True
    assert calls["softmax_dim"] == -1
    assert calls["tokenizer_batch"] == (
        ["OGDC reports higher production"],
        {
            "add_special_tokens": True,
            "padding": True,
            "truncation": True,
            "max_length": 8,
            "return_tensors": "pt",
        },
    )
    assert len(predictions) == 1
    assert predictions[0].probabilities == {
        "negative": 0.1,
        "neutral": 0.2,
        "positive": 0.7,
    }
    assert predictions[0].predicted_label == "positive"
    assert predictions[0].tokenizer_token_count == 4
    assert predictions[0].token_truncation_applied is False
