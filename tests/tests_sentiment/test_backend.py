from __future__ import annotations

import builtins
from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

import paisa.sentiment.backend as backend_module
from paisa.sentiment.backend import (
    ModelSnapshotIntegrityError,
    SentimentBackendIdentity,
    ModelLabelMappingError,
    TransformersFinBERTBackend,
    map_probabilities_by_label,
    validate_finbert_label_mapping,
    validate_immutable_revision,
    verify_reviewed_snapshot,
)
from paisa.sentiment.config import (
    FINBERT_REVIEWED_SNAPSHOT_SHA256,
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
            final_token_count=4,
            truncation_side="right",
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
            final_token_count=4,
            truncation_side="right",
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


def test_production_backend_rejects_unreviewed_repository_or_revision():
    with pytest.raises(ValueError, match="repository is fixed"):
        TransformersFinBERTBackend(model_repository="example/arbitrary-model")
    with pytest.raises(ValueError, match="revision is fixed"):
        TransformersFinBERTBackend(model_revision="1" * 40)
    with pytest.raises(ValueError, match="tokenizer revision is fixed"):
        TransformersFinBERTBackend(tokenizer_revision="2" * 40)


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

        def to(self, _device):
            return self

        def sum(self, *, dim):
            assert dim == 1
            return FakeTensor([sum(row) for row in self.rows])

    class FakeTokenizer:
        def __init__(self):
            self.batch_tensor = FakeDeviceTensor()
            self.truncation_side = "left"

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
            return {
                "input_ids": self.batch_tensor,
                "attention_mask": FakeTensor([[1, 1, 1, 1]]),
            }

    tokenizer = FakeTokenizer()

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, repository, **kwargs):
            calls["tokenizer_load"] = (repository, kwargs)
            calls["tokenizer_load_files"] = {
                path.name for path in Path(repository).iterdir()
            }
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
            calls["model_load_files"] = {
                path.name for path in Path(repository).iterdir()
            }
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
    fake_huggingface_hub = ModuleType("huggingface_hub")

    def hf_hub_download(*, repo_id, filename, revision, cache_dir):
        calls.setdefault("snapshot_downloads", []).append(
            (repo_id, filename, revision, cache_dir)
        )
        path = cache / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(filename.encode("utf-8"))
        return str(path)

    fake_huggingface_hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_huggingface_hub)

    def verified_inventory(files, *, expected_sha256, trusted_root):
        assert set(files) == set(expected_sha256)
        assert trusted_root is not None
        return tuple(
            {"path": name, "sha256": digest, "size_bytes": 1}
            for name, digest in sorted(expected_sha256.items())
        )

    monkeypatch.setattr(
        backend_module, "verify_reviewed_snapshot", verified_inventory
    )

    cache = tmp_path / "model-cache"
    cache.mkdir()
    (cache / "added_tokens.json").write_text(
        '{"[UNREVIEWED]": 30522}\n', encoding="utf-8"
    )
    backend = TransformersFinBERTBackend(cache_dir=cache)
    predictions = backend.score_batch(
        ["OGDC reports higher production"], maximum_sequence_length=8
    )

    isolated_root = Path(calls["tokenizer_load"][0])
    assert isolated_root == Path(calls["model_load"][0])
    assert isolated_root != cache
    assert isolated_root.is_dir()
    common = {"trust_remote_code": False, "local_files_only": True}
    assert calls["tokenizer_load"][1] == common
    assert calls["model_load"][1] == {"use_safetensors": True, **common}
    assert calls["tokenizer_load_files"] == set(FINBERT_REVIEWED_SNAPSHOT_SHA256)
    assert calls["model_load_files"] == set(FINBERT_REVIEWED_SNAPSHOT_SHA256)
    assert "added_tokens.json" not in calls["tokenizer_load_files"]
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
    assert predictions[0].final_token_count == 4
    assert predictions[0].truncation_side == "right"
    assert predictions[0].token_truncation_applied is False
    assert tokenizer.truncation_side == "right"
    assert {item[1] for item in calls["snapshot_downloads"]} == set(
        FINBERT_REVIEWED_SNAPSHOT_SHA256
    )
    assert backend.snapshot_inventory


def test_reviewed_snapshot_inventory_rejects_changed_or_external_bytes(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    expected = {"config.json": "a" * 64}
    config = cache / "config.json"
    config.write_text("changed", encoding="utf-8")

    with pytest.raises(ModelSnapshotIntegrityError, match="hash mismatch"):
        verify_reviewed_snapshot(
            {"config.json": config},
            expected_sha256=expected,
            trusted_root=cache,
        )

    external = tmp_path / "external.json"
    external.write_text("outside", encoding="utf-8")
    config.unlink()
    config.symlink_to(external)
    observed = __import__("hashlib").sha256(external.read_bytes()).hexdigest()
    with pytest.raises(ModelSnapshotIntegrityError, match="escapes"):
        verify_reviewed_snapshot(
            {"config.json": config},
            expected_sha256={"config.json": observed},
            trusted_root=cache,
        )


def test_backend_identity_hash_binds_every_snapshot_file():
    identity = TransformersFinBERTBackend().identity
    changed = dict(identity.snapshot_files_sha256)
    changed["vocab.txt"] = "0" * 64
    altered = SentimentBackendIdentity(
        model_repository=identity.model_repository,
        model_revision=identity.model_revision,
        tokenizer_repository=identity.tokenizer_repository,
        tokenizer_revision=identity.tokenizer_revision,
        original_labels_by_index=identity.original_labels_by_index,
        canonical_probability_index_mapping=identity.canonical_probability_index_mapping,
        license_identifier=identity.license_identifier,
        license_url=identity.license_url,
        snapshot_files_sha256=changed,
    )

    assert identity.identity_sha256 != altered.identity_sha256
