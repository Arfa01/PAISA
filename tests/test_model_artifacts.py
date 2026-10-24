from __future__ import annotations

import json

import pandas as pd
import pytest

from paisa.experiments.artifacts import (
    detect_runtime_environment,
    load_model_artifact,
    make_zip_from_directory,
    sha256_file,
    source_fingerprint,
    validate_model_manifest,
    write_model_artifact,
)
from paisa.experiments.registry import ModelArtifactRegistry


def _complete_manifest(model_id: str, *, algorithm: str = "random_forest") -> dict:
    split_metrics = {
        "accuracy": 1.0,
        "balanced_accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "matthews_corrcoef": 1.0,
        "roc_auc": 1.0,
        "brier_score": 0.0,
        "confusion_matrix": [[1, 0], [0, 1]],
        "class_distribution": {"0": 1, "1": 1},
        "observed_classes": [0, 1],
        "complete_class_coverage": True,
        "selection_eligible": True,
        "metric_warnings": [],
        "support": 2,
        "positive_label": 1,
        "average": "binary",
        "zero_division": 0,
    }
    return {
        "schema_version": "1.0",
        "model_id": model_id,
        "algorithm": algorithm,
        "model_configuration": {"random_state": 42},
        "feature_set": {
            "feature_set_id": "technical-daily-v1",
            "columns": ["close", "daily_return"],
            "feature_code_version": "a" * 40,
            "transforms": [
                {
                    "name": "median_imputer",
                    "parameters": {"strategy": "median"},
                    "fit_scope": "train_only",
                }
            ],
        },
        "training_dates": {
            "data_feature_start": "2025-01-01",
            "data_feature_end": "2025-12-31",
            "train_target_start": "2025-01-02",
            "train_target_end": "2025-08-01",
            "validation_target_start": "2025-08-04",
            "validation_target_end": "2025-10-01",
            "test_target_start": "2025-10-02",
            "test_target_end": "2025-12-31",
        },
        "target_definition": {
            "name": "target_next_day_up",
            "positive_class_condition": "close[target_date] > close[feature_date]",
            "class_labels": {"0": "down_or_flat", "1": "up"},
        },
        "prediction_horizon": {"value": 1, "unit": "trading_session"},
        "random_seed": 42,
        "metrics": {
            "train": dict(split_metrics),
            "validation": dict(split_metrics),
            "test": dict(split_metrics),
        },
        "created_at": "2026-10-19T12:00:00Z",
        "symbols": ["OGDC"],
        "data_provenance": {
            "provider": "fixture",
            "requested_symbols": ["OGDC"],
            "included_symbols": ["OGDC"],
            "requested_start": "2025-01-01",
            "requested_end": "2025-12-31",
            "actual_feature_start": "2025-01-01",
            "actual_feature_end": "2025-12-31",
            "dataset_sha256": "b" * 64,
            "input_schema_version": "1.0",
            "price_adjustment": "unknown",
            "dividends_included": False,
            "corporate_actions_policy": "not_modeled",
        },
        "class_mapping": {
            "estimator_classes": [0, 1],
            "probability_fields_by_class": {
                "0": "probability_down",
                "1": "probability_up",
            },
        },
        "code_version": "a" * 40,
        "dirty_tree": False,
        "source_fingerprint": "a" * 40,
        "library_versions": {"scikit-learn": "1.9.1"},
    }


def test_model_artifact_round_trip_and_collision_are_immutable(tmp_path):
    root = tmp_path / "models"
    predictions = pd.DataFrame({"probability_up": [0.25, 0.75]})
    model_dir = write_model_artifact(
        root,
        "M-001-fixture",
        {"estimator": "fixture", "classes": [0, 1]},
        _complete_manifest("M-001-fixture"),
        extra_files={"predictions.csv": predictions, "metrics.json": {"accuracy": 1.0}},
    )

    loaded, manifest = load_model_artifact(model_dir)
    assert loaded == {"estimator": "fixture", "classes": [0, 1]}
    assert manifest["artifact_path"] == "M-001-fixture/model.joblib"
    assert manifest["artifact_sha256"] == sha256_file(model_dir / "model.joblib")
    assert pd.read_csv(model_dir / "predictions.csv").equals(predictions)

    original_manifest = (model_dir / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_model_artifact(
            root,
            "M-001-fixture",
            {"estimator": "replacement"},
            _complete_manifest("M-001-fixture"),
        )
    assert (model_dir / "manifest.json").read_bytes() == original_manifest
    assert not list(root.glob(".M-001-fixture.*.tmp"))


def test_model_loader_accepts_matching_manifest_and_artifact_trust_anchors(tmp_path):
    model_dir = write_model_artifact(
        tmp_path / "models",
        "M-001-anchored",
        {"estimator": "fixture", "classes": [0, 1]},
        _complete_manifest("M-001-anchored"),
    )
    manifest_sha256 = sha256_file(model_dir / "manifest.json")
    artifact_sha256 = sha256_file(model_dir / "model.joblib")

    loaded, manifest = load_model_artifact(
        model_dir,
        expected_manifest_sha256=manifest_sha256,
        expected_artifact_sha256=artifact_sha256,
    )

    assert loaded == {"estimator": "fixture", "classes": [0, 1]}
    assert manifest["artifact_sha256"] == artifact_sha256


def test_model_loader_rejects_wrong_manifest_anchor_before_deserialization(
    tmp_path, monkeypatch
):
    model_dir = write_model_artifact(
        tmp_path / "models",
        "M-002-manifest-anchor",
        {"estimator": "fixture"},
        _complete_manifest("M-002-manifest-anchor"),
    )
    monkeypatch.setattr(
        "paisa.experiments.artifacts.joblib.load",
        lambda handle: pytest.fail("untrusted model bytes must not be deserialized"),
    )

    with pytest.raises(ValueError, match="manifest checksum.*trust anchor"):
        load_model_artifact(
            model_dir,
            expected_manifest_sha256="0" * 64,
            expected_artifact_sha256=sha256_file(model_dir / "model.joblib"),
        )


def test_model_loader_rejects_wrong_artifact_trust_anchor(tmp_path, monkeypatch):
    model_dir = write_model_artifact(
        tmp_path / "models",
        "M-003-artifact-anchor",
        {"estimator": "fixture"},
        _complete_manifest("M-003-artifact-anchor"),
    )
    monkeypatch.setattr(
        "paisa.experiments.artifacts.joblib.load",
        lambda handle: pytest.fail("untrusted model bytes must not be deserialized"),
    )

    with pytest.raises(ValueError, match="artifact checksum.*trust anchor"):
        load_model_artifact(
            model_dir,
            expected_manifest_sha256=sha256_file(model_dir / "manifest.json"),
            expected_artifact_sha256="0" * 64,
        )


def test_model_loader_rejects_checksum_and_unsafe_artifact_path(tmp_path):
    root = tmp_path / "models"
    model_dir = write_model_artifact(
        root,
        "M-002-fixture",
        {"estimator": "fixture"},
        _complete_manifest("M-002-fixture"),
    )
    (model_dir / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_model_artifact(model_dir)

    second = write_model_artifact(
        root,
        "M-003-fixture",
        {"estimator": "fixture"},
        _complete_manifest("M-003-fixture"),
    )
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_path"] = "../M-002-fixture/model.joblib"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="safe relative"):
        load_model_artifact(second)


def test_model_registry_and_zip_refuse_reuse(tmp_path):
    registry = ModelArtifactRegistry(tmp_path / "models")
    registry.register(
        "M-004-fixture",
        {"algorithm": "xgboost"},
        _complete_manifest("M-004-fixture", algorithm="xgboost"),
    )
    assert registry.contains("M-004-fixture")
    assert registry.list_model_ids() == ("M-004-fixture",)
    loaded, manifest = registry.load("M-004-fixture")
    assert loaded["algorithm"] == "xgboost"
    assert manifest["model_id"] == "M-004-fixture"

    source = tmp_path / "source"
    source.mkdir()
    (source / "result.txt").write_text("immutable", encoding="utf-8")
    archive = make_zip_from_directory(source, tmp_path / "result.zip")
    assert archive.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        make_zip_from_directory(source, archive)


def test_source_fingerprint_is_content_and_path_sensitive(tmp_path):
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    initial = source_fingerprint([second, first], root=tmp_path)
    assert initial == source_fingerprint([first, second], root=tmp_path)
    first.write_text("value = 3\n", encoding="utf-8")
    assert source_fingerprint([first, second], root=tmp_path) != initial


def test_manifest_validator_rejects_incomplete_contract():
    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_manifest({"schema_version": "1.0", "model_id": "M-001"})


def test_manifest_validator_requires_complete_metric_contract():
    manifest = _complete_manifest("M-001-metrics")
    del manifest["metrics"]["validation"]["brier_score"]
    with pytest.raises(ValueError, match="missing required fields"):
        validate_model_manifest({
            **manifest,
            "artifact_path": "M-001-metrics/model.joblib",
            "artifact_sha256": "c" * 64,
        })


def test_runtime_environment_reports_the_actual_local_platform(monkeypatch):
    monkeypatch.setattr("paisa.experiments.artifacts.platform.system", lambda: "Darwin")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    monkeypatch.delenv("KAGGLE_URL_BASE", raising=False)
    assert detect_runtime_environment()["runtime_environment"] == "darwin"


def test_manifest_accepts_explicitly_undefined_single_class_metrics():
    manifest = _complete_manifest("M-001-single-class")
    metrics = manifest["metrics"]["test"]
    metrics.update(
        {
            "accuracy": 1.0,
            "balanced_accuracy": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matthews_corrcoef": None,
            "roc_auc": None,
            "brier_score": 0.025,
            "confusion_matrix": [[2, 0], [0, 0]],
            "class_distribution": {"0": 2, "1": 0},
            "observed_classes": [0],
            "complete_class_coverage": False,
            "selection_eligible": False,
            "metric_warnings": ["single_class_actuals"],
        }
    )
    validated = validate_model_manifest(
        {
            **manifest,
            "artifact_path": "M-001-single-class/model.joblib",
            "artifact_sha256": "d" * 64,
        }
    )
    assert validated["metrics"]["test"]["balanced_accuracy"] is None
