from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest

from paisa.experiments import artifacts as artifact_module
from paisa.experiments.artifacts import validate_model_manifest
from paisa.experiments.experiment_config import TARGET_NAME, get_variant
from paisa.experiments.registry import ModelArtifactRegistry
from paisa.experiments.trainer import fit_variant


def _complete_manifest(model_id: str) -> dict:
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
        "algorithm": "random_forest",
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
        "artifact_path": f"{model_id}/model.joblib",
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
            "dataset_sha256": "c" * 64,
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


def _adversarial_training_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [10.0, 11.0, 10.5, 12.0, 11.5, 13.0],
            "next_close": [11.0, 10.5, 12.0, 11.5, 13.0, 12.5],
            TARGET_NAME: [1, 0, 1, 0, 1, 0],
            "actual_class": [1, 0, 1, 0, 1, 0],
            "ticker": ["OGDC"] * 6,
            "feature_date": pd.bdate_range("2025-01-01", periods=6),
        }
    )


@pytest.mark.parametrize(
    "feature_columns",
    [
        ["next_close"],
        [TARGET_NAME],
        ["actual_class"],
        ["ticker"],
        ["feature_date"],
        ["source"],
        ["date"],
        ["close", "close"],
    ],
    ids=[
        "next-prefix",
        "target-prefix",
        "actual-class",
        "ticker",
        "feature-date",
        "source",
        "date",
        "duplicate",
    ],
)
def test_public_fit_boundary_rejects_leaky_metadata_and_duplicate_features(
    feature_columns,
):
    with pytest.raises(ValueError, match="target, future, or metadata|duplicates"):
        fit_variant(
            get_variant("M-001"),
            _adversarial_training_rows(),
            feature_columns=feature_columns,
        )


@pytest.mark.parametrize(
    "feature_columns",
    [
        ["next_close"],
        [TARGET_NAME],
        ["actual_class"],
        ["ticker"],
        ["feature_date"],
        ["source"],
        ["date"],
        ["close", "close"],
    ],
    ids=[
        "next-prefix",
        "target-prefix",
        "actual-class",
        "ticker",
        "feature-date",
        "source",
        "date",
        "duplicate",
    ],
)
def test_manifest_boundary_rejects_leaky_metadata_and_duplicate_features(
    feature_columns,
):
    manifest = _complete_manifest("M-001-adversarial")
    manifest["artifact_sha256"] = "b" * 64
    manifest["feature_set"]["columns"] = feature_columns
    with pytest.raises(ValueError, match="feature_set.columns are invalid"):
        validate_model_manifest(manifest)


def _registry_with_model(tmp_path: Path, model_id: str) -> tuple[ModelArtifactRegistry, Path]:
    registry = ModelArtifactRegistry(tmp_path / "registry")
    model_dir = registry.register(
        model_id,
        {"estimator": "fixture", "classes": [0, 1]},
        _complete_manifest(model_id),
    )
    return registry, model_dir


def test_registry_rejects_model_directory_symlink_outside_root(tmp_path):
    external_registry = ModelArtifactRegistry(tmp_path / "external")
    external_dir = external_registry.register(
        "M-001-external",
        {"estimator": "external"},
        _complete_manifest("M-001-external"),
    )
    registry_root = tmp_path / "registry"
    registry_root.mkdir()
    link = registry_root / "M-001-external"
    try:
        os.symlink(external_dir, link, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - unsupported Windows setup
        pytest.skip(f"Symlinks are unavailable: {exc}")

    registry = ModelArtifactRegistry(registry_root)
    with pytest.raises(ValueError, match="symlink|outside"):
        registry.load("M-001-external")


def test_registry_rejects_external_manifest_symlink(tmp_path):
    registry, model_dir = _registry_with_model(tmp_path, "M-002-manifest-link")
    manifest_path = model_dir / "manifest.json"
    external_manifest = tmp_path / "external-manifest.json"
    shutil.copy2(manifest_path, external_manifest)
    manifest_path.unlink()
    try:
        os.symlink(external_manifest, manifest_path)
    except OSError as exc:  # pragma: no cover - unsupported Windows setup
        pytest.skip(f"Symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="manifest.*symlink"):
        registry.load("M-002-manifest-link")


def test_registry_rejects_external_model_binary_symlink(tmp_path):
    registry, model_dir = _registry_with_model(tmp_path, "M-003-binary-link")
    binary_path = model_dir / "model.joblib"
    external_binary = tmp_path / "external-model.joblib"
    shutil.copy2(binary_path, external_binary)
    binary_path.unlink()
    try:
        os.symlink(external_binary, binary_path)
    except OSError as exc:  # pragma: no cover - unsupported Windows setup
        pytest.skip(f"Symlinks are unavailable: {exc}")

    with pytest.raises(ValueError, match="artifact.*symlink"):
        registry.load("M-003-binary-link")


def test_registry_rejects_nested_artifact_directory_symlink(tmp_path):
    registry, model_dir = _registry_with_model(tmp_path, "M-003-nested-link")
    binary_path = model_dir / "model.joblib"
    external_directory = tmp_path / "external-directory"
    external_directory.mkdir()
    shutil.copy2(binary_path, external_directory / "model.joblib")
    binary_path.unlink()
    nested = model_dir / "nested"
    try:
        os.symlink(external_directory, nested, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - unsupported Windows setup
        pytest.skip(f"Symlinks are unavailable: {exc}")

    manifest_path = model_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_path"] = "M-003-nested-link/nested/model.joblib"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact.*symlink"):
        registry.load("M-003-nested-link")


def test_registry_checksums_and_deserializes_one_open_file(tmp_path, monkeypatch):
    registry, _ = _registry_with_model(tmp_path, "M-004-open-handle")
    real_load = artifact_module.joblib.load
    received_open_file = False

    def load_from_open_file(source):
        nonlocal received_open_file
        received_open_file = hasattr(source, "read") and hasattr(source, "seek")
        return real_load(source)

    monkeypatch.setattr(artifact_module.joblib, "load", load_from_open_file)
    loaded, _ = registry.load("M-004-open-handle")

    assert received_open_file
    assert loaded == {"estimator": "fixture", "classes": [0, 1]}
