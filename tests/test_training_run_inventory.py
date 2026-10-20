from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paisa.experiments.artifacts import (
    assert_training_run_artifacts_unchanged,
    seal_training_run_artifacts,
    validate_training_run_artifacts,
)
from paisa.experiments.trainer import run_training_stage


def _price_fixture(rows: int = 140) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100.0 + index * 0.08 + 2.0 * np.sin(index / 2.7)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=rows),
            "ticker": "OGDC",
            "open": close * (1.0 + 0.002 * np.cos(index)),
            "close": close,
            "volume": 100_000 + (index % 17) * 1_000,
            "source": "deterministic-fixture",
        }
    )


@pytest.fixture(scope="module")
def completed_training_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("training-run-inventory")
    prices_path = root / "prices.csv"
    _price_fixture().to_csv(prices_path, index=False)
    summary = run_training_stage(
        start="2025-01-01",
        prices_path=prices_path,
        output_dir=root / "artifacts",
        repo_root=Path(__file__).resolve().parents[1],
        run_id="inventory-contract",
        create_zip=False,
        model_ids=["M-001"],
    )
    return Path(summary["artifact_directory"])


def _manifest(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))


def _inventory_paths(manifest: dict) -> list[str]:
    return [entry["path"] for entry in manifest["artifact_inventory"]["files"]]


def test_training_run_has_complete_deterministic_inventory_and_source_provenance(
    completed_training_run: Path,
) -> None:
    evidence = validate_training_run_artifacts(completed_training_run)
    manifest = _manifest(completed_training_run)
    paths = _inventory_paths(manifest)
    actual = sorted(
        path.relative_to(completed_training_run).as_posix()
        for path in completed_training_run.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    )

    assert manifest["artifact_inventory"]["algorithm"] == "sha256"
    assert manifest["artifact_inventory"]["root"] == "."
    assert manifest["artifact_inventory"]["excludes"] == ["run_manifest.json"]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))
    assert paths == actual
    assert {
        "data_manifest.json",
        "split_manifest.json",
        "frozen_input/OGDC_prices.csv",
        "frozen_input/OGDC_model_rows.csv",
        "reports/environment.json",
        "reports/final_summary.json",
        "reports/validation_leaderboard.csv",
        "reports/test_leaderboard.csv",
    }.issubset(paths)

    model_record = manifest["models"][0]
    model_root = f"models/{model_record['model_id']}"
    assert {
        f"{model_root}/manifest.json",
        f"{model_root}/model.joblib",
        f"{model_root}/config.json",
        f"{model_root}/feature_columns.json",
        f"{model_root}/metrics.json",
        f"{model_root}/confusion_matrix.csv",
        f"{model_root}/feature_importance.csv",
        f"{model_root}/predictions.csv",
        f"{model_root}/predictions_all_splits.csv",
    }.issubset(paths)

    model_manifest = json.loads(
        (completed_training_run / model_record["manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    for field in ("code_version", "dirty_tree", "source_fingerprint"):
        assert manifest[field] == model_manifest[field]
        assert getattr(evidence, field) == manifest[field]
    assert evidence.artifact_sha256 == tuple(
        (entry["path"], entry["sha256"])
        for entry in manifest["artifact_inventory"]["files"]
    )
    assert dict(evidence.model_ids) == {
        model_record["variant_id"]: model_record["model_id"]
    }
    assert dict(evidence.prediction_paths) == manifest["prediction_paths"]
    assert_training_run_artifacts_unchanged(evidence)


@pytest.mark.parametrize(
    "relative_path",
    [
        "MODEL_MANIFEST",
        "MODEL_CONFIG",
        "MODEL_METRICS",
        "MODEL_PREDICTIONS",
        "split_manifest.json",
    ],
)
def test_training_run_validator_rejects_any_inventoried_artifact_change(
    completed_training_run: Path,
    tmp_path: Path,
    relative_path: str,
) -> None:
    copied = tmp_path / "copied-run"
    shutil.copytree(completed_training_run, copied)
    model_id = _manifest(copied)["models"][0]["model_id"]
    resolved = {
        "MODEL_MANIFEST": f"models/{model_id}/manifest.json",
        "MODEL_CONFIG": f"models/{model_id}/config.json",
        "MODEL_METRICS": f"models/{model_id}/metrics.json",
        "MODEL_PREDICTIONS": f"models/{model_id}/predictions.csv",
    }.get(relative_path, relative_path)
    target = copied / resolved
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_training_run_artifacts(copied)


def test_training_run_validator_rejects_uninventoried_extra_file(
    completed_training_run: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied-run"
    shutil.copytree(completed_training_run, copied)
    (copied / "untracked-result.txt").write_text("not inventoried", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected artifact|exactly match"):
        validate_training_run_artifacts(copied)


def test_captured_training_evidence_rejects_coordinated_reseal(
    completed_training_run: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied-run"
    shutil.copytree(completed_training_run, copied)
    evidence = validate_training_run_artifacts(copied)
    model_id = _manifest(copied)["models"][0]["model_id"]
    config_path = copied / "models" / model_id / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["review_tamper"] = True
    config_path.write_text(json.dumps(config), encoding="utf-8")

    # A current-state validator can accept an explicitly resealed owned run,
    # but evidence captured before an untrusted callback must remain immutable.
    seal_training_run_artifacts(copied)
    validate_training_run_artifacts(copied)
    with pytest.raises(RuntimeError, match="run manifest changed"):
        assert_training_run_artifacts_unchanged(evidence)


def test_training_run_validator_cross_checks_model_source_provenance(
    completed_training_run: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "copied-run"
    shutil.copytree(completed_training_run, copied)
    manifest = _manifest(copied)
    manifest["source_fingerprint"] = "0" * 64
    (copied / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="source_fingerprint"):
        validate_training_run_artifacts(copied)
