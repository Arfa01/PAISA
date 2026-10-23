from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from paisa.experiments.artifacts import (
    seal_training_run_artifacts,
    validate_training_run_artifacts,
    write_json,
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
            "source": "nested-checksum-fixture",
        }
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def training_run_with_nested_checksums(
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    root = tmp_path_factory.mktemp("training-nested-checksums")
    prices_path = root / "prices.csv"
    _price_fixture().to_csv(prices_path, index=False)
    summary = run_training_stage(
        start="2025-01-01",
        prices_path=prices_path,
        output_dir=root / "artifacts",
        repo_root=Path(__file__).resolve().parents[1],
        run_id="nested-checksum-contract",
        create_zip=False,
        model_ids=["M-001"],
    )
    run_dir = Path(summary["artifact_directory"])
    run_manifest = _read_json(run_dir / "run_manifest.json")
    model_record = run_manifest["models"][0]
    model_manifest_path = run_dir / model_record["manifest_path"]
    model_dir = model_manifest_path.parent
    model_manifest = _read_json(model_manifest_path)

    preprocessing_payload = b"fixture train-only preprocessor\n"
    preprocessing_path = model_dir / "preprocessing" / "preprocessor.joblib"
    preprocessing_path.parent.mkdir(parents=True)
    preprocessing_path.write_bytes(preprocessing_payload)
    scaler_payload = b"fixture train-only scaler\n"
    scaler_path = model_dir / "preprocessing" / "scaler.joblib"
    scaler_path.write_bytes(scaler_payload)
    preprocessing_inventory = {
        "algorithm": "sha256",
        "files": [
            {
                "path": "preprocessing/preprocessor.joblib",
                "sha256": hashlib.sha256(preprocessing_payload).hexdigest(),
            },
            {
                "path": "preprocessing/scaler.joblib",
                "sha256": hashlib.sha256(scaler_payload).hexdigest(),
            },
        ],
    }
    write_json(
        preprocessing_inventory,
        model_dir / "preprocessing" / "checksums.json",
    )

    checkpoint_payload = b"fixture validation checkpoint\n"
    checkpoint_relative = "checkpoints/M-001.weights.bin"
    checkpoint_path = run_dir / checkpoint_relative
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(checkpoint_payload)
    training_metadata = {
        "model_checkpoint": {
            "path": checkpoint_relative,
            "sha256": hashlib.sha256(checkpoint_payload).hexdigest(),
            "loaded_after_fit": True,
        }
    }
    write_json(training_metadata, model_dir / "training_metadata.json")

    model_manifest["preprocessing_artifacts"] = preprocessing_inventory
    model_manifest["training_metadata"] = training_metadata
    write_json(model_manifest, model_manifest_path)
    seal_training_run_artifacts(run_dir)
    return run_dir


def test_training_run_reconciles_valid_nested_checksum_claims(
    training_run_with_nested_checksums: Path,
) -> None:
    evidence = validate_training_run_artifacts(training_run_with_nested_checksums)

    assert dict(evidence.model_ids).keys() == {"M-001"}


def test_training_run_rejects_resealed_false_preprocessing_checksum(
    training_run_with_nested_checksums: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "false-preprocessing-checksum"
    shutil.copytree(training_run_with_nested_checksums, copied)
    run_manifest = _read_json(copied / "run_manifest.json")
    model_manifest_path = copied / run_manifest["models"][0]["manifest_path"]
    model_manifest = _read_json(model_manifest_path)
    model_manifest["preprocessing_artifacts"]["files"][0]["sha256"] = "0" * 64
    write_json(model_manifest, model_manifest_path)
    write_json(
        model_manifest["preprocessing_artifacts"],
        model_manifest_path.parent / "preprocessing" / "checksums.json",
    )

    with pytest.raises(ValueError, match="preprocessing_artifacts checksum disagrees"):
        seal_training_run_artifacts(copied)


def test_training_run_rejects_omitted_preprocessing_checksum_claim(
    training_run_with_nested_checksums: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "omitted-preprocessing-checksum"
    shutil.copytree(training_run_with_nested_checksums, copied)
    run_manifest = _read_json(copied / "run_manifest.json")
    model_manifest_path = copied / run_manifest["models"][0]["manifest_path"]
    model_manifest = _read_json(model_manifest_path)
    omitted = model_manifest["preprocessing_artifacts"]["files"].pop()
    write_json(model_manifest, model_manifest_path)
    write_json(
        model_manifest["preprocessing_artifacts"],
        model_manifest_path.parent / "preprocessing" / "checksums.json",
    )

    assert (model_manifest_path.parent / omitted["path"]).is_file()
    with pytest.raises(ValueError, match="must exactly match"):
        seal_training_run_artifacts(copied)


def test_training_run_rejects_uninventoried_preprocessing_checksum_claim(
    training_run_with_nested_checksums: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "unexpected-preprocessing-checksum"
    shutil.copytree(training_run_with_nested_checksums, copied)
    run_manifest = _read_json(copied / "run_manifest.json")
    model_manifest_path = copied / run_manifest["models"][0]["manifest_path"]
    model_manifest = _read_json(model_manifest_path)
    model_manifest["preprocessing_artifacts"]["files"].append(
        {
            "path": "preprocessing/unpublished.joblib",
            "sha256": "0" * 64,
        }
    )
    write_json(model_manifest, model_manifest_path)
    write_json(
        model_manifest["preprocessing_artifacts"],
        model_manifest_path.parent / "preprocessing" / "checksums.json",
    )

    with pytest.raises(ValueError, match="unexpected_claims"):
        seal_training_run_artifacts(copied)


def test_training_run_rejects_resealed_false_checkpoint_checksum(
    training_run_with_nested_checksums: Path,
    tmp_path: Path,
) -> None:
    copied = tmp_path / "false-checkpoint-checksum"
    shutil.copytree(training_run_with_nested_checksums, copied)
    run_manifest = _read_json(copied / "run_manifest.json")
    model_manifest_path = copied / run_manifest["models"][0]["manifest_path"]
    model_manifest = _read_json(model_manifest_path)
    model_manifest["training_metadata"]["model_checkpoint"]["sha256"] = "0" * 64
    write_json(model_manifest, model_manifest_path)
    write_json(
        model_manifest["training_metadata"],
        model_manifest_path.parent / "training_metadata.json",
    )

    with pytest.raises(ValueError, match="checkpoint checksum disagrees"):
        seal_training_run_artifacts(copied)
