from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess
import os

import pytest

from paisa.sentiment.artifacts import (
    SentimentRunCollisionError,
    artifact_inventory,
    make_sentiment_run_id,
    publish_staged_run,
    validate_sentiment_run_id,
    write_json,
)


def test_sentiment_run_id_is_safe_and_time_scoped() -> None:
    value = make_sentiment_run_id(datetime(2026, 10, 22, 13, 4, 5, tzinfo=timezone.utc))

    assert value.startswith("SENT-20261022-130405-")
    assert validate_sentiment_run_id(value) == value
    with pytest.raises(ValueError):
        validate_sentiment_run_id("../SENT-unsafe")


def test_artifact_inventory_is_exact_and_checksum_bound(tmp_path: Path) -> None:
    write_json(tmp_path / "quality_report.json", {"status": "PASS"})

    inventory = artifact_inventory(
        tmp_path,
        expected_files=("quality_report.json",),
    )

    assert inventory[0]["path"] == "quality_report.json"
    assert len(inventory[0]["sha256"]) == 64
    assert inventory[0]["size_bytes"] > 0

    (tmp_path / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact set mismatch"):
        artifact_inventory(tmp_path, expected_files=("quality_report.json",))


def test_artifact_inventory_rejects_symlinked_payload(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text('{"secret": true}\n', encoding="utf-8")
    os.symlink(outside, tmp_path / "quality_report.json")

    with pytest.raises(ValueError, match="regular files"):
        artifact_inventory(tmp_path, expected_files=("quality_report.json",))


def test_publish_is_atomically_no_replace_during_check_use_gap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "SENT-20261022-130405-deadbeef"
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "evidence.txt").write_text("new", encoding="utf-8")

    from paisa.sentiment import artifacts

    original = artifacts._rename_noreplace

    expected = artifact_inventory(staging, expected_files=("evidence.txt",))

    def race(source: Path, target: Path, **kwargs) -> None:
        target.mkdir()
        (target / "incumbent.txt").write_text("old", encoding="utf-8")
        original(source, target, **kwargs)

    monkeypatch.setattr(artifacts, "_rename_noreplace", race)
    with pytest.raises(SentimentRunCollisionError):
        publish_staged_run(staging, destination, expected_inventory=expected)

    assert (destination / "incumbent.txt").read_text(encoding="utf-8") == "old"
    assert staging.is_dir()


def test_publish_refuses_immutable_run_collision(tmp_path: Path) -> None:
    destination = tmp_path / "SENT-20261022-130405-deadbeef"
    destination.mkdir()
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "evidence.txt").write_text("new", encoding="utf-8")
    expected = artifact_inventory(staging, expected_files=("evidence.txt",))

    with pytest.raises(SentimentRunCollisionError):
        publish_staged_run(staging, destination, expected_inventory=expected)

    assert staging.exists()


def test_publish_rejects_staging_root_symlink_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "SENT-20261022-130405-deadbeef"
    staging = tmp_path / ".staging"
    staging.mkdir()
    (staging / "evidence.txt").write_text("sealed", encoding="utf-8")
    expected = artifact_inventory(staging, expected_files=("evidence.txt",))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "evidence.txt").write_text("external", encoding="utf-8")
    displaced = tmp_path / ".displaced"

    from paisa.sentiment import artifacts

    original = artifacts._rename_noreplace

    def swap_root(source: Path, target: Path, **kwargs) -> None:
        source.rename(displaced)
        source.symlink_to(outside, target_is_directory=True)
        original(source, target, **kwargs)

    monkeypatch.setattr(artifacts, "_rename_noreplace", swap_root)
    with pytest.raises(RuntimeError, match="staging root changed"):
        publish_staged_run(staging, destination, expected_inventory=expected)

    assert not destination.exists()
    assert staging.is_symlink()


def test_publish_rejects_leaf_replacement_after_initial_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "SENT-20261022-130405-deadbeef"
    staging = tmp_path / ".staging"
    staging.mkdir()
    evidence = staging / "evidence.txt"
    evidence.write_text("sealed", encoding="utf-8")
    expected = artifact_inventory(staging, expected_files=("evidence.txt",))

    from paisa.sentiment import artifacts

    original = artifacts._rename_noreplace

    def replace_leaf(source: Path, target: Path, **kwargs) -> None:
        evidence.unlink()
        evidence.write_text("different", encoding="utf-8")
        original(source, target, **kwargs)

    monkeypatch.setattr(artifacts, "_rename_noreplace", replace_leaf)
    with pytest.raises(RuntimeError, match="artifact bytes changed"):
        publish_staged_run(staging, destination, expected_inventory=expected)

    assert staging.is_dir()
    assert not destination.exists()


def test_generated_model_weights_and_cache_are_ignored_by_git() -> None:
    repository = Path(__file__).resolve().parents[2]
    for relative in (
        "data/sentiment_model_cache/model.safetensors",
        "models/finbert/model.safetensors",
        "models/finbert/pytorch_model.bin",
        "models/finbert/weights.pt",
        "models/finbert/weights.pth",
    ):
        completed = subprocess.run(
            ["git", "check-ignore", "--quiet", "--no-index", relative],
            cwd=repository,
            check=False,
        )
        assert completed.returncode == 0, relative
