from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import subprocess

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


def test_publish_refuses_immutable_run_collision(tmp_path: Path) -> None:
    destination = tmp_path / "SENT-20261022-130405-deadbeef"
    destination.mkdir()
    staging = tmp_path / ".staging"
    staging.mkdir()

    with pytest.raises(SentimentRunCollisionError):
        publish_staged_run(staging, destination)

    assert staging.exists()


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
