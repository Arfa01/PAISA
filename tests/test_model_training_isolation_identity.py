from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from uuid import UUID

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from paisa.api.main import app
from paisa.config import get_paths
from paisa.experiments.trainer import run_training_stage
from paisa.pipeline import run_v0_pipeline


def _price_fixture(rows: int = 120) -> pd.DataFrame:
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


def _snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_default_acquisition_preserves_canonical_v0_data_models_and_api_stocks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A training fetch must not publish into the canonical v0 directories."""

    import paisa.api.main as api_main

    repo = tmp_path / "repository"
    data_dir = repo / "data"
    model_dir = repo / "models"
    run_v0_pipeline(
        symbols=["OGDC", "HBL"],
        provider_name="sample",
        start=date(2021, 1, 1),
        end=date(2022, 12, 31),
        train=True,
        data_dir=data_dir,
        model_dir=model_dir,
    )
    monkeypatch.setattr(
        api_main,
        "paths",
        get_paths(data_dir=data_dir, model_dir=model_dir),
    )
    client = TestClient(app)
    before_stocks = client.get("/stocks")
    assert before_stocks.status_code == 200
    assert before_stocks.json() == {"count": 2, "stocks": ["HBL", "OGDC"]}
    before = _snapshot_tree(repo)

    summary = run_training_stage(
        provider="sample",
        start="2021-01-01",
        output_dir=repo / "artifacts" / "model_training",
        repo_root=repo,
        run_id="isolated-acquisition",
        create_zip=False,
        model_ids=["M-001"],
    )

    after = _snapshot_tree(repo)
    # The newly published run is intentionally excluded from the legacy snapshot.
    after_legacy = {
        name: payload
        for name, payload in after.items()
        if not name.startswith("artifacts/")
    }
    assert after_legacy == before
    after_stocks = client.get("/stocks")
    assert after_stocks.status_code == 200
    assert after_stocks.json() == before_stocks.json()
    assert Path(summary["prediction_paths"]["M-001"]).is_file()


def test_registered_model_ids_are_unique_across_independent_run_roots(
    tmp_path: Path,
) -> None:
    """Invocation identity prevents foreign-key collisions across registries."""

    prices_path = tmp_path / "prices.csv"
    _price_fixture().to_csv(prices_path, index=False)

    first = run_training_stage(
        start="2025-01-01",
        prices_path=prices_path,
        output_dir=tmp_path / "registry-a",
        repo_root=tmp_path,
        run_id="same-run-id",
        create_zip=False,
        model_ids=["M-001"],
    )
    second = run_training_stage(
        start="2025-01-01",
        prices_path=prices_path,
        output_dir=tmp_path / "registry-b",
        repo_root=tmp_path,
        run_id="same-run-id",
        create_zip=False,
        model_ids=["M-001"],
    )

    assert first["model_ids"]["M-001"] != second["model_ids"]["M-001"]
    assert first["invocation_id"] != second["invocation_id"]

    for summary in (first, second):
        assert str(UUID(summary["invocation_id"])) == summary["invocation_id"]
        model_id = summary["model_ids"]["M-001"]
        run_dir = Path(summary["artifact_directory"])
        run_manifest = json.loads(
            (run_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        model_manifest = json.loads(
            (run_dir / "models" / model_id / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        configuration = json.loads(
            (run_dir / "models" / model_id / "config.json").read_text(
                encoding="utf-8"
            )
        )
        assert run_manifest["invocation_id"] == summary["invocation_id"]
        assert model_manifest["invocation_id"] == summary["invocation_id"]
        assert configuration["invocation_id"] == summary["invocation_id"]
        assert summary["invocation_id"] in model_id
