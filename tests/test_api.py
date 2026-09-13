from __future__ import annotations

from datetime import date

from fastapi.testclient import TestClient

from paisa.api.main import app
from paisa.pipeline import run_v0_pipeline


def test_api_can_read_generated_outputs(tmp_path, monkeypatch):
    # paisa.api.main reads env at import time, so patch module-level paths too.
    import paisa.api.main as api_main
    from paisa.config import get_paths

    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(api_main, "paths", get_paths(data_dir=data_dir, model_dir=model_dir))

    run_v0_pipeline(
        symbols=["OGDC", "HBL"],
        provider_name="sample",
        start=date(2021, 1, 1),
        end=date(2022, 12, 31),
        train=True,
        data_dir=data_dir,
        model_dir=model_dir,
    )

    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["prices_ready"] is True

    stocks = client.get("/stocks")
    assert stocks.status_code == 200
    assert set(stocks.json()["stocks"]) == {"HBL", "OGDC"}

    pred = client.get("/predict/OGDC")
    assert pred.status_code == 200
    assert pred.json()["prediction"]["predicted_direction"] in {"up", "down"}
