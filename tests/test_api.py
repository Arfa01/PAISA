from __future__ import annotations

import json
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

    status = client.get("/pipeline/status")
    assert status.status_code == 200
    assert status.json() == json.loads(
        (data_dir / "processed" / "pipeline_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    stocks = client.get("/stocks")
    assert stocks.status_code == 200
    assert set(stocks.json()) == {"count", "stocks"}
    assert set(stocks.json()["stocks"]) == {"HBL", "OGDC"}

    prices = client.get("/stocks/ogdc/prices?limit=3")
    assert prices.status_code == 200
    assert set(prices.json()) == {"ticker", "count", "rows"}
    assert prices.json()["ticker"] == "OGDC"
    assert prices.json()["count"] == 3
    assert len(prices.json()["rows"]) == 3
    assert [row["date"] for row in prices.json()["rows"]] == sorted(
        row["date"] for row in prices.json()["rows"]
    )

    features = client.get("/stocks/OGDC/features?limit=3")
    assert features.status_code == 200
    assert set(features.json()) == {"ticker", "count", "rows"}
    assert features.json()["ticker"] == "OGDC"
    assert features.json()["count"] == len(features.json()["rows"]) == 3

    dataset = client.get("/stocks/OGDC/dataset?limit=3")
    assert dataset.status_code == 200
    assert set(dataset.json()) == {"ticker", "count", "rows"}
    assert dataset.json()["ticker"] == "OGDC"
    assert dataset.json()["count"] == len(dataset.json()["rows"]) == 3

    metrics = client.get("/models/baseline/metrics")
    assert metrics.status_code == 200
    assert metrics.json() == json.loads(
        (model_dir / "baseline_metrics.json").read_text(encoding="utf-8")
    )

    pred = client.get("/predict/OGDC")
    assert pred.status_code == 200
    assert pred.json()["prediction"]["predicted_direction"] in {"up", "down"}
