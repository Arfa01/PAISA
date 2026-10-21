from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
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
    health_payload = health.json()
    expected_price_rows = len(
        pd.read_csv(data_dir / "processed" / "psx_prices.csv")
    )
    assert health_payload == {
        "status": "ok",
        "data_dir": "legacy-v0-data",
        "model_dir": "legacy-v0-models",
        "prices_ready": True,
        "dataset_ready": True,
        "model_ready": True,
        "price_rows": expected_price_rows,
        "stocks": ["HBL", "OGDC"],
    }
    serialized_health = health.text
    assert "/Users/" not in serialized_health
    assert re.search(r"[A-Za-z]:[\\/]", serialized_health) is None
    assert str(Path(__file__).resolve().parents[1]) not in serialized_health
    assert str(tmp_path) not in serialized_health
    assert not any(
        suffix in serialized_health.lower()
        for suffix in (".joblib", ".keras", ".onnx", ".pt", ".pth")
    )

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


def test_health_openapi_and_error_response_do_not_expose_paths(
    tmp_path,
    monkeypatch,
):
    import paisa.api.main as api_main
    from paisa.config import get_paths

    data_dir = tmp_path / "private-owner" / "data"
    model_dir = tmp_path / "private-owner" / "models"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "psx_prices.csv").write_text(
        "date,ticker,open,high,low,close,volume,source\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api_main,
        "paths",
        get_paths(data_dir=data_dir, model_dir=model_dir),
    )
    secret_path = "/Users/private-owner/models/private.joblib"

    def fail_read(*_args, **_kwargs):
        raise RuntimeError(f"could not read {secret_path}")

    monkeypatch.setattr(api_main.pd, "read_csv", fail_read)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/health")
    assert response.status_code == 500
    assert secret_path not in response.text
    assert "/Users/" not in response.text
    assert ".joblib" not in response.text

    openapi = client.get("/openapi.json")
    assert openapi.status_code == 200
    health_operation = openapi.json()["paths"]["/health"]["get"]
    assert health_operation["responses"]["200"]["content"]["application/json"][
        "schema"
    ] == {"$ref": "#/components/schemas/HealthResponse"}


def test_health_filters_unsafe_stock_labels(tmp_path, monkeypatch) -> None:
    import paisa.api.main as api_main
    from paisa.config import get_paths

    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True)
    (processed_dir / "psx_prices.csv").write_text(
        "ticker\nOGDC\n/Users/private-owner/secret\nC:\\private\\secret\n"
        "model.joblib\nweights.ONNX\ncheckpoint.safetensors\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        api_main,
        "paths",
        get_paths(data_dir=data_dir, model_dir=tmp_path / "models"),
    )

    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["price_rows"] == 6
    assert response.json()["stocks"] == ["OGDC"]
    assert "/Users/" not in response.text
    assert re.search(r"[A-Za-z]:[\\/]", response.text) is None
    assert ".joblib" not in response.text


def test_production_preview_origin_is_allowed_by_cors() -> None:
    client = TestClient(app)
    response = client.options(
        "/api/v1/vertical-slices",
        headers={
            "Origin": "http://127.0.0.1:4173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "http://127.0.0.1:4173"
    )
