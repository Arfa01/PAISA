from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from paisa.api.router_v1 import router as results_v1_router
from paisa.api.schemas import HealthResponse
from paisa.config import get_paths
from paisa.features import FEATURE_COLUMNS

paths = get_paths(data_dir=os.getenv("PAISA_DATA_DIR"), model_dir=os.getenv("PAISA_MODEL_DIR"))
app = FastAPI(title="PAISA v0 API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HEALTH_DATA_LABEL = "legacy-v0-data"
HEALTH_MODEL_LABEL = "legacy-v0-models"
_SAFE_HEALTH_STOCK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_MODEL_BINARY_SUFFIXES = (
    ".joblib",
    ".pkl",
    ".pickle",
    ".keras",
    ".h5",
    ".hdf5",
    ".onnx",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
)


def _safe_health_stock(value: object) -> str | None:
    ticker = str(value)
    if not _SAFE_HEALTH_STOCK.fullmatch(ticker):
        return None
    if ticker.lower().endswith(_MODEL_BINARY_SUFFIXES):
        return None
    return ticker


def _path(name: str) -> Path:
    mapping = {
        "prices": paths.processed_dir / "psx_prices.csv",
        "features": paths.processed_dir / "psx_features.csv",
        "dataset": paths.processed_dir / "ml_dataset.csv",
        "metadata": paths.processed_dir / "pipeline_metadata.json",
        "model": paths.model_dir / "baseline_random_forest.joblib",
        "metrics": paths.model_dir / "baseline_metrics.json",
        "importance": paths.model_dir / "baseline_feature_importance.csv",
    }
    return mapping[name]


def _read_csv_or_404(name: str) -> pd.DataFrame:
    path = _path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{path.name} not found. Run scripts/run_v0_pipeline.py first.")
    return pd.read_csv(path)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.replace({pd.NA: None}).to_json(orient="records", date_format="iso"))


@app.get(
    "/health",
    response_model=HealthResponse,
    response_model_exclude_none=True,
    response_description="Legacy pipeline readiness without local filesystem paths.",
)
def health() -> HealthResponse:
    prices_exists = _path("prices").exists()
    dataset_exists = _path("dataset").exists()
    model_exists = _path("model").exists()
    price_rows: int | None = None
    stocks: list[str] | None = None
    if prices_exists:
        prices = pd.read_csv(_path("prices"))
        price_rows = int(len(prices))
        stock_labels = (
            _safe_health_stock(raw_ticker)
            for raw_ticker in prices["ticker"].dropna().unique()
        )
        stocks = sorted(ticker for ticker in stock_labels if ticker is not None)
    return HealthResponse(
        status="ok",
        data_dir=HEALTH_DATA_LABEL,
        model_dir=HEALTH_MODEL_LABEL,
        prices_ready=prices_exists,
        dataset_ready=dataset_exists,
        model_ready=model_exists,
        price_rows=price_rows,
        stocks=stocks,
    )


@app.get("/pipeline/status")
def pipeline_status() -> dict[str, Any]:
    path = _path("metadata")
    if not path.exists():
        raise HTTPException(status_code=404, detail="pipeline_metadata.json not found. Run pipeline first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/stocks")
def list_stocks() -> dict[str, Any]:
    prices = _read_csv_or_404("prices")
    stocks = sorted(prices["ticker"].dropna().unique().tolist())
    return {"count": len(stocks), "stocks": stocks}


@app.get("/stocks/{ticker}/prices")
def get_prices(ticker: str, limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    prices = _read_csv_or_404("prices")
    rows = prices[prices["ticker"].str.upper() == ticker.upper()].sort_values("date").tail(limit)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No price rows for {ticker}.")
    return {"ticker": ticker.upper(), "count": int(len(rows)), "rows": _records(rows)}


@app.get("/stocks/{ticker}/features")
def get_features(ticker: str, limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    features = _read_csv_or_404("features")
    rows = features[features["ticker"].str.upper() == ticker.upper()].sort_values("date").tail(limit)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No feature rows for {ticker}.")
    return {"ticker": ticker.upper(), "count": int(len(rows)), "rows": _records(rows)}


@app.get("/stocks/{ticker}/dataset")
def get_dataset(ticker: str, limit: int = Query(default=200, ge=1, le=5000)) -> dict[str, Any]:
    dataset = _read_csv_or_404("dataset")
    rows = dataset[dataset["ticker"].str.upper() == ticker.upper()].sort_values("date").tail(limit)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No ML dataset rows for {ticker}.")
    return {"ticker": ticker.upper(), "count": int(len(rows)), "rows": _records(rows)}


@app.get("/models/baseline/metrics")
def baseline_metrics() -> dict[str, Any]:
    path = _path("metrics")
    if not path.exists():
        raise HTTPException(status_code=404, detail="baseline_metrics.json not found. Run pipeline with --train.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/predict/{ticker}")
def predict(ticker: str) -> dict[str, Any]:
    model_path = _path("model")
    if not model_path.exists():
        raise HTTPException(status_code=404, detail="baseline_random_forest.joblib not found. Run pipeline with --train.")

    features = _read_csv_or_404("features")
    rows = features[features["ticker"].str.upper() == ticker.upper()].sort_values("date")
    rows = rows.dropna(subset=FEATURE_COLUMNS)
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No complete feature row available for {ticker}.")

    artifact = joblib.load(model_path)
    model = artifact["model"]
    feature_columns = artifact.get("feature_columns", FEATURE_COLUMNS)
    latest = rows.iloc[-1]
    x = latest[feature_columns].to_frame().T
    predicted_class = int(model.predict(x)[0])
    probability = model.predict_proba(x)[0]
    confidence = float(max(probability))
    direction = "up" if predicted_class == 1 else "down"

    importance = artifact.get("feature_importance", [])[:5]
    top_factors = [
        {
            "feature": item["feature"],
            "importance": round(float(item["importance"]), 6),
            "latest_value": None if pd.isna(latest[item["feature"]]) else float(latest[item["feature"]]),
        }
        for item in importance
        if item["feature"] in latest.index
    ]

    return {
        "ticker": ticker.upper(),
        "as_of_date": str(latest["date"]),
        "model": "baseline_random_forest",
        "prediction": {
            "target": "next_trading_day_direction",
            "predicted_direction": direction,
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": {
                "down": round(float(probability[0]), 4),
                "up": round(float(probability[1]), 4) if len(probability) > 1 else None,
            },
        },
        "explanation": {
            "method": "RandomForest feature importance v0; SHAP comes in a later version.",
            "top_factors": top_factors,
        },
        "disclaimer": "Academic prototype only. This is not financial advice or a buy/sell recommendation.",
    }


# The additive v1 router reads immutable result artifacts only.  Legacy routes
# above deliberately retain their existing behavior and response shapes.
app.include_router(results_v1_router)
