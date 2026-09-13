# PAISA v0 — Predictive AI Stock Assessment

PAISA v0 is the first implementation slice of the FYP: a PSX-first stock-data layer that fetches historical market data, cleans it, builds technical features and labels, trains a baseline model, and exposes the outputs through a FastAPI backend.

This is **not financial advice** and does not produce buy/sell recommendations. It is an academic prototype for data engineering, ML evaluation, and dashboard integration.

## What v0 includes

- PSX historical data ingestion through `psx-dps` provider.
- Optional full-OHLCV ingestion through `pypsx-toolkit` provider.
- Manual CSV provider for cases where data access is blocked.
- Sample provider for offline tests and API demos.
- Standard price schema: `date, ticker, open, high, low, close, volume, source`.
- Technical features: returns, lagged close, moving averages, volatility, RSI, MACD, volume change.
- Supervised label: `target_next_day_up`.
- Chronological train/validation/test split.
- Baseline RandomForest model.
- FastAPI endpoints for prices, features, dataset, metrics, and prediction.
- Pytest tests for pipeline, preprocessing, and API.

## Recommended first run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Offline smoke test with fake data
python scripts/run_v0_pipeline.py --provider sample --symbols OGDC HBL MCB --start 2021-01-01 --end 2023-12-31 --train

# Run tests
pytest

# Start API
uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/stocks
- http://127.0.0.1:8000/stocks/OGDC/prices
- http://127.0.0.1:8000/models/baseline/metrics
- http://127.0.0.1:8000/predict/OGDC
- http://127.0.0.1:8000/docs

## PSX run

```bash
source .venv/bin/activate
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train
```

The `psx-dps` provider uses the public PSX Data Portal time-series endpoint:

```text
https://dps.psx.com.pk/timeseries/eod/{SYMBOL}
```

Observed response rows are shaped like:

```text
[unix_timestamp, close_price, volume, open_price]
```

So this provider gives `open`, `close`, and `volume`; `high` and `low` may remain missing. That is acceptable for v0 because the first baseline model is trained on open/close/volume-derived indicators.

## Optional full-OHLCV run

`pypsx-toolkit` advertises 10 years of PSX OHLCV data as pandas DataFrames. If it installs successfully on your Mac, use:

```bash
source .venv/bin/activate
python scripts/run_v0_pipeline.py --provider pypsx --symbols OGDC HBL MCB --start 2021-01-01 --train
```

## Manual CSV fallback

If both providers fail, download one CSV per ticker and place files here:

```text
data/external/manual_psx/OGDC.csv
data/external/manual_psx/HBL.csv
data/external/manual_psx/MCB.csv
```

Then run:

```bash
python scripts/run_v0_pipeline.py --provider manual-csv --manual-csv-dir data/external/manual_psx --symbols OGDC HBL MCB --start 2021-01-01 --train
```

Accepted column variants include `date`, `open`, `open_price`, `close`, `close_price`, `volume`, `high`, and `low`.

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check whether generated data/model artifacts exist. |
| `GET /pipeline/status` | View latest pipeline metadata. |
| `GET /stocks` | List tickers available in processed data. |
| `GET /stocks/{ticker}/prices` | Return cleaned historical price rows. |
| `GET /stocks/{ticker}/features` | Return feature-engineered rows. |
| `GET /stocks/{ticker}/dataset` | Return ML-ready feature + label rows. |
| `GET /models/baseline/metrics` | Return RandomForest split metrics. |
| `GET /predict/{ticker}` | Return next-trading-day up/down prediction for latest complete row. |

## Folder structure

```text
paisa-v0/
├── data/
│   ├── raw/          # generated raw provider outputs
│   ├── processed/    # generated prices/features/dataset/metadata
│   ├── external/     # manually downloaded CSVs
│   └── metadata/     # stock universe
├── docs/
├── models/           # generated model artifacts
├── scripts/
├── src/paisa/
│   ├── api/
│   ├── models/
│   ├── providers/
│   ├── features.py
│   ├── pipeline.py
│   └── preprocessing.py
└── tests/
```

## Definition of Done for v0

v0 is considered successful when:

1. `python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train` completes.
2. `data/raw/psx/` contains raw CSVs for each requested ticker.
3. `data/processed/psx_prices.csv` exists and has standardized columns.
4. `data/processed/psx_features.csv` exists and contains engineered indicators.
5. `data/processed/ml_dataset.csv` exists and contains `target_next_day_up`.
6. `models/baseline_random_forest.joblib` exists.
7. `models/baseline_metrics.json` contains accuracy, precision, recall, and F1 for chronological train/validation/test splits.
8. `pytest` passes.
9. `GET /health` returns `prices_ready=true`, `dataset_ready=true`, and `model_ready=true`.
10. `GET /predict/OGDC` returns a JSON prediction with direction, confidence, top factors, and the academic disclaimer.

## Next versions

- v1: add news ingestion and daily sentiment aggregation.
- v2: compare price-only vs price+sentiment models.
- v3: add SHAP explanations.
- v4: add React + Plotly dashboard.
- v5: add sector/event relationship graph.
