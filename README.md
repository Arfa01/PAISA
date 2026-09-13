# PAISA v0/v0.1 — Predictive AI Stock Assessment

PAISA is an academic Final Year Project prototype for AI-assisted stock market analysis. The current implementation is focused on the PSX stock-data foundation: fetching historical data, cleaning it, building technical features and labels, training baseline models, checking data quality, and exposing results through FastAPI.

This project is **not financial advice** and does not produce buy/sell recommendations.

## Start here

- Exact clone-and-run instructions: [`RUN_FROM_GITHUB.md`](RUN_FROM_GITHUB.md)
- Full project background and team context: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)
- API endpoint notes: [`docs/api_design.md`](docs/api_design.md)
- Data-quality workflow: [`docs/data_quality_report_v0.md`](docs/data_quality_report_v0.md)
- Model-comparison workflow: [`docs/model_comparison_v0.md`](docs/model_comparison_v0.md)

## What the current version includes

- PSX historical data ingestion through the `psx-dps` provider.
- Optional `pypsx-toolkit` provider.
- Manual CSV fallback provider.
- Sample provider for offline tests and demos.
- Standard price schema: `date, ticker, open, high, low, close, volume, source`.
- Technical features: returns, lagged close, moving averages, volatility, RSI, MACD, and volume change.
- Supervised label: `target_next_day_up`.
- Chronological train/validation/test splitting.
- Baseline Random Forest model.
- Data-quality report generation.
- Price-only model comparison: Logistic Regression, Random Forest, and Histogram Gradient Boosting.
- FastAPI endpoints for prices, features, ML dataset, data quality, model metrics, model comparison, and prediction.
- Pytest coverage for preprocessing, pipeline, API, data quality, and model comparison.

## Quick local run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Starter run:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train --quality-report
```

Expanded v0.1 run:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols-from-csv data/metadata/stock_universe.csv --max-priority 2 --start 2021-01-01 --train --quality-report
```

Run model comparison:

```bash
python scripts/run_model_comparison.py
```

Run tests:

```bash
pytest
```

Start API:

```bash
python scripts/serve_api.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

## API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check generated data/model readiness. |
| `GET /pipeline/status` | View latest pipeline metadata. |
| `GET /stocks` | List tickers available in processed data. |
| `GET /stocks/{ticker}/prices` | Return cleaned historical prices. |
| `GET /stocks/{ticker}/features` | Return feature-engineered rows. |
| `GET /stocks/{ticker}/dataset` | Return ML-ready rows with labels. |
| `GET /quality/data` | Return full data-quality summary. |
| `GET /quality/tickers` | Return per-ticker quality rows. |
| `GET /models/baseline/metrics` | Return baseline Random Forest metrics. |
| `GET /models/comparison` | Return saved price-only model comparison results. |
| `GET /predict/{ticker}` | Return next-trading-day up/down prediction JSON. |

## Generated files

Generated data/model files are ignored by Git and should be regenerated locally.

```text
data/raw/psx/
data/processed/psx_prices.csv
data/processed/psx_features.csv
data/processed/ml_dataset.csv
data/processed/data_quality_report.json
data/processed/data_quality_report.csv
models/baseline_random_forest.joblib
models/baseline_metrics.json
models/model_comparison_price_only.csv
models/model_comparison_price_only.json
models/model_comparison_price_only.joblib
```

## Current limitation

The current `psx-dps` provider has been observed to provide date, open, close, and volume. High/low may be missing for this source, so the first baseline feature set avoids depending on high/low.

## Recommended next milestone

Do not jump directly to LSTM or the dashboard. The next milestone is:

```text
v1 = news ingestion + company/date matching + daily sentiment features
```

After v1, compare:

```text
price-only models vs price + sentiment models
```

That comparison is one of the main FYP requirements.
