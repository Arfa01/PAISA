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

`pypsx-toolkit` advertises 10 years of PSX OHLCV data as pandas DataFrames. If
both its installation and import verification succeed, use:

```bash
source .venv/bin/activate
pip install -r requirements-provider-pypsx.txt
python -c "import pypsx_toolkit"
python scripts/run_v0_pipeline.py --provider pypsx --symbols OGDC HBL MCB --start 2021-01-01 --train
```

The optional provider is intentionally excluded from `requirements.txt` because
its upstream distribution can install successfully but fail to import on some
platforms. The default `psx-dps`, `manual-csv`, and `sample` providers do not
require it.

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

## Versioned OGDC model experiments

The experiment trainer reuses the daily PAISA feature pipeline without
replacing the legacy v0 baseline. It trains four variants on one shared,
chronological, leakage-safe split:

- `M-001`: Random Forest with the existing baseline configuration.
- `M-002`: conservative Random Forest.
- `M-003`: more flexible Random Forest.
- `M-004`: XGBoost using the same features, target dates, and split dates.

Install the default dependencies and run the unified command:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/train_ogdc_models.py
```

The defaults are OGDC, daily data, the `psx-dps` provider, a requested start
date of `2021-01-01`, random seed `42`, and immutable output below
`artifacts/model_training/`. Use `--help` to see the explicit overrides.
Generated model binaries, frozen inputs, and ZIP archives are ignored by Git.

Each model's `predictions.csv` follows the architecture prediction-record
contract and can be passed directly to the backtesting CLI. The experiment
configuration must reference the same `model_id` and the desired test-date
window:

```bash
python scripts/run_backtest.py \
  --predictions artifacts/model_training/<run_id>/models/<model_id>/predictions.csv \
  --prices data/processed/psx_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

Open [`notebooks/PAISA_OGDC_Model_Training.ipynb`](notebooks/PAISA_OGDC_Model_Training.ipynb)
for the Colab workflow. The optional
[`notebooks/PAISA_OGDC_LSTM_Training.ipynb`](notebooks/PAISA_OGDC_LSTM_Training.ipynb)
keeps deep-learning dependencies separate from the CPU pipeline. See
[`docs/model_training_stage.md`](docs/model_training_stage.md) for target,
split, metric, artifact, and provider details.

## Versioned OGDC backtesting

The backtesting engine consumes a standardized prediction CSV, daily OGDC
prices, and an immutable experiment configuration. It does not train models and
does not change the existing v0 pipeline.

```bash
python scripts/run_backtest.py \
  --predictions path/to/predictions.csv \
  --prices path/to/OGDC_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

`--runs-dir` is optional and defaults to `runs/`. Each successful run creates a
new `runs/<experiment_id>/` directory containing the normalized configuration,
predictions, signals, trades, daily portfolio snapshots, benchmark ledgers,
metrics, an equity-curve PNG, and a drawdown PNG. Existing experiment IDs and
portfolio IDs are rejected rather than overwritten.

See [the backtesting runbook](docs/BACKTESTING.md) for the exact CSV schemas,
configuration, execution timing, accounting rules, metric formulas, and output
tree. All backtests are academic simulations and are not financial advice.

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
PAISA/
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

## Current roadmap

The earlier v1–v5 sequence placed news and sentiment before the dashboard. The
dated OGDC vertical-slice sprint supersedes that order for 19–24 October 2026;
news, sentiment, SHAP, and relationship graphs are deferred. See
[`docs/OCT19_24_SPRINT.md`](docs/OCT19_24_SPRINT.md) for the active scope and
dependency order.
