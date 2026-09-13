# Run PAISA from GitHub

These are the exact steps to run the current PAISA backend from a fresh clone.

## 1. Clone the repo

```bash
git clone https://github.com/Arfa01/PAISA.git
cd PAISA
```

## 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. Generate PSX data, features, dataset, baseline model, and quality report

Starter 3-stock run:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train --quality-report
```

Recommended v0.1 expanded run:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols-from-csv data/metadata/stock_universe.csv --max-priority 2 --start 2021-01-01 --train --quality-report
```

This generates:

```text
data/raw/psx/
data/processed/psx_prices.csv
data/processed/psx_features.csv
data/processed/ml_dataset.csv
data/processed/data_quality_report.json
data/processed/data_quality_report.csv
docs/data_quality_report_v0.md
models/baseline_random_forest.joblib
models/baseline_metrics.json
```

## 5. Run price-only model comparison

```bash
python scripts/run_model_comparison.py
```

This generates:

```text
models/model_comparison_price_only.csv
models/model_comparison_price_only.json
models/model_comparison_price_only.joblib
```

## 6. Run tests

```bash
pytest
```

Expected:

```text
all tests pass
```

## 7. Start the API

Use either command:

```bash
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

or:

```bash
python scripts/serve_api.py
```

## 8. Open the API in the browser

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/stocks
http://127.0.0.1:8000/stocks/OGDC/prices
http://127.0.0.1:8000/stocks/OGDC/features
http://127.0.0.1:8000/stocks/OGDC/dataset
http://127.0.0.1:8000/quality/data
http://127.0.0.1:8000/quality/tickers
http://127.0.0.1:8000/models/baseline/metrics
http://127.0.0.1:8000/models/comparison
http://127.0.0.1:8000/predict/OGDC
http://127.0.0.1:8000/docs
```

## Notes

- Generated CSV/model files are ignored by Git and should be regenerated locally.
- v0/v0.1 is an academic prototype and does not give buy/sell advice.
- The current PSX provider may not return high/low values; it currently works with open, close, and volume.
