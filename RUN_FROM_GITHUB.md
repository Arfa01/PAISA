# Run PAISA v0 from GitHub

These are the exact steps to run the current PAISA v0 backend from a fresh clone.

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

The default install supports `psx-dps`, `manual-csv`, and `sample`. To try the
optional full-OHLCV provider, install its dependencies separately:

```bash
pip install -r requirements-provider-pypsx.txt
python -c "import pypsx_toolkit"
```

`pypsx-toolkit` is excluded from the default requirements because its upstream
package can install successfully but fail to import on some platforms. A failed
optional-provider install or import does not block the standard PAISA workflow.

## 4. Generate PSX data, features, dataset, and baseline model

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train
```

This generates:

```text
data/raw/psx/
data/processed/psx_prices.csv
data/processed/psx_features.csv
data/processed/ml_dataset.csv
models/baseline_random_forest.joblib
models/baseline_metrics.json
```

## 5. Run tests

```bash
pytest
```

Expected: all collected tests pass. The count will grow as components are added.

## 6. Start the API

Use this command:

```bash
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

## 7. Open the API in the browser

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/stocks
http://127.0.0.1:8000/stocks/OGDC/prices
http://127.0.0.1:8000/stocks/OGDC/features
http://127.0.0.1:8000/stocks/OGDC/dataset
http://127.0.0.1:8000/models/baseline/metrics
http://127.0.0.1:8000/predict/OGDC
http://127.0.0.1:8000/docs
```

## Current working PSX command

This was the command used successfully for v0:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train
```

## Notes

- Generated CSV/model files are ignored by Git and should be regenerated locally.
- v0 is an academic prototype and does not give buy/sell advice.
- The current PSX provider may not return high/low values; it currently works with open, close, and volume.
