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

## Run versioned OGDC model experiments

The unified trainer uses OGDC daily data and a chronological train/validation/test
split. It preserves the v0 baseline model and writes each new run to a unique,
immutable artifact directory.

```bash
python scripts/train_ogdc_models.py
```

The command defaults to:

```text
symbol=OGDC
start=2021-01-01
provider=psx-dps
output-dir=artifacts/model_training
random-state=42
```

For the same command with every option visible:

```bash
python scripts/train_ogdc_models.py \
  --symbol OGDC \
  --start 2021-01-01 \
  --provider psx-dps \
  --output-dir artifacts/model_training \
  --random-state 42
```

It trains three Random Forest variants and one XGBoost variant, writes
contract-compatible prediction CSV files, compares them with majority-class and
previous-direction baselines, and selects a provisional champion using
validation metrics only. Generated artifacts and ZIP archives are ignored by
Git.

For Colab, use:

```text
notebooks/PAISA_OGDC_Model_Training.ipynb
```

The separate optional LSTM workflow is:

```text
notebooks/PAISA_OGDC_LSTM_Training.ipynb
```

TensorFlow/Keras is not part of the default CPU installation. See
[`docs/model_training_stage.md`](docs/model_training_stage.md) for artifact and
data-provider details.

## Run a versioned OGDC backtest

Backtesting is separate from model training. Supply contract-compatible model
predictions, the corresponding daily OGDC prices, and an immutable experiment
configuration:

```bash
python scripts/run_backtest.py \
  --predictions path/to/predictions.csv \
  --prices path/to/OGDC_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

`--runs-dir` is optional and defaults to `runs/`. A successful command writes
`runs/<experiment_id>/`. It fails if that experiment directory already exists
or any configured strategy or benchmark portfolio ID has already been used;
choose new IDs instead of deleting or overwriting an old run.

The required input columns, complete configuration example, after-close to
next-session-open timing, accounting rules, metric formulas, and generated
artifact tree are documented in
[`docs/BACKTESTING.md`](docs/BACKTESTING.md).

A generated model prediction file can be supplied directly:

```bash
python scripts/run_backtest.py \
  --predictions artifacts/model_training/<run_id>/models/<model_id>/predictions.csv \
  --prices data/processed/psx_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

The experiment configuration's `model_id` and evaluation window must match the
selected prediction file. The backtester reads only the standardized prediction
fields; training diagnostics remain model artifacts.

All backtest output is an academic simulation and is not financial advice.

## Notes

- Generated CSV/model files are ignored by Git and should be regenerated locally.
- v0 is an academic prototype and does not give buy/sell advice.
- The current PSX provider may not return high/low values; it currently works with open, close, and volume.
