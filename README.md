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
window. The normal multi-symbol `data/processed/psx_prices.csv` is accepted;
the backtester selects the configured symbol before session validation:

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
keeps deep-learning dependencies separate from the CPU pipeline. It calls the
reusable optional LSTM trainer and registers one checksum-protected inference
bundle containing both the Keras model and its train-fitted scaler; its
`predictions.csv` uses the same backtesting contract as the CPU models. See
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

## OGDC end-to-end vertical slice

The vertical-slice runner joins the versioned components into one reproducible
real-data workflow: fetch and clean daily OGDC data, train `M-001` through
`M-004`, pass each model's untouched test predictions directly to an isolated
portfolio backtest, calculate cash-only and buy-and-hold benchmarks, and publish
one comparison report.

Install the normal, machine-learning, backtesting, and test dependencies from
the aggregate requirements file, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python -m pytest
python scripts/run_ogdc_vertical_slice.py
```

The live command uses `psx-dps`. It never substitutes the sample provider or
synthetic data. If the live request fails, a cached price file is accepted only
with one of two explicit provenance records:

- A current schema-1.0 `paisa.pipeline.run_v0_pipeline` record produced with
  that exact CSV. This existing path remains accepted unchanged.
- A complete, checksum-bound verified-legacy bundle produced by
  `paisa.legacy_cache.migrate_legacy_psx_cache`. A pre-schema CSV and its old
  metadata are not accepted directly.

Both paths bind the price bytes, real provider, daily frequency, row count, and
OGDC coverage and require positive raw-acquisition evidence with no OGDC
provider error. Source values in the CSV must agree with the provider. A CSV
alone, stale or incomplete metadata, relabelled fixture data, or mixed/blank
provenance is rejected.

### Migrate the verified pre-schema PSX cache

The migration command performs no live acquisition and never changes the
legacy files. Run it once with the exact source paths and the repository's
standard verified-cache destination:

```bash
python scripts/migrate_legacy_psx_cache.py \
  --legacy-prices \
  "<legacy-repository>/data/processed/psx_prices.csv" \
  --legacy-metadata \
  "<legacy-repository>/data/processed/pipeline_metadata.json" \
  --legacy-raw-dir \
  "<legacy-repository>/data/raw" \
  --output-dir \
  "data/external/verified_legacy_psx_cache"
```

By default, the command looks recursively below the inferred legacy
`data/raw/` directory for exact `<SYMBOL>_psx_dps_raw.csv` filenames. If one
exists for every requested symbol, it snapshots each source file, strictly
validates every raw row, rebuilds prices with the current cleaner, and requires
one-to-one row accounting plus an exact match with the legacy processed data.
Missing, partial, invalid, duplicate, or processed-only raw evidence is
rejected; it can never produce a verified-real bundle.

The immutable destination contains exactly the migrated `psx_prices.csv`, new
schema-1.0 `pipeline_metadata.json`, byte-preserved
`legacy_pipeline_metadata.json`, and `migration_manifest.json`. The manifest
records original and migrated SHA-256 values, coverage checks, timestamps,
migration strategy, evidence strength, and limitations. The producer remains
the distinct legacy migrator; it is never represented as a new pipeline or
live-provider run. An existing output directory is not overwritten.

After a successful migration, run the controlled real-cache workflow with:

```bash
python scripts/run_ogdc_vertical_slice.py \
  --cached-prices \
  "data/external/verified_legacy_psx_cache/psx_prices.csv" \
  --run-id \
  "VSL-20261020-OGDC-real-cache-02"
```

When the live request returns HTTP 404, that error remains in the run evidence.
The fallback is labelled `acquisition_mode="verified_legacy_psx_cache"`,
`live_fetch_succeeded=false`, `original_provider="psx_dps"`, and
`legacy_metadata_preserved=true`; it is not described as a successful live
fetch. Migration rejects missing evidence, invalid prices or volumes,
duplicate ticker/date rows, wrong or mixed providers, sample/synthetic/fixture/
mock/test sources, coverage mismatches, non-positive raw counts, and provider
errors.

The selected CSV and provenance metadata are first copied to an
invocation-private working snapshot before either validation or training, so
later changes to the source cache cannot change the run. On success, that exact
CSV and its `pipeline_metadata.json` are retained under the published bundle's
`input_snapshot/` directory as checksum-inventoried input evidence. For a
verified-legacy run, the snapshot also retains `legacy_pipeline_metadata.json`
and `migration_manifest.json`. A model-training run is published only after
all four backtests and persisted-evidence checks pass. The runner rechecks the
trainer's frozen price file and all four prediction-file checksums after
backtesting and immediately before publication. The complete training run also
has a sorted, exact-file SHA-256 inventory covering its frozen inputs, split
and data manifests, reports, model manifests, configurations, metrics,
predictions, and binaries; its source version fields and inventory are checked
before and after backtesting. Model IDs and prediction paths are derived from
the validated run manifest, so a caller summary cannot redirect backtesting to
external files. A callback therefore cannot change model evidence mid-run; a
handled late failure rolls back artifacts owned by that invocation.
Successful outputs are immutable under
`artifacts/vertical_slice/<vertical_slice_run_id>/`, and an existing run ID is
never overwritten. Each completed bundle retains the four full backtesting
trees under `experiments/<experiment_id>/`; aggregate metrics and curves are
rebuilt and reconciled from those persisted ledgers rather than trusted only
from in-memory results.

The common experiment allocates at most 50% of equity to OGDC. That same cap
also applies to buy-and-hold, so this is a 50%-allocated buy-and-hold benchmark
with the balance left in cash, not a fully invested benchmark. The model target
measures next-session close versus feature-date close, while simulated trades
execute at the next observed session's open and earn returns from that point
onward. Prediction quality and simulated portfolio profitability are therefore
reported separately.

The verified-legacy path establishes a strict internal evidence chain, not
independent market-data authenticity. Legacy raw captures have no provider
signature or retained HTTP-response capture, and a SHA-256 digest detects byte
changes but is not a digital signature. See the runbook for the full limits.

See [the OGDC vertical-slice runbook](docs/OGDC_VERTICAL_SLICE.md) for the exact
assumptions, fallback rules, artifact tree, report semantics, and limitations.
All outputs are academic simulations and are not financial advice.

## API endpoints

### Existing v0 API

The original v0 routes and field names remain unchanged. The `/health`
`data_dir` and `model_dir` values are intentionally safe logical labels rather
than host filesystem paths; readiness booleans, row counts, and available-stock
reporting are unchanged. In particular, `/stocks` and
`/stocks/{ticker}/prices` still depend on the generated
`data/processed/psx_prices.csv` compatibility file and return `404` when it is
absent.

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

`GET /health` returns `status`, the logical labels `data_dir: "legacy-v0-data"`
and `model_dir: "legacy-v0-models"`, and the `prices_ready`, `dataset_ready`,
and `model_ready` booleans. When the legacy price CSV exists, it also returns
`price_rows` and sorted `stocks`. It never returns repository, artifact, model,
or user-directory paths.

### Read-only v1 experiment-results API

The additive `/api/v1` API reads completed, immutable vertical-slice artifacts.
It does not train a model, rerun a backtest, recalculate business metrics, write
artifacts, perform real trading, or deserialize/serve a fitted model binary.
A completed vertical slice owns its historical-data snapshot; dashboards should
therefore request prices through that run rather than through a mutable processed
data path.

From the repository root, install and start it with the local real run as the
explicit default when that uncommitted artifact exists:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02

PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

`PAISA_VERTICAL_SLICE_ROOT` and `PAISA_MODEL_TRAINING_ROOT` default to the
values shown. `PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID` is optional, but
`GET /api/v1/leaderboard` returns `409` if it is unset instead of guessing a
run. Run-specific URLs do not require a default. Configured relative paths are
resolved from the repository root.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/vertical-slices` | List completed, checksum-valid vertical-slice runs with pagination. |
| `GET /api/v1/vertical-slices/{run_id}` | Return run identity, dates, models, experiments, and shared assumptions. |
| `GET /api/v1/vertical-slices/{run_id}/summary` | Return the persisted final summary. |
| `GET /api/v1/vertical-slices/{run_id}/stocks/{ticker}/prices` | Return historical daily prices from that run's checksum-verified input snapshot. |
| `GET /api/v1/vertical-slices/{run_id}/leaderboard` | Return the combined model/portfolio leaderboard. |
| `GET /api/v1/leaderboard` | Return the explicitly configured default run's leaderboard. |
| `GET /api/v1/vertical-slices/{run_id}/models` | Return validation and test model metrics. |
| `GET /api/v1/vertical-slices/{run_id}/portfolios` | Return strategy, cash-only, and buy-and-hold metrics. |
| `GET /api/v1/vertical-slices/{run_id}/equity-curves` | Return paginated daily equity rows. |
| `GET /api/v1/vertical-slices/{run_id}/drawdowns` | Return paginated daily drawdown rows. |
| `GET /api/v1/vertical-slices/{run_id}/source` | Return filtered input/source provenance. |
| `GET /api/v1/vertical-slices/{run_id}/experiments` | List experiment configurations and result summaries. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}` | Return one persisted configuration and result. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/trades` | Return paginated strategy trades. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/snapshots` | Return paginated strategy portfolio snapshots. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/benchmarks` | Return paginated cash-only and buy-and-hold snapshots and buy-and-hold trades. |

The run listing and large-table endpoints accept `limit` (default `200`,
maximum `1000`) and `offset` (default `0`) and return `total`, `limit`,
`offset`, and deterministically ordered `rows`. For example:

```bash
curl -s http://127.0.0.1:8000/api/v1/vertical-slices
curl -s http://127.0.0.1:8000/api/v1/leaderboard
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/stocks/OGDC/prices?limit=5&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/equity-curves?limit=25&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/trades?limit=25&offset=0"
```

All v1 responses use explicit Pydantic schemas and ISO dates. PKR amounts are
fixed-scale strings such as `"105987.04"`, preserving decimal meaning.
Undefined metrics are JSON `null`; `NaN` and infinity are never emitted. Every
applicable run response retains warnings and the `"Academic simulation; not
financial advice."` disclaimer. The API explicitly identifies `actual_class`
as evaluation-only and never uses it for a signal, execution, or accounting.

The historical-price endpoint has its own `limit` range (`1` through `5000`,
default `500`) and accepts `offset`, inclusive `start_date`, and inclusive
`end_date`. Date filters apply before pagination, `total` is the filtered count
before pagination, rows are chronological, prices are decimal strings, and
unavailable `high`/`low` values are JSON `null`. The React dashboard must use
this run-scoped route, not legacy `/stocks/{ticker}/prices`.

Run and experiment IDs accept only safe identifier characters. The repository
rejects traversal, symlink escape, non-`PASS` or incomplete runs, missing or
extra inventoried files, checksum mismatches, and invalid referenced training
runs. Clients choose registered IDs, never filenames or filesystem paths, and
v1 responses do not reveal absolute local paths. The price route selects only
the manifest-bound `input_snapshot` CSV; it never reads
`data/processed/psx_prices.csv` and never loads a model binary. See
[`docs/api_design.md`](docs/api_design.md) for response examples and error
semantics.

## React results dashboard

The dashboard under [`frontend/`](frontend/) presents the completed OGDC
vertical slice with React, TypeScript, and Plotly. It reads the PAISA API; it
does not train models, execute backtests or trades, recompute financial
metrics, read artifact files directly, or load fitted model binaries. Runtime
API failures are shown as errors and are never silently replaced with fixture
data. All results remain academic simulations and are not financial advice.

Node.js 22 LTS is recommended (22.12 or newer; Node.js 24 is also supported).
The current frontend has been exercised with Node.js 22.20.0. Install and
configure it from the repository root:

```bash
cd frontend
npm install
cp .env.example .env.local
```

The example environment file configures the browser client with:

```text
VITE_PAISA_API_BASE_URL=http://127.0.0.1:8000
```

Do not put filesystem paths in this variable; it must be the HTTP(S) base URL
of the PAISA API. Start the backend from the repository root in one terminal:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

The configured immutable vertical-slice and model-training artifact roots must
already contain that completed run. In a second terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`.

The Historical OGDC view uses the currently selected run and requests
`GET /api/v1/vertical-slices/{selected_run_id}/stocks/OGDC/prices`. It retrieves
all available rows through documented pages of at most `5000`; the backend
serves the selected run's immutable, checksum-verified input snapshot rather
than `data/processed/psx_prices.csv`. API failures remain visible and the
dashboard does not substitute fixture, sample, synthetic, or fake prices.

Frontend quality and production-build commands are:

```bash
cd frontend
npm run test
npm run build
npm run lint
```

The production build is written to the ignored `frontend/dist/` directory.
See [`frontend/README.md`](frontend/README.md) for the complete integrated-demo
workflow and frontend behavior.

## Folder structure

```text
PAISA/
├── data/
│   ├── raw/          # generated raw provider outputs
│   ├── processed/    # generated prices/features/dataset/metadata
│   ├── external/     # manually downloaded CSVs
│   └── metadata/     # stock universe
├── docs/
├── frontend/         # React + TypeScript read-only results dashboard
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
