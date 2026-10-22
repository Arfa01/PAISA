# Run PAISA from GitHub

These are the exact steps to run the current PAISA backend and read-only results
dashboard from a fresh clone.

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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
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

Article-level FinBERT inference is also an optional, separate runtime. Its
pinned weights are large and recorded as CC BY-NC-SA 4.0, so PAISA limits this
configuration to non-commercial academic use. Install it only when running the
sentiment stage:

```bash
python -m pip install -r requirements-sentiment.txt
```

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

The legacy `GET /stocks` and `GET /stocks/{ticker}/prices` routes read
`data/processed/psx_prices.csv`. They intentionally return `404` if that v0
pipeline output is absent; the immutable-run API below does not depend on it.

## 4a. Run the separate news-ingestion foundation

This workflow is independent of prices, model training, portfolios, API, and
dashboard state. It requests configured publisher RSS/Atom feed URLs, rejects
redirects, and never fetches linked article pages. Relative entry links are
resolved against the feed URL while their original values remain in raw
evidence. To acquire a configured feed:

```bash
python scripts/run_news_pipeline.py \
  --provider rss \
  --config configs/news/ogdc_sources.example.json \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

Or import a permitted local dataset, which will be labelled `imported`:

```bash
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /absolute/path/to/news.csv \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

After valid argument parsing, one structured result prints the immutable run ID
and locations. `FAIL` exits non-zero. Existing run IDs are never overwritten,
provider failures never fall back to fixtures, and generated
databases/artifacts remain outside Git. See
[`docs/NEWS_INGESTION.md`](docs/NEWS_INGESTION.md) before configuring a source.

## 4b. Run article-level FinBERT sentiment inference

This stage reads canonical normalized articles from one completed news run. It
does not fetch publishers, aggregate daily sentiment, join stock prices,
retrain a stock model, produce a trading signal, or update the dashboard.

The default model and tokenizer are both pinned to
`ProsusAI/finbert@0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`.
The upstream output indices are mapped explicitly as `0=positive`,
`1=negative`, and `2=neutral`; persisted PAISA probabilities use the canonical
order `negative`, `neutral`, `positive`.

Inspect all resource and device controls first:

```bash
python scripts/run_sentiment_inference.py --help
```

Then run a bounded CPU invocation against the accepted PR #12 news run:

```bash
python scripts/run_sentiment_inference.py \
  --source-run-id NEWS-20261022-131242-5ddffd2d \
  --database data/news/paisa_news.sqlite3 \
  --output-dir artifacts/sentiment_inference \
  --model-cache data/sentiment_model_cache \
  --device cpu \
  --batch-size 8 \
  --max-input-characters 4000 \
  --max-token-length 512 \
  --max-articles 500
```

The exact `SENT-...` run ID and body-free summary print on completion. The
destination is immutable: never reuse an existing run ID or directory.
Downloaded model files, SQLite state, and generated artifacts remain ignored by
Git. Public manifests contain portable cache/database labels, not absolute
private paths or credentials. A failed model download/load exits non-zero and
never silently selects another classifier.

The reviewed config, safetensors, tokenizer config, vocabulary, and
special-token files are SHA-256 verified before and after load. Current SQLite
schema `6` records their identity, each ordered input's terminal
generated/reused/error association, post-truncation token evidence, and both
recovery seals before it permits `PASS`.

If an earlier process was interrupted and the exact ID remains `RUNNING`, first
confirm no process still owns it, then retry that same ID with
`--discard-stale-running`. The option never applies to finalized runs; a new ID
cannot reuse output owned by the unfinished run.

The accepted PR #12 ingestion manifest and artifact pack record schema version
`3`; those historical artifacts predate the schema-4 sealing evidence and
cannot be passed off as current news verification evidence. Before the
retained sentiment execution, the working `data/news/paisa_news.sqlite3` also
used schema version `3`. The migration repaired only the exact recognized
legacy signature after validating every affected row, then applied schemas 4
and 5; all other drift fails closed. The retained reference database records
schema version `5`; current code advances a verified working copy to schema
version `6` before new sentiment inference. Its ignored, uncommitted
pre-migration copy is
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3`, verified at
schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check. Do not describe database compatibility as
retroactive artifact certification.

The retained final-code reference execution is
`SENT-20261022-172211-real0002`, sourced from
`NEWS-20261022-131242-5ddffd2d`. It completed with `PASS` on CPU in
`7.116756208240986` seconds against a preserved schema-3 database copy, which
migrated to schema version `5`. That is the retained reference-run schema;
current code advances a verified database to schema `6` for sentiment run
certification.
It requested and scored `30` unique canonical articles, skipped `0`, failed
`0`, and recorded `any_articles_failed=false`. One scored article was linked
to OGDC. The label counts were `negative=11`, `neutral=9`, and `positive=10`;
`17` inputs were truncated, `0` used title only, and all `30` carried `en-us`
language evidence. The canonical availability range was
`2026-10-22T11:52:32Z` through `2026-10-22T13:12:43.088883Z`.

That run used the pinned model and tokenizer revision above under Python
`3.13.7`, PyTorch `2.10.0`, Transformers `5.20.2`, Tokenizers `0.23.3`,
Hugging Face Hub `1.32.0`, and Safetensors `0.8.0`. Its generated artifact
directory is
`artifacts/sentiment_inference/SENT-20261022-172211-real0002`; it is not
committed.

Sentiment outputs were generated, but classification accuracy was not evaluated in this run.
The retained sentiment evidence covers October 19--22, 2026, whereas retained
OGDC prices end on September 15, 2026, so there are **zero overlapping dates**.
This run cannot support a combined price-plus-sentiment experiment or portfolio
conclusion. Obtain an overlapping historical news corpus before attempting that
research stage.
This short snapshot is insufficient for stock-model integration. See
[`docs/FINBERT_SENTIMENT.md`](docs/FINBERT_SENTIMENT.md),
[`docs/SENTIMENT_DATA_CONTRACT.md`](docs/SENTIMENT_DATA_CONTRACT.md), and
[`docs/SENTIMENT_TIME_ALIGNMENT.md`](docs/SENTIMENT_TIME_ALIGNMENT.md).

## 5. Run tests

```bash
python -m pytest
```

Expected: all collected tests pass. The count will grow as components are added.

## 6. Start the API

Activate the environment and set the v1 artifact roots before starting the
server. The first two variables below are their repository-relative defaults.
The third variable selects the only run that the unqualified v1 leaderboard
may use:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

The local example expects the completed run at
`artifacts/vertical_slice/VSL-20261020-OGDC-main-02` and its referenced training
run below `artifacts/model_training`. These generated artifacts are deliberately
not committed, so generate them with the vertical-slice workflow or copy the
complete immutable artifact roots to the configured locations before using the
v1 endpoints. Both roots may instead be absolute paths. Relative paths resolve
from the repository root, and resolved runs must stay within those roots.

`PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID` may be omitted when only explicit
`/vertical-slices/{run_id}/...` URLs are used. If it is absent,
`GET /api/v1/leaderboard` returns `409`; it never chooses whichever directory
happens to be newest or first.

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

These legacy v0 routes and their response shapes are unchanged. The two stock
routes shown above require `data/processed/psx_prices.csv`.

## 8. Open the read-only v1 result API

Use `VSL-20261020-OGDC-main-02` and its first experiment as the local real-run
example:

```text
http://127.0.0.1:8000/api/v1/vertical-slices
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/summary
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/stocks/OGDC/prices?limit=5&offset=0
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/leaderboard
http://127.0.0.1:8000/api/v1/leaderboard
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/models
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/portfolios
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/equity-curves?limit=25&offset=0
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/drawdowns?limit=25&offset=0
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/source
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/trades?limit=25&offset=0
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/snapshots?limit=25&offset=0
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/benchmarks?limit=25&offset=0
```

Equivalent command-line checks include:

```bash
curl -s http://127.0.0.1:8000/api/v1/vertical-slices
curl -s http://127.0.0.1:8000/api/v1/leaderboard
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/stocks/OGDC/prices?limit=5&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/equity-curves?limit=25&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/trades?limit=25&offset=0"
```

Abridged representative fields from
`equity-curves?limit=1&offset=0` are:

```json
{
  "schema_version": "1.0",
  "run_id": "VSL-20261020-OGDC-main-02",
  "source_mode": "verified_legacy_psx_cache",
  "data_date_range": {"start": "2021-09-16", "end": "2026-09-15"},
  "warnings": [
    "The prediction target is next-session close-to-close direction; simulated portfolio returns begin at the next-session open. These are distinct measurements."
  ],
  "limitations": [
    "actual_class is evaluation-only and is never used for signals, execution, or portfolio accounting."
  ],
  "disclaimer": "Academic simulation; not financial advice.",
  "total": 1110,
  "limit": 1,
  "offset": 0,
  "rows": [
    {
      "date": "2025-12-15",
      "series_id": "BUY-AND-HOLD",
      "portfolio_type": "buy_and_hold",
      "model_id": null,
      "portfolio_id": "VSL-20261020-OGDC-main-02-m-001-buy-and-hold",
      "cash": "100000.00",
      "shares": 0,
      "market_price": "278.67",
      "total_equity": "100000.00",
      "drawdown": 0.0
    }
  ]
}
```

`limit` defaults to `200`, must be from `1` through `1000`, and `offset`
defaults to `0`. Pagination applies to the run listing, equity curves,
drawdowns, strategy trades, strategy snapshots, and each benchmark series.
Rows are stable and deterministically ordered, and every paginated object
reports its full `total`.

Historical prices are the exception: their `limit` defaults to `500` and may be
from `1` through `5000`. `offset` is non-negative; optional `start_date` and
`end_date` are inclusive ISO dates. Filtering happens before pagination, so
`total` reports all matching rows before `limit` and `offset` are applied. The
dashboard must use the selected run's v1 price URL rather than the legacy stock
route.

Dates are ISO strings. PKR money uses fixed two-decimal strings such as
`"100000.00"`; historical price strings preserve exact source precision with at
least two fractional digits. Neither is converted through binary floating
point. Undefined metrics such as a cash-only Sharpe ratio are `null`; the
service never emits `NaN` or infinity.

Before returning v1 data, PAISA requires a matching safe run ID, a completed
`PASS` manifest, an exact artifact inventory, valid SHA-256 checksums, and a
valid referenced model-training integrity root. Traversal, symlink escape,
unsafe experiment IDs, arbitrary filenames, incomplete/failed runs, or changed
artifacts are rejected. The v1 routes expose no absolute local paths and never
load or serve joblib files. They do not retrain, backtest, write artifacts,
perform real trading, or turn `actual_class` into a decision. Every v1 run
response retains the academic-simulation disclaimer.

For historical prices, the repository identifies the exact CSV through the
completed run manifest and its checksum-bound `input_snapshot`. It never reads
`data/processed/psx_prices.csv`, accepts a client filename/path, or loads a model
binary for this request.

See [`docs/api_design.md`](docs/api_design.md) for the complete endpoint and
error contract.

## 9. Install and configure the React dashboard

The dashboard recommends Node.js 22 LTS (22.12 or newer) and also supports
Node.js 24; Node.js 22.20.0 is the current tested version. From the repository
root, install the frontend dependencies and copy
the committed example configuration:

```bash
cd frontend
npm ci
cp .env.example .env.local
```

`frontend/.env.example` contains:

```text
VITE_PAISA_API_BASE_URL=http://127.0.0.1:8000
```

`VITE_PAISA_API_BASE_URL` must be the PAISA HTTP(S) API base URL. Do not use an
absolute local filesystem path. Local `.env` files are ignored by Git; the
example file remains committed.

The dashboard consumes completed immutable v1 results, including OGDC prices
from the selected run's v1 price endpoint. It does not train, predict, run a
backtest, calculate portfolio metrics, read artifact files, deserialize model
binaries, or execute trades. An API failure remains a visible error; the
running application never silently falls back to fixture data. Test fixtures
are limited to automated tests.

## 10. Run the integrated dashboard

Before starting the applications, make sure the configured immutable artifact
roots contain `VSL-20261020-OGDC-main-02` and its referenced model-training run.

The Historical OGDC view calls
`GET /api/v1/vertical-slices/{selected_run_id}/stocks/OGDC/prices`, using the
currently selected run ID. It requests every available page with the documented
maximum `limit=5000`; the backend reads the immutable run's checksum-verified
`input_snapshot` through the API, not `data/processed/psx_prices.csv`. A request
failure remains visible in the view, with no fixture, sample, synthetic, or fake
price fallback.

From the repository root, start the backend in the first terminal:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

Start the frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Open the dashboard at:

```text
http://127.0.0.1:5173
```

The development server uses the configured API at
`http://127.0.0.1:8000`. The dashboard keeps the API disclaimer visible and
presents predictions, trades, benchmarks, and portfolio results only as an
academic simulation, not financial advice.

In Vite development, React Strict Mode may start an Effect request, cancel it
during its development-only cleanup check, and then issue the successful
request. The cancelled entry can appear in browser Network tools; it is
intentionally ignored and is not an application failure or an infinite retry
loop. Production builds do not perform this development-only Effect replay.

## 11. Test, lint, and build the frontend

Run each command from `frontend/`:

```bash
npm run test
npm run build
npm run lint
VITE_PAISA_API_BASE_URL=http://127.0.0.1:8000 npm run build
npm run preview -- --host 127.0.0.1 --port 4173
```

`npm run build` produces the ignored `frontend/dist/` directory. Do not commit
`node_modules/`, `dist/`, local environment files, real model artifacts, or
real vertical-slice artifacts. See [`frontend/README.md`](frontend/README.md)
for the frontend-specific runbook.

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
[`docs/model_training_stage.md`](docs/model_training_stage.md) for the optional
installation, chronological sequence rules, immutable model-plus-scaler bundle,
shared prediction contract, and data-provider details.

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
fields; training diagnostics remain model artifacts. A canonical multi-symbol
price file is supported directly because the configured symbol is selected
before market-session validation.

All backtest output is an academic simulation and is not financial advice.

## Notes

- Generated CSV/model files are ignored by Git and should be regenerated locally.
- v0 is an academic prototype and does not give buy/sell advice.
- The current PSX provider may not return high/low values; it currently works with open, close, and volume.
