# PAISA API design

Base URL during local development:

```text
http://127.0.0.1:8000
```

PAISA exposes two additive interfaces:

- The original unversioned v0 routes provide pipeline and baseline-prediction
  compatibility. Their routes and response shapes are unchanged.
- `/api/v1` provides read-only access to completed immutable model,
  experiment, portfolio, benchmark, vertical-slice, and run-owned historical
  price results.

Every prediction, signal, trade, portfolio value, benchmark, and return shown
by either interface is an academic simulation and is not financial advice.

## Install, configure, and start

From the repository root:

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

Open the generated OpenAPI UI at
`http://127.0.0.1:8000/docs`.

The local real-run example is
`artifacts/vertical_slice/VSL-20261020-OGDC-main-02`. It also requires the
referenced immutable training run below `artifacts/model_training`. Runtime
artifacts are intentionally ignored by Git and must not be committed.

### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PAISA_VERTICAL_SLICE_ROOT` | `artifacts/vertical_slice` | Root containing immutable vertical-slice run directories. |
| `PAISA_MODEL_TRAINING_ROOT` | `artifacts/model_training` | Root containing training runs referenced by vertical-slice manifests. |
| `PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID` | unset | Optional completed run used only by `GET /api/v1/leaderboard`. |

Relative roots resolve from the repository root. Absolute roots are supported,
but a run and every resolved artifact must remain below the configured root.
Repository caches are partitioned by the complete current root/default-run
configuration, so changing an environment variable does not reuse reads from
the previous roots.

The default run is deliberately explicit. When
`PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID` is unset, `GET /api/v1/leaderboard`
returns HTTP `409`; it does not infer a run from directory order or timestamps.
All URLs containing an explicit `{run_id}` remain available without a default.

## v1 endpoint catalog

All routes below use `GET` and are beneath `/api/v1`.

| Path | Result |
|---|---|
| `/vertical-slices` | Paginated completed, checksum-valid `PASS` runs. |
| `/vertical-slices/{run_id}` | One run's identity, date range, registered models, experiments, and shared assumptions. |
| `/vertical-slices/{run_id}/summary` | Persisted final model/portfolio summary and interpretation. |
| `/vertical-slices/{run_id}/stocks/{ticker}/prices` | Chronological daily prices from the run's immutable input snapshot. |
| `/vertical-slices/{run_id}/leaderboard` | Combined model and strategy-portfolio leaderboard for that run. |
| `/leaderboard` | Leaderboard for the explicitly configured default run. |
| `/vertical-slices/{run_id}/models` | Algorithms, registered model IDs, validation/test metrics, and split/class counts. |
| `/vertical-slices/{run_id}/portfolios` | Strategy, cash-only, and buy-and-hold portfolio metrics. |
| `/vertical-slices/{run_id}/equity-curves` | Paginated daily strategy and benchmark equity curves. |
| `/vertical-slices/{run_id}/drawdowns` | Paginated daily strategy and benchmark drawdowns. |
| `/vertical-slices/{run_id}/source` | Filtered provider, acquisition, checksum, pipeline, and migration evidence. |
| `/vertical-slices/{run_id}/experiments` | Experiment IDs, model bindings, date windows, and result summaries. |
| `/vertical-slices/{run_id}/experiments/{experiment_id}` | One persisted experiment configuration and result. |
| `/vertical-slices/{run_id}/experiments/{experiment_id}/trades` | Paginated strategy trade ledger. |
| `/vertical-slices/{run_id}/experiments/{experiment_id}/snapshots` | Paginated strategy portfolio ledger. |
| `/vertical-slices/{run_id}/experiments/{experiment_id}/benchmarks` | Paginated cash-only snapshots, buy-and-hold snapshots, and buy-and-hold trades. |

The API never accepts a client-provided artifact filename or filesystem path.
`run_id`, `experiment_id`, and `ticker` are identifiers, not path fragments.

## Pagination and deterministic ordering

These endpoints accept query parameters `limit` and `offset`:

- `/vertical-slices`
- `/equity-curves`
- `/drawdowns`
- `/trades`
- `/snapshots`
- `/benchmarks`

`limit` defaults to `200`, has a minimum of `1`, and has a maximum of `1000`.
`offset` defaults to `0` and cannot be negative. Invalid query values return
FastAPI's HTTP `422` validation response. Every paginated collection includes
`total`, `limit`, `offset`, and `rows`; each of the three benchmark collections
has its own count and page fields.

`/vertical-slices/{run_id}/stocks/{ticker}/prices` has a route-specific default
`limit` of `500` and maximum of `5000`. Its `offset` is also non-negative. It
accepts optional inclusive ISO `start_date` and `end_date`; filters apply first,
then `total` is calculated, and only then are `offset` and `limit` applied.

Ordering is stable:

- run and experiment listings sort by identifier;
- model and portfolio tables sort by their declared stable identities;
- leaderboard rows retain the producer's persisted deterministic validation
  ranking;
- curves and drawdowns sort by date, series ID, then portfolio ID;
- historical prices sort chronologically and reject duplicate ticker/date rows;
- trades sort by execution date, then trade ID; and
- snapshots sort by date, then portfolio ID.

## Response representation

The v1 routes use explicit Pydantic response models. Run-scoped responses
include this context:

- `schema_version` and `run_id`;
- `source_mode` and the ISO `data_date_range`;
- persisted `warnings` plus API-enforced `limitations` where applicable; and
- `disclaimer: "Academic simulation; not financial advice."`.

Run details add ISO evaluation dates, registered model IDs and algorithms,
experiment assumptions, validation/test metrics, portfolio metrics, and
cash-only and buy-and-hold results as appropriate for the route.

Calendar dates use `YYYY-MM-DD`; timestamps are ISO 8601 strings. Exact PKR
money values are JSON strings with two fractional digits, for example
`"105987.04"`. Historical prices preserve the snapshot's exact decimal
precision and display at least two fractional digits. Neither representation is
converted through binary floating point.
Undefined metrics are JSON `null`—for example, a cash-only Sharpe ratio—not
`NaN`, positive infinity, or negative infinity.

Historical-price rows contain `date`, `open`, nullable `high`, nullable `low`,
`close`, integer `volume`, and non-empty `source`. Price values are decimal
strings created directly from the snapshot text rather than binary floats. The
response `data_date_range` describes the complete validated ticker snapshot;
`total` describes date-filter matches before pagination. Open and close must be
positive, supplied high/low values must be positive and finite, and volume must
be non-negative.

`actual_class` is evaluation-only and is never used for signals, execution, or
portfolio accounting. The API carries that limitation in responses rather
than presenting evaluation labels as an input to a simulated decision.

### Run-scoped historical prices

```bash
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/stocks/OGDC/prices?limit=5&offset=0"
```

An abridged response is:

```json
{
  "schema_version": "1.0",
  "run_id": "VSL-20261020-OGDC-main-02",
  "ticker": "OGDC",
  "source_mode": "verified_legacy_psx_cache",
  "data_date_range": {"start": "2021-09-16", "end": "2026-09-15"},
  "total": 1239,
  "limit": 5,
  "offset": 0,
  "rows": [
    {
      "date": "2021-09-16",
      "open": "88.09",
      "high": null,
      "low": null,
      "close": "89.17",
      "volume": 4201784,
      "source": "psx_dps"
    }
  ],
  "warnings": [
    "The prediction target is next-session close-to-close direction; simulated portfolio returns begin at the next-session open. These are distinct measurements."
  ],
  "disclaimer": "Academic simulation; not financial advice."
}
```

The completed vertical slice owns this snapshot. A dashboard must address its
selected `run_id`; it must not substitute the legacy mutable stock route.

### Representative completed-run listing

```bash
curl -s http://127.0.0.1:8000/api/v1/vertical-slices
```

A response containing the local real run includes:

```json
{
  "schema_version": "1.0",
  "total": 1,
  "limit": 200,
  "offset": 0,
  "rows": [
    {
      "schema_version": "1.0",
      "run_id": "VSL-20261020-OGDC-main-02",
      "status": "PASS",
      "symbol": "OGDC",
      "completed_at": "2026-10-20T19:18:20Z",
      "source_mode": "verified_legacy_psx_cache",
      "data_date_range": {
        "start": "2021-09-16",
        "end": "2026-09-15"
      },
      "model_training_run_id": "TRN-20261020-191815-624662-nogit",
      "model_count": 4,
      "experiment_count": 4
    }
  ],
  "disclaimer": "Academic simulation; not financial advice."
}
```

The `total` above reflects a root containing only that run; other valid local
runs increase it. Failed, incomplete, linked, malformed, or checksum-invalid
directories are not advertised by the listing.

### Explicit and default leaderboards

```bash
curl -s http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/leaderboard
curl -s http://127.0.0.1:8000/api/v1/leaderboard
```

Both commands address the same run when the example default is configured. An
abridged leaderboard row is:

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
  "total": 4,
  "rows": [
    {
      "model_id": "M-001",
      "registered_model_id": "M-001-6ce88dc384de975e-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9",
      "algorithm": "random_forest",
      "validation_balanced_accuracy": 0.5302041802585478,
      "validation_f1": 0.4585987261146497,
      "test_accuracy": 0.5163043478260869,
      "test_balanced_accuracy": 0.5122887864823349,
      "test_f1": 0.22608695652173913,
      "portfolio_id": "VSL-20261020-OGDC-main-02-m-001-strategy",
      "ending_portfolio_value": "99298.24",
      "total_return": -0.0070176,
      "maximum_drawdown": 0.02199656516669476,
      "volatility": 0.028772855911356206,
      "sharpe_ratio": -0.3208542436221641,
      "number_of_trades": 2,
      "fees_paid": "99.06",
      "result_versus_buy_and_hold": -0.0694965
    }
  ]
}
```

The snippet is abbreviated: a live response contains all persisted warnings,
limitations, and all four rows.

### Pagination, Decimal strings, and `null`

```bash
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/equity-curves?limit=1&offset=0"
curl -s http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/portfolios
```

Representative pagination fields and the first deterministically ordered
equity row are:

```json
{
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

The portfolio table represents an undefined cash-only risk-adjusted metric as
JSON `null` while retaining exact money:

```json
{
  "portfolio_type": "cash_only",
  "model_id": "CASH-ONLY",
  "portfolio_id": "VSL-20261020-OGDC-main-02-m-001-cash-only",
  "ending_portfolio_value": "100000.00",
  "volatility": 0.0,
  "sharpe_ratio": null,
  "fees_paid": "0.00"
}
```

### Experiment ledger examples

```bash
curl -s http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments
curl -s http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/trades?limit=25&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/snapshots?limit=25&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/benchmarks?limit=25&offset=0"
```

Trade records keep `signal_date` distinct from `execution_date`. Benchmark
responses contain three separately paginated objects: `cash_only`,
`buy_and_hold`, and `buy_and_hold_trades`.

## Error contract

| Condition | Status | Behavior |
|---|---:|---|
| Malformed `run_id`, `experiment_id`, or `ticker`, including traversal text | `400` | Reject before constructing an artifact path. |
| Historical `start_date` later than `end_date` | `400` | Reject the inverted range. |
| Unknown safe run, experiment ID, or snapshot ticker | `404` | Return a clear `detail` message. |
| Default leaderboard requested without a configured default | `409` | Do not guess a run. |
| Incomplete/non-`PASS` run, invalid snapshot values, or integrity failure | `409` | Return no result data from that run. |
| Invalid ISO query date or pagination bound | `422` | Standard FastAPI request-validation response. |

Malformed identifiers that reach the endpoint return `400`. Encoded traversal
containing `/` may instead fail route matching and return `404`; neither case is
allowed to select an artifact.

For example, without `PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID`:

```json
{
  "detail": "PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID is not configured"
}
```

## Read-only and integrity guarantees

The v1 artifact repository:

- accepts only identifiers containing letters, digits, `.`, `_`, and `-`,
  starting with a letter or digit;
- resolves configured roots once per configuration and requires every selected
  run and training run to stay beneath its root;
- rejects symlinked run directories, symlinks anywhere in an artifact tree,
  special files, path traversal, and root escape;
- requires matching schema/run identities, a `PASS` run and final summary, an
  exact sorted artifact inventory, and valid SHA-256 values for every file;
- validates the referenced model-training run manifest, source provenance,
  exact inventory, model bindings, and prediction checksums transitively;
- reads only known JSON/CSV result artifacts and filters source metadata so
  absolute local paths are never returned;
- selects historical prices only by matching the run manifest's input checksum
  to exactly one inventoried `input_snapshot` CSV, then verifies the bytes read;
- never uses `data/processed/psx_prices.csv` for a v1 historical-price request;
- never deserializes or serves joblib/Keras/XGBoost model binaries;
- never modifies a model, experiment, portfolio, benchmark, or manifest;
- never retrains, predicts, reruns a backtest, or recomputes business metrics
  during an HTTP request; and
- never performs real trading or labels a simulated result as financial advice.

Validated reads may be cached. A cached run is reused only while both the
vertical-slice and referenced training artifact-tree signatures remain
unchanged. A configured-root/default-run change selects a separate repository
cache entry.

## Legacy v0 endpoints

### `GET /health`

Confirms whether the v0 data and model artifacts are ready without exposing the
host filesystem. Its HTTP `200` response uses the explicit `HealthResponse`
schema:

```json
{
  "status": "ok",
  "data_dir": "legacy-v0-data",
  "model_dir": "legacy-v0-models",
  "prices_ready": true,
  "dataset_ready": true,
  "model_ready": true,
  "price_rows": 1042,
  "stocks": ["HBL", "OGDC"]
}
```

`data_dir` and `model_dir` are compatibility keys containing stable logical
labels, never absolute or shortened local paths. `price_rows` and `stocks` are
omitted when the legacy price CSV is absent. Errors use FastAPI's generic
server response and do not echo local exception or path details.

### `GET /pipeline/status`

Returns the latest v0 pipeline metadata: provider, requested symbols, row
counts, errors, date range, and metrics.

### `GET /stocks`

Returns available tickers from `data/processed/psx_prices.csv`. This compatibility
route returns `404` when that generated file is absent.

### `GET /stocks/{ticker}/prices`

Returns standardized historical price rows from
`data/processed/psx_prices.csv`. Its response and most-recent-row behavior are
intentionally unchanged; dashboards backed by completed runs should use the v1
run-scoped endpoint instead.

### `GET /stocks/{ticker}/features`

Returns engineered feature rows.

### `GET /stocks/{ticker}/dataset`

Returns complete rows used for supervised model training.

### `GET /models/baseline/metrics`

Returns baseline RandomForest metrics.

### `GET /predict/{ticker}`

Returns a v0 next-trading-day up/down prediction using the latest complete
feature row. This compatibility endpoint does not issue a buy/sell
recommendation. Its existing response shape is unchanged.
