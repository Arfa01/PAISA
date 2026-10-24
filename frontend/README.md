# PAISA results dashboard

This React, TypeScript, Vite, and Plotly application presents completed PAISA
OGDC vertical-slice results from the real PAISA API.

It also exposes the independently loaded Unified Six-Model Comparison view for
completed, checksum-valid M-001 through M-006 comparison packs. The view reads
authoritative API values and never selects a cross-criterion model, recomputes
metrics, or loads fitted model binaries.

It is a read-only presentation layer. It does not train models, create
predictions, execute backtests or trades, recompute backend financial metrics,
read artifacts from the filesystem, or load model binaries. API errors and
empty responses remain visible; the running dashboard never silently replaces
them with fixture data. All predictions, trades, benchmarks, and returns are
academic simulations and are not financial advice.

## Requirements

- Node.js 22 LTS is recommended (22.12 or newer). Node.js 24 is also supported;
  the current tested version is Node.js 22.20.0.
- The Python environment and backend dependencies documented in the repository
  [`README.md`](../README.md).
- A completed immutable vertical-slice run and its referenced model-training
  run under the configured artifact roots, including its checksum-verified
  input snapshot.

## Install

From the repository root:

```bash
cd frontend
npm install
```

`node_modules/` is local generated content and must not be committed.

## Configure the API

Create a local Vite environment file from the committed example:

```bash
cd frontend
cp .env.example .env.local
```

The example value is:

```text
VITE_PAISA_API_BASE_URL=http://127.0.0.1:8000
```

The value must be an HTTP(S) base URL and must not be an absolute filesystem
path. Restart the Vite development server after changing it. Local environment
files are ignored by Git; `.env.example` is intentionally committed.

## Prepare backend data

The immutable dashboard views read completed v1 artifacts below
`artifacts/vertical_slice` and `artifacts/model_training`. The example commands
below select `VSL-20261020-OGDC-main-02`; that run and its referenced training
run must already exist at those roots. Generated artifacts are intentionally
not committed to Git.

The Historical OGDC view uses the currently selected run and requests:

```text
GET /api/v1/vertical-slices/{selected_run_id}/stocks/OGDC/prices
```

It retrieves all available rows using the endpoint's documented maximum
`limit=5000` and continues through additional pages when necessary. The backend
reads the selected run's immutable, checksum-verified input snapshot through
the API; the frontend never reads `data/processed` files directly. If the API
request fails, the historical view shows the error and does not substitute
fixture, sample, synthetic, or fake prices.

## Run the integrated demo

Start the backend from the repository root in the first terminal:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
export PAISA_UNIFIED_COMPARISON_ROOT=artifacts/model_comparisons
export PAISA_DEFAULT_UNIFIED_COMPARISON_RUN_ID=CMP-20261024-OGDC-price-only-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

This exposes the API at `http://127.0.0.1:8000` and its OpenAPI UI at
`http://127.0.0.1:8000/docs`.

Start the frontend in a second terminal:

```bash
cd frontend
npm run dev
```

Open:

```text
http://127.0.0.1:5173
```

The application loads the vertical-slice run list for the existing views. The
Unified comparison navigation item loads its own completed comparison inventory
only when opened, uses the API-declared default when configured, and otherwise
uses the first deterministic completed row. It remains available when the
vertical-slice inventory is empty or unavailable. A configured default that is
missing or fails integrity checks is shown as an error rather than silently
replaced. The browser never chooses a run by reading local directories and does
not use test fixtures as a runtime fallback.

The unified view reads these v1 resources:

```text
GET /api/v1/unified-comparisons
GET /api/v1/unified-comparisons/{comparison_id}
GET /api/v1/unified-comparisons/{comparison_id}/sources
GET /api/v1/unified-comparisons/{comparison_id}/summary
GET /api/v1/unified-comparisons/{comparison_id}/prediction-metrics
GET /api/v1/unified-comparisons/{comparison_id}/validation-metrics
GET /api/v1/unified-comparisons/{comparison_id}/portfolio-metrics
GET /api/v1/unified-comparisons/{comparison_id}/prediction-rankings
GET /api/v1/unified-comparisons/{comparison_id}/portfolio-rankings
GET /api/v1/unified-comparisons/{comparison_id}/common-date-contract
GET /api/v1/unified-comparisons/{comparison_id}/confusion-matrices
```

The dashboard retains the exact term **shared project holdout**. It presents
validation selection, shared-holdout prediction metrics, and simulated
portfolio outcomes separately because better classification performance does
not necessarily produce better portfolio performance.

## Frontend commands

Run these commands from `frontend/`:

```bash
# Start the development server on http://127.0.0.1:5173
npm run dev

# Run the Vitest and React Testing Library suite once
npm run test

# Type-check and create the production bundle in dist/
npm run build

# Serve the production bundle on http://127.0.0.1:4173
npm run preview -- --host 127.0.0.1 --port 4173

# Run ESLint with warnings treated as failures
npm run lint
```

The generated `dist/` and `coverage/` directories are ignored and must not be
committed.

## Runtime behavior

- The dashboard uses the API by default; there is no implicit fixture mode.
- Requests have timeouts, structured error handling, and cancellation or stale
  response protection where applicable.
- In Vite development, React Strict Mode intentionally replays Effect setup and
  cleanup. An initial request may therefore appear as cancelled in browser
  Network tools before the subsequent request completes successfully. Cleanup
  aborts that stale request; it is ignored rather than shown or logged as an
  application failure, and it does not create an automatic retry loop.
- Paginated API collections are fetched through their registered run and
  experiment identifiers, never client-supplied filesystem paths.
- Exact PKR values remain fixed-scale strings for display. Undefined metrics
  render as `N/A`, never `NaN` or infinity.
- Unified-comparison responses must contain M-001 through M-006 exactly once,
  the two controlled portfolio benchmarks, a non-combined outcome contract,
  and the exact academic disclaimer before they are rendered.
- Model-selection, simulated portfolio, lowest-drawdown, and buy-and-hold
  results stay visibly separate. Test accuracy is not used to relabel the
  validation-selected prediction model.
- Warnings, limitations, provenance context, and the API-provided academic
  disclaimer remain visible.

Automated tests may use mock responses to exercise UI states. Those fixtures
are isolated to the test environment and do not replace failed production or
development API requests.
