# PAISA Team Handoff

This handoff follows the approved proposal responsibilities. All current and
future predictions, signals, trades, and returns are academic simulations and
are not financial advice.

## Approved ownership

### Arfa

- Stock-market data collection and integration.
- Market/news feature engineering, dataset preparation, and label generation.
- Model training, controlled performance evaluation, and comparative reports.
- Web-interface design, development, and visualization.
- Documentation and reporting.

### Waqas

- Financial-news collection and integration.
- NLP, company/entity extraction and linking.
- Knowledge-graph and relationship modelling.
- Machine-learning model development.
- Backend API design, development, and integration.
- Documentation and reporting.

### Shared

System tests, integration review, final improvements, supervisor evidence, and
documentation are shared. The ML boundary is deliberately collaborative: Waqas
develops the news/NLP and new model components; Arfa owns dataset integration,
training, evaluation, and result presentation; both review temporal leakage and
contracts.

## Checkout and environment

Work on a task branch, never directly on `main`:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
PAISA_TASK_BRANCH="feature/news-ingestion"
git switch -c "$PAISA_TASK_BRANCH"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

cd frontend
npm install
cp .env.example .env.local
cd ..
```

The bare `python` command is available after activating `.venv`; on the current
machine it is not available outside that environment.

## Current component entry points

| Component | Primary entry points |
|---|---|
| Providers and legacy data pipeline | `src/paisa/providers/`, `src/paisa/pipeline.py`, `src/paisa/preprocessing.py`, `src/paisa/features.py`, `scripts/run_v0_pipeline.py` |
| Experiment data and targets | `src/paisa/experiments/training_features.py`, `src/paisa/experiments/training_targets.py`, `src/paisa/splits.py` |
| Model training and evaluation | `src/paisa/experiments/trainer.py`, `src/paisa/experiments/evaluation.py`, `src/paisa/experiments/artifacts.py`, `src/paisa/experiments/registry.py`, `scripts/train_ogdc_models.py` |
| Optional LSTM | `src/paisa/experiments/lstm.py`, `notebooks/PAISA_OGDC_LSTM_Training.ipynb` |
| Decision policy and portfolio | `src/paisa/trading/policy.py`, `src/paisa/trading/execution.py`, `src/paisa/trading/portfolio.py`, `src/paisa/trading/models.py` |
| Backtesting and benchmarks | `src/paisa/backtesting/engine.py`, `src/paisa/backtesting/metrics.py`, `src/paisa/backtesting/benchmarks.py`, `src/paisa/backtesting/reports.py`, `scripts/run_backtest.py` |
| End-to-end orchestration | `src/paisa/vertical_slice.py`, `scripts/run_ogdc_vertical_slice.py` |
| Legacy-cache migration | `src/paisa/legacy_cache.py`, `scripts/migrate_legacy_psx_cache.py` |
| FastAPI | `src/paisa/api/main.py`, `src/paisa/api/router_v1.py`, `src/paisa/api/schemas.py`, `src/paisa/api/artifact_repository.py` |
| React dashboard | `frontend/src/App.tsx`, `frontend/src/components/`, `frontend/src/api/client.ts`, `frontend/src/api/schemas.ts` |
| Tests | `tests/`, `frontend/src/**/*.test.ts`, `frontend/src/**/*.test.tsx` |

New news, sentiment, entity, and relationship modules should be additive under
`src/paisa/` and should expose explicit artifact contracts before the API or UI
depends on them. Do not place NLP logic in API handlers or React components.

## Retained artifacts

The supervisor run is generated local content and must not be committed:

```text
artifacts/vertical_slice/VSL-20261020-OGDC-main-02/
artifacts/model_training/TRN-20261020-191815-624662-nogit/
```

Important evidence inside the vertical-slice directory:

```text
manifest.json
final_summary.json
model_metrics.csv
portfolio_metrics.csv
combined_leaderboard.csv
daily_equity_curves.csv
model_training_run_reference.json
input_snapshot/
experiment_configs/
experiments/<experiment_id>/
```

Other generated roots are `runs/<experiment_id>/` for standalone backtests and
`artifacts/lstm/<model_id>/` for optional LSTM registration. Never overwrite an
existing model, experiment, portfolio, or run ID. Do not commit datasets, model
binaries, run artifacts, `frontend/dist/`, local environments, or secrets.

## Run and inspect the current demo

Terminal 1, from the repository root:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Open:

- Swagger/OpenAPI: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:5173`
- Default-run leaderboard: `http://127.0.0.1:8000/api/v1/leaderboard`
- Run summary: `http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/summary`
- Source provenance: `http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/source`

The versioned API surface is:

```text
GET /api/v1/vertical-slices
GET /api/v1/vertical-slices/{run_id}
GET /api/v1/vertical-slices/{run_id}/summary
GET /api/v1/vertical-slices/{run_id}/stocks/{ticker}/prices
GET /api/v1/vertical-slices/{run_id}/leaderboard
GET /api/v1/leaderboard
GET /api/v1/vertical-slices/{run_id}/models
GET /api/v1/vertical-slices/{run_id}/portfolios
GET /api/v1/vertical-slices/{run_id}/equity-curves
GET /api/v1/vertical-slices/{run_id}/drawdowns
GET /api/v1/vertical-slices/{run_id}/source
GET /api/v1/vertical-slices/{run_id}/experiments
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/trades
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/snapshots
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/benchmarks
```

The compatibility API remains `/health`, `/pipeline/status`, `/stocks`, the
stock price/feature/dataset routes, `/models/baseline/metrics`, and
`/predict/{ticker}`. Do not change their schemas while extending `/api/v1`.

## Task handoff: Arfa

1. Own the price/news feature join and declare publication cutoffs before adding
   a column to `src/paisa/experiments/training_features.py`.
2. Keep the current price-only feature set as the control; prepare paired
   price-only and price-plus-sentiment configurations with identical splits.
3. Train and evaluate the paired Random Forest/XGBoost/LSTM experiments without
   using test results for selection.
4. Add SHAP result views only after Waqas exposes a validated artifact/API
   schema; explain model behavior, not causality.
5. Extend dashboard components and client schemas without recalculating backend
   metrics in React.
6. Lead additional-stock data quality, evaluation, screenshots, and supervisor
   reporting.

## Task handoff: Waqas

1. Implement source-attributed financial-news ingestion with offline fixtures
   and separate provider/network checks.
2. Build the ticker/company alias registry and reviewable entity-linking output.
3. Implement pinned FinBERT inference and leakage-safe daily aggregation with
   publication-time cutoffs.
4. Develop the LSTM and relationship/knowledge-graph components behind the
   current prediction and artifact contracts.
5. Extend `src/paisa/api/schemas.py`, `src/paisa/api/artifact_repository.py`,
   and `src/paisa/api/router_v1.py` additively for news, sentiment,
   explanations, and relationships; keep GET requests read-only.
6. Provide contract fixtures and API tests for each new result before dashboard
   integration.

## Commands to continue experiments

```bash
# Legacy/offline pipeline smoke
python scripts/run_v0_pipeline.py --provider sample --symbols OGDC HBL MCB --start 2021-01-01 --end 2023-12-31 --train

# Four versioned price-only models from the retained, standardized snapshot
PAISA_NEW_TRAINING_RUN_ID="TRN-20261022-team-handoff-01"
python scripts/train_ogdc_models.py \
  --prices artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/psx_prices.csv \
  --run-id "$PAISA_NEW_TRAINING_RUN_ID"

# Full OGDC vertical slice with a new immutable ID. The command still attempts
# live PSX-DPS first, then validates this complete checksum-bound cache bundle.
PAISA_NEW_VERTICAL_RUN_ID="VSL-20261022-OGDC-team-handoff-01"
python scripts/run_ogdc_vertical_slice.py \
  --cached-prices artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/psx_prices.csv \
  --run-id "$PAISA_NEW_VERTICAL_RUN_ID"

# Inspect supported overrides before any real run
python scripts/run_ogdc_vertical_slice.py --help
python scripts/run_backtest.py --help
```

The quoted branch and run IDs above are executable examples; replace them with
unused, task-specific values. Do not rerun with an existing ID. The retained
supervisor run must remain unchanged.

## Required verification before handoff

From the repository root with `.venv` active:

```bash
python -m pytest
python -m pip check
cd frontend
npm run test
npm run lint
npm run build
cd ..
git diff --check
```

Report the exact pass/fail/skip counts and warnings, branch and commit, generated
artifact IDs, assumptions, known limitations, and any contract changes. Keep
provider-dependent network checks separate from the default offline suite.
