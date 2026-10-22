# PAISA supervisor demonstration script

Demo target: `VSL-20261020-OGDC-main-02`

Expected duration: 7–8 minutes

Scope: completed OGDC daily-data vertical slice

PAISA is an academic simulation. Its predictions, signals, trades, portfolio
curves, and simulated returns are not financial advice.

## Preflight before the meeting

Run these checks from the repository root. The real run and its referenced
training run are generated, ignored artifacts; they are deliberately not stored
in Git.

```bash
git status --short --branch
test -f artifacts/vertical_slice/VSL-20261020-OGDC-main-02/manifest.json
test -f artifacts/vertical_slice/VSL-20261020-OGDC-main-02/final_summary.json
test -f artifacts/model_training/TRN-20261020-191815-624662-nogit/run_manifest.json

source .venv/bin/activate
python -m pip check
python -m pytest

cd frontend
npm install
npm run test
npm run lint
npm run build
cp .env.example .env.local
cd ..
```

The frontend requires Node.js 22.12 or newer in the Node 22 line; Node 24 is
also supported. Confirm that `frontend/.env.local` contains:

```text
VITE_PAISA_API_BASE_URL=http://127.0.0.1:8000
```

Close unrelated browser tabs, set browser zoom to 100%, and keep two terminals
visible. Do not rerun training during the demonstration: the point is to inspect
the completed immutable result.

## Start the applications

Terminal 1, from the repository root:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2, from the repository root:

```bash
cd frontend
npm run dev
```

Open these tabs before the supervisor arrives:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Dashboard: <http://127.0.0.1:5173>
- Optional health check: <http://127.0.0.1:8000/health>

If the dashboard reports no completed runs, first verify both artifact roots and
then open
<http://127.0.0.1:8000/api/v1/vertical-slices>. The API publishes only completed,
checksum-valid runs whose referenced training artifact tree also validates.

## Timed demonstration flow

### 0:00–0:40 — Frame the objective

Show the dashboard Overview page with
`VSL-20261020-OGDC-main-02` selected.

Say:

> PAISA now has one complete, reproducible vertical slice for daily OGDC data.
> It takes checksum-bound historical input through leakage-safe model evaluation,
> converts predictions into policy signals, simulates independent portfolios,
> serves immutable results through FastAPI, and presents them in a read-only
> React dashboard. This is an academic simulation, not financial advice.

Point out the run status `PASS`, symbol `OGDC`, the common split counts (853
train, 182 validation, and 184 test rows), and the separate headline cards for
prediction selection, portfolio performance, drawdown, and buy-and-hold.

### 0:40–1:30 — Prove the API and artifact boundary in Swagger

Switch to <http://127.0.0.1:8000/docs>. Expand and execute, in this order:

1. `GET /api/v1/vertical-slices`
2. `GET /api/v1/vertical-slices/{run_id}/summary`, using
   `VSL-20261020-OGDC-main-02`
3. `GET /api/v1/vertical-slices/{run_id}/source`, using the same run ID

Direct URLs, if needed:

```text
http://127.0.0.1:8000/api/v1/vertical-slices
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/summary
http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/source
```

Say:

> These are read-only artifact endpoints. A request does not retrain a model,
> run a backtest, deserialize a model binary, or change a portfolio. The artifact
> repository checks the run's PASS manifest, exact SHA-256 inventory, safe IDs,
> and referenced model-training integrity root before returning schema-validated
> JSON.

Expected signal: the selected run is returned with the academic-simulation
disclaimer and without an absolute local filesystem path.

### 1:30–2:20 — Explain the historical OGDC input

Return to the dashboard and select **Historical OGDC**.

Say:

> This view reads the selected run's immutable input snapshot through the v1
> API; it does not read the mutable legacy `data/processed` file. The completed
> run contains 1,239 daily OGDC observations from 16 September 2021 through
> 15 September 2026. The charts show the recorded daily close and volume.

Then state the provenance precisely:

> The live PSX-DPS request failed with HTTP 404 for this run. PAISA therefore
> used a verified legacy PSX-DPS cache rebuilt from captured raw CSV files. The
> reconstructed prices, preserved legacy metadata, migration manifest, and
> pipeline metadata are checksum-bound. This is verified real historical input,
> but it is not a fresh live-endpoint acquisition.

Do not imply full OHLCV coverage: the captured source supplies open, close, and
volume; high and low are absent.

### 2:20–3:50 — Explain the four models and validation versus test

Select **Model comparison**.

Introduce the variants:

- `M-001`: the existing baseline Random Forest configuration.
- `M-002`: a conservative Random Forest configuration.
- `M-003`: a more flexible Random Forest configuration.
- `M-004`: a deterministic CPU XGBoost configuration.

Say:

> All four models use the same ordered feature rows, one-session target,
> chronological target-date split, and random seed. Learned transforms are fit
> on training data only. The target is whether the next observed session's close
> is above the feature-date close.

Point to the validation and held-out test charts, then say:

> `M-001` is the validation-selected prediction model. It was selected by
> validation balanced accuracy, then validation F1 and accuracy as tie-breakers;
> its validation balanced accuracy is 0.530204 and validation F1 is 0.458599.
> Test metrics are reported after selection and never used to rename the
> champion. `M-002`, for example, has the highest test accuracy here at 0.527174,
> but that is not a valid basis for going back and selecting it.

Emphasize that accuracy measures classification agreement, not the economic
result of a thresholded, fee-paying strategy.

### 3:50–5:40 — Explain independent portfolios and benchmarks

Select **Portfolio comparison**.

Say:

> Each model receives its own newly funded PKR 100,000 strategy ledger and unique
> portfolio ID. Cash, shares, signals, trades, fees, and snapshots are never
> shared. All strategies use the same 0.60 buy threshold, 0.40 sell threshold,
> 50% maximum allocation, whole shares, 10-basis-point fee, zero modeled
> slippage, next-session-open execution, and daily-close valuation.

Read the outcomes without calling any model a universal winner:

- `M-002` is the best simulated model portfolio: PKR 105,987.04, or
  5.98704% total return, with 5.66996% maximum drawdown.
- `M-003` has the lowest model-portfolio drawdown, 1.18467%, but its return is
  negative at -0.83393%.
- `M-004` overtrades relative to the others: 34 executed trades and PKR 1,751.06
  in fees, for only 0.19005% total return.
- Controlled buy-and-hold ends at PKR 106,247.89, or 6.24789% total return,
  with 12.79045% maximum drawdown and PKR 49.75 in fees.

Say:

> No model portfolio beats buy-and-hold in total return. However, buy-and-hold's
> drawdown is more than twice `M-002`'s drawdown. That is a return-versus-risk
> observation for this controlled period, not evidence that either approach will
> generalize. This benchmark also follows the shared 50% allocation rule; it is
> not a conventional fully invested benchmark.

Point briefly to both the equity and drawdown charts. Explain that the cash-only
line is the no-trade control and remains PKR 100,000.

### 5:40–6:40 — Inspect trades, fees, timing, and provenance

Select **Experiment detail**, choose `M-004`, and show **Execution assumptions**
and **Trade history**.

Say:

> A prediction is formed after the feature-date close. The policy sees only the
> probability and current position, creates a BUY, HOLD, or SELL signal, and any
> eligible trade executes at the next observed session's open. The ledger then
> values holdings at that session's close. `actual_class` is evaluation-only; it
> cannot enter policy or accounting. Every trade row preserves signal date,
> execution date, quantity, reference price, execution price, fee, remaining
> cash, and remaining shares.

Then select **Data & limitations** and point to acquisition evidence, migration
status, checksums, warnings, and the persistent disclaimer.

### 6:40–7:40 — State limitations and next-semester plan

Say:

> The conclusion is deliberately narrow: one stock, daily data, this date
> period, this technical feature set, these models, this 60/40 policy, and these
> execution and fee assumptions. Price adjustment and corporate-action treatment
> are unknown; dividends are not modeled. The classification target is
> close-to-close, while portfolio returns begin at the next open, so prediction
> accuracy and simulated profitability answer different questions.

Also state what is not implemented: fresh reliable live ingestion for this run,
sentiment, a knowledge graph, a production database, and broker integration.

Close with:

> Next semester we will first add financial-news ingestion, company/entity
> linking, FinBERT sentiment, and daily sentiment aggregation. We will compare
> price-only with price-plus-sentiment under the same chronological split, add an
> LSTM under that contract, use SHAP for explanations, then broaden to more
> stocks and sector/event relationships. Reinforcement learning is explicitly
> deferred until those data, evaluation, and explanation components are stable.

## Expected result checkpoints

| Checkpoint | Expected value |
|---|---|
| Run | `VSL-20261020-OGDC-main-02`, `PASS` |
| Referenced training run | `TRN-20261020-191815-624662-nogit` |
| Input | 1,239 OGDC daily rows, 2021-09-16 to 2026-09-15 |
| Test window | feature dates 2025-12-15 to 2026-09-14; target dates 2025-12-16 to 2026-09-15 |
| Validation-selected prediction model | `M-001` |
| Best simulated model portfolio | `M-002`, PKR 105,987.04, return 5.98704% |
| Lowest model-portfolio drawdown | `M-003`, drawdown 1.18467% |
| Highest model trading/fees | `M-004`, 34 trades, PKR 1,751.06 |
| Buy-and-hold | PKR 106,247.89, return 6.24789%, drawdown 12.79045% |
| Any model beat buy-and-hold return? | No |

## Recovery shortcuts

- Backend not reachable: confirm Terminal 1 still shows Uvicorn listening on
  `http://127.0.0.1:8000`.
- Dashboard cannot reach the API: confirm
  `VITE_PAISA_API_BASE_URL=http://127.0.0.1:8000`, then restart `npm run dev`.
- Run list is empty or returns 409: verify the exact vertical-slice and training
  run roots; never create placeholder artifacts.
- Historical view fails while a legacy stock endpoint works: use the run-scoped
  `/api/v1/vertical-slices/.../stocks/OGDC/prices` endpoint. The dashboard must
  not fall back to `/stocks/OGDC/prices`.
- One cancelled browser request in React development mode is expected from
  Strict Mode setup/cleanup replay; the following successful request is the
  operative one.
