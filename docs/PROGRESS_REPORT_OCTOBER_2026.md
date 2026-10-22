# PAISA Progress Report — October 2026

Report date: 22 October 2026

Supervisor evidence run: `VSL-20261020-OGDC-main-02` (`PASS`)

Training run: `TRN-20261020-191815-624662-nogit`

All reported predictions, trades, portfolio values, and returns are academic
simulations and are not financial advice.

## Objective

Deliver a reproducible end-to-end daily-data vertical slice for OGDC: prepare
historical PSX data, train comparable models without look-ahead leakage, emit
standard prediction artifacts, run independent simulated portfolios and common
benchmarks, expose retained results through FastAPI, and present them in a
React/Plotly dashboard.

## Completed work

- Preserved the legacy v0 pipeline and API while adding immutable versioned
  experiment contracts.
- Implemented chronological target-date splits, train-only learned transforms,
  explicit feature/target/signal/execution dates, fixed seeds, checksums, and
  collision refusal.
- Trained three Random Forest variants (`M-001` to `M-003`) and one XGBoost
  variant (`M-004`) on the same OGDC observations.
- Passed each model's untouched test predictions through the same separate
  decision policy and independent portfolio engine.
- Added cash-only and 50%-allocated buy-and-hold benchmarks, fees, trade
  ledgers, daily snapshots, drawdowns, and aggregate reports.
- Added a read-only FastAPI artifact API and React dashboard for history,
  model/portfolio comparisons, experiment details, and provenance.
- Preserved the completed run and its model-training reference as immutable,
  checksum-inventoried local artifacts.

## Data and split evidence

| Item | Retained value |
|---|---|
| Price source | `psx_dps`, recovered through `verified_legacy_psx_cache` |
| Live fetch | Failed with HTTP 404; no silent sample/synthetic substitution |
| OGDC price rows | 1,239 |
| Price coverage | 2021-09-16 to 2026-09-15 |
| Usable feature coverage | 2021-10-13 to 2026-09-14 |
| Train / validation / test | 853 / 182 / 184 observations |
| Test feature window | 2025-12-15 to 2026-09-14 |
| Test target window | 2025-12-16 to 2026-09-15 |
| Initial cash | PKR 100,000 per independent portfolio |
| Policy | Buy at `>= 0.60`, sell at `<= 0.40`, otherwise hold |
| Execution and cost | Next observed session open; 10 bps fee; zero modeled slippage |
| Exposure | Long-only, whole shares, maximum 50% allocation |

The cache was rebuilt from preserved legacy raw captures and bound to source
metadata and a migration manifest with SHA-256 inventories. This is auditable
internal provenance, not a fresh live endpoint response or independent provider
signature. The captured source has no `high`/`low` values.

## Real experiment results

Prediction metrics and simulated profitability answer different questions. The
validation-selected model was frozen using validation metrics; test metrics did
not select it. Portfolio outcomes additionally depend on thresholds, position
state, next-open prices, the return path, and fees.

| Model | Algorithm | Validation balanced accuracy | Validation F1 | Test accuracy | Test balanced accuracy | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| `M-001` | Random Forest | 53.0204% | 45.8599% | 51.6304% | 51.2289% | 22.6087% |
| `M-002` | Random Forest | 50.0846% | 51.3369% | 52.7174% | 52.4459% | 36.4964% |
| `M-003` | Random Forest | 51.3350% | 42.8571% | 51.0870% | 50.6794% | 21.0526% |
| `M-004` | XGBoost | 49.4503% | 48.8889% | 49.4565% | 49.2910% | 40.0000% |

`M-001` is the validation-selected prediction model because the selection rule
prioritizes validation balanced accuracy, then validation F1 and accuracy.

| Portfolio | Ending value | Total return | Maximum drawdown | Trades | Fees |
|---|---:|---:|---:|---:|---:|
| `M-001` strategy | PKR 99,298.24 | -0.70176% | 2.19966% | 2 | PKR 99.06 |
| `M-002` strategy | PKR 105,987.04 | 5.98704% | 5.66996% | 3 | PKR 160.95 |
| `M-003` strategy | PKR 99,166.07 | -0.83393% | 1.18467% | 2 | PKR 98.93 |
| `M-004` strategy | PKR 100,190.05 | 0.19005% | 7.12784% | 34 | PKR 1,751.06 |
| Cash-only | PKR 100,000.00 | 0.00000% | 0.00000% | 0 | PKR 0.00 |
| 50%-allocated buy-and-hold | PKR 106,247.89 | 6.24789% | 12.79045% | 1 | PKR 49.75 |

`M-002` is the best model portfolio, but it trails buy-and-hold by 0.26085
percentage points. Buy-and-hold produces the highest return but also has a much
higher drawdown than `M-002`. `M-003` has the lowest model-portfolio drawdown
while losing money. `M-004` overtrades and incurs the largest fees. No model
portfolio beats buy-and-hold in total return.

These conclusions apply only to OGDC, this historical period, daily frequency,
the retained price-only feature set, one chronological split, the fixed policy,
50% allocation cap, fee assumptions, and next-open execution convention.

## Test evidence

The complete branch was verified on 22 October 2026 after the documentation
pack was prepared:

| Command | Result |
|---|---|
| `python -m pytest` | `276 passed, 1 skipped`; one `StarletteDeprecationWarning` |
| `python -m pip check` | No broken requirements found |
| `cd frontend && npm run test` | 8 files and 57 tests passed |
| `cd frontend && npm run lint` | Passed with zero allowed warnings |
| `cd frontend && npm run build` | Passed; main bundle 1,441.02 kB (462.91 kB gzip) with a non-failing `>500 kB` chunk warning |
| `git diff --check` | Passed |

The skipped test is the optional real TensorFlow LSTM round-trip when TensorFlow
is not installed. The default RF/XGBoost vertical slice does not require
TensorFlow.

## Screenshots to capture

- [ ] Backend terminal showing successful startup with
      `PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02`.
- [ ] Swagger at `http://127.0.0.1:8000/docs` with the `/api/v1` run, model,
      portfolio, experiment, and provenance routes visible.
- [ ] Run summary response showing `PASS`, the retained run/training IDs, common
      split counts, and the academic disclaimer.
- [ ] Dashboard overview with the selected run, OGDC, evaluation window,
      warnings, and disclaimer visible.
- [ ] Historical OGDC chart and source context showing the retained date range
      and unavailable `high`/`low` fields.
- [ ] Model comparison showing `M-001` as the validation-selected model and all
      four validation/test metric rows.
- [ ] Portfolio comparison showing all independent model curves, cash-only, and
      50%-allocated buy-and-hold.
- [ ] `M-002` experiment detail with ending value, return, drawdown, three trades,
      and PKR 160.95 fees.
- [ ] `M-004` detail showing 34 trades and PKR 1,751.06 fees.
- [ ] Provenance view showing `verified_legacy_psx_cache`, failed live fetch,
      checksums, warnings, and the legacy-cache limitation.

## Current risks and limitations

- Evidence covers one stock and daily observations only; it does not establish
  generalization across stocks, sectors, periods, or market regimes.
- The retained run's live provider request failed, so this demonstration depends
  on the local verified legacy bundle being present under the configured roots.
- `high` and `low` are absent. Corporate-action adjustment is unknown and
  dividends are not modeled, so neither strategy nor benchmark is total return.
- The classifier predicts feature-close to next-close direction, while a new
  position can execute only at the next open.
- No news, sentiment, entity linking, knowledge graph, or SHAP evidence exists.
- There is no production database, authentication, broker integration, live
  trading, or scheduled retraining.
- The optional LSTM has a contract and notebook but is not represented in this
  retained run; its real TensorFlow smoke is skipped in the default environment.
- The frontend production build passes, but its main JavaScript chunk exceeds
  Vite's 500 kB advisory threshold.

An older `PROJECT_CONTEXT.md` status still describes the baseline-only phase,
and `docs/DEFINITION_OF_DONE.md` still requires successful fresh live acquisition
and an executed LSTM smoke. The current code is ahead of the former, while the
retained run does not satisfy those two strict checks in the latter.

## Next milestones

1. Add provenance-rich financial-news ingestion and reviewed company/entity
   linking.
2. Add pinned FinBERT inference and leakage-safe daily sentiment aggregation.
3. Run matched price-only versus price-plus-sentiment comparisons.
4. Execute LSTM under the same frozen split, prediction, policy, and backtest
   contracts.
5. Add SHAP artifacts and API/UI explanations.
6. Expand to additional PSX stocks, then dated sector/event relationships.

Reinforcement learning remains explicitly deferred until these data,
supervised-model, evaluation, and explanation components are stable.
