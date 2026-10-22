# PAISA Current Implementation Status

Status date: 22 October 2026

Code baseline: `origin/main` at `826c371` after PR #10

Retained demonstration run: `VSL-20261020-OGDC-main-02`

PAISA is an academic research prototype. Its predictions, signals, trades,
portfolio curves, and simulated returns are not financial advice.

## Implemented and working

| Area | Current implementation and evidence |
|---|---|
| Stock-data foundation | The legacy v0 path implements PSX acquisition through `psx-dps`, with manual-CSV and offline sample alternatives; cleans the canonical daily schema; creates backward-looking technical features and `target_next_day_up`; trains the compatibility Random Forest; and preserves the original FastAPI routes. The acquisition path is implemented, but the live endpoint returned HTTP 404 during the retained run, so fresh live ingestion is not currently demonstrated. The optional `pypsx` provider remains separate from the default installation. |
| Leakage-safe experiment data | The versioned path distinguishes `feature_date` and next-session `target_date`, splits chronologically by target date, fits learned transforms on training data only, records feature order and seeds, and preserves immutable input and model evidence. |
| Versioned model comparison | `M-001`, `M-002`, and `M-003` are Random Forest variants; `M-004` is XGBoost. All four use the same daily OGDC feature set, target, split dates, and untouched test observations. Each has a unique registered model identity and checksum-protected artifacts. |
| Decision and simulation | Predictions pass through a separate threshold policy into separate long-only portfolio ledgers. The retained run uses buy at `probability_up >= 0.60`, sell at `probability_up <= 0.40`, a 50% maximum allocation, whole shares, 10-basis-point execution fees, next-session-open execution, and close valuation. |
| Independent comparisons | Each model has its own experiment, portfolio, trades, cash, holdings, and snapshots. Cash-only and 50%-allocated buy-and-hold benchmarks use the common test window and cost assumptions. |
| Immutable vertical slice | `scripts/run_ogdc_vertical_slice.py` joins acquisition, preparation, training, predictions, policies, backtests, benchmarks, evidence validation, and reports without overwriting an existing run ID. |
| Results API | The additive `/api/v1` FastAPI surface reads completed, checksum-valid artifacts. It lists runs and exposes run metadata, source provenance, historical prices, model metrics, portfolio metrics, curves, experiments, trades, snapshots, and benchmarks. It does not retrain models or recalculate accounting in a GET request. |
| Dashboard | The React/TypeScript/Plotly application reads only the API and shows run overview, immutable OGDC history, model comparison, portfolio comparison, experiment detail, trades, snapshots, provenance, limitations, and the academic-simulation disclaimer. It has no model, policy, or accounting business logic. |
| Verification | On 22 October 2026 the pre-documentation branch baseline passed `276` Python tests with `1` optional TensorFlow test skipped, all `57` frontend tests, frontend lint, frontend build, and `pip check`. |

## Retained real-run evidence

The completed `PASS` run `VSL-20261020-OGDC-main-02` references training run
`TRN-20261020-191815-624662-nogit`. It contains 1,239 daily OGDC price rows
covering 2021-09-16 through 2026-09-15. The usable feature range is
2021-10-13 through 2026-09-14, with `853` train, `182` validation, and `184`
test observations. The test feature window is 2025-12-15 through 2026-09-14;
its corresponding target window is 2025-12-16 through 2026-09-15.

The run is real-data evidence from a verified reconstruction of the legacy PSX
cache, not a fresh live download. The attempted live PSX-DPS request returned
HTTP 404. The fallback retains the original metadata, rebuilt raw-capture
evidence, migration manifest, and SHA-256 inventory. This establishes internal
integrity and reproducibility, but not independent provider authentication.

The run reports separate conclusions:

- `M-001` is the validation-selected prediction model.
- `M-002` has the highest model-portfolio ending value: PKR 105,987.04, a
  5.98704% simulated return.
- `M-003` has the lowest model-portfolio drawdown: 1.18467%, but its simulated
  return is -0.83393%.
- `M-004` makes 34 trades and pays PKR 1,751.06 in fees; it returns 0.19005%.
- The 50%-allocated buy-and-hold benchmark ends at PKR 106,247.89, a 6.24789%
  return with a 12.79045% maximum drawdown.
- No model portfolio beats buy-and-hold in total return in this run.

## Current research prototypes

- The four price-only classifiers are controlled comparative experiments, not
  evidence of performance on other stocks, periods, horizons, or regimes.
- The policy thresholds and portfolio simulator are research assumptions, not a
  recommendation engine or execution system.
- The dashboard and artifact API are demonstration and analysis surfaces, not a
  production deployment or current market-data service.
- The optional LSTM code and Colab notebook implement the common artifact and
  chronological-split contract. TensorFlow is intentionally outside the default
  runtime, and LSTM is not one of the four models in the retained demonstration
  run.
- Stored Random Forest feature importance is preliminary explainability. SHAP
  explanations have not been implemented.
- The verified-cache migration is a provenance-preserving continuity mechanism;
  it does not turn the retained data into a fresh or cryptographically signed
  provider response.

## Progress against the approved proposal

| Approved objective | Status |
|---|---|
| Collect, preprocess, and structure historical stock data | Partly complete. The reusable pipeline supports multiple symbols, while the retained vertical slice and its conclusions are OGDC-only and daily-only. |
| Continuously update financial news, economic reports, and global events | Not implemented. Current workflows are explicit commands, not scheduled feeds. |
| Train and compare multiple prediction models | Implemented for three Random Forest configurations and one XGBoost configuration on one frozen OGDC split. The optional LSTM workflow is not promoted into this run. |
| Compare models with and without sentiment | Not implemented. The current comparison is price-only. |
| Deploy the best-performing model as the prediction engine | Not complete. The v1 API deliberately serves retained results and does not load `M-001` for live inference; the legacy `/predict/{ticker}` route remains the v0 baseline compatibility path. |
| Add event-driven company/sector reasoning | Not implemented. |
| Deliver predictions, visualizations, explanations, and current feeds | Partly complete. Historical results and provenance are visible in FastAPI and React; sentiment, event explanations, SHAP, and current feeds remain. |

## Remaining work

The next research sequence is financial-news ingestion, company/entity linking,
FinBERT sentiment, daily sentiment aggregation, price-only versus
price-plus-sentiment comparison, LSTM under the same split contract, SHAP,
additional stocks, and sector/event relationships. Production storage,
authentication, scheduled ingestion/retraining, cloud deployment, broker
integration, and live trading are also absent. Reinforcement learning is
explicitly deferred until the data, sentiment, supervised-model, evaluation,
and explanation components are stable.

## Documentation inconsistencies to keep visible

- `PROJECT_CONTEXT.md` contains an older baseline-only status and an older
  multi-symbol date range. It is useful proposal history, but it is not the
  current implementation inventory or the evidence source for this run.
- `docs/DEFINITION_OF_DONE.md` still has unchecked acceptance boxes and requires
  a successful live provider fetch plus an executed TensorFlow LSTM smoke run.
  The retained run instead records a failed live request, a verified legacy
  cache, and no LSTM model. Those two strict checks must not be represented as
  complete.
- The proposal's “best-performing model” is not a single result here: `M-001`
  wins the validation-based prediction selection, while `M-002` produces the
  best model-portfolio simulation. The system correctly preserves that
  distinction.
