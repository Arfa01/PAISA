# PAISA OGDC Vertical-Slice Sprint: 19–24 October 2026

## Sprint objective

Deliver a demonstrable, end-to-end, daily-data vertical slice for `OGDC` by
24 October 2026. The slice must compare versioned model variants and their
independent simulated portfolios, expose the results through FastAPI, and show
them in a simple React/Plotly dashboard.

This document is a delivery plan. The `sprint/oct19-foundation` task establishes
contracts, guardrails, and dependency boundaries only; it does not implement
training, backtesting, API endpoints, or the frontend.

All predictions and portfolio results are academic simulations and are not
financial advice.

## Current repository status at sprint start

The repository already has a working PAISA v0 path:

- Daily PSX ingestion through `psx-dps`, with `sample` and `manual-csv`
  fallbacks. At audit time the optional `pypsx` provider was incorrectly bundled
  into the default requirements; the foundation change moves it to a separate,
  explicitly optional install.
- A canonical price table with
  `date, ticker, open, high, low, close, volume, source`.
- Backward-looking technical features plus the binary
  `target_next_day_up` label.
- A chronological 70/15/15 train/validation/test split.
- One fixed-configuration Random Forest baseline.
- CSV, JSON, and joblib artifacts at fixed v0 paths.
- File-backed FastAPI endpoints for health, pipeline status, stock data,
  baseline metrics, and a latest prediction.
- Three automated tests covering preprocessing, the offline sample pipeline,
  and the main API happy path. On 19 October 2026 the suite passed with
  `3 passed` and one upstream Starlette/httpx deprecation warning.

Important limitations at sprint start:

- There is no model or experiment registry, immutable artifact ID, XGBoost or
  LSTM path, portfolio simulator, backtest, benchmark, experiment API, or
  dashboard.
- Current model outputs use fixed names and are overwritten on rerun.
- The current split assigns rows by feature date even though the label uses the
  next trading session. A boundary row can therefore use a target from the next
  split unless the new experiment path splits by target date or purges the
  boundary.
- Features include the current session close and volume. They are valid only
  for an after-close signal whose simulated execution occurs no earlier than
  the next trading session.
- The main PSX endpoint may omit `high` and `low`; it must not be treated as full
  OHLCV. Previously observed source coverage began on 14 September 2021, even
  when January 2021 was requested.
- `pypsx-toolkit` packaging is unreliable on some platforms and is not required
  for the standard `psx-dps`, `manual-csv`, or `sample` paths.

## Mandatory scope for 19–24 October

### 1. Reproducible experiment foundation

- Implement the interfaces in [ARCHITECTURE_CONTRACTS.md](ARCHITECTURE_CONTRACTS.md).
- Give every trained variant a unique, immutable `model_id` and every run a
  unique, immutable `experiment_id` and `portfolio_id`.
- Store manifests, predictions, trades, snapshots, and results in versioned,
  non-overwriting locations while retaining the v0 paths/endpoints needed for
  compatibility. Every new model must be preserved canonically under its unique
  ID before any explicit compatibility alias/copy is considered; such an alias
  may never replace the canonical artifact of an earlier run.
- Record configuration, ordered features, target and horizon, seeds, data and
  evaluation dates, code/dependency provenance, metrics, and artifact paths.

### 2. OGDC daily dataset

- Fetch and prepare daily `OGDC` data using `psx-dps`; retain `manual-csv` as a
  fallback and `sample` for offline tests.
- Validate schema, sort order, duplicates, missing values, actual date coverage,
  and the mapping from each feature date to the next observed trading date.
- Freeze the vertical-slice target as next-trading-day direction:
  `close[target_date] > close[feature_date]`.
- Split chronologically by target date, or purge the horizon-sized boundary, so
  no label crosses train/validation/test boundaries.

### 3. Model training and prediction records

- Train reproducible Random Forest and XGBoost variants through one shared
  experiment interface.
- Provide a Colab-compatible LSTM notebook that uses the same dataset contract,
  chronological evaluation policy, and export manifest.
- Persist class probabilities using the estimator's explicit class mapping;
  do not assume positional probability columns.
- Emit a prediction record for each eligible evaluation row without reading its
  future label when producing the prediction.

### 4. Independent simulated portfolios and backtesting

- Keep the model, decision policy, and portfolio engine separate.
- Run at least three independent model/configuration experiments. Each must own
  separate initial cash, holdings, trades, and daily snapshots.
- Use after-close signals and execute no earlier than the next observed trading
  session using a documented price, fee, sizing, and missing-price policy.
- Prevent negative cash, short positions, and allocation above the configured
  maximum unless a later contract explicitly enables them.
- Add cash-only and buy-and-hold benchmarks over the same evaluation dates and
  with documented, comparable cost assumptions.

### 5. Comparison metrics

- Report prediction metrics and portfolio metrics together without conflating
  them.
- Include ending equity, total return, maximum drawdown, annualized volatility,
  Sharpe ratio, trade count, win rate, and benchmark comparison.
- Make metric formulas, annualization factor, risk-free rate, null/zero cases,
  and trade-win accounting explicit and test them.

### 6. FastAPI experiment surface

- Add read-only endpoints to list models and experiments and retrieve one
  experiment's configuration, predictions, metrics, trades, and portfolio
  curve.
- Validate identifiers and return clear `404`/validation responses for missing
  or malformed resources.
- Preserve all existing v0 endpoints and response behavior.

### 7. React/Plotly dashboard

- Show the `OGDC` price history.
- Compare model prediction metrics.
- Plot independent portfolio equity curves alongside cash-only and buy-and-hold.
- Show the selected model/configuration, evaluation window, and a visible
  “academic simulation — not financial advice” disclaimer.
- Keep the dashboard read-only; it must not contain portfolio or model logic.

### 8. Quality, workflow documentation, and demonstration

- Add unit tests for identifiers, artifact collision handling, leakage guards,
  policy thresholds, accounting, metrics, API behavior, and benchmark alignment.
- Add an offline end-to-end test for the complete experiment path.
- Document exact setup, data, training, backtest, API, frontend, and test
  commands from a fresh clone.
- Prepare one reproducible supervisor demonstration using named, retained
  experiment IDs rather than hand-edited output.

## Optional/stretch scope

Only begin these items after every mandatory acceptance check is green:

- Additional Random Forest/XGBoost hyperparameter variants.
- Confidence calibration or a neutral/no-trade band study.
- Walk-forward evaluation beyond the single frozen chronological split.
- Dashboard filters, trade markers, downloadable results, or visual polish.
- CI execution of the offline suite and artifact-schema validation.
- A documented second PSX data-source comparison or provider health report.
- Extra portfolio statistics such as Sortino ratio, turnover, or exposure.

## Dependency order

1. Freeze the schemas, dates, IDs, target, execution semantics, and storage
   layout.
2. Prepare and validate the OGDC dataset and trading-date mapping.
3. Implement leakage-safe chronological splits and immutable model storage.
4. Train Random Forest and XGBoost; the LSTM notebook may proceed in parallel
   once steps 1–2 are stable.
5. Materialize prediction records from frozen model artifacts.
6. Apply decision policies to predictions.
7. Execute each policy in its own portfolio ledger and generate both
   benchmarks.
8. Calculate and persist experiment metrics.
9. Expose persisted results through FastAPI.
10. Render the API results in the dashboard.
11. Run compatibility, unit, integration, and end-to-end tests; then rehearse
    the documented demonstration from a clean environment.

Later stages must consume the preceding contract rather than import internal
implementation details. In particular, the dashboard depends on the API, the
API depends on persisted experiment results, and portfolio simulation depends
on predictions—not on estimator objects.

## Branch plan

No work is committed directly to `main`. Keep one concern per branch and merge
only reviewed, green branches into `sprint/oct19-integration` before a final
reviewed merge to `main`. The foundation branch remains documentation and
dependency work only.

| Branch | Responsibility |
|---|---|
| `sprint/oct19-foundation` | Contracts, sprint plan, dependency separation, and definition of done only. |
| `sprint/oct19-integration` | Reviewed sprint integration and end-to-end stabilization; no direct feature development. |
| `feature/ogdc-data-slice` | OGDC validation, trading-date mapping, and leakage-safe split. |
| `feature/versioned-models` | Immutable model registry, Random Forest, XGBoost, and prediction records. |
| `feature/lstm-colab` | Colab notebook and compatible LSTM export manifest. |
| `feature/portfolio-backtests` | Policies, independent ledgers, metrics, cash, and buy-and-hold benchmarks. |
| `feature/experiment-api` | Read-only experiment endpoints while preserving v0 routes. |
| `feature/ogdc-dashboard` | React/Plotly presentation consuming only the API. |
| `test/oct24-demo` | Cross-component tests, runbook validation, and supervisor demo fixture. |

If a branch needs a contract change, update the contract and its consumers in a
dedicated reviewed change before depending on the new shape.

## Definition of done

The sprint is done only when every item in
[DEFINITION_OF_DONE.md](DEFINITION_OF_DONE.md) has reproducible evidence. In
summary:

- Existing v0 tests and interfaces still work.
- OGDC can be fetched, prepared, and evaluated without future information.
- At least two algorithms train and every variant has a unique model ID.
- At least three configurations run with independent portfolios.
- Cash-only and buy-and-hold comparisons use the same evaluation window.
- Results are retrievable through FastAPI and visible in the dashboard.
- A fresh-clone runbook reproduces the workflow without overwriting artifacts.

“Code complete,” screenshots without retained experiment IDs, or locally edited
result files do not satisfy this definition.

## Daily milestones

| Date | Milestone and evidence |
|---|---|
| **19 Oct** | Merge foundation documents and dependency split; agree on contracts, branch ownership, target, execution timing, and acceptance evidence. |
| **20 Oct** | Produce validated OGDC data, trading-date mapping, leakage-safe splits, immutable ID/storage utilities, and their tests. |
| **21 Oct** | Train and retain Random Forest and XGBoost variants; emit validated prediction records and comparison metrics. |
| **22 Oct** | Complete the Colab LSTM notebook; run at least three independent portfolios plus cash and buy-and-hold; verify accounting and metric tests. |
| **23 Oct** | Expose experiment results through FastAPI and connect the first React/Plotly price, metric, and equity views. |
| **24 Oct** | Run the fresh-clone workflow, full suite, artifact-collision and leakage checks; fix documentation; rehearse and record the supervisor demonstration IDs. |

## Known risks and mitigations

| Risk | Mitigation |
|---|---|
| PSX endpoint outage or changed response | Keep raw-response validation, retries with bounded failure, manual CSV fallback, and an offline sample path. |
| Incomplete `high`/`low` values | Use only documented available fields; never invent OHLC values; fix one next-session execution-price rule. |
| Source coverage starts later than requested | Report actual coverage in manifests and fail acceptance if the usable evaluation window is too short. |
| Label crosses a split boundary | Assign/screen by `target_date` or purge the final horizon row from each earlier split; test the invariant. |
| Same-close backtest leakage | Generate signals after `feature_date` close and execute on a later observed session only. |
| Optional `pypsx` package installs but cannot import | Keep it outside the default requirements and treat `psx-dps` as the primary provider. |
| XGBoost/TensorFlow installation or Colab drift | Pin and document sprint-tested versions on their feature branches; keep LSTM dependencies out of the core runtime. |
| Small/noisy single-stock sample | Present uncertainty and benchmark results; do not market weak accuracy or returns as investment value. |
| Fixed v0 paths conflict with versioning | Add immutable registries and compatibility reads/aliases; never silently replace an existing ID. |
| Portfolio configurations accidentally share state | Construct a new ledger per `portfolio_id` and test state isolation explicitly. |
| Metric ambiguity creates misleading comparisons | Centralize formulas and require identical dates, price rules, and cost assumptions for benchmarks. |
| Six-day integration pressure | Freeze interfaces on 19 October, parallelize only behind those interfaces, and cut stretch work first. |

## Deferred features

The following are outside the 19–24 October vertical slice:

- News ingestion, sentiment analysis, FinBERT, and entity extraction.
- Knowledge graphs and cross-company/sector/event reasoning.
- SHAP or other advanced explainability beyond stored feature importance.
- Multi-stock portfolio optimization and broad-universe training.
- Intraday data, live trading, brokerage integration, or real-money execution.
- Production databases, authentication, user accounts, cloud deployment, and
  scheduled retraining.
- Corporate-action-adjusted research beyond documenting the source's price
  semantics.
- Investment recommendations or any claim that simulated results are financial
  advice.
