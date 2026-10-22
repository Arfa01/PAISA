# PAISA architecture overview

This document describes the implemented OGDC daily-data vertical slice. The
components are intentionally separated: data preparation owns market and feature
records; a model owns predictions; a policy owns signals; a portfolio owns cash,
holdings, trades, and valuation; the API validates and delivers immutable
artifacts; and the dashboard only presents API responses.

All predictions, signals, trades, portfolio curves, and returns are academic
simulations and are not financial advice.

## Component contracts

| Component | Consumes | Produces | Does not own |
|---|---|---|---|
| Data and feature pipeline | Provider rows and acquisition evidence | Standard daily prices, backward-looking features, next-session labels, provenance | Model choice, signals, balances |
| Model training | Frozen model rows and chronological split contract | Immutable model bundle, metrics, test prediction records | Trading policy or portfolio state |
| Prediction artifact | Registered model identity and evaluation rows | One record per model/ticker/feature date/target date | Trade decisions or cash |
| Decision policy | `probability_up`, thresholds, current position | BUY, HOLD, or SELL signal and target allocation | Actual label, estimator fitting, execution price |
| Portfolio/backtest | Signals, daily open/close bars, immutable experiment config | Trades, snapshots, metrics, independent benchmark ledgers | Feature calculation or inference |
| FastAPI artifact API | Completed run and referenced training trees | Schema-validated, read-only JSON | Retraining, backtesting, binary serving, writes |
| React dashboard | Versioned API responses | Tables, charts, warnings, provenance, disclaimer | Financial calculations or fallback data |

The normative schema details remain in
[`ARCHITECTURE_CONTRACTS.md`](ARCHITECTURE_CONTRACTS.md). The diagrams below show
how the implemented code realizes those boundaries.

## 1. Data pipeline

```mermaid
flowchart LR
    LIVE[PSX-DPS daily endpoint] --> ACQ[Provider acquisition]
    RAW[Verified legacy raw captures] --> MIG[Legacy-cache migration]
    MIG --> EVIDENCE[Checksum-bound cache evidence]
    ACQ --> SELECT{Usable live OGDC data?}
    EVIDENCE --> SELECT
    SELECT -->|live success| CLEAN[Standardize daily price schema]
    SELECT -->|live failure plus verified cache| CLEAN
    CLEAN --> PRICES["price rows: date, ticker, open, close, volume, source"]
    PRICES --> SNAP[Immutable input snapshot]
    PRICES --> FEATURES[Backward-looking technical features]
    FEATURES --> TARGET[Next observed session target mapping]
    TARGET --> ROWS[Model rows with feature_date and target_date]

    subgraph data_boundary["Data contract boundary"]
        SNAP
        ROWS
    end
```

Implemented entry points:

- `scripts/run_v0_pipeline.py` and `src/paisa/pipeline.py` orchestrate providers,
  standardization, feature creation, labels, and legacy v0 outputs.
- `src/paisa/preprocessing.py` standardizes source frames.
- `src/paisa/features.py` builds technical features and the v0 ML dataset.
- `scripts/migrate_legacy_psx_cache.py` and `src/paisa/legacy_cache.py` create and
  verify the legacy-cache evidence bundle.
- `src/paisa/vertical_slice.py` freezes the exact selected price and provenance
  bytes before model training.

For `VSL-20261020-OGDC-main-02`, the live request failed and the selected input
was a verified legacy PSX-DPS reconstruction: 1,239 OGDC daily rows covering
2021-09-16 through 2026-09-15. It is not represented as a successful fresh
fetch.

## 2. Model training

```mermaid
flowchart TB
    ROWS[Frozen common model rows] --> SPLIT[Chronological split by target_date]
    SPLIT --> TRAIN[Train: 853 rows]
    SPLIT --> VALID[Validation: 182 rows]
    SPLIT --> TEST[Test: 184 rows]
    TRAIN --> FIT[Fit transforms on train only]
    FIT --> M1[M-001 baseline Random Forest]
    FIT --> M2[M-002 conservative Random Forest]
    FIT --> M3[M-003 flexible Random Forest]
    FIT --> M4[M-004 XGBoost]
    M1 --> VEVAL[Validation metrics]
    M2 --> VEVAL
    M3 --> VEVAL
    M4 --> VEVAL
    VEVAL --> SELECT[Select provisional champion without test data]
    SELECT --> FREEZE[Freeze configurations and selection]
    FREEZE --> TEVAL[Evaluate all variants once on held-out test]
    TEVAL --> REG[Immutable training-run registry]

    subgraph training_boundary["Training artifact boundary"]
        REG --> ROOT[run_manifest.json and exact SHA-256 inventory]
        REG --> BUNDLES[model binary, manifest, config, features, metrics]
        REG --> REPORTS[validation and test leaderboards]
    end
```

The versioned CPU training entry point is `scripts/train_ogdc_models.py`, backed by
`src/paisa/experiments/trainer.py`. Feature, split, target, and evaluation rules
live in `training_features.py`, `training_targets.py`, and `evaluation.py`.
`M-001` is the validation-selected model for the completed run. Test metrics are
reporting evidence, not a model-selection input.

## 3. Prediction artifacts

```mermaid
flowchart LR
    MODEL[Registered model manifest] --> ID[Immutable registered model_id]
    TEST[Held-out test rows] --> INFER[Model inference]
    ID --> INFER
    INFER --> PRED[predictions.csv]
    PRED --> CONTRACT{Prediction-record validation}
    CONTRACT -->|valid| BACKTEST[Experiment input]
    CONTRACT -->|invalid identity, date, probability, or duplicate| FAIL[Fail closed]
    LABEL[actual_class joined after outcome is known] -. evaluation only .-> PRED

    subgraph prediction_boundary["Prediction record contract"]
        FIELDS[schema_version, model_id, ticker]
        DATES[feature_date distinct from target_date]
        VALUES[predicted_class, probability_up, probability_down]
        ACTUAL[actual_class nullable and evaluation-only]
    end
    PRED --> FIELDS
    PRED --> DATES
    PRED --> VALUES
    PRED --> ACTUAL
```

The test `predictions.csv` for each registered model is stored below
`artifacts/model_training/<training_run_id>/models/<registered_model_id>/`.
The vertical slice derives those paths and model bindings from the validated
training root; caller-provided paths cannot redirect the run. Probabilities must
be finite, sum to one within tolerance, and retain the estimator's declared class
mapping.

## 4. Decision policy

```mermaid
flowchart TB
    P[Prediction probability_up] --> SAFE[DecisionInput]
    POS[Current flat or invested state] --> SAFE
    SAFE --> CHECK{ConfidenceThresholdPolicy}
    CHECK -->|probability_up >= 0.60 and flat| BUY[BUY toward 50% allocation]
    CHECK -->|probability_up <= 0.40 and invested| SELL[SELL full position]
    CHECK -->|neutral band or redundant transition| HOLD[HOLD]
    BUY --> SIGNAL[SignalRecord at feature-date close]
    SELL --> SIGNAL
    HOLD --> SIGNAL
    LABEL[actual_class] -. prohibited .-> SAFE
    PRICE[future or execution price] -. prohibited .-> SAFE
```

`src/paisa/trading/policy.py` implements the inclusive confidence thresholds and
long-only target-allocation state transition. `DecisionInput` intentionally has
no actual label or market-price field. Repeated buys do not pyramid an existing
position; a sell while already in cash becomes HOLD.

## 5. Portfolio and backtesting engine

```mermaid
flowchart LR
    CONFIG[Immutable experiment config] --> ENGINE[BacktestEngine]
    SIGNAL[Signal after feature-date close] --> ENGINE
    BARS[Daily OGDC open and close] --> ENGINE
    ENGINE --> EXEC[Execute on target-date open]
    EXEC --> FEE[Apply 10 bps fee and whole-share rounding]
    FEE --> LEDGER[Unique strategy portfolio ledger]
    LEDGER --> SNAP[Value at target-date close]
    SNAP --> METRICS[Return, drawdown, volatility, Sharpe, trades, fees]

    BARS --> CASH[Independent cash-only ledger]
    BARS --> BAH[Independent 50%-allocated buy-and-hold ledger]
    CASH --> COMPARE[Common-period comparison]
    BAH --> COMPARE
    METRICS --> COMPARE

    subgraph experiment_boundary["Per-experiment immutable outputs"]
        P1[predictions.csv and signals.csv]
        P2[trades.csv and portfolio_snapshots.csv]
        P3[benchmark ledgers]
        P4[metrics.json, result.json, charts]
    end
    COMPARE --> P1
    COMPARE --> P2
    COMPARE --> P3
    COMPARE --> P4
```

`src/paisa/backtesting/engine.py` validates chronological prediction/session
alignment and orchestrates one strategy. `src/paisa/trading/execution.py` and
`portfolio.py` own fills and accounting. `src/paisa/backtesting/benchmarks.py`
creates separately funded cash-only and buy-and-hold ledgers.
`src/paisa/experiments/runner.py` persists one atomic experiment. No portfolio
ID, cash balance, holding, trade, or snapshot is shared between experiments.

The classification target measures feature-close to next-session-close
direction; the simulation enters at the next-session open. That deliberate
difference is carried as a warning because accuracy and profitability measure
different things.

## 6. FastAPI artifact API

```mermaid
flowchart TB
    ENV[Configured vertical and training artifact roots] --> REPO[ArtifactRepository]
    REPO --> PASS[Require completed PASS manifest]
    PASS --> VINV[Validate exact vertical file inventory and SHA-256 values]
    VINV --> TREF[Validate referenced training root and registered model bindings]
    TREF --> SAFE[Reject traversal, links, unsafe IDs, unexpected files, non-finite data]
    SAFE --> LOAD[Read approved JSON and CSV artifacts only]
    LOAD --> SCHEMA[Pydantic response schemas]
    SCHEMA --> ROUTES["/api/v1 read-only routes"]

    BINARY[model binaries] -. never load or serve .-> ROUTES
    WRITE[training, backtesting, or artifact writes] -. never perform in GET .-> ROUTES

    ROUTES --> RUNS[run list and detail]
    ROUTES --> RESULTS[summary, models, portfolios, curves]
    ROUTES --> AUDIT[source, experiments, trades, snapshots, benchmarks]
```

The application is `src/paisa/api/main.py`. The additive v1 router is
`src/paisa/api/router_v1.py`; secure artifact access is implemented in
`src/paisa/api/artifact_repository.py`; response contracts are in
`src/paisa/api/schemas.py`.

The service exposes, among others:

```text
GET /api/v1/vertical-slices
GET /api/v1/vertical-slices/{run_id}
GET /api/v1/vertical-slices/{run_id}/summary
GET /api/v1/vertical-slices/{run_id}/stocks/{ticker}/prices
GET /api/v1/vertical-slices/{run_id}/models
GET /api/v1/vertical-slices/{run_id}/portfolios
GET /api/v1/vertical-slices/{run_id}/equity-curves
GET /api/v1/vertical-slices/{run_id}/drawdowns
GET /api/v1/vertical-slices/{run_id}/source
GET /api/v1/vertical-slices/{run_id}/experiments
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/trades
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/snapshots
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/benchmarks
```

Legacy v0 routes remain compatible, but the dashboard uses the selected run's
v1 price endpoint rather than `data/processed/psx_prices.csv`.

## 7. React dashboard

```mermaid
flowchart TB
    START[React application] --> LIST[Fetch completed vertical-slice runs]
    LIST --> PICK[Select run_id]
    PICK --> CORE[Fetch detail, summary, leaderboard, models, portfolios, source, experiments]
    CORE --> VALIDATE[Zod schema and identity validation]
    VALIDATE --> OVERVIEW[Overview]
    VALIDATE --> HISTORY[Historical OGDC]
    VALIDATE --> MODELS[Model comparison]
    VALIDATE --> PORTS[Portfolio comparison]
    VALIDATE --> EXP[Experiment detail]
    VALIDATE --> PROV[Data and limitations]
    PORTS --> LAZY1[Paginated equity and drawdown rows]
    EXP --> LAZY2[Paginated trades, snapshots, benchmarks]
    HISTORY --> LAZY3[Run-scoped historical price pages]

    ERROR[Network, timeout, schema, identity, or integrity error] --> NOFALLBACK[No runtime fixture fallback]
    NOFALLBACK --> VISIBLE[Visible error and retry]
    OVERVIEW --> DISCLAIMER[Persistent academic-simulation disclaimer]
    HISTORY --> DISCLAIMER
    MODELS --> DISCLAIMER
    PORTS --> DISCLAIMER
    EXP --> DISCLAIMER
    PROV --> DISCLAIMER
```

`frontend/src/App.tsx` coordinates run selection and views.
`frontend/src/api/client.ts` owns HTTP access, timeouts, pagination, cancellation,
Zod validation, identity checks, and local-path rejection. Components below
`frontend/src/components/` render the six read-only views. The browser never
loads artifacts or model binaries and does not recompute model or portfolio
metrics.

## End-to-end publication and read path

```mermaid
sequenceDiagram
    participant V as Vertical-slice runner
    participant T as Model-training registry
    participant E as Four experiment runners
    participant A as Immutable vertical artifact
    participant API as FastAPI artifact API
    participant UI as React dashboard

    V->>T: Train M-001 through M-004 from frozen input
    T-->>V: Validated run manifest, model IDs, test predictions
    loop One independent experiment per model
        V->>E: Prediction records plus config plus common prices
        E-->>V: Strategy and benchmark ledgers, metrics, plots
    end
    V->>V: Reload ledgers and reconcile identities and metrics
    V->>A: Atomically publish PASS run with exact checksum inventory
    UI->>API: Request completed run by run_id
    API->>A: Validate vertical and referenced training integrity
    A-->>API: Approved JSON and CSV values
    API-->>UI: Versioned read-only responses and disclaimer
```

The completed demonstration run is stored locally at
`artifacts/vertical_slice/VSL-20261020-OGDC-main-02/` and references
`artifacts/model_training/TRN-20261020-191815-624662-nogit/`. These directories
contain runtime data and model artifacts, are ignored by Git, and must not be
committed.
