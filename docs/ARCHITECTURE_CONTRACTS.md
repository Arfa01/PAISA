# PAISA Architecture Contracts

Status: foundation contract for the 19–24 October 2026 OGDC vertical slice.

These contracts define the stable boundary between model training, predictions,
decision policies, portfolio simulation, result delivery, and presentation.
Implementations may add fields, but they must not change the meaning or type of
an existing field without introducing a new `schema_version` and a documented
migration.

All outputs are academic simulations and are not financial advice.

## Component boundaries

| Component | Owns | Must not own |
|---|---|---|
| Data/feature pipeline | Clean daily prices, backward-looking features, labels, trading-date mapping, chronological splits | Model choice, trading signals, cash or holdings |
| Model trainer | Fitted estimator, immutable model manifest, evaluation predictions and prediction metrics | Buy/sell policy or portfolio state |
| Decision policy | Conversion of a prediction and thresholds into a signal/target allocation | Estimator training, trade execution, balances |
| Portfolio engine | Execution rules, fees, cash, shares, trades, daily valuation | Feature calculation or model inference |
| Experiment runner | Immutable references binding one model, policy/configuration, portfolio, and evaluation window | Shared mutable state between experiments |
| FastAPI | Validation and read-only delivery of registered artifacts/results | Retraining or recomputing accounting during a GET request |
| Dashboard | Read-only presentation of API responses | Model, policy, metric, or portfolio business logic |

A model may serve many experiments. Each experiment owns one strategy
`portfolio_id` plus two benchmark portfolio IDs (`cash_only` and
`buy_and_hold`). Every ledger is independent, and no portfolio ID may be shared
between experiments or roles.

## Common representation rules

- Every persisted manifest/result carries `schema_version`, initially `"1.0"`.
- IDs are non-empty, immutable, filesystem-safe strings. Recommended characters
  are lowercase ASCII letters, digits, `.`, `_`, and `-`. Generating an ID whose
  path already exists is an error; writers must never replace it silently.
- Calendar dates use ISO `YYYY-MM-DD`. Timestamps use ISO 8601 UTC with a `Z`
  suffix.
- `ticker`/`symbol` is uppercase; the first supported value is `OGDC`.
- Ratios and rates are decimal fractions unless the field explicitly says
  `bps`. For example, `0.01` means one percent.
- Monetary values and prices use the `PKR decimal` type for this slice: decimal
  arithmetic internally and a fixed-scale string with exactly two fractional
  digits when persisted to JSON or CSV (for example, `"1250.50"`). Every money,
  price, and fee calculation is quantized to PKR 0.01 with decimal
  `ROUND_HALF_UP` at the point specified by the accounting rule. Readers must
  parse these fields as decimals, never binary floating-point values.
- Share quantities are non-negative whole numbers for PSX simulation.
- Fields are required unless explicitly described as nullable. Unknown values
  are `null`, not zero, an empty string, or a fabricated value.
- Collections have a deterministic order. Feature columns retain their declared
  order, and dated records sort by date then stable identifier.
- Writes are atomic. A completed model/experiment directory is immutable.
- Runtime data, model binaries, and experiment outputs stay ignored by Git; only
  small approved fixtures and schemas belong in the repository.

## Target, dates, and leakage rules

The vertical slice uses one daily prediction target:

```text
target_next_day_up = 1 if close[target_date] > close[feature_date], else 0
```

`target_date` is the next observed trading session for the same ticker, not
calendar date plus one. The prediction horizon is exactly one trading session.

The timeline is normative:

1. Features for `feature_date` may use only information available by that
   session's close.
2. The model emits a prediction and the policy emits a signal after that close.
3. A trade executes no earlier than the next observed session, using the price
   field declared by the experiment's execution policy.
4. `actual_class` may be joined only after the target session is known and may
   be used for evaluation only—not for signal or execution decisions.

All splits are chronological. A row belongs to a split according to its
`target_date`, or the implementation must purge enough boundary rows that its
target cannot fall in a later split. Learned preprocessing is fitted on the
training split only. Hyperparameters, probability thresholds, and variant
selection may use training and validation data only. Freeze the selected
variant before evaluating the held-out test split once; test metrics must not
be used to select or tune a model/configuration. Random time-series splitting
is prohibited.

## Storage layout and immutability

New code uses unique directories such as:

```text
models/{model_id}/manifest.json
models/{model_id}/model.<format>
experiments/{experiment_id}/config.json
experiments/{experiment_id}/predictions.csv
experiments/{experiment_id}/portfolio_snapshots.csv
experiments/{experiment_id}/trades.csv
experiments/{experiment_id}/result.json
experiments/{experiment_id}/benchmarks/cash_only_snapshots.csv
experiments/{experiment_id}/benchmarks/buy_and_hold_snapshots.csv
experiments/{experiment_id}/benchmarks/buy_and_hold_trades.csv
```

The exact model extension is algorithm-specific. All paths in manifests are
POSIX-style paths relative to the configured model or experiment artifact root;
absolute paths are prohibited. The host resolves that root from configuration,
so artifacts may live outside the checkout without making a manifest
machine-specific. Checksums are calculated after writing. Creating a model or
experiment fails if its ID or destination already exists.

Existing v0 files such as `models/baseline_random_forest.joblib` and existing v0
API response shapes remain legacy compatibility interfaces. A versioned writer
must first preserve every new model under its immutable ID and must never write,
copy, or alias it onto a fixed v0 path in a way that replaces a previous
canonical artifact. Existing v0 code may retain its legacy fixed-name behavior
for compatibility tests, but new registered runs must not use those paths as
canonical storage.

### Training-run integrity root

Every versioned training-run directory has a `run_manifest.json` integrity
root. It records the run and invocation IDs plus the run-level `code_version`,
`dirty_tree`, and `source_fingerprint`; those source fields must exactly match
every registered model manifest. Its `artifact_inventory` is a sorted,
duplicate-free array of relative POSIX paths and SHA-256 digests for every
other regular file in the run. The inventory is exact-set: missing, additional,
linked, special, or byte-modified artifacts invalidate the run. It covers, at
minimum, frozen inputs, split and data manifests, reports, each model manifest,
fitted binary, complete configuration, feature metadata, metrics, and
prediction files.

Only `run_manifest.json` itself is excluded from its embedded inventory to
avoid a self-referential digest. A consuming workflow must separately capture
that root file's SHA-256, validate the complete inventory before use, and
reassert the unchanged root digest and payload set after delegated processing
and immediately before publication. The OGDC vertical slice persists the exact
root digest, source provenance, and inventory in
`model_training_run_reference.json`. Consumers derive the variant-to-model ID
mapping and prediction paths from this validated root; a caller-supplied
summary must match those bindings exactly and cannot redirect reads to an
external file. A controlled path/provenance rewrite may reseal only an
orchestrator-owned run that has not been published; an existing or
caller-supplied run is never silently resealed.

### Unified price-model comparison pack

The authoritative derived M-001 through M-006 price-only comparison is
`CMP-20261024-OGDC-price-only-02`. It is a read-only evidence transform, not a
training, inference, production-policy, or production-backtesting component.
`CMP-20261024-OGDC-price-only-01` is retained only as a superseded,
non-portable historical draft and is not acceptable publication evidence.

The builder accepts only the explicitly allowlisted completed training,
vertical-slice, and LSTM roots and validates their integrity roots and
relationships. Training and LSTM source manifests are copied byte-for-byte.
The vertical-slice raw manifest is deliberately excluded because it embeds
host-local paths; the pack instead carries a digest-allowlisted, redacted
certificate containing the source-manifest name and SHA-256, an allowlisted
projection of semantic assertions, and the complete source artifact inventory.
The source record distinguishes byte-copy roots from this deterministic
redacted projection.

The portable evidence includes persisted split and prediction rows, two raw
market CSVs, one policy configuration and one signal CSV for each of the six
models, and each independent model and benchmark ledger. The two market files
must prove the same ordered 185-session opening-and-holdout price vector. Raw
market prices and persisted policy outputs are therefore required evidence;
fitted model binaries, caches, and provider credentials are excluded.

Prediction metrics are recomputed from copied prediction rows. The
majority-class baseline derives its class from training labels, while the
previous-direction baseline uses the persisted prior row whose `target_date`
equals the scored row's `feature_date`. Portfolio verification independently
replays the policy, next-session-open execution, allocation constraints, fees,
trades, close marks, cash, holdings, realized and unrealized P&L, returns, and
drawdowns with exact `Decimal` arithmetic and the declared rounding rule. It
compares that reconstruction row-by-row and trade-by-trade with copied evidence
without importing or invoking the production backtest runner.

The metrics contract exposes exactly two logical benchmark rows: cash-only and
controlled buy-and-hold. Each logical row carries six ledger-provenance records
so all independent model-associated benchmark ledgers remain auditable even
when their economics agree. Every strategy and benchmark ledger retains a
distinct portfolio ID and snapshot ledger. Provenance binds each benchmark to
its associated model experiment and configuration; buy-and-hold includes trade
evidence, while cash-only intentionally has no trade file.

Validation-only selection remains separate from descriptive results on the
**shared project holdout**. Balanced accuracy, F1, simulated return, drawdown,
fees, buy-and-hold, and the validation-selected model are separate reported
outcomes; the pack must not collapse them into one cross-criterion score or
ranking.

The comparison directory uses an exact SHA-256 inventory that excludes only
its own `manifest.json`. Publication refuses existing directory, ZIP, or
checksum destinations. The deterministic ZIP contains one ID-matched top-level
directory and canonical member metadata; its adjacent checksum sidecar is
mandatory and is authenticated before archive extraction. The offline verifier
recursively scans every text member and rejects defined host-path and
credential-like patterns, including file URIs, URL credentials, authorization
values, and known token or key formats. It also rejects unsafe paths, links,
special or additional files, digest changes, source-identity changes,
assumption drift, date/label misalignment, persisted-metric drift, and any
mismatch between the human report and semantic replay. Normal CLI output identifies artifacts
relative to an explicit path base; default failures are concise and omit
tracebacks, while opt-in debug output is still sanitized.

This comparison does not integrate sentiment and has a tabular/report-only
presentation contract. It contains no plots, and comparison build and verify
paths must not import Matplotlib. The complete schema, commands, and canonical
source digests are documented in
[`UNIFIED_PRICE_MODEL_COMPARISON.md`](UNIFIED_PRICE_MODEL_COMPARISON.md).

## A. Model artifact contract

The model artifact is the manifest plus its referenced fitted binary. The
manifest is JSON.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Required contract version; initially `"1.0"`. |
| `model_id` | string | Globally unique immutable ID for this trained variant. |
| `algorithm` | string enum | `random_forest`, `xgboost`, or `lstm` for this sprint. |
| `model_configuration` | object | Complete, JSON-serializable hyperparameters and algorithm settings; omitted defaults are not allowed. |
| `feature_set` | object | Exact structure below: ID, ordered columns, feature-code version, and transforms. |
| `training_dates` | object | Exact inclusive feature coverage and target-date split boundaries defined below. |
| `target_definition` | object | Exact target name, positive-class condition, and class labels defined below. |
| `prediction_horizon` | object | `{ "value": 1, "unit": "trading_session" }` for the vertical slice. |
| `random_seed` | integer | Seed used by the model and every stochastic transform. |
| `metrics` | object | Exact per-split prediction metric structure defined below. Test metrics come from the held-out chronological test split. |
| `artifact_path` | string | Unique path, relative to the configured model artifact root, to the fitted binary inside this model's directory. |
| `created_at` | UTC timestamp | Time at which the completed artifact was registered. |
| `symbols` | array of strings | Training universe; `['OGDC']` for the first vertical slice unless explicitly documented otherwise. |
| `data_provenance` | object | Exact provider, coverage, fingerprint, symbol, price-adjustment, dividend, and corporate-action fields defined below. |
| `class_mapping` | object | Exact estimator class order and output probability-field mapping defined below. |
| `code_version` | string | Git commit SHA that produced the artifact. |
| `dirty_tree` | boolean | Whether tracked or untracked source changes existed; production/demo artifacts should use `false`. |
| `source_fingerprint` | string | SHA-256 of the relevant source/notebook content when `dirty_tree` is true; otherwise the commit SHA may be repeated. |
| `library_versions` | object | Relevant runtime versions, including the algorithm library. |
| `artifact_sha256` | string | SHA-256 digest of the referenced fitted binary. |

The following nested structures are required in schema `1.0`:

```json
{
  "feature_set": {
    "feature_set_id": "technical-daily-v1",
    "columns": ["ordered", "feature", "names"],
    "feature_code_version": "git-sha-or-source-sha256",
    "transforms": [
      {
        "name": "transform_name",
        "parameters": {},
        "fit_scope": "train_only"
      }
    ]
  },
  "training_dates": {
    "data_feature_start": "YYYY-MM-DD",
    "data_feature_end": "YYYY-MM-DD",
    "train_target_start": "YYYY-MM-DD",
    "train_target_end": "YYYY-MM-DD",
    "validation_target_start": "YYYY-MM-DD",
    "validation_target_end": "YYYY-MM-DD",
    "test_target_start": "YYYY-MM-DD",
    "test_target_end": "YYYY-MM-DD"
  },
  "target_definition": {
    "name": "target_next_day_up",
    "positive_class_condition": "close[target_date] > close[feature_date]",
    "class_labels": {"0": "down_or_flat", "1": "up"}
  },
  "class_mapping": {
    "estimator_classes": [0, 1],
    "probability_fields_by_class": {
      "0": "probability_down",
      "1": "probability_up"
    }
  }
}
```

`feature_set.columns` is an ordered, duplicate-free subset of the existing
PAISA technical feature set. Training and manifest validation both reject
future/label fields (`next_*`, `target_*`, and `actual_class`) and audit or
identity metadata such as dates, ticker, and source.

Every target-date split is non-empty, internally ordered, non-overlapping, and
strictly later than the preceding split. Each of `train`, `validation`, and
`test` under `metrics` requires numeric `accuracy`, `precision`, `recall`, and
`f1`, integer `support`, `positive_label: 1`, `average: "binary"`, and
`zero_division: 0`. Precision, recall, and F1 are therefore recorded as zero
when their denominator is zero.

Balanced accuracy, Matthews correlation coefficient, and ROC-AUC require both
actual classes to be observed. When a split contains one class, these fields
are `null`, `selection_eligible` is `false`, and `metric_warnings` contains
`single_class_actuals`; ordinary accuracy is never substituted for balanced
accuracy. A provisional champion may be selected only when the validation
split contains both classes. If no validation result is eligible, training
fails closed instead of consulting test metrics or reporting a misleading
champion.

`data_provenance` requires `provider`, `requested_symbols`, `included_symbols`,
`requested_start`, `requested_end`, `actual_feature_start`,
`actual_feature_end`, `dataset_sha256`, `input_schema_version`,
`price_adjustment` (`adjusted`, `unadjusted`, or `unknown`),
`dividends_included` (boolean), and `corporate_actions_policy`. The current PSX
source does not establish an adjusted-price series and the sprint does not model
dividends or splits, so manifests must record `price_adjustment: "unknown"`,
`dividends_included: false`, and `corporate_actions_policy: "not_modeled"` unless
the provider is independently verified. This limitation must accompany any
buy-and-hold interpretation.

The `model_id` must change whenever the algorithm, configuration, ordered
feature set, training data fingerprint/window, target/horizon, or seed changes.
A random UUID is acceptable only if the immutable manifest carries all of that
provenance. Loading executable formats such as joblib is permitted only from a
trusted local registry entry whose checksum matches. The resolved model
directory must remain below the configured registry root; model directories,
manifests, artifact binaries, and nested artifact path components must not be
symlinks. The loader hashes and deserializes through one already-open file
handle so path replacement cannot swap in a different file between those
operations.

## B. Prediction record contract

Prediction records are stored in a tabular format with one row per model,
ticker, feature date, and target date.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Initially `"1.0"`. |
| `model_id` | string | Foreign key to one immutable model manifest. |
| `ticker` | string | Uppercase symbol; initially `OGDC`. |
| `feature_date` | date | Session whose information produced the features. |
| `target_date` | date | Next observed trading session evaluated by the target. Must be later than `feature_date`. |
| `predicted_class` | integer enum | `0` for down-or-flat or `1` for up, decoded through `class_mapping`. |
| `probability_up` | number | Probability for class `1`, in `[0, 1]`. |
| `probability_down` | number | Probability for class `0`, in `[0, 1]`. |
| `actual_class` | nullable integer enum | `0` or `1` once known; otherwise `null`. Evaluation-only field. |

The uniqueness key is
`(model_id, ticker, feature_date, target_date)`. Probabilities come from the
estimator's declared class order, must be finite, and must satisfy
`abs(probability_up + probability_down - 1) <= 1e-6`. `predicted_class` is `1`
when `probability_up > probability_down` and `0` otherwise, so an exact tie maps
to down-or-flat. This model-level class is independent of the experiment's
trading thresholds. A historical experiment first generates predictions
without consulting labels, then joins `actual_class` in a separate evaluation
step before writing its final immutable prediction table.

## C. Experiment configuration contract

An experiment configuration is immutable JSON and binds a frozen model to one
decision policy and one independent portfolio.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Initially `"1.0"`. |
| `experiment_id` | string | Globally unique immutable experiment identifier. |
| `model_id` | string | Foreign key to an existing model artifact. |
| `portfolio_id` | string | Unique strategy-ledger identifier used by this experiment and no other. |
| `benchmark_portfolio_ids` | object | Unique `cash_only` and `buy_and_hold` ledger IDs owned by this experiment. |
| `symbol` | string | `OGDC` for the initial slice. |
| `initial_cash` | PKR decimal | Strictly positive opening cash. |
| `thresholds` | object | Exact probability-to-action rule defined below. |
| `maximum_allocation` | number | Maximum fraction of equity invested, in `(0, 1]`. |
| `transaction_cost` | object | Exact non-negative proportional fee rule defined below. |
| `evaluation_dates` | object | Exact inclusive feature- and target-date window defined below. |
| `execution_policy` | object | Exact timing, price, slippage, allocation, rounding, valuation, and ending-position rules defined below. |
| `metric_assumptions` | object | Exact annualization factor and annual risk-free rate used by result metrics. |
| `created_at` | UTC timestamp | Registration time. |

Schema `1.0` requires these nested keys:

```json
{
  "thresholds": {
    "buy_probability_up_gte": 0.60,
    "sell_probability_up_lte": 0.40
  },
  "benchmark_portfolio_ids": {
    "cash_only": "unique-cash-benchmark-id",
    "buy_and_hold": "unique-buy-hold-benchmark-id"
  },
  "transaction_cost": {
    "type": "proportional_bps",
    "value": 10
  },
  "evaluation_dates": {
    "feature_start": "YYYY-MM-DD",
    "feature_end": "YYYY-MM-DD",
    "target_start": "YYYY-MM-DD",
    "target_end": "YYYY-MM-DD"
  },
  "execution_policy": {
    "signal_timing": "after_close",
    "execution_session": "next_trading_session",
    "reference_price_field": "open",
    "slippage_bps": 0,
    "missing_execution_price": "fail",
    "allocation_mode": "state_transition_to_maximum",
    "whole_share_rounding": "floor",
    "valuation_price_field": "close",
    "missing_valuation_price": "fail",
    "ending_position": "mark_to_market"
  },
  "metric_assumptions": {
    "annualization_factor": 252,
    "risk_free_rate_annual": 0.0
  }
}
```

Both thresholds lie in `[0, 1]`, and the sell threshold is strictly below the
buy threshold. All three portfolio IDs are distinct and unused by any other
experiment. `feature_start <= feature_end`, `target_start <= target_end`, and
each feature row's target lies inside the declared target window.
`slippage_bps` lies in `[0, 10_000)`, `annualization_factor` is positive, and
`risk_free_rate_annual > -1`.

The initial policy is deterministic and long-only:

- When shares are zero and `probability_up` reaches the buy threshold, buy once
  toward `maximum_allocation`. While shares are positive, further buy signals do
  not rebalance or add shares.
- When shares are positive and `probability_up` reaches the sell threshold,
  sell all shares. A sell signal while already in cash does nothing. All other
  cases hold and create no trade record.
- For a buy, `fill_price` is the next-session open multiplied by
  `1 + slippage_bps / 10_000`; for a sell it is multiplied by
  `1 - slippage_bps / 10_000`. Quantize the fill price to PKR 0.01 using
  `ROUND_HALF_UP` and store it as `execution_price`.
- With fee rate `f = transaction_cost.value / 10_000`, choose the greatest whole
  quantity no larger than both
  `floor(cash / (fill_price * (1 + f)))` and
  `floor(maximum_allocation * pre_trade_equity / fill_price)`. Calculate
  `fee = quantity * fill_price * f`, quantize the fee to PKR 0.01, and decrement
  quantity if rounding would make cash negative. A zero quantity creates no
  trade. For sizing and compliance, allocation means
  `fill_notional / pre_trade_equity`; later price movement may move the marked
  position above the original fraction and does not trigger rebalancing.
- The schema `1.0` provider policy fails the experiment on a missing next-session
  open or valuation close; it never substitutes or backfills a later price.

Thresholds control the decision policy, not the model's stored
`predicted_class`. Portfolio code may not inspect `actual_class`.

## D. Portfolio snapshot contract

One snapshot records the independently owned ledger after that session's
eligible execution and end-of-session valuation.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Initially `"1.0"`. |
| `portfolio_id` | string | Foreign key to the strategy or benchmark portfolio ID registered in the experiment configuration. |
| `date` | date | Trading session represented by the snapshot. |
| `cash` | PKR decimal | Non-negative cash after any trade and fee. |
| `shares` | integer | Non-negative whole shares held after any trade. |
| `market_price` | PKR decimal | Finite, positive close used for end-of-session valuation. |
| `position_value` | PKR decimal | Exactly `shares * market_price`, quantized to PKR 0.01. |
| `total_equity` | PKR decimal | Exactly `cash + position_value` at accounting precision. |
| `realized_profit_loss` | PKR decimal | Cumulative net P/L from closed position episodes, including buy and sell fees. |
| `unrealized_profit_loss` | PKR decimal | Mark-to-market P/L on open inventory relative to its fee-aware cost basis. |

The uniqueness key is `(portfolio_id, date)`. Snapshots are append-only and
strictly chronological. Write an opening, pre-trade snapshot on
`evaluation_dates.feature_start` with `cash = total_equity = initial_cash`, zero
shares/position/P&L, and that session's valuation close. Then write exactly one
post-execution, end-of-session snapshot for every later observed trading session
through `evaluation_dates.target_end`, including no-trade days. Every later
snapshot reconciles from the prior state through that session's recorded trade,
if any. Cross-portfolio balances or holdings are forbidden.

Schema `1.0` permits only full position episodes: one buy from zero shares and a
later sell of all shares; partial exits and adding to an open position are not
supported. The open-position cost basis is
`buy_quantity * buy_execution_price + buy_fee`. While open, unrealized P/L is
`position_value - open_cost_basis`.
When the position closes, episode P/L is `sell notional - sell fee - open cost
basis`; add it to cumulative realized P/L and reset unrealized P/L to zero. An
episode is winning only when this net value is greater than zero.

## E. Trade record contract

A trade record represents an executed order. `HOLD` is not a trade and does not
create a row.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Initially `"1.0"`. |
| `trade_id` | string | Globally unique immutable trade identifier. |
| `portfolio_id` | string | Foreign key to exactly one strategy or benchmark ledger registered in the experiment configuration. |
| `signal_date` | date | Feature session after whose close the decision was made. |
| `execution_date` | date | Observed trading session on which execution occurs; strictly later than `signal_date`. |
| `action` | string enum | `BUY` or `SELL`. |
| `quantity` | integer | Strictly positive whole-share quantity. |
| `reference_price` | PKR decimal | Unmodified next-session price from the configured source field. |
| `execution_price` | PKR decimal | Finite, positive slippage-adjusted fill price before fee. |
| `fee` | PKR decimal | Non-negative transaction cost charged for this trade. |
| `cash_after_trade` | PKR decimal | Non-negative reconciled ledger cash immediately after execution. |
| `shares_after_trade` | integer | Non-negative reconciled holdings immediately after execution. |

For a buy, cash decreases by `quantity * execution_price + fee`; for a sell, it
increases by `quantity * execution_price - fee`. Quantize notional and the final
cash balance to PKR 0.01 with `ROUND_HALF_UP` after the already quantized fill
price and fee are applied. A trade may not create negative cash, sell more
shares than are held, or make fill notional exceed
`maximum_allocation * pre_trade_equity` after rounding.

## F. Experiment result contract

The result is immutable JSON derived from the experiment's persisted
predictions and portfolio ledger.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Initially `"1.0"`. |
| `experiment_id` | string | Foreign key to the immutable configuration. |
| `model_id` | string | Copied foreign key for direct comparison. |
| `portfolio_id` | string | Copied foreign key to the independently reconciled ledger. |
| `prediction_metrics` | object | Exact metric structure below, computed only on rows whose `actual_class` is known. |
| `ending_equity` | PKR decimal | Final snapshot's `total_equity`. |
| `total_return` | number | `(ending_equity / initial_cash) - 1`. |
| `maximum_drawdown` | number | Non-negative worst peak-to-trough fraction: `max(1 - equity / running_peak)`. |
| `volatility` | nullable number | Sample standard deviation of daily portfolio returns times `sqrt(annualization_factor)`; `null` when fewer than two returns exist. |
| `sharpe_ratio` | nullable number | Annualized excess daily return divided by daily-return standard deviation, using the recorded annual risk-free rate; `null` for insufficient rows or zero volatility. |
| `number_of_trades` | integer | Count of executed `BUY` and `SELL` trade rows; never counts hold signals. |
| `win_rate` | nullable number | Profitable fully closed position episodes divided by all fully closed episodes, net of fees; `null` when none close. |
| `benchmark_comparison` | object | Exact cash-only/buy-and-hold identities, artifact paths, values, and differences defined below. |

`prediction_metrics` requires numeric `accuracy`, `precision`, `recall`, and
`f1`, integer `support`, `positive_label: 1`, `average: "binary"`, and
`zero_division: 0`. It uses the experiment evaluation window only and does not
replace the frozen model manifest's split metrics.

`benchmark_comparison` requires this shape (PKR fields use fixed-scale strings):

```json
{
  "cash_only": {
    "portfolio_id": "unique-benchmark-portfolio-id",
    "snapshots_path": "relative/path/to/cash_only_snapshots.csv",
    "ending_equity": "1000000.00",
    "total_return": 0.0
  },
  "buy_and_hold": {
    "portfolio_id": "unique-benchmark-portfolio-id",
    "snapshots_path": "relative/path/to/buy_and_hold_snapshots.csv",
    "trades_path": "relative/path/to/buy_and_hold_trades.csv",
    "ending_equity": "1000000.00",
    "total_return": 0.0
  },
  "excess_return_vs_cash_only": 0.0,
  "excess_return_vs_buy_and_hold": 0.0
}
```

Each nested benchmark `portfolio_id` must equal the corresponding value in the
experiment configuration. Return differences use the strategy as the left-hand
side: `excess_return_vs_X = experiment.total_return - X.total_return`.

For consecutive chronological snapshot equities `E_t`, daily simple return is
`r_t = E_t / E_(t-1) - 1`. Let `N` be the recorded annualization factor
(`252` for this sprint), `rf_annual` the recorded annual risk-free rate, and
`rf_daily = (1 + rf_annual)^(1/N) - 1`. Volatility is
`sample_stddev(r_t, ddof=1) * sqrt(N)`. Sharpe is
`sqrt(N) * (mean(r_t) - rf_daily) / sample_stddev(r_t, ddof=1)`. Volatility and
Sharpe are `null` with fewer than two returns; Sharpe is also `null` when sample
standard deviation is zero. Maximum drawdown is zero for a one-snapshot series.

Cash-only holds `initial_cash`, places no trades, and has zero return unless a
future explicit interest policy says otherwise. It persists one snapshot per
evaluation session using a unique benchmark `portfolio_id`. Buy-and-hold uses a
different unique benchmark `portfolio_id`, purchases as many whole shares as
allowed by that experiment's `maximum_allocation` on the first eligible
execution session, pays the same transaction-cost/slippage rule, then holds and
persists snapshots through the final date. It is not silently force-liquidated
at the end. Because maximum allocation is part of a configuration, experiments
with different maximum allocations intentionally have distinct buy-and-hold
benchmarks. Both benchmark curves follow the Portfolio Snapshot contract and are
available to the API/dashboard through the recorded paths.

## Existing v0 migration gaps

The contracts above describe new versioned paths; they do not claim the current
v0 implementation is already compliant. Future implementation branches must
address these gaps additively:

- `features.py` currently calculates the next-day label but does not persist a
  `target_date`. Add that column without removing current v0 columns.
- `chronological_split()` currently partitions on the feature `date`. Do not use
  it unchanged for registered experiments; split on persisted `target_date` or
  implement and test the required purge.
- The sample provider seeds data with Python's process-randomized `hash(symbol)`.
  Registered experiments need a stable seed derived from a recorded hash such
  as SHA-256; keep any compatibility behavior isolated to v0.
- Existing fixed-name artifacts overwrite on a v0 rerun. New training code must
  write the immutable model directory first and may not treat a fixed-name v0
  file as the registered artifact.

## Required validation and tests

Implementations of these contracts must test at least:

- Schema validation, enum/range checks, referential integrity, and deterministic
  serialization.
- Collision refusal for existing model and experiment IDs.
- Ordered feature identity, data fingerprints, checksum verification, and
  explicit estimator class mapping.
- Chronological target-date split membership and boundary purging.
- Training/validation-only selection and a single post-freeze test evaluation.
- `feature_date < target_date`, `signal_date < execution_date`, and absence of
  label access in policy/execution code.
- Threshold boundary behavior and missing-price behavior.
- Independent state for multiple portfolios using the same model.
- Cash/share/trade/snapshot reconciliation, fees, maximum allocation, and
  insufficient-cash cases.
- Every metric formula, empty/constant-series behavior, and benchmark date/cost
  alignment.
- Backward compatibility for existing v0 tests and API routes.
