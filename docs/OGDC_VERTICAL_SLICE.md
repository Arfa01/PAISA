# OGDC end-to-end vertical slice

The OGDC vertical-slice command proves one complete path through PAISA's
versioned components:

```text
real daily PSX OGDC prices
  -> technical features and next-session labels
  -> M-001, M-002, M-003, and M-004 training
  -> untouched test prediction records
  -> four independent strategy portfolios
  -> cash-only and buy-and-hold benchmarks
  -> one model-and-portfolio comparison report
```

This workflow is an academic simulation. Its predictions, signals, trades,
portfolio curves, benchmark comparisons, and simulated returns are not
financial advice.

## Install and validate

Python 3.10 or later is required. `requirements.txt` includes the normal data
pipeline dependencies, the Random Forest and XGBoost dependencies, Matplotlib
for headless charts, and the test dependencies. The optional `pypsx-toolkit`
package and optional LSTM dependencies are not needed.

From the repository root, create a clean environment and validate it before a
live run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest
python -m pip check
```

Run the full workflow with:

```bash
python scripts/run_ogdc_vertical_slice.py
```

Use `python scripts/run_ogdc_vertical_slice.py --help` to inspect any explicit
path or run-identity overrides supported by the current command. Generated
data, fitted models, portfolio records, and plots remain runtime artifacts and
must not be committed.

## Fixed experiment assumptions

Every model portfolio receives the same evaluation observations and execution
rules. Only the fitted model and its probabilities may differ.

| Assumption | Value |
|---|---|
| Symbol and frequency | `OGDC`, daily observed PSX sessions |
| Initial cash | PKR 100,000 per independent portfolio |
| Buy rule | `probability_up >= 0.60` |
| Sell rule | `probability_up <= 0.40` |
| Neutral band | Hold when `0.40 < probability_up < 0.60` |
| Position direction | Long only |
| Maximum allocation | `0.50` of pre-trade equity |
| Quantity | Whole shares, rounded down |
| Transaction cost | 10 basis points per execution |
| Slippage | 0 basis points for the controlled run |
| Signal time | After feature-date close |
| Execution | Next observed trading-session open |
| Valuation | Every observed session's close |
| Final position | Mark to market; no forced liquidation |
| Metric annualization | 252 sessions; 0% annual risk-free rate |

The runner verifies that all four prediction files have identical ordered
feature and target dates, and that all portfolios receive identical OGDC open
and close observations. Each strategy uses its own immutable experiment and
portfolio identity; cash, shares, trades, and snapshots are never shared
between model experiments.

## Live-data and cached-real fallback rules

The command first requests OGDC history from the real `psx-dps` provider. A
sample provider, generated fixture, synthetic series, or silent provider switch
is never a valid substitute for a live vertical-slice run.

If the live provider is unavailable, the command must:

1. Stop the live acquisition attempt and retain the complete provider error.
2. Require `pipeline_metadata.json` beside the configured cached price CSV.
   The CSV and metadata are one provenance pair; neither is accepted alone.
3. Require the metadata's completed `prices_artifact` record to identify the
   exact CSV filename and SHA-256, the real provider, daily frequency, total
   row count, and per-symbol row count and date coverage. Its OGDC coverage and
   source must match the rows actually read. The metadata must also identify a
   completed `paisa.pipeline.run_v0_pipeline` producer, contain a valid
   generation timestamp and transformation counts, and record positive raw
   acquisition rows for OGDC without an OGDC provider error.
4. Reject the pair if OGDC is absent, required daily price fields are invalid,
   any source value is null or blank, the file mixes providers, the provider is
   sample/fixture/synthetic, or any checksum, row-count, provider, or coverage
   claim disagrees with the CSV. Merely relabelling fixture rows is therefore
   insufficient because their bytes no longer match the producer record.
5. If the cached pair qualifies as real data, record that cached-real fallback
   was used, preserve the live-provider error, and document the actual first
   and last OGDC dates.
6. Fail the workflow if no qualifying cached-real provenance pair is
   available.

Before these checks, the runner copies the selected price CSV and its metadata
to an invocation-private working snapshot. Validation and model training both
consume that snapshot, not the mutable original cache. The runner verifies the
snapshot fingerprint around training so an external cache edit cannot create
a time-of-check/time-of-use mismatch. The snapshot is private while the run is
being built, but it is not discarded after success: the exact CSV and, for a
real-data run, its `pipeline_metadata.json` remain in the immutable published
bundle under `input_snapshot/`. The manifest checksum inventory and final
report record this input evidence and its price and provenance fingerprints.

The final summary identifies the effective source and date range. A cached run
must not be described as a successful live fetch, and fixture-based tests must
never be labeled as real-data runs.

## Immutable run behavior

Every invocation receives a unique `vertical_slice_run_id`. A completed output
directory, training run, model artifact, experiment ID, or portfolio ID is
immutable. If an intended destination or identity already exists, the command
fails instead of replacing it; rerun with a new identity.

The four model-training variants are:

- `M-001`: Random Forest using the PAISA v0 baseline configuration.
- `M-002`: conservative Random Forest configuration.
- `M-003`: flexible Random Forest configuration.
- `M-004`: XGBoost using the deterministic CPU histogram configuration.

`M-001` through `M-004` are stable variant labels. Every fitted artifact also
has a longer immutable registered model ID. Prediction records and backtest
configurations use that registered ID; reports retain the stable variant label
so the four rows remain recognizable.

A model or backtest failure prevents a successful final status. The model run
is first written to an invocation-owned pending location and is not published
under its immutable public run ID until all four backtests, their persisted
ledgers, benchmarks, combined tables, and plots have completed and passed their
consistency checks. A handled late failure removes the invocation's pending or
newly published model run along with the staged vertical bundle; it never
deletes a pre-existing run. The command prints a final `PASS` only after those
checks and publications succeed. The frozen training-price file and all four
prediction files are fingerprinted before backtesting, rechecked after every
experiment has been reconciled, and rechecked again immediately before
publication. Any in-place mutation aborts the transaction.

## Output artifacts

A successful run publishes this comparison bundle:

```text
artifacts/vertical_slice/<vertical_slice_run_id>/
├── manifest.json
├── model_training_run_reference.json
├── input_snapshot/
│   ├── <price_csv_filename>.csv
│   └── pipeline_metadata.json  # real-data runs
├── experiment_configs/
│   ├── M-001.json
│   ├── M-002.json
│   ├── M-003.json
│   └── M-004.json
├── experiments/
│   └── <experiment_id>/
│       ├── config.json
│       ├── predictions.csv
│       ├── signals.csv
│       ├── trades.csv
│       ├── portfolio_snapshots.csv
│       ├── metrics.json
│       ├── result.json
│       ├── equity_curve.png
│       ├── drawdown.png
│       └── benchmarks/
│           ├── cash_only_snapshots.csv
│           ├── buy_and_hold_snapshots.csv
│           └── buy_and_hold_trades.csv
├── model_metrics.csv
├── portfolio_metrics.csv
├── combined_leaderboard.csv
├── daily_equity_curves.csv
├── model_metric_comparison.png
├── portfolio_equity_comparison.png
├── drawdown_comparison.png
└── final_summary.json
```

The files have these roles:

- `manifest.json` records the vertical-slice identity, status, common
  assumptions, data provenance and fingerprint, date coverage, participating
  model and portfolio identities, common split evidence, and generated file
  references.
- `model_training_run_reference.json` points to the immutable model-training
  run and its four registered model artifacts and prediction files.
- `input_snapshot/` retains the exact price CSV that was snapshotted before
  validation and consumed by training. Real-data runs also retain the matching
  producer-written `pipeline_metadata.json`; fixture-only tests may omit it.
  These are immutable evidence for reproducing which input bytes passed the
  producer-attestation and integrity checks and were used; both are covered by
  the manifest when present.
- `experiment_configs/` contains one normalized, immutable configuration per
  model portfolio.
- `experiments/` retains the complete immutable backtest evidence for each of
  the four model portfolios: exact prediction input, generated signals, trade
  ledger, daily strategy snapshots, benchmark ledgers, result/metric records,
  and per-experiment plots. Each file is included in the vertical manifest's
  checksum inventory.
- `model_metrics.csv` keeps the train, validation, and untouched-test
  prediction evidence separate from portfolio outcomes.
- `portfolio_metrics.csv` contains one row for each of the four model
  strategies plus the cash-only and buy-and-hold comparison portfolios.
- `combined_leaderboard.csv` joins model context to strategy outcomes without
  treating test accuracy as the portfolio-selection rule.
- `daily_equity_curves.csv` is the dated source table used for the combined
  equity and drawdown charts.
- Before publication, the runner reloads the persisted experiment files,
  reconstructs their typed records, recomputes the result metrics, checks the
  strategy and benchmark identities and common market observations, and then
  builds the aggregate portfolio table and curves from that persisted evidence.
  A mismatch between `result.json`, `metrics.json`, or any retained ledger
  fails the run instead of publishing an inconsistent summary.
- The three PNG files compare model metrics, the six logical equity series,
  and their drawdowns. They are presentation artifacts; the CSV and JSON files
  remain the auditable source values.
- `final_summary.json` is published only for a completed `PASS` and records key
  run facts, separately named comparison results, warnings, and the
  academic-simulation disclaimer. A failed command prints a structured `FAIL`
  with the full error to standard error and publishes no partial run directory.

The combined leaderboard includes:

```text
model_id
algorithm
validation_balanced_accuracy
validation_f1
test_accuracy
test_balanced_accuracy
test_f1
ending_portfolio_value
total_return
maximum_drawdown
volatility
sharpe_ratio
number_of_trades
fees_paid
result_versus_buy_and_hold
```

Additional identity or audit columns, such as `variant_id`, `portfolio_id`, or
`experiment_id`, may be retained without changing those required meanings.
Monetary values are PKR amounts, return and risk values are decimal fractions,
and `result_versus_buy_and_hold` is the strategy total return minus the
buy-and-hold total return.

## How to read the comparison

The final report names separate results rather than one universal winner:

- The best validation prediction model is chosen using validation balanced
  accuracy, then validation F1, validation accuracy, and model ID as
  deterministic tie breakers. Test metrics are not used for that selection.
- The best test-period simulated portfolio is the model portfolio with the
  highest ending value or total return; these rankings are equivalent because
  every portfolio starts with the same cash.
- The lowest-drawdown model portfolio is the one with the smallest maximum
  drawdown.
- The buy-and-hold result is reported independently.
- The report states whether any model portfolio's total return exceeded
  buy-and-hold over the common test window.

Do not infer that the model with the best test accuracy must have the best
portfolio result. Trading thresholds, neutral signals, overnight price gaps,
entry timing, fees, position state, and the path of returns all affect the
simulation.

### The 50%-allocation benchmark

The same `maximum_allocation=0.50`, fee, slippage, whole-share, execution, and
valuation rules apply to the buy-and-hold benchmark. It buys up to half of
initial equity at the first eligible next-session open and leaves the remaining
cash uninvested through the test period. It is therefore a controlled
50%-allocated benchmark, not a conventional 100%-invested buy-and-hold return.
The cash-only benchmark remains at PKR 100,000 and makes no trades.

### Prediction target versus portfolio return

The current binary target is close to close:

```text
target_next_day_up = 1
when close[target_date] > close[feature_date], otherwise 0
```

Features and the signal become available only after `feature_date` closes. A
simulated trade cannot use that close; it executes at `target_date` open. The
portfolio then earns the price movement from its actual execution onward and
is valued at daily closes. Overnight movement between the feature close and
next open affects the available execution price but is not a realizable return
for a newly opened position in this simulation.

Consequently, prediction accuracy measures close-to-close directional
classification, while portfolio metrics measure a thresholded, stateful,
fee-aware strategy entered at next-session opens. They are related evidence,
not the same measurement, and the report must not claim otherwise.

## Limitations and interpretation

- PSX DPS supplies the fields needed here (`open`, `close`, and `volume`) but
  may omit `high` and `low`; the vertical slice does not invent them.
- Current provenance does not establish a fully adjusted total-return price
  series. Dividends and corporate actions are not modeled, so buy-and-hold and
  model returns must be interpreted with that limitation.
- Cached-file qualification depends on the producer-written
  `pipeline_metadata.json` and its checksum-bound `prices_artifact` entry. This
  detects post-production edits, stale metadata, mixed/blank sources, and
  simple relabelling, but cannot prove that an already-compromised producer or
  a coordinated replacement of both files reported truthful market data.
- The run covers one stock, one daily horizon, one historical split, fixed
  thresholds, and one controlled cost model. It is not evidence of live
  execution quality or future returns.
- Reported Sharpe ratio, volatility, drawdown, fees, and trade count describe
  the simulated test window only.
- No API, dashboard, news, sentiment, LSTM promotion, SHAP analysis, or
  reinforcement-learning component is part of this command.

All generated results are academic simulations and are not financial advice.
