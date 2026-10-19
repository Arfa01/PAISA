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
2. Check `data/processed/psx_prices.csv` for existing OGDC rows.
3. Reject the file if OGDC is absent, the required daily price fields are
   invalid, or its provenance identifies sample, fixture, or synthetic data.
4. If the cached rows qualify as real data, record that cached-real fallback
   was used, preserve the live-provider error, and document the actual first
   and last OGDC dates.
5. Fail the workflow if no qualifying cached-real OGDC data is available.

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

A model or backtest failure prevents a successful final status. The command
prints a final `PASS` only after all four models, all portfolio simulations,
benchmarks, combined tables, and plots have completed and passed their
consistency checks.

## Output artifacts

A successful run publishes this comparison bundle:

```text
artifacts/vertical_slice/<vertical_slice_run_id>/
├── manifest.json
├── model_training_run_reference.json
├── experiment_configs/
│   ├── M-001.json
│   ├── M-002.json
│   ├── M-003.json
│   └── M-004.json
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
- `experiment_configs/` contains one normalized, immutable configuration per
  model portfolio.
- `model_metrics.csv` keeps the train, validation, and untouched-test
  prediction evidence separate from portfolio outcomes.
- `portfolio_metrics.csv` contains one row for each of the four model
  strategies plus the cash-only and buy-and-hold comparison portfolios.
- `combined_leaderboard.csv` joins model context to strategy outcomes without
  treating test accuracy as the portfolio-selection rule.
- `daily_equity_curves.csv` is the dated source table used for the combined
  equity and drawdown charts.
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
- Cached-file qualification validates the standardized source field, OGDC
  coverage, dates, and numeric market fields. It cannot independently
  authenticate a deliberately mislabeled local CSV, so the cache remains a
  trusted local input and its SHA-256 is recorded for auditability.
- The run covers one stock, one daily horizon, one historical split, fixed
  thresholds, and one controlled cost model. It is not evidence of live
  execution quality or future returns.
- Reported Sharpe ratio, volatility, drawdown, fees, and trade count describe
  the simulated test window only.
- No API, dashboard, news, sentiment, LSTM promotion, SHAP analysis, or
  reinforcement-learning component is part of this command.

All generated results are academic simulations and are not financial advice.
