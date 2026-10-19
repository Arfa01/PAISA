# PAISA OGDC Backtesting

The versioned backtesting path consumes standardized model predictions and
daily market prices. It does not train a model. Each experiment owns an
independent strategy portfolio plus independent cash-only and buy-and-hold
benchmark portfolios.

All predictions, signals, trades, metrics, and charts are academic simulations
and are not financial advice.

## Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements-backtesting.txt` adds Matplotlib for deterministic, headless PNG
reports. The accounting engine itself uses the existing core dependencies.

## Run an experiment

```bash
python scripts/run_backtest.py \
  --predictions path/to/predictions.csv \
  --prices path/to/OGDC_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

`--runs-dir` is optional and defaults to `runs/`. The output is written to
`<runs-dir>/<experiment_id>/`. The command fails rather than replace an
existing experiment directory. To run the same inputs again, choose a new
immutable `experiment_id` and new strategy and benchmark portfolio IDs.

## Prediction input

`predictions.csv` follows the prediction-record contract in
[ARCHITECTURE_CONTRACTS.md](ARCHITECTURE_CONTRACTS.md). It has one row per
model, ticker, feature date, and target date, in this column order:

| Column | Type | Rule |
|---|---|---|
| `schema_version` | string | Must be `1.0`. |
| `model_id` | string | Must equal the configuration's immutable model ID. |
| `ticker` | string | Uppercase; `OGDC` for the first slice. |
| `feature_date` | date | ISO `YYYY-MM-DD`; the session whose close completes the features. |
| `target_date` | date | ISO `YYYY-MM-DD`; the next observed trading session after `feature_date`. |
| `predicted_class` | integer | `0` for down-or-flat or `1` for up. |
| `probability_up` | number | Finite value in `[0, 1]`. |
| `probability_down` | number | Finite value in `[0, 1]`; the two probabilities sum to one within `1e-6`. |
| `actual_class` | nullable integer | `0`, `1`, or empty; used only for prediction evaluation. |

The uniqueness key is
`(model_id, ticker, feature_date, target_date)`. Duplicate keys are invalid and
are not silently deduplicated. Rows are processed in feature-date order. The
policy and execution engine never inspect `actual_class`.

Example:

```csv
schema_version,model_id,ticker,feature_date,target_date,predicted_class,probability_up,probability_down,actual_class
1.0,M-001,OGDC,2026-09-01,2026-09-02,1,0.670000,0.330000,1
1.0,M-001,OGDC,2026-09-02,2026-09-03,0,0.350000,0.650000,0
```

The runner verifies that every prediction's `model_id` matches the immutable
ID in the experiment configuration. It deliberately does not load executable
model binaries. Until the versioned model-registry branch is integrated, the
producer of a prediction file is responsible for resolving that ID to its
trusted model manifest; this is the remaining model/backtest integration
boundary, not permission to relabel an existing run.

## Price input

The price input uses the existing PAISA daily price schema. It may be an
OGDC-only extract or the canonical multi-symbol `psx_prices.csv`; the runner
selects the configured symbol before duplicate-session and evaluation-window
validation. Backtesting requires these columns:

| Column | Type | Rule |
|---|---|---|
| `date` | date | ISO `YYYY-MM-DD`, strictly chronological after sorting. |
| `ticker` | string | Uppercase; rows for the configured symbol are selected. |
| `open` | PKR decimal | Finite and positive; used for next-session execution. |
| `close` | PKR decimal | Finite and positive; used for end-of-session valuation. |

The existing optional columns `high`, `low`, `volume`, and `source` may remain
in the file. There must be exactly one row for each `(ticker, date)`. Duplicate
market dates, missing next-session opens, and missing valuation closes are
errors. The file must cover every observed trading session in the configured
evaluation window.

Example:

```csv
date,ticker,open,high,low,close,volume,source
2026-09-01,OGDC,225.00,,,226.25,1000000,fixture
2026-09-02,OGDC,227.00,,,228.50,1100000,fixture
2026-09-03,OGDC,226.50,,,225.75,900000,fixture
```

## Experiment configuration

The JSON configuration follows the experiment contract. Monetary values are
fixed-scale PKR strings. Rates and allocation values are decimal fractions;
transaction cost and slippage values are basis points.

```json
{
  "schema_version": "1.0",
  "experiment_id": "ogdc-m001-60-40-v1",
  "model_id": "M-001",
  "portfolio_id": "ogdc-m001-60-40-strategy-v1",
  "benchmark_portfolio_ids": {
    "cash_only": "ogdc-m001-60-40-cash-v1",
    "buy_and_hold": "ogdc-m001-60-40-buy-hold-v1"
  },
  "symbol": "OGDC",
  "initial_cash": "100000.00",
  "thresholds": {
    "buy_probability_up_gte": 0.60,
    "sell_probability_up_lte": 0.40
  },
  "maximum_allocation": 0.80,
  "transaction_cost": {
    "type": "proportional_bps",
    "value": 10
  },
  "evaluation_dates": {
    "feature_start": "2026-09-01",
    "feature_end": "2026-09-29",
    "target_start": "2026-09-02",
    "target_end": "2026-09-30"
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
  },
  "created_at": "2026-10-19T09:00:00Z"
}
```

All three portfolio IDs must be distinct and unused by another experiment.
The sell threshold must be lower than the buy threshold. `initial_cash` must be
positive, `maximum_allocation` must be in `(0, 1]`, and fees and slippage must
be non-negative.

## Timing and leakage rules

For a prediction made from trading session `t`:

1. Features use information available no later than the close of
   `feature_date`.
2. The confidence policy creates a signal after that close.
3. `target_date` must be the next observed trading session for OGDC.
4. An eligible order executes on `target_date` at its open, adjusted for the
   configured slippage. It never executes on `feature_date`.
5. The portfolio is valued at that session's close.
6. `actual_class`, when present, is joined for evaluation only and cannot affect
   a signal, quantity, fill, or balance.

The engine does not backfill a missing execution or valuation price from a
later session. It fails the experiment instead.

## Policies and accounting

### Confidence threshold policy

- Emit `BUY` when `probability_up` is greater than or equal to
  `buy_probability_up_gte`.
- Emit `SELL` when `probability_up` is less than or equal to
  `sell_probability_up_lte`.
- Emit `HOLD` otherwise.

The boundary comparisons are inclusive. `HOLD` is a signal but is not a trade.
A sell signal while the portfolio is already in cash also creates no trade.

### Target-allocation policy

The schema `1.0` policy is a long-only state transition:

- A `BUY` from zero shares moves toward `maximum_allocation` of pre-trade
  equity.
- Repeated `BUY` signals while shares are already held do not add shares or
  rebalance the position.
- A `SELL` closes the complete position and moves its target allocation to
  zero.
- Quantities are non-negative whole shares. There is no borrowing, short
  selling, or fractional-share execution.

For a buy, the execution price is the next-session open multiplied by
`1 + slippage_bps / 10_000`. For a sell, it is multiplied by
`1 - slippage_bps / 10_000`. The execution price and fees are rounded to PKR
0.01 using `ROUND_HALF_UP`.

With fee rate `f = transaction_cost.value / 10_000`, a buy uses the greatest
whole quantity no larger than both:

```text
floor(cash / (execution_price * (1 + f)))
floor(maximum_allocation * pre_trade_equity / execution_price)
```

The fee is `quantity * execution_price * f`. Quantity is reduced if monetary
rounding would otherwise make cash negative. A zero quantity produces no trade.
A sell can never exceed the shares held.

### Benchmarks

- **Cash only:** retains the same initial cash, places no trades, and writes one
  snapshot per evaluation session.
- **Buy and hold:** on the first eligible execution session, buys as many whole
  shares as the same maximum-allocation, slippage, and fee settings permit,
  then holds through the final close. It is not force-liquidated.

Both benchmarks use the same market observations and valuation dates as the
strategy and own separate portfolio IDs and ledgers.

## Metrics

Metrics use chronologically ordered end-of-session snapshots:

| Metric | Definition |
|---|---|
| `initial_equity` | Opening snapshot equity. |
| `ending_equity` | Final snapshot equity. |
| `total_return` | `ending_equity / initial_equity - 1`. |
| `daily_returns` | For consecutive equities, `E_t / E_(t-1) - 1`. |
| `maximum_drawdown` | Largest non-negative `1 - equity / running_peak`. |
| `volatility` | Sample standard deviation of daily returns (`ddof=1`) multiplied by the square root of the annualization factor; `null` with fewer than two returns. |
| `sharpe_ratio` | `sqrt(N) * (mean(daily_return) - rf_daily) / sample_stddev(daily_return)`, where `rf_daily = (1 + rf_annual)^(1/N) - 1`; `null` for insufficient returns or zero standard deviation. |
| `number_of_trades` | Count of executed `BUY` and `SELL` rows; holds are excluded. |
| `win_rate` | Positive net closed-position episodes divided by all closed episodes; `null` when no episode closes. |
| `average_gain` | Mean net PKR profit among profitable closed episodes; `null` when there are none. |
| `average_loss` | Mean net PKR loss among losing closed episodes, retained as a negative value; `null` when there are none. |
| `turnover` | Sum of absolute executed fill notional divided by mean daily total equity. |
| `fees_paid` | Sum of all strategy trade fees. |
| `time_in_market` | Fraction of post-opening evaluation snapshots whose share balance is greater than zero; zero when there are no such sessions. |
| `comparison_with_buy_and_hold` | Strategy total return minus buy-and-hold total return. |

PKR fields are persisted as strings with exactly two fractional digits. Win
accounting is net of both buy and sell fees. The result also includes the
contracted prediction metrics on rows with a known `actual_class` and the cash
and buy-and-hold identities, paths, ending equities, returns, and excess-return
comparisons.

## Output artifacts

A successful run produces this immutable tree:

```text
runs/<experiment_id>/
├── config.json
├── predictions.csv
├── signals.csv
├── trades.csv
├── portfolio_snapshots.csv
├── metrics.json
├── result.json
├── equity_curve.png
├── drawdown.png
└── benchmarks/
    ├── cash_only_snapshots.csv
    ├── buy_and_hold_snapshots.csv
    └── buy_and_hold_trades.csv
```

`config.json` is the normalized immutable configuration. `predictions.csv` is
the validated experiment input. `signals.csv` records decisions, including
holds; `trades.csv` records only executions. `portfolio_snapshots.csv` contains
one reconciled row per evaluation session. `metrics.json` contains the complete
experiment result and benchmark comparison. `result.json` contains the same
deterministic payload under the stable name defined by the architecture
contract.

`equity_curve.png` compares the strategy, cash-only, and buy-and-hold curves.
`drawdown.png` presents their drawdowns. Charts use the same persisted snapshots
as the metrics and carry the academic-simulation disclaimer.

Files are written in deterministic column and row order. A run is published
only after every artifact has been produced successfully. A completed directory
is immutable: an existing experiment ID, experiment path, or reused portfolio
ID is a hard error, never an overwrite prompt. Short-lived exclusive identity
reservations also prevent concurrent processes from claiming the same
experiment or portfolio ID before either has finished publishing.

Generated run artifacts are ignored by Git. Commit only small deterministic
fixtures under `tests/fixtures/`; do not commit real prediction histories,
portfolio outputs, or charts.
