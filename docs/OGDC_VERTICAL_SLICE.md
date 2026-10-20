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

If the live provider is unavailable, the command retains the complete provider
error and permits only one of the following checksum-bound evidence paths. The
two producer identities remain distinct.

### Current schema-1.0 pipeline cache

The existing path remains accepted unchanged. It requires the cached CSV and
its sibling schema-1.0 `pipeline_metadata.json`, both from the completed
`paisa.pipeline.run_v0_pipeline` producer. The runner must:

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

### Verified migration of the pre-schema PAISA cache

A pre-schema cache is not accepted directly. Migrate the specified legacy
prices and metadata into the standard destination with this exact command:

```bash
python scripts/migrate_legacy_psx_cache.py \
  --legacy-prices \
  "<legacy-repository>/data/processed/psx_prices.csv" \
  --legacy-metadata \
  "<legacy-repository>/data/processed/pipeline_metadata.json" \
  --legacy-raw-dir \
  "<legacy-repository>/data/raw" \
  --output-dir \
  "data/external/verified_legacy_psx_cache"
```

The migrator performs no network request or live acquisition. It computes the
source-file hashes before work, checks them again before publication, and never
edits the original CSV or metadata. The destination is immutable: if it already
exists, the command fails rather than overwriting or merging it.

The command automatically infers the legacy `data/raw/` directory from the
processed CSV and searches it recursively. Only exact
`<SYMBOL>_psx_dps_raw.csv` filenames count as raw captures. If a capture exists
for every symbol requested by the legacy metadata, the command rebuilds the
standard price table with the current cleaning logic and requires it to match
the validated legacy processed table. Every source is first copied and hashed
through one open descriptor; parsing uses only that private snapshot. Raw rows
must have valid unique dates, finite positive required prices, and finite,
non-negative whole-number volume, and cleaning must preserve every row. A
missing or partial set, invalid or duplicate rows, ambiguous duplicate
captures, a requested explicit raw directory that is missing, or a raw rebuild
that disagrees with the processed evidence causes a failure. Processed-only
migration is never accepted as verified real data.

The completed bundle has four files:

```text
data/external/verified_legacy_psx_cache/
├── psx_prices.csv
├── pipeline_metadata.json
├── legacy_pipeline_metadata.json
└── migration_manifest.json
```

`legacy_pipeline_metadata.json` is a byte-for-byte copy of the original. The
new `pipeline_metadata.json` uses schema 1.0 but truthfully identifies its
producer as `paisa.legacy_cache.migrate_legacy_psx_cache`, never
`paisa.pipeline.run_v0_pipeline`. `migration_manifest.json` binds the original
and migrated SHA-256 values, original and migration timestamps, migration
strategy, per-symbol coverage, evidence strength, and limitations.

Migration validates all of the following before publishing:

- Required price columns; numeric `open`, `close`, and `volume`; positive
  prices; non-negative volume; and unique ticker/date observations.
- One exact expected real source, `psx_dps`, across every row. Wrong, blank,
  mixed, sample, synthetic, fixture, mock, or test sources are rejected.
- Legacy provider `psx_dps`, `OGDC` in `symbols_requested`, positive OGDC raw
  acquisition evidence, and no OGDC provider error.
- The legacy total row count and global date range against the CSV, plus
  computed per-symbol coverage and agreement between requested symbols and
  observed symbols. Missing evidence is an error; the migrator never guesses.
- Checksums for the migrated CSV, schema-1.0 metadata, and preserved legacy
  metadata. A checksum disagreement invalidates bundle validation, and the
  runner separately inventories the migration manifest it consumed.

After migration, invoke the real-data runner with the new cache and controlled
run ID:

```bash
python scripts/run_ogdc_vertical_slice.py \
  --cached-prices \
  "data/external/verified_legacy_psx_cache/psx_prices.csv" \
  --run-id \
  "VSL-20261020-OGDC-real-cache-02"
```

The runner still tries live PSX DPS first. If that attempt returns HTTP 404,
the full error is retained in the manifest and final summary. A migrated-cache
fallback is labelled in all output evidence as:

```text
acquisition_mode = "verified_legacy_psx_cache"
live_fetch_succeeded = false
original_provider = "psx_dps"
legacy_metadata_preserved = true
```

It must never be called a successful live fetch. The current schema-1.0
pipeline-produced cache remains valid through its original, separate producer
path.

Before validation, the runner copies every required input-evidence file to an
invocation-private working snapshot. Validation and model training consume
that snapshot, not the mutable source cache. The runner verifies its
fingerprints around training, so an external cache edit cannot create a
time-of-check/time-of-use mismatch. A successful verified-legacy run retains:

```text
artifacts/vertical_slice/<vertical_slice_run_id>/input_snapshot/
├── psx_prices.csv
├── pipeline_metadata.json
├── legacy_pipeline_metadata.json
└── migration_manifest.json
```

A current-pipeline run retains only its exact price CSV and
`pipeline_metadata.json`. The manifest checksum inventory and final report bind
all retained input evidence and record its price and provenance fingerprints.

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
prediction files are part of a sorted exact-file SHA-256 inventory covering
every training payload other than the self-referential `run_manifest.json`:
frozen inputs, data and split manifests, reports, model manifests, model
configurations, metrics, predictions, and binaries. The run manifest records
the common code version, dirty-tree flag, and source fingerprint and is itself
hash-bound by the vertical-slice reference. The full inventory is checked
before backtesting, after every experiment has been reconciled, and immediately
before publication. The model IDs and prediction paths consumed by backtesting
must exactly match the validated root-manifest bindings; external summary paths
are rejected. Any missing, extra, redirected, or modified file aborts the
transaction.

## Output artifacts

A successful run publishes this comparison bundle:

```text
artifacts/vertical_slice/<vertical_slice_run_id>/
├── manifest.json
├── model_training_run_reference.json
├── input_snapshot/
│   ├── <price_csv_filename>.csv
│   ├── pipeline_metadata.json  # real-data runs
│   ├── legacy_pipeline_metadata.json  # verified-legacy runs only
│   └── migration_manifest.json  # verified-legacy runs only
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
  producer-written `pipeline_metadata.json`; verified-legacy runs additionally
  retain the byte-preserved `legacy_pipeline_metadata.json` and the
  checksum-binding `migration_manifest.json`. Fixture-only tests may omit
  provenance files. These are immutable evidence for reproducing which input
  bytes passed the producer-attestation and integrity checks and were used;
  every retained file is covered by the manifest.
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
- Cached-file qualification depends on the producer-written or migrated
  `pipeline_metadata.json` and its checksum-bound evidence. This detects
  post-production edits, stale metadata, mixed/blank sources, and simple
  relabelling, but cannot prove that an already-compromised producer or a
  coordinated replacement of all evidence reported truthful market data.
- Legacy raw files do not contain a cryptographic provider signature or a
  retained HTTP request/response capture. Their filename, columns, values, and
  agreement with the legacy processed data provide internal consistency, not
  independent external authenticity.
- The captured PSX-DPS rows have blank `high` and `low` fields. Migration
  verifies them when supplied, but this legacy run's features and simulations
  rely on the validated `open`, `close`, and `volume` observations.
- SHA-256 checksums detect byte changes within the recorded evidence chain;
  they are integrity digests, not digital signatures or third-party
  attestations.
- The run covers one stock, one daily horizon, one historical split, fixed
  thresholds, and one controlled cost model. It is not evidence of live
  execution quality or future returns.
- Reported Sharpe ratio, volatility, drawdown, fees, and trade count describe
  the simulated test window only.
- No API, dashboard, news, sentiment, LSTM promotion, SHAP analysis, or
  reinforcement-learning component is part of this command.

All generated results are academic simulations and are not financial advice.
