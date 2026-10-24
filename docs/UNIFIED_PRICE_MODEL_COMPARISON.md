# PAISA unified price-only model comparison

This document defines the `PAISA-RM-003` derived-artifact contract for the
price-only comparison of `M-001` through `M-006`. The canonical comparison ID
is `CMP-20261024-OGDC-price-only-02`, and its output root is
`artifacts/model_comparisons/`.

`CMP-20261024-OGDC-price-only-01` is a superseded, non-portable historical
draft. Its copied vertical-slice manifest disclosed producer-host paths, and
its verifier did not establish every portfolio transition it claimed. Do not
use, relabel, or distribute `-01` as the authoritative comparison. The
hardened `-02` ID is immutable and must be generated afresh.

The comparison is an academic simulation, not financial advice. It does not
demonstrate future performance.

The required evaluation terminology and consumer rules are summarized in
[`SHARED_HOLDOUT_POLICY.md`](SHARED_HOLDOUT_POLICY.md). The detailed artifact
contract in this document remains authoritative.

## Contract boundary

The builder is a deterministic evidence transformer. It validates completed
source runs, copies the row-level predictions, signals, two independently
retained market sources, and portfolio ledgers needed for portable replay,
derives the two reference baselines, and recomputes every published metric. It
does **not** train or deserialize a model, run model inference, rerun a
backtest, change a decision threshold, or mutate a source run. It does not
integrate or evaluate sentiment features.

The pack keeps three questions separate:

1. Which model wins the declared validation-only selection rule?
2. Which model has the strongest prediction metric on the shared project
   holdout?
3. Which fixed-policy simulated portfolio has the strongest return, drawdown,
   or fee outcome?

These answers are not combined into a cross-criterion score or ranking. The
persisted summary records `outcomes_are_not_combined=true`.

## Authoritative source roots

Only the following completed local sources are accepted for the canonical
pack. Generated artifacts are Git-ignored and must be restored from their
checksum-verified custodian bundle before the build.

| Role | Required run ID | Integrity root | Required integrity-root SHA-256 |
|---|---|---|---|
| Baseline training and M-001--M-004 prediction evidence | `TRN-20261020-191815-624662-nogit` | `run_manifest.json` | `8d5113bc8b67f43a76639d7397b905217bda1f94321183f0438e33448f689023` |
| M-001--M-004 portfolio and benchmark evidence | `VSL-20261020-OGDC-main-02` | `manifest.json` | `2ce2f7581a5baff8487b09ae695213745c704f11715e6bd5abd0e99009337f31` |
| M-005--M-006 prediction, portfolio, and benchmark evidence | `LSTM-20261024-OGDC-price-only-02` | `run_manifest.json` | `e475e077f375071c79ce8743425831d8cd9cf5ceb37f42eca0acfa65a100f3d0` |

The raw vertical-slice manifest is still authenticated by its authoritative
SHA-256, but it is not copied into the portable pack because its historical
diagnostic text contains producer-host paths. Instead, the builder writes a
deterministic allowlisted projection named
`baseline_vertical_manifest_certificate.json`. The certificate binds the
original manifest name and digest, stable source assertions, and the complete
source artifact inventory. Its own digest is independently allowlisted, so
neither an edited raw manifest nor an edited projection can be substituted.

`LSTM-20261024-OGDC-price-only-01` is historical pre-hardening evidence. It is
not an accepted input, a fallback, or a source of canonical values. A generic
artifact-integrity pass does not override this explicit allowlist.

The stable variant-to-model mapping is:

| Variant | Family | Registered model ID |
|---|---|---|
| `M-001` | Random Forest, PAISA v0 baseline | `M-001-6ce88dc384de975e-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-002` | Random Forest, conservative | `M-002-5a0160ff1b7a8043-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-003` | Random Forest, flexible | `M-003-23efd4aed873bf08-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-004` | XGBoost, CPU histogram | `M-004-6c60cc92f77a495a-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-005` | 20-session LSTM, baseline recurrent | `M-005-1fee47047b61cc43-81548726-c0c4e1fd-50f7-4fa4-9963-01b6b3f85a46` |
| `M-006` | 20-session LSTM, regularized recurrent | `M-006-67bcc0f398cc963a-81548726-c0c4e1fd-50f7-4fa4-9963-01b6b3f85a46` |

Before staging output, the builder validates all three roots, including their
exact IDs, completion state, inventories, nested checksums, and relationships.
During staged semantic replay it confirms model mappings, ticker, target,
horizon, chronological split, policy, fee, execution, and date contracts. It
repeats source-root validation after generation and before the atomic
publication boundary. A mismatch or source mutation fails closed.

## Shared project holdout

The comparison key is `target_date`. All six models have the same exact
184-date intersection, so no model loses a row:

- target dates: `2025-12-16` through `2026-09-15`;
- corresponding feature dates: `2025-12-15` through `2026-09-14`;
- ticker: `OGDC`;
- target: next-session close is greater than feature-date close;
- prediction horizon: one trading session.

This period is the **shared project holdout**. Its dates and M-001 through M-004
results were exposed before the LSTM comparison. It is reused comparative
evidence, not a fresh independent estimate.

The builder also requires all six models to share the 182 validation target
dates from `2025-03-24` through `2025-12-15`. Model selection uses only those
validation rows. Shared project holdout rankings are descriptive and never
change the validation-selected label.

## Build command

Run from the repository root in the default PAISA environment. TensorFlow is
not required because the builder never loads or retrains either LSTM.

```bash
python scripts/build_unified_price_model_comparison.py \
  --comparison-id CMP-20261024-OGDC-price-only-02 \
  --output-dir artifacts/model_comparisons \
  --training-run artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --vertical-slice artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --lstm-run artifacts/lstm_experiments/LSTM-20261024-OGDC-price-only-02
```

The destination directory, ZIP, and checksum sidecar are immutable. If any
published target already exists, the build fails instead of overwriting it.
Use `--no-zip` only when the caller intentionally wants the verified directory
without the portable archive and checksum sidecar.

The default workflow verifies the staged directory, ZIP, and sidecar, then
revalidates all three sources. It publishes the checksum sidecar, ZIP, and then
the comparison directory, whose no-replace rename is the final publication
boundary. An interruption can leave an inert sidecar or a complete verified
ZIP/sidecar pair, but never an unauthenticated ZIP or a partially written
comparison directory. Inspect an orphan explicitly and use a new ID; never
silently overwrite it.

Successful CLI output contains the comparison ID and paths relative to the
declared output root, not checkout or home-directory paths. Failures emit
sanitized JSON without a traceback by default. `--debug` adds a sanitized
traceback; path and credential-like material remains redacted. The build and
verification import path does not import Matplotlib or another plotting
backend.

## Pack layout

The successful default build publishes these three siblings:

```text
artifacts/model_comparisons/
├── CMP-20261024-OGDC-price-only-02/
├── CMP-20261024-OGDC-price-only-02.zip
└── CMP-20261024-OGDC-price-only-02.zip.sha256
```

The authoritative `-02` ZIP SHA-256 is
`8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`,
and the verified total file count is `65`: `manifest.json` plus the 64 entries
in its exact inventory. The final immutable directory and ZIP/sidecar pair pass
portable and external-source verification. The pack remains Git-ignored and
must be preserved or restored separately from a clean clone.

The comparison directory has this exact structure. `<variant>` means each of
`M-001`, `M-002`, `M-003`, `M-004`, `M-005`, and `M-006`.

```text
CMP-20261024-OGDC-price-only-02/
├── manifest.json
├── final_summary.json
├── comparison_contract.json
├── source_roots.json
├── date_alignment.json
├── metrics.json
├── rankings.json
├── report.md
├── derived/
│   └── reference_baseline_predictions.csv
└── evidence/
    ├── predictions/
    │   ├── M-001/predictions_all_splits.csv
    │   ├── M-002/predictions_all_splits.csv
    │   ├── M-003/predictions_all_splits.csv
    │   ├── M-004/predictions_all_splits.csv
    │   ├── M-005/predictions_all_splits.csv
    │   └── M-006/predictions_all_splits.csv
    ├── splits/
    │   ├── baseline_split_manifest.json
    │   └── lstm_split_manifest.json
    ├── market_data/
    │   ├── baseline_psx_prices.csv
    │   └── lstm_ogdc_prices.csv
    ├── portfolios/
    │   └── <variant>/
    │       ├── config.json
    │       ├── signals.csv
    │       ├── portfolio_snapshots.csv
    │       ├── trades.csv
    │       └── benchmarks/
    │           ├── cash_only_snapshots.csv
    │           ├── buy_and_hold_snapshots.csv
    │           └── buy_and_hold_trades.csv
    └── source_manifests/
        ├── baseline_training_run_manifest.json
        ├── baseline_vertical_manifest_certificate.json
        ├── lstm_run_manifest.json
        └── lstm_artifact_manifest.json
```

No fitted model, model binary, cache, or secret belongs in the pack. The only
raw market files are the two checksum-bound sources required to prove identical
session opens and closes across the baseline and LSTM families. All authored
paths are normalized relative POSIX paths. Every portable member must be UTF-8
text, and the verifier recursively rejects defined host-path and
credential-like patterns, including POSIX home/root paths, Windows drive or UNC
paths, file URIs, URL credentials, authorization values, known token or key
formats, and credential-like assignments. Symlinks, special files, traversal
components, and unexpected files are invalid.

## Schema 1.0

Every comparison-authored JSON document carries `schema_version="1.0"`;
byte-copied source manifests retain their original source contracts, while the
vertical-slice certificate is a schema-`1.0` deterministic projection.
Identifiers and ordered arrays are deterministic, authored JSON is UTF-8 and
rejects non-finite numeric values, dates use `YYYY-MM-DD`, and SHA-256 values
are lowercase 64-character hexadecimal strings. Authored comparison JSON omits
timestamps so equivalent inputs produce deterministic bytes.

### `manifest.json`

This is the pack integrity root. Its exact top-level fields are
`schema_version`, `artifact_type`, `comparison_id`, `status`, `inventory`,
`source_roots`, `final_summary`, and `disclaimer`. `status` must be `PASS`, the
artifact type must be `paisa_unified_price_model_comparison`, the disclaimer
must be `Academic simulation; not financial advice.`, the two document
pointers are fixed relative paths, and `inventory` uses SHA-256 with `root="."`
and `excludes=["manifest.json"]`. Its `files` array is
path-sorted and contains exactly every other regular file in the comparison
directory. Each entry contains only relative `path` and `sha256`. Missing,
additional, duplicate, linked, unsafe, or byte-modified files invalidate the
pack.

### `source_roots.json`

This file binds the three allowlisted source roles. Each source record contains
`role`, `artifact_type`, exact `run_id`, `integrity_root_name`,
`integrity_root_sha256`, `payload_file_count`, `integrity_root_pack_path`,
`integrity_root_pack_sha256`, and `integrity_root_derivation`. Top-level
`relationships` records the required source binding. Every ordered
`evidence_files` entry contains exactly
`source_role`, `source_path`, `pack_path`, `sha256`, `purpose`, and
`derivation="byte_copy"`; it maps each safe source-relative path to its safe
pack-relative path and SHA-256. `source_revalidation` records the
pre-generation and pre-publication checks. The document stores no source
checkout's absolute path. The training and LSTM manifests under
`evidence/source_manifests/` must match their source digests byte for byte.
The vertical role instead names the digest-allowlisted deterministic redacted
certificate, while retaining the authoritative raw-manifest digest as the
source integrity root.
For the LSTM source, the allowlisted `run_manifest.json` inventory must bind
the copied `artifact_manifest.json` digest, and that nested inventory must
exactly equal the remaining run-level entries. Neither layer is trusted on its
own.

### `comparison_contract.json`

This file freezes the assumptions whose mismatch must stop the build. Its
top-level fields are `schema_version`, `comparison_id`, `symbol`,
`source_frequency`, `models`, `reference_baselines`, `portfolio_benchmarks`,
`target_definition`, `alignment_key`, `evaluation_scopes`, `selection_policy`,
`reference_baseline_policy`, `portfolio_assumptions`,
`metric_recomputation`, `presentation`, and `disclaimer`:

- the six stable variant IDs in their canonical order; the copied source
  manifests bind each variant to its registered model ID;
- ticker `OGDC`, daily frequency, target definition, positive class, and
  one-trading-session horizon;
- exact chronological validation and shared project holdout target dates and
  the validation-only selection rule;
- the two reference-baseline definitions;
- PKR 100,000 initial cash, `0.60` buy threshold, `0.40` sell threshold, 50%
  maximum allocation, whole-share floor sizing, 10-basis-point transaction
  fee, zero slippage, long-only state, after-close signal, next-session-open
  execution, close valuation, and 252-session annualization;
- separate model portfolios plus cash-only and controlled buy-and-hold
  benchmarks;
- full semantic portfolio replay from persisted probabilities, signals,
  source-bound market opens/closes, trades, and snapshots; and
- tabular/report-only presentation with no generated comparison plots.

### `date_alignment.json`

This is the date-intersection proof. Its top-level fields are
`schema_version`, `comparison_id`, `alignment_key`, `validation`, and
`shared_project_holdout`. Each evaluation-scope object contains `terminology`,
`common_count`, feature/target start and end, `original_counts`,
`common_counts`, `dropped_target_dates`, and ordered `common_keys`. Each common
key records `ticker`, `feature_date`, `target_date`, and `actual_class`. The
shared project holdout additionally records `previously_exposed_through`. The
canonical shared project holdout has 184 dates and every dropped-date list is
empty. Feature dates, actual labels, and ticker must agree for each retained
target date.

### Copied prediction evidence

`evidence/predictions/<variant>/predictions_all_splits.csv` is copied from the
verified source and is not rewritten. M-001 through M-004 retain their source
`split` column. M-005 and M-006 retain the nine-column prediction-record schema;
their split membership is proved from the bound split/date evidence. The
prediction-record columns are:

```text
schema_version,model_id,ticker,feature_date,target_date,predicted_class,
probability_up,probability_down,actual_class
```

Probabilities must be finite, be in `[0, 1]`, sum to one within `1e-6`, and
produce `predicted_class` by choosing class 1 only when `probability_up` is
strictly greater than `probability_down`; a tie maps to class 0. The verifier
derives validation and shared project holdout metrics from these rows rather
than trusting a copied leaderboard.

### Copied split evidence

`evidence/splits/baseline_split_manifest.json` and
`evidence/splits/lstm_split_manifest.json` are byte copies of the verified
sources. Their `target_dates` arrays are non-empty, unique, and chronological.
The baseline manifest supplies training labels for the majority rule; the two
manifests must contain exactly equal validation dates and exactly equal shared
project holdout dates.

### `derived/reference_baseline_predictions.csv`

This file contains the row-level validation and shared project holdout
predictions used to recompute the majority-class and previous-direction
baselines. Its columns are:

```text
evaluation_scope,baseline_id,schema_version,model_id,ticker,feature_date,
target_date,predicted_class,probability_up,probability_down,actual_class
```

`majority-class-baseline` is frozen from training labels only.
`previous-direction-baseline` uses the immediately preceding realized daily
direction; it does not look at the scored target-date close. The file contains
both validation and shared project holdout scopes, with deterministic ordering.

### Copied market and signal evidence

`evidence/market_data/baseline_psx_prices.csv` and
`evidence/market_data/lstm_ogdc_prices.csv` are byte copies of the two
independently checksum-bound source files. The verifier selects `OGDC` from the
declared feature start through target end and requires both sources to contain
the exact same ordered 185-session `(date, open, close)` vector: the opening
feature session plus every one of the 184 shared project holdout target
sessions. Missing, duplicate, reordered, or changed observations fail closed,
including a changed price on a day when a portfolio is flat.

Each `evidence/portfolios/<variant>/signals.csv` is also a source-exact byte
copy with this ordered schema:

```text
schema_version,signal_id,portfolio_id,model_id,ticker,signal_date,target_date,
action,probability_up,target_allocation,reason
```

The verifier derives the expected signal from the persisted `probability_up`,
the configured inclusive `0.60`/`0.40` thresholds, and the preceding share
state. It requires an exact signal row for every target session and never uses
`actual_class` to make a policy decision.

### Copied portfolio evidence

Every `<variant>` portfolio directory contains source-exact config, signals,
snapshots, trades, and benchmark ledgers. Snapshot columns are:

```text
schema_version,portfolio_id,date,cash,shares,market_price,position_value,
total_equity,realized_profit_loss,unrealized_profit_loss
```

Trade columns are:

```text
schema_version,trade_id,portfolio_id,signal_date,execution_date,action,
quantity,reference_price,execution_price,fee,cash_after_trade,
shares_after_trade
```

Using exact `Decimal` arithmetic, PKR `0.01` `ROUND_HALF_UP` money rounding,
and whole-share floor sizing, the verifier independently replays every policy
and ledger transition. It proves the opening PKR 100,000 flat state, all 184
next-observed-session decisions, next-session-open fills, zero slippage,
10-basis-point fees, 50% allocation sizing, cash/share and cost-basis changes,
realized and unrealized P/L, close-valued snapshots, cash-only invariance, and
the controlled benchmark's single first-session buy. This is semantic evidence
validation, not a call to the production backtest runner. A persisted portfolio
summary is never the primary metric input.

### `metrics.json`

This file has deliberately separate ordered collections:

| Field | Exact members | Meaning |
|---|---|---|
| `validation_model_metrics` | `M-001` through `M-006` | Recomputed validation prediction metrics used by the declared selection rule. |
| `validation_reference_baseline_metrics` | `majority-class-baseline`, `previous-direction-baseline` | Recomputed validation reference context; never used to select a model. |
| `shared_project_holdout_prediction_metrics` | `M-001` through `M-006`, `majority-class-baseline`, `previous-direction-baseline` | Recomputed prediction metrics on the exact 184 common target dates. |
| `model_portfolio_metrics` | `M-001` through `M-006` | Recomputed fixed-policy model portfolio outcomes. |
| `benchmark_portfolio_metrics` | Exactly two logical rows: `cash-only` and `buy-and-hold` | One reconciled economic metric record per benchmark. Each row carries six `ledger_provenance` records naming the independent variant, model, experiment, portfolio, config, snapshot, and optional trade evidence. |
| `reference_baseline_training` | Training row count, class counts, and frozen majority class | Provenance for the majority-class prediction rule. |

Prediction records include row/support counts, class coverage and
distributions, confusion matrix, accuracy, balanced accuracy, precision,
recall, F1, Matthews correlation coefficient, ROC-AUC, Brier score, and log
loss where defined, plus expected calibration error and calibration bins.
Portfolio records use `initial_equity`, `ending_equity`, `total_return`,
`maximum_drawdown`, `volatility`, `sharpe_ratio`, `number_of_trades`,
`win_rate`, `average_gain`, `average_loss`, `turnover`, `fees_paid`, and
`time_in_market`; model records also include return differences against both
benchmarks. Undefined values are JSON `null`, never `NaN` or infinity.
Ordering and tie-break rules are persisted and deterministic. The verifier
compares all six retained snapshot copies of each benchmark row by row and
compares trades where present before emitting the single logical benchmark
record.

### `rankings.json`

This file keeps each deterministic ordering separate:
`validation_model_selection`, `lstm_validation_preference`,
`shared_project_holdout_balanced_accuracy`, `shared_project_holdout_f1`,
`simulated_model_portfolio_return`, `simulated_model_portfolio_drawdown`, and
`simulated_model_portfolio_fees`. A ranking entry records its one-based rank,
competitor identity, registered model identity where applicable, and the
metric values needed to reproduce its declared tie breaks.

### `final_summary.json` and `report.md`

`final_summary.json` names, without combining, all required outcomes under
`reported_outcomes`:

- `validation_selected_model`;
- `lstm_validation_preferred_model`;
- `best_shared_holdout_balanced_accuracy`;
- `best_shared_holdout_f1`;
- `best_simulated_model_portfolio`;
- `lowest_drawdown_model_portfolio`;
- `lowest_fee_model_portfolio`;
- `cash_only_benchmark`;
- `buy_and_hold_result`;
- `any_model_beat_buy_and_hold`.

Its other top-level fields are `schema_version`, `artifact_type`,
`comparison_id`, `status`, `symbol`, `path_base`, `holdout_context`,
`outcomes_are_not_combined`, `reported_outcomes`, `artifact_paths`,
`limitations`, and `disclaimer`. `outcomes_are_not_combined` must be `true`.
The relative artifact paths lead to source identities, assumptions, date
evidence, metrics, and rankings; the summary does not duplicate those
documents or add a cross-criterion selection. `report.md` renders the same
persisted facts for human review; it is not a second source of metrics.

For the canonical evidence, the separately reported outcomes are:

| Outcome | Result |
|---|---|
| Validation-selected model | `M-001`, validation balanced accuracy `0.5302041802585478` |
| LSTM-family validation-preferred model | `M-005`, validation balanced accuracy `0.4792799323426362` |
| Best shared project holdout balanced accuracy | `M-006`, `0.544251447477254` |
| Best shared project holdout F1 | `M-005`, `0.6396761133603239` |
| Best simulated model portfolio | `M-002`, PKR `105987.04`, return `0.0598704` |
| Lowest-drawdown model portfolio | `M-003`, maximum drawdown `0.0118467` |
| Lowest-fee model portfolio | `M-003`, fees PKR `98.93` |
| Controlled buy-and-hold result | PKR `106247.89`, return `0.0624789` |
| Did any model beat buy-and-hold? | `false` |

The cash-only benchmark remains PKR 100,000 with zero return, drawdown, trades,
and fees. The buy-and-hold comparator uses the same 50% maximum-allocation
constraint as the model portfolios; it is not a conventional fully invested
benchmark.

## Offline verification

Verify an unpacked comparison without the source runs:

```bash
python scripts/verify_unified_price_model_comparison.py \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02
```

This verifies safe paths, the exact manifest inventory, checksums, schemas,
contracts, the digest-allowlisted redacted vertical certificate, copied source
manifests, recursive portable-content hygiene, date alignment, row-level
predictions, baseline predictions, the two raw market files, all six signal
files, every model and benchmark ledger, full `Decimal` semantic replay,
recomputed metrics, rankings, summary, and report consistency. Packs are
bounded to 256 files and 64 MiB of uncompressed payload. Verification does not
load a fitted model, invoke inference, rerun a production backtest, require
TensorFlow, or import Matplotlib or another plotting backend.

For stronger local custody verification, provide all three original source
roots together. Supplying only some of them is invalid.

```bash
python scripts/verify_unified_price_model_comparison.py \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02 \
  --training-run artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --vertical-slice artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --lstm-run artifacts/lstm_experiments/LSTM-20261024-OGDC-price-only-02
```

The external-source mode reruns the three source validators and compares their
current integrity roots with the digests sealed in `source_roots.json`.

## Portable ZIP and checksum

The ZIP transports the exact file bytes of the verified comparison directory.
The checksum sidecar is outside the archive so that it can authenticate the
complete ZIP without a circular self-checksum. It contains exactly one strict
ASCII record: the lowercase 64-character ZIP SHA-256, two spaces, the ZIP
filename, and a newline. The archive contains one top-level directory named
for the comparison ID. File members are sorted, normalized regular files with
fixed `0644` mode and `1980-01-01 00:00:00` timestamp; directory entries,
comments, extra fields, duplicate names, encryption, and unsafe compression
ratios are rejected. Verify both together offline:

```bash
python scripts/verify_unified_price_model_comparison.py \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02.zip \
  --checksum artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02.zip.sha256
```

ZIP verification authenticates the archive before reading it, rejects unsafe
or duplicate members and linked/special entries, extracts into a temporary
directory, and runs the same semantic verifier as the directory workflow. The
sidecar is mandatory for ZIP verification. The archive and sidecar can
therefore be moved together to another checkout and verified without the three
original source roots.

## Limitations

This is one stock, one historical period, six fitted price-only configurations,
one chronological split, and one fixed trading policy. The shared project
holdout was previously exposed, the sample is small, and the result may be
sensitive to market regime, thresholds, class balance, fees, execution rules,
and unmodeled dividends or corporate-action adjustment. The pack makes no
causal claim and does not show that a model will generalize to another period,
stock, feature family, or portfolio policy.

This comparison is price-only: it does not integrate, aggregate, or evaluate
sentiment. Its presentation contract is tabular/report-only; the pack contains
no plots and the build and verification paths do not import Matplotlib.

Validation selection, shared project holdout prediction rankings, and
simulated portfolio rankings answer different questions. They must not be
collapsed into one cross-criterion score or ranking. All predictions, trades,
returns, and portfolio values are academic simulations and are not financial
advice.
