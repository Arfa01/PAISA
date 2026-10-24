# PAISA Current Implementation Status

Status date: 24 October 2026

Feature base: `origin/main` at `a80594d` after PR #17

Retained demonstration run: `VSL-20261020-OGDC-main-02`

PAISA is an academic research prototype. Its predictions, signals, trades,
portfolio curves, and simulated returns are not financial advice.

## Implemented and working

| Area | Current implementation and evidence |
|---|---|
| Stock-data foundation | The legacy v0 path implements PSX acquisition through `psx-dps`, with manual-CSV and offline sample alternatives; cleans the canonical daily schema; creates backward-looking technical features and `target_next_day_up`; trains the compatibility Random Forest; and preserves the original FastAPI routes. The acquisition path is implemented, but the live endpoint returned HTTP 404 during the retained run, so fresh live ingestion is not currently demonstrated. The optional `pypsx` provider remains separate from the default installation. |
| Leakage-safe experiment data | The versioned path distinguishes `feature_date` and next-session `target_date`, splits chronologically by target date, fits learned transforms on training data only, records feature order and seeds, and preserves immutable input and model evidence. |
| Versioned model comparison | `M-001`, `M-002`, and `M-003` are Random Forest variants; `M-004` is XGBoost. All four use the same daily OGDC feature set, target, split dates, and held-out test observations. Each has a unique registered model identity and checksum-protected artifacts. The later M-005/M-006 comparison reused those 184 dates and the already-exposed baseline outcomes, making the period the shared project holdout. |
| Price-only LSTM comparison | The isolated optional-TensorFlow workflow adds `M-005`, a 20-session/64-unit baseline with dropout `0.20`, and `M-006`, the same sequence length and hidden width with stronger dropout and recurrent regularization. It reuses the frozen OGDC input and split evidence, fits preprocessing on training data only, chooses the preferred variant from validation evidence, evaluates test data post-freeze, and publishes immutable run artifacts. Certified CPU run `LSTM-20261024-OGDC-price-only-02` completed from clean commit `2bd20edebac9f5a063e2e10cc0394343067d8c7f`; its local, Git-ignored `final_summary.json` is authoritative. |
| Unified price-only comparison | The evidence-only builder accepts exactly `VSL-20261020-OGDC-main-02`, `TRN-20261020-191815-624662-nogit`, and `LSTM-20261024-OGDC-price-only-02`; recomputes prediction and reference-baseline metrics; and reconstructs policy, execution, trades, accounting, returns, drawdowns, and fees with exact `Decimal` arithmetic from two raw market files and six signal files. It preserves separate validation, shared project holdout, and portfolio outcomes and publishes collision-refusing directory/ZIP/checksum artifacts. Authoritative local identity `CMP-20261024-OGDC-price-only-02` verifies offline without TensorFlow, model loading, retraining, inference, the production backtest runner, or Matplotlib. It is price-only and tabular/report-only. |
| Tree-model SHAP explainability | The additive `paisa.explainability` producer explains only M-001 through M-004 on the exact 184-date shared project holdout. It validates fixed source identities and checksums, loads models only from the trusted training root, applies only the fitted median imputer, uses a deterministic 100-row training background, and emits class-1 probability-space local/global attributions with additivity gates of `5e-5` for Random Forest and `1e-6` for XGBoost. It stages and verifies an immutable directory plus deterministic ZIP/sidecar; the portable verifier reloads no executable model. Official retained run `XAI-20261024-140320-OGDC-tree-v1`, produced from clean commit `2cd4145e5e01459d13d186a6f3b697e979821a9d`, has quality status `PASS`; its ZIP SHA-256 is `7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`. All four `184 × 19` probability-space explanations pass additivity and provide all seven audit categories. |
| Decision and simulation | Predictions pass through a separate threshold policy into separate long-only portfolio ledgers. The retained run uses buy at `probability_up >= 0.60`, sell at `probability_up <= 0.40`, a 50% maximum allocation, whole shares, 10-basis-point execution fees, next-session-open execution, and close valuation. |
| Independent comparisons | Each model has its own experiment, portfolio, trades, cash, holdings, and snapshots. Cash-only and 50%-allocated buy-and-hold benchmarks use the common test window and cost assumptions. |
| Immutable vertical slice | `scripts/run_ogdc_vertical_slice.py` joins acquisition, preparation, training, predictions, policies, backtests, benchmarks, evidence validation, and reports without overwriting an existing run ID. |
| News-ingestion foundation | The isolated `paisa.news` package captures configured RSS/Atom responses or permitted manual CSV imports; preserves raw bytes and acquisition metadata; writes normalized schema `1.1` while retaining explicit read compatibility for `1.0`; keeps publication, source observation, fetch, and defensible availability timestamps distinct; applies layered deterministic deduplication; links boundary-safe controlled OGDC aliases; persists canonical and per-ingestion evidence in local SQLite; and publishes immutable quality reports. The ingestion command does not run sentiment or alter the price/model stack. |
| Historical-news source and coverage audit | The news package now also exposes a bounded metadata-only GDELT DOC 2.0 adapter and a licence-note-gated manual import path. A separate checksum-validating audit consumes immutable news, price, and exact target-split evidence; enforces the checksum-bound source-permission policy; maps each canonical article's conservative `available_at` to a configurable market cutoff; reports occurrence/canonical, all-news/OGDC/sector, split/month/source/language/completeness/gap metrics; and publishes collision-refusing `COVERAGE-...` artifacts under explicit readiness thresholds. It does not aggregate sentiment or alter models, policies, portfolios, APIs, or the dashboard. |
| Contemporaneous historical evidence | The isolated `paisa.evidence` package adds official GDELT GKG 2.1 raw-archive metadata and an offline PSX OGDC announcement contract. It preserves batch/publication/import timing, separates `gdelt_metadata`, `exchange_announcement`, and `publisher_news`, publishes immutable `EVID-...` packs, and audits four separate/combined profiles into immutable `ECOV-...` packs. Public PSX automation is prohibited and the controlled manual path remains `REVIEW_REQUIRED`. No aggregation or downstream model/API/dashboard integration is included. |
| Article sentiment foundation | The isolated sentiment stage reads canonical normalized articles from a completed ingestion run, applies an exact pinned `ProsusAI/finbert` model/tokenizer revision, maps the upstream class indices explicitly, records bounded text/token policy and runtime identity, persists immutable article probabilities/errors in SQLite, publishes checksum-inventoried `SENT-...` artifacts, and exposes a leakage-safe decision-cutoff filter. It does not aggregate by day or alter stock models, policies, portfolios, API, or dashboard. |
| Results API | The additive `/api/v1` FastAPI surface reads completed, checksum-valid artifacts. It lists runs and exposes run metadata, source provenance, historical prices, model metrics, portfolio metrics, curves, experiments, trades, snapshots, and benchmarks. It does not retrain models or recalculate accounting in a GET request. |
| Dashboard | The React/TypeScript/Plotly application reads only the API and shows run overview, immutable OGDC history, model comparison, portfolio comparison, experiment detail, trades, snapshots, provenance, limitations, and the academic-simulation disclaimer. It has no model, policy, or accounting business logic. |
| Verification | Independent verification of reviewed PR head `4d7084b` produced `609 passed, 1 skipped` in the artifact-present checkout, where only optional TensorFlow/LSTM skipped and the retained-artifact integration ran. Its clean Git clone produced `608 passed, 2 skipped`; optional TensorFlow/LSTM and the absent local-real-run artifact integration skipped. These remain historical pre-fix results. The hardened certified LSTM source commit `2bd20edebac9f5a063e2e10cc0394343067d8c7f` completed with `1036 passed, 1 skipped` in the backend and `57 passed` in the frontend. Before `CMP-20261024-OGDC-price-only-02` hardening, the unified comparison branch completed with `1082 passed, 1 skipped` in the backend and `57 passed` in the frontend. Final validation for the tree-SHAP implementation and official run completed with `1339 passed, 2 skipped, 6 warnings` in the backend and `132 passed` across 10 frontend test files, plus passing frontend lint and build. |

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

### Accepted LSTM execution

The certified `PASS` run `LSTM-20261024-OGDC-price-only-02` was executed on CPU
from clean source commit `2bd20edebac9f5a063e2e10cc0394343067d8c7f` with
Python 3.13 and TensorFlow 2.21. It produced 834 training, 182 validation, and
184 test sequences. Both variants trained for 7 epochs and recorded best
validation epoch 1. Validation preferred `M-005` at `0.479280` balanced
accuracy over `M-006` at `0.474931`.

Only after that preference was frozen, the common 184-date test reported
`M-005` at `0.520087` and `M-006` at `0.544251` balanced accuracy. `M-006`
therefore exceeded all `M-001`–`M-004` test balanced accuracies, whose best was
`M-002` at `0.524459`; this post-freeze result does not make `M-006` the
validation-selected variant.

The same date window and the M-001–M-004 outcomes were already exposed before
the LSTM run. “Post-freeze” therefore describes the M-005/M-006 selection
order; this is repeated evaluation on the shared project holdout.

The `M-005` portfolio ended at PKR 96,359.29 (-3.64071%) and the `M-006`
portfolio at PKR 96,110.91 (-3.88909%). Neither beat buy-and-hold at PKR
106,247.89 (+6.24789%) or the best baseline model portfolio, `M-002`, at PKR
105,987.04 (+5.98704%). The generated checksum-inventoried run directory is
local and ignored by Git; its `final_summary.json` is the source of record.
These are academic simulation results, not financial advice. Follow
[`LSTM_PRICE_ONLY_EXPERIMENTS.md`](LSTM_PRICE_ONLY_EXPERIMENTS.md).

### Authoritative unified comparison evidence

`CMP-20261024-OGDC-price-only-02` is the authoritative hardened comparison
identity. Its immutable ZIP SHA-256 is
`8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`,
and its verified total file count is `65`.
`CMP-20261024-OGDC-price-only-01` is a superseded, non-portable historical
draft and is not current publication evidence.

The pack replaces the host-path-bearing vertical manifest with a
digest-allowlisted redacted certificate, then recursively scans every text
member for non-portable paths and credential material. It carries two raw
market files and six signal files and performs independent full `Decimal`
semantic replay of policy thresholds, next-open execution, allocations, fees,
trades, close valuation, cash, holdings, P&L, returns, and drawdowns. It does
not run inference or invoke the production backtest.

The metrics artifact contains exactly two logical benchmark rows, cash-only
and controlled buy-and-hold. Each row retains six ledger-provenance records for
the independent model-associated benchmark evidence. CLI success output uses
relative names and an explicit path base; default failures omit tracebacks and
opt-in debug diagnostics are sanitized. This comparison does not integrate
sentiment, includes no plots, and does not import Matplotlib; its presentation
is tabular/report-only. Selection, shared project holdout prediction metrics,
and portfolio outcomes remain distinct findings.

### Tree-model SHAP execution status

The generator, schema-1.0 portable pack, offline verifier, and deterministic
fixture/smoke path are implemented for M-001 through M-004. Every official
model explanation is class `1` in probability space using interventional Tree
SHAP, 100 evenly spaced chronological training rows, and the model's original
fitted median imputer. Random Forest and XGBoost output shapes are normalized
separately, and every one of the 184 direct probabilities must reconstruct
within its declared tolerance (`5e-5` for Random Forest, `1e-6` for XGBoost).

Official retained execution `XAI-20261024-140320-OGDC-tree-v1` completed from
clean commit `2cd4145e5e01459d13d186a6f3b697e979821a9d` with run and quality
status `PASS`. Its deterministic ZIP SHA-256 is
`7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`.
All models use a `184 × 19` matrix in positive-class probability space, and
all seven audit categories are available for every model.

| Model | Top five features | Maximum / mean absolute additivity error |
|---|---|---:|
| `M-001` | `close_lag_1`, `return_volatility_5`, `close`, `close_lag_5`, `ma_20` | `5.78484978308591e-06` / `5.781760213738571e-06` |
| `M-002` | `return_volatility_5`, `close_lag_5`, `close`, `ma_20`, `rsi_14` | `4.302598055616613e-09` / `7.864803434113863e-10` |
| `M-003` | `close`, `ma_20`, `close_lag_1`, `close_lag_5`, `return_volatility_5` | `2.7466758000471714e-05` / `2.7464541568790406e-05` |
| `M-004` | `ma_20`, `close`, `close_lag_5`, `return_volatility_5`, `open` | `1.5920226559273942e-07` / `3.5509845686483396e-08` |

Every one of the 184 rows per model passes tolerance. The recorded rank
agreements are `M-001`/`M-002` (`0.8228070175438595`, four top-five features
shared), `M-001`/`M-003` (`0.8210526315789475`, all five shared), and
`M-002`/`M-003` (`0.7543859649122807`, four shared). The recorded rank
disagreements are the `M-004` pairings: `M-001`/`M-004`
(`0.5964912280701754`), `M-003`/`M-004` (`0.6175438596491227`), and
`M-002`/`M-004` (`0.7`), each with four shared top-five features. The runtime
records Python `3.13.7`, SHAP `0.54.1`, NumPy `2.3.5`, pandas `2.3.3`,
scikit-learn `1.9.1`, XGBoost `3.4.1`, joblib `1.6.0`, and Matplotlib `3.11.2`.
Follow
[`TREE_MODEL_SHAP_EXPLAINABILITY.md`](TREE_MODEL_SHAP_EXPLAINABILITY.md),
[`EXPLAINABILITY_DATA_CONTRACT.md`](EXPLAINABILITY_DATA_CONTRACT.md), and
[`SHAP_INTERPRETATION_POLICY.md`](SHAP_INTERPRETATION_POLICY.md).

### Accepted news-ingestion execution

The accepted network-acquired RSS run `NEWS-20261022-131242-5ddffd2d`
reached run/quality status `PASS`/`PASS` against one configured Dawn Business
feed. It recorded `30` raw and `30` normalized occurrences, `29` duplicates,
`1` OGDC-linked occurrence, `0` missing publication times, and a publication
range from `2026-10-19T18:56:40Z` through `2026-10-22T13:07:31Z`. The source
response was acquired at `2026-10-22T13:12:43.088883Z`, after the run start at
`2026-10-22T13:12:42.772512Z`, and its declared and observed SHA-256 values
match. Evidence is under
`artifacts/news_ingestion/NEWS-20261022-131242-5ddffd2d/` and is ignored by
Git.

The `29` duplicates match canonical evidence retained from earlier acquisitions;
one newly observed feed article became new canonical evidence. The accepted run
still retains every raw response and normalized occurrence. The earlier
`NEWS-20261022-115232-03d0fc4c` run remains immutable but is superseded because
its acquisition timestamp was taken too early. The intermediate corrected run
`NEWS-20261022-120012-31d811b1` is also retained but superseded because its
manifest reported an obsolete database schema version. The later
`NEWS-20261022-120133-43a51981` run remains immutable but is superseded by the
storage-integrity hardening, while `NEWS-20261022-130023-470cb7a7` predates the
final time-evidence, secret-redaction, and recoverable-finalization hardening.
The accepted manifest reports finalization state `COMMITTED` and SQLite schema
version `3`. Its ingestion artifact tree remains historical: it predates the
schema-4 sealing fingerprint, the schema-5 sentiment tables, the current
schema-6 sentiment certification plus schema-7 source-observation contract, and the
completion/deduplication-context files, and must not be described as passing
the current news verifier. Before sentiment execution the working
`data/news/paisa_news.sqlite3` also used schema version `3`; it is now schema
version `5` for the retained reference execution. Current code migrates a
verified database onward through schema `6`, which adds hash-bound model files,
post-truncation evidence, explicit run associations, and certified
finalization, and schema `7`, which stores nullable audit-only
`source_observed_at` separately from eligibility under
`news-availability-v1`. An ignored, uncommitted pre-migration copy at
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3` was verified as
schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check. The accepted run demonstrates only
one publisher's current feed snapshot; it does not establish historical
completeness, stable future availability, licensing beyond the publisher's
feed terms, sentiment quality, or predictive/model value.

### Historical news-price coverage audit

The existing historical schema-`1.0` / `news-price-coverage-v2` preflight
`COVERAGE-20261023-100000-preflight003` validates all five retained legacy
`PASS` RSS evidence packs against `VSL-20261020-OGDC-main-02` and
`TRN-20261020-191815-624662-nogit`. Across the five packs it observes `150`
occurrences, `31` unique canonicals, and `119` duplicate occurrences. There are
`5` OGDC-linked occurrences representing `1` unique OGDC canonical. All
retained news is from Dawn Business in `en-us`; its publication range is
2026-10-19 through 2026-10-22 and its conservative availability range is on
2026-10-22. Dawn reuse remains `LICENSING_REVIEW_REQUIRED`; the existing
ignored snapshot is private audit evidence and is not approved for new
acquisition, publication, export, or corpus reuse.

All five selected manifests are `PASS` with zero provider errors and zero
rejected records, so `ingestion_completeness.passed` is true. That hard
prerequisite is reported separately from source permission; it does not erase
the unresolved Dawn permission marker or replace the more fundamental overlap
status. The checksum-bound permission policy records every selected RSS run as
`REVIEW_REQUIRED`. The SQLite database is fingerprinted, but all five packs are
legacy unsealed, so `database_provenance.passed` is false and their run IDs are
reported in `unsealed_run_ids`. OGDC status is reconciled against each pack's
checksum-bound `article_entity_links.csv`, and the audit preserves the exact
link rows in its OGDC-specific output.

The checksum-validated stock evidence contains `1,239` observed price sessions
from 2021-09-16 through 2026-09-15. Exact target-date arrays contain `853`
train, `182` validation, and `184` test sessions. Neither publication dates nor
cutoff-eligible availability intersect that stock period, so all-news and
OGDC-linked coverage is `0 / 853`, `0 / 182`, and `0 / 184`; the immutable
readiness result is `NO_OVERLAP`. Energy-sector linking is unsupported by the
current news contract and is reported as unsupported, not inferred from
general news. Generated evidence is under
`artifacts/news_coverage/COVERAGE-20261023-100000-preflight003/` and remains
outside Git. That preflight predates the current coverage completion seal and is
descriptive rather than the official final. The retained official RSS artifact
is `artifacts/news_coverage/COVERAGE-20261023-120000-real0002/`. Its verifier
returned `PASS` at the producing revision; it contains `11` payload artifacts
and `450` mapping rows, and retains `NO_OVERLAP`. Its manifest SHA-256 is
`57d1f2e3b099318062e4f094c127de7d130953e38b76564684c7427fa3139b54`
and its inventory SHA-256 is
`1316d7fa9164bb8d091664cc102f8a5b4ad89d4760c7caca7fe62d13142d29a8`.

### Live GDELT acquisition and separate preflight

The completed metadata-only GDELT run `NEWS-20261023-060000-gdelt-ogdc2` is
sealed `PARTIAL` with `WARN` quality and verifies
`PASS (DATABASE_AND_ARTIFACT_BOUND)`. It retained `656` raw rows and `656`
normalized canonical rows unique under the current exact
URL/content/title-source-time rules, with no exact-rule duplicate relationships
or rejected records. A conservative publisher document-ID diagnostic flags
`17` candidate groups (`36` rows, `19` beyond one per group) that remain
unmerged pending adjudication. Exact checksum-bound entity evidence links `42`
rows to OGDC, and the returned
metadata represents `93` source domains. All `656` records preserve missing
`published_at`; PAISA does not turn GDELT `seendate` into publisher publication
time. Raw `seendate` evidence spans `2023-06-18T12:15:00Z` through
`2024-06-14T15:00:00Z`, while conservative `available_at` spans
`2026-10-23T06:26:32.440746Z` through `2026-10-23T06:29:25.858147Z` because the
later acquisition time controls eligibility. The `PARTIAL` status records `22`
provider errors: `15` `ReadTimeout` failures, `4` quarantined out-of-window responses,
and `3` exhausted HTTP 429 responses.
The machine-readable source-permission decision allowlists this exact run under
its reviewed, permission-relevant metadata-only controls; a future GDELT run
requires a new review and policy update.
This retained run uses the prior news schema and its provider errors identify
only configured root windows. New schema-`1.1` acquisition records exact
request-tile/attempt context, bounded DNS/request deadlines, and audited
`Retry-After` decisions; those facts require regeneration and are not imputed
into the retained pack.

The separate historical schema-`1.0` / `news-price-coverage-v2` preflight
`COVERAGE-20261023-110000-gdeltpre1` passes source permission and database
provenance but fails ingestion completeness because that selected run is
`PARTIAL` and has `22` provider errors. Publication and availability
calendar-overlap ranges are `null`; all-news and exact-OGDC coverage is `0%`.
The historical schema-`1.0` pack stored unsupported energy as `0%`; current
schema `1.1` emits `null`/`N/A` rather than a measured zero. Readiness is
`NO_OVERLAP`.
The preflight remains historical. The retained official GDELT artifact is
`artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/`; its verifier
returned `PASS` at the producing revision, it contains `11` payload artifacts
and `1,968` mapping rows, and it retains `NO_OVERLAP`. Its manifest SHA-256 is
`9bb6385196d79befde21d422eb28a78d8aa5222b0863ea8d44b330fad3e0968a`
and its inventory SHA-256 is
`51d170ce3a0e26aa743d37b9e32335bfbfe0e1bdf9abee89e0d3683b748e0673`.
Both official artifacts were generated from clean commit
`041c557edf5b15660fd2f02d47ed7f9b8b4152a7` with `dirty_tree=false`. The RSS
result records source-permission and legacy database-provenance failure but
passing ingestion completeness; the GDELT result records passing permission and
database provenance but failing completeness with `22` provider errors. Neither
official result permits daily aggregation.
The current coverage contract is schema `1.1` /
`news-price-coverage-v3`. Its semantic verifier intentionally rejects the
retained v2 coverage packs; current-verifier evidence requires restored source
inputs, recomputation, and new immutable coverage run IDs.
For v3, default verification reopens those exact inputs and reports
`SOURCE_EVIDENCE_BOUND`; explicit `--pack-only` reports the weaker
`PACK_SEMANTIC` authority and cannot substitute for source replay.

The adapter never fetches linked publisher pages or stores publisher bodies.
GDELT metadata use remains subject to the reviewed source-permission policy,
required attribution, service limits, and underlying publisher rights. No
licensed manual historical dataset was supplied in this task.

### Contemporaneous GKG and PSX evidence status

The new GKG path uses raw GKG 2.1 archives rather than GDELT DOC. It accepts
only `WEB=1`, treats the batch timestamp in `GKGRECORDID` as the defensible
historical `available_at`, retains `V2.1DATE` as publication time, verifies
archive size and checksums, and never follows publisher URLs. The canonical
post-fix execution `EVID-20261023-175847-822c5685` processed the official
`2026-09-04 09:00`, `09:15`, and `09:30` UTC 15-minute partitions within the
requested `2025-12-16`--`2026-09-15` interval. That is only three adjacent
partitions, or 45 minutes, and is not the full test period. Its three raw
captures contain `2,957` scanned and timestamp-valid rows, `0` rejected,
`2,957` unmatched, `0` matched, and `0` duplicate rows. The verifier returned
`PASS` with basis `RAW_GDELT_GKG_ZIP_SEMANTIC_REPLAY`; this is bounded
execution evidence, not full-period completeness.

The PSX path accepts only exact `OGDC` announcement metadata supplied offline.
Timed rows require the official offset-aware posting time. Date-only rows keep
timestamps null and become eligible strictly on the next observed trading
session. The reviewed permission decision is still `REVIEW_REQUIRED`, so no
authorized input was imported and the count is `0`; public-page automation is
not used.

The canonical paired coverage run is `ECOV-20261023-175904-be50a499`. Its `656`
mapping rows are verified by portable sealed-record/stock-calendar
recomputation with basis `PORTABLE_RECORD_CALENDAR_RECOMPUTATION`; that basis
is deliberately not source-archive-bound. The four coverage profiles are
`gdelt_gkg_ogdc`, `psx_ogdc_announcements`, `combined_ogdc_evidence`, and
`general_financial_news`. The restored publisher-news evidence contributes
`656` general inputs, including `42` exact OGDC-linked inputs to the combined
profile. None is cutoff-eligible inside the immutable model period. Every
profile reports `0 / 853` train, `0 / 182` validation, and `0 / 184` test
overlap and readiness `NO_OVERLAP`. These labels express caller-defined gate
satisfaction only, not statistical sufficiency, predictive value, or financial
merit. These are academic simulation results, not financial advice.

Both canonical packs were produced from clean commit
`ec6a47b6922d1858d727845ab7de37be719a1dea`; pack directories are mode `0700`
and files are mode `0600`. Their immutable identities are:

| Pack | Manifest SHA-256 | Payload-inventory SHA-256 |
|---|---|---|
| `EVID-20261023-175847-822c5685` | `15ae19a7d186bd1417545914e3898b23ba80e1f1adb31bb7f8e8018f01f633a4` | `636b096f9bedaed73756c7175b35676555049251c6a3b14f8ed609e960c50888` |
| `ECOV-20261023-175904-be50a499` | `6b35a6c6c7a9a83b20bc93099fbfaff05743e49150ea0dc2cfa3ceb1e6667250` | `baf14a5d30f60fd7c5fad2d1daa75322007b43e2257e0a1a1ecf498efc23b826` |

The earlier retained pair is immutable audit history and is superseded, not
deleted: `EVID-20261023-163905-gkgfinal` has manifest
`a335bcf657ca855e52aebc2a0bcd4721e85434716aa8d6fa13a405cc0d42991e`
and payload inventory
`1675a8902f4c0175972392d04a207f09157d034ec1b118eb4c449e2ac9d77e7e`;
`ECOV-20261023-164000-final001` has manifest
`d4c31758d39e1a97ee7629e487fe04a801b82ce291923e47392b552f4f99335c`
and payload inventory
`e01f33b278bc780d02e74fa5dc88abbba803f12ffc8e90b95562ffb272010e4d`.
The local attempts `EVID-20261023-163553-gkgfinal`,
`EVID-20261023-213000-gkgraw01`, `EVID-20261023-213500-gkgraw02`, and the
post-review pre-final `EVID-20261023-175415-5c1b0c48` are likewise
noncanonical/superseded and must be preserved rather than relabeled. The team
handoff contains the checksum-bound preservation, absent-target-only restore,
and current verifier commands.

### FinBERT article-sentiment implementation

The implemented sentiment configuration pins both model and tokenizer to
`ProsusAI/finbert` revision
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. The recorded model/weight license
is CC BY-NC-SA 4.0, so this PAISA configuration is limited to non-commercial
academic use. The model's original index order is `0=positive`, `1=negative`,
and `2=neutral`; PAISA maps it explicitly into the stable probability order
`negative`, `neutral`, `positive`.

The accepted source run represents 30 unique canonical articles, one linked to
OGDC, and all carry `en-us` language evidence. Twenty-nine canonical articles
were first available in an earlier retained run; one was first available in the
accepted run. Sentiment inherits each canonical article's original
`published_at`, `available_at`, and `fetched_at`; the inference timestamp does
not replace those values.

The retained real execution completed as follows. These are generated local
artifacts and database records, not committed repository files.

| Real-run result | Measured value |
|---|---|
| Sentiment run / source run | `SENT-20261022-172211-real0002` / `NEWS-20261022-131242-5ddffd2d` |
| Status | `PASS` |
| Requested / scored / unique canonical | `30 / 30 / 30` |
| Skipped / failed / any articles failed | `0 / 0 / false` |
| OGDC-linked articles scored | `1` |
| Negative / neutral / positive counts | `11 / 9 / 10` |
| Truncated inputs / title-only inputs | `17 / 0` |
| Language evidence | `en-us=30` |
| Availability range | `2026-10-22T11:52:32Z` to `2026-10-22T13:12:43.088883Z` |
| Device / duration | `cpu / 7.116756208240986 seconds` |
| Model and tokenizer revision | `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` |
| Runtime | Python `3.13.7`; PyTorch `2.10.0`; Transformers `5.20.2`; Tokenizers `0.23.3`; Hugging Face Hub `1.32.0`; Safetensors `0.8.0` |
| Database schema for retained reference run / current code | `5 / 7` |
| Artifact directory | `artifacts/sentiment_inference/SENT-20261022-172211-real0002` |

Sentiment outputs were generated, but classification accuracy was not evaluated in this run.

The retained sentiment evidence covers October 19--22, 2026, but the retained
OGDC price evidence ends on September 15, 2026. These sources have **zero
overlapping dates**. The current evidence therefore cannot form even one
price-plus-sentiment observation and cannot support a combined experiment or
portfolio conclusion. That work requires an overlapping historical news
corpus.

The narrow legacy compatibility path validates the exact recognized schema-3
signature and affected rows, transactionally restores the later constraints,
and applies migrations 4 and 5. It does not recreate missing artifacts or
retroactively validate the retained news pack. The working database is now
schema version `5` (and is upgraded through `6` to `7` when opened by current
code), while
the ignored pre-migration copy above preserves the
verified schema-3 state. Even after a successful execution, this single short
RSS snapshot will not be sufficient for stock-model integration,
classifier-accuracy claims, or price-plus-sentiment comparison.

## Current research prototypes

- The four price-only classifiers are controlled comparative experiments, not
  evidence of performance on other stocks, periods, horizons, or regimes.
- The policy thresholds and portfolio simulator are research assumptions, not a
  recommendation engine or execution system.
- The dashboard and artifact API are demonstration and analysis surfaces, not a
  production deployment or current market-data service.
- The controlled M-005/M-006 price-only LSTM workflow and thin Colab notebook
  implement the common artifact and chronological-split contract. TensorFlow is
  intentionally outside the default runtime. The workflow supports CPU and
  Colab GPU, with optional Apple Metal, but does not promise bit-identical
  results across devices. The separate certified LSTM run is retained locally;
  neither LSTM is part of the four-model vertical-slice run above.
- Stored Random Forest impurity importance remains preliminary evidence. The
  tree-SHAP producer, portable verifier, and authoritative real explanation
  pack are complete for M-001 through M-004, but any API/dashboard
  presentation remains pending.
- News entity linking is a deterministic first stage for controlled OGDC
  aliases only. It is not general NER or event reasoning.
- FinBERT output is article-level classifier evidence, not a stock prediction,
  causal explanation, trading signal, or accuracy result. No human-labelled
  evaluation set, daily aggregation, or stock-feature join is included.
- The verified-cache migration is a provenance-preserving continuity mechanism;
  it does not turn the retained data into a fresh or cryptographically signed
  provider response.

## Progress against the approved proposal

| Approved objective | Status |
|---|---|
| Collect, preprocess, and structure historical stock data | Partly complete. The reusable pipeline supports multiple symbols, while the retained vertical slice and its conclusions are OGDC-only and daily-only. |
| Continuously update financial news, economic reports, and global events | Partly complete. Reproducible explicit RSS/Atom and imported-CSV runs now preserve news evidence, timing, quality, and failures. Scheduling, broad economic/global source coverage, and production operations are not implemented. |
| Train and compare multiple prediction models | Implemented for three Random Forest configurations and one XGBoost configuration on one frozen OGDC split. The separate M-005/M-006 price-only LSTM comparison has also been executed and certified under the same target-date split contract. `M-005` was preferred on validation; the stronger `M-006` test balanced accuracy is a post-freeze result, not a selection decision. |
| Compare models with and without sentiment | Not implemented. Article-level inference is implemented, but daily aggregation and the controlled stock-feature comparison remain. The current stock comparison is price-only. |
| Deploy the best-performing model as the prediction engine | Not complete. The v1 API deliberately serves retained results and does not load `M-001` for live inference; the legacy `/predict/{ticker}` route remains the v0 baseline compatibility path. |
| Add event-driven company/sector reasoning | Partly complete only at the ingestion boundary: reviewed OGDC aliases are linked deterministically with evidence. General entities, sectors, events, graphs, and propagation are not implemented. |
| Deliver predictions, visualizations, explanations, and current feeds | Partly complete. Historical results and provenance are visible in FastAPI and React; the tree-SHAP artifact/verifier layer and official real run are complete, but its API/dashboard view remains pending, while sentiment/event explanations and current feeds remain. |

## Remaining work

The next research sequence begins with written permission for the retained
Dawn fields or another licensed source, then acquisition/import of news whose
publication and defensible cutoff availability overlap the frozen OGDC period.
Only after a new immutable audit passes the configured coverage gate should the
team review article sentiment and define daily aggregation. Price-only versus
price-plus-sentiment comparison, read-only presentation of the verified SHAP
evidence, additional stocks, and sector/event relationships remain later work.
Production storage, authentication, scheduled
ingestion/retraining, cloud deployment, broker integration, and live trading
are also absent. Reinforcement learning is explicitly deferred until the data,
sentiment, supervised-model, evaluation, and explanation components are stable.

## Documentation inconsistencies to keep visible

- `PROJECT_CONTEXT.md` contains an older baseline-only status and an older
  multi-symbol date range. It is useful proposal history, but it is not the
  current implementation inventory or the evidence source for this run.
- `docs/DEFINITION_OF_DONE.md` still has unchecked acceptance boxes and requires
  a successful primary PSX price-provider fetch plus an executed TensorFlow
  LSTM smoke run. The retained vertical-slice run records the failed PSX request
  and verified legacy price cache, while the separate GDELT news acquisition
  above and the certified M-005/M-006 run did complete. The GDELT success still
  does not satisfy the primary price-provider acceptance check.
- The proposal's “best-performing model” is not a single result here. In the
  four-model run, `M-001` wins validation-based prediction selection and
  `M-002` produces the best model-portfolio simulation. In the separate LSTM
  run, validation prefers `M-005`, while `M-006` has the highest fair-test
  balanced accuracy across all six models. Test evidence did not revise either
  validation decision, and neither LSTM portfolio beats `M-002` or buy-and-hold.
