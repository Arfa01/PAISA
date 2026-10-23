# PAISA project state — 24 October 2026

Repository: `Arfa01/PAISA`

Snapshot base: `origin/main` at `a80594d80a884dcef129f2d4a9c896217c6793c5`
after merge of PR #17

Current comparison work: `feature/unified-price-model-comparison`

PAISA is an academic research prototype. Every prediction, signal, trade,
portfolio value, curve, and return is an academic simulation and is **not
financial advice**.

This file is the dated continuation snapshot. For detailed contracts and
evidence, follow [Current Implementation Status](CURRENT_IMPLEMENTATION_STATUS.md),
[Team Handoff](TEAM_HANDOFF.md),
[LSTM Price-Only Experiments](LSTM_PRICE_ONLY_EXPERIMENTS.md), and
[Unified Price-Only Model Comparison](UNIFIED_PRICE_MODEL_COMPARISON.md).

## Snapshot boundary

- Git contains source, tests, contracts, small fixtures, configurations, and
  notebooks. It does not contain the retained real datasets, fitted weights,
  SQLite databases, or generated run directories.
- The artifact IDs below identify immutable, Git-ignored evidence retained in
  the project checkout or a custodian export. A clean clone must restore or
  regenerate them; their mention in documentation is not a download guarantee.
- Historical coverage-v2 packs passed the verifier at their producing revision.
  Current schema-`1.1` / coverage-v3 code intentionally does not certify them;
  current authority requires a fresh ID and source-bound recomputation. Do not
  describe an old v2 pack as a current-verifier pass.

## Completed components

| Component | State at this snapshot |
|---|---|
| Stock data and v0 compatibility | Daily PSX acquisition adapters, cleaning, technical features, next-session target, compatibility Random Forest, and legacy FastAPI routes remain available. Sample/manual fallbacks are explicit. |
| Leakage-safe experiment foundation | Chronological target-date splits, train-only learned transforms, ordered features, fixed seeds, immutable manifests, exact inventories, and distinct feature/target/signal/execution dates are enforced. |
| Price-only baselines | Three Random Forest variants (`M-001`–`M-003`) and one XGBoost variant (`M-004`) use the same frozen OGDC evidence and comparison contract. |
| Price-only LSTM | The optional TensorFlow workflow trains `M-005` and `M-006`, selects on validation evidence, performs one post-freeze test pass, backtests independent ledgers, and publishes a sealed run. |
| Unified price-only comparison | A deterministic read-only builder and offline verifier recompute M-001–M-006, reference-baseline, model-portfolio, cash-only, and controlled buy-and-hold metrics from checksum-bound persisted evidence. The portable evidence now includes two raw market files and six policy-signal files so exact `Decimal` policy, execution, accounting, and benchmark semantics can be replayed without retraining, inference, or invoking the production backtest runner. The output remains price-only and tabular/report-only. |
| Policy, portfolio, and backtesting | Threshold policy, next-session-open execution, fees, independent strategy ledgers, cash-only and controlled buy-and-hold benchmarks, trades, snapshots, drawdowns, and reports are implemented. |
| OGDC vertical slice | One collision-refusing command connects acquisition, preparation, training, prediction, policy, simulation, validation, and reports for daily OGDC data. |
| Results delivery | A read-only `/api/v1` validates immutable vertical-slice artifacts; the React/TypeScript/Plotly dashboard presents history, model and portfolio comparisons, experiment detail, provenance, limitations, and the disclaimer. |
| News foundation | RSS/Atom, reviewed manual CSV, and bounded metadata-only GDELT DOC adapters preserve raw evidence, normalized records, timing, deterministic deduplication, controlled OGDC links, SQLite state, and immutable quality packs. |
| Historical coverage audit | News evidence can be mapped leakage-safely to the immutable stock calendar and exact split dates under explicit permission, provenance, completeness, and coverage gates. |
| Article sentiment | Pinned `ProsusAI/finbert` inference produces immutable article-level class probabilities with model, tokenizer, text-policy, timing, runtime, and checksum evidence. It does not aggregate by day or emit a trading signal. |
| Contemporaneous evidence | Bounded GDELT GKG 2.1 raw-archive evidence, controlled offline PSX-announcement import, and separate/combined evidence coverage profiles are implemented without fetching publisher pages. |
| Offline verification | Backend, frontend, artifact-integrity, security-boundary, documentation-contract, and notebook-contract tests are present. Live providers and real weights remain outside the default suite. |

Component ownership remains strict: models emit predictions, policies emit
decisions, portfolios own accounting, APIs read validated artifacts, and the
dashboard presents results without recomputing them.

## Merged milestone pull requests

| PR | Merge commit | Milestone |
|---:|---|---|
| #3 | `c71822e` | Versioned backtesting engine |
| #4 | `630936a` | Reproducible model experiments |
| #5 | `36a5d90` | OGDC end-to-end vertical-slice runner |
| #6 | `3b82245` | Verified legacy PSX-cache provenance |
| #7 | `768cbb2` | Read-only experiment-results API |
| #8 | `a443927` | React results dashboard |
| #9 | `4e2c5aa` | Run-scoped price-history API |
| #10 | `826c371` | Post-merge vertical-slice smoke polish |
| #11 | `418d6b5` | Supervisor demo and handoff pack |
| #12 | `578eeec` | News-ingestion foundation |
| #13 | `9dc4b3a` | FinBERT sentiment foundation |
| #14 | `31fe4ad` | Historical-news coverage audit |
| #15 | `cf897e1` | Contemporaneous historical evidence |
| #16 | `d7015cb` | Price-only LSTM experiments and certified evidence |
| #17 | `a80594d` | Project continuation roadmap and offline CI |

Tags currently retained in Git are:

- `v0.0.1` at `781b702`;
- `v0.1.0-ogdc-vertical-slice` at `826c371`;
- `v0.2.0-news-ingestion-foundation` at `578eeec`;
- `v0.3.0-finbert-sentiment-foundation` at `9dc4b3a`; and
- `v0.4.0-historical-news-coverage-audit` at `31fe4ad`;
- `v0.6.0-lstm-price-only` at `d7015cb`; and
- `v0.7.0-continuation-ci` at `a80594d`.

The LSTM tag points to `main` after PRs #15 and #16. PR #15 has no separate
milestone tag of its own.

## Canonical retained evidence and IDs

These paths are local, generated, and ignored by Git.

| Evidence | Canonical identity | Authority and notes |
|---|---|---|
| OGDC vertical slice | `VSL-20261020-OGDC-main-02` | Completed `PASS` demonstration run. |
| Baseline training | `TRN-20261020-191815-624662-nogit` | Frozen input, split, registered models, predictions, reports, and exact inventory used by the vertical slice and LSTM comparison. |
| LSTM comparison | `LSTM-20261024-OGDC-price-only-02` | Certified `PASS` result. Its `run_manifest.json` uses schema `1.0`; its authoritative `final_summary.json` uses schema `1.1` and carries the final status. The earlier `LSTM-20261024-OGDC-price-only-01` run is immutable pre-hardening history, not the certified result. Run-manifest SHA-256: `e475e077f375071c79ce8743425831d8cd9cf5ceb37f42eca0acfa65a100f3d0`; ZIP SHA-256: `f9d3ef10bb80528388a9137e7f169a7d870e7429f779f46e074a6254a26d8821`. |
| Unified price-only comparison | `CMP-20261024-OGDC-price-only-02` | Authoritative hardened pack derived only from the three authoritative price-only sources. It uses a digest-allowlisted redacted vertical certificate, recursively enforces portable-content hygiene, and replays all model and benchmark portfolio semantics from two raw market files and six signals. Verified file count: `65`; ZIP SHA-256: `8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`. |
| Superseded comparison draft | `CMP-20261024-OGDC-price-only-01` | Non-portable historical draft retained only as provenance. It predates the hardened certificate, recursive hygiene, and complete independent portfolio-semantic replay requirements and must not be distributed or cited as current authority. |
| Accepted RSS acquisition | `NEWS-20261022-131242-5ddffd2d` | Historical accepted run, retained for evidence and sentiment input. It predates current news sealing/schema contracts and must not be relabelled as a current verifier pass. |
| Retained GDELT DOC acquisition | `NEWS-20261023-060000-gdelt-ogdc2` | Sealed `PARTIAL/WARN`; its exact permission-reviewed metadata controls are allowlisted, but acquisition is incomplete. |
| Historical RSS coverage | `COVERAGE-20261023-120000-real0002` | Immutable coverage-v2 `NO_OVERLAP`; passed only at its producing revision. |
| Historical GDELT coverage | `COVERAGE-20261023-130000-gdelt0001` | Immutable coverage-v2 `NO_OVERLAP`; passed only at its producing revision. |
| Article sentiment | `SENT-20261022-172211-real0002` | `PASS`, sourced from the accepted RSS run, using FinBERT revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. |
| Canonical GKG evidence | `EVID-20261023-175847-822c5685` | Current sealed pack; raw GKG ZIP semantic replay returned `PASS`. |
| Canonical evidence coverage | `ECOV-20261023-175904-be50a499` | Current schema-`1.1` sealed pack; portable record/calendar recomputation returned `PASS`, while every profile remained `NO_OVERLAP`. |

Registered model identities are:

| Variant | Registered model ID |
|---|---|
| `M-001` | `M-001-6ce88dc384de975e-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-002` | `M-002-5a0160ff1b7a8043-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-003` | `M-003-23efd4aed873bf08-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-004` | `M-004-6c60cc92f77a495a-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` |
| `M-005` | `M-005-1fee47047b61cc43-81548726-c0c4e1fd-50f7-4fa4-9963-01b6b3f85a46` |
| `M-006` | `M-006-67bcc0f398cc963a-81548726-c0c4e1fd-50f7-4fa4-9963-01b6b3f85a46` |

## Current research results

### Price-only OGDC baseline

- The retained real-data evidence has `1,239` daily OGDC sessions from
  2021-09-16 through 2026-09-15. The chronological target-date split contains
  `853` train, `182` validation, and `184` test observations.
- `M-001` is the validation-selected baseline predictor with validation balanced
  accuracy `0.530204`. `M-002` has the highest baseline test balanced accuracy,
  `0.524459`; the test result did not select it.
- `M-002` is the best baseline model portfolio at PKR `105,987.04`
  (`+5.98704%`). The controlled 50%-allocated buy-and-hold benchmark ends at
  PKR `106,247.89` (`+6.24789%`). No model portfolio beats buy-and-hold.

### Price-only LSTM

- The 20-session sequence contract yields `834 / 182 / 184`
  train/validation/test examples. Both variants stopped after seven epochs; the
  best validation checkpoint was epoch one.
- Validation selects `M-005` (`0.479280` balanced accuracy) over `M-006`
  (`0.474931`). After selection was frozen, test balanced accuracy was
  `0.520087` for `M-005` and `0.544251` for `M-006`.
- `M-006` has the highest balanced accuracy across all six models on the common
  184-date test intersection, but it is not the validation-selected LSTM. The
  dates and baseline outcomes were already exposed, so this is a shared project
  holdout rather than a fresh independent evaluation.
- The LSTM portfolios end at PKR `96,359.29` (`-3.64071%`) for `M-005` and PKR
  `96,110.91` (`-3.88909%`) for `M-006`. Neither beats `M-002` or buy-and-hold.

### Unified price-only comparison

- `CMP-20261024-OGDC-price-only-02` is the authoritative comparison identity.
  Its immutable ZIP SHA-256 is
  `8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`,
  and its verified total file count is `65`.
  `CMP-20261024-OGDC-price-only-01` is a superseded, non-portable historical
  draft.
- The validation-only rule selects `M-001` across all six models at balanced
  accuracy `0.5302041802585478`; within the LSTM family, validation prefers
  `M-005` at `0.4792799323426362`.
- On the 184-date shared project holdout, `M-006` has the best model balanced
  accuracy (`0.544251447477254`) and `M-005` has the best model F1
  (`0.6396761133603239`). The majority-class baseline records balanced
  accuracy `0.5` and F1 `0.0`; the previous-direction baseline records
  `0.5271771239513174` and `0.5245901639344263`, respectively.
- `M-002` is the best simulated model portfolio at PKR `105,987.04`
  (`+5.98704%`). `M-003` has both the lowest model drawdown (`1.18467%`) and
  lowest model fees (PKR `98.93`). Cash remains PKR `100,000.00`; controlled
  buy-and-hold ends at PKR `106,247.89` (`+6.24789%`). No model beats
  buy-and-hold in simulated total return.
- These are separate outcomes. Shared project holdout prediction rankings do
  not replace validation selection, and portfolio results do not select a
  prediction model.
- The hardened verifier binds the comparison to a digest-allowlisted redacted
  vertical certificate, two raw market files, six signal files, and all
  independent ledgers. It performs full `Decimal` policy, execution,
  accounting, return, drawdown, and fee replay without model inference or a
  production backtest. Metrics expose exactly two logical benchmark rows,
  cash-only and controlled buy-and-hold, each retaining six ledger-provenance
  records.
- Portable-content hygiene is recursive. Default CLI results use relative
  artifact names, failures omit a traceback, and opt-in debug diagnostics are
  sanitized. The price-only pack has no sentiment integration, no plots, and
  no Matplotlib import; presentation is tabular/report-only.

### News, sentiment, and contemporaneous evidence

- The accepted RSS run contains 30 occurrences, 29 duplicates against retained
  canonical evidence, and one OGDC-linked occurrence. Its October 2026 evidence
  is later than the retained price period.
- FinBERT scored all 30 requested canonical articles: 11 negative, 9 neutral,
  and 10 positive; 17 inputs were truncated and one article was OGDC-linked.
  No labelled sentiment-accuracy evaluation was performed.
- The retained GDELT DOC run contains 656 normalized rows from 93 domains and 42
  exact OGDC links, but all 656 publisher publication timestamps are missing and
  22 provider errors make the run incomplete. Conservative availability is the
  October 2026 acquisition time, not the historical `seendate`.
- The canonical GKG proof scanned 2,957 timestamp-valid rows from only three
  adjacent 15-minute partitions and found zero exact OGDC matches. It proves the
  bounded mechanism, not full-period coverage.
- Every canonical evidence-coverage profile reports zero eligible sessions in
  all three splits and `NO_OVERLAP`. Article sentiment therefore cannot yet be
  joined to the retained price experiment.

The last certified LSTM checkout reported `1,036 passed, 1 skipped` backend
tests, 57 passing frontend tests, and passing frontend lint/build, `pip check`,
`compileall`, and diff checks. That is retained validation evidence for source
commit `2bd20ed`. Before the `-02` hardening changes, this comparison branch
reported `1,082 passed, 1 skipped` backend tests with six dependency
deprecation warnings, 57 passing frontend tests, and passing frontend
lint/build, `pip check`, `compileall`, focused comparison/documentation tests,
and diff checks. Final hardened-regression results must replace this historical
branch checkpoint before publication. The frontend build retains its known
large-chunk warning.

## Active and open work

- The hardened unified comparison pack is active on
  `feature/unified-price-model-comparison`; its generated canonical artifacts
  remain local and Git-ignored. The verified `-02` custody digest is
  `8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`,
  and its total file count is `65`.
- GitHub reports zero open issues at this snapshot.
- Legacy PR #1 (`feature/v0.1-data-quality`) and PR #2
  (`feature/ogdc-rf-training-stage`) remain open. Their work appears superseded
  by later merged milestones, but they require repository-owner review; do not
  close or merge them automatically.
- PR #16 is merged. No LSTM training change remains under review.

## Known blockers and limits

1. **Historical news permission and overlap:** Dawn reuse remains
   `REVIEW_REQUIRED`; the permitted GDELT acquisition is incomplete and cannot
   be backdated; PSX announcement import remains `REVIEW_REQUIRED`; and current
   evidence has no defensible cutoff availability inside the frozen OGDC period.
   Daily aggregation and price-plus-sentiment work must stay blocked until a new
   immutable coverage run passes the configured gate.
2. **Coverage authority:** retained coverage-v2 packs are useful historical
   evidence, but current v3 verification requires restored exact inputs,
   recomputation, and fresh IDs. `PACK_SEMANTIC` fallback is weaker than
   source-bound verification.
3. **Primary price acquisition:** the live PSX-DPS endpoint returned HTTP 404.
   The retained price run is a checksum-bound reconstruction of a legacy real
   cache, not a fresh provider response or third-party signature.
4. **Research scope:** conclusions cover one stock, daily data, one split, one
   period, one fixed trading policy, and a reused shared project holdout. Verified
   `high`/`low` values are absent, corporate-action adjustment is unknown, and
   dividends are not modeled.
5. **Missing research layers:** SHAP, general entity/sector/event graphs,
   multi-stock evaluation, daily sentiment aggregation, and controlled
   price-plus-sentiment comparison are not complete.
6. **Delivery gap:** the certified LSTM run remains separate from the
   vertical-slice API and dashboard, while the hardened unified comparison is
   a local, Git-ignored derived artifact pending final `-02` regeneration. There
   is no fixture-backed read-only
   LSTM or unified-comparison API and no unified six-model dashboard
   presentation.
7. **Operational scope:** there is no production database service,
   authentication, scheduled ingestion/retraining, broker integration, live
   trading, or production deployment evidence.

## Exact next safe task

Create one narrow, fixture-backed PR for **read-only LSTM artifact integration**:

1. define an additive configured root and validated repository contract for
   immutable `artifacts/lstm_experiments/<run_id>/` packs;
2. validate a small committed fixture for schema, exact inventory, checksums,
   safe relative paths, run/model identities, completion status, and the
   academic disclaimer without loading model binaries;
3. expose additive list/detail/summary GET endpoints with typed response schemas;
4. add offline tests for valid fixtures, absent optional roots, malformed IDs,
   traversal/symlink attempts, missing or extra files, checksum changes, and
   contradictory identities; and
5. document the environment variable and endpoints.

Keep the PR independent of the retained private artifacts. It must not retrain a
model, mutate or reseal `VSL-20261020-OGDC-main-02`, recompute metrics in a GET
request, change the legacy API, or add a combined dashboard ranking. The unified
comparison was produced early under an explicit evidence-only task; that does
not satisfy the fixture-backed LSTM read gate or authorize using the private
comparison pack in clean-clone tests.

This task is unblocked by the historical-news permission problem and preserves
the existing model, policy, portfolio, API, and presentation boundaries.
