# PAISA Next Phase Plan

This plan extends the completed price-only OGDC vertical slice without changing
its leakage, artifact, policy, portfolio, API, or presentation contracts. All
outputs remain academic simulations and are not financial advice.

## Prioritized research sequence

### 1. Financial-news ingestion

**Foundation and historical-coverage audit implemented.** PAISA now has
explicit RSS/Atom and manual-CSV adapters, raw response/import evidence, a
versioned normalized record, distinct publication/fetch/availability time,
deterministic deduplication, local SQLite, immutable run packs, and data-quality
reports. The reviewed historical path adds a bounded GDELT DOC 2.0 adapter that
stores returned metadata and source URLs only; it never follows publisher links
or stores publisher bodies. A manual CSV remains acceptable only when its
provider, provenance, field-level storage permissions, academic-use basis,
attribution, and retention/redistribution limits are documented. Network checks
remain outside the default offline suite and bulk article data remains outside
Git.

The permission decision and candidates are recorded in
`docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md` and executed through the
checksum-bound `configs/news/historical_source_permissions.json`. The reusable
audit in `src/paisa/news/coverage.py` and
`scripts/audit_news_price_overlap.py` follows
`docs/NEWS_PRICE_OVERLAP_CONTRACT.md`, uses the immutable stock and split
manifests, applies `configs/news_coverage/default_readiness.json`, maps only
`available_at` evidence through the configured session cutoff, and publishes
collision-refusing `COVERAGE-...` packs. Its readiness labels are engineering
and data-coverage gates, not evidence of statistical sufficiency or predictive
value.

The existing historical schema-`1.0` / `news-price-coverage-v2` preflight at
`artifacts/news_coverage/COVERAGE-20261023-100000-preflight003/` covers the five
retained legacy `PASS` Dawn Business ingestion packs: `150` occurrences
representing `31` canonical articles and `119` duplicate occurrences. It
contains `1` OGDC-linked canonical article represented by `5` occurrences, and
no deterministic sector evidence, so energy-sector coverage is explicitly
unsupported. All records are from Dawn; publication evidence spans 2026-10-19
through 2026-10-22 and availability is on 2026-10-22. The immutable OGDC price
evidence contains `1,239` observed sessions from 2021-09-16 through 2026-09-15,
with exact split counts of `853` train, `182` validation, and `184` test
sessions. Publication overlap and cutoff-eligible availability overlap are both
zero, and the readiness status is `NO_OVERLAP`. This preflight predates the
current coverage completion seal. The retained official RSS artifact is
`artifacts/news_coverage/COVERAGE-20261023-120000-real0002/`; it verified `PASS`
at its producing revision with `11` payload artifacts and `450` mapping rows
and retains `NO_OVERLAP`.
Ingestion completeness passes, while source permission and legacy database
provenance remain failed prerequisites.

The later live metadata-only acquisition
`NEWS-20261023-060000-gdelt-ogdc2` is a separate sealed `PARTIAL/WARN` pack that
verifies `PASS (DATABASE_AND_ARTIFACT_BOUND)`. It retained `656` raw rows and
`656` normalized canonical rows unique under the current exact
URL/content/title-source-time rules, including `42` exact OGDC-linked rows
across `93` returned source domains, with zero rejected records. A conservative
publisher document-ID diagnostic separately flags `17` candidate duplicate
groups (`36` rows); these are not silently merged. Every canonical keeps
`published_at` missing. Its raw GDELT `seendate` range is
`2023-06-18T12:15:00Z`--`2024-06-14T15:00:00Z`, but conservative `available_at`
is `2026-10-23T06:26:32.440746Z`--`2026-10-23T06:29:25.858147Z`. The pack is
partial because it preserves `22` provider errors (`15` `ReadTimeout` failures, `4`
out-of-window quarantines, and `3` exhausted HTTP 429 responses).
That retained news run predates schema `1.1`: its `seendate` range remains raw
evidence, and its error rows identify root windows rather than exact attempts.
New acquisitions preserve audit-only `source_observed_at` under
`news-availability-v1`, fail closed on missing/malformed historical observation
time, bind every error to exact request-tile/attempt fingerprints, share one
bounded DNS/request deadline, and record whether `Retry-After` was applied or
refused above the wait ceiling.
Those fields require a new immutable acquisition and are never imputed into the
old run.

Its historical schema-`1.0` / `news-price-coverage-v2` preflight
`COVERAGE-20261023-110000-gdeltpre1` passes source permission and database
provenance, fails ingestion completeness, reports `null` publication and
availability calendar-overlap ranges and `0%` coverage for the supported
all-news and OGDC populations, and returns
`NO_OVERLAP`. That preflight remains historical. The retained official GDELT
artifact is `artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/`; it
verified `PASS` at its producing revision with `11` payload artifacts and
`1,968` mapping rows and retains
`NO_OVERLAP`. Permission and database provenance pass, while ingestion
completeness fails with `22` provider errors. Both official artifacts were
generated from clean commit `041c557edf5b15660fd2f02d47ed7f9b8b4152a7`
with `dirty_tree=false`.
Both coverage packs remain immutable historical v2 evidence. The current
schema-`1.1` / `news-price-coverage-v3` semantic verifier requires restored
source inputs, recomputation, and new run IDs; it does not certify the old
packs.
Default v3 verification reports `SOURCE_EVIDENCE_BOUND` only after reopening
the inputs; explicit `--pack-only` reports weaker `PACK_SEMANTIC` authority.

Remaining work is to obtain permitted, complete, sufficiently covered evidence
whose availability overlaps the immutable price/split period; the official
audits are already sealed and must remain immutable. The current policy permits
only the exact retained GDELT run
`NEWS-20261023-060000-gdelt-ogdc2` under its permission-relevant controls; it is
incomplete and cannot be backdated for model features, and a future acquisition
requires a new review and allowlist update. A historical CSV imported now has
the same timing constraint because the contract uses the later
acquisition/import time in `available_at`. Scheduling,
retention/backup policy, and production operations also remain. An explicit run
is not a continuously updating service.

### 1a. Contemporaneous GKG and official-event evidence

**Foundation implemented; downstream work remains blocked.** PAISA now keeps
three normalized evidence types separate: GDELT GKG `gdelt_metadata`, official
PSX `exchange_announcement`, and existing `publisher_news`. Coverage reports
`gdelt_gkg_ogdc`, `psx_ogdc_announcements`,
`combined_ogdc_evidence`, and `general_financial_news` independently before any
future aggregation.

The bounded GKG 2.1 raw-archive execution used the official 15-minute
partitions at `2026-09-04 09:00`, `09:15`, and `09:30` UTC inside the requested
`2025-12-16`--`2026-09-15` interval. It scanned `2,957` rows and found no exact
OGDC match. GKG batch time controls availability, `V2.1DATE` remains source
publication time, only `WEB=1` is retained, and publisher pages are never
fetched. This subset validates the bounded mechanism but is not a full history.

The PSX manual-import decision remains `REVIEW_REQUIRED`, so `0` announcement
rows were admitted and automation stays blocked. A date-only official event
must retain null timestamps and map strictly to the next observed session; it
must never receive an invented same-day time. The publisher-news input supplies
`656` general records and `42` exact OGDC-linked combined records, but all four
profiles still measure `0 / 853`, `0 / 182`, and `0 / 184` overlap across
train, validation, and test and therefore remain `NO_OVERLAP`.

Next, obtain separately reviewed source authorization and a sufficiently broad,
historically contemporaneous corpus. Re-run immutable evidence and coverage
packs under fresh IDs, then require nonzero split coverage and the configured
readiness gates before designing aggregation. Do not proceed to features,
sentiment joins, models, signals, portfolios, API, or dashboard work from the
current result. This is an academic simulation and not financial advice.

### 2. Company and entity linking

**First OGDC stage implemented.** Boundary-aware controlled aliases now record
matched alias, method, location, and confidence without fuzzy or forced
matching. Continue by creating a versioned registry for PSX tickers, legal company names,
subsidiaries, sectors, industries, places, and event actors. Preserve the raw
mention, linked identifier, method, confidence, and unresolved state; do not
force ambiguous mentions to a company.

Acceptance gate: reviewed fixtures cover OGDC aliases, ambiguous mentions,
unresolved entities, and effective-date changes without future leakage.

### 3. FinBERT sentiment

**Article-level foundation implemented; real CPU execution retained.** The
implementation pins the `ProsusAI/finbert` model and tokenizer to immutable
revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`, records its CC BY-NC-SA 4.0
license/non-commercial academic-use boundary, maps the original
`0=positive`, `1=negative`, `2=neutral` ordering explicitly, and persists
canonical negative/neutral/positive probabilities. It records deterministic
text selection, hashes, character/token counts, truncation, inherited timing,
runtime versions, resource settings, structured errors, SQLite identities, and
immutable run artifacts. The reusable cutoff filter performs no aggregation.

Run `SENT-20261022-172211-real0002` completed with `PASS` on CPU against
`NEWS-20261022-131242-5ddffd2d`: all `30` requested unique canonical articles
were scored, none were skipped or failed, one was OGDC-linked, the label counts
were `negative=11`, `neutral=9`, and `positive=10`, and `17` inputs were
truncated. Its retained execution used schema version `5`; current code advances
verified storage through schema `6` sentiment certification to the current
schema `7` audit-only source-observation contract. Generated artifacts remain
outside Git at
`artifacts/sentiment_inference/SENT-20261022-172211-real0002`.

Remaining gate: review the retained audit sample and outputs before defining
daily aggregation. The accepted ingestion artifacts still record their
historical schema version `3`, while the working
`data/news/paisa_news.sqlite3` was retained at schema version `5` and is
upgraded through `6` to `7` by current code. Preserve the ignored,
uncommitted pre-migration copy at
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3`, verified as
schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check. The narrow compatibility path is not retroactive
news verification.
Deterministic fixtures and the retained sentiment manifest must prove model,
input, output, and timing identity.

Sentiment outputs were generated, but classification accuracy was not evaluated in this run.
This short snapshot is insufficient for stock-model integration. FinBERT
output remains a feature source, not a trading signal or an accuracy claim.

### 4. Daily sentiment aggregation

**Not implemented and currently blocked by coverage.** The retained RSS audit
and the live-GDELT preflight are both `NO_OVERLAP`; the GDELT input also fails
the ingestion-completeness prerequisite. Do not begin aggregation merely
because article-level FinBERT output exists. First obtain permitted overlapping
evidence and pass the configured engineering/data-coverage gates; even
`READY_FOR_AGGREGATION` would not prove statistical sufficiency.

Aggregate article-level scores by ticker and trading date using only information
available by the declared feature cutoff. Define after-close and non-trading-day
handling, source counts, confidence weighting, missing-news indicators, and
lookback windows before running experiments.

Acceptance gate: tests prove no article with `available_at` after a decision
cutoff enters that feature row and no validation/test statistics are used to
fit transforms.

### 5. Price-only versus price-plus-sentiment comparison

Freeze paired configurations that differ only by the added sentiment features.
Use identical target definitions, target-date split boundaries, seeds,
evaluation rows, decision policy, execution rules, initial cash, and fees.
Select configurations on validation evidence and evaluate the untouched test
split once. Report prediction and portfolio metrics separately.

Acceptance gate: one comparison report records the ordered feature sets and
proves matched dates and assumptions; inconclusive or negative results are
reported without promotion language.

### 6. LSTM under the same split contract

Run the existing optional LSTM workflow with the frozen daily dataset contract.
Fit scaling on training data only, build sequences without crossing temporal
boundaries, register the model and scaler together, and emit the same prediction
record used by the backtester. Compare it under the same validation-selection
and untouched-test policy; do not give it a different window or cost model.

Acceptance gate: an actual TensorFlow smoke run and one retained immutable LSTM
experiment pass checksum, prediction-schema, leakage, and backtest validation.

### 7. SHAP explanations

Add model-appropriate SHAP analysis for retained model artifacts. Record the
background/reference data, explainer configuration, class/output explained, and
library versions. Explanations must describe model behavior, not causality or
financial advice.

Acceptance gate: local and aggregate explanations reconcile with the selected
model and ordered features, remain stable on fixtures, and are served through an
additive API schema before being displayed by the dashboard.

### 8. Additional stocks

Expand only after the OGDC news/sentiment comparison is reproducible. Start with
a small documented PSX set, audit date coverage and missing fields per symbol,
and keep per-model portfolios independent. Do not reuse one stock's scaler,
state, or conclusions for another without an explicit pooled-model design.

Acceptance gate: every stock has a data-quality/provenance report, chronological
split evidence, immutable experiment IDs, and stock-specific conclusions.

### 9. Sector and event relationships

Build dated company-sector-event edges from the reviewed entity-linking output.
Keep evidence, direction, confidence, and validity dates on every relationship.
Evaluate whether relationship features improve the same controlled comparison
before adding interactive graph views.

Acceptance gate: relationships are traceable to sources, temporally valid at
the feature date, and tested for ambiguous, conflicting, and expired evidence.

## Cross-cutting rules

- Preserve distinct `feature_date`, `target_date`, `signal_date`, and
  `execution_date` fields.
- A news record may enter a future feature only when its UTC `available_at` is
  no later than the declared decision cutoff; never backfill missing publication
  times with historical guesses.
- Fit every learned transformation on training data only and split time series
  chronologically by target date.
- Preserve price-only baselines and use immutable experiment/model IDs; never
  overwrite `VSL-20261020-OGDC-main-02` or its referenced artifacts.
- Preserve sentiment model/output identities; never replace results created by
  a different model/tokenizer revision or text-policy configuration.
- Preserve source-permission evidence and the immutable news-coverage audit;
  metadata access never grants publisher-body reuse rights, and an import is
  never relabelled as live acquisition.
- Keep models, policies, portfolios, APIs, and dashboard presentation separate.
- Keep bulk data, fitted binaries, secrets, and run outputs out of Git.
- Document missing data, negative findings, and provider failures explicitly.

## Reinforcement learning deferral

Reinforcement learning is not part of the next phase. It remains deferred until
news ingestion, entity linking, FinBERT, daily aggregation, supervised paired
comparisons, the common LSTM contract, SHAP, multi-stock data quality, and
sector/event relationships are stable and reproducible. Before any RL work,
PAISA must also have an agreed reward definition, transaction-cost and action
constraints, walk-forward evaluation, leakage tests, and supervised/benchmark
baselines. RL must not be used to bypass unresolved data or evaluation issues.
