# News-price overlap contract

- Coverage artifact schema: `1.1`
- Coverage algorithm: `news-price-coverage-v3`
- Readiness configuration schema: `1.0`

The readiness-policy schema is intentionally versioned independently. Earlier
schema-`1.0` coverage packs remain historical byte evidence, but they lack the
current semantic and unsupported-population contract and must be regenerated
before the current verifier will certify them.

This contract defines a reproducible coverage audit between completed PAISA
news evidence and immutable daily OGDC stock/model evidence. It emits
all-population session aggregates, article-to-session detail for OGDC-linked
canonicals, and descriptive coverage metrics only. It does not define a daily
sentiment feature, prediction, signal, trade, or portfolio decision.

## Input boundary and proof requirements

An audit consumes:

1. one or more immutable `NEWS-...` ingestion packs with a checksum inventory;
2. one completed `VSL-...` vertical-slice pack and its frozen price snapshot;
3. the exact completed `TRN-...` model-training pack referenced by the vertical
   slice, including `split_manifest.json`;
4. a reviewed historical-source assessment and its checksum-bound,
   machine-readable permission policy; and
5. an explicit cutoff, market timezone, and readiness policy; and
6. only for supported energy coverage, an external checksum-bound versioned
   energy-sector link contract over the exact occurrence evidence.

The implementation must reject a required input when its declared identity,
relative path, regular-file/root-containment check, or SHA-256 does not match
the immutable manifest chain; declared byte sizes are also enforced where the
source inventory carries them. Every declared news-pack artifact is checked.
For the larger vertical-slice and training packs, the exact consumed price,
training-data-manifest, and split-manifest files are checked against their
inventories and cross-bound fingerprints. A vertical slice and model-training
run must reference the same frozen stock evidence and training identity. The
audit reads the full price period from the validated stock rows and reads
train, validation, and test membership from the exact `target_dates` arrays in
`split_manifest.json`. Displayed summaries and hard-coded date ranges are not
the source of truth.

When a news pack contains `article_entity_links.csv`, that checksum-bound file
is the authoritative OGDC-link evidence. Link IDs and required evidence fields
must be complete and unique, every linked article must exist in the normalized
table, and the linked ticker set must exactly equal that article's normalized
`mentioned_tickers`. A mismatch is rejected rather than trusting either copy.

Historical ingestion evidence is not silently upgraded or rewritten. Current
packs should carry the present completion seal. The five retained legacy RSS
packs predate that seal and therefore cannot pass the current news verifier;
they are accepted only through an explicit legacy compatibility boundary that
validates every file in their manifest checksum inventory and records
`completion_marker_present=false`. This proves retained bytes, not current-
contract completion or source permission. The audit is read-only with respect
to all source packs and SQLite databases. When a database is supplied, the
audit first creates one consistent read-only SQLite online-backup image. Every
completion-seal check and the published database SHA-256 use that same immutable
image, so a concurrent change to the live database cannot create a verification/
fingerprint mismatch.

In `--real-mode`, any input or transitive source marked `fixture`, `mock`,
`test`, `generated`, or `synthetic` is rejected. Missing provenance is not
treated as real provenance. A manual import may be accepted when it is labelled
`imported`, its raw evidence is retained, and the source assessment documents a
legitimate acquisition and usage basis; it is never relabelled as live.

`configs/news/historical_source_permissions.json` is the executable permission
boundary for the reviewed prose assessment. It binds the assessment path and
SHA-256, keys decisions by provider and acquisition mode, and may bind allowed
run IDs, an exact licensing-note SHA-256, and permission-relevant provider
controls. The audit fails on a policy/assessment checksum or label mismatch and
persists the policy fingerprint plus each run's decision. `PERMITTED` is
required for `READY_FOR_AGGREGATION`; `REVIEW_REQUIRED`, `UNASSESSED`,
`UNVERIFIED_POLICY`, run-scope mismatch, note mismatch, or provider-control
mismatch remains a hard readiness blocker. The current policy permits only the
exact GDELT run `NEWS-20261023-060000-gdelt-ogdc2` when its reviewed
metadata-only controls and licensing note match; a future run requires a new
review and explicit allowlist update. Retained RSS and dataset-specific manual
CSV evidence remain review-required.

Selected ingestion evidence may be `PASS` or retained `PARTIAL`, but that
distinction remains visible. For each run the audit records
`provider_error_count`, `rejected_record_count`, and whether ingestion is
complete. Completeness requires a `PASS` manifest and zero retained provider
errors. A `PARTIAL` or otherwise incomplete selected run may contribute
descriptive coverage, but it cannot support `READY_FOR_AGGREGATION`.

## Article identities and populations

Counts use the following terms consistently:

| Term | Meaning |
|---|---|
| occurrence | One normalized record associated with one ingestion run. Repeated acquisition remains a separate occurrence. |
| canonical article | The first retained article identity after the ordered news deduplication rules. |
| duplicate relationship | One occurrence whose `duplicate_of` identifies a canonical article. |
| represented canonical | A distinct canonical identity represented by at least one selected occurrence, including a canonical owned by an earlier completed run. |
| OGDC-linked article | A represented canonical with deterministic controlled-alias link evidence for `OGDC`; general market context is insufficient. |
| energy-sector-linked article | A represented canonical with deterministic, versioned energy-sector link evidence. Keyword guesses do not qualify. |

`total_news_occurrences` counts occurrences. Coverage-day canonical counts and
session assignments use distinct represented canonical identities so repeated
acquisition cannot inflate covered-day percentages. The coverage pack reports
the represented-canonical and aggregate duplicate-occurrence totals. Detailed
within-run/historical duplicate scopes and rule evidence remain in the
checksum-validated ingestion packs; the coverage pack does not restate them.

If the selected evidence has no deterministic sector-link contract, the
energy-sector population is `UNSUPPORTED`. It is not an empty supported
population and must not be conflated with all financial news or OGDC-linked
news.

## Time fields

All comparison instants are timezone-aware and normalized to UTC. The audit
preserves these different meanings:

| Field | Meaning in this audit |
|---|---|
| `published_at` | Complete source-supplied publication evidence; nullable. |
| `published_timezone` | Original offset/name evidence retained by ingestion. |
| `source_observed_at` | Nullable upstream index/provider observation evidence retained for audit only. It never participates in eligibility under `news-availability-v1`. |
| `fetched_at` / `imported_at` | When PAISA obtained or imported the retained occurrence. |
| `available_at` | Earliest defensible PAISA availability under the news contract. |
| session cutoff | Configured local decision clock on one observed stock-session date, converted to UTC. |

Missing, invalid, date-only, or unjustifiably naive publication evidence stays
null. The audit must not invent a publication time from `available_at`,
`fetched_at`, import time, source order, file metadata, or a stock date.

The existing news contract defines availability conservatively as the maximum
of acquisition/import time, valid publication time, and valid provider
availability evidence. Therefore a historical article imported today may have
historical publication overlap but no historical cutoff-eligible availability
overlap. Both facts are reported; publication overlap is never promoted into
leakage-safe availability.

## Publication overlap

Publication overlap is a descriptive corpus metric. For non-null
`published_at`, the audit converts the instant to the configured market
timezone and compares its local calendar date with observed OGDC trading dates
and the validated price-period bounds.

The report includes:

- earliest and latest `published_at`;
- the intersection of publication-date range and price-date range;
- the actual local calendar-date overlap with the price-period bounds; and
- publication-time completeness.

Publication overlap does not assign an article to a historical feature row and
does not contribute to cutoff-eligible coverage unless `available_at` also
passes the session rule below.

## Cutoff-eligible session mapping

Let the validated observed OGDC trading dates be ordered as
`d[0] < d[1] < ... < d[n]`. For each date, construct `cutoff[i]` by combining
`d[i]` with the configured decision clock in the configured IANA market
timezone, then convert that instant to UTC. The clock accepts only whole-second
precision; fractional seconds are rejected rather than rounded away in the
sealed configuration.

A represented canonical article is mapped once, using the canonical article's
first retained `available_at`, to the first observed session whose cutoff is at
or after availability. For sessions after the first, membership is exactly:

```text
cutoff[i - 1] < article.available_at <= cutoff[i]
```

The first bounded session includes only evidence at or after the configured
audit-window start and at or before `cutoff[0]`; earlier evidence is reported as
pre-window rather than silently carried forward. Evidence later than
`cutoff[n]` is post-window and remains unmapped.

Consequences are deterministic:

- availability exactly at a cutoff is included for that session;
- availability after a cutoff is excluded from that session and considered for
  the next observed session;
- weekend and exchange-holiday availability falls between observed cutoffs and
  is carried to the next observed session;
- no later article can map backward to an earlier trading date; and
- filesystem order, publication date, inference time, and query time do not
  affect assignment.

The audit applies this assignment once per represented canonical and persists
the resulting per-session population counts in `overlap_by_trading_day.csv`.
`article_session_mappings.csv` is the exhaustive handoff for later aggregation:
it contains one row for every selected occurrence crossed with every declared
population. Each row states whether that population is supported, whether the
occurrence belongs to it, the occurrence's own cutoff assignment, and the
population-specific canonical representative and canonical assignment. This is
important when a later duplicate supplies the first defensible OGDC link: the
general occurrence remains non-OGDC and the OGDC canonical assignment uses the
later linked occurrence rather than moving evidence backward. Rows retain IDs,
timestamps, content checksums, and raw-evidence references but no article text.

For OGDC-linked canonicals, `ogdc_linked_articles.csv` additionally records the
representative occurrence, `available_at`, assigned trading date, mapping
status, arrival class, occurrence count, the exact
`link_basis=checksum_bound_article_entity_links`, and stable JSON
`link_evidence` copied from the validated entity-link rows. Each evidence
object retains `ingestion_run_id`, `link_id`, `article_id`, `ticker`,
`company_name`, `matched_alias`, `match_method`, `match_location`, and
`confidence`. The audit does not calculate sentiment aggregates.

The audit separately counts records first available before/at a same-date
cutoff and after a same-date cutoff. The latter are never credited to that
date, even when their `published_at` is earlier.

## Split coverage

Split membership comes from the validated `split_manifest.json` target-date
arrays. For each of `train`, `validation`, and `test`, and for every supported
article population, calculate at least:

- observed trading-day count;
- days with one or more cutoff-eligible canonical articles;
- coverage percentage;
- cutoff-eligible canonical articles and articles per trading day;
- zero-article trading days;
- longest consecutive zero-article gap; and
- all consecutive missing intervals.

The denominator is the exact count of observed target dates assigned to that
split, not weekdays, calendar days, a generated exchange calendar, or summary
boundary subtraction. If a manifest target date is absent from the validated
stock observations, validation fails rather than changing the denominator.

All-financial-news, OGDC-linked, and supported deterministic energy-sector
metrics are separate. General-news coverage must never be presented as
OGDC-specific coverage.

## Required aggregate metrics

The immutable report records, at minimum:

- occurrence count, represented-canonical count, and OGDC-linked canonical
  count;
- the aggregate duplicate-occurrence count and represented-canonical count;
- first and last publication and availability instants;
- publication-date overlap and cutoff-eligible availability overlap as
  separate ranges/counts;
- train, validation, and test coverage for every supported population;
- articles per trading day and zero-article trading days;
- consecutive missing intervals;
- monthly coverage;
- source and language coverage;
- publication-time and availability-time completeness;
- before/at-cutoff, after-cutoff, weekend/holiday carry, pre-window, and
  post-window counts; and
- exact input fingerprints, effective configuration, limitations, and
  unsupported populations.

Monthly and source/language tables count occurrences and distinct canonicals in
separate fields. Neither table may collapse duplicate evidence without saying
which identity it counts.

## Readiness policy

Readiness is derived from a validated, persisted configuration. The effective
cutoff/timezone/readiness configuration is hashed into
`effective_readiness_policy_sha256` in both `run_manifest.json` and
`final_summary.json`; changing a threshold requires a new audit identity.
Readiness-configuration schema `1.0` configures all-news and OGDC-linked minimum
covered-day percentages for every train/validation/test split, the maximum
all-news missing-session gap, minimum distinct source count, and publication/
availability timestamp completeness. Minimum canonical counts and a required
energy-sector population are not configurable in this schema and must not be
claimed as gates.

Validation rejects booleans/non-integers for integer fields, requires every
percentage threshold to be positive, and requires ready coverage thresholds to
be at least as strict as their partial counterparts. Configurable ready floors
are `5%` all-news coverage in each split, `1%` OGDC coverage in each split,
`1%` publication/availability completeness, and two distinct sources. In
addition, a non-configurable engineering guard requires at least two covered
sessions in every split for both all-news and OGDC. These floors prevent sparse
evidence from being configured as ready; they do not establish statistical
sufficiency.

Status meanings are:

| Status | Meaning |
|---|---|
| `NO_OVERLAP` | Neither publication dates nor cutoff-eligible `available_at` evidence overlap the validated price period. |
| `INSUFFICIENT_OVERLAP` | Some publication overlap exists, but there is no usable cutoff-eligible overlap or the minimum usable evidence gate fails. |
| `PARTIALLY_COVERED` | Cutoff-eligible evidence exists and the partial thresholds pass, but one or more readiness gates fail; this is also the maximum status when otherwise-ready source evidence still requires permission review, contains incomplete ingestion runs, or lacks authoritative database-bound completion evidence. |
| `READY_FOR_AGGREGATION` | Every configured engineering/data-coverage gate passes for all required supported populations and splits, every source decision is permitted, all selected ingestion runs are complete, and every selected current-contract pack replayed against the fingerprinted SQLite seal. |

Source permission, ingestion completeness, and database provenance are hard
prerequisites rather than configurable coverage thresholds.
`source_permission_evidence` reports unresolved review notes.
`ingestion_completeness` reports `passed`, the sorted `incomplete_run_ids`, and
aggregate `provider_error_count` and `rejected_record_count`.
`database_provenance` reports `passed`,
`authoritative_completion_seal_available`, `database_fingerprinted`, sorted
`unsealed_run_ids`, and the reason. Omitting the SQLite fingerprint creates a
valid artifact-only replay but demotes an otherwise-ready result. A database
fingerprint alone is not sufficient: a legacy pack accepted through
`--allow-legacy-unsealed` has no current completion/database seal and remains
non-authoritative even when that database was supplied. These prerequisites do
not replace the more fundamental `NO_OVERLAP` or `INSUFFICIENT_OVERLAP` result
when usable coverage is absent. In particular, zero cutoff-eligible coverage
is `NO_OVERLAP` when publication dates also do not intersect the price period
and `INSUFFICIENT_OVERLAP` when publication dates do intersect it.

`READY_FOR_AGGREGATION` means only that the configured data-engineering gates
passed, no unresolved source-permission marker remained, and all selected
ingestion runs were complete, with authoritative database-bound completion
evidence for every selected pack. It is not proof of statistical power,
representativeness, classifier accuracy, causal relevance, predictive
improvement, profitability, or fitness for financial decisions. Energy-sector
support is reported explicitly and is never inferred from general news. It
requires an explicit checksum-bound, versioned energy-link contract; arbitrary
`mentioned_sectors` values cannot enable it. Without that contract, energy
coverage metrics and gaps are `null`/omitted rather than measured zeroes.
Readiness-configuration schema `1.0` does not make energy a readiness gate.
The contract checksum covers the exact ordered occurrence/article/canonical
identities, content hashes, and normalized sector lists, and the offline
verifier recomputes it from the mapping artifact.

The final summary persists every configured threshold and the decisive overlap
reason. When cutoff-eligible evidence exists, it also reports the observed
values and pass/fail result for each evaluated readiness gate. A `NO_OVERLAP`
short circuit therefore carries an empty threshold-check map rather than
pretending that coverage gates were evaluated.

## Artifact contract

A completed run owns exactly one collision-refusing `COVERAGE-...` directory:

```text
artifacts/news_coverage/<run_id>/
├── run_manifest.json
├── source_assessment_reference.json
├── imported_or_acquired_news_reference.json
├── stock_artifact_reference.json
├── overlap_by_trading_day.csv
├── overlap_by_month.csv
├── split_coverage.csv
├── missing_intervals.csv
├── article_session_mappings.csv
├── ogdc_linked_articles.csv
├── quality_report.json
├── final_summary.json
└── completion.json
```

The files have these roles:

- `run_manifest.json` records the run identity, status, producer/code identity,
  effective cutoff/timezone/readiness policy and its SHA-256, completion time,
  and the sorted SHA-256/size inventory of every payload file.
- `source_assessment_reference.json` fingerprints the reviewed assessment and
  its machine-readable permission policy, then records the exact permission
  decision and provider/mode/licensing evidence associated with each selected
  run. It does not convert an assessment finding such as
  `LICENSING_REVIEW_REQUIRED` into authorization.
- `imported_or_acquired_news_reference.json` fingerprints every selected news
  pack and the one immutable SQLite online-backup image used for all seal
  verification, and records provider mode, data classes, licensing/usage notes,
  occurrence/canonical counts, and the optional checksum-bound versioned
  energy-link contract without copying prohibited text. Visible WAL state is
  folded into that single consistent snapshot.
- `stock_artifact_reference.json` fingerprints the vertical slice, frozen price
  file, model-training run, and split manifest.
- `overlap_by_trading_day.csv` contains one row per observed price session with
  its model-split label (or `outside_model_splits`), cutoff, and separate
  population counts. Unsupported-population count cells are empty.
- `overlap_by_month.csv` contains calendar-month totals and covered-session
  percentages by population, plus `population_supported`; measured fields are
  empty when support is false.
- `split_coverage.csv` contains the exact split denominators and coverage
  metrics consumed by the readiness gates, plus `population_supported` and
  empty measured fields for unsupported populations.
- `missing_intervals.csv` contains inclusive first/last session dates and
  observed-session length for every zero-article run. It has no rows for an
  unsupported population because an unmeasured gap is not a zero.
- `article_session_mappings.csv` contains every occurrence/population pair,
  membership and support flags, the occurrence assignment, and the
  population-specific canonical representative and assignment. It also keeps
  the sanitized source URL and the optional reviewed-publisher document key so
  candidate duplicate groups can be independently reconciled. Its normalized
  sector list supports independent energy-contract checksum replay. It is the
  checksum-bound, article-text-free handoff for a later aggregation component.
- `ogdc_linked_articles.csv` contains only permitted metadata, deterministic
  link evidence, timing, canonical and representative occurrence identities,
  occurrence totals, and assigned session. Its `link_basis` is
  `checksum_bound_article_entity_links`, and `link_evidence` is stable JSON of
  the validated per-run evidence rows; it is not a sentiment dataset.
- `quality_report.json` records timestamp and ingestion completeness, aggregate
  duplicate, timing, source, language, unsupported-population, database-
  provenance, and validation findings. Canonical identity is explicitly
  qualified to the current exact rules; reviewed publisher-document patterns
  report candidate duplicate groups separately without merging them.
- `final_summary.json` contains the readiness result, effective readiness-policy
  SHA-256, decisive gate reasons, source-permission, ingestion-completeness, and
  database-provenance evidence, core coverage facts, limitations, and the
  academic-use disclaimer.
- `completion.json` is written last as the non-circular commit point. It binds
  the exact `run_manifest.json` and `final_summary.json` bytes plus the payload
  inventory digest and count. The manifest and completion marker are excluded
  from the payload inventory because one cannot checksum the other without a
  circular dependency.

Every payload artifact is checksum-bound by the run evidence. Publication first
recomputes the complete coverage result from the supplied `NewsEvidence`,
`StockEvidence`, and `CoverageConfig`; a caller cannot publish a separately
constructed or subsequently altered `CoverageComputation`. It then performs
the same byte and internal semantic reconciliation used by pack-only
verification before the staging directory is renamed. Source-bound
verification remains a distinct replay against the persisted inputs.
References and paths in public artifacts are portable and relative to declared
roots. API keys, authorization values, credential-bearing URLs, host usernames,
and host-specific absolute paths are prohibited. When coverage CSV text begins
with a spreadsheet-formula prefix (`=`, `+`, `-`, `@`, tab, or carriage
return), publication adds a reversible leading apostrophe and verification
decodes it before semantic comparison. Existing run paths or
identities cause failure; a per-run exclusive publication reservation
serializes cooperating publishers, and the final staging rename uses the
operating system's atomic no-replace primitive. A run is never overwritten or
merged in place. The rename is the externally visible commit point. If removal
of the private reservation file fails after that rename, publication returns
the committed PASS directory and emits a cleanup warning instead of falsely
reporting that the run failed.

Verification is offline in both modes. The default is source-bound: it reopens
the exact news runs and SQLite state plus stock, split, assessment, and
permission evidence, recomputes coverage, and compares the results. Replace
these valid example IDs with the immutable IDs produced or restored in your
checkout:

```bash
PAISA_COVERAGE_RUN_DIR="artifacts/news_coverage/COVERAGE-20261023-190000-example-v3"
PAISA_NEWS_RUN_DIR="artifacts/news_ingestion/NEWS-20261023-180000-example-source"
PAISA_VERTICAL_RUN_DIR="artifacts/vertical_slice/VSL-20261020-OGDC-main-02"
PAISA_TRAINING_RUN_DIR="artifacts/model_training/TRN-20261020-191815-624662-nogit"
python scripts/verify_news_coverage.py \
  --run-dir "$PAISA_COVERAGE_RUN_DIR" \
  --news-run-dir "$PAISA_NEWS_RUN_DIR" \
  --news-database data/news/paisa_news.sqlite3 \
  --vertical-slice-dir "$PAISA_VERTICAL_RUN_DIR" \
  --model-training-dir "$PAISA_TRAINING_RUN_DIR" \
  --source-assessment docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md \
  --source-permission-policy configs/news/historical_source_permissions.json
```

The equivalent API is `paisa.news.coverage.verify_coverage_run(...)`, with the
source paths supplied as required keyword arguments. Success reports
`verification_basis=SOURCE_EVIDENCE_BOUND` and `source_evidence_bound=true`.
When sealed references declare database or permission-policy evidence, those
exact inputs are mandatory. A pack with supported energy coverage also
requires the external versioned contract through
`--energy-sector-link-contract`; it is rejected when absent, mismatched, or
supplied for an unsupported pack.

If the referenced source evidence is unavailable, a caller may explicitly ask
for the weaker pack-semantic check:

```bash
python scripts/verify_news_coverage.py \
  --run-dir "$PAISA_COVERAGE_RUN_DIR" \
  --pack-only
```

The API is `paisa.news.coverage.verify_coverage_pack(run_directory)`. It
reports `verification_basis=PACK_SEMANTIC` and `source_evidence_bound=false`;
its PASS must never be represented as source-bound correctness. Pack-semantic
verification rejects manifest/payload tampering, missing or extra payloads,
unsafe paths, symlinks, identity/configuration disagreement, and an absent or
inconsistent completion marker. It parses and reconciles the summary, quality
report, article mappings, trading-day/month/split tables, missing intervals,
OGDC rows, effective policy fingerprint, portable input fingerprint, and
sealed reference metadata. Source-bound mode additionally reloads the source
bytes and requires replayed outputs to match. Neither mode re-downloads a
provider response; both use captured evidence. Neither is a digital signature
or proof of who produced the files. Both modes reject historical schema-`1.0`
/ v2 packs under current code.

`run_manifest.json.status=PASS` means that the audit itself completed and its
artifacts were published successfully. It is independent of
`readiness_status`; a valid completed audit may therefore be `NO_OVERLAP`.

## Determinism and validation

Re-running the calculation over identical captured news packs, stock/split
bytes, cutoff/timezone, readiness policy, and code identity must reproduce the
same data tables and metrics. Execution timestamps and the new run ID are
provenance fields and do not alter assignments. All tables use stable ordering
and canonical timestamp/JSON serialization before checksumming.

Automated validation covers zero, partial, and complete overlap; exact split
coverage; weekend carry; after-cutoff exclusion; future-news leakage; OGDC and
general-news separation; missing publication time; timezone conversion;
duplicates and canonical counts; missing gaps; readiness states; real-mode
fixture rejection; immutable run collisions; checksum disagreement; portable
path and credential hygiene; completion-seal/manifest tampering; and
deterministic replay from captured evidence.

The retained October 2026 real news evidence is later than the stock artifact
ending September 15, 2026, and the existing
`COVERAGE-20261023-100000-preflight003` evidence therefore reports
`NO_OVERLAP`. That preflight predates the current coverage completion seal. The
sealed official `COVERAGE-20261023-120000-real0002` artifact verified `PASS` at
its producing revision and retained `NO_OVERLAP`; the separate sealed GDELT
result `COVERAGE-20261023-130000-gdelt0001` did likewise. Both are historical
schema-`1.0` / `news-price-coverage-v2` evidence and the current verifier
requires regeneration under new run IDs before certification. Fixture results
may test other statuses but must
never replace or be presented as the real evidence.
