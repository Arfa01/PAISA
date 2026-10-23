# News-price overlap contract

Schema version: `1.0`

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
5. an explicit cutoff, market timezone, and readiness policy.

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
timezone, then convert that instant to UTC.

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
Schema `1.0` configures all-news and OGDC-linked minimum
covered-day percentages for every train/validation/test split, the maximum
all-news missing-session gap, minimum distinct source count, and publication/
availability timestamp completeness. Minimum canonical counts and a required
energy-sector population are not configurable in this schema and must not be
claimed as gates.

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
support is reported explicitly and is never inferred from general news; schema
`1.0` does not make it a readiness gate.

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
  and occurrence/canonical counts without copying prohibited text. Visible WAL
  state is folded into that single consistent snapshot.
- `stock_artifact_reference.json` fingerprints the vertical slice, frozen price
  file, model-training run, and split manifest.
- `overlap_by_trading_day.csv` contains one row per observed price session with
  its model-split label (or `outside_model_splits`), cutoff, and separate
  population counts.
- `overlap_by_month.csv` contains calendar-month totals and covered-session
  percentages by population.
- `split_coverage.csv` contains the exact split denominators and coverage
  metrics consumed by the readiness gates.
- `missing_intervals.csv` contains inclusive first/last session dates and
  observed-session length for every zero-article run.
- `article_session_mappings.csv` contains every occurrence/population pair,
  membership and support flags, the occurrence assignment, and the
  population-specific canonical representative and assignment. It is the
  checksum-bound, text-free handoff for a later aggregation component.
- `ogdc_linked_articles.csv` contains only permitted metadata, deterministic
  link evidence, timing, canonical and representative occurrence identities,
  occurrence totals, and assigned session. Its `link_basis` is
  `checksum_bound_article_entity_links`, and `link_evidence` is stable JSON of
  the validated per-run evidence rows; it is not a sentiment dataset.
- `quality_report.json` records timestamp and ingestion completeness, aggregate
  duplicate, timing, source, language, unsupported-population, database-
  provenance, and validation findings.
- `final_summary.json` contains the readiness result, effective readiness-policy
  SHA-256, decisive gate reasons, source-permission, ingestion-completeness, and
  database-provenance evidence, core coverage facts, limitations, and the
  academic-use disclaimer.
- `completion.json` is written last as the non-circular commit point. It binds
  the exact `run_manifest.json` and `final_summary.json` bytes plus the payload
  inventory digest and count. The manifest and completion marker are excluded
  from the payload inventory because one cannot checksum the other without a
  circular dependency.

Every payload artifact is checksum-bound by the run evidence, and publication
performs the same full verification before the staging directory is renamed.
References and paths in public artifacts are portable and relative to declared
roots. API keys, authorization values, credential-bearing URLs, host usernames,
and host-specific absolute paths are prohibited. Existing run paths or
identities cause failure; a per-run exclusive publication reservation
serializes cooperating publishers, and the final staging rename uses the
operating system's atomic no-replace primitive. A run is never overwritten or
merged in place.

Verify a completed pack offline with:

```bash
python scripts/verify_news_coverage.py \
  --run-dir artifacts/news_coverage/<run_id>
```

The equivalent API is
`paisa.news.coverage.verify_coverage_run(run_directory)`. Verification rejects
manifest or payload tampering, missing or extra payloads, unsafe paths,
symlinks, identity/configuration disagreement, and an absent or inconsistent
completion marker. This seal provides artifact-integrity and completion
consistency; it is not a digital signature or proof of who produced the files.

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
sealed official `COVERAGE-20261023-120000-real0002` artifact verifies `PASS` and
retains `NO_OVERLAP`; the separate sealed GDELT result
`COVERAGE-20261023-130000-gdelt0001` also verifies `PASS` and retains
`NO_OVERLAP`. Fixture results may test other statuses but must never replace or
be presented as the real evidence.
