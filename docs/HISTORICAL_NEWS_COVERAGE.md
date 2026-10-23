# Historical news coverage and overlap audit

PAISA's historical-news workflow measures whether retained financial-news
evidence overlaps the immutable OGDC daily-price and model-split evidence. It
does not aggregate sentiment, train a price-plus-sentiment model, change a
decision policy or portfolio, or claim that news improves prediction. All
outputs are academic data-engineering evidence and are not financial advice.

Newly generated packs use coverage schema `1.1` and
`news-price-coverage-v3`. The independent readiness configuration remains
schema `1.0`. Retained schema-`1.0` coverage packs are historical and require
regeneration before the current semantic verifier will certify them.

## Permission comes before acquisition

Review [`HISTORICAL_NEWS_SOURCE_ASSESSMENT.md`](HISTORICAL_NEWS_SOURCE_ASSESSMENT.md)
before configuring a source. A source is eligible only when its documented
access method and the intended storage, automated retrieval, attribution, and
academic use are reasonably permitted. PAISA does not bypass authentication,
paywalls, CAPTCHAs, or rate limits and does not scrape pages whose terms prohibit
that access.

The prose assessment is paired with
`configs/news/historical_source_permissions.json`. That machine-readable policy
checksum-binds the reviewed assessment and records provider/mode decisions,
allowed scope, allowlisted run IDs, licensing-note hashes, and
permission-relevant provider controls. The audit persists the policy
fingerprint and each selected run's decision. The current policy permits only
the exact GDELT run `NEWS-20261023-060000-gdelt-ogdc2` when its reviewed
metadata-only controls and licensing note match. A future GDELT run fails closed
until it is reviewed and explicitly allowlisted. Retained RSS and
dataset-specific manual CSV evidence remain `REVIEW_REQUIRED` and cannot support
`READY_FOR_AGGREGATION` until the policy is reviewed and replaced with
appropriately scoped evidence.

The supported historical paths are deliberately narrow:

1. Import a legitimately obtained CSV through the existing `manual-csv`
   provider. The operator remains responsible for the dataset's license,
   attribution, and permitted stored fields.
2. Use a reviewed archive/API adapter only after recording its access and usage
   basis in the source assessment. The initial configurable archive path is
   GDELT, subject to its published API documentation, service limits, and the
   rights of the underlying publishers. GDELT metadata access does not grant
   PAISA permission to republish publisher body text. Provider behavior and
   terms can change, so the assessment must be reviewed at execution time.

Credentials are supplied through the provider's documented runtime mechanism,
never committed in configuration or copied into artifacts. A provider failure
must remain a failure; fixture, generated, or synthetic articles are never a
historical-data fallback.

## Import a permitted historical CSV

The existing importer preserves the supplied CSV and normalized row evidence:

```bash
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /absolute/path/to/permitted_historical_news.csv \
  --source-name reviewed-dataset-name \
  --licensing-usage-note "Licence, academic-use basis, permitted fields, retention, and attribution reviewed on YYYY-MM-DD" \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

The input should preserve a source record ID or URL, title, permitted summary
or body, complete publication time with timezone evidence, source availability
time when the provider supplies one, language, source identity, and licensing
notes outside the article text. See [`NEWS_DATA_CONTRACT.md`](NEWS_DATA_CONTRACT.md)
for accepted fields and normalization rules. Missing `published_at` is retained
as missing; a date-only value is not converted to an invented midnight.

An imported row is always `imported`, never `real` or live retrieval. Do not
rename, edit, or relabel fixture data to make it acceptable to a real audit.
The licensing/usage note is required for a manual import and is retained in
provider configuration and row-level raw evidence; it is evidence of the
operator's review, not a licence created by PAISA.

## Acquire bounded GDELT metadata

The selected automated provider uses the public GDELT DOC 2.0 ArticleList API
in metadata-only mode. Review and edit the bounded query windows, permission
basis, licensing note, and required attribution before every real run:

```bash
python scripts/run_news_pipeline.py \
  --provider gdelt-doc \
  --config configs/news/gdelt_ogdc_history.example.json \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

The adapter uses a fixed official endpoint, explicit UTC windows, sequential
bounded requests, deterministic saturation splitting, and bounded retry/backoff
for rate limits and server errors. It rejects redirects and never fetches the
returned publisher URLs, bodies, or images. A capped result that cannot be
split safely fails closed instead of being presented as complete coverage.

### Retained live GDELT evidence

The completed metadata-only acquisition `NEWS-20261023-060000-gdelt-ogdc2` is
a sealed `PARTIAL` pack with `WARN` quality. Offline verification against the
database and artifact seals returns `PASS (DATABASE_AND_ARTIFACT_BOUND)`. It
preserves `656` raw rows and `656` normalized canonical rows unique under the
current exact URL/content/title-source-time rules, including `42` exact
checksum-bound OGDC-linked rows across `93` returned source domains. There are
no exact-rule duplicate relationships and no rejected records. This is not an
article-level uniqueness claim: the conservative reviewed-publisher document-ID
diagnostic flags `17` candidate groups (`36` rows, or `19` beyond one per
group) for later adjudication. Candidate reporting does not silently rewrite
the immutable retained rows, and the `42` OGDC links are unaffected.

All `656` canonicals preserve missing `published_at`; GDELT `seendate` is not
relabeled as publisher publication time. The raw source-observation evidence
spans `2023-06-18T12:15:00Z` through `2024-06-14T15:00:00Z`, while conservative
`available_at` spans `2026-10-23T06:26:32.440746Z` through
`2026-10-23T06:29:25.858147Z`. The pack is `PARTIAL` because it records `22`
provider errors: `15` `ReadTimeout` failures, `4` out-of-window responses quarantined
rather than accepted, and `3` exhausted HTTP 429 responses. A valid seal proves
the retained pack's integrity; it does not erase incomplete acquisition.
This retained schema-`1.0` pack preserves the observation range in raw
evidence. New schema-`1.1` runs expose it as nullable `source_observed_at` in
normalized SQLite/CSV and coverage artifacts. The field remains audit-only and
cannot change an eligibility mapping under `news-availability-v1`.
The retained provider-error rows also predate schema-`1.1` request contexts and
therefore identify only their configured root window. Regenerate the GDELT run
to obtain exact capture/attempt, split-path, request-bound, fingerprint, and
retry-delay evidence for failures; the old `22`-error pack is not retroactively
certified with facts it did not retain.

### Historical publication is not historical availability

The news time contract calculates:

```text
available_at = max(imported_at/fetched_at, valid published_at,
                   valid provider availability evidence)
```

Consequently, importing an old article today can establish that its recorded
publication date overlaps an old price period, but it cannot establish that
PAISA possessed the article at a historical decision cutoff. The current import
time will be later and therefore controls `available_at`. Calling a historical
API today has the same limitation unless a future, separately reviewed evidence
contract establishes contemporaneous availability without weakening the
anti-leakage rule.

The audit reports publication-date overlap and cutoff-eligible availability
overlap separately. It never treats publication overlap alone as proof that a
historical feature was available for leakage-safe model training.

## Run the overlap audit

First identify completed, immutable inputs:

- one or more completed news-ingestion run directories;
- the completed vertical-slice directory containing the frozen price snapshot;
- the exact model-training directory containing `split_manifest.json`; and
- the reviewed source assessment plus its checksum-bound machine-readable
  permission policy.

`--news-run-dir` is repeatable. Every supplied run remains a distinct occurrence
source while canonical identities prevent repeated acquisition from inflating
trading-day coverage. Prefer an explicit run list for a reproducible retained
audit; `--news-root` is a convenience for exploratory audits and will also
discover future `PASS`/`PARTIAL` children.

**Artifact-present retained example:** the command below requires the ignored
news/database evidence and retained `VSL-...`/`TRN-...` pair. They are not in a
Git-only clone. Restore a checksum-verified custodian export as documented in
[`TEAM_HANDOFF.md`](TEAM_HANDOFF.md), or substitute regenerated inputs and a
fresh coverage run ID.

```bash
python scripts/audit_news_price_overlap.py \
  --news-run-dir artifacts/news_ingestion/NEWS-20261022-115232-03d0fc4c \
  --news-run-dir artifacts/news_ingestion/NEWS-20261022-120012-31d811b1 \
  --news-run-dir artifacts/news_ingestion/NEWS-20261022-120133-43a51981 \
  --news-run-dir artifacts/news_ingestion/NEWS-20261022-130023-470cb7a7 \
  --news-run-dir artifacts/news_ingestion/NEWS-20261022-131242-5ddffd2d \
  --vertical-slice-dir artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --model-training-dir artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --source-assessment docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md \
  --source-permission-policy configs/news/historical_source_permissions.json \
  --output-dir artifacts/news_coverage \
  --run-id COVERAGE-20261023-120000-example001 \
  --decision-cutoff 15:30:00 \
  --market-timezone Asia/Karachi \
  --real-mode \
  --allow-legacy-unsealed \
  --readiness-config configs/news_coverage/default_readiness.json
```

Use `python scripts/audit_news_price_overlap.py --help` for the authoritative
option syntax. The decision cutoff is a local market clock time interpreted in
the named IANA timezone; all persisted comparison timestamps remain UTC.
Cutoffs accept `HH:MM` or `HH:MM:SS` at whole-second precision. Fractional
seconds are rejected so serialized and hashed configuration never loses
assignment-relevant precision.

The retained RSS and live-GDELT packs have separate sealed schema-`1.0`
historical audits. These artifact-present commands document their original
verification entrypoints, but current schema-`1.1` code intentionally rejects
them and requires regeneration under new run IDs:

```bash
python scripts/verify_news_coverage.py \
  --run-dir artifacts/news_coverage/COVERAGE-20261023-120000-real0002
python scripts/verify_news_coverage.py \
  --run-dir artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001
```

At their producing revision, both verifiers returned `PASS`. Both runs were generated from clean commit
`041c557edf5b15660fd2f02d47ed7f9b8b4152a7` with `dirty_tree=false`, contain
`11` payload artifacts, and retain `NO_OVERLAP`:

- `COVERAGE-20261023-120000-real0002` contains `450` mapping rows. Source
  permission and legacy database provenance fail; ingestion completeness
  passes. Manifest SHA-256 is
  `57d1f2e3b099318062e4f094c127de7d130953e38b76564684c7427fa3139b54`;
  inventory SHA-256 is
  `1316d7fa9164bb8d091664cc102f8a5b4ad89d4760c7caca7fe62d13142d29a8`.
- `COVERAGE-20261023-130000-gdelt0001` contains `1,968` mapping rows. Source
  permission and database provenance pass; ingestion completeness fails with
  `22` provider errors. Manifest SHA-256 is
  `9bb6385196d79befde21d422eb28a78d8aa5222b0863ea8d44b330fad3e0968a`;
  inventory SHA-256 is
  `51d170ce3a0e26aa743d37b9e32335bfbfe0e1bdf9abee89e0d3683b748e0673`.

For an independent reproduction, use the same documented inputs and options
with a fresh unique `--run-id`; never reuse either retained official ID.

Real mode is the default. It rejects any news or stock evidence marked fixture,
mock, test, generated, or synthetic, and it does not infer legitimacy from a
filename. `--allow-fixture` exists only for deterministic automated tests and
must never be used for a real-data readiness claim.

Before computing metrics, the command validates the selected news packs, the
vertical-slice manifest, the frozen stock snapshot, the model-training run, the
split manifest, their declared identities, and their SHA-256 bindings. Split
membership is read from the exact `target_dates` arrays in the immutable
`split_manifest.json`; command-line dates are not a substitute.

For OGDC attribution, the audit also validates checksum-bound
`article_entity_links.csv` rows and requires their exact ticker sets to match
the normalized articles. `ogdc_linked_articles.csv` then carries
`link_basis=checksum_bound_article_entity_links` plus stable JSON
`link_evidence`; a normalized ticker alone is not treated as sufficient proof.
The separate `article_session_mappings.csv` is exhaustive: every selected news
occurrence is crossed with every declared population and records occurrence
membership, support, its own session assignment, and the population-specific
canonical representative/assignment needed by a later aggregation stage.

The five retained legacy RSS packs predate the current completion marker and
cannot pass the current news verifier. The audit accepts them only through a
visible `--allow-legacy-unsealed` boundary: it validates every file in each
declared checksum inventory, records that the completion marker is absent, and
makes no claim that this upgrades them to the current ingestion contract or
resolves source permission. New packs should carry the current completion seal.
Even when the current SQLite file is fingerprinted, these legacy packs have no
current database-bound completion seal. They therefore appear in
`database_provenance.unsealed_run_ids`, make that prerequisite fail, and cannot
support `READY_FOR_AGGREGATION`.

By default the command opens the current SQLite database read-only and creates
one consistent private image with SQLite's online-backup API. Visible WAL state
is folded into that image. Every selected completion seal is checked against
that exact immutable snapshot, and the published database SHA-256 fingerprints
the same bytes; a concurrent live write therefore cannot split verification
from provenance. `--news-database ''` deliberately omits the snapshot for
artifact-only replay and creates a different audit identity. The
`database_provenance` block marks that replay as non-authoritative, and missing
database provenance prevents an otherwise-ready result from becoming
`READY_FOR_AGGREGATION`. Its fields distinguish whether a database was
fingerprinted from whether every selected pack has an authoritative completion/
database seal; it must not be described as the retained real run.

## Session assignment

For every observed OGDC session, the audit constructs a cutoff timestamp from
the session date, decision clock, and market timezone. A unique canonical
article is eligible for the first observed session whose interval contains its
`available_at`:

```text
previous observed session cutoff < available_at <= current session cutoff
```

This makes the boundary inclusive without allowing future evidence to move
backward. News first available after a session cutoff moves to the next
observed session. News first available on a weekend or exchange holiday is
carried to the next observed session. Evidence beyond the final observed cutoff
is reported as unmapped and never forced into the price period.

Publication-date overlap is a descriptive corpus measure only. The leakage-safe
coverage and readiness measures use the availability-to-cutoff mapping. Missing
publication time remains missing and affects completeness metrics; it is never
reconstructed from availability, import time, a filesystem timestamp, or a
price date.

See [`NEWS_PRICE_OVERLAP_CONTRACT.md`](NEWS_PRICE_OVERLAP_CONTRACT.md) for the
complete counting and mapping contract.

## Immutable output

Each successful invocation publishes a new collision-refusing directory:

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

The run manifest records portable relative references, the effective cutoff,
timezone, readiness policy, input fingerprints, code identity, and a sorted
SHA-256 payload inventory. `completion.json` is written last and binds the
exact manifest and final-summary bytes plus that inventory digest and count;
the manifest and completion marker are intentionally outside the payload
inventory to avoid a circular checksum. Both the run manifest and final summary
record the effective cutoff/timezone/readiness configuration hash as
`effective_readiness_policy_sha256`. Existing run IDs and directories are
never overwritten; publication uses a per-run exclusive reservation and an
atomic no-replace staging rename. Generated coverage packs remain ignored by
Git. Public artifacts must not contain host-specific absolute paths, API keys,
authorization headers, query credentials, or article text that the source did
not permit PAISA to store. Coverage CSV publication also neutralizes
spreadsheet-formula prefixes (`=`, `+`, `-`, `@`, tab, or carriage return) in
untrusted cells with a reversible leading apostrophe; semantic verification
decodes that representation before comparison.

Current verification is offline but source-bound by default. It reopens the
exact captured news packs, SQLite database, stock prices, split manifest,
assessment, and permission policy; recomputes coverage; and compares every
result artifact. Repeat `--news-run-dir` for the exact ordered set used by the
audit (or use `--news-root` for deliberate discovery). Replace these valid
example IDs with the immutable IDs produced or restored in your checkout:

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

The source-bound API is `paisa.news.coverage.verify_coverage_run(...)`; its
required keyword arguments identify the exact news, vertical-slice, training,
assessment, database, and policy evidence. A successful result reports
`verification_basis=SOURCE_EVIDENCE_BOUND` and
`source_evidence_bound=true`.
If the sealed references declare a SQLite database or permission policy,
source-bound verification fails unless that exact input is supplied. If the
pack measures the optional energy population, also pass the exact external
versioned contract with `--energy-sector-link-contract`; supplying one for an
unsupported pack, or omitting one for a supported pack, fails closed.

When source evidence is genuinely unavailable, the deliberately weaker
sealed-pack check must be requested explicitly:

```bash
python scripts/verify_news_coverage.py \
  --run-dir "$PAISA_COVERAGE_RUN_DIR" \
  --pack-only
```

Its API is `paisa.news.coverage.verify_coverage_pack(run_directory)`. It checks
bytes and internal semantic reconciliation, but reports
`verification_basis=PACK_SEMANTIC` and `source_evidence_bound=false`; it must
not be described as source-bound correctness or accepted in place of source
replay. Both modes reject historical schema-`1.0` / v2 packs under current
code. Neither mode is a digital signature or producer-authentication
mechanism.

Use the recorded input fingerprints and effective configuration to reproduce
the same audit from captured evidence without another provider request. A
changed source pack, stock snapshot, split manifest, cutoff, timezone, or
readiness policy is a different audit and requires a new run ID.

## Reading the result

The report keeps these populations separate:

- all retained financial-news canonicals;
- canonicals with deterministic OGDC link evidence; and
- energy-sector canonicals only when deterministic sector evidence exists.

General financial news is never reported as OGDC-specific news. The current
news contract does not infer sectors, so energy-sector coverage is explicitly
`UNSUPPORTED` rather than guessed from broad keywords. A populated
`mentioned_sectors` field alone cannot enable support: a checksum-bound,
explicit versioned `EnergySectorLinkContract` is required. Unsupported energy
counts, percentages, gaps, and readiness inputs are serialized as `null`/empty
CSV cells, not as measured zeroes. The contract digest covers the exact
occurrence/article/canonical identities, content hashes, and normalized sector
lists and is replayed by offline verification.

The final readiness status is one of:

- `NO_OVERLAP`;
- `INSUFFICIENT_OVERLAP`;
- `PARTIALLY_COVERED`; or
- `READY_FOR_AGGREGATION`.

The thresholds are explicit configuration for engineering and data-coverage
gating. They are not proof of sample size, sentiment quality, predictive value,
or statistical sufficiency. Configuration validation rejects booleans and
non-integers for integer fields, requires positive percentage thresholds,
requires ready thresholds to be at least as strict as partial thresholds, and
enforces ready floors of `5%` all-news coverage, `1%` OGDC coverage, `1%`
timestamp completeness, and two distinct sources. A non-configurable guard also
requires at least two cutoff-eligible covered sessions in every model split for
both all-news and OGDC. These are engineering sanity floors, not statistical
power claims; a small amount of overlap cannot be configured as
`READY_FOR_AGGREGATION`.

Source permission is a separate prerequisite, reported in
`source_permission_evidence`. Evidence still marked for permission review
cannot produce `READY_FOR_AGGREGATION`, even if every configured coverage gate
passes. Ingestion completeness is another hard prerequisite. The
`ingestion_completeness` block reports sorted `incomplete_run_ids` plus aggregate
`provider_error_count` and `rejected_record_count`; any selected `PARTIAL` or
otherwise incomplete run prevents an otherwise-ready result. These
prerequisites are joined by `database_provenance`: a supplied SQLite database
fingerprint and a current database-bound completion seal for every selected pack
are required for `READY_FOR_AGGREGATION`, while an artifact-only or legacy-
unsealed replay remains valid descriptive evidence. They do not hide a more
basic overlap result: no publication-date or cutoff-eligible overlap is
`NO_OVERLAP`, while publication overlap without any cutoff-eligible canonical
is `INSUFFICIENT_OVERLAP`.

For the retained October 19--22, 2026 news snapshot and the OGDC price artifact
ending September 15, 2026, the existing preflight
`COVERAGE-20261023-100000-preflight003` correctly reports `NO_OVERLAP`. It
predates the current coverage completion seal and is descriptive rather than
the official final. The sealed official RSS audit
`COVERAGE-20261023-120000-real0002` verified `PASS` at its producing revision
and retains `NO_OVERLAP`; it remains separate even if a later permitted
historical dataset produces publication overlap.

The historical schema-`1.0` / `news-price-coverage-v2` preflight
`COVERAGE-20261023-110000-gdeltpre1` for
`NEWS-20261023-060000-gdelt-ogdc2` is also `NO_OVERLAP`, for a different evidence
profile. Source permission and database
provenance pass, but ingestion completeness fails because the selected pack is
`PARTIAL` and records `22` provider errors. With all `published_at` values
missing and every conservative `available_at` after the price period, the
publication and availability calendar-overlap ranges are both `null`; all-news
and OGDC-linked coverage values are `0%`. That historical schema-`1.0` pack also
stored unsupported energy as `0%`; regeneration under schema `1.1` instead
emits `null`/`N/A`. This preflight remains descriptive historical evidence. The
sealed official `COVERAGE-20261023-130000-gdelt0001` artifact verified `PASS`
at its producing revision, retains the same `NO_OVERLAP` readiness result, and
does not unblock aggregation. Both retained official packs are schema `1.0` /
`news-price-coverage-v2`; the current schema-`1.1` /
`news-price-coverage-v3` semantic verifier requires recomputation under new run
IDs and intentionally does not certify them.

## Explicitly out of scope

This workflow does not implement daily sentiment aggregation, stock-feature
generation, model training, model comparison, portfolio changes, knowledge
graphs, SHAP, reinforcement learning, or dashboard presentation. It must not be
used to claim that sentiment predicts returns.
