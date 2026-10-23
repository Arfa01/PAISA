# Historical news coverage and overlap audit

PAISA's historical-news workflow measures whether retained financial-news
evidence overlaps the immutable OGDC daily-price and model-split evidence. It
does not aggregate sentiment, train a price-plus-sentiment model, change a
decision policy or portfolio, or claim that news improves prediction. All
outputs are academic data-engineering evidence and are not financial advice.

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
preserves `656` raw records and `656` normalized unique canonicals, including
`42` exact checksum-bound OGDC-linked canonicals across `93` returned source
domains. There are no duplicates and no rejected records.

All `656` canonicals preserve missing `published_at`; GDELT `seendate` is not
relabeled as publisher publication time. The raw source-observation evidence
spans `2023-06-18T12:15:00Z` through `2024-06-14T15:00:00Z`, while conservative
`available_at` spans `2026-10-23T06:26:32.440746Z` through
`2026-10-23T06:29:25.858147Z`. The pack is `PARTIAL` because it records `22`
provider errors: `15` `ReadTimeout` failures, `4` out-of-window responses quarantined
rather than accepted, and `3` exhausted HTTP 429 responses. A valid seal proves
the retained pack's integrity; it does not erase incomplete acquisition.

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

The retained live GDELT pack has a separate planned final audit identity. If the
target directory is still absent, run:

```bash
python scripts/audit_news_price_overlap.py \
  --news-run-dir artifacts/news_ingestion/NEWS-20261023-060000-gdelt-ogdc2 \
  --vertical-slice-dir artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --model-training-dir artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --source-assessment docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md \
  --source-permission-policy configs/news/historical_source_permissions.json \
  --output-dir artifacts/news_coverage \
  --run-id COVERAGE-20261023-130000-gdelt0001 \
  --decision-cutoff 15:30:00 \
  --market-timezone Asia/Karachi \
  --real-mode \
  --readiness-config configs/news_coverage/default_readiness.json
```

The intended immutable directory is
`artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/`. It must not
overwrite or be conflated with the planned RSS final at
`artifacts/news_coverage/COVERAGE-20261023-120000-real0002/`.

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
not permit PAISA to store.

Verify the complete published pack without provider or database access:

```bash
python scripts/verify_news_coverage.py \
  --run-dir artifacts/news_coverage/<run_id>
```

The same check is available as
`paisa.news.coverage.verify_coverage_run(run_directory)`. It rejects changed,
missing, extra, or unsafe files and cross-file identity/configuration drift.
The seal is an integrity/completion check, not a digital signature or producer
authentication mechanism.

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
`UNSUPPORTED` rather than guessed from broad keywords.

The final readiness status is one of:

- `NO_OVERLAP`;
- `INSUFFICIENT_OVERLAP`;
- `PARTIALLY_COVERED`; or
- `READY_FOR_AGGREGATION`.

The thresholds are explicit configuration for engineering and data-coverage
gating. They are not proof of sample size, sentiment quality, predictive value,
or statistical sufficiency. A small amount of overlap must not automatically
produce `READY_FOR_AGGREGATION`.

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
the official final. The planned official RSS audit
`COVERAGE-20261023-120000-real0002` must be retained separately if published,
even if a later permitted historical dataset produces publication overlap.

The `news-price-coverage-v2` preflight
`COVERAGE-20261023-110000-gdeltpre1` for
`NEWS-20261023-060000-gdelt-ogdc2` is also `NO_OVERLAP`, for a different evidence
profile. Source permission and database
provenance pass, but ingestion completeness fails because the selected pack is
`PARTIAL` and records `22` provider errors. With all `published_at` values
missing and every conservative `available_at` after the price period, the
publication and availability calendar-overlap ranges are both `null`; all
all-news, OGDC-linked, and unsupported energy split/population coverage values
are `0%`. This preflight is descriptive evidence, not the intended final
`COVERAGE-20261023-130000-gdelt0001` artifact, and it does not unblock
aggregation.

## Explicitly out of scope

This workflow does not implement daily sentiment aggregation, stock-feature
generation, model training, model comparison, portfolio changes, knowledge
graphs, SHAP, reinforcement learning, or dashboard presentation. It must not be
used to claim that sentiment predicts returns.
