# PAISA v0 — Predictive AI Stock Assessment

PAISA began as a PSX-first stock-data layer and now also includes additive,
reproducible financial-news ingestion and article-level financial-sentiment
inference foundations. The price/model/backtest/API and dashboard contracts
remain separate from news acquisition and sentiment evidence.

This is **not financial advice** and does not produce buy/sell recommendations. It is an academic prototype for data engineering, ML evaluation, and dashboard integration.

## Project documentation

Use the following entry points for the current vertical slice and its evidence.
[`CURRENT_IMPLEMENTATION_STATUS.md`](docs/CURRENT_IMPLEMENTATION_STATUS.md) is
the canonical current implementation and evidence inventory;
[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) is retained only as historical
proposal and early-v0 context.

| Document | Purpose |
|---|---|
| [Current implementation status](docs/CURRENT_IMPLEMENTATION_STATUS.md) | Canonical status of implemented code, retained evidence, and incomplete scope. |
| [October 2026 progress report](docs/PROGRESS_REPORT_OCTOBER_2026.md) | Dated delivery summary, verified results, test evidence, and next milestones. |
| [Results interpretation](docs/RESULTS_INTERPRETATION.md) | Defensible interpretation of prediction and simulated portfolio results. |
| [Price-only LSTM experiments](docs/LSTM_PRICE_ONLY_EXPERIMENTS.md) | Reproducible M-005/M-006 execution, comparison, artifact, and reporting runbook. |
| [LSTM sequence contract](docs/LSTM_SEQUENCE_CONTRACT.md) | Normative 20-session target alignment, split-boundary, and train-only preprocessing rules. |
| [Architecture overview](docs/ARCHITECTURE_OVERVIEW.md) | End-to-end component, data-flow, timing, and trust-boundary overview. |
| [Known limitations](docs/KNOWN_LIMITATIONS.md) | Data, methodology, operational, and research limitations. |
| [Supervisor demo script](docs/SUPERVISOR_DEMO_SCRIPT.md) | Fail-fast preflight and timed demonstration walkthrough. |
| [Team handoff](docs/TEAM_HANDOFF.md) | Ownership, exact demo-artifact restore, regeneration, and continuation commands. |
| [Next-phase plan](docs/NEXT_PHASE_PLAN.md) | Ordered plan for news, sentiment, LSTM, explainability, and expansion. |
| [News ingestion runbook](docs/NEWS_INGESTION.md) | RSS/manual acquisition, immutable artifacts, quality policy, and reproduction commands. |
| [News data contract](docs/NEWS_DATA_CONTRACT.md) | Versioned normalized record, provider evidence, deduplication, entity-link, and SQLite semantics. |
| [News time alignment](docs/NEWS_TIME_ALIGNMENT.md) | Publication/acquisition/availability rules and the future-feature cutoff boundary. |
| [Historical news source assessment](docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md) | Dated access, permission, storage, attribution, cost, and coverage assessment for candidate archives and import sources. |
| [Historical news coverage runbook](docs/HISTORICAL_NEWS_COVERAGE.md) | Permitted acquisition/import paths and the reproducible price-overlap audit workflow. |
| [News-price overlap contract](docs/NEWS_PRICE_OVERLAP_CONTRACT.md) | Checksum validation, cutoff mapping, population/counting rules, artifacts, and configurable readiness states. |
| [GDELT GKG historical evidence](docs/GDELT_GKG_HISTORICAL_EVIDENCE.md) | Raw GKG 2.1 permission, archive, timestamp, matching, and bounded-execution contract. |
| [PSX announcement source assessment](docs/PSX_ANNOUNCEMENT_SOURCE_ASSESSMENT.md) | Reviewed automation prohibition and controlled manual-import permission status. |
| [PSX announcement data contract](docs/PSX_ANNOUNCEMENT_DATA_CONTRACT.md) | Exact-OGDC normalized fields, authorization boundary, timestamp precision, and immutable artifacts. |
| [Contemporaneous evidence time policy](docs/CONTEMPORANEOUS_EVIDENCE_TIME_POLICY.md) | GKG batch, official-event, date-only, cutoff, profile, and leakage rules. |
| [FinBERT sentiment runbook](docs/FINBERT_SENTIMENT.md) | Pinned model/license, deterministic text policy, resource controls, artifacts, and execution commands. |
| [Sentiment data contract](docs/SENTIMENT_DATA_CONTRACT.md) | Article-level probabilities, model/input identity, immutable artifacts, and SQLite semantics. |
| [Sentiment time alignment](docs/SENTIMENT_TIME_ALIGNMENT.md) | Inherited news time, decision-cutoff filtering, and future aggregation boundary. |
| [Historical project context](PROJECT_CONTEXT.md) | Original proposal and superseded PAISA v0 snapshot; not a current-status source. |

## What v0 includes

- PSX historical data ingestion through `psx-dps` provider.
- Optional full-OHLCV ingestion through `pypsx-toolkit` provider.
- Manual CSV provider for cases where data access is blocked.
- Sample provider for offline tests and API demos.
- Standard price schema: `date, ticker, open, high, low, close, volume, source`.
- Technical features: returns, lagged close, moving averages, volatility, RSI, MACD, volume change.
- Supervised label: `target_next_day_up`.
- Chronological train/validation/test split.
- Baseline RandomForest model.
- FastAPI endpoints for prices, features, dataset, metrics, and prediction.
- Pytest tests for pipeline, preprocessing, and API.
- Separate RSS/Atom, metadata-only GDELT DOC 2.0, and licensed manual-CSV
  news adapters with immutable raw evidence and explicit usage notes.
- Versioned normalized news records, deterministic deduplication, first-stage
  OGDC alias linking, SQLite persistence, and quality/coverage reports.
- A checksum-validated, immutable news-versus-price coverage audit that maps
  conservative `available_at` evidence to configurable market cutoffs and
  reports all-news, OGDC-linked, and supported sector populations separately.
- Reproducible article-level FinBERT inference with immutable model/tokenizer
  revisions, explicit class mapping, bounded batching, SQLite persistence, and
  leakage-safe availability filtering.

## Recommended first run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Offline smoke test with fake data
python scripts/run_v0_pipeline.py --provider sample --symbols OGDC HBL MCB --start 2021-01-01 --end 2023-12-31 --train

# Run tests
python -m pytest

# Start API
uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

- http://127.0.0.1:8000/health
- http://127.0.0.1:8000/stocks
- http://127.0.0.1:8000/stocks/OGDC/prices
- http://127.0.0.1:8000/models/baseline/metrics
- http://127.0.0.1:8000/predict/OGDC
- http://127.0.0.1:8000/docs

## PSX run

```bash
source .venv/bin/activate
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train
```

The `psx-dps` provider uses the public PSX Data Portal time-series endpoint:

```text
https://dps.psx.com.pk/timeseries/eod/{SYMBOL}
```

Observed response rows are shaped like:

```text
[unix_timestamp, close_price, volume, open_price]
```

So this provider gives `open`, `close`, and `volume`; `high` and `low` may remain missing. That is acceptable for v0 because the first baseline model is trained on open/close/volume-derived indicators.

## Optional full-OHLCV run

`pypsx-toolkit` advertises 10 years of PSX OHLCV data as pandas DataFrames. If
both its installation and import verification succeed, use:

```bash
source .venv/bin/activate
pip install -r requirements-provider-pypsx.txt
python -c "import pypsx_toolkit"
python scripts/run_v0_pipeline.py --provider pypsx --symbols OGDC HBL MCB --start 2021-01-01 --train
```

The optional provider is intentionally excluded from `requirements.txt` because
its upstream distribution can install successfully but fail to import on some
platforms. The default `psx-dps`, `manual-csv`, and `sample` providers do not
require it.

## Manual CSV fallback

If both providers fail, download one CSV per ticker and place files here:

```text
data/external/manual_psx/OGDC.csv
data/external/manual_psx/HBL.csv
data/external/manual_psx/MCB.csv
```

Then run:

```bash
python scripts/run_v0_pipeline.py --provider manual-csv --manual-csv-dir data/external/manual_psx --symbols OGDC HBL MCB --start 2021-01-01 --train
```

Accepted column variants include `date`, `open`, `open_price`, `close`, `close_price`, `volume`, `high`, and `low`.

## Financial-news ingestion foundation

News acquisition is an explicit, separate workflow. The ingestion command does
not run FinBERT, add sentiment to a stock dataset, retrain a model, or change
any API/dashboard result. The RSS adapter requests explicitly configured publisher feed URLs,
rejects redirects, and never fetches linked article pages. Relative entry links
are resolved against the feed URL while the unresolved value remains in raw
evidence:

```bash
python scripts/run_news_pipeline.py \
  --provider rss \
  --config configs/news/ogdc_sources.example.json \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

A permitted local dataset can instead be imported and is always labelled
`imported`, never live:

```bash
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /absolute/path/to/news.csv \
  --licensing-usage-note "Dataset licence and permitted stored fields reviewed on YYYY-MM-DD" \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

The selected automated archive path is the public GDELT DOC 2.0 API in
metadata-only mode. The example intentionally queries only OGDC. Before any new
acquisition, review and update the dated source assessment and its companion
policy, including the assessment checksum and exact run allowlist; then review
the bounded query windows. The adapter never follows publisher links or stores
publisher bodies:

```bash
python scripts/run_news_pipeline.py \
  --provider gdelt-doc \
  --config configs/news/gdelt_ogdc_history.example.json \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

GDELT `seendate` is source-observation evidence, not a publisher publication
time. New schema-`1.1` runs preserve it separately as `source_observed_at`,
which is audit-only under `news-availability-v1` and excluded from cutoff
eligibility. Normalized
`available_at` is therefore controlled by the later PAISA fetch for these
metadata-only records; an archive query made today cannot be backdated into an
earlier model cutoff.

The completed live metadata-only acquisition
`NEWS-20261023-060000-gdelt-ogdc2` is retained separately from the RSS evidence.
It is a sealed historical `PARTIAL` run with `WARN` quality. Its
news-ingestion seal verified `PASS` with `DATABASE_AND_ARTIFACT_BOUND`
authority at the producing revision; the coverage claims below are separate
historical evidence. It preserves `656` raw records and
`656` normalized canonical rows that are unique only under the current exact
URL/content/title-source-time rules, including `42` exact OGDC-linked rows
across `93` returned source domains, with no exact-rule duplicate relationships
or rejected records. This is not a claim of article-level uniqueness: the
conservative reviewed-publisher document-ID diagnostic flags `17` candidate
groups (`36` rows, or `19` rows beyond one per group) for later adjudication;
the `42` OGDC matches and zero-overlap result are unchanged. All `656` records
keep `published_at` missing. Raw GDELT `seendate`
evidence spans `2023-06-18T12:15:00Z` through `2024-06-14T15:00:00Z`, while
conservative `available_at` spans `2026-10-23T06:26:32.440746Z` through
`2026-10-23T06:29:25.858147Z`. The run remains `PARTIAL` because it records
`22` provider errors: `15` `ReadTimeout` failures, `4` quarantined out-of-window
responses, and `3` exhausted HTTP 429 responses.
That immutable run predates schema `1.1`; new acquisitions additionally bind
errors to exact request tiles/attempts and sanitized fingerprints, quarantine
missing or malformed historical observation times, share one bounded
DNS/request deadline, and record applied or policy-refused `Retry-After`
durations. Those facts are not retroactively imputed into the retained run.
The permission policy permits this exact run under the reviewed,
permission-relevant metadata-only controls. It does not grant permission to a
future run automatically.

After valid argument parsing, the command prints one structured summary and
creates a collision-refusing `artifacts/news_ingestion/<run_id>/` evidence
pack. Paths must remain under `--storage-root` unless external storage is
explicitly allowed. `PARTIAL` returns non-zero unless `--allow-partial` is
given. Completed consumers require the SQLite seal and the final
`completion.json`; an artifact manifest alone is not proof of completion.
Generated news databases, raw responses, and run outputs are ignored by Git.

Verify a retained pack offline with:

```bash
python scripts/verify_news_run.py \
  --run-dir artifacts/news_ingestion/<run_id> \
  --database data/news/paisa_news.sqlite3
```

Omit `--database` only for artifact-only offline replay; that mode verifies the
retained evidence but is not authoritative proof of committed completion. The
coverage audit reports that distinction in `database_provenance`, and a missing
database fingerprint or a selected legacy pack without a current database-bound
completion seal prevents an otherwise-ready result from becoming
`READY_FOR_AGGREGATION`.

See the
[news runbook](docs/NEWS_INGESTION.md),
[record contract](docs/NEWS_DATA_CONTRACT.md), and
[time-alignment contract](docs/NEWS_TIME_ALIGNMENT.md).

## Historical news-price coverage audit

The coverage audit consumes immutable news packs plus the retained OGDC price
and exact target-split manifests. It does not run sentiment, build a stock
feature, retrain a model, change a policy/portfolio, or update the dashboard.
Real mode rejects fixture, mock, test, generated, or synthetic inputs and
validates declared checksums before calculating coverage. The machine-readable
permission policy checksum-binds the reviewed assessment, permission-relevant
provider controls, and any exact allowlisted run identities; unresolved
decisions cannot support readiness:

**Artifact-present retained example:** the following command requires the
ignored news/database evidence and the retained `VSL-...`/`TRN-...` pair. A
fresh clone must first restore checksum-verified custodian exports using
[`docs/TEAM_HANDOFF.md`](docs/TEAM_HANDOFF.md), or regenerate every input and
the coverage output under new IDs.

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
  --readiness-config configs/news_coverage/default_readiness.json \
  --market-timezone Asia/Karachi \
  --decision-cutoff 15:30:00 \
  --real-mode \
  --allow-legacy-unsealed \
  --output-dir artifacts/news_coverage
```

The retained five-pack RSS evidence from October 2026 is later than the frozen
price series, which ends on 2026-09-15. The existing
historical schema-`1.0` / `news-price-coverage-v2` preflight
`COVERAGE-20261023-100000-preflight003` therefore reports `NO_OVERLAP`; fixture
results never replace that finding. The explicit legacy flag is required
because these five retained packs predate completion seals. It validates their
full declared checksum inventories but neither upgrades them to the current
contract nor resolves source-permission review. The current policy marks RSS
evidence `REVIEW_REQUIRED`, and the legacy packs remain non-authoritative even
when the SQLite file is fingerprinted because they lack current database-bound
completion seals. For current packs, the audit verifies every seal against one
immutable SQLite online-backup image and fingerprints those same bytes. The
preflight's checksum-bound `article_session_mappings.csv` carries the exhaustive
occurrence/population and canonical session assignments needed by a later
aggregation component. The retained official RSS artifact is
`artifacts/news_coverage/COVERAGE-20261023-120000-real0002/`. At its producing
revision it verified `PASS`, contained `11` payload artifacts and `450` mapping
rows, and retained
`NO_OVERLAP`: ingestion completeness passes, while source permission and legacy
database provenance do not. Its completion marker binds manifest SHA-256
`57d1f2e3b099318062e4f094c127de7d130953e38b76564684c7427fa3139b54`
and inventory SHA-256
`1316d7fa9164bb8d091664cc102f8a5b4ad89d4760c7caca7fe62d13142d29a8`.
See the [coverage
runbook](docs/HISTORICAL_NEWS_COVERAGE.md) and
[overlap contract](docs/NEWS_PRICE_OVERLAP_CONTRACT.md) for the exact session
mapping, population, artifact, and readiness rules.

The live GDELT pack has a separate historical schema-`1.0` /
`news-price-coverage-v2` preflight,
`COVERAGE-20261023-110000-gdeltpre1`. Source permission and database provenance
pass, but ingestion completeness fails because the selected run is `PARTIAL`
and contains `22` provider errors. Publication and
availability calendar-overlap ranges are both `null`, and the supported
all-news and OGDC-linked populations report `0%` train/validation/test
coverage. Its old schema encoded unsupported energy-sector metrics as `0%`;
that value is not a measurement. Readiness is `NO_OVERLAP`. The preflight
remains historical; the retained
official GDELT artifact is
`artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/`. At its producing
revision it verified `PASS`, contained `11` payload artifacts and `1,968`
mapping rows, and retained
`NO_OVERLAP`: source permission and database provenance pass, while ingestion
completeness fails with `22` provider errors. Its completion marker binds
manifest SHA-256
`9bb6385196d79befde21d422eb28a78d8aa5222b0863ea8d44b330fad3e0968a`
and inventory SHA-256
`51d170ce3a0e26aa743d37b9e32335bfbfe0e1bdf9abee89e0d3683b748e0673`.
Both official packs were generated from clean code commit
`041c557edf5b15660fd2f02d47ed7f9b8b4152a7` with `dirty_tree=false`.
They remain immutable historical evidence. The current coverage contract is
schema `1.1` / `news-price-coverage-v3`; its semantic verifier intentionally
does not certify these v2 packs. Restore their ignored inputs and regenerate
under new run IDs to obtain current-verifier evidence.
For a current v3 pack, `scripts/verify_news_coverage.py` is source-bound by
default and requires the referenced news, SQLite, vertical-slice, training,
assessment, and permission-policy inputs. `--pack-only` explicitly selects a
weaker internal semantic check. The modes report `SOURCE_EVIDENCE_BOUND` and
`PACK_SEMANTIC` respectively; a pack-only PASS is not source-evidence-bound.
Strict mode also requires every database, permission-policy, and optional
energy-sector link contract declared by the sealed references.
Neither the historical preflights nor the retained official results authorize
daily sentiment aggregation.

## Contemporaneous historical evidence

PAISA now has a second, separate evidence path for contemporaneous historical
sources. It uses official GDELT GKG 2.1 raw archives, not the GDELT DOC API,
and a controlled offline import boundary for official PSX OGDC announcements.
The normalized evidence types remain distinct: `gdelt_metadata`,
`exchange_announcement`, and `publisher_news`. Coverage is reported separately
for `gdelt_gkg_ogdc`, `psx_ogdc_announcements`,
`combined_ogdc_evidence`, and `general_financial_news`.

GKG records use the archive batch time from `GKGRECORDID` as `available_at`,
retain `V2.1DATE` as the source publication timestamp, accept only `WEB=1`, and
never dereference publisher URLs. Official PSX rows require the exact `OGDC`
ticker. A timed announcement keeps its official offset-aware posting time; a
date-only announcement keeps both publication and availability timestamps null
and becomes eligible only on the next observed trading session. No same-day
time is invented.

The canonical post-fix GKG execution
`EVID-20261023-175847-822c5685` processed the three official 15-minute
partitions at `2026-09-04 09:00`, `09:15`, and `09:30` UTC within the requested
`2025-12-16` through `2026-09-15` interval. This is only three adjacent
partitions, or 45 minutes, and is not the full test period. Its three raw ZIP
captures contain `2,957` scanned and timestamp-valid rows, `0` rejected,
`2,957` unmatched, `0` matched, and `0` duplicate rows. Verification passes
with basis `RAW_GDELT_GKG_ZIP_SEMANTIC_REPLAY`. This narrow subset is a
reproducible acquisition check, not a claim of full-period completeness. PSX
remains `REVIEW_REQUIRED`; no authorized announcement input was supplied, so
its retained count is `0` and public-page automation remains blocked. The
existing publisher-news input adds
`656` general records, including `42` exact OGDC-linked records in the combined
profile. Nevertheless, cutoff-eligible overlap is `0 / 853` train,
`0 / 182` validation, and `0 / 184` test for every profile. All four readiness
statuses are therefore `NO_OVERLAP`.

The paired canonical coverage pack is `ECOV-20261023-175904-be50a499`. It has
`656` mapping rows, recomputes the sealed records against the sealed stock
calendar with basis `PORTABLE_RECORD_CALENDAR_RECOMPUTATION`, and reports
`NO_OVERLAP` for all four profiles. Its readiness labels mean
caller-defined gate satisfaction only, not statistical sufficiency or
predictive value. Both canonical packs were produced from clean commit
`ec6a47b6922d1858d727845ab7de37be719a1dea`; their directories are mode `0700`
and payload files are mode `0600`. Exact manifest and payload-inventory hashes,
the retained superseded pair, and checksum-bound preservation and restoration
commands are recorded in [the team handoff](docs/TEAM_HANDOFF.md#canonical-contemporaneous-evidence-pair).

Run and verify the bounded GKG path with a fresh immutable ID. `--no-resume`
forces a new acquisition rather than reusing an earlier checkpoint:

```bash
.venv/bin/python scripts/run_gdelt_gkg_evidence.py \
  --config configs/evidence/gdelt_gkg_test_subset.json \
  --permission-policy configs/evidence/source_permissions.json \
  --output-dir artifacts/contemporaneous_evidence \
  --run-id "EVID-<UTC-timestamp>-<unique-suffix>" \
  --no-resume

.venv/bin/python scripts/verify_evidence_run.py \
  --run-dir "artifacts/contemporaneous_evidence/<generated-gkg-run-id>"
```

Resume checkpoints contain raw archive bytes and therefore use the same local
security boundary as generated evidence. The checkpoint directory must be a
real, link-free `0700` directory, and its checkpoint and cache files must be
regular `0600` files. A repository-local `--checkpoint-dir` must also be
Git-ignored and contain no tracked files. PAISA rejects older checkpoints with
public modes; secure or remove them before retrying. Verification failures are
reported as generic structured JSON and do not echo host filesystem paths.

The PSX importer is intentionally unusable under the current
`REVIEW_REQUIRED` decision. Only after the reviewed policy is updated to bind
the exact authorization attestation and authorization reference hashes may an
operator run this offline command:

```bash
.venv/bin/python scripts/import_psx_announcements.py \
  --input /absolute/path/to/authorized-ogdc-announcements.csv \
  --authorization-attestation-file /absolute/path/to/attestation.txt \
  --authorization-reference-file /absolute/path/to/authorization-reference.txt \
  --permission-policy configs/evidence/source_permissions.json \
  --output-dir artifacts/contemporaneous_evidence
```

Create and verify the immutable four-profile coverage pack after restoring the
ignored stock, split, publisher-news, and GKG artifacts:

```bash
.venv/bin/python scripts/audit_contemporaneous_evidence.py \
  --gdelt-run-dir "artifacts/contemporaneous_evidence/<generated-gkg-run-id>" \
  --publisher-news-run-dir artifacts/news_ingestion/NEWS-20261023-060000-gdelt-ogdc2 \
  --vertical-slice-dir artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --model-training-dir artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --output-dir artifacts/evidence_coverage \
  --run-id "ECOV-<UTC-timestamp>-<unique-suffix>"

.venv/bin/python scripts/verify_evidence_coverage.py \
  --run-dir "artifacts/evidence_coverage/<generated-coverage-run-id>"
```

The coverage verifier proves the sealed pack's checksums and performs portable
record/calendar recomputation; it remains explicitly not source-archive-bound.
These are academic data-engineering results for a simulation, not financial
advice, and
they do not authorize aggregation, modeling, signals, trading, portfolio, API,
or dashboard changes.

## Article-level FinBERT sentiment foundation

Sentiment inference is a separate, downstream workflow. It reads canonical
normalized articles represented by a completed news run, never fetches article
pages, and does not modify news availability time. The default is the exact
`ProsusAI/finbert` model and tokenizer revision
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. Its original labels are mapped
explicitly from `0=positive`, `1=negative`, and `2=neutral` into PAISA's
canonical `negative`, `neutral`, `positive` output order.

The pinned weights/model are recorded as CC BY-NC-SA 4.0 and are restricted in
PAISA to non-commercial academic use. Install the separate runtime and inspect
the bounded CLI before downloading the model:

```bash
python -m pip install -r requirements-sentiment.txt
python scripts/run_sentiment_inference.py --help
```

An example CPU run against the accepted news evidence is:

```bash
python scripts/run_sentiment_inference.py \
  --source-run-id NEWS-20261022-131242-5ddffd2d \
  --database data/news/paisa_news.sqlite3 \
  --output-dir artifacts/sentiment_inference \
  --model-cache data/sentiment_model_cache \
  --device cpu \
  --batch-size 8 \
  --max-input-characters 4000 \
  --max-token-length 512 \
  --max-articles 500
```

Every `SENT-...` directory is collision-refusing and contains the pinned model
and input manifests, article probabilities, structured errors, deterministic
audit sample, quality report, checksummed run manifest, and final summary. The
pipeline stores input hashes and source references instead of another copy of
article bodies. CPU is the default; an explicitly available CUDA or Apple Metal
device is recorded when selected. A model/download failure is non-zero and is
never replaced with a generic classifier.

The loader verifies reviewed SHA-256 hashes for the configuration, safetensors
weights, tokenizer configuration, vocabulary, and special-token map before and
after model loading. SQLite schema `6` introduced those hashes, ordered input
and generated/reused/error associations, post-truncation token counts, and both
artifact recovery seals before `PASS` is allowed. Current schema `7` preserves
those guarantees and adds nullable audit-only `source_observed_at`. An
interrupted `RUNNING` run is never reused globally. After confirming it is
stale, retry its exact `--run-id` with `--discard-stale-running`; finalized runs
cannot be discarded.

The current stage emits article-level sentiment only. It does not aggregate by
day, join prices, retrain a stock model, emit a trading signal, or alter the API
or dashboard. A later consumer must first enforce `available_at <=
decision_cutoff`. See the [FinBERT runbook](docs/FINBERT_SENTIMENT.md),
[sentiment record contract](docs/SENTIMENT_DATA_CONTRACT.md), and
[sentiment time contract](docs/SENTIMENT_TIME_ALIGNMENT.md).

## Versioned OGDC model experiments

The experiment trainer reuses the daily PAISA feature pipeline without
replacing the legacy v0 baseline. It trains four variants on one shared,
chronological, leakage-safe split:

- `M-001`: Random Forest with the existing baseline configuration.
- `M-002`: conservative Random Forest.
- `M-003`: more flexible Random Forest.
- `M-004`: XGBoost using the same features, target dates, and split dates.

The optional price-only recurrent comparison adds two frozen 20-session,
64-unit LSTM variants without changing the four CPU models:

- `M-005`: baseline LSTM with dropout `0.20`.
- `M-006`: the same sequence length and hidden width with stronger dropout and
  recurrent regularization.

On the canonical frozen split, the sequence contract produces 834 training,
182 validation, and 184 test examples. Validation results select and report the
preferred variant; the test set is evaluated only after that choice is frozen.

Install the default dependencies and run the unified command:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/train_ogdc_models.py
```

The defaults are OGDC, daily data, the `psx-dps` provider, a requested start
date of `2021-01-01`, random seed `42`, and immutable output below
`artifacts/model_training/`. Use `--help` to see the explicit overrides.
Generated model binaries, frozen inputs, and ZIP archives are ignored by Git.

Each model's `predictions.csv` follows the architecture prediction-record
contract and can be passed directly to the backtesting CLI. The experiment
configuration must reference the same `model_id` and the desired test-date
window. The normal multi-symbol `data/processed/psx_prices.csv` is accepted;
the backtester selects the configured symbol before session validation:

```bash
python scripts/run_backtest.py \
  --predictions artifacts/model_training/<run_id>/models/<model_id>/predictions.csv \
  --prices data/processed/psx_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

Open [`notebooks/PAISA_OGDC_Model_Training.ipynb`](notebooks/PAISA_OGDC_Model_Training.ipynb)
for the CPU/Colab workflow. The thin optional
[`notebooks/PAISA_OGDC_LSTM_Experiments.ipynb`](notebooks/PAISA_OGDC_LSTM_Experiments.ipynb)
installs TensorFlow separately and invokes the reusable experiment runner; it
does not duplicate model, split, or backtest logic. The same workflow can be
run directly:

```bash
python -m pip install -r requirements-lstm.txt
python scripts/run_lstm_price_only_experiments.py \
  --prices artifacts/model_training/<baseline_training_run>/frozen_input/OGDC_prices.csv \
  --output-dir artifacts/lstm_experiments \
  --run-id <new_lstm_run_id> \
  --baseline-training-run artifacts/model_training/<baseline_training_run> \
  --baseline-vertical-slice artifacts/vertical_slice/<baseline_vertical_slice>
```

CPU and Colab GPU execution are supported; Apple Metal is optional. Fixed
seeds and recorded runtime metadata improve reproducibility, but bit-identical
results are not promised across devices. Generated `final_summary.json` is the
authoritative source for the completed run ID and real metrics. See the
[LSTM runbook](docs/LSTM_PRICE_ONLY_EXPERIMENTS.md), [sequence
contract](docs/LSTM_SEQUENCE_CONTRACT.md), and [training-stage
documentation](docs/model_training_stage.md).

## Versioned OGDC backtesting

The backtesting engine consumes a standardized prediction CSV, daily OGDC
prices, and an immutable experiment configuration. It does not train models and
does not change the existing v0 pipeline.

```bash
python scripts/run_backtest.py \
  --predictions path/to/predictions.csv \
  --prices path/to/OGDC_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

`--runs-dir` is optional and defaults to `runs/`. Each successful run creates a
new `runs/<experiment_id>/` directory containing the normalized configuration,
predictions, signals, trades, daily portfolio snapshots, benchmark ledgers,
metrics, an equity-curve PNG, and a drawdown PNG. Existing experiment IDs and
portfolio IDs are rejected rather than overwritten.

See [the backtesting runbook](docs/BACKTESTING.md) for the exact CSV schemas,
configuration, execution timing, accounting rules, metric formulas, and output
tree. All backtests are academic simulations and are not financial advice.

## OGDC end-to-end vertical slice

The vertical-slice runner joins the versioned components into one reproducible
real-data workflow: fetch and clean daily OGDC data, train `M-001` through
`M-004`, pass each model's untouched test predictions directly to an isolated
portfolio backtest, calculate cash-only and buy-and-hold benchmarks, and publish
one comparison report.

Install the normal, machine-learning, backtesting, and test dependencies from
the aggregate requirements file, then run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pip check
python -m pytest
python scripts/run_ogdc_vertical_slice.py
```

The live command uses `psx-dps`. It never substitutes the sample provider or
synthetic data. If the live request fails, a cached price file is accepted only
with one of two explicit provenance records:

- A current schema-1.0 `paisa.pipeline.run_v0_pipeline` record produced with
  that exact CSV. This existing path remains accepted unchanged.
- A complete, checksum-bound verified-legacy bundle produced by
  `paisa.legacy_cache.migrate_legacy_psx_cache`. A pre-schema CSV and its old
  metadata are not accepted directly.

Both paths bind the price bytes, real provider, daily frequency, row count, and
OGDC coverage and require positive raw-acquisition evidence with no OGDC
provider error. Source values in the CSV must agree with the provider. A CSV
alone, stale or incomplete metadata, relabelled fixture data, or mixed/blank
provenance is rejected.

### Migrate the verified pre-schema PSX cache

The migration command performs no live acquisition and never changes the
legacy files. Run it once with the exact source paths and the repository's
standard verified-cache destination:

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

By default, the command looks recursively below the inferred legacy
`data/raw/` directory for exact `<SYMBOL>_psx_dps_raw.csv` filenames. If one
exists for every requested symbol, it snapshots each source file, strictly
validates every raw row, rebuilds prices with the current cleaner, and requires
one-to-one row accounting plus an exact match with the legacy processed data.
Missing, partial, invalid, duplicate, or processed-only raw evidence is
rejected; it can never produce a verified-real bundle.

The immutable destination contains exactly the migrated `psx_prices.csv`, new
schema-1.0 `pipeline_metadata.json`, byte-preserved
`legacy_pipeline_metadata.json`, and `migration_manifest.json`. The manifest
records original and migrated SHA-256 values, coverage checks, timestamps,
migration strategy, evidence strength, and limitations. The producer remains
the distinct legacy migrator; it is never represented as a new pipeline or
live-provider run. An existing output directory is not overwritten.

After a successful migration, run the controlled real-cache workflow with:

```bash
python scripts/run_ogdc_vertical_slice.py \
  --cached-prices \
  "data/external/verified_legacy_psx_cache/psx_prices.csv" \
  --run-id \
  "VSL-20261020-OGDC-real-cache-02"
```

When the live request returns HTTP 404, that error remains in the run evidence.
The fallback is labelled `acquisition_mode="verified_legacy_psx_cache"`,
`live_fetch_succeeded=false`, `original_provider="psx_dps"`, and
`legacy_metadata_preserved=true`; it is not described as a successful live
fetch. Migration rejects missing evidence, invalid prices or volumes,
duplicate ticker/date rows, wrong or mixed providers, sample/synthetic/fixture/
mock/test sources, coverage mismatches, non-positive raw counts, and provider
errors.

The selected CSV and provenance metadata are first copied to an
invocation-private working snapshot before either validation or training, so
later changes to the source cache cannot change the run. On success, that exact
CSV and its `pipeline_metadata.json` are retained under the published bundle's
`input_snapshot/` directory as checksum-inventoried input evidence. For a
verified-legacy run, the snapshot also retains `legacy_pipeline_metadata.json`
and `migration_manifest.json`. A model-training run is published only after
all four backtests and persisted-evidence checks pass. The runner rechecks the
trainer's frozen price file and all four prediction-file checksums after
backtesting and immediately before publication. The complete training run also
has a sorted, exact-file SHA-256 inventory covering its frozen inputs, split
and data manifests, reports, model manifests, configurations, metrics,
predictions, and binaries; its source version fields and inventory are checked
before and after backtesting. Model IDs and prediction paths are derived from
the validated run manifest, so a caller summary cannot redirect backtesting to
external files. A callback therefore cannot change model evidence mid-run; a
handled late failure rolls back artifacts owned by that invocation.
Successful outputs are immutable under
`artifacts/vertical_slice/<vertical_slice_run_id>/`, and an existing run ID is
never overwritten. Each completed bundle retains the four full backtesting
trees under `experiments/<experiment_id>/`; aggregate metrics and curves are
rebuilt and reconciled from those persisted ledgers rather than trusted only
from in-memory results.

The common experiment allocates at most 50% of equity to OGDC. That same cap
also applies to buy-and-hold, so this is a 50%-allocated buy-and-hold benchmark
with the balance left in cash, not a fully invested benchmark. The model target
measures next-session close versus feature-date close, while simulated trades
execute at the next observed session's open and earn returns from that point
onward. Prediction quality and simulated portfolio profitability are therefore
reported separately.

The verified-legacy path establishes a strict internal evidence chain, not
independent market-data authenticity. Legacy raw captures have no provider
signature or retained HTTP-response capture, and a SHA-256 digest detects byte
changes but is not a digital signature. See the runbook for the full limits.

See [the OGDC vertical-slice runbook](docs/OGDC_VERTICAL_SLICE.md) for the exact
assumptions, fallback rules, artifact tree, report semantics, and limitations.
All outputs are academic simulations and are not financial advice.

## API endpoints

### Existing v0 API

The original v0 routes and field names remain unchanged. The `/health`
`data_dir` and `model_dir` values are intentionally safe logical labels rather
than host filesystem paths; readiness booleans, row counts, and available-stock
reporting are unchanged. In particular, `/stocks` and
`/stocks/{ticker}/prices` still depend on the generated
`data/processed/psx_prices.csv` compatibility file and return `404` when it is
absent.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Check whether generated data/model artifacts exist. |
| `GET /pipeline/status` | View latest pipeline metadata. |
| `GET /stocks` | List tickers available in processed data. |
| `GET /stocks/{ticker}/prices` | Return cleaned historical price rows. |
| `GET /stocks/{ticker}/features` | Return feature-engineered rows. |
| `GET /stocks/{ticker}/dataset` | Return ML-ready feature + label rows. |
| `GET /models/baseline/metrics` | Return RandomForest split metrics. |
| `GET /predict/{ticker}` | Return next-trading-day up/down prediction for latest complete row. |

`GET /health` returns `status`, the logical labels `data_dir: "legacy-v0-data"`
and `model_dir: "legacy-v0-models"`, and the `prices_ready`, `dataset_ready`,
and `model_ready` booleans. When the legacy price CSV exists, it also returns
`price_rows` and sorted `stocks`. It never returns repository, artifact, model,
or user-directory paths.

### Read-only v1 experiment-results API

The additive `/api/v1` API reads completed, immutable vertical-slice artifacts.
It does not train a model, rerun a backtest, recalculate business metrics, write
artifacts, perform real trading, or deserialize/serve a fitted model binary.
A completed vertical slice owns its historical-data snapshot; dashboards should
therefore request prices through that run rather than through a mutable processed
data path.

From the repository root, install and start it with the local real run as the
explicit default when that uncommitted artifact exists:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02

PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

`PAISA_VERTICAL_SLICE_ROOT` and `PAISA_MODEL_TRAINING_ROOT` default to the
values shown. `PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID` is optional, but
`GET /api/v1/leaderboard` returns `409` if it is unset instead of guessing a
run. Run-specific URLs do not require a default. Configured relative paths are
resolved from the repository root.

| Endpoint | Purpose |
|---|---|
| `GET /api/v1/vertical-slices` | List completed, checksum-valid vertical-slice runs with pagination. |
| `GET /api/v1/vertical-slices/{run_id}` | Return run identity, dates, models, experiments, and shared assumptions. |
| `GET /api/v1/vertical-slices/{run_id}/summary` | Return the persisted final summary. |
| `GET /api/v1/vertical-slices/{run_id}/stocks/{ticker}/prices` | Return historical daily prices from that run's checksum-verified input snapshot. |
| `GET /api/v1/vertical-slices/{run_id}/leaderboard` | Return the combined model/portfolio leaderboard. |
| `GET /api/v1/leaderboard` | Return the explicitly configured default run's leaderboard. |
| `GET /api/v1/vertical-slices/{run_id}/models` | Return validation and test model metrics. |
| `GET /api/v1/vertical-slices/{run_id}/portfolios` | Return strategy, cash-only, and buy-and-hold metrics. |
| `GET /api/v1/vertical-slices/{run_id}/equity-curves` | Return paginated daily equity rows. |
| `GET /api/v1/vertical-slices/{run_id}/drawdowns` | Return paginated daily drawdown rows. |
| `GET /api/v1/vertical-slices/{run_id}/source` | Return filtered input/source provenance. |
| `GET /api/v1/vertical-slices/{run_id}/experiments` | List experiment configurations and result summaries. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}` | Return one persisted configuration and result. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/trades` | Return paginated strategy trades. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/snapshots` | Return paginated strategy portfolio snapshots. |
| `GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/benchmarks` | Return paginated cash-only and buy-and-hold snapshots and buy-and-hold trades. |

The run listing and large-table endpoints accept `limit` (default `200`,
maximum `1000`) and `offset` (default `0`) and return `total`, `limit`,
`offset`, and deterministically ordered `rows`. For example:

```bash
curl -s http://127.0.0.1:8000/api/v1/vertical-slices
curl -s http://127.0.0.1:8000/api/v1/leaderboard
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/stocks/OGDC/prices?limit=5&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/equity-curves?limit=25&offset=0"
curl -s "http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/experiments/VSL-20261020-OGDC-main-02-m-001/trades?limit=25&offset=0"
```

All v1 responses use explicit Pydantic schemas and ISO dates. PKR amounts are
fixed-scale strings such as `"105987.04"`, preserving decimal meaning.
Undefined metrics are JSON `null`; `NaN` and infinity are never emitted. Every
applicable run response retains warnings and the `"Academic simulation; not
financial advice."` disclaimer. The API explicitly identifies `actual_class`
as evaluation-only and never uses it for a signal, execution, or accounting.

The historical-price endpoint has its own `limit` range (`1` through `5000`,
default `500`) and accepts `offset`, inclusive `start_date`, and inclusive
`end_date`. Date filters apply before pagination, `total` is the filtered count
before pagination, rows are chronological, prices are decimal strings, and
unavailable `high`/`low` values are JSON `null`. The React dashboard must use
this run-scoped route, not legacy `/stocks/{ticker}/prices`.

Run and experiment IDs accept only safe identifier characters. The repository
rejects traversal, symlink escape, non-`PASS` or incomplete runs, missing or
extra inventoried files, checksum mismatches, and invalid referenced training
runs. Clients choose registered IDs, never filenames or filesystem paths, and
v1 responses do not reveal absolute local paths. The price route selects only
the manifest-bound `input_snapshot` CSV; it never reads
`data/processed/psx_prices.csv` and never loads a model binary. See
[`docs/api_design.md`](docs/api_design.md) for response examples and error
semantics.

## React results dashboard

The dashboard under [`frontend/`](frontend/) presents the completed OGDC
vertical slice with React, TypeScript, and Plotly. It reads the PAISA API; it
does not train models, execute backtests or trades, recompute financial
metrics, read artifact files directly, or load fitted model binaries. Runtime
API failures are shown as errors and are never silently replaced with fixture
data. All results remain academic simulations and are not financial advice.

Node.js 22 LTS is recommended (22.12 or newer; Node.js 24 is also supported).
The current frontend has been exercised with Node.js 22.20.0. Install and
configure it from the repository root:

```bash
cd frontend
npm ci
if [ ! -e .env.local ] && [ ! -L .env.local ]; then
  cp .env.example .env.local
else
  printf 'Keeping existing frontend/.env.local\n'
fi
```

The example environment file configures the browser client with:

```text
VITE_PAISA_API_BASE_URL=http://127.0.0.1:8000
```

Do not put filesystem paths in this variable; it must be the HTTP(S) base URL
of the PAISA API. Start the backend from the repository root in one terminal:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

The configured immutable vertical-slice and model-training artifact roots must
already contain that completed run. In a second terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`.

The Historical OGDC view uses the currently selected run and requests
`GET /api/v1/vertical-slices/{selected_run_id}/stocks/OGDC/prices`. It retrieves
all available rows through documented pages of at most `5000`; the backend
serves the selected run's immutable, checksum-verified input snapshot rather
than `data/processed/psx_prices.csv`. API failures remain visible and the
dashboard does not substitute fixture, sample, synthetic, or fake prices.

Frontend quality and production-build commands are:

```bash
cd frontend
npm run test
npm run build
npm run lint
```

The production build is written to the ignored `frontend/dist/` directory.
See [`frontend/README.md`](frontend/README.md) for the complete integrated-demo
workflow and frontend behavior.

## Folder structure

```text
PAISA/
├── data/
│   ├── raw/          # generated raw provider outputs
│   ├── processed/    # generated prices/features/dataset/metadata
│   ├── external/     # manually downloaded CSVs
│   ├── news/         # generated local news SQLite database
│   └── metadata/     # stock universe
├── configs/news/     # versioned news-provider examples
├── configs/news_coverage/ # versioned engineering/readiness thresholds
├── docs/
├── frontend/         # React + TypeScript read-only results dashboard
├── models/           # generated model artifacts
├── scripts/
├── src/paisa/
│   ├── api/
│   ├── models/
│   ├── news/         # isolated news providers, contracts, storage, and pipeline
│   ├── providers/
│   ├── features.py
│   ├── pipeline.py
│   └── preprocessing.py
└── tests/
```

## Definition of Done for v0

v0 is considered successful when:

1. `python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train` completes.
2. `data/raw/psx/` contains raw CSVs for each requested ticker.
3. `data/processed/psx_prices.csv` exists and has standardized columns.
4. `data/processed/psx_features.csv` exists and contains engineered indicators.
5. `data/processed/ml_dataset.csv` exists and contains `target_next_day_up`.
6. `models/baseline_random_forest.joblib` exists.
7. `models/baseline_metrics.json` contains accuracy, precision, recall, and F1 for chronological train/validation/test splits.
8. `pytest` passes.
9. `GET /health` returns `prices_ready=true`, `dataset_ready=true`, and `model_ready=true`.
10. `GET /predict/OGDC` returns a JSON prediction with direction, confidence, top factors, and the academic disclaimer.

## Current roadmap

The price-only OGDC vertical slice, dashboard, additive news ingestion,
deterministic OGDC linking, article-level FinBERT inference, historical archive
adapter, reproducible coverage audit, and price-only M-005/M-006 LSTM experiment
workflow are implemented. The LSTM runbook deliberately does not claim a real
run ID or metrics before execution; use its generated `final_summary.json` as
the source of record. Current retained news has no price-period overlap, and
the permitted archive cannot retroactively prove historical cutoff
availability. A licensed overlapping corpus with defensible timing remains the
gate before daily aggregation or controlled price-plus-sentiment experiments.
SHAP and relationship graphs also remain future work. See
[`docs/NEXT_PHASE_PLAN.md`](docs/NEXT_PHASE_PLAN.md) for the dependency order.
