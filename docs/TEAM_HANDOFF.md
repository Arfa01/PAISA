# PAISA Team Handoff

This handoff follows the approved proposal responsibilities. All current and
future predictions, signals, trades, and returns are academic simulations and
are not financial advice.

## Approved ownership

### Arfa

- Stock-market data collection and integration.
- Market/news feature engineering, dataset preparation, and label generation.
- Model training, controlled performance evaluation, and comparative reports.
- Web-interface design, development, and visualization.
- Documentation and reporting.

### Waqas

- Financial-news collection and integration.
- NLP, company/entity extraction and linking.
- Knowledge-graph and relationship modelling.
- Machine-learning model development.
- Backend API design, development, and integration.
- Documentation and reporting.

### Shared

System tests, integration review, final improvements, supervisor evidence, and
documentation are shared. The ML boundary is deliberately collaborative: Waqas
develops the news/NLP and new model components; Arfa owns dataset integration,
training, evaluation, and result presentation; both review temporal leakage and
contracts.

## Checkout and environment

Work on a task branch, never directly on `main`:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
PAISA_TASK_BRANCH="feature/task-summary"
git switch -c "$PAISA_TASK_BRANCH"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

cd frontend
npm ci
cp .env.example .env.local
cd ..
```

The bare `python` command is available after activating `.venv`; on the current
machine it is not available outside that environment.

Install the large optional transformer runtime only for sentiment work. The
pinned FinBERT weights/model are recorded as CC BY-NC-SA 4.0 and PAISA limits
this configuration to non-commercial academic use:

```bash
python -m pip install -r requirements-sentiment.txt
```

## Current component entry points

| Component | Primary entry points |
|---|---|
| Providers and legacy data pipeline | `src/paisa/providers/`, `src/paisa/pipeline.py`, `src/paisa/preprocessing.py`, `src/paisa/features.py`, `scripts/run_v0_pipeline.py` |
| Experiment data and targets | `src/paisa/experiments/training_features.py`, `src/paisa/experiments/training_targets.py`, `src/paisa/splits.py` |
| Model training and evaluation | `src/paisa/experiments/trainer.py`, `src/paisa/experiments/evaluation.py`, `src/paisa/experiments/artifacts.py`, `src/paisa/experiments/registry.py`, `scripts/train_ogdc_models.py` |
| Optional LSTM | `src/paisa/experiments/lstm.py`, `notebooks/PAISA_OGDC_LSTM_Training.ipynb` |
| Decision policy and portfolio | `src/paisa/trading/policy.py`, `src/paisa/trading/execution.py`, `src/paisa/trading/portfolio.py`, `src/paisa/trading/models.py` |
| Backtesting and benchmarks | `src/paisa/backtesting/engine.py`, `src/paisa/backtesting/metrics.py`, `src/paisa/backtesting/benchmarks.py`, `src/paisa/backtesting/reports.py`, `scripts/run_backtest.py` |
| End-to-end orchestration | `src/paisa/vertical_slice.py`, `scripts/run_ogdc_vertical_slice.py` |
| Legacy-cache migration | `src/paisa/legacy_cache.py`, `scripts/migrate_legacy_psx_cache.py` |
| News acquisition and evidence | `src/paisa/news/`, `scripts/run_news_pipeline.py`, `configs/news/`, `docs/NEWS_*.md` |
| Historical source and coverage audit | `src/paisa/news/coverage.py`, `src/paisa/news/providers/gdelt_doc.py`, `scripts/audit_news_price_overlap.py`, `configs/news/gdelt_ogdc_history.example.json`, `configs/news/historical_source_permissions.json`, `configs/news_coverage/`, `docs/HISTORICAL_NEWS_*.md`, `docs/NEWS_PRICE_OVERLAP_CONTRACT.md` |
| Article-level sentiment | `src/paisa/sentiment/`, `scripts/run_sentiment_inference.py`, `docs/FINBERT_SENTIMENT.md`, `docs/SENTIMENT_*.md` |
| FastAPI | `src/paisa/api/main.py`, `src/paisa/api/router_v1.py`, `src/paisa/api/schemas.py`, `src/paisa/api/artifact_repository.py` |
| React dashboard | `frontend/src/App.tsx`, `frontend/src/components/`, `frontend/src/api/client.ts`, `frontend/src/api/schemas.ts` |
| Tests | `tests/`, `frontend/src/**/*.test.ts`, `frontend/src/**/*.test.tsx` |

New news, sentiment, entity, and relationship modules should be additive under
`src/paisa/` and should expose explicit artifact contracts before the API or UI
depends on them. Do not place NLP logic in API handlers or React components.

## Retained artifacts

The supervisor run is generated local content and must not be committed:

```text
artifacts/vertical_slice/VSL-20261020-OGDC-main-02/
artifacts/model_training/TRN-20261020-191815-624662-nogit/
```

Important evidence inside the vertical-slice directory:

```text
manifest.json
final_summary.json
model_metrics.csv
portfolio_metrics.csv
combined_leaderboard.csv
daily_equity_curves.csv
model_training_run_reference.json
input_snapshot/
experiment_configs/
experiments/<experiment_id>/
```

Other generated roots are `runs/<experiment_id>/` for standalone backtests,
`artifacts/lstm/<model_id>/` for optional LSTM registration,
`artifacts/news_ingestion/<run_id>/` plus `data/news/paisa_news.sqlite3` for
news evidence, `artifacts/news_coverage/<run_id>/` for immutable price-overlap
audits, and `artifacts/sentiment_inference/<run_id>/` for article-level sentiment
evidence. Never overwrite an existing model, experiment, portfolio, or run ID.
Do not commit datasets, downloaded transformer weights, model binaries, run
artifacts, `frontend/dist/`, local environments, or secrets.

The accepted local RSS run `NEWS-20261022-131242-5ddffd2d` is final
non-fixture evidence for this foundation: one source, `30` raw and `30`
normalized occurrences, `29` cross-run duplicates, `1` OGDC-linked
occurrence, publication range `2026-10-19T18:56:40Z` through
`2026-10-22T13:07:31Z`, `0` missing publication times, and `PASS` quality.
Its response acquisition timestamp is later than run start and the response
checksum was verified. The manifest records finalization state `COMMITTED` and
SQLite schema version `3`.
The duplicate count reflects canonical evidence from retained earlier
acquisitions, not discarded input; all acquisitions are preserved. Generated
evidence remains outside Git.

### Historical source and coverage status

The permission review is
`docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md`, with executable decisions in
`configs/news/historical_source_permissions.json`. It selects GDELT DOC 2.0
only for bounded metadata and source-URL retrieval: retain the GDELT-returned
title, URL, observation/source metadata, request evidence, and checksums, but
never follow publisher links or store publisher bodies. Cite GDELT and re-check
its current terms and limits before every future real acquisition. The current
decision allowlists only `NEWS-20261023-060000-gdelt-ogdc2` and binds its
permission-relevant metadata-only controls; a new run requires another review
and explicit policy update. Publisher rights remain separate. The licensed
manual-CSV path is also supported, but each real
import must record its provider, acquisition and licence evidence, field-level
storage permissions, academic-use basis, attribution, and retention/
redistribution limits. An import is never described as live retrieval.

The completed live metadata-only run `NEWS-20261023-060000-gdelt-ogdc2` is
sealed `PARTIAL/WARN` and verifies `PASS (DATABASE_AND_ARTIFACT_BOUND)`. It
contains `656` raw and `656` normalized unique canonical records, `42` exact
OGDC-linked canonicals, and `93` returned source domains, with no duplicates or
rejected records. Every record preserves missing `published_at`. Raw GDELT
`seendate` evidence spans `2023-06-18T12:15:00Z` through
`2024-06-14T15:00:00Z`; conservative `available_at` is the later range
`2026-10-23T06:26:32.440746Z` through `2026-10-23T06:29:25.858147Z`. The run is
partial because it records `22` provider errors: `15` `ReadTimeout` failures, `4`
quarantined out-of-window responses, and `3` exhausted HTTP 429 responses.

The historical audit implementation and interface are defined by
`docs/HISTORICAL_NEWS_COVERAGE.md` and
`docs/NEWS_PRICE_OVERLAP_CONTRACT.md`. It validates immutable news, vertical-
slice, price, training, and exact split-manifest checksums; applies the
checksum-bound permission policy and the versioned thresholds in
`configs/news_coverage/default_readiness.json`; keeps
publication overlap separate from cutoff-eligible `available_at` overlap; and
publishes a unique `COVERAGE-...` directory without overwriting prior evidence.

The existing `news-price-coverage-v2` preflight is
`artifacts/news_coverage/COVERAGE-20261023-100000-preflight003/`. Its inputs are
the five retained legacy `PASS` ingestion packs
(`NEWS-20261022-115232-03d0fc4c`,
`NEWS-20261022-120012-31d811b1`, `NEWS-20261022-120133-43a51981`,
`NEWS-20261022-130023-470cb7a7`, and
`NEWS-20261022-131242-5ddffd2d`). Together they contain `150` occurrences,
`31` represented canonical articles, and `119` duplicate occurrences. One
canonical is deterministically OGDC-linked and is represented by `5`
occurrences. The only source is Dawn Business. Publication evidence spans
2026-10-19 through 2026-10-22; availability evidence is on 2026-10-22. No
deterministic sector link exists, so energy-sector coverage is `UNSUPPORTED`
rather than inferred from general news.

All five selected manifests have zero provider errors and zero rejected
records, so the reported `ingestion_completeness` prerequisite passes. Source
permission is a separate hard prerequisite and remains unresolved for Dawn;
the machine-readable policy records each RSS run as `REVIEW_REQUIRED`. The
SQLite database is copied once into an immutable online-backup image used for
every seal check and fingerprint, but all five packs are legacy unsealed, so
`database_provenance.passed` is false and their run IDs are reported in
`unsealed_run_ids`. None of these prerequisites overrides the more fundamental
overlap status below.
OGDC linkage is reconciled against each pack's checksum-bound
`article_entity_links.csv`, and `ogdc_linked_articles.csv` retains the exact
JSON link evidence rather than only a generic link label. The exhaustive
`article_session_mappings.csv` records every occurrence/population pair and its
population-specific canonical assignment for the later aggregation stage.

The compared OGDC artifact contains `1,239` observed daily sessions from
2021-09-16 through 2026-09-15. Exact split-manifest denominators are `853`
train, `182` validation, and `184` test sessions. The audit found zero
publication-date overlap and zero cutoff-eligible availability overlap; every
split has zero covered sessions and the readiness result is `NO_OVERLAP`.
Daily sentiment aggregation and every price-plus-sentiment experiment remain
blocked until a permitted corpus has overlapping defensible availability and
passes the configured engineering/data-coverage gates. These findings are
academic data-engineering results, not financial advice, and even a later
`READY_FOR_AGGREGATION` result would not prove statistical or predictive value.

The RSS preflight predates the current coverage completion seal and is
descriptive rather than the official final. The retained official RSS artifact
is `artifacts/news_coverage/COVERAGE-20261023-120000-real0002/`. It verifies
`PASS`, contains `11` payload artifacts and `450` mapping rows, and retains
`NO_OVERLAP`; source permission and legacy database provenance fail, while
ingestion completeness passes. Its manifest SHA-256 is
`57d1f2e3b099318062e4f094c127de7d130953e38b76564684c7427fa3139b54`
and its inventory SHA-256 is
`1316d7fa9164bb8d091664cc102f8a5b4ad89d4760c7caca7fe62d13142d29a8`.

The GDELT run has a distinct `news-price-coverage-v2` preflight,
`COVERAGE-20261023-110000-gdeltpre1`. Source permission and database provenance
pass; ingestion completeness fails because the pack is
`PARTIAL` with `22` provider errors. Publication and availability
calendar-overlap ranges are `null`, every split/population coverage value is
`0%`, and readiness is `NO_OVERLAP`. That preflight remains historical. The
retained official GDELT artifact is
`artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/`. It verifies
`PASS`, contains `11` payload artifacts and `1,968` mapping rows, and retains
`NO_OVERLAP`; source permission and database provenance pass, while ingestion
completeness fails with `22` provider errors. Its manifest SHA-256 is
`9bb6385196d79befde21d422eb28a78d8aa5222b0863ea8d44b330fad3e0968a`
and its inventory SHA-256 is
`51d170ce3a0e26aa743d37b9e32335bfbfe0e1bdf9abee89e0d3683b748e0673`.
Both official artifacts were generated from clean commit
`041c557edf5b15660fd2f02d47ed7f9b8b4152a7` with `dirty_tree=false`.

The article-level sentiment implementation pins the model and tokenizer to
`ProsusAI/finbert@0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` and explicitly
maps `0=positive`, `1=negative`, and `2=neutral` into PAISA's canonical
negative/neutral/positive output order. Its weights/model are recorded as CC
BY-NC-SA 4.0 and are limited here to non-commercial academic work. It owns
separate immutable `SENT-...` runs and SQLite tables; it does not change news
evidence, stock models, policy, portfolios, API, or dashboard.

The retained real execution is `SENT-20261022-172211-real0002`, sourced from
`NEWS-20261022-131242-5ddffd2d`. It completed with `PASS` on CPU in
`7.116756208240986` seconds against a preserved schema-3 database copy, which
migrated to schema version `5` for that retained execution. Current code
advances a verified database to schema `6` for sentiment evidence
certification.
It requested and scored all `30` unique canonical articles, with `0` skipped,
`0` failed, and `any_articles_failed=false`; one scored article was OGDC-linked.
The label counts were `negative=11`, `neutral=9`, and `positive=10`; `17`
inputs were truncated, none used title only, and all `30` carried `en-us`
language evidence. Canonical availability spans `2026-10-22T11:52:32Z` to
`2026-10-22T13:12:43.088883Z`.

The run used model and tokenizer revision
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` under Python `3.13.7`, PyTorch
`2.10.0`, Transformers `5.20.2`, Tokenizers `0.23.3`, Hugging Face Hub
`1.32.0`, and Safetensors `0.8.0`. Its generated, uncommitted artifacts are at
`artifacts/sentiment_inference/SENT-20261022-172211-real0002`.

Sentiment outputs were generated, but classification accuracy was not evaluated in this run.
The short single-publisher snapshot is insufficient for stock-model integration.

At ingestion completion, the accepted manifest and artifact pack recorded
schema version `3`; those artifacts remain historical and predate the schema-4
sealing evidence, the schema-5 sentiment tables, and the current schema-6
certification contract. Before sentiment execution, the
working `data/news/paisa_news.sqlite3` also used schema version `3`. The
compatibility migration accepted only that exact known signature, validated
the affected rows, transactionally restored the later constraints, verified
integrity, and advanced through schema 4 to schema 5 for the retained run.
Current code then advances verified storage to schema `6`; validation and DDL
remain under one write lock.

The ignored, uncommitted pre-migration copy
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3` was verified as
schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check. Preserve it and the historical ingestion artifact
pack, and never describe database compatibility as retroactive artifact
validation.

### Restore the exact supervisor demo pair

The authoritative source for the exact historical demo is the project
lead/artifact custodian's retained demo checkout. Git does not contain either
generated root. The custodian must export both roots together and transfer the
archive plus its checksum sidecar through approved shared storage; transferring
only the vertical-slice directory leaves its referenced training evidence
unavailable.

From the retained checkout, create the transfer package without modifying the
source artifacts. Set the staging directory to approved storage outside the
repository so the generated archive cannot be mistaken for a Git-tracked
project file:

<!-- BEGIN DEMO_ARTIFACT_EXPORT -->
```bash
set -euo pipefail
PAISA_DEMO_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"
DEMO_ARCHIVE_NAME="PAISA-demo-VSL-20261020-OGDC-main-02.tar.gz"
DEMO_ARCHIVE="$PAISA_DEMO_TRANSFER_DIR/$DEMO_ARCHIVE_NAME"

test -d "$PAISA_DEMO_TRANSFER_DIR"
if [ -e "$DEMO_ARCHIVE" ] || [ -L "$DEMO_ARCHIVE" ] || \
   [ -e "$DEMO_ARCHIVE.sha256" ] || [ -L "$DEMO_ARCHIVE.sha256" ]; then
  printf 'ERROR: refusing to overwrite an existing transfer package: %s\n' \
    "$DEMO_ARCHIVE" >&2
  exit 1
fi

test -f artifacts/vertical_slice/VSL-20261020-OGDC-main-02/manifest.json
test -f artifacts/vertical_slice/VSL-20261020-OGDC-main-02/final_summary.json
test -f artifacts/model_training/TRN-20261020-191815-624662-nogit/run_manifest.json

tar -czf "$DEMO_ARCHIVE" \
  artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  artifacts/model_training/TRN-20261020-191815-624662-nogit
(
  cd "$PAISA_DEMO_TRANSFER_DIR"
  shasum -a 256 "$DEMO_ARCHIVE_NAME" > "$DEMO_ARCHIVE_NAME.sha256"
)
```
<!-- END DEMO_ARTIFACT_EXPORT -->

Transfer both `PAISA-demo-VSL-20261020-OGDC-main-02.tar.gz` and
`PAISA-demo-VSL-20261020-OGDC-main-02.tar.gz.sha256`. From the receiving
repository root, refuse to overwrite any existing run, verify the archive, and
then restore it. Point `PAISA_DEMO_TRANSFER_DIR` at the approved storage
location containing the received files.

<!-- BEGIN DEMO_ARTIFACT_RESTORE -->
```bash
set -euo pipefail
PAISA_DEMO_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"
DEMO_ARCHIVE_NAME="PAISA-demo-VSL-20261020-OGDC-main-02.tar.gz"
DEMO_ARCHIVE="$PAISA_DEMO_TRANSFER_DIR/$DEMO_ARCHIVE_NAME"

require_absent_restore_target() {
  if [ -e "$1" ] || [ -L "$1" ]; then
    printf 'ERROR: refusing to overwrite existing demo artifact path: %s\n' \
      "$1" >&2
    return 1
  fi
}

require_absent_restore_target artifacts/vertical_slice/VSL-20261020-OGDC-main-02
require_absent_restore_target artifacts/model_training/TRN-20261020-191815-624662-nogit
(
  cd "$PAISA_DEMO_TRANSFER_DIR"
  shasum -a 256 -c "$DEMO_ARCHIVE_NAME.sha256"
)
tar -xzf "$DEMO_ARCHIVE" -C .

test -f artifacts/vertical_slice/VSL-20261020-OGDC-main-02/manifest.json
test -f artifacts/vertical_slice/VSL-20261020-OGDC-main-02/final_summary.json
test -f artifacts/model_training/TRN-20261020-191815-624662-nogit/run_manifest.json
```
<!-- END DEMO_ARTIFACT_RESTORE -->

With the Python environment installed, validate the restored pair through the
same artifact API used by the demonstration:

```bash
source .venv/bin/activate
python -m pytest -q \
  tests/test_experiment_results_api.py::test_local_real_run_all_endpoints_when_available
```

This test must pass rather than skip. A checksum failure, missing file, or skip
means the exact demo pair is not ready.

### Regenerate a replacement run with new IDs

Regeneration is a separate recovery path and cannot reproduce the exact
historical demo IDs, source provenance, or reported metrics. Use it only to
produce a newly reviewed replacement run. Never assign the historical run IDs
to regenerated output.

Start with either a complete verified-cache directory supplied through
approved shared storage, or the original processed CSV, original pipeline
metadata, and complete raw captures for every symbol. To build a new verified
cache from the original evidence:

```bash
set -euo pipefail
source .venv/bin/activate

PAISA_LEGACY_REPOSITORY="/absolute/path/to/legacy/paisa-v0"
PAISA_VERIFIED_CACHE="data/external/verified_legacy_psx_cache-team-handoff-01"

python scripts/migrate_legacy_psx_cache.py \
  --legacy-prices "$PAISA_LEGACY_REPOSITORY/data/processed/psx_prices.csv" \
  --legacy-metadata "$PAISA_LEGACY_REPOSITORY/data/processed/pipeline_metadata.json" \
  --legacy-raw-dir "$PAISA_LEGACY_REPOSITORY/data/raw" \
  --output-dir "$PAISA_VERIFIED_CACHE"
```

A transferred verified-cache directory must contain `psx_prices.csv`,
`pipeline_metadata.json`, `legacy_pipeline_metadata.json`, and
`migration_manifest.json`. Then publish a new vertical slice and its new
training run under unused IDs:

```bash
set -euo pipefail
source .venv/bin/activate

PAISA_VERIFIED_CACHE="data/external/verified_legacy_psx_cache-team-handoff-01"
PAISA_NEW_VERTICAL_RUN_ID="VSL-20261022-OGDC-replacement-01"

python scripts/run_ogdc_vertical_slice.py \
  --cached-prices "$PAISA_VERIFIED_CACHE/psx_prices.csv" \
  --output-dir artifacts/vertical_slice \
  --model-output-dir artifacts/model_training \
  --run-id "$PAISA_NEW_VERTICAL_RUN_ID"
```

The runner attempts live PSX-DPS first and uses the verified cache only when
that attempt fails. Record the new training run ID printed in the result,
validate the new pair, review all metrics and limitations, and update the local
`PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID` used for the replacement demonstration.

## Run and inspect the current demo

Terminal 1, from the repository root:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd frontend
npm run dev
```

Open:

- Swagger/OpenAPI: `http://127.0.0.1:8000/docs`
- Dashboard: `http://127.0.0.1:5173`
- Default-run leaderboard: `http://127.0.0.1:8000/api/v1/leaderboard`
- Run summary: `http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/summary`
- Source provenance: `http://127.0.0.1:8000/api/v1/vertical-slices/VSL-20261020-OGDC-main-02/source`

The versioned API surface is:

```text
GET /api/v1/vertical-slices
GET /api/v1/vertical-slices/{run_id}
GET /api/v1/vertical-slices/{run_id}/summary
GET /api/v1/vertical-slices/{run_id}/stocks/{ticker}/prices
GET /api/v1/vertical-slices/{run_id}/leaderboard
GET /api/v1/leaderboard
GET /api/v1/vertical-slices/{run_id}/models
GET /api/v1/vertical-slices/{run_id}/portfolios
GET /api/v1/vertical-slices/{run_id}/equity-curves
GET /api/v1/vertical-slices/{run_id}/drawdowns
GET /api/v1/vertical-slices/{run_id}/source
GET /api/v1/vertical-slices/{run_id}/experiments
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/trades
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/snapshots
GET /api/v1/vertical-slices/{run_id}/experiments/{experiment_id}/benchmarks
```

The compatibility API remains `/health`, `/pipeline/status`, `/stocks`, the
stock price/feature/dataset routes, `/models/baseline/metrics`, and
`/predict/{ticker}`. Do not change their schemas while extending `/api/v1`.

## Task handoff: Arfa

1. Keep the price/news feature join blocked while the retained coverage result
   is `NO_OVERLAP`. After a permitted overlapping corpus passes the configured
   engineering gate, declare decision cutoffs before adding a column to
   `src/paisa/experiments/training_features.py`; include an article only when
   `available_at <= decision_cutoff`.
2. Keep the current price-only feature set as the control; prepare paired
   price-only and price-plus-sentiment configurations with identical splits.
3. Train and evaluate the paired Random Forest/XGBoost/LSTM experiments without
   using test results for selection.
4. Add SHAP result views only after Waqas exposes a validated artifact/API
   schema; explain model behavior, not causality.
5. Extend dashboard components and client schemas without recalculating backend
   metrics in React.
6. Lead additional-stock data quality, evaluation, screenshots, and supervisor
   reporting.

## Task handoff: Waqas

1. Operate and extend the source-attributed news-ingestion foundation with the
   source assessment, metadata-only GDELT boundary, licensed manual imports,
   offline fixtures, separate provider/network checks, and immutable coverage
   audits. Do not treat publisher metadata access as permission to store bodies.
2. Expand the current OGDC aliases into a versioned ticker/company registry and
   reviewable entity-linking output.
3. Review and operate the pinned FinBERT article-inference foundation; inspect
   the retained real-run artifacts and audit sample before implementing
   leakage-safe daily aggregation with the
   `available_at <= decision_cutoff` rule.
4. Develop the LSTM and relationship/knowledge-graph components behind the
   current prediction and artifact contracts.
5. Extend `src/paisa/api/schemas.py`, `src/paisa/api/artifact_repository.py`,
   and `src/paisa/api/router_v1.py` additively for news, sentiment,
   explanations, and relationships; keep GET requests read-only.
6. Provide contract fixtures and API tests for each new result before dashboard
   integration.

## Commands to continue experiments

```bash
# Legacy/offline pipeline smoke
python scripts/run_v0_pipeline.py --provider sample --symbols OGDC HBL MCB --start 2021-01-01 --end 2023-12-31 --train

# Review the news CLI and source policy, then use a new immutable run ID.
python scripts/run_news_pipeline.py --help
python scripts/run_news_pipeline.py \
  --provider rss \
  --config configs/news/ogdc_sources.example.json \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3

# Historical metadata only. Re-check the source assessment and current GDELT
# terms first. This adapter does not fetch publisher bodies or bypass limits.
python scripts/run_news_pipeline.py \
  --provider gdelt-doc \
  --config configs/news/gdelt_ogdc_history.example.json \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3

# A manual historical CSV is permitted only with documented provenance and
# field-level licence/storage rights. It remains an import, never a live fetch.
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /absolute/path/to/licensed_historical_news.csv \
  --source-name reviewed-dataset-name \
  --licensing-usage-note "Licence, permitted fields, academic use, retention, and attribution reviewed on YYYY-MM-DD" \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3

# Verify the two retained official audits; never rerun with these immutable IDs.
python scripts/verify_news_coverage.py \
  --run-dir artifacts/news_coverage/COVERAGE-20261023-120000-real0002
python scripts/verify_news_coverage.py \
  --run-dir artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001

# For an independent reproduction, reuse the documented inputs and options but
# choose a fresh, unique COVERAGE-... value for --run-id.

# Pinned article-level FinBERT inference. Review the license and CLI first,
# use a copy of legacy retained storage during migration testing, and never
# reuse an existing SENT run ID or artifact directory.
python -m pip install -r requirements-sentiment.txt
python scripts/run_sentiment_inference.py --help
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

# Four versioned price-only models from the retained, standardized snapshot
PAISA_NEW_TRAINING_RUN_ID="TRN-20261022-team-handoff-01"
python scripts/train_ogdc_models.py \
  --prices artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/psx_prices.csv \
  --run-id "$PAISA_NEW_TRAINING_RUN_ID"

# Full OGDC vertical slice with a new immutable ID. The command still attempts
# live PSX-DPS first, then validates this complete checksum-bound cache bundle.
PAISA_NEW_VERTICAL_RUN_ID="VSL-20261022-OGDC-team-handoff-01"
python scripts/run_ogdc_vertical_slice.py \
  --cached-prices artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/psx_prices.csv \
  --run-id "$PAISA_NEW_VERTICAL_RUN_ID"

# Inspect supported overrides before any real run
python scripts/run_ogdc_vertical_slice.py --help
python scripts/run_backtest.py --help
```

The quoted branch and run IDs above are executable examples; replace them with
unused, task-specific values. Do not rerun with an existing ID. The retained
supervisor run must remain unchanged.

## Required verification before handoff

From the repository root with `.venv` active:

```bash
python -m pytest
python -m pip check
cd frontend
npm ci
npm run test
npm run lint
npm run build
cd ..
git diff --check
```

Latest exact validation: Python `609 passed, 1 skipped`; frontend `57 passed`,
with lint and production build passing; `pip check`, `compileall`, and
`git diff --check` also pass. The single Python skip is the documented
retained-artifact integration check unavailable in a clean checkout.

Report the exact pass/fail/skip counts and warnings, branch and commit, generated
artifact IDs, assumptions, known limitations, and any contract changes. Keep
provider-dependent network checks separate from the default offline suite.
