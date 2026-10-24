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
if [ ! -e .env.local ] && [ ! -L .env.local ]; then
  cp .env.example .env.local
else
  printf 'Keeping existing frontend/.env.local\n'
fi
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

Run tree-model explanations only in a separate Python 3.12+ / NumPy 2+
environment. Python 3.13 matches the retained model runtime; the default PAISA
environment continues to support Python 3.10+.

```bash
python3.13 -m venv .venv-explainability
source .venv-explainability/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-explainability.txt
python -m pip install --no-deps -e .
python -m pip check
deactivate
```

## Current component entry points

| Component | Primary entry points |
|---|---|
| Providers and legacy data pipeline | `src/paisa/providers/`, `src/paisa/pipeline.py`, `src/paisa/preprocessing.py`, `src/paisa/features.py`, `scripts/run_v0_pipeline.py` |
| Experiment data and targets | `src/paisa/experiments/training_features.py`, `src/paisa/experiments/training_targets.py`, `src/paisa/splits.py` |
| Model training and evaluation | `src/paisa/experiments/trainer.py`, `src/paisa/experiments/evaluation.py`, `src/paisa/experiments/artifacts.py`, `src/paisa/experiments/registry.py`, `scripts/train_ogdc_models.py` |
| Price-only LSTM experiments | `src/paisa/experiments/lstm.py`, `src/paisa/experiments/lstm_runner.py`, `scripts/run_lstm_price_only_experiments.py`, `notebooks/PAISA_OGDC_LSTM_Experiments.ipynb`, `docs/LSTM_PRICE_ONLY_EXPERIMENTS.md`, `docs/LSTM_SEQUENCE_CONTRACT.md` |
| Tree-model SHAP evidence | `src/paisa/explainability/tree_shap.py`, `scripts/run_tree_shap_explainability.py`, `scripts/verify_explainability_run.py`, `docs/TREE_MODEL_SHAP_EXPLAINABILITY.md`, `docs/EXPLAINABILITY_DATA_CONTRACT.md`, `docs/SHAP_INTERPRETATION_POLICY.md` |
| Decision policy and portfolio | `src/paisa/trading/policy.py`, `src/paisa/trading/execution.py`, `src/paisa/trading/portfolio.py`, `src/paisa/trading/models.py` |
| Backtesting and benchmarks | `src/paisa/backtesting/engine.py`, `src/paisa/backtesting/metrics.py`, `src/paisa/backtesting/benchmarks.py`, `src/paisa/backtesting/reports.py`, `scripts/run_backtest.py` |
| End-to-end orchestration | `src/paisa/vertical_slice.py`, `scripts/run_ogdc_vertical_slice.py` |
| Legacy-cache migration | `src/paisa/legacy_cache.py`, `scripts/migrate_legacy_psx_cache.py` |
| News acquisition and evidence | `src/paisa/news/`, `scripts/run_news_pipeline.py`, `configs/news/`, `docs/NEWS_*.md` |
| Historical source and coverage audit | `src/paisa/news/coverage.py`, `src/paisa/news/providers/gdelt_doc.py`, `scripts/audit_news_price_overlap.py`, `configs/news/gdelt_ogdc_history.example.json`, `configs/news/historical_source_permissions.json`, `configs/news_coverage/`, `docs/HISTORICAL_NEWS_*.md`, `docs/NEWS_PRICE_OVERLAP_CONTRACT.md` |
| Contemporaneous GKG/PSX evidence | `src/paisa/evidence/`, `scripts/run_gdelt_gkg_evidence.py`, `scripts/import_psx_announcements.py`, `scripts/audit_contemporaneous_evidence.py`, `configs/evidence/`, `docs/GDELT_GKG_HISTORICAL_EVIDENCE.md`, `docs/PSX_ANNOUNCEMENT_*.md`, `docs/CONTEMPORANEOUS_EVIDENCE_TIME_POLICY.md` |
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
`artifacts/lstm_experiments/<run_id>/` for the immutable M-005/M-006
price-only LSTM comparison (with `final_summary.json` as the run source of
record),
`artifacts/news_ingestion/<run_id>/` plus `data/news/paisa_news.sqlite3` for
news evidence, `artifacts/news_coverage/<run_id>/` for immutable price-overlap
audits, and `artifacts/sentiment_inference/<run_id>/` for article-level sentiment
evidence. Contemporaneous source packs use
`artifacts/contemporaneous_evidence/<run_id>/`, while four-profile audits use
`artifacts/evidence_coverage/<run_id>/`. Tree-model explanations use
`artifacts/explainability/<run_id>/`, `<run_id>.zip`, and
`<run_id>.zip.sha256`. The accepted local pack is
`XAI-20261024-140320-OGDC-tree-v1`; its ZIP SHA-256 is
`7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`.
Never overwrite an existing model, experiment, portfolio, or run ID.
Do not commit datasets, downloaded transformer weights, model binaries, run
artifacts, `frontend/dist/`, local environments, or secrets.

The accepted explanation pack was produced from clean implementation commit
`2cd4145e5e01459d13d186a6f3b697e979821a9d`. Its 24-file inventory and quality
report are `PASS`; directory-plus-checksum, ZIP-plus-sidecar, and extracted
offline verification passed. The manifest records Python 3.13.7, NumPy 2.3.5,
pandas 2.3.3, scikit-learn 1.9.1, XGBoost 3.4.1, joblib 1.6.0, Matplotlib
3.11.2, and SHAP 0.54.1. All M-001--M-004 results cover the exact 184-row
shared project holdout and 19 transformed features in positive-class
probability space. Every row passed its family tolerance; all seven audit
categories were available for every model.

The model-wise maximum/mean absolute additivity errors were
`5.78484978308591e-06` / `5.781760213738571e-06` for M-001,
`4.302598055616613e-09` / `7.864803434113863e-10` for M-002,
`2.7466758000471714e-05` / `2.7464541568790406e-05` for M-003, and
`1.5920226559273942e-07` / `3.5509845686483396e-08` for M-004. The code-stage
validation recorded 1,339 backend tests passed with two skipped and six
warnings, 111 focused tests passed, 132 frontend tests passed, and passing
doctor, dependency, bytecode-compilation, frontend lint, frontend build, and
diff checks.

The ordered top five features are M-001 (`close_lag_1`,
`return_volatility_5`, `close`, `close_lag_5`, `ma_20`), M-002
(`return_volatility_5`, `close_lag_5`, `close`, `ma_20`, `rsi_14`), M-003
(`close`, `ma_20`, `close_lag_1`, `close_lag_5`, `return_volatility_5`), and
M-004 (`ma_20`, `close`, `close_lag_5`, `return_volatility_5`, `open`). The
Spearman/top-five-overlap agreement pairs are M-001/M-002
`0.8228070175438595`/4, M-001/M-003 `0.8210526315789475`/5, and M-002/M-003
`0.7543859649122807`/4; the disagreement pairs are M-001/M-004
`0.5964912280701754`/4, M-003/M-004 `0.6175438596491227`/4, and M-002/M-004
`0.7`/4. These attributions describe fitted-model behavior on a previously
exposed shared holdout. They cover one stock, one period, one feature family,
and four tree configurations; they do not establish causality, profitability,
or future results. Academic simulation; not financial advice.

The accepted local LSTM run is `LSTM-20261024-OGDC-price-only-02`. It was
executed on CPU from clean source commit
`2bd20edebac9f5a063e2e10cc0394343067d8c7f` with Python 3.13 and TensorFlow
2.21 and produced 834/182/184 train/validation/test sequences. Both variants
trained for 7 epochs and recorded best validation epoch 1. Validation preferred
`M-005` at `0.479280` balanced accuracy over `M-006` at `0.474931`. On the
post-freeze common 184-date test, `M-005` scored `0.520087` and `M-006` scored
`0.544251`; the latter exceeded all four baseline test balanced accuracies
(best: `M-002`, `0.524459`) but is not the validation-selected variant.

The `M-005` and `M-006` portfolios ended at PKR 96,359.29 (-3.64071%) and PKR
96,110.91 (-3.88909%), respectively. Neither beat buy-and-hold at PKR
106,247.89 (+6.24789%) or the best baseline model portfolio, `M-002`, at PKR
105,987.04 (+5.98704%). The run's generated checksum-inventoried artifacts are
local and ignored by Git; `final_summary.json` remains authoritative. These
results are an academic simulation, not financial advice.

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
sealed historical `PARTIAL/WARN` evidence. Its news-ingestion seal verifies
`PASS (DATABASE_AND_ARTIFACT_BOUND)` when the ignored pack and database are
present. It
contains `656` raw rows and `656` normalized canonical rows unique under the
current exact URL/content/title-source-time rules, `42` exact OGDC-linked rows,
and `93` returned source domains, with no exact-rule duplicate relationships or
rejected records. A conservative publisher document-ID diagnostic flags `17`
candidate groups (`36` rows) for later adjudication, so the count is not an
article-level uniqueness claim. Every record preserves missing `published_at`.
Raw GDELT
`seendate` evidence spans `2023-06-18T12:15:00Z` through
`2024-06-14T15:00:00Z`; conservative `available_at` is the later range
`2026-10-23T06:26:32.440746Z` through `2026-10-23T06:29:25.858147Z`. The run is
partial because it records `22` provider errors: `15` `ReadTimeout` failures, `4`
quarantined out-of-window responses, and `3` exhausted HTTP 429 responses.
It predates news schema `1.1`, so its source observation remains raw evidence
and its errors identify root windows rather than exact request attempts. New
acquisitions preserve audit-only `source_observed_at` under
`news-availability-v1`, quarantine missing/malformed historical observation
times, bind errors to exact request-tile/attempt fingerprints, enforce one
bounded DNS/request deadline, and record applied or refused `Retry-After`
decisions. Do not backfill those facts into this immutable retained run.

The historical audit implementation and interface are defined by
`docs/HISTORICAL_NEWS_COVERAGE.md` and
`docs/NEWS_PRICE_OVERLAP_CONTRACT.md`. It validates immutable news, vertical-
slice, price, training, and exact split-manifest checksums; applies the
checksum-bound permission policy and the versioned thresholds in
`configs/news_coverage/default_readiness.json`; keeps
publication overlap separate from cutoff-eligible `available_at` overlap; and
publishes a unique `COVERAGE-...` directory without overwriting prior evidence.

The existing historical schema-`1.0` / `news-price-coverage-v2` preflight is
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
is `artifacts/news_coverage/COVERAGE-20261023-120000-real0002/`. It verified
`PASS` at its producing revision, contains `11` payload artifacts and `450`
mapping rows, and retains
`NO_OVERLAP`; source permission and legacy database provenance fail, while
ingestion completeness passes. Its manifest SHA-256 is
`57d1f2e3b099318062e4f094c127de7d130953e38b76564684c7427fa3139b54`
and its inventory SHA-256 is
`1316d7fa9164bb8d091664cc102f8a5b4ad89d4760c7caca7fe62d13142d29a8`.

The GDELT run has a distinct historical schema-`1.0` /
`news-price-coverage-v2` preflight,
`COVERAGE-20261023-110000-gdeltpre1`. Source permission and database provenance
pass; ingestion completeness fails because the pack is
`PARTIAL` with `22` provider errors. Publication and availability
calendar-overlap ranges are `null`, and supported all-news/OGDC split coverage
is `0%`. Its old schema encoded unsupported energy as `0%`; that was not a
measurement. Readiness is `NO_OVERLAP`. That preflight remains historical. The
retained official GDELT artifact is
`artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/`. It verified
`PASS` at its producing revision, contains `11` payload artifacts and `1,968`
mapping rows, and retains
`NO_OVERLAP`; source permission and database provenance pass, while ingestion
completeness fails with `22` provider errors. Its manifest SHA-256 is
`9bb6385196d79befde21d422eb28a78d8aa5222b0863ea8d44b330fad3e0968a`
and its inventory SHA-256 is
`51d170ce3a0e26aa743d37b9e32335bfbfe0e1bdf9abee89e0d3683b748e0673`.
Both official artifacts were generated from clean commit
`041c557edf5b15660fd2f02d47ed7f9b8b4152a7` with `dirty_tree=false`.
They are immutable historical v2 evidence. The current schema-`1.1` /
`news-price-coverage-v3` semantic verifier requires recomputation from restored
source inputs under new IDs and intentionally does not certify the old packs.
For a current v3 pack, the CLI defaults to full source replay and reports
`SOURCE_EVIDENCE_BOUND`. The explicit `--pack-only` fallback reports
`PACK_SEMANTIC`; it checks internal consistency without reopening inputs and is
not equivalent authority. Strict replay requires any database, permission
policy, and optional energy-sector contract declared by the sealed references.

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
advances a verified database through schema `6` sentiment certification to
schema `7` audit-only source-observation provenance.
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
sealing evidence, the schema-5 sentiment tables, schema-6 certification, and
the current schema-7 source-observation contract. Before sentiment execution, the
working `data/news/paisa_news.sqlite3` also used schema version `3`. The
compatibility migration accepted only that exact known signature, validated
the affected rows, transactionally restored the later constraints, verified
integrity, and advanced through schema 4 to schema 5 for the retained run.
Current code then advances verified storage through schema `6` certification
to schema `7` audit-only source-observation provenance; validation and DDL
remain under one write lock.

The ignored, uncommitted pre-migration copy
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3` was verified as
schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check. Preserve it and the historical ingestion artifact
pack, and never describe database compatibility as retroactive artifact
validation.

### Contemporaneous evidence handoff

This path is separate from GDELT DOC ingestion. GKG uses official raw GKG 2.1
archives, accepts only `WEB=1`, assigns availability from the batch timestamp
encoded in `GKGRECORDID`, retains `V2.1DATE` as publication time, verifies
bounded archives, and never dereferences publisher URLs. The canonical
post-fix execution covered the official `2026-09-04 09:00`, `09:15`, and
`09:30` UTC 15-minute partitions inside the requested
`2025-12-16`--`2026-09-15` interval. This is only three adjacent partitions,
or 45 minutes, not the full test period. Its three captures contain `2,957`
scanned and timestamp-valid rows, `0` rejected, `2,957` unmatched, `0` matched,
and `0` duplicates. Do not describe this check as full historical coverage.

The PSX path is an offline exact-`OGDC` import only. Timed announcements retain
their official offset-aware posting time. Date-only announcements retain null
timestamps and are eligible only on the next observed trading session. The
current reviewed status is `REVIEW_REQUIRED`, no authorized input was supplied,
and its count remains `0`; do not automate the public pages or weaken the two
authorization-hash checks.

The evidence types are `gdelt_metadata`, `exchange_announcement`, and
`publisher_news`. Their separate/combined profiles are `gdelt_gkg_ogdc`,
`psx_ogdc_announcements`, `combined_ogdc_evidence`, and
`general_financial_news`. The restored publisher-news run contributes `656`
general records and `42` exact OGDC-linked combined records, but every profile
has `0 / 853` train, `0 / 182` validation, and `0 / 184` test overlap. All four
readiness results are `NO_OVERLAP`; aggregation and downstream model work stay
blocked. Readiness expresses caller-defined gate satisfaction only, not
statistical sufficiency, predictive value, or financial merit. These are
academic simulation results and are not financial advice.

#### Canonical contemporaneous evidence pair

The canonical post-fix pair was produced from clean commit
`ec6a47b6922d1858d727845ab7de37be719a1dea`. Its private pack directories are
mode `0700` and files are mode `0600`:

| Pack | Manifest SHA-256 | Payload-inventory SHA-256 | Verification basis |
|---|---|---|---|
| `EVID-20261023-175847-822c5685` | `15ae19a7d186bd1417545914e3898b23ba80e1f1adb31bb7f8e8018f01f633a4` | `636b096f9bedaed73756c7175b35676555049251c6a3b14f8ed609e960c50888` | `RAW_GDELT_GKG_ZIP_SEMANTIC_REPLAY` |
| `ECOV-20261023-175904-be50a499` | `6b35a6c6c7a9a83b20bc93099fbfaff05743e49150ea0dc2cfa3ceb1e6667250` | `baf14a5d30f60fd7c5fad2d1daa75322007b43e2257e0a1a1ecf498efc23b826` | `PORTABLE_RECORD_CALENDAR_RECOMPUTATION` |

The coverage pack has `656` mapping rows and all four profiles are
`NO_OVERLAP`. Portable record/calendar recomputation rechecks the sealed
normalized evidence and stock calendar; it remains explicitly not
source-archive-bound.

Retain the previous pair as superseded audit evidence; do not modify or delete
it:

| Superseded pack | Manifest SHA-256 | Payload-inventory SHA-256 |
|---|---|---|
| `EVID-20261023-163905-gkgfinal` | `a335bcf657ca855e52aebc2a0bcd4721e85434716aa8d6fa13a405cc0d42991e` | `1675a8902f4c0175972392d04a207f09157d034ec1b118eb4c449e2ac9d77e7e` |
| `ECOV-20261023-164000-final001` | `d4c31758d39e1a97ee7629e487fe04a801b82ce291923e47392b552f4f99335c` | `e01f33b278bc780d02e74fa5dc88abbba803f12ffc8e90b95562ffb272010e4d` |

The local attempts `EVID-20261023-163553-gkgfinal`,
`EVID-20261023-213000-gkgraw01`, `EVID-20261023-213500-gkgraw02`, and the
post-review pre-final `EVID-20261023-175415-5c1b0c48` are also
noncanonical/superseded. Preserve them in place; never present them as the
canonical pair.

Before export and again after restore, run this schema-neutral integrity check
from the repository root. It verifies each known manifest and completion
identity, the exact private flat-pack file set and modes, the canonical digest
of the embedded payload inventory, and every inventoried payload size and
SHA-256 without trying to reinterpret a legacy contract:

```bash
.venv/bin/python - <<'PY'
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

expected = {
    Path("artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685"): (
        "15ae19a7d186bd1417545914e3898b23ba80e1f1adb31bb7f8e8018f01f633a4",
        "636b096f9bedaed73756c7175b35676555049251c6a3b14f8ed609e960c50888",
        "7c7c58a7d97e278dbf7ec3411818301ad49509b8ebb48fbdda17c1c769982515",
    ),
    Path("artifacts/evidence_coverage/ECOV-20261023-175904-be50a499"): (
        "6b35a6c6c7a9a83b20bc93099fbfaff05743e49150ea0dc2cfa3ceb1e6667250",
        "baf14a5d30f60fd7c5fad2d1daa75322007b43e2257e0a1a1ecf498efc23b826",
        "d3a98c57d33c36c1d0fa1a41bedfef13b3c5fa32dcfcdae2bb25bccf9991e265",
    ),
    Path("artifacts/contemporaneous_evidence/EVID-20261023-163905-gkgfinal"): (
        "a335bcf657ca855e52aebc2a0bcd4721e85434716aa8d6fa13a405cc0d42991e",
        "1675a8902f4c0175972392d04a207f09157d034ec1b118eb4c449e2ac9d77e7e",
        "ba95e027fde05175d189e243268360487295ae01d8d0e3309084ae2feed1d742",
    ),
    Path("artifacts/evidence_coverage/ECOV-20261023-164000-final001"): (
        "d4c31758d39e1a97ee7629e487fe04a801b82ce291923e47392b552f4f99335c",
        "e01f33b278bc780d02e74fa5dc88abbba803f12ffc8e90b95562ffb272010e4d",
        "7a7bcdafe3887b712dd468abd0868c7b1836d2ea92bdc2352fd30e4488f98641",
    ),
}

def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")


for root, (manifest_sha, inventory_sha, completion_sha) in expected.items():
    root_stat = root.lstat()
    require(stat.S_ISDIR(root_stat.st_mode), f"not a real directory: {root}")
    require(stat.S_IMODE(root_stat.st_mode) == 0o700, f"wrong mode: {root}")
    observed_names = set()
    for child in root.iterdir():
        child_stat = child.lstat()
        require(stat.S_ISREG(child_stat.st_mode), f"not a regular file: {child}")
        require(stat.S_IMODE(child_stat.st_mode) == 0o600, f"wrong mode: {child}")
        observed_names.add(child.name)
    raw_manifest = (root / "run_manifest.json").read_bytes()
    require(
        hashlib.sha256(raw_manifest).hexdigest() == manifest_sha,
        f"manifest identity differs: {root}",
    )
    manifest = json.loads(raw_manifest)
    inventory = manifest["payload_artifact_inventory"]
    inventory_paths = [item["path"] for item in inventory]
    require(
        all(Path(path).name == path for path in inventory_paths),
        f"non-flat inventory path: {root}",
    )
    require(
        len(set(inventory_paths)) == len(inventory_paths),
        f"duplicate inventory path: {root}",
    )
    require(
        manifest["inventory_excludes"] == ["completion.json", "run_manifest.json"],
        f"unexpected inventory exclusions: {root}",
    )
    require(
        observed_names == set(inventory_paths) | {"completion.json", "run_manifest.json"},
        f"file set differs: {root}",
    )
    canonical_inventory = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    require(
        hashlib.sha256(canonical_inventory).hexdigest() == inventory_sha,
        f"inventory identity differs: {root}",
    )
    require(
        manifest["payload_artifact_inventory_sha256"] == inventory_sha,
        f"manifest inventory seal differs: {root}",
    )
    raw_completion = (root / "completion.json").read_bytes()
    require(
        hashlib.sha256(raw_completion).hexdigest() == completion_sha,
        f"completion identity differs: {root}",
    )
    for item in inventory:
        payload_path = root / item["path"]
        payload = payload_path.read_bytes()
        require(len(payload) == item["size_bytes"], f"size differs: {payload_path}")
        require(
            hashlib.sha256(payload).hexdigest() == item["sha256"],
            f"payload identity differs: {payload_path}",
        )
    print(f"PASS {root}")
PY
```

Use fresh immutable IDs and `--no-resume` for each new GKG acquisition. Never
reuse an existing pack ID:

```bash
.venv/bin/python scripts/run_gdelt_gkg_evidence.py \
  --config configs/evidence/gdelt_gkg_test_subset.json \
  --permission-policy configs/evidence/source_permissions.json \
  --output-dir artifacts/contemporaneous_evidence \
  --run-id "EVID-<UTC-timestamp>-<unique-suffix>" \
  --no-resume

.venv/bin/python scripts/verify_evidence_run.py \
  --run-dir "artifacts/contemporaneous_evidence/<generated-gkg-run-id>"

# This fails closed while the policy remains REVIEW_REQUIRED. Run it only after
# reviewed policy bytes bind both supplied authorization values by SHA-256.
.venv/bin/python scripts/import_psx_announcements.py \
  --input /absolute/path/to/authorized-ogdc-announcements.csv \
  --authorization-attestation-file /absolute/path/to/attestation.txt \
  --authorization-reference-file /absolute/path/to/authorization-reference.txt \
  --permission-policy configs/evidence/source_permissions.json \
  --output-dir artifacts/contemporaneous_evidence

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

The GKG verifier replays the retained raw ZIP members and their semantics. The
coverage verifier recomputes mappings from the sealed normalized records and
stock calendar, but it is not source-archive-bound; retain the checksum-bound
input references with the pack.

Preserve the canonical and retained superseded pairs together. Run the portable
four-pack integrity check above first. Create the tar only in approved storage
outside the repository, refuse to replace an existing archive, verify the four
known manifests before export, and create the checksum sidecar from inside the
transfer directory so it remains portable:

```bash
set -euo pipefail
umask 077
PAISA_EVIDENCE_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"
PAIR_ARCHIVE_NAME="PAISA-contemporaneous-evidence-20261023.tar.gz"
PAIR_ARCHIVE="$PAISA_EVIDENCE_TRANSFER_DIR/$PAIR_ARCHIVE_NAME"

if test -L "$PAISA_EVIDENCE_TRANSFER_DIR"; then
  printf 'ERROR: transfer directory cannot be a symlink: %s\n' \
    "$PAISA_EVIDENCE_TRANSFER_DIR" >&2
  exit 1
fi
mkdir -p "$PAISA_EVIDENCE_TRANSFER_DIR"
chmod 700 "$PAISA_EVIDENCE_TRANSFER_DIR"
test ! -e "$PAIR_ARCHIVE" && test ! -L "$PAIR_ARCHIVE"
test ! -e "$PAIR_ARCHIVE.sha256" && test ! -L "$PAIR_ARCHIVE.sha256"

printf '%s  %s\n' \
  '15ae19a7d186bd1417545914e3898b23ba80e1f1adb31bb7f8e8018f01f633a4' \
  'artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685/run_manifest.json' \
  '6b35a6c6c7a9a83b20bc93099fbfaff05743e49150ea0dc2cfa3ceb1e6667250' \
  'artifacts/evidence_coverage/ECOV-20261023-175904-be50a499/run_manifest.json' \
  'a335bcf657ca855e52aebc2a0bcd4721e85434716aa8d6fa13a405cc0d42991e' \
  'artifacts/contemporaneous_evidence/EVID-20261023-163905-gkgfinal/run_manifest.json' \
  'd4c31758d39e1a97ee7629e487fe04a801b82ce291923e47392b552f4f99335c' \
  'artifacts/evidence_coverage/ECOV-20261023-164000-final001/run_manifest.json' \
  | shasum -a 256 -c -

COPYFILE_DISABLE=1 tar -czf "$PAIR_ARCHIVE" \
  artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685 \
  artifacts/evidence_coverage/ECOV-20261023-175904-be50a499 \
  artifacts/contemporaneous_evidence/EVID-20261023-163905-gkgfinal \
  artifacts/evidence_coverage/ECOV-20261023-164000-final001
chmod 600 "$PAIR_ARCHIVE"

(
  cd "$PAISA_EVIDENCE_TRANSFER_DIR"
  shasum -a 256 "$PAIR_ARCHIVE_NAME" > "$PAIR_ARCHIVE_NAME.sha256"
)
chmod 600 "$PAIR_ARCHIVE.sha256"
```

Restore only into a checkout where all four exact target directories are
absent. Never merge an archive into an existing pack. Verify the transferred
archive checksum, then use the committed restore helper. The helper accepts
only the exact four checksum-bound pack manifests and their sealed inventory,
rejects every duplicate, unexpected, traversal, absolute, link, and special
member, extracts regular files into private staging, verifies the staged
canonical packs, and atomically publishes each expected root without
replacement. It never extracts an untrusted tar directly into the checkout.
Afterward, run the current semantic verifiers against the canonical pair:

Publication is an explicit sequence of four atomic no-replace renames because
portable filesystems do not offer one atomic multi-directory transaction. The
helper preflights every destination again immediately before publication. If a
concurrent process creates a later destination, the helper preserves any
earlier verified pack already committed, refuses to replace the collision, and
reports failure; inspect the reported repository state before retrying.

```bash
set -euo pipefail
umask 077
PAISA_EVIDENCE_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"
PAIR_ARCHIVE_NAME="PAISA-contemporaneous-evidence-20261023.tar.gz"
PAIR_ARCHIVE="$PAISA_EVIDENCE_TRANSFER_DIR/$PAIR_ARCHIVE_NAME"

if test -L "$PAISA_EVIDENCE_TRANSFER_DIR" \
  || ! test -d "$PAISA_EVIDENCE_TRANSFER_DIR" \
  || test -L "$PAIR_ARCHIVE" \
  || ! test -f "$PAIR_ARCHIVE" \
  || test -L "$PAIR_ARCHIVE.sha256" \
  || ! test -f "$PAIR_ARCHIVE.sha256"; then
  printf 'ERROR: transfer inputs must be regular files in a real directory\n' >&2
  exit 1
fi

(
  cd "$PAISA_EVIDENCE_TRANSFER_DIR"
  shasum -a 256 -c "$PAIR_ARCHIVE_NAME.sha256"
)

for target in \
  artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685 \
  artifacts/evidence_coverage/ECOV-20261023-175904-be50a499 \
  artifacts/contemporaneous_evidence/EVID-20261023-163905-gkgfinal \
  artifacts/evidence_coverage/ECOV-20261023-164000-final001
do
  if test -e "$target" || test -L "$target"; then
    printf 'ERROR: refusing to overwrite existing evidence pack: %s\n' "$target" >&2
    exit 1
  fi
done

.venv/bin/python scripts/restore_evidence_bundle.py \
  --archive "$PAIR_ARCHIVE" \
  --destination-root .

printf '%s  %s\n' \
  '15ae19a7d186bd1417545914e3898b23ba80e1f1adb31bb7f8e8018f01f633a4' \
  'artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685/run_manifest.json' \
  '6b35a6c6c7a9a83b20bc93099fbfaff05743e49150ea0dc2cfa3ceb1e6667250' \
  'artifacts/evidence_coverage/ECOV-20261023-175904-be50a499/run_manifest.json' \
  'a335bcf657ca855e52aebc2a0bcd4721e85434716aa8d6fa13a405cc0d42991e' \
  'artifacts/contemporaneous_evidence/EVID-20261023-163905-gkgfinal/run_manifest.json' \
  'd4c31758d39e1a97ee7629e487fe04a801b82ce291923e47392b552f4f99335c' \
  'artifacts/evidence_coverage/ECOV-20261023-164000-final001/run_manifest.json' \
  | shasum -a 256 -c -

.venv/bin/python scripts/verify_evidence_run.py \
  --run-dir artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685
.venv/bin/python scripts/verify_evidence_coverage.py \
  --run-dir artifacts/evidence_coverage/ECOV-20261023-175904-be50a499
```

After extraction, also rerun the schema-neutral four-pack integrity block above.
It is the legacy-compatible payload verification step; the two current CLIs
then add full semantic verification for the canonical pair.

The current verifiers intentionally reject the superseded pair's older sealed
contracts. Preserve and identify that pair through its known manifest and
payload-inventory hashes; do not rewrite it to make a current verifier pass.

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

COPYFILE_DISABLE=1 tar -czf "$DEMO_ARCHIVE" \
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
location containing the received files. The restore helper rejects traversal,
absolute paths, links, special files, duplicates, and every member outside the
two named roots before extracting regular files into private staging. It then
publishes each root atomically without replacement.

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
.venv/bin/python scripts/restore_artifact_bundle.py \
  --archive "$DEMO_ARCHIVE" \
  --destination-root . \
  --allow-directory artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --allow-directory artifacts/model_training/TRN-20261020-191815-624662-nogit

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

### Restore or regenerate ignored news evidence

Git contains the news code, policies, and documentation, but not
`artifacts/news_ingestion/`, `artifacts/news_coverage/`, or
`data/news/paisa_news.sqlite3`. There is no project-managed public download URL
or predetermined archive checksum. The project lead/artifact custodian is the
only documented source for the exact retained bytes. The custodian must create
the checksum sidecar from the archive actually transferred through approved
shared storage; recipients must not trust a checksum copied from prose.

From the retained checkout, export the complete news evidence set without
modifying it:

<!-- BEGIN NEWS_ARTIFACT_EXPORT -->
```bash
set -euo pipefail
PAISA_NEWS_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"
NEWS_ARCHIVE_NAME="PAISA-news-evidence-20261023.tar.gz"
NEWS_ARCHIVE="$PAISA_NEWS_TRANSFER_DIR/$NEWS_ARCHIVE_NAME"

test -d "$PAISA_NEWS_TRANSFER_DIR"
test -d artifacts/news_ingestion
test -d artifacts/news_coverage
test -f data/news/paisa_news.sqlite3
test -f artifacts/news_ingestion/NEWS-20261023-060000-gdelt-ogdc2/completion.json
test -f artifacts/news_coverage/COVERAGE-20261023-120000-real0002/completion.json
test -f artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/completion.json
if [ -e "$NEWS_ARCHIVE" ] || [ -L "$NEWS_ARCHIVE" ] || \
   [ -e "$NEWS_ARCHIVE.sha256" ] || [ -L "$NEWS_ARCHIVE.sha256" ]; then
  printf 'ERROR: refusing to overwrite news transfer package: %s\n' \
    "$NEWS_ARCHIVE" >&2
  exit 1
fi

COPYFILE_DISABLE=1 tar -czf "$NEWS_ARCHIVE" \
  artifacts/news_ingestion \
  artifacts/news_coverage \
  data/news/paisa_news.sqlite3
(
  cd "$PAISA_NEWS_TRANSFER_DIR"
  shasum -a 256 "$NEWS_ARCHIVE_NAME" > "$NEWS_ARCHIVE_NAME.sha256"
)
```
<!-- END NEWS_ARTIFACT_EXPORT -->

Transfer both archive and sidecar. Restore only into a checkout where the
generated targets are absent; never merge the archive into another working
database or artifact root. The allowlisted helper performs the same private
staging and unsafe-member rejection used for the demo pair; the explicit file
and directory roots are the complete permitted extraction surface:

<!-- BEGIN NEWS_ARTIFACT_RESTORE -->
```bash
set -euo pipefail
PAISA_NEWS_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"
NEWS_ARCHIVE_NAME="PAISA-news-evidence-20261023.tar.gz"
NEWS_ARCHIVE="$PAISA_NEWS_TRANSFER_DIR/$NEWS_ARCHIVE_NAME"

for target in \
  artifacts/news_ingestion \
  artifacts/news_coverage \
  data/news/paisa_news.sqlite3
do
  if [ -e "$target" ] || [ -L "$target" ]; then
    printf 'ERROR: refusing to overwrite news evidence path: %s\n' \
      "$target" >&2
    exit 1
  fi
done
(
  cd "$PAISA_NEWS_TRANSFER_DIR"
  shasum -a 256 -c "$NEWS_ARCHIVE_NAME.sha256"
)
.venv/bin/python scripts/restore_artifact_bundle.py \
  --archive "$NEWS_ARCHIVE" \
  --destination-root . \
  --allow-directory artifacts/news_ingestion \
  --allow-directory artifacts/news_coverage \
  --allow-file data/news/paisa_news.sqlite3

test -f artifacts/news_ingestion/NEWS-20261023-060000-gdelt-ogdc2/completion.json
test -f artifacts/news_coverage/COVERAGE-20261023-120000-real0002/completion.json
test -f artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/completion.json
test -f data/news/paisa_news.sqlite3
```
<!-- END NEWS_ARTIFACT_RESTORE -->

The two retained official coverage packs use schema `1.0` /
`news-price-coverage-v2`. They verified `PASS` only at their producing revision,
`041c557edf5b15660fd2f02d47ed7f9b8b4152a7`; current schema-`1.1` /
`news-price-coverage-v3` verification intentionally rejects them. Preserve the
restored packs for audit. To create current evidence, work on copies of the
restored inputs, restore the `VSL-...`/`TRN-...` pair above, and run the current
coverage CLI with a new `COVERAGE-...` ID. A current coverage regeneration from
an old ingestion pack cannot invent a missing `source_observed_at`. Obtaining
that first-class field requires a new permitted acquisition/import under a new
`NEWS-...` ID and, for GDELT, a newly reviewed permission-policy entry.

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
3. Preserve the certified `LSTM-20261024-OGDC-price-only-02` evidence and its
   validation-only selection decision. Do not promote `M-006` from its stronger
   post-freeze test balanced accuracy. After the sentiment coverage gate passes,
   train paired price-only and price-plus-sentiment experiments without using
   test results for selection.
4. Use verified pack `XAI-20261024-140320-OGDC-tree-v1` when adding SHAP result
   views after Waqas exposes the validated read-only API schema. Consume
   persisted evidence without recalculating SHAP in React; explain model
   behavior, not causality.
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
4. Review the certified local M-005/M-006 run against the current prediction
   and artifact contracts, and operate future runs only under new immutable
   IDs; develop the separate
   relationship/knowledge-graph component only after its contract is written.
5. Extend `src/paisa/api/schemas.py`, `src/paisa/api/artifact_repository.py`,
   and `src/paisa/api/router_v1.py` additively for news, sentiment,
   explanations, and relationships; keep GET requests read-only. For SHAP,
   validate a small portable fixture with the offline verifier and never load a
   model in a GET request.
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

# The retained coverage-v2 packs verified only at their producing revision.
# Current code requires regenerated schema-1.1 / coverage-v3 evidence under a
# fresh, unique COVERAGE-... value; never reuse an old immutable run ID. Replace
# these syntactically valid example IDs with the exact restored input IDs.
PAISA_COVERAGE_RUN_ID="COVERAGE-20261023-190000-handoff-v3"
PAISA_COVERAGE_RUN_DIR="artifacts/news_coverage/$PAISA_COVERAGE_RUN_ID"
PAISA_NEWS_RUN_DIR="artifacts/news_ingestion/NEWS-20261023-180000-handoff-source"
PAISA_VERTICAL_RUN_DIR="artifacts/vertical_slice/VSL-20261020-OGDC-main-02"
PAISA_TRAINING_RUN_DIR="artifacts/model_training/TRN-20261020-191815-624662-nogit"

# Generate the new immutable v3 pack first. This example includes the explicit
# compatibility flag required for a checksum-verified, pre-seal RSS news pack;
# omit that flag when every selected news run has a current completion seal.
python scripts/audit_news_price_overlap.py \
  --news-run-dir "$PAISA_NEWS_RUN_DIR" \
  --news-database data/news/paisa_news.sqlite3 \
  --vertical-slice-dir "$PAISA_VERTICAL_RUN_DIR" \
  --model-training-dir "$PAISA_TRAINING_RUN_DIR" \
  --source-assessment docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md \
  --source-permission-policy configs/news/historical_source_permissions.json \
  --allow-legacy-unsealed \
  --output-dir artifacts/news_coverage \
  --run-id "$PAISA_COVERAGE_RUN_ID"

# Then perform the strict source-bound replay against the exact same inputs.
python scripts/verify_news_coverage.py \
  --run-dir "$PAISA_COVERAGE_RUN_DIR" \
  --news-run-dir "$PAISA_NEWS_RUN_DIR" \
  --news-database data/news/paisa_news.sqlite3 \
  --vertical-slice-dir "$PAISA_VERTICAL_RUN_DIR" \
  --model-training-dir "$PAISA_TRAINING_RUN_DIR" \
  --source-assessment docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md \
  --source-permission-policy configs/news/historical_source_permissions.json \
  --allow-legacy-unsealed

# Only when source evidence is unavailable: this is a weaker internal semantic
# check and its PASS must not be reported as source-evidence-bound.
python scripts/verify_news_coverage.py \
  --run-dir "$PAISA_COVERAGE_RUN_DIR" \
  --pack-only

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

# Optional TensorFlow price-only comparison. The accepted run is
# LSTM-20261024-OGDC-price-only-02; the command below is for a new run ID only.
# M-005 and M-006 both use the frozen OGDC evidence; validation selects/reports
# and test is post-freeze.
# CPU and Colab GPU are supported, Apple Metal is optional, and results are not
# promised to be bit-identical across devices.
python -m pip install -r requirements-lstm.txt
PAISA_NEW_LSTM_RUN_ID="LSTM-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_lstm_price_only_experiments.py \
  --prices artifacts/model_training/TRN-20261020-191815-624662-nogit/frozen_input/OGDC_prices.csv \
  --output-dir artifacts/lstm_experiments \
  --run-id "$PAISA_NEW_LSTM_RUN_ID" \
  --baseline-training-run artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --baseline-vertical-slice artifacts/vertical_slice/VSL-20261020-OGDC-main-02

# Optional isolated tree-SHAP evidence for retained M-001 through M-004.
# SHAP 0.54.1 requires Python 3.12+ and NumPy 2+; use the dedicated
# explainability environment and never replace an existing XAI run.
source .venv-explainability/bin/activate
python scripts/run_tree_shap_explainability.py \
  --model-training-run \
  artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --comparison-run \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02 \
  --vertical-slice-run \
  artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --output-dir artifacts/explainability

# Verify the accepted immutable local evidence pack. Replace only when
# verifying the fresh run ID emitted by the producer above.
PAISA_XAI_RUN_ID="XAI-20261024-140320-OGDC-tree-v1"
python scripts/verify_explainability_run.py \
  "artifacts/explainability/$PAISA_XAI_RUN_ID"
python scripts/verify_explainability_run.py \
  "artifacts/explainability/$PAISA_XAI_RUN_ID.zip" \
  --checksum \
  "artifacts/explainability/$PAISA_XAI_RUN_ID.zip.sha256"
deactivate

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
python -m pytest tests/test_lstm_notebook.py
python -m pytest
python -m pip check
python -m compileall src scripts
cd frontend
npm ci
npm run test
npm run lint
npm run build
cd ..
git diff --check origin/main...HEAD
```

Independent validation of reviewed PR head `4d7084b` produced two historical
pre-fix matrices:

- artifact-present checkout: Python `609 passed, 1 skipped`; the only skip was
  optional TensorFlow/LSTM, and the local retained-artifact integration test
  passed;
- clean Git clone: Python `608 passed, 2 skipped`; optional TensorFlow/LSTM and
  the absent local-real-run artifact integration test skipped.

That review also reported frontend `57 passed`, passing lint/build,
`pip check`, `compileall`, and `git diff --check origin/main...HEAD`. Do not
present those counts as validation of later changes; report the exact results
from the checkout actually tested.

The hardened certified LSTM source commit
`2bd20edebac9f5a063e2e10cc0394343067d8c7f` completed with `1036 passed, 1
skipped` in the backend, `57 passed` in the frontend, and passing frontend
lint/build, `pip check`, bytecode compilation, and diff checks. The skip is the
Windows-only junction test on macOS. Those are the accepted counts for that
clean source commit; future changes require their own exact verification.

Report the exact pass/fail/skip counts and warnings, branch and commit, generated
artifact IDs, assumptions, known limitations, and any contract changes. Keep
provider-dependent network checks separate from the default offline suite.
