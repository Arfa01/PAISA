# PAISA reproduction commands

This runbook collects the supported commands for continuing PAISA from a clean
checkout. Run them from the repository root unless a block explicitly changes
directory. Predictions, signals, trades, portfolios, and returns produced by
these commands are academic simulations and are not financial advice.

## Execution profiles and safety boundaries

There are two distinct profiles:

- **Offline validation** installs only `requirements.txt`, uses deterministic
  fixtures and mocks, does not call a live provider, does not download model
  weights, and does not require any private generated artifact.
- **Local research reproduction** may use network providers, ignored SQLite
  data, ignored immutable artifacts, and optional model runtimes. Review source
  permission and licensing before acquisition, and never commit generated data,
  model binaries, caches, databases, credentials, or artifact directories.

The Python package supports Python 3.10 or newer. The certified LSTM run used
Python 3.13.7, and TensorFlow 2.21 is supported here only on CPython 3.10 through
3.13. The optional `shap==0.54.1` explainability runtime requires Python 3.12+
and NumPy 2+. Its certified direct versions are pinned in
`constraints-explainability.txt`, and dedicated Python 3.13.7 CI exercises
real Tree SHAP against small Random Forest and XGBoost estimators. The frontend
supports Node `^22.12.0`, `^24.0.0`, or `>=26.0.0`; Node
22.20.0 is the version recorded in the current dashboard runbook.

All generated run destinations are immutable. Every producer below requires a
new immutable ID. Replace every
`REPLACE-WITH-A-NEW-UNUSED-ID` value before running a producer. A collision must
fail; do not delete, overwrite, reseal, or relabel an earlier run.

## Default Python setup

The default environment contains the backend, baseline/boosted model stack,
backtesting, and test dependencies. It intentionally excludes TensorFlow,
FinBERT, and the optional `pypsx-toolkit` provider. The development-only
PyYAML dependency supports structural GitHub Actions policy tests; it is not a
runtime dependency of the PAISA API or research pipelines.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

Run the read-only environment diagnostic after setup. It reports required
commands and dependencies, optional local evidence, and exact next commands; it
does not install, download, create, or change data or artifacts.

```bash
python scripts/doctor.py
```

The optional dependency sets are deliberately separate:

```bash
# Only for article-level FinBERT inference; downloads are not part of CI.
python -m pip install -r requirements-sentiment.txt

# Only in a CPython 3.10--3.13 LSTM environment or Colab runtime.
python -m pip install -r requirements-lstm.txt

# Only in a separate CPython 3.12+ / NumPy 2+ explainability environment.
python -m pip install \
  -c constraints-explainability.txt \
  -r requirements.txt \
  -r requirements-explainability.txt

# Optional full-OHLCV provider; not required for the official psx-dps path.
python -m pip install -r requirements-provider-pypsx.txt
python -c "import pypsx_toolkit"
```

For the cleanest separation, create `.venv-lstm` or a Colab runtime for
TensorFlow and `.venv-explainability` for SHAP instead of adding either optional
stack to the default development environment. A matching explainability setup
is:

```bash
python3.13 -m venv .venv-explainability
source .venv-explainability/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -c constraints-explainability.txt \
  -r requirements.txt \
  -r requirements-explainability.txt
python -m pip install --no-deps -e .
python -m pip check
deactivate
```

## Offline validation and CI-equivalent checks

These commands do not invoke the OGDC runner, news providers, sentiment
inference, or LSTM training. The default tests inject fixtures or fake provider
sessions. In a fresh clone, the optional TensorFlow smoke test and retained-real
artifact integration test may skip; report the counts actually observed.

```bash
source .venv/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m pip check
python -m compileall -q src scripts tests
python -m pytest
git diff --check
git diff --check origin/main...HEAD
```

Frontend dependencies are restored from the committed lockfile. The production
build is generated under ignored `frontend/dist/`.

```bash
cd frontend
npm ci
npm run test
npm run lint
npm run build
cd ..
git diff --check
git diff --check origin/main...HEAD
git status --short
```

CI must use only the default Python dependency set and the npm lockfile. It must
not install `requirements-lstm.txt` or `requirements-sentiment.txt`, invoke live
providers, restore private artifacts, or fetch real model weights.

## Start the API and dashboard

The v1 results API requires a complete, checksum-valid vertical-slice run and
its referenced model-training run. These generated roots are private,
Git-ignored local artifacts; restore both together from the project artifact
custodian or create a new pair before starting the demo.

Terminal 1, from the repository root:

```bash
source .venv/bin/activate
export PAISA_VERTICAL_SLICE_ROOT=artifacts/vertical_slice
export PAISA_MODEL_TRAINING_ROOT=artifacts/model_training
export PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID=VSL-20261020-OGDC-main-02
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

The retained ID above works only when the ignored
`artifacts/vertical_slice/VSL-20261020-OGDC-main-02/` and referenced
`artifacts/model_training/TRN-20261020-191815-624662-nogit/` directories are
present. An explicit `/api/v1/vertical-slices/{run_id}/...` request does not
require the default-run variable.

Terminal 2:

```bash
cd frontend
npm ci
if [ ! -e .env.local ] && [ ! -L .env.local ]; then
  cp .env.example .env.local
else
  printf 'Keeping existing frontend/.env.local\n'
fi
npm run dev
```

Open `http://127.0.0.1:5173`. The dashboard reads API results and never trains a
model, reads an artifact directly, or substitutes fixture prices after an API
failure.

## Read-only artifact verification

### Price, training, and LSTM artifact inventories

The following validators check checksums and schemas without deserializing a
fitted model. They require the ignored artifact directories to be present.

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path

from paisa.experiments.artifacts import validate_training_run_artifacts
from paisa.experiments.lstm_runner import validate_vertical_slice_artifacts

vertical = Path("artifacts/vertical_slice/VSL-20261020-OGDC-main-02")
training = Path("artifacts/model_training/TRN-20261020-191815-624662-nogit")
lstm = Path("artifacts/lstm_experiments/LSTM-20261024-OGDC-price-only-02")

vertical_result = validate_vertical_slice_artifacts(vertical)
training_result = validate_training_run_artifacts(training)
lstm_result = validate_training_run_artifacts(lstm)

print(f"PASS vertical slice: {vertical_result.directory.name}")
print(f"PASS model training: {training_result.run_id}")
print(f"PASS LSTM experiment: {lstm_result.run_id}")
PY
```

### News, contemporaneous evidence, and coverage artifacts

Database-bound news verification is authoritative only when the exact SQLite
evidence for that run is present. Omitting `--database` performs the weaker
artifact-only replay.

```bash
PAISA_NEWS_RUN_ID="NEWS-20261023-060000-gdelt-ogdc2"
python scripts/verify_news_run.py \
  --run-dir "artifacts/news_ingestion/$PAISA_NEWS_RUN_ID" \
  --database data/news/paisa_news.sqlite3

python scripts/verify_evidence_run.py \
  --run-dir artifacts/contemporaneous_evidence/EVID-20261023-175847-822c5685

python scripts/verify_evidence_coverage.py \
  --run-dir artifacts/evidence_coverage/ECOV-20261023-175904-be50a499
```

These verifiers are offline and read-only. The contemporaneous coverage
verifier reconciles its sealed records and stock calendar but is not a replay
of the original remote GDELT archive.

## Official OGDC vertical slice

The official vertical slice is daily OGDC data through the real `psx-dps`
provider. It trains M-001 through M-004, creates independent portfolios, and
publishes an immutable comparison. It never silently substitutes sample or
synthetic data.

Review the CLI, select a fresh ID, then run:

```bash
source .venv/bin/activate
python scripts/run_ogdc_vertical_slice.py --help

PAISA_NEW_VERTICAL_RUN_ID="VSL-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_ogdc_vertical_slice.py \
  --start 2021-01-01 \
  --output-dir artifacts/vertical_slice \
  --model-output-dir artifacts/model_training \
  --run-id "$PAISA_NEW_VERTICAL_RUN_ID"
```

When live PSX-DPS is unavailable, only a checksum-bound, real-provider cache
with its required sibling provenance is acceptable. The retained vertical
slice contains such a frozen input snapshot; fixtures and bare CSV files are
rejected.

```bash
PAISA_NEW_VERTICAL_RUN_ID="VSL-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_ogdc_vertical_slice.py \
  --cached-prices artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/psx_prices.csv \
  --output-dir artifacts/vertical_slice \
  --model-output-dir artifacts/model_training \
  --run-id "$PAISA_NEW_VERTICAL_RUN_ID"
```

The second command still attempts the live provider first. If it falls back,
the provider error and verified-cache acquisition mode remain explicit in the
new evidence.

## News ingestion

News ingestion is separate from sentiment, stock models, signals, portfolios,
the API, and the dashboard. Existing destinations are never overwritten, and a
provider failure never falls back to fixtures. Review
`docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md`, the source terms, allowed stored
fields, attribution, and retention before any network or manual acquisition.

### Configured RSS or Atom feeds

The RSS adapter fetches only configured feeds and never follows article links.

```bash
PAISA_NEW_NEWS_RUN_ID="NEWS-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_news_pipeline.py \
  --provider rss \
  --config configs/news/ogdc_sources.example.json \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3 \
  --run-id "$PAISA_NEW_NEWS_RUN_ID"
```

### Licensed manual CSV import

Use only a dataset whose licence and field-level storage rights have been
reviewed. Record a concrete permission note; an absolute path is recommended.

```bash
PAISA_NEW_NEWS_RUN_ID="NEWS-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /absolute/path/to/licensed_historical_news.csv \
  --source-name reviewed-dataset-name \
  --licensing-usage-note "Licence, permitted fields, academic use, retention, and attribution reviewed on YYYY-MM-DD" \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3 \
  --run-id "$PAISA_NEW_NEWS_RUN_ID"
```

### Bounded GDELT DOC metadata discovery

The example is metadata-only and OGDC-bounded. It does not fetch publisher
pages. Before a new acquisition, review the dated assessment, query windows,
and permission policy. The existing policy allowlist is tied to an exact prior
run and is not a provider-wide permission grant; a new run needs a new reviewed
decision before it can support a coverage-readiness claim.

```bash
PAISA_NEW_NEWS_RUN_ID="NEWS-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_news_pipeline.py \
  --provider gdelt-doc \
  --config configs/news/gdelt_ogdc_history.example.json \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3 \
  --run-id "$PAISA_NEW_NEWS_RUN_ID"
```

After any completed ingestion, verify the exact new run against its SQLite
commit:

```bash
PAISA_NEW_NEWS_RUN_ID="NEWS-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/verify_news_run.py \
  --run-dir "artifacts/news_ingestion/$PAISA_NEW_NEWS_RUN_ID" \
  --database data/news/paisa_news.sqlite3
```

## Article-level FinBERT sentiment

This optional stage needs `requirements-sentiment.txt` and the exact reviewed
FinBERT snapshot. It may download large model files during an authorized local
setup; CI must not run it. The model licence is CC BY-NC-SA 4.0 and its use here
is limited to non-commercial academic work. The stage emits article-level
probabilities only: it does not aggregate by day, join prices, retrain a stock
model, produce a signal, or update a portfolio or dashboard.

```bash
source .venv/bin/activate
python -m pip install -r requirements-sentiment.txt
python scripts/run_sentiment_inference.py --help

PAISA_NEW_SENTIMENT_RUN_ID="SENT-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_sentiment_inference.py \
  --source-run-id NEWS-20261022-131242-5ddffd2d \
  --database data/news/paisa_news.sqlite3 \
  --output-dir artifacts/sentiment_inference \
  --model-cache data/sentiment_model_cache \
  --device cpu \
  --batch-size 8 \
  --max-input-characters 4000 \
  --max-token-length 512 \
  --max-articles 500 \
  --run-id "$PAISA_NEW_SENTIMENT_RUN_ID"
```

The source news pack, matching SQLite evidence, and reviewed cached model bytes
must exist. A download or integrity failure is fatal and never selects another
classifier. Reusing an existing finalized `SENT-...` ID is forbidden.

## Current news-price coverage v3

The retained `COVERAGE-20261023-120000-real0002` and
`COVERAGE-20261023-130000-gdelt0001` packs are historical schema-`1.0`,
`news-price-coverage-v2` evidence. They verified at their producing revisions,
but the current schema-`1.1`, `news-price-coverage-v3` verifier intentionally
rejects them. Never edit or reseal a v2 pack to make it pass.

The current authoritative workflow restores the exact source evidence,
publishes a new v3 run under a fresh ID, and performs source-bound replay. The
example below uses the permission-allowlisted retained GDELT ingestion pack; it
still has no contemporaneous overlap with the retained price period and does
not unblock daily sentiment aggregation.

```bash
PAISA_COVERAGE_RUN_ID="COVERAGE-REPLACE-WITH-A-NEW-UNUSED-V3-ID"
PAISA_COVERAGE_RUN_DIR="artifacts/news_coverage/$PAISA_COVERAGE_RUN_ID"
PAISA_NEWS_RUN_DIR="artifacts/news_ingestion/NEWS-20261023-060000-gdelt-ogdc2"
PAISA_VERTICAL_RUN_DIR="artifacts/vertical_slice/VSL-20261020-OGDC-main-02"
PAISA_TRAINING_RUN_DIR="artifacts/model_training/TRN-20261020-191815-624662-nogit"

python scripts/audit_news_price_overlap.py \
  --news-run-dir "$PAISA_NEWS_RUN_DIR" \
  --news-database data/news/paisa_news.sqlite3 \
  --vertical-slice-dir "$PAISA_VERTICAL_RUN_DIR" \
  --model-training-dir "$PAISA_TRAINING_RUN_DIR" \
  --source-assessment docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md \
  --source-permission-policy configs/news/historical_source_permissions.json \
  --readiness-config configs/news_coverage/default_readiness.json \
  --market-timezone Asia/Karachi \
  --decision-cutoff 15:30:00 \
  --real-mode \
  --output-dir artifacts/news_coverage \
  --run-id "$PAISA_COVERAGE_RUN_ID"

python scripts/verify_news_coverage.py \
  --run-dir "$PAISA_COVERAGE_RUN_DIR" \
  --news-run-dir "$PAISA_NEWS_RUN_DIR" \
  --news-database data/news/paisa_news.sqlite3 \
  --vertical-slice-dir "$PAISA_VERTICAL_RUN_DIR" \
  --model-training-dir "$PAISA_TRAINING_RUN_DIR" \
  --source-assessment docs/HISTORICAL_NEWS_SOURCE_ASSESSMENT.md \
  --source-permission-policy configs/news/historical_source_permissions.json
```

A successful strict replay reports `verification_basis=SOURCE_EVIDENCE_BOUND`
and `source_evidence_bound=true`. If a selected old news pack predates the
completion seal, both commands require the explicit `--allow-legacy-unsealed`
flag; that compatibility boundary does not upgrade the source pack. Do not use
`--allow-fixture` for a real readiness claim.

When source evidence is genuinely unavailable, this explicit fallback performs
only weaker sealed-pack semantic verification:

```bash
PAISA_COVERAGE_RUN_DIR="artifacts/news_coverage/COVERAGE-REPLACE-WITH-AN-EXISTING-V3-ID"
python scripts/verify_news_coverage.py \
  --run-dir "$PAISA_COVERAGE_RUN_DIR" \
  --pack-only
```

A `PACK_SEMANTIC` pass is not source-bound authority and must not be reported as
such. Coverage thresholds are engineering/data gates, not evidence that
sentiment predicts prices or returns.

## Price-only LSTM M-005 and M-006

LSTM is an optional, offline training workflow. It requires TensorFlow but does
not download data or weights. Use CPython 3.10 through 3.13, preferably a
separate environment or the maintained Colab notebook. The input training and
vertical-slice artifacts are private and ignored by Git.

```bash
python3.13 -m venv .venv-lstm
source .venv-lstm/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lstm.txt
python -m pip install --no-deps -e .
python scripts/run_lstm_price_only_experiments.py --help

export PAISA_NEW_LSTM_RUN_ID="LSTM-REPLACE-WITH-A-NEW-UNUSED-ID"
python scripts/run_lstm_price_only_experiments.py \
  --prices artifacts/model_training/TRN-20261020-191815-624662-nogit/frozen_input/OGDC_prices.csv \
  --output-dir artifacts/lstm_experiments \
  --run-id "$PAISA_NEW_LSTM_RUN_ID" \
  --baseline-training-run artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --baseline-vertical-slice artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --device cpu
```

The reusable Colab entry point is
`notebooks/PAISA_OGDC_LSTM_Experiments.ipynb`. It calls the repository runner
and displays saved evidence rather than reimplementing the split, sequence,
evaluation, or portfolio logic. Preserve the certified
`LSTM-20261024-OGDC-price-only-02` run unchanged; every reproduction uses a new
ID.

Verify the newly generated LSTM pack without loading its Keras model:

```bash
python - <<'PY'
import os
from pathlib import Path

from paisa.experiments.artifacts import validate_training_run_artifacts

run_id = os.environ["PAISA_NEW_LSTM_RUN_ID"]
result = validate_training_run_artifacts(Path("artifacts/lstm_experiments") / run_id)
print(f"PASS LSTM experiment: {result.run_id}")
PY
```

## Unified price-only comparison

The retained `PAISA-RM-003` comparison derives M-001 through M-006 metrics
from persisted prediction, raw market, signal, and portfolio evidence. It
neither retrains models, invokes inference, nor calls the production backtest
runner. Its 184 dates are the reused **shared project holdout**, not a fresh
independent holdout. The comparison is price-only and tabular/report-only: it
does not integrate sentiment, emit plots, or import Matplotlib. See
[`UNIFIED_PRICE_MODEL_COMPARISON.md`](UNIFIED_PRICE_MODEL_COMPARISON.md) for the
schema and interpretation contract.

Use only `CMP-20261024-OGDC-price-only-02` for current generation and
verification. `CMP-20261024-OGDC-price-only-01` is a superseded, non-portable
historical draft. The final `-02` ZIP SHA-256 is
`8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`,
and its verified total file count is `65`.

Build the canonical immutable directory, portable ZIP, and ZIP checksum from
the three authoritative completed source roots:

```bash
source .venv/bin/activate
python scripts/build_unified_price_model_comparison.py \
  --comparison-id CMP-20261024-OGDC-price-only-02 \
  --output-dir artifacts/model_comparisons \
  --training-run artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --vertical-slice artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --lstm-run artifacts/lstm_experiments/LSTM-20261024-OGDC-price-only-02
```

Existing targets are never overwritten. Verify the published directory and
rebind it to the original sources:

```bash
python scripts/verify_unified_price_model_comparison.py \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02 \
  --training-run artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --vertical-slice artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --lstm-run artifacts/lstm_experiments/LSTM-20261024-OGDC-price-only-02
```

The ZIP/sidecar pair is portable and can be verified without the source roots:

```bash
python scripts/verify_unified_price_model_comparison.py \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02.zip \
  --checksum artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02.zip.sha256
```

The hardened verifier authenticates a digest-allowlisted redacted vertical
certificate, recursively rejects non-portable paths and credential material,
and requires exactly two raw market files plus six signal files. It then
performs independent full `Decimal` replay of model and benchmark policy,
execution, accounting, return, drawdown, and fee semantics. Metrics retain
exactly two logical benchmark rows; each row names six independent ledger
provenance records.

Normal command output reports artifact names relative to an explicit path base
so it can be copied into support logs. Default failures are concise JSON and do
not include a traceback. Pass `--debug` only for local diagnosis; its traceback
is still sanitized before display.

Run the full offline Python and frontend validation matrix before handoff:

```bash
source .venv/bin/activate
python -m pip check
python -m compileall -q src scripts tests
python -m pytest

cd frontend
npm ci
npm run test
npm run lint
npm run build
cd ..

git diff --check
git diff --check origin/main...HEAD
git status --short
```

## Tree-model SHAP explainability

The optional explainability producer consumes only the authoritative
M-001--M-004 training run, the authoritative unified comparison, and its bound
vertical slice. It does not retrain, tune, recalibrate, explain M-005/M-006,
modify a source, or create a portfolio. The exact 184 rows are the reused
**shared project holdout**, not a fresh independent evaluation.

Official runs require `shap==0.54.1` in the separate Python 3.12+ / NumPy 2+
environment documented above. `constraints-explainability.txt` pins the
certified direct NumPy, pandas, scikit-learn, XGBoost, joblib, Matplotlib, and
SHAP versions while both requirements files declare the environment; it is not
a standalone or fully hashed lock. The dedicated artifact-free Python 3.13.7
CI job runs real `shap.TreeExplainer` calls. Random Forest returns an explicit
class dimension and reconstructs within `5e-5`; XGBoost returns a
two-dimensional class-1 matrix and reconstructs within `1e-6`.

Restore the three generated, Git-ignored source roots before running:

```bash
source .venv-explainability/bin/activate
python scripts/run_tree_shap_explainability.py \
  --model-training-run \
  artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --comparison-run \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02 \
  --vertical-slice-run \
  artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --output-dir artifacts/explainability
```

The command validates fixed source integrity roots, trusted model paths,
manifest/model checksums, registered IDs, estimator classes, class order,
ordered features, dates, labels, and fitted preprocessing before explanation.
It uses interventional Tree SHAP in class-1 probability space with 100
deterministic training-background rows. Every one of the 184 rows per model
must satisfy its estimator-family additivity tolerance: `5e-5` for Random
Forest and `1e-6` for XGBoost.

The producer stages and verifies all files before collision-refusing,
no-replace publication. Record the returned run ID, then verify the directory:

```bash
# Accepted retained pack only; new runs also need the independent digest below.
PAISA_XAI_RUN_ID="XAI-20261024-140320-OGDC-tree-v1"
python scripts/verify_explainability_run.py \
  "artifacts/explainability/$PAISA_XAI_RUN_ID" \
  --require-authenticated
```

Verify a transferred deterministic ZIP with its sidecar:

```bash
# Accepted retained pack only; new runs also need the independent digest below.
PAISA_XAI_RUN_ID="XAI-20261024-140320-OGDC-tree-v1"
python scripts/verify_explainability_run.py \
  "artifacts/explainability/$PAISA_XAI_RUN_ID.zip" \
  --checksum \
  "artifacts/explainability/$PAISA_XAI_RUN_ID.zip.sha256" \
  --require-authenticated
```

The verifier is offline and model-free. It validates the exact inventory and
SHA-256 values, source/model/feature/date contracts, all local rows,
probability-space reconstruction, global importance, cross-model ranks,
deterministic audit selection, summaries, plots, portable paths, canonical ZIP
metadata, and prohibited causal/advice wording. It rejects missing and added
files, tampering, absolute paths, symlinks, executable model payloads, unsafe
ZIP members, and any false `PASS`.

The result deliberately separates integrity from provenance. `status: PASS`
and `integrity_status: PASS` mean that the pack is internally consistent.
`checksum_verified`/`sidecar_checksum_verified` mean only that an archive
matched its adjacent transport sidecar. Require
`authentication_status: AUTHENTICATED` for official use. The retained run is
anchored by a committed manifest and ZIP digest allowlist; a new run needs an
independently obtained `artifact_manifest.json` digest supplied with
`--expected-manifest-sha256` and enforced with `--require-authenticated`.
PNG checks are structural; the authenticated manifest binds the accepted plot
bytes without claiming that the verifier rerendered them.

The accepted local real pack is
`artifacts/explainability/XAI-20261024-140320-OGDC-tree-v1/` with its sibling
ZIP and sidecar. Its clean code version/source fingerprint is
`2cd4145e5e01459d13d186a6f3b697e979821a9d`, its committed
`artifact_manifest.json` SHA-256 is
`cd43f76a208bed6b549d6dbfb3b4e468dd91db021f4543f1cfefdbbc7a661fda`, its ZIP
SHA-256 is
`7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`, and
its exact 24-file inventory reports `PASS`. Authenticated directory, ZIP, and
extracted-directory verification all passed; the sidecar transport-integrity
check also passed. The recorded runtime is Python 3.13.7, NumPy 2.3.5, pandas
2.3.3, scikit-learn 1.9.1,
XGBoost 3.4.1, joblib 1.6.0, Matplotlib 3.11.2, and SHAP 0.54.1.

All M-001--M-004 results contain 184 rows and 19 transformed features in
positive-class probability space. Maximum/mean absolute additivity errors were
`5.78484978308591e-06` / `5.781760213738571e-06`,
`4.302598055616613e-09` / `7.864803434113863e-10`,
`2.7466758000471714e-05` / `2.7464541568790406e-05`, and
`1.5920226559273942e-07` / `3.5509845686483396e-08`, respectively. Every one
of the 184 rows per model passed its `5e-5` Random Forest or `1e-6` XGBoost
tolerance, and all seven audit categories were available for each model.

The strongest rank agreements were M-001/M-002 (`0.8228070175438595`, four
top-five overlaps), M-001/M-003 (`0.8210526315789475`, five), and M-002/M-003
(`0.7543859649122807`, four). The reported disagreements were M-001/M-004
(`0.5964912280701754`, four), M-003/M-004 (`0.6175438596491227`, four), and
M-002/M-004 (`0.7`, four). Generated explanation directories and ZIPs stay
out of Git. See
[`TREE_MODEL_SHAP_EXPLAINABILITY.md`](TREE_MODEL_SHAP_EXPLAINABILITY.md),
[`EXPLAINABILITY_DATA_CONTRACT.md`](EXPLAINABILITY_DATA_CONTRACT.md), and
[`SHAP_INTERPRETATION_POLICY.md`](SHAP_INTERPRETATION_POLICY.md). Explanations
describe fitted-model behavior on a previously exposed shared project holdout,
not economic causality, profitability, or future results. Evidence covers one
stock, one period, one feature family, and four tree configurations. Academic
simulation; not financial advice.

## Final handoff checks

Run the complete validation matrix, report the observed results, and confirm
that only intended source/documentation changes remain:

```bash
source .venv/bin/activate
python scripts/doctor.py
python -m pip check
python -m compileall -q src scripts tests
python -m pytest

cd frontend
npm ci
npm run test
npm run lint
npm run build
cd ..

git diff --check
git diff --check origin/main...HEAD
git status --short
```

Do not copy historical test counts into a handoff. Record the counts and skips
from the checkout that was actually tested, along with the exact source commit,
configuration, artifact IDs, date ranges, feature order, target definition,
prediction horizon, random seeds, and metrics.
