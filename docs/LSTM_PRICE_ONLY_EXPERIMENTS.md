# PAISA OGDC Price-Only LSTM Experiments

This runbook covers the controlled `M-005` and `M-006` recurrent experiments.
They use price and technical features only: no article, sentiment, event, or
future-market field enters either model. The experiment is an academic
simulation and is not financial advice.

The authoritative outcome of any execution is its checksum-inventoried
`final_summary.json`; documentation text is not a substitute for that artifact.

## Controlled variants

| Variant | Frozen label | Controlled configuration |
|---|---|---|
| `M-005` | `LSTM | 20-session sequence | baseline recurrent configuration | frozen` | One 64-unit LSTM layer, 20-session input, dropout `0.20`, and sigmoid binary output. |
| `M-006` | `LSTM | 20-session sequence | regularized recurrent configuration | frozen` | The same ordered features, 20-session input, and 64 recurrent units, with stronger dropout/recurrent regularization. |

The two variants retain the same data, feature order, sequence length, target,
split dates, random-seed policy, optimizer family, prediction schema, trading
rules, and portfolio assumptions. Their immutable model manifests contain the
complete realized layer, regularization, optimizer, learning-rate, batch-size,
early-stopping, and class-weight configuration. A neural-network label is not
evidence of superiority.

## Optional runtime

TensorFlow/Keras remains outside `requirements.txt`, so normal PAISA CPU tests
and the Random Forest/XGBoost pipeline do not depend on it. Install the default
environment first, then add a compatible TensorFlow build only in the runtime
used for this experiment. TensorFlow 2.21 supports CPython 3.10 through 3.13;
do not create this environment with Python 3.14. For example, select Python
3.13 explicitly:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lstm.txt
python -m pip install -e .
```

`requirements-lstm.txt` includes the normal PAISA requirements and pins the
supported TensorFlow release range for this workflow.

CPU execution is supported. A Colab GPU may be used when available. Apple
Metal is optional, macOS-only, and requires a TensorFlow-visible GPU through a
separately installed/configured plugin such as `tensorflow-metal`; it is never
a requirement for tests or the real run. Use `--device gpu` for a generic GPU
request, `--device metal` only for that macOS policy, or the default
`--device auto` to use TensorFlow's available runtime. The artifact records the
actual Python, TensorFlow/Keras and NumPy versions, effective device, duration,
seeds, and determinism status. Deterministic operations are requested where
supported, but PAISA does not claim bit-identical CPU/GPU/Metal results.

## Required immutable inputs

For the canonical comparison, use the retained daily OGDC input and the exact
baseline evidence it belongs to:

```text
artifacts/model_training/TRN-20261020-191815-624662-nogit/
artifacts/vertical_slice/VSL-20261020-OGDC-main-02/
```

These generated artifacts are ignored by Git. A fresh clone must restore a
checksum-verified custodian bundle or regenerate equivalent evidence with new
immutable IDs. Do not silently fetch replacement data and call it the canonical
comparison. The accepted training run's `frozen_input/OGDC_prices.csv` contains
1,239 sessions from 2021-09-16 through 2026-09-15.

## Run from the command line

Choose a new run ID. Existing destinations fail rather than being overwritten:

```bash
python scripts/run_lstm_price_only_experiments.py \
  --prices artifacts/model_training/TRN-20261020-191815-624662-nogit/frozen_input/OGDC_prices.csv \
  --output-dir artifacts/lstm_experiments \
  --run-id LSTM-REPLACE-WITH-A-NEW-UNUSED-ID \
  --baseline-training-run artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --baseline-vertical-slice artifacts/vertical_slice/VSL-20261020-OGDC-main-02
```

Use `--no-zip` only when an archive is not needed. Run
`python scripts/run_lstm_price_only_experiments.py --help` for the exact CLI
contract.

The equivalent reusable entry point is:

```python
from paisa.experiments.lstm_runner import run_lstm_price_only_experiments

summary = run_lstm_price_only_experiments(
    prices_path="path/to/OGDC_prices.csv",
    output_dir="artifacts/lstm_experiments",
    run_id="LSTM-NEW-UNUSED-ID",
    baseline_training_run="path/to/TRN-run",
    baseline_vertical_slice="path/to/VSL-run",
    create_zip=True,
    device="auto",
)
```

The Colab/CPU notebook
[`PAISA_OGDC_LSTM_Experiments.ipynb`](../notebooks/PAISA_OGDC_LSTM_Experiments.ipynb)
calls this reusable entry point and only displays saved evidence. It does not
reimplement sequence, model, metric, or portfolio logic.

## Selection and post-freeze LSTM test policy

The common target-date split is chronological. Learned preprocessing is fit on
training rows only. Early stopping, learning-rate decisions, epoch selection,
checkpoint selection, and variant preference use validation evidence only.
The test split is evaluated after the controlled configurations are frozen and
is never used to tune or relabel a champion.

This timing guarantee applies to the M-005/M-006 selection process only. The
same 184-date holdout, from 2025-12-16 through 2026-09-15, and its M-001 through
M-004 results had already been exposed by the earlier baseline experiment. It
is therefore the reused shared project holdout, not a fresh independent sample.

On the canonical data, the 20-session contract yields 834 training, 182
validation, and 184 test sequences. All baseline and LSTM validation/test dates
match, but the runner still proves and persists the common-date intersection.
See [the sequence contract](LSTM_SEQUENCE_CONTRACT.md) for boundary details.

`validation_preferred_variant` means only the result of the declared
validation rule. It is not renamed because of test accuracy or simulated
portfolio return. Test results and portfolio outcomes are reported separately.

## Backtesting contract

Each test `predictions.csv` is passed directly to the existing portfolio engine
with no schema conversion. It has exactly:

```text
schema_version,model_id,ticker,feature_date,target_date,predicted_class,
probability_up,probability_down,actual_class
```

Probabilities must be finite, lie in `[0, 1]`, and sum to one within `1e-6`.
Backtests retain the baseline rules: PKR 100,000 initial cash, buy at `0.60`,
sell at `0.40`, 50% maximum allocation, 10-basis-point transaction cost, zero
slippage, long-only state transitions, next-session-open execution, close
valuation, and a 50%-allocated buy-and-hold benchmark. Every variant owns an
independent strategy and benchmark portfolio.

## Artifact layout

Each successful execution creates one new immutable directory:

```text
artifacts/lstm_experiments/<run_id>/
├── run_manifest.json
├── frozen_input/
│   └── OGDC_prices.csv
├── split_reference.json
├── preprocessing/
├── models/
│   ├── <M-005-registered_model_id>/
│   └── <M-006-registered_model_id>/
├── validation_leaderboard.csv
├── test_leaderboard.csv
├── training_history.csv
├── comparison_plots/
├── final_summary.json
└── artifact_manifest.json
```

`run_manifest.json` maps the stable experiment variant IDs `M-005` and `M-006`
to their unique immutable `registered_model_id` directory names. Consumers must
follow that mapping instead of assuming the variant ID is the directory name.

Persisted `final_summary.json` uses schema `1.1`. Its
`artifact_directory=<run_id>`, optional `zip_path=<run_id>.zip`, and
`path_base=output_root` fields are portable rather than machine-specific
absolute paths. It reports LSTM test evidence under
`lstm_metrics.post_freeze_lstm_test` and
`post_freeze_lstm_test_ranking`; `shared_test_holdout_context` records the
prior M-001 through M-004 exposure. The Python runner's returned summary still
resolves the artifact and ZIP paths for immediate local use and reports
`path_base=absolute` for those in-memory/CLI values.

When ZIP output is requested, the complete ZIP is published first and the run
directory rename is the final publication boundary. A failed or interrupted
final rename can therefore leave a complete orphan ZIP, but it cannot expose a
public `PASS` run that references a missing ZIP. Collision refusal remains in
force; an operator must inspect an orphan explicitly and use a new run ID or
remove it deliberately rather than having failure cleanup risk deleting a
concurrently replaced file.

Per-model evidence includes the registered model identity, fitted model,
preprocessing artifacts, ordered features, sequence length, split dates,
configuration/architecture, parameter count, training history, predictions,
metrics, confusion matrix, environment/source provenance, and checksums. The
run-level manifest binds those files and rejects collisions or later mutation.
Validation also reconciles each advertised preprocessing and best-validation
checkpoint digest with the corresponding run-inventory entry and actual file;
resealing cannot legitimize a false nested checksum claim.

## Certified real run

The retained local run `LSTM-20261024-OGDC-price-only-02` completed with status
`PASS`. It was produced from clean source commit
`2bd20edebac9f5a063e2e10cc0394343067d8c7f` on CPU with Python `3.13.7`,
TensorFlow `2.21.0`, NumPy `2.3.5`, and seed `42`. Generated run artifacts and
the ZIP are intentionally ignored by Git; the checksum-sealed local directory
and its `final_summary.json` are the result source of record.

The earlier immutable `LSTM-20261024-OGDC-price-only-01` run is retained as
historical pre-hardening evidence. Its schema `1.0` summary does not demonstrate
the portable-path, publication-order, or revised shared-holdout contracts; it
was not overwritten or relabeled as the certified result.

The frozen input contains 1,239 OGDC daily sessions from 2021-09-16 through
2026-09-15 and produces 1,219 labeled technical-feature rows with 19 ordered
features. Target-date membership is 853/182/184 for
train/validation/test. The 20-session window retains 834/182/184 sequences:
the first 19 training targets are reported as dropped, while all validation and
test targets remain. The retained target ranges are 2021-11-11 through
2025-03-21, 2025-03-24 through 2025-12-15, and 2025-12-16 through 2026-09-15.

Both one-layer models have 64 recurrent units, a sigmoid output, 21,569
parameters, Adam at `0.001`, batch size 32, and 50 requested epochs. `M-005`
uses dropout `0.20`; `M-006` uses dropout `0.40`, recurrent dropout `0.10`, and
kernel/recurrent L2 `0.0001`. Early stopping completed both fits after 7 epochs,
with epoch 1 as the best validation checkpoint. Training durations were 2.503
seconds for M-005 and 4.036 seconds for M-006 on this host. The registered IDs
are:

- `M-005-1fee47047b61cc43-81548726-c0c4e1fd-50f7-4fa4-9963-01b6b3f85a46`
- `M-006-67bcc0f398cc963a-81548726-c0c4e1fd-50f7-4fa4-9963-01b6b3f85a46`

### Validation-only selection metrics

The declared validation rule prefers `M-005`. Test and portfolio evidence did
not change that label.

| Variant | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | ROC-AUC | Brier | ECE | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M-005 | 0.472527 | 0.479280 | 0.476190 | 0.786517 | 0.593220 | -0.052561 | 0.453425 | 0.262189 | 0.092499 | `[[16, 77], [19, 70]]` |
| M-006 | 0.472527 | 0.474931 | 0.468468 | 0.584270 | 0.520000 | -0.051383 | 0.446176 | 0.260540 | 0.082305 | `[[34, 59], [37, 52]]` |

### Post-freeze LSTM test metrics

Each frozen model made one test prediction pass after validation selection was
recorded. These dates and the M-001 through M-004 outcomes were already known,
so “post-freeze” describes LSTM evaluation order rather than a globally
untouched project test set.

| Variant | Accuracy | Balanced accuracy | Precision | Recall | F1 | MCC | ROC-AUC | Brier | ECE | Confusion matrix `[[TN, FP], [FN, TP]]` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| M-005 | 0.516304 | 0.520087 | 0.506410 | 0.868132 | 0.639676 | 0.055921 | 0.562094 | 0.250776 | 0.054840 | `[[16, 77], [12, 79]]` |
| M-006 | 0.543478 | 0.544251 | 0.533333 | 0.615385 | 0.571429 | 0.089395 | 0.557249 | 0.249100 | 0.070173 | `[[44, 49], [35, 56]]` |

All six models share the same 184 test target dates; no model loses a date in
the fair intersection. On that exact set, balanced accuracy ranks M-006
`0.544251`, M-002 `0.524459`, M-005 `0.520087`, M-001 `0.512289`, M-003
`0.506794`, and M-004 `0.492910`. Thus M-006 exceeds every M-001 through M-004
baseline on this one predefined test metric. This does not replace the
validation-selected M-005, establish general superiority, or imply that the
LSTM portfolio performs better.

### Fixed-policy portfolio results

| Variant/reference | Ending value (PKR) | Total return | Max drawdown | Volatility | Sharpe | Trades | Fees (PKR) | Difference vs buy-and-hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M-005 | 96,359.29 | -3.64071% | 14.65748% | 12.53449% | -0.342422 | 3 | 146.08 | -9.88860 percentage points |
| M-006 | 96,110.91 | -3.88909% | 15.46586% | 12.43528% | -0.374556 | 3 | 146.72 | -10.13698 percentage points |
| Buy-and-hold | 106,247.89 | 6.24789% | — | — | — | — | — | 0 |

Neither LSTM beats buy-and-hold or the best model portfolio. On the same dates
and rules, M-002 is the best model portfolio at PKR 105,987.04 and 5.98704%,
still below buy-and-hold by 0.26085 percentage points.

The run-level validator accepted 147 checksum-inventoried payload files. The
persisted summary uses schema `1.1`, repository-independent relative paths, and
the post-freeze/shared-holdout field names; neither the run directory nor its
ZIP contains the originating absolute workspace path. The comparison
leaderboard has one canonical `registered_model_id` column. Both
184-row prediction files use the exact nine-column backtester schema, contain
finite complementary probabilities, and cover 2025-12-16 through 2026-09-15.
Reloading each saved bundle reproduced all 182 validation probabilities exactly
without a second test inference. The complete ZIP passed integrity testing and
has SHA-256
`f9d3ef10bb80528388a9137e7f169a7d870e7429f779f46e074a6254a26d8821`.
The published `run_manifest.json` has SHA-256
`e475e077f375071c79ce8743425831d8cd9cf5ceb37f42eca0acfa65a100f3d0`.

One stock, one period, and one trading policy cannot establish general market
superiority or future performance. Unknown corporate-action adjustment,
unmodeled dividends, small sample size, class/regime sensitivity, threshold
sensitivity, and cross-device numeric variation remain limitations.

## Verification

From an activated environment, run:

```bash
python -m pytest
python -m pip check
python -m compileall src scripts
git diff --check

cd frontend
npm ci
npm run test
npm run lint
npm run build
```

The default test suite must remain offline. TensorFlow smoke training is bounded
to a tiny fixture and skips only when the optional dependency is absent. For
the certified run above, the TensorFlow-enabled full suite completed with 1,036
passing backend tests and one Windows-only junction test skipped on macOS; the
frontend completed 57 tests plus lint and build.
