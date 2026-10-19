# PAISA versioned OGDC model-training stage

This stage is the reproducible model boundary for the OGDC vertical slice. It
reuses PAISA's existing daily PSX cleaning and technical-feature pipeline and
does not replace the legacy v0 baseline model. It does not own decision rules,
portfolio balances, trade execution, API routes, or dashboard logic.

All outputs are academic experiments and are not financial advice.

## Models

The default full run trains the following immutable variants from identical
modelling rows and split dates. A deliberately selected `--model-id` subset
uses the same preparation and artifact contracts:

| Model | Algorithm | Purpose |
|---|---|---|
| `M-001` | Random Forest | Existing baseline configuration. |
| `M-002` | Random Forest | Conservative configuration. |
| `M-003` | Random Forest | More flexible configuration. |
| `M-004` | XGBoost classifier | Same dates, features, target, and splits as the Random Forest variants. |

The run also evaluates two non-trained references:

- Majority class, learned from the training split only.
- Previous direction, using only information available on the feature date.

The provisional champion is selected from validation metrics only, and only
when both actual classes occur in that split. For a one-class split, balanced
accuracy, Matthews correlation coefficient, and ROC-AUC are undefined, the
metrics carry `single_class_actuals`, and selection fails closed. The test split
remains untouched until configurations and the champion-selection rule have
been frozen; test results never select or tune a model.

## Target and timing contract

The experimental target follows `docs/ARCHITECTURE_CONTRACTS.md`:

```text
target_next_day_up = 1 if close[target_date] > close[feature_date], else 0
```

`feature_date` is trading session `t`. `target_date` is the next observed OGDC
trading session, not the next calendar date. A feature row uses information
available no later than the close of `feature_date`. The prediction is formed
after that close and can execute no earlier than `target_date` at the price
defined by the later backtesting experiment.

Rows are split chronologically. Split assignment follows target dates, with
boundary purging where needed so a future label cannot cross from one split
into another. Rows are never shuffled. Imputers or other learned transforms are
fit on training rows only.

## Install and train

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/train_ogdc_models.py
```

The explicit equivalent is:

```bash
python scripts/train_ogdc_models.py \
  --symbol OGDC \
  --start 2021-01-01 \
  --provider psx-dps \
  --output-dir artifacts/model_training \
  --random-state 42
```

Use `--prices data/processed/psx_prices.csv` to train from an existing cleaned
price file instead of fetching. Use `--run-id` only to supply a new immutable
identifier; an existing destination fails. `--no-zip` is useful for quick local
smoke tests. Run `python scripts/train_ogdc_models.py --help` for details.
`pypsx-toolkit` remains an optional provider and is not required by the default
installation.

When `--prices` is omitted, training still reuses the v0 provider and cleaning
pipeline, but runs it in an isolated temporary acquisition directory. Only the
normalized symbol data is copied into the run's `frozen_input/`; canonical
`data/`, `models/`, and the stocks exposed by the v0 API are never replaced by
a model-training fetch.

## Saved evidence

Each successful run creates a new directory below:

```text
artifacts/model_training/<run_id>/
```

The run records, at minimum:

- A per-invocation UUID that makes registered model IDs globally unique even
  when the same run ID, data, configuration, and seed are used in independent
  artifact roots. This UUID does not affect fitting; the recorded random seed
  and deterministic semantic metadata remain the reproducibility inputs.
- Ordered feature columns and feature-set version.
- Complete model configuration and random seed.
- Data fingerprint, provider, symbol, and coverage.
- Inclusive train, validation, and test target-date ranges.
- Target definition and one-session horizon.
- Predictions and class probabilities for every split.
- Accuracy, balanced accuracy, precision, recall, F1, Matthews correlation
  coefficient, ROC-AUC when valid, Brier score, confusion matrix, and class
  distributions.
- Feature importance where the estimator supports it.
- Fitted model binary, checksum, manifest, tabular leaderboards, and a ZIP of
  the complete run.

Writers reject an existing run or model destination rather than overwriting it.
Generated inputs, binaries, results, and ZIP files are ignored by Git.

## Prediction handoff to backtesting

Every per-model test `predictions.csv` includes the prediction-record fields:

```text
schema_version,model_id,ticker,feature_date,target_date,predicted_class,
probability_up,probability_down,actual_class
```

`actual_class` is evaluation-only. It is not available to feature generation,
model inference, policies, or execution. Probabilities map through the saved
class order, sum to one within contract tolerance, and an exact tie maps to
class `0`.

The CSV can be passed directly to the backtester. Supply the cleaned daily
price file and an experiment configuration whose `model_id`, symbol, and
evaluation dates match the chosen prediction rows:

```bash
python scripts/run_backtest.py \
  --predictions artifacts/model_training/<run_id>/models/<model_id>/predictions.csv \
  --prices data/processed/psx_prices.csv \
  --config path/to/experiment.json \
  --runs-dir runs
```

The price CSV may be the canonical multi-symbol `psx_prices.csv`; the
backtester selects the configured symbol before checking duplicate sessions and
evaluation dates. It does not require a separate OGDC-only conversion.

Backtesting remains independent: training does not create signals, trades, or
portfolio state.

## Colab notebooks

The primary notebook is:

```text
notebooks/PAISA_OGDC_Model_Training.ipynb
```

It clones or uses the repository, installs the normal dependencies, invokes the
same reusable trainer used by the CLI, displays saved metrics, creates
presentation-only plots from the saved leaderboards, and exposes the generated
ZIP for download. Training logic remains in `src/paisa/`; plot files are not
part of the immutable run artifact.

The optional deep-learning workflow is separate:

```text
notebooks/PAISA_OGDC_LSTM_Training.ipynb
```

Its TensorFlow/Keras installation is notebook-local and is not part of
`requirements.txt`, so it cannot block the Random Forest/XGBoost CPU pipeline.
The notebook calls `paisa.experiments.lstm.train_and_register_lstm`; model
construction, sequence preparation, evaluation, and persistence are not hidden
in notebook cells. The helper fits `StandardScaler` on training rows only and
uses earlier chronological rows as validation/test context without changing
target split membership or including a row after the prediction's
`feature_date`. Keras training always uses `shuffle=False`.

Each optional LSTM run creates an immutable model registry directory below
`artifacts/lstm/<model_id>/`. Its checksum-protected `lstm_bundle.joblib`
contains both the Keras v3 model bytes and the fitted scaler, along with ordered
features and sequence length needed for raw-feature inference. `manifest.json`
uses the same complete model-artifact contract as the CPU models, including
configuration, transforms, split date ranges and metrics, target/horizon,
class mapping, source/data provenance, dependencies, artifact path, and
checksum. The saved test `predictions.csv` is validated through the shared
prediction-record contract and remains a direct backtesting input. TensorFlow
is required only to train or invoke this optional bundle; importing and testing
the CPU pipeline does not import TensorFlow.

## Provider limitations

The public PSX endpoint can be unavailable or reject automated requests. A
provider failure is reported as a failed acquisition; the trainer does not
silently substitute sample rows for a real-data run. See
[`training_when_live_psx_is_blocked.md`](training_when_live_psx_is_blocked.md)
for the existing-processed-data path.

The current provider does not independently establish adjusted prices or full
corporate-action and dividend treatment. Manifests therefore retain the
contract's `price_adjustment: "unknown"`, `dividends_included: false`, and
`corporate_actions_policy: "not_modeled"` limitations unless a source is later
verified.
