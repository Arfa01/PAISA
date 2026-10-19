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

The provisional champion is selected from validation metrics only. The test
split remains untouched until configurations and the champion-selection rule
have been frozen; test results never select or tune a model.

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

## Saved evidence

Each successful run creates a new directory below:

```text
artifacts/model_training/<run_id>/
```

The run records, at minimum:

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
It uses the shared data-preparation function and must preserve the same target,
chronological splits, prediction schema, and artifact contract before an LSTM
is promoted beyond an optional experiment.

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
