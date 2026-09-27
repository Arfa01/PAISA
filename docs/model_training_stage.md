# PAISA model-training stage: frozen OGDC Random Forest variants

This stage is a separate experimental layer. It does **not** replace the existing PAISA v0 baseline model.

## Purpose

The stage creates reproducible Random Forest variants for one stock, OGDC, using daily PSX technical data only. It freezes the input dataset, builds a paper-trading-aligned next-session target, trains multiple immutable variants, compares them on the same chronological splits, and saves all artifacts under a unique run directory.

## What it does

- Uses the existing PAISA PSX pipeline with `provider=psx-dps` by default.
- Requests OGDC data from `2021-01-01`.
- Does not use synthetic fallback data.
- Preserves the v0 target but uses a separate experimental target:
  - signal after close of day `t`
  - earliest future execution at open of day `t+1`
  - target measures `close(t+1) / open(t+1) - 1`
- Creates short, week, and month feature contexts.
- Uses a shared chronological train/validation/test split with boundary purging.
- Trains five immutable Random Forest variants:
  - M-001: short context, CFG-A
  - M-002: week context, CFG-A
  - M-003: month context, CFG-A
  - M-004: month context, CFG-B conservative
  - M-005: month context, CFG-C flexible
- Selects a provisional champion using validation balanced accuracy and validation F1 as tie breaker.
- Saves model files, predictions, metrics, manifests, plots, and a ZIP archive.

## Run locally

```bash
source .venv/bin/activate
python scripts/train_ogdc_rf_variants.py --symbol OGDC --start 2021-01-01 --provider psx-dps --output-dir artifacts/model_training --random-state 42
```

## If `psx-dps` returns 404

The DPS endpoint is undocumented and may temporarily reject requests even when the symbol is valid. The provider now sends browser-like headers and retries the endpoint with and without a trailing slash.

After pulling the latest branch, retry the same command. If it still returns 404, use the alternate real PSX provider instead of sample/synthetic data:

```bash
python scripts/train_ogdc_rf_variants.py --symbol OGDC --start 2021-01-01 --provider pypsx --output-dir artifacts/model_training --random-state 42
```

Do not use `--provider sample` for the real training-stage artifact.

## Run in Colab/Kaggle

Use:

```text
notebooks/train_ogdc_rf_variants_colab_kaggle.ipynb
```

## Output

Each run creates:

```text
artifacts/model_training/<run_id>/
```

and a ZIP archive:

```text
artifacts/paisa_<run_id>.zip
```

In Colab, the ZIP is placed in `/content/`. In Kaggle, the ZIP is placed in `/kaggle/working/`.

## Not included yet

This stage does not perform portfolio simulation, live trading, news analysis, sentiment analysis, fundamental analysis, reinforcement learning, XGBoost, LSTM, SHAP, or financial advice.
