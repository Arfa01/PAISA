# Training when live PSX providers are blocked

The OGDC Random Forest training stage normally uses a live real PSX provider:

```bash
python scripts/train_ogdc_rf_variants.py \
  --symbol OGDC \
  --start 2021-01-01 \
  --provider psx-dps \
  --output-dir artifacts/model_training \
  --random-state 42
```

If `psx-dps` returns `403`/`404` and `pypsx-toolkit` fails to import on the local machine, use the existing processed PSX files that were generated from a previous successful real PSX run.

This is not a sample-data fallback. The script refuses to run if the existing processed rows are marked as `sample`.

## Required local files

```text
data/processed/psx_prices.csv
data/processed/psx_features.csv
```

These files must contain OGDC rows from a real PSX provider such as `psx_dps`.

## Run command

```bash
python scripts/train_ogdc_rf_variants_from_existing_processed.py \
  --symbol OGDC \
  --start 2021-01-01 \
  --output-dir artifacts/model_training \
  --random-state 42
```

Then run:

```bash
pytest
```

## Expected output

A successful run creates:

```text
artifacts/model_training/<run_id>/
```

and a ZIP archive:

```text
artifacts/paisa_<run_id>.zip
```

Check:

```text
reports/final_pass_fail_checklist.json
reports/final_summary.json
reports/validation_leaderboard.csv
reports/test_leaderboard.csv
reports/baselines.csv
```

## Why this exists

The training-stage requirements say not to generate fake replacement data and not to report a successful run if real data acquisition fails. This script keeps that rule by using only already-existing processed PSX files and by rejecting sample/synthetic sources.
