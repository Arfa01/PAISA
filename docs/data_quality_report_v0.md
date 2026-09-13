# PAISA v0 Data Quality Report

This file is generated/updated by:

```bash
python scripts/generate_data_quality_report.py
```

or automatically during the pipeline when `--quality-report` is used:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols-from-csv data/metadata/stock_universe.csv --max-priority 2 --start 2021-01-01 --train --quality-report
```

## What this report checks

The report verifies whether the PSX data layer is safe enough to use before adding stronger ML, news, sentiment, and dashboard work.

It checks:

- number of rows per ticker
- start and end date per ticker
- missing open, high, low, close, and volume values
- duplicate ticker/date rows
- largest calendar gap between available trading rows
- number of ML-ready rows per ticker
- next-day up/down label balance

## Expected v0 limitations

The current `psx-dps` source may not provide high and low values. That is acceptable for v0 because the first feature set uses open, close, volume, and technical indicators derived from close/volume. It should still be documented in every report.

## Blocking issues

Treat these as problems that must be fixed before model comparison:

- missing close values
- missing volume values
- duplicate ticker/date rows
- zero ML-ready dataset rows for a ticker
- very small row count for a ticker
- extremely imbalanced labels

## Output files

Running the report creates:

```text
data/processed/data_quality_report.json
data/processed/data_quality_report.csv
docs/data_quality_report_v0.md
```

Generated files inside `data/processed/` are ignored by Git and should be regenerated locally.
