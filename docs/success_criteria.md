# PAISA v0 success criteria

Use this checklist after every major change.

## Data ingestion

- [ ] Pipeline accepts at least 3 PSX symbols.
- [ ] Each requested symbol is saved as a raw CSV under `data/raw/psx/`.
- [ ] Failed symbols are reported in `data/processed/pipeline_metadata.json`.
- [ ] Pipeline does not silently continue with an empty dataset.

## Cleaning

- [ ] `data/processed/psx_prices.csv` exists.
- [ ] Columns are exactly: `date, ticker, open, high, low, close, volume, source`.
- [ ] Dates are sorted per ticker.
- [ ] Duplicate ticker-date rows are removed.
- [ ] `open`, `close`, and `volume` are numeric and non-empty.

## Feature engineering and label generation

- [ ] `data/processed/psx_features.csv` exists.
- [ ] Moving averages, lag features, returns, volatility, RSI, and MACD columns exist.
- [ ] `data/processed/ml_dataset.csv` exists.
- [ ] `target_next_day_up` exists and contains only 0/1 values.
- [ ] The final row without a known next-day target is not used for training.

## Model training

- [ ] The split is chronological, not random.
- [ ] `models/baseline_random_forest.joblib` exists.
- [ ] `models/baseline_metrics.json` exists.
- [ ] Metrics include accuracy, precision, recall, and F1.
- [ ] `models/baseline_feature_importance.csv` exists.

## Backend API

- [ ] `/health` returns `status: ok`.
- [ ] `/stocks` lists processed tickers.
- [ ] `/stocks/{ticker}/prices` returns cleaned price rows.
- [ ] `/stocks/{ticker}/features` returns feature rows.
- [ ] `/stocks/{ticker}/dataset` returns ML-ready rows.
- [ ] `/models/baseline/metrics` returns model metrics.
- [ ] `/predict/{ticker}` returns direction, confidence, top factors, and disclaimer.

## Testing

- [ ] `pytest` passes.
- [ ] Sample provider works offline.
- [ ] PSX provider works on a real internet connection.
