# PAISA v0/v0.1 API design

Base URL during local development:

```text
http://127.0.0.1:8000
```

## Endpoints

### `GET /health`

Confirms whether the data and model artifacts are ready.

### `GET /pipeline/status`

Returns latest pipeline metadata: provider, requested symbols, row counts, errors, date range, and baseline metrics.

### `GET /stocks`

Returns available tickers.

### `GET /stocks/{ticker}/prices`

Returns standardized historical price rows.

### `GET /stocks/{ticker}/features`

Returns engineered feature rows.

### `GET /stocks/{ticker}/dataset`

Returns complete rows used for supervised model training.

### `GET /quality/data`

Returns an overall data-quality summary plus per-ticker quality checks. Use this before trusting model results.

### `GET /quality/tickers`

Returns only the per-ticker quality rows. This is useful for a future dashboard table.

### `GET /models/baseline/metrics`

Returns baseline RandomForest metrics.

### `GET /models/comparison`

Returns saved price-only model comparison results from `scripts/run_model_comparison.py`.

### `GET /predict/{ticker}`

Returns a v0 next-trading-day up/down prediction using latest complete feature row.

Important: This endpoint does not issue buy/sell recommendations.
