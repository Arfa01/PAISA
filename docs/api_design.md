# PAISA v0 API design

Base URL during local development:

```text
http://127.0.0.1:8000
```

## Endpoints

### `GET /health`

Confirms whether the data and model artifacts are ready.

### `GET /pipeline/status`

Returns latest pipeline metadata: provider, requested symbols, row counts, errors, date range, and metrics.

### `GET /stocks`

Returns available tickers.

### `GET /stocks/{ticker}/prices`

Returns standardized historical price rows.

### `GET /stocks/{ticker}/features`

Returns engineered feature rows.

### `GET /stocks/{ticker}/dataset`

Returns complete rows used for supervised model training.

### `GET /models/baseline/metrics`

Returns baseline RandomForest metrics.

### `GET /predict/{ticker}`

Returns a v0 next-trading-day up/down prediction using latest complete feature row.

Important: This endpoint does not issue buy/sell recommendations.
