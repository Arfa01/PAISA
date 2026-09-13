# PAISA Project Context

## Project name

**PAISA — Predictive AI Stock Assessment**

## Project type

Final Year Project for BS Computer Science at COMSATS University Islamabad, Lahore Campus.

Team:

- Arfa Riaz — FA23-BCS-035 — Group Leader
- Waqas Ul Hasan — FA23-BCS-167

## Main idea

PAISA is an AI-assisted stock market analysis system for retail investors. The project combines historical stock data with financial news, sentiment analysis, event-driven reasoning, and explainable prediction outputs.

The system should not directly say “buy” or “sell.” It should show AI-assisted insight for academic and educational use.

## Proposal objectives

The proposal defines these main objectives:

1. Collect, preprocess, and structure historical stock market data from 2021 to 2026.
2. Build training, validation, and test splits.
3. Build a continuously updating data pipeline for financial news, economic reports, and global events.
4. Train and compare multiple machine learning models for stock movement prediction.
5. Compare models with and without sentiment-based features.
6. Deploy the best-performing model as the prediction engine.
7. Add event-driven reasoning to show affected companies/sectors and possible propagation.
8. Build a user-facing dashboard with predictions, visualizations, explanations, and current data feeds.

## Proposed methodology

The proposal has four technical parts:

### 1. Data collection and preprocessing

Collect stock and news data from trusted APIs or open online sources. Clean and standardize the data, handle missing/inconsistent records, and align stock timestamps with news timestamps.

### 2. NLP and sentiment analysis

Use NLP tools to extract company names, sectors, industries, geographical signals, event signals, and positive/negative sentiment from financial news.

Suggested tools: FinBERT / HuggingFace Transformers.

### 3. Knowledge graph / relationship modelling

Model relationships between companies, sectors, industries, events, and affected entities. The goal is to represent how one event may affect more than one company through market, sector, or supply-chain connections.

Suggested tools: NetworkX or Neo4j.

### 4. Prediction, evaluation, explainability, and dashboard

Train supervised ML models for stock trend prediction. Compare model performance using accuracy, precision, recall, and F1-score. Add explanations such as feature importance or SHAP. Serve results through an API and web dashboard.

Suggested models: Random Forest, XGBoost, LSTM.

Suggested backend/frontend: FastAPI or Flask, React.js, Plotly.

## Team responsibilities from proposal

Arfa:

- Stock market data collection and integration
- Feature engineering from news and market data
- Dataset preparation and label generation
- Model training and performance evaluation
- Web interface design and visualization planning
- Web interface development
- Documentation and reporting

Waqas:

- News data collection and integration
- NLP processing and entity extraction from news
- Knowledge graph construction and relationship mapping
- Machine learning model development
- Backend API design and integration planning
- Backend API development and integration
- Documentation and reporting

Both:

- System testing
- Final improvements
- Documentation/reporting

## Current implementation status: v0

v0 is the first working implementation slice. It focuses on the stock-data foundation.

v0 does this:

1. Fetches PSX stock data.
2. Saves raw ticker-level data.
3. Cleans data into a standard schema.
4. Generates price/volume technical indicators.
5. Generates the next-trading-day up/down label.
6. Creates an ML-ready dataset.
7. Splits data chronologically.
8. Trains a baseline Random Forest model.
9. Saves metrics and feature importance.
10. Serves data and prediction outputs through FastAPI.

## Current data source

v0 uses the `psx-dps` provider as the PSX-first source.

The provider uses the observed endpoint format:

```text
https://dps.psx.com.pk/timeseries/eod/{SYMBOL}
```

Observed row format:

```text
[unix_timestamp, close_price, volume, open_price]
```

So the current source reliably gives:

- date
- open
- close
- volume

High and low may be unavailable from this source and should be documented as a v0 limitation.

## Current successful local run

This command worked locally:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train
```

Observed result:

```text
provider: psx_dps
symbols: OGDC, HBL, MCB
raw rows per symbol: 1239
prices rows: 3717
features rows: 3717
ML dataset rows: 3657
date range: 2021-09-14 to 2026-09-11
errors: {}
```

The FastAPI endpoints worked when started with:

```bash
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

or:

```bash
python scripts/serve_api.py
```

## Current generated files

Generated locally after running the pipeline:

```text
data/raw/psx/
data/processed/psx_prices.csv
data/processed/psx_features.csv
data/processed/ml_dataset.csv
data/processed/pipeline_metadata.json
models/baseline_random_forest.joblib
models/baseline_metrics.json
models/baseline_feature_importance.csv
models/baseline_confusion_matrix.csv
```

Generated data/model files are ignored by Git so teammates can regenerate them locally.

## Current v0.1 branch

Branch:

```text
feature/v0.1-data-quality
```

Purpose:

```text
Harden the stock-data foundation before starting news/sentiment work.
```

v0.1 adds:

1. Expanded stock universe file with priority levels.
2. Pipeline option to read symbols from CSV.
3. Pipeline option to generate a data-quality report.
4. Data-quality module and report generator.
5. API endpoints for data-quality output.
6. Price-only model-comparison workflow.
7. API endpoint for saved model-comparison output.
8. Tests for data quality and model comparison.
9. Cleaner run instructions and project documentation.

## v0.1 commands

From a fresh clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Run expanded PSX pipeline:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols-from-csv data/metadata/stock_universe.csv --max-priority 2 --start 2021-01-01 --train --quality-report
```

Run model comparison:

```bash
python scripts/run_model_comparison.py
```

Run tests:

```bash
pytest
```

Start API:

```bash
python scripts/serve_api.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

## v0.1 API endpoints

```text
GET /health
GET /pipeline/status
GET /stocks
GET /stocks/{ticker}/prices
GET /stocks/{ticker}/features
GET /stocks/{ticker}/dataset
GET /quality/data
GET /quality/tickers
GET /models/baseline/metrics
GET /models/comparison
GET /predict/{ticker}
GET /docs
```

## v0 model status

The first PSX baseline model is weak but acceptable for v0 because the purpose of v0 was to prove the pipeline, not final prediction accuracy.

Observed PSX baseline result:

```text
test accuracy: about 0.517
test F1: about 0.440
```

This means the current baseline is mainly a benchmark to improve against.

## Current limitations

Not implemented yet:

- News collection
- Sentiment analysis
- Entity extraction
- Knowledge graph
- SHAP explanations
- XGBoost
- LSTM
- React dashboard
- Database integration
- Scheduled automatic updates

Known data limitation:

- Current PSX endpoint may start from 2021-09 for tested symbols, not January 2021.
- Current PSX endpoint may not provide high/low values.

## Definition of Done for v0

v0 is successful when:

1. PSX pipeline runs successfully.
2. Raw ticker data is saved.
3. `psx_prices.csv` is generated.
4. `psx_features.csv` is generated.
5. `ml_dataset.csv` is generated.
6. `target_next_day_up` exists.
7. Baseline Random Forest model trains.
8. Accuracy, precision, recall, and F1 are saved.
9. `pytest` passes.
10. FastAPI starts successfully.
11. `/docs` shows the API UI.
12. `/predict/{ticker}` returns prediction JSON.

## Definition of Done for v0.1

v0.1 is successful when:

1. Expanded priority-2 stock universe pipeline runs.
2. Data-quality report is generated.
3. Report shows row counts, date coverage, missing values, duplicates, and label balance.
4. Price-only model comparison runs.
5. `/quality/data`, `/quality/tickers`, and `/models/comparison` work.
6. Tests pass.
7. Teammate can clone and run using `RUN_FROM_GITHUB.md`.

## Next recommended versions

### v1 — News ingestion

Collect financial news linked to companies, dates, and sectors.

### v2 — Sentiment features

Use FinBERT/HuggingFace to generate daily sentiment features per company.

### v3 — Price-only vs price+sentiment comparison

Compare whether sentiment actually improves accuracy, precision, recall, and F1.

### v4 — Explainability

Add SHAP or improved model-specific explanations.

### v5 — Dashboard

Build React + Plotly dashboard for stock selection, charts, predictions, quality summaries, model results, and explanations.

## Main rule going forward

Do not jump straight to LSTM or dashboard.

Follow this order:

```text
Reliable PSX data → data quality → price-only baseline comparison → news collection → sentiment features → price+sentiment comparison → explainability → dashboard
```
