# PAISA Project Context

## Project name

**PAISA — Predictive AI Stock Assessment**

## Project type

Final Year Project for BS Computer Science at COMSATS University Islamabad, Lahore Campus.

Team:

- Arfa Riaz — FA23-BCS-035 — Group Leader
- Waqas Ul Hasan — FA23-BCS-167

## Main idea

PAISA is an AI-based stock market analysis system for retail investors. The goal is to combine historical stock data with financial news, sentiment analysis, and event-driven reasoning so that users can view stock trends, predictions, and explanations in a simple dashboard.

The system should not directly give financial advice such as “buy” or “sell.” It should provide AI-assisted prediction and explanation for academic/research purposes.

## Problem being solved

Retail investors often rely on incomplete information, intuition, social media hype, or scattered online opinions. Existing retail platforms usually show charts and basic historical prices, but they do not provide a complete accessible system with:

- machine-learning-based stock trend prediction
- sentiment analysis
- event-driven reasoning
- relationship mapping between companies/sectors/events
- explainable predictions for non-expert users

PAISA aims to fill this gap by building one integrated platform.

## Proposal objectives

The proposal defines these core objectives:

1. Collect, preprocess, and structure historical stock market data from 2021 to 2026.
2. Build a continuously updating data pipeline for financial news, economic reports, and global events.
3. Train and compare multiple machine learning models for stock movement prediction.
4. Compare model performance with and without sentiment-based features.
5. Deploy the best-performing model as the prediction engine.
6. Add event-driven reasoning to show which companies/sectors are affected by events.
7. Build a user-facing digital platform with predictions, visualizations, explanations, and current data feeds.

## Proposed technical components

### 1. Data collection and preprocessing

Collect stock and news data from trusted APIs/open online sources. Clean and standardize it. Handle missing or inconsistent records. Synchronize timestamps between stock data and news data.

### 2. NLP and sentiment analysis

Use NLP tools to extract:

- company names
- industries
- sectors
- geographical regions
- sentiment tone
- event signals

Suggested tools from proposal: FinBERT / HuggingFace Transformers.

### 3. Knowledge graph / relationship modelling

Map relationships between:

- companies
- sectors
- industries
- real-world events
- affected entities

Suggested tools from proposal: NetworkX or Neo4j.

### 4. Prediction models

Train and compare supervised ML models for stock trend prediction.

Models mentioned in the proposal:

- Random Forest
- XGBoost
- LSTM

Metrics mentioned in the proposal:

- accuracy
- precision
- recall
- F1-score

### 5. Explainability

Predictions should include reasons. The proposal mentions explainability methods such as SHAP.

### 6. Backend and dashboard

The backend should retrieve data, run prediction models, and return prediction + explanation. The dashboard should allow users to view historical trends, predicted trends, and analytical summaries.

Suggested stack:

- Python
- Pandas / NumPy
- MySQL or MongoDB
- FastAPI or Flask
- React.js
- Plotly
- scikit-learn
- SHAP

## Team responsibilities from proposal

Arfa:

- stock market data collection and integration
- feature engineering from news and market data
- dataset preparation and label generation
- model training and performance evaluation
- web interface design and visualization planning
- web interface development
- documentation and reporting

Waqas:

- news data collection and integration
- NLP processing and entity extraction from news
- knowledge graph construction and relationship mapping
- machine learning model development
- backend API design and integration planning
- backend API development and integration
- documentation and reporting

Both:

- system testing
- final improvements
- documentation/reporting

## Current implementation status: PAISA v0

v0 is the first working implementation slice. It is focused on the stock-data foundation.

Current v0 does the following:

1. Fetches PSX stock data.
2. Saves raw ticker data.
3. Cleans stock data into a standard schema.
4. Generates technical indicators.
5. Generates the next-day movement label.
6. Creates an ML-ready dataset.
7. Splits the dataset chronologically.
8. Trains a baseline Random Forest model.
9. Saves model metrics and feature importance.
10. Serves data and predictions through FastAPI.

## Current data source

v0 uses the `psx-dps` provider as the main PSX-first source.

The provider currently works with the public PSX DPS time-series endpoint format:

```text
https://dps.psx.com.pk/timeseries/eod/{SYMBOL}
```

Observed row format:

```text
[unix_timestamp, close_price, volume, open_price]
```

Because of this, the current v0 reliably uses:

- date
- open
- close
- volume

High and low may remain empty if the source does not provide them.

## Current successful local run

The following PSX run worked locally:

```bash
python scripts/run_v0_pipeline.py --provider psx-dps --symbols OGDC HBL MCB --start 2021-01-01 --train
```

Observed successful output:

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

The API endpoints also work when started with:

```bash
PYTHONPATH=src python -m uvicorn paisa.api.main:app --reload --host 127.0.0.1 --port 8000
```

The `/docs` endpoint shows the FastAPI UI for accessing all other endpoints.

## Current generated files

After running the PSX pipeline, these generated files should exist locally:

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

Generated data and model files are intentionally ignored by Git. Teammates should regenerate them locally after cloning.

## Current API endpoints

```text
GET /health
GET /pipeline/status
GET /stocks
GET /stocks/{ticker}/prices
GET /stocks/{ticker}/features
GET /stocks/{ticker}/dataset
GET /models/baseline/metrics
GET /predict/{ticker}
GET /docs
```

## Current model status

v0 trains a baseline Random Forest model.

The baseline model is only a first benchmark. It is not expected to be highly accurate yet.

Observed PSX model result:

```text
test accuracy: about 0.517
test F1: about 0.440
```

This means the current model is only a weak baseline. The value of v0 is that the pipeline works end-to-end.

## Important limitations of v0

v0 does not yet include:

- news collection
- sentiment analysis
- entity extraction
- knowledge graph
- SHAP explanations
- XGBoost comparison
- LSTM model
- React dashboard
- database integration
- scheduled automatic updates

Current PSX data begins from 2021-09-14 for the tested symbols, not January 2021. This should be documented as a current data-source limitation.

## Definition of Done for v0

v0 is considered successful when:

1. The PSX pipeline command runs successfully.
2. Raw ticker data is saved.
3. Cleaned `psx_prices.csv` is generated.
4. Feature-engineered `psx_features.csv` is generated.
5. ML-ready `ml_dataset.csv` is generated.
6. `target_next_day_up` exists.
7. Baseline Random Forest model is trained.
8. Accuracy, precision, recall, and F1 are saved.
9. `pytest` passes.
10. FastAPI starts successfully.
11. `/docs` displays the API UI.
12. `/predict/{ticker}` returns a prediction JSON.

## Immediate next milestone: v0.1

Before adding sentiment or dashboard, v0 should be hardened.

Recommended next tasks:

1. Push current v0 to GitHub.
2. Add `RUN_FROM_GITHUB.md` and `PROJECT_CONTEXT.md` to the repo.
3. Expand stock universe from 3 tickers to 10–20 PSX tickers.
4. Run the pipeline on the larger stock list.
5. Create a simple data-quality report.
6. Check missing values, date coverage, duplicate rows, and per-ticker row counts.
7. Document PSX source limitations.
8. Confirm the exact prediction target with the supervisor: next-day up/down or another horizon.
9. Improve the baseline experiment so results are reproducible and easy to compare.

## Recommended next versions

### v0.1 — Data quality and larger stock universe

Goal: Make PSX stock pipeline more reliable and document data quality.

### v0.2 — Better baseline experiments

Goal: Train and compare simple ML models using price-only technical indicators.

Suggested models:

- Logistic Regression
- Random Forest
- XGBoost, if added

### v1 — News pipeline

Goal: Collect financial news linked to companies/sectors/dates.

### v2 — Sentiment features

Goal: Use FinBERT/HuggingFace model to generate daily sentiment features per stock.

### v3 — Model comparison with and without sentiment

Goal: Compare price-only models against price + sentiment models.

### v4 — Explainability

Goal: Add SHAP or feature-importance explanations for predictions.

### v5 — Dashboard

Goal: Build React + Plotly dashboard for stock selection, charts, predictions, sentiment, and explanations.

## Main rule going forward

Do not jump directly to the dashboard or LSTM. Keep the project incremental:

```text
Reliable data → reproducible dataset → baseline model → comparison models → news/sentiment → explainability → dashboard
```
