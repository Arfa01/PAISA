# PAISA v0.1 Price-Only Model Comparison

Run this after the PSX pipeline has generated `data/processed/ml_dataset.csv`:

```bash
python scripts/run_model_comparison.py
```

or:

```bash
make compare
```

## Models compared in v0.1

This first comparison intentionally uses lightweight models that work with the current dependency list:

- Logistic Regression
- Random Forest
- Histogram Gradient Boosting

## Why XGBoost/LSTM are not here yet

The FYP proposal includes Random Forest, XGBoost, and LSTM. v0.1 is still focused on validating the data and experiment flow. XGBoost and LSTM should be added after:

1. the PSX stock universe is stable,
2. data-quality reports are clean,
3. the train/validation/test split is accepted,
4. the prediction target is confirmed by the supervisor/team.

## Output files

Running the comparison creates:

```text
models/model_comparison_price_only.csv
models/model_comparison_price_only.json
models/model_comparison_price_only.joblib
```

Generated model files are ignored by Git and should be regenerated locally.

## How to evaluate the result

Use the validation and test rows, not the training rows, to compare models. Training scores are mainly useful for detecting overfitting.

Important metrics:

- accuracy
- precision
- recall
- F1-score

The proposal requires comparison using these standard classification metrics and later comparison with and without sentiment-based features.
