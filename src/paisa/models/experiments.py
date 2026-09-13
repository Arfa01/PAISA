from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paisa.features import FEATURE_COLUMNS
from paisa.splits import chronological_split


def _scores(y_true: pd.Series, y_pred: list[int] | pd.Series) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def _candidate_models(random_state: int) -> dict[str, Any]:
    """Return lightweight price-only models that work with the current requirements.

    XGBoost and LSTM are intentionally not included in v0.1 so that teammates can
    run the comparison after a normal `pip install -r requirements.txt`. They
    should be added in later milestones once the data pipeline is validated.
    """
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=4,
            random_state=random_state,
            class_weight="balanced_subsample",
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            random_state=random_state,
        ),
    }


def run_price_only_model_comparison(
    dataset: pd.DataFrame,
    model_dir: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train comparable price-only models and save a model comparison table."""
    if dataset.empty:
        raise ValueError("Cannot run model comparison because dataset is empty.")
    missing = [col for col in FEATURE_COLUMNS + ["target_next_day_up"] if col not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset missing model columns: {missing}")

    train, validation, test = chronological_split(dataset)
    if train["target_next_day_up"].nunique() < 2:
        raise ValueError("Training split has only one target class. Add more symbols/dates.")

    model_path_dir = Path(model_dir)
    model_path_dir.mkdir(parents=True, exist_ok=True)

    split_frames = {"train": train, "validation": validation, "test": test}
    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    for model_name, model in _candidate_models(random_state).items():
        model.fit(train[FEATURE_COLUMNS], train["target_next_day_up"])
        artifacts[model_name] = model

        for split_name, frame in split_frames.items():
            if frame.empty:
                continue
            y_true = frame["target_next_day_up"]
            y_pred = model.predict(frame[FEATURE_COLUMNS])
            score = _scores(y_true, y_pred)
            rows.append(
                {
                    "model": model_name,
                    "feature_set": "price_only_technical_indicators",
                    "split": split_name,
                    "rows": int(len(frame)),
                    "start_date": str(pd.to_datetime(frame["date"]).min().date()),
                    "end_date": str(pd.to_datetime(frame["date"]).max().date()),
                    **score,
                }
            )

    comparison = pd.DataFrame(rows).sort_values(["split", "f1", "accuracy"], ascending=[True, False, False])
    comparison.to_csv(model_path_dir / "model_comparison_price_only.csv", index=False)

    result = {
        "experiment_name": "price_only_model_comparison_v0_1",
        "feature_columns": FEATURE_COLUMNS,
        "models": list(_candidate_models(random_state).keys()),
        "splits": {name: int(len(frame)) for name, frame in split_frames.items()},
        "results": comparison.to_dict(orient="records"),
        "important_note": "This is a price-only comparison. Sentiment, XGBoost, LSTM, and SHAP are later milestones.",
    }
    (model_path_dir / "model_comparison_price_only.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    joblib.dump({"models": artifacts, "feature_columns": FEATURE_COLUMNS, "results": result}, model_path_dir / "model_comparison_price_only.joblib")
    return result
