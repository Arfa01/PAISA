from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from paisa.features import FEATURE_COLUMNS
from paisa.splits import chronological_split


def _metrics(y_true: pd.Series, y_pred: list[int] | pd.Series) -> dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }


def train_baseline_random_forest(
    dataset: pd.DataFrame,
    model_dir: str | Path,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and save v0 RandomForest baseline on chronological train/val/test splits."""
    if dataset.empty:
        raise ValueError("Cannot train baseline model because dataset is empty.")
    missing = [col for col in FEATURE_COLUMNS + ["target_next_day_up"] if col not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset missing model columns: {missing}")

    model_path_dir = Path(model_dir)
    model_path_dir.mkdir(parents=True, exist_ok=True)

    train, validation, test = chronological_split(dataset)
    if train["target_next_day_up"].nunique() < 2:
        raise ValueError("Training split has only one target class. Add more symbols/dates.")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train["target_next_day_up"])

    split_frames = {"train": train, "validation": validation, "test": test}
    metrics: dict[str, Any] = {
        "model_name": "baseline_random_forest",
        "feature_columns": FEATURE_COLUMNS,
        "split_row_counts": {name: int(len(frame)) for name, frame in split_frames.items()},
        "split_date_ranges": {},
        "scores": {},
    }

    for name, frame in split_frames.items():
        if frame.empty:
            continue
        y_true = frame["target_next_day_up"]
        y_pred = model.predict(frame[FEATURE_COLUMNS])
        metrics["scores"][name] = _metrics(y_true, y_pred)
        metrics["split_date_ranges"][name] = {
            "start": str(pd.to_datetime(frame["date"]).min().date()),
            "end": str(pd.to_datetime(frame["date"]).max().date()),
        }
        if name == "test":
            pd.DataFrame(confusion_matrix(y_true, y_pred), index=["actual_down", "actual_up"], columns=["pred_down", "pred_up"]).to_csv(
                model_path_dir / "baseline_confusion_matrix.csv"
            )

    importances = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(model_path_dir / "baseline_feature_importance.csv", index=False)

    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "feature_importance": importances.to_dict(orient="records"),
    }
    joblib.dump(artifact, model_path_dir / "baseline_random_forest.joblib")
    (model_path_dir / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
