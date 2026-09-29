from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def binary_classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, probability_up: np.ndarray) -> dict[str, Any]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    prob_arr = np.asarray(probability_up, dtype=float)

    labels_present = sorted(np.unique(y_true_arr).tolist())
    metrics: dict[str, Any] = {
        "row_count": int(len(y_true_arr)),
        "accuracy": _safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": _safe_float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "precision": _safe_float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": _safe_float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": _safe_float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "matthews_corrcoef": _safe_float(matthews_corrcoef(y_true_arr, y_pred_arr)),
        "roc_auc": None,
        "brier_score": _safe_float(brier_score_loss(y_true_arr, prob_arr)),
        "log_loss": None,
        "class_counts": {str(k): int(v) for k, v in pd.Series(y_true_arr).value_counts().sort_index().to_dict().items()},
        "predicted_class_counts": {str(k): int(v) for k, v in pd.Series(y_pred_arr).value_counts().sort_index().to_dict().items()},
        "confusion_matrix": confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).astype(int).tolist(),
    }

    if len(labels_present) == 2:
        metrics["roc_auc"] = _safe_float(roc_auc_score(y_true_arr, prob_arr))
        proba_2col = np.column_stack([1.0 - prob_arr, prob_arr])
        metrics["log_loss"] = _safe_float(log_loss(y_true_arr, proba_2col, labels=[0, 1]))

    return metrics


def make_prediction_frame(
    rows: pd.DataFrame,
    ticker: str,
    model_id: str,
    split_name: str,
    y_pred: np.ndarray,
    probability_up: np.ndarray,
) -> pd.DataFrame:
    probability_up = np.asarray(probability_up, dtype=float)
    y_pred = np.asarray(y_pred).astype(int)
    result = pd.DataFrame(
        {
            "feature_date": pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d"),
            "target_date": pd.to_datetime(rows["target_date"]).dt.strftime("%Y-%m-%d"),
            "ticker": ticker.upper(),
            "model_id": model_id,
            "actual_class": rows["target_next_session_up"].astype(int).to_numpy(),
            "predicted_class": y_pred,
            "probability_up": probability_up,
            "probability_down": 1.0 - probability_up,
            "is_correct": (y_pred == rows["target_next_session_up"].astype(int).to_numpy()).astype(int),
            "feature_open": rows["open"].to_numpy(),
            "feature_close": rows["close"].to_numpy(),
            "next_open": rows["next_open"].to_numpy(),
            "next_close": rows["next_close"].to_numpy(),
            "target_next_session_return": rows["target_next_session_return"].to_numpy(),
            "split": split_name,
        }
    )
    return result


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for split_name, group in predictions.groupby("split", sort=False):
        summary[split_name] = binary_classification_metrics(
            group["actual_class"].astype(int),
            group["predicted_class"].astype(int).to_numpy(),
            group["probability_up"].astype(float).to_numpy(),
        )
    if "train" in summary and "validation" in summary:
        summary["train_to_validation_gaps"] = {
            "accuracy_gap": _gap(summary, "accuracy"),
            "f1_gap": _gap(summary, "f1"),
            "balanced_accuracy_gap": _gap(summary, "balanced_accuracy"),
        }
    return summary


def _gap(summary: dict[str, dict[str, Any]], metric: str) -> float | None:
    train_value = summary["train"].get(metric)
    validation_value = summary["validation"].get(metric)
    if train_value is None or validation_value is None:
        return None
    return round(float(train_value) - float(validation_value), 6)


def predictions_to_leaderboard(metrics_by_model: dict[str, dict[str, Any]], split: str) -> pd.DataFrame:
    records = []
    for model_id, model_metrics in metrics_by_model.items():
        split_metrics = model_metrics["metrics"].get(split, {})
        record = {
            "model_id": model_id,
            "label": model_metrics.get("label"),
            "feature_set": model_metrics.get("feature_set_name"),
            "rf_config": model_metrics.get("rf_config_name"),
        }
        for metric_name in [
            "row_count",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "matthews_corrcoef",
            "roc_auc",
            "brier_score",
            "log_loss",
        ]:
            record[metric_name] = split_metrics.get(metric_name)
        records.append(record)

    frame = pd.DataFrame(records)
    if not frame.empty and "balanced_accuracy" in frame.columns:
        frame = frame.sort_values(
            ["balanced_accuracy", "f1", "accuracy"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)
    return frame


def select_provisional_champion(validation_leaderboard: pd.DataFrame) -> dict[str, Any]:
    if validation_leaderboard.empty:
        raise ValueError("Validation leaderboard is empty; cannot select champion.")
    row = validation_leaderboard.iloc[0].to_dict()
    return {
        "selection_rule": "Highest validation balanced accuracy; validation F1 as tie breaker.",
        "model_id": row["model_id"],
        "label": row["label"],
        "validation_balanced_accuracy": row.get("balanced_accuracy"),
        "validation_f1": row.get("f1"),
    }


def make_reference_baselines(test_rows: pd.DataFrame, train_rows: pd.DataFrame) -> pd.DataFrame:
    if test_rows.empty:
        raise ValueError("Cannot calculate reference baselines because test rows are empty.")

    y_test = test_rows["target_next_session_up"].astype(int)
    train_counts = train_rows["target_next_session_up"].astype(int).value_counts()
    majority_class = int(train_counts.sort_values(ascending=False).index[0])

    baselines = []
    majority_pred = np.full(len(test_rows), majority_class, dtype=int)
    majority_prob = np.full(len(test_rows), float(majority_class), dtype=float)
    majority_metrics = binary_classification_metrics(y_test, majority_pred, majority_prob)
    baselines.append({"baseline": "majority_class_from_training", **majority_metrics})

    previous_direction_pred = (pd.to_numeric(test_rows["daily_return"], errors="coerce").fillna(0) > 0).astype(int).to_numpy()
    previous_direction_prob = previous_direction_pred.astype(float)
    previous_metrics = binary_classification_metrics(y_test, previous_direction_pred, previous_direction_prob)
    baselines.append({"baseline": "previous_daily_direction", **previous_metrics})

    return pd.DataFrame(baselines)


def add_cumulative_performance(test_predictions: pd.DataFrame) -> pd.DataFrame:
    frames = []
    ordered = test_predictions.sort_values(["model_id", "feature_date"]).copy()
    for model_id, group in ordered.groupby("model_id", sort=False):
        g = group.copy().reset_index(drop=True)
        g["daily_correctness"] = g["is_correct"].astype(int)
        g["cumulative_correct_predictions"] = g["daily_correctness"].cumsum()
        g["cumulative_incorrect_predictions"] = (1 - g["daily_correctness"]).cumsum()
        g["cumulative_accuracy"] = g["cumulative_correct_predictions"] / (np.arange(len(g)) + 1)
        g["rolling_5_trading_day_hit_rate"] = g["daily_correctness"].rolling(5, min_periods=1).mean()
        g["rolling_20_trading_day_hit_rate"] = g["daily_correctness"].rolling(20, min_periods=1).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
