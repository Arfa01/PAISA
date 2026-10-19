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

from paisa.trading.models import PredictionRecord


PREDICTION_SCHEMA_VERSION = "1.0"
PREDICTION_COLUMNS = [
    "schema_version",
    "model_id",
    "ticker",
    "feature_date",
    "target_date",
    "predicted_class",
    "probability_up",
    "probability_down",
    "actual_class",
]
_TARGET_COLUMN_CANDIDATES = ("target_next_day_up", "target_next_session_up")


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


def _validated_binary_array(values: Any, *, field_name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if result.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    try:
        parsed = result.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only binary values") from exc
    if not np.isin(parsed, [0, 1]).all() or not np.equal(result, parsed).all():
        raise ValueError(f"{field_name} must contain only 0 and 1")
    return parsed


def binary_classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    probability_up: np.ndarray,
) -> dict[str, Any]:
    """Return deterministic schema-1.0 metrics for one chronological split."""

    y_true_arr = _validated_binary_array(y_true, field_name="y_true")
    y_pred_arr = _validated_binary_array(y_pred, field_name="y_pred")
    prob_arr = np.asarray(probability_up, dtype=float)
    if prob_arr.ndim != 1 or len(prob_arr) != len(y_true_arr):
        raise ValueError("probability_up must be one-dimensional and match y_true")
    if len(y_pred_arr) != len(y_true_arr):
        raise ValueError("y_pred must have the same length as y_true")
    if not np.isfinite(prob_arr).all() or ((prob_arr < 0) | (prob_arr > 1)).any():
        raise ValueError("probability_up must contain only finite values in [0, 1]")

    labels_present = sorted(np.unique(y_true_arr).tolist())
    class_distribution = {
        "0": int((y_true_arr == 0).sum()),
        "1": int((y_true_arr == 1).sum()),
    }
    predicted_distribution = {
        "0": int((y_pred_arr == 0).sum()),
        "1": int((y_pred_arr == 1).sum()),
    }
    balanced_accuracy = (
        accuracy_score(y_true_arr, y_pred_arr)
        if len(labels_present) == 1
        else balanced_accuracy_score(y_true_arr, y_pred_arr)
    )
    matthews = (
        0.0
        if len(labels_present) == 1
        else matthews_corrcoef(y_true_arr, y_pred_arr)
    )
    metrics: dict[str, Any] = {
        "row_count": int(len(y_true_arr)),
        "support": int(len(y_true_arr)),
        "positive_label": 1,
        "average": "binary",
        "zero_division": 0,
        "accuracy": _safe_float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": _safe_float(balanced_accuracy),
        "precision": _safe_float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": _safe_float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": _safe_float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "matthews_corrcoef": _safe_float(matthews),
        "roc_auc": None,
        "brier_score": _safe_float(brier_score_loss(y_true_arr, prob_arr)),
        "log_loss": None,
        "class_distribution": class_distribution,
        # Compatibility aliases retained for the existing reports.
        "class_counts": class_distribution,
        "predicted_class_distribution": predicted_distribution,
        "predicted_class_counts": predicted_distribution,
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
    split_name: str | None = None,
    y_pred: np.ndarray | None = None,
    probability_up: np.ndarray | None = None,
    *,
    target_column: str | None = None,
) -> pd.DataFrame:
    """Build the exact prediction-record boundary consumed by backtesting.

    ``split_name`` and ``y_pred`` remain accepted for compatibility with the
    earlier trainer.  Split membership belongs in an internal evaluation table,
    not the stable prediction record.  The contract class is always derived
    from probabilities, including the required down-or-flat result on a tie.
    """

    del split_name, y_pred
    if probability_up is None:
        raise ValueError("probability_up is required")
    probability = np.asarray(probability_up, dtype=float)
    if probability.ndim != 1 or len(probability) != len(rows):
        raise ValueError("probability_up must be one-dimensional and match rows")

    actual_column = target_column
    if actual_column is None:
        actual_column = next(
            (candidate for candidate in _TARGET_COLUMN_CANDIDATES if candidate in rows.columns),
            None,
        )
    if actual_column is not None and actual_column not in rows.columns:
        raise ValueError(f"Target column is missing: {actual_column}")

    feature_column = "feature_date" if "feature_date" in rows.columns else "date"
    if feature_column not in rows.columns or "target_date" not in rows.columns:
        raise ValueError("rows must include feature/date and target_date columns")

    probability_down = 1.0 - probability
    predicted = (probability > probability_down).astype(int)
    actual: Any
    if actual_column is None:
        actual = [None] * len(rows)
    else:
        actual = rows[actual_column].to_numpy()
    result = pd.DataFrame(
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "model_id": model_id,
            "ticker": ticker.upper(),
            "feature_date": pd.to_datetime(rows[feature_column]).dt.strftime("%Y-%m-%d"),
            "target_date": pd.to_datetime(rows["target_date"]).dt.strftime("%Y-%m-%d"),
            "predicted_class": predicted,
            "probability_up": probability,
            "probability_down": probability_down,
            "actual_class": actual,
        }
    )
    return validate_prediction_frame(result)


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize, and deterministically order a prediction table."""

    missing = sorted(set(PREDICTION_COLUMNS) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(PREDICTION_COLUMNS))
    if missing or extra:
        raise ValueError(
            f"Prediction columns must exactly match schema 1.0; missing={missing}, extra={extra}"
        )
    data = frame[PREDICTION_COLUMNS].copy()
    if data.empty:
        raise ValueError("Prediction frame must contain at least one row")
    if not (data["schema_version"].astype(str) == PREDICTION_SCHEMA_VERSION).all():
        raise ValueError("Every prediction must use schema_version 1.0")

    data["feature_date"] = pd.to_datetime(data["feature_date"], errors="raise")
    data["target_date"] = pd.to_datetime(data["target_date"], errors="raise")
    if not (data["target_date"] > data["feature_date"]).all():
        raise ValueError("Every target_date must be later than feature_date")

    for column in ("probability_up", "probability_down"):
        data[column] = pd.to_numeric(data[column], errors="raise")
    probabilities = data[["probability_up", "probability_down"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Prediction probabilities must be finite and in [0, 1]")
    if not np.isclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6).all():
        raise ValueError("probability_up and probability_down must sum to 1 within 1e-6")

    data["predicted_class"] = pd.to_numeric(data["predicted_class"], errors="raise")
    expected = (data["probability_up"] > data["probability_down"]).astype(int)
    if not (data["predicted_class"] == expected).all():
        raise ValueError("predicted_class must be derived from probabilities; ties map to 0")
    data["predicted_class"] = data["predicted_class"].astype(int)

    actual_values: list[int | None] = []
    for value in data["actual_class"]:
        if pd.isna(value) or value == "":
            actual_values.append(None)
            continue
        if isinstance(value, (bool, np.bool_)):
            raise ValueError("actual_class must be 0, 1, or null")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("actual_class must be 0, 1, or null") from exc
        if parsed not in (0, 1) or float(value) != parsed:
            raise ValueError("actual_class must be 0, 1, or null")
        actual_values.append(parsed)
    data["actual_class"] = actual_values

    key = ["model_id", "ticker", "feature_date", "target_date"]
    if data.duplicated(key).any():
        raise ValueError("Prediction uniqueness key is duplicated")

    normalized_records = []
    for row in data.to_dict(orient="records"):
        record = PredictionRecord.from_mapping(row)
        normalized_records.append(record.to_dict())
    normalized = pd.DataFrame(normalized_records, columns=PREDICTION_COLUMNS)
    if normalized.duplicated(key).any():
        raise ValueError("Prediction uniqueness key is duplicated after normalization")
    return normalized.sort_values(
        ["model_id", "ticker", "feature_date", "target_date"], kind="mergesort"
    ).reset_index(drop=True)


def summarize_predictions(predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    groups = (
        predictions.groupby("split", sort=False)
        if "split" in predictions.columns
        else [("all", predictions)]
    )
    for split_name, group in groups:
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
            "configuration": model_metrics.get("configuration_name")
            or model_metrics.get("rf_config_name"),
            "algorithm": model_metrics.get("algorithm"),
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
            ["balanced_accuracy", "f1", "accuracy", "model_id"],
            ascending=[False, False, False, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    return frame


def select_provisional_champion(validation_leaderboard: pd.DataFrame) -> dict[str, Any]:
    if validation_leaderboard.empty:
        raise ValueError("Validation leaderboard is empty; cannot select champion.")
    row = validation_leaderboard.iloc[0].to_dict()
    return {
        "selection_rule": (
            "Highest validation balanced accuracy; then validation F1; then "
            "validation accuracy; then ascending model_id as deterministic tie breakers."
        ),
        "model_id": row["model_id"],
        "label": row["label"],
        "validation_balanced_accuracy": row.get("balanced_accuracy"),
        "validation_f1": row.get("f1"),
        "validation_accuracy": row.get("accuracy"),
    }


def make_reference_baselines(test_rows: pd.DataFrame, train_rows: pd.DataFrame) -> pd.DataFrame:
    if test_rows.empty:
        raise ValueError("Cannot calculate reference baselines because test rows are empty.")

    target_column = next(
        (column for column in _TARGET_COLUMN_CANDIDATES if column in test_rows.columns),
        None,
    )
    if target_column is None or target_column not in train_rows.columns:
        raise ValueError("Train/test rows do not contain the same supported target column")
    y_test = test_rows[target_column].astype(int)
    train_counts = train_rows[target_column].astype(int).value_counts()
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
        if "is_correct" not in g.columns:
            if g["actual_class"].isna().any():
                raise ValueError("Cumulative performance requires known actual_class values")
            g["is_correct"] = (
                g["predicted_class"].astype(int) == g["actual_class"].astype(int)
            ).astype(int)
        g["daily_correctness"] = g["is_correct"].astype(int)
        g["cumulative_correct_predictions"] = g["daily_correctness"].cumsum()
        g["cumulative_incorrect_predictions"] = (1 - g["daily_correctness"]).cumsum()
        g["cumulative_accuracy"] = g["cumulative_correct_predictions"] / (np.arange(len(g)) + 1)
        g["rolling_5_trading_day_hit_rate"] = g["daily_correctness"].rolling(5, min_periods=1).mean()
        g["rolling_20_trading_day_hit_rate"] = g["daily_correctness"].rolling(20, min_periods=1).mean()
        frames.append(g)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
