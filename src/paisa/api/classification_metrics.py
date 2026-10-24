"""Pure reconciliation helpers for persisted binary classification metrics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


# Persisted producer values are full-precision JSON floats.  A small tolerance
# allows harmless serializer/platform round-off without accepting rounded or
# otherwise contradictory metric claims.
METRIC_REL_TOLERANCE = 1e-9
METRIC_ABS_TOLERANCE = 1e-9


def binary_metrics_from_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
) -> dict[str, float | None]:
    """Derive producer-compatible metrics from a ``labels=[0, 1]`` matrix.

    The PAISA producer uses binary averaging with positive label 1 and
    ``zero_division=0``.  It deliberately emits null balanced accuracy and MCC
    when the actual observations contain only one class.
    """

    true_negative, false_positive = confusion_matrix[0]
    false_negative, true_positive = confusion_matrix[1]
    support = true_negative + false_positive + false_negative + true_positive
    if support <= 0:
        raise ValueError("confusion matrix support must be positive")

    actual_negative = true_negative + false_positive
    actual_positive = false_negative + true_positive
    predicted_positive = false_positive + true_positive
    accuracy = (true_negative + true_positive) / support
    precision = (
        true_positive / predicted_positive if predicted_positive else 0.0
    )
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1_denominator = (2 * true_positive) + false_positive + false_negative
    f1 = (2 * true_positive) / f1_denominator if f1_denominator else 0.0

    if actual_negative and actual_positive:
        balanced_accuracy = (
            (true_negative / actual_negative) + (true_positive / actual_positive)
        ) / 2
        matthews_denominator = math.sqrt(
            (true_positive + false_positive)
            * (true_positive + false_negative)
            * (true_negative + false_positive)
            * (true_negative + false_negative)
        )
        matthews_corrcoef = (
            (
                (true_positive * true_negative)
                - (false_positive * false_negative)
            )
            / matthews_denominator
            if matthews_denominator
            else 0.0
        )
    else:
        balanced_accuracy = None
        matthews_corrcoef = None

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matthews_corrcoef": matthews_corrcoef,
    }


def validate_confusion_derived_metrics(
    metrics: Mapping[str, float | None],
    confusion_matrix: Sequence[Sequence[int]],
    *,
    field: str,
) -> None:
    """Reject reported classification metrics that contradict their matrix."""

    expected = binary_metrics_from_confusion_matrix(confusion_matrix)
    for metric_name, expected_value in expected.items():
        reported_value = metrics.get(metric_name)
        if expected_value is None:
            if reported_value is not None:
                raise ValueError(
                    f"{field}.{metric_name} must be null for single-class actuals"
                )
            continue
        if reported_value is None or not math.isclose(
            float(reported_value),
            expected_value,
            rel_tol=METRIC_REL_TOLERANCE,
            abs_tol=METRIC_ABS_TOLERANCE,
        ):
            raise ValueError(
                f"{field}.{metric_name} contradicts the confusion matrix"
            )
