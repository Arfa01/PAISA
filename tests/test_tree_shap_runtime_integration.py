"""Artifact-free integration checks for the certified Tree SHAP runtime."""

from __future__ import annotations

import importlib.metadata

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import paisa.explainability.tree_shap as tree_shap


shap = pytest.importorskip(
    "shap",
    reason="Tree SHAP runtime integration requires requirements-explainability.txt",
)

CERTIFIED_PACKAGE_VERSIONS = {
    "numpy": "2.3.5",
    "pandas": "2.3.3",
    "scikit-learn": "1.9.1",
    "xgboost": "3.4.1",
    "joblib": "1.6.0",
    "matplotlib": "3.11.2",
    "shap": "0.54.1",
}

FEATURES = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
        [0.0, 1.0, 0.0, 1.0, 0.50, -0.25],
        [1.0, 0.0, 1.0, 0.0, 1.00, 1.00],
        [1.0, 1.0, 1.0, 1.0, 1.50, 0.75],
        [0.2, 0.8, 0.1, 0.9, 0.60, -0.125],
        [0.8, 0.2, 0.9, 0.1, 0.90, 0.875],
        [0.3, 0.7, 0.2, 0.8, 0.65, 0.00],
        [0.7, 0.3, 0.8, 0.2, 0.85, 0.75],
    ],
    dtype=float,
)
LABELS = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)


def test_certified_explainability_package_versions_are_installed() -> None:
    observed = {
        distribution: importlib.metadata.version(distribution)
        for distribution in CERTIFIED_PACKAGE_VERSIONS
    }
    assert observed == CERTIFIED_PACKAGE_VERSIONS
    assert tree_shap.SHAP_PIN == CERTIFIED_PACKAGE_VERSIONS["shap"]


@pytest.mark.parametrize(
    ("algorithm", "estimator", "raw_shape", "expected_value_shape"),
    [
        (
            "random_forest",
            RandomForestClassifier(
                n_estimators=7,
                max_depth=3,
                random_state=314159,
                n_jobs=1,
            ),
            (8, 6, 2),
            (2,),
        ),
        (
            "xgboost",
            XGBClassifier(
                n_estimators=7,
                max_depth=3,
                learning_rate=0.2,
                random_state=314159,
                n_jobs=1,
                verbosity=0,
            ),
            (8, 6),
            (),
        ),
    ],
    ids=("random-forest", "xgboost"),
)
def test_real_tree_explainer_reconstructs_positive_class_probability(
    algorithm: str,
    estimator: RandomForestClassifier | XGBClassifier,
    raw_shape: tuple[int, ...],
    expected_value_shape: tuple[int, ...],
) -> None:
    estimator.fit(FEATURES, LABELS)
    np.testing.assert_array_equal(estimator.classes_, np.array([0, 1]))
    positive_index = tree_shap._class_index(estimator.classes_)

    probabilities = np.asarray(estimator.predict_proba(FEATURES), dtype=float)
    assert probabilities.shape == (len(FEATURES), 2)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        1.0,
        atol=tree_shap.PREDICTION_REPLAY_TOLERANCE,
        rtol=0,
    )

    explainer = shap.TreeExplainer(
        estimator,
        data=FEATURES,
        feature_perturbation="interventional",
        model_output="probability",
    )
    raw_values = explainer.shap_values(FEATURES, check_additivity=True)
    assert np.asarray(raw_values).shape == raw_shape
    assert np.asarray(explainer.expected_value).shape == expected_value_shape

    normalized = tree_shap.normalize_positive_class_shap(
        raw_values,
        row_count=len(FEATURES),
        feature_count=FEATURES.shape[1],
        class_index=positive_index,
    )
    expected_value = tree_shap.normalize_expected_value(
        explainer.expected_value,
        class_index=positive_index,
    )
    assert normalized.shape == FEATURES.shape
    assert np.isfinite(normalized).all()
    assert np.isfinite(expected_value)

    reconstructed = expected_value + normalized.sum(axis=1)
    tolerance = tree_shap.ADDITIVITY_TOLERANCES[algorithm]
    np.testing.assert_allclose(
        reconstructed,
        probabilities[:, positive_index],
        atol=tolerance,
        rtol=0,
    )
