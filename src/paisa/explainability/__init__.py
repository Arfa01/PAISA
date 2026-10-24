"""Portable model-explainability artifacts for PAISA research models."""

from .tree_shap import (
    ACADEMIC_DISCLAIMER,
    ADDITIVITY_TOLERANCES,
    EXPECTED_MODELS,
    MODEL_ORDER,
    SHAP_PIN,
    build_tree_shap_explainability,
    normalize_expected_value,
    normalize_positive_class_shap,
    verify_explainability_pack,
)

__all__ = [
    "ACADEMIC_DISCLAIMER",
    "ADDITIVITY_TOLERANCES",
    "EXPECTED_MODELS",
    "MODEL_ORDER",
    "SHAP_PIN",
    "build_tree_shap_explainability",
    "normalize_expected_value",
    "normalize_positive_class_shap",
    "verify_explainability_pack",
]
