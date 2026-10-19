from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from sklearn.ensemble import RandomForestClassifier

from paisa.experiments.experiment_config import (
    MODEL_VARIANTS,
    MODEL_VARIANTS_BY_ID,
    RF_CONFIGS,
    TARGET_POSITIVE_CLASS_CONDITION,
    get_model_variant,
)


def test_required_model_variants_are_unique_frozen_and_in_stable_order():
    assert [variant.model_id for variant in MODEL_VARIANTS] == [
        "M-001",
        "M-002",
        "M-003",
        "M-004",
    ]
    assert [variant.algorithm for variant in MODEL_VARIANTS] == [
        "random_forest",
        "random_forest",
        "random_forest",
        "xgboost",
    ]
    assert len(MODEL_VARIANTS_BY_ID) == len(MODEL_VARIANTS)
    assert {variant.feature_set_name for variant in MODEL_VARIANTS} == {
        "FEATURE_MONTH"
    }

    with pytest.raises(FrozenInstanceError):
        MODEL_VARIANTS[0].model_id = "replacement"  # type: ignore[misc]
    with pytest.raises(TypeError):
        MODEL_VARIANTS[0].model_configuration["n_estimators"] = 1  # type: ignore[index]


def test_m001_exactly_preserves_explicit_paisa_v0_baseline_settings():
    configuration = dict(get_model_variant("M-001").model_configuration)
    assert configuration["n_estimators"] == 200
    assert configuration["max_depth"] == 8
    assert configuration["min_samples_leaf"] == 4
    assert configuration["random_state"] == 42
    assert configuration["class_weight"] == "balanced_subsample"
    assert configuration["n_jobs"] == -1

    # Every sklearn parameter is persisted rather than relying on library
    # defaults that may change between artifact-producing environments.
    estimator_parameters = RandomForestClassifier(**configuration).get_params()
    assert set(configuration) == set(estimator_parameters)


def test_conservative_flexible_and_xgboost_configurations_are_distinct():
    baseline = get_model_variant("M-001")
    conservative = get_model_variant("M-002")
    flexible = get_model_variant("M-003")
    xgboost = get_model_variant("M-004")

    assert conservative.model_configuration["max_depth"] == 4
    assert conservative.model_configuration["min_samples_leaf"] == 10
    assert flexible.model_configuration["max_depth"] is None
    assert flexible.model_configuration["min_samples_leaf"] == 2
    assert len({
        tuple(sorted(dict(variant.model_configuration).items()))
        for variant in (baseline, conservative, flexible)
    }) == 3
    assert xgboost.algorithm == "xgboost"
    assert xgboost.model_configuration["objective"] == "binary:logistic"
    assert xgboost.model_configuration["tree_method"] == "hist"
    assert TARGET_POSITIVE_CLASS_CONDITION == (
        "close[target_date] > close[feature_date]"
    )


def test_serialized_configuration_is_detached_from_frozen_registry():
    serialized = get_model_variant("M-001").to_dict()
    serialized["model_configuration"]["n_estimators"] = 1
    assert RF_CONFIGS["CFG-A"]["n_estimators"] == 200
