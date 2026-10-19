from __future__ import annotations

from paisa.experiments.experiment_config import MODEL_VARIANTS, RF_CONFIGS
from paisa.experiments.training_features import FEATURE_MONTH, FEATURE_SETS, validate_feature_sets


def test_variant_ids_are_unique_and_required_four_exist():
    ids = [variant.model_id for variant in MODEL_VARIANTS]
    assert ids == ["M-001", "M-002", "M-003", "M-004"]
    assert len(ids) == len(set(ids))
    assert [variant.algorithm for variant in MODEL_VARIANTS] == [
        "random_forest",
        "random_forest",
        "random_forest",
        "xgboost",
    ]


def test_feature_sets_are_valid_and_month_matches_current_paisa_features():
    validate_feature_sets()
    assert set(FEATURE_SETS["FEATURE_SHORT"].columns).issubset(set(FEATURE_MONTH))
    assert set(FEATURE_SETS["FEATURE_WEEK"].columns).issubset(set(FEATURE_MONTH))
    assert FEATURE_SETS["FEATURE_MONTH"].max_lookback_trading_days == 20
    assert {variant.feature_set_name for variant in MODEL_VARIANTS} == {"FEATURE_MONTH"}


def test_rf_configs_are_frozen_expected_configs():
    assert RF_CONFIGS["CFG-A"]["n_estimators"] == 200
    assert RF_CONFIGS["CFG-A"]["max_depth"] == 8
    assert RF_CONFIGS["CFG-B"]["max_depth"] == 4
    assert RF_CONFIGS["CFG-C"]["max_depth"] is None
