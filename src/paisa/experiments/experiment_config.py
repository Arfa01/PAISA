from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

RANDOM_SEED = 42
DEFAULT_SYMBOL = "OGDC"
DEFAULT_PROVIDER = "psx-dps"
DEFAULT_START = "2021-01-01"
SOURCE_FREQUENCY = "daily EOD"
TRAINING_MODE = "frozen"
TARGET_NAME = "target_next_session_up"
TARGET_DEFINITION = (
    "For a signal produced after trading day t has completed, the earliest "
    "future paper-trading execution is the open of trading day t+1. The target "
    "is therefore 1 when close(t+1) > open(t+1), else 0. Zero return is class 0."
)
PROBABILITY_STATUS = "uncalibrated_random_forest_probability"

RF_CONFIGS: dict[str, dict[str, Any]] = {
    "CFG-A": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "CFG-B": {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
    "CFG-C": {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "bootstrap": True,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
    },
}


@dataclass(frozen=True)
class ModelVariant:
    model_id: str
    label: str
    feature_set_name: str
    rf_config_name: str
    parent_model_id: str | None
    max_lookback_trading_days: int

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["algorithm"] = "RandomForestClassifier"
        data["training_mode"] = TRAINING_MODE
        data["probability_status"] = PROBABILITY_STATUS
        data["target_name"] = TARGET_NAME
        data["target_definition"] = TARGET_DEFINITION
        data["rf_params"] = RF_CONFIGS[self.rf_config_name]
        return data


MODEL_VARIANTS: list[ModelVariant] = [
    ModelVariant(
        model_id="M-001",
        label="RF | short context | CFG-A | frozen",
        feature_set_name="FEATURE_SHORT",
        rf_config_name="CFG-A",
        parent_model_id=None,
        max_lookback_trading_days=1,
    ),
    ModelVariant(
        model_id="M-002",
        label="RF | week context | CFG-A | frozen",
        feature_set_name="FEATURE_WEEK",
        rf_config_name="CFG-A",
        parent_model_id="M-001",
        max_lookback_trading_days=5,
    ),
    ModelVariant(
        model_id="M-003",
        label="RF | month context | CFG-A | frozen",
        feature_set_name="FEATURE_MONTH",
        rf_config_name="CFG-A",
        parent_model_id="M-002",
        max_lookback_trading_days=20,
    ),
    ModelVariant(
        model_id="M-004",
        label="RF | month context | CFG-B conservative | frozen",
        feature_set_name="FEATURE_MONTH",
        rf_config_name="CFG-B",
        parent_model_id="M-003",
        max_lookback_trading_days=20,
    ),
    ModelVariant(
        model_id="M-005",
        label="RF | month context | CFG-C flexible | frozen",
        feature_set_name="FEATURE_MONTH",
        rf_config_name="CFG-C",
        parent_model_id="M-003",
        max_lookback_trading_days=20,
    ),
]


def variants_as_dicts() -> list[dict[str, Any]]:
    return [variant.to_dict() for variant in MODEL_VARIANTS]


def get_variant(model_id: str) -> ModelVariant:
    for variant in MODEL_VARIANTS:
        if variant.model_id == model_id:
            return variant
    raise KeyError(f"Unknown model_id: {model_id}")
