"""Frozen model-variant definitions for the OGDC experiment pipeline.

``model_id`` values here identify the four requested variants. The artifact
registry may derive a run/data-specific resolved ID, but every manifest must
retain this stable variant ID and its complete configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


RANDOM_SEED = 42
DEFAULT_SYMBOL = "OGDC"
DEFAULT_PROVIDER = "psx-dps"
DEFAULT_START = "2021-01-01"
SOURCE_FREQUENCY = "daily"
TRAINING_MODE = "frozen"
FEATURE_SET_ID = "technical-daily-v1"
TARGET_NAME = "target_next_day_up"
TARGET_DEFINITION = (
    "target_next_day_up = 1 when close[target_date] > close[feature_date], "
    "otherwise 0; target_date is the next observed trading session."
)
TARGET_POSITIVE_CLASS_CONDITION = "close[target_date] > close[feature_date]"
PREDICTION_HORIZON = MappingProxyType(
    {"value": 1, "unit": "trading_session"}
)
PROBABILITY_STATUS = "uncalibrated_classifier_probability"


def _random_forest_parameters(**overrides: Any) -> dict[str, Any]:
    """Return a complete, explicit RandomForestClassifier configuration."""

    parameters: dict[str, Any] = {
        "bootstrap": True,
        "ccp_alpha": 0.0,
        "class_weight": "balanced_subsample",
        "criterion": "gini",
        "max_depth": 8,
        "max_features": "sqrt",
        "max_leaf_nodes": None,
        "max_samples": None,
        "min_impurity_decrease": 0.0,
        "min_samples_leaf": 4,
        "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0,
        "monotonic_cst": None,
        "n_estimators": 200,
        "n_jobs": -1,
        "oob_score": False,
        "random_state": RANDOM_SEED,
        "verbose": 0,
        "warm_start": False,
    }
    parameters.update(overrides)
    return parameters


RF_CONFIGS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        # Exact PAISA v0 baseline values where train_baseline_random_forest
        # explicitly overrides sklearn defaults; remaining defaults are made
        # explicit for a complete reproducibility record.
        "CFG-A": MappingProxyType(_random_forest_parameters()),
        "CFG-B": MappingProxyType(
            _random_forest_parameters(
                n_estimators=300,
                max_depth=4,
                min_samples_leaf=10,
                min_samples_split=10,
            )
        ),
        "CFG-C": MappingProxyType(
            _random_forest_parameters(
                n_estimators=400,
                max_depth=None,
                min_samples_leaf=2,
            )
        ),
    }
)

XGB_CONFIGS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "XGB-CPU": MappingProxyType(
            {
                "base_score": 0.5,
                "booster": "gbtree",
                "colsample_bylevel": 1.0,
                "colsample_bynode": 1.0,
                "colsample_bytree": 0.9,
                "enable_categorical": False,
                "eval_metric": "logloss",
                "gamma": 0.0,
                "grow_policy": "depthwise",
                "learning_rate": 0.05,
                "max_bin": 256,
                "max_delta_step": 0,
                "max_depth": 4,
                "max_leaves": 0,
                "min_child_weight": 1.0,
                "n_estimators": 300,
                "n_jobs": 1,
                "objective": "binary:logistic",
                "random_state": RANDOM_SEED,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "sampling_method": "uniform",
                "scale_pos_weight": 1.0,
                "subsample": 0.9,
                "tree_method": "hist",
                "validate_parameters": True,
                "verbosity": 0,
            }
        )
    }
)


@dataclass(frozen=True, slots=True)
class ModelVariant:
    """An immutable algorithm/configuration choice in stable training order."""

    model_id: str
    label: str
    algorithm: str
    feature_set_name: str
    configuration_name: str
    parent_model_id: str | None
    max_lookback_trading_days: int = 20
    random_seed: int = RANDOM_SEED

    def __post_init__(self) -> None:
        if not self.model_id or any(char.isspace() for char in self.model_id):
            raise ValueError("model_id must be a non-empty, whitespace-free identifier")
        if self.algorithm not in {"random_forest", "xgboost"}:
            raise ValueError("algorithm must be 'random_forest' or 'xgboost'")
        configurations = RF_CONFIGS if self.algorithm == "random_forest" else XGB_CONFIGS
        if self.configuration_name not in configurations:
            raise ValueError(
                f"Unknown {self.algorithm} configuration: {self.configuration_name}"
            )
        if self.model_configuration.get("random_state") != self.random_seed:
            raise ValueError("Configuration random_state must match variant random_seed")

    @property
    def model_configuration(self) -> Mapping[str, Any]:
        configurations = RF_CONFIGS if self.algorithm == "random_forest" else XGB_CONFIGS
        return configurations[self.configuration_name]

    @property
    def rf_config_name(self) -> str | None:
        """Compatibility alias used by the earlier RF-only trainer."""

        return self.configuration_name if self.algorithm == "random_forest" else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "label": self.label,
            "algorithm": self.algorithm,
            "feature_set_name": self.feature_set_name,
            "feature_set_id": FEATURE_SET_ID,
            "configuration_name": self.configuration_name,
            "model_configuration": deepcopy(dict(self.model_configuration)),
            "parent_model_id": self.parent_model_id,
            "max_lookback_trading_days": self.max_lookback_trading_days,
            "random_seed": self.random_seed,
            "training_mode": TRAINING_MODE,
            "probability_status": PROBABILITY_STATUS,
            "target_name": TARGET_NAME,
            "target_definition": TARGET_DEFINITION,
            "prediction_horizon": dict(PREDICTION_HORIZON),
        }


MODEL_VARIANTS: tuple[ModelVariant, ...] = (
    ModelVariant(
        model_id="M-001",
        label="Random Forest | PAISA v0 baseline configuration",
        algorithm="random_forest",
        feature_set_name="FEATURE_MONTH",
        configuration_name="CFG-A",
        parent_model_id=None,
    ),
    ModelVariant(
        model_id="M-002",
        label="Random Forest | conservative configuration",
        algorithm="random_forest",
        feature_set_name="FEATURE_MONTH",
        configuration_name="CFG-B",
        parent_model_id="M-001",
    ),
    ModelVariant(
        model_id="M-003",
        label="Random Forest | flexible configuration",
        algorithm="random_forest",
        feature_set_name="FEATURE_MONTH",
        configuration_name="CFG-C",
        parent_model_id="M-001",
    ),
    ModelVariant(
        model_id="M-004",
        label="XGBoost | CPU histogram configuration",
        algorithm="xgboost",
        feature_set_name="FEATURE_MONTH",
        configuration_name="XGB-CPU",
        parent_model_id="M-001",
    ),
)

_variants_by_id = {variant.model_id: variant for variant in MODEL_VARIANTS}
if len(_variants_by_id) != len(MODEL_VARIANTS):
    raise RuntimeError("Model variant IDs must be unique")
MODEL_VARIANTS_BY_ID: Mapping[str, ModelVariant] = MappingProxyType(_variants_by_id)


def variants_as_dicts() -> list[dict[str, Any]]:
    return [variant.to_dict() for variant in MODEL_VARIANTS]


def get_variant(model_id: str) -> ModelVariant:
    try:
        return MODEL_VARIANTS_BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model_id: {model_id}") from exc


def get_model_variant(model_id: str) -> ModelVariant:
    """Descriptive alias for :func:`get_variant`."""

    return get_variant(model_id)


def model_variants() -> tuple[ModelVariant, ...]:
    return MODEL_VARIANTS


__all__ = [
    "DEFAULT_PROVIDER",
    "DEFAULT_START",
    "DEFAULT_SYMBOL",
    "FEATURE_SET_ID",
    "MODEL_VARIANTS",
    "MODEL_VARIANTS_BY_ID",
    "ModelVariant",
    "PREDICTION_HORIZON",
    "PROBABILITY_STATUS",
    "RANDOM_SEED",
    "RF_CONFIGS",
    "SOURCE_FREQUENCY",
    "TARGET_DEFINITION",
    "TARGET_NAME",
    "TARGET_POSITIVE_CLASS_CONDITION",
    "TRAINING_MODE",
    "XGB_CONFIGS",
    "get_model_variant",
    "get_variant",
    "model_variants",
    "variants_as_dicts",
]
