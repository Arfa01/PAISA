"""Experimental training layer for PAISA.

This package is intentionally separate from the v0 baseline trainer so that
new model variants can be added without overwriting earlier PAISA artifacts.
"""

from paisa.experiments.experiment_config import MODEL_VARIANTS, RF_CONFIGS
from paisa.experiments.trainer import run_training_stage

__all__ = ["MODEL_VARIANTS", "RF_CONFIGS", "run_training_stage"]
