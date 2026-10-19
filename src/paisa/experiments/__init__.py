"""Experiment boundaries with lazy public imports.

Keeping package import side-effect free is important for the training CLI: the
portfolio report stack imports Matplotlib, which may print first-run cache
messages before a command emits its machine-readable JSON summary.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "ExperimentRun",
    "ModelArtifactRegistry",
    "ModelRegistry",
    "PortfolioRegistry",
    "RegisteredPortfolios",
    "run_experiment",
]


def __getattr__(name: str) -> Any:
    if name in {"ModelArtifactRegistry", "ModelRegistry"}:
        from paisa.experiments.registry import ModelArtifactRegistry, ModelRegistry

        return {
            "ModelArtifactRegistry": ModelArtifactRegistry,
            "ModelRegistry": ModelRegistry,
        }[name]
    if name in {"PortfolioRegistry", "RegisteredPortfolios"}:
        from paisa.experiments.portfolio_registry import (
            PortfolioRegistry,
            RegisteredPortfolios,
        )

        return {
            "PortfolioRegistry": PortfolioRegistry,
            "RegisteredPortfolios": RegisteredPortfolios,
        }[name]
    if name in {"ExperimentRun", "run_experiment"}:
        from paisa.experiments.runner import ExperimentRun, run_experiment

        return {"ExperimentRun": ExperimentRun, "run_experiment": run_experiment}[name]
    raise AttributeError(name)
