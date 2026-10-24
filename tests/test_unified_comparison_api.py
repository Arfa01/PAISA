from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping
from urllib.parse import quote

import joblib
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import paisa.api.artifact_repository as artifact_repository_module
from paisa.api.artifact_repository import (
    ACADEMIC_DISCLAIMER,
    ArtifactIntegrityError,
    ArtifactRepository,
    InvalidIdentifierError,
    RepositorySettings,
    get_artifact_repository,
    settings_from_environment,
)
from paisa.api.classification_metrics import binary_metrics_from_confusion_matrix
from paisa.api.main import app
from paisa.api.schemas import (
    UnifiedBenchmarkPortfolioMetric,
    UnifiedConfusionMatrixRow,
    UnifiedModelPortfolioMetric,
    UnifiedPredictionMetricRow,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_ID = "CMP-api-unified-fixture"
MODEL_IDS = tuple(f"M-{index:03d}" for index in range(1, 7))
REFERENCE_BASELINES = ("majority-class-baseline", "previous-direction-baseline")
FIXTURE_SOURCE_AUTHORITY = {
    "baseline_training": {
        "run_id": "TRN-api-fixture",
        "manifest_sha256": (
            "b0a75925b804b60c17aa5d9dcac6c86d2ad14dad986a39d1ad068d9a02bdbf44"
        ),
    },
    "baseline_vertical_slice": {
        "run_id": "VSL-api-fixture",
        "manifest_sha256": "2" * 64,
        "portable_certificate_sha256": (
            "6d79bd8980f0dfe756f9a5ab79c81a0cea68e6eb589d3f71847e521da0ab9920"
        ),
    },
    "lstm": {
        "run_id": "LSTM-api-fixture",
        "manifest_sha256": (
            "b4978b8d73d97e20158bb1825c5ef11467b2c7905675996c77852a037e4d2713"
        ),
    },
}
PORTFOLIO_FILES = (
    "config.json",
    "signals.csv",
    "portfolio_snapshots.csv",
    "trades.csv",
    "benchmarks/cash_only_snapshots.csv",
    "benchmarks/buy_and_hold_snapshots.csv",
    "benchmarks/buy_and_hold_trades.csv",
)
AUTHORED_FILES = {
    "comparison_contract.json",
    "date_alignment.json",
    "derived/reference_baseline_predictions.csv",
    "final_summary.json",
    "metrics.json",
    "rankings.json",
    "report.md",
    "source_roots.json",
}
EVIDENCE_FILES = {
    "evidence/market_data/baseline_psx_prices.csv",
    "evidence/market_data/lstm_ogdc_prices.csv",
    "evidence/source_manifests/baseline_training_run_manifest.json",
    "evidence/source_manifests/baseline_vertical_manifest_certificate.json",
    "evidence/source_manifests/lstm_artifact_manifest.json",
    "evidence/source_manifests/lstm_run_manifest.json",
    "evidence/splits/baseline_split_manifest.json",
    "evidence/splits/lstm_split_manifest.json",
    *(f"evidence/predictions/{model}/predictions_all_splits.csv" for model in MODEL_IDS),
    *(
        f"evidence/portfolios/{model}/{relative}"
        for model in MODEL_IDS
        for relative in PORTFOLIO_FILES
    ),
}
EXPECTED_FILES = AUTHORED_FILES | EVIDENCE_FILES


def _write_json(path: Path, payload: object, *, allow_nan: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=allow_nan) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _registered_model_id(model_id: str) -> str:
    return f"{model_id}-api-fixture-registered-model"


def _algorithm(model_id: str) -> str:
    if model_id == "M-004":
        return "xgboost"
    if model_id in {"M-005", "M-006"}:
        return "lstm"
    return "random_forest"


def _label(model_id: str) -> str:
    return {
        "M-001": "PAISA v0 baseline",
        "M-002": "conservative",
        "M-003": "flexible",
        "M-004": "CPU histogram",
        "M-005": "baseline recurrent",
        "M-006": "regularized recurrent",
    }[model_id]


def _prediction_metric(model_id: str, scope: str, index: int) -> dict[str, Any]:
    row_count = 2
    confusion_matrix = {
        1: [[0, 1], [1, 0]],
        2: [[1, 0], [1, 0]],
        3: [[0, 1], [1, 0]],
        4: [[0, 1], [0, 1]],
        5: [[1, 0], [1, 0]],
        6: [[1, 0], [0, 1]],
    }[index]
    classification = binary_metrics_from_confusion_matrix(confusion_matrix)
    predicted_class_counts = {
        "0": confusion_matrix[0][0] + confusion_matrix[1][0],
        "1": confusion_matrix[0][1] + confusion_matrix[1][1],
    }
    return {
        "competitor_id": model_id,
        "competitor_type": "model",
        "registered_model_id": _registered_model_id(model_id),
        "algorithm": _algorithm(model_id),
        "label": _label(model_id),
        "evaluation_scope": scope,
        "row_count": row_count,
        "support": row_count,
        "class_counts": {"0": 1, "1": 1},
        "class_distribution": {"0": 1, "1": 1},
        "complete_class_coverage": True,
        "observed_classes": [0, 1],
        "predicted_class_counts": predicted_class_counts,
        "predicted_class_distribution": predicted_class_counts,
        "selection_eligible": True,
        "average": "binary",
        "positive_label": 1,
        "zero_division": 0,
        "confusion_matrix": confusion_matrix,
        **classification,
        "roc_auc": 0.75,
        "brier_score": 0.2,
        "log_loss": 0.6,
        "expected_calibration_error": 0.1,
        "calibration": {
            "method": "equal_width",
            "n_bins": 1,
            "row_count": row_count,
            "expected_calibration_error": 0.1,
            "bins": [],
        },
        "metric_warnings": [],
    }


def _baseline_metric(baseline_id: str, scope: str, index: int) -> dict[str, Any]:
    row = _prediction_metric("M-001", scope, index)
    confusion_matrix = (
        [[1, 0], [1, 0]]
        if baseline_id == "majority-class-baseline"
        else [[0, 1], [1, 0]]
    )
    predicted_class_counts = {
        "0": confusion_matrix[0][0] + confusion_matrix[1][0],
        "1": confusion_matrix[0][1] + confusion_matrix[1][1],
    }
    row.update(
        {
            "competitor_id": baseline_id,
            "competitor_type": "reference_baseline",
            "registered_model_id": None,
            "algorithm": None,
            "label": None,
            "selection_eligible": True,
            "confusion_matrix": confusion_matrix,
            "predicted_class_counts": predicted_class_counts,
            "predicted_class_distribution": predicted_class_counts,
            **binary_metrics_from_confusion_matrix(confusion_matrix),
        }
    )
    return row


def _portfolio_metric(model_id: str, index: int) -> dict[str, Any]:
    total_return = index / 100
    return {
        "competitor_id": model_id,
        "competitor_type": "model",
        "registered_model_id": _registered_model_id(model_id),
        "experiment_id": f"EXP-{model_id}",
        "portfolio_id": f"PORT-{model_id}",
        "initial_equity": "100000.00",
        "ending_equity": f"{100000 + index * 1000:.2f}",
        "total_return": total_return,
        "maximum_drawdown": 0.08 - index / 200,
        "volatility": 0.1 + index / 100,
        "sharpe_ratio": None if model_id == "M-001" else 0.2 + index / 10,
        "number_of_trades": index,
        "win_rate": None if model_id == "M-001" else 0.5,
        "average_gain": None,
        "average_loss": None,
        "turnover": 0.1 * index,
        "fees_paid": f"{index * 10:.2f}",
        "time_in_market": 0.15 * index,
        "excess_return_vs_cash_only": total_return,
        "excess_return_vs_buy_and_hold": total_return - 0.07,
    }


def _benchmark_metric(benchmark_id: str) -> dict[str, Any]:
    buy_and_hold = benchmark_id == "buy-and-hold"
    return {
        "benchmark_id": benchmark_id,
        "competitor_type": "portfolio_benchmark",
        "initial_equity": "100000.00",
        "ending_equity": "107000.00" if buy_and_hold else "100000.00",
        "total_return": 0.07 if buy_and_hold else 0.0,
        "maximum_drawdown": 0.12 if buy_and_hold else 0.0,
        "volatility": 0.18 if buy_and_hold else 0.0,
        "sharpe_ratio": 0.55 if buy_and_hold else None,
        "number_of_trades": 1 if buy_and_hold else 0,
        "win_rate": None,
        "average_gain": None,
        "average_loss": None,
        "turnover": 0.48 if buy_and_hold else 0.0,
        "fees_paid": "50.00" if buy_and_hold else "0.00",
        "time_in_market": 1.0 if buy_and_hold else 0.0,
        "ledger_provenance": [
            {
                "variant_id": model_id,
                "registered_model_id": _registered_model_id(model_id),
                "experiment_id": f"EXP-{model_id}",
                "portfolio_id": f"PORT-{model_id}-{benchmark_id}",
                "config_path": f"evidence/portfolios/{model_id}/config.json",
                "snapshots_path": (
                    f"evidence/portfolios/{model_id}/benchmarks/"
                    f"{'buy_and_hold' if buy_and_hold else 'cash_only'}_snapshots.csv"
                ),
                "trades_path": (
                    f"evidence/portfolios/{model_id}/benchmarks/"
                    "buy_and_hold_trades.csv"
                    if buy_and_hold
                    else None
                ),
            }
            for model_id in MODEL_IDS
        ],
    }


def _prediction_rankings(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    tie_breakers = (
        ("balanced_accuracy", "f1", "accuracy")
        if metric == "balanced_accuracy"
        else ("f1", "balanced_accuracy", "accuracy")
    )
    ordered = sorted(
        rows,
        key=lambda row: (
            *(-float(row[key]) for key in tie_breakers),
            row["registered_model_id"] or row["competitor_id"],
        ),
    )
    return [
        {
            "rank": rank,
            "competitor_id": row["competitor_id"],
            "registered_model_id": row["registered_model_id"],
            "balanced_accuracy": row["balanced_accuracy"],
            "f1": row["f1"],
            "accuracy": row["accuracy"],
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def _portfolio_rankings(
    rows: list[dict[str, Any]], metric: str, *, reverse: bool
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row[metric]) if reverse else float(row[metric]),
            row["competitor_id"],
        ),
    )
    ranked: list[dict[str, Any]] = []
    for rank, row in enumerate(ordered, start=1):
        result = {
            "rank": rank,
            "competitor_id": row["competitor_id"],
            "registered_model_id": row["registered_model_id"],
        }
        if metric == "total_return":
            result.update(
                {
                    "total_return": row["total_return"],
                    "sharpe_ratio": row["sharpe_ratio"],
                    "maximum_drawdown": row["maximum_drawdown"],
                }
            )
        elif metric == "maximum_drawdown":
            result.update(
                {
                    "maximum_drawdown": row["maximum_drawdown"],
                    "total_return": row["total_return"],
                }
            )
        elif metric == "fees_paid":
            result.update(
                {
                    "fees_paid": row["fees_paid"],
                    "number_of_trades": row["number_of_trades"],
                }
            )
        else:  # pragma: no cover - fixture authoring guard
            raise KeyError(metric)
        ranked.append(result)
    return ranked


def _fixture_payloads() -> dict[str, object]:
    validation = [
        _prediction_metric(model_id, "validation", index)
        for index, model_id in enumerate(MODEL_IDS, start=1)
    ]
    holdout = [
        _prediction_metric(model_id, "shared_project_holdout", index)
        for index, model_id in enumerate(MODEL_IDS, start=1)
    ]
    portfolios = [
        _portfolio_metric(model_id, index)
        for index, model_id in enumerate(MODEL_IDS, start=1)
    ]
    validation_baselines = [
        _baseline_metric(baseline_id, "validation", index)
        for index, baseline_id in enumerate(REFERENCE_BASELINES, start=1)
    ]
    holdout_baselines = [
        _baseline_metric(baseline_id, "shared_project_holdout", index)
        for index, baseline_id in enumerate(REFERENCE_BASELINES, start=1)
    ]
    benchmarks = [_benchmark_metric(name) for name in ("cash-only", "buy-and-hold")]
    rankings = {
        "schema_version": "1.0",
        "comparison_id": COMPARISON_ID,
        "validation_model_selection": _prediction_rankings(
            validation, "balanced_accuracy"
        ),
        "lstm_validation_preference": _prediction_rankings(
            validation[-2:], "balanced_accuracy"
        ),
        "shared_project_holdout_balanced_accuracy": _prediction_rankings(
            holdout, "balanced_accuracy"
        ),
        "shared_project_holdout_f1": _prediction_rankings(holdout, "f1"),
        "simulated_model_portfolio_return": _portfolio_rankings(
            portfolios, "total_return", reverse=True
        ),
        "simulated_model_portfolio_drawdown": _portfolio_rankings(
            portfolios, "maximum_drawdown", reverse=False
        ),
        "simulated_model_portfolio_fees": _portfolio_rankings(
            portfolios, "fees_paid", reverse=False
        ),
    }
    validation_common_keys = [
        {
            "ticker": "OGDC",
            "feature_date": "2025-12-29",
            "target_date": "2025-12-30",
            "actual_class": 1,
        },
        {
            "ticker": "OGDC",
            "feature_date": "2025-12-30",
            "target_date": "2025-12-31",
            "actual_class": 0,
        },
    ]
    holdout_common_keys = [
        {
            "ticker": "OGDC",
            "feature_date": "2026-01-02",
            "target_date": "2026-01-05",
            "actual_class": 1,
        },
        {
            "ticker": "OGDC",
            "feature_date": "2026-01-05",
            "target_date": "2026-01-06",
            "actual_class": 0,
        },
    ]

    def alignment_scope(terminology: str) -> dict[str, Any]:
        common_keys = (
            holdout_common_keys
            if terminology == "shared project holdout"
            else validation_common_keys
        )
        result: dict[str, Any] = {
            "terminology": terminology,
            "common_count": 2,
            "feature_start": common_keys[0]["feature_date"],
            "feature_end": common_keys[-1]["feature_date"],
            "target_start": common_keys[0]["target_date"],
            "target_end": common_keys[-1]["target_date"],
            "original_counts": dict.fromkeys(MODEL_IDS, 2),
            "common_counts": dict.fromkeys(MODEL_IDS, 2),
            "dropped_target_dates": {model_id: [] for model_id in MODEL_IDS},
            "common_keys": common_keys,
        }
        if terminology == "shared project holdout":
            result["previously_exposed_through"] = list(MODEL_IDS[:4])
        return result

    best_portfolio = portfolios[-1]
    lowest_drawdown = portfolios[-1]
    lowest_fees = portfolios[0]
    prediction_outcome_keys = (
        "registered_model_id",
        "accuracy",
        "balanced_accuracy",
        "f1",
    )

    def prediction_outcome(row: dict[str, Any], *, criterion: str, scope: str) -> dict[str, Any]:
        return {
            "scope": scope,
            "variant_id": row["competitor_id"],
            **{key: row[key] for key in prediction_outcome_keys},
            "criterion": criterion,
        }

    def portfolio_outcome(row: dict[str, Any], *, criterion: str) -> dict[str, Any]:
        return {
            "variant_id": row["competitor_id"],
            "registered_model_id": row["registered_model_id"],
            "portfolio_id": row["portfolio_id"],
            "ending_equity": row["ending_equity"],
            "total_return": row["total_return"],
            "maximum_drawdown": row["maximum_drawdown"],
            "fees_paid": row["fees_paid"],
            "criterion": criterion,
        }

    summary = {
        "schema_version": "1.0",
        "artifact_type": "paisa_unified_price_model_comparison",
        "comparison_id": COMPARISON_ID,
        "status": "PASS",
        "symbol": "OGDC",
        "path_base": "comparison_root",
        "holdout_context": {
            "term": "shared project holdout",
            "common_count": 2,
            "target_start": "2026-01-05",
            "target_end": "2026-01-06",
            "previously_exposed_through": list(MODEL_IDS[:4]),
        },
        "outcomes_are_not_combined": True,
        "reported_outcomes": {
            "validation_selected_model": prediction_outcome(
                validation[-1],
                criterion="validation balanced accuracy, then F1, accuracy, and model ID",
                scope="all_six_models",
            ),
            "lstm_validation_preferred_model": prediction_outcome(
                validation[-1],
                criterion="validation balanced accuracy, then F1, accuracy, and model ID",
                scope="M-005_and_M-006_only",
            ),
            "best_shared_holdout_balanced_accuracy": prediction_outcome(
                holdout[-1],
                criterion="highest shared project holdout balanced accuracy",
                scope="shared project holdout",
            ),
            "best_shared_holdout_f1": prediction_outcome(
                holdout[-1],
                criterion="highest shared project holdout F1",
                scope="shared project holdout",
            ),
            "best_simulated_model_portfolio": portfolio_outcome(
                best_portfolio, criterion="highest simulated total return"
            ),
            "lowest_drawdown_model_portfolio": portfolio_outcome(
                lowest_drawdown, criterion="lowest simulated maximum drawdown"
            ),
            "lowest_fee_model_portfolio": portfolio_outcome(
                lowest_fees, criterion="lowest simulated fees paid"
            ),
            "cash_only_benchmark": {
                key: benchmarks[0][key]
                for key in (
                    "initial_equity",
                    "ending_equity",
                    "total_return",
                    "maximum_drawdown",
                    "volatility",
                    "sharpe_ratio",
                    "number_of_trades",
                    "fees_paid",
                )
            },
            "buy_and_hold_result": {
                key: benchmarks[1][key]
                for key in (
                    "initial_equity",
                    "ending_equity",
                    "total_return",
                    "maximum_drawdown",
                    "volatility",
                    "sharpe_ratio",
                    "number_of_trades",
                    "fees_paid",
                )
            },
            "any_model_beat_buy_and_hold": {
                "value": False,
                "comparison_metric": "simulated total return",
                "best_model_variant_id": best_portfolio["competitor_id"],
                "best_model_minus_buy_and_hold": -0.01,
            },
        },
        "artifact_paths": {
            "source_roots": "source_roots.json",
            "comparison_contract": "comparison_contract.json",
            "date_alignment": "date_alignment.json",
            "metrics": "metrics.json",
            "rankings": "rankings.json",
            "reference_baseline_predictions": "derived/reference_baseline_predictions.csv",
            "human_report": "report.md",
        },
        "limitations": [
            "Better classification performance did not necessarily produce better portfolio performance."
        ],
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    contract = {
        "schema_version": "1.0",
        "comparison_id": COMPARISON_ID,
        "symbol": "OGDC",
        "source_frequency": "daily",
        "models": list(MODEL_IDS),
        "reference_baselines": list(REFERENCE_BASELINES),
        "portfolio_benchmarks": ["cash-only", "buy-and-hold"],
        "target_definition": {
            "name": "target_next_day_up",
            "positive_class_condition": "close[target_date] > close[feature_date]",
            "prediction_horizon": {"value": 1, "unit": "trading_session"},
        },
        "alignment_key": ["ticker", "feature_date", "target_date"],
        "evaluation_scopes": {
            "validation": {"target_dates": ["2025-12-30", "2025-12-31"]},
            "shared_project_holdout": {
                "term": "shared project holdout",
                "target_dates": ["2026-01-05", "2026-01-06"],
                "previously_exposed_through": list(MODEL_IDS[:4]),
            },
        },
        "selection_policy": {
            "primary": "balanced_accuracy",
            "tie_breakers": ["f1", "accuracy", "model_id_ascending"],
            "test_metrics_used_for_selection": False,
        },
        "reference_baseline_policy": {
            "majority_class": "training-only majority",
            "previous_direction": "previous realized direction",
        },
        "portfolio_assumptions": {
            "symbol": "OGDC",
            "initial_cash": "100000.00",
            "thresholds": {
                "buy_probability_up_gte": 0.6,
                "sell_probability_up_lte": 0.4,
            },
            "maximum_allocation": 0.5,
            "transaction_cost": {"type": "proportional_bps", "value": 10.0},
            "evaluation_dates": {
                "feature_start": "2026-01-02",
                "feature_end": "2026-01-05",
                "target_start": "2026-01-05",
                "target_end": "2026-01-06",
            },
            "execution_policy": {
                "signal_timing": "after_close",
                "execution_session": "next_trading_session",
                "reference_price_field": "open",
                "slippage_bps": 0.0,
                "allocation_mode": "state_transition_to_maximum",
                "whole_share_rounding": "floor",
                "valuation_price_field": "close",
                "ending_position": "mark_to_market",
            },
            "metric_assumptions": {
                "annualization_factor": 252,
                "risk_free_rate_annual": 0.0,
            },
        },
        "metric_recomputation": {
            "prediction_source": "persisted prediction rows",
            "portfolio_source": "persisted ledgers",
            "portfolio_semantic_replay": True,
            "models_retrained": False,
            "backtests_rerun": False,
        },
        "presentation": {"format": "tabular_report_only", "plots_included": False},
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    metrics = {
        "schema_version": "1.0",
        "comparison_id": COMPARISON_ID,
        "validation_model_metrics": validation,
        "validation_reference_baseline_metrics": validation_baselines,
        "shared_project_holdout_prediction_metrics": [*holdout, *holdout_baselines],
        "model_portfolio_metrics": portfolios,
        "benchmark_portfolio_metrics": benchmarks,
        "reference_baseline_training": {
            "training_row_count": 4,
            "class_counts": {"0": 2, "1": 2},
            "majority_class": 0,
        },
    }
    date_alignment = {
        "schema_version": "1.0",
        "comparison_id": COMPARISON_ID,
        "alignment_key": ["ticker", "feature_date", "target_date"],
        "validation": alignment_scope("validation"),
        "shared_project_holdout": alignment_scope("shared project holdout"),
    }
    return {
        "comparison_contract.json": contract,
        "date_alignment.json": date_alignment,
        "final_summary.json": summary,
        "metrics.json": metrics,
        "rankings.json": rankings,
    }


def _reseal_pack(run_directory: Path) -> None:
    files = sorted(
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    )
    manifest = {
        "schema_version": "1.0",
        "artifact_type": "paisa_unified_price_model_comparison",
        "comparison_id": run_directory.name,
        "status": "PASS",
        "inventory": {
            "algorithm": "sha256",
            "root": ".",
            "excludes": ["manifest.json"],
            "files": [
                {"path": relative, "sha256": _sha256(run_directory / relative)}
                for relative in files
            ],
        },
        "source_roots": "source_roots.json",
        "final_summary": "final_summary.json",
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    _write_json(run_directory / "manifest.json", manifest)


def _make_comparison_pack(root: Path) -> Path:
    run_directory = root / COMPARISON_ID
    payloads = _fixture_payloads()
    for relative in EXPECTED_FILES:
        path = run_directory / relative
        if relative in payloads:
            _write_json(path, payloads[relative])
        elif relative == "source_roots.json":
            continue
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                _write_json(path, {"schema_version": "1.0", "fixture": relative})
            else:
                path.write_text("fixture\n", encoding="utf-8")

    training_manifest_relative = (
        "evidence/source_manifests/baseline_training_run_manifest.json"
    )
    vertical_certificate_relative = (
        "evidence/source_manifests/baseline_vertical_manifest_certificate.json"
    )
    lstm_manifest_relative = "evidence/source_manifests/lstm_run_manifest.json"
    lstm_artifact_manifest_relative = (
        "evidence/source_manifests/lstm_artifact_manifest.json"
    )

    def binding(relative: str) -> tuple[str, str, str]:
        fixed = {
            training_manifest_relative: (
                "baseline_training",
                "run_manifest.json",
                "source_integrity_root",
            ),
            lstm_manifest_relative: (
                "lstm",
                "run_manifest.json",
                "source_integrity_root",
            ),
            lstm_artifact_manifest_relative: (
                "lstm",
                "artifact_manifest.json",
                "source_payload_inventory",
            ),
            "evidence/splits/baseline_split_manifest.json": (
                "baseline_training",
                "split_manifest.json",
                "chronological_split_membership",
            ),
            "evidence/splits/lstm_split_manifest.json": (
                "lstm",
                "split_manifest.json",
                "lstm_sequence_split_membership",
            ),
            "evidence/market_data/baseline_psx_prices.csv": (
                "baseline_vertical_slice",
                "input_snapshot/psx_prices.csv",
                "portfolio_market_observations",
            ),
            "evidence/market_data/lstm_ogdc_prices.csv": (
                "lstm",
                "frozen_input/OGDC_prices.csv",
                "portfolio_market_observations",
            ),
        }
        if relative in fixed:
            return fixed[relative]
        parts = Path(relative).parts
        if parts[:2] == ("evidence", "predictions"):
            model_id = parts[2]
            role = "baseline_training" if model_id in MODEL_IDS[:4] else "lstm"
            return (
                role,
                f"models/{_registered_model_id(model_id)}/predictions_all_splits.csv",
                "model_prediction_rows",
            )
        if parts[:2] == ("evidence", "portfolios"):
            model_id = parts[2]
            suffix = "/".join(parts[3:])
            if model_id in MODEL_IDS[:4]:
                role = "baseline_vertical_slice"
                prefix = f"experiments/VSL-api-fixture-{model_id.lower()}"
            else:
                role = "lstm"
                prefix = f"backtests/LSTM-api-fixture-{model_id.lower()}"
            return (
                role,
                f"{prefix}/{suffix}",
                "portfolio_configuration"
                if suffix == "config.json"
                else "portfolio_ledger",
            )
        raise AssertionError(f"Missing fixture evidence binding for {relative}")

    def source_inventory(
        files: dict[str, str], *, excludes: list[str]
    ) -> dict[str, object]:
        return {
            "algorithm": "sha256",
            "root": ".",
            "excludes": excludes,
            "files": [
                {"path": path, "sha256": digest}
                for path, digest in sorted(files.items())
            ],
        }

    source_files: dict[str, dict[str, str]] = {
        "baseline_training": {},
        "baseline_vertical_slice": {},
        "lstm": {},
    }
    copied_root_files = {
        training_manifest_relative,
        vertical_certificate_relative,
        lstm_manifest_relative,
        lstm_artifact_manifest_relative,
    }
    for relative in sorted(EVIDENCE_FILES - copied_root_files):
        role, source_path, _purpose = binding(relative)
        source_files[role][source_path] = _sha256(run_directory / relative)

    lstm_artifact_manifest_path = run_directory / lstm_artifact_manifest_relative
    _write_json(
        lstm_artifact_manifest_path,
        source_inventory(
            source_files["lstm"],
            excludes=["artifact_manifest.json", "run_manifest.json"],
        ),
    )
    source_files["lstm"]["artifact_manifest.json"] = _sha256(
        lstm_artifact_manifest_path
    )

    training_manifest_path = run_directory / training_manifest_relative
    _write_json(
        training_manifest_path,
        {
            "schema_version": "1.0",
            "run_id": "TRN-api-fixture",
            "artifact_inventory": source_inventory(
                source_files["baseline_training"],
                excludes=["run_manifest.json"],
            ),
            "models": [
                {
                    "variant_id": model_id,
                    "model_id": _registered_model_id(model_id),
                }
                for model_id in MODEL_IDS[:4]
            ],
        },
    )
    training_source_digest = _sha256(training_manifest_path)
    vertical_source_digest = "2" * 64
    vertical_certificate_path = run_directory / vertical_certificate_relative
    _write_json(
        vertical_certificate_path,
        {
            "schema_version": "1.0",
            "artifact_type": "paisa_source_integrity_certificate",
            "run_id": "VSL-api-fixture",
            "source_role": "baseline_vertical_slice",
            "source_manifest": {
                "name": "manifest.json",
                "sha256": vertical_source_digest,
            },
            "projection_policy": "allowlisted_fields_v1",
            "raw_manifest_included": False,
            "source_assertions": {
                "schema_version": "1.0",
                "status": "PASS",
                "symbol": "OGDC",
                "model_training_run_id": "TRN-api-fixture",
                "registered_model_ids": {
                    model_id: _registered_model_id(model_id)
                    for model_id in MODEL_IDS[:4]
                },
                "disclaimer": ACADEMIC_DISCLAIMER,
            },
            "artifact_inventory": source_inventory(
                source_files["baseline_vertical_slice"],
                excludes=["manifest.json"],
            ),
            "disclaimer": ACADEMIC_DISCLAIMER,
        },
    )
    lstm_manifest_path = run_directory / lstm_manifest_relative
    _write_json(
        lstm_manifest_path,
        {
            "schema_version": "1.0",
            "run_id": "LSTM-api-fixture",
            "artifact_inventory": source_inventory(
                source_files["lstm"], excludes=["run_manifest.json"]
            ),
            "model_ids": {
                model_id: _registered_model_id(model_id)
                for model_id in MODEL_IDS[4:]
            },
            "baseline_training_run": {
                "run_id": "TRN-api-fixture",
                "manifest_sha256": training_source_digest,
            },
            "baseline_vertical_slice": {
                "run_id": "VSL-api-fixture",
                "manifest_sha256": vertical_source_digest,
            },
        },
    )

    source_maps = {
        "baseline_training": {
            "run_manifest.json": training_source_digest,
            **source_files["baseline_training"],
        },
        "baseline_vertical_slice": {
            "manifest.json": vertical_source_digest,
            **source_files["baseline_vertical_slice"],
        },
        "lstm": {
            "run_manifest.json": _sha256(lstm_manifest_path),
            **source_files["lstm"],
        },
    }
    evidence_records = []
    for relative in sorted(EVIDENCE_FILES - {vertical_certificate_relative}):
        role, source_path, purpose = binding(relative)
        digest = _sha256(run_directory / relative)
        assert source_maps[role][source_path] == digest
        evidence_records.append(
            {
                "source_role": role,
                "source_path": source_path,
                "pack_path": relative,
                "sha256": digest,
                "purpose": purpose,
                "derivation": "byte_copy",
            }
        )
    source_roots = {
        "schema_version": "1.0",
        "comparison_id": COMPARISON_ID,
        "sources": [
            {
                "role": "baseline_training",
                "artifact_type": "model_training_run",
                "run_id": "TRN-api-fixture",
                "integrity_root_name": "run_manifest.json",
                "integrity_root_sha256": training_source_digest,
                "payload_file_count": len(source_files["baseline_training"]),
                "integrity_root_pack_path": (
                    "evidence/source_manifests/baseline_training_run_manifest.json"
                ),
                "integrity_root_pack_sha256": _sha256(
                    run_directory
                    / "evidence/source_manifests/baseline_training_run_manifest.json"
                ),
                "integrity_root_derivation": "byte_copy",
            },
            {
                "role": "baseline_vertical_slice",
                "artifact_type": "vertical_slice",
                "run_id": "VSL-api-fixture",
                "integrity_root_name": "manifest.json",
                "integrity_root_sha256": vertical_source_digest,
                "payload_file_count": len(source_files["baseline_vertical_slice"]),
                "integrity_root_pack_path": (
                    "evidence/source_manifests/baseline_vertical_manifest_certificate.json"
                ),
                "integrity_root_pack_sha256": _sha256(
                    run_directory
                    / "evidence/source_manifests/baseline_vertical_manifest_certificate.json"
                ),
                "integrity_root_derivation": "deterministic_redacted_projection",
            },
            {
                "role": "lstm",
                "artifact_type": "lstm_price_only_experiment",
                "run_id": "LSTM-api-fixture",
                "integrity_root_name": "run_manifest.json",
                "integrity_root_sha256": _sha256(
                    run_directory / "evidence/source_manifests/lstm_run_manifest.json"
                ),
                "payload_file_count": len(source_files["lstm"]),
                "integrity_root_pack_path": "evidence/source_manifests/lstm_run_manifest.json",
                "integrity_root_pack_sha256": _sha256(
                    run_directory / "evidence/source_manifests/lstm_run_manifest.json"
                ),
                "integrity_root_derivation": "byte_copy",
            },
        ],
        "relationships": {
            "vertical_slice_uses_training_run": True,
            "lstm_uses_training_run": True,
            "lstm_uses_vertical_slice": True,
        },
        "evidence_files": evidence_records,
        "source_revalidation": {
            "validated_before_generation": True,
            "revalidated_before_publication": True,
        },
    }
    observed_authority = {
        source["role"]: {
            "run_id": source["run_id"],
            "manifest_sha256": source["integrity_root_sha256"],
            **(
                {
                    "portable_certificate_sha256": source[
                        "integrity_root_pack_sha256"
                    ]
                }
                if source["role"] == "baseline_vertical_slice"
                else {}
            ),
        }
        for source in source_roots["sources"]
    }
    assert observed_authority == FIXTURE_SOURCE_AUTHORITY
    _write_json(run_directory / "source_roots.json", source_roots)
    _reseal_pack(run_directory)
    assert {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
    } == EXPECTED_FILES | {"manifest.json"}
    return run_directory


def _settings(
    tmp_path: Path,
    comparison_root: Path,
    *,
    default_id: str | None = COMPARISON_ID,
    authorize_fixture: bool = True,
    manifest_authority: Mapping[str, str] | None = None,
) -> RepositorySettings:
    vertical_root = tmp_path / "vertical"
    training_root = tmp_path / "training"
    vertical_root.mkdir(exist_ok=True)
    training_root.mkdir(exist_ok=True)
    source_authority = (
        {
            role: dict(identity)
            for role, identity in FIXTURE_SOURCE_AUTHORITY.items()
        }
        if authorize_fixture
        else None
    )
    effective_manifest_authority = (
        dict(manifest_authority)
        if manifest_authority is not None
        else {
            COMPARISON_ID: _sha256(
                comparison_root / COMPARISON_ID / "manifest.json"
            )
        }
    )
    return RepositorySettings(
        repository_root=REPOSITORY_ROOT,
        vertical_slice_root=vertical_root,
        model_training_root=training_root,
        default_vertical_slice_run_id=None,
        unified_comparison_root=comparison_root.resolve(),
        default_unified_comparison_run_id=default_id,
        unified_comparison_source_authority=source_authority,
        unified_comparison_manifest_authority=effective_manifest_authority,
    )


@contextmanager
def _api_client(settings: RepositorySettings) -> Iterator[TestClient]:
    repository = ArtifactRepository(settings)
    previous = app.dependency_overrides.get(get_artifact_repository)

    def override_repository() -> ArtifactRepository:
        return repository

    app.dependency_overrides[get_artifact_repository] = override_repository
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_artifact_repository, None)
        else:
            app.dependency_overrides[get_artifact_repository] = previous


@pytest.fixture
def comparison_pack(tmp_path: Path) -> tuple[Path, Path]:
    if not artifact_repository_module._SUPPORTS_SECURE_DIR_FD:
        pytest.skip(
            "Unified comparison reads fail closed without secure dir-fd traversal"
        )
    comparison_root = tmp_path / "comparisons"
    comparison_root.mkdir()
    return comparison_root, _make_comparison_pack(comparison_root)


def _all_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _all_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_strings(nested)


def _endpoint_paths() -> tuple[str, ...]:
    base = f"/api/v1/unified-comparisons/{COMPARISON_ID}"
    return (
        "/api/v1/unified-comparisons",
        "/api/v1/unified-comparisons/default",
        base,
        f"{base}/sources",
        f"{base}/summary",
        f"{base}/prediction-metrics",
        f"{base}/validation-metrics",
        f"{base}/portfolio-metrics",
        f"{base}/prediction-rankings",
        f"{base}/portfolio-rankings",
        f"{base}/common-date-contract",
        f"{base}/confusion-matrices",
    )


def test_all_unified_comparison_endpoints_expose_the_validated_pack(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, _run_directory = comparison_pack
    settings = _settings(tmp_path, comparison_root)

    with _api_client(settings) as client:
        responses = {path: client.get(path) for path in _endpoint_paths()}

    assert {path: response.status_code for path, response in responses.items()} == {
        path: 200 for path in _endpoint_paths()
    }
    for response in responses.values():
        payload = response.json()
        assert payload["schema_version"] == "1.0"
        assert payload["disclaimer"] == ACADEMIC_DISCLAIMER
        assert re.search(r"(?<![A-Za-z])(?:NaN|-?Infinity)(?![A-Za-z])", response.text) is None
        json.dumps(payload, allow_nan=False)

    listing = responses["/api/v1/unified-comparisons"].json()
    assert listing["total"] == 1
    assert listing["default_comparison_id"] == COMPARISON_ID
    assert listing["rows"] == [
        {
            "schema_version": "1.0",
            "comparison_id": COMPARISON_ID,
            "status": "PASS",
            "symbol": "OGDC",
            "model_count": 6,
            "validation_count": 2,
            "shared_holdout_count": 2,
            "shared_holdout_target_start": "2026-01-05",
            "shared_holdout_target_end": "2026-01-06",
            "holdout_term": "shared project holdout",
            "is_default": True,
        }
    ]

    default_detail = responses["/api/v1/unified-comparisons/default"].json()
    explicit_detail = responses[
        f"/api/v1/unified-comparisons/{COMPARISON_ID}"
    ].json()
    assert default_detail == explicit_detail
    assert explicit_detail["presentation"] == {
        "format": "tabular_report_only",
        "plots_included": False,
        "data_sources": [
            "prediction-metrics",
            "validation-metrics",
            "portfolio-metrics",
            "prediction-rankings",
            "portfolio-rankings",
            "common-date-contract",
            "confusion-matrices",
        ],
    }


def test_exactly_six_models_once_and_shared_holdout_language(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, _run_directory = comparison_pack
    base = f"/api/v1/unified-comparisons/{COMPARISON_ID}"
    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(base).json()
        prediction = client.get(f"{base}/prediction-metrics").json()
        validation = client.get(f"{base}/validation-metrics").json()
        portfolios = client.get(f"{base}/portfolio-metrics").json()
        dates = client.get(f"{base}/common-date-contract").json()
        matrices = client.get(f"{base}/confusion-matrices").json()

    assert tuple(row["variant_id"] for row in detail["models"]) == MODEL_IDS
    for rows, key in (
        (prediction["models"], "competitor_id"),
        (validation["models"], "competitor_id"),
        (portfolios["models"], "variant_id"),
        (matrices["validation_models"], "competitor_id"),
        (matrices["shared_project_holdout_models"], "competitor_id"),
    ):
        ids = [row[key] for row in rows]
        assert len(ids) == 6
        assert set(ids) == set(MODEL_IDS)
        assert len(ids) == len(set(ids))

    assert {row["algorithm"] for row in detail["models"]} == {
        "random_forest",
        "xgboost",
        "lstm",
    }
    assert all(row["configuration_family"] for row in detail["models"])
    assert prediction["holdout_term"] == "shared project holdout"
    assert all(
        row["evaluation_scope"] == "shared project holdout"
        for row in prediction["models"]
    )
    assert dates["shared_project_holdout"]["terminology"] == (
        "shared project holdout"
    )
    assert dates["shared_project_holdout"]["common_counts"] == dict.fromkeys(
        MODEL_IDS, 2
    )


def test_summary_keeps_outcomes_separate_and_preserves_nullable_portfolio_metrics(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, _run_directory = comparison_pack
    base = f"/api/v1/unified-comparisons/{COMPARISON_ID}"
    with _api_client(_settings(tmp_path, comparison_root)) as client:
        summary = client.get(f"{base}/summary").json()
        predictions = client.get(f"{base}/prediction-metrics").json()
        portfolios = client.get(f"{base}/portfolio-metrics").json()

    assert summary["outcomes_are_not_combined"] is True
    assert set(summary["reported_outcomes"]) == {
        "validation_selected_model",
        "lstm_validation_preferred_model",
        "best_shared_holdout_balanced_accuracy",
        "best_shared_holdout_f1",
        "best_simulated_model_portfolio",
        "lowest_drawdown_model_portfolio",
        "lowest_fee_model_portfolio",
        "cash_only_benchmark",
        "buy_and_hold_result",
        "any_model_beat_buy_and_hold",
    }
    serialized = json.dumps(summary, sort_keys=True).lower()
    assert "universal winner" not in serialized
    assert "universal_winner" not in serialized
    assert summary["reported_outcomes"]["any_model_beat_buy_and_hold"]["value"] is False
    assert all(row["roc_auc"] is not None for row in predictions["models"])
    assert next(
        row for row in portfolios["models"] if row["variant_id"] == "M-001"
    )["sharpe_ratio"] is None
    cash = next(
        row for row in portfolios["benchmarks"] if row["benchmark_id"] == "cash-only"
    )
    assert cash["sharpe_ratio"] is None


def test_sources_are_redacted_and_model_binaries_are_never_loaded(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root, _run_directory = comparison_pack

    def unexpected_load(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("joblib.load must never be called by comparison endpoints")

    monkeypatch.setattr(joblib, "load", unexpected_load)
    with _api_client(_settings(tmp_path, comparison_root)) as client:
        responses = [client.get(path) for path in _endpoint_paths()]

    assert all(response.status_code == 200 for response in responses)
    for response in responses:
        payload = response.json()
        text = "\n".join(_all_strings(payload))
        lowered = text.lower()
        assert str(tmp_path) not in text
        assert "/users/" not in lowered
        assert "/home/" not in lowered
        assert "file://" not in lowered
        assert "c:\\" not in lowered
        assert ".joblib" not in lowered
        assert ".pkl" not in lowered
        assert ".keras" not in lowered

    source_payload = responses[_endpoint_paths().index(
        f"/api/v1/unified-comparisons/{COMPARISON_ID}/sources"
    )].json()
    assert len(source_payload["sources"]) == 3
    assert all("integrity_root_pack_path" not in row for row in source_payload["sources"])


def test_checksum_tamper_is_rejected_and_omitted_from_listing(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    metrics_path = run_directory / "metrics.json"
    metrics_path.write_bytes(metrics_path.read_bytes() + b" ")

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409
    assert "checksum" not in detail.text.lower() or "integrity" in detail.text.lower()


@pytest.mark.parametrize("status", ["PENDING", "FAIL"])
def test_incomplete_or_failed_runs_are_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path, status: str
) -> None:
    comparison_root, run_directory = comparison_pack
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    _write_json(manifest_path, manifest)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["total"] == 0
    assert detail.status_code == 409


def test_nonfinite_json_is_rejected_even_when_resealed(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["validation_model_metrics"][0]["roc_auc"] = float("nan")
    _write_json(metrics_path, metrics, allow_nan=True)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


def test_resealed_probability_metric_is_rejected_by_manifest_authority(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    trusted_manifest_digest = _sha256(run_directory / "manifest.json")
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["validation_model_metrics"][0]["roc_auc"] = 0.99
    _write_json(metrics_path, metrics)
    _reseal_pack(run_directory)

    settings = _settings(
        tmp_path,
        comparison_root,
        manifest_authority={COMPARISON_ID: trusted_manifest_digest},
    )
    with _api_client(settings) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


@pytest.mark.parametrize(
    ("collection", "field", "value"),
    [
        ("validation_model_metrics", "accuracy", 1.5),
        ("validation_model_metrics", "f1", -0.1),
        ("validation_model_metrics", "matthews_corrcoef", 1.01),
        ("validation_model_metrics", "brier_score", 1.01),
        ("validation_model_metrics", "log_loss", -0.01),
        ("model_portfolio_metrics", "maximum_drawdown", -0.01),
        ("model_portfolio_metrics", "volatility", -0.01),
        ("model_portfolio_metrics", "number_of_trades", -1),
        ("model_portfolio_metrics", "win_rate", 5.0),
        ("model_portfolio_metrics", "turnover", -0.01),
        ("model_portfolio_metrics", "fees_paid", "-0.01"),
        ("model_portfolio_metrics", "time_in_market", 1.01),
    ],
)
def test_resealed_out_of_domain_metrics_are_rejected(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    collection: str,
    field: str,
    value: object,
) -> None:
    comparison_root, run_directory = comparison_pack
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics[collection][0][field] = value
    _write_json(metrics_path, metrics)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


def test_resealed_confusion_counts_must_reconcile(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["validation_model_metrics"][0]["confusion_matrix"] = [
        [1, 0],
        [1, 0],
    ]
    _write_json(metrics_path, metrics)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


@pytest.mark.parametrize(
    ("collection", "field", "value"),
    [
        ("validation_model_metrics", "accuracy", 0.25),
        ("validation_reference_baseline_metrics", "f1", 0.25),
    ],
)
def test_resealed_within_range_metrics_must_reconcile_with_confusion_matrix(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    collection: str,
    field: str,
    value: float,
) -> None:
    comparison_root, run_directory = comparison_pack
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics[collection][0][field] = value
    _write_json(metrics_path, metrics)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("selection_eligible", False),
        ("metric_warnings", ["single_class_actuals"]),
        ("roc_auc", None),
        ("log_loss", None),
        ("brier_score", None),
        ("expected_calibration_error", None),
    ],
)
def test_resealed_metrics_must_preserve_producer_class_coverage_semantics(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    comparison_root, run_directory = comparison_pack
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["validation_model_metrics"][0][field] = value
    _write_json(metrics_path, metrics)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def _prediction_response_row() -> dict[str, Any]:
    raw = _prediction_metric("M-001", "validation", 1)
    return {
        "competitor_id": raw["competitor_id"],
        "competitor_type": raw["competitor_type"],
        "registered_model_id": raw["registered_model_id"],
        "algorithm": raw["algorithm"],
        "configuration_family": raw["label"],
        "evaluation_scope": raw["evaluation_scope"],
        **{
            key: raw[key]
            for key in (
                "row_count",
                "support",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "matthews_corrcoef",
                "roc_auc",
                "brier_score",
                "log_loss",
                "expected_calibration_error",
                "selection_eligible",
                "metric_warnings",
                "class_counts",
                "predicted_class_counts",
                "confusion_matrix",
            )
        },
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("accuracy", -0.01),
        ("balanced_accuracy", 1.01),
        ("precision", 1.01),
        ("recall", -0.01),
        ("f1", 99.0),
        ("matthews_corrcoef", -1.01),
        ("roc_auc", 1.01),
        ("brier_score", -0.01),
        ("log_loss", -0.01),
        ("expected_calibration_error", 1.01),
    ],
)
def test_prediction_response_schema_rejects_out_of_domain_metrics(
    field: str, value: float
) -> None:
    row = _prediction_response_row()
    row[field] = value

    with pytest.raises(ValidationError):
        UnifiedPredictionMetricRow.model_validate(row)


def test_prediction_response_schema_accepts_metric_boundaries() -> None:
    lower = _prediction_response_row()
    lower.update(
        {
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matthews_corrcoef": -1.0,
            "roc_auc": 0.0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "expected_calibration_error": 0.0,
            "confusion_matrix": [[0, 1], [1, 0]],
        }
    )
    assert UnifiedPredictionMetricRow.model_validate(lower)

    upper = _prediction_response_row()
    for field in (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
    ):
        upper[field] = 1.0
    upper["matthews_corrcoef"] = 1.0
    upper["confusion_matrix"] = [[1, 0], [0, 1]]
    assert UnifiedPredictionMetricRow.model_validate(upper)


def test_prediction_response_schema_reconciles_confusion_counts() -> None:
    row = _prediction_response_row()
    row["predicted_class_counts"] = {"0": 2, "1": 0}

    with pytest.raises(ValidationError):
        UnifiedPredictionMetricRow.model_validate(row)

    with pytest.raises(ValidationError):
        UnifiedConfusionMatrixRow.model_validate(
            {
                "competitor_id": "M-001",
                "competitor_type": "model",
                "registered_model_id": _registered_model_id("M-001"),
                "evaluation_scope": "validation",
                "class_labels": [0, 1],
                "confusion_matrix": [[1, 0], [0, 0]],
                "support": 2,
            }
        )


@pytest.mark.parametrize("competitor_type", ["model", "reference_baseline"])
def test_prediction_response_schema_rejects_within_range_metric_contradictions(
    competitor_type: str,
) -> None:
    row = _prediction_response_row()
    if competitor_type == "reference_baseline":
        row.update(
            {
                "competitor_id": "majority-class-baseline",
                "competitor_type": competitor_type,
                "registered_model_id": None,
                "algorithm": None,
                "configuration_family": None,
            }
        )
    row["accuracy"] = 0.25

    with pytest.raises(ValidationError, match="confusion matrix"):
        UnifiedPredictionMetricRow.model_validate(row)


def test_prediction_response_schema_enforces_single_class_semantics() -> None:
    row = _prediction_response_row()
    row.update(
        {
            "accuracy": 1.0,
            "balanced_accuracy": None,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "matthews_corrcoef": None,
            "roc_auc": None,
            "log_loss": None,
            "selection_eligible": False,
            "metric_warnings": ["single_class_actuals"],
            "class_counts": {"0": 2, "1": 0},
            "predicted_class_counts": {"0": 2, "1": 0},
            "confusion_matrix": [[2, 0], [0, 0]],
        }
    )
    assert UnifiedPredictionMetricRow.model_validate(row)

    for field, value in (
        ("selection_eligible", True),
        ("metric_warnings", []),
        ("roc_auc", 0.5),
        ("log_loss", 0.5),
    ):
        contradictory = dict(row)
        contradictory[field] = value
        with pytest.raises(ValidationError):
            UnifiedPredictionMetricRow.model_validate(contradictory)


def test_prediction_response_schema_rejects_unsafe_classification_counts() -> None:
    row = _prediction_response_row()
    oversized = 10_000_001
    row.update(
        {
            "row_count": oversized,
            "support": oversized,
            "class_counts": {"0": oversized - 1, "1": 1},
            "predicted_class_counts": {"0": oversized - 1, "1": 1},
            "confusion_matrix": [[oversized - 1, 0], [0, 1]],
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "matthews_corrcoef": 1.0,
        }
    )

    with pytest.raises(ValidationError):
        UnifiedPredictionMetricRow.model_validate(row)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maximum_drawdown", -0.01),
        ("volatility", -0.01),
        ("number_of_trades", -1),
        ("win_rate", 1.01),
        ("turnover", -0.01),
        ("fees_paid", "-0.01"),
        ("time_in_market", 1.01),
    ],
)
def test_portfolio_response_schema_rejects_out_of_domain_metrics(
    field: str, value: object
) -> None:
    raw = _portfolio_metric("M-001", 1)
    raw["variant_id"] = raw.pop("competitor_id")
    raw.pop("competitor_type")
    raw[field] = value

    with pytest.raises(ValidationError):
        UnifiedModelPortfolioMetric.model_validate(raw)


def test_benchmark_response_schema_rejects_out_of_domain_metrics() -> None:
    raw = _benchmark_metric("buy-and-hold")
    raw.pop("competitor_type")
    raw.pop("ledger_provenance")
    raw["ledger_count"] = 6
    raw["fees_paid"] = "-1.00"

    with pytest.raises(ValidationError):
        UnifiedBenchmarkPortfolioMetric.model_validate(raw)


@pytest.mark.parametrize(
    "unsafe_text",
    [
        "//server/share/model.keras",
        "models/private.joblib",
        "models/private.joblib.",
        "models/private.joblib/metadata",
        r"models\private.joblib\metadata",
        "models/private.safetensors/index.json",
        "file:///private/model.joblib",
        "file:///private/report.csv",
        "path=/etc/passwd",
        r"path:C:\secret",
        "source:file:///tmp/x",
        r"source:file:C:\private\report.csv",
        "note,/Users/private/report.csv",
        "note;//server/share/report.csv",
        "See https://example.test/report?source=/Users/private/report.csv",
        "See https://example.test/report?source=%2FUsers%2Fprivate%2Freport.csv",
        "See https://example.test/models/private%2Ejoblib",
    ],
)
def test_renderable_public_text_rejects_paths_and_model_binaries(
    comparison_pack: tuple[Path, Path], tmp_path: Path, unsafe_text: str
) -> None:
    comparison_root, run_directory = comparison_pack
    summary_path = run_directory / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["limitations"].append(unsafe_text)
    _write_json(summary_path, summary)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        summary_response = client.get(
            f"/api/v1/unified-comparisons/{COMPARISON_ID}/summary"
        )

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert summary_response.status_code == 409
    assert unsafe_text not in summary_response.text


def test_renderable_public_text_allows_web_urls(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    summary_path = run_directory / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    allowed_text = [
        "See https://models.example.test/usr/local/share/model-card.html",
        "See wss://models.example.test/usr/local/share/model-card.html",
        "See ftp://models.example.test/usr/local/share/model-card.html",
        "profile:academic_research",
    ]
    summary["limitations"].extend(allowed_text)
    _write_json(summary_path, summary)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        summary_response = client.get(
            f"/api/v1/unified-comparisons/{COMPARISON_ID}/summary"
        )

    assert summary_response.status_code == 200
    assert all(
        item in summary_response.json()["limitations"] for item in allowed_text
    )


def test_malformed_and_traversal_identifiers_cannot_escape_root(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, _run_directory = comparison_pack
    settings = _settings(tmp_path, comparison_root)
    repository = ArtifactRepository(settings)
    with pytest.raises(InvalidIdentifierError):
        repository.load_unified_comparison_detail("../outside")

    with _api_client(settings) as client:
        malformed = client.get("/api/v1/unified-comparisons/bad%5Cid")
        traversal = client.get(
            f"/api/v1/unified-comparisons/{quote('../outside', safe='')}"
        )

    assert malformed.status_code == 400
    assert traversal.status_code in {400, 404}
    assert str(tmp_path) not in malformed.text
    assert str(tmp_path) not in traversal.text


def test_run_symlink_escape_is_rejected_and_omitted(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    outside = tmp_path / "outside-run"
    shutil.move(run_directory, outside)
    os.symlink(outside, run_directory, target_is_directory=True)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


def test_member_symlink_escape_is_rejected_and_omitted(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    member = run_directory / "metrics.json"
    outside = tmp_path / "outside-metrics.json"
    outside.write_bytes(member.read_bytes())
    member.unlink()
    os.symlink(outside, member)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


@pytest.mark.skipif(
    os.name == "nt"
    or not hasattr(os, "O_NONBLOCK")
    or not hasattr(os, "mkfifo")
    or os.mkfifo not in os.supports_dir_fd,
    reason="FIFO descriptor-relative swap is a POSIX regression test",
)
def test_regular_file_fifo_swap_is_opened_nonblocking_and_rejected(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root, _run_directory = comparison_pack
    repository = ArtifactRepository(_settings(tmp_path, comparison_root))
    real_stat = artifact_repository_module.os.stat
    real_open = artifact_repository_module.os.open
    swapped = False
    saw_nonblocking_open = False

    def swapping_stat(
        path: object,
        *,
        dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> os.stat_result:
        nonlocal swapped
        observed = real_stat(
            path,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
        )
        if (
            path == "metrics.json"
            and dir_fd is not None
            and follow_symlinks is False
            and not swapped
        ):
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(path, mode=0o600, dir_fd=dir_fd)
            swapped = True
        return observed

    def guarded_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal saw_nonblocking_open
        if path == "metrics.json" and dir_fd is not None and swapped:
            saw_nonblocking_open = bool(flags & os.O_NONBLOCK)
            if not saw_nonblocking_open:
                raise OSError("refusing a potentially blocking FIFO open")
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifact_repository_module.os, "stat", swapping_stat)
    monkeypatch.setattr(artifact_repository_module.os, "open", guarded_open)

    with pytest.raises(ArtifactIntegrityError):
        repository.load_unified_summary(COMPARISON_ID)

    assert swapped is True
    assert saw_nonblocking_open is True


def test_comparison_tree_entry_limit_is_enforced_during_scan(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    overflow = run_directory / "overflow"
    overflow.mkdir()
    for index in range(artifact_repository_module._UNIFIED_MAX_TREE_ENTRIES):
        (overflow / f"entry-{index:03d}").touch()
    repository = ArtifactRepository(_settings(tmp_path, comparison_root))

    with pytest.raises(ArtifactIntegrityError) as captured:
        repository.load_unified_summary(COMPARISON_ID)

    assert captured.value.__cause__ is not None
    assert "entry limit" in str(captured.value.__cause__)


def test_comparison_tree_depth_limit_is_enforced_before_recursion(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    nested = run_directory
    for index in range(artifact_repository_module._UNIFIED_MAX_TREE_DEPTH + 1):
        nested = nested / f"depth-{index}"
        nested.mkdir()
    repository = ArtifactRepository(_settings(tmp_path, comparison_root))

    with pytest.raises(ArtifactIntegrityError) as captured:
        repository.load_unified_summary(COMPARISON_ID)

    assert captured.value.__cause__ is not None
    assert "depth limit" in str(captured.value.__cause__)


@pytest.mark.skipif(os.name == "nt", reason="Covered by the Windows junction test")
def test_configured_root_symlink_escape_is_rejected(tmp_path: Path) -> None:
    outside_root = tmp_path / "outside-comparisons"
    outside_root.mkdir()
    _make_comparison_pack(outside_root)
    linked_root = tmp_path / "linked-comparisons"
    os.symlink(outside_root, linked_root, target_is_directory=True)
    vertical_root = tmp_path / "vertical"
    training_root = tmp_path / "training"
    vertical_root.mkdir()
    training_root.mkdir()
    settings = RepositorySettings(
        repository_root=REPOSITORY_ROOT,
        vertical_slice_root=vertical_root,
        model_training_root=training_root,
        default_vertical_slice_run_id=None,
        unified_comparison_root=linked_root,
        default_unified_comparison_run_id=COMPARISON_ID,
    )

    with _api_client(settings) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 409
    assert detail.status_code == 409
    assert str(outside_root) not in listing.text
    assert str(outside_root) not in detail.text


def test_configured_root_ancestor_swap_cannot_redirect_a_pinned_read(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root, _run_directory = comparison_pack
    outside_root = tmp_path / "outside-comparisons"
    outside_run = _make_comparison_pack(outside_root)
    outside_summary_path = outside_run / "final_summary.json"
    outside_summary = json.loads(outside_summary_path.read_text(encoding="utf-8"))
    outside_summary["limitations"].append("outside-root-sentinel")
    _write_json(outside_summary_path, outside_summary)
    _reseal_pack(outside_run)

    settings = _settings(tmp_path, comparison_root)
    repository = ArtifactRepository(settings)
    displaced_root = tmp_path / "displaced-comparisons"
    real_open = artifact_repository_module.os.open
    swapped = False

    def swapping_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == COMPARISON_ID and dir_fd is not None and not swapped:
            comparison_root.rename(displaced_root)
            os.symlink(outside_root, comparison_root, target_is_directory=True)
            swapped = True
        if dir_fd is None:
            return real_open(path, flags, mode)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(artifact_repository_module.os, "open", swapping_open)

    summary = repository.load_unified_summary(COMPARISON_ID)

    assert swapped is True
    assert "outside-root-sentinel" not in summary["limitations"]


def test_detail_projects_a_single_validated_snapshot(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root, _run_directory = comparison_pack
    repository = ArtifactRepository(_settings(tmp_path, comparison_root))
    validate = repository._validate_unified_comparison
    calls = 0

    def counted_validate(comparison_id: str, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return validate(comparison_id, **kwargs)

    monkeypatch.setattr(
        repository, "_validate_unified_comparison", counted_validate
    )

    detail = repository.load_unified_comparison_detail(COMPARISON_ID)

    assert detail["comparison_id"] == COMPARISON_ID
    assert calls == 1


def test_cached_signature_walk_oserror_fails_closed_without_leaking_details(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root, _run_directory = comparison_pack
    settings = _settings(tmp_path, comparison_root)

    with _api_client(settings) as client:
        warm = client.get(
            f"/api/v1/unified-comparisons/{COMPARISON_ID}/summary"
        )
        assert warm.status_code == 200

        def failing_signature(_descriptor: int) -> tuple[tuple[object, ...], ...]:
            raise OSError("private signature-walk detail")

        monkeypatch.setattr(
            artifact_repository_module,
            "_descriptor_tree_signature",
            failing_signature,
        )
        response = client.get(
            f"/api/v1/unified-comparisons/{COMPARISON_ID}/summary"
        )

    assert response.status_code == 409
    assert "private signature-walk detail" not in response.text


def test_root_enumeration_oserror_fails_closed_without_leaking_details(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root, _run_directory = comparison_pack
    settings = _settings(tmp_path, comparison_root)

    with _api_client(settings) as client:
        def failing_scandir(_descriptor: int) -> object:
            raise OSError("private root-enumeration detail")

        monkeypatch.setattr(
            artifact_repository_module.os,
            "scandir",
            failing_scandir,
        )
        response = client.get("/api/v1/unified-comparisons")

    assert response.status_code == 409
    assert "private root-enumeration detail" not in response.text


def test_internally_consistent_non_authoritative_sources_are_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, _run_directory = comparison_pack
    settings = _settings(tmp_path, comparison_root, authorize_fixture=False)

    with _api_client(settings) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


def test_repository_snapshots_and_deep_freezes_source_authority(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, _run_directory = comparison_pack
    mutable_authority = {
        role: dict(identity)
        for role, identity in FIXTURE_SOURCE_AUTHORITY.items()
    }
    settings = replace(
        _settings(tmp_path, comparison_root),
        unified_comparison_source_authority=mutable_authority,
    )
    repository = ArtifactRepository(settings)

    assert repository.load_unified_summary(COMPARISON_ID)["status"] == "PASS"
    mutable_authority["lstm"]["run_id"] = "LSTM-no-longer-authorized"

    assert repository.load_unified_summary(COMPARISON_ID)["status"] == "PASS"
    with pytest.raises(TypeError):
        repository.settings.unified_comparison_source_authority["lstm"][  # type: ignore[index]
            "run_id"
        ] = "LSTM-cannot-mutate-snapshot"
    with pytest.raises(ArtifactIntegrityError):
        ArtifactRepository(settings).load_unified_summary(COMPARISON_ID)


def test_resealed_evidence_substitution_remains_bound_to_authoritative_source(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    evidence_relative = "evidence/predictions/M-006/predictions_all_splits.csv"
    evidence_path = run_directory / evidence_relative
    evidence_path.write_text(
        "pre-hardening-lstm-01-substitution\n", encoding="utf-8"
    )
    substituted_digest = _sha256(evidence_path)

    source_path = run_directory / "source_roots.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    for evidence in sources["evidence_files"]:
        if evidence["pack_path"] == evidence_relative:
            evidence["sha256"] = substituted_digest
            break
    else:  # pragma: no cover - fixture authoring guard
        raise AssertionError("M-006 prediction evidence binding is missing")
    _write_json(source_path, sources)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409
    assert "pre-hardening-lstm-01-substitution" not in detail.text


def test_consistently_rewritten_sources_cannot_self_authorize(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    training_path = (
        run_directory
        / "evidence/source_manifests/baseline_training_run_manifest.json"
    )
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["run_id"] = "TRN-rewritten"
    _write_json(training_path, training)
    training_digest = _sha256(training_path)

    vertical_path = (
        run_directory
        / "evidence/source_manifests/baseline_vertical_manifest_certificate.json"
    )
    vertical = json.loads(vertical_path.read_text(encoding="utf-8"))
    vertical["run_id"] = "VSL-rewritten"
    vertical["source_manifest"]["sha256"] = "3" * 64
    vertical["source_assertions"]["model_training_run_id"] = "TRN-rewritten"
    for entry in vertical["artifact_inventory"]["files"]:
        entry["path"] = entry["path"].replace(
            "experiments/VSL-api-fixture-",
            "experiments/VSL-rewritten-",
        )
    _write_json(vertical_path, vertical)
    vertical_certificate_digest = _sha256(vertical_path)

    lstm_artifact_path = (
        run_directory
        / "evidence/source_manifests/lstm_artifact_manifest.json"
    )
    lstm_artifact = json.loads(lstm_artifact_path.read_text(encoding="utf-8"))
    for entry in lstm_artifact["files"]:
        entry["path"] = entry["path"].replace(
            "backtests/LSTM-api-fixture-",
            "backtests/LSTM-rewritten-",
        )
    _write_json(lstm_artifact_path, lstm_artifact)
    lstm_artifact_digest = _sha256(lstm_artifact_path)

    lstm_path = run_directory / "evidence/source_manifests/lstm_run_manifest.json"
    lstm = json.loads(lstm_path.read_text(encoding="utf-8"))
    lstm["run_id"] = "LSTM-rewritten"
    lstm["baseline_training_run"] = {
        "run_id": "TRN-rewritten",
        "manifest_sha256": training_digest,
    }
    lstm["baseline_vertical_slice"] = {
        "run_id": "VSL-rewritten",
        "manifest_sha256": "3" * 64,
    }
    for entry in lstm["artifact_inventory"]["files"]:
        entry["path"] = entry["path"].replace(
            "backtests/LSTM-api-fixture-",
            "backtests/LSTM-rewritten-",
        )
        if entry["path"] == "artifact_manifest.json":
            entry["sha256"] = lstm_artifact_digest
    _write_json(lstm_path, lstm)
    lstm_digest = _sha256(lstm_path)

    source_path = run_directory / "source_roots.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    rewritten_authority = {
        "baseline_training": {
            "run_id": "TRN-rewritten",
            "manifest_sha256": training_digest,
        },
        "baseline_vertical_slice": {
            "run_id": "VSL-rewritten",
            "manifest_sha256": "3" * 64,
            "portable_certificate_sha256": vertical_certificate_digest,
        },
        "lstm": {
            "run_id": "LSTM-rewritten",
            "manifest_sha256": lstm_digest,
        },
    }
    for source in sources["sources"]:
        authority = rewritten_authority[source["role"]]
        source["run_id"] = authority["run_id"]
        source["integrity_root_sha256"] = authority["manifest_sha256"]
        source["integrity_root_pack_sha256"] = (
            authority.get("portable_certificate_sha256")
            or authority["manifest_sha256"]
        )
    rewritten_evidence_hashes = {
        training_path.relative_to(run_directory).as_posix(): training_digest,
        lstm_artifact_path.relative_to(run_directory).as_posix(): (
            lstm_artifact_digest
        ),
        lstm_path.relative_to(run_directory).as_posix(): lstm_digest,
    }
    for evidence in sources["evidence_files"]:
        evidence["source_path"] = evidence["source_path"].replace(
            "experiments/VSL-api-fixture-",
            "experiments/VSL-rewritten-",
        ).replace(
            "backtests/LSTM-api-fixture-",
            "backtests/LSTM-rewritten-",
        )
        if evidence["pack_path"] in rewritten_evidence_hashes:
            evidence["sha256"] = rewritten_evidence_hashes[evidence["pack_path"]]
    _write_json(source_path, sources)
    _reseal_pack(run_directory)

    rewritten_settings = replace(
        _settings(tmp_path, comparison_root),
        unified_comparison_source_authority=rewritten_authority,
    )
    with _api_client(rewritten_settings) as client:
        internally_consistent = client.get(
            f"/api/v1/unified-comparisons/{COMPARISON_ID}"
        )
    with _api_client(_settings(tmp_path, comparison_root)) as client:
        authoritative = client.get(
            f"/api/v1/unified-comparisons/{COMPARISON_ID}"
        )

    assert internally_consistent.status_code == 200
    assert authoritative.status_code == 409


def test_missing_secure_descriptor_support_fails_closed(
    comparison_pack: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_root, _run_directory = comparison_pack
    monkeypatch.setattr(
        artifact_repository_module, "_SUPPORTS_SECURE_DIR_FD", False
    )

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 409
    assert detail.status_code == 409


@pytest.mark.skipif(os.name != "nt", reason="Windows junction behavior")
def test_configured_root_windows_junction_fails_closed(tmp_path: Path) -> None:
    outside_root = tmp_path / "junction-target"
    outside_root.mkdir()
    _make_comparison_pack(outside_root)
    junction = tmp_path / "comparison-junction"
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(outside_root)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip("This Windows environment cannot create directory junctions")
    vertical_root = tmp_path / "vertical"
    training_root = tmp_path / "training"
    vertical_root.mkdir()
    training_root.mkdir()
    settings = RepositorySettings(
        repository_root=REPOSITORY_ROOT,
        vertical_slice_root=vertical_root,
        model_training_root=training_root,
        default_vertical_slice_run_id=None,
        unified_comparison_root=junction,
        default_unified_comparison_run_id=COMPARISON_ID,
    )
    try:
        with _api_client(settings) as client:
            listing = client.get("/api/v1/unified-comparisons")
            detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")
        assert listing.status_code == 409
        assert detail.status_code == 409
    finally:
        os.rmdir(junction)


def test_resealed_source_digest_contradiction_is_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    source_path = run_directory / "source_roots.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    sources["sources"][0]["integrity_root_sha256"] = "0" * 64
    _write_json(source_path, sources)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


def test_resealed_source_run_identity_contradiction_is_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    source_path = run_directory / "source_roots.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    sources["sources"][0]["run_id"] = "TRN-wrong"
    _write_json(source_path, sources)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_resealed_summary_ranking_identity_contradiction_is_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    summary_path = run_directory / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    outcome = summary["reported_outcomes"]["best_shared_holdout_f1"]
    outcome["variant_id"] = "M-001"
    outcome["registered_model_id"] = _registered_model_id("M-001")
    _write_json(summary_path, summary)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


def test_resealed_nonwinning_ranking_metric_contradiction_is_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    rankings_path = run_directory / "rankings.json"
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    rankings["validation_model_selection"][1]["accuracy"] = 0.123456
    _write_json(rankings_path, rankings)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_resealed_duplicate_registered_model_identity_is_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    metrics_path = run_directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    duplicate_id = _registered_model_id("M-001")
    metrics["validation_model_metrics"][1]["registered_model_id"] = duplicate_id
    metrics["shared_project_holdout_prediction_metrics"][1][
        "registered_model_id"
    ] = duplicate_id
    _write_json(metrics_path, metrics)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_resealed_overlapping_validation_and_holdout_is_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    alignment_path = run_directory / "date_alignment.json"
    alignment = json.loads(alignment_path.read_text(encoding="utf-8"))
    holdout = alignment["shared_project_holdout"]
    validation = alignment["validation"]
    for key in (
        "feature_start",
        "feature_end",
        "target_start",
        "target_end",
        "common_keys",
    ):
        validation[key] = holdout[key]
    _write_json(alignment_path, alignment)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_resealed_source_payload_count_contradiction_is_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    source_path = run_directory / "source_roots.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    sources["sources"][0]["payload_file_count"] = 999
    _write_json(source_path, sources)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_resealed_source_metadata_cannot_claim_a_universal_winner(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    source_path = run_directory / "source_roots.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    sources["evidence_files"][0]["purpose"] = "universal winner"
    _write_json(source_path, sources)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_resealed_holdout_summary_scope_must_use_contract_term(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    summary_path = run_directory / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["reported_outcomes"]["best_shared_holdout_f1"]["scope"] = "test"
    _write_json(summary_path, summary)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_windows_absolute_source_paths_are_rejected(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    source_path = run_directory / "source_roots.json"
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    sources["evidence_files"][0]["source_path"] = "C:/Users/alice/private.csv"
    _write_json(source_path, sources)
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert detail.status_code == 409


def test_executable_model_members_are_rejected_even_when_inventoried(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, run_directory = comparison_pack
    model_binary = run_directory / "evidence/models/unsafe.joblib"
    model_binary.parent.mkdir(parents=True)
    model_binary.write_bytes(b"not actually a model")
    _reseal_pack(run_directory)

    with _api_client(_settings(tmp_path, comparison_root)) as client:
        listing = client.get("/api/v1/unified-comparisons")
        detail = client.get(f"/api/v1/unified-comparisons/{COMPARISON_ID}")

    assert listing.status_code == 200
    assert listing.json()["rows"] == []
    assert detail.status_code == 409


@pytest.mark.skipif(
    not artifact_repository_module._SUPPORTS_SECURE_DIR_FD,
    reason="Unified comparison reads fail closed without secure dir-fd traversal",
)
def test_default_is_explicit_and_empty_root_is_not_guessed(tmp_path: Path) -> None:
    comparison_root = tmp_path / "comparisons"
    comparison_root.mkdir()
    _make_comparison_pack(comparison_root)
    settings = _settings(tmp_path, comparison_root, default_id=None)

    with _api_client(settings) as client:
        listing = client.get("/api/v1/unified-comparisons")
        default = client.get("/api/v1/unified-comparisons/default")

    assert listing.status_code == 200
    assert listing.json()["total"] == 1
    assert listing.json()["default_comparison_id"] is None
    assert listing.json()["rows"][0]["is_default"] is False
    assert default.status_code == 409


def test_environment_configures_comparison_root_and_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "configured-comparisons"
    monkeypatch.setenv("PAISA_UNIFIED_COMPARISON_ROOT", str(configured))
    monkeypatch.setenv("PAISA_DEFAULT_UNIFIED_COMPARISON_RUN_ID", COMPARISON_ID)

    settings = settings_from_environment()

    assert settings.unified_comparison_root == configured
    assert settings.default_unified_comparison_run_id == COMPARISON_ID


def test_existing_api_routes_remain_available(
    comparison_pack: tuple[Path, Path], tmp_path: Path
) -> None:
    comparison_root, _run_directory = comparison_pack
    with _api_client(_settings(tmp_path, comparison_root)) as client:
        health = client.get("/health")
        vertical_slices = client.get("/api/v1/vertical-slices")
        openapi = client.get("/openapi.json")

    assert health.status_code == 200
    assert vertical_slices.status_code == 200
    assert vertical_slices.json()["rows"] == []
    assert openapi.status_code == 200
    paths = set(openapi.json()["paths"])
    assert {
        "/health",
        "/pipeline/status",
        "/stocks",
        "/api/v1/vertical-slices",
        "/api/v1/leaderboard",
        "/api/v1/unified-comparisons",
        "/api/v1/unified-comparisons/{comparison_id}/summary",
    } <= paths
