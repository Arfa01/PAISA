from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import stat
import struct
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
import zipfile
import zlib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import paisa.explainability.tree_shap as tree_shap
from paisa.experiments.artifacts import sha256_file, write_json


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RUN_ID = "XAI-20261024-120000-OGDC-tree-v1-fixture"
FEATURE_COLUMNS = tree_shap.EXPECTED_FEATURE_COLUMNS
PNG_FIXTURE = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000b4944415478da636000020000050001e9fadcd80000000049454e44ae426082"
)
HOLDOUT_ROWS_CSV = """\
2025-12-15,2025-12-16,0
2025-12-16,2025-12-17,0
2025-12-17,2025-12-18,0
2025-12-18,2025-12-19,0
2025-12-19,2025-12-22,0
2025-12-22,2025-12-23,0
2025-12-23,2025-12-24,1
2025-12-24,2025-12-26,1
2025-12-26,2025-12-29,1
2025-12-29,2025-12-30,1
2025-12-30,2025-12-31,1
2025-12-31,2026-01-01,1
2026-01-01,2026-01-02,1
2026-01-02,2026-01-05,1
2026-01-05,2026-01-06,0
2026-01-06,2026-01-07,1
2026-01-07,2026-01-08,1
2026-01-08,2026-01-09,0
2026-01-09,2026-01-12,0
2026-01-12,2026-01-13,1
2026-01-13,2026-01-14,1
2026-01-14,2026-01-15,1
2026-01-15,2026-01-16,1
2026-01-16,2026-01-19,0
2026-01-19,2026-01-20,1
2026-01-20,2026-01-21,0
2026-01-21,2026-01-22,1
2026-01-22,2026-01-23,0
2026-01-23,2026-01-26,0
2026-01-26,2026-01-27,1
2026-01-27,2026-01-28,1
2026-01-28,2026-01-29,0
2026-01-29,2026-01-30,1
2026-01-30,2026-02-02,1
2026-02-02,2026-02-03,1
2026-02-03,2026-02-04,0
2026-02-04,2026-02-06,0
2026-02-06,2026-02-09,0
2026-02-09,2026-02-10,0
2026-02-10,2026-02-11,0
2026-02-11,2026-02-12,0
2026-02-12,2026-02-13,0
2026-02-13,2026-02-16,0
2026-02-16,2026-02-17,1
2026-02-17,2026-02-18,1
2026-02-18,2026-02-19,0
2026-02-19,2026-02-20,1
2026-02-20,2026-02-23,0
2026-02-23,2026-02-24,1
2026-02-24,2026-02-25,0
2026-02-25,2026-02-26,1
2026-02-26,2026-02-27,0
2026-02-27,2026-03-02,0
2026-03-02,2026-03-03,1
2026-03-03,2026-03-04,1
2026-03-04,2026-03-05,1
2026-03-05,2026-03-06,0
2026-03-06,2026-03-09,0
2026-03-09,2026-03-10,1
2026-03-10,2026-03-11,0
2026-03-11,2026-03-12,0
2026-03-12,2026-03-13,0
2026-03-13,2026-03-16,0
2026-03-16,2026-03-17,1
2026-03-17,2026-03-18,1
2026-03-18,2026-03-19,0
2026-03-19,2026-03-24,0
2026-03-24,2026-03-25,1
2026-03-25,2026-03-26,0
2026-03-26,2026-03-27,0
2026-03-27,2026-03-30,0
2026-03-30,2026-03-31,1
2026-03-31,2026-04-01,1
2026-04-01,2026-04-02,0
2026-04-02,2026-04-03,0
2026-04-03,2026-04-06,1
2026-04-06,2026-04-07,0
2026-04-07,2026-04-08,1
2026-04-08,2026-04-09,0
2026-04-09,2026-04-10,1
2026-04-10,2026-04-13,0
2026-04-13,2026-04-14,1
2026-04-14,2026-04-15,1
2026-04-15,2026-04-16,1
2026-04-16,2026-04-17,1
2026-04-17,2026-04-20,1
2026-04-20,2026-04-21,0
2026-04-21,2026-04-22,0
2026-04-22,2026-04-23,0
2026-04-23,2026-04-24,1
2026-04-24,2026-04-27,0
2026-04-27,2026-04-28,0
2026-04-28,2026-04-29,0
2026-04-29,2026-04-30,0
2026-04-30,2026-05-04,1
2026-05-04,2026-05-05,1
2026-05-05,2026-05-06,1
2026-05-06,2026-05-07,1
2026-05-07,2026-05-08,0
2026-05-08,2026-05-11,0
2026-05-11,2026-05-12,0
2026-05-12,2026-05-13,0
2026-05-13,2026-05-14,0
2026-05-14,2026-05-15,0
2026-05-15,2026-05-18,0
2026-05-18,2026-05-19,1
2026-05-19,2026-05-20,1
2026-05-20,2026-05-21,1
2026-05-21,2026-05-22,0
2026-05-22,2026-05-25,1
2026-05-25,2026-05-29,1
2026-05-29,2026-06-01,0
2026-06-01,2026-06-02,1
2026-06-02,2026-06-03,0
2026-06-03,2026-06-04,1
2026-06-04,2026-06-05,0
2026-06-05,2026-06-08,0
2026-06-08,2026-06-09,1
2026-06-09,2026-06-10,0
2026-06-10,2026-06-11,0
2026-06-11,2026-06-12,1
2026-06-12,2026-06-15,1
2026-06-15,2026-06-16,1
2026-06-16,2026-06-17,1
2026-06-17,2026-06-18,1
2026-06-18,2026-06-19,0
2026-06-19,2026-06-22,1
2026-06-22,2026-06-23,1
2026-06-23,2026-06-24,1
2026-06-24,2026-06-29,0
2026-06-29,2026-06-30,0
2026-06-30,2026-07-01,1
2026-07-01,2026-07-02,1
2026-07-02,2026-07-03,1
2026-07-03,2026-07-06,1
2026-07-06,2026-07-07,0
2026-07-07,2026-07-08,0
2026-07-08,2026-07-09,0
2026-07-09,2026-07-10,1
2026-07-10,2026-07-13,0
2026-07-13,2026-07-14,0
2026-07-14,2026-07-15,1
2026-07-15,2026-07-16,1
2026-07-16,2026-07-17,0
2026-07-17,2026-07-20,1
2026-07-20,2026-07-21,0
2026-07-21,2026-07-22,1
2026-07-22,2026-07-23,0
2026-07-23,2026-07-24,0
2026-07-24,2026-07-27,1
2026-07-27,2026-07-28,0
2026-07-28,2026-07-29,0
2026-07-29,2026-07-30,0
2026-07-30,2026-07-31,1
2026-07-31,2026-08-03,1
2026-08-03,2026-08-04,0
2026-08-04,2026-08-05,1
2026-08-05,2026-08-06,1
2026-08-06,2026-08-07,1
2026-08-07,2026-08-10,1
2026-08-10,2026-08-11,0
2026-08-11,2026-08-12,0
2026-08-12,2026-08-13,0
2026-08-13,2026-08-17,1
2026-08-17,2026-08-18,0
2026-08-18,2026-08-19,0
2026-08-19,2026-08-20,1
2026-08-20,2026-08-21,1
2026-08-21,2026-08-24,1
2026-08-24,2026-08-25,0
2026-08-25,2026-08-27,0
2026-08-27,2026-08-28,1
2026-08-28,2026-08-31,1
2026-08-31,2026-09-01,1
2026-09-01,2026-09-02,0
2026-09-02,2026-09-03,0
2026-09-03,2026-09-04,1
2026-09-04,2026-09-07,0
2026-09-07,2026-09-08,1
2026-09-08,2026-09-09,1
2026-09-09,2026-09-10,0
2026-09-10,2026-09-11,1
2026-09-11,2026-09-14,0
2026-09-14,2026-09-15,0
"""
BACKGROUND_TARGET_DATES_CSV = """\
2021-10-14,2021-10-27,2021-11-09,2021-11-19,2021-12-02,2021-12-15,2021-12-27,2022-01-07,2022-01-19,2022-02-01,
2022-02-14,2022-02-24,2022-03-09,2022-03-21,2022-04-04,2022-04-15,2022-04-27,2022-05-17,2022-05-27,2022-06-09,
2022-06-22,2022-07-04,2022-07-20,2022-08-01,2022-08-16,2022-08-29,2022-09-08,2022-09-21,2022-10-03,2022-10-14,
2022-10-27,2022-11-08,2022-11-22,2022-12-05,2022-12-15,2022-12-28,2023-01-09,2023-01-20,2023-02-02,2023-02-14,
2023-02-27,2023-03-09,2023-03-22,2023-04-05,2023-04-18,2023-05-05,2023-05-17,2023-05-30,2023-06-12,2023-06-22,
2023-07-10,2023-07-20,2023-08-03,2023-08-17,2023-08-29,2023-09-11,2023-09-21,2023-10-05,2023-10-18,2023-10-30,
2023-11-13,2023-11-23,2023-12-06,2023-12-19,2024-01-01,2024-01-12,2024-01-25,2024-02-07,2024-02-21,2024-03-04,
2024-03-15,2024-03-28,2024-04-15,2024-04-26,2024-05-09,2024-05-22,2024-06-05,2024-06-20,2024-07-03,2024-07-15,
2024-07-30,2024-08-12,2024-08-23,2024-09-05,2024-09-18,2024-10-01,2024-10-14,2024-10-24,2024-11-06,2024-11-18,
2024-11-29,2024-12-12,2024-12-24,2025-01-07,2025-01-17,2025-01-30,2025-02-13,2025-02-25,2025-03-10,2025-03-21
"""


def _holdout_rows() -> list[dict[str, Any]]:
    rows = []
    for line in HOLDOUT_ROWS_CSV.splitlines():
        feature_date, target_date, actual_class = line.split(",")
        rows.append(
            {
                "ticker": tree_shap.TICKER,
                "feature_date": feature_date,
                "target_date": target_date,
                "actual_class": int(actual_class),
            }
        )
    assert len(rows) == tree_shap.ROW_COUNT
    assert tree_shap._canonical_json_sha256(rows) == tree_shap.HOLDOUT_ROWS_SHA256
    return rows


def _background_target_dates() -> list[str]:
    dates = BACKGROUND_TARGET_DATES_CSV.replace("\n", "").split(",")
    assert len(dates) == tree_shap.BACKGROUND_ROW_COUNT
    assert (
        tree_shap._canonical_json_sha256(dates)
        == tree_shap.BACKGROUND_TARGET_DATES_SHA256
    )
    return dates


def _holdout_dates() -> tuple[list[str], list[str]]:
    rows = _holdout_rows()
    return (
        [str(row["feature_date"]) for row in rows],
        [str(row["target_date"]) for row in rows],
    )


def _classification(index: int, *, perfect: bool = False) -> tuple[int, int, float]:
    # The normal fixture includes every confusion-matrix category. The perfect
    # fixture deliberately leaves false-positive/false-negative audit slots empty.
    actual = int(_holdout_rows()[index]["actual_class"])
    if perfect:
        predicted = actual
    elif index == 1:
        predicted = 1  # deterministic false positive
    elif index == 7:
        predicted = 0  # deterministic false negative
    else:
        predicted = actual
    # Deliberately asymmetric, exactly representable binary fractions keep the
    # signed global means away from a serialization-sensitive zero tie.
    probability_up = 0.75 if predicted == 1 else 0.3125
    return actual, predicted, probability_up


def _local_rows(
    *,
    perfect: bool = False,
    failing_model: str | None = None,
) -> pd.DataFrame:
    feature_dates, target_dates = _holdout_dates()
    records: list[dict[str, Any]] = []
    expected_value = 0.5
    for model_id in tree_shap.MODEL_ORDER:
        contract = tree_shap.EXPECTED_MODELS[model_id]
        for row_index, (feature_date, target_date) in enumerate(
            zip(feature_dates, target_dates, strict=True)
        ):
            actual, predicted, probability_up = _classification(
                row_index, perfect=perfect
            )
            probability_down = 1.0 - probability_up
            difference = probability_up - expected_value
            weight_total = sum(range(1, len(FEATURE_COLUMNS) + 1))
            contributions = {
                feature_name: difference
                * (len(FEATURE_COLUMNS) - feature_index)
                / weight_total
                for feature_index, feature_name in enumerate(FEATURE_COLUMNS)
            }
            if failing_model == model_id and row_index == 0:
                contributions[FEATURE_COLUMNS[-1]] += 1e-3
            reconstructed = expected_value + sum(contributions.values())
            additivity_error = reconstructed - probability_up
            order = sorted(
                FEATURE_COLUMNS,
                key=lambda feature: (-abs(contributions[feature]), feature),
            )
            ranks = {feature: rank for rank, feature in enumerate(order, start=1)}
            for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
                contribution = contributions[feature_name]
                records.append(
                    {
                        "schema_version": tree_shap.SCHEMA_VERSION,
                        "model_id": model_id,
                        "registered_model_id": contract["registered_model_id"],
                        "algorithm": contract["algorithm"],
                        "ticker": tree_shap.TICKER,
                        "feature_date": feature_date,
                        "target_date": target_date,
                        "predicted_class": predicted,
                        "probability_up": probability_up,
                        "probability_down": probability_down,
                        "actual_class": actual,
                        "output_space": contract["output_space"],
                        "expected_value": expected_value,
                        "direct_model_output": probability_up,
                        "reconstructed_model_output": reconstructed,
                        "additivity_error": additivity_error,
                        "feature_name": feature_name,
                        "feature_value": float(
                            row_index * len(FEATURE_COLUMNS) + feature_index
                        ),
                        "shap_value": contribution,
                        "absolute_contribution_rank": ranks[feature_name],
                        "contribution_direction": (
                            "increased_model_output"
                            if contribution > 0
                            else "decreased_model_output"
                            if contribution < 0
                            else "neutral"
                        ),
                    }
                )
    return pd.DataFrame(records, columns=tree_shap.LOCAL_COLUMNS)


def _explainer_contracts() -> list[dict[str, Any]]:
    return [
        {
            "schema_version": tree_shap.SCHEMA_VERSION,
            "model_id": model_id,
            "registered_model_id": contract["registered_model_id"],
            "explainer_type": "shap.TreeExplainer",
            "algorithm": contract["algorithm"],
            "estimator_class": contract["estimator_class"],
            "class_explained": 1,
            "estimator_class_order": [0, 1],
            "expected_value": 0.5,
            "output_space": "probability",
            "feature_perturbation": "interventional",
            "background_data_policy": (
                "100 evenly spaced rows from the chronological training split; "
                "original fitted median imputer only"
            ),
            "background_row_count": tree_shap.BACKGROUND_ROW_COUNT,
            "background_matrix_sha256": tree_shap.TRANSFORMED_BACKGROUND_SHA256,
            "shap_version": tree_shap.SHAP_PIN,
            "feature_count": len(FEATURE_COLUMNS),
            "row_count": tree_shap.ROW_COUNT,
            "additivity_tolerance": tree_shap.ADDITIVITY_TOLERANCES[
                contract["algorithm"]
            ],
        }
        for model_id, contract in tree_shap.EXPECTED_MODELS.items()
    ]


def _fixture_feature_matrix() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [
                float(row_index * len(FEATURE_COLUMNS) + feature_index)
                for feature_index in range(len(FEATURE_COLUMNS))
            ]
            for row_index in range(tree_shap.ROW_COUNT)
        ],
        columns=FEATURE_COLUMNS,
    )


def _fixture_transformed_holdout_sha256() -> str:
    return tree_shap._frame_sha256(_fixture_feature_matrix())


def _feature_contract(*, leaky: bool = False) -> dict[str, Any]:
    feature_dates, target_dates = _holdout_dates()
    holdout_rows = _holdout_rows()
    background_dates = _background_target_dates()
    columns = ["open", "target_next_day_up"] if leaky else list(FEATURE_COLUMNS)
    return {
        "schema_version": tree_shap.SCHEMA_VERSION,
        "ticker": tree_shap.TICKER,
        "holdout_terminology": tree_shap.HOLDOUT_TERM,
        "row_count": tree_shap.ROW_COUNT,
        "feature_date_count": tree_shap.ROW_COUNT,
        "target_date_count": tree_shap.ROW_COUNT,
        "feature_dates": feature_dates,
        "target_dates": target_dates,
        "holdout_rows": holdout_rows,
        "holdout_rows_sha256": tree_shap.HOLDOUT_ROWS_SHA256,
        "feature_value_space": "output of the original fitted median imputer",
        "preprocessing_refit": False,
        "background_policy": (
            "100 chronological training rows selected at evenly spaced deterministic "
            "indices; fitted preprocessing only"
        ),
        "background_target_dates": background_dates,
        "background_target_dates_sha256": (
            tree_shap.BACKGROUND_TARGET_DATES_SHA256
        ),
        "source_dataset_sha256": tree_shap.MODEL_ROWS_SHA256,
        "models": [
            {
                "model_id": model_id,
                "registered_model_id": contract["registered_model_id"],
                "ordered_feature_columns": columns,
                "feature_count": len(columns),
                "row_count": tree_shap.ROW_COUNT,
                "transformed_matrix_sha256": _fixture_transformed_holdout_sha256(),
                "target_columns_in_matrix": False,
                "actual_class_in_matrix": False,
                "future_values_in_matrix": False,
            }
            for model_id, contract in tree_shap.EXPECTED_MODELS.items()
        ],
    }


def _validate_fixture_feature_contract(
    payload: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    with patch.object(
        tree_shap,
        "TRANSFORMED_HOLDOUT_SHA256",
        _fixture_transformed_holdout_sha256(),
    ):
        return tree_shap._validate_feature_contract(payload)


def _source_references() -> dict[str, Any]:
    return {
        "schema_version": tree_shap.SCHEMA_VERSION,
        "sources": [
            {
                "role": "model_training",
                "run_id": tree_shap.TRAINING_RUN_ID,
                "integrity_root_name": "run_manifest.json",
                "integrity_root_sha256": tree_shap.TRAINING_RUN_SHA256,
                "source_fingerprint": tree_shap.TRAINING_SOURCE_FINGERPRINT,
                "dataset_sha256": tree_shap.MODEL_ROWS_SHA256,
            },
            {
                "role": "unified_comparison",
                "run_id": tree_shap.COMPARISON_RUN_ID,
                "integrity_root_name": "manifest.json",
                "integrity_root_sha256": tree_shap.COMPARISON_MANIFEST_SHA256,
            },
            {
                "role": "vertical_slice",
                "run_id": tree_shap.VERTICAL_SLICE_RUN_ID,
                "integrity_root_name": "manifest.json",
                "integrity_root_sha256": tree_shap.VERTICAL_SLICE_MANIFEST_SHA256,
            },
        ],
        "models": [
            {
                "model_id": model_id,
                "registered_model_id": contract["registered_model_id"],
                "algorithm": contract["algorithm"],
                "model_binary_sha256": contract["model_binary_sha256"],
                "model_manifest_sha256": contract["model_manifest_sha256"],
                "comparison_prediction_sha256": contract[
                    "comparison_prediction_sha256"
                ],
            }
            for model_id, contract in tree_shap.EXPECTED_MODELS.items()
        ],
        "portable_paths": True,
        "model_binaries_included": False,
    }


def _run_manifest() -> dict[str, Any]:
    return {
        "schema_version": tree_shap.SCHEMA_VERSION,
        "artifact_type": tree_shap.ARTIFACT_TYPE,
        "run_id": FIXTURE_RUN_ID,
        "status": "PASS",
        "created_at": "2026-10-24T12:00:00Z",
        "ticker": tree_shap.TICKER,
        "holdout_terminology": tree_shap.HOLDOUT_TERM,
        "model_ids": list(tree_shap.MODEL_ORDER),
        "registered_model_ids": [
            tree_shap.EXPECTED_MODELS[model_id]["registered_model_id"]
            for model_id in tree_shap.MODEL_ORDER
        ],
        "package_versions": {
            key: tree_shap.SHAP_PIN if key == "shap" else "fixture"
            for key in tree_shap.PACKAGE_VERSION_KEYS
        },
        "background_selection": (
            "100 chronological training rows selected at evenly spaced deterministic "
            "indices; fitted preprocessing only"
        ),
        "random_seed": None,
        "code_provenance": {
            "code_version": "fixture-code",
            "dirty_tree": False,
            "source_fingerprint": "fixture-source-fingerprint",
        },
        "source_references": "source_run_references.json",
        "explainer_contract": "explainer_contract.json",
        "feature_matrix_contract": "feature_matrix_contract.json",
        "final_summary": "final_summary.json",
        "disclaimer": tree_shap.ACADEMIC_DISCLAIMER,
    }


def _reseal(pack: Path) -> None:
    tree_shap._write_artifact_manifest(pack, run_id=pack.name)


def _verify_fixture_pack(pack: str | Path, **kwargs: Any) -> dict[str, Any]:
    with patch.object(
        tree_shap,
        "TRANSFORMED_HOLDOUT_SHA256",
        _fixture_transformed_holdout_sha256(),
    ):
        return tree_shap.verify_explainability_pack(pack, **kwargs)


def _build_pack(
    parent: Path,
    *,
    failing_model: str | None = None,
) -> Path:
    pack = parent / FIXTURE_RUN_ID
    pack.mkdir()
    for directory in ("local_explanations", "summaries", "plots"):
        (pack / directory).mkdir()

    local = _local_rows(failing_model=failing_model)
    contracts = _explainer_contracts()
    global_frame = tree_shap.compute_global_importance(local)
    cross = tree_shap.compute_cross_model_rankings(global_frame)
    audit = tree_shap._select_audit_rows(local)
    additivity = tree_shap._additivity_report(local, contracts)
    agreements = tree_shap._rank_agreements(global_frame)

    for model_id in tree_shap.MODEL_ORDER:
        tree_shap._write_csv(
            local.loc[local["model_id"] == model_id].reset_index(drop=True),
            pack / "local_explanations" / f"{model_id}.csv",
        )
        write_json(
            tree_shap._model_summary(
                model_id,
                local=local,
                global_frame=global_frame,
                audit=audit,
                agreements=agreements,
            ),
            pack / "summaries" / f"{model_id}.json",
        )
        (pack / "plots" / f"{model_id}-global-importance.png").write_bytes(
            PNG_FIXTURE
        )
    (pack / "plots" / "cross-model-rank-comparison.png").write_bytes(PNG_FIXTURE)

    tree_shap._write_csv(global_frame, pack / "global_feature_importance.csv")
    tree_shap._write_csv(cross, pack / "cross_model_feature_rankings.csv")
    tree_shap._write_csv(audit, pack / "audit_sample.csv")
    write_json(_source_references(), pack / "source_run_references.json")
    write_json(
        {
            "schema_version": tree_shap.SCHEMA_VERSION,
            "models": contracts,
            "output_space_rule": (
                "Attributions and direct outputs are positive-class probabilities. "
                "No raw-margin or log-odds values are mixed into this run."
            ),
        },
        pack / "explainer_contract.json",
    )
    write_json(_feature_contract(), pack / "feature_matrix_contract.json")
    write_json(additivity, pack / "additivity_report.json")
    write_json(
        {
            "schema_version": tree_shap.SCHEMA_VERSION,
            "status": "PASS",
            "checks": [
                {"check": check, "status": "PASS"}
                for check in tree_shap.QUALITY_CHECKS
            ],
            "disclaimer": tree_shap.ACADEMIC_DISCLAIMER,
        },
        pack / "quality_report.json",
    )
    versions = _run_manifest()["package_versions"]
    write_json(
        tree_shap._final_summary(
            run_id=FIXTURE_RUN_ID,
            versions=versions,
            global_frame=global_frame,
            additivity=additivity,
            audit=audit,
            agreements=agreements,
        ),
        pack / "final_summary.json",
    )
    write_json(_run_manifest(), pack / "run_manifest.json")
    _reseal(pack)
    return pack


def _write_sidecar(zip_path: Path) -> Path:
    digest = sha256_file(zip_path)
    sidecar = zip_path.with_name(f"{zip_path.name}.sha256")
    sidecar.write_text(f"{digest}  {zip_path.name}\n", encoding="ascii")
    return sidecar


def _canonical_zip_info(name: str, *, mode: int = 0o600) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def test_explainability_dependency_declares_exact_shap_pin():
    requirements = REPOSITORY_ROOT / "requirements-explainability.txt"
    assert requirements.is_file()
    declared = {
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert f"shap=={tree_shap.SHAP_PIN}" in declared
    assert tree_shap.SHAP_PIN == "0.54.1"


def test_runtime_version_gate_fails_closed_on_wrong_shap(monkeypatch):
    monkeypatch.setattr(
        tree_shap.importlib.metadata,
        "version",
        lambda distribution: "0.53.0" if distribution == "shap" else "1.0",
    )
    with pytest.raises(RuntimeError, match=r"require shap==0\.54\.1"):
        tree_shap._versions()


def test_authoritative_constants_and_model_identities_are_frozen():
    assert tree_shap.MODEL_ORDER == ("M-001", "M-002", "M-003", "M-004")
    assert tree_shap.ROW_COUNT == 184
    assert tree_shap.TICKER == "OGDC"
    assert tree_shap.HOLDOUT_TERM == "shared project holdout"
    assert tree_shap.TRAINING_RUN_ID == "TRN-20261020-191815-624662-nogit"
    assert tree_shap.COMPARISON_RUN_ID == "CMP-20261024-OGDC-price-only-02"
    assert tree_shap.VERTICAL_SLICE_RUN_ID == "VSL-20261020-OGDC-main-02"
    assert {
        model_id: contract["registered_model_id"]
        for model_id, contract in tree_shap.EXPECTED_MODELS.items()
    } == {
        "M-001": (
            "M-001-6ce88dc384de975e-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
        "M-002": (
            "M-002-5a0160ff1b7a8043-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
        "M-003": (
            "M-003-23efd4aed873bf08-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
        "M-004": (
            "M-004-6c60cc92f77a495a-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9"
        ),
    }
    assert [
        tree_shap.EXPECTED_MODELS[model_id]["algorithm"]
        for model_id in tree_shap.MODEL_ORDER
    ] == ["random_forest", "random_forest", "random_forest", "xgboost"]
    assert {
        contract["output_space"] for contract in tree_shap.EXPECTED_MODELS.values()
    } == {"probability"}
    assert tree_shap.ADDITIVITY_TOLERANCES == {
        "random_forest": 5e-5,
        "xgboost": 1e-6,
    }
    assert (
        tree_shap.TRANSFORMED_HOLDOUT_SHA256
        == "4add3d0aa5269e3df8f66f36311a9a53321066b67f38e1d715d847991ed26a46"
    )
    assert (
        tree_shap.TRANSFORMED_BACKGROUND_SHA256
        == "ea1242cd06841ca64a74774d329aab838d4e62677c97689f62760bd4f2ae48b7"
    )


def test_run_id_generation_is_utc_deterministic_and_validation_is_strict():
    instant = datetime(2026, 10, 24, 7, 8, 9, tzinfo=timezone.utc)
    assert (
        tree_shap.make_explainability_run_id(instant)
        == "XAI-20261024-070809-OGDC-tree-v1"
    )
    assert tree_shap._safe_run_id(FIXTURE_RUN_ID) == FIXTURE_RUN_ID
    with pytest.raises(ValueError, match="run_id must match"):
        tree_shap._safe_run_id("XAI-invalid")


def test_trusted_run_path_rejects_non_symlink_root_escape(tmp_path):
    trusted_root = tmp_path / "trusted-model-training"
    trusted_root.mkdir()
    outside_run = tmp_path / tree_shap.TRAINING_RUN_ID
    outside_run.mkdir()

    with pytest.raises(ValueError, match="outside its configured trusted root"):
        tree_shap._path_inside(
            outside_run,
            trusted_root,
            field="Model-training run",
        )


@pytest.mark.parametrize(
    ("value", "class_index", "expected"),
    [
        (0.25, 1, 0.25),
        ([0.25], 1, 0.25),
        ([0.75, 0.25], 1, 0.25),
        (np.array([[0.75, 0.25]]), 1, 0.25),
    ],
)
def test_expected_value_normalization_handles_binary_conventions(
    value, class_index, expected
):
    assert tree_shap.normalize_expected_value(value, class_index=class_index) == expected


@pytest.mark.parametrize(
    ("value", "class_index"), [([0.5, 0.5], 2), ([0.5, np.nan], 1)]
)
def test_expected_value_normalization_rejects_invalid_positive_class(
    value, class_index
):
    with pytest.raises(ValueError, match="positive-class|finite"):
        tree_shap.normalize_expected_value(value, class_index=class_index)


def test_positive_class_shap_normalizes_rf_list_and_xgb_2d_outputs():
    positive = np.arange(12, dtype=float).reshape(4, 3)
    negative = -positive
    list_result = tree_shap.normalize_positive_class_shap(
        [negative, positive], row_count=4, feature_count=3, class_index=1
    )
    matrix_result = tree_shap.normalize_positive_class_shap(
        positive, row_count=4, feature_count=3, class_index=1
    )
    np.testing.assert_array_equal(list_result, positive)
    np.testing.assert_array_equal(matrix_result, positive)


@pytest.mark.parametrize(
    ("estimator", "shape_convention"),
    [
        (
            RandomForestClassifier(
                n_estimators=5,
                max_depth=2,
                random_state=20261024,
                n_jobs=1,
            ),
            "binary_list",
        ),
        (
            XGBClassifier(
                n_estimators=5,
                max_depth=2,
                learning_rate=0.2,
                random_state=20261024,
                n_jobs=1,
                verbosity=0,
            ),
            "positive_matrix",
        ),
    ],
    ids=["random-forest", "xgboost"],
)
def test_tiny_fitted_tree_models_exercise_probability_and_shape_contract(
    estimator, shape_convention
):
    features = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.2, 0.8, 0.1, 0.9],
            [0.8, 0.2, 0.9, 0.1],
            [0.3, 0.7, 0.2, 0.8],
            [0.7, 0.3, 0.8, 0.2],
        ],
        dtype=float,
    )
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=int)
    estimator.fit(features, labels)

    positive_index = tree_shap._class_index(estimator.classes_)
    probabilities = np.asarray(estimator.predict_proba(features), dtype=float)
    assert probabilities.shape == (len(features), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)

    expected_value = float(probabilities[:, positive_index].mean())
    positive_contributions = np.zeros_like(features)
    positive_contributions[:, 0] = (
        probabilities[:, positive_index] - expected_value
    )
    raw_values = (
        [-positive_contributions, positive_contributions]
        if shape_convention == "binary_list"
        else positive_contributions
    )
    normalized = tree_shap.normalize_positive_class_shap(
        raw_values,
        row_count=len(features),
        feature_count=features.shape[1],
        class_index=positive_index,
    )
    reconstructed = expected_value + normalized.sum(axis=1)
    np.testing.assert_allclose(
        reconstructed,
        probabilities[:, positive_index],
        atol=1e-12,
        rtol=0,
    )


@pytest.mark.parametrize(
    "values",
    [
        lambda base: np.stack((-base, base), axis=2),
        lambda base: np.stack((-base, base), axis=0),
        lambda base: np.stack((-base, base), axis=1),
    ],
    ids=["rows-features-classes", "classes-rows-features", "rows-classes-features"],
)
def test_positive_class_shap_normalizes_supported_three_dimensional_layouts(values):
    positive = np.arange(12, dtype=float).reshape(4, 3)
    normalized = tree_shap.normalize_positive_class_shap(
        values(positive), row_count=4, feature_count=3, class_index=1
    )
    np.testing.assert_array_equal(normalized, positive)


def test_positive_class_shap_supports_explanation_values_attribute():
    expected = np.arange(12, dtype=float).reshape(4, 3)
    explanation = SimpleNamespace(values=expected)
    np.testing.assert_array_equal(
        tree_shap.normalize_positive_class_shap(
            explanation, row_count=4, feature_count=3, class_index=1
        ),
        expected,
    )


@pytest.mark.parametrize(
    "values",
    [
        np.arange(8, dtype=float).reshape(2, 2, 2),
        np.zeros((4, 2)),
        np.array([[1.0, np.nan, 2.0]] * 4),
        [],
    ],
    ids=["ambiguous", "wrong-feature-count", "non-finite", "empty-list"],
)
def test_positive_class_shap_rejects_ambiguous_malformed_or_nonfinite_values(values):
    with pytest.raises(ValueError):
        tree_shap.normalize_positive_class_shap(
            values, row_count=4 if np.asarray(values).size != 8 else 2,
            feature_count=3 if np.asarray(values).size != 8 else 2,
            class_index=1,
        )


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((5, 4, 3)),
        np.zeros((3, 5, 4)),
        np.zeros((5, 3, 4)),
    ],
    ids=["rows-features-classes", "classes-rows-features", "rows-classes-features"],
)
def test_positive_class_shap_rejects_nonbinary_three_dimensional_class_axes(values):
    with pytest.raises(ValueError, match="binary|two classes|shape"):
        tree_shap.normalize_positive_class_shap(
            values,
            row_count=5,
            feature_count=4,
            class_index=1,
        )


def test_positive_class_shap_rejects_a_three_class_list_output():
    class_values = np.zeros((5, 4))
    with pytest.raises(ValueError, match="exactly two classes"):
        tree_shap.normalize_positive_class_shap(
            [class_values, class_values, class_values],
            row_count=5,
            feature_count=4,
            class_index=1,
        )


def test_estimator_class_mapping_requires_exact_binary_order():
    assert tree_shap._class_index([0, 1]) == 1
    for classes in ([1, 0], [0.5, 1.5]):
        with pytest.raises(ValueError, match=r"exactly \[0, 1\]"):
            tree_shap._class_index(classes)


def test_explainer_rejects_fractional_fitted_estimator_classes_before_inference():
    estimator_type = type(
        "RandomForestClassifier",
        (),
        {
            "classes_": np.array([0.5, 1.5]),
            "predict_proba": lambda self, values: np.zeros((len(values), 2)),
        },
    )
    estimator = estimator_type()
    imputer = SimpleNamespace(transform=lambda frame: frame.to_numpy(float))
    pipeline = SimpleNamespace(
        named_steps={"imputer": imputer, "model": estimator},
        classes_=np.array([0, 1]),
    )
    contract = tree_shap.EXPECTED_MODELS["M-001"]
    manifest = {
        "variant_id": "M-001",
        "model_id": contract["registered_model_id"],
        "algorithm": contract["algorithm"],
        "target_definition": {
            "positive_class_condition": tree_shap.TARGET_DEFINITION
        },
        "prediction_horizon": {"value": 1, "unit": "trading_session"},
        "class_mapping": {"estimator_classes": [0, 1]},
        "feature_set": {"columns": list(FEATURE_COLUMNS)},
    }
    model_rows = pd.DataFrame(columns=FEATURE_COLUMNS)

    with pytest.raises(ValueError, match=r"class order.*(?:invalid|exactly \[0, 1\])"):
        tree_shap._explain_model(
            variant_id="M-001",
            pipeline=pipeline,
            manifest=manifest,
            model_rows=model_rows,
            background_rows=model_rows,
            predictions=pd.DataFrame(),
        )


def test_global_importance_recomputes_signed_metrics_counts_shares_and_ties():
    records: list[dict[str, Any]] = []
    feature_values = {"alpha": [2.0, -1.0, 0.0], "beta": [1.0, 1.0, -1.0]}
    for model_id in tree_shap.MODEL_ORDER:
        contract = tree_shap.EXPECTED_MODELS[model_id]
        for feature_name, values in feature_values.items():
            for value in values:
                records.append(
                    {
                        "model_id": model_id,
                        "registered_model_id": contract["registered_model_id"],
                        "algorithm": contract["algorithm"],
                        "output_space": contract["output_space"],
                        "feature_name": feature_name,
                        "shap_value": value,
                    }
                )
    result = tree_shap.compute_global_importance(pd.DataFrame(records))
    model = result.loc[result["model_id"] == "M-001"].set_index("feature_name")
    assert model.loc["alpha", "mean_absolute_shap_value"] == pytest.approx(1.0)
    assert model.loc["alpha", "signed_mean_shap_value"] == pytest.approx(1 / 3)
    assert model.loc["alpha", "median_absolute_shap_value"] == pytest.approx(1.0)
    assert model.loc["alpha", "maximum_absolute_shap_value"] == pytest.approx(2.0)
    assert model.loc["alpha", "positive_contribution_count"] == 1
    assert model.loc["alpha", "negative_contribution_count"] == 1
    assert model.loc["alpha", "share_of_total_absolute_attribution"] == pytest.approx(
        0.5
    )
    assert model.loc["alpha", "feature_rank"] == 1
    assert model.loc["beta", "feature_rank"] == 2
    assert model["share_of_total_absolute_attribution"].sum() == pytest.approx(1.0)


def test_cross_model_table_keeps_output_spaces_and_magnitudes_separate():
    rows: list[dict[str, Any]] = []
    for model_id in tree_shap.MODEL_ORDER:
        is_xgb = model_id == "M-004"
        for feature_name, rank in (("open", 2 if is_xgb else 1), ("close", 1 if is_xgb else 2)):
            rows.append(
                {
                    "model_id": model_id,
                    "feature_name": feature_name,
                    "feature_rank": rank,
                    "mean_absolute_shap_value": (
                        1000.0 + rank if is_xgb else 0.01 * rank
                    ),
                    "output_space": "raw_margin" if is_xgb else "probability",
                }
            )
    result = tree_shap.compute_cross_model_rankings(pd.DataFrame(rows)).set_index(
        "feature_name"
    )
    assert result.loc["open", "random_forest_mean_rank"] == 1.0
    assert result.loc["open", "M_004_rank"] == 2
    assert result.loc["open", "M_004_output_space"] == "raw_margin"
    assert result.loc["open", "M_004_mean_absolute_shap_value"] == 1002.0
    assert not any(column.startswith("all_model_mean_absolute") for column in result)


def test_generated_plots_include_the_exact_academic_disclaimer(tmp_path, monkeypatch):
    saved_figure_text: list[list[str]] = []

    def capture_figure_text(figure, _path, **_kwargs):
        saved_figure_text.append([item.get_text() for item in figure.texts])

    monkeypatch.setattr("matplotlib.figure.Figure.savefig", capture_figure_text)
    tree_shap._plot_global_importance(
        pd.DataFrame(
            [
                {
                    "model_id": "M-001",
                    "feature_name": "close",
                    "feature_rank": 1,
                    "mean_absolute_shap_value": 0.25,
                }
            ]
        ),
        tmp_path / "global.png",
        model_id="M-001",
    )
    tree_shap._plot_cross_model_ranks(
        pd.DataFrame(
            [
                {
                    "feature_name": "close",
                    "random_forest_mean_rank": 1.0,
                    "M_001_rank": 1,
                    "M_002_rank": 1,
                    "M_003_rank": 1,
                    "M_004_rank": 1,
                }
            ]
        ),
        tmp_path / "cross-model.png",
    )

    assert saved_figure_text == [
        [tree_shap.ACADEMIC_DISCLAIMER],
        [tree_shap.ACADEMIC_DISCLAIMER],
    ]


def test_audit_sample_is_deterministic_complete_and_date_tie_broken():
    local = _local_rows()
    first = tree_shap._select_audit_rows(local)
    second = tree_shap._select_audit_rows(local.sample(frac=1, random_state=17))
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == len(tree_shap.MODEL_ORDER) * len(tree_shap.AUDIT_CATEGORIES)
    assert set(first["category"]) == set(tree_shap.AUDIT_CATEGORIES)
    assert first["available"].all()
    feature_dates, target_dates = _holdout_dates()
    m1 = first.loc[first["model_id"] == "M-001"].set_index("category")
    assert m1.loc["highest_confidence_class_1", "target_date"] == target_dates[1]
    assert m1.loc["highest_confidence_class_0", "target_date"] == target_dates[0]
    assert m1.loc["false_positive", "target_date"] == target_dates[1]
    assert m1.loc["false_negative", "target_date"] == target_dates[7]
    assert m1.loc["largest_absolute_additivity_error", "feature_date"] == feature_dates[0]


def test_audit_sample_marks_unavailable_confusion_categories_explicitly():
    audit = tree_shap._select_audit_rows(_local_rows(perfect=True))
    unavailable = audit.loc[audit["category"].isin(["false_positive", "false_negative"])]
    assert not unavailable["available"].any()
    assert set(unavailable["unavailability_reason"]) == {
        "no_prediction_matches_category"
    }
    assert set(unavailable["target_date"]) == {""}


def test_prediction_collapse_rejects_rank_and_metadata_inconsistency():
    local = _local_rows().loc[lambda frame: frame["model_id"] == "M-001"].copy()
    bad_rank = local.copy()
    bad_rank.loc[bad_rank.index[:2], "absolute_contribution_rank"] = 1
    with pytest.raises(ValueError, match="complete permutation"):
        tree_shap._prediction_rows(bad_rank)

    bad_metadata = local.copy()
    bad_metadata.loc[bad_metadata.index[0], "probability_up"] = 0.8
    with pytest.raises(ValueError, match="metadata changes"):
        tree_shap._prediction_rows(bad_metadata)


def test_additivity_report_counts_passes_and_failures_without_false_pass():
    contracts = _explainer_contracts()
    passing = tree_shap._additivity_report(_local_rows(), contracts)
    assert passing["status"] == "PASS"
    assert all(record["row_count_failing_tolerance"] == 0 for record in passing["models"])

    failing = tree_shap._additivity_report(
        _local_rows(failing_model="M-004"), contracts
    )
    assert failing["status"] == "FAIL"
    by_model = {record["model_id"]: record for record in failing["models"]}
    assert by_model["M-004"]["row_count_failing_tolerance"] == 1
    assert by_model["M-004"]["maximum_absolute_error"] == pytest.approx(1e-3)


def test_feature_and_explainer_contracts_enforce_exact_nonleaky_columns():
    validated = _validate_fixture_feature_contract(_feature_contract())
    assert validated == {model_id: FEATURE_COLUMNS for model_id in tree_shap.MODEL_ORDER}

    with pytest.raises(ValueError, match="target, future, or metadata"):
        _validate_fixture_feature_contract(_feature_contract(leaky=True))

    contracts = tree_shap._validate_explainer_contract(
        {
            "schema_version": tree_shap.SCHEMA_VERSION,
            "models": _explainer_contracts(),
            "output_space_rule": (
                "Attributions and direct outputs are positive-class probabilities. "
                "No raw-margin or log-odds values are mixed into this run."
            ),
        }
    )
    assert [record["model_id"] for record in contracts] == list(tree_shap.MODEL_ORDER)
    raw_margin = deepcopy(contracts)
    raw_margin[-1]["output_space"] = "raw_margin"
    with pytest.raises(ValueError, match="[Ee]xplainer contract is invalid"):
        tree_shap._validate_explainer_contract(
            {
                "schema_version": tree_shap.SCHEMA_VERSION,
                "models": raw_margin,
                "output_space_rule": "mixed",
            }
        )


def test_feature_contract_rejects_incomplete_184_date_coverage():
    payload = _feature_contract()
    payload["feature_dates"].pop()
    payload["target_dates"].pop()
    payload["holdout_rows"].pop()

    with pytest.raises(ValueError, match="holdout contract is invalid"):
        _validate_fixture_feature_contract(payload)


def test_feature_contract_rejects_non_authoritative_feature_order():
    payload = _feature_contract()
    columns = payload["models"][0]["ordered_feature_columns"]
    columns[0], columns[1] = columns[1], columns[0]

    with pytest.raises(ValueError, match="M-001 feature-matrix contract is invalid"):
        _validate_fixture_feature_contract(payload)


def test_feature_contract_rejects_actual_class_in_explainer_columns():
    payload = _feature_contract()
    payload["models"][0]["ordered_feature_columns"][-1] = "actual_class"

    with pytest.raises(ValueError, match="target, future, or metadata"):
        _validate_fixture_feature_contract(payload)


@pytest.mark.parametrize(
    ("record_kind", "field", "value", "message"),
    [
        (
            "training",
            "integrity_root_name",
            "manifest.json",
            "Source dataset fingerprint is not authoritative",
        ),
        (
            "training",
            "source_fingerprint",
            "0" * 40,
            "Source dataset fingerprint is not authoritative",
        ),
        (
            "model",
            "comparison_prediction_sha256",
            "0" * 64,
            "M-001 source-model fingerprint is invalid",
        ),
        (
            "model",
            "registered_model_id",
            "M-001-not-the-registered-model",
            "M-001 source-model fingerprint is invalid",
        ),
    ],
)
def test_source_references_reject_non_authoritative_names_and_fingerprints(
    record_kind, field, value, message
):
    payload = _source_references()
    record = payload["sources"][0] if record_kind == "training" else payload["models"][0]
    record[field] = value
    with pytest.raises(ValueError, match=message):
        tree_shap._validate_source_references(payload)


def test_feature_contract_rejects_declared_and_content_holdout_fingerprint_changes():
    wrong_declaration = _feature_contract()
    wrong_declaration["holdout_rows_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="holdout contract is invalid"):
        _validate_fixture_feature_contract(wrong_declaration)

    changed_rows = _feature_contract()
    changed_rows["holdout_rows"][0]["actual_class"] = 1
    with pytest.raises(ValueError, match="authoritative holdout"):
        _validate_fixture_feature_contract(changed_rows)


def test_fixture_only_directory_pack_replays_to_pass(tmp_path):
    pack = _build_pack(tmp_path)
    result = _verify_fixture_pack(pack)
    assert result == {
        "status": "PASS",
        "run_id": FIXTURE_RUN_ID,
        "verified_file_count": len(tree_shap._expected_layout()),
        "models": list(tree_shap.MODEL_ORDER),
        "row_count_per_model": tree_shap.ROW_COUNT,
        "shap_version": tree_shap.SHAP_PIN,
        "disclaimer": tree_shap.ACADEMIC_DISCLAIMER,
        "pack": FIXTURE_RUN_ID,
        "path_base": "input_parent",
        "checksum_verified": False,
        "zip_sha256": None,
        "verification_scope": "portable_evidence",
    }


def test_verifier_rejects_checksum_tamper_and_resealed_semantic_tamper(tmp_path):
    pack = _build_pack(tmp_path)
    summary = pack / "final_summary.json"
    summary.write_text(summary.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        _verify_fixture_pack(pack)

    write_json(
        tree_shap._final_summary(
            run_id=FIXTURE_RUN_ID,
            versions=_run_manifest()["package_versions"],
            global_frame=pd.read_csv(pack / "global_feature_importance.csv"),
            additivity=json.loads((pack / "additivity_report.json").read_text()),
            audit=pd.read_csv(pack / "audit_sample.csv", keep_default_na=False),
            agreements=tree_shap._rank_agreements(
                pd.read_csv(pack / "global_feature_importance.csv")
            ),
        ),
        summary,
    )
    global_path = pack / "global_feature_importance.csv"
    global_frame = pd.read_csv(global_path)
    global_frame.loc[0, "mean_absolute_shap_value"] += 0.25
    global_frame.to_csv(global_path, index=False, lineterminator="\n")
    _reseal(pack)
    with pytest.raises(ValueError, match="Global feature importance numeric column"):
        _verify_fixture_pack(pack)


def test_verifier_rejects_unexpected_files_even_when_inventory_is_resealed(tmp_path):
    pack = _build_pack(tmp_path)
    (pack / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    _reseal(pack)
    with pytest.raises(ValueError, match="layout is incomplete or unexpected"):
        _verify_fixture_pack(pack)


def test_verifier_rejects_extra_quality_report_fields_after_reseal(tmp_path):
    pack = _build_pack(tmp_path)
    quality_path = pack / "quality_report.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    quality["undeclared"] = "not allowed"
    write_json(quality, quality_path)
    _reseal(pack)
    with pytest.raises(ValueError, match="quality report is invalid"):
        _verify_fixture_pack(pack)


def test_verifier_rejects_wrong_local_columns_even_when_inventory_is_resealed(tmp_path):
    pack = _build_pack(tmp_path)
    path = pack / "local_explanations" / "M-001.csv"
    frame = pd.read_csv(path)
    frame = frame[[*frame.columns[1:], frame.columns[0]]]
    frame.to_csv(path, index=False, lineterminator="\n")
    _reseal(pack)
    with pytest.raises(ValueError, match="local explanation columns are invalid"):
        _verify_fixture_pack(pack)


def test_verifier_recomputes_transformed_matrix_fingerprint_from_local_values(tmp_path):
    pack = _build_pack(tmp_path)
    path = pack / "local_explanations" / "M-001.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "feature_value"] += 1.0
    frame.to_csv(path, index=False, lineterminator="\n")
    _reseal(pack)
    with pytest.raises(ValueError, match="transformed feature matrix is inconsistent"):
        _verify_fixture_pack(pack)


def test_verifier_rejects_resealed_additivity_failure_before_accepting_pass(tmp_path):
    pack = _build_pack(tmp_path, failing_model="M-001")
    with pytest.raises(ValueError, match="additivity exceeds"):
        _verify_fixture_pack(pack)


@pytest.mark.parametrize(
    ("filename", "payload", "message"),
    [
        ("notes.txt", "/Users/example/private/input.csv", "absolute host path"),
        ("notes.txt", "the feature caused the stock to rise", "causal claim"),
        ("notes.txt", "this output is financial advice", "advice wording"),
        ("notes.txt", "api_key=not-a-real-secret-value", "credential data"),
        ("notes.txt", "prefix:/tmp/secret", "host-path or credential data"),
        ("model.joblib", "not really a model", "executable model"),
    ],
)
def test_portability_scan_rejects_paths_claims_advice_and_model_payloads(
    tmp_path, filename, payload, message
):
    (tmp_path / filename).write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        tree_shap._assert_portable_content(tmp_path)


def test_portability_scan_allows_only_the_exact_academic_disclaimer(tmp_path):
    (tmp_path / "disclaimer.txt").write_text(
        tree_shap.ACADEMIC_DISCLAIMER, encoding="utf-8"
    )
    tree_shap._assert_portable_content(tmp_path)


def test_portability_scan_rejects_invalid_png_and_symlink(tmp_path):
    (tmp_path / "plot.png").write_bytes(b"not-a-png")
    with pytest.raises(ValueError, match="valid PNG"):
        tree_shap._assert_portable_content(tmp_path)

    (tmp_path / "plot.png").unlink()
    target = tmp_path / "target.txt"
    target.write_text("safe", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        tree_shap._assert_portable_content(tmp_path)


@pytest.mark.parametrize("corruption", ["trailing-bytes", "crc"], ids=str)
def test_portability_scan_rejects_png_trailing_bytes_and_crc_corruption(
    tmp_path, corruption
):
    payload = bytearray(PNG_FIXTURE)
    if corruption == "trailing-bytes":
        payload.extend(b"unexpected-trailing-data")
    else:
        payload[-1] ^= 0x01
    (tmp_path / "plot.png").write_bytes(payload)

    with pytest.raises(ValueError, match="PNG"):
        tree_shap._assert_portable_content(tmp_path)


def test_portability_scan_rejects_invalid_png_scanline_filter(tmp_path):
    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x05\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )
    (tmp_path / "plot.png").write_bytes(payload)
    with pytest.raises(ValueError, match="scanline filter"):
        tree_shap._assert_portable_content(tmp_path)


def test_portability_scan_rejects_an_oversized_directory_file(tmp_path):
    oversized = tmp_path / "oversized.txt"
    with oversized.open("wb") as stream:
        stream.seek(tree_shap._MAX_ARTIFACT_FILE_BYTES)
        stream.write(b"x")

    with pytest.raises(ValueError, match="size|large|limit"):
        tree_shap._assert_portable_content(tmp_path)


@pytest.mark.parametrize(
    "text",
    [
        "/Users/a/file",
        "/home/a/file",
        "x,/tmp/secret",
        "x=(/tmp/secret)",
        r"C:\Users\a\file",
        "C:/Users/a/file",
        r"x,C:\Users\a\file",
        "x=(D:/data/private.csv)",
        r"\\server\share\file.csv",
        r"\\?\C:\Users\a\file.csv",
        r"\\.\PhysicalDrive0",
        "file:///tmp/a",
    ],
)
def test_absolute_host_path_detection_covers_embedded_posix_windows_unc_and_uris(text):
    assert tree_shap._contains_absolute_path(text)


@pytest.mark.parametrize("path", ["/absolute", "../escape", "a/../b", r"a\\b"])
def test_relative_path_and_inventory_parsing_reject_unsafe_paths(path):
    with pytest.raises(ValueError, match="relative POSIX path"):
        tree_shap._safe_relative(path, field="fixture")
    inventory = {
        "algorithm": "sha256",
        "root": ".",
        "excludes": ["artifact_manifest.json"],
        "files": [{"path": path, "sha256": "a" * 64}],
    }
    with pytest.raises(ValueError):
        tree_shap._parse_inventory(inventory)


def test_inventory_parser_rejects_unsorted_duplicate_and_invalid_digest_entries():
    base = {
        "algorithm": "sha256",
        "root": ".",
        "excludes": ["artifact_manifest.json"],
    }
    with pytest.raises(ValueError, match="sorted"):
        tree_shap._parse_inventory(
            {
                **base,
                "files": [
                    {"path": "b", "sha256": "b" * 64},
                    {"path": "a", "sha256": "a" * 64},
                ],
            }
        )
    with pytest.raises(ValueError, match="unique"):
        tree_shap._parse_inventory(
            {
                **base,
                "files": [
                    {"path": "a", "sha256": "a" * 64},
                    {"path": "a", "sha256": "b" * 64},
                ],
            }
        )
    with pytest.raises(ValueError, match="digest"):
        tree_shap._parse_inventory(
            {**base, "files": [{"path": "a", "sha256": "not-a-digest"}]}
        )


def test_valid_deterministic_zip_and_sidecar_verify_offline(tmp_path):
    pack = _build_pack(tmp_path)
    archive = tmp_path / f"{FIXTURE_RUN_ID}.zip"
    digest = tree_shap._create_deterministic_zip(
        pack, archive, run_id=FIXTURE_RUN_ID
    )
    sidecar = _write_sidecar(archive)
    assert sidecar.read_text(encoding="ascii") == f"{digest}  {archive.name}\n"
    result = _verify_fixture_pack(archive)
    assert result["status"] == "PASS"
    assert result["checksum_verified"] is True
    assert result["zip_sha256"] == digest


def test_directory_checksum_verification_binds_exact_sibling_zip_contents(tmp_path):
    pack = _build_pack(tmp_path)
    archive = tmp_path / f"{FIXTURE_RUN_ID}.zip"
    tree_shap._create_deterministic_zip(pack, archive, run_id=FIXTURE_RUN_ID)
    sidecar = _write_sidecar(archive)

    quality_report = pack / "quality_report.json"
    quality_report.write_text(
        quality_report.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    _reseal(pack)

    with pytest.raises(
        ValueError, match="directory contents do not match the checksummed ZIP"
    ):
        _verify_fixture_pack(pack, checksum=sidecar)


def test_deterministic_zip_bytes_are_reproducible(tmp_path):
    pack = _build_pack(tmp_path)
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    first_digest = tree_shap._create_deterministic_zip(
        pack, first, run_id=FIXTURE_RUN_ID
    )
    second_digest = tree_shap._create_deterministic_zip(
        pack, second, run_id=FIXTURE_RUN_ID
    )
    assert first_digest == second_digest
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("placement", ["prepended", "trailing"])
def test_authenticated_zip_rejects_bytes_outside_the_canonical_archive(
    tmp_path, placement
):
    pack = _build_pack(tmp_path)
    archive = tmp_path / f"{FIXTURE_RUN_ID}.zip"
    tree_shap._create_deterministic_zip(pack, archive, run_id=FIXTURE_RUN_ID)
    canonical = archive.read_bytes()
    marker = b"noncanonical-bytes"
    archive.write_bytes(
        marker + canonical if placement == "prepended" else canonical + marker
    )
    _write_sidecar(archive)

    with pytest.raises(ValueError, match="ZIP"):
        _verify_fixture_pack(archive)


def test_canonical_zip_rejects_bytes_hidden_after_a_member_deflate_stream(tmp_path):
    archive_path = tmp_path / "hidden-member-bytes.zip"
    member_name = f"{FIXTURE_RUN_ID}/file.txt"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(_canonical_zip_info(member_name), b"hello")
    payload = bytearray(archive_path.read_bytes())
    eocd_offset = len(payload) - 22
    _, _, _, _, _, central_size, central_offset, _ = struct.unpack_from(
        "<4s4H2LH", payload, eocd_offset
    )
    compressed_size = struct.unpack_from("<L", payload, 18)[0]
    filename_length, extra_length = struct.unpack_from("<2H", payload, 26)
    data_end = 30 + filename_length + extra_length + compressed_size
    marker = b"SECRET-MODEL-BYTES"
    payload[data_end:data_end] = marker
    delta = len(marker)
    struct.pack_into("<L", payload, 18, compressed_size + delta)
    shifted_central = central_offset + delta
    struct.pack_into(
        "<L", payload, shifted_central + 20, compressed_size + delta
    )
    shifted_eocd = eocd_offset + delta
    struct.pack_into("<L", payload, shifted_eocd + 12, central_size)
    struct.pack_into("<L", payload, shifted_eocd + 16, shifted_central)

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        _, members = tree_shap._validated_zip_members(archive)
        with pytest.raises(ValueError, match="member stream is not canonical"):
            tree_shap._validate_canonical_zip_bytes(bytes(payload), members)


def test_zip_checksum_sidecar_is_mandatory_strict_and_tamper_evident(tmp_path):
    pack = _build_pack(tmp_path)
    archive = tmp_path / f"{FIXTURE_RUN_ID}.zip"
    tree_shap._create_deterministic_zip(pack, archive, run_id=FIXTURE_RUN_ID)
    with pytest.raises(FileNotFoundError, match="requires its SHA-256 sidecar"):
        _verify_fixture_pack(archive)

    sidecar = archive.with_name(f"{archive.name}.sha256")
    sidecar.write_text(f"{'0' * 64} *{archive.name}\n", encoding="ascii")
    with pytest.raises(ValueError, match="strict SHA-256 record"):
        _verify_fixture_pack(archive)

    sidecar.write_text(f"{'0' * 64}  {archive.name}\n", encoding="ascii")
    with pytest.raises(ValueError, match="does not match its sidecar"):
        _verify_fixture_pack(archive)


@pytest.mark.parametrize(
    "member_name",
    [
        f"{FIXTURE_RUN_ID}/../escape.txt",
        "/absolute/path.txt",
        f"{FIXTURE_RUN_ID}/a\\b.txt",
    ],
)
def test_zip_member_validation_rejects_traversal_absolute_and_backslash_paths(
    tmp_path, member_name
):
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(_canonical_zip_info(member_name), b"unsafe")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="relative POSIX path"):
            tree_shap._validated_zip_members(archive)


def test_zip_member_validation_rejects_noncanonical_permissions(tmp_path):
    archive_path = tmp_path / "mode.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            _canonical_zip_info(f"{FIXTURE_RUN_ID}/file.txt", mode=0o644),
            b"unsafe mode",
        )
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="metadata is not canonical"):
            tree_shap._validated_zip_members(archive)


def test_zip_member_validation_rejects_noncanonical_dos_attributes(tmp_path):
    archive_path = tmp_path / "attributes.zip"
    info = _canonical_zip_info(f"{FIXTURE_RUN_ID}/file.txt")
    info.external_attr |= 0x01
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(info, b"unsafe DOS attribute")
    with zipfile.ZipFile(archive_path) as archive:
        with pytest.raises(ValueError, match="metadata is not canonical"):
            tree_shap._validated_zip_members(archive)


def test_authenticated_zip_rejects_an_unexpected_member(tmp_path):
    pack = _build_pack(tmp_path)
    archive_path = tmp_path / f"{FIXTURE_RUN_ID}.zip"
    tree_shap._create_deterministic_zip(
        pack, archive_path, run_id=FIXTURE_RUN_ID
    )
    with zipfile.ZipFile(archive_path, "a") as archive:
        archive.writestr(
            _canonical_zip_info(f"{FIXTURE_RUN_ID}/zz-unexpected.txt"),
            b"unexpected",
        )
    _write_sidecar(archive_path)
    with pytest.raises(ValueError, match="inventory must exactly match files"):
        _verify_fixture_pack(archive_path)


def _fake_sources(tmp_path: Path) -> SimpleNamespace:
    roots = []
    for name in ("training", "comparison", "vertical"):
        root = tmp_path / name
        root.mkdir()
        roots.append(root)
    return SimpleNamespace(
        training_root=roots[0], comparison_root=roots[1], vertical_root=roots[2]
    )


def _stub_build_prelude(monkeypatch, sources: SimpleNamespace) -> None:
    monkeypatch.setattr(
        tree_shap,
        "_versions",
        lambda: {"python": "fixture", "shap": tree_shap.SHAP_PIN},
    )
    monkeypatch.setattr(tree_shap, "_validate_sources", lambda **kwargs: sources)


def test_build_refuses_existing_destination_without_touching_it(tmp_path, monkeypatch):
    sources = _fake_sources(tmp_path)
    _stub_build_prelude(monkeypatch, sources)
    output = tmp_path / "output"
    output.mkdir()
    existing = output / FIXTURE_RUN_ID
    existing.mkdir()
    sentinel = existing / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing collision"):
        tree_shap.build_tree_shap_explainability(
            model_training_run="unused",
            comparison_run="unused",
            vertical_slice_run="unused",
            output_dir=output,
            run_id=FIXTURE_RUN_ID,
            create_zip=False,
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not any(".staging-" in child.name for child in output.iterdir())


def test_failed_build_removes_staging_and_publishes_no_false_pass(
    tmp_path, monkeypatch
):
    sources = _fake_sources(tmp_path)
    _stub_build_prelude(monkeypatch, sources)
    monkeypatch.setattr(
        tree_shap,
        "_load_holdout",
        lambda evidence: (_ for _ in ()).throw(RuntimeError("fixture failure")),
    )
    output = tmp_path / "output"

    with pytest.raises(RuntimeError, match="fixture failure"):
        tree_shap.build_tree_shap_explainability(
            model_training_run="unused",
            comparison_run="unused",
            vertical_slice_run="unused",
            output_dir=output,
            run_id=FIXTURE_RUN_ID,
            create_zip=True,
        )
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not (output / FIXTURE_RUN_ID).exists()
    assert not (output / f"{FIXTURE_RUN_ID}.zip").exists()
    assert not (output / f"{FIXTURE_RUN_ID}.zip.sha256").exists()


def test_staged_verification_failure_precedes_publication_and_cleans_up(
    tmp_path, monkeypatch
):
    sources = _fake_sources(tmp_path)
    sources.training = SimpleNamespace()
    _stub_build_prelude(monkeypatch, sources)
    all_local = _local_rows()
    contracts = {
        contract["model_id"]: contract for contract in _explainer_contracts()
    }
    explanations = {
        model_id: tree_shap.ModelExplanation(
            variant_id=model_id,
            registered_model_id=tree_shap.EXPECTED_MODELS[model_id][
                "registered_model_id"
            ],
            algorithm=tree_shap.EXPECTED_MODELS[model_id]["algorithm"],
            estimator_class=tree_shap.EXPECTED_MODELS[model_id]["estimator_class"],
            output_space="probability",
            expected_value=0.5,
            tolerance=tree_shap.ADDITIVITY_TOLERANCES[
                tree_shap.EXPECTED_MODELS[model_id]["algorithm"]
            ],
            feature_columns=FEATURE_COLUMNS,
            transformed_matrix_sha256=_fixture_transformed_holdout_sha256(),
            local_rows=all_local.loc[
                all_local["model_id"] == model_id
            ].reset_index(drop=True),
            contract=contracts[model_id],
        )
        for model_id in tree_shap.MODEL_ORDER
    }
    feature_dates, target_dates = _holdout_dates()
    holdout_contract = {
        "feature_dates": feature_dates,
        "target_dates": target_dates,
        "holdout_rows": _holdout_rows(),
        "background_target_dates": _background_target_dates(),
        "background_policy": (
            "100 chronological training rows selected at evenly spaced deterministic "
            "indices; fitted preprocessing only"
        ),
    }
    monkeypatch.setattr(
        tree_shap,
        "_load_holdout",
        lambda _sources: (
            pd.DataFrame(),
            pd.DataFrame(),
            {model_id: pd.DataFrame() for model_id in tree_shap.MODEL_ORDER},
            holdout_contract,
        ),
    )
    monkeypatch.setattr(tree_shap, "load_model_artifact", lambda *args, **kwargs: (object(), {}))
    monkeypatch.setattr(
        tree_shap,
        "_explain_model",
        lambda *, variant_id, **kwargs: explanations[variant_id],
    )
    monkeypatch.setattr(
        tree_shap, "assert_training_run_artifacts_unchanged", lambda _training: None
    )
    monkeypatch.setattr(
        tree_shap, "_source_references", lambda *args, **kwargs: _source_references()
    )
    monkeypatch.setattr(
        tree_shap,
        "_feature_matrix_contract",
        lambda **kwargs: _feature_contract(),
    )
    monkeypatch.setattr(
        tree_shap,
        "_plot_global_importance",
        lambda _frame, path, **kwargs: path.write_bytes(PNG_FIXTURE),
    )
    monkeypatch.setattr(
        tree_shap,
        "_plot_cross_model_ranks",
        lambda _frame, path: path.write_bytes(PNG_FIXTURE),
    )
    monkeypatch.setattr(
        tree_shap,
        "code_provenance",
        lambda _root: {
            "code_version": "fixture-code",
            "dirty_tree": False,
            "source_fingerprint": "fixture-source-fingerprint",
        },
    )
    output = tmp_path / "output"
    final_dir = output / FIXTURE_RUN_ID
    final_zip = output / f"{FIXTURE_RUN_ID}.zip"
    final_sidecar = output / f"{FIXTURE_RUN_ID}.zip.sha256"
    observed_staging_paths: list[Path] = []

    def fail_staged_verification(pack, **_kwargs):
        staged_pack = Path(pack)
        observed_staging_paths.append(staged_pack)
        assert ".staging-" in staged_pack.parent.name
        assert staged_pack.name == FIXTURE_RUN_ID
        assert staged_pack.is_dir()
        assert not final_dir.exists()
        assert not final_zip.exists()
        assert not final_sidecar.exists()
        raise RuntimeError("forced staged verification failure")

    monkeypatch.setattr(
        tree_shap, "verify_explainability_pack", fail_staged_verification
    )

    with pytest.raises(RuntimeError, match="forced staged verification failure"):
        tree_shap.build_tree_shap_explainability(
            model_training_run="unused",
            comparison_run="unused",
            vertical_slice_run="unused",
            output_dir=output,
            run_id=FIXTURE_RUN_ID,
            create_zip=True,
        )

    assert len(observed_staging_paths) == 1
    assert output.is_dir()
    assert list(output.iterdir()) == []
    assert not final_dir.exists()
    assert not final_zip.exists()
    assert not final_sidecar.exists()


def test_recursive_private_permissions_cover_nested_pack(tmp_path):
    root = tmp_path / "pack"
    nested = root / "nested"
    nested.mkdir(parents=True)
    artifact = nested / "artifact.txt"
    artifact.write_text("fixture", encoding="utf-8")
    root.chmod(0o755)
    nested.chmod(0o755)
    artifact.chmod(0o644)

    tree_shap._make_tree_private(root)

    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(nested.stat().st_mode) == 0o700
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_exclusive_file_publication_refuses_collision(tmp_path):
    staged = tmp_path / "staged.zip"
    destination = tmp_path / "final.zip"
    staged.write_bytes(b"new")
    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        tree_shap._publish_staged_file(staged, destination)
    assert destination.read_bytes() == b"existing"
    assert staged.read_bytes() == b"new"


def test_publication_rollback_refuses_to_unlink_a_replaced_inode(tmp_path):
    destination = tmp_path / "published.zip"
    original = tmp_path / "original.zip"
    destination.write_bytes(b"published")
    metadata = destination.lstat()
    identity = (metadata.st_dev, metadata.st_ino, metadata.st_size)

    destination.rename(original)
    destination.write_bytes(b"replacement")

    with pytest.raises(
        RuntimeError, match="Refusing to roll back a changed publication target"
    ):
        tree_shap._rollback_published_file(destination, identity=identity)

    assert destination.read_bytes() == b"replacement"
    assert original.read_bytes() == b"published"


def test_json_reader_rejects_nonfinite_constants(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="Non-finite JSON constant"):
        tree_shap._read_json(path, field="fixture")
