"""Reproducible, immutable OGDC model-experiment training.

The module keeps model fitting separate from portfolio decisions. It reuses
PAISA's existing daily technical features, assigns rows to chronological
splits by ``target_date``, and emits prediction records that can be passed
directly to the backtesting command.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from paisa.experiments.artifacts import (
    detect_runtime_environment,
    get_git_commit_sha,
    load_model_artifact,
    make_run_id,
    make_zip_from_directory,
    package_versions,
    sha256_file,
    sha256_frame,
    to_jsonable,
    utc_now_z,
    write_json,
    write_model_artifact,
)
from paisa.experiments.evaluation import (
    PREDICTION_COLUMNS,
    binary_classification_metrics,
    make_prediction_frame,
    make_reference_baselines,
    select_provisional_champion,
    validate_prediction_frame,
)
from paisa.experiments.experiment_config import (
    DEFAULT_PROVIDER,
    DEFAULT_START,
    DEFAULT_SYMBOL,
    FEATURE_SET_ID,
    MODEL_VARIANTS,
    PREDICTION_HORIZON,
    RANDOM_SEED,
    SOURCE_FREQUENCY,
    TARGET_NAME,
    TARGET_POSITIVE_CLASS_CONDITION,
    ModelVariant,
    get_variant,
)
from paisa.experiments.training_features import (
    DatasetSplits,
    TECHNICAL_FEATURE_COLUMNS,
    chronological_target_split,
    prepare_model_frame,
)
from paisa.experiments.training_targets import add_next_session_target
from paisa.features import build_features
from paisa.pipeline import run_v0_pipeline


@dataclass(frozen=True, slots=True)
class PreparedTrainingData:
    """Shared model rows and split boundaries for every algorithm variant."""

    prices: pd.DataFrame
    features: pd.DataFrame
    model_frame: pd.DataFrame
    splits: DatasetSplits
    feature_columns: tuple[str, ...]
    dataset_sha256: str
    symbol: str


@dataclass(frozen=True, slots=True)
class FittedVariant:
    """A fitted estimator plus the identity and inputs needed for evaluation."""

    variant: ModelVariant
    model_id: str
    pipeline: Pipeline
    feature_columns: tuple[str, ...]
    model_configuration: dict[str, Any]


@dataclass(frozen=True, slots=True)
class VariantEvaluation:
    """Contract predictions and metrics for one chronological split."""

    split_name: str
    predictions: pd.DataFrame
    metrics: dict[str, Any]


def set_deterministic_seeds(seed: int = RANDOM_SEED) -> None:
    """Seed Python and NumPy; estimators receive the same explicit seed."""

    random.seed(seed)
    np.random.seed(seed)


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalise_prices(prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    required = {"date", "open", "close", "volume"}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"Price input is missing required columns: {missing}")

    ticker = symbol.strip().upper()
    if not ticker:
        raise ValueError("symbol must be non-empty")
    frame = prices.copy()
    if "ticker" not in frame.columns:
        frame["ticker"] = ticker
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame = frame[frame["ticker"] == ticker].copy()
    if frame.empty:
        raise ValueError(f"No price rows found for {ticker}")
    if "source" not in frame.columns:
        frame["source"] = "input-csv"

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in ("open", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    if frame[["date", "open", "close", "volume"]].isna().any().any():
        raise ValueError("Price input contains missing or non-finite required values")
    if (frame[["open", "close"]] <= 0).any().any() or (frame["volume"] < 0).any():
        raise ValueError("Prices must be positive and volume must be non-negative")
    frame = frame.sort_values("date", kind="mergesort").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Price input contains duplicate trading dates")
    return frame


def prepare_training_data(
    prices: pd.DataFrame,
    symbol: str = DEFAULT_SYMBOL,
) -> PreparedTrainingData:
    """Build the common leakage-safe feature rows and target-date splits.

    This is the reusable data-preparation entry point for the CPU models and
    the optional LSTM notebook. No learned transform is fitted here.
    """

    normalised = _normalise_prices(prices, symbol)
    features = build_features(normalised)
    targeted = add_next_session_target(features, ticker=symbol)
    model_frame = prepare_model_frame(
        targeted,
        ticker=symbol,
        feature_columns=TECHNICAL_FEATURE_COLUMNS,
    )
    splits = chronological_target_split(model_frame)
    fingerprint_frame = model_frame.copy()
    for column in ("feature_date", "target_date"):
        fingerprint_frame[column] = pd.to_datetime(
            fingerprint_frame[column]
        ).dt.strftime("%Y-%m-%d")
    return PreparedTrainingData(
        prices=normalised,
        features=targeted,
        model_frame=model_frame,
        splits=splits,
        feature_columns=tuple(TECHNICAL_FEATURE_COLUMNS),
        dataset_sha256=sha256_frame(fingerprint_frame),
        symbol=symbol.strip().upper(),
    )


def _build_estimator(variant: ModelVariant, random_state: int) -> Any:
    params = dict(variant.model_configuration)
    params["random_state"] = random_state
    if variant.algorithm == "random_forest":
        return RandomForestClassifier(**params)
    if variant.algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - clean-install diagnostic
            raise RuntimeError(
                "XGBoost is required for M-004. Install requirements-ml.txt."
            ) from exc
        return XGBClassifier(**params)
    raise ValueError(f"Unsupported algorithm: {variant.algorithm}")


def fit_variant(
    variant: ModelVariant,
    train_rows: pd.DataFrame,
    *,
    model_id: str | None = None,
    feature_columns: Sequence[str] = TECHNICAL_FEATURE_COLUMNS,
    random_state: int = RANDOM_SEED,
) -> FittedVariant:
    """Fit one variant on training rows only, with train-fitted imputation."""

    columns = tuple(feature_columns)
    missing = sorted((set(columns) | {TARGET_NAME}) - set(train_rows.columns))
    if missing:
        raise ValueError(f"Training rows are missing required columns: {missing}")
    target = train_rows[TARGET_NAME].astype(int)
    if target.nunique() != 2:
        raise ValueError("Training target must contain both classes")

    set_deterministic_seeds(random_state)
    estimator = _build_estimator(variant, random_state)
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )
    pipeline.fit(train_rows[list(columns)], target)
    complete_configuration = to_jsonable(estimator.get_params(deep=False))
    return FittedVariant(
        variant=variant,
        model_id=model_id or variant.model_id,
        pipeline=pipeline,
        feature_columns=columns,
        model_configuration=complete_configuration,
    )


def evaluate_variant(
    fitted: FittedVariant,
    rows: pd.DataFrame,
    split_name: str,
    symbol: str = DEFAULT_SYMBOL,
) -> VariantEvaluation:
    """Evaluate a frozen fitted variant without exposing labels to inference."""

    probabilities = fitted.pipeline.predict_proba(rows[list(fitted.feature_columns)])
    classes = [int(value) for value in fitted.pipeline.classes_]
    if classes != [0, 1]:
        raise AssertionError(f"Unexpected estimator class order: {classes}")
    probability_up = probabilities[:, classes.index(1)]
    predicted = (probability_up > (1.0 - probability_up)).astype(int)
    predictions = make_prediction_frame(
        rows=rows,
        ticker=symbol,
        model_id=fitted.model_id,
        split_name=split_name,
        y_pred=predicted,
        probability_up=probability_up,
        target_column=TARGET_NAME,
    )
    predictions = validate_prediction_frame(predictions)
    metrics = binary_classification_metrics(
        rows[TARGET_NAME].astype(int).to_numpy(),
        predictions["predicted_class"].astype(int).to_numpy(),
        predictions["probability_up"].astype(float).to_numpy(),
    )
    return VariantEvaluation(
        split_name=split_name,
        predictions=predictions,
        metrics=metrics,
    )


def _variant_model_id(
    variant: ModelVariant,
    prepared: PreparedTrainingData,
    random_state: int,
    run_id: str,
) -> str:
    semantic_payload = {
        "variant": variant.to_dict(),
        "features": list(prepared.feature_columns),
        "dataset_sha256": prepared.dataset_sha256,
        "target": TARGET_POSITIVE_CLASS_CONDITION,
        "seed": random_state,
    }
    semantic_digest = hashlib.sha256(
        json.dumps(to_jsonable(semantic_payload), sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]
    return f"{variant.model_id}-{semantic_digest}-{run_digest}"


def _date_string(series: pd.Series, which: str) -> str:
    values = pd.to_datetime(series, errors="raise")
    value = values.min() if which == "min" else values.max()
    return value.strftime("%Y-%m-%d")


def _training_dates(prepared: PreparedTrainingData) -> dict[str, str]:
    splits = prepared.splits
    return {
        "data_feature_start": _date_string(prepared.model_frame["feature_date"], "min"),
        "data_feature_end": _date_string(prepared.model_frame["feature_date"], "max"),
        "train_target_start": _date_string(splits.train["target_date"], "min"),
        "train_target_end": _date_string(splits.train["target_date"], "max"),
        "validation_target_start": _date_string(splits.validation["target_date"], "min"),
        "validation_target_end": _date_string(splits.validation["target_date"], "max"),
        "test_target_start": _date_string(splits.test["target_date"], "min"),
        "test_target_end": _date_string(splits.test["target_date"], "max"),
    }


def _git_dirty(repo: Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _source_fingerprint(repo: Path, code_version: str, dirty: bool) -> str:
    if not dirty:
        return code_version
    digest = hashlib.sha256()
    roots = [repo / "src" / "paisa", repo / "scripts"]
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            digest.update(path.relative_to(repo).as_posix().encode("utf-8"))
            digest.update(path.read_bytes())
    for notebook in (
        repo / "notebooks" / "PAISA_OGDC_Model_Training.ipynb",
        repo / "notebooks" / "PAISA_OGDC_LSTM_Training.ipynb",
    ):
        if notebook.is_file():
            digest.update(notebook.relative_to(repo).as_posix().encode("utf-8"))
            digest.update(notebook.read_bytes())
    return digest.hexdigest()


def _feature_importance(fitted: FittedVariant) -> pd.DataFrame:
    estimator = fitted.pipeline.named_steps["model"]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        return pd.DataFrame(columns=["feature", "importance", "rank"])
    frame = pd.DataFrame(
        {"feature": fitted.feature_columns, "importance": np.asarray(values, dtype=float)}
    )
    frame["rank"] = frame["importance"].rank(
        ascending=False, method="dense"
    ).astype(int)
    return frame.sort_values(["rank", "feature"], kind="mergesort").reset_index(
        drop=True
    )


def _confusion_frame(metrics_by_split: dict[str, dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for split_name in ("train", "validation", "test"):
        matrix = metrics_by_split[split_name]["confusion_matrix"]
        for actual in (0, 1):
            records.append(
                {
                    "split": split_name,
                    "actual_class": actual,
                    "predicted_down": int(matrix[actual][0]),
                    "predicted_up": int(matrix[actual][1]),
                }
            )
    return pd.DataFrame(records)


def _leaderboard(
    fitted_models: Sequence[FittedVariant],
    evaluations: dict[str, dict[str, VariantEvaluation]],
    split_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fitted in fitted_models:
        metrics = evaluations[fitted.model_id][split_name].metrics
        records.append(
            {
                "model_id": fitted.model_id,
                "variant_id": fitted.variant.model_id,
                "label": fitted.variant.label,
                "algorithm": fitted.variant.algorithm,
                **{
                    name: metrics.get(name)
                    for name in (
                        "support",
                        "accuracy",
                        "balanced_accuracy",
                        "precision",
                        "recall",
                        "f1",
                        "matthews_corrcoef",
                        "roc_auc",
                        "brier_score",
                    )
                },
            }
        )
    return pd.DataFrame(records).sort_values(
        ["balanced_accuracy", "f1", "accuracy", "model_id"],
        ascending=[False, False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def _select_variants(model_ids: Iterable[str] | None) -> tuple[ModelVariant, ...]:
    if model_ids is None:
        return tuple(MODEL_VARIANTS)
    requested = list(model_ids)
    if not requested:
        raise ValueError("model_ids must contain at least one variant")
    if len(requested) != len(set(requested)):
        raise ValueError("model_ids must not contain duplicates")
    return tuple(get_variant(model_id) for model_id in requested)


def _safe_run_id(value: str) -> str:
    run_id = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError(
            "run_id must start with a letter or digit and contain only letters, "
            "digits, '.', '_', and '-'"
        )
    return run_id


def _load_or_fetch_prices(
    *,
    repo: Path,
    symbol: str,
    start: str,
    provider: str,
    prices_path: str | Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    if prices_path is not None:
        source_path = Path(prices_path).expanduser().resolve()
        prices = pd.read_csv(source_path)
        provenance_provider = "input-csv"
        if "source" in prices.columns:
            sources = sorted(
                str(value) for value in prices["source"].dropna().unique()
            )
            if sources:
                provenance_provider = ",".join(sources)
        acquisition = {
            "mode": "existing_prices_csv",
            "path": str(source_path),
            "provider": provenance_provider,
        }
    else:
        acquisition = run_v0_pipeline(
            symbols=[symbol],
            provider_name=provider,
            start=date.fromisoformat(start),
            train=False,
            data_dir=repo / "data",
            model_dir=repo / "models",
        )
        source_path = repo / "data" / "processed" / "psx_prices.csv"
        prices = pd.read_csv(source_path)
        provenance_provider = str(acquisition.get("provider", provider))

    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    prices = prices[prices["date"].dt.date >= date.fromisoformat(start)].copy()
    if prices.empty:
        raise ValueError(f"No prices remain at or after requested start date {start}")
    return prices, acquisition, provenance_provider


def _baseline_hashes(repo: Path) -> dict[str, str]:
    paths = (
        repo / "models" / "baseline_random_forest.joblib",
        repo / "models" / "baseline_metrics.json",
        repo / "models" / "baseline_feature_importance.csv",
        repo / "models" / "baseline_confusion_matrix.csv",
    )
    return {
        path.relative_to(repo).as_posix(): sha256_file(path)
        for path in paths
        if path.exists()
    }


def run_training_stage(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START,
    provider: str = DEFAULT_PROVIDER,
    output_dir: str | Path = "artifacts/model_training",
    random_state: int = RANDOM_SEED,
    repo_root: str | Path | None = None,
    prices_path: str | Path | None = None,
    run_id: str | None = None,
    create_zip: bool = True,
    model_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Train and atomically register immutable OGDC model variants.

    If ``prices_path`` is omitted, the existing PSX pipeline fetches daily data.
    A provided CSV is useful for reproducible offline runs and CLI tests.
    """

    set_deterministic_seeds(random_state)
    repo = Path(repo_root).resolve() if repo_root else _repo_root_from_file()
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        output_root = repo / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_variants = _select_variants(model_ids)
    resolved_run_id = _safe_run_id(run_id or make_run_id(repo))
    final_run_dir = output_root / resolved_run_id
    if final_run_dir.exists():
        raise FileExistsError(
            f"Run directory already exists; refusing to overwrite: {final_run_dir}"
        )
    zip_path = output_root / f"{resolved_run_id}.zip"
    if create_zip and zip_path.exists():
        raise FileExistsError(f"ZIP already exists; refusing to overwrite: {zip_path}")

    baseline_before = _baseline_hashes(repo)
    prices, acquisition, provenance_provider = _load_or_fetch_prices(
        repo=repo,
        symbol=symbol.strip().upper(),
        start=start,
        provider=provider,
        prices_path=prices_path,
    )
    prepared = prepare_training_data(prices, symbol=symbol)
    environment = detect_runtime_environment()
    environment["package_versions"] = package_versions()
    code_version = get_git_commit_sha(repo) or "unknown"
    dirty_tree = _git_dirty(repo)
    source_fingerprint = _source_fingerprint(repo, code_version, dirty_tree)
    training_dates = _training_dates(prepared)

    stage_dir = Path(
        tempfile.mkdtemp(prefix=f".{resolved_run_id}-", dir=output_root)
    )
    try:
        (stage_dir / "frozen_input").mkdir()
        (stage_dir / "reports").mkdir()
        models_root = stage_dir / "models"
        models_root.mkdir()
        prepared.prices.to_csv(
            stage_dir / "frozen_input" / f"{prepared.symbol}_prices.csv", index=False
        )
        prepared.model_frame.to_csv(
            stage_dir / "frozen_input" / f"{prepared.symbol}_model_rows.csv",
            index=False,
        )

        split_manifest = {
            "schema_version": "1.0",
            "method": "chronological_target_date_70_15_15_no_shuffle",
            "boundaries": prepared.splits.boundaries.to_dict(),
            "row_counts": {
                name: int(len(frame))
                for name, frame in prepared.splits.as_dict().items()
            },
            "target_dates": {
                name: pd.to_datetime(frame["target_date"])
                .dt.strftime("%Y-%m-%d")
                .tolist()
                for name, frame in prepared.splits.as_dict().items()
            },
        }
        data_manifest = {
            "schema_version": "1.0",
            "symbol": prepared.symbol,
            "provider": provenance_provider,
            "requested_start": start,
            "actual_feature_start": training_dates["data_feature_start"],
            "actual_feature_end": training_dates["data_feature_end"],
            "price_rows": int(len(prepared.prices)),
            "model_rows": int(len(prepared.model_frame)),
            "dataset_sha256": prepared.dataset_sha256,
            "acquisition": acquisition,
        }
        write_json(split_manifest, stage_dir / "split_manifest.json")
        write_json(data_manifest, stage_dir / "data_manifest.json")
        write_json(environment, stage_dir / "reports" / "environment.json")

        fitted_models: list[FittedVariant] = []
        evaluations: dict[str, dict[str, VariantEvaluation]] = {}
        for variant in selected_variants:
            registered_id = _variant_model_id(
                variant, prepared, random_state, resolved_run_id
            )
            fitted = fit_variant(
                variant,
                prepared.splits.train,
                model_id=registered_id,
                feature_columns=prepared.feature_columns,
                random_state=random_state,
            )
            fitted_models.append(fitted)
            evaluations[registered_id] = {
                "train": evaluate_variant(
                    fitted, prepared.splits.train, "train", prepared.symbol
                ),
                "validation": evaluate_variant(
                    fitted,
                    prepared.splits.validation,
                    "validation",
                    prepared.symbol,
                ),
            }

        validation_leaderboard = _leaderboard(
            fitted_models, evaluations, "validation"
        )
        champion = select_provisional_champion(validation_leaderboard)
        champion["selection_data"] = "validation_only"
        champion["test_metrics_consulted"] = False
        write_json(
            champion, stage_dir / "reports" / "provisional_champion.json"
        )

        # Test evaluation happens only after the validation-only choice is frozen.
        for fitted in fitted_models:
            evaluations[fitted.model_id]["test"] = evaluate_variant(
                fitted, prepared.splits.test, "test", prepared.symbol
            )
        test_leaderboard = _leaderboard(fitted_models, evaluations, "test")

        validation_leaderboard.to_csv(
            stage_dir / "reports" / "validation_leaderboard.csv", index=False
        )
        test_leaderboard.to_csv(
            stage_dir / "reports" / "test_leaderboard.csv", index=False
        )
        baseline_frames = []
        for split_name, split_rows in (
            ("validation", prepared.splits.validation),
            ("test", prepared.splits.test),
        ):
            baseline = make_reference_baselines(
                split_rows,
                prepared.splits.train,
            )
            baseline.insert(0, "split", split_name)
            baseline_frames.append(baseline)
        pd.concat(baseline_frames, ignore_index=True).to_csv(
            stage_dir / "reports" / "baselines.csv", index=False
        )

        artifact_records: list[dict[str, Any]] = []
        prediction_paths: dict[str, str] = {}
        for fitted in fitted_models:
            per_split = evaluations[fitted.model_id]
            metrics = {
                name: per_split[name].metrics
                for name in ("train", "validation", "test")
            }
            all_predictions = []
            for split_name in ("train", "validation", "test"):
                frame = per_split[split_name].predictions.copy()
                frame["split"] = split_name
                all_predictions.append(frame)
            predictions_all = pd.concat(all_predictions, ignore_index=True)
            test_predictions = per_split["test"].predictions
            assert list(test_predictions.columns) == list(PREDICTION_COLUMNS)

            manifest = {
                "schema_version": "1.0",
                "model_id": fitted.model_id,
                "variant_id": fitted.variant.model_id,
                "algorithm": fitted.variant.algorithm,
                "model_configuration": fitted.model_configuration,
                "feature_set": {
                    "feature_set_id": FEATURE_SET_ID,
                    "columns": list(fitted.feature_columns),
                    "feature_code_version": source_fingerprint,
                    "transforms": [
                        {
                            "name": "median_imputer",
                            "parameters": {"strategy": "median"},
                            "fit_scope": "train_only",
                        }
                    ],
                },
                "training_dates": training_dates,
                "target_definition": {
                    "name": TARGET_NAME,
                    "positive_class_condition": TARGET_POSITIVE_CLASS_CONDITION,
                    "class_labels": {"0": "down_or_flat", "1": "up"},
                },
                "prediction_horizon": dict(PREDICTION_HORIZON),
                "random_seed": random_state,
                "metrics": metrics,
                "created_at": utc_now_z(),
                "symbols": [prepared.symbol],
                "data_provenance": {
                    "provider": provenance_provider,
                    "requested_symbols": [prepared.symbol],
                    "included_symbols": [prepared.symbol],
                    "requested_start": start,
                    "requested_end": _date_string(prepared.prices["date"], "max"),
                    "actual_feature_start": training_dates["data_feature_start"],
                    "actual_feature_end": training_dates["data_feature_end"],
                    "dataset_sha256": prepared.dataset_sha256,
                    "input_schema_version": "paisa-price-v0",
                    "price_adjustment": "unknown",
                    "dividends_included": False,
                    "corporate_actions_policy": "not_modeled",
                },
                "class_mapping": {
                    "estimator_classes": [0, 1],
                    "probability_fields_by_class": {
                        "0": "probability_down",
                        "1": "probability_up",
                    },
                },
                "code_version": code_version,
                "dirty_tree": dirty_tree,
                "source_fingerprint": source_fingerprint,
                "library_versions": environment["package_versions"],
            }
            configuration = fitted.variant.to_dict()
            configuration["registered_model_id"] = fitted.model_id
            configuration["model_configuration"] = fitted.model_configuration
            configuration["random_seed"] = random_state
            extra_files = {
                "config.json": configuration,
                "feature_columns.json": list(fitted.feature_columns),
                "metrics.json": metrics,
                "confusion_matrix.csv": _confusion_frame(metrics),
                "feature_importance.csv": _feature_importance(fitted),
                "predictions.csv": test_predictions,
                "predictions_all_splits.csv": predictions_all,
            }
            model_dir = write_model_artifact(
                models_root=models_root,
                model_id=fitted.model_id,
                artifact=fitted.pipeline,
                manifest=manifest,
                extra_files=extra_files,
            )
            _, stored_manifest = load_model_artifact(model_dir)
            artifact_records.append(
                {
                    "model_id": fitted.model_id,
                    "variant_id": fitted.variant.model_id,
                    "manifest_path": f"models/{fitted.model_id}/manifest.json",
                    "artifact_sha256": stored_manifest["artifact_sha256"],
                }
            )
            prediction_paths[fitted.variant.model_id] = (
                f"models/{fitted.model_id}/predictions.csv"
            )

        if _baseline_hashes(repo) != baseline_before:
            raise AssertionError("The legacy v0 baseline artifacts were modified")

        run_manifest = {
            "schema_version": "1.0",
            "run_id": resolved_run_id,
            "created_at": utc_now_z(),
            "symbol": prepared.symbol,
            "source_frequency": SOURCE_FREQUENCY,
            "target_definition": TARGET_POSITIVE_CLASS_CONDITION,
            "random_seed": random_state,
            "data_manifest": "data_manifest.json",
            "split_manifest": "split_manifest.json",
            "models": artifact_records,
            "provisional_champion": champion,
            "selection_policy": "validation metrics only; test evaluated after freeze",
            "prediction_paths": prediction_paths,
            "disclaimer": "Academic model output; not financial advice.",
        }
        write_json(run_manifest, stage_dir / "run_manifest.json")

        persisted_summary = to_jsonable({
            "run_id": resolved_run_id,
            "artifact_directory": str(final_run_dir),
            "model_ids": {
                fitted.variant.model_id: fitted.model_id for fitted in fitted_models
            },
            "prediction_paths": {
                variant_id: str(final_run_dir / relative_path)
                for variant_id, relative_path in prediction_paths.items()
            },
            "provisional_champion": champion,
            "validation_leaderboard": validation_leaderboard.to_dict(
                orient="records"
            ),
            "test_leaderboard": test_leaderboard.to_dict(orient="records"),
            "zip_path": str(zip_path) if create_zip else None,
            "disclaimer": "Academic model output; not financial advice.",
        })
        write_json(
            persisted_summary, stage_dir / "reports" / "final_summary.json"
        )

        stage_dir.rename(final_run_dir)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise

    if create_zip:
        make_zip_from_directory(final_run_dir, zip_path)
    return persisted_summary


__all__ = [
    "FittedVariant",
    "PreparedTrainingData",
    "VariantEvaluation",
    "evaluate_variant",
    "fit_variant",
    "prepare_training_data",
    "run_training_stage",
    "set_deterministic_seeds",
]
