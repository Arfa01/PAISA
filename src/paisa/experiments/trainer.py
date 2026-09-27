from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from paisa.experiments.artifacts import (
    detect_runtime_environment,
    ensure_dir,
    file_size_mb,
    get_git_commit_sha,
    make_run_id,
    make_zip_from_directory,
    package_versions,
    sha256_file,
    sha256_frame,
    split_rows_by_dates,
    to_jsonable,
    write_json,
)
from paisa.experiments.evaluation import (
    add_cumulative_performance,
    make_prediction_frame,
    make_reference_baselines,
    predictions_to_leaderboard,
    select_provisional_champion,
    summarize_predictions,
)
from paisa.experiments.experiment_config import (
    DEFAULT_PROVIDER,
    DEFAULT_START,
    DEFAULT_SYMBOL,
    MODEL_VARIANTS,
    PROBABILITY_STATUS,
    RANDOM_SEED,
    RF_CONFIGS,
    SOURCE_FREQUENCY,
    TARGET_DEFINITION,
    TARGET_NAME,
    TRAINING_MODE,
)
from paisa.experiments.training_features import (
    FEATURE_SETS,
    build_common_modelling_rows,
    get_feature_columns,
)
from paisa.experiments.training_targets import add_next_session_target, target_class_distribution
from paisa.pipeline import run_v0_pipeline


def set_deterministic_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[3]


def run_training_stage(
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START,
    provider: str = DEFAULT_PROVIDER,
    output_dir: str | Path = "artifacts/model_training",
    random_state: int = RANDOM_SEED,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the frozen OGDC Random Forest variant training stage.

    This function intentionally does not write to models/baseline_random_forest.joblib.
    Every variant is saved under a unique artifacts/model_training/<run_id>/ directory.
    """
    set_deterministic_seeds(random_state)
    repo = Path(repo_root).resolve() if repo_root else _repo_root_from_file()
    output_root = (repo / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()

    baseline_files = [
        repo / "models" / "baseline_random_forest.joblib",
        repo / "models" / "baseline_metrics.json",
        repo / "models" / "baseline_feature_importance.csv",
        repo / "models" / "baseline_confusion_matrix.csv",
    ]
    baseline_before = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in baseline_files
        if path.exists()
    }

    run_id = make_run_id(repo)
    run_dir = output_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists; refusing to overwrite: {run_dir}")

    dirs = {
        "root": run_dir,
        "frozen_input": run_dir / "frozen_input",
        "reports": run_dir / "reports",
        "plots": run_dir / "plots",
        "models": run_dir / "models",
    }
    for path in dirs.values():
        ensure_dir(path)

    environment = detect_runtime_environment()
    environment["package_versions"] = package_versions()
    write_json(environment, dirs["reports"] / "environment.json")

    acquisition = _run_real_psx_acquisition(repo, symbol=symbol, start=start, provider=provider)
    prices_path = repo / "data" / "processed" / "psx_prices.csv"
    features_path = repo / "data" / "processed" / "psx_features.csv"

    expected_absent = [str(p.relative_to(repo)) for p in [prices_path, features_path] if not p.exists()]
    if expected_absent:
        raise RuntimeError(f"PSX acquisition did not produce expected files: {expected_absent}")

    prices = pd.read_csv(prices_path)
    features = pd.read_csv(features_path)

    prices = prices[prices["ticker"].astype(str).str.upper() == symbol.upper()].copy()
    features = features[features["ticker"].astype(str).str.upper() == symbol.upper()].copy()

    if prices.empty or features.empty:
        raise RuntimeError(f"No {symbol.upper()} rows found after PSX acquisition.")

    frozen_prices_path = dirs["frozen_input"] / f"{symbol.upper()}_prices.csv"
    frozen_features_path = dirs["frozen_input"] / f"{symbol.upper()}_features_from_v0.csv"
    prices.to_csv(frozen_prices_path, index=False)
    features.to_csv(frozen_features_path, index=False)

    features_with_target = add_next_session_target(features, ticker=symbol)
    quality_report = _run_data_quality_checks(features_with_target, symbol=symbol)
    write_json(quality_report, run_dir / "data_quality_report.json")

    modelling_rows = build_common_modelling_rows(features_with_target, ticker=symbol)
    removal_count = int(modelling_rows.attrs.get("common_rows_removed_for_month_feature_or_target_completeness", 0))
    quality_report["removed_rows"]["month_feature_or_target_completeness"] = removal_count
    write_json(quality_report, run_dir / "data_quality_report.json")

    frozen_modelling_path = dirs["frozen_input"] / f"{symbol.upper()}_modelling_rows.csv"
    modelling_rows.to_csv(frozen_modelling_path, index=False)

    data_manifest = _build_data_manifest(
        prices=prices,
        modelling_rows=modelling_rows,
        prices_path=frozen_prices_path,
        symbol=symbol,
        provider=provider,
        requested_start=start,
        repo=repo,
    )
    write_json(data_manifest, run_dir / "data_manifest.json")

    split_manifest = make_chronological_split_manifest(modelling_rows)
    write_json(split_manifest, run_dir / "split_manifest.json")

    split_frames = {
        "train": split_rows_by_dates(modelling_rows, split_manifest["splits"]["train"]["dates"]),
        "validation": split_rows_by_dates(modelling_rows, split_manifest["splits"]["validation"]["dates"]),
        "test": split_rows_by_dates(modelling_rows, split_manifest["splits"]["test"]["dates"]),
    }
    validate_split_integrity(split_frames, split_manifest)

    model_results: dict[str, dict[str, Any]] = {}
    all_predictions: list[pd.DataFrame] = []
    artifact_manifests = []

    for variant in MODEL_VARIANTS:
        result = train_one_variant(
            variant=variant,
            split_frames=split_frames,
            symbol=symbol,
            run_dir=run_dir,
            data_manifest=data_manifest,
            split_manifest=split_manifest,
            random_state=random_state,
            environment=environment,
        )
        model_results[variant.model_id] = result["metrics_summary"]
        all_predictions.append(result["predictions"])
        artifact_manifests.append(result["artifact_manifest"])

    predictions_all = pd.concat(all_predictions, ignore_index=True)
    validation_leaderboard = predictions_to_leaderboard(model_results, split="validation")
    test_leaderboard = predictions_to_leaderboard(model_results, split="test")
    champion = select_provisional_champion(validation_leaderboard)

    validation_leaderboard.to_csv(dirs["reports"] / "validation_leaderboard.csv", index=False)
    test_leaderboard.to_csv(dirs["reports"] / "test_leaderboard.csv", index=False)
    write_json(champion, dirs["reports"] / "provisional_champion.json")

    baselines = make_reference_baselines(split_frames["test"], split_frames["train"])
    baselines.to_csv(dirs["reports"] / "baselines.csv", index=False)

    cumulative = add_cumulative_performance(predictions_all[predictions_all["split"] == "test"])
    cumulative.to_csv(dirs["reports"] / "cumulative_performance.csv", index=False)

    _make_all_plots(
        modelling_rows=modelling_rows,
        prices=prices,
        split_manifest=split_manifest,
        validation_leaderboard=validation_leaderboard,
        test_leaderboard=test_leaderboard,
        cumulative=cumulative,
        predictions_all=predictions_all,
        run_dir=run_dir,
    )

    run_manifest = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "repository_git_commit_sha": get_git_commit_sha(repo),
        "symbol": symbol.upper(),
        "provider": provider,
        "requested_start_date": start,
        "training_mode": TRAINING_MODE,
        "random_seed": random_state,
        "source_frequency": SOURCE_FREQUENCY,
        "target_name": TARGET_NAME,
        "target_definition": TARGET_DEFINITION,
        "probability_status": PROBABILITY_STATUS,
        "acquisition": acquisition,
        "data_manifest": "data_manifest.json",
        "split_manifest": "split_manifest.json",
        "data_quality_report": "data_quality_report.json",
        "model_variants": [variant.to_dict() for variant in MODEL_VARIANTS],
        "artifact_manifests": artifact_manifests,
        "provisional_champion": champion,
        "warnings": [
            "The PSX DPS source may not provide high and low values.",
            "Random Forest probabilities are uncalibrated and must not be described as guaranteed real-world confidence.",
            "This run uses technical PSX data only; no portfolio simulation, news, sentiment, fundamentals, reinforcement learning, or financial advice is included.",
        ],
    }
    write_json(run_manifest, run_dir / "run_manifest.json")

    pass_fail = validate_run_outputs(
        run_dir=run_dir,
        split_frames=split_frames,
        model_results=model_results,
        baseline_before=baseline_before,
        repo=repo,
    )
    write_json(pass_fail, dirs["reports"] / "final_pass_fail_checklist.json")
    if not pass_fail["overall_pass"]:
        run_manifest["status"] = "FAILED_VALIDATION"
        write_json(run_manifest, run_dir / "run_manifest.json")
        raise AssertionError("Training run failed validation checks. See reports/final_pass_fail_checklist.json.")

    zip_base = _zip_output_base(environment, repo)
    zip_path = zip_base / f"paisa_{run_id}.zip"
    zip_path = make_zip_from_directory(run_dir, zip_path)
    zip_info = {"path": str(zip_path), "size_mb": file_size_mb(zip_path)}
    write_json(zip_info, dirs["reports"] / "zip_info.json")

    final_summary = {
        "run_id": run_id,
        "git_commit_sha": get_git_commit_sha(repo),
        "data_date_range": data_manifest["date_range"],
        "common_modelling_rows": int(len(modelling_rows)),
        "train_validation_test_counts": {
            split: int(len(frame)) for split, frame in split_frames.items()
        },
        "class_distributions": {
            split: {str(k): int(v) for k, v in frame["target_next_session_up"].value_counts().sort_index().to_dict().items()}
            for split, frame in split_frames.items()
        },
        "model_labels": {variant.model_id: variant.label for variant in MODEL_VARIANTS},
        "validation_leaderboard": validation_leaderboard.to_dict(orient="records"),
        "provisional_champion": champion,
        "final_test_leaderboard": test_leaderboard.to_dict(orient="records"),
        "baseline_results": baselines.to_dict(orient="records"),
        "artifact_directory": str(run_dir),
        "zip_path": str(zip_path),
        "zip_size_mb": file_size_mb(zip_path),
        "test_pass_fail_status": pass_fail["overall_pass"],
        "important_warnings": run_manifest["warnings"],
        "final_statement": (
            "This run trained frozen Random Forest variants using technical PSX data only. "
            "It did not perform portfolio simulation, live trading, news analysis, fundamental analysis, "
            "reinforcement learning, or financial advice."
        ),
    }
    write_json(final_summary, dirs["reports"] / "final_summary.json")
    print(json.dumps(to_jsonable(final_summary), indent=2))
    print(final_summary["final_statement"])
    return final_summary


def _run_real_psx_acquisition(repo: Path, symbol: str, start: str, provider: str) -> dict[str, Any]:
    print(f"Running PAISA v0 acquisition with provider={provider}, symbol={symbol}, start={start}")
    try:
        metadata = run_v0_pipeline(
            symbols=[symbol.upper()],
            provider_name=provider,
            start=date.fromisoformat(start),
            train=False,
            data_dir=repo / "data",
            model_dir=repo / "models",
        )
    except Exception as exc:
        expected = [
            repo / "data" / "processed" / "psx_prices.csv",
            repo / "data" / "processed" / "psx_features.csv",
        ]
        absent = [str(path) for path in expected if not path.exists()]
        print("PSX acquisition failed. Training workflow is stopped.")
        print(f"Useful error: {type(exc).__name__}: {exc}")
        print(f"Expected files absent: {absent}")
        raise
    return metadata


def _build_data_manifest(
    prices: pd.DataFrame,
    modelling_rows: pd.DataFrame,
    prices_path: Path,
    symbol: str,
    provider: str,
    requested_start: str,
    repo: Path,
) -> dict[str, Any]:
    prices_copy = prices.copy()
    prices_copy["date"] = pd.to_datetime(prices_copy["date"], errors="raise")
    null_counts = {col: int(val) for col, val in prices_copy.isna().sum().to_dict().items()}
    duplicate_count = int(prices_copy["date"].duplicated().sum())
    return {
        "provider": provider,
        "ticker": symbol.upper(),
        "requested_start_date": requested_start,
        "actual_minimum_date": str(prices_copy["date"].min().date()),
        "actual_maximum_date": str(prices_copy["date"].max().date()),
        "date_range": {
            "start": str(prices_copy["date"].min().date()),
            "end": str(prices_copy["date"].max().date()),
        },
        "row_count": int(len(prices_copy)),
        "modelling_row_count": int(len(modelling_rows)),
        "column_names": list(prices_copy.columns),
        "null_counts": null_counts,
        "duplicate_date_count": duplicate_count,
        "data_file_sha256": sha256_file(prices_path),
        "modelling_rows_sha256": sha256_frame(modelling_rows),
        "repository_git_commit_sha": get_git_commit_sha(repo),
        "generation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "source_frequency": SOURCE_FREQUENCY,
        "warning": "High and low may be unavailable from the current PSX DPS source.",
    }


def _run_data_quality_checks(frame: pd.DataFrame, symbol: str) -> dict[str, Any]:
    data = frame.copy()
    removed_rows: dict[str, int] = {}

    data["date"] = pd.to_datetime(data["date"], errors="raise")
    data = data.sort_values("date").reset_index(drop=True)

    checks: dict[str, bool] = {}
    checks["date_parse_success"] = True
    checks["chronologically_sorted"] = data["date"].is_monotonic_increasing
    checks["unique_trading_dates"] = not data["date"].duplicated().any()
    checks["open_close_numeric_positive"] = bool(
        (pd.to_numeric(data["open"], errors="coerce") > 0).all()
        and (pd.to_numeric(data["close"], errors="coerce") > 0).all()
    )
    checks["volume_numeric_nonnegative"] = bool((pd.to_numeric(data["volume"], errors="coerce") >= 0).all())
    checks["enough_rows_for_20_day_features"] = len(data) > 40
    numeric = data.select_dtypes(include=[np.number])
    checks["no_infinite_values"] = bool(~np.isinf(numeric.to_numpy()).any())
    known_target = data.dropna(subset=["target_next_session_up"])
    checks["target_has_both_classes"] = bool(known_target["target_next_session_up"].astype(int).nunique() == 2)

    required_features = [*FEATURE_SETS["FEATURE_MONTH"].columns, "target_next_session_up"]
    modelling_candidate = data.replace([np.inf, -np.inf], np.nan).dropna(subset=required_features)
    removed_rows["missing_required_month_features_or_target"] = int(len(data) - len(modelling_candidate))
    checks["final_modelling_rows_have_no_missing_required_features"] = bool(
        not modelling_candidate[required_features].isna().any().any()
    )

    duplicate_date_count = int(data["date"].duplicated().sum())
    class_distribution = target_class_distribution(known_target)

    failed = [name for name, ok in checks.items() if not bool(ok)]
    if failed:
        raise AssertionError(f"Data-quality checks failed before training: {failed}")

    return {
        "ticker": symbol.upper(),
        "row_count_before_common_filter": int(len(data)),
        "duplicate_date_count": duplicate_date_count,
        "class_distribution_known_targets": class_distribution,
        "removed_rows": removed_rows,
        "checks": checks,
        "status": "PASS",
    }


def make_chronological_split_manifest(modelling_rows: pd.DataFrame) -> dict[str, Any]:
    ordered_dates = pd.to_datetime(modelling_rows["date"]).dt.strftime("%Y-%m-%d").drop_duplicates().tolist()
    if len(ordered_dates) < 60:
        raise ValueError("Not enough common modelling dates to create 70/15/15 split with purge boundaries.")

    n = len(ordered_dates)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    if train_end <= 2 or validation_end <= train_end + 2 or n <= validation_end + 2:
        raise ValueError("Split boundaries are too small after purge calculation.")

    train_purge = ordered_dates[train_end - 1]
    validation_purge = ordered_dates[validation_end - 1]

    train_dates = ordered_dates[: train_end - 1]
    validation_dates = ordered_dates[train_end : validation_end - 1]
    test_dates = ordered_dates[validation_end:]

    split_dates = {
        "train": train_dates,
        "validation": validation_dates,
        "test": test_dates,
    }
    purged_dates = [
        {"date": train_purge, "reason": "one-session purge between train and validation"},
        {"date": validation_purge, "reason": "one-session purge between validation and test"},
    ]

    splits: dict[str, Any] = {}
    for split_name, dates in split_dates.items():
        rows = split_rows_by_dates(modelling_rows, dates)
        splits[split_name] = {
            "dates": dates,
            "row_count": int(len(rows)),
            "start_date": dates[0],
            "end_date": dates[-1],
            "class_counts": {str(k): int(v) for k, v in rows["target_next_session_up"].value_counts().sort_index().to_dict().items()},
        }

    return {
        "split_method": "shared chronological date split with one feature-date purge at each boundary",
        "target_horizon": "one future trading session",
        "approximate_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
        "purged_dates": purged_dates,
        "splits": splits,
    }


def validate_split_integrity(split_frames: dict[str, pd.DataFrame], split_manifest: dict[str, Any]) -> None:
    train = split_frames["train"]
    validation = split_frames["validation"]
    test = split_frames["test"]

    for name, frame in split_frames.items():
        if frame.empty:
            raise AssertionError(f"{name} split is empty.")
        if frame["date"].duplicated().any():
            raise AssertionError(f"{name} split contains duplicate dates.")

    train_dates = set(pd.to_datetime(train["date"]).dt.strftime("%Y-%m-%d"))
    validation_dates = set(pd.to_datetime(validation["date"]).dt.strftime("%Y-%m-%d"))
    test_dates = set(pd.to_datetime(test["date"]).dt.strftime("%Y-%m-%d"))

    if train_dates & validation_dates or train_dates & test_dates or validation_dates & test_dates:
        raise AssertionError("Train/validation/test date overlap exists.")

    if not pd.to_datetime(train["date"]).max() < pd.to_datetime(validation["date"]).min():
        raise AssertionError("max(train feature date) must be < min(validation feature date).")
    if not pd.to_datetime(validation["date"]).max() < pd.to_datetime(test["date"]).min():
        raise AssertionError("max(validation feature date) must be < min(test feature date).")

    validation_block = validation_dates
    test_block = test_dates
    train_target_dates = set(pd.to_datetime(train["target_date"]).dt.strftime("%Y-%m-%d"))
    validation_target_dates = set(pd.to_datetime(validation["target_date"]).dt.strftime("%Y-%m-%d"))

    if train_target_dates & validation_block:
        raise AssertionError("Training target_date overlaps validation feature-date block; purge failed.")
    if validation_target_dates & test_block:
        raise AssertionError("Validation target_date overlaps test feature-date block; purge failed.")

    purged = {item["date"] for item in split_manifest["purged_dates"]}
    all_used = train_dates | validation_dates | test_dates
    if purged & all_used:
        raise AssertionError("Purged dates must not appear in any split.")


def train_one_variant(
    variant: Any,
    split_frames: dict[str, pd.DataFrame],
    symbol: str,
    run_dir: Path,
    data_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    random_state: int,
    environment: dict[str, Any],
) -> dict[str, Any]:
    feature_columns = get_feature_columns(variant.feature_set_name)
    feature_spec = FEATURE_SETS[variant.feature_set_name]
    rf_params = dict(RF_CONFIGS[variant.rf_config_name])
    rf_params["random_state"] = random_state

    train_rows = split_frames["train"]
    validation_rows = split_frames["validation"]

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(**rf_params)),
        ]
    )
    model.fit(train_rows[feature_columns], train_rows[TARGET_NAME].astype(int))

    predictions = []
    metrics_summary: dict[str, Any] = {
        "model_id": variant.model_id,
        "label": variant.label,
        "feature_set_name": variant.feature_set_name,
        "feature_set_metadata_label": feature_spec.metadata_label,
        "rf_config_name": variant.rf_config_name,
        "parent_model_id": variant.parent_model_id,
        "metrics": {},
    }

    for split_name, rows in split_frames.items():
        probability = model.predict_proba(rows[feature_columns])
        class_index = list(model.classes_).index(1)
        probability_up = probability[:, class_index]
        y_pred = (probability_up >= 0.5).astype(int)
        pred_frame = make_prediction_frame(
            rows=rows,
            ticker=symbol,
            model_id=variant.model_id,
            split_name=split_name,
            y_pred=y_pred,
            probability_up=probability_up,
        )
        predictions.append(pred_frame)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics_summary["metrics"] = summarize_predictions(prediction_frame)
    metrics_summary["train_to_validation_gaps"] = metrics_summary["metrics"].get("train_to_validation_gaps", {})

    model_dir = run_dir / "models" / variant.model_id
    ensure_dir(model_dir)
    model_path = model_dir / "model.joblib"

    artifact = {
        "pipeline": model,
        "model_id": variant.model_id,
        "label": variant.label,
        "parent_model_id": variant.parent_model_id,
        "algorithm": "RandomForestClassifier",
        "model_hyperparameters": rf_params,
        "feature_set_name": variant.feature_set_name,
        "feature_set_metadata_label": feature_spec.metadata_label,
        "ordered_feature_columns": feature_columns,
        "max_lookback_trading_days": variant.max_lookback_trading_days,
        "target_definition": TARGET_DEFINITION,
        "ticker": symbol.upper(),
        "data_cutoff": data_manifest["actual_maximum_date"],
        "data_sha256": data_manifest["modelling_rows_sha256"],
        "split_manifest_reference": "../../split_manifest.json",
        "random_seed": random_state,
        "training_mode": TRAINING_MODE,
        "probability_calibration_status": PROBABILITY_STATUS,
        "repository_git_commit_sha": data_manifest["repository_git_commit_sha"],
        "package_versions": environment["package_versions"],
    }
    joblib.dump(artifact, model_path)

    prediction_frame.to_csv(model_dir / "predictions.csv", index=False)
    write_json(variant.to_dict(), model_dir / "config.json")
    write_json(feature_columns, model_dir / "feature_columns.json")
    write_json(metrics_summary, model_dir / "metrics.json")

    confusion_records = []
    for split_name, split_metrics in metrics_summary["metrics"].items():
        if split_name == "train_to_validation_gaps":
            continue
        cm = split_metrics["confusion_matrix"]
        confusion_records.append({"split": split_name, "actual": 0, "predicted_0": cm[0][0], "predicted_1": cm[0][1]})
        confusion_records.append({"split": split_name, "actual": 1, "predicted_0": cm[1][0], "predicted_1": cm[1][1]})
    pd.DataFrame(confusion_records).to_csv(model_dir / "confusion_matrices.csv", index=False)

    importance = _feature_importance_frame(model, feature_columns, validation_rows)
    importance.to_csv(model_dir / "feature_importance.csv", index=False)

    artifact_manifest = {
        "model_id": variant.model_id,
        "model_dir": str(model_dir.relative_to(run_dir)),
        "files": sorted(path.name for path in model_dir.iterdir() if path.is_file()),
        "model_sha256": sha256_file(model_path),
        "prediction_rows": int(len(prediction_frame)),
        "training_mode": TRAINING_MODE,
        "probability_status": PROBABILITY_STATUS,
    }
    write_json(artifact_manifest, model_dir / "artifact_manifest.json")

    return {
        "predictions": prediction_frame,
        "metrics_summary": metrics_summary,
        "artifact_manifest": artifact_manifest,
    }


def _feature_importance_frame(model: Pipeline, feature_columns: list[str], validation_rows: pd.DataFrame) -> pd.DataFrame:
    rf = model.named_steps["model"]
    built_in = getattr(rf, "feature_importances_", np.zeros(len(feature_columns)))

    perm = permutation_importance(
        model,
        validation_rows[feature_columns],
        validation_rows[TARGET_NAME].astype(int),
        n_repeats=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scoring="balanced_accuracy",
    )

    frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "built_in_importance": built_in,
            "permutation_importance_mean": perm.importances_mean,
            "permutation_importance_std": perm.importances_std,
        }
    )
    frame["rank"] = frame["permutation_importance_mean"].rank(ascending=False, method="dense").astype(int)
    return frame.sort_values(["rank", "feature"]).reset_index(drop=True)


def validate_run_outputs(
    run_dir: Path,
    split_frames: dict[str, pd.DataFrame],
    model_results: dict[str, dict[str, Any]],
    baseline_before: dict[str, str],
    repo: Path,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    common_dates = {
        split: pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d").tolist()
        for split, frame in split_frames.items()
    }
    target_values = {
        split: frame["target_next_session_up"].astype(int).tolist()
        for split, frame in split_frames.items()
    }

    add("data_dates_sorted_unique", all(frame["date"].is_monotonic_increasing and not frame["date"].duplicated().any() for frame in split_frames.values()))
    validate_split_integrity(split_frames, json.loads((run_dir / "split_manifest.json").read_text(encoding="utf-8")))
    add("no_train_validation_test_overlap", True)
    add("boundary_purging_applied", True)

    model_dirs = [run_dir / "models" / model_id for model_id in model_results]
    add("no_two_model_ids_share_artifact_dir", len({str(p) for p in model_dirs}) == len(model_dirs))

    required_model_files = {
        "model.joblib",
        "config.json",
        "feature_columns.json",
        "metrics.json",
        "predictions.csv",
        "confusion_matrices.csv",
        "feature_importance.csv",
        "artifact_manifest.json",
    }
    missing_by_model = {}
    reload_checks = {}
    same_dates_checks = {}
    same_targets_checks = {}
    split_date_checks = {}

    for model_id, model_dir in zip(model_results, model_dirs):
        existing_files = {path.name for path in model_dir.iterdir() if path.is_file()}
        missing = sorted(required_model_files - existing_files)
        missing_by_model[model_id] = missing

        artifact = joblib.load(model_dir / "model.joblib")
        predictions = pd.read_csv(model_dir / "predictions.csv")
        features = json.loads((model_dir / "feature_columns.json").read_text(encoding="utf-8"))
        add(f"{model_id}_no_target_or_next_columns_in_features", not any(c.startswith("target_") or c.startswith("next_") or c == "target_date" for c in features))

        first_test = predictions[predictions["split"] == "test"].head(5)
        if not first_test.empty:
            test_rows = split_frames["test"].head(len(first_test))
            proba = artifact["pipeline"].predict_proba(test_rows[features])
            class_index = list(artifact["pipeline"].classes_).index(1)
            reproduced = proba[:, class_index]
            reload_checks[model_id] = bool(np.allclose(reproduced, first_test["probability_up"].to_numpy(), rtol=1e-10, atol=1e-10))
        else:
            reload_checks[model_id] = False

        for split, dates in common_dates.items():
            model_dates = predictions[predictions["split"] == split]["feature_date"].tolist()
            same_dates_checks[f"{model_id}_{split}"] = model_dates == dates
            model_targets = predictions[predictions["split"] == split]["actual_class"].astype(int).tolist()
            same_targets_checks[f"{model_id}_{split}"] = model_targets == target_values[split]
            split_date_checks[f"{model_id}_{split}"] = model_dates == dates

    add("every_model_directory_has_required_files", all(not v for v in missing_by_model.values()), json.dumps(missing_by_model))
    add("reloaded_models_reproduce_first_five_test_probabilities", all(reload_checks.values()), json.dumps(reload_checks))
    add("rerun_same_snapshot_reproduces_probabilities_proxy", all(reload_checks.values()), "Saved/reloaded pipeline reproduces stored probabilities for the first five test rows.")
    add("all_model_variants_use_identical_sample_dates", all(same_dates_checks.values()), json.dumps(same_dates_checks))
    add("all_model_variants_use_identical_target_values", all(same_targets_checks.values()), json.dumps(same_targets_checks))
    add("split_dates_identical_across_variants", all(split_date_checks.values()), json.dumps(split_date_checks))

    target_date_after_feature = all((pd.to_datetime(frame["target_date"]) > pd.to_datetime(frame["date"])).all() for frame in split_frames.values())
    add("target_date_after_feature_date", target_date_after_feature)

    baseline_after = {
        str(path.relative_to(repo)): sha256_file(path)
        for path in [
            repo / "models" / "baseline_random_forest.joblib",
            repo / "models" / "baseline_metrics.json",
            repo / "models" / "baseline_feature_importance.csv",
            repo / "models" / "baseline_confusion_matrix.csv",
        ]
        if path.exists()
    }
    add("existing_baseline_files_not_overwritten", baseline_before == baseline_after)

    overall = all(item["passed"] for item in checks.values())
    return {"overall_pass": overall, "checks": checks}


def _make_all_plots(
    modelling_rows: pd.DataFrame,
    prices: pd.DataFrame,
    split_manifest: dict[str, Any],
    validation_leaderboard: pd.DataFrame,
    test_leaderboard: pd.DataFrame,
    cumulative: pd.DataFrame,
    predictions_all: pd.DataFrame,
    run_dir: Path,
) -> None:
    plots_dir = ensure_dir(run_dir / "plots")

    prices_plot = prices.copy()
    prices_plot["date"] = pd.to_datetime(prices_plot["date"])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prices_plot["date"], prices_plot["close"], label="Close")
    for split, alpha in [("train", 0.08), ("validation", 0.12), ("test", 0.16)]:
        dates = pd.to_datetime(split_manifest["splits"][split]["dates"])
        ax.axvspan(dates.min(), dates.max(), alpha=alpha, label=split)
    for item in split_manifest["purged_dates"]:
        ax.axvline(pd.to_datetime(item["date"]), linestyle="--", linewidth=1, label=f"purge {item['date']}")
    ax.set_title("OGDC closing-price history with chronological split regions")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / "price_and_splits.png", dpi=160)
    plt.close(fig)

    class_rows = []
    for split, meta in split_manifest["splits"].items():
        for cls, count in meta["class_counts"].items():
            class_rows.append({"split": split, "class": cls, "count": count})
    class_df = pd.DataFrame(class_rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    class_df.pivot(index="split", columns="class", values="count").plot(kind="bar", ax=ax)
    ax.set_title("Target class distribution by split")
    ax.set_xlabel("Split")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(plots_dir / "class_distribution.png", dpi=160)
    plt.close(fig)

    _leaderboard_plot(validation_leaderboard, "Validation model comparison", plots_dir / "validation_model_comparison.png")
    _leaderboard_plot(test_leaderboard, "Final test model comparison", plots_dir / "test_model_comparison.png")

    if not cumulative.empty:
        for metric, filename, title in [
            ("cumulative_accuracy", "cumulative_accuracy.png", "Cumulative test accuracy over time"),
            ("rolling_5_trading_day_hit_rate", "rolling_5_day_hit_rate.png", "Rolling 5-trading-day hit rate"),
            ("rolling_20_trading_day_hit_rate", "rolling_20_day_hit_rate.png", "Rolling 20-trading-day hit rate"),
        ]:
            fig, ax = plt.subplots(figsize=(12, 5))
            for model_id, group in cumulative.groupby("model_id", sort=False):
                ax.plot(pd.to_datetime(group["feature_date"]), group[metric], label=model_id)
            ax.set_title(title)
            ax.set_xlabel("Feature date")
            ax.set_ylabel(metric.replace("_", " "))
            ax.legend(loc="best")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(plots_dir / filename, dpi=160)
            plt.close(fig)

    for model_id, group in predictions_all.groupby("model_id", sort=False):
        test_group = group[group["split"] == "test"].copy()
        if not test_group.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(pd.to_datetime(test_group["feature_date"]), test_group["probability_up"], label="probability_up")
            ax.scatter(
                pd.to_datetime(test_group["feature_date"]),
                test_group["actual_class"],
                s=12,
                label="actual direction",
            )
            ax.set_title(f"{model_id}: probability-up timeline against actual direction")
            ax.set_xlabel("Feature date")
            ax.set_ylabel("Probability / class")
            ax.legend(loc="best")
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(plots_dir / f"probability_up_timeline_{model_id}.png", dpi=160)
            plt.close(fig)

        cm = group[group["split"] == "test"].copy()
        if not cm.empty:
            matrix = pd.crosstab(cm["actual_class"], cm["predicted_class"], dropna=False).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            fig, ax = plt.subplots(figsize=(4.8, 4.2))
            image = ax.imshow(matrix.to_numpy())
            ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
            ax.set_yticks([0, 1], labels=["Actual 0", "Actual 1"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, int(matrix.iloc[i, j]), ha="center", va="center")
            ax.set_title(f"{model_id}: test confusion matrix")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(plots_dir / f"confusion_matrix_{model_id}.png", dpi=160)
            plt.close(fig)

        importance_path = run_dir / "models" / model_id / "feature_importance.csv"
        if importance_path.exists():
            importance = pd.read_csv(importance_path).sort_values("permutation_importance_mean", ascending=True).tail(12)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(importance["feature"], importance["permutation_importance_mean"])
            ax.set_title(f"{model_id}: validation permutation feature importance")
            ax.set_xlabel("Permutation importance mean")
            fig.tight_layout()
            fig.savefig(plots_dir / f"feature_importance_{model_id}.png", dpi=160)
            plt.close(fig)


def _leaderboard_plot(leaderboard: pd.DataFrame, title: str, path: Path) -> None:
    metrics = ["accuracy", "balanced_accuracy", "f1"]
    frame = leaderboard[["model_id", *metrics]].set_index("model_id")
    fig, ax = plt.subplots(figsize=(10, 5))
    frame.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _zip_output_base(environment: dict[str, Any], repo: Path) -> Path:
    env = environment.get("runtime_environment")
    if env == "colab":
        return Path("/content")
    if env == "kaggle":
        return Path("/kaggle/working")
    return repo / "artifacts"


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train frozen OGDC Random Forest variants for PAISA.")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    parser.add_argument("--output-dir", default="artifacts/model_training")
    parser.add_argument("--random-state", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    run_training_stage(
        symbol=args.symbol,
        start=args.start,
        provider=args.provider,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    _main()
