"""End-to-end orchestration for the real-data OGDC vertical slice.

The vertical slice deliberately composes the existing data, model-training,
and portfolio-experiment boundaries.  It does not train estimators or perform
accounting itself.  A completed top-level directory is published atomically
only after all four model experiments and the combined reports succeed.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import traceback
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from paisa.backtesting.metrics import calculate_drawdowns, calculate_portfolio_metrics
from paisa.experiments.artifacts import (
    make_run_id,
    sha256_file,
    to_jsonable,
    utc_now_z,
    write_json,
)
from paisa.experiments.evaluation import binary_classification_metrics
from paisa.experiments.experiment_config import (
    DEFAULT_START,
    DEFAULT_SYMBOL,
    MODEL_VARIANTS,
    RANDOM_SEED,
)
from paisa.experiments.runner import load_predictions, run_experiment
from paisa.experiments.trainer import run_training_stage
from paisa.pipeline import run_v0_pipeline
from paisa.trading.models import ExperimentConfig


VERTICAL_SLICE_SCHEMA_VERSION = "1.0"
VERTICAL_SLICE_PREFIX = "VSL"
REQUIRED_VARIANT_IDS = tuple(variant.model_id for variant in MODEL_VARIANTS)
ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_REQUIRED_PRICE_COLUMNS = {"date", "ticker", "open", "close", "volume", "source"}
_REAL_PRICE_SOURCES = {"psx_dps", "pypsx_toolkit"}
_NON_REAL_SOURCE_MARKERS = ("sample", "synthetic", "fixture", "mock", "test")


@dataclass(frozen=True, slots=True)
class PriceAcquisition:
    """A price file plus explicit provenance supplied to the orchestrator.

    ``is_real_data=False`` exists solely for fixture-based automated tests.  It
    is persisted in the result so a fixture run can never be described as a
    real-data run.  The production acquisition function always returns true.
    """

    prices_path: str | Path
    source: str
    mode: str
    date_start: str | None = None
    date_end: str | None = None
    row_count: int | None = None
    is_real_data: bool = True
    live_error: str | None = None


@dataclass(frozen=True, slots=True)
class ValidatedPrices:
    """Validated OGDC coverage and provenance for one standardized price CSV."""

    path: Path
    source: str
    date_start: str
    date_end: str
    row_count: int


class LiveDataUnavailableError(RuntimeError):
    """Raised when live PSX acquisition and the validated real cache both fail."""


class LiveAcquisitionError(RuntimeError):
    """Raised only when the live PSX provider cannot supply usable OGDC rows."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _absolute_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _safe_run_id(value: str) -> str:
    run_id = str(value).strip()
    if not _SAFE_ID.fullmatch(run_id):
        raise ValueError(
            "vertical_slice_run_id must start with a letter or digit and contain "
            "only letters, digits, '.', '_', and '-'"
        )
    return run_id


def _inspect_price_file(
    path: str | Path,
    *,
    symbol: str,
    start: str,
    require_real_source: bool,
) -> ValidatedPrices:
    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Price file does not exist: {source_path}")

    frame = pd.read_csv(source_path)
    missing = sorted(_REQUIRED_PRICE_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(
            f"Price file {source_path} is missing required columns: {missing}"
        )

    ticker = str(symbol).strip().upper()
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    selected = frame.loc[frame["ticker"].eq(ticker)].copy()
    if selected.empty:
        raise ValueError(f"Price file {source_path} contains no rows for {ticker}")

    selected["date"] = pd.to_datetime(selected["date"], errors="raise")
    selected = selected.loc[
        selected["date"].dt.date >= date.fromisoformat(start)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"Price file {source_path} contains no {ticker} rows on or after {start}"
        )
    if selected["date"].duplicated().any():
        duplicates = (
            selected.loc[selected["date"].duplicated(keep=False), "date"]
            .dt.strftime("%Y-%m-%d")
            .unique()
            .tolist()
        )
        raise ValueError(f"Price file has duplicate {ticker} dates: {duplicates}")

    for column in ("open", "close", "volume"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    numeric = selected[["open", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Price file contains missing or non-finite open/close/volume")
    if (selected[["open", "close"]] <= 0).any().any():
        raise ValueError("Price file open and close values must be strictly positive")
    if (selected["volume"] < 0).any():
        raise ValueError("Price file volume values must be non-negative")

    sources = sorted(
        {
            str(value).strip().lower()
            for value in selected["source"].dropna().tolist()
            if str(value).strip()
        }
    )
    if not sources:
        raise ValueError("Price file must identify a non-empty source for OGDC")
    blocked = [
        source
        for source in sources
        if any(marker in source for marker in _NON_REAL_SOURCE_MARKERS)
    ]
    if require_real_source and blocked:
        raise ValueError(
            "Refusing to label sample, synthetic, fixture, mock, or test data as "
            f"real PSX data; rejected sources: {blocked}"
        )
    if require_real_source:
        unknown = [source for source in sources if source not in _REAL_PRICE_SOURCES]
        if unknown:
            raise ValueError(
                "Cached real-data fallback requires an explicit PSX source from "
                f"{sorted(_REAL_PRICE_SOURCES)}; found {unknown}"
            )

    selected = selected.sort_values("date", kind="mergesort")
    return ValidatedPrices(
        path=source_path,
        source=",".join(sources),
        date_start=selected["date"].iloc[0].strftime("%Y-%m-%d"),
        date_end=selected["date"].iloc[-1].strftime("%Y-%m-%d"),
        row_count=int(len(selected)),
    )


def validate_real_price_file(
    path: str | Path,
    *,
    symbol: str = DEFAULT_SYMBOL,
    start: str = DEFAULT_START,
) -> ValidatedPrices:
    """Validate that a standardized file is real PSX data, never a fixture."""

    return _inspect_price_file(
        path,
        symbol=symbol,
        start=start,
        require_real_source=True,
    )


def acquire_live_ogdc_prices(
    *,
    symbol: str,
    start: str,
    work_dir: str | Path,
) -> PriceAcquisition:
    """Fetch and clean OGDC with the real PSX DPS provider exactly once."""

    acquisition_root = Path(work_dir).resolve()
    acquisition_root.mkdir(parents=True, exist_ok=False)
    try:
        metadata = run_v0_pipeline(
            symbols=[symbol],
            provider_name="psx-dps",
            start=date.fromisoformat(start),
            train=False,
            data_dir=acquisition_root / "data",
            model_dir=acquisition_root / "models",
        )
    except RuntimeError as exc:
        # ``run_v0_pipeline`` uses this error when no requested symbol could be
        # fetched and cleaned.  Keep that provider-specific failure distinct
        # from later validation/programming errors so only an unavailable live
        # source is allowed to open the cached-real fallback path.
        raise LiveAcquisitionError(
            f"Live PSX DPS acquisition failed for {symbol}: {exc}"
        ) from exc
    prices_path = acquisition_root / "data" / "processed" / "psx_prices.csv"
    validated = validate_real_price_file(prices_path, symbol=symbol, start=start)
    provider = str(metadata.get("provider", validated.source))
    if provider != "psx_dps":
        raise ValueError(
            f"Live acquisition must use provider 'psx_dps'; received {provider!r}"
        )
    return PriceAcquisition(
        prices_path=validated.path,
        source=validated.source,
        mode="live_psx_dps",
        date_start=validated.date_start,
        date_end=validated.date_end,
        row_count=validated.row_count,
        is_real_data=True,
    )


def _resolve_prices(
    *,
    symbol: str,
    start: str,
    work_dir: Path,
    cached_prices_path: Path,
    acquire_fn: Callable[..., PriceAcquisition],
    custom_acquirer: bool,
) -> tuple[PriceAcquisition, ValidatedPrices]:
    try:
        acquisition = acquire_fn(symbol=symbol, start=start, work_dir=work_dir)
    except Exception as exc:
        if not custom_acquirer and not isinstance(exc, LiveAcquisitionError):
            raise
        # Custom acquirers are an explicit test seam; an exception from one is
        # treated as the simulated live-provider failure.  In production, only
        # the narrowly typed provider-unavailable error may reach fallback.
        live_error = traceback.format_exc()
    else:
        if not isinstance(acquisition, PriceAcquisition):
            raise TypeError("acquire_fn must return PriceAcquisition")
        inspected = _inspect_price_file(
            acquisition.prices_path,
            symbol=symbol,
            start=start,
            require_real_source=acquisition.is_real_data,
        )
        if not acquisition.is_real_data and not custom_acquirer:
            raise ValueError("The production acquisition path must return real PSX data")
        normalized = PriceAcquisition(
            prices_path=inspected.path,
            source=inspected.source,
            mode=acquisition.mode,
            date_start=inspected.date_start,
            date_end=inspected.date_end,
            row_count=inspected.row_count,
            is_real_data=acquisition.is_real_data,
            live_error=acquisition.live_error,
        )
        return normalized, inspected

    try:
        fallback = validate_real_price_file(
            cached_prices_path,
            symbol=symbol,
            start=start,
        )
    except Exception:
        fallback_error = traceback.format_exc()
        raise LiveDataUnavailableError(
            "Live PSX DPS acquisition failed and no validated real cached price "
            "file is available.\n\nLIVE ACQUISITION ERROR:\n"
            f"{live_error}\nCACHED FALLBACK ERROR:\n{fallback_error}"
        ) from None

    return (
        PriceAcquisition(
            prices_path=fallback.path,
            source=fallback.source,
            mode="cached_real_fallback",
            date_start=fallback.date_start,
            date_end=fallback.date_end,
            row_count=fallback.row_count,
            is_real_data=True,
            live_error=live_error,
        ),
        fallback,
    )


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _validate_training_summary(
    summary: Any,
    *,
    repo: Path,
) -> tuple[dict[str, Any], Path, dict[str, str], dict[str, Path]]:
    value = dict(_require_mapping(summary, "training summary"))
    required = {
        "run_id",
        "invocation_id",
        "artifact_directory",
        "model_ids",
        "prediction_paths",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Training summary is missing required fields: {missing}")

    model_ids_raw = _require_mapping(value["model_ids"], "training summary model_ids")
    prediction_paths_raw = _require_mapping(
        value["prediction_paths"], "training summary prediction_paths"
    )
    if set(model_ids_raw) != set(REQUIRED_VARIANT_IDS):
        raise ValueError(
            "Training did not produce exactly M-001, M-002, M-003, and M-004"
        )
    if set(prediction_paths_raw) != set(REQUIRED_VARIANT_IDS):
        raise ValueError(
            "Training summary does not contain test prediction paths for all four models"
        )

    model_ids = {key: str(model_ids_raw[key]) for key in REQUIRED_VARIANT_IDS}
    if len(set(model_ids.values())) != len(REQUIRED_VARIANT_IDS):
        raise ValueError("Every trained variant must have a unique registered model ID")

    artifact_directory = _absolute_path(value["artifact_directory"], base=repo)
    if not artifact_directory.is_dir():
        raise FileNotFoundError(
            f"Training artifact directory does not exist: {artifact_directory}"
        )
    prediction_paths = {
        key: _absolute_path(prediction_paths_raw[key], base=repo)
        for key in REQUIRED_VARIANT_IDS
    }
    missing_predictions = [
        str(path) for path in prediction_paths.values() if not path.is_file()
    ]
    if missing_predictions:
        raise FileNotFoundError(
            f"Training prediction artifacts are missing: {missing_predictions}"
        )
    return value, artifact_directory, model_ids, prediction_paths


def _read_training_contract(
    *,
    summary: Mapping[str, Any],
    artifact_directory: Path,
    model_ids: Mapping[str, str],
    prediction_paths: Mapping[str, Path],
) -> tuple[
    pd.DataFrame,
    dict[str, int],
    dict[str, dict[str, int]],
    dict[str, str],
    tuple[str, ...],
]:
    run_manifest_path = artifact_directory / "run_manifest.json"
    split_manifest_path = artifact_directory / "split_manifest.json"
    data_manifest_path = artifact_directory / "data_manifest.json"
    for path in (run_manifest_path, split_manifest_path, data_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required training contract file is missing: {path}")

    run_manifest = _load_json(run_manifest_path)
    if run_manifest.get("run_id") != summary["run_id"]:
        raise ValueError("Training summary run_id does not match run_manifest.json")
    if run_manifest.get("invocation_id") != summary["invocation_id"]:
        raise ValueError("Training invocation_id does not match run_manifest.json")

    split_manifest = _load_json(split_manifest_path)
    counts_raw = _require_mapping(split_manifest.get("row_counts"), "split row_counts")
    if set(counts_raw) != {"train", "validation", "test"}:
        raise ValueError("Training split counts must contain train, validation, and test")
    split_counts = {name: int(counts_raw[name]) for name in ("train", "validation", "test")}
    if any(value <= 0 for value in split_counts.values()):
        raise ValueError("Every training split count must be positive")
    target_dates_raw = _require_mapping(
        split_manifest.get("target_dates"), "split target_dates"
    )
    test_target_dates_value = target_dates_raw.get("test")
    if not isinstance(test_target_dates_value, list):
        raise ValueError("Training split manifest must contain a test target-date list")
    test_target_dates = tuple(str(value) for value in test_target_dates_value)
    if len(test_target_dates) != split_counts["test"]:
        raise ValueError("Test target-date count does not match the split manifest")
    for value in test_target_dates:
        date.fromisoformat(value)

    records: list[dict[str, Any]] = []
    common_distributions: dict[str, dict[str, int]] | None = None
    for variant in MODEL_VARIANTS:
        variant_id = variant.model_id
        registered_id = model_ids[variant_id]
        model_dir = artifact_directory / "models" / registered_id
        manifest_path = model_dir / "manifest.json"
        manifest = _load_json(manifest_path)
        if manifest.get("model_id") != registered_id:
            raise ValueError(f"Manifest model_id mismatch for {variant_id}")
        if manifest.get("variant_id") != variant_id:
            raise ValueError(f"Manifest variant_id mismatch for {variant_id}")
        if manifest.get("algorithm") != variant.algorithm:
            raise ValueError(f"Manifest algorithm mismatch for {variant_id}")

        metrics = _require_mapping(manifest.get("metrics"), f"{variant_id} metrics")
        distributions: dict[str, dict[str, int]] = {}
        for split_name in ("train", "validation", "test"):
            split_metrics = _require_mapping(
                metrics.get(split_name), f"{variant_id} {split_name} metrics"
            )
            if int(split_metrics.get("support", -1)) != split_counts[split_name]:
                raise ValueError(
                    f"{variant_id} {split_name} support does not match split manifest"
                )
            distribution_raw = _require_mapping(
                split_metrics.get("class_distribution"),
                f"{variant_id} {split_name} class_distribution",
            )
            distribution = {
                "0": int(distribution_raw.get("0", -1)),
                "1": int(distribution_raw.get("1", -1)),
            }
            if min(distribution.values()) < 0 or sum(distribution.values()) != split_counts[split_name]:
                raise ValueError(f"Invalid {variant_id} {split_name} class distribution")
            distributions[split_name] = distribution

        if common_distributions is None:
            common_distributions = distributions
        elif distributions != common_distributions:
            raise ValueError("All four models must use identical target class distributions")

        validation = _require_mapping(metrics.get("validation"), "validation metrics")
        test = _require_mapping(metrics.get("test"), "test metrics")
        records.append(
            {
                "model_id": variant_id,
                "registered_model_id": registered_id,
                "algorithm": variant.algorithm,
                "validation_accuracy": validation.get("accuracy"),
                "validation_balanced_accuracy": validation.get("balanced_accuracy"),
                "validation_f1": validation.get("f1"),
                "test_accuracy": test.get("accuracy"),
                "test_balanced_accuracy": test.get("balanced_accuracy"),
                "test_f1": test.get("f1"),
                "train_count": split_counts["train"],
                "validation_count": split_counts["validation"],
                "test_count": split_counts["test"],
                "train_class_0": distributions["train"]["0"],
                "train_class_1": distributions["train"]["1"],
                "validation_class_0": distributions["validation"]["0"],
                "validation_class_1": distributions["validation"]["1"],
                "test_class_0": distributions["test"]["0"],
                "test_class_1": distributions["test"]["1"],
                "predictions_path": str(prediction_paths[variant_id]),
                "manifest_path": str(manifest_path),
            }
        )

    assert common_distributions is not None
    data_manifest = _load_json(data_manifest_path)
    data_details = {
        "provider": str(data_manifest.get("provider", "")),
        "dataset_sha256": str(data_manifest.get("dataset_sha256", "")),
        "actual_feature_start": str(data_manifest.get("actual_feature_start", "")),
        "actual_feature_end": str(data_manifest.get("actual_feature_end", "")),
    }
    return (
        pd.DataFrame(records),
        split_counts,
        common_distributions,
        data_details,
        test_target_dates,
    )


def _common_prediction_dates(
    prediction_paths: Mapping[str, Path],
    model_ids: Mapping[str, str],
    *,
    symbol: str,
    expected_test_count: int,
    expected_target_dates: Sequence[str],
) -> tuple[dict[str, str], dict[str, tuple[Any, ...]]]:
    reference_pairs: tuple[tuple[date, date], ...] | None = None
    loaded: dict[str, tuple[Any, ...]] = {}
    for variant_id in REQUIRED_VARIANT_IDS:
        predictions = load_predictions(prediction_paths[variant_id])
        if len(predictions) != expected_test_count:
            raise ValueError(
                f"{variant_id} predictions count does not match the common test count"
            )
        if any(record.model_id != model_ids[variant_id] for record in predictions):
            raise ValueError(f"{variant_id} prediction model identity is inconsistent")
        if any(record.ticker != symbol for record in predictions):
            raise ValueError(f"{variant_id} predictions contain a non-{symbol} ticker")
        pairs = tuple((record.feature_date, record.target_date) for record in predictions)
        if tuple(record.target_date.isoformat() for record in predictions) != tuple(
            expected_target_dates
        ):
            raise ValueError(
                f"{variant_id} prediction target dates do not match split_manifest.json"
            )
        if reference_pairs is None:
            reference_pairs = pairs
        elif pairs != reference_pairs:
            raise ValueError("All four model experiments must use identical test dates")
        loaded[variant_id] = predictions

    if not reference_pairs:
        raise ValueError("The common test prediction window must not be empty")
    return (
        {
            "feature_start": reference_pairs[0][0].isoformat(),
            "feature_end": reference_pairs[-1][0].isoformat(),
            "target_start": reference_pairs[0][1].isoformat(),
            "target_end": reference_pairs[-1][1].isoformat(),
        },
        loaded,
    )


def _reconcile_prediction_metrics(
    *,
    predictions_by_variant: Mapping[str, tuple[Any, ...]],
    model_metrics: pd.DataFrame,
    class_distributions: Mapping[str, Mapping[str, int]],
) -> None:
    """Verify persisted test predictions reproduce persisted test metrics."""

    expected_distribution = dict(class_distributions["test"])
    rows = model_metrics.set_index("model_id")
    for variant_id in REQUIRED_VARIANT_IDS:
        predictions = predictions_by_variant[variant_id]
        if any(record.actual_class is None for record in predictions):
            raise ValueError(
                f"{variant_id} test predictions must retain every evaluation label"
            )
        actual = np.asarray([record.actual_class for record in predictions], dtype=int)
        predicted = np.asarray(
            [record.predicted_class for record in predictions], dtype=int
        )
        probability_up = np.asarray(
            [float(record.probability_up) for record in predictions], dtype=float
        )
        recalculated = binary_classification_metrics(
            actual, predicted, probability_up
        )
        if recalculated["class_distribution"] != expected_distribution:
            raise ValueError(
                f"{variant_id} prediction labels do not match the test class distribution"
            )
        persisted = rows.loc[variant_id]
        for metric_name, column_name in (
            ("accuracy", "test_accuracy"),
            ("balanced_accuracy", "test_balanced_accuracy"),
            ("f1", "test_f1"),
        ):
            calculated = recalculated[metric_name]
            expected = persisted[column_name]
            if calculated is None or pd.isna(expected):
                if not (calculated is None and pd.isna(expected)):
                    raise ValueError(
                        f"{variant_id} persisted {metric_name} does not match predictions"
                    )
            elif not math.isclose(
                float(calculated), float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{variant_id} persisted {metric_name} does not match predictions"
                )


def _experiment_config_mapping(
    *,
    vertical_slice_run_id: str,
    variant_id: str,
    registered_model_id: str,
    evaluation_dates: Mapping[str, str],
    created_at: str,
) -> dict[str, Any]:
    slug = variant_id.lower()
    experiment_id = f"{vertical_slice_run_id}-{slug}"
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "model_id": registered_model_id,
        "portfolio_id": f"{experiment_id}-strategy",
        "benchmark_portfolio_ids": {
            "cash_only": f"{experiment_id}-cash-only",
            "buy_and_hold": f"{experiment_id}-buy-and-hold",
        },
        "symbol": DEFAULT_SYMBOL,
        "initial_cash": "100000.00",
        "thresholds": {
            "buy_probability_up_gte": 0.60,
            "sell_probability_up_lte": 0.40,
        },
        "maximum_allocation": 0.50,
        "transaction_cost": {"type": "proportional_bps", "value": 10},
        "evaluation_dates": dict(evaluation_dates),
        "execution_policy": {
            "signal_timing": "after_close",
            "execution_session": "next_trading_session",
            "reference_price_field": "open",
            "slippage_bps": 0,
            "missing_execution_price": "fail",
            "allocation_mode": "state_transition_to_maximum",
            "whole_share_rounding": "floor",
            "valuation_price_field": "close",
            "missing_valuation_price": "fail",
            "ending_position": "mark_to_market",
        },
        "metric_assumptions": {
            "annualization_factor": 252,
            "risk_free_rate_annual": 0.0,
        },
        "created_at": created_at,
    }


def _portfolio_metric_record(
    *,
    portfolio_type: str,
    model_id: str,
    portfolio_id: str,
    metrics: Mapping[str, Any],
    result_versus_buy_and_hold: float,
) -> dict[str, Any]:
    return {
        "portfolio_type": portfolio_type,
        "model_id": model_id,
        "portfolio_id": portfolio_id,
        "ending_portfolio_value": metrics["ending_equity"],
        "total_return": metrics["total_return"],
        "maximum_drawdown": metrics["maximum_drawdown"],
        "volatility": metrics["volatility"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "number_of_trades": metrics["number_of_trades"],
        "fees_paid": metrics["fees_paid"],
        "result_versus_buy_and_hold": result_versus_buy_and_hold,
    }


def _economic_snapshot_signature(outcome: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            snapshot.date,
            snapshot.cash,
            snapshot.shares,
            snapshot.market_price,
            snapshot.total_equity,
        )
        for snapshot in outcome.snapshots
    )


def _economic_trade_signature(outcome: Any) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            trade.signal_date,
            trade.execution_date,
            trade.action,
            trade.quantity,
            trade.reference_price,
            trade.execution_price,
            trade.fee,
        )
        for trade in outcome.trades
    )


def _run_four_experiments(
    *,
    vertical_slice_run_id: str,
    model_ids: Mapping[str, str],
    prediction_paths: Mapping[str, Path],
    prices_path: Path,
    evaluation_dates: Mapping[str, str],
    config_root: Path,
    runs_root: Path,
    backtest_fn: Callable[..., Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config_root.mkdir(parents=True, exist_ok=False)
    created_at = utc_now_z()
    outcomes: dict[str, Any] = {}
    model_portfolio_records: list[dict[str, Any]] = []
    reference_market: tuple[tuple[Any, ...], ...] | None = None
    reference_cash: tuple[tuple[Any, ...], ...] | None = None
    reference_buy_hold: tuple[tuple[Any, ...], ...] | None = None
    reference_buy_hold_trades: tuple[tuple[Any, ...], ...] | None = None

    for variant_id in REQUIRED_VARIANT_IDS:
        raw_config = _experiment_config_mapping(
            vertical_slice_run_id=vertical_slice_run_id,
            variant_id=variant_id,
            registered_model_id=model_ids[variant_id],
            evaluation_dates=evaluation_dates,
            created_at=created_at,
        )
        config = ExperimentConfig.from_mapping(raw_config)
        config_path = config_root / f"{variant_id}.json"
        write_json(config.to_dict(), config_path)
        completed = backtest_fn(
            predictions_path=prediction_paths[variant_id],
            prices_path=prices_path,
            config_path=config_path,
            runs_root=runs_root,
        )
        for attribute in ("config", "strategy", "cash_only", "buy_and_hold", "metrics"):
            if not hasattr(completed, attribute):
                raise TypeError(f"backtest_fn result is missing {attribute}")
        if completed.config.model_id != model_ids[variant_id]:
            raise ValueError(f"Backtest model identity mismatch for {variant_id}")
        if completed.config.portfolio_id != config.portfolio_id:
            raise ValueError(f"Backtest portfolio identity mismatch for {variant_id}")

        market_signature = tuple(
            (bar.date, bar.open, bar.close) for bar in completed.strategy.market_bars
        )
        cash_signature = _economic_snapshot_signature(completed.cash_only)
        buy_hold_signature = _economic_snapshot_signature(completed.buy_and_hold)
        buy_hold_trade_signature = _economic_trade_signature(completed.buy_and_hold)
        if reference_market is None:
            reference_market = market_signature
            reference_cash = cash_signature
            reference_buy_hold = buy_hold_signature
            reference_buy_hold_trades = buy_hold_trade_signature
        elif (
            market_signature != reference_market
            or cash_signature != reference_cash
            or buy_hold_signature != reference_buy_hold
            or buy_hold_trade_signature != reference_buy_hold_trades
        ):
            raise AssertionError(
                "Every model experiment must receive identical dates, prices, and benchmarks"
            )

        outcomes[variant_id] = completed
        metrics = completed.metrics
        model_portfolio_records.append(
            _portfolio_metric_record(
                portfolio_type="model",
                model_id=variant_id,
                portfolio_id=config.portfolio_id,
                metrics=metrics,
                result_versus_buy_and_hold=float(
                    metrics["comparison_with_buy_and_hold"]
                ),
            )
        )

    first = outcomes[REQUIRED_VARIANT_IDS[0]]
    metric_options = {
        "annualization_factor": first.config.metric_assumptions.annualization_factor,
        "risk_free_rate_annual": float(
            first.config.metric_assumptions.risk_free_rate_annual
        ),
    }
    cash_metrics = calculate_portfolio_metrics(
        first.cash_only.snapshots, first.cash_only.trades, **metric_options
    )
    buy_hold_metrics = calculate_portfolio_metrics(
        first.buy_and_hold.snapshots, first.buy_and_hold.trades, **metric_options
    )
    benchmark_records = [
        _portfolio_metric_record(
            portfolio_type="cash_only",
            model_id="CASH-ONLY",
            portfolio_id=first.cash_only.portfolio_id,
            metrics=cash_metrics,
            result_versus_buy_and_hold=float(
                cash_metrics["total_return"] - buy_hold_metrics["total_return"]
            ),
        ),
        _portfolio_metric_record(
            portfolio_type="buy_and_hold",
            model_id="BUY-AND-HOLD",
            portfolio_id=first.buy_and_hold.portfolio_id,
            metrics=buy_hold_metrics,
            result_versus_buy_and_hold=0.0,
        ),
    ]
    portfolio_metrics = pd.DataFrame([*model_portfolio_records, *benchmark_records])
    daily_curves = _daily_equity_frame(outcomes)
    return outcomes, portfolio_metrics, daily_curves


def _daily_equity_frame(outcomes: Mapping[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def append_series(
        series_id: str,
        portfolio_type: str,
        model_id: str,
        outcome: Any,
    ) -> None:
        drawdowns = {
            item["date"]: item["drawdown"]
            for item in calculate_drawdowns(outcome.snapshots)
        }
        for snapshot in outcome.snapshots:
            rows.append(
                {
                    "date": snapshot.date.isoformat(),
                    "series_id": series_id,
                    "portfolio_type": portfolio_type,
                    "model_id": model_id,
                    "portfolio_id": snapshot.portfolio_id,
                    "cash": format(snapshot.cash, ".2f"),
                    "shares": snapshot.shares,
                    "market_price": format(snapshot.market_price, ".2f"),
                    "total_equity": format(snapshot.total_equity, ".2f"),
                    "drawdown": drawdowns[snapshot.date.isoformat()],
                }
            )

    for variant_id in REQUIRED_VARIANT_IDS:
        append_series(variant_id, "model", variant_id, outcomes[variant_id].strategy)
    first = outcomes[REQUIRED_VARIANT_IDS[0]]
    append_series("CASH-ONLY", "cash_only", "", first.cash_only)
    append_series("BUY-AND-HOLD", "buy_and_hold", "", first.buy_and_hold)
    return pd.DataFrame(rows)


def build_combined_leaderboard(
    model_metrics: pd.DataFrame,
    portfolio_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Join prediction and simulated-portfolio metrics by stable model variant."""

    models = model_metrics.copy()
    portfolios = portfolio_metrics.copy()
    if "variant_id" in models.columns:
        if "model_id" in models.columns and "registered_model_id" not in models.columns:
            models["registered_model_id"] = models["model_id"]
        models["model_id"] = models["variant_id"]
    if "portfolio_type" in portfolios.columns:
        portfolios = portfolios.loc[portfolios["portfolio_type"].eq("model")].copy()
    rename = {
        "ending_equity": "ending_portfolio_value",
        "comparison_with_buy_and_hold": "result_versus_buy_and_hold",
    }
    portfolios = portfolios.rename(
        columns={key: value for key, value in rename.items() if value not in portfolios.columns}
    )

    model_columns = [
        "model_id",
        "algorithm",
        "validation_balanced_accuracy",
        "validation_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
    ]
    portfolio_columns = [
        "model_id",
        "ending_portfolio_value",
        "total_return",
        "maximum_drawdown",
        "volatility",
        "sharpe_ratio",
        "number_of_trades",
        "fees_paid",
        "result_versus_buy_and_hold",
    ]
    if "registered_model_id" in models.columns:
        model_columns.insert(1, "registered_model_id")
    if "portfolio_id" in portfolios.columns:
        portfolio_columns.insert(1, "portfolio_id")
    missing_models = sorted(set(model_columns) - set(models.columns))
    missing_portfolios = sorted(set(portfolio_columns) - set(portfolios.columns))
    if missing_models or missing_portfolios:
        raise ValueError(
            "Cannot build combined leaderboard; missing model columns "
            f"{missing_models}, missing portfolio columns {missing_portfolios}"
        )
    if models["model_id"].duplicated().any() or portfolios["model_id"].duplicated().any():
        raise ValueError("Combined leaderboard inputs must have one row per model_id")

    combined = models[model_columns].merge(
        portfolios[portfolio_columns],
        on="model_id",
        how="inner",
        validate="one_to_one",
    )
    if set(combined["model_id"]) != set(REQUIRED_VARIANT_IDS):
        raise ValueError("Combined leaderboard must contain all four required models")
    combined = combined.sort_values(
        ["validation_balanced_accuracy", "validation_f1", "model_id"],
        ascending=[False, False, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return combined


def _plot_model_metrics(path: Path, model_metrics: pd.DataFrame) -> None:
    frame = model_metrics.set_index("model_id").loc[list(REQUIRED_VARIANT_IDS)]
    metric_specs = (
        ("validation_balanced_accuracy", "Validation balanced accuracy"),
        ("validation_f1", "Validation F1"),
        ("test_accuracy", "Test accuracy"),
        ("test_balanced_accuracy", "Test balanced accuracy"),
        ("test_f1", "Test F1"),
    )
    figure = Figure(figsize=(11, 6), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    positions = np.arange(len(frame))
    width = 0.16
    for index, (column, label) in enumerate(metric_specs):
        values = [np.nan if value is None else float(value) for value in frame[column]]
        axis.bar(positions + (index - 2) * width, values, width=width, label=label)
    axis.set_xticks(positions, frame.index.tolist())
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    axis.set_title("OGDC model metrics — prediction quality only")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    figure.text(0.5, 0.01, ACADEMIC_DISCLAIMER, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(str(path), metadata={"Software": "PAISA"})


def _plot_portfolio_curves(path: Path, daily_curves: pd.DataFrame) -> None:
    figure = Figure(figsize=(11, 6), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    for series_id, group in daily_curves.groupby("series_id", sort=False):
        axis.plot(
            pd.to_datetime(group["date"]),
            pd.to_numeric(group["total_equity"]),
            label=series_id,
            linewidth=1.7,
        )
    axis.set_title("OGDC test-period simulated portfolio equity")
    axis.set_xlabel("Trading session")
    axis.set_ylabel("Equity (PKR)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    figure.autofmt_xdate()
    figure.text(0.5, 0.01, ACADEMIC_DISCLAIMER, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(str(path), metadata={"Software": "PAISA"})


def _plot_drawdowns(path: Path, daily_curves: pd.DataFrame) -> None:
    figure = Figure(figsize=(11, 6), dpi=120)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(1, 1, 1)
    for series_id, group in daily_curves.groupby("series_id", sort=False):
        axis.plot(
            pd.to_datetime(group["date"]),
            pd.to_numeric(group["drawdown"]),
            label=series_id,
            linewidth=1.7,
        )
    axis.set_title("OGDC test-period simulated drawdown")
    axis.set_xlabel("Trading session")
    axis.set_ylabel("Drawdown")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, ncols=2)
    figure.autofmt_xdate()
    figure.text(0.5, 0.01, ACADEMIC_DISCLAIMER, ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    canvas.print_png(str(path), metadata={"Software": "PAISA"})


def _best_validation_model(model_metrics: pd.DataFrame) -> dict[str, Any]:
    eligible = model_metrics.loc[
        model_metrics["validation_balanced_accuracy"].notna()
    ].copy()
    if eligible.empty:
        raise ValueError("No model is eligible for validation-based comparison")
    row = eligible.sort_values(
        [
            "validation_balanced_accuracy",
            "validation_f1",
            "validation_accuracy",
            "model_id",
        ],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).iloc[0]
    return {
        "model_id": row["model_id"],
        "registered_model_id": row["registered_model_id"],
        "algorithm": row["algorithm"],
        "validation_balanced_accuracy": row["validation_balanced_accuracy"],
        "validation_f1": row["validation_f1"],
        "selection_basis": "validation balanced accuracy, then validation F1 and accuracy",
    }


def _best_portfolio(portfolio_metrics: pd.DataFrame) -> dict[str, Any]:
    models = portfolio_metrics.loc[portfolio_metrics["portfolio_type"].eq("model")].copy()
    models["_sharpe"] = pd.to_numeric(models["sharpe_ratio"], errors="coerce")
    row = models.sort_values(
        ["total_return", "_sharpe", "maximum_drawdown", "model_id"],
        ascending=[False, False, True, True],
        na_position="last",
        kind="mergesort",
    ).iloc[0]
    return {
        "model_id": row["model_id"],
        "portfolio_id": row["portfolio_id"],
        "ending_portfolio_value": row["ending_portfolio_value"],
        "total_return": row["total_return"],
        "sharpe_ratio": None if pd.isna(row["_sharpe"]) else float(row["_sharpe"]),
        "selection_basis": "highest test-period simulated portfolio total return",
    }


def _lowest_drawdown_portfolio(portfolio_metrics: pd.DataFrame) -> dict[str, Any]:
    models = portfolio_metrics.loc[portfolio_metrics["portfolio_type"].eq("model")]
    row = models.sort_values(
        ["maximum_drawdown", "total_return", "model_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]
    return {
        "model_id": row["model_id"],
        "portfolio_id": row["portfolio_id"],
        "maximum_drawdown": row["maximum_drawdown"],
        "total_return": row["total_return"],
    }


def _artifact_manifest(stage_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(stage_dir).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(stage_dir.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]


def run_ogdc_vertical_slice(
    *,
    output_dir: str | Path = "artifacts/vertical_slice",
    model_output_dir: str | Path = "artifacts/model_training",
    cached_prices_path: str | Path = "data/processed/psx_prices.csv",
    start: str = DEFAULT_START,
    vertical_slice_run_id: str | None = None,
    random_state: int = RANDOM_SEED,
    acquire_fn: Callable[..., PriceAcquisition] | None = None,
    train_fn: Callable[..., Mapping[str, Any]] | None = None,
    backtest_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run and atomically publish the complete OGDC research workflow.

    Dependency-injection hooks are intended for offline fixture tests.  The
    default path always attempts the real ``psx-dps`` provider first and only
    accepts the canonical cache after strict real-source validation.
    """

    repo = _repo_root()
    symbol = DEFAULT_SYMBOL
    date.fromisoformat(start)
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("random_state must be an integer")

    output_root = _absolute_path(output_dir, base=repo)
    model_output_root = _absolute_path(model_output_dir, base=repo)
    cached_path = _absolute_path(cached_prices_path, base=repo)
    output_root.mkdir(parents=True, exist_ok=True)
    model_output_root.mkdir(parents=True, exist_ok=True)
    run_id = _safe_run_id(vertical_slice_run_id or make_run_id(repo, prefix=VERTICAL_SLICE_PREFIX))
    destination = output_root / run_id
    if destination.exists():
        raise FileExistsError(
            f"Vertical-slice run already exists; refusing to overwrite: {destination}"
        )

    acquisition_callable = acquire_fn or acquire_live_ogdc_prices
    training_callable = train_fn or run_training_stage
    backtesting_callable = backtest_fn or run_experiment
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.tmp-", dir=output_root))
    work_dir = stage_dir / ".work"
    work_dir.mkdir()

    try:
        acquisition, _ = _resolve_prices(
            symbol=symbol,
            start=start,
            work_dir=work_dir / "live-acquisition",
            cached_prices_path=cached_path,
            acquire_fn=acquisition_callable,
            custom_acquirer=acquire_fn is not None,
        )

        training_summary_raw = training_callable(
            symbol=symbol,
            start=start,
            provider="psx-dps",
            prices_path=acquisition.prices_path,
            output_dir=model_output_root,
            random_state=random_state,
            repo_root=repo,
            create_zip=False,
            model_ids=list(REQUIRED_VARIANT_IDS),
        )
        (
            training_summary,
            training_dir,
            model_ids,
            prediction_paths,
        ) = _validate_training_summary(training_summary_raw, repo=repo)

        (
            model_metrics,
            split_counts,
            class_distributions,
            training_data,
            test_target_dates,
        ) = _read_training_contract(
            summary=training_summary,
            artifact_directory=training_dir,
            model_ids=model_ids,
            prediction_paths=prediction_paths,
        )
        evaluation_dates, predictions_by_variant = _common_prediction_dates(
            prediction_paths,
            model_ids,
            symbol=symbol,
            expected_test_count=split_counts["test"],
            expected_target_dates=test_target_dates,
        )
        _reconcile_prediction_metrics(
            predictions_by_variant=predictions_by_variant,
            model_metrics=model_metrics,
            class_distributions=class_distributions,
        )

        frozen_prices = training_dir / "frozen_input" / f"{symbol}_prices.csv"
        if not frozen_prices.is_file():
            raise FileNotFoundError(
                f"Training run did not preserve its exact price input: {frozen_prices}"
            )
        frozen_validation = _inspect_price_file(
            frozen_prices,
            symbol=symbol,
            start=start,
            require_real_source=acquisition.is_real_data,
        )

        outcomes, portfolio_metrics, daily_curves = _run_four_experiments(
            vertical_slice_run_id=run_id,
            model_ids=model_ids,
            prediction_paths=prediction_paths,
            prices_path=frozen_prices,
            evaluation_dates=evaluation_dates,
            config_root=stage_dir / "experiment_configs",
            runs_root=work_dir / "backtests",
            backtest_fn=backtesting_callable,
        )
        if len({outcome.config.portfolio_id for outcome in outcomes.values()}) != 4:
            raise AssertionError("Each model experiment must have a unique portfolio ID")

        combined = build_combined_leaderboard(model_metrics, portfolio_metrics)
        model_metrics.to_csv(stage_dir / "model_metrics.csv", index=False)
        portfolio_metrics.to_csv(stage_dir / "portfolio_metrics.csv", index=False)
        combined.to_csv(stage_dir / "combined_leaderboard.csv", index=False)
        daily_curves.to_csv(stage_dir / "daily_equity_curves.csv", index=False)
        _plot_model_metrics(stage_dir / "model_metric_comparison.png", model_metrics)
        _plot_portfolio_curves(
            stage_dir / "portfolio_equity_comparison.png", daily_curves
        )
        _plot_drawdowns(stage_dir / "drawdown_comparison.png", daily_curves)

        training_reference = {
            "schema_version": VERTICAL_SLICE_SCHEMA_VERSION,
            "training_run_id": training_summary["run_id"],
            "training_invocation_id": training_summary["invocation_id"],
            "artifact_directory": str(training_dir),
            "run_manifest_path": str(training_dir / "run_manifest.json"),
            "run_manifest_sha256": sha256_file(training_dir / "run_manifest.json"),
            "frozen_prices_path": str(frozen_prices),
            "frozen_prices_sha256": sha256_file(frozen_prices),
            "model_ids": model_ids,
            "prediction_paths": {
                key: str(value) for key, value in prediction_paths.items()
            },
            "prediction_sha256": {
                key: sha256_file(value) for key, value in prediction_paths.items()
            },
        }
        write_json(
            training_reference, stage_dir / "model_training_run_reference.json"
        )

        best_validation = _best_validation_model(model_metrics)
        champion_registered = _require_mapping(
            training_summary.get("provisional_champion", {}),
            "provisional_champion",
        ).get("model_id")
        if champion_registered and champion_registered != best_validation["registered_model_id"]:
            raise AssertionError(
                "Validation-model selection disagrees with the frozen training champion"
            )
        best_portfolio = _best_portfolio(portfolio_metrics)
        lowest_drawdown = _lowest_drawdown_portfolio(portfolio_metrics)
        buy_hold_row = portfolio_metrics.loc[
            portfolio_metrics["portfolio_type"].eq("buy_and_hold")
        ].iloc[0]
        model_rows = portfolio_metrics.loc[
            portfolio_metrics["portfolio_type"].eq("model")
        ]
        beat_buy_hold = bool(
            (model_rows["total_return"] > float(buy_hold_row["total_return"])).any()
        )

        warnings = [
            "The prediction target is next-session close-to-close direction; simulated "
            "portfolio returns begin at the next-session open. These are distinct measurements.",
            "PSX prices have unknown adjustment/corporate-action treatment and dividends are not modeled.",
            "The buy-and-hold benchmark uses the shared 50% maximum-allocation rule.",
        ]
        if acquisition.live_error:
            warnings.append(
                "Live PSX acquisition failed; a separately validated cached real-data file was used."
            )
            warnings.append(
                "Cached-file provenance is validated from its standardized source field and "
                "content checks; a deliberately mislabeled local file cannot be independently authenticated."
            )
        if not acquisition.is_real_data:
            warnings.append(
                "Fixture-based test run: this output is not a real-data vertical-slice result."
            )

        artifact_paths = {
            "manifest": str(destination / "manifest.json"),
            "model_training_run_reference": str(
                destination / "model_training_run_reference.json"
            ),
            "experiment_configs": str(destination / "experiment_configs"),
            "model_metrics": str(destination / "model_metrics.csv"),
            "portfolio_metrics": str(destination / "portfolio_metrics.csv"),
            "combined_leaderboard": str(destination / "combined_leaderboard.csv"),
            "daily_equity_curves": str(destination / "daily_equity_curves.csv"),
            "model_metric_comparison": str(
                destination / "model_metric_comparison.png"
            ),
            "portfolio_equity_comparison": str(
                destination / "portfolio_equity_comparison.png"
            ),
            "drawdown_comparison": str(destination / "drawdown_comparison.png"),
            "final_summary": str(destination / "final_summary.json"),
        }
        final_summary = to_jsonable(
            {
                "schema_version": VERTICAL_SLICE_SCHEMA_VERSION,
                "status": "PASS",
                "vertical_slice_run_id": run_id,
                "completed_at": utc_now_z(),
                "symbol": symbol,
                "data_source": {
                    "source": frozen_validation.source,
                    "mode": acquisition.mode,
                    "is_real_data": acquisition.is_real_data,
                    "frozen_prices_sha256": sha256_file(frozen_prices),
                    "actual_ogdc_date_range": {
                        "start": frozen_validation.date_start,
                        "end": frozen_validation.date_end,
                    },
                    "row_count": frozen_validation.row_count,
                    "live_acquisition_error": acquisition.live_error,
                },
                "model_training_run_id": training_summary["run_id"],
                "common_split_counts": split_counts,
                "class_distributions": class_distributions,
                "common_test_dates": evaluation_dates,
                "best_validation_prediction_model": best_validation,
                "best_test_period_simulated_portfolio": best_portfolio,
                "lowest_drawdown_portfolio": lowest_drawdown,
                "buy_and_hold_result": {
                    "portfolio_id": buy_hold_row["portfolio_id"],
                    "ending_portfolio_value": buy_hold_row[
                        "ending_portfolio_value"
                    ],
                    "total_return": buy_hold_row["total_return"],
                    "maximum_drawdown": buy_hold_row["maximum_drawdown"],
                    "fees_paid": buy_hold_row["fees_paid"],
                },
                "any_model_portfolio_beat_buy_and_hold": beat_buy_hold,
                "selection_policy": (
                    "Prediction model reporting is validation-based; simulated portfolio "
                    "results are reported separately. Test accuracy never selects a winner."
                ),
                "target_interpretation": (
                    "Model classification measures feature-close to next-session-close direction."
                ),
                "portfolio_return_interpretation": (
                    "Trading simulation executes after the signal at next-session open and values daily at close."
                ),
                "training_data": training_data,
                "warnings": warnings,
                "artifact_paths": artifact_paths,
                "disclaimer": ACADEMIC_DISCLAIMER,
            }
        )
        write_json(final_summary, stage_dir / "final_summary.json")

        if work_dir.exists():
            shutil.rmtree(work_dir)
        manifest = {
            "schema_version": VERTICAL_SLICE_SCHEMA_VERSION,
            "vertical_slice_run_id": run_id,
            "status": "PASS",
            "created_at": final_summary["completed_at"],
            "symbol": symbol,
            "data": final_summary["data_source"],
            "model_training_run_id": training_summary["run_id"],
            "model_variants": list(REQUIRED_VARIANT_IDS),
            "registered_model_ids": dict(model_ids),
            "portfolio_ids": {
                variant_id: {
                    "strategy": outcomes[variant_id].config.portfolio_id,
                    "cash_only": outcomes[
                        variant_id
                    ].config.benchmark_portfolio_ids.cash_only,
                    "buy_and_hold": outcomes[
                        variant_id
                    ].config.benchmark_portfolio_ids.buy_and_hold,
                }
                for variant_id in REQUIRED_VARIANT_IDS
            },
            "common_split_counts": split_counts,
            "class_distributions": class_distributions,
            "common_test_dates": evaluation_dates,
            "shared_experiment_assumptions": {
                "initial_cash_pkr": "100000.00",
                "buy_probability_up_gte": 0.60,
                "sell_probability_up_lte": 0.40,
                "maximum_allocation": 0.50,
                "long_only": True,
                "whole_shares": True,
                "transaction_cost_bps": 10,
                "slippage_bps": 0,
                "signal_timing": "after feature-date close",
                "execution": "next observed trading-session open",
                "valuation": "daily close",
                "ending_position": "mark_to_market",
            },
            "artifacts": _artifact_manifest(stage_dir),
            "disclaimer": ACADEMIC_DISCLAIMER,
        }
        write_json(manifest, stage_dir / "manifest.json")

        if destination.exists():
            raise FileExistsError(
                f"Vertical-slice run already exists; refusing to overwrite: {destination}"
            )
        os.rename(stage_dir, destination)
        return final_summary
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise


__all__ = [
    "ACADEMIC_DISCLAIMER",
    "LiveAcquisitionError",
    "LiveDataUnavailableError",
    "PriceAcquisition",
    "REQUIRED_VARIANT_IDS",
    "ValidatedPrices",
    "acquire_live_ogdc_prices",
    "build_combined_leaderboard",
    "run_ogdc_vertical_slice",
    "validate_real_price_file",
]
