from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from paisa.experiments.artifacts import (
    seal_training_run_artifacts,
    validate_training_run_artifacts,
)
from paisa.experiments.evaluation import PREDICTION_COLUMNS
from paisa.experiments.runner import run_experiment
from paisa.legacy_cache import migrate_legacy_psx_cache
from paisa.pipeline import run_v0_pipeline
from paisa.vertical_slice import (
    PriceAcquisition,
    build_combined_leaderboard,
    run_ogdc_vertical_slice,
    validate_real_price_file,
)


REQUIRED_OUTPUTS = {
    "manifest.json",
    "model_training_run_reference.json",
    "input_snapshot",
    "experiment_configs",
    "experiments",
    "model_metrics.csv",
    "portfolio_metrics.csv",
    "combined_leaderboard.csv",
    "daily_equity_curves.csv",
    "model_metric_comparison.png",
    "portfolio_equity_comparison.png",
    "drawdown_comparison.png",
    "final_summary.json",
}
EXPECTED_VARIANTS = {"M-001", "M-002", "M-003", "M-004"}
EXPECTED_EXPERIMENT_FILES = {
    "config.json",
    "predictions.csv",
    "signals.csv",
    "trades.csv",
    "portfolio_snapshots.csv",
    "metrics.json",
    "result.json",
    "equity_curve.png",
    "drawdown.png",
    "benchmarks/cash_only_snapshots.csv",
    "benchmarks/buy_and_hold_snapshots.csv",
    "benchmarks/buy_and_hold_trades.csv",
}


def _price_fixture(
    rows: int = 160,
    *,
    source: str = "deterministic_fixture",
) -> pd.DataFrame:
    index = np.arange(rows)
    close = 100.0 + index * 0.08 + 2.0 * np.sin(index / 2.7)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-01", periods=rows),
            "ticker": "OGDC",
            "open": close * (1.0 + 0.002 * np.cos(index)),
            "high": np.nan,
            "low": np.nan,
            "close": close,
            "volume": 100_000 + (index % 17) * 1_000,
            "source": source,
        }
    )


def _fixture_acquisition(*, symbol: str, start: str, work_dir: Path) -> PriceAcquisition:
    assert symbol == "OGDC"
    assert start <= "2025-01-01"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    prices = _price_fixture()
    prices_path = work_dir / "OGDC_fixture_prices.csv"
    prices.to_csv(prices_path, index=False)
    return PriceAcquisition(
        prices_path=prices_path,
        source="deterministic_fixture",
        mode="fixture_test",
        date_start="2025-01-01",
        date_end=pd.Timestamp(prices["date"].max()).strftime("%Y-%m-%d"),
        row_count=len(prices),
        is_real_data=False,
        live_error=None,
    )


def _pipeline_produced_psx_contract_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, pd.DataFrame]:
    """Exercise the real pipeline writer with an offline provider test double."""

    prices = _price_fixture(source="psx_dps")

    class OfflinePsxProvider:
        source_name = "psx_dps"

        def fetch_history(self, symbol: str, start=None, end=None):
            assert symbol == "OGDC"
            return prices.drop(columns=["ticker", "source"]).copy()

    monkeypatch.setattr(
        "paisa.pipeline.get_provider",
        lambda *_args, **_kwargs: OfflinePsxProvider(),
    )
    data_dir = tmp_path / "pipeline-data"
    run_v0_pipeline(
        symbols=["OGDC"],
        provider_name="psx-dps",
        start=date(2025, 1, 1),
        train=False,
        data_dir=data_dir,
        model_dir=tmp_path / "pipeline-models",
    )
    processed = data_dir / "processed"
    cached = processed / "psx_prices.csv"
    return cached, processed / "pipeline_metadata.json", pd.read_csv(cached)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_price_provenance(
    prices_path: Path,
    *,
    provider: str = "psx_dps",
    metadata_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write the pipeline provenance shape for one test price artifact."""

    frame = pd.read_csv(prices_path)
    dates = pd.to_datetime(frame["date"], errors="raise")
    tickers = frame["ticker"].astype(str).str.strip().str.upper()
    symbols: dict[str, dict[str, Any]] = {}
    for symbol in sorted(tickers.unique()):
        selected_dates = dates.loc[tickers.eq(symbol)]
        symbols[symbol] = {
            "row_count": int(len(selected_dates)),
            "date_start": selected_dates.min().strftime("%Y-%m-%d"),
            "date_end": selected_dates.max().strftime("%Y-%m-%d"),
            "source": provider,
        }
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "producer": {
            "component": "paisa.pipeline.run_v0_pipeline",
            "schema_version": "1.0",
            "status": "complete",
        },
        "generated_at": "2026-10-20T00:00:00",
        "provider": provider,
        "symbols_requested": sorted(symbols),
        "raw_row_counts": {
            symbol: details["row_count"] for symbol, details in symbols.items()
        },
        "errors": {},
        "prices_rows": int(len(frame)),
        "features_rows": int(len(frame)),
        "ml_dataset_rows": max(0, int(len(frame)) - 30),
        "date_range": {
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d"),
        },
        "prices_artifact": {
            "schema_version": "1.0",
            "status": "complete",
            "path": prices_path.name,
            "sha256": _sha256(prices_path),
            "provider": provider,
            "frequency": "daily",
            "row_count": int(len(frame)),
            "symbols": symbols,
        },
        "files": {
            "prices": str(prices_path),
            "features": str(prices_path.parent / "psx_features.csv"),
            "dataset": str(prices_path.parent / "ml_dataset.csv"),
            "model_dir": str(prices_path.parent / "models"),
        },
        "model_metrics": None,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    metadata_path = prices_path.parent / "pipeline_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return metadata_path


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@dataclass(frozen=True)
class CompletedVerticalSlice:
    artifact_path: Path
    output_dir: Path
    model_output_dir: Path
    summary: dict[str, Any]
    backtest_calls: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CompletedVerifiedLegacySlice:
    artifact_path: Path
    output_dir: Path
    model_output_dir: Path
    bundle_path: Path
    summary: dict[str, Any]
    live_error_text: str


def _migrated_legacy_price_bundle(root: Path) -> Path:
    """Create an old-format PSX cache and migrate it through the public API."""

    processed = root / "legacy" / "data" / "processed"
    raw_dir = root / "legacy" / "data" / "raw"
    raw_psx_dir = raw_dir / "psx"
    processed.mkdir(parents=True)
    raw_psx_dir.mkdir(parents=True)
    prices = _price_fixture(source="psx_dps")
    prices_path = processed / "psx_prices.csv"
    prices.to_csv(prices_path, index=False)
    prices.drop(columns=["ticker", "source"]).to_csv(
        raw_psx_dir / "OGDC_psx_dps_raw.csv",
        index=False,
    )
    dates = pd.to_datetime(prices["date"], errors="raise")
    metadata = {
        "generated_at": "2026-09-15T23:46:30",
        "provider": "psx_dps",
        "symbols_requested": ["OGDC"],
        "raw_row_counts": {"OGDC": len(prices)},
        "errors": {},
        "prices_rows": len(prices),
        "features_rows": len(prices),
        "ml_dataset_rows": len(prices) - 30,
        "date_range": {
            "start": dates.min().strftime("%Y-%m-%d"),
            "end": dates.max().strftime("%Y-%m-%d"),
        },
        "files": {
            "prices": str(prices_path),
            "features": str(processed / "psx_features.csv"),
            "dataset": str(processed / "ml_dataset.csv"),
            "model_dir": str(root / "legacy" / "models"),
        },
        "model_metrics": None,
    }
    metadata_path = processed / "pipeline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    bundle = root / "verified-legacy-cache"
    migrate_legacy_psx_cache(
        prices_path,
        metadata_path,
        bundle,
        legacy_raw_dir=raw_dir,
        expected_provider="psx_dps",
        required_symbol="OGDC",
    )
    return bundle


@pytest.fixture(scope="module")
def completed_vertical_slice(tmp_path_factory: pytest.TempPathFactory) -> CompletedVerticalSlice:
    root = tmp_path_factory.mktemp("ogdc-vertical-slice")
    output_dir = root / "vertical"
    model_output_dir = root / "model-training"
    calls: list[dict[str, Any]] = []

    def capture_backtest(**kwargs: Any):
        prediction_path = Path(kwargs["predictions_path"])
        prices_path = Path(kwargs["prices_path"])
        before = _sha256(prediction_path)
        prediction_frame = pd.read_csv(prediction_path)
        completed = run_experiment(**kwargs)
        calls.append(
            {
                **kwargs,
                "variant_id": Path(kwargs["config_path"]).stem,
                "prediction_sha256_before": before,
                "prediction_sha256_after": _sha256(prediction_path),
                "prediction_columns": prediction_frame.columns.tolist(),
                "prediction_date_pairs": list(
                    zip(
                        prediction_frame["feature_date"],
                        prediction_frame["target_date"],
                        strict=True,
                    )
                ),
                "prices_sha256": _sha256(prices_path),
                "completed": completed,
            }
        )
        return completed

    summary = run_ogdc_vertical_slice(
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        cached_prices_path=root / "missing-cached-prices.csv",
        vertical_slice_run_id="pytest-smoke",
        acquire_fn=_fixture_acquisition,
        backtest_fn=capture_backtest,
    )
    return CompletedVerticalSlice(
        artifact_path=output_dir / "pytest-smoke",
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        summary=summary,
        backtest_calls=tuple(calls),
    )


@pytest.fixture(scope="module")
def completed_verified_legacy_slice(
    tmp_path_factory: pytest.TempPathFactory,
) -> CompletedVerifiedLegacySlice:
    root = tmp_path_factory.mktemp("ogdc-verified-legacy-slice")
    bundle = _migrated_legacy_price_bundle(root)
    output_dir = root / "vertical"
    model_output_dir = root / "model-training"
    live_error_text = (
        "PSX DPS request failed with HTTP 404 Not Found for "
        "https://dps.psx.com.pk/historical; response body: route not found"
    )

    def live_404(**_: Any) -> PriceAcquisition:
        raise RuntimeError(live_error_text)

    summary = run_ogdc_vertical_slice(
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        cached_prices_path=bundle / "psx_prices.csv",
        vertical_slice_run_id="verified-legacy-smoke",
        acquire_fn=live_404,
    )
    return CompletedVerifiedLegacySlice(
        artifact_path=output_dir / "verified-legacy-smoke",
        output_dir=output_dir,
        model_output_dir=model_output_dir,
        bundle_path=bundle,
        summary=summary,
        live_error_text=live_error_text,
    )


def _experiment_configs(run: CompletedVerticalSlice) -> list[dict[str, Any]]:
    paths = sorted((run.artifact_path / "experiment_configs").glob("*.json"))
    assert {path.stem for path in paths} == EXPECTED_VARIANTS
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_all_four_experiments_use_identical_dates_market_prices_and_rules(
    completed_vertical_slice: CompletedVerticalSlice,
):
    configs = _experiment_configs(completed_vertical_slice)
    for field in (
        "initial_cash",
        "thresholds",
        "maximum_allocation",
        "transaction_cost",
        "evaluation_dates",
        "execution_policy",
        "metric_assumptions",
    ):
        assert all(config[field] == configs[0][field] for config in configs[1:])

    assert configs[0]["initial_cash"] == "100000.00"
    assert configs[0]["thresholds"] == {
        "buy_probability_up_gte": 0.6,
        "sell_probability_up_lte": 0.4,
    }
    assert configs[0]["maximum_allocation"] == 0.5
    assert configs[0]["transaction_cost"] == {
        "type": "proportional_bps",
        "value": 10.0,
    }
    assert configs[0]["execution_policy"]["slippage_bps"] == 0.0

    calls = completed_vertical_slice.backtest_calls
    assert len(calls) == 4
    assert len({Path(call["prices_path"]).resolve() for call in calls}) == 1
    assert len({call["prices_sha256"] for call in calls}) == 1
    common_market = tuple(
        (bar.date, bar.open, bar.close)
        for bar in calls[0]["completed"].strategy.market_bars
    )
    assert common_market
    for call in calls[1:]:
        assert tuple(
            (bar.date, bar.open, bar.close)
            for bar in call["completed"].strategy.market_bars
        ) == common_market

    date_pairs = [call["prediction_date_pairs"] for call in calls]
    assert all(pairs == date_pairs[0] for pairs in date_pairs[1:])


def test_each_model_experiment_has_a_unique_portfolio_id(
    completed_vertical_slice: CompletedVerticalSlice,
):
    configs = _experiment_configs(completed_vertical_slice)
    strategy_ids = [config["portfolio_id"] for config in configs]
    assert len(strategy_ids) == len(set(strategy_ids)) == 4

    all_ids = []
    for config in configs:
        all_ids.extend(
            [
                config["portfolio_id"],
                config["benchmark_portfolio_ids"]["cash_only"],
                config["benchmark_portfolio_ids"]["buy_and_hold"],
            ]
        )
    assert len(all_ids) == len(set(all_ids)) == 12


def test_orchestrator_refuses_to_overwrite_an_old_run(
    completed_vertical_slice: CompletedVerticalSlice,
):
    before = _tree_hashes(completed_vertical_slice.artifact_path)

    def acquisition_must_not_run(**_: Any) -> PriceAcquisition:
        raise AssertionError("collision must be rejected before acquisition")

    with pytest.raises(FileExistsError, match="overwrite|already exists|exists"):
        run_ogdc_vertical_slice(
            output_dir=completed_vertical_slice.output_dir,
            model_output_dir=completed_vertical_slice.model_output_dir,
            cached_prices_path=completed_vertical_slice.output_dir / "missing.csv",
            vertical_slice_run_id="pytest-smoke",
            acquire_fn=acquisition_must_not_run,
        )

    assert _tree_hashes(completed_vertical_slice.artifact_path) == before


def test_failed_model_prevents_a_false_successful_final_report(tmp_path: Path):
    def failed_training(**_: Any) -> dict[str, Any]:
        raise RuntimeError("M-003 training sentinel failure")

    with pytest.raises(RuntimeError, match="M-003 training sentinel failure"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="failed-model-run",
            acquire_fn=_fixture_acquisition,
            train_fn=failed_training,
        )

    final_run = tmp_path / "vertical" / "failed-model-run"
    assert not final_run.exists()
    assert not list((tmp_path / "vertical").glob("**/final_summary.json"))


def test_live_data_failure_never_uses_synthetic_cached_data(tmp_path: Path):
    cached = tmp_path / "psx_prices.csv"
    _price_fixture(source="sample").to_csv(cached, index=False)
    train_called = False

    def live_failure(**_: Any) -> PriceAcquisition:
        raise RuntimeError("live provider sentinel HTTP 403")

    def training_must_not_run(**_: Any) -> dict[str, Any]:
        nonlocal train_called
        train_called = True
        raise AssertionError("synthetic prices must not reach training")

    with pytest.raises(Exception) as failure:
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=cached,
            vertical_slice_run_id="reject-sample-cache",
            acquire_fn=live_failure,
            train_fn=training_must_not_run,
        )

    message = str(failure.value).lower()
    assert "live provider sentinel http 403" in message
    assert "sample" in message or "synthetic" in message or "real" in message
    assert train_called is False
    assert not (tmp_path / "vertical" / "reject-sample-cache").exists()


def test_pipeline_produced_price_pair_confirms_ogdc_and_date_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cached, _, prices = _pipeline_produced_psx_contract_fixture(tmp_path, monkeypatch)

    validated = validate_real_price_file(cached, symbol="OGDC", start="2025-02-03")

    assert validated.source == "psx_dps"
    assert validated.date_start == "2025-02-03"
    assert validated.date_end == pd.Timestamp(prices["date"].max()).strftime("%Y-%m-%d")
    assert validated.row_count == int((prices["date"] >= "2025-02-03").sum())
    assert validated.prices_sha256 == _sha256(cached)


def test_relabelled_fixture_cache_is_rejected_by_acquisition_checksum(tmp_path: Path):
    cached = tmp_path / "psx_prices.csv"
    prices = _price_fixture(source="sample")
    prices.to_csv(cached, index=False)
    _write_price_provenance(cached, provider="psx_dps")
    prices["source"] = "psx_dps"
    prices.to_csv(cached, index=False)

    with pytest.raises(ValueError, match="checksum|SHA-256"):
        validate_real_price_file(cached)


@pytest.mark.parametrize(
    ("bad_source", "message"),
    [(None, "missing|null"), ("   ", "blank|non-empty")],
    ids=["null-source", "blank-source"],
)
def test_every_cached_ogdc_row_requires_nonblank_source(
    tmp_path: Path,
    bad_source: str | None,
    message: str,
):
    cached = tmp_path / "psx_prices.csv"
    prices = _price_fixture(source="psx_dps")
    prices.loc[0, "source"] = bad_source
    prices.to_csv(cached, index=False)
    _write_price_provenance(cached)

    with pytest.raises(ValueError, match=message):
        validate_real_price_file(cached)


def test_cached_ogdc_rows_cannot_mix_allowed_real_providers(tmp_path: Path):
    cached = tmp_path / "psx_prices.csv"
    prices = _price_fixture(source="psx_dps")
    prices.loc[0, "source"] = "pypsx_toolkit"
    prices.to_csv(cached, index=False)
    _write_price_provenance(cached, provider="psx_dps")

    with pytest.raises(ValueError, match="source|provider|mixed"):
        validate_real_price_file(cached)


def test_cached_price_mutation_after_acquisition_invalidates_provenance(tmp_path: Path):
    cached = tmp_path / "psx_prices.csv"
    prices = _price_fixture(source="psx_dps")
    prices.to_csv(cached, index=False)
    _write_price_provenance(cached)
    prices.loc[0, "close"] += 1.0
    prices.to_csv(cached, index=False)

    with pytest.raises(ValueError, match="checksum|SHA-256"):
        validate_real_price_file(cached)


@pytest.mark.parametrize(
    "metadata_mutation",
    [
        lambda metadata: metadata.pop("producer"),
        lambda metadata: metadata["raw_row_counts"].pop("OGDC"),
        lambda metadata: metadata["errors"].update({"OGDC": "provider failed"}),
    ],
    ids=["missing-producer", "missing-raw-rows", "provider-error"],
)
def test_cached_metadata_must_record_a_successful_pipeline_acquisition(
    tmp_path: Path,
    metadata_mutation: Any,
):
    cached = tmp_path / "psx_prices.csv"
    _price_fixture(source="psx_dps").to_csv(cached, index=False)
    metadata_path = _write_price_provenance(cached)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_mutation(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="producer|acquisition|metadata"):
        validate_real_price_file(cached)


def test_cached_prices_are_snapshotted_before_training(tmp_path: Path):
    cached = tmp_path / "psx_prices.csv"
    original = _price_fixture(source="psx_dps")
    original.to_csv(cached, index=False)
    _write_price_provenance(cached)
    accepted_hash = _sha256(cached)

    def live_failure(**_: Any) -> PriceAcquisition:
        raise RuntimeError("live provider unavailable")

    def inspect_snapshot_then_stop(**kwargs: Any) -> dict[str, Any]:
        supplied = Path(kwargs["prices_path"])
        assert supplied.resolve() != cached.resolve()
        replacement = original.copy()
        replacement["source"] = "sample"
        replacement.to_csv(cached, index=False)
        assert _sha256(supplied) == accepted_hash
        assert set(pd.read_csv(supplied)["source"]) == {"psx_dps"}
        raise RuntimeError("snapshot verified")

    with pytest.raises(RuntimeError, match="snapshot verified"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=cached,
            vertical_slice_run_id="snapshot-cache",
            acquire_fn=live_failure,
            train_fn=inspect_snapshot_then_stop,
        )

    assert not (tmp_path / "vertical" / "snapshot-cache").exists()


def test_mutating_the_private_snapshot_during_training_aborts_the_run(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    reference = json.loads(
        (
            completed_vertical_slice.artifact_path
            / "model_training_run_reference.json"
        ).read_text(encoding="utf-8")
    )

    def mutate_snapshot(**kwargs: Any) -> dict[str, Any]:
        supplied = Path(kwargs["prices_path"])
        prices = pd.read_csv(supplied)
        prices.loc[0, "close"] = float(prices.loc[0, "close"]) + 1.0
        prices.to_csv(supplied, index=False)
        return {
            "run_id": reference["training_run_id"],
            "invocation_id": reference["training_invocation_id"],
            "artifact_directory": reference["artifact_directory"],
            "model_ids": reference["model_ids"],
            "prediction_paths": reference["prediction_paths"],
        }

    with pytest.raises(RuntimeError, match="snapshot changed"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="mutated-private-snapshot",
            acquire_fn=_fixture_acquisition,
            train_fn=mutate_snapshot,
        )

    assert not (tmp_path / "vertical" / "mutated-private-snapshot").exists()


def test_mutating_snapshotted_provenance_during_training_aborts_the_run(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    cached = tmp_path / "psx_prices.csv"
    _price_fixture(source="psx_dps").to_csv(cached, index=False)
    _write_price_provenance(cached)
    reference = json.loads(
        (
            completed_vertical_slice.artifact_path
            / "model_training_run_reference.json"
        ).read_text(encoding="utf-8")
    )

    def live_failure(**_: Any) -> PriceAcquisition:
        raise RuntimeError("forced live outage")

    def mutate_provenance(**kwargs: Any) -> dict[str, Any]:
        metadata_path = Path(kwargs["prices_path"]).parent / "pipeline_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["generated_at"] = "2099-01-01T00:00:00"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return {
            "run_id": reference["training_run_id"],
            "invocation_id": reference["training_invocation_id"],
            "artifact_directory": reference["artifact_directory"],
            "model_ids": reference["model_ids"],
            "prediction_paths": reference["prediction_paths"],
        }

    with pytest.raises(RuntimeError, match="provenance snapshot changed"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=cached,
            vertical_slice_run_id="mutated-provenance-snapshot",
            acquire_fn=live_failure,
            train_fn=mutate_provenance,
        )

    assert not (tmp_path / "vertical" / "mutated-provenance-snapshot").exists()


def test_invalid_live_result_does_not_open_cached_fallback(tmp_path: Path):
    cached = tmp_path / "cached.csv"
    _price_fixture(source="psx_dps").to_csv(cached, index=False)
    invalid_live = tmp_path / "invalid-live.csv"
    invalid = _price_fixture(source="psx_dps")
    invalid.loc[0, "open"] = -1
    invalid.to_csv(invalid_live, index=False)
    _write_price_provenance(invalid_live)
    train_called = False

    def invalid_acquisition(**_: Any) -> PriceAcquisition:
        return PriceAcquisition(
            prices_path=invalid_live,
            source="psx_dps",
            mode="live_psx_dps",
            is_real_data=True,
        )

    def training_must_not_run(**_: Any) -> dict[str, Any]:
        nonlocal train_called
        train_called = True
        raise AssertionError("invalid live data must fail before cached fallback")

    with pytest.raises(ValueError, match="strictly positive"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=cached,
            vertical_slice_run_id="invalid-live-result",
            acquire_fn=invalid_acquisition,
            train_fn=training_must_not_run,
        )

    assert train_called is False
    assert not (tmp_path / "vertical" / "invalid-live-result").exists()


def test_combined_leaderboard_calculations_and_separate_rankings_are_correct():
    model_metrics = pd.DataFrame(
        [
            {
                "model_id": "M-001",
                "registered_model_id": "registered-001",
                "algorithm": "random_forest",
                "validation_accuracy": 0.70,
                "validation_balanced_accuracy": 0.75,
                "validation_f1": 0.72,
                "test_accuracy": 0.51,
                "test_balanced_accuracy": 0.52,
                "test_f1": 0.50,
            },
            {
                "model_id": "M-002",
                "registered_model_id": "registered-002",
                "algorithm": "random_forest",
                "validation_accuracy": 0.60,
                "validation_balanced_accuracy": 0.65,
                "validation_f1": 0.64,
                "test_accuracy": 0.91,
                "test_balanced_accuracy": 0.90,
                "test_f1": 0.89,
            },
            {
                "model_id": "M-003",
                "registered_model_id": "registered-003",
                "algorithm": "random_forest",
                "validation_accuracy": 0.58,
                "validation_balanced_accuracy": 0.57,
                "validation_f1": 0.56,
                "test_accuracy": 0.59,
                "test_balanced_accuracy": 0.58,
                "test_f1": 0.57,
            },
            {
                "model_id": "M-004",
                "registered_model_id": "registered-004",
                "algorithm": "xgboost",
                "validation_accuracy": 0.55,
                "validation_balanced_accuracy": 0.54,
                "validation_f1": 0.53,
                "test_accuracy": 0.56,
                "test_balanced_accuracy": 0.55,
                "test_f1": 0.54,
            },
        ]
    )
    portfolio_metrics = pd.DataFrame(
        [
            {
                "model_id": "M-001",
                "registered_model_id": "registered-001",
                "portfolio_id": "portfolio-001",
                "portfolio_type": "model",
                "ending_portfolio_value": 110_000.0,
                "total_return": 0.10,
                "maximum_drawdown": 0.03,
                "volatility": 0.12,
                "sharpe_ratio": 1.1,
                "number_of_trades": 4,
                "fees_paid": 50.0,
                "result_versus_buy_and_hold": 0.02,
            },
            {
                "model_id": "M-002",
                "registered_model_id": "registered-002",
                "portfolio_id": "portfolio-002",
                "portfolio_type": "model",
                "ending_portfolio_value": 105_000.0,
                "total_return": 0.05,
                "maximum_drawdown": 0.01,
                "volatility": 0.08,
                "sharpe_ratio": 0.8,
                "number_of_trades": 2,
                "fees_paid": 25.0,
                "result_versus_buy_and_hold": -0.03,
            },
            {
                "model_id": "M-003",
                "registered_model_id": "registered-003",
                "portfolio_id": "portfolio-003",
                "portfolio_type": "model",
                "ending_portfolio_value": 103_000.0,
                "total_return": 0.03,
                "maximum_drawdown": 0.02,
                "volatility": 0.09,
                "sharpe_ratio": 0.6,
                "number_of_trades": 3,
                "fees_paid": 30.0,
                "result_versus_buy_and_hold": -0.05,
            },
            {
                "model_id": "M-004",
                "registered_model_id": "registered-004",
                "portfolio_id": "portfolio-004",
                "portfolio_type": "model",
                "ending_portfolio_value": 101_000.0,
                "total_return": 0.01,
                "maximum_drawdown": 0.04,
                "volatility": 0.11,
                "sharpe_ratio": 0.2,
                "number_of_trades": 5,
                "fees_paid": 60.0,
                "result_versus_buy_and_hold": -0.07,
            },
        ]
    )

    leaderboard = build_combined_leaderboard(model_metrics, portfolio_metrics)

    assert leaderboard["model_id"].tolist() == [
        "M-001",
        "M-002",
        "M-003",
        "M-004",
    ]
    first = leaderboard.set_index("model_id").loc["M-001"]
    second = leaderboard.set_index("model_id").loc["M-002"]
    assert first["validation_balanced_accuracy"] == pytest.approx(0.75)
    assert first["validation_f1"] == pytest.approx(0.72)
    assert first["ending_portfolio_value"] == pytest.approx(110_000.0)
    assert first["result_versus_buy_and_hold"] == pytest.approx(0.02)
    assert second["test_accuracy"] == pytest.approx(0.91)
    # Higher test accuracy does not reorder or replace the validation choice.
    assert leaderboard.iloc[0]["model_id"] == "M-001"


def test_model_predictions_reach_backtester_without_schema_conversion(
    completed_vertical_slice: CompletedVerticalSlice,
):
    calls = completed_vertical_slice.backtest_calls
    assert len(calls) == 4
    for call in calls:
        prediction_path = Path(call["predictions_path"])
        assert prediction_path.name == "predictions.csv"
        assert prediction_path.parent.parent.name == "models"
        assert call["prediction_sha256_before"] == call["prediction_sha256_after"]
        assert call["prediction_columns"] == PREDICTION_COLUMNS

    reference = json.loads(
        (completed_vertical_slice.artifact_path / "model_training_run_reference.json").read_text(
            encoding="utf-8"
        )
    )
    referenced_paths = {
        variant_id: Path(path).resolve()
        for variant_id, path in reference["prediction_paths"].items()
    }
    assert set(referenced_paths) == EXPECTED_VARIANTS
    for call in calls:
        public_path = referenced_paths[call["variant_id"]]
        assert public_path.is_file()
        assert _sha256(public_path) == call["prediction_sha256_before"]


def test_backtester_cannot_replace_the_supplied_model_predictions(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    reference = json.loads(
        (
            completed_vertical_slice.artifact_path
            / "model_training_run_reference.json"
        ).read_text(encoding="utf-8")
    )
    calls = 0

    def existing_training(**_: Any) -> dict[str, Any]:
        return {
            "run_id": reference["training_run_id"],
            "invocation_id": reference["training_invocation_id"],
            "artifact_directory": reference["artifact_directory"],
            "model_ids": reference["model_ids"],
            "prediction_paths": reference["prediction_paths"],
        }

    def replacing_backtester(**kwargs: Any):
        nonlocal calls
        calls += 1
        changed = pd.read_csv(kwargs["predictions_path"])
        original_up = changed["probability_up"].astype(float)
        changed["probability_up"] = 1.0 - original_up
        changed["probability_down"] = original_up
        changed["predicted_class"] = (
            changed["probability_up"] > changed["probability_down"]
        ).astype(int)
        replacement = tmp_path / f"replacement-{calls}.csv"
        changed.to_csv(replacement, index=False)
        return run_experiment(
            predictions_path=replacement,
            prices_path=kwargs["prices_path"],
            config_path=kwargs["config_path"],
            runs_root=kwargs["runs_root"],
        )

    with pytest.raises(ValueError, match="differs|prediction"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="replaced-model-predictions",
            acquire_fn=_fixture_acquisition,
            train_fn=existing_training,
            backtest_fn=replacing_backtester,
        )

    assert calls == 4
    assert not (tmp_path / "vertical" / "replaced-model-predictions").exists()


def test_backtester_cannot_mutate_a_training_prediction_after_using_it(
    tmp_path: Path,
):
    calls = 0

    def mutating_backtester(**kwargs: Any):
        nonlocal calls
        completed = run_experiment(**kwargs)
        calls += 1
        if calls == 1:
            path = Path(kwargs["predictions_path"])
            changed = pd.read_csv(path)
            original_up = float(changed.loc[0, "probability_up"])
            changed.loc[0, "probability_up"] = 1.0 - original_up
            changed.loc[0, "probability_down"] = original_up
            changed.to_csv(path, index=False)
        return completed

    output_dir = tmp_path / "vertical"
    model_output_dir = tmp_path / "models"
    with pytest.raises(RuntimeError, match="prediction artifact changed"):
        run_ogdc_vertical_slice(
            output_dir=output_dir,
            model_output_dir=model_output_dir,
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="mutated-training-predictions",
            acquire_fn=_fixture_acquisition,
            backtest_fn=mutating_backtester,
        )

    assert calls == 4
    assert not (output_dir / "mutated-training-predictions").exists()
    assert model_output_dir.is_dir()
    assert not any(model_output_dir.iterdir())


def test_backtester_cannot_mutate_the_trainer_frozen_price_evidence(
    tmp_path: Path,
):
    calls = 0

    def mutating_backtester(**kwargs: Any):
        nonlocal calls
        if calls == 0:
            path = Path(kwargs["prices_path"])
            changed = pd.read_csv(path)
            changed.loc[0, "open"] = float(changed.loc[0, "open"]) + 1.0
            changed.to_csv(path, index=False)
        calls += 1
        return run_experiment(**kwargs)

    output_dir = tmp_path / "vertical"
    model_output_dir = tmp_path / "models"
    with pytest.raises(RuntimeError, match="price snapshot changed"):
        run_ogdc_vertical_slice(
            output_dir=output_dir,
            model_output_dir=model_output_dir,
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="mutated-frozen-prices",
            acquire_fn=_fixture_acquisition,
            backtest_fn=mutating_backtester,
        )

    assert calls == 4
    assert not (output_dir / "mutated-frozen-prices").exists()
    assert model_output_dir.is_dir()
    assert not any(model_output_dir.iterdir())


def test_stale_prediction_metrics_prevent_backtesting(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    reference = json.loads(
        (completed_vertical_slice.artifact_path / "model_training_run_reference.json").read_text(
            encoding="utf-8"
        )
    )
    source_training = Path(reference["artifact_directory"])
    copied_training = tmp_path / "copied-training"
    shutil.copytree(source_training, copied_training)
    model_ids = dict(reference["model_ids"])
    prediction_paths = {
        variant_id: str(
            copied_training / "models" / registered_id / "predictions.csv"
        )
        for variant_id, registered_id in model_ids.items()
    }
    tampered_path = Path(prediction_paths["M-001"])
    tampered = pd.read_csv(tampered_path)
    tampered.loc[0, "actual_class"] = 1 - int(tampered.loc[0, "actual_class"])
    tampered.to_csv(tampered_path, index=False)
    # This test intentionally exercises semantic reconciliation after a
    # coordinated, trusted reseal. Untrusted byte changes are covered by the
    # dedicated inventory-tampering regression below.
    seal_training_run_artifacts(copied_training)
    run_manifest = json.loads(
        (copied_training / "run_manifest.json").read_text(encoding="utf-8")
    )

    def stale_training(**_: Any) -> dict[str, Any]:
        return {
            "run_id": run_manifest["run_id"],
            "invocation_id": run_manifest["invocation_id"],
            "artifact_directory": str(copied_training),
            "model_ids": model_ids,
            "prediction_paths": prediction_paths,
        }

    def backtest_must_not_run(**_: Any):
        raise AssertionError("stale prediction evidence must fail before backtesting")

    with pytest.raises(ValueError, match="class distribution|does not match predictions"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing-cache.csv",
            vertical_slice_run_id="stale-predictions",
            acquire_fn=_fixture_acquisition,
            train_fn=stale_training,
            backtest_fn=backtest_must_not_run,
        )


def test_training_manifest_tamper_is_rejected_before_backtesting(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    reference = json.loads(
        (
            completed_vertical_slice.artifact_path
            / "model_training_run_reference.json"
        ).read_text(encoding="utf-8")
    )
    copied_training = tmp_path / "copied-training"
    shutil.copytree(Path(reference["artifact_directory"]), copied_training)
    run_manifest = json.loads(
        (copied_training / "run_manifest.json").read_text(encoding="utf-8")
    )
    model_ids = dict(reference["model_ids"])
    prediction_paths = {
        variant_id: str(
            copied_training / "models" / registered_id / "predictions.csv"
        )
        for variant_id, registered_id in model_ids.items()
    }
    manifest_path = (
        copied_training / "models" / model_ids["M-001"] / "manifest.json"
    )
    model_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_manifest["review_tamper"] = "validation evidence changed"
    manifest_path.write_text(json.dumps(model_manifest), encoding="utf-8")

    def tampered_training(**_: Any) -> dict[str, Any]:
        return {
            "run_id": run_manifest["run_id"],
            "invocation_id": run_manifest["invocation_id"],
            "artifact_directory": str(copied_training),
            "model_ids": model_ids,
            "prediction_paths": prediction_paths,
        }

    def backtest_must_not_run(**_: Any):
        raise AssertionError("tampered training evidence must fail before backtesting")

    with pytest.raises(ValueError, match="inventory|checksum|artifact"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="tampered-model-manifest",
            acquire_fn=_fixture_acquisition,
            train_fn=tampered_training,
            backtest_fn=backtest_must_not_run,
        )


def test_external_prediction_paths_are_rejected_before_backtesting(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    reference = json.loads(
        (
            completed_vertical_slice.artifact_path
            / "model_training_run_reference.json"
        ).read_text(encoding="utf-8")
    )
    external_root = tmp_path / "external-predictions"
    external_root.mkdir()
    external_paths: dict[str, str] = {}
    for variant_id, source in reference["prediction_paths"].items():
        destination = external_root / f"{variant_id}.csv"
        shutil.copyfile(source, destination)
        external_paths[variant_id] = str(destination)

    def external_training(**_: Any) -> dict[str, Any]:
        return {
            "run_id": reference["training_run_id"],
            "invocation_id": reference["training_invocation_id"],
            "artifact_directory": reference["artifact_directory"],
            "model_ids": reference["model_ids"],
            "prediction_paths": external_paths,
        }

    def backtest_must_not_run(**_: Any):
        raise AssertionError("external predictions must fail before backtesting")

    with pytest.raises(ValueError, match="prediction paths|checksum-inventoried"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="external-predictions",
            acquire_fn=_fixture_acquisition,
            train_fn=external_training,
            backtest_fn=backtest_must_not_run,
        )


def test_unsafe_summary_model_id_is_rejected_before_artifact_reads(
    completed_vertical_slice: CompletedVerticalSlice,
    tmp_path: Path,
):
    reference = json.loads(
        (
            completed_vertical_slice.artifact_path
            / "model_training_run_reference.json"
        ).read_text(encoding="utf-8")
    )
    unsafe_model_ids = dict(reference["model_ids"])
    unsafe_model_ids["M-001"] = "../outside-run"

    def unsafe_training(**_: Any) -> dict[str, Any]:
        return {
            "run_id": reference["training_run_id"],
            "invocation_id": reference["training_invocation_id"],
            "artifact_directory": reference["artifact_directory"],
            "model_ids": unsafe_model_ids,
            "prediction_paths": reference["prediction_paths"],
        }

    def backtest_must_not_run(**_: Any):
        raise AssertionError("unsafe model IDs must fail before backtesting")

    with pytest.raises(ValueError, match="model ID|filesystem-safe"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="unsafe-summary-model-id",
            acquire_fn=_fixture_acquisition,
            train_fn=unsafe_training,
            backtest_fn=backtest_must_not_run,
        )


def test_backtester_cannot_mutate_an_inventoried_model_manifest(tmp_path: Path):
    calls = 0

    def mutating_backtester(**kwargs: Any):
        nonlocal calls
        completed = run_experiment(**kwargs)
        calls += 1
        if calls == 1:
            prediction_path = Path(kwargs["predictions_path"])
            training_root = prediction_path.parents[2]
            manifest_path = prediction_path.parent / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["review_tamper"] = str(training_root)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return completed

    output_dir = tmp_path / "vertical"
    model_output_dir = tmp_path / "models"
    with pytest.raises(RuntimeError, match="training|manifest|artifact|inventory"):
        run_ogdc_vertical_slice(
            output_dir=output_dir,
            model_output_dir=model_output_dir,
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="mutated-model-manifest",
            acquire_fn=_fixture_acquisition,
            backtest_fn=mutating_backtester,
        )

    assert calls == 4
    assert not (output_dir / "mutated-model-manifest").exists()
    assert model_output_dir.is_dir()
    assert not any(model_output_dir.iterdir())


def test_complete_fixture_based_end_to_end_smoke_run(
    completed_vertical_slice: CompletedVerticalSlice,
):
    artifact = completed_vertical_slice.artifact_path
    assert artifact.is_dir()
    assert REQUIRED_OUTPUTS.issubset({path.name for path in artifact.iterdir()})
    assert {path.stem for path in (artifact / "experiment_configs").glob("*.json")} == EXPECTED_VARIANTS

    final_summary = json.loads((artifact / "final_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert completed_vertical_slice.summary == final_summary
    assert final_summary["status"] == "PASS"
    assert "not financial advice" in final_summary["disclaimer"].lower()
    assert final_summary["data_source"]["source"] == "deterministic_fixture"
    assert final_summary["data_source"]["mode"] == "fixture_test"
    assert final_summary["data_source"]["acquisition_mode"] == "fixture_test"
    assert final_summary["data_source"]["is_real_data"] is False
    assert final_summary["data_source"]["live_fetch_succeeded"] is False
    assert final_summary["data_source"]["legacy_metadata_preserved"] is False
    assert manifest["status"] == "PASS"
    assert manifest["data"] == final_summary["data_source"]
    assert set(manifest["registered_model_ids"]) == EXPECTED_VARIANTS
    assert set(manifest["portfolio_ids"]) == EXPECTED_VARIANTS
    assert len(
        {
            portfolio_id
            for identities in manifest["portfolio_ids"].values()
            for portfolio_id in identities.values()
        }
    ) == 12
    for key in (
        "best_validation_prediction_model",
        "best_test_period_simulated_portfolio",
        "lowest_drawdown_portfolio",
        "buy_and_hold_result",
        "any_model_portfolio_beat_buy_and_hold",
    ):
        assert key in final_summary
    assert "distinct measurements" in " ".join(final_summary["warnings"])
    assert sum(final_summary["common_split_counts"].values()) > 0
    for split_name, split_count in final_summary["common_split_counts"].items():
        assert sum(final_summary["class_distributions"][split_name].values()) == split_count

    training_reference = json.loads(
        (artifact / "model_training_run_reference.json").read_text(encoding="utf-8")
    )
    training_evidence = validate_training_run_artifacts(
        training_reference["artifact_directory"],
        expected_run_id=training_reference["training_run_id"],
        expected_invocation_id=training_reference["training_invocation_id"],
    )
    assert training_reference["run_manifest_sha256"] == _sha256(
        Path(training_reference["run_manifest_path"])
    )
    assert (
        training_reference["run_manifest_sha256"]
        == training_evidence.run_manifest_sha256
    )
    assert training_reference["training_source_provenance"] == {
        "code_version": training_evidence.code_version,
        "dirty_tree": training_evidence.dirty_tree,
        "source_fingerprint": training_evidence.source_fingerprint,
    }
    assert training_reference["training_artifact_inventory"]["files"] == [
        {"path": path, "sha256": checksum}
        for path, checksum in training_evidence.artifact_sha256
    ]

    model_metrics = pd.read_csv(artifact / "model_metrics.csv")
    portfolio_metrics = pd.read_csv(artifact / "portfolio_metrics.csv")
    leaderboard = pd.read_csv(artifact / "combined_leaderboard.csv")
    equity = pd.read_csv(artifact / "daily_equity_curves.csv")
    assert set(model_metrics["model_id"]) == EXPECTED_VARIANTS
    assert set(leaderboard["model_id"]) == EXPECTED_VARIANTS
    assert {
        "model_id",
        "registered_model_id",
        "portfolio_id",
        "algorithm",
        "validation_balanced_accuracy",
        "validation_f1",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_f1",
        "ending_portfolio_value",
        "total_return",
        "maximum_drawdown",
        "volatility",
        "sharpe_ratio",
        "number_of_trades",
        "fees_paid",
        "result_versus_buy_and_hold",
    }.issubset(leaderboard.columns)
    assert len(portfolio_metrics) == 6
    assert portfolio_metrics["portfolio_id"].nunique() == 6
    assert equity["portfolio_id"].nunique() == 6

    buy_hold_return = float(
        portfolio_metrics.loc[
            portfolio_metrics["portfolio_type"].eq("buy_and_hold"), "total_return"
        ].iloc[0]
    )
    model_portfolios = portfolio_metrics.loc[
        portfolio_metrics["portfolio_type"].eq("model")
    ].set_index("model_id")
    model_scores = model_metrics.set_index("model_id")
    for row in leaderboard.to_dict(orient="records"):
        portfolio = model_portfolios.loc[row["model_id"]]
        model = model_scores.loc[row["model_id"]]
        assert row["ending_portfolio_value"] == pytest.approx(
            float(portfolio["ending_portfolio_value"])
        )
        assert row["result_versus_buy_and_hold"] == pytest.approx(
            float(portfolio["total_return"]) - buy_hold_return
        )
        assert row["validation_balanced_accuracy"] == pytest.approx(
            float(model["validation_balanced_accuracy"])
        )

    for filename in (
        "model_metric_comparison.png",
        "portfolio_equity_comparison.png",
        "drawdown_comparison.png",
    ):
        assert (artifact / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_successful_run_retains_and_checksums_all_experiment_ledgers(
    completed_vertical_slice: CompletedVerticalSlice,
):
    artifact = completed_vertical_slice.artifact_path
    experiment_root = artifact / "experiments"
    experiment_dirs = sorted(path for path in experiment_root.iterdir() if path.is_dir())
    assert len(experiment_dirs) == 4

    retained_portfolios: set[str] = set()
    for experiment_dir in experiment_dirs:
        files = {
            path.relative_to(experiment_dir).as_posix()
            for path in experiment_dir.rglob("*")
            if path.is_file()
        }
        assert files == EXPECTED_EXPERIMENT_FILES
        config = json.loads((experiment_dir / "config.json").read_text(encoding="utf-8"))
        result = json.loads((experiment_dir / "result.json").read_text(encoding="utf-8"))
        metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
        assert result == metrics
        assert result["experiment_id"] == config["experiment_id"] == experiment_dir.name
        assert result["model_id"] == config["model_id"]
        retained_portfolios.update(
            {
                config["portfolio_id"],
                config["benchmark_portfolio_ids"]["cash_only"],
                config["benchmark_portfolio_ids"]["buy_and_hold"],
            }
        )
    assert len(retained_portfolios) == 12

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    retained_files = {
        path.relative_to(artifact).as_posix(): _sha256(path)
        for path in experiment_root.rglob("*")
        if path.is_file()
    }
    assert retained_files
    assert all(manifest_paths[path] == digest for path, digest in retained_files.items())
    assert not any(path.startswith(".work/") for path in manifest_paths)


def test_aggregate_reports_reconcile_to_retained_strategy_ledgers(
    completed_vertical_slice: CompletedVerticalSlice,
):
    artifact = completed_vertical_slice.artifact_path
    portfolio_metrics = pd.read_csv(artifact / "portfolio_metrics.csv").set_index(
        "model_id"
    )
    curves = pd.read_csv(artifact / "daily_equity_curves.csv")
    configs = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (artifact / "experiment_configs").glob("*.json")
    }
    for variant_id, config in configs.items():
        experiment_dir = artifact / "experiments" / config["experiment_id"]
        result = json.loads((experiment_dir / "result.json").read_text(encoding="utf-8"))
        snapshots = pd.read_csv(experiment_dir / "portfolio_snapshots.csv")
        trades = pd.read_csv(experiment_dir / "trades.csv")
        aggregate = portfolio_metrics.loc[variant_id]
        assert aggregate["ending_portfolio_value"] == pytest.approx(
            float(result["ending_equity"])
        )
        assert aggregate["total_return"] == pytest.approx(result["total_return"])
        assert aggregate["maximum_drawdown"] == pytest.approx(
            result["maximum_drawdown"]
        )
        assert int(aggregate["number_of_trades"]) == len(trades)
        strategy_curve = curves.loc[curves["series_id"].eq(variant_id)].reset_index(
            drop=True
        )
        assert strategy_curve["date"].tolist() == snapshots["date"].tolist()
        assert pd.to_numeric(strategy_curve["total_equity"]).tolist() == pytest.approx(
            pd.to_numeric(snapshots["total_equity"]).tolist()
        )


def test_pipeline_produced_cached_pair_completes_vertical_slice_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    cached, metadata, _ = _pipeline_produced_psx_contract_fixture(tmp_path, monkeypatch)

    def live_failure(**_: Any) -> PriceAcquisition:
        raise RuntimeError("forced live outage")

    summary = run_ogdc_vertical_slice(
        output_dir=tmp_path / "vertical",
        model_output_dir=tmp_path / "models",
        cached_prices_path=cached,
        vertical_slice_run_id="cached-real-contract",
        acquire_fn=live_failure,
    )

    assert summary["status"] == "PASS"
    assert summary["data_source"]["mode"] == "cached_real_fallback"
    assert summary["data_source"]["acquisition_mode"] == "cached_real_fallback"
    assert summary["data_source"]["is_real_data"] is True
    assert summary["data_source"]["live_fetch_succeeded"] is False
    assert summary["data_source"]["original_provider"] == "psx_dps"
    assert summary["data_source"]["legacy_metadata_preserved"] is False
    assert summary["data_source"]["producer_component"] == (
        "paisa.pipeline.run_v0_pipeline"
    )
    assert "forced live outage" in summary["data_source"]["live_acquisition_error"]
    assert summary["data_source"]["input_prices_sha256"] == _sha256(cached)
    retained_input = Path(summary["artifact_paths"]["input_snapshot"])
    assert _sha256(retained_input / cached.name) == _sha256(cached)
    assert _sha256(retained_input / "pipeline_metadata.json") == _sha256(metadata)
    training_reference = json.loads(
        Path(summary["artifact_paths"]["model_training_run_reference"]).read_text(
            encoding="utf-8"
        )
    )
    assert summary["data_source"]["frozen_prices_sha256"] == _sha256(
        training_reference["frozen_prices_path"]
    )
    assert summary["data_source"]["provenance_sha256"] == _sha256(metadata)


def test_verified_legacy_cache_completes_after_live_404_with_truthful_labels(
    completed_verified_legacy_slice: CompletedVerifiedLegacySlice,
):
    run = completed_verified_legacy_slice
    summary = run.summary
    artifact = run.artifact_path
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    training_reference = json.loads(
        (artifact / "model_training_run_reference.json").read_text(encoding="utf-8")
    )

    assert summary["status"] == "PASS"
    data = summary["data_source"]
    assert data["source"] == "psx_dps"
    assert data["mode"] == "verified_legacy_psx_cache"
    assert data["acquisition_mode"] == "verified_legacy_psx_cache"
    assert data["is_real_data"] is True
    assert data["live_fetch_succeeded"] is False
    assert data["original_provider"] == "psx_dps"
    assert data["legacy_metadata_preserved"] is True
    assert data["producer_component"] == (
        "paisa.legacy_cache.migrate_legacy_psx_cache"
    )
    assert data["migration_strategy"] == "rebuild_from_legacy_raw_captures"
    assert data["provenance_strength"] == (
        "raw_capture_rebuild_matches_legacy_processed_cache"
    )
    assert run.live_error_text in data["live_acquisition_error"]
    assert "HTTP 404 Not Found" in data["live_acquisition_error"]
    assert "response body: route not found" in data["live_acquisition_error"]

    assert manifest["status"] == "PASS"
    assert manifest["data"] == data
    assert manifest["data"]["mode"] == "verified_legacy_psx_cache"
    assert manifest["data"]["acquisition_mode"] == "verified_legacy_psx_cache"
    assert manifest["data"]["live_fetch_succeeded"] is False
    assert manifest["data"]["original_provider"] == "psx_dps"
    assert manifest["data"]["legacy_metadata_preserved"] is True

    assert training_reference["acquisition_mode"] == "verified_legacy_psx_cache"
    assert training_reference["live_fetch_succeeded"] is False
    assert training_reference["original_provider"] == "psx_dps"
    assert training_reference["legacy_metadata_preserved"] is True
    assert training_reference["provenance_producer_component"] == (
        "paisa.legacy_cache.migrate_legacy_psx_cache"
    )
    assert set(training_reference["additional_provenance_sha256"]) == {
        "legacy_pipeline_metadata",
        "migration_manifest",
    }

    expected_bundle_files = {
        "psx_prices.csv",
        "pipeline_metadata.json",
        "legacy_pipeline_metadata.json",
        "migration_manifest.json",
    }
    snapshot = artifact / "input_snapshot"
    assert {path.name for path in snapshot.iterdir()} == expected_bundle_files
    for filename in expected_bundle_files:
        assert _sha256(snapshot / filename) == _sha256(run.bundle_path / filename)


@pytest.mark.parametrize(
    "evidence_filename",
    ["migration_manifest.json", "legacy_pipeline_metadata.json"],
)
def test_mutating_verified_legacy_evidence_during_training_aborts_pass(
    completed_verified_legacy_slice: CompletedVerifiedLegacySlice,
    tmp_path: Path,
    evidence_filename: str,
):
    bundle = _migrated_legacy_price_bundle(tmp_path / "source")
    reference = json.loads(
        (
            completed_verified_legacy_slice.artifact_path
            / "model_training_run_reference.json"
        ).read_text(encoding="utf-8")
    )

    def live_404(**_: Any) -> PriceAcquisition:
        raise RuntimeError("PSX DPS HTTP 404 mutation-test sentinel")

    def mutate_evidence(**kwargs: Any) -> dict[str, Any]:
        evidence_path = Path(kwargs["prices_path"]).parent / evidence_filename
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        evidence["tampered_during_training"] = True
        evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
        return {
            "run_id": reference["training_run_id"],
            "invocation_id": reference["training_invocation_id"],
            "artifact_directory": reference["artifact_directory"],
            "model_ids": reference["model_ids"],
            "prediction_paths": reference["prediction_paths"],
        }

    with pytest.raises(RuntimeError, match="Authenticated .* snapshot changed"):
        run_ogdc_vertical_slice(
            output_dir=tmp_path / "vertical",
            model_output_dir=tmp_path / "models",
            cached_prices_path=bundle / "psx_prices.csv",
            vertical_slice_run_id=f"mutated-{Path(evidence_filename).stem}",
            acquire_fn=live_404,
            train_fn=mutate_evidence,
        )

    assert not (tmp_path / "vertical").joinpath(
        f"mutated-{Path(evidence_filename).stem}"
    ).exists()


def test_corrupt_fourth_experiment_rolls_back_model_and_vertical_publication(
    tmp_path: Path,
):
    calls = 0

    def corrupt_fourth_result(**kwargs: Any):
        nonlocal calls
        completed = run_experiment(**kwargs)
        calls += 1
        if calls == 4:
            result_path = completed.artifact_path / "result.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["total_return"] = float(result["total_return"]) + 0.5
            result_path.write_text(json.dumps(result), encoding="utf-8")
        return completed

    output_dir = tmp_path / "vertical"
    model_output_dir = tmp_path / "models"
    preexisting = model_output_dir / "preexisting-run"
    preexisting.mkdir(parents=True)
    (preexisting / "sentinel.txt").write_text("keep me", encoding="utf-8")
    before = _tree_hashes(model_output_dir)
    with pytest.raises(ValueError, match="result|metrics|persisted|reconcile"):
        run_ogdc_vertical_slice(
            output_dir=output_dir,
            model_output_dir=model_output_dir,
            cached_prices_path=tmp_path / "missing.csv",
            vertical_slice_run_id="late-corruption",
            acquire_fn=_fixture_acquisition,
            backtest_fn=corrupt_fourth_result,
        )

    assert calls == 4
    assert not (output_dir / "late-corruption").exists()
    assert model_output_dir.is_dir()
    assert _tree_hashes(model_output_dir) == before
    assert {path.name for path in model_output_dir.iterdir()} == {"preexisting-run"}
