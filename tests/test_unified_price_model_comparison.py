from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import stat
from typing import Any, Callable
import zipfile

import pandas as pd
import pytest

import paisa.experiments.artifacts as artifact_module
import paisa.experiments.runner as runner_module
import paisa.experiments.trainer as trainer_module
import paisa.experiments.unified_comparison as comparison
from paisa.experiments.artifacts import TrainingRunEvidence, sha256_file, write_json
from paisa.experiments.evaluation import PREDICTION_COLUMNS
from paisa.experiments.lstm_runner import DirectoryInventoryEvidence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_RUN = (
    REPOSITORY_ROOT
    / "artifacts/model_training/TRN-20261020-191815-624662-nogit"
)
VERTICAL_SLICE = (
    REPOSITORY_ROOT
    / "artifacts/vertical_slice/VSL-20261020-OGDC-main-02"
)
LSTM_RUN = (
    REPOSITORY_ROOT
    / "artifacts/lstm_experiments/LSTM-20261024-OGDC-price-only-02"
)


@dataclass(frozen=True)
class PortableFixture:
    output_root: Path
    pack: Path
    archive: Path
    checksum: Path
    authority: dict[str, dict[str, str]]
    snapshots: comparison.SourceSnapshots


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_rows(model_id: str, probabilities: list[float]) -> list[dict[str, Any]]:
    feature_dates = (
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
    )
    target_dates = (
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
    )
    actual = (0, 1, 0, 1, 1, 0)
    return [
        {
            "schema_version": "1.0",
            "model_id": model_id,
            "ticker": "OGDC",
            "feature_date": feature_date,
            "target_date": target_date,
            "predicted_class": int(probability > 0.5),
            "probability_up": probability,
            "probability_down": 1.0 - probability,
            "actual_class": actual_class,
        }
        for feature_date, target_date, probability, actual_class in zip(
            feature_dates,
            target_dates,
            probabilities,
            actual,
            strict=True,
        )
    ]


def _experiment_config(
    *, variant_id: str, model_id: str, experiment_id: str
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "model_id": model_id,
        "portfolio_id": f"fixture-{variant_id.lower()}-strategy",
        "benchmark_portfolio_ids": {
            "cash_only": f"fixture-{variant_id.lower()}-cash",
            "buy_and_hold": f"fixture-{variant_id.lower()}-buy-hold",
        },
        "symbol": "OGDC",
        "initial_cash": "100000.00",
        "thresholds": {
            "buy_probability_up_gte": 0.6,
            "sell_probability_up_lte": 0.4,
        },
        "maximum_allocation": 0.5,
        "transaction_cost": {"type": "proportional_bps", "value": 10},
        "evaluation_dates": {
            "feature_start": "2026-01-08",
            "feature_end": "2026-01-09",
            "target_start": "2026-01-09",
            "target_end": "2026-01-12",
        },
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
        "created_at": "2026-10-24T00:00:00Z",
    }


def _inventory(root: Path, *, excluded: set[str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), sha256_file(path))
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )


def _inventory_payload(
    entries: tuple[tuple[str, str], ...], *, excludes: list[str]
) -> dict[str, Any]:
    return {
        "algorithm": "sha256",
        "root": ".",
        "excludes": excludes,
        "files": [
            {"path": relative, "sha256": checksum}
            for relative, checksum in entries
        ],
    }


def _make_synthetic_sources(root: Path) -> tuple[
    comparison.SourceSnapshots,
    dict[str, dict[str, str]],
]:
    training = root / comparison.AUTHORITATIVE_SOURCE_ROOTS["baseline_training"][
        "run_id"
    ]
    vertical = root / comparison.AUTHORITATIVE_SOURCE_ROOTS[
        "baseline_vertical_slice"
    ]["run_id"]
    lstm = root / comparison.AUTHORITATIVE_SOURCE_ROOTS["lstm"]["run_id"]
    for directory in (training, vertical, lstm):
        directory.mkdir(parents=True)

    model_ids = {
        variant_id: f"{variant_id}-offline-fixture-model"
        for variant_id in comparison.MODEL_VARIANTS
    }
    probabilities = {
        "M-001": [0.2, 0.8, 0.3, 0.7, 0.8, 0.2],
        "M-002": [0.3, 0.7, 0.4, 0.6, 0.7, 0.3],
        "M-003": [0.4, 0.6, 0.7, 0.3, 0.6, 0.4],
        "M-004": [0.6, 0.4, 0.2, 0.8, 0.4, 0.6],
        "M-005": [0.2, 0.8, 0.6, 0.4, 0.7, 0.3],
        "M-006": [0.2, 0.8, 0.4, 0.6, 0.3, 0.7],
    }
    split_labels = ["train", "train", "validation", "validation", "test", "test"]
    for variant_id in comparison.BASELINE_VARIANTS:
        rows = _prediction_rows(model_ids[variant_id], probabilities[variant_id])
        rows_with_splits = [
            {**row, "split": split}
            for row, split in zip(rows, split_labels, strict=True)
        ]
        _write_csv(
            training
            / "models"
            / model_ids[variant_id]
            / "predictions_all_splits.csv",
            [*PREDICTION_COLUMNS, "split"],
            rows_with_splits,
        )
    for variant_id in comparison.LSTM_VARIANTS:
        rows = _prediction_rows(model_ids[variant_id], probabilities[variant_id])
        _write_csv(
            lstm
            / "models"
            / model_ids[variant_id]
            / "predictions_all_splits.csv",
            list(PREDICTION_COLUMNS),
            rows,
        )

    split_manifest = {
        "target_dates": {
            "train": ["2026-01-05", "2026-01-06"],
            "validation": ["2026-01-07", "2026-01-08"],
            "test": ["2026-01-09", "2026-01-12"],
        }
    }
    write_json(split_manifest, training / "split_manifest.json")
    write_json(split_manifest, lstm / "split_manifest.json")

    prices = root / "prices.csv"
    _write_csv(
        prices,
        ["date", "ticker", "open", "close"],
        [
            {"date": "2026-01-08", "ticker": "OGDC", "open": "100.00", "close": "100.00"},
            {"date": "2026-01-09", "ticker": "OGDC", "open": "102.00", "close": "103.00"},
            {"date": "2026-01-12", "ticker": "OGDC", "open": "104.00", "close": "101.00"},
        ],
    )
    for variant_id in comparison.MODEL_VARIANTS:
        if variant_id in comparison.BASELINE_VARIANTS:
            experiment_id = (
                f"{vertical.name}-{variant_id.lower()}"
            )
            runs_root = vertical / "experiments"
        else:
            experiment_id = f"{lstm.name}-{variant_id.lower()}"
            runs_root = lstm / "backtests"
        inputs = root / "backtest-inputs" / variant_id
        inputs.mkdir(parents=True)
        prediction_path = inputs / "predictions.csv"
        _write_csv(
            prediction_path,
            list(PREDICTION_COLUMNS),
            _prediction_rows(model_ids[variant_id], probabilities[variant_id])[4:],
        )
        config_path = inputs / "config.json"
        write_json(
            _experiment_config(
                variant_id=variant_id,
                model_id=model_ids[variant_id],
                experiment_id=experiment_id,
            ),
            config_path,
        )
        runner_module.run_experiment(
            predictions_path=prediction_path,
            prices_path=prices,
            config_path=config_path,
            runs_root=runs_root,
        )

    training_entries = _inventory(training, excluded={"run_manifest.json"})
    write_json(
        {
            "models": [
                {"variant_id": variant_id, "model_id": model_ids[variant_id]}
                for variant_id in comparison.BASELINE_VARIANTS
            ],
            "artifact_inventory": _inventory_payload(
                training_entries, excludes=["run_manifest.json"]
            ),
        },
        training / "run_manifest.json",
    )
    training_sha = sha256_file(training / "run_manifest.json")

    vertical_entries = _inventory(vertical, excluded={"manifest.json"})
    write_json(
        {
            "artifacts": [
                {"path": relative, "sha256": checksum}
                for relative, checksum in vertical_entries
            ]
        },
        vertical / "manifest.json",
    )
    vertical_sha = sha256_file(vertical / "manifest.json")

    lstm_entries = _inventory(
        lstm, excluded={"artifact_manifest.json", "run_manifest.json"}
    )
    write_json(
        _inventory_payload(
            lstm_entries, excludes=["artifact_manifest.json", "run_manifest.json"]
        ),
        lstm / "artifact_manifest.json",
    )
    lstm_root_entries = _inventory(lstm, excluded={"run_manifest.json"})
    write_json(
        {
            "model_ids": {
                variant_id: model_ids[variant_id]
                for variant_id in comparison.LSTM_VARIANTS
            },
            "artifact_inventory": _inventory_payload(
                lstm_root_entries, excludes=["run_manifest.json"]
            ),
        },
        lstm / "run_manifest.json",
    )
    lstm_sha = sha256_file(lstm / "run_manifest.json")

    training_evidence = TrainingRunEvidence(
        run_directory=training.resolve(),
        run_id=training.name,
        invocation_id="offline-fixture-invocation",
        run_manifest_sha256=training_sha,
        artifact_sha256=training_entries,
        code_version="offline-fixture",
        dirty_tree=False,
        source_fingerprint="f" * 64,
        model_ids=tuple(
            (variant_id, model_ids[variant_id])
            for variant_id in comparison.BASELINE_VARIANTS
        ),
        prediction_paths=tuple(
            (
                variant_id,
                f"models/{model_ids[variant_id]}/predictions_all_splits.csv",
            )
            for variant_id in comparison.BASELINE_VARIANTS
        ),
    )
    vertical_evidence = DirectoryInventoryEvidence(
        directory=vertical.resolve(),
        manifest_sha256=vertical_sha,
        files=vertical_entries,
    )
    lstm_evidence = comparison.LSTMRunEvidence(
        directory=lstm.resolve(),
        run_id=lstm.name,
        run_manifest_sha256=lstm_sha,
        files=lstm_root_entries,
        model_ids=tuple(
            (variant_id, model_ids[variant_id])
            for variant_id in comparison.LSTM_VARIANTS
        ),
        baseline_training_run_id=training.name,
        baseline_training_manifest_sha256=training_sha,
        baseline_vertical_slice_id=vertical.name,
        baseline_vertical_manifest_sha256=vertical_sha,
    )
    authority = {
        "baseline_training": {
            "run_id": training.name,
            "manifest_sha256": training_sha,
        },
        "baseline_vertical_slice": {
            "run_id": vertical.name,
            "manifest_sha256": vertical_sha,
        },
        "lstm": {"run_id": lstm.name, "manifest_sha256": lstm_sha},
    }
    return (
        comparison.SourceSnapshots(
            training=training_evidence,
            vertical=vertical_evidence,
            lstm=lstm_evidence,
        ),
        authority,
    )


@pytest.fixture(scope="module")
def portable_fixture(tmp_path_factory: pytest.TempPathFactory) -> PortableFixture:
    root = tmp_path_factory.mktemp("unified-source-fixture").resolve()
    snapshots, authority = _make_synthetic_sources(root / "sources")
    output_root = root / "comparisons"
    patch = pytest.MonkeyPatch()
    patch.setattr(comparison, "AUTHORITATIVE_SOURCE_ROOTS", authority)
    patch.setattr(
        comparison,
        "_capture_source_snapshots",
        lambda **_kwargs: snapshots,
    )
    patch.setattr(comparison, "_assert_source_snapshots_unchanged", lambda *_args: None)
    try:
        result = comparison.build_unified_price_model_comparison(
            comparison_id="CMP-offline-unified-fixture",
            output_dir=output_root,
            training_run=snapshots.training.run_directory,
            vertical_slice=snapshots.vertical.directory,
            lstm_run=snapshots.lstm.directory,
        )
    finally:
        patch.undo()
    return PortableFixture(
        output_root=output_root,
        pack=Path(result["artifact_directory"]),
        archive=Path(result["zip_path"]),
        checksum=Path(result["checksum_path"]),
        authority=authority,
        snapshots=snapshots,
    )


def _trust_fixture(
    monkeypatch: pytest.MonkeyPatch, fixture: PortableFixture
) -> None:
    monkeypatch.setattr(comparison, "AUTHORITATIVE_SOURCE_ROOTS", fixture.authority)


def _copy_pack(fixture: PortableFixture, destination: Path) -> Path:
    target = destination / fixture.pack.name
    shutil.copytree(fixture.pack, target)
    return target


def _reseal_pack(pack: Path) -> None:
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inventory"]["files"] = [
        {
            "path": path.relative_to(pack).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(pack.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file() and path != manifest_path
    ]
    write_json(manifest, manifest_path)


def _replace_inventory_checksum(
    payload: dict[str, Any], *, source_path: str, checksum: str
) -> None:
    entries = payload["artifact_inventory"]["files"]
    matching = [entry for entry in entries if entry["path"] == source_path]
    assert len(matching) == 1
    matching[0]["sha256"] = checksum


def _replace_source_evidence_checksum(
    payload: dict[str, Any], *, pack_path: str, checksum: str
) -> None:
    entries = payload["evidence_files"]
    matching = [entry for entry in entries if entry["pack_path"] == pack_path]
    assert len(matching) == 1
    matching[0]["sha256"] = checksum


def _reseal_copied_source_evidence(
    pack: Path,
    fixture: PortableFixture,
    *,
    role: str,
    evidence_pack_path: str,
) -> dict[str, dict[str, str]]:
    source_roots_path = pack / "source_roots.json"
    source_roots = json.loads(source_roots_path.read_text(encoding="utf-8"))
    evidence_records = [
        entry
        for entry in source_roots["evidence_files"]
        if entry["pack_path"] == evidence_pack_path
    ]
    assert len(evidence_records) == 1
    evidence_record = evidence_records[0]
    assert evidence_record["source_role"] == role
    evidence_checksum = sha256_file(pack / evidence_pack_path)
    evidence_record["sha256"] = evidence_checksum

    if role == "baseline_training":
        manifest_pack_path = (
            "evidence/source_manifests/baseline_training_run_manifest.json"
        )
        manifest_path = pack / manifest_pack_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _replace_inventory_checksum(
            manifest,
            source_path=evidence_record["source_path"],
            checksum=evidence_checksum,
        )
        write_json(manifest, manifest_path)
        root_checksum = sha256_file(manifest_path)
        _replace_source_evidence_checksum(
            source_roots,
            pack_path=manifest_pack_path,
            checksum=root_checksum,
        )
    elif role == "lstm":
        nested_pack_path = (
            "evidence/source_manifests/lstm_artifact_manifest.json"
        )
        nested_path = pack / nested_pack_path
        nested = json.loads(nested_path.read_text(encoding="utf-8"))
        nested_entries = nested["files"]
        matching = [
            entry
            for entry in nested_entries
            if entry["path"] == evidence_record["source_path"]
        ]
        assert len(matching) == 1
        matching[0]["sha256"] = evidence_checksum
        write_json(nested, nested_path)
        nested_checksum = sha256_file(nested_path)

        manifest_pack_path = "evidence/source_manifests/lstm_run_manifest.json"
        manifest_path = pack / manifest_pack_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _replace_inventory_checksum(
            manifest,
            source_path=evidence_record["source_path"],
            checksum=evidence_checksum,
        )
        _replace_inventory_checksum(
            manifest,
            source_path="artifact_manifest.json",
            checksum=nested_checksum,
        )
        write_json(manifest, manifest_path)
        root_checksum = sha256_file(manifest_path)
        _replace_source_evidence_checksum(
            source_roots,
            pack_path=nested_pack_path,
            checksum=nested_checksum,
        )
        _replace_source_evidence_checksum(
            source_roots,
            pack_path=manifest_pack_path,
            checksum=root_checksum,
        )
    else:
        raise AssertionError(f"Unsupported fixture role: {role}")

    matching_roots = [
        source for source in source_roots["sources"] if source["role"] == role
    ]
    assert len(matching_roots) == 1
    matching_roots[0]["integrity_root_sha256"] = root_checksum
    write_json(source_roots, source_roots_path)
    _reseal_pack(pack)
    authority = deepcopy(fixture.authority)
    authority[role]["manifest_sha256"] = root_checksum
    return authority


def _configure_fixture_builder(
    monkeypatch: pytest.MonkeyPatch, fixture: PortableFixture
) -> None:
    _trust_fixture(monkeypatch, fixture)
    monkeypatch.setattr(
        comparison,
        "_capture_source_snapshots",
        lambda **_kwargs: fixture.snapshots,
    )
    monkeypatch.setattr(comparison, "_assert_source_snapshots_unchanged", lambda *_: None)


def test_actual_authoritative_sources_build_and_verify_without_retraining(
    tmp_path: Path,
) -> None:
    if not all(path.is_dir() for path in (TRAINING_RUN, VERTICAL_SLICE, LSTM_RUN)):
        pytest.skip("optional local authoritative comparison sources are absent")
    output_root = tmp_path.resolve() / "authoritative-comparison"
    result = comparison.build_unified_price_model_comparison(
        comparison_id="CMP-pytest-authoritative",
        output_dir=output_root,
        training_run=TRAINING_RUN,
        vertical_slice=VERTICAL_SLICE,
        lstm_run=LSTM_RUN,
    )
    verified = comparison.verify_unified_comparison_pack(
        result["zip_path"],
        checksum=result["checksum_path"],
        training_run=TRAINING_RUN,
        vertical_slice=VERTICAL_SLICE,
        lstm_run=LSTM_RUN,
    )
    assert verified["status"] == "PASS"
    assert verified["checksum_verified"] is True
    assert verified["source_verification"] == (
        "portable_evidence_and_external_sources"
    )
    assert result["holdout_context"]["common_count"] == 184
    assert result["reported_outcomes"]["validation_selected_model"][
        "variant_id"
    ] == "M-001"


def test_directory_zip_and_checksum_are_independently_verifiable(
    portable_fixture: PortableFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trust_fixture(monkeypatch, portable_fixture)
    directory = comparison.verify_unified_comparison_pack(portable_fixture.pack)
    directory_with_checksum = comparison.verify_unified_comparison_pack(
        portable_fixture.pack, checksum=portable_fixture.checksum
    )
    archive = comparison.verify_unified_comparison_pack(portable_fixture.archive)

    assert directory["status"] == "PASS"
    assert directory["checksum_verified"] is False
    assert directory["source_verification"] == "portable_evidence"
    assert directory_with_checksum["checksum_verified"] is True
    assert archive["checksum_verified"] is True
    assert archive["zip_sha256"] == portable_fixture.checksum.read_text(
        encoding="ascii"
    ).split()[0]
    assert directory["comparison_id"] == archive["comparison_id"]


def test_builder_refuses_an_existing_directory_zip_or_checksum(
    portable_fixture: PortableFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fixture_builder(monkeypatch, portable_fixture)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        comparison.build_unified_price_model_comparison(
            comparison_id=portable_fixture.pack.name,
            output_dir=portable_fixture.output_root,
            training_run=portable_fixture.snapshots.training.run_directory,
            vertical_slice=portable_fixture.snapshots.vertical.directory,
            lstm_run=portable_fixture.snapshots.lstm.directory,
        )


@pytest.mark.parametrize("existing_suffix", [".zip", ".zip.sha256"])
def test_directory_only_build_reserves_the_complete_immutable_id_namespace(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_suffix: str,
) -> None:
    _configure_fixture_builder(monkeypatch, portable_fixture)
    comparison_id = "CMP-existing-portable-sibling"
    output_root = tmp_path.resolve() / existing_suffix.removeprefix(".")
    output_root.mkdir()
    (output_root / f"{comparison_id}{existing_suffix}").write_text(
        "occupied\n", encoding="ascii"
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        comparison.build_unified_price_model_comparison(
            comparison_id=comparison_id,
            output_dir=output_root,
            training_run=portable_fixture.snapshots.training.run_directory,
            vertical_slice=portable_fixture.snapshots.vertical.directory,
            lstm_run=portable_fixture.snapshots.lstm.directory,
            create_zip=False,
        )


def test_builder_never_loads_models_retrains_or_reruns_backtests(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fixture_builder(monkeypatch, portable_fixture)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("comparison generation must consume persisted evidence only")

    monkeypatch.setattr(artifact_module.joblib, "load", forbidden)
    monkeypatch.setattr(runner_module, "run_experiment", forbidden)
    monkeypatch.setattr(trainer_module, "run_training_stage", forbidden)
    result = comparison.build_unified_price_model_comparison(
        comparison_id="CMP-offline-no-model-execution",
        output_dir=tmp_path.resolve() / "no-execution-output",
        training_run=portable_fixture.snapshots.training.run_directory,
        vertical_slice=portable_fixture.snapshots.vertical.directory,
        lstm_run=portable_fixture.snapshots.lstm.directory,
        create_zip=False,
    )
    pack = Path(result["artifact_directory"])
    assert result["zip_path"] is None
    assert not any(
        path.suffix.lower() in {".joblib", ".keras", ".h5", ".pkl"}
        for path in pack.rglob("*")
        if path.is_file()
    )


def test_lstm_01_is_rejected_as_historical_evidence(tmp_path: Path) -> None:
    run = tmp_path / "LSTM-20261024-OGDC-price-only-01"
    run.mkdir()
    write_json(
        {
            "schema_version": "1.0",
            "run_id": run.name,
            "artifact_manifest": "artifact_manifest.json",
        },
        run / "run_manifest.json",
    )
    write_json({}, run / "artifact_manifest.json")

    with pytest.raises(ValueError, match="historical evidence only"):
        comparison.validate_lstm_run_artifacts(run)


def test_corrupted_evidence_is_rejected_before_semantic_use(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trust_fixture(monkeypatch, portable_fixture)
    pack = _copy_pack(portable_fixture, tmp_path)
    evidence = pack / "evidence/predictions/M-001/predictions_all_splits.csv"
    evidence.write_bytes(evidence.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        comparison.verify_unified_comparison_pack(pack)


def test_resealed_nested_lstm_inventory_is_rejected_when_not_bound_by_run_root(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trust_fixture(monkeypatch, portable_fixture)
    pack = _copy_pack(portable_fixture, tmp_path)
    nested_path = pack / "evidence/source_manifests/lstm_artifact_manifest.json"
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    nested["files"][0]["sha256"] = "0" * 64
    write_json(nested, nested_path)
    _reseal_pack(pack)

    with pytest.raises(ValueError, match="not bound by its run manifest"):
        comparison.verify_unified_comparison_pack(pack)


def test_incomplete_manifest_inventory_is_rejected(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trust_fixture(monkeypatch, portable_fixture)
    pack = _copy_pack(portable_fixture, tmp_path)
    manifest_path = pack / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["inventory"]["files"].pop()
    write_json(manifest, manifest_path)

    with pytest.raises(ValueError, match="inventory is not exact"):
        comparison.verify_unified_comparison_pack(pack)


@pytest.mark.parametrize(
    ("relative", "mutate", "message"),
    [
        (
            "metrics.json",
            lambda payload: payload["shared_project_holdout_prediction_metrics"][0].__setitem__(
                "accuracy", 0.0
            ),
            "Persisted metrics.json does not match replayed evidence",
        ),
        (
            "date_alignment.json",
            lambda payload: payload["shared_project_holdout"].__setitem__(
                "target_end", "2099-12-31"
            ),
            "Persisted date_alignment.json does not match replayed evidence",
        ),
        (
            "comparison_contract.json",
            lambda payload: payload["portfolio_assumptions"].__setitem__(
                "maximum_allocation", 0.75
            ),
            "Persisted comparison_contract.json does not match replayed evidence",
        ),
    ],
)
def test_resealed_semantic_json_tampering_is_rejected_by_replay(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    mutate: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    _trust_fixture(monkeypatch, portable_fixture)
    pack = _copy_pack(portable_fixture, tmp_path)
    path = pack / relative
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    write_json(payload, path)
    _reseal_pack(pack)

    with pytest.raises(ValueError, match=message):
        comparison.verify_unified_comparison_pack(pack)


def test_resealed_reference_label_tampering_is_rejected_by_replay(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _trust_fixture(monkeypatch, portable_fixture)
    pack = _copy_pack(portable_fixture, tmp_path)
    path = pack / "derived/reference_baseline_predictions.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "actual_class"] = 1 - int(frame.loc[0, "actual_class"])
    frame.to_csv(path, index=False, lineterminator="\n")
    _reseal_pack(pack)

    with pytest.raises(
        ValueError, match="Reference baseline predictions do not match replayed evidence"
    ):
        comparison.verify_unified_comparison_pack(pack)


@pytest.mark.parametrize(
    ("case", "role", "relative", "message"),
    [
        (
            "wrong-ticker",
            "baseline_training",
            "evidence/predictions/M-001/predictions_all_splits.csv",
            "Prediction ticker must be OGDC for M-001",
        ),
        (
            "wrong-split",
            "baseline_training",
            "evidence/predictions/M-002/predictions_all_splits.csv",
            "Baseline split membership mismatch for M-002",
        ),
        (
            "missing-scope-row",
            "lstm",
            "evidence/predictions/M-006/predictions_all_splits.csv",
            "exactly cover declared split dates for M-006",
        ),
    ],
)
def test_resealed_prediction_evidence_must_match_ticker_split_and_scope_contracts(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    role: str,
    relative: str,
    message: str,
) -> None:
    pack = _copy_pack(portable_fixture, tmp_path)
    path = pack / relative
    frame = pd.read_csv(path)
    if case == "wrong-ticker":
        frame.loc[0, "ticker"] = "HBL"
    elif case == "wrong-split":
        validation_index = frame.index[frame["split"] == "validation"][0]
        frame.loc[validation_index, "split"] = "test"
    elif case == "missing-scope-row":
        frame = frame.loc[frame["target_date"] != "2026-01-12"].copy()
    else:
        raise AssertionError(case)
    frame.to_csv(path, index=False, lineterminator="\n")
    authority = _reseal_copied_source_evidence(
        pack,
        portable_fixture,
        role=role,
        evidence_pack_path=relative,
    )
    monkeypatch.setattr(comparison, "AUTHORITATIVE_SOURCE_ROOTS", authority)

    with pytest.raises(ValueError, match=message):
        comparison.verify_unified_comparison_pack(pack)


def test_prediction_ranking_uses_registered_model_id_for_final_tie_break() -> None:
    rows = [
        {
            "competitor_id": "M-001",
            "registered_model_id": "z-registered-model",
            "balanced_accuracy": 0.5,
            "f1": 0.5,
            "accuracy": 0.5,
        },
        {
            "competitor_id": "M-006",
            "registered_model_id": "a-registered-model",
            "balanced_accuracy": 0.5,
            "f1": 0.5,
            "accuracy": 0.5,
        },
    ]

    ranking = comparison._rank_prediction_rows(
        rows, primary="balanced_accuracy"
    )

    assert [row["competitor_id"] for row in ranking] == ["M-006", "M-001"]


@pytest.mark.parametrize(
    ("primary", "expected_order", "expected_fields"),
    [
        (
            "total_return",
            ["M-002", "M-001"],
            {"total_return", "sharpe_ratio", "maximum_drawdown"},
        ),
        (
            "maximum_drawdown",
            ["M-002", "M-001"],
            {"maximum_drawdown", "total_return"},
        ),
        (
            "fees_paid",
            ["M-002", "M-001"],
            {"fees_paid", "number_of_trades"},
        ),
    ],
)
def test_portfolio_ranking_persists_declared_tie_break_values(
    primary: str,
    expected_order: list[str],
    expected_fields: set[str],
) -> None:
    rows = [
        {
            "competitor_id": "M-001",
            "registered_model_id": "model-001",
            "total_return": 0.1,
            "sharpe_ratio": 0.5,
            "maximum_drawdown": 0.2,
            "fees_paid": "10.00",
            "number_of_trades": 2,
        },
        {
            "competitor_id": "M-002",
            "registered_model_id": "model-002",
            "total_return": 0.1,
            "sharpe_ratio": 0.6,
            "maximum_drawdown": 0.1,
            "fees_paid": "10.00",
            "number_of_trades": 1,
        },
    ]

    ranking = comparison._rank_portfolios(rows, primary=primary)

    assert [row["competitor_id"] for row in ranking] == expected_order
    assert expected_fields.issubset(ranking[0])


def test_resealed_pack_rejects_an_unexpected_authored_file(
    portable_fixture: PortableFixture,
    tmp_path: Path,
) -> None:
    pack = _copy_pack(portable_fixture, tmp_path)
    (pack / "extra.txt").write_text("untrusted payload\n", encoding="utf-8")
    _reseal_pack(pack)

    with pytest.raises(ValueError, match="file set is not exact"):
        comparison.verify_unified_comparison_pack(pack)


def test_zip_without_a_sidecar_is_rejected_before_archive_extraction(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / portable_fixture.archive.name
    shutil.copyfile(portable_fixture.archive, archive)

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("ZIP extraction must not start before sidecar verification")

    monkeypatch.setattr(comparison, "_verify_zip_archive", forbidden)

    with pytest.raises(FileNotFoundError, match="requires its SHA-256 sidecar"):
        comparison.verify_unified_comparison_pack(archive)


def test_zip_rejects_windows_drive_member_before_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_id = "CMP-windows-drive-path"
    archive = tmp_path / f"{comparison_id}.zip"
    member = zipfile.ZipInfo(
        f"{comparison_id}/D:/outside.txt",
        date_time=(1980, 1, 1, 0, 0, 0),
    )
    member.compress_type = zipfile.ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = (stat.S_IFREG | 0o644) << 16
    with zipfile.ZipFile(
        archive,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as output:
        output.writestr(member, b"untrusted payload\n", compresslevel=9)
    checksum = tmp_path / f"{archive.name}.sha256"
    checksum.write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii"
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("Unsafe ZIP member must be rejected before extraction")

    monkeypatch.setattr(comparison.shutil, "copyfileobj", forbidden)

    with pytest.raises(ValueError, match="portable relative POSIX path"):
        comparison.verify_unified_comparison_pack(archive)


def test_corrupted_zip_with_a_matching_sidecar_is_rejected(
    portable_fixture: PortableFixture,
    tmp_path: Path,
) -> None:
    archive = tmp_path / portable_fixture.archive.name
    payload = portable_fixture.archive.read_bytes()
    archive.write_bytes(payload[:-32])
    checksum = tmp_path / portable_fixture.checksum.name
    checksum.write_text(
        f"{sha256_file(archive)}  {archive.name}\n", encoding="ascii"
    )

    with pytest.raises(ValueError, match="malformed or corrupt"):
        comparison.verify_unified_comparison_pack(archive)


def test_corrupted_checksum_sidecar_is_rejected(
    portable_fixture: PortableFixture,
    tmp_path: Path,
) -> None:
    archive = tmp_path / portable_fixture.archive.name
    checksum = tmp_path / portable_fixture.checksum.name
    shutil.copyfile(portable_fixture.archive, archive)
    checksum.write_text(f"{'0' * 64}  {archive.name}\n", encoding="ascii")

    with pytest.raises(ValueError, match="ZIP checksum does not match"):
        comparison.verify_unified_comparison_pack(archive, checksum=checksum)


@pytest.mark.parametrize("failure_call", [1, 2])
def test_publication_failure_never_leaves_a_zip_without_its_sidecar(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_call: int,
) -> None:
    _configure_fixture_builder(monkeypatch, portable_fixture)
    original_publish = comparison._publish_staged_file
    calls = 0

    def fail_during_portable_publication(staged: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise RuntimeError("injected portable publication failure")
        original_publish(staged, destination)

    monkeypatch.setattr(
        comparison, "_publish_staged_file", fail_during_portable_publication
    )
    output_root = tmp_path.resolve() / f"failure-{failure_call}"
    comparison_id = f"CMP-publication-failure-{failure_call}"

    with pytest.raises(RuntimeError, match="injected portable publication failure"):
        comparison.build_unified_price_model_comparison(
            comparison_id=comparison_id,
            output_dir=output_root,
            training_run=portable_fixture.snapshots.training.run_directory,
            vertical_slice=portable_fixture.snapshots.vertical.directory,
            lstm_run=portable_fixture.snapshots.lstm.directory,
        )

    archive = output_root / f"{comparison_id}.zip"
    checksum = output_root / f"{comparison_id}.zip.sha256"
    assert not archive.exists() or checksum.is_file()
    assert not (output_root / comparison_id).exists()
    if failure_call == 1:
        assert not archive.exists()
        assert not checksum.exists()
    else:
        assert not archive.exists()
        assert checksum.is_file()


def test_late_directory_publication_failure_leaves_complete_verifiable_pair(
    portable_fixture: PortableFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_fixture_builder(monkeypatch, portable_fixture)
    original_rename = comparison.atomic_rename_noreplace

    def fail_directory_publication(source: Path, destination: Path) -> None:
        if source.is_dir():
            raise RuntimeError("injected directory publication failure")
        original_rename(source, destination)

    monkeypatch.setattr(
        comparison, "atomic_rename_noreplace", fail_directory_publication
    )
    output_root = tmp_path.resolve() / "late-directory-failure"
    comparison_id = "CMP-late-directory-failure"

    with pytest.raises(RuntimeError, match="injected directory publication failure"):
        comparison.build_unified_price_model_comparison(
            comparison_id=comparison_id,
            output_dir=output_root,
            training_run=portable_fixture.snapshots.training.run_directory,
            vertical_slice=portable_fixture.snapshots.vertical.directory,
            lstm_run=portable_fixture.snapshots.lstm.directory,
        )

    archive = output_root / f"{comparison_id}.zip"
    checksum = output_root / f"{comparison_id}.zip.sha256"
    assert archive.is_file()
    assert checksum.is_file()
    assert not (output_root / comparison_id).exists()
    verified = comparison.verify_unified_comparison_pack(archive)
    assert verified["status"] == "PASS"
    assert verified["checksum_verified"] is True
