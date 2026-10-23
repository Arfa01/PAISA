from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import pytest

import paisa.experiments.lstm_runner as runner
from paisa.experiments.artifacts import (
    validate_training_run_artifacts as validate_sealed_training_run,
)
from paisa.experiments.evaluation import PREDICTION_COLUMNS, make_prediction_frame
from paisa.experiments.trainer import prepare_training_data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_frame(model_id: str, probability_up: list[float]) -> pd.DataFrame:
    probability = np.asarray(probability_up, dtype=float)
    return pd.DataFrame(
        {
            "schema_version": "1.0",
            "model_id": model_id,
            "ticker": "OGDC",
            "feature_date": ["2026-01-05", "2026-01-06", "2026-01-07"],
            "target_date": ["2026-01-06", "2026-01-07", "2026-01-08"],
            "predicted_class": (probability > 0.5).astype(int),
            "probability_up": probability,
            "probability_down": 1.0 - probability,
            "actual_class": [1, 0, 1],
        },
        columns=PREDICTION_COLUMNS,
    )


def test_lstm_variants_are_the_exact_controlled_experiments():
    assert [variant.variant_id for variant in runner.LSTM_VARIANTS] == [
        "M-005",
        "M-006",
    ]
    assert [variant.label for variant in runner.LSTM_VARIANTS] == [
        "LSTM | 20-session sequence | baseline recurrent configuration | frozen",
        "LSTM | 20-session sequence | regularized recurrent configuration | frozen",
    ]
    baseline, regularized = [
        variant.training_config for variant in runner.LSTM_VARIANTS
    ]
    for config in (baseline, regularized):
        assert config.sequence_length == 20
        assert config.units == 64
        assert config.second_layer_units is None
        assert config.learning_rate == 0.001
        assert config.class_weight is None
    assert (baseline.dropout, baseline.recurrent_dropout) == (0.20, 0.0)
    assert (baseline.kernel_l2, baseline.recurrent_l2) == (0.0, 0.0)
    assert (regularized.dropout, regularized.recurrent_dropout) == (0.40, 0.10)
    assert (regularized.kernel_l2, regularized.recurrent_l2) == (0.0001, 0.0001)


def test_canonical_1239_session_fixture_has_834_182_184_sequences():
    row_count = 1_239
    dates = pd.bdate_range("2021-01-01", periods=row_count)
    indexes = np.arange(row_count)
    close = 100.0 + 0.03 * indexes + np.where(indexes % 2 == 0, 2.0, -2.0)
    prices = pd.DataFrame(
        {
            "date": dates,
            "ticker": "OGDC",
            "open": close + np.where(indexes % 3 == 0, 0.2, -0.2),
            "close": close,
            "volume": 1_000_000 + (indexes % 7) * 1_000 + indexes,
            "source": "offline-canonical-shape-fixture",
        }
    )
    prepared = prepare_training_data(prices, symbol="OGDC")
    assert {name: len(frame) for name, frame in prepared.splits.as_dict().items()} == {
        "train": 853,
        "validation": 182,
        "test": 184,
    }
    preprocessor = runner.LSTMPreprocessor.fit(
        prepared.splits.train, prepared.feature_columns
    )
    counts = {}
    for split_name, rows in prepared.splits.as_dict().items():
        batch = runner.build_lstm_sequence_batch(
            prepared.model_frame,
            rows,
            preprocessor=preprocessor,
            feature_columns=prepared.feature_columns,
            sequence_length=20,
            require_every_target=split_name != "train",
        )
        counts[split_name] = len(batch.rows)
    assert counts == {"train": 834, "validation": 182, "test": 184}


def test_metal_device_policy_requires_darwin_and_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
):
    gpu_tf = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda: [
                SimpleNamespace(name="GPU:0", device_type="GPU")
            ]
        )
    )
    cpu_tf = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda: [
                SimpleNamespace(name="CPU:0", device_type="CPU")
            ]
        )
    )
    hidden_gpu_tf = SimpleNamespace(
        config=SimpleNamespace(
            list_physical_devices=lambda: [
                SimpleNamespace(name="GPU:0", device_type="GPU")
            ],
            get_visible_devices=lambda _device_type: [],
        )
    )

    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="only on macOS"):
        runner._configure_device(gpu_tf, "metal")

    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError, match="TensorFlow-visible GPU"):
        runner._configure_device(cpu_tf, "metal")
    with pytest.raises(RuntimeError, match="TensorFlow-visible GPU"):
        runner._configure_device(hidden_gpu_tf, "metal")
    with pytest.raises(RuntimeError, match="not visible"):
        runner._configure_device(hidden_gpu_tf, "gpu")
    runner._configure_device(gpu_tf, "metal")
    runner._configure_device(gpu_tf, "gpu")


def test_vertical_slice_inventory_is_exact_and_detects_mutation(tmp_path: Path):
    root = tmp_path / "vertical"
    root.mkdir()
    (root / "a.txt").write_text("alpha\n", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.json").write_text("{}\n", encoding="utf-8")
    entries = [
        {"path": "a.txt", "sha256": _sha256(root / "a.txt")},
        {"path": "nested/b.json", "sha256": _sha256(nested / "b.json")},
    ]
    (root / "manifest.json").write_text(
        json.dumps({"artifacts": entries}, indent=2) + "\n", encoding="utf-8"
    )

    evidence = runner.validate_vertical_slice_artifacts(root)
    assert dict(evidence.files) == {
        "a.txt": entries[0]["sha256"],
        "nested/b.json": entries[1]["sha256"],
    }

    (root / "a.txt").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        runner.assert_vertical_slice_artifacts_unchanged(evidence)


def test_price_freeze_rejects_bytes_changed_after_baseline_checksum(tmp_path: Path):
    source = tmp_path / "prices.csv"
    source.write_text("date,open,close,volume\n2026-01-01,1,1,1\n", encoding="utf-8")
    expected = _sha256(source)
    source.write_text("mutated before freeze\n", encoding="utf-8")
    stage = tmp_path / "stage"
    stage.mkdir()

    with pytest.raises(ValueError, match="checksum-bound baseline"):
        runner._freeze_verified_prices(
            source,
            stage,
            expected_sha256=expected,
        )


def test_baseline_data_provenance_is_checksum_bound_and_common(tmp_path: Path):
    root = tmp_path / "training"
    root.mkdir()
    data_manifest = {
        "provider": "psx-fixture",
        "requested_start": "2021-01-01",
    }
    (root / "data_manifest.json").write_text(
        json.dumps(data_manifest), encoding="utf-8"
    )
    model_ids = {
        variant_id: f"{variant_id}-registered"
        for variant_id in ("M-001", "M-002", "M-003", "M-004")
    }
    inventory = [("data_manifest.json", _sha256(root / "data_manifest.json"))]
    for variant_id, model_id in model_ids.items():
        model_dir = root / "models" / model_id
        model_dir.mkdir(parents=True)
        manifest_path = model_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "data_provenance": {
                        "provider": "psx-fixture",
                        "requested_start": "2021-01-01",
                        "requested_end": "2026-09-15",
                        "input_schema_version": "paisa-price-v0",
                    }
                }
            ),
            encoding="utf-8",
        )
        relative = manifest_path.relative_to(root).as_posix()
        inventory.append((relative, _sha256(manifest_path)))
    evidence = SimpleNamespace(
        run_directory=root,
        run_id="TRN-checksum-bound",
        run_manifest_sha256="a" * 64,
        artifact_sha256=tuple(sorted(inventory)),
        model_ids=tuple(sorted(model_ids.items())),
    )

    provenance = runner._load_baseline_data_provenance(evidence)

    assert provenance.provider == "psx-fixture"
    assert provenance.requested_start == "2021-01-01"
    assert provenance.requested_end == "2026-09-15"
    assert provenance.input_schema_version == "paisa-price-v0"
    assert dict(provenance.source_files)["data_manifest.json"] == inventory[0][1]

    first_manifest = root / "models" / model_ids["M-001"] / "manifest.json"
    first_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum changed"):
        runner._load_baseline_data_provenance(evidence)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"provider": "  "}, "provider must be non-empty"),
        ({"input_schema_version": ""}, "input_schema_version must be non-empty"),
        ({"requested_start": "01/01/2024"}, "exact YYYY-MM-DD"),
        (
            {"requested_start": "2026-01-02", "requested_end": "2026-01-01"},
            "must not be later",
        ),
        (
            {"data_provenance_extra_fields": {"provider": "override"}},
            "cannot override required fields",
        ),
    ],
)
def test_lstm_registration_rejects_invalid_or_overridden_provenance(
    tmp_path: Path, overrides: dict[str, object], message: str
):
    prepared = _prepared_sequence_fixture()
    prepared.prices = pd.DataFrame(
        {"date": pd.bdate_range("2026-01-01", periods=55)}
    )
    arguments = {
        "prepared": prepared,
        "model": _FakeModel(),
        "config": runner.LSTM_VARIANTS[0].training_config,
        "models_root": tmp_path / "models",
        "repo_root": Path(__file__).resolve().parents[1],
        "model_id": "M-005-provenance-validation",
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        runner.register_fitted_lstm(**arguments)
    assert not (tmp_path / "models").exists()


def test_runner_refuses_collision_before_training_and_preserves_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prices = tmp_path / "prices.csv"
    prices.write_text("date,open,close,volume\n", encoding="utf-8")
    training_root = tmp_path / "training"
    vertical_root = tmp_path / "vertical"
    training_root.mkdir()
    vertical_root.mkdir()
    training_marker = training_root / "marker.txt"
    vertical_marker = vertical_root / "marker.txt"
    training_marker.write_text("training\n", encoding="utf-8")
    vertical_marker.write_text("vertical\n", encoding="utf-8")
    training_evidence = SimpleNamespace(
        run_directory=training_root.resolve(),
        run_id="TRN-test",
        invocation_id="training-invocation",
        run_manifest_sha256="a" * 64,
        artifact_sha256=(("frozen_input/OGDC_prices.csv", _sha256(prices)),),
    )
    vertical_evidence = runner.DirectoryInventoryEvidence(
        directory=vertical_root.resolve(),
        manifest_sha256="b" * 64,
        files=(),
    )
    monkeypatch.setattr(
        runner, "validate_training_run_artifacts", lambda *_args, **_kwargs: training_evidence
    )
    monkeypatch.setattr(
        runner, "validate_vertical_slice_artifacts", lambda *_args, **_kwargs: vertical_evidence
    )
    monkeypatch.setattr(runner, "_validate_baseline_relationship", lambda *_args: None)

    output = tmp_path / "output"
    destination = output / "LSTM-existing"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_lstm_price_only_experiments(
            prices_path=prices,
            output_dir=output,
            run_id="LSTM-existing",
            baseline_training_run=training_root,
            baseline_vertical_slice=vertical_root,
            create_zip=False,
            tensorflow_module=object(),
        )

    assert training_marker.read_text(encoding="utf-8") == "training\n"
    assert vertical_marker.read_text(encoding="utf-8") == "vertical\n"


def test_runner_cleans_staging_after_late_failure_and_preserves_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dates = pd.bdate_range("2024-01-02", periods=120)
    indexes = np.arange(len(dates))
    close = 100.0 + np.where(indexes % 2 == 0, 2.0, -2.0) + indexes * 0.02
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": dates,
            "ticker": "OGDC",
            "open": close + 0.1,
            "close": close,
            "volume": 1_000_000 + indexes,
            "source": "offline-fixture",
        }
    ).to_csv(prices_path, index=False)
    training_root = tmp_path / "training"
    vertical_root = tmp_path / "vertical"
    training_root.mkdir()
    vertical_root.mkdir()
    training_marker = training_root / "marker.txt"
    vertical_marker = vertical_root / "marker.txt"
    training_marker.write_text("training baseline\n", encoding="utf-8")
    vertical_marker.write_text("vertical baseline\n", encoding="utf-8")
    training_evidence = SimpleNamespace(
        run_directory=training_root.resolve(),
        run_id="TRN-late-failure",
        invocation_id="baseline-invocation",
        run_manifest_sha256="a" * 64,
        artifact_sha256=(
            ("frozen_input/OGDC_prices.csv", _sha256(prices_path)),
        ),
    )
    vertical_evidence = runner.DirectoryInventoryEvidence(
        directory=vertical_root.resolve(),
        manifest_sha256="b" * 64,
        files=(),
    )
    provenance = runner.BaselineDataProvenance(
        provider="fixture",
        requested_start="2024-01-01",
        requested_end="2024-12-31",
        input_schema_version="fixture-v1",
        training_run_id="TRN-late-failure",
        training_run_manifest_sha256="a" * 64,
        source_files=(("data_manifest.json", "c" * 64),),
    )
    monkeypatch.setattr(
        runner, "validate_training_run_artifacts", lambda *_args, **_kwargs: training_evidence
    )
    monkeypatch.setattr(
        runner, "validate_vertical_slice_artifacts", lambda *_args, **_kwargs: vertical_evidence
    )
    monkeypatch.setattr(runner, "_validate_baseline_relationship", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "_validate_prepared_against_baseline",
        lambda *_args: {"method": "fixture", "boundaries": {}},
    )
    monkeypatch.setattr(
        runner, "_load_baseline_data_provenance", lambda *_args: provenance
    )
    monkeypatch.setattr(runner, "_load_baseline_predictions", lambda *_args: {})
    monkeypatch.setattr(
        runner,
        "assert_training_run_artifacts_unchanged",
        lambda *_args: assert_marker(training_marker, "training baseline\n"),
    )
    monkeypatch.setattr(
        runner,
        "assert_vertical_slice_artifacts_unchanged",
        lambda *_args: assert_marker(vertical_marker, "vertical baseline\n"),
    )

    def fail_after_staging(*, stage_dir, **_kwargs):
        assert stage_dir.is_dir()
        (stage_dir / "partial.txt").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected late failure")

    monkeypatch.setattr(runner, "_fit_until_validation_freeze", fail_after_staging)
    output_root = tmp_path / "runs"
    with pytest.raises(RuntimeError, match="injected late failure"):
        runner.run_lstm_price_only_experiments(
            prices_path=prices_path,
            output_dir=output_root,
            run_id="LSTM-late-failure",
            baseline_training_run=training_root,
            baseline_vertical_slice=vertical_root,
            create_zip=False,
            device="cpu",
            tensorflow_module=_fake_tensorflow(),
        )

    assert output_root.is_dir()
    assert list(output_root.iterdir()) == []
    assert training_marker.read_text(encoding="utf-8") == "training baseline\n"
    assert vertical_marker.read_text(encoding="utf-8") == "vertical baseline\n"


def test_zip_publish_failure_rolls_back_just_published_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage_parent = tmp_path / ".stage"
    stage_dir = stage_parent / "LSTM-transaction-test"
    stage_dir.mkdir(parents=True)
    (stage_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    staged_zip = stage_parent / "LSTM-transaction-test.zip"
    staged_zip.write_bytes(b"complete staged zip")
    final_run = tmp_path / "LSTM-transaction-test"
    final_zip = tmp_path / "LSTM-transaction-test.zip"

    def fail_final_zip_rename(_staged: Path, _final: Path) -> None:
        raise OSError("injected final ZIP rename failure")

    monkeypatch.setattr(runner, "_publish_staged_zip", fail_final_zip_rename)
    with pytest.raises(OSError, match="injected final ZIP rename failure"):
        runner._publish_staged_artifacts(
            stage_dir=stage_dir,
            staged_zip=staged_zip,
            final_run_dir=final_run,
            final_zip=final_zip,
        )

    assert not final_run.exists()
    assert not final_zip.exists()
    assert staged_zip.read_bytes() == b"complete staged zip"


class _FakeModel:
    def __init__(self) -> None:
        self.predict_lengths: list[int] = []
        self.loaded_checkpoint: Path | None = None
        self.fit_validation_values: np.ndarray | None = None
        self.fit_validation_targets: np.ndarray | None = None
        self.optimizer = SimpleNamespace(get_config=lambda: {"learning_rate": 0.001})
        self.loss = "binary_crossentropy"

    def fit(self, values, targets, **kwargs):
        assert kwargs["shuffle"] is False
        assert kwargs["class_weight"] is None
        assert len(kwargs["callbacks"]) == 2
        assert len(values) == len(targets)
        validation_values, validation_targets = kwargs["validation_data"]
        self.fit_validation_values = np.asarray(validation_values).copy()
        self.fit_validation_targets = np.asarray(validation_targets).copy()
        checkpoint = kwargs["callbacks"][1]
        Path(checkpoint.filepath).write_bytes(b"fake-best-validation-weights")
        return SimpleNamespace(
            history={
                "loss": [0.8, 0.7],
                "accuracy": [0.5, 0.6],
                "val_loss": [0.75, 0.65],
                "val_accuracy": [0.5, 0.6],
            }
        )

    def predict(self, values, verbose=0):
        assert verbose == 0
        self.predict_lengths.append(len(values))
        return np.asarray(
            [0.8 if index % 2 else 0.2 for index in range(len(values))],
            dtype=float,
        ).reshape(-1, 1)

    def count_params(self):
        return 1234

    def load_weights(self, path):
        self.loaded_checkpoint = Path(path)
        assert self.loaded_checkpoint.read_bytes() == b"fake-best-validation-weights"

    def save(self, path):
        Path(path).write_bytes(b"fake-keras-v3-model")

    def get_config(self):
        return {
            "name": "fake_lstm",
            "layers": [{"class_name": "LSTM"}, {"class_name": "Dense"}],
        }


class _FakeEarlyStopping:
    def __init__(self, **kwargs) -> None:
        assert kwargs == {
            "monitor": "val_loss",
            "mode": "min",
            "patience": 6,
            "restore_best_weights": True,
        }
        self.stopped_epoch = 0


class _FakeModelCheckpoint:
    def __init__(self, **kwargs) -> None:
        assert kwargs["monitor"] == "val_loss"
        assert kwargs["mode"] == "min"
        assert kwargs["save_best_only"] is True
        assert kwargs["save_weights_only"] is True
        self.filepath = kwargs["filepath"]


def _fake_tensorflow():
    return SimpleNamespace(
        __version__="fake-tensorflow",
        keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: None),
            utils=SimpleNamespace(set_random_seed=lambda _seed: None),
            callbacks=SimpleNamespace(
                EarlyStopping=_FakeEarlyStopping,
                ModelCheckpoint=_FakeModelCheckpoint,
            ),
            models=SimpleNamespace(
                load_model=lambda _path, compile=False: _FakeModel()
            ),
        ),
        config=SimpleNamespace(
            experimental=SimpleNamespace(enable_op_determinism=lambda: None),
            list_physical_devices=lambda: [
                SimpleNamespace(name="CPU:0", device_type="CPU")
            ],
        ),
    )


def _prepared_sequence_fixture():
    row_count = 55
    feature_dates = pd.bdate_range("2026-01-01", periods=row_count)
    target_dates = pd.bdate_range("2026-01-02", periods=row_count)
    frame = pd.DataFrame(
        {
            "feature_date": feature_dates,
            "target_date": target_dates,
            "open": np.linspace(100.0, 120.0, row_count),
            "close": np.linspace(101.0, 121.0, row_count),
            "target_next_day_up": np.arange(row_count) % 2,
        }
    )
    splits = SimpleNamespace(
        train=frame.iloc[:35].reset_index(drop=True),
        validation=frame.iloc[35:43].reset_index(drop=True),
        test=frame.iloc[43:].reset_index(drop=True),
    )
    return SimpleNamespace(
        model_frame=frame,
        splits=splits,
        feature_columns=("open", "close"),
        dataset_sha256="c" * 64,
    )


def test_validation_freeze_precedes_one_test_prediction_per_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    models: list[_FakeModel] = []

    def build_model(*_args, **_kwargs):
        model = _FakeModel()
        models.append(model)
        return model

    monkeypatch.setattr(runner, "build_lstm_model", build_model)
    events: list[dict[str, object]] = []
    states, decision = runner._fit_until_validation_freeze(
        prepared=_prepared_sequence_fixture(),
        stage_dir=tmp_path,
        run_id="LSTM-test",
        invocation_id="invocation-test",
        tf=_fake_tensorflow(),
        requested_device="cpu",
        events=events,
    )

    assert decision["test_metrics_consulted"] is False
    assert (tmp_path / "validation_leaderboard.csv").is_file()
    assert (tmp_path / "validation_freeze_decision.json").is_file()
    assert [model.predict_lengths for model in models] == [[16, 8], [16, 8]]
    assert all(model.loaded_checkpoint is not None for model in models)
    for model, state in zip(models, states, strict=True):
        np.testing.assert_array_equal(
            model.fit_validation_values, state.validation_batch.values
        )
        np.testing.assert_array_equal(
            model.fit_validation_targets, state.validation_batch.targets
        )
        assert len(state.validation_batch.values) == 8
        checkpoint = state.training_metadata["model_checkpoint"]
        assert checkpoint["loaded_after_fit"] is True
        assert checkpoint["path"] == f"checkpoints/{state.variant.variant_id}.weights.h5"
        assert len(checkpoint["sha256"]) == 64
    assert not any(event["event"].startswith("test_") for event in events)

    runner._evaluate_test_once(
        states=states,
        prepared=_prepared_sequence_fixture(),
        events=events,
        stage_dir=tmp_path,
    )

    assert [model.predict_lengths for model in models] == [
        [16, 8, 12],
        [16, 8, 12],
    ]
    for model, state in zip(models, states, strict=True):
        assert state.test_batch is not None
        assert len(state.test_batch.values) == 12
        assert model.fit_validation_values is not None
        assert model.fit_validation_values.shape != state.test_batch.values.shape
    names = [event["event"] for event in events]
    freeze_index = names.index("validation_selection_frozen")
    assert all(
        index > freeze_index
        for index, name in enumerate(names)
        if name.startswith("test_")
    )
    saved_events = json.loads(
        (tmp_path / "selection_events.json").read_text(encoding="utf-8")
    )["events"]
    assert [event["sequence"] for event in saved_events] == list(
        range(1, len(saved_events) + 1)
    )


def test_all_six_fair_predictions_use_identical_rules_and_unique_portfolios(
    tmp_path: Path,
):
    probabilities = {
        "M-001": [0.7, 0.3, 0.8],
        "M-002": [0.6, 0.2, 0.9],
        "M-003": [0.8, 0.4, 0.7],
        "M-004": [0.65, 0.35, 0.75],
        "M-005": [0.72, 0.28, 0.82],
        "M-006": [0.68, 0.32, 0.78],
    }
    predictions = {
        variant_id: _prediction_frame(
            f"{variant_id}-registered", variant_probabilities
        )
        for variant_id, variant_probabilities in probabilities.items()
    }
    prices = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
            "ticker": ["OGDC"] * 4,
            "open": [100.0, 102.0, 99.0, 97.0],
            "close": [101.0, 103.0, 98.0, 100.0],
        }
    ).to_csv(prices, index=False)
    stage = tmp_path / "stage"
    stage.mkdir()

    metrics, portfolios, backtests, evidence = runner._run_fair_comparison(
        predictions_by_variant=predictions,
        stage_dir=stage,
        run_id="LSTM-comparison-test",
        prices_path=prices,
    )

    assert set(metrics["variant_id"]) == set(probabilities)
    assert set(portfolios["variant_id"]) == set(probabilities)
    assert set(backtests) == set(probabilities)
    assert evidence["common_count"] == 3
    configs = [
        json.loads(
            (stage / "experiment_configs" / f"{variant_id}.json").read_text(
                encoding="utf-8"
            )
        )
        for variant_id in sorted(probabilities)
    ]
    for config in configs:
        assert config["initial_cash"] == "100000.00"
        assert config["thresholds"] == {
            "buy_probability_up_gte": 0.6,
            "sell_probability_up_lte": 0.4,
        }
        assert config["maximum_allocation"] == 0.5
        assert config["transaction_cost"]["value"] == 10.0
        assert config["execution_policy"]["slippage_bps"] == 0.0
        assert config["execution_policy"]["execution_session"] == "next_trading_session"
        assert config["execution_policy"]["reference_price_field"] == "open"
        assert config["execution_policy"]["valuation_price_field"] == "close"
    portfolio_ids = {
        portfolio_id
        for config in configs
        for portfolio_id in (
            config["portfolio_id"],
            config["benchmark_portfolio_ids"]["cash_only"],
            config["benchmark_portfolio_ids"]["buy_and_hold"],
        )
    }
    assert len(portfolio_ids) == 18
    assert (stage / "comparison_plots" / "prediction_metric_comparison.png").is_file()
    assert (stage / "comparison_plots" / "portfolio_equity_comparison.png").is_file()


def test_fake_end_to_end_run_has_required_structure_and_valid_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dates = pd.bdate_range("2024-01-02", periods=120)
    close = 100.0 + np.sin(np.arange(len(dates)) / 2.0) * 5.0 + np.arange(
        len(dates)
    ) * 0.02
    prices_frame = pd.DataFrame(
        {
            "date": dates,
            "ticker": "OGDC",
            "open": close - np.where(np.arange(len(dates)) % 2, 0.5, -0.5),
            "close": close,
            "volume": 1_000_000 + np.arange(len(dates)) * 100,
            "source": "offline-fixture",
        }
    )
    prices_path = tmp_path / "prices.csv"
    prices_frame.to_csv(prices_path, index=False)
    prepared = prepare_training_data(prices_frame, symbol="OGDC")
    test_probabilities = np.where(
        np.arange(len(prepared.splits.test)) % 2, 0.7, 0.3
    )
    baseline_predictions = {
        variant_id: make_prediction_frame(
            prepared.splits.test,
            ticker="OGDC",
            model_id=f"{variant_id}-fixture-registered",
            probability_up=test_probabilities,
            target_column="target_next_day_up",
        )
        for variant_id in ("M-001", "M-002", "M-003", "M-004")
    }

    training_root = tmp_path / "baseline-training"
    vertical_root = tmp_path / "baseline-vertical"
    training_root.mkdir()
    vertical_root.mkdir()
    training_marker = training_root / "marker.txt"
    vertical_marker = vertical_root / "marker.txt"
    training_marker.write_text("sealed training baseline\n", encoding="utf-8")
    vertical_marker.write_text("sealed vertical baseline\n", encoding="utf-8")
    training_evidence = SimpleNamespace(
        run_directory=training_root.resolve(),
        run_id="TRN-fixture",
        invocation_id="baseline-invocation",
        run_manifest_sha256="a" * 64,
        artifact_sha256=(
            ("frozen_input/OGDC_prices.csv", _sha256(prices_path)),
        ),
        model_ids=tuple(
            (variant_id, f"{variant_id}-fixture-registered")
            for variant_id in baseline_predictions
        ),
        prediction_paths=(),
    )
    vertical_evidence = runner.DirectoryInventoryEvidence(
        directory=vertical_root.resolve(),
        manifest_sha256="b" * 64,
        files=(),
    )
    baseline_provenance = runner.BaselineDataProvenance(
        provider="checksum-bound-fixture-provider",
        requested_start="2023-12-01",
        requested_end="2024-07-01",
        input_schema_version="fixture-price-v9",
        training_run_id="TRN-fixture",
        training_run_manifest_sha256="a" * 64,
        source_files=(
            ("data_manifest.json", "c" * 64),
            ("models/M-001-fixture-registered/manifest.json", "d" * 64),
        ),
    )

    def validate_training(path, **kwargs):
        if Path(path).resolve() == training_root.resolve():
            return training_evidence
        return validate_sealed_training_run(path, **kwargs)

    monkeypatch.setattr(runner, "validate_training_run_artifacts", validate_training)
    monkeypatch.setattr(
        runner, "validate_vertical_slice_artifacts", lambda *_args: vertical_evidence
    )
    monkeypatch.setattr(runner, "_validate_baseline_relationship", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "_validate_prepared_against_baseline",
        lambda *_args: {"method": "fixture", "boundaries": {}},
    )
    monkeypatch.setattr(
        runner,
        "_load_baseline_data_provenance",
        lambda *_args: baseline_provenance,
    )
    monkeypatch.setattr(
        runner, "_load_baseline_predictions", lambda *_args: baseline_predictions
    )
    monkeypatch.setattr(
        runner,
        "assert_training_run_artifacts_unchanged",
        lambda *_args: assert_marker(training_marker, "sealed training baseline\n"),
    )
    monkeypatch.setattr(
        runner,
        "assert_vertical_slice_artifacts_unchanged",
        lambda *_args: assert_marker(vertical_marker, "sealed vertical baseline\n"),
    )
    monkeypatch.setattr(runner, "build_lstm_model", lambda *_args, **_kwargs: _FakeModel())
    original_price_bytes = prices_path.read_bytes()
    original_read_csv = runner.pd.read_csv
    source_mutated = False

    def read_csv_and_mutate_source(path, *args, **kwargs):
        nonlocal source_mutated
        frame = original_read_csv(path, *args, **kwargs)
        candidate = Path(path)
        if candidate.parent.name == "frozen_input" and not source_mutated:
            prices_path.write_text("mutated after frozen read\n", encoding="utf-8")
            source_mutated = True
        return frame

    monkeypatch.setattr(runner.pd, "read_csv", read_csv_and_mutate_source)
    original_run_experiment = runner.run_experiment
    backtest_price_paths: list[Path] = []

    def run_experiment_with_frozen_prices(**kwargs):
        candidate = Path(kwargs["prices_path"])
        assert candidate.parent.name == "frozen_input"
        assert candidate.name == "OGDC_prices.csv"
        backtest_price_paths.append(candidate)
        return original_run_experiment(**kwargs)

    monkeypatch.setattr(runner, "run_experiment", run_experiment_with_frozen_prices)
    original_publish = runner._publish_staged_artifacts
    zip_was_complete_before_publication = False

    def publish_only_after_zip_is_complete(**kwargs):
        nonlocal zip_was_complete_before_publication
        assert kwargs["stage_dir"].is_dir()
        assert not kwargs["final_run_dir"].exists()
        staged_zip = kwargs["staged_zip"]
        assert staged_zip is not None and staged_zip.is_file()
        with zipfile.ZipFile(staged_zip) as archive:
            staged_names = archive.namelist()
        assert staged_names
        assert all(name.startswith("LSTM-fixture-run/") for name in staged_names)
        zip_was_complete_before_publication = True
        return original_publish(**kwargs)

    monkeypatch.setattr(
        runner, "_publish_staged_artifacts", publish_only_after_zip_is_complete
    )

    summary = runner.run_lstm_price_only_experiments(
        prices_path=prices_path,
        output_dir=tmp_path / "runs",
        run_id="LSTM-fixture-run",
        baseline_training_run=training_root,
        baseline_vertical_slice=vertical_root,
        create_zip=True,
        device="cpu",
        tensorflow_module=_fake_tensorflow(),
    )

    run_dir = Path(summary["artifact_directory"])
    zip_path = Path(summary["zip_path"])
    required = {
        "run_manifest.json",
        "artifact_manifest.json",
        "data_manifest.json",
        "split_manifest.json",
        "split_reference.json",
        "validation_leaderboard.csv",
        "test_leaderboard.csv",
        "training_history.csv",
        "fair_common_dates.json",
        "validation_reproduction_checks.json",
        "comparison_metrics.csv",
        "portfolio_metrics.csv",
        "final_summary.json",
    }
    assert required.issubset({path.name for path in run_dir.iterdir()})
    assert zip_was_complete_before_publication is True
    assert source_mutated is True
    assert prices_path.read_text(encoding="utf-8") == "mutated after frozen read\n"
    assert len(backtest_price_paths) == 6
    assert len({str(path) for path in backtest_price_paths}) == 1
    frozen_prices = run_dir / "frozen_input" / "OGDC_prices.csv"
    assert frozen_prices.read_bytes() == original_price_bytes
    data_manifest = json.loads(
        (run_dir / "data_manifest.json").read_text(encoding="utf-8")
    )
    assert data_manifest["prices_path"] == "frozen_input/OGDC_prices.csv"
    assert run_dir / data_manifest["prices_path"] == frozen_prices
    assert data_manifest["prices_sha256"] == _sha256(frozen_prices)
    assert zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
    assert names
    assert all(name.startswith("LSTM-fixture-run/") for name in names)
    assert "LSTM-fixture-run/frozen_input/OGDC_prices.csv" in names
    assert {path.name for path in (tmp_path / "runs").iterdir()} == {
        "LSTM-fixture-run",
        "LSTM-fixture-run.zip",
    }
    evidence = validate_sealed_training_run(run_dir)
    assert dict(evidence.model_ids).keys() == {"M-005", "M-006"}
    assert all((run_dir / "models" / model_id).is_dir() for _, model_id in evidence.model_ids)
    artifact_manifest = json.loads(
        (run_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert artifact_manifest["excludes"] == [
        "artifact_manifest.json",
        "run_manifest.json",
    ]
    checkpoint_metadata = json.loads(
        next((run_dir / "models").glob("*/training_metadata.json")).read_text(
            encoding="utf-8"
        )
    )["model_checkpoint"]
    checkpoint_path = run_dir / checkpoint_metadata["path"]
    assert _sha256(checkpoint_path) == checkpoint_metadata["sha256"]
    run_manifest = json.loads(
        (run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert run_manifest["source_fingerprint_inputs"] == list(
        runner._SOURCE_FINGERPRINT_INPUTS
    )
    assert run_manifest["baseline_training_run"]["data_provenance"][
        "provider"
    ] == "checksum-bound-fixture-provider"
    for model_manifest_path in (run_dir / "models").glob("*/manifest.json"):
        model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
        provenance = model_manifest["data_provenance"]
        assert provenance["provider"] == "checksum-bound-fixture-provider"
        assert provenance["requested_start"] == "2023-12-01"
        assert provenance["requested_end"] == "2024-07-01"
        assert provenance["input_schema_version"] == "fixture-price-v9"
        assert provenance["upstream_baseline"]["training_run_manifest_sha256"] == (
            "a" * 64
        )
        reproduction = json.loads(
            (model_manifest_path.parent / "validation_reproduction_check.json").read_text(
                encoding="utf-8"
            )
        )
        assert reproduction["status"] == "PASS"
        assert reproduction["split"] == "validation"
        assert reproduction["test_split_inference_performed"] is False
        assert reproduction["tolerance_match"] is True
    final_summary = json.loads(
        (run_dir / "final_summary.json").read_text(encoding="utf-8")
    )
    assert set(final_summary["lstm_metrics"]) == {
        "validation",
        "untouched_test",
    }
    for split_metrics in final_summary["lstm_metrics"].values():
        assert set(split_metrics) == {"M-005", "M-006"}
        for metrics in split_metrics.values():
            assert {
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "matthews_corrcoef",
                "roc_auc",
                "brier_score",
                "confusion_matrix",
                "calibration",
            }.issubset(metrics)
    assert set(final_summary["lstm_portfolio_metrics"]) == {"M-005", "M-006"}
    for metrics in final_summary["lstm_portfolio_metrics"].values():
        assert {
            "ending_portfolio_value",
            "total_return",
            "maximum_drawdown",
            "volatility",
            "sharpe_ratio",
            "number_of_trades",
            "fees_paid",
            "result_versus_buy_and_hold",
        }.issubset(metrics)
    assert training_marker.read_text(encoding="utf-8") == "sealed training baseline\n"
    assert vertical_marker.read_text(encoding="utf-8") == "sealed vertical baseline\n"


def assert_marker(path: Path, expected: str) -> None:
    assert path.read_text(encoding="utf-8") == expected
