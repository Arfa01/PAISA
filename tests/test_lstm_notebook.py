from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

import nbformat
import pytest

from paisa.experiments.lstm_runner import run_lstm_price_only_experiments

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPOSITORY_ROOT / "notebooks" / "PAISA_OGDC_LSTM_Experiments.ipynb"
CLI_PATH = REPOSITORY_ROOT / "scripts" / "run_lstm_price_only_experiments.py"


def _notebook() -> dict[str, object]:
    return nbformat.read(NOTEBOOK_PATH, as_version=4)


def _cli_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("paisa_lstm_cli_contract", CLI_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _joined_source(notebook: dict[str, object], *, code_only: bool = False) -> str:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    selected = [
        cell
        for cell in cells
        if isinstance(cell, dict)
        and (not code_only or cell.get("cell_type") == "code")
    ]
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in selected
    )


def test_lstm_experiment_notebook_is_valid_unexecuted_nbformat_json() -> None:
    notebook = _notebook()

    nbformat.validate(notebook)
    assert notebook["nbformat"] == 4
    assert int(notebook["nbformat_minor"]) >= 5
    assert isinstance(notebook["metadata"], dict)
    cells = notebook["cells"]
    assert isinstance(cells, list) and cells
    assert {cell["cell_type"] for cell in cells} <= {"markdown", "code"}
    assert all(isinstance(cell.get("source"), str) for cell in cells)
    code_cells = [cell for cell in cells if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell.get("execution_count") is None for cell in code_cells)
    assert all(cell.get("outputs") == [] for cell in code_cells)
    for index, cell in enumerate(code_cells):
        compile(
            "".join(cell.get("source", [])),
            f"{NOTEBOOK_PATH}:code-cell-{index}",
            "exec",
        )


def test_lstm_experiment_notebook_calls_reusable_runner_and_names_cli() -> None:
    source = _joined_source(_notebook())

    assert (
        "from paisa.experiments.lstm_runner import "
        "run_lstm_price_only_experiments"
    ) in source
    assert "summary = run_lstm_price_only_experiments(" in source
    assert "scripts/run_lstm_price_only_experiments.py" in source
    assert "requirements-lstm.txt" in source
    for keyword in (
        "prices_path=PRICES_PATH",
        "output_dir=OUTPUT_DIR",
        "run_id=RUN_ID",
        "baseline_training_run=BASELINE_TRAINING_RUN",
        "baseline_vertical_slice=BASELINE_VERTICAL_SLICE",
        "create_zip=True",
    ):
        assert keyword in source
    for artifact in (
        "validation_leaderboard.csv",
        "test_leaderboard.csv",
        "final_summary.json",
    ):
        assert artifact in source
    assert "not financial advice" in source.lower()


def test_lstm_experiment_notebook_discloses_shared_test_holdout_reuse() -> None:
    source = _joined_source(_notebook())
    normalized = source.lower()

    assert "post-freeze lstm test" in normalized
    assert "shared project holdout" in normalized
    assert "m-001 through m-004" in normalized
    assert "not a globally untouched test" in normalized
    assert "untouched-test" not in normalized


def test_lstm_runner_callable_and_cli_contracts_match_the_notebook() -> None:
    signature = inspect.signature(run_lstm_price_only_experiments)
    parameters = signature.parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )
    required = {
        "prices_path",
        "output_dir",
        "run_id",
        "baseline_training_run",
        "baseline_vertical_slice",
    }
    assert {
        name
        for name, parameter in parameters.items()
        if parameter.default is inspect.Parameter.empty
    } == required
    assert parameters["create_zip"].default is True
    assert parameters["device"].default == "auto"
    assert parameters["tensorflow_module"].default is None

    cli = _cli_module()
    defaults = cli.parse_args(
        [
            "--prices",
            "prices.csv",
            "--run-id",
            "LSTM-test",
            "--baseline-training-run",
            "training-run",
            "--baseline-vertical-slice",
            "vertical-run",
        ]
    )
    assert defaults.output_dir == "artifacts/lstm_experiments"
    assert defaults.device == "auto"
    assert defaults.no_zip is False

    overrides = cli.parse_args(
        [
            "--prices",
            "prices.csv",
            "--output-dir",
            "custom-output",
            "--run-id",
            "LSTM-test",
            "--baseline-training-run",
            "training-run",
            "--baseline-vertical-slice",
            "vertical-run",
            "--device",
            "cpu",
            "--no-zip",
        ]
    )
    assert overrides.output_dir == "custom-output"
    assert overrides.device == "cpu"
    assert overrides.no_zip is True
    with pytest.raises(SystemExit):
        cli.parse_args(
            [
                "--prices",
                "prices.csv",
                "--run-id",
                "LSTM-test",
                "--baseline-training-run",
                "training-run",
                "--baseline-vertical-slice",
                "vertical-run",
                "--device",
                "cuda",
            ]
        )


def test_lstm_experiment_notebook_does_not_duplicate_business_logic() -> None:
    code = _joined_source(_notebook(), code_only=True)

    forbidden_fragments = (
        "tf.keras.Sequential",
        "keras.Sequential",
        "tf.keras.layers",
        "layers.LSTM",
        "model.compile(",
        "model.fit(",
        "StandardScaler(",
        "SimpleImputer(",
        "build_lstm_sequence_batch(",
        "BacktestEngine(",
        "calculate_portfolio_metrics(",
        "binary_classification_metrics(",
    )
    for fragment in forbidden_fragments:
        assert fragment not in code
