"""Portable, immutable comparison packs for PAISA price-only models.

The builder in this module is deliberately a reader of completed evidence. It
never loads a fitted estimator, performs inference, or reruns a backtest.
Prediction metrics and portfolio semantics are independently replayed from
persisted, checksum-bound rows.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from paisa.backtesting.metrics import calculate_portfolio_metrics
from paisa.evidence.artifact_security import (
    atomic_rename_noreplace,
    prepare_output_root,
    reject_symlink_components,
)
from paisa.evidence.diagnostics import contains_sensitive_text
from paisa.experiments.artifacts import (
    TrainingRunEvidence,
    assert_training_run_artifacts_unchanged,
    sha256_file,
    to_jsonable,
    validate_training_run_artifacts,
    write_json,
)
from paisa.experiments.evaluation import (
    PREDICTION_COLUMNS,
    binary_classification_metrics,
    validate_prediction_frame,
)
from paisa.experiments.lstm_comparison import (
    FairCommonDateAlignment,
    align_predictions_on_common_dates,
    calibration_information,
)
from paisa.experiments.unified_authority import (
    AUTHORITATIVE_SOURCE_ROOTS,
    expected_evidence_bindings,
)
from paisa.trading.models import (
    Action,
    BPS_DENOMINATOR,
    ExperimentConfig,
    MarketBar,
    PortfolioSnapshot,
    SignalRecord,
    TradeRecord,
    money,
    stable_id,
)


SCHEMA_VERSION = "1.0"
ARTIFACT_TYPE = "paisa_unified_price_model_comparison"
ACADEMIC_DISCLAIMER = "Academic simulation; not financial advice."
HOLDOUT_NAME = "shared project holdout"
MODEL_VARIANTS = ("M-001", "M-002", "M-003", "M-004", "M-005", "M-006")
BASELINE_VARIANTS = MODEL_VARIANTS[:4]
LSTM_VARIANTS = MODEL_VARIANTS[4:]
MAJORITY_BASELINE = "majority-class-baseline"
PREVIOUS_DIRECTION_BASELINE = "previous-direction-baseline"
BENCHMARKS = ("cash-only", "buy-and-hold")
MODEL_DESCRIPTIONS = {
    "M-001": {"algorithm": "random_forest", "label": "PAISA v0 baseline"},
    "M-002": {"algorithm": "random_forest", "label": "conservative"},
    "M-003": {"algorithm": "random_forest", "label": "flexible"},
    "M-004": {"algorithm": "xgboost", "label": "CPU histogram"},
    "M-005": {"algorithm": "lstm", "label": "20-session baseline recurrent"},
    "M-006": {"algorithm": "lstm", "label": "20-session regularized recurrent"},
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{number}" for number in range(1, 10)),
    *(f"LPT{number}" for number in range(1, 10)),
}
_SOURCE_ROLES = ("baseline_training", "baseline_vertical_slice", "lstm")
_PACK_REQUIRED_FILES = {
    "comparison_contract.json",
    "date_alignment.json",
    "derived/reference_baseline_predictions.csv",
    "final_summary.json",
    "metrics.json",
    "rankings.json",
    "report.md",
    "source_roots.json",
}
_PORTFOLIO_FILES = (
    "config.json",
    "signals.csv",
    "portfolio_snapshots.csv",
    "trades.csv",
    "benchmarks/cash_only_snapshots.csv",
    "benchmarks/buy_and_hold_snapshots.csv",
    "benchmarks/buy_and_hold_trades.csv",
)
_MAX_ZIP_FILES = 256
_MAX_ZIP_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
_SIGNAL_FIELDS = (
    "schema_version",
    "signal_id",
    "portfolio_id",
    "model_id",
    "ticker",
    "signal_date",
    "target_date",
    "action",
    "probability_up",
    "target_allocation",
    "reason",
)
_TRADE_FIELDS = (
    "schema_version",
    "trade_id",
    "portfolio_id",
    "signal_date",
    "execution_date",
    "action",
    "quantity",
    "reference_price",
    "execution_price",
    "fee",
    "cash_after_trade",
    "shares_after_trade",
)
_SNAPSHOT_FIELDS = (
    "schema_version",
    "portfolio_id",
    "date",
    "cash",
    "shares",
    "market_price",
    "position_value",
    "total_equity",
    "realized_profit_loss",
    "unrealized_profit_loss",
)


@dataclass(frozen=True, slots=True)
class LSTMRunEvidence:
    """Checksum-verified identity and root inventory for one retained LSTM run."""

    directory: Path
    run_id: str
    run_manifest_sha256: str
    files: tuple[tuple[str, str], ...]
    model_ids: tuple[tuple[str, str], ...]
    baseline_training_run_id: str
    baseline_training_manifest_sha256: str
    baseline_vertical_slice_id: str
    baseline_vertical_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class DirectoryInventoryEvidence:
    """Checksum baseline for the retained vertical-slice directory."""

    directory: Path
    manifest_sha256: str
    files: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SourceSnapshots:
    """The three source-root identities captured before pack generation."""

    training: TrainingRunEvidence
    vertical: DirectoryInventoryEvidence
    lstm: LSTMRunEvidence


@dataclass(frozen=True, slots=True)
class EvidenceFile:
    """One exact source file copied into the portable comparison pack."""

    source_role: str
    source_path: str
    pack_path: str
    sha256: str
    purpose: str


@dataclass(frozen=True, slots=True)
class ReplayResults:
    """All deterministic values replayed from portable row-level evidence."""

    model_registry: Mapping[str, str]
    contract: Mapping[str, Any]
    date_alignment: Mapping[str, Any]
    metrics: Mapping[str, Any]
    rankings: Mapping[str, Any]
    final_summary: Mapping[str, Any]
    baseline_predictions: pd.DataFrame


def _safe_id(value: object, *, field: str) -> str:
    result = str(value).strip()
    device_name = result.split(".", 1)[0].upper()
    if (
        not _SAFE_ID.fullmatch(result)
        or result in {".", ".."}
        or result.rstrip(" .") != result
        or device_name in _WINDOWS_RESERVED_NAMES
    ):
        raise ValueError(f"{field} must be a filesystem-safe identifier")
    return result


def _safe_relative(value: object, *, field: str) -> str:
    raw = str(value)
    if "\\" in raw or ":" in raw:
        raise ValueError(f"{field} must be a portable relative POSIX path")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or not candidate.parts or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    for part in candidate.parts:
        if part.rstrip(" .") != part:
            raise ValueError(f"{field} must be a portable relative POSIX path")
        device_name = part.split(".", 1)[0].upper()
        if device_name in _WINDOWS_RESERVED_NAMES:
            raise ValueError(f"{field} must be a portable relative POSIX path")
    return candidate.as_posix()


def _read_json(path: Path, *, field: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{field} is missing or not a regular file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"Non-finite JSON constant is forbidden: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} must contain valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a JSON object")
    return value


def _regular_files(root: Path, *, exclude: set[str] | None = None) -> dict[str, Path]:
    excluded = exclude or set()
    files: dict[str, Path] = {}
    for candidate in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"Artifacts must not contain symlinks: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"Artifact must be a regular file: {relative}")
        if relative not in excluded:
            files[relative] = candidate
    return files


def validate_vertical_slice_artifacts(
    directory: str | Path,
) -> DirectoryInventoryEvidence:
    """Validate the retained vertical slice without importing plotting modules."""

    candidate = Path(directory)
    if candidate.is_symlink():
        raise ValueError("Baseline vertical-slice directory must not be a symlink")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError("Baseline vertical-slice directory is missing")
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path, field="Vertical-slice manifest")
    raw_entries = manifest.get("artifacts")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Vertical-slice manifest artifacts must be a non-empty array")

    parsed: list[tuple[str, str]] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"path", "sha256"}:
            raise ValueError(
                "Every vertical-slice artifact entry must contain exactly path and sha256"
            )
        relative = _safe_relative(
            raw_entry["path"], field=f"artifacts[{index}].path"
        )
        checksum = str(raw_entry["sha256"])
        if not _SHA256.fullmatch(checksum):
            raise ValueError(f"Invalid vertical-slice SHA-256 for {relative}")
        parsed.append((relative, checksum))
    if parsed != sorted(parsed) or len({item[0] for item in parsed}) != len(parsed):
        raise ValueError("Vertical-slice artifact inventory must be sorted and unique")

    actual = _regular_files(root, exclude={"manifest.json"})
    expected_paths = {relative for relative, _ in parsed}
    if set(actual) != expected_paths:
        raise ValueError(
            "Vertical-slice artifact paths must exactly match its manifest; "
            f"missing={sorted(expected_paths - set(actual))}, "
            f"unexpected={sorted(set(actual) - expected_paths)}"
        )
    for relative, expected in parsed:
        if sha256_file(actual[relative]) != expected:
            raise ValueError(f"Vertical-slice artifact checksum mismatch: {relative}")
    return DirectoryInventoryEvidence(
        directory=root,
        manifest_sha256=sha256_file(manifest_path),
        files=tuple(parsed),
    )


def assert_vertical_slice_artifacts_unchanged(
    evidence: DirectoryInventoryEvidence,
) -> None:
    current = validate_vertical_slice_artifacts(evidence.directory)
    if (
        current.manifest_sha256 != evidence.manifest_sha256
        or current.files != evidence.files
    ):
        raise RuntimeError("Baseline vertical-slice artifacts changed during the run")


def _read_csv_rows(path: Path, *, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Portable CSV evidence is missing")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(
                f"Portable CSV has an unexpected schema: {path.name}"
            )
        return [dict(row) for row in reader]


def _load_experiment_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_mapping(_read_json(path, field="Experiment config"))


def _load_signals(path: Path) -> tuple[SignalRecord, ...]:
    return tuple(
        SignalRecord.from_mapping(row)
        for row in _read_csv_rows(path, expected_columns=_SIGNAL_FIELDS)
    )


def _load_trades(path: Path) -> tuple[TradeRecord, ...]:
    return tuple(
        TradeRecord.from_mapping(row)
        for row in _read_csv_rows(path, expected_columns=_TRADE_FIELDS)
    )


def _load_portfolio_snapshots(path: Path) -> tuple[PortfolioSnapshot, ...]:
    return tuple(
        PortfolioSnapshot.from_mapping(row)
        for row in _read_csv_rows(path, expected_columns=_SNAPSHOT_FIELDS)
    )


def _load_market_bars(path: Path) -> tuple[MarketBar, ...]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("Portable market evidence is missing")
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = tuple(reader.fieldnames or ())
        required = {"date", "ticker", "open", "close"}
        if not required.issubset(columns):
            raise ValueError("Portable market evidence is missing required columns")
        return tuple(MarketBar.from_mapping(row) for row in reader)


def _parse_sha256_inventory(
    value: object,
    *,
    field: str,
    expected_excludes: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    if set(value) != {"algorithm", "root", "excludes", "files"}:
        raise ValueError(
            f"{field} must contain exactly algorithm, root, excludes, and files"
        )
    if value["algorithm"] != "sha256" or value["root"] != ".":
        raise ValueError(f"{field} must use SHA-256 relative to the artifact root")
    if value["excludes"] != list(expected_excludes):
        raise ValueError(f"{field} has an invalid self-exclusion policy")
    raw_files = value["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError(f"{field}.files must be a non-empty array")
    parsed: list[tuple[str, str]] = []
    for index, raw_entry in enumerate(raw_files):
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {"path", "sha256"}:
            raise ValueError(
                f"{field}.files[{index}] must contain exactly path and sha256"
            )
        relative = _safe_relative(raw_entry["path"], field=f"{field}.files[{index}].path")
        checksum = str(raw_entry["sha256"])
        if not _SHA256.fullmatch(checksum):
            raise ValueError(f"{field} contains an invalid SHA-256 for {relative}")
        parsed.append((relative, checksum))
    if parsed != sorted(parsed) or len({path for path, _ in parsed}) != len(parsed):
        raise ValueError(f"{field}.files must be sorted and duplicate-free")
    return tuple(parsed)


def validate_lstm_run_artifacts(
    directory: str | Path,
    *,
    expected_run_id: str | None = None,
) -> LSTMRunEvidence:
    """Validate the complete retained LSTM run without loading model binaries."""

    candidate = Path(directory)
    if candidate.is_symlink():
        raise ValueError("LSTM run directory must not be a symlink")
    root = candidate.resolve(strict=True)
    if not root.is_dir():
        raise FileNotFoundError("LSTM run directory is missing")

    run_manifest_path = root / "run_manifest.json"
    artifact_manifest_path = root / "artifact_manifest.json"
    run_manifest = _read_json(run_manifest_path, field="LSTM run manifest")
    artifact_manifest = _read_json(
        artifact_manifest_path, field="LSTM artifact manifest"
    )
    if run_manifest.get("schema_version") != "1.0":
        raise ValueError("LSTM run manifest schema_version must be '1.0'")
    run_id = _safe_id(run_manifest.get("run_id"), field="LSTM run_id")
    if run_id != root.name:
        raise ValueError("LSTM run_id must match its immutable directory")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError("LSTM run_id does not match the expected source")
    if run_id == "LSTM-20261024-OGDC-price-only-01":
        raise ValueError("The pre-hardening LSTM-01 run is historical evidence only")
    if run_manifest.get("artifact_manifest") != "artifact_manifest.json":
        raise ValueError("LSTM run must reference artifact_manifest.json")

    sealed_run = validate_training_run_artifacts(root, expected_run_id=run_id)
    root_inventory = sealed_run.artifact_sha256
    root_expected = dict(root_inventory)
    artifact_manifest_sha256 = root_expected.get("artifact_manifest.json")
    if (
        artifact_manifest_sha256 is None
        or artifact_manifest_sha256 != sha256_file(artifact_manifest_path)
    ):
        raise ValueError(
            "LSTM artifact manifest is not checksum-bound by run_manifest.json"
        )

    nested_inventory = _parse_sha256_inventory(
        artifact_manifest,
        field="LSTM artifact inventory",
        expected_excludes=("artifact_manifest.json", "run_manifest.json"),
    )
    expected_nested_inventory = tuple(
        (relative, checksum)
        for relative, checksum in root_inventory
        if relative != "artifact_manifest.json"
    )
    if nested_inventory != expected_nested_inventory:
        raise ValueError(
            "LSTM artifact manifest must exactly match the checksum-bound run inventory"
        )

    final_summary = _read_json(root / "final_summary.json", field="LSTM final summary")
    if (
        final_summary.get("schema_version") != "1.1"
        or final_summary.get("status") != "PASS"
        or final_summary.get("run_id") != run_id
        or final_summary.get("artifact_directory") != run_id
        or final_summary.get("path_base") != "output_root"
        or final_summary.get("disclaimer") != ACADEMIC_DISCLAIMER
    ):
        raise ValueError("LSTM final summary is not a portable completed result")

    raw_model_ids = run_manifest.get("model_ids")
    raw_prediction_paths = run_manifest.get("prediction_paths")
    if not isinstance(raw_model_ids, Mapping) or set(raw_model_ids) != set(LSTM_VARIANTS):
        raise ValueError("LSTM run must map exactly M-005 and M-006 model IDs")
    if not isinstance(raw_prediction_paths, Mapping) or set(raw_prediction_paths) != set(
        LSTM_VARIANTS
    ):
        raise ValueError("LSTM run must map exactly M-005 and M-006 predictions")
    model_ids: list[tuple[str, str]] = []
    for variant_id in LSTM_VARIANTS:
        model_id = _safe_id(raw_model_ids[variant_id], field=f"{variant_id} model_id")
        expected_test_path = f"models/{model_id}/predictions.csv"
        expected_all_path = f"models/{model_id}/predictions_all_splits.csv"
        if (
            _safe_relative(
                raw_prediction_paths[variant_id], field=f"{variant_id} prediction path"
            )
            != expected_test_path
            or expected_test_path not in root_expected
            or expected_all_path not in root_expected
        ):
            raise ValueError(f"LSTM {variant_id} prediction paths are not inventoried")
        model_ids.append((variant_id, model_id))
    if dict(sealed_run.model_ids) != dict(model_ids):
        raise ValueError("LSTM registered model identities disagree with sealed evidence")

    training_reference = run_manifest.get("baseline_training_run")
    vertical_reference = run_manifest.get("baseline_vertical_slice")
    if not isinstance(training_reference, Mapping) or not isinstance(
        vertical_reference, Mapping
    ):
        raise ValueError("LSTM run is missing baseline source bindings")
    training_id = _safe_id(
        training_reference.get("run_id"), field="LSTM baseline training run_id"
    )
    training_sha = str(training_reference.get("manifest_sha256", ""))
    vertical_id = _safe_id(
        vertical_reference.get("run_id"), field="LSTM baseline vertical run_id"
    )
    vertical_sha = str(vertical_reference.get("manifest_sha256", ""))
    if not _SHA256.fullmatch(training_sha) or not _SHA256.fullmatch(vertical_sha):
        raise ValueError("LSTM baseline source bindings require SHA-256 digests")

    return LSTMRunEvidence(
        directory=root,
        run_id=run_id,
        run_manifest_sha256=sealed_run.run_manifest_sha256,
        files=root_inventory,
        model_ids=tuple(model_ids),
        baseline_training_run_id=training_id,
        baseline_training_manifest_sha256=training_sha,
        baseline_vertical_slice_id=vertical_id,
        baseline_vertical_manifest_sha256=vertical_sha,
    )


def assert_lstm_run_artifacts_unchanged(evidence: LSTMRunEvidence) -> None:
    """Revalidate an LSTM root and reject any mid-build source mutation."""

    current = validate_lstm_run_artifacts(
        evidence.directory, expected_run_id=evidence.run_id
    )
    if current != evidence:
        raise RuntimeError("LSTM source artifacts changed during comparison generation")


def _capture_source_snapshots(
    *,
    training_run: str | Path,
    vertical_slice: str | Path,
    lstm_run: str | Path,
) -> SourceSnapshots:
    training = validate_training_run_artifacts(training_run)
    vertical = validate_vertical_slice_artifacts(vertical_slice)
    lstm = validate_lstm_run_artifacts(lstm_run)

    observed_roots = {
        "baseline_training": {
            "run_id": training.run_id,
            "manifest_sha256": training.run_manifest_sha256,
        },
        "baseline_vertical_slice": {
            "run_id": vertical.directory.name,
            "manifest_sha256": vertical.manifest_sha256,
        },
        "lstm": {
            "run_id": lstm.run_id,
            "manifest_sha256": lstm.run_manifest_sha256,
        },
    }
    expected_roots = {
        role: {
            "run_id": values["run_id"],
            "manifest_sha256": values["manifest_sha256"],
        }
        for role, values in AUTHORITATIVE_SOURCE_ROOTS.items()
    }
    if observed_roots != expected_roots:
        raise ValueError(
            "Unified comparison accepts only the three authoritative completed sources"
        )

    training_models = dict(training.model_ids)
    if set(training_models) != set(BASELINE_VARIANTS):
        raise ValueError("Baseline training source must contain exactly M-001 through M-004")
    vertical_manifest = _read_json(
        vertical.directory / "manifest.json", field="Vertical-slice manifest"
    )
    if (
        vertical_manifest.get("schema_version") != "1.0"
        or vertical_manifest.get("status") != "PASS"
        or vertical_manifest.get("vertical_slice_run_id") != vertical.directory.name
        or vertical_manifest.get("model_training_run_id") != training.run_id
        or vertical_manifest.get("symbol") != "OGDC"
        or vertical_manifest.get("disclaimer") != ACADEMIC_DISCLAIMER
    ):
        raise ValueError("Vertical-slice source is not the completed OGDC baseline")
    if vertical_manifest.get("registered_model_ids") != training_models:
        raise ValueError("Vertical-slice model identities do not match training evidence")
    vertical_summary = _read_json(
        vertical.directory / "final_summary.json", field="Vertical-slice final summary"
    )
    if vertical_summary.get("status") != "PASS":
        raise ValueError("Vertical-slice final summary must have PASS status")
    training_reference = _read_json(
        vertical.directory / "model_training_run_reference.json",
        field="Vertical-slice training reference",
    )
    if (
        training_reference.get("run_manifest_sha256")
        != training.run_manifest_sha256
        or str(training_reference.get("artifact_directory", "")).rstrip("/").split("/")[
            -1
        ]
        != training.run_id
    ):
        raise ValueError("Vertical slice is not bound to the selected training source")

    if (
        lstm.baseline_training_run_id != training.run_id
        or lstm.baseline_training_manifest_sha256 != training.run_manifest_sha256
        or lstm.baseline_vertical_slice_id != vertical.directory.name
        or lstm.baseline_vertical_manifest_sha256 != vertical.manifest_sha256
    ):
        raise ValueError("LSTM source is not bound to the selected baseline sources")
    return SourceSnapshots(training=training, vertical=vertical, lstm=lstm)


def _assert_source_snapshots_unchanged(snapshots: SourceSnapshots) -> None:
    assert_training_run_artifacts_unchanged(snapshots.training)
    assert_vertical_slice_artifacts_unchanged(snapshots.vertical)
    assert_lstm_run_artifacts_unchanged(snapshots.lstm)


def _validate_output_separation(output_root: Path, snapshots: SourceSnapshots) -> None:
    """Reject publishing at, above, or below any retained evidence source."""

    output = output_root.resolve(strict=False)
    for source in (
        snapshots.training.run_directory,
        snapshots.vertical.directory,
        snapshots.lstm.directory,
    ):
        source_root = source.resolve(strict=True)
        if output == source_root or output in source_root.parents or source_root in output.parents:
            raise ValueError("Comparison output must be separate from all source roots")


def _source_root_for_role(snapshots: SourceSnapshots, role: str) -> Path:
    if role == "baseline_training":
        return snapshots.training.run_directory
    if role == "baseline_vertical_slice":
        return snapshots.vertical.directory
    if role == "lstm":
        return snapshots.lstm.directory
    raise KeyError(role)


def _source_inventory_for_role(
    snapshots: SourceSnapshots, role: str
) -> dict[str, str]:
    if role == "baseline_training":
        return dict(snapshots.training.artifact_sha256)
    if role == "baseline_vertical_slice":
        return dict(snapshots.vertical.files)
    if role == "lstm":
        return dict(snapshots.lstm.files)
    raise KeyError(role)


def _source_manifest_specs(snapshots: SourceSnapshots) -> list[EvidenceFile]:
    specs = [
        EvidenceFile(
            source_role="baseline_training",
            source_path="run_manifest.json",
            pack_path="evidence/source_manifests/baseline_training_run_manifest.json",
            sha256=snapshots.training.run_manifest_sha256,
            purpose="source_integrity_root",
        ),
        EvidenceFile(
            source_role="lstm",
            source_path="run_manifest.json",
            pack_path="evidence/source_manifests/lstm_run_manifest.json",
            sha256=snapshots.lstm.run_manifest_sha256,
            purpose="source_integrity_root",
        ),
    ]
    lstm_root_inventory = dict(snapshots.lstm.files)
    if "artifact_manifest.json" not in lstm_root_inventory:
        raise ValueError("LSTM root inventory is missing artifact_manifest.json")
    specs.append(
        EvidenceFile(
            source_role="lstm",
            source_path="artifact_manifest.json",
            pack_path="evidence/source_manifests/lstm_artifact_manifest.json",
            sha256=lstm_root_inventory["artifact_manifest.json"],
            purpose="source_payload_inventory",
        )
    )
    return specs


def _evidence_specs(snapshots: SourceSnapshots) -> list[EvidenceFile]:
    specs = _source_manifest_specs(snapshots)
    training_inventory = _source_inventory_for_role(snapshots, "baseline_training")
    vertical_inventory = _source_inventory_for_role(snapshots, "baseline_vertical_slice")
    lstm_inventory = _source_inventory_for_role(snapshots, "lstm")

    def add(
        role: str,
        source_path: str,
        pack_path: str,
        purpose: str,
        inventory: Mapping[str, str],
    ) -> None:
        if source_path not in inventory:
            raise ValueError(f"Required {role} evidence is not inventoried: {source_path}")
        specs.append(
            EvidenceFile(
                source_role=role,
                source_path=source_path,
                pack_path=pack_path,
                sha256=inventory[source_path],
                purpose=purpose,
            )
        )

    add(
        "baseline_training",
        "split_manifest.json",
        "evidence/splits/baseline_split_manifest.json",
        "chronological_split_membership",
        training_inventory,
    )
    add(
        "lstm",
        "split_manifest.json",
        "evidence/splits/lstm_split_manifest.json",
        "lstm_sequence_split_membership",
        lstm_inventory,
    )
    add(
        "baseline_vertical_slice",
        "input_snapshot/psx_prices.csv",
        "evidence/market_data/baseline_psx_prices.csv",
        "portfolio_market_observations",
        vertical_inventory,
    )
    add(
        "lstm",
        "frozen_input/OGDC_prices.csv",
        "evidence/market_data/lstm_ogdc_prices.csv",
        "portfolio_market_observations",
        lstm_inventory,
    )

    for variant_id, registered_id in snapshots.training.model_ids:
        source_path = f"models/{registered_id}/predictions_all_splits.csv"
        add(
            "baseline_training",
            source_path,
            f"evidence/predictions/{variant_id}/predictions_all_splits.csv",
            "model_prediction_rows",
            training_inventory,
        )
        experiment = f"{snapshots.vertical.directory.name}-{variant_id.lower()}"
        for relative in _PORTFOLIO_FILES:
            source_path = f"experiments/{experiment}/{relative}"
            add(
                "baseline_vertical_slice",
                source_path,
                f"evidence/portfolios/{variant_id}/{relative}",
                "portfolio_configuration" if relative == "config.json" else "portfolio_ledger",
                vertical_inventory,
            )

    for variant_id, registered_id in snapshots.lstm.model_ids:
        source_path = f"models/{registered_id}/predictions_all_splits.csv"
        add(
            "lstm",
            source_path,
            f"evidence/predictions/{variant_id}/predictions_all_splits.csv",
            "model_prediction_rows",
            lstm_inventory,
        )
        experiment = f"{snapshots.lstm.run_id}-{variant_id.lower()}"
        for relative in _PORTFOLIO_FILES:
            source_path = f"backtests/{experiment}/{relative}"
            add(
                "lstm",
                source_path,
                f"evidence/portfolios/{variant_id}/{relative}",
                "portfolio_configuration" if relative == "config.json" else "portfolio_ledger",
                lstm_inventory,
            )

    pack_paths = [spec.pack_path for spec in specs]
    if len(pack_paths) != len(set(pack_paths)):
        raise AssertionError("Portable evidence paths must be unique")
    return sorted(specs, key=lambda spec: spec.pack_path)


def _write_vertical_manifest_certificate(
    stage_dir: Path, snapshots: SourceSnapshots
) -> tuple[str, str]:
    """Write a safe projection bound to the source manifest's immutable digest."""

    pack_path = (
        "evidence/source_manifests/baseline_vertical_manifest_certificate.json"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "paisa_source_integrity_certificate",
        "source_role": "baseline_vertical_slice",
        "run_id": snapshots.vertical.directory.name,
        "source_manifest": {
            "name": "manifest.json",
            "sha256": snapshots.vertical.manifest_sha256,
        },
        "projection_policy": "allowlisted_fields_v1",
        "raw_manifest_included": False,
        "source_assertions": {
            "schema_version": "1.0",
            "status": "PASS",
            "symbol": "OGDC",
            "model_training_run_id": snapshots.training.run_id,
            "registered_model_ids": dict(snapshots.training.model_ids),
            "disclaimer": ACADEMIC_DISCLAIMER,
        },
        "artifact_inventory": {
            "algorithm": "sha256",
            "root": ".",
            "excludes": ["manifest.json"],
            "files": [
                {"path": relative, "sha256": checksum}
                for relative, checksum in snapshots.vertical.files
            ],
        },
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    destination = stage_dir / pack_path
    write_json(payload, destination)
    certificate_sha256 = sha256_file(destination)
    expected = AUTHORITATIVE_SOURCE_ROOTS["baseline_vertical_slice"].get(
        "portable_certificate_sha256"
    )
    if certificate_sha256 != expected:
        raise ValueError("Vertical-slice certificate does not match the allowlisted projection")
    return pack_path, certificate_sha256


def _copy_evidence(
    stage_dir: Path,
    snapshots: SourceSnapshots,
    specs: Sequence[EvidenceFile],
) -> None:
    for spec in specs:
        source = _source_root_for_role(snapshots, spec.source_role) / spec.source_path
        if source.is_symlink() or not source.is_file():
            raise FileNotFoundError(f"Source evidence is missing: {spec.source_path}")
        if sha256_file(source) != spec.sha256:
            raise RuntimeError(f"Source evidence changed before copy: {spec.source_path}")
        destination = stage_dir / spec.pack_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != spec.sha256:
            raise RuntimeError(f"Copied evidence checksum mismatch: {spec.pack_path}")


def _source_roots_payload(
    comparison_id: str,
    snapshots: SourceSnapshots,
    specs: Sequence[EvidenceFile],
    *,
    vertical_certificate_path: str,
    vertical_certificate_sha256: str,
) -> dict[str, Any]:
    roots = [
        {
            "role": "baseline_training",
            "artifact_type": "model_training_run",
            "run_id": snapshots.training.run_id,
            "integrity_root_name": "run_manifest.json",
            "integrity_root_sha256": snapshots.training.run_manifest_sha256,
            "payload_file_count": len(snapshots.training.artifact_sha256),
            "integrity_root_pack_path": (
                "evidence/source_manifests/baseline_training_run_manifest.json"
            ),
            "integrity_root_pack_sha256": snapshots.training.run_manifest_sha256,
            "integrity_root_derivation": "byte_copy",
        },
        {
            "role": "baseline_vertical_slice",
            "artifact_type": "vertical_slice",
            "run_id": snapshots.vertical.directory.name,
            "integrity_root_name": "manifest.json",
            "integrity_root_sha256": snapshots.vertical.manifest_sha256,
            "payload_file_count": len(snapshots.vertical.files),
            "integrity_root_pack_path": vertical_certificate_path,
            "integrity_root_pack_sha256": vertical_certificate_sha256,
            "integrity_root_derivation": "deterministic_redacted_projection",
        },
        {
            "role": "lstm",
            "artifact_type": "lstm_price_only_experiment",
            "run_id": snapshots.lstm.run_id,
            "integrity_root_name": "run_manifest.json",
            "integrity_root_sha256": snapshots.lstm.run_manifest_sha256,
            "payload_file_count": len(snapshots.lstm.files),
            "integrity_root_pack_path": "evidence/source_manifests/lstm_run_manifest.json",
            "integrity_root_pack_sha256": snapshots.lstm.run_manifest_sha256,
            "integrity_root_derivation": "byte_copy",
        },
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "sources": roots,
        "relationships": {
            "vertical_slice_uses_training_run": True,
            "lstm_uses_training_run": True,
            "lstm_uses_vertical_slice": True,
        },
        "evidence_files": [
            {
                "source_role": spec.source_role,
                "source_path": spec.source_path,
                "pack_path": spec.pack_path,
                "sha256": spec.sha256,
                "purpose": spec.purpose,
                "derivation": "byte_copy",
            }
            for spec in specs
        ],
        "source_revalidation": {
            "validated_before_generation": True,
            "revalidated_before_publication": True,
        },
    }


def _model_registry(pack_root: Path) -> dict[str, str]:
    training_manifest = _read_json(
        pack_root / "evidence/source_manifests/baseline_training_run_manifest.json",
        field="Copied baseline training manifest",
    )
    lstm_manifest = _read_json(
        pack_root / "evidence/source_manifests/lstm_run_manifest.json",
        field="Copied LSTM run manifest",
    )
    raw_training_models = training_manifest.get("models")
    if not isinstance(raw_training_models, list):
        raise ValueError("Copied training manifest must contain model records")
    registry: dict[str, str] = {}
    for raw_record in raw_training_models:
        if not isinstance(raw_record, Mapping):
            raise ValueError("Copied training model record must be an object")
        variant_id = str(raw_record.get("variant_id", ""))
        model_id = str(raw_record.get("model_id", ""))
        if variant_id in registry:
            raise ValueError("Copied training manifest repeats a variant")
        registry[variant_id] = _safe_id(model_id, field=f"{variant_id} model_id")
    raw_lstm_models = lstm_manifest.get("model_ids")
    if not isinstance(raw_lstm_models, Mapping):
        raise ValueError("Copied LSTM manifest must contain model_ids")
    for variant_id, raw_model_id in raw_lstm_models.items():
        variant = str(variant_id)
        if variant in registry:
            raise ValueError("Copied source manifests repeat a variant")
        registry[variant] = _safe_id(raw_model_id, field=f"{variant} model_id")
    if set(registry) != set(MODEL_VARIANTS):
        raise ValueError("Portable source manifests must bind exactly M-001 through M-006")
    return {variant: registry[variant] for variant in MODEL_VARIANTS}


def _target_dates(split_manifest: Mapping[str, Any], split: str) -> tuple[str, ...]:
    raw_target_dates = split_manifest.get("target_dates")
    if not isinstance(raw_target_dates, Mapping):
        raise ValueError("Split manifest must contain target_dates")
    raw_values = raw_target_dates.get(split)
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(f"Split manifest must contain non-empty {split} target dates")
    values = tuple(str(value) for value in raw_values)
    parsed = pd.to_datetime(pd.Series(values), format="%Y-%m-%d", errors="raise")
    if parsed.duplicated().any() or not parsed.is_monotonic_increasing:
        raise ValueError(f"{split} target dates must be unique and chronological")
    return values


def _load_all_prediction_frames(
    pack_root: Path,
    registry: Mapping[str, str],
    *,
    baseline_targets: Mapping[str, Sequence[str]],
    lstm_targets: Mapping[str, Sequence[str]],
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    exact_prediction_columns = set(PREDICTION_COLUMNS)
    for variant_id in MODEL_VARIANTS:
        path = (
            pack_root
            / "evidence"
            / "predictions"
            / variant_id
            / "predictions_all_splits.csv"
        )
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Portable prediction evidence is missing for {variant_id}")
        raw = pd.read_csv(path)
        expected_columns = (
            exact_prediction_columns | {"split"}
            if variant_id in BASELINE_VARIANTS
            else exact_prediction_columns
        )
        if set(raw.columns) != expected_columns:
            raise ValueError(f"Unexpected all-split prediction schema for {variant_id}")
        normalized = validate_prediction_frame(raw.loc[:, PREDICTION_COLUMNS])
        if set(normalized["model_id"]) != {registry[variant_id]}:
            raise ValueError(f"Prediction model identity mismatch for {variant_id}")
        if normalized["actual_class"].isna().any():
            raise ValueError("Completed comparison predictions require every actual_class")
        if set(normalized["ticker"].astype(str)) != {"OGDC"}:
            raise ValueError(f"Prediction ticker must be OGDC for {variant_id}")

        declared_targets = (
            baseline_targets if variant_id in BASELINE_VARIANTS else lstm_targets
        )
        target_to_split: dict[str, str] = {}
        for split in ("train", "validation", "test"):
            for target_date in declared_targets[split]:
                if target_date in target_to_split:
                    raise ValueError("Declared prediction split dates must not overlap")
                target_to_split[str(target_date)] = split
        observed_target_dates = normalized["target_date"].astype(str).tolist()
        if (
            len(observed_target_dates) != len(target_to_split)
            or set(observed_target_dates) != set(target_to_split)
        ):
            raise ValueError(
                f"Prediction rows must exactly cover declared split dates for {variant_id}"
            )
        if variant_id in BASELINE_VARIANTS:
            split_values = set(raw["split"].astype(str))
            if split_values != {"train", "validation", "test"}:
                raise ValueError(f"Baseline split labels are incomplete for {variant_id}")
            for target_date, split in raw[["target_date", "split"]].itertuples(
                index=False, name=None
            ):
                if target_to_split.get(str(target_date)) != str(split):
                    raise ValueError(
                        f"Baseline split membership mismatch for {variant_id}"
                    )
        frames[variant_id] = normalized
    return frames


def _scope_alignment(
    frames: Mapping[str, pd.DataFrame], target_dates: Sequence[str]
) -> FairCommonDateAlignment:
    selected: dict[str, pd.DataFrame] = {}
    allowed = set(target_dates)
    for variant_id, frame in frames.items():
        scoped = frame.loc[frame["target_date"].astype(str).isin(allowed)].copy()
        if scoped.empty:
            raise ValueError(f"{variant_id} has no rows in the requested comparison scope")
        selected[variant_id] = scoped
    return align_predictions_on_common_dates(selected)


def _require_exact_scope_alignment(
    alignment: FairCommonDateAlignment,
    *,
    target_dates: Sequence[str],
    scope: str,
) -> None:
    expected_dates = tuple(str(value) for value in target_dates)
    expected_count = len(expected_dates)
    if alignment.common_target_dates != expected_dates:
        raise ValueError(f"{scope} common target dates must match the split manifest")
    if len(alignment.common_keys) != expected_count:
        raise ValueError(f"{scope} must contain exactly one common key per target date")
    for variant_id in MODEL_VARIANTS:
        if (
            alignment.original_counts[variant_id] != expected_count
            or alignment.common_counts[variant_id] != expected_count
            or alignment.dropped_target_dates[variant_id]
        ):
            raise ValueError(
                f"{scope} rows must align exactly across all six models"
            )


def _alignment_payload(
    alignment: FairCommonDateAlignment,
    *,
    terminology: str,
) -> dict[str, Any]:
    first = alignment.frame_for(MODEL_VARIANTS[0]).set_index(
        ["ticker", "feature_date", "target_date"]
    )
    keys: list[dict[str, Any]] = []
    for ticker, feature_date, target_date in alignment.common_keys:
        actual = int(first.loc[(ticker, feature_date, target_date), "actual_class"])
        keys.append(
            {
                "ticker": ticker,
                "feature_date": feature_date,
                "target_date": target_date,
                "actual_class": actual,
            }
        )
    return {
        "terminology": terminology,
        "common_count": len(alignment.common_keys),
        "feature_start": min(key[1] for key in alignment.common_keys),
        "feature_end": max(key[1] for key in alignment.common_keys),
        "target_start": min(key[2] for key in alignment.common_keys),
        "target_end": max(key[2] for key in alignment.common_keys),
        "original_counts": dict(alignment.original_counts),
        "common_counts": dict(alignment.common_counts),
        "dropped_target_dates": {
            variant: list(values)
            for variant, values in alignment.dropped_target_dates.items()
        },
        "common_keys": keys,
    }


def _prediction_metric_record(
    *,
    competitor_id: str,
    competitor_type: str,
    registered_model_id: str | None,
    evaluation_scope: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    normalized = validate_prediction_frame(frame.loc[:, PREDICTION_COLUMNS])
    actual = normalized["actual_class"].astype(int).to_numpy()
    metrics = binary_classification_metrics(
        actual,
        normalized["predicted_class"].astype(int).to_numpy(),
        normalized["probability_up"].astype(float).to_numpy(),
    )
    calibration = calibration_information(
        actual, normalized["probability_up"].astype(float).to_numpy()
    )
    return to_jsonable(
        {
            "competitor_id": competitor_id,
            "competitor_type": competitor_type,
            "registered_model_id": registered_model_id,
            "evaluation_scope": evaluation_scope,
            **metrics,
            "expected_calibration_error": calibration[
                "expected_calibration_error"
            ],
            "calibration": calibration,
        }
    )


def _reference_baseline_frame(
    *,
    baseline_id: str,
    evaluation_scope: str,
    scored_rows: pd.DataFrame,
    context_rows: pd.DataFrame,
    majority_class: int,
) -> pd.DataFrame:
    scored = validate_prediction_frame(scored_rows.loc[:, PREDICTION_COLUMNS])
    if baseline_id == MAJORITY_BASELINE:
        predicted = np.full(len(scored), majority_class, dtype=int)
    elif baseline_id == PREVIOUS_DIRECTION_BASELINE:
        lookup: dict[tuple[str, str], int] = {}
        for ticker, target_date, actual in context_rows[
            ["ticker", "target_date", "actual_class"]
        ].itertuples(index=False, name=None):
            key = (str(ticker), str(target_date))
            if key in lookup:
                raise ValueError("Previous-direction context contains duplicate target keys")
            lookup[key] = int(actual)
        previous: list[int] = []
        for ticker, feature_date in scored[["ticker", "feature_date"]].itertuples(
            index=False, name=None
        ):
            key = (str(ticker), str(feature_date))
            if key not in lookup:
                raise ValueError(
                    "Previous-direction baseline is missing the immediately prior row"
                )
            previous.append(lookup[key])
        predicted = np.asarray(previous, dtype=int)
    else:
        raise KeyError(baseline_id)
    result = scored.copy()
    result["model_id"] = baseline_id
    result["predicted_class"] = predicted
    result["probability_up"] = predicted.astype(float)
    result["probability_down"] = 1.0 - result["probability_up"]
    normalized = validate_prediction_frame(result)
    normalized.insert(0, "baseline_id", baseline_id)
    normalized.insert(0, "evaluation_scope", evaluation_scope)
    return normalized


def _rank_prediction_rows(
    rows: Sequence[Mapping[str, Any]], *, primary: str
) -> list[dict[str, Any]]:
    if primary == "balanced_accuracy":
        keys = ("balanced_accuracy", "f1", "accuracy")
    elif primary == "f1":
        keys = ("f1", "balanced_accuracy", "accuracy")
    else:
        raise KeyError(primary)

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        numeric: list[float] = []
        for key in keys:
            value = row.get(key)
            numeric.append(float("inf") if value is None else -float(value))
        model_tie_break = row.get("registered_model_id") or row["competitor_id"]
        return (*numeric, str(model_tie_break))

    ordered = sorted(rows, key=sort_key)
    return [
        {
            "rank": index,
            "competitor_id": row["competitor_id"],
            "registered_model_id": row.get("registered_model_id"),
            primary: row.get(primary),
            **{key: row.get(key) for key in keys if key != primary},
        }
        for index, row in enumerate(ordered, start=1)
    ]


def _portfolio_assumption_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": config["symbol"],
        "initial_cash": config["initial_cash"],
        "thresholds": config["thresholds"],
        "maximum_allocation": config["maximum_allocation"],
        "transaction_cost": config["transaction_cost"],
        "evaluation_dates": config["evaluation_dates"],
        "execution_policy": config["execution_policy"],
        "metric_assumptions": config["metric_assumptions"],
    }


def _validate_expected_portfolio_assumptions(signature: Mapping[str, Any]) -> None:
    expected = {
        "symbol": "OGDC",
        "initial_cash": "100000.00",
        "thresholds": {
            "buy_probability_up_gte": 0.6,
            "sell_probability_up_lte": 0.4,
        },
        "maximum_allocation": 0.5,
        "transaction_cost": {"type": "proportional_bps", "value": 10.0},
        "execution_policy": {
            "signal_timing": "after_close",
            "execution_session": "next_trading_session",
            "reference_price_field": "open",
            "slippage_bps": 0.0,
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
    }
    for key, value in expected.items():
        if signature.get(key) != value:
            raise ValueError(f"Portfolio assumption mismatch for {key}")


def _metric_without_daily_returns(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "daily_returns"}


def _validate_ledger_identity(
    snapshots: Sequence[Any], trades: Sequence[Any], *, portfolio_id: str
) -> None:
    if not snapshots:
        raise ValueError(f"Portfolio {portfolio_id} has no snapshots")
    if any(snapshot.portfolio_id != portfolio_id for snapshot in snapshots):
        raise ValueError(f"Portfolio snapshot identity mismatch for {portfolio_id}")
    if any(trade.portfolio_id != portfolio_id for trade in trades):
        raise ValueError(f"Portfolio trade identity mismatch for {portfolio_id}")


def _raw_holdout_probabilities(
    pack_root: Path,
    *,
    variant_id: str,
    registered_model_id: str,
    common_keys: Sequence[tuple[str, str, str]],
) -> dict[tuple[str, str, str], Decimal]:
    path = pack_root / f"evidence/predictions/{variant_id}/predictions_all_splits.csv"
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        expected = tuple(PREDICTION_COLUMNS) + (
            ("split",) if variant_id in BASELINE_VARIANTS else ()
        )
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(f"Prediction CSV schema mismatch for {variant_id}")
        wanted = set(common_keys)
        result: dict[tuple[str, str, str], Decimal] = {}
        for row in reader:
            key = (row["ticker"], row["feature_date"], row["target_date"])
            if key not in wanted:
                continue
            if row["model_id"] != registered_model_id or key in result:
                raise ValueError(f"Prediction identity mismatch for {variant_id}")
            probability = Decimal(row["probability_up"])
            if not probability.is_finite() or not Decimal("0") <= probability <= Decimal("1"):
                raise ValueError(f"Prediction probability is invalid for {variant_id}")
            result[key] = probability
    if set(result) != set(common_keys):
        raise ValueError(f"Prediction probability coverage mismatch for {variant_id}")
    return result


def _portfolio_market_sessions(
    pack_root: Path, holdout_alignment: FairCommonDateAlignment
) -> tuple[MarketBar, ...]:
    keys = tuple(
        (str(ticker), str(feature_date), str(target_date))
        for ticker, feature_date, target_date in holdout_alignment.common_keys
    )
    if not keys:
        raise ValueError("Portfolio replay requires shared project holdout rows")
    session_dates = (keys[0][1], *(key[2] for key in keys))
    if len(session_dates) != len(set(session_dates)):
        raise ValueError("Portfolio session vector must be unique")
    for index, key in enumerate(keys):
        if key[0] != "OGDC" or key[1] != session_dates[index] or key[2] != session_dates[index + 1]:
            raise ValueError("Shared project holdout rows are not consecutive sessions")

    start = date.fromisoformat(session_dates[0])
    end = date.fromisoformat(session_dates[-1])

    def select(path: Path) -> tuple[MarketBar, ...]:
        selected = tuple(
            bar
            for bar in _load_market_bars(path)
            if bar.ticker == "OGDC" and start <= bar.date <= end
        )
        if tuple(bar.date.isoformat() for bar in selected) != session_dates:
            raise ValueError("Market evidence does not exactly cover the portfolio sessions")
        return selected

    baseline = select(
        pack_root / "evidence/market_data/baseline_psx_prices.csv"
    )
    lstm = select(pack_root / "evidence/market_data/lstm_ogdc_prices.csv")
    if baseline != lstm:
        raise ValueError("Baseline and LSTM market observations disagree")
    expected_classes = tuple(
        int(target.close > feature.close)
        for feature, target in zip(baseline, baseline[1:])
    )
    for variant_id in MODEL_VARIANTS:
        observed_classes = tuple(
            int(value)
            for value in holdout_alignment.frame_for(variant_id)["actual_class"]
        )
        if observed_classes != expected_classes:
            raise ValueError(
                f"Persisted target labels disagree with market closes for {variant_id}"
            )
    return baseline


def _expected_signal(
    *,
    config: ExperimentConfig,
    probability_up: Decimal,
    feature_date: date,
    target_date: date,
    shares: int,
) -> SignalRecord:
    if probability_up >= config.thresholds.buy_probability_up_gte:
        if shares == 0:
            action, allocation, reason = (
                Action.BUY,
                config.maximum_allocation,
                "buy_threshold_reached",
            )
        else:
            action, allocation, reason = Action.HOLD, None, "already_in_position"
    elif probability_up <= config.thresholds.sell_probability_up_lte:
        if shares > 0:
            action, allocation, reason = (
                Action.SELL,
                Decimal("0"),
                "sell_threshold_reached",
            )
        else:
            action, allocation, reason = Action.HOLD, None, "already_in_cash"
    else:
        action, allocation, reason = Action.HOLD, None, "inside_neutral_band"
    signal_id = stable_id(
        "signal",
        config.portfolio_id,
        config.model_id,
        config.symbol,
        feature_date.isoformat(),
        target_date.isoformat(),
        action.value,
    )
    return SignalRecord(
        signal_id=signal_id,
        portfolio_id=config.portfolio_id,
        model_id=config.model_id,
        ticker=config.symbol,
        signal_date=feature_date,
        target_date=target_date,
        action=action,
        probability_up=probability_up,
        target_allocation=allocation,
        reason=reason,
    )


def _expected_trade_and_state(
    *,
    config: ExperimentConfig,
    signal: SignalRecord,
    bar: MarketBar,
    cash: Decimal,
    shares: int,
    open_cost_basis: Decimal,
    realized: Decimal,
) -> tuple[TradeRecord | None, Decimal, int, Decimal, Decimal]:
    if signal.action is Action.HOLD:
        return None, cash, shares, open_cost_basis, realized
    slippage = config.execution_policy.slippage_bps / BPS_DENOMINATOR
    multiplier = Decimal("1") + slippage if signal.action is Action.BUY else Decimal("1") - slippage
    execution_price = money(bar.open * multiplier, field_name="execution_price")
    fee_rate = config.transaction_cost.value / BPS_DENOMINATOR

    if signal.action is Action.BUY:
        equity = money(cash + Decimal(shares) * bar.open)
        cash_limit = cash / (execution_price * (Decimal("1") + fee_rate))
        allocation_limit = config.maximum_allocation * equity / execution_price
        quantity = min(
            int(cash_limit.to_integral_value(rounding=ROUND_FLOOR)),
            int(allocation_limit.to_integral_value(rounding=ROUND_FLOOR)),
        )
        while quantity > 0:
            notional = money(Decimal(quantity) * execution_price)
            fee = money(Decimal(quantity) * execution_price * fee_rate)
            if (
                money(notional + fee) <= cash
                and notional <= config.maximum_allocation * equity
            ):
                break
            quantity -= 1
        if quantity == 0:
            return None, cash, shares, open_cost_basis, realized
        notional = money(Decimal(quantity) * execution_price)
        fee = money(Decimal(quantity) * execution_price * fee_rate)
        total_cost = money(notional + fee)
        cash_after = money(cash - total_cost)
        shares_after = quantity
        cost_after = total_cost
        realized_after = realized
    else:
        quantity = shares
        if quantity <= 0:
            raise ValueError("SELL signal cannot execute while the replay is flat")
        notional = money(Decimal(quantity) * execution_price)
        fee = money(Decimal(quantity) * execution_price * fee_rate)
        net_proceeds = money(notional - fee)
        cash_after = money(cash + net_proceeds)
        shares_after = 0
        realized_after = money(realized + money(net_proceeds - open_cost_basis))
        cost_after = Decimal("0.00")

    trade_id = stable_id(
        "trade",
        signal.portfolio_id,
        signal.signal_id,
        bar.date.isoformat(),
        signal.action.value,
        quantity,
        execution_price,
        fee,
    )
    trade = TradeRecord(
        trade_id=trade_id,
        portfolio_id=signal.portfolio_id,
        signal_date=signal.signal_date,
        execution_date=bar.date,
        action=signal.action,
        quantity=quantity,
        reference_price=bar.open,
        execution_price=execution_price,
        fee=fee,
        cash_after_trade=cash_after,
        shares_after_trade=shares_after,
    )
    return trade, cash_after, shares_after, cost_after, realized_after


def _expected_snapshot(
    *,
    portfolio_id: str,
    bar: MarketBar,
    cash: Decimal,
    shares: int,
    open_cost_basis: Decimal,
    realized: Decimal,
) -> PortfolioSnapshot:
    position = money(Decimal(shares) * bar.close)
    return PortfolioSnapshot(
        portfolio_id=portfolio_id,
        date=bar.date,
        cash=cash,
        shares=shares,
        market_price=bar.close,
        position_value=position,
        total_equity=money(cash + position),
        realized_profit_loss=realized,
        unrealized_profit_loss=(
            money(position - open_cost_basis) if shares else Decimal("0.00")
        ),
    )


def _replay_strategy_ledger(
    *,
    variant_id: str,
    config: ExperimentConfig,
    bars: Sequence[MarketBar],
    probabilities: Mapping[tuple[str, str, str], Decimal],
    signals: Sequence[SignalRecord],
    snapshots: Sequence[PortfolioSnapshot],
    trades: Sequence[TradeRecord],
) -> None:
    expected_count = len(bars) - 1
    if len(signals) != expected_count or len(snapshots) != len(bars):
        raise ValueError(f"Strategy ledger row count mismatch for {variant_id}")
    cash = config.initial_cash
    shares = 0
    open_cost_basis = Decimal("0.00")
    realized = Decimal("0.00")
    if snapshots[0] != _expected_snapshot(
        portfolio_id=config.portfolio_id,
        bar=bars[0],
        cash=cash,
        shares=shares,
        open_cost_basis=open_cost_basis,
        realized=realized,
    ):
        raise ValueError(f"Opening portfolio state mismatch for {variant_id}")

    trade_index = 0
    for index, (feature_bar, target_bar) in enumerate(zip(bars, bars[1:])):
        key = (
            config.symbol,
            feature_bar.date.isoformat(),
            target_bar.date.isoformat(),
        )
        signal = _expected_signal(
            config=config,
            probability_up=probabilities[key],
            feature_date=feature_bar.date,
            target_date=target_bar.date,
            shares=shares,
        )
        if signals[index] != signal:
            raise ValueError(f"Persisted signal semantics mismatch for {variant_id}")
        expected_trade, cash, shares, open_cost_basis, realized = (
            _expected_trade_and_state(
                config=config,
                signal=signal,
                bar=target_bar,
                cash=cash,
                shares=shares,
                open_cost_basis=open_cost_basis,
                realized=realized,
            )
        )
        if expected_trade is not None:
            if trade_index >= len(trades) or trades[trade_index] != expected_trade:
                raise ValueError(f"Persisted trade semantics mismatch for {variant_id}")
            trade_index += 1
        elif trade_index < len(trades) and trades[trade_index].execution_date <= target_bar.date:
            raise ValueError(f"Unexpected trade without an executable signal for {variant_id}")
        expected_snapshot = _expected_snapshot(
            portfolio_id=config.portfolio_id,
            bar=target_bar,
            cash=cash,
            shares=shares,
            open_cost_basis=open_cost_basis,
            realized=realized,
        )
        if snapshots[index + 1] != expected_snapshot:
            raise ValueError(f"Portfolio state transition mismatch for {variant_id}")
    if trade_index != len(trades):
        raise ValueError(f"Unexpected trailing trades for {variant_id}")


def _replay_cash_ledger(
    *, config: ExperimentConfig, bars: Sequence[MarketBar], snapshots: Sequence[PortfolioSnapshot]
) -> None:
    if len(snapshots) != len(bars):
        raise ValueError("Cash-only benchmark session count mismatch")
    for bar, snapshot in zip(bars, snapshots, strict=True):
        expected = _expected_snapshot(
            portfolio_id=config.benchmark_portfolio_ids.cash_only,
            bar=bar,
            cash=config.initial_cash,
            shares=0,
            open_cost_basis=Decimal("0.00"),
            realized=Decimal("0.00"),
        )
        if snapshot != expected:
            raise ValueError("Cash-only benchmark must remain independently funded and flat")


def _replay_buy_and_hold_ledger(
    *,
    config: ExperimentConfig,
    bars: Sequence[MarketBar],
    snapshots: Sequence[PortfolioSnapshot],
    trades: Sequence[TradeRecord],
) -> None:
    if len(snapshots) != len(bars) or len(trades) != 1 or len(bars) < 2:
        raise ValueError("Buy-and-hold benchmark must contain one controlled entry")
    portfolio_id = config.benchmark_portfolio_ids.buy_and_hold
    cash = config.initial_cash
    shares = 0
    open_cost_basis = Decimal("0.00")
    realized = Decimal("0.00")
    if snapshots[0] != _expected_snapshot(
        portfolio_id=portfolio_id,
        bar=bars[0],
        cash=cash,
        shares=shares,
        open_cost_basis=open_cost_basis,
        realized=realized,
    ):
        raise ValueError("Buy-and-hold opening state mismatch")
    signal = SignalRecord(
        signal_id=stable_id(
            "signal",
            portfolio_id,
            "buy-and-hold",
            config.symbol,
            bars[0].date.isoformat(),
            bars[1].date.isoformat(),
            Action.BUY.value,
        ),
        portfolio_id=portfolio_id,
        model_id=config.model_id,
        ticker=config.symbol,
        signal_date=bars[0].date,
        target_date=bars[1].date,
        action=Action.BUY,
        probability_up=Decimal("1"),
        target_allocation=config.maximum_allocation,
        reason="buy_and_hold_entry",
    )
    expected_trade, cash, shares, open_cost_basis, realized = _expected_trade_and_state(
        config=config,
        signal=signal,
        bar=bars[1],
        cash=cash,
        shares=shares,
        open_cost_basis=open_cost_basis,
        realized=realized,
    )
    if expected_trade is None or trades[0] != expected_trade:
        raise ValueError("Buy-and-hold entry does not match first-target open execution")
    for index, bar in enumerate(bars[1:], start=1):
        if snapshots[index] != _expected_snapshot(
            portfolio_id=portfolio_id,
            bar=bar,
            cash=cash,
            shares=shares,
            open_cost_basis=open_cost_basis,
            realized=realized,
        ):
            raise ValueError("Buy-and-hold benchmark state transition mismatch")


def _normalized_snapshots(
    snapshots: Sequence[PortfolioSnapshot],
) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            item.date,
            item.cash,
            item.shares,
            item.market_price,
            item.position_value,
            item.total_equity,
            item.realized_profit_loss,
            item.unrealized_profit_loss,
        )
        for item in snapshots
    )


def _normalized_trades(trades: Sequence[TradeRecord]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            item.signal_date,
            item.execution_date,
            item.action,
            item.quantity,
            item.reference_price,
            item.execution_price,
            item.fee,
            item.cash_after_trade,
            item.shares_after_trade,
        )
        for item in trades
    )


def _portfolio_replay(
    pack_root: Path,
    registry: Mapping[str, str],
    *,
    holdout_alignment: FairCommonDateAlignment,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model_rows: list[dict[str, Any]] = []
    benchmark_candidates: dict[str, list[dict[str, Any]]] = {
        benchmark_id: [] for benchmark_id in BENCHMARKS
    }
    expected_signature: dict[str, Any] | None = None
    all_portfolio_ids: list[str] = []

    common_feature_dates = [key[1] for key in holdout_alignment.common_keys]
    common_target_dates = [key[2] for key in holdout_alignment.common_keys]
    expected_evaluation_dates = {
        "feature_start": min(common_feature_dates),
        "feature_end": max(common_feature_dates),
        "target_start": min(common_target_dates),
        "target_end": max(common_target_dates),
    }
    market_bars = _portfolio_market_sessions(pack_root, holdout_alignment)
    common_keys = tuple(
        (str(ticker), str(feature_date), str(target_date))
        for ticker, feature_date, target_date in holdout_alignment.common_keys
    )

    for variant_id in MODEL_VARIANTS:
        base = pack_root / "evidence" / "portfolios" / variant_id
        config = _load_experiment_config(base / "config.json")
        config_payload = config.to_dict()
        if config.model_id != registry[variant_id]:
            raise ValueError(f"Portfolio model identity mismatch for {variant_id}")
        signature = _portfolio_assumption_signature(config_payload)
        _validate_expected_portfolio_assumptions(signature)
        if signature["evaluation_dates"] != expected_evaluation_dates:
            raise ValueError(f"Portfolio evaluation dates mismatch for {variant_id}")
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError("All six model portfolios must use identical assumptions")

        probabilities = _raw_holdout_probabilities(
            pack_root,
            variant_id=variant_id,
            registered_model_id=registry[variant_id],
            common_keys=common_keys,
        )
        signals = _load_signals(base / "signals.csv")
        strategy_snapshots = _load_portfolio_snapshots(
            base / "portfolio_snapshots.csv"
        )
        strategy_trades = _load_trades(base / "trades.csv")
        cash_snapshots = _load_portfolio_snapshots(
            base / "benchmarks" / "cash_only_snapshots.csv"
        )
        buy_hold_snapshots = _load_portfolio_snapshots(
            base / "benchmarks" / "buy_and_hold_snapshots.csv"
        )
        buy_hold_trades = _load_trades(
            base / "benchmarks" / "buy_and_hold_trades.csv"
        )
        _validate_ledger_identity(
            strategy_snapshots, strategy_trades, portfolio_id=config.portfolio_id
        )
        _validate_ledger_identity(
            cash_snapshots,
            (),
            portfolio_id=config.benchmark_portfolio_ids.cash_only,
        )
        _validate_ledger_identity(
            buy_hold_snapshots,
            buy_hold_trades,
            portfolio_id=config.benchmark_portfolio_ids.buy_and_hold,
        )
        all_portfolio_ids.extend(
            (
                config.portfolio_id,
                config.benchmark_portfolio_ids.cash_only,
                config.benchmark_portfolio_ids.buy_and_hold,
            )
        )

        _replay_strategy_ledger(
            variant_id=variant_id,
            config=config,
            bars=market_bars,
            probabilities=probabilities,
            signals=signals,
            snapshots=strategy_snapshots,
            trades=strategy_trades,
        )
        _replay_cash_ledger(
            config=config, bars=market_bars, snapshots=cash_snapshots
        )
        _replay_buy_and_hold_ledger(
            config=config,
            bars=market_bars,
            snapshots=buy_hold_snapshots,
            trades=buy_hold_trades,
        )

        metric_options = {
            "annualization_factor": config.metric_assumptions.annualization_factor,
            "risk_free_rate_annual": float(
                config.metric_assumptions.risk_free_rate_annual
            ),
        }
        strategy_metrics = calculate_portfolio_metrics(
            strategy_snapshots, strategy_trades, **metric_options
        )
        cash_metrics = calculate_portfolio_metrics(cash_snapshots, (), **metric_options)
        buy_hold_metrics = calculate_portfolio_metrics(
            buy_hold_snapshots, buy_hold_trades, **metric_options
        )
        model_rows.append(
            to_jsonable(
                {
                    "competitor_id": variant_id,
                    "competitor_type": "model",
                    "registered_model_id": registry[variant_id],
                    "experiment_id": config.experiment_id,
                    "portfolio_id": config.portfolio_id,
                    **_metric_without_daily_returns(strategy_metrics),
                    "excess_return_vs_cash_only": (
                        strategy_metrics["total_return"] - cash_metrics["total_return"]
                    ),
                    "excess_return_vs_buy_and_hold": (
                        strategy_metrics["total_return"]
                        - buy_hold_metrics["total_return"]
                    ),
                }
            )
        )
        config_path = f"evidence/portfolios/{variant_id}/config.json"
        for benchmark_id, portfolio_id, metrics, snapshots, trades, relative in (
            (
                "cash-only",
                config.benchmark_portfolio_ids.cash_only,
                cash_metrics,
                cash_snapshots,
                (),
                "benchmarks/cash_only_snapshots.csv",
            ),
            (
                "buy-and-hold",
                config.benchmark_portfolio_ids.buy_and_hold,
                buy_hold_metrics,
                buy_hold_snapshots,
                buy_hold_trades,
                "benchmarks/buy_and_hold_snapshots.csv",
            ),
        ):
            benchmark_candidates[benchmark_id].append(
                {
                    "metrics": to_jsonable(_metric_without_daily_returns(metrics)),
                    "normalized_snapshots": _normalized_snapshots(snapshots),
                    "normalized_trades": _normalized_trades(trades),
                    "provenance": {
                        "variant_id": variant_id,
                        "registered_model_id": registry[variant_id],
                        "experiment_id": config.experiment_id,
                        "portfolio_id": portfolio_id,
                        "config_path": config_path,
                        "snapshots_path": f"evidence/portfolios/{variant_id}/{relative}",
                        "trades_path": (
                            None
                            if benchmark_id == "cash-only"
                            else (
                                f"evidence/portfolios/{variant_id}/benchmarks/"
                                "buy_and_hold_trades.csv"
                            )
                        ),
                    },
                }
            )

    if len(all_portfolio_ids) != 18 or len(set(all_portfolio_ids)) != 18:
        raise ValueError("Every model and benchmark must retain an independent portfolio ID")
    assert expected_signature is not None

    benchmark_rows: list[dict[str, Any]] = []
    for benchmark_id in BENCHMARKS:
        selected = benchmark_candidates[benchmark_id]
        if len(selected) != len(MODEL_VARIANTS):
            raise ValueError(f"Exactly six {benchmark_id} ledgers are required")
        reference = selected[0]
        if any(
            candidate["metrics"] != reference["metrics"]
            or candidate["normalized_snapshots"]
            != reference["normalized_snapshots"]
            or candidate["normalized_trades"] != reference["normalized_trades"]
            for candidate in selected[1:]
        ):
            raise ValueError(
                f"Independent {benchmark_id} ledgers disagree row by row"
            )
        benchmark_rows.append(
            {
                "benchmark_id": benchmark_id,
                "competitor_type": "portfolio_benchmark",
                **reference["metrics"],
                "ledger_provenance": [
                    candidate["provenance"] for candidate in selected
                ],
            }
        )

    return model_rows, to_jsonable(benchmark_rows), expected_signature


def _rank_portfolios(
    rows: Sequence[Mapping[str, Any]], *, primary: str
) -> list[dict[str, Any]]:
    if primary == "total_return":
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row["total_return"]),
                float("inf")
                if row["sharpe_ratio"] is None
                else -float(row["sharpe_ratio"]),
                float(row["maximum_drawdown"]),
                str(row["competitor_id"]),
            ),
        )
    elif primary == "maximum_drawdown":
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["maximum_drawdown"]),
                -float(row["total_return"]),
                str(row["competitor_id"]),
            ),
        )
    elif primary == "fees_paid":
        ordered = sorted(
            rows,
            key=lambda row: (
                Decimal(str(row["fees_paid"])),
                int(row["number_of_trades"]),
                str(row["competitor_id"]),
            ),
        )
    else:
        raise KeyError(primary)
    tie_break_fields = {
        "total_return": ("sharpe_ratio", "maximum_drawdown"),
        "maximum_drawdown": ("total_return",),
        "fees_paid": ("number_of_trades",),
    }[primary]
    return [
        {
            "rank": index,
            "competitor_id": row["competitor_id"],
            "registered_model_id": row["registered_model_id"],
            primary: row[primary],
            **{field: row[field] for field in tie_break_fields},
        }
        for index, row in enumerate(ordered, start=1)
    ]


def _selected_prediction_outcome(
    row: Mapping[str, Any], *, criterion: str
) -> dict[str, Any]:
    return {
        "variant_id": row["competitor_id"],
        "registered_model_id": row["registered_model_id"],
        "criterion": criterion,
        "balanced_accuracy": row["balanced_accuracy"],
        "f1": row["f1"],
        "accuracy": row["accuracy"],
    }


def _selected_portfolio_outcome(
    row: Mapping[str, Any], *, criterion: str
) -> dict[str, Any]:
    return {
        "variant_id": row["competitor_id"],
        "registered_model_id": row["registered_model_id"],
        "portfolio_id": row["portfolio_id"],
        "criterion": criterion,
        "ending_equity": row["ending_equity"],
        "total_return": row["total_return"],
        "maximum_drawdown": row["maximum_drawdown"],
        "fees_paid": row["fees_paid"],
    }


def _replay_pack_evidence(pack_root: Path, *, comparison_id: str) -> ReplayResults:
    registry = _model_registry(pack_root)
    baseline_split = _read_json(
        pack_root / "evidence/splits/baseline_split_manifest.json",
        field="Copied baseline split manifest",
    )
    lstm_split = _read_json(
        pack_root / "evidence/splits/lstm_split_manifest.json",
        field="Copied LSTM split manifest",
    )
    train_target_dates = _target_dates(baseline_split, "train")
    validation_target_dates = _target_dates(baseline_split, "validation")
    holdout_target_dates = _target_dates(baseline_split, "test")
    lstm_train_target_dates = _target_dates(lstm_split, "train")
    lstm_validation_target_dates = _target_dates(lstm_split, "validation")
    lstm_holdout_target_dates = _target_dates(lstm_split, "test")
    if lstm_validation_target_dates != validation_target_dates:
        raise ValueError("Baseline and LSTM validation target dates do not match")
    if lstm_holdout_target_dates != holdout_target_dates:
        raise ValueError("Baseline and LSTM shared project holdout dates do not match")

    frames = _load_all_prediction_frames(
        pack_root,
        registry,
        baseline_targets={
            "train": train_target_dates,
            "validation": validation_target_dates,
            "test": holdout_target_dates,
        },
        lstm_targets={
            "train": lstm_train_target_dates,
            "validation": lstm_validation_target_dates,
            "test": lstm_holdout_target_dates,
        },
    )
    validation_alignment = _scope_alignment(frames, validation_target_dates)
    holdout_alignment = _scope_alignment(frames, holdout_target_dates)
    _require_exact_scope_alignment(
        validation_alignment,
        target_dates=validation_target_dates,
        scope="Validation",
    )
    _require_exact_scope_alignment(
        holdout_alignment,
        target_dates=holdout_target_dates,
        scope=HOLDOUT_NAME.capitalize(),
    )
    date_alignment = {
        "schema_version": SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "alignment_key": ["ticker", "feature_date", "target_date"],
        "validation": _alignment_payload(
            validation_alignment, terminology="validation"
        ),
        "shared_project_holdout": {
            **_alignment_payload(
                holdout_alignment, terminology=HOLDOUT_NAME
            ),
            "previously_exposed_through": list(BASELINE_VARIANTS),
        },
    }

    validation_model_metrics: list[dict[str, Any]] = []
    holdout_model_metrics: list[dict[str, Any]] = []
    for variant_id in MODEL_VARIANTS:
        validation_model_metrics.append(
            {
                **_prediction_metric_record(
                    competitor_id=variant_id,
                    competitor_type="model",
                    registered_model_id=registry[variant_id],
                    evaluation_scope="validation",
                    frame=validation_alignment.frame_for(variant_id),
                ),
                **MODEL_DESCRIPTIONS[variant_id],
            }
        )
        holdout_model_metrics.append(
            {
                **_prediction_metric_record(
                    competitor_id=variant_id,
                    competitor_type="model",
                    registered_model_id=registry[variant_id],
                    evaluation_scope="shared_project_holdout",
                    frame=holdout_alignment.frame_for(variant_id),
                ),
                **MODEL_DESCRIPTIONS[variant_id],
            }
        )

    context = frames["M-001"]
    training = context.loc[
        context["target_date"].astype(str).isin(set(train_target_dates))
    ].copy()
    if len(training) != len(train_target_dates):
        raise ValueError("Training prediction context does not match the split manifest")
    training_counts = training["actual_class"].astype(int).value_counts().to_dict()
    majority_count = max(training_counts.values())
    majority_class = min(
        int(label) for label, count in training_counts.items() if count == majority_count
    )
    baseline_prediction_frames: list[pd.DataFrame] = []
    baseline_metrics: list[dict[str, Any]] = []
    for scope, alignment in (
        ("validation", validation_alignment),
        ("shared_project_holdout", holdout_alignment),
    ):
        scored_rows = alignment.frame_for("M-001")
        for baseline_id in (MAJORITY_BASELINE, PREVIOUS_DIRECTION_BASELINE):
            baseline_frame = _reference_baseline_frame(
                baseline_id=baseline_id,
                evaluation_scope=scope,
                scored_rows=scored_rows,
                context_rows=context,
                majority_class=majority_class,
            )
            baseline_prediction_frames.append(baseline_frame)
            baseline_metrics.append(
                _prediction_metric_record(
                    competitor_id=baseline_id,
                    competitor_type="reference_baseline",
                    registered_model_id=None,
                    evaluation_scope=scope,
                    frame=baseline_frame.loc[:, PREDICTION_COLUMNS],
                )
            )
    baseline_predictions = pd.concat(
        baseline_prediction_frames, ignore_index=True
    ).sort_values(
        ["evaluation_scope", "baseline_id", "ticker", "target_date"],
        kind="mergesort",
    ).reset_index(drop=True)

    validation_baseline_metrics = [
        row for row in baseline_metrics if row["evaluation_scope"] == "validation"
    ]
    holdout_baseline_metrics = [
        row
        for row in baseline_metrics
        if row["evaluation_scope"] == "shared_project_holdout"
    ]
    shared_holdout_metrics = [*holdout_model_metrics, *holdout_baseline_metrics]

    model_portfolios, benchmark_portfolios, portfolio_assumptions = _portfolio_replay(
        pack_root, registry, holdout_alignment=holdout_alignment
    )
    validation_ranking = _rank_prediction_rows(
        validation_model_metrics, primary="balanced_accuracy"
    )
    lstm_validation_ranking = _rank_prediction_rows(
        [
            row
            for row in validation_model_metrics
            if row["competitor_id"] in LSTM_VARIANTS
        ],
        primary="balanced_accuracy",
    )
    shared_ba_ranking = _rank_prediction_rows(
        holdout_model_metrics, primary="balanced_accuracy"
    )
    shared_f1_ranking = _rank_prediction_rows(holdout_model_metrics, primary="f1")
    portfolio_return_ranking = _rank_portfolios(
        model_portfolios, primary="total_return"
    )
    drawdown_ranking = _rank_portfolios(
        model_portfolios, primary="maximum_drawdown"
    )
    fee_ranking = _rank_portfolios(model_portfolios, primary="fees_paid")
    rankings = {
        "schema_version": SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "validation_model_selection": validation_ranking,
        "lstm_validation_preference": lstm_validation_ranking,
        "shared_project_holdout_balanced_accuracy": shared_ba_ranking,
        "shared_project_holdout_f1": shared_f1_ranking,
        "simulated_model_portfolio_return": portfolio_return_ranking,
        "simulated_model_portfolio_drawdown": drawdown_ranking,
        "simulated_model_portfolio_fees": fee_ranking,
    }

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "validation_model_metrics": validation_model_metrics,
        "validation_reference_baseline_metrics": validation_baseline_metrics,
        "shared_project_holdout_prediction_metrics": shared_holdout_metrics,
        "model_portfolio_metrics": model_portfolios,
        "benchmark_portfolio_metrics": benchmark_portfolios,
        "reference_baseline_training": {
            "training_row_count": len(training),
            "class_counts": {
                str(label): int(training_counts.get(label, 0)) for label in (0, 1)
            },
            "majority_class": majority_class,
        },
    }

    validation_winner = next(
        row
        for row in validation_model_metrics
        if row["competitor_id"] == validation_ranking[0]["competitor_id"]
    )
    lstm_validation_winner = next(
        row
        for row in validation_model_metrics
        if row["competitor_id"] == lstm_validation_ranking[0]["competitor_id"]
    )
    balanced_accuracy_winner = next(
        row
        for row in holdout_model_metrics
        if row["competitor_id"] == shared_ba_ranking[0]["competitor_id"]
    )
    f1_winner = next(
        row
        for row in holdout_model_metrics
        if row["competitor_id"] == shared_f1_ranking[0]["competitor_id"]
    )
    best_portfolio = next(
        row
        for row in model_portfolios
        if row["competitor_id"] == portfolio_return_ranking[0]["competitor_id"]
    )
    lowest_drawdown = next(
        row
        for row in model_portfolios
        if row["competitor_id"] == drawdown_ranking[0]["competitor_id"]
    )
    lowest_fee = next(
        row
        for row in model_portfolios
        if row["competitor_id"] == fee_ranking[0]["competitor_id"]
    )
    cash_rows = [
        row for row in benchmark_portfolios if row["benchmark_id"] == "cash-only"
    ]
    buy_hold_rows = [
        row
        for row in benchmark_portfolios
        if row["benchmark_id"] == "buy-and-hold"
    ]
    cash_result = cash_rows[0]
    buy_hold_result = buy_hold_rows[0]
    any_model_beat = any(
        float(row["total_return"]) > float(buy_hold_result["total_return"])
        for row in model_portfolios
    )

    outcomes = {
        "validation_selected_model": {
            "scope": "all_six_models",
            **_selected_prediction_outcome(
                validation_winner,
                criterion=(
                    "validation balanced accuracy, then F1, accuracy, and model ID"
                ),
            ),
        },
        "lstm_validation_preferred_model": {
            "scope": "M-005_and_M-006_only",
            **_selected_prediction_outcome(
                lstm_validation_winner,
                criterion=(
                    "validation balanced accuracy, then F1, accuracy, and model ID"
                ),
            ),
        },
        "best_shared_holdout_balanced_accuracy": {
            "scope": HOLDOUT_NAME,
            **_selected_prediction_outcome(
                balanced_accuracy_winner,
                criterion="highest shared project holdout balanced accuracy",
            ),
        },
        "best_shared_holdout_f1": {
            "scope": HOLDOUT_NAME,
            **_selected_prediction_outcome(
                f1_winner, criterion="highest shared project holdout F1"
            ),
        },
        "best_simulated_model_portfolio": _selected_portfolio_outcome(
            best_portfolio, criterion="highest simulated total return"
        ),
        "lowest_drawdown_model_portfolio": _selected_portfolio_outcome(
            lowest_drawdown, criterion="lowest simulated maximum drawdown"
        ),
        "lowest_fee_model_portfolio": _selected_portfolio_outcome(
            lowest_fee, criterion="lowest simulated fees paid"
        ),
        "cash_only_benchmark": {
            key: cash_result[key]
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
            key: buy_hold_result[key]
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
            "value": any_model_beat,
            "comparison_metric": "simulated total return",
            "best_model_variant_id": best_portfolio["competitor_id"],
            "best_model_minus_buy_and_hold": (
                float(best_portfolio["total_return"])
                - float(buy_hold_result["total_return"])
            ),
        },
    }

    contract = {
        "schema_version": SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "symbol": "OGDC",
        "source_frequency": "daily",
        "models": list(MODEL_VARIANTS),
        "reference_baselines": [MAJORITY_BASELINE, PREVIOUS_DIRECTION_BASELINE],
        "portfolio_benchmarks": list(BENCHMARKS),
        "target_definition": {
            "name": "target_next_day_up",
            "positive_class_condition": "close[target_date] > close[feature_date]",
            "prediction_horizon": {"value": 1, "unit": "trading_session"},
        },
        "alignment_key": ["ticker", "feature_date", "target_date"],
        "evaluation_scopes": {
            "validation": {"target_dates": list(validation_target_dates)},
            "shared_project_holdout": {
                "term": HOLDOUT_NAME,
                "target_dates": list(holdout_target_dates),
                "previously_exposed_through": list(BASELINE_VARIANTS),
            },
        },
        "selection_policy": {
            "primary": "balanced_accuracy",
            "tie_breakers": ["f1", "accuracy", "model_id_ascending"],
            "test_metrics_used_for_selection": False,
        },
        "reference_baseline_policy": {
            "majority_class": "most frequent training actual_class; ascending class tie-break",
            "previous_direction": (
                "actual_class from the prior persisted row whose target_date equals "
                "the scored feature_date"
            ),
        },
        "portfolio_assumptions": portfolio_assumptions,
        "metric_recomputation": {
            "prediction_source": "persisted prediction rows",
            "portfolio_source": (
                "persisted signals, source-bound market observations, snapshots, and trades"
            ),
            "portfolio_semantic_replay": True,
            "models_retrained": False,
            "backtests_rerun": False,
        },
        "presentation": {
            "format": "tabular_report_only",
            "plots_included": False,
        },
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    final_summary = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "comparison_id": comparison_id,
        "status": "PASS",
        "symbol": "OGDC",
        "path_base": "comparison_root",
        "holdout_context": {
            "term": HOLDOUT_NAME,
            "common_count": len(holdout_alignment.common_keys),
            "target_start": min(holdout_target_dates),
            "target_end": max(holdout_target_dates),
            "previously_exposed_through": list(BASELINE_VARIANTS),
        },
        "outcomes_are_not_combined": True,
        "reported_outcomes": outcomes,
        "artifact_paths": {
            "source_roots": "source_roots.json",
            "comparison_contract": "comparison_contract.json",
            "date_alignment": "date_alignment.json",
            "metrics": "metrics.json",
            "rankings": "rankings.json",
            "reference_baseline_predictions": (
                "derived/reference_baseline_predictions.csv"
            ),
            "human_report": "report.md",
        },
        "limitations": [
            "One stock and one historical period do not establish general superiority.",
            "The shared project holdout and baseline results were exposed previously.",
            "Corporate-action adjustment is unknown and dividends are not modeled.",
            "Results depend on one fixed policy, execution model, fee, and split regime.",
            "This price-only comparison does not integrate or evaluate sentiment features.",
            "The portable pack is tabular/report-only and does not include plots.",
            "Simulated results do not predict future performance.",
        ],
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    return ReplayResults(
        model_registry=registry,
        contract=to_jsonable(contract),
        date_alignment=to_jsonable(date_alignment),
        metrics=to_jsonable(metrics),
        rankings=to_jsonable(rankings),
        final_summary=to_jsonable(final_summary),
        baseline_predictions=baseline_predictions,
    )


def _format_number(value: object, *, money: bool = False) -> str:
    if value is None:
        return "n/a"
    if money:
        return f"PKR {float(value):,.2f}"
    return f"{float(value):.6f}"


def _render_report(
    replay: ReplayResults,
    *,
    source_roots: Mapping[str, Any],
) -> str:
    """Render the deterministic human-readable view of the verified results."""

    metrics = replay.metrics
    outcomes = replay.final_summary["reported_outcomes"]
    alignment = replay.date_alignment["shared_project_holdout"]
    source_lines = [
        (
            f"- `{source['role']}`: `{source['run_id']}` "
            f"(`{source['integrity_root_sha256']}`)"
        )
        for source in source_roots["sources"]
    ]

    validation_rows = metrics["validation_model_metrics"]
    holdout_rows = metrics["shared_project_holdout_prediction_metrics"]
    portfolio_rows = metrics["model_portfolio_metrics"]
    benchmark_rows = metrics["benchmark_portfolio_metrics"]

    prediction_table = [
        "| Competitor | Type | Balanced accuracy | F1 | Accuracy |",
        "|---|---|---:|---:|---:|",
    ]
    for row in holdout_rows:
        prediction_table.append(
            "| {competitor_id} | {competitor_type} | {balanced_accuracy} | {f1} | "
            "{accuracy} |".format(
                competitor_id=row["competitor_id"],
                competitor_type=row["competitor_type"],
                balanced_accuracy=_format_number(row["balanced_accuracy"]),
                f1=_format_number(row["f1"]),
                accuracy=_format_number(row["accuracy"]),
            )
        )

    validation_table = [
        "| Model | Balanced accuracy | F1 | Accuracy |",
        "|---|---:|---:|---:|",
    ]
    for row in validation_rows:
        validation_table.append(
            "| {competitor_id} | {balanced_accuracy} | {f1} | {accuracy} |".format(
                competitor_id=row["competitor_id"],
                balanced_accuracy=_format_number(row["balanced_accuracy"]),
                f1=_format_number(row["f1"]),
                accuracy=_format_number(row["accuracy"]),
            )
        )

    portfolio_table = [
        "| Model | Ending equity | Total return | Maximum drawdown | Fees paid |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in portfolio_rows:
        portfolio_table.append(
            "| {competitor_id} | {ending_equity} | {total_return} | "
            "{maximum_drawdown} | {fees_paid} |".format(
                competitor_id=row["competitor_id"],
                ending_equity=_format_number(row["ending_equity"], money=True),
                total_return=_format_number(row["total_return"]),
                maximum_drawdown=_format_number(row["maximum_drawdown"]),
                fees_paid=_format_number(row["fees_paid"], money=True),
            )
        )

    benchmark_table = [
        "| Benchmark | Ending equity | Total return | Maximum drawdown | Fees paid |",
        "|---|---:|---:|---:|---:|",
    ]
    for benchmark_id in BENCHMARKS:
        row = next(
            item for item in benchmark_rows if item["benchmark_id"] == benchmark_id
        )
        benchmark_table.append(
            "| {benchmark_id} | {ending_equity} | {total_return} | "
            "{maximum_drawdown} | {fees_paid} |".format(
                benchmark_id=benchmark_id,
                ending_equity=_format_number(row["ending_equity"], money=True),
                total_return=_format_number(row["total_return"]),
                maximum_drawdown=_format_number(row["maximum_drawdown"]),
                fees_paid=_format_number(row["fees_paid"], money=True),
            )
        )

    validation_selected = outcomes["validation_selected_model"]
    best_ba = outcomes["best_shared_holdout_balanced_accuracy"]
    best_f1 = outcomes["best_shared_holdout_f1"]
    best_portfolio = outcomes["best_simulated_model_portfolio"]
    lowest_drawdown = outcomes["lowest_drawdown_model_portfolio"]
    lowest_fee = outcomes["lowest_fee_model_portfolio"]
    buy_hold = outcomes["buy_and_hold_result"]
    beat_buy_hold = outcomes["any_model_beat_buy_and_hold"]

    sections = [
        f"# Unified price-only model comparison: {replay.final_summary['comparison_id']}",
        "",
        "Status: **PASS**",
        "",
        ACADEMIC_DISCLAIMER,
        "",
        "This pack recomputes prediction and simulated-portfolio metrics from "
        "persisted evidence. It does not retrain a model or rerun a backtest. "
        "Criteria remain separate; no cross-criterion score is produced.",
        "",
        "## Authoritative sources",
        "",
        *source_lines,
        "",
        "## Comparison scope",
        "",
        f"The {HOLDOUT_NAME} contains {alignment['common_count']} common target dates "
        f"from {alignment['target_start']} through {alignment['target_end']}. "
        "M-001 through M-004 had prior exposure to this period, which is why it is "
        f"described only as the {HOLDOUT_NAME}.",
        "",
        "## Separately reported outcomes",
        "",
        f"- Validation-selected model: **{validation_selected['variant_id']}** "
        f"(balanced accuracy {_format_number(validation_selected['balanced_accuracy'])}).",
        f"- Best shared project holdout balanced accuracy: **{best_ba['variant_id']}** "
        f"({_format_number(best_ba['balanced_accuracy'])}).",
        f"- Best shared project holdout F1: **{best_f1['variant_id']}** "
        f"({_format_number(best_f1['f1'])}).",
        f"- Best simulated model portfolio: **{best_portfolio['variant_id']}** "
        f"({_format_number(best_portfolio['ending_equity'], money=True)}; return "
        f"{_format_number(best_portfolio['total_return'])}).",
        f"- Lowest-drawdown model portfolio: **{lowest_drawdown['variant_id']}** "
        f"({_format_number(lowest_drawdown['maximum_drawdown'])}).",
        f"- Lowest-fee model portfolio: **{lowest_fee['variant_id']}** "
        f"({_format_number(lowest_fee['fees_paid'], money=True)}).",
        f"- Buy-and-hold result: {_format_number(buy_hold['ending_equity'], money=True)} "
        f"(return {_format_number(buy_hold['total_return'])}).",
        f"- Any model beat buy-and-hold: **{'yes' if beat_buy_hold['value'] else 'no'}**.",
        "",
        "## Validation metrics",
        "",
        *validation_table,
        "",
        f"## {HOLDOUT_NAME.title()} prediction metrics",
        "",
        *prediction_table,
        "",
        "## Simulated model portfolios",
        "",
        *portfolio_table,
        "",
        "## Benchmarks",
        "",
        *benchmark_table,
        "",
        "Each model configuration retains separate strategy, cash-only, and "
        "buy-and-hold portfolio identities. Each benchmark row is one logical "
        "result reconciled row by row from six independently identified ledgers; "
        "their relative paths and IDs are recorded in metrics.json.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in replay.final_summary["limitations"]],
        "",
        ACADEMIC_DISCLAIMER,
        "",
    ]
    return "\n".join(sections)


def _write_replayed_outputs(
    stage_dir: Path,
    replay: ReplayResults,
    *,
    source_roots: Mapping[str, Any],
) -> None:
    write_json(replay.contract, stage_dir / "comparison_contract.json")
    write_json(replay.date_alignment, stage_dir / "date_alignment.json")
    write_json(replay.metrics, stage_dir / "metrics.json")
    write_json(replay.rankings, stage_dir / "rankings.json")
    write_json(replay.final_summary, stage_dir / "final_summary.json")
    baseline_path = stage_dir / "derived/reference_baseline_predictions.csv"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    replay.baseline_predictions.to_csv(baseline_path, index=False, lineterminator="\n")
    (stage_dir / "report.md").write_text(
        _render_report(replay, source_roots=source_roots), encoding="utf-8"
    )


def _write_pack_manifest(stage_dir: Path, *, comparison_id: str) -> dict[str, Any]:
    files = _regular_files(stage_dir, exclude={"manifest.json"})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "comparison_id": comparison_id,
        "status": "PASS",
        "inventory": {
            "algorithm": "sha256",
            "root": ".",
            "excludes": ["manifest.json"],
            "files": [
                {"path": relative, "sha256": sha256_file(files[relative])}
                for relative in sorted(files)
            ],
        },
        "source_roots": "source_roots.json",
        "final_summary": "final_summary.json",
        "disclaimer": ACADEMIC_DISCLAIMER,
    }
    write_json(manifest, stage_dir / "manifest.json")
    return manifest


def _expected_evidence_bindings(
    registry: Mapping[str, str],
) -> dict[str, tuple[str, str, str]]:
    return expected_evidence_bindings(
        registry,
        portfolio_files=_PORTFOLIO_FILES,
    )


def _source_checksum_maps(pack_root: Path) -> tuple[dict[str, dict[str, str]], dict[str, int]]:
    training_manifest = _read_json(
        pack_root / "evidence/source_manifests/baseline_training_run_manifest.json",
        field="Copied baseline training manifest",
    )
    training_inventory = dict(
        _parse_sha256_inventory(
            training_manifest.get("artifact_inventory"),
            field="Copied baseline training inventory",
            expected_excludes=("run_manifest.json",),
        )
    )

    vertical_certificate_path = (
        pack_root
        / "evidence/source_manifests/baseline_vertical_manifest_certificate.json"
    )
    vertical_certificate = _read_json(
        vertical_certificate_path, field="Vertical-slice integrity certificate"
    )
    if set(vertical_certificate) != {
        "schema_version",
        "artifact_type",
        "source_role",
        "run_id",
        "source_manifest",
        "projection_policy",
        "raw_manifest_included",
        "source_assertions",
        "artifact_inventory",
        "disclaimer",
    }:
        raise ValueError("Vertical-slice integrity certificate has an unexpected schema")
    source_manifest = vertical_certificate["source_manifest"]
    if not isinstance(source_manifest, Mapping) or set(source_manifest) != {
        "name",
        "sha256",
    }:
        raise ValueError("Vertical-slice source manifest certificate is malformed")
    vertical_manifest_sha256 = str(source_manifest["sha256"])
    if (
        vertical_certificate["schema_version"] != SCHEMA_VERSION
        or vertical_certificate["artifact_type"]
        != "paisa_source_integrity_certificate"
        or vertical_certificate["source_role"] != "baseline_vertical_slice"
        or vertical_certificate["run_id"]
        != AUTHORITATIVE_SOURCE_ROOTS["baseline_vertical_slice"]["run_id"]
        or source_manifest["name"] != "manifest.json"
        or vertical_manifest_sha256
        != AUTHORITATIVE_SOURCE_ROOTS["baseline_vertical_slice"]["manifest_sha256"]
        or vertical_certificate["projection_policy"] != "allowlisted_fields_v1"
        or vertical_certificate["raw_manifest_included"] is not False
        or vertical_certificate["disclaimer"] != ACADEMIC_DISCLAIMER
    ):
        raise ValueError("Vertical-slice integrity certificate identity mismatch")
    vertical_inventory = dict(
        _parse_sha256_inventory(
            vertical_certificate["artifact_inventory"],
            field="Certified vertical-slice inventory",
            expected_excludes=("manifest.json",),
        )
    )

    lstm_run_manifest_path = (
        pack_root / "evidence/source_manifests/lstm_run_manifest.json"
    )
    lstm_run_manifest = _read_json(
        lstm_run_manifest_path, field="Copied LSTM run manifest"
    )
    lstm_root_inventory = dict(
        _parse_sha256_inventory(
            lstm_run_manifest.get("artifact_inventory"),
            field="Copied LSTM root inventory",
            expected_excludes=("run_manifest.json",),
        )
    )
    lstm_inventory_manifest_path = (
        pack_root / "evidence/source_manifests/lstm_artifact_manifest.json"
    )
    lstm_inventory_manifest = _read_json(
        lstm_inventory_manifest_path,
        field="Copied LSTM artifact manifest",
    )
    lstm_nested_inventory = dict(
        _parse_sha256_inventory(
            lstm_inventory_manifest,
            field="Copied LSTM artifact inventory",
            expected_excludes=("artifact_manifest.json", "run_manifest.json"),
        )
    )
    artifact_manifest_sha256 = sha256_file(lstm_inventory_manifest_path)
    if lstm_root_inventory.get("artifact_manifest.json") != artifact_manifest_sha256:
        raise ValueError(
            "Copied LSTM artifact manifest is not bound by its run manifest"
        )
    expected_lstm_nested = {
        relative: checksum
        for relative, checksum in lstm_root_inventory.items()
        if relative != "artifact_manifest.json"
    }
    if lstm_nested_inventory != expected_lstm_nested:
        raise ValueError("Copied LSTM nested and root inventories disagree")
    maps = {
        "baseline_training": {
            "run_manifest.json": sha256_file(
                pack_root
                / "evidence/source_manifests/baseline_training_run_manifest.json"
            ),
            **training_inventory,
        },
        "baseline_vertical_slice": {
            "manifest.json": vertical_manifest_sha256,
            **vertical_inventory,
        },
        "lstm": {
            "run_manifest.json": sha256_file(lstm_run_manifest_path),
            **lstm_root_inventory,
        },
    }
    counts = {
        "baseline_training": len(training_inventory),
        "baseline_vertical_slice": len(vertical_inventory),
        "lstm": len(lstm_root_inventory),
    }
    return maps, counts


def _validate_source_roots_payload(
    pack_root: Path,
    *,
    comparison_id: str,
    inventory: Mapping[str, str],
) -> dict[str, Any]:
    payload = _read_json(pack_root / "source_roots.json", field="Source roots")
    if set(payload) != {
        "schema_version",
        "comparison_id",
        "sources",
        "relationships",
        "evidence_files",
        "source_revalidation",
    }:
        raise ValueError("source_roots.json has an unexpected schema")
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["comparison_id"] != comparison_id
    ):
        raise ValueError("source_roots.json identity mismatch")
    if payload["relationships"] != {
        "vertical_slice_uses_training_run": True,
        "lstm_uses_training_run": True,
        "lstm_uses_vertical_slice": True,
    }:
        raise ValueError("Source relationship assertions are incomplete")
    if payload["source_revalidation"] != {
        "validated_before_generation": True,
        "revalidated_before_publication": True,
    }:
        raise ValueError("Source revalidation assertions are incomplete")

    raw_sources = payload["sources"]
    if not isinstance(raw_sources, list) or len(raw_sources) != len(_SOURCE_ROLES):
        raise ValueError("Exactly three source roots are required")
    source_maps, payload_counts = _source_checksum_maps(pack_root)
    expected_root_paths = {
        "baseline_training": (
            "run_manifest.json",
            "evidence/source_manifests/baseline_training_run_manifest.json",
            "model_training_run",
            "byte_copy",
        ),
        "baseline_vertical_slice": (
            "manifest.json",
            "evidence/source_manifests/baseline_vertical_manifest_certificate.json",
            "vertical_slice",
            "deterministic_redacted_projection",
        ),
        "lstm": (
            "run_manifest.json",
            "evidence/source_manifests/lstm_run_manifest.json",
            "lstm_price_only_experiment",
            "byte_copy",
        ),
    }
    if [source.get("role") for source in raw_sources if isinstance(source, Mapping)] != list(
        _SOURCE_ROLES
    ):
        raise ValueError("Source roots must use the canonical role order")
    for source in raw_sources:
        if not isinstance(source, Mapping) or set(source) != {
            "role",
            "artifact_type",
            "run_id",
            "integrity_root_name",
            "integrity_root_sha256",
            "payload_file_count",
            "integrity_root_pack_path",
            "integrity_root_pack_sha256",
            "integrity_root_derivation",
        }:
            raise ValueError("Each source root has an unexpected schema")
        role = str(source["role"])
        expected_identity = AUTHORITATIVE_SOURCE_ROOTS[role]
        root_name, pack_path, artifact_type, derivation = expected_root_paths[role]
        if (
            source["run_id"] != expected_identity["run_id"]
            or source["integrity_root_sha256"]
            != expected_identity["manifest_sha256"]
            or source["integrity_root_name"] != root_name
            or source["integrity_root_pack_path"] != pack_path
            or source["artifact_type"] != artifact_type
            or source["payload_file_count"] != payload_counts[role]
            or source["integrity_root_derivation"] != derivation
        ):
            raise ValueError(f"Authoritative source binding mismatch for {role}")
        pack_sha256 = str(source["integrity_root_pack_sha256"])
        if (
            not _SHA256.fullmatch(pack_sha256)
            or inventory.get(pack_path) != pack_sha256
            or (derivation == "byte_copy" and pack_sha256 != source["integrity_root_sha256"])
            or (
                role == "baseline_vertical_slice"
                and pack_sha256
                != expected_identity.get("portable_certificate_sha256")
            )
        ):
            raise ValueError(f"Copied integrity root mismatch for {role}")

    registry = _model_registry(pack_root)
    vertical_certificate = _read_json(
        pack_root
        / "evidence/source_manifests/baseline_vertical_manifest_certificate.json",
        field="Vertical-slice integrity certificate",
    )
    if vertical_certificate.get("source_assertions") != {
        "schema_version": "1.0",
        "status": "PASS",
        "symbol": "OGDC",
        "model_training_run_id": AUTHORITATIVE_SOURCE_ROOTS["baseline_training"][
            "run_id"
        ],
        "registered_model_ids": {
            variant_id: registry[variant_id] for variant_id in BASELINE_VARIANTS
        },
        "disclaimer": ACADEMIC_DISCLAIMER,
    }:
        raise ValueError("Vertical-slice certificate assertions are incomplete")
    expected_bindings = _expected_evidence_bindings(registry)
    raw_evidence = payload["evidence_files"]
    if not isinstance(raw_evidence, list):
        raise ValueError("evidence_files must be an array")
    observed: dict[str, tuple[str, str, str]] = {}
    previous_pack_path = ""
    for index, entry in enumerate(raw_evidence):
        if not isinstance(entry, Mapping) or set(entry) != {
            "source_role",
            "source_path",
            "pack_path",
            "sha256",
            "purpose",
            "derivation",
        }:
            raise ValueError(f"evidence_files[{index}] has an unexpected schema")
        role = str(entry["source_role"])
        if role not in _SOURCE_ROLES:
            raise ValueError(f"evidence_files[{index}] has an unknown source role")
        source_path = _safe_relative(
            entry["source_path"], field=f"evidence_files[{index}].source_path"
        )
        pack_path = _safe_relative(
            entry["pack_path"], field=f"evidence_files[{index}].pack_path"
        )
        if not pack_path.startswith("evidence/") or pack_path <= previous_pack_path:
            raise ValueError("Evidence pack paths must be sorted, unique, and under evidence/")
        previous_pack_path = pack_path
        checksum = str(entry["sha256"])
        if (
            not _SHA256.fullmatch(checksum)
            or entry["derivation"] != "byte_copy"
            or inventory.get(pack_path) != checksum
            or source_maps[role].get(source_path) != checksum
        ):
            raise ValueError(f"Evidence checksum or derivation mismatch: {pack_path}")
        observed[pack_path] = (role, source_path, str(entry["purpose"]))
    if observed != expected_bindings:
        missing = sorted(set(expected_bindings) - set(observed))
        unexpected = sorted(set(observed) - set(expected_bindings))
        raise ValueError(
            "Portable evidence binding set mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    actual_evidence = {
        path for path in inventory if path.startswith("evidence/")
    }
    expected_evidence = {
        *expected_bindings,
        "evidence/source_manifests/baseline_vertical_manifest_certificate.json",
    }
    if actual_evidence != expected_evidence:
        raise ValueError("Pack evidence files must exactly match source_roots.json")
    return payload


def _assert_no_prohibited_claims(pack_root: Path) -> None:
    forbidden = ("globally untouched", "universal winner", "universal_winner")
    for relative, path in _regular_files(pack_root).items():
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Portable comparison member is not UTF-8 text: {relative}"
            ) from exc
        text = raw_text.lower()
        for phrase in forbidden:
            if phrase in text:
                raise ValueError(f"Comparison pack contains prohibited claim: {phrase}")
        if contains_sensitive_text(raw_text):
            raise ValueError(
                f"Portable comparison member contains host-path or credential data: {relative}"
            )


def _compare_replayed_outputs(
    pack_root: Path,
    *,
    replay: ReplayResults,
    source_roots: Mapping[str, Any],
) -> None:
    expected_json = {
        "comparison_contract.json": replay.contract,
        "date_alignment.json": replay.date_alignment,
        "metrics.json": replay.metrics,
        "rankings.json": replay.rankings,
        "final_summary.json": replay.final_summary,
    }
    for relative, expected in expected_json.items():
        actual = _read_json(pack_root / relative, field=relative)
        if actual != expected:
            raise ValueError(f"Persisted {relative} does not match replayed evidence")
    expected_baselines = replay.baseline_predictions.to_csv(
        index=False, lineterminator="\n"
    )
    actual_baselines = (
        pack_root / "derived/reference_baseline_predictions.csv"
    ).read_text(encoding="utf-8")
    if actual_baselines != expected_baselines:
        raise ValueError("Reference baseline predictions do not match replayed evidence")
    expected_report = _render_report(replay, source_roots=source_roots)
    if (pack_root / "report.md").read_text(encoding="utf-8") != expected_report:
        raise ValueError("Human report does not match replayed evidence")


def _verify_pack_directory(pack_root: Path) -> dict[str, Any]:
    if pack_root.is_symlink() or not pack_root.is_dir():
        raise ValueError("Comparison pack must be a real directory")
    manifest = _read_json(pack_root / "manifest.json", field="Comparison manifest")
    if set(manifest) != {
        "schema_version",
        "artifact_type",
        "comparison_id",
        "status",
        "inventory",
        "source_roots",
        "final_summary",
        "disclaimer",
    }:
        raise ValueError("Comparison manifest has an unexpected schema")
    comparison_id = _safe_id(manifest["comparison_id"], field="comparison_id")
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["artifact_type"] != ARTIFACT_TYPE
        or manifest["status"] != "PASS"
        or manifest["source_roots"] != "source_roots.json"
        or manifest["final_summary"] != "final_summary.json"
        or manifest["disclaimer"] != ACADEMIC_DISCLAIMER
        or pack_root.name != comparison_id
    ):
        raise ValueError("Comparison manifest identity or completion state is invalid")
    parsed_inventory = _parse_sha256_inventory(
        manifest["inventory"],
        field="Comparison inventory",
        expected_excludes=("manifest.json",),
    )
    inventory = dict(parsed_inventory)
    actual_files = _regular_files(pack_root, exclude={"manifest.json"})
    if len(actual_files) + 1 > _MAX_ZIP_FILES:
        raise ValueError("Comparison pack contains too many files")
    if sum(path.stat().st_size for path in actual_files.values()) > _MAX_ZIP_UNCOMPRESSED_BYTES:
        raise ValueError("Comparison pack is too large")
    if set(actual_files) != set(inventory):
        raise ValueError(
            "Comparison inventory is not exact; "
            f"missing={sorted(set(inventory) - set(actual_files))}, "
            f"unexpected={sorted(set(actual_files) - set(inventory))}"
        )
    authored_files = {
        relative for relative in actual_files if not relative.startswith("evidence/")
    }
    if authored_files != _PACK_REQUIRED_FILES:
        raise ValueError(
            "Comparison-authored file set is not exact; "
            f"missing={sorted(_PACK_REQUIRED_FILES - authored_files)}, "
            f"unexpected={sorted(authored_files - _PACK_REQUIRED_FILES)}"
        )
    for relative, expected in parsed_inventory:
        if sha256_file(actual_files[relative]) != expected:
            raise ValueError(f"Comparison checksum mismatch: {relative}")

    source_roots = _validate_source_roots_payload(
        pack_root, comparison_id=comparison_id, inventory=inventory
    )
    replay = _replay_pack_evidence(pack_root, comparison_id=comparison_id)
    _compare_replayed_outputs(pack_root, replay=replay, source_roots=source_roots)
    _assert_no_prohibited_claims(pack_root)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": ARTIFACT_TYPE,
        "comparison_id": comparison_id,
        "status": "PASS",
        "verified_file_count": len(actual_files) + 1,
        "holdout_term": HOLDOUT_NAME,
        "disclaimer": ACADEMIC_DISCLAIMER,
    }


def _create_deterministic_zip(source_dir: Path, zip_path: Path) -> str:
    if os.path.lexists(zip_path):
        raise FileExistsError(f"ZIP already exists; refusing to overwrite: {zip_path}")
    files = _regular_files(source_dir)
    with zipfile.ZipFile(
        zip_path,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        strict_timestamps=True,
    ) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(
                f"{source_dir.name}/{relative}", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, files[relative].read_bytes(), compresslevel=9)
    return sha256_file(zip_path)


def _read_checksum_sidecar(sidecar: Path, *, zip_path: Path) -> str:
    if sidecar.name != f"{zip_path.name}.sha256":
        raise ValueError("Checksum sidecar name must match the ZIP filename")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise FileNotFoundError("Checksum sidecar is missing or not a regular file")
    try:
        payload = sidecar.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Checksum sidecar must contain ASCII") from exc
    match = re.fullmatch(
        rf"([0-9a-f]{{64}})  {re.escape(zip_path.name)}\n", payload
    )
    if match is None:
        raise ValueError("Checksum sidecar must contain one strict SHA-256 record")
    expected = match.group(1)
    if sha256_file(zip_path) != expected:
        raise ValueError("ZIP checksum does not match its sidecar")
    return expected


def _validated_zip_members(archive: zipfile.ZipFile) -> tuple[str, list[zipfile.ZipInfo]]:
    if archive.comment:
        raise ValueError("Comparison ZIP must not contain an archive comment")
    members = archive.infolist()
    if not members or len(members) > _MAX_ZIP_FILES:
        raise ValueError("Comparison ZIP has an invalid member count")
    names = [member.filename for member in members]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ValueError("Comparison ZIP members must be sorted and duplicate-free")
    total_size = 0
    roots: set[str] = set()
    for index, member in enumerate(members):
        name = _safe_relative(member.filename, field=f"ZIP member {index}")
        if name != member.filename or member.is_dir():
            raise ValueError("Comparison ZIP may contain only normalized regular files")
        path = PurePosixPath(name)
        if len(path.parts) < 2:
            raise ValueError("Every ZIP member must be under one comparison directory")
        roots.add(path.parts[0])
        mode = (member.external_attr >> 16) & 0xFFFF
        if (
            member.flag_bits & 0x1
            or member.compress_type != zipfile.ZIP_DEFLATED
            or member.date_time != (1980, 1, 1, 0, 0, 0)
            or member.create_system != 3
            or not stat.S_ISREG(mode)
            or stat.S_IMODE(mode) != 0o644
            or member.extra
        ):
            raise ValueError(f"ZIP member metadata is not canonical: {name}")
        total_size += member.file_size
        if total_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
            raise ValueError("Comparison ZIP exceeds the uncompressed size limit")
        if member.file_size and (
            member.compress_size == 0
            or member.file_size / member.compress_size > 1000
        ):
            raise ValueError("Comparison ZIP has an unsafe compression ratio")
    if len(roots) != 1:
        raise ValueError("Comparison ZIP must contain exactly one top-level directory")
    comparison_id = _safe_id(next(iter(roots)), field="ZIP comparison_id")
    return comparison_id, members


def _verify_zip_archive(zip_path: Path) -> dict[str, Any]:
    if zip_path.is_symlink() or not zip_path.is_file():
        raise ValueError("Comparison ZIP must be a regular non-symlink file")
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            comparison_id, members = _validated_zip_members(archive)
            if zip_path.name != f"{comparison_id}.zip":
                raise ValueError("Comparison ZIP filename must match its immutable ID")
            with tempfile.TemporaryDirectory(prefix="paisa-comparison-verify-") as temp:
                temp_root = Path(temp).resolve(strict=True)
                for member in members:
                    relative = PurePosixPath(member.filename)
                    destination = temp_root.joinpath(*relative.parts).resolve(
                        strict=False
                    )
                    try:
                        destination.relative_to(temp_root)
                    except ValueError as exc:
                        raise ValueError(
                            "Comparison ZIP member escapes the extraction root"
                        ) from exc
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member, "r") as source, destination.open("xb") as target:
                        shutil.copyfileobj(source, target)
                pack_root = temp_root / comparison_id
                result = _verify_pack_directory(pack_root)
                expected_names = [
                    f"{comparison_id}/{relative}"
                    for relative in sorted(_regular_files(pack_root))
                ]
                if [member.filename for member in members] != expected_names:
                    raise ValueError("ZIP member list does not match the verified pack")
                return result
    except zipfile.BadZipFile as exc:
        raise ValueError("Comparison ZIP is malformed or corrupt") from exc


def _verify_external_sources(
    *,
    training_run: str | Path | None,
    vertical_slice: str | Path | None,
    lstm_run: str | Path | None,
) -> str:
    provided = (training_run, vertical_slice, lstm_run)
    if not any(value is not None for value in provided):
        return "portable_evidence"
    if not all(value is not None for value in provided):
        raise ValueError("External verification requires all three source roots")
    snapshots = _capture_source_snapshots(
        training_run=training_run,
        vertical_slice=vertical_slice,
        lstm_run=lstm_run,
    )
    _assert_source_snapshots_unchanged(snapshots)
    return "portable_evidence_and_external_sources"


def verify_unified_comparison_pack(
    pack: str | Path,
    *,
    checksum: str | Path | None = None,
    training_run: str | Path | None = None,
    vertical_slice: str | Path | None = None,
    lstm_run: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a comparison directory or deterministic ZIP entirely offline."""

    candidate = reject_symlink_components(
        Path(pack).expanduser(), label="Unified comparison pack"
    )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError("Unified comparison pack does not exist") from exc

    checksum_path: Path | None = None
    zip_sha256: str | None = None
    if resolved.is_dir():
        result = _verify_pack_directory(resolved)
        if checksum is not None:
            sibling_zip = resolved.parent / f"{resolved.name}.zip"
            checksum_path = reject_symlink_components(
                Path(checksum).expanduser(), label="Comparison checksum sidecar"
            ).resolve(strict=True)
            zip_sha256 = _read_checksum_sidecar(checksum_path, zip_path=sibling_zip)
            zip_result = _verify_zip_archive(sibling_zip)
            if zip_result["comparison_id"] != result["comparison_id"]:
                raise ValueError("Directory and ZIP comparison IDs disagree")
    elif resolved.is_file() and resolved.suffix == ".zip":
        requested_sidecar = Path(checksum).expanduser() if checksum is not None else None
        automatic_sidecar = resolved.with_name(f"{resolved.name}.sha256")
        sidecar_candidate = requested_sidecar or automatic_sidecar
        if requested_sidecar is None and not automatic_sidecar.exists():
            raise FileNotFoundError(
                "Comparison ZIP verification requires its SHA-256 sidecar"
            )
        checksum_path = reject_symlink_components(
            sidecar_candidate, label="Comparison checksum sidecar"
        ).resolve(strict=True)
        zip_sha256 = _read_checksum_sidecar(checksum_path, zip_path=resolved)
        result = _verify_zip_archive(resolved)
    else:
        raise ValueError("Pack must be a comparison directory or .zip file")

    source_verification = _verify_external_sources(
        training_run=training_run,
        vertical_slice=vertical_slice,
        lstm_run=lstm_run,
    )
    return {
        **result,
        "pack": resolved.name,
        "path_base": "input_parent",
        "checksum_verified": checksum_path is not None,
        "zip_sha256": zip_sha256,
        "source_verification": source_verification,
    }


def _entry_identity(path: Path) -> tuple[int, int, int, int]:
    metadata = path.lstat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_size),
    )


def _publish_staged_file(staged: Path, destination: Path) -> None:
    identity = _entry_identity(staged)
    if not stat.S_ISREG(identity[2]) or staged.is_symlink():
        raise ValueError("Staged publication file must be regular and non-symlink")
    atomic_rename_noreplace(staged, destination)
    if _entry_identity(destination) != identity:
        raise RuntimeError("Published file identity changed during publication")


def _publish_staged_pack(
    *,
    stage_dir: Path,
    final_dir: Path,
    staged_zip: Path | None,
    final_zip: Path | None,
    staged_checksum: Path | None,
    final_checksum: Path | None,
) -> None:
    targets = (final_dir, final_zip, final_checksum)
    if any(target is not None and os.path.lexists(target) for target in targets):
        raise FileExistsError("Comparison output appeared during staging")
    if (staged_zip is None) != (final_zip is None) or (
        staged_checksum is None
    ) != (final_checksum is None):
        raise ValueError("Staged and final portable files must be paired")
    if (staged_zip is None) != (staged_checksum is None):
        raise ValueError("ZIP and checksum must be published together")
    stage_identity = _entry_identity(stage_dir)
    if not stat.S_ISDIR(stage_identity[2]) or stage_dir.is_symlink():
        raise ValueError("Staged comparison must be a real directory")
    if staged_zip is not None:
        assert final_zip is not None
        assert staged_checksum is not None
        assert final_checksum is not None
        # Publish the checksum first. A late error may leave an inert sidecar or
        # a complete ZIP/sidecar pair, but never an unauthenticated public ZIP.
        _publish_staged_file(staged_checksum, final_checksum)
        _publish_staged_file(staged_zip, final_zip)
    atomic_rename_noreplace(stage_dir, final_dir)


def build_unified_price_model_comparison(
    *,
    comparison_id: str,
    output_dir: str | Path,
    training_run: str | Path,
    vertical_slice: str | Path,
    lstm_run: str | Path,
    create_zip: bool = True,
) -> dict[str, Any]:
    """Build an immutable comparison solely from the three completed sources."""

    resolved_id = _safe_id(comparison_id, field="comparison_id")
    snapshots = _capture_source_snapshots(
        training_run=training_run,
        vertical_slice=vertical_slice,
        lstm_run=lstm_run,
    )
    repo_root = Path(__file__).resolve().parents[3]
    output_candidate = reject_symlink_components(
        Path(output_dir).expanduser(), label="Unified comparison output root"
    )
    _validate_output_separation(output_candidate, snapshots)
    output_root = prepare_output_root(
        output_candidate,
        repository_root=repo_root,
        label="Unified comparison output root",
    )
    _validate_output_separation(output_root, snapshots)

    final_dir = output_root / resolved_id
    canonical_zip = output_root / f"{resolved_id}.zip"
    canonical_checksum = output_root / f"{resolved_id}.zip.sha256"
    for target in (final_dir, canonical_zip, canonical_checksum):
        if os.path.lexists(target):
            raise FileExistsError(f"Output already exists; refusing to overwrite: {target}")
    final_zip = canonical_zip if create_zip else None
    final_checksum = canonical_checksum if create_zip else None

    stage_parent = Path(
        tempfile.mkdtemp(prefix=f".{resolved_id}-", dir=output_root)
    )
    stage_dir = stage_parent / resolved_id
    stage_dir.mkdir()
    staged_zip = stage_parent / f"{resolved_id}.zip" if create_zip else None
    staged_checksum = (
        stage_parent / f"{resolved_id}.zip.sha256" if create_zip else None
    )
    committed = False
    replay: ReplayResults | None = None
    zip_sha256: str | None = None
    try:
        specs = _evidence_specs(snapshots)
        _copy_evidence(stage_dir, snapshots, specs)
        certificate_path, certificate_sha256 = _write_vertical_manifest_certificate(
            stage_dir, snapshots
        )
        source_roots = _source_roots_payload(
            resolved_id,
            snapshots,
            specs,
            vertical_certificate_path=certificate_path,
            vertical_certificate_sha256=certificate_sha256,
        )
        write_json(source_roots, stage_dir / "source_roots.json")
        replay = _replay_pack_evidence(stage_dir, comparison_id=resolved_id)
        _write_replayed_outputs(stage_dir, replay, source_roots=source_roots)
        _write_pack_manifest(stage_dir, comparison_id=resolved_id)
        _verify_pack_directory(stage_dir)

        if staged_zip is not None:
            assert staged_checksum is not None
            zip_sha256 = _create_deterministic_zip(stage_dir, staged_zip)
            staged_checksum.write_text(
                f"{zip_sha256}  {staged_zip.name}\n", encoding="ascii"
            )
            _read_checksum_sidecar(staged_checksum, zip_path=staged_zip)
            _verify_zip_archive(staged_zip)

        _assert_source_snapshots_unchanged(snapshots)
        _publish_staged_pack(
            stage_dir=stage_dir,
            final_dir=final_dir,
            staged_zip=staged_zip,
            final_zip=final_zip,
            staged_checksum=staged_checksum,
            final_checksum=final_checksum,
        )
        committed = True
        try:
            stage_parent.rmdir()
        except OSError:
            pass
    except Exception:
        if not committed and stage_parent.exists():
            shutil.rmtree(stage_parent)
        _assert_source_snapshots_unchanged(snapshots)
        raise

    assert replay is not None
    return {
        **replay.final_summary,
        "artifact_directory": final_dir.name,
        "zip_path": final_zip.name if final_zip is not None else None,
        "checksum_path": final_checksum.name if final_checksum is not None else None,
        "zip_sha256": zip_sha256,
        "path_base": "output_root",
    }


__all__ = [
    "ACADEMIC_DISCLAIMER",
    "AUTHORITATIVE_SOURCE_ROOTS",
    "HOLDOUT_NAME",
    "build_unified_price_model_comparison",
    "validate_lstm_run_artifacts",
    "verify_unified_comparison_pack",
]
