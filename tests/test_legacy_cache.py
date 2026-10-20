from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

from paisa.legacy_cache import (
    migrate_legacy_psx_cache,
    validate_verified_legacy_bundle,
)


EXPECTED_PROVIDER = "psx_dps"
EXPECTED_PRODUCER = "paisa.legacy_cache.migrate_legacy_psx_cache"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _nested_values(value: Any, key: str) -> list[Any]:
    matches: list[Any] = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            if child_key == key:
                matches.append(child_value)
            matches.extend(_nested_values(child_value, key))
    elif isinstance(value, list):
        for child in value:
            matches.extend(_nested_values(child, key))
    return matches


@dataclass(frozen=True)
class LegacyFixture:
    root: Path
    prices_path: Path
    metadata_path: Path
    raw_dir: Path
    prices: pd.DataFrame


def _market_rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "HBL": [
            {
                "date": "2025-01-02",
                "open": 112.00,
                "high": 113.10,
                "low": 111.40,
                "close": 112.80,
                "volume": 420_000,
            },
            {
                "date": "2025-01-03",
                "open": 112.90,
                "high": 114.00,
                "low": 112.30,
                "close": 113.60,
                "volume": 515_000,
            },
            {
                "date": "2025-01-06",
                "open": 113.75,
                "high": 114.20,
                "low": 112.80,
                "close": 113.10,
                "volume": 385_000,
            },
        ],
        "OGDC": [
            {
                "date": "2025-01-02",
                "open": 208.50,
                "high": 211.40,
                "low": 207.70,
                "close": 210.25,
                "volume": 1_240_000,
            },
            {
                "date": "2025-01-03",
                "open": 210.75,
                "high": 212.60,
                "low": 209.80,
                "close": 211.90,
                "volume": 1_310_000,
            },
            {
                "date": "2025-01-06",
                "open": 211.20,
                "high": 213.00,
                "low": 210.10,
                "close": 212.35,
                "volume": 1_180_000,
            },
        ],
    }


def _write_legacy_fixture(
    root: Path,
    *,
    source: str = EXPECTED_PROVIDER,
    metadata_mutation: Callable[[dict[str, Any]], None] | None = None,
) -> LegacyFixture:
    processed_dir = root / "data" / "processed"
    raw_dir = root / "data" / "raw"
    raw_psx_dir = raw_dir / "psx"
    processed_dir.mkdir(parents=True)
    raw_psx_dir.mkdir(parents=True)

    standardized: list[pd.DataFrame] = []
    rows_by_symbol = _market_rows()
    for symbol, records in rows_by_symbol.items():
        raw = pd.DataFrame.from_records(records)
        raw.to_csv(raw_psx_dir / f"{symbol}_{EXPECTED_PROVIDER}_raw.csv", index=False)
        symbol_frame = raw.copy()
        symbol_frame.insert(1, "ticker", symbol)
        symbol_frame["source"] = source
        standardized.append(symbol_frame)

    prices = (
        pd.concat(standardized, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    prices = prices[
        ["date", "ticker", "open", "high", "low", "close", "volume", "source"]
    ]
    prices_path = processed_dir / "psx_prices.csv"
    prices.to_csv(prices_path, index=False)

    metadata: dict[str, Any] = {
        "generated_at": "2026-09-15T23:46:30",
        "provider": EXPECTED_PROVIDER,
        "symbols_requested": ["OGDC", "HBL"],
        "raw_row_counts": {symbol: len(rows) for symbol, rows in rows_by_symbol.items()},
        "errors": {},
        "prices_rows": len(prices),
        "features_rows": len(prices),
        "ml_dataset_rows": 0,
        "date_range": {"start": "2025-01-02", "end": "2025-01-06"},
        "files": {
            "prices": str(prices_path),
            "features": str(processed_dir / "psx_features.csv"),
            "dataset": str(processed_dir / "ml_dataset.csv"),
            "model_dir": str(root / "models"),
        },
        "model_metrics": None,
    }
    if metadata_mutation is not None:
        metadata_mutation(metadata)
    metadata_path = processed_dir / "pipeline_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return LegacyFixture(root, prices_path, metadata_path, raw_dir, prices)


def _migrate(fixture: LegacyFixture, output_dir: Path) -> dict[str, Any]:
    return migrate_legacy_psx_cache(
        fixture.prices_path,
        fixture.metadata_path,
        output_dir,
        legacy_raw_dir=fixture.raw_dir,
        expected_provider=EXPECTED_PROVIDER,
        required_symbol="OGDC",
    )


def test_migration_rebuilds_and_authenticates_bundle_from_real_provider_raw_files(
    tmp_path: Path,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    output_dir = tmp_path / "verified"

    returned_manifest = _migrate(fixture, output_dir)

    expected_files = {
        "psx_prices.csv",
        "pipeline_metadata.json",
        "legacy_pipeline_metadata.json",
        "migration_manifest.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected_files

    migrated = pd.read_csv(output_dir / "psx_prices.csv")
    pd.testing.assert_frame_equal(migrated, fixture.prices, check_dtype=False)
    assert set(migrated["source"]) == {EXPECTED_PROVIDER}

    metadata = json.loads(
        (output_dir / "pipeline_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["schema_version"] == "1.0"
    assert metadata["producer"] == {
        "component": EXPECTED_PRODUCER,
        "schema_version": "1.0",
        "status": "complete",
    }
    assert metadata["prices_artifact"]["sha256"] == _sha256(
        output_dir / "psx_prices.csv"
    )
    assert metadata["prices_artifact"]["row_count"] == len(fixture.prices)
    assert metadata["prices_artifact"]["symbols"]["OGDC"] == {
        "row_count": 3,
        "date_start": "2025-01-02",
        "date_end": "2025-01-06",
        "source": EXPECTED_PROVIDER,
    }
    assert metadata["generated_at"] == "2026-09-15T23:46:30"
    assert metadata["acquisition_mode"] == "verified_legacy_psx_cache"
    assert metadata["live_fetch_succeeded"] is False
    assert metadata["live_acquisition_performed"] is False
    assert metadata["original_provider"] == EXPECTED_PROVIDER
    assert metadata["legacy_metadata_preserved"] is True
    assert metadata["migration"]["strategy"] == (
        "rebuild_from_legacy_raw_captures"
    )
    assert metadata["migration"]["original_generated_at"] == (
        "2026-09-15T23:46:30"
    )
    assert metadata["migration"]["provenance_strength"]
    assert metadata["migration"]["limitations"]
    assert metadata["migration_timestamp"] != metadata["generated_at"]

    persisted_manifest = json.loads(
        (output_dir / "migration_manifest.json").read_text(encoding="utf-8")
    )
    returned_persisted_fields = dict(returned_manifest)
    assert Path(returned_persisted_fields.pop("output_directory")) == output_dir
    assert returned_persisted_fields == persisted_manifest
    assert persisted_manifest["acquisition_mode"] == "verified_legacy_psx_cache"
    assert persisted_manifest["live_fetch_succeeded"] is False
    assert persisted_manifest["live_acquisition_performed"] is False
    assert persisted_manifest["original_provider"] == EXPECTED_PROVIDER
    assert persisted_manifest["legacy_metadata_preserved"] is True
    assert persisted_manifest["migration_strategy"] == (
        "rebuild_from_legacy_raw_captures"
    )
    assert persisted_manifest["provenance_strength"]
    assert persisted_manifest["limitations"]
    assert persisted_manifest["original_artifacts"]["pipeline_metadata"][
        "generated_at"
    ] == "2026-09-15T23:46:30"
    for bundled_name in (
        "psx_prices.csv",
        "pipeline_metadata.json",
        "legacy_pipeline_metadata.json",
    ):
        assert _sha256(output_dir / bundled_name) in set(
            _nested_values(persisted_manifest, "sha256")
        )

    validated = validate_verified_legacy_bundle(
        output_dir / "psx_prices.csv", required_symbol="OGDC"
    )
    assert validated


def test_migration_leaves_all_legacy_input_bytes_unchanged(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    before = _tree_bytes(fixture.root)

    _migrate(fixture, tmp_path / "verified")

    assert _tree_bytes(fixture.root) == before
    assert (
        tmp_path / "verified" / "legacy_pipeline_metadata.json"
    ).read_bytes() == fixture.metadata_path.read_bytes()


def test_migration_uses_explicit_weaker_processed_only_mode_when_raw_is_absent(
    tmp_path: Path,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    original_prices_hash = _sha256(fixture.prices_path)
    shutil.rmtree(fixture.raw_dir)
    output_dir = tmp_path / "verified"

    manifest = migrate_legacy_psx_cache(
        fixture.prices_path,
        fixture.metadata_path,
        output_dir,
    )

    assert manifest["migration_strategy"] == "validated_legacy_processed_cache_copy"
    assert manifest["provenance_strength"] == (
        "processed_cache_and_legacy_metadata_only"
    )
    assert manifest["validation"]["raw_rebuild_matches_legacy_processed_cache"] is False
    assert manifest["original_artifacts"]["raw_captures"] == {}
    assert _sha256(output_dir / "psx_prices.csv") == original_prices_hash
    validate_verified_legacy_bundle(output_dir / "psx_prices.csv")


def test_migration_rejects_missing_legacy_metadata(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    missing = fixture.metadata_path.with_name("missing_metadata.json")

    with pytest.raises(ValueError, match="missing.*metadata"):
        migrate_legacy_psx_cache(
            fixture.prices_path,
            missing,
            tmp_path / "verified",
            legacy_raw_dir=fixture.raw_dir,
        )


def test_migration_rejects_wrong_legacy_provider(tmp_path: Path):
    fixture = _write_legacy_fixture(
        tmp_path / "legacy",
        metadata_mutation=lambda metadata: metadata.update(
            {"provider": "pypsx_toolkit"}
        ),
    )

    with pytest.raises(ValueError, match="provider|psx_dps"):
        _migrate(fixture, tmp_path / "verified")


@pytest.mark.parametrize(
    "non_real_source",
    ["sample", "synthetic", "fixture", "mock", "test"],
)
def test_migration_rejects_non_real_source_markers(
    tmp_path: Path,
    non_real_source: str,
):
    fixture = _write_legacy_fixture(
        tmp_path / f"legacy-{non_real_source}", source=non_real_source
    )

    with pytest.raises(ValueError, match="source|provider|real"):
        _migrate(fixture, tmp_path / f"verified-{non_real_source}")


def test_migration_rejects_mixed_price_sources(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    prices = pd.read_csv(fixture.prices_path)
    prices.loc[0, "source"] = "pypsx_toolkit"
    prices.to_csv(fixture.prices_path, index=False)

    with pytest.raises(ValueError, match="mixed|source|provider"):
        _migrate(fixture, tmp_path / "verified")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda metadata: metadata.update({"prices_rows": 999}), "row|count"),
        (
            lambda metadata: metadata.update(
                {"date_range": {"start": "2025-01-03", "end": "2025-01-06"}}
            ),
            "date|range",
        ),
    ],
    ids=["row-count", "date-range"],
)
def test_migration_rejects_csv_metadata_coverage_mismatch(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
):
    fixture = _write_legacy_fixture(
        tmp_path / "legacy", metadata_mutation=mutation
    )

    with pytest.raises(ValueError, match=message):
        _migrate(fixture, tmp_path / "verified")


@pytest.mark.parametrize("bad_raw_count", [None, 0, -1], ids=["missing", "zero", "negative"])
def test_migration_rejects_missing_or_nonpositive_ogdc_raw_evidence(
    tmp_path: Path,
    bad_raw_count: int | None,
):
    def mutate(metadata: dict[str, Any]) -> None:
        if bad_raw_count is None:
            metadata["raw_row_counts"].pop("OGDC")
        else:
            metadata["raw_row_counts"]["OGDC"] = bad_raw_count

    fixture = _write_legacy_fixture(
        tmp_path / "legacy", metadata_mutation=mutate
    )

    with pytest.raises(ValueError, match="raw|row|OGDC|acquisition"):
        _migrate(fixture, tmp_path / "verified")


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("raw_row_counts", True),
        ("raw_row_counts", 3.7),
        ("prices_rows", True),
        ("prices_rows", 6.9),
    ],
    ids=["raw-bool", "raw-float", "prices-bool", "prices-float"],
)
def test_migration_rejects_noninteger_count_evidence(
    tmp_path: Path,
    field: str,
    bad_value: Any,
):
    def mutate(metadata: dict[str, Any]) -> None:
        if field == "raw_row_counts":
            metadata[field]["OGDC"] = bad_value
        else:
            metadata[field] = bad_value

    fixture = _write_legacy_fixture(
        tmp_path / "legacy", metadata_mutation=mutate
    )

    with pytest.raises(ValueError, match="integer|row|count"):
        _migrate(fixture, tmp_path / "verified")


def test_migration_rejects_legacy_ogdc_provider_error(tmp_path: Path):
    fixture = _write_legacy_fixture(
        tmp_path / "legacy",
        metadata_mutation=lambda metadata: metadata["errors"].update(
            {"OGDC": "provider returned HTTP 404"}
        ),
    )

    with pytest.raises(ValueError, match="OGDC|error|acquisition"):
        _migrate(fixture, tmp_path / "verified")


def test_bundle_validation_rejects_modified_migrated_prices(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    output_dir = tmp_path / "verified"
    _migrate(fixture, output_dir)
    prices_path = output_dir / "psx_prices.csv"
    prices = pd.read_csv(prices_path)
    prices.loc[0, "close"] += 1.0
    prices.to_csv(prices_path, index=False)

    with pytest.raises(ValueError, match="checksum|hash|modified|SHA-256"):
        validate_verified_legacy_bundle(prices_path, required_symbol="OGDC")


def test_bundle_validation_rejects_modified_generated_metadata(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    output_dir = tmp_path / "verified"
    _migrate(fixture, output_dir)
    metadata_path = output_dir / "pipeline_metadata.json"
    metadata_path.write_bytes(metadata_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="checksum|hash|modified|SHA-256"):
        validate_verified_legacy_bundle(
            output_dir / "psx_prices.csv", metadata_path=metadata_path
        )


def test_bundle_validation_rejects_modified_preserved_legacy_metadata(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    output_dir = tmp_path / "verified"
    _migrate(fixture, output_dir)
    preserved = output_dir / "legacy_pipeline_metadata.json"
    preserved.write_bytes(preserved.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="metadata|hash|SHA-256"):
        validate_verified_legacy_bundle(output_dir / "psx_prices.csv")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda manifest: manifest["migrated_artifacts"]["prices"].update(
            {"row_count": 999}
        ),
        lambda manifest: manifest["validation"].update(
            {"required_columns_verified": False}
        ),
        lambda manifest: manifest["original_artifacts"]["raw_captures"][
            "OGDC"
        ].update({"sha256": "0" * 64}),
    ],
    ids=["migrated-row-count", "validation-claim", "raw-capture-hash"],
)
def test_bundle_validation_rejects_modified_migration_manifest(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    output_dir = tmp_path / "verified"
    _migrate(fixture, output_dir)
    manifest_path = output_dir / "migration_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest|evidence|row|checksum"):
        validate_verified_legacy_bundle(output_dir / "psx_prices.csv")


def test_existing_destination_is_never_overwritten(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    output_dir = tmp_path / "verified"
    _migrate(fixture, output_dir)
    before = _tree_bytes(output_dir)

    with pytest.raises(FileExistsError, match="exist|overwrite|destination"):
        _migrate(fixture, output_dir)

    assert _tree_bytes(output_dir) == before


def test_raw_rebuild_allows_equivalent_original_csv_serialization(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    reordered = pd.read_csv(fixture.prices_path).iloc[::-1].reset_index(drop=True)
    reordered.to_csv(fixture.prices_path, index=False)
    original_hash = _sha256(fixture.prices_path)
    output_dir = tmp_path / "verified"

    manifest = _migrate(fixture, output_dir)

    assert manifest["migration_strategy"] == "rebuild_from_legacy_raw_captures"
    assert manifest["original_artifacts"]["prices"]["sha256"] == original_hash
    assert manifest["migrated_artifacts"]["prices"]["sha256"] != original_hash
    validate_verified_legacy_bundle(output_dir / "psx_prices.csv")


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("open", "not-a-number", "open|numeric"),
        ("close", 0, "close|positive"),
        ("high", "not-a-number", "high|numeric"),
        ("low", -1, "low|positive"),
        ("volume", -1, "volume|non.?negative"),
    ],
)
def test_migration_rejects_invalid_market_values(
    tmp_path: Path,
    column: str,
    bad_value: Any,
    message: str,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    prices = pd.read_csv(fixture.prices_path)
    if isinstance(bad_value, str):
        prices[column] = prices[column].astype("object")
    prices.loc[0, column] = bad_value
    prices.to_csv(fixture.prices_path, index=False)

    with pytest.raises(ValueError, match=message):
        _migrate(fixture, tmp_path / "verified")


def test_migration_rejects_duplicate_symbol_dates(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    prices = pd.read_csv(fixture.prices_path)
    prices = pd.concat([prices, prices.iloc[[0]]], ignore_index=True)
    prices.to_csv(fixture.prices_path, index=False)
    metadata = json.loads(fixture.metadata_path.read_text(encoding="utf-8"))
    metadata["prices_rows"] = len(prices)
    fixture.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|unique|ticker|date"):
        _migrate(fixture, tmp_path / "verified")
