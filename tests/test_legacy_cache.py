from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pytest

import paisa.legacy_cache as legacy_cache_module
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


def _raw_path(fixture: LegacyFixture, symbol: str = "OGDC") -> Path:
    return fixture.raw_dir / "psx" / f"{symbol}_{EXPECTED_PROVIDER}_raw.csv"


def _set_raw_count(fixture: LegacyFixture, symbol: str, count: int) -> None:
    metadata = json.loads(fixture.metadata_path.read_text(encoding="utf-8"))
    metadata["raw_row_counts"][symbol] = count
    fixture.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _rewrite_as_processed_only_bundle(bundle: Path) -> None:
    """Turn a strong fixture bundle into the weak schema accepted before this fix."""

    metadata_path = bundle / "pipeline_metadata.json"
    manifest_path = bundle / "migration_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    original_artifacts = manifest["original_artifacts"]
    original_artifacts["prices"]["sha256"] = manifest["migrated_artifacts"][
        "prices"
    ]["sha256"]
    original_artifacts["raw_directory"] = None
    original_artifacts["raw_captures"] = {}
    validation = manifest["validation"]
    validation["raw_rebuild_matches_legacy_processed_cache"] = False
    validation.pop("raw_one_to_one_row_accounting_verified", None)

    strategy = "validated_legacy_processed_cache_copy"
    strength = "processed_cache_and_legacy_metadata_only"
    manifest["migration_strategy"] = strategy
    manifest["provenance_strength"] = strength
    metadata["migration"]["strategy"] = strategy
    metadata["migration"]["provenance_strength"] = strength
    metadata["legacy_evidence"] = {
        "original_artifacts": original_artifacts,
        "validation": validation,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manifest["migrated_artifacts"]["pipeline_metadata"]["sha256"] = _sha256(
        metadata_path
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    assert persisted_manifest["validation"][
        "raw_one_to_one_row_accounting_verified"
    ] is True
    for raw_record in persisted_manifest["original_artifacts"][
        "raw_captures"
    ].values():
        assert raw_record["row_count"] == raw_record["cleaned_row_count"]
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


def test_migration_rejects_processed_only_input_when_raw_is_absent(
    tmp_path: Path,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    shutil.rmtree(fixture.raw_dir)
    before = _tree_bytes(fixture.root)
    output_dir = tmp_path / "verified"

    with pytest.raises(ValueError, match="raw|processed|verified|certif"):
        migrate_legacy_psx_cache(
            fixture.prices_path,
            fixture.metadata_path,
            output_dir,
        )

    assert not output_dir.exists()
    assert _tree_bytes(fixture.root) == before


def test_bundle_validation_rejects_preexisting_processed_only_bundle(tmp_path: Path):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    output_dir = tmp_path / "verified"
    _migrate(fixture, output_dir)
    _rewrite_as_processed_only_bundle(output_dir)

    with pytest.raises(ValueError, match="raw|processed|strategy|provenance"):
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


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("date", "not-a-date", "raw|date|one-to-one"),
        ("open", "not-a-number", "raw|open|finite|numeric|one-to-one"),
        ("open", 0, "raw|open|positive|one-to-one"),
        ("close", -1, "raw|close|positive|one-to-one"),
        ("volume", -1, "raw|volume|non.?negative|one-to-one"),
    ],
    ids=[
        "invalid-date",
        "non-finite-open",
        "non-positive-open",
        "non-positive-close",
        "negative-volume",
    ],
)
def test_migration_rejects_raw_rows_that_cleaning_would_silently_drop(
    tmp_path: Path,
    column: str,
    bad_value: Any,
    message: str,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    raw_path = _raw_path(fixture)
    raw = pd.read_csv(raw_path)
    extra = raw.iloc[[-1]].copy()
    extra.loc[:, "date"] = "2025-01-07"
    if isinstance(bad_value, str):
        extra[column] = extra[column].astype("object")
    extra.loc[:, column] = bad_value
    raw = pd.concat([raw, extra], ignore_index=True)
    raw.to_csv(raw_path, index=False)
    _set_raw_count(fixture, "OGDC", len(raw))
    source_bytes = _tree_bytes(fixture.root)
    output_dir = tmp_path / "verified"

    with pytest.raises(ValueError, match=message):
        _migrate(fixture, output_dir)

    assert not output_dir.exists()
    assert _tree_bytes(fixture.root) == source_bytes


def test_migration_rejects_raw_duplicate_that_cleaning_would_silently_collapse(
    tmp_path: Path,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    raw_path = _raw_path(fixture)
    raw = pd.read_csv(raw_path)
    raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    raw.to_csv(raw_path, index=False)
    _set_raw_count(fixture, "OGDC", len(raw))
    output_dir = tmp_path / "verified"

    with pytest.raises(ValueError, match="raw|duplicate|date|one-to-one"):
        _migrate(fixture, output_dir)

    assert not output_dir.exists()


def test_migration_rejects_fractional_raw_volume_instead_of_rounding_it(
    tmp_path: Path,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    raw_path = _raw_path(fixture)
    raw = pd.read_csv(raw_path)
    raw["volume"] = raw["volume"].astype(float)
    raw.loc[0, "volume"] = float(raw.loc[0, "volume"]) + 0.4
    raw.to_csv(raw_path, index=False)
    output_dir = tmp_path / "verified"

    with pytest.raises(ValueError, match="raw|volume|integer|whole|round"):
        _migrate(fixture, output_dir)

    assert not output_dir.exists()


def test_raw_hash_is_bound_to_private_snapshot_bytes_used_for_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    raw_path = _raw_path(fixture)
    raw_a_bytes = raw_path.read_bytes()
    raw_b = pd.read_csv(raw_path)
    raw_b.loc[0, "close"] = float(raw_b.loc[0, "close"]) + 7.5
    raw_b_bytes = raw_b.to_csv(index=False).encode("utf-8")

    processed = pd.read_csv(fixture.prices_path)
    target = (processed["ticker"] == "OGDC") & (
        processed["date"] == raw_b.loc[0, "date"]
    )
    processed.loc[target, "close"] = raw_b.loc[0, "close"]
    processed.to_csv(fixture.prices_path, index=False)

    real_read_csv = pd.read_csv
    raced_original = False

    def race_original_raw(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal raced_original
        if Path(path).resolve() == raw_path.resolve():
            raced_original = True
            raw_path.write_bytes(raw_b_bytes)
            try:
                return real_read_csv(path, *args, **kwargs)
            finally:
                raw_path.write_bytes(raw_a_bytes)
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(legacy_cache_module.pd, "read_csv", race_original_raw)
    output_dir = tmp_path / "verified"

    with pytest.raises(ValueError, match="reproduce|match|raw|processed"):
        _migrate(fixture, output_dir)

    assert raced_original is False
    assert raw_path.read_bytes() == raw_a_bytes
    assert not output_dir.exists()


def test_processed_hash_is_bound_to_private_snapshot_bytes_used_for_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    prices_a_bytes = fixture.prices_path.read_bytes()
    prices_b = pd.read_csv(fixture.prices_path)
    target = prices_b["ticker"] == "OGDC"
    first_index = prices_b.index[target][0]
    prices_b.loc[first_index, "close"] = (
        float(prices_b.loc[first_index, "close"]) + 7.5
    )
    prices_b_bytes = prices_b.to_csv(index=False).encode("utf-8")

    raw_path = _raw_path(fixture)
    raw = pd.read_csv(raw_path)
    raw.loc[0, "close"] = prices_b.loc[first_index, "close"]
    raw.to_csv(raw_path, index=False)

    real_read_csv = pd.read_csv
    raced_original = False

    def race_original_prices(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        nonlocal raced_original
        if Path(path).resolve() == fixture.prices_path.resolve():
            raced_original = True
            fixture.prices_path.write_bytes(prices_b_bytes)
            try:
                return real_read_csv(path, *args, **kwargs)
            finally:
                fixture.prices_path.write_bytes(prices_a_bytes)
        return real_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(legacy_cache_module.pd, "read_csv", race_original_prices)
    output_dir = tmp_path / "verified"

    with pytest.raises(ValueError, match="reproduce|match|raw|processed"):
        _migrate(fixture, output_dir)

    assert raced_original is False
    assert fixture.prices_path.read_bytes() == prices_a_bytes
    assert not output_dir.exists()


def test_metadata_hash_is_bound_to_private_snapshot_bytes_used_for_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fixture = _write_legacy_fixture(tmp_path / "legacy")
    metadata_a_bytes = fixture.metadata_path.read_bytes()
    metadata_b = json.loads(metadata_a_bytes)
    metadata_b["errors"] = {"UNRELATED": "race sentinel"}
    metadata_b_bytes = json.dumps(metadata_b, indent=2).encode("utf-8")
    real_load_json = legacy_cache_module._load_json
    raced_original = False

    def race_original_metadata(path: Path, *, label: str) -> dict[str, Any]:
        nonlocal raced_original
        if Path(path).resolve() == fixture.metadata_path.resolve():
            raced_original = True
            fixture.metadata_path.write_bytes(metadata_b_bytes)
            try:
                return real_load_json(path, label=label)
            finally:
                fixture.metadata_path.write_bytes(metadata_a_bytes)
        return real_load_json(path, label=label)

    monkeypatch.setattr(legacy_cache_module, "_load_json", race_original_metadata)
    output_dir = tmp_path / "verified"

    _migrate(fixture, output_dir)

    generated = json.loads(
        (output_dir / "pipeline_metadata.json").read_text(encoding="utf-8")
    )
    assert raced_original is False
    assert generated["errors"] == {}
    assert fixture.metadata_path.read_bytes() == metadata_a_bytes
