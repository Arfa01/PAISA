"""Truthful migration and validation for pre-schema PAISA PSX caches.

The compatibility path in this module deliberately does not pretend that an
old cache was produced by the current pipeline.  It preserves the old metadata,
rebuilds from captured provider rows when complete raw evidence is available,
and publishes a checksum-bound schema-1.0 bundle under a distinct producer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from paisa.experiments.artifacts import sha256_file, utc_now_z, write_json
from paisa.preprocessing import (
    COLUMN_ALIASES,
    REQUIRED_OUTPUT_COLUMNS,
    combine_price_frames,
    standardize_price_frame,
)


LEGACY_CACHE_SCHEMA_VERSION = "1.0"
LEGACY_CACHE_PRODUCER = "paisa.legacy_cache.migrate_legacy_psx_cache"
LEGACY_ACQUISITION_MODE = "verified_legacy_psx_cache"
PIPELINE_METADATA_FILENAME = "pipeline_metadata.json"
LEGACY_METADATA_FILENAME = "legacy_pipeline_metadata.json"
MIGRATION_MANIFEST_FILENAME = "migration_manifest.json"
PRICES_FILENAME = "psx_prices.csv"

_NON_REAL_SOURCE_MARKERS = ("sample", "synthetic", "fixture", "mock", "test")
_LIMITATIONS = [
    (
        "This is a verified reconstruction of a legacy cache, not a new live "
        "provider acquisition."
    ),
    (
        "The captured raw CSV files contain no provider signature, request URL, "
        "HTTP headers, or independently timestamped response hash; provider "
        "attribution rests on the preserved legacy metadata, exact provider-specific "
        "filenames, and reproducibility through PAISA cleaning logic."
    ),
    (
        "The bundle is checksum-bound but not cryptographically signed by an "
        "external authority."
    ),
    (
        "The captured PSX-DPS rows do not provide high/low values; the verified "
        "open, close, and volume fields are the inputs used by this workflow."
    ),
]


def _error(message: str) -> ValueError:
    return ValueError(f"Legacy PSX cache validation failed: {message}")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
    except FileNotFoundError:
        raise _error(f"missing {label}: {path}") from None
    except json.JSONDecodeError as exc:
        raise _error(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise _error(f"{label} root must be an object: {path}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(f"{label} must be an object")
    return value


def _json_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _error(f"{label} must be an integer")
    return value


def _parse_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{label} must be a non-empty ISO-8601 timestamp")
    timestamp = value.strip()
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _error(f"{label} must be a valid ISO-8601 timestamp") from exc
    return timestamp


def _stable_sha256(path: Path, *, label: str) -> str:
    if not path.is_file():
        raise _error(f"missing {label}: {path}")
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise _error(f"{label} changed while it was being hashed: {path}")
    return digest


def _snapshot_and_hash(source: Path, destination: Path, *, label: str) -> str:
    """Hash and copy exactly the bytes read through one stable descriptor."""

    if not source.is_file():
        raise _error(f"missing {label}: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    try:
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            before = os.fstat(input_file.fileno())
            copied = 0
            while chunk := input_file.read(1024 * 1024):
                digest.update(chunk)
                output_file.write(chunk)
                copied += len(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
            after = os.fstat(input_file.fileno())
    except FileExistsError:
        raise _error(f"private snapshot destination already exists: {destination}") from None

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or copied != before.st_size:
        raise _error(f"{label} changed while its private snapshot was created: {source}")
    return digest.hexdigest()


def _raw_column(frame: pd.DataFrame, field: str, *, label: str) -> Any | None:
    aliases = COLUMN_ALIASES[field]
    matches = [
        column
        for column in frame.columns
        if str(column).strip().lower().replace(" ", "_").replace("-", "_")
        in aliases
    ]
    if len(matches) > 1:
        raise _error(f"{label} has ambiguous columns for {field!r}: {matches}")
    return matches[0] if matches else None


def _raw_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _validate_raw_capture(frame: pd.DataFrame, *, symbol: str) -> None:
    """Reject provider evidence that normal cleaning would repair or discard."""

    label = f"legacy raw capture for {symbol}"
    if frame.empty:
        raise _error(f"{label} cannot be empty")
    columns = {
        field: _raw_column(frame, field, label=label)
        for field in ("date", "open", "high", "low", "close", "volume")
    }
    missing = [
        field
        for field in ("date", "open", "close", "volume")
        if columns[field] is None
    ]
    if missing:
        raise _error(f"{label} is missing required columns: {missing}")

    dates = pd.to_datetime(frame[columns["date"]], errors="coerce", utc=False)
    if dates.isna().any():
        raise _error(f"{label} contains invalid dates")
    normalized_dates = dates.dt.date
    if normalized_dates.duplicated().any():
        raise _error(f"{label} contains duplicate trading dates")

    numeric: dict[str, pd.Series] = {}
    for field in ("open", "close", "volume"):
        values = _raw_numeric(frame[columns[field]])
        if values.isna().any() or not all(
            math.isfinite(float(value)) for value in values
        ):
            raise _error(f"{label} column {field!r} must be finite and numeric")
        numeric[field] = values
    if (numeric["open"] <= 0).any() or (numeric["close"] <= 0).any():
        raise _error(f"{label} open and close prices must be strictly positive")
    if (numeric["volume"] < 0).any():
        raise _error(f"{label} volume must be non-negative")
    if (numeric["volume"] % 1 != 0).any():
        raise _error(f"{label} volume must be a whole integer and cannot be rounded")

    for field in ("high", "low"):
        column = columns[field]
        if column is None:
            continue
        original = frame[column]
        present = original.notna() & original.astype("string").str.strip().ne("")
        values = _raw_numeric(original)
        if values.loc[present].isna().any() or not all(
            math.isfinite(float(value)) for value in values.loc[present]
        ):
            raise _error(
                f"{label} column {field!r} must be finite and numeric when provided"
            )
        if (values.loc[present] <= 0).any():
            raise _error(
                f"{label} column {field!r} must be strictly positive when provided"
            )


def _validate_price_frame(
    frame: pd.DataFrame,
    *,
    expected_provider: str,
    label: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    missing = sorted(set(REQUIRED_OUTPUT_COLUMNS) - set(frame.columns))
    if missing:
        raise _error(f"{label} is missing required columns: {missing}")
    if frame.empty:
        raise _error(f"{label} cannot be empty")

    validated = frame.loc[:, REQUIRED_OUTPUT_COLUMNS].copy()
    validated["date"] = pd.to_datetime(validated["date"], errors="coerce")
    if validated["date"].isna().any():
        raise _error(f"{label} contains invalid dates")
    validated["ticker"] = validated["ticker"].astype("string").str.strip().str.upper()
    if validated["ticker"].isna().any() or validated["ticker"].eq("").any():
        raise _error(f"{label} contains blank or missing tickers")
    if validated.duplicated(["ticker", "date"]).any():
        raise _error(f"{label} contains duplicate ticker/date rows")

    for column in ("open", "close", "volume"):
        numeric = pd.to_numeric(validated[column], errors="coerce")
        if numeric.isna().any() or not all(math.isfinite(float(value)) for value in numeric):
            raise _error(f"{label} column {column!r} must be finite and numeric")
        validated[column] = numeric
    if (validated[["open", "close"]] <= 0).any().any():
        raise _error(f"{label} open and close prices must be strictly positive")
    if (validated["volume"] < 0).any():
        raise _error(f"{label} volume must be non-negative")
    for column in ("high", "low"):
        original = validated[column]
        present = original.notna() & original.astype("string").str.strip().ne("")
        numeric = pd.to_numeric(original, errors="coerce")
        if numeric.loc[present].isna().any() or not all(
            math.isfinite(float(value)) for value in numeric.loc[present]
        ):
            raise _error(
                f"{label} column {column!r} must be numeric when provided"
            )
        if (numeric.loc[present] <= 0).any():
            raise _error(
                f"{label} column {column!r} must be strictly positive when provided"
            )
        validated[column] = numeric

    sources = validated["source"].astype("string")
    if sources.isna().any() or sources.str.strip().eq("").any():
        raise _error(f"{label} contains blank or missing source values")
    observed_sources = sorted(sources.str.strip().unique().tolist())
    blocked_sources = [
        source
        for source in observed_sources
        if any(marker in source.lower() for marker in _NON_REAL_SOURCE_MARKERS)
    ]
    if blocked_sources:
        raise _error(
            f"{label} contains sample, synthetic, fixture, mock, or test sources: "
            f"{blocked_sources}"
        )
    if observed_sources != [expected_provider]:
        raise _error(
            f"{label} source values must all be exactly {expected_provider!r}; "
            f"found {observed_sources}"
        )
    validated["source"] = sources.str.strip()

    coverage: dict[str, dict[str, Any]] = {}
    for symbol, rows in validated.groupby("ticker", sort=True):
        coverage[str(symbol)] = {
            "row_count": int(len(rows)),
            "date_start": rows["date"].min().strftime("%Y-%m-%d"),
            "date_end": rows["date"].max().strftime("%Y-%m-%d"),
            "source": expected_provider,
        }
    return validated, coverage


def _validate_legacy_metadata(
    metadata: Mapping[str, Any],
    *,
    coverage: Mapping[str, Mapping[str, Any]],
    row_count: int,
    date_start: str,
    date_end: str,
    expected_provider: str,
    required_symbol: str,
) -> dict[str, Any]:
    provider = metadata.get("provider")
    if provider != expected_provider:
        raise _error(
            f"legacy metadata provider must be exactly {expected_provider!r}; "
            f"found {provider!r}"
        )

    requested_raw = metadata.get("symbols_requested")
    if not isinstance(requested_raw, list) or not requested_raw:
        raise _error("legacy metadata symbols_requested must be a non-empty list")
    requested = [str(value).strip().upper() for value in requested_raw]
    if any(not value for value in requested):
        raise _error("legacy metadata symbols_requested contains a blank symbol")
    if len(set(requested)) != len(requested):
        raise _error("legacy metadata symbols_requested contains duplicates")
    if required_symbol not in requested:
        raise _error(
            f"legacy metadata symbols_requested does not include {required_symbol}"
        )
    if set(requested) != set(coverage):
        raise _error(
            "legacy metadata requested-symbol coverage does not match the price CSV"
        )

    raw_counts_source = _mapping(
        metadata.get("raw_row_counts"), "legacy metadata raw_row_counts"
    )
    raw_counts: dict[str, int] = {}
    for symbol in requested:
        try:
            raw_count = _json_integer(
                raw_counts_source[symbol],
                f"legacy metadata raw_row_counts[{symbol!r}]",
            )
        except KeyError as exc:
            raise _error(
                f"legacy metadata is missing raw acquisition evidence for {symbol}"
            ) from exc
        if raw_count <= 0:
            raise _error(
                f"legacy metadata raw acquisition evidence for {symbol} must be positive"
            )
        if raw_count < int(coverage[symbol]["row_count"]):
            raise _error(
                f"legacy raw row count for {symbol} is smaller than cleaned coverage"
            )
        raw_counts[symbol] = raw_count

    errors = _mapping(metadata.get("errors"), "legacy metadata errors")
    error_symbols = {str(key).strip().upper() for key in errors}
    if required_symbol in error_symbols:
        raise _error(f"legacy metadata records a provider error for {required_symbol}")

    try:
        claimed_rows = _json_integer(
            metadata["prices_rows"], "legacy metadata prices_rows"
        )
    except KeyError as exc:
        raise _error("legacy metadata prices_rows is missing or invalid") from exc
    if claimed_rows != row_count:
        raise _error(
            f"legacy metadata price row count {claimed_rows} does not match CSV {row_count}"
        )

    claimed_range = _mapping(metadata.get("date_range"), "legacy metadata date_range")
    if {
        "start": str(claimed_range.get("start", "")),
        "end": str(claimed_range.get("end", "")),
    } != {"start": date_start, "end": date_end}:
        raise _error("legacy metadata date range does not match the price CSV")

    generated_at = metadata.get("generated_at")
    original_generated_at = (
        _parse_timestamp(generated_at, "legacy metadata generated_at")
        if generated_at is not None
        else None
    )
    return {
        "provider": expected_provider,
        "symbols_requested": requested,
        "raw_row_counts": raw_counts,
        "errors": dict(errors),
        "original_generated_at": original_generated_at,
    }


def _discover_raw_files(
    *,
    legacy_prices_path: Path,
    symbols: list[str],
    provider: str,
    legacy_raw_dir: Path | None,
) -> tuple[Path | None, dict[str, Path]]:
    explicit = legacy_raw_dir is not None
    raw_root = (
        legacy_raw_dir.expanduser().resolve()
        if legacy_raw_dir is not None
        else legacy_prices_path.parent.parent / "raw"
    )
    if not raw_root.is_dir():
        if explicit:
            raise _error(f"legacy raw directory does not exist: {raw_root}")
        return None, {}

    discovered: dict[str, Path] = {}
    for symbol in symbols:
        filename = f"{symbol}_{provider}_raw.csv"
        matches = sorted(path.resolve() for path in raw_root.rglob(filename) if path.is_file())
        if len(matches) > 1:
            raise _error(f"multiple raw captures found for {symbol}: {matches}")
        if matches:
            discovered[symbol] = matches[0]

    if discovered and set(discovered) != set(symbols):
        missing = sorted(set(symbols) - set(discovered))
        raise _error(
            "partial legacy raw evidence is ambiguous; missing provider captures for "
            f"{missing}"
        )
    if explicit and not discovered:
        raise _error(
            f"no exact *_{{provider}}_raw.csv captures found under {raw_root}"
        )
    return raw_root.resolve(), discovered


def _normalized_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.loc[:, REQUIRED_OUTPUT_COLUMNS].copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    normalized["ticker"] = normalized["ticker"].astype(str).str.strip().str.upper()
    normalized["source"] = normalized["source"].astype(str).str.strip()
    for column in ("open", "high", "low", "close", "volume"):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    return normalized.sort_values(["ticker", "date"], kind="mergesort").reset_index(
        drop=True
    )


def _rebuild_from_raw(
    *,
    raw_files: Mapping[str, Path],
    original_raw_files: Mapping[str, Path],
    raw_hashes: Mapping[str, str],
    raw_row_counts: Mapping[str, int],
    provider: str,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    artifacts: dict[str, dict[str, Any]] = {}
    for symbol in sorted(raw_files):
        raw_path = raw_files[symbol]
        raw = pd.read_csv(raw_path)
        if len(raw) != raw_row_counts[symbol]:
            raise _error(
                f"raw capture row count for {symbol} does not match legacy metadata"
            )
        _validate_raw_capture(raw, symbol=symbol)
        cleaned = standardize_price_frame(raw, ticker=symbol, source=provider)
        if len(cleaned) != len(raw):
            raise _error(
                f"raw capture for {symbol} does not have one-to-one raw-to-cleaned "
                "row accounting"
            )
        frames.append(cleaned)
        original_path = original_raw_files[symbol]
        artifacts[symbol] = {
            "path": str(original_path),
            "filename": original_path.name,
            "sha256": raw_hashes[symbol],
            "row_count": int(len(raw)),
            "cleaned_row_count": int(len(cleaned)),
            "date_start": pd.to_datetime(cleaned["date"]).min().strftime("%Y-%m-%d"),
            "date_end": pd.to_datetime(cleaned["date"]).max().strftime("%Y-%m-%d"),
        }
    combined = combine_price_frames(frames)
    if len(combined) != sum(record["row_count"] for record in artifacts.values()):
        raise _error("combined raw reconstruction is not one-to-one with raw evidence")
    return combined, artifacts


def _prices_artifact(
    *,
    path: Path,
    provider: str,
    coverage: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": LEGACY_CACHE_SCHEMA_VERSION,
        "status": "complete",
        "path": path.name,
        "sha256": sha256_file(path),
        "provider": provider,
        "frequency": "daily",
        "row_count": int(sum(int(value["row_count"]) for value in coverage.values())),
        "symbols": {key: dict(value) for key, value in coverage.items()},
    }


def migrate_legacy_psx_cache(
    legacy_prices_path: str | Path,
    legacy_metadata_path: str | Path,
    output_dir: str | Path,
    *,
    legacy_raw_dir: str | Path | None = None,
    expected_provider: str = "psx_dps",
    required_symbol: str = "OGDC",
) -> dict[str, Any]:
    """Create one immutable verified-legacy bundle without changing originals."""

    prices_source = Path(legacy_prices_path).expanduser().resolve()
    metadata_source = Path(legacy_metadata_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    raw_dir = (
        Path(legacy_raw_dir).expanduser().resolve()
        if legacy_raw_dir is not None
        else None
    )
    provider = str(expected_provider).strip()
    symbol = str(required_symbol).strip().upper()
    if not provider or provider.lower() != provider:
        raise _error("expected_provider must be a non-empty canonical lowercase value")
    if any(marker in provider for marker in _NON_REAL_SOURCE_MARKERS):
        raise _error("expected_provider cannot identify sample or test data")
    if not symbol:
        raise _error("required_symbol cannot be blank")
    if destination.exists():
        raise FileExistsError(
            f"Legacy cache destination already exists; refusing to overwrite: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    snapshots = stage / ".source-snapshots"
    try:
        original_prices_hash = _snapshot_and_hash(
            prices_source,
            snapshots / "legacy_prices.csv",
            label="legacy price CSV",
        )
        original_metadata_hash = _snapshot_and_hash(
            metadata_source,
            snapshots / "legacy_pipeline_metadata.json",
            label="legacy pipeline metadata",
        )
        legacy_prices = pd.read_csv(snapshots / "legacy_prices.csv")
        validated_prices, original_coverage = _validate_price_frame(
            legacy_prices,
            expected_provider=provider,
            label="legacy price CSV",
        )
        dates = validated_prices["date"]
        global_start = dates.min().strftime("%Y-%m-%d")
        global_end = dates.max().strftime("%Y-%m-%d")
        legacy_metadata = _load_json(
            snapshots / "legacy_pipeline_metadata.json",
            label="legacy pipeline metadata",
        )
        evidence = _validate_legacy_metadata(
            legacy_metadata,
            coverage=original_coverage,
            row_count=len(validated_prices),
            date_start=global_start,
            date_end=global_end,
            expected_provider=provider,
            required_symbol=symbol,
        )

        raw_root, original_raw_files = _discover_raw_files(
            legacy_prices_path=prices_source,
            symbols=evidence["symbols_requested"],
            provider=provider,
            legacy_raw_dir=raw_dir,
        )
        if not original_raw_files:
            raise _error(
                "complete legacy raw captures are required; processed-only evidence "
                "cannot be certified as verified real PSX data"
            )
        raw_files: dict[str, Path] = {}
        raw_hashes: dict[str, str] = {}
        for raw_symbol, original_raw_path in original_raw_files.items():
            snapshot_path = snapshots / "raw" / original_raw_path.name
            raw_hashes[raw_symbol] = _snapshot_and_hash(
                original_raw_path,
                snapshot_path,
                label=f"legacy raw capture for {raw_symbol}",
            )
            raw_files[raw_symbol] = snapshot_path

        migrated_prices, raw_artifacts = _rebuild_from_raw(
            raw_files=raw_files,
            original_raw_files=original_raw_files,
            raw_hashes=raw_hashes,
            raw_row_counts=evidence["raw_row_counts"],
            provider=provider,
        )
        rebuilt_validated, rebuilt_coverage = _validate_price_frame(
            migrated_prices,
            expected_provider=provider,
            label="raw-capture reconstruction",
        )
        try:
            pd.testing.assert_frame_equal(
                _normalized_comparison(rebuilt_validated),
                _normalized_comparison(validated_prices),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            raise _error(
                "current cleaning of legacy raw captures does not reproduce the legacy "
                "processed price evidence"
            ) from exc
        strategy = "rebuild_from_legacy_raw_captures"
        provenance_strength = "raw_capture_rebuild_matches_legacy_processed_cache"
        coverage = rebuilt_coverage
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise

    try:
        migrated_prices_path = stage / PRICES_FILENAME
        migrated_prices.to_csv(migrated_prices_path, index=False)
        migrated_frame = pd.read_csv(migrated_prices_path)
        _, migrated_coverage = _validate_price_frame(
            migrated_frame,
            expected_provider=provider,
            label="migrated price CSV",
        )
        if migrated_coverage != coverage:
            raise _error("migrated price coverage changed while the bundle was written")

        preserved_metadata_path = stage / LEGACY_METADATA_FILENAME
        shutil.copyfile(
            snapshots / "legacy_pipeline_metadata.json", preserved_metadata_path
        )
        if sha256_file(preserved_metadata_path) != original_metadata_hash:
            raise _error("preserved legacy metadata is not byte-for-byte identical")

        migration_timestamp = utc_now_z()
        artifact = _prices_artifact(
            path=migrated_prices_path,
            provider=provider,
            coverage=migrated_coverage,
        )
        original_artifacts: dict[str, Any] = {
            "prices": {
                "path": str(prices_source),
                "sha256": original_prices_hash,
                "row_count": int(len(validated_prices)),
            },
            "pipeline_metadata": {
                "path": str(metadata_source),
                "sha256": original_metadata_hash,
                "generated_at": evidence["original_generated_at"],
            },
            "raw_directory": str(raw_root) if raw_root is not None else None,
            "raw_captures": raw_artifacts,
        }
        validation_record: dict[str, Any] = {
            "required_columns_verified": True,
            "numeric_open_close_volume_verified": True,
            "positive_prices_and_nonnegative_volume_verified": True,
            "unique_ticker_date_verified": True,
            "single_expected_real_provider_verified": True,
            "legacy_metadata_counts_and_dates_verified": True,
            "per_symbol_coverage": migrated_coverage,
            "raw_rebuild_matches_legacy_processed_cache": True,
            "raw_one_to_one_row_accounting_verified": True,
        }
        generated_metadata: dict[str, Any] = {
            "schema_version": LEGACY_CACHE_SCHEMA_VERSION,
            "producer": {
                "component": LEGACY_CACHE_PRODUCER,
                "schema_version": LEGACY_CACHE_SCHEMA_VERSION,
                "status": "complete",
            },
            "generated_at": evidence["original_generated_at"],
            "migration_timestamp": migration_timestamp,
            "provider": provider,
            "original_provider": provider,
            "acquisition_mode": LEGACY_ACQUISITION_MODE,
            "live_fetch_succeeded": False,
            "live_acquisition_performed": False,
            "legacy_metadata_preserved": True,
            "symbols_requested": evidence["symbols_requested"],
            "raw_row_counts": evidence["raw_row_counts"],
            "errors": evidence["errors"],
            "prices_rows": int(len(migrated_frame)),
            "date_range": {"start": global_start, "end": global_end},
            "prices_artifact": artifact,
            "files": {
                "prices": PRICES_FILENAME,
                "legacy_metadata": LEGACY_METADATA_FILENAME,
                "migration_manifest": MIGRATION_MANIFEST_FILENAME,
            },
            "migration": {
                "strategy": strategy,
                "provenance_strength": provenance_strength,
                "original_generated_at": evidence["original_generated_at"],
                "limitations": list(_LIMITATIONS),
            },
            "legacy_evidence": {
                "original_artifacts": original_artifacts,
                "validation": validation_record,
            },
        }
        generated_metadata_path = stage / PIPELINE_METADATA_FILENAME
        write_json(generated_metadata, generated_metadata_path)

        manifest: dict[str, Any] = {
            "schema_version": LEGACY_CACHE_SCHEMA_VERSION,
            "producer": {
                "component": LEGACY_CACHE_PRODUCER,
                "schema_version": LEGACY_CACHE_SCHEMA_VERSION,
                "status": "complete",
            },
            "status": "complete",
            "migration_timestamp": migration_timestamp,
            "acquisition_mode": LEGACY_ACQUISITION_MODE,
            "live_fetch_succeeded": False,
            "live_acquisition_performed": False,
            "original_provider": provider,
            "legacy_metadata_preserved": True,
            "migration_strategy": strategy,
            "provenance_strength": provenance_strength,
            "limitations": list(_LIMITATIONS),
            "original_artifacts": original_artifacts,
            "migrated_artifacts": {
                "prices": {
                    "path": PRICES_FILENAME,
                    "sha256": artifact["sha256"],
                    "row_count": artifact["row_count"],
                },
                "pipeline_metadata": {
                    "path": PIPELINE_METADATA_FILENAME,
                    "sha256": sha256_file(generated_metadata_path),
                },
                "legacy_pipeline_metadata": {
                    "path": LEGACY_METADATA_FILENAME,
                    "sha256": sha256_file(preserved_metadata_path),
                },
            },
            "validation": validation_record,
        }
        manifest_path = stage / MIGRATION_MANIFEST_FILENAME
        write_json(manifest, manifest_path)

        shutil.rmtree(snapshots)
        validate_verified_legacy_bundle(
            migrated_prices_path,
            metadata_path=generated_metadata_path,
            required_symbol=symbol,
        )
        if _stable_sha256(prices_source, label="legacy price CSV") != original_prices_hash:
            raise _error("original legacy price CSV changed during migration")
        if (
            _stable_sha256(metadata_source, label="legacy pipeline metadata")
            != original_metadata_hash
        ):
            raise _error("original legacy metadata changed during migration")
        for raw_record in raw_artifacts.values():
            raw_path = Path(str(raw_record["path"]))
            if _stable_sha256(raw_path, label="legacy raw capture") != raw_record["sha256"]:
                raise _error(f"original legacy raw capture changed during migration: {raw_path}")
        if destination.exists():
            raise FileExistsError(
                f"Legacy cache destination already exists; refusing to overwrite: {destination}"
            )
        os.rename(stage, destination)
        manifest["output_directory"] = str(destination)
        return manifest
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _verified_artifact(
    artifacts: Mapping[str, Any],
    name: str,
    *,
    expected_filename: str,
    actual_path: Path,
) -> str:
    record = _mapping(artifacts.get(name), f"migration manifest {name}")
    if record.get("path") != expected_filename:
        raise _error(f"migration manifest {name} path is invalid")
    expected_hash = str(record.get("sha256", "")).strip().lower()
    if len(expected_hash) != 64 or any(char not in "0123456789abcdef" for char in expected_hash):
        raise _error(f"migration manifest {name} SHA-256 is invalid")
    actual_hash = _stable_sha256(actual_path, label=name)
    if actual_hash != expected_hash:
        raise _error(f"{name} SHA-256 does not match migration manifest")
    return actual_hash


def validate_verified_legacy_bundle(
    prices_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    required_symbol: str = "OGDC",
    frame: pd.DataFrame | None = None,
    prices_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a complete migrated bundle and return normalized provenance labels."""

    prices = Path(prices_path).expanduser().resolve()
    metadata = (
        Path(metadata_path).expanduser().resolve()
        if metadata_path is not None
        else prices.parent / PIPELINE_METADATA_FILENAME
    )
    bundle_dir = prices.parent
    manifest_path = bundle_dir / MIGRATION_MANIFEST_FILENAME
    preserved_metadata_path = bundle_dir / LEGACY_METADATA_FILENAME

    metadata_hash = _stable_sha256(metadata, label="migrated pipeline metadata")
    manifest_hash = _stable_sha256(manifest_path, label="migration manifest")
    manifest = _load_json(manifest_path, label="migration manifest")
    if _stable_sha256(manifest_path, label="migration manifest") != manifest_hash:
        raise _error("migration manifest changed while it was read")
    if manifest.get("schema_version") != LEGACY_CACHE_SCHEMA_VERSION:
        raise _error("migration manifest must use schema version 1.0")
    expected_producer = {
        "component": LEGACY_CACHE_PRODUCER,
        "schema_version": LEGACY_CACHE_SCHEMA_VERSION,
        "status": "complete",
    }
    if manifest.get("producer") != expected_producer or manifest.get("status") != "complete":
        raise _error("migration manifest does not contain the verified legacy producer")
    if manifest.get("acquisition_mode") != LEGACY_ACQUISITION_MODE:
        raise _error("migration manifest acquisition mode is invalid")
    if manifest.get("live_fetch_succeeded") is not False or manifest.get(
        "live_acquisition_performed"
    ) is not False:
        raise _error("a legacy migration cannot claim a successful live fetch")
    if manifest.get("legacy_metadata_preserved") is not True:
        raise _error("migration manifest must attest preserved legacy metadata")
    if manifest.get("original_provider") != "psx_dps":
        raise _error("migration manifest must identify original provider psx_dps")
    migration_strategy = manifest.get("migration_strategy")
    if migration_strategy != "rebuild_from_legacy_raw_captures":
        raise _error(
            "verified legacy bundles require raw-capture reconstruction; "
            "processed-only migration is not certifiable"
        )
    if manifest.get("provenance_strength") != (
        "raw_capture_rebuild_matches_legacy_processed_cache"
    ):
        raise _error("migration manifest provenance strength is inconsistent")
    limitations = manifest.get("limitations")
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(value, str) and value.strip() for value in limitations
    ):
        raise _error("migration manifest must disclose non-empty provenance limitations")
    manifest_timestamp = _parse_timestamp(
        manifest.get("migration_timestamp"), "manifest migration timestamp"
    )

    artifacts = _mapping(
        manifest.get("migrated_artifacts"), "migration manifest migrated_artifacts"
    )
    migrated_prices_record = _mapping(
        artifacts.get("prices"), "migration manifest prices"
    )
    manifest_prices_hash = _verified_artifact(
        artifacts,
        "prices",
        expected_filename=PRICES_FILENAME,
        actual_path=prices,
    )
    verified_metadata_hash = _verified_artifact(
        artifacts,
        "pipeline_metadata",
        expected_filename=PIPELINE_METADATA_FILENAME,
        actual_path=metadata,
    )
    if verified_metadata_hash != metadata_hash:
        raise _error("migrated pipeline metadata changed while it was validated")
    preserved_hash = _verified_artifact(
        artifacts,
        "legacy_pipeline_metadata",
        expected_filename=LEGACY_METADATA_FILENAME,
        actual_path=preserved_metadata_path,
    )

    generated = _load_json(metadata, label="migrated pipeline metadata")
    if _stable_sha256(metadata, label="migrated pipeline metadata") != metadata_hash:
        raise _error("migrated pipeline metadata changed while it was read")
    if generated.get("schema_version") != LEGACY_CACHE_SCHEMA_VERSION:
        raise _error("migrated pipeline metadata must use schema version 1.0")
    if generated.get("producer") != expected_producer:
        raise _error("migrated metadata producer is not the verified legacy migrator")
    provider = str(generated.get("provider", "")).strip()
    if provider != "psx_dps" or generated.get("original_provider") != provider:
        raise _error("verified legacy metadata must identify original provider psx_dps")
    if generated.get("acquisition_mode") != LEGACY_ACQUISITION_MODE:
        raise _error("migrated metadata acquisition mode is invalid")
    if generated.get("live_fetch_succeeded") is not False or generated.get(
        "live_acquisition_performed"
    ) is not False:
        raise _error("migrated metadata cannot claim a successful live fetch")
    if generated.get("legacy_metadata_preserved") is not True:
        raise _error("migrated metadata must attest preserved legacy metadata")
    _parse_timestamp(generated.get("migration_timestamp"), "migration timestamp")
    if generated.get("generated_at") is not None:
        _parse_timestamp(generated.get("generated_at"), "original generation timestamp")

    observed_prices_hash = prices_sha256 or _stable_sha256(
        prices, label="migrated price CSV"
    )
    if observed_prices_hash != manifest_prices_hash:
        raise _error("migrated price CSV changed while it was validated")
    observed_frame = frame.copy() if frame is not None else pd.read_csv(prices)
    validated, coverage = _validate_price_frame(
        observed_frame,
        expected_provider=provider,
        label="migrated price CSV",
    )
    if _json_integer(
        migrated_prices_record.get("row_count"),
        "migration manifest migrated price row_count",
    ) != len(validated):
        raise _error("migrated price row count does not match the price CSV")
    dates = validated["date"]
    evidence = _validate_legacy_metadata(
        generated,
        coverage=coverage,
        row_count=len(validated),
        date_start=dates.min().strftime("%Y-%m-%d"),
        date_end=dates.max().strftime("%Y-%m-%d"),
        expected_provider=provider,
        required_symbol=str(required_symbol).strip().upper(),
    )
    artifact = _mapping(generated.get("prices_artifact"), "prices_artifact")
    expected_artifact = _prices_artifact(
        path=prices,
        provider=provider,
        coverage=coverage,
    )
    if dict(artifact) != expected_artifact:
        raise _error("migrated prices_artifact does not match the exact price CSV")
    files = _mapping(generated.get("files"), "migrated metadata files")
    if files != {
        "prices": PRICES_FILENAME,
        "legacy_metadata": LEGACY_METADATA_FILENAME,
        "migration_manifest": MIGRATION_MANIFEST_FILENAME,
    }:
        raise _error("migrated metadata file bindings are invalid")

    preserved_legacy = _load_json(
        preserved_metadata_path, label="preserved legacy pipeline metadata"
    )
    if (
        _stable_sha256(
            preserved_metadata_path, label="preserved legacy pipeline metadata"
        )
        != preserved_hash
    ):
        raise _error("preserved legacy metadata changed while it was read")
    original_artifacts = _mapping(
        manifest.get("original_artifacts"), "migration manifest original_artifacts"
    )
    original_prices = _mapping(
        original_artifacts.get("prices"), "migration manifest original prices"
    )
    original_prices_hash = str(original_prices.get("sha256", "")).strip().lower()
    if len(original_prices_hash) != 64 or any(
        character not in "0123456789abcdef" for character in original_prices_hash
    ):
        raise _error("original price SHA-256 is invalid")
    original_prices_location = Path(str(original_prices.get("path", "")))
    if not original_prices_location.is_absolute():
        raise _error("original price path must be absolute evidence")
    original_price_rows = _json_integer(
        original_prices.get("row_count"),
        "migration manifest original price row_count",
    )
    if original_price_rows != len(validated):
        raise _error("original price row count does not match migrated evidence")
    original_metadata = _mapping(
        original_artifacts.get("pipeline_metadata"),
        "migration manifest original pipeline metadata",
    )
    if str(original_metadata.get("sha256", "")).lower() != preserved_hash:
        raise _error("preserved legacy metadata hash does not match original evidence")
    if not Path(str(original_metadata.get("path", ""))).is_absolute():
        raise _error("original metadata path must be absolute evidence")
    preserved_evidence = _validate_legacy_metadata(
        preserved_legacy,
        coverage=coverage,
        row_count=len(validated),
        date_start=dates.min().strftime("%Y-%m-%d"),
        date_end=dates.max().strftime("%Y-%m-%d"),
        expected_provider=provider,
        required_symbol=str(required_symbol).strip().upper(),
    )
    if preserved_evidence["original_generated_at"] != evidence["original_generated_at"]:
        raise _error("original generation timestamp was not preserved truthfully")
    if original_metadata.get("generated_at") != evidence["original_generated_at"]:
        raise _error("manifest did not preserve the original generation timestamp")
    if generated.get("migration_timestamp") != manifest_timestamp:
        raise _error("migration timestamps disagree across the bundle")
    migration = _mapping(generated.get("migration"), "migrated metadata migration")
    if migration.get("strategy") != migration_strategy:
        raise _error("migration strategy disagrees across the bundle")
    if migration.get("provenance_strength") != manifest.get("provenance_strength"):
        raise _error("provenance strength disagrees across the bundle")
    if migration.get("limitations") != limitations:
        raise _error("provenance limitations disagree across the bundle")

    raw_captures = _mapping(
        original_artifacts.get("raw_captures"), "migration manifest raw captures"
    )
    raw_directory = Path(str(original_artifacts.get("raw_directory", "")))
    if not raw_directory.is_absolute():
        raise _error("raw-capture migration must record an absolute raw directory")
    if set(raw_captures) != set(coverage):
        raise _error("raw-capture evidence does not cover every migrated symbol")
    for raw_symbol, raw_value in raw_captures.items():
        raw_record = _mapping(raw_value, f"raw-capture evidence for {raw_symbol}")
        if raw_record.get("filename") != f"{raw_symbol}_{provider}_raw.csv":
            raise _error(f"raw-capture filename is invalid for {raw_symbol}")
        raw_location = Path(str(raw_record.get("path", "")))
        if (
            not raw_location.is_absolute()
            or raw_location.name != raw_record.get("filename")
            or not raw_location.is_relative_to(raw_directory)
        ):
            raise _error(f"raw-capture path is invalid for {raw_symbol}")
        raw_hash = str(raw_record.get("sha256", "")).strip().lower()
        if len(raw_hash) != 64 or any(
            character not in "0123456789abcdef" for character in raw_hash
        ):
            raise _error(f"raw-capture SHA-256 is invalid for {raw_symbol}")
        raw_rows = _json_integer(
            raw_record.get("row_count"),
            f"raw-capture row count for {raw_symbol}",
        )
        cleaned_rows = _json_integer(
            raw_record.get("cleaned_row_count"),
            f"raw-capture cleaned row count for {raw_symbol}",
        )
        if raw_rows != evidence["raw_row_counts"][raw_symbol]:
            raise _error(f"raw-capture row count is invalid for {raw_symbol}")
        if raw_rows != cleaned_rows:
            raise _error(
                f"raw-capture rows are not one-to-one with cleaned rows for {raw_symbol}"
            )
        if cleaned_rows != coverage[raw_symbol]["row_count"]:
            raise _error(f"raw-capture cleaned coverage is invalid for {raw_symbol}")
        if {
            "date_start": raw_record.get("date_start"),
            "date_end": raw_record.get("date_end"),
        } != {
            "date_start": coverage[raw_symbol]["date_start"],
            "date_end": coverage[raw_symbol]["date_end"],
        }:
            raise _error(f"raw-capture date coverage is invalid for {raw_symbol}")

    expected_validation = {
        "required_columns_verified": True,
        "numeric_open_close_volume_verified": True,
        "positive_prices_and_nonnegative_volume_verified": True,
        "unique_ticker_date_verified": True,
        "single_expected_real_provider_verified": True,
        "legacy_metadata_counts_and_dates_verified": True,
        "per_symbol_coverage": coverage,
        "raw_rebuild_matches_legacy_processed_cache": True,
        "raw_one_to_one_row_accounting_verified": True,
    }
    if manifest.get("validation") != expected_validation:
        raise _error("migration manifest validation evidence is inconsistent")
    legacy_evidence = _mapping(
        generated.get("legacy_evidence"), "migrated metadata legacy_evidence"
    )
    if legacy_evidence != {
        "original_artifacts": original_artifacts,
        "validation": expected_validation,
    }:
        raise _error("checksum-bound legacy evidence disagrees with the manifest")

    if _stable_sha256(manifest_path, label="migration manifest") != manifest_hash:
        raise _error("migration manifest changed during validation")
    if _stable_sha256(metadata, label="migrated pipeline metadata") != metadata_hash:
        raise _error("migrated pipeline metadata changed during validation")
    if (
        _stable_sha256(
            preserved_metadata_path, label="preserved legacy pipeline metadata"
        )
        != preserved_hash
    ):
        raise _error("preserved legacy metadata changed during validation")

    return {
        "provider": provider,
        "producer_component": LEGACY_CACHE_PRODUCER,
        "acquisition_mode": LEGACY_ACQUISITION_MODE,
        "live_fetch_succeeded": False,
        "original_provider": provider,
        "legacy_metadata_preserved": True,
        "pipeline_metadata_path": metadata,
        "pipeline_metadata_sha256": metadata_hash,
        "migration_manifest_path": manifest_path,
        "migration_manifest_sha256": manifest_hash,
        "legacy_metadata_path": preserved_metadata_path,
        "legacy_metadata_sha256": preserved_hash,
        "migration_strategy": migration_strategy,
        "provenance_strength": manifest.get("provenance_strength"),
        "limitations": list(limitations),
    }


__all__ = [
    "LEGACY_ACQUISITION_MODE",
    "LEGACY_CACHE_PRODUCER",
    "LEGACY_METADATA_FILENAME",
    "MIGRATION_MANIFEST_FILENAME",
    "PIPELINE_METADATA_FILENAME",
    "PRICES_FILENAME",
    "migrate_legacy_psx_cache",
    "validate_verified_legacy_bundle",
]
