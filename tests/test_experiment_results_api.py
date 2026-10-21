from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import quote

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import paisa.api.main as api_main
from paisa.api.artifact_repository import (
    ACADEMIC_DISCLAIMER,
    ArtifactIntegrityError,
    ArtifactRepository,
    RepositorySettings,
    get_artifact_repository,
)
from paisa.api.main import app
from paisa.api.router_v1 import _assert_no_absolute_local_paths
from paisa.config import get_paths
from paisa.vertical_slice import PriceAcquisition, run_ogdc_vertical_slice


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_RUN_ID = "api-fixture-run"
REAL_RUN_ID = "VSL-20261020-OGDC-main-02"
MONEY_PATTERN = re.compile(r"^-?(?:0|[1-9][0-9]*)\.[0-9]{2}$")


@dataclass(frozen=True)
class GeneratedArtifacts:
    vertical_root: Path
    training_root: Path
    run_directory: Path
    experiment_ids: tuple[str, ...]


def _price_fixture(rows: int = 160) -> pd.DataFrame:
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
            "source": "deterministic_fixture",
        }
    )


def _offline_acquisition(*, symbol: str, start: str, work_dir: Path) -> PriceAcquisition:
    assert symbol == "OGDC"
    assert start <= "2025-01-01"
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    frame = _price_fixture()
    prices_path = work_dir / "OGDC_fixture_prices.csv"
    frame.to_csv(prices_path, index=False)
    return PriceAcquisition(
        prices_path=prices_path,
        source="deterministic_fixture",
        mode="fixture_test",
        date_start="2025-01-01",
        date_end=pd.Timestamp(frame["date"].max()).strftime("%Y-%m-%d"),
        row_count=len(frame),
        is_real_data=False,
        live_error=None,
    )


@pytest.fixture(scope="module")
def generated_artifacts(
    tmp_path_factory: pytest.TempPathFactory,
) -> GeneratedArtifacts:
    root = tmp_path_factory.mktemp("experiment-results-api")
    vertical_root = root / "vertical-slices"
    training_root = root / "model-training"
    summary = run_ogdc_vertical_slice(
        output_dir=vertical_root,
        model_output_dir=training_root,
        cached_prices_path=root / "missing-cache.csv",
        vertical_slice_run_id=FIXTURE_RUN_ID,
        acquire_fn=_offline_acquisition,
    )
    run_directory = vertical_root / FIXTURE_RUN_ID
    experiment_ids = tuple(
        sorted(
            path.name
            for path in (run_directory / "experiments").iterdir()
            if path.is_dir()
        )
    )
    assert summary["status"] == "PASS"
    assert len(experiment_ids) == 4
    return GeneratedArtifacts(
        vertical_root=vertical_root,
        training_root=training_root,
        run_directory=run_directory,
        experiment_ids=experiment_ids,
    )


def _settings(
    artifacts: GeneratedArtifacts,
    *,
    vertical_root: Path | None = None,
    default_run_id: str | None = FIXTURE_RUN_ID,
) -> RepositorySettings:
    return RepositorySettings(
        repository_root=REPOSITORY_ROOT,
        vertical_slice_root=vertical_root or artifacts.vertical_root,
        model_training_root=artifacts.training_root,
        default_vertical_slice_run_id=default_run_id,
    )


@contextmanager
def _api_client(settings: RepositorySettings) -> Iterator[TestClient]:
    repository = ArtifactRepository(settings)
    previous = app.dependency_overrides.get(get_artifact_repository)

    def override_repository() -> ArtifactRepository:
        return repository

    app.dependency_overrides[get_artifact_repository] = override_repository
    try:
        with TestClient(app) as client:
            yield client
    finally:
        if previous is None:
            app.dependency_overrides.pop(get_artifact_repository, None)
        else:
            app.dependency_overrides[get_artifact_repository] = previous


def _copy_valid_run(artifacts: GeneratedArtifacts, destination_root: Path) -> Path:
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / FIXTURE_RUN_ID
    shutil.copytree(artifacts.run_directory, destination)
    return destination


def _refresh_vertical_checksum(run_directory: Path, relative_path: str) -> None:
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = run_directory / relative_path
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    entry = next(
        item for item in manifest["artifacts"] if item["path"] == relative_path
    )
    entry["sha256"] = digest
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _price_snapshot_relative(run_directory: Path) -> str:
    manifest = json.loads((run_directory / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest["data"]["input_prices_sha256"]
    matches = [
        item["path"]
        for item in manifest["artifacts"]
        if item["sha256"] == digest
        and item["path"].startswith("input_snapshot/")
        and item["path"].count("/") == 1
        and item["path"].lower().endswith(".csv")
    ]
    assert len(matches) == 1
    return matches[0]


def _read_price_snapshot(run_directory: Path) -> tuple[str, list[str], list[dict[str, str]]]:
    relative = _price_snapshot_relative(run_directory)
    with (run_directory / relative).open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        assert reader.fieldnames is not None
        return relative, list(reader.fieldnames), list(reader)


def _write_and_reseal_price_snapshot(
    run_directory: Path,
    relative: str,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    snapshot_path = run_directory / relative
    with snapshot_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    snapshot_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()

    summary_path = run_directory / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["data_source"]["input_prices_sha256"] = snapshot_digest
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    reference_path = run_directory / "model_training_run_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["input_prices_sha256"] = snapshot_digest
    reference_path.write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data"]["input_prices_sha256"] = snapshot_digest
    updated_digests = {
        relative: snapshot_digest,
        "final_summary.json": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "model_training_run_reference.json": hashlib.sha256(
            reference_path.read_bytes()
        ).hexdigest(),
    }
    for item in manifest["artifacts"]:
        if item["path"] in updated_digests:
            item["sha256"] = updated_digests[item["path"]]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _add_pipeline_price_contract(
    run_directory: Path,
    *,
    mutation: str,
) -> None:
    relative, _fieldnames, rows = _read_price_snapshot(run_directory)
    snapshot_path = run_directory / relative
    snapshot_digest = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    dates = sorted(row["date"] for row in rows)
    metadata = {
        "schema_version": "1.0",
        "prices_artifact": {
            "schema_version": "1.0",
            "status": "complete",
            "frequency": "daily",
            "path": Path(relative).name,
            "sha256": snapshot_digest,
            "row_count": len(rows),
            "symbols": {
                "OGDC": {
                    "row_count": len(rows),
                    "date_start": dates[0],
                    "date_end": dates[-1],
                    "source": "deterministic_fixture",
                }
            },
        },
    }
    contract = metadata["prices_artifact"]
    if mutation == "path-traversal":
        contract["path"] = "../outside.csv"
    elif mutation == "checksum":
        contract["sha256"] = "0" * 64
    elif mutation == "row-count":
        contract["row_count"] = len(rows) + 1
    elif mutation == "symbol-coverage":
        contract["symbols"]["OGDC"]["date_end"] = dates[-2]
    elif mutation != "valid":  # pragma: no cover - test helper misuse
        raise AssertionError(f"unknown pipeline contract mutation: {mutation}")

    metadata_path = run_directory / "input_snapshot" / "pipeline_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = run_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "path": "input_snapshot/pipeline_metadata.json",
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
    )
    manifest["artifacts"].sort(key=lambda item: item["path"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _all_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _all_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_strings(nested)


def _endpoint_urls(run_id: str, experiment_id: str) -> tuple[str, ...]:
    base = f"/api/v1/vertical-slices/{run_id}"
    experiment = f"{base}/experiments/{experiment_id}"
    return (
        "/api/v1/vertical-slices",
        base,
        f"{base}/summary",
        f"{base}/stocks/OGDC/prices?limit=7&offset=1",
        f"{base}/leaderboard",
        f"{base}/models",
        f"{base}/portfolios",
        f"{base}/equity-curves?limit=7&offset=1",
        f"{base}/drawdowns?limit=7&offset=1",
        f"{base}/source",
        f"{base}/experiments",
        experiment,
        f"{experiment}/trades?limit=2&offset=0",
        f"{experiment}/snapshots?limit=2&offset=1",
        f"{experiment}/benchmarks?limit=2&offset=1",
        "/api/v1/leaderboard",
    )


def test_fixture_artifacts_support_every_v1_endpoint_without_exposing_paths(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    experiment_id = generated_artifacts.experiment_ids[0]
    with _api_client(_settings(generated_artifacts)) as client:
        for url in _endpoint_urls(FIXTURE_RUN_ID, experiment_id):
            response = client.get(url)
            assert response.status_code == 200, (url, response.text)
            payload = response.json()
            assert payload["disclaimer"] == ACADEMIC_DISCLAIMER
            assert str(generated_artifacts.vertical_root) not in response.text
            assert str(generated_artifacts.training_root) not in response.text
            assert not any(value.startswith("/") for value in _all_strings(payload))
            assert "NaN" not in response.text
            assert "Infinity" not in response.text
            assert "Traceback" not in response.text


def test_run_scoped_price_history_returns_explicit_safe_schema(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    _relative, _fieldnames, source_rows = _read_price_snapshot(
        generated_artifacts.run_directory
    )
    expected = sorted(source_rows, key=lambda row: row["date"])[0]
    url = f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/ogdc/prices"

    with _api_client(_settings(generated_artifacts)) as client:
        response = client.get(f"{url}?limit=5&offset=0")
        default_page = client.get(url)

    assert response.status_code == default_page.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == "1.0"
    assert payload["run_id"] == FIXTURE_RUN_ID
    assert payload["ticker"] == "OGDC"
    assert payload["source_mode"] == "fixture_test"
    assert payload["data_date_range"] == {
        "start": min(row["date"] for row in source_rows),
        "end": max(row["date"] for row in source_rows),
    }
    assert payload["total"] == len(source_rows)
    assert payload["limit"] == 5
    assert payload["offset"] == 0
    assert len(payload["rows"]) == 5
    first = payload["rows"][0]
    assert first["date"] == expected["date"]
    assert Decimal(first["open"]) == Decimal(expected["open"])
    assert Decimal(first["close"]) == Decimal(expected["close"])
    assert len(first["open"].partition(".")[2]) >= 2
    assert len(first["close"].partition(".")[2]) >= 2
    assert first["high"] is None
    assert first["low"] is None
    assert isinstance(first["volume"], int)
    assert first["source"] == "deterministic_fixture"
    assert payload["disclaimer"] == ACADEMIC_DISCLAIMER
    assert default_page.json()["limit"] == 500
    assert "NaN" not in response.text
    assert "Infinity" not in response.text
    assert str(generated_artifacts.vertical_root) not in response.text
    assert str(generated_artifacts.training_root) not in response.text
    assert not any(value.startswith("/") for value in _all_strings(payload))


def test_price_history_is_deterministically_ordered_and_paginated(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "reordered" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative, fieldnames, rows = _read_price_snapshot(copied_run)
    _write_and_reseal_price_snapshot(
        copied_run, relative, fieldnames, list(reversed(rows))
    )
    base = f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        first = client.get(f"{base}?limit=5000")
        second = client.get(f"{base}?limit=5000")
        page = client.get(f"{base}?limit=4&offset=3")

    assert first.status_code == second.status_code == page.status_code == 200
    full_rows = first.json()["rows"]
    dates = [row["date"] for row in full_rows]
    assert dates == sorted(dates)
    assert len(dates) == len(set(dates))
    assert second.json()["rows"] == full_rows
    assert page.json()["total"] == len(full_rows)
    assert page.json()["rows"] == full_rows[3:7]


def test_price_history_applies_inclusive_date_filters_before_pagination(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    base = f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
    with _api_client(_settings(generated_artifacts)) as client:
        full = client.get(f"{base}?limit=5000")
        assert full.status_code == 200
        dates = [row["date"] for row in full.json()["rows"]]
        start = dates[20]
        end = dates[39]
        start_only = client.get(f"{base}?limit=5000&start_date={start}")
        end_only = client.get(f"{base}?limit=5000&end_date={end}")
        combined = client.get(
            f"{base}?limit=5000&start_date={start}&end_date={end}"
        )
        paged = client.get(
            f"{base}?limit=3&offset=2&start_date={start}&end_date={end}"
        )

    assert start_only.status_code == end_only.status_code == 200
    assert combined.status_code == paged.status_code == 200
    assert start_only.json()["total"] == len(dates) - 20
    assert start_only.json()["rows"][0]["date"] == start
    assert end_only.json()["total"] == 40
    assert end_only.json()["rows"][-1]["date"] == end
    assert combined.json()["total"] == 20
    assert [row["date"] for row in combined.json()["rows"]] == dates[20:40]
    assert paged.json()["total"] == 20
    assert [row["date"] for row in paged.json()["rows"]] == dates[22:25]


def test_price_history_query_validation_is_documented_and_stable(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    base = f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
    with _api_client(_settings(generated_artifacts)) as client:
        invalid_range = client.get(
            f"{base}?start_date=2025-03-01&end_date=2025-02-01"
        )
        invalid_date = client.get(f"{base}?start_date=not-a-date")
        invalid_limits = [
            client.get(f"{base}?limit=0"),
            client.get(f"{base}?limit=5001"),
            client.get(f"{base}?offset=-1"),
        ]

    assert invalid_range.status_code == 400
    assert invalid_range.json()["detail"] == "start_date must be on or before end_date"
    assert invalid_date.status_code == 422
    assert all(response.status_code == 422 for response in invalid_limits)


def test_price_history_distinguishes_unknown_ticker_run_and_incomplete_runs(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "statuses" / "vertical"
    _copy_valid_run(generated_artifacts, vertical_root)
    (vertical_root / "incomplete-run").mkdir()
    failed = vertical_root / "failed-run"
    failed.mkdir()
    (failed / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "vertical_slice_run_id": "failed-run",
                "status": "FAIL",
            }
        ),
        encoding="utf-8",
    )
    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        unknown_ticker = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/ZZZZ/prices"
        )
        unknown_run = client.get(
            "/api/v1/vertical-slices/unknown-run/stocks/OGDC/prices"
        )
        incomplete = client.get(
            "/api/v1/vertical-slices/incomplete-run/stocks/OGDC/prices"
        )
        failed_response = client.get(
            "/api/v1/vertical-slices/failed-run/stocks/OGDC/prices"
        )

    assert unknown_ticker.status_code == unknown_run.status_code == 404
    assert incomplete.status_code == failed_response.status_code == 409
    for response in (unknown_ticker, unknown_run, incomplete, failed_response):
        assert str(vertical_root) not in response.text


@pytest.mark.parametrize("mutation", ["snapshot-bytes", "inventory-checksum"])
def test_price_history_rejects_corruption_and_checksum_mismatch(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    mutation: str,
) -> None:
    vertical_root = tmp_path / mutation / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative = _price_snapshot_relative(copied_run)
    if mutation == "snapshot-bytes":
        snapshot = copied_run / relative
        snapshot.write_bytes(snapshot.read_bytes() + b"\n")
    else:
        manifest_path = copied_run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        next(
            item for item in manifest["artifacts"] if item["path"] == relative
        )["sha256"] = "0" * 64
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
    assert response.status_code == 409
    assert "integrity" in response.json()["detail"].lower()
    assert str(vertical_root) not in response.text


@pytest.mark.parametrize("delete_snapshot", [False, True])
def test_price_history_rejects_missing_or_uninventoried_snapshot(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    delete_snapshot: bool,
) -> None:
    case = "missing-snapshot" if delete_snapshot else "uninventoried-snapshot"
    vertical_root = tmp_path / case / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative = _price_snapshot_relative(copied_run)
    if delete_snapshot:
        (copied_run / relative).unlink()
    manifest_path = copied_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = [
        item for item in manifest["artifacts"] if item["path"] != relative
    ]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
    assert response.status_code == 409
    assert "snapshot" in response.json()["detail"].lower() or "integrity" in response.json()[
        "detail"
    ].lower()


@pytest.mark.parametrize(
    "mutation", ["path-traversal", "checksum", "row-count", "symbol-coverage"]
)
def test_price_history_rejects_invalid_input_snapshot_contract(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    mutation: str,
) -> None:
    vertical_root = tmp_path / f"contract-{mutation}" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    _add_pipeline_price_contract(copied_run, mutation=mutation)

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
    assert response.status_code == 409
    assert str(vertical_root) not in response.text


def test_price_history_requires_metadata_for_a_real_data_run(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "missing-real-metadata" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    manifest_path = copied_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data"]["is_real_data"] = True
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
    assert response.status_code == 409
    assert "metadata" in response.json()["detail"].lower()


def test_price_history_rejects_internal_and_request_path_traversal(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "traversal" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative = _price_snapshot_relative(copied_run)
    manifest_path = copied_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(item for item in manifest["artifacts"] if item["path"] == relative)[
        "path"
    ] = "input_snapshot/../outside.csv"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        internal = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
        request_path = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/"
            "%2E%2E%2Foutside/prices"
        )
        malformed_ticker = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/bad%5Cticker/prices"
        )
    assert internal.status_code == 409
    assert request_path.status_code in {400, 404}
    assert malformed_ticker.status_code == 400
    assert "ticker" in malformed_ticker.json()["detail"]
    assert str(vertical_root) not in internal.text
    assert str(vertical_root) not in request_path.text
    assert str(vertical_root) not in malformed_ticker.text


def test_price_history_rejects_snapshot_symlink_escape(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "symlink" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative = _price_snapshot_relative(copied_run)
    snapshot = copied_run / relative
    external = tmp_path / "outside-prices.csv"
    external.write_bytes(snapshot.read_bytes())
    snapshot.unlink()
    try:
        snapshot.symlink_to(external)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
    assert response.status_code == 409
    assert str(external) not in response.text


def test_inventoried_reader_rejects_parent_symlink_swap(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertical_root = tmp_path / "symlink-race" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    repository = ArtifactRepository(
        _settings(generated_artifacts, vertical_root=vertical_root)
    )
    validated = repository._validate_run(FIXTURE_RUN_ID)
    relative = _price_snapshot_relative(copied_run)

    snapshot_directory = copied_run / Path(relative).parts[0]
    held_directory = copied_run / "held-input-snapshot"
    external_directory = tmp_path / "outside-input-snapshot"
    external_directory.mkdir()
    shutil.copy2(
        copied_run / relative,
        external_directory / Path(relative).name,
    )

    real_open = os.open
    swapped = False

    def swap_parent_before_open(
        path: Any, flags: int, *args: Any, **kwargs: Any
    ) -> int:
        nonlocal swapped
        path_text = os.fspath(path)
        if not swapped and (
            path_text == "input_snapshot" or path_text.endswith(relative)
        ):
            snapshot_directory.rename(held_directory)
            snapshot_directory.symlink_to(external_directory, target_is_directory=True)
            swapped = True
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_parent_before_open)
    with pytest.raises(ArtifactIntegrityError):
        repository._csv_artifact(validated, relative)
    assert swapped is True


def test_inventoried_json_is_rechecked_after_run_validation(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "json-reread" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    _add_pipeline_price_contract(copied_run, mutation="valid")
    repository = ArtifactRepository(
        _settings(generated_artifacts, vertical_root=vertical_root)
    )
    validated = repository._validate_run(FIXTURE_RUN_ID)

    metadata_path = copied_run / "input_snapshot" / "pipeline_metadata.json"
    metadata_path.write_text(
        metadata_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    with pytest.raises(ArtifactIntegrityError):
        repository._json_artifact(validated, "input_snapshot/pipeline_metadata.json")


def test_price_history_rejects_duplicate_ticker_dates(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "duplicate-date" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative, fieldnames, rows = _read_price_snapshot(copied_run)
    rows[1]["date"] = rows[0]["date"]
    _write_and_reseal_price_snapshot(copied_run, relative, fieldnames, rows)

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
    assert response.status_code == 409
    assert "absolute" not in response.text.lower()


def test_price_history_validates_primary_coverage_for_any_requested_ticker(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "primary-coverage" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative, fieldnames, rows = _read_price_snapshot(copied_run)
    second_symbol_rows = [dict(row, ticker="HBL") for row in rows]
    _write_and_reseal_price_snapshot(
        copied_run, relative, fieldnames, rows + second_symbol_rows
    )

    manifest_path = copied_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["data"]["row_count"] = 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/HBL/prices"
        )
    assert response.status_code == 409


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("date", "not-a-date"),
        ("open", "0"),
        ("open", "NaN"),
        ("open", "1e1000000"),
        ("close", "-1"),
        ("close", "Infinity"),
        ("high", "Infinity"),
        ("low", "1e-1000000"),
        ("volume", "-1"),
        ("volume", "1.5"),
        ("source", ""),
    ],
)
def test_price_history_rejects_invalid_snapshot_values(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    vertical_root = tmp_path / f"invalid-{field}-{value}" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    relative, fieldnames, rows = _read_price_snapshot(copied_run)
    rows[0][field] = value
    _write_and_reseal_price_snapshot(copied_run, relative, fieldnames, rows)

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices"
        )
    assert response.status_code == 409
    assert "NaN" not in response.text
    assert "Infinity" not in response.text
    assert str(vertical_root) not in response.text


def test_price_history_never_reads_legacy_prices_or_model_binaries(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        api_main,
        "paths",
        get_paths(data_dir=tmp_path / "legacy-data", model_dir=tmp_path / "models"),
    )

    def reject_joblib_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("run-scoped price API must not deserialize joblib")

    monkeypatch.setattr("joblib.load", reject_joblib_load)
    with _api_client(_settings(generated_artifacts)) as client:
        legacy = client.get("/stocks/OGDC/prices")
        versioned = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/stocks/OGDC/prices?limit=1"
        )
    assert legacy.status_code == 404
    assert versioned.status_code == 200
    assert versioned.json()["rows"][0]["source"] == "deterministic_fixture"


def test_price_history_openapi_documents_success_and_all_errors() -> None:
    path = "/api/v1/vertical-slices/{run_id}/stocks/{ticker}/prices"
    operation = app.openapi()["paths"][path]["get"]
    assert set(operation["responses"]) == {"200", "400", "404", "409", "422"}
    assert operation["responses"]["200"]["content"]["application/json"]["schema"][
        "$ref"
    ].endswith("/StockPricesResponse")


def test_lists_only_completed_checksum_valid_pass_runs(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "vertical"
    _copy_valid_run(generated_artifacts, vertical_root)
    (vertical_root / "incomplete-run").mkdir()
    failed = vertical_root / "failed-run"
    failed.mkdir()
    (failed / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "vertical_slice_run_id": "failed-run",
                "status": "FAIL",
            }
        ),
        encoding="utf-8",
    )

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        listing = client.get("/api/v1/vertical-slices")
        assert listing.status_code == 200
        assert listing.json()["total"] == 1
        assert listing.json()["limit"] == 200
        assert listing.json()["offset"] == 0
        assert [row["run_id"] for row in listing.json()["rows"]] == [FIXTURE_RUN_ID]

        for run_id in ("incomplete-run", "failed-run"):
            rejected = client.get(f"/api/v1/vertical-slices/{run_id}")
            assert rejected.status_code == 409
            assert "PASS" in rejected.json()["detail"] or "completed" in rejected.json()["detail"]


def test_default_leaderboard_requires_explicit_completed_run(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    with _api_client(_settings(generated_artifacts, default_run_id=None)) as client:
        response = client.get("/api/v1/leaderboard")
    assert response.status_code == 409
    assert "PAISA_DEFAULT_VERTICAL_SLICE_RUN_ID" in response.json()["detail"]


@pytest.mark.parametrize("run_id", ["bad$id", ".hidden", "a" * 201])
def test_malformed_run_ids_are_rejected(
    generated_artifacts: GeneratedArtifacts,
    run_id: str,
) -> None:
    encoded = quote(run_id, safe="")
    with _api_client(_settings(generated_artifacts)) as client:
        response = client.get(f"/api/v1/vertical-slices/{encoded}")
    assert response.status_code == 400
    assert "run_id" in response.json()["detail"]


def test_path_traversal_attempts_cannot_select_runs_or_experiments(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    attempts = (
        "/api/v1/vertical-slices/%2E%2E%2Foutside",
        (
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/experiments/"
            "%2E%2E%2Fmanifest.json"
        ),
        f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/experiments/bad%5Cid",
    )
    with _api_client(_settings(generated_artifacts)) as client:
        responses = [client.get(url) for url in attempts]
    assert all(response.status_code in {400, 404} for response in responses)
    assert all(response.status_code != 200 for response in responses)
    assert all(str(REPOSITORY_ROOT) not in response.text for response in responses)


@pytest.mark.parametrize(
    "value",
    [
        "/tmp",
        "/",
        "file:///etc/passwd",
        r"\\server\share\secret.csv",
        r"`\\server\share\secret.csv`",
        r"\Users\name\secret.csv",
        r"C:\\secret.csv",
        "`/tmp/private`",
        "path->/tmp/private",
        "path|/tmp/private",
        "path-/tmp/private",
    ],
)
def test_response_guard_rejects_all_absolute_local_path_forms(value: str) -> None:
    with pytest.raises(ArtifactIntegrityError):
        _assert_no_absolute_local_paths(value)


def test_response_guard_allows_public_web_urls() -> None:
    _assert_no_absolute_local_paths("See https://example.test/api/v1/results for details")


def test_response_guard_scans_mapping_keys() -> None:
    with pytest.raises(ArtifactIntegrityError):
        _assert_no_absolute_local_paths({"/tmp/private.csv": "safe value"})


def test_checksum_valid_artifact_path_text_is_still_not_exposed(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "absolute-path" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    summary_path = copied_run / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["warnings"].append("Producer debug output was written to /tmp/private.log")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_vertical_checksum(copied_run, "final_summary.json")

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}")
    assert response.status_code == 409
    assert "/tmp/private.log" not in response.text


def test_symlink_escape_inside_run_is_rejected(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    vertical_root = tmp_path / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    external = tmp_path / "outside.csv"
    external.write_bytes((copied_run / "combined_leaderboard.csv").read_bytes())
    linked_artifact = copied_run / "combined_leaderboard.csv"
    linked_artifact.unlink()
    try:
        linked_artifact.symlink_to(external)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symbolic links are unavailable: {exc}")

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        detail = client.get(f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}")
        listing = client.get("/api/v1/vertical-slices")
    assert detail.status_code == 409
    assert listing.status_code == 200
    assert listing.json()["rows"] == []


def test_leaderboard_parsing_is_typed_and_deterministically_ranked(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    with (generated_artifacts.run_directory / "combined_leaderboard.csv").open(
        encoding="utf-8", newline=""
    ) as source:
        raw_rows = list(csv.DictReader(source))
    expected_ids = [
        row["model_id"]
        for row in sorted(
            raw_rows,
            key=lambda row: (
                row["validation_balanced_accuracy"].strip() == "",
                -float(row["validation_balanced_accuracy"] or 0.0),
                -float(row["validation_f1"]),
                row["model_id"],
            ),
        )
    ]

    with _api_client(_settings(generated_artifacts)) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/leaderboard"
        )
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert response.json()["total"] == 4
    assert [row["model_id"] for row in rows] == expected_ids
    assert all(isinstance(row["validation_f1"], float) for row in rows)
    assert all(MONEY_PATTERN.fullmatch(row["ending_portfolio_value"]) for row in rows)
    assert all(MONEY_PATTERN.fullmatch(row["fees_paid"]) for row in rows)


def test_money_is_exact_fixed_scale_and_undefined_metrics_are_json_null(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    with _api_client(_settings(generated_artifacts)) as client:
        detail = client.get(f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}")
        portfolios = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/portfolios"
        )
    assert detail.status_code == portfolios.status_code == 200
    assert (
        detail.json()["shared_experiment_assumptions"]["initial_cash_pkr"]
        == "100000.00"
    )
    for row in portfolios.json()["rows"]:
        assert isinstance(row["ending_portfolio_value"], str)
        assert isinstance(row["fees_paid"], str)
        assert MONEY_PATTERN.fullmatch(row["ending_portfolio_value"])
        assert MONEY_PATTERN.fullmatch(row["fees_paid"])
    cash_only = next(
        row
        for row in portfolios.json()["rows"]
        if row["portfolio_type"] == "cash_only"
    )
    assert cash_only["ending_portfolio_value"] == "100000.00"
    assert cash_only["fees_paid"] == "0.00"
    assert cash_only["sharpe_ratio"] is None
    assert '"sharpe_ratio":null' in portfolios.text
    assert "NaN" not in portfolios.text
    assert "Infinity" not in portfolios.text


def test_equity_curves_are_stably_ordered_and_paginated(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    url = (
        f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/equity-curves"
        "?limit=1000&offset=0"
    )
    with _api_client(_settings(generated_artifacts)) as client:
        first = client.get(url)
        second = client.get(url)
    assert first.status_code == second.status_code == 200
    assert first.json()["rows"] == second.json()["rows"]
    payload = first.json()
    keys = [
        (row["date"], row["series_id"], row["portfolio_id"])
        for row in payload["rows"]
    ]
    assert keys == sorted(keys)
    assert payload["total"] == len(payload["rows"])
    assert payload["limit"] == 1000
    assert payload["offset"] == 0


def test_trade_pagination_has_stable_total_and_evaluation_warning(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    selected_id: str | None = None
    with _api_client(_settings(generated_artifacts)) as client:
        for experiment_id in generated_artifacts.experiment_ids:
            candidate = client.get(
                f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/experiments/"
                f"{experiment_id}/trades?limit=1&offset=0"
            )
            assert candidate.status_code == 200
            if candidate.json()["total"] >= 2:
                selected_id = experiment_id
                first = candidate
                break
        assert selected_id is not None, "fixture should generate at least two trades"
        second = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/experiments/"
            f"{selected_id}/trades?limit=1&offset=1"
        )

    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["total"] == second_payload["total"]
    assert first_payload["limit"] == second_payload["limit"] == 1
    assert first_payload["offset"] == 0
    assert second_payload["offset"] == 1
    assert len(first_payload["rows"]) == len(second_payload["rows"]) == 1
    assert first_payload["rows"][0]["trade_id"] != second_payload["rows"][0]["trade_id"]
    assert "evaluation-only" in first_payload["actual_class_usage"]


def test_benchmark_response_contains_cash_buy_hold_snapshots_and_trades(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    experiment_id = generated_artifacts.experiment_ids[0]
    url = (
        f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/experiments/"
        f"{experiment_id}/benchmarks?limit=2&offset=1"
    )
    with _api_client(_settings(generated_artifacts)) as client:
        response = client.get(url)
    assert response.status_code == 200
    payload = response.json()
    assert payload["experiment_id"] == experiment_id
    assert payload["cash_only"]["portfolio_id"] != payload["buy_and_hold"]["portfolio_id"]
    for name in ("cash_only", "buy_and_hold"):
        series = payload[name]
        assert series["total"] > 2
        assert series["limit"] == 2
        assert series["offset"] == 1
        assert len(series["rows"]) == 2
        assert MONEY_PATTERN.fullmatch(series["ending_equity"])
        assert [row["date"] for row in series["rows"]] == sorted(
            row["date"] for row in series["rows"]
        )
    trades = payload["buy_and_hold_trades"]
    assert trades["total"] >= 1
    assert trades["portfolio_id"] == payload["buy_and_hold"]["portfolio_id"]


def test_academic_disclaimer_and_actual_class_limitation_are_preserved(
    generated_artifacts: GeneratedArtifacts,
) -> None:
    experiment_id = generated_artifacts.experiment_ids[0]
    with _api_client(_settings(generated_artifacts)) as client:
        listing = client.get("/api/v1/vertical-slices").json()
        detail = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/experiments/{experiment_id}"
        ).json()
    assert "not financial advice" in listing["disclaimer"].lower()
    assert "not financial advice" in detail["disclaimer"].lower()
    assert "evaluation-only" in detail["actual_class_usage"]
    assert any("actual_class" in item for item in detail["limitations"])


def test_legacy_health_endpoint_shape_is_unchanged(
    generated_artifacts: GeneratedArtifacts,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        api_main,
        "paths",
        get_paths(data_dir=tmp_path / "legacy-data", model_dir=tmp_path / "legacy-models"),
    )
    with _api_client(_settings(generated_artifacts)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "data_dir": str(tmp_path / "legacy-data"),
        "model_dir": str(tmp_path / "legacy-models"),
        "prices_ready": False,
        "dataset_ready": False,
        "model_ready": False,
    }


@pytest.mark.parametrize("mutation", ["artifact-bytes", "manifest-checksum"])
def test_corrupted_or_checksum_invalid_run_is_rejected(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    mutation: str,
) -> None:
    vertical_root = tmp_path / mutation / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    if mutation == "artifact-bytes":
        leaderboard = copied_run / "combined_leaderboard.csv"
        leaderboard.write_bytes(leaderboard.read_bytes() + b"\n")
    else:
        manifest_path = copied_run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            item
            for item in manifest["artifacts"]
            if item["path"] == "combined_leaderboard.csv"
        )
        entry["sha256"] = "0" * 64
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/leaderboard"
        )
        listing = client.get("/api/v1/vertical-slices")
    assert response.status_code == 409
    assert "integrity" in response.json()["detail"]
    assert listing.status_code == 200
    assert listing.json()["total"] == 0


def test_read_only_v1_routes_never_deserialize_model_binaries(
    generated_artifacts: GeneratedArtifacts,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_joblib_load(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("read-only results API must not deserialize joblib")

    monkeypatch.setattr("joblib.load", reject_joblib_load)
    with _api_client(_settings(generated_artifacts)) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/leaderboard"
        )
    assert response.status_code == 200


@pytest.mark.parametrize("limit", [0, 1001])
def test_large_table_limits_are_validated_before_artifact_access(
    generated_artifacts: GeneratedArtifacts,
    limit: int,
) -> None:
    with _api_client(_settings(generated_artifacts)) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/equity-curves?limit={limit}"
        )
    assert response.status_code == 422

    with _api_client(_settings(generated_artifacts)) as client:
        listing = client.get(f"/api/v1/vertical-slices?limit={limit}")
    assert listing.status_code == 422


def test_unsupported_nested_schema_versions_are_rejected(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
) -> None:
    relative_path = "model_training_run_reference.json"
    vertical_root = tmp_path / "nested-schema" / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    artifact_path = copied_run / relative_path
    value = json.loads(artifact_path.read_text(encoding="utf-8"))
    value["schema_version"] = "2.0"
    artifact_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_vertical_checksum(copied_run, relative_path)

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}")
    assert response.status_code == 409


@pytest.mark.parametrize("field", ["schema_version", "portfolio_id"])
def test_trade_contract_versions_and_portfolio_identities_are_enforced(
    generated_artifacts: GeneratedArtifacts,
    tmp_path: Path,
    field: str,
) -> None:
    vertical_root = tmp_path / field / "vertical"
    copied_run = _copy_valid_run(generated_artifacts, vertical_root)
    experiment_id: str | None = None
    for candidate in generated_artifacts.experiment_ids:
        candidate_path = (
            generated_artifacts.run_directory
            / "experiments"
            / candidate
            / "trades.csv"
        )
        with candidate_path.open(encoding="utf-8", newline="") as source:
            if any(csv.DictReader(source)):
                experiment_id = candidate
                break
    assert experiment_id is not None
    relative_path = f"experiments/{experiment_id}/trades.csv"
    artifact_path = copied_run / relative_path
    with artifact_path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames
        rows = list(reader)
    assert fieldnames is not None and rows
    rows[0][field] = "2.0" if field == "schema_version" else "wrong-portfolio"
    with artifact_path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _refresh_vertical_checksum(copied_run, relative_path)

    with _api_client(
        _settings(generated_artifacts, vertical_root=vertical_root)
    ) as client:
        response = client.get(
            f"/api/v1/vertical-slices/{FIXTURE_RUN_ID}/experiments/"
            f"{experiment_id}/trades"
        )
    assert response.status_code == 409


def test_local_real_run_all_endpoints_when_available() -> None:
    run_directory = (
        REPOSITORY_ROOT / "artifacts" / "vertical_slice" / REAL_RUN_ID
    )
    if not run_directory.is_dir():
        pytest.skip(f"optional local real run is absent: {run_directory}")

    settings = RepositorySettings(
        repository_root=REPOSITORY_ROOT,
        vertical_slice_root=run_directory.parent,
        model_training_root=REPOSITORY_ROOT / "artifacts" / "model_training",
        default_vertical_slice_run_id=REAL_RUN_ID,
    )
    with _api_client(settings) as client:
        experiments = client.get(
            f"/api/v1/vertical-slices/{REAL_RUN_ID}/experiments"
        )
        assert experiments.status_code == 200, experiments.text
        experiment_id = experiments.json()["rows"][0]["experiment_id"]
        for url in _endpoint_urls(REAL_RUN_ID, experiment_id):
            response = client.get(url)
            assert response.status_code == 200, (url, response.text)
            assert response.json()["disclaimer"] == ACADEMIC_DISCLAIMER
            assert str(REPOSITORY_ROOT) not in response.text
            assert "Traceback" not in response.text
        first_page = client.get(
            f"/api/v1/vertical-slices/{REAL_RUN_ID}/stocks/OGDC/prices"
            "?limit=5&offset=0"
        )
        complete = client.get(
            f"/api/v1/vertical-slices/{REAL_RUN_ID}/stocks/OGDC/prices"
            "?limit=5000&offset=0"
        )

    assert first_page.status_code == complete.status_code == 200
    assert len(first_page.json()["rows"]) == 5
    payload = complete.json()
    assert payload["total"] == 1239
    assert payload["rows"][0]["date"] == "2021-09-16"
    assert payload["rows"][-1] == {
        "date": "2026-09-15",
        "open": "317.00",
        "high": None,
        "low": None,
        "close": "314.87",
        "volume": 1499010,
        "source": "psx_dps",
    }
    assert all(row["source"] == "psx_dps" for row in payload["rows"])
    assert all(row["high"] is None and row["low"] is None for row in payload["rows"])
