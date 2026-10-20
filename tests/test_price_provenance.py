from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import pytest

from paisa.pipeline import _build_prices_artifact, run_v0_pipeline


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_binds_price_provenance_to_exact_saved_csv(tmp_path):
    data_dir = tmp_path / "data"
    metadata = run_v0_pipeline(
        symbols=["OGDC", "HBL"],
        provider_name="sample",
        start=date(2025, 1, 1),
        end=date(2025, 6, 30),
        train=False,
        data_dir=data_dir,
        model_dir=tmp_path / "models",
    )

    prices_path = data_dir / "processed" / "psx_prices.csv"
    persisted = json.loads(
        (data_dir / "processed" / "pipeline_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    prices = pd.read_csv(prices_path, parse_dates=["date"])
    artifact = metadata["prices_artifact"]

    assert persisted["prices_artifact"] == artifact
    assert metadata["schema_version"] == "1.0"
    assert metadata["producer"] == {
        "component": "paisa.pipeline.run_v0_pipeline",
        "schema_version": "1.0",
        "status": "complete",
    }
    assert metadata["provider"] == "sample"
    assert metadata["prices_rows"] == len(prices)
    assert artifact == {
        "schema_version": "1.0",
        "status": "complete",
        "path": "psx_prices.csv",
        "sha256": _sha256(prices_path),
        "provider": "sample",
        "frequency": "daily",
        "row_count": len(prices),
        "symbols": {
            symbol: {
                "row_count": len(rows),
                "date_start": rows["date"].min().strftime("%Y-%m-%d"),
                "date_end": rows["date"].max().strftime("%Y-%m-%d"),
                "source": "sample",
            }
            for symbol, rows in prices.groupby("ticker", sort=True)
        },
    }


def test_price_provenance_checksum_no_longer_matches_after_csv_mutation(tmp_path):
    data_dir = tmp_path / "data"
    metadata = run_v0_pipeline(
        symbols=["OGDC"],
        provider_name="sample",
        start=date(2025, 1, 1),
        end=date(2025, 6, 30),
        train=False,
        data_dir=data_dir,
        model_dir=tmp_path / "models",
    )
    prices_path = data_dir / "processed" / "psx_prices.csv"
    original_checksum = metadata["prices_artifact"]["sha256"]

    prices = pd.read_csv(prices_path)
    prices.loc[0, "source"] = "psx_dps"
    prices.to_csv(prices_path, index=False)

    assert _sha256(prices_path) != original_checksum


@pytest.mark.parametrize("invalid_source", [None, "", "   "])
def test_price_provenance_rejects_blank_or_null_source_rows(
    tmp_path,
    invalid_source,
):
    prices_path = tmp_path / "psx_prices.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "ticker": ["OGDC", "OGDC"],
            "source": ["sample", invalid_source],
        }
    ).to_csv(prices_path, index=False)

    with pytest.raises(ValueError, match="blank or null source"):
        _build_prices_artifact(prices_path, provider="sample")


def test_price_provenance_rejects_mixed_sources(tmp_path):
    prices_path = tmp_path / "psx_prices.csv"
    pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "ticker": ["OGDC", "OGDC"],
            "source": ["sample", "psx_dps"],
        }
    ).to_csv(prices_path, index=False)

    with pytest.raises(ValueError, match="do not match provider"):
        _build_prices_artifact(prices_path, provider="sample")
