from __future__ import annotations

from datetime import date

import pandas as pd

from paisa.data_quality import summarize_data_quality, write_data_quality_report
from paisa.pipeline import run_v0_pipeline


def test_data_quality_report_summarizes_pipeline_outputs(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"

    run_v0_pipeline(
        symbols=["OGDC", "HBL"],
        provider_name="sample",
        start=date(2021, 1, 1),
        end=date(2022, 12, 31),
        train=True,
        data_dir=data_dir,
        model_dir=model_dir,
    )

    report = write_data_quality_report(
        prices_path=data_dir / "processed" / "psx_prices.csv",
        features_path=data_dir / "processed" / "psx_features.csv",
        dataset_path=data_dir / "processed" / "ml_dataset.csv",
        output_dir=data_dir / "processed",
        docs_dir=tmp_path / "docs",
    )

    assert report["tickers_count"] == 2
    assert report["price_rows"] > 400
    assert report["dataset_rows"] > 300
    assert (data_dir / "processed" / "data_quality_report.json").exists()
    assert (data_dir / "processed" / "data_quality_report.csv").exists()
    assert (tmp_path / "docs" / "data_quality_report_v0.md").exists()

    rows = pd.read_csv(data_dir / "processed" / "data_quality_report.csv")
    assert set(rows["ticker"]) == {"HBL", "OGDC"}
    assert (rows["missing_close"] == 0).all()
    assert (rows["duplicate_dates"] == 0).all()


def test_data_quality_detects_duplicate_dates():
    prices = pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-01-01"],
            "ticker": ["OGDC", "OGDC"],
            "open": [100.0, 101.0],
            "high": [101.0, 102.0],
            "low": [99.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 2000],
            "source": ["unit", "unit"],
        }
    )
    report = summarize_data_quality(prices)
    assert report["tickers"][0]["duplicate_dates"] == 1
    assert any("duplicate" in issue for issue in report["issues"])
