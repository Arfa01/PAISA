from __future__ import annotations

from datetime import date

import pandas as pd

from paisa.features import FEATURE_COLUMNS
from paisa.pipeline import run_v0_pipeline


def test_sample_pipeline_produces_expected_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"

    metadata = run_v0_pipeline(
        symbols=["OGDC", "HBL"],
        provider_name="sample",
        start=date(2021, 1, 1),
        end=date(2022, 12, 31),
        train=True,
        data_dir=data_dir,
        model_dir=model_dir,
    )

    assert metadata["prices_rows"] > 400
    assert metadata["ml_dataset_rows"] > 300
    assert (data_dir / "processed" / "psx_prices.csv").exists()
    assert (data_dir / "processed" / "psx_features.csv").exists()
    assert (data_dir / "processed" / "ml_dataset.csv").exists()
    assert (model_dir / "baseline_random_forest.joblib").exists()
    assert (model_dir / "baseline_metrics.json").exists()

    dataset = pd.read_csv(data_dir / "processed" / "ml_dataset.csv")
    for col in FEATURE_COLUMNS + ["target_next_day_up"]:
        assert col in dataset.columns
    assert set(dataset["target_next_day_up"].unique()).issubset({0, 1})
