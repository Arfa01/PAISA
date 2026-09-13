from __future__ import annotations

from datetime import date

import pandas as pd

from paisa.models.experiments import run_price_only_model_comparison
from paisa.pipeline import run_v0_pipeline


def test_price_only_model_comparison_outputs_table(tmp_path):
    data_dir = tmp_path / "data"
    model_dir = tmp_path / "models"

    run_v0_pipeline(
        symbols=["OGDC", "HBL"],
        provider_name="sample",
        start=date(2021, 1, 1),
        end=date(2022, 12, 31),
        train=False,
        data_dir=data_dir,
        model_dir=model_dir,
    )

    dataset = pd.read_csv(data_dir / "processed" / "ml_dataset.csv")
    result = run_price_only_model_comparison(dataset=dataset, model_dir=model_dir)

    assert result["experiment_name"] == "price_only_model_comparison_v0_1"
    assert set(result["models"]) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert (model_dir / "model_comparison_price_only.csv").exists()
    assert (model_dir / "model_comparison_price_only.json").exists()
    assert (model_dir / "model_comparison_price_only.joblib").exists()
