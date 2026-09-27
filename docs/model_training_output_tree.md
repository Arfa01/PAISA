# Expected output tree for OGDC RF training stage

A successful run creates a unique immutable directory:

```text
artifacts/model_training/<run_id>/
├── run_manifest.json
├── data_manifest.json
├── split_manifest.json
├── data_quality_report.json
├── frozen_input/
│   ├── OGDC_prices.csv
│   ├── OGDC_features_from_v0.csv
│   └── OGDC_modelling_rows.csv
├── reports/
│   ├── validation_leaderboard.csv
│   ├── test_leaderboard.csv
│   ├── baselines.csv
│   ├── cumulative_performance.csv
│   ├── environment.json
│   ├── provisional_champion.json
│   ├── final_pass_fail_checklist.json
│   ├── final_summary.json
│   └── zip_info.json
├── plots/
│   ├── price_and_splits.png
│   ├── class_distribution.png
│   ├── validation_model_comparison.png
│   ├── test_model_comparison.png
│   ├── rolling_5_day_hit_rate.png
│   ├── rolling_20_day_hit_rate.png
│   ├── cumulative_accuracy.png
│   ├── confusion_matrix_M-001.png
│   ├── confusion_matrix_M-002.png
│   ├── confusion_matrix_M-003.png
│   ├── confusion_matrix_M-004.png
│   ├── confusion_matrix_M-005.png
│   ├── probability_up_timeline_M-001.png
│   ├── probability_up_timeline_M-002.png
│   ├── probability_up_timeline_M-003.png
│   ├── probability_up_timeline_M-004.png
│   ├── probability_up_timeline_M-005.png
│   ├── feature_importance_M-001.png
│   ├── feature_importance_M-002.png
│   ├── feature_importance_M-003.png
│   ├── feature_importance_M-004.png
│   └── feature_importance_M-005.png
└── models/
    ├── M-001/
    │   ├── model.joblib
    │   ├── config.json
    │   ├── feature_columns.json
    │   ├── metrics.json
    │   ├── predictions.csv
    │   ├── confusion_matrices.csv
    │   ├── feature_importance.csv
    │   └── artifact_manifest.json
    ├── M-002/
    │   └── same file pattern as M-001
    ├── M-003/
    │   └── same file pattern as M-001
    ├── M-004/
    │   └── same file pattern as M-001
    └── M-005/
        └── same file pattern as M-001
```

A successful run also creates a ZIP archive:

```text
artifacts/paisa_<run_id>.zip
```

In Colab the ZIP path is:

```text
/content/paisa_<run_id>.zip
```

In Kaggle the ZIP path is:

```text
/kaggle/working/paisa_<run_id>.zip
```
