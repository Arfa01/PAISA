# Expected output tree for OGDC model experiments

A successful run publishes one unique immutable directory and, unless
`--no-zip` is supplied, one adjacent ZIP archive:

```text
artifacts/model_training/
├── <run_id>/
│   ├── run_manifest.json
│   ├── data_manifest.json
│   ├── split_manifest.json
│   ├── frozen_input/
│   │   ├── OGDC_prices.csv
│   │   └── OGDC_model_rows.csv
│   ├── reports/
│   │   ├── environment.json
│   │   ├── validation_leaderboard.csv
│   │   ├── test_leaderboard.csv
│   │   ├── baselines.csv
│   │   ├── provisional_champion.json
│   │   └── final_summary.json
│   └── models/
│       ├── M-001-<version>/
│       │   ├── model.joblib
│       │   ├── manifest.json
│       │   ├── config.json
│       │   ├── feature_columns.json
│       │   ├── metrics.json
│       │   ├── confusion_matrix.csv
│       │   ├── feature_importance.csv
│       │   ├── predictions.csv
│       │   └── predictions_all_splits.csv
│       ├── M-002-<version>/
│       │   └── same file pattern as M-001
│       ├── M-003-<version>/
│       │   └── same file pattern as M-001
│       └── M-004-<version>/
│           └── same file pattern as M-001
└── <run_id>.zip
```

`M-001`, `M-002`, and `M-003` are the stable Random Forest variant IDs;
`M-004` is the XGBoost variant. Their registered model IDs combine a
deterministic semantic digest of the data, feature order, configuration,
target, and seed with the run ID digest and a per-invocation UUID. The UUID
prevents collisions when identical reproducible runs are created in separate
artifact roots; it is identity metadata and is not used as model randomness.
The complete registered ID and invocation UUID are persisted in the model and
run manifests, and the registered ID appears in every prediction row.

Each model's `predictions.csv` contains only the untouched test split in the
prediction-record schema required by `scripts/run_backtest.py`, including
`schema_version`. `predictions_all_splits.csv` is retained for training and
evaluation audit work and adds the split name.

The training library saves evaluation evidence as tables rather than generated
plot files. The Colab notebook reads the leaderboards and creates presentation
plots in memory. Those plots are not part of the immutable run tree.

The writer fails before publishing if `<run_id>/`, `<run_id>.zip`, or a model
destination already exists. The complete `artifacts/` tree and generated ZIP
files are ignored by Git.
