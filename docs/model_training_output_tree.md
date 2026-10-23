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

`run_manifest.json` is the integrity root for the directory. It records the
run-level `code_version`, `dirty_tree`, and `source_fingerprint` and requires
those values to match every registered model manifest. Its deterministic
`artifact_inventory` lists the relative POSIX path and SHA-256 digest of every
other regular file in the run, including frozen inputs, split and data
manifests, reports, model manifests, fitted binaries, configurations, metrics,
and predictions. The listed paths must exactly equal the files on disk; missing
files, additional files, links, special files, duplicate paths, unsorted paths,
or checksum changes invalidate the run. When a model advertises nested
preprocessing or validation-checkpoint checksums, the validator also reconciles
those claims with their checksum-inventoried files; a coordinated metadata
rewrite and reseal cannot substitute a false nested digest.

The root manifest excludes only itself because a file cannot contain its own
final digest. A consumer captures the SHA-256 of `run_manifest.json` together
with its listed inventory, then reasserts both the unchanged root digest and
the exact payload set after loading or delegated processing. Model IDs and
prediction paths used downstream must be derived from that validated manifest;
an accompanying summary is accepted only when its bindings match exactly. A
controlled metadata rewrite of an unpublished, runner-owned directory must
reseal and revalidate the run before its final evidence baseline is captured.
Existing or third-party runs are never silently resealed.

Each model's `predictions.csv` contains only the held-out test split in the
prediction-record schema required by `scripts/run_backtest.py`, including
`schema_version`. `predictions_all_splits.csv` is retained for training and
evaluation audit work and adds the split name.

The training library saves evaluation evidence as tables rather than generated
plot files. The Colab notebook reads the leaderboards and creates presentation
plots in memory. Those plots are not part of the immutable run tree.

The writer seals and validates the staged directory before publication, and
fails before publishing if `<run_id>/`, `<run_id>.zip`, or a model destination
already exists. The adjacent ZIP is transport packaging rather than a member
of the run directory, so it is not part of the run inventory. The complete
`artifacts/` tree and generated ZIP files are ignored by Git.
