# PAISA tree-model SHAP explainability

PAISA produces immutable SHAP evidence for the four retained tree-based,
price-only OGDC models. The explanation workflow is a post-training research
stage: it does not retrain, tune, recalibrate, select, or replace a model, and it
does not create signals, trades, portfolios, or dashboard state.

The outputs are an academic simulation and model-behavior analysis, not
financial advice. A feature attribution is not evidence that the feature caused
a market movement or that acting on the model would be profitable.

## Scope

The workflow explains exactly these fitted models:

| Stable ID | Registered model ID | Algorithm |
|---|---|---|
| `M-001` | `M-001-6ce88dc384de975e-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | Baseline Random Forest |
| `M-002` | `M-002-5a0160ff1b7a8043-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | Conservative Random Forest |
| `M-003` | `M-003-23efd4aed873bf08-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | Flexible Random Forest |
| `M-004` | `M-004-6c60cc92f77a495a-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | XGBoost classifier |

`M-005` and `M-006` are LSTMs and are deliberately outside this tree-explainer
contract. The workflow covers one ticker (`OGDC`), daily technical features,
one next-session binary target, and the exact 184-date shared project holdout.
It is not a general explanation service for client-supplied models.

## Authoritative inputs

An official run accepts only the following completed source artifacts:

| Role | Run ID | Integrity root | Required SHA-256 |
|---|---|---|---|
| Tree-model training | `TRN-20261020-191815-624662-nogit` | `run_manifest.json` | `8d5113bc8b67f43a76639d7397b905217bda1f94321183f0438e33448f689023` |
| Unified comparison | `CMP-20261024-OGDC-price-only-02` | `manifest.json` | `e4b0b12bb565fa6df1f23b7e8855f84569ffcefb795c36c4bfeb4a1e2c82849d` |
| Vertical-slice evidence | `VSL-20261020-OGDC-main-02` | `manifest.json` | `2ce2f7581a5baff8487b09ae695213745c704f11715e6bd5abd0e99009337f31` |
| Frozen modelling rows | training-run `frozen_input/OGDC_model_rows.csv` | file bytes | `5ba9ce1c99af8873163c78ae9466c9a7de8042bb1deb3a56eeac525f12deb08e` |

The fixed training source fingerprint is
`3b82245209925d7a4678fe7b276a3e58b5dac73d`. The canonical digest of the
ordered ticker, feature-date, target-date, and actual-class holdout keys is
`730755ddb7993d829ddd4d7f5afaec4d5ca5617311d642f4a33722d251a5a85a`.

The builder validates the cross-run bindings before loading a model and checks
the source artifacts again immediately before publication. It refuses a source
identity or digest substitution even if the substitute has a superficially
compatible schema.

## Isolated runtime and SHAP pin

Official explainability runs require `shap==0.54.1`. The authoritative model
manifests record Python `3.13.7`, NumPy `2.3.5`, pandas `2.3.3`,
scikit-learn `1.9.1`, XGBoost `3.4.1`, and joblib `1.6.0` for the retained
serialization environment.

SHAP 0.54.1 itself requires Python 3.12 or newer and NumPy 2 or newer. That is
compatible with the retained model runtime but narrower than PAISA's general
Python 3.10+ and NumPy 1.26+ package floor. Keep explainability in a separate
Python 3.12+ environment—Python 3.13 is the closest match. The default Python
unit-test job does not install SHAP; a separate required explainability-runtime
CI job installs and exercises the certified stack.

```bash
python3.13 -m venv .venv-explainability
source .venv-explainability/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  -c constraints-explainability.txt \
  -r requirements.txt \
  -r requirements-explainability.txt
python -m pip install --no-deps -e .
python -m pip check
```

`constraints-explainability.txt` pins the certified direct versions of NumPy,
pandas, scikit-learn, XGBoost, joblib, Matplotlib, and SHAP. It is a pip
constraints file, not a standalone requirements file or a full hashed lock.
The real-estimator integration runs in a dedicated artifact-free Python 3.13.7
CI job: it installs that constrained runtime, imports and asserts SHAP
`0.54.1`, fits tiny Random Forest and XGBoost classifiers, constructs the
production-style `TreeExplainer`, normalizes the observed SHAP and expected-
value shapes, and reconstructs every class-1 probability. With seed `314159`,
eight rows, and six features, CI asserts this live-library contract:

| Estimator | `shap_values` shape | `explainer.expected_value` shape | Reconstruction gate |
|---|---:|---:|---:|
| Random Forest | `(8, 6, 2)` | `(2,)` | `8/8` within `5e-5` |
| XGBoost | `(8, 6)` | scalar `()` | `8/8` within `1e-6` |

This difference is part of the implementation contract: Random Forest may
return an explicit class dimension, while binary XGBoost may return one
two-dimensional class-1 matrix. The normalizer also rejects ambiguous or
non-finite list, two-dimensional, or class-dimension outputs.

## Trusted model loading

Joblib model files are executable serialized artifacts. The public verifier
never loads them. The producing command loads a retained model only after it:

1. resolves the selected training run inside the configured trusted
   model-training root;
2. rejects path traversal, symlinks, nested-run substitution, and root escape;
3. verifies the training inventory and authoritative run digest;
4. verifies the stable and registered model IDs, manifest digest, model binary
   digest, algorithm, estimator class, target, horizon, class order, and ordered
   features; and
5. opens the model through the existing trusted artifact loader rather than a
   path supplied by an API client.

No model binary is copied into the explanation pack. Portable files use
repository-independent relative names and must not contain absolute host
paths.

## Feature and background matrices

The holdout keys come from the verified unified comparison and must contain
exactly 184 ordered `OGDC` rows. `feature_date`, `target_date`, and
`actual_class` must agree across M-001 through M-004. The corresponding feature
rows come from the frozen training input, and each model's ordered features
come from its immutable manifest.

The only preprocessing applied is the model's already-fitted, train-only median
imputer. Nothing is refitted. Target fields, `actual_class`, dates, ticker,
source, next-day values, and future values are forbidden from the explainer
matrix.

The interventional background contains 100 deterministic, evenly spaced rows
from the 853-row chronological training split. Those rows are transformed by
the same fitted imputer. The run records the ordered holdout dates, ordered
feature columns, transformed-matrix digests, background target dates, and
background-matrix digest.

See [the explainability data contract](EXPLAINABILITY_DATA_CONTRACT.md) for the
complete schemas and leakage rules.

## Explainer and output space

Every model uses `shap.TreeExplainer` with:

- positive class `1` (`up`);
- estimator class order exactly `[0, 1]`;
- `feature_perturbation="interventional"`;
- `model_output="probability"`;
- the deterministic 100-row training background; and
- the original fitted median imputer only.

Consequently, `direct_model_output` is the model's `probability_up`, and SHAP
values are class-1 probability-space contributions. No raw margin or log-odds
value is mixed into the run. The pack still keeps model-specific magnitudes and
does not average them into a universal importance score.

For prediction row `i`, the signed error is:

```text
reconstructed_i = expected_value + sum_feature(shap_value_i,feature)
additivity_error_i = reconstructed_i - direct_model_output_i
```

The documented absolute tolerance is `5e-5` for Random Forest and `1e-6` for
XGBoost. The interventional probability transform accumulates small
floating-point error differently across estimator families. An official run
fails if any of the 184 rows for any model exceeds its declared tolerance.
Additivity checks are not suppressed to force publication.

## Generated evidence

A successful run is staged, internally verified, and only then published with
collision-refusing no-replace operations under an ID of the form
`XAI-YYYYMMDD-HHMMSS-OGDC-tree-v1`:

```text
artifacts/explainability/<run_id>/
├── run_manifest.json
├── source_run_references.json
├── explainer_contract.json
├── feature_matrix_contract.json
├── additivity_report.json
├── global_feature_importance.csv
├── cross_model_feature_rankings.csv
├── local_explanations/
│   ├── M-001.csv
│   ├── M-002.csv
│   ├── M-003.csv
│   └── M-004.csv
├── audit_sample.csv
├── summaries/
│   ├── M-001.json
│   ├── M-002.json
│   ├── M-003.json
│   └── M-004.json
├── plots/
│   ├── M-001-global-importance.png
│   ├── M-002-global-importance.png
│   ├── M-003-global-importance.png
│   ├── M-004-global-importance.png
│   └── cross-model-rank-comparison.png
├── quality_report.json
├── final_summary.json
└── artifact_manifest.json
```

The producer also writes `<run_id>.zip` and `<run_id>.zip.sha256`. ZIP members
are sorted, use fixed timestamps and private file modes, and sit below one
top-level run directory. The ZIP contains explanation evidence only—no source
models, virtual environment, cache, secret, database, or unrelated file.

## Run and verify

The three source directories are generated, Git-ignored local evidence and
must be restored from a checksum-verified custodian export before execution.
The command refuses an existing run ID or destination. Omitting `--run-id`
uses a fresh UTC timestamp-derived ID.

```bash
source .venv-explainability/bin/activate
python scripts/run_tree_shap_explainability.py \
  --model-training-run \
  artifacts/model_training/TRN-20261020-191815-624662-nogit \
  --comparison-run \
  artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02 \
  --vertical-slice-run \
  artifacts/vertical_slice/VSL-20261020-OGDC-main-02 \
  --output-dir artifacts/explainability
```

The directory verifier recomputes global importance, cross-model ranks,
additivity summaries, and deterministic audit selection from the long-form
local rows. It checks the exact inventory, every digest, source identities,
model identities, dates, features, portability, and wording without loading a
model:

```bash
PAISA_XAI_RUN_ID="XAI-20261024-140320-OGDC-tree-v1"
python scripts/verify_explainability_run.py \
  "artifacts/explainability/$PAISA_XAI_RUN_ID" \
  --require-authenticated
```

After transferring the ZIP and sidecar together, verification remains offline:

```bash
PAISA_XAI_RUN_ID="XAI-20261024-140320-OGDC-tree-v1"
python scripts/verify_explainability_run.py \
  "artifacts/explainability/$PAISA_XAI_RUN_ID.zip" \
  --checksum \
  "artifacts/explainability/$PAISA_XAI_RUN_ID.zip.sha256" \
  --require-authenticated
```

The adjacent sidecar is a transport-corruption check, not a provenance trust
anchor. `status: PASS`/`integrity_status: PASS` mean the pack is internally
consistent. `authentication_status: AUTHENTICATED` means its raw
`artifact_manifest.json` bytes matched either the committed official allowlist
or a caller-supplied digest obtained through an independent channel. For a new
run, pass that value with `--expected-manifest-sha256` together with
`--require-authenticated`. The verifier validates PNG encoding and structure;
it does not rerender plots. Authentication proves that the accepted plot pixels
are the exact manifest-bound bytes identified by the trust anchor; it does not
by itself prove semantic correspondence between plots and their source tables.

## Global, local, and audit views

The global table reports, by model and feature, mean absolute attribution,
signed mean, median absolute value, maximum absolute value, rank, share of the
model's total absolute attribution, and positive/negative contribution counts.
The cross-model table compares ranks and retains each model's own magnitude and
output-space label.

The local files use one row per model, prediction, and feature. They retain the
dates, predicted and actual classes, both class probabilities, expected value,
direct and reconstructed outputs, signed error, transformed feature value,
SHAP value, absolute contribution rank, and direction.

The deterministic audit table attempts seven categories for every model:
highest-confidence class 1, highest-confidence class 0, correct positive,
correct negative, false positive, false negative, and largest absolute
additivity error. An unavailable category is recorded as unavailable rather
than replaced by a hand-picked example.

Interpret all of these fields under
[the SHAP interpretation policy](SHAP_INTERPRETATION_POLICY.md).

## Real execution status

Official retained run `XAI-20261024-140320-OGDC-tree-v1` completed with status
`PASS` and quality status `PASS`. It was produced from clean code commit
`2cd4145e5e01459d13d186a6f3b697e979821a9d`. Its independently committed
`artifact_manifest.json` SHA-256 is
`cd43f76a208bed6b549d6dbfb3b4e468dd91db021f4543f1cfefdbbc7a661fda`; the
deterministic ZIP SHA-256 is
`7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`.
Directory and ZIP verification returned internal `PASS` and authenticated
`AUTHENTICATED` without loading an executable model.

Every model was explained in positive-class probability space over exactly 184
shared project holdout rows and 19 ordered technical features. Each model used
the same deterministic 100-row chronological training background. All `184`
rows for each model passed its declared additivity tolerance, with zero failing
rows:

| Model | Algorithm | Maximum absolute error | Mean absolute error | Tolerance | Passing rows |
|---|---|---:|---:|---:|---:|
| `M-001` | Random Forest | `5.78484978308591e-06` | `5.781760213738571e-06` | `5e-5` | `184/184` |
| `M-002` | Random Forest | `4.302598055616613e-09` | `7.864803434113863e-10` | `5e-5` | `184/184` |
| `M-003` | Random Forest | `2.7466758000471714e-05` | `2.7464541568790406e-05` | `5e-5` | `184/184` |
| `M-004` | XGBoost | `1.5920226559273942e-07` | `3.5509845686483396e-08` | `1e-6` | `184/184` |

The official global mean-absolute-attribution rankings begin with:

| Model | Top five features, in rank order |
|---|---|
| `M-001` | `close_lag_1`, `return_volatility_5`, `close`, `close_lag_5`, `ma_20` |
| `M-002` | `return_volatility_5`, `close_lag_5`, `close`, `ma_20`, `rsi_14` |
| `M-003` | `close`, `ma_20`, `close_lag_1`, `close_lag_5`, `return_volatility_5` |
| `M-004` | `ma_20`, `close`, `close_lag_5`, `return_volatility_5`, `open` |

The summary classifies the three Random-Forest pairings as rank agreements:

| Pair | Spearman rank correlation | Top-five overlap |
|---|---:|---:|
| `M-001` / `M-002` | `0.8228070175438595` | `4/5` |
| `M-001` / `M-003` | `0.8210526315789475` | `5/5` |
| `M-002` / `M-003` | `0.7543859649122807` | `4/5` |

The three comparisons involving XGBoost are the recorded rank disagreements:

| Pair | Spearman rank correlation | Top-five overlap |
|---|---:|---:|
| `M-001` / `M-004` | `0.5964912280701754` | `4/5` |
| `M-003` / `M-004` | `0.6175438596491227` | `4/5` |
| `M-002` / `M-004` | `0.7` | `4/5` |

All seven deterministic audit categories are available for every model:
highest-confidence class 1, highest-confidence class 0, correct positive,
correct negative, false positive, false negative, and largest absolute
additivity error.

The recorded runtime is Python `3.13.7`, SHAP `0.54.1`, NumPy `2.3.5`, pandas
`2.3.3`, scikit-learn `1.9.1`, XGBoost `3.4.1`, joblib `1.6.0`, and Matplotlib
`3.11.2`. Final validation completed with `1350 passed, 2 skipped, 6 warnings`
in the backend and `132 passed` across 10 frontend test files, plus passing
frontend lint and build.

## Limitations

- The 184 dates are a previously exposed shared project holdout, not a fresh
  independent evaluation.
- SHAP explains the fitted model relative to the declared background; it does
  not discover economic causes or validate the input features as market
  mechanisms.
- Correlated technical indicators can divide or redistribute attribution.
- Evidence covers one stock, one historical period, one feature family, and
  four fitted tree configurations.
- Explanations do not measure prediction quality, portfolio value, robustness
  to regime change, or live performance.
- Feature importance does not establish profitability or future performance.
- The generator has no read-only API or dashboard view in this scope. Those are
  separate additive components and must consume only verified persisted
  evidence.
