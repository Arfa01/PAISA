# PAISA explainability data contract

This document defines schema `1.0` for PAISA's immutable tree-model SHAP
evidence. It is normative for packs produced by
`paisa.explainability.tree_shap` and for the offline verifier. The contract is
limited to M-001 through M-004 on the authoritative OGDC shared project
holdout.

All outputs are academic model-behavior evidence, not financial advice. The
contract does not grant permission to describe attribution as economic
causality.

## Fixed identity and source contract

| Field | Required value |
|---|---|
| `schema_version` | `1.0` |
| `artifact_type` | `paisa_tree_shap_explainability` |
| ticker | `OGDC` |
| holdout term | `shared project holdout` |
| holdout rows | exactly `184` |
| stable model IDs | ordered `M-001`, `M-002`, `M-003`, `M-004` |
| class explained | integer `1` (`up`) |
| class order | exactly `[0, 1]` |
| output space | `probability` |
| SHAP version | exactly `0.54.1` |
| prediction replay tolerance | absolute `1e-6`, zero relative tolerance |
| numerical tolerance | `5e-5` for `random_forest`; `1e-6` for `xgboost` |
| disclaimer | `Academic simulation; not financial advice.` |

The required source runs and integrity roots are:

| Source role | Required run | Integrity-root SHA-256 |
|---|---|---|
| `model_training` | `TRN-20261020-191815-624662-nogit` | `8d5113bc8b67f43a76639d7397b905217bda1f94321183f0438e33448f689023` |
| `unified_comparison` | `CMP-20261024-OGDC-price-only-02` | `e4b0b12bb565fa6df1f23b7e8855f84569ffcefb795c36c4bfeb4a1e2c82849d` |
| `vertical_slice` | `VSL-20261020-OGDC-main-02` | `2ce2f7581a5baff8487b09ae695213745c704f11715e6bd5abd0e99009337f31` |

The frozen modelling dataset SHA-256 is
`5ba9ce1c99af8873163c78ae9466c9a7de8042bb1deb3a56eeac525f12deb08e`.
The training source fingerprint is
`3b82245209925d7a4678fe7b276a3e58b5dac73d`, and the canonical digest of
the ordered 184 holdout-key records is
`730755ddb7993d829ddd4d7f5afaec4d5ca5617311d642f4a33722d251a5a85a`.
`source_run_references.json` also binds each registered model ID, algorithm,
model binary digest, model-manifest digest, and authoritative
comparison-prediction source digest. The digest is recorded in the evidence;
the source prediction payload need not be copied into the portable pack. Paths
in this file are logical relative names; absolute host paths are forbidden.

## Model identities

| Stable ID | Registered model ID | Algorithm | Estimator class |
|---|---|---|---|
| `M-001` | `M-001-6ce88dc384de975e-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | `random_forest` | `RandomForestClassifier` |
| `M-002` | `M-002-5a0160ff1b7a8043-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | `random_forest` | `RandomForestClassifier` |
| `M-003` | `M-003-23efd4aed873bf08-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | `random_forest` | `RandomForestClassifier` |
| `M-004` | `M-004-6c60cc92f77a495a-b04d73fd-b4a31806-0346-4786-8247-f8922092e2f9` | `xgboost` | `XGBClassifier` |

A stable ID, registered ID, algorithm, estimator class, binary digest, or
manifest digest mismatch is fatal. Model IDs are data, not aliases that a
caller may remap.

After trusted loading, the producer replays both class probabilities from each
fitted pipeline and requires them to match the checksum-bound comparison
predictions within absolute `1e-6` and zero relative tolerance. This inference
replay gate is separate from the estimator-family SHAP additivity tolerances.

## Feature-matrix contract

The target is `target_next_day_up`, whose positive condition is
`close[target_date] > close[feature_date]` at a horizon of one trading session.
The ordered feature list for every retained model is:

1. `open`
2. `close`
3. `volume`
4. `daily_return`
5. `log_return`
6. `close_lag_1`
7. `close_lag_5`
8. `close_lag_10`
9. `volume_lag_1`
10. `ma_5`
11. `ma_10`
12. `ma_20`
13. `return_volatility_5`
14. `return_volatility_10`
15. `volume_change`
16. `rsi_14`
17. `macd_line`
18. `macd_signal`
19. `macd_hist`

The producer nevertheless reads the order from each immutable model manifest
and records it per model. It never relies on this document as an executable
substitute for the manifest.

`feature_matrix_contract.json` contains:

- exactly 184 ordered feature dates and 184 ordered target dates;
- one `holdout_rows` record per date with ticker and evaluation-only
  `actual_class`;
- `holdout_rows_sha256` equal to the fixed canonical digest above;
- the source dataset digest;
- the deterministic background policy and its 100 target dates;
- feature-value space (`output of the original fitted median imputer`);
- `preprocessing_refit: false`; and
- for each model, registered ID, ordered columns, feature and row counts,
  transformed-matrix digest, and explicit false flags for target columns,
  `actual_class`, and future values in the matrix.

Forbidden explainer-input fields include `feature_date`, `target_date`, ticker,
source, `actual_class`, `target_next_day_up`, fields prefixed `target_`, and
fields prefixed `next_`. The producer fails if the four prediction files do not
agree on ordered dates and actual classes.

## Preprocessing and background contract

The trusted model must be a fitted two-step pipeline named `imputer`, then
`model`. The producer calls `transform` on the existing fitted median imputer;
it does not call `fit`, `fit_transform`, feature selection, calibration, or
training.

The SHAP background is 100 chronological training rows selected at evenly
spaced deterministic indices from the 853-row training split. It is not sampled
from validation or the shared project holdout. Background features use the same
ordered columns and fitted imputer as their model. Every transformed value must
be finite.

## Explainer contract

`explainer_contract.json` has top-level fields `schema_version`, `models`, and
`output_space_rule`. Each of its four ordered model records contains:

- `model_id` and `registered_model_id`;
- `explainer_type: shap.TreeExplainer`;
- algorithm and estimator class;
- `class_explained: 1` and `estimator_class_order: [0, 1]`;
- finite `expected_value`;
- `output_space: probability`;
- `feature_perturbation: interventional`;
- background policy, row count, and matrix SHA-256;
- `shap_version: 0.54.1`;
- feature count and `row_count: 184`; and
- algorithm-specific `additivity_tolerance` (`5e-5` for Random Forest,
  `1e-6` for XGBoost).

List-of-class SHAP results, two-dimensional binary results, and unambiguous
three-dimensional class results normalize to an `(184, feature_count)` class-1
matrix. Missing, ambiguous, incorrectly shaped, or non-finite results fail.

## Local explanation schema

Each `local_explanations/M-00N.csv` is long-form: one row per prediction and
feature. With 184 predictions and 19 features, the authoritative models are
expected to produce 3,496 rows each. The verifier derives the expected count
from the contract rather than accepting a hard-coded CSV length.

| Column | Type and meaning |
|---|---|
| `schema_version` | String `1.0`. |
| `model_id` | Stable ID `M-001` through `M-004`. |
| `registered_model_id` | Exact immutable registered ID. |
| `algorithm` | `random_forest` or `xgboost`. |
| `ticker` | String `OGDC`. |
| `feature_date` | ISO date of the information set. |
| `target_date` | ISO date of the next observed session being evaluated. |
| `predicted_class` | Integer `0` or `1`. |
| `probability_up` | Finite class-1 probability in `[0, 1]`. |
| `probability_down` | Finite class-0 probability in `[0, 1]`; the two probabilities sum to one within `1e-6`. |
| `actual_class` | Integer `0` or `1`, retained for evaluation only and never passed to the explainer. |
| `output_space` | String `probability`. |
| `expected_value` | Finite class-1 probability-space base value. |
| `direct_model_output` | Direct fitted-model class-1 probability; equals `probability_up`. |
| `reconstructed_model_output` | `expected_value + sum(shap_value)` for the prediction. |
| `additivity_error` | Signed `reconstructed_model_output - direct_model_output`. |
| `feature_name` | One ordered manifest feature. |
| `feature_value` | Finite value after the fitted median imputer. |
| `shap_value` | Finite class-1 probability-space contribution. |
| `absolute_contribution_rank` | Complete `1..feature_count` permutation per prediction, ordered by descending absolute contribution then feature name. |
| `contribution_direction` | `increased_model_output`, `decreased_model_output`, or `neutral`, derived from the SHAP sign. |

Metadata repeated across the feature rows for one prediction must be identical.
Each target date must contain every feature exactly once in manifest order.

## Additivity schema

`additivity_report.json` records the signed-error definition and one record per
model with:

- model and registered IDs;
- output space;
- mean and maximum absolute error;
- row counts passing and failing tolerance; and
- numerical tolerance.

Every row must pass. The report and official run status are `PASS` only when
all four failing counts are zero.

## Global importance schema

`global_feature_importance.csv` contains one record per model and feature:

| Column | Meaning |
|---|---|
| `mean_absolute_shap_value` | Mean attribution magnitude across 184 rows. |
| `signed_mean_shap_value` | Mean signed class-1 contribution. |
| `median_absolute_shap_value` | Median attribution magnitude. |
| `maximum_absolute_shap_value` | Largest observed magnitude. |
| `feature_rank` | Deterministic descending rank by mean absolute value, then feature name. |
| `share_of_total_absolute_attribution` | Feature absolute-attribution sum divided by the model-wide total. |
| `positive_contribution_count` | Rows whose contribution is positive. |
| `negative_contribution_count` | Rows whose contribution is negative. |

The table also retains schema, stable and registered IDs, algorithm, output
space, and feature name. Shares and magnitudes are calculated within each
model; the schema does not define a pooled universal importance.

`cross_model_feature_rankings.csv` retains each model's rank, mean absolute
value, and output-space label, plus Random Forest mean/range and all-model rank
range/standard deviation. The verifier recomputes the file from local rows.

## Audit-sample schema

`audit_sample.csv` contains exactly one record for each model/category pair:

- `highest_confidence_class_1`;
- `highest_confidence_class_0`;
- `correct_positive`;
- `correct_negative`;
- `false_positive`;
- `false_negative`; and
- `largest_absolute_additivity_error`.

Selection is deterministic, with target date as the stable tie-breaker. A
category with no eligible prediction has `available: false` and
`unavailability_reason: no_prediction_matches_category`; its dates, classes,
probabilities, and error remain empty. Fabricated substitutes are forbidden.

## Summaries and plots

Each `summaries/M-00N.json` is a deterministic projection of verified local and
global rows. It contains top features, top positive and negative signed means,
pairwise rank agreements/disagreements, and the top local contributors for each
audit category. Its interpretation statement must describe association with
the model output rather than causality.

PNG files are presentation derivatives. Global plots display the leading mean
absolute values for one model in class-1 probability space. The cross-model
plot displays ranks. Every PNG visibly carries the exact academic-simulation
and not-financial-advice disclaimer so it remains qualified when viewed apart
from the pack. The CSV/JSON evidence, not pixels, is the semantic source of
record.

## Run, quality, and artifact manifests

`run_manifest.json` binds the run ID, creation time, ticker, holdout term,
stable and registered model order, installed package versions, background
policy, code provenance, contract filenames, and disclaimer. SHAP must be
exactly `0.54.1`. Other installed versions are recorded rather than hidden.

`quality_report.json` is `PASS` only after source identity, trusted loading,
checksums, model/class/feature contracts, exact dates, leakage exclusions,
fitted preprocessing, finite SHAP values, additivity, audit selection,
portability, executable-model exclusion, and wording policy pass.

`artifact_manifest.json` inventories every other file by normalized relative
path and SHA-256. The verifier rejects both missing and unexpected files; it
does not accept an internally consistent subset.

## Portable verification contract

The offline verifier needs no model binary, joblib deserialization, XGBoost
runtime, retraining, or inference. For a directory or ZIP it:

1. validates the run and artifact schemas;
2. requires the exact layout and inventory;
3. validates every file digest;
4. rejects symlinks, absolute paths, executable-model suffixes, malformed PNGs,
   and forbidden causal/advice wording;
5. validates source fingerprints, model identities, ordered columns, and the
   184-date alignment;
6. recomputes prediction-level additivity from local feature rows;
7. recomputes global importance, cross-model rankings, audit selection,
   summaries, and final summary; and
8. for ZIP input, validates the sidecar, canonical member metadata, one safe
   top-level run directory, size limits, and extraction containment.

Any mismatch fails verification. The verifier cannot attest that model
inference originally ran correctly on an untrusted host; it attests that the
portable pack is internally consistent with this sealed contract and its fixed
source identities.

## Accepted schema-1.0 instance

The accepted local pack is `XAI-20261024-140320-OGDC-tree-v1`, produced from
clean code version/source fingerprint
`2cd4145e5e01459d13d186a6f3b697e979821a9d`. Its exact inventory contains 24
files, its quality status is `PASS`, and its ZIP SHA-256 is
`7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`.
Directory-plus-checksum, ZIP-plus-sidecar, and extracted offline verification
all passed. The manifest records Python 3.13.7, NumPy 2.3.5, pandas 2.3.3,
scikit-learn 1.9.1, XGBoost 3.4.1, joblib 1.6.0, Matplotlib 3.11.2, and SHAP
0.54.1.

All four model records contain 184 rows and 19 transformed features in
positive-class probability space. M-001 through M-004 respectively report
maximum/mean absolute additivity errors of
`5.78484978308591e-06` / `5.781760213738571e-06`,
`4.302598055616613e-09` / `7.864803434113863e-10`,
`2.7466758000471714e-05` / `2.7464541568790406e-05`, and
`1.5920226559273942e-07` / `3.5509845686483396e-08`. Their tolerances are
`5e-5`, `5e-5`, `5e-5`, and `1e-6`; all 184 rows per model passed, with zero
failures. All seven audit categories were available for every model.

This is evidence about one stock, one period, one feature family, and four
fitted tree configurations on a previously exposed shared project holdout. It
does not establish causality, profitability, or future results. Academic
simulation; not financial advice.

## Immutability and permissions

Run IDs match `XAI-YYYYMMDD-HHMMSS-OGDC-tree-v1` with an optional safe suffix.
The output root, destination, and intermediate paths must not be symlinks. The
producer writes private staging files, verifies them, creates a deterministic
ZIP and sidecar, rechecks sources, and publishes without replacement. Any
collision fails. A failed build removes its staging tree and leaves no final
`PASS` directory.

Generated directories, ZIPs, and local environments remain ignored and must
not be committed. A schema change requires a new schema version and verifier;
old immutable packs are never rewritten in place.
