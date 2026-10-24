# PAISA SHAP interpretation policy

This policy governs prose, tables, plots, API fields, dashboard labels, reports,
and demonstrations derived from PAISA tree-model SHAP evidence. It applies to
M-001 through M-004 on the OGDC shared project holdout.

The evidence is an academic simulation and model-behavior analysis, not
financial advice. It is not evidence of future performance.

## The claim SHAP supports

For this workflow, SHAP decomposes one fitted model's class-1 probability for
one retained row relative to a declared training-background distribution. It
supports statements such as:

- “This feature increased the model output for this prediction.”
- “This feature decreased the model output for this prediction.”
- “This feature contributed toward the model's class-1 probability.”
- “This feature was associated with this fitted-model prediction.”
- “This feature had the largest mean absolute attribution for M-002 over these
  184 rows.”
- “The two models ranked this feature differently on the shared project
  holdout.”

Always name the model, output space, class, population, and evidence window
when the surrounding context does not already establish them.

## Output-space meaning

All four retained tree models are explained in positive-class probability
space. Class `1` means the fitted model assigns the next observed close a
higher value than the feature-date close. `direct_model_output` is therefore
`probability_up`.

For one row:

```text
class-1 probability = expected value + sum of feature SHAP values
```

A positive SHAP value moves the reconstruction above the expected class-1
probability; a negative value moves it below. The stored unit is a probability
on the `0..1` scale. Multiplying by 100 expresses the same numerical
contribution in probability points, but it must not be described as the result
of actually changing that feature or as a causal treatment effect.

`probability_down` is retained for context, but the SHAP values explain class
1. Do not reverse a sign and present it as a separately computed class-0
explanation.

No raw-margin or log-odds attribution is present in schema 1.0. If a future
schema uses another output space, it must label and verify that space and must
not compare or pool its magnitudes with probability-space values.

## Additivity is a numerical check, not a validity proof

The expected value plus all feature contributions must reconstruct the direct
class-1 probability within the declared strict absolute tolerance: `5e-5` for
Random Forest and `1e-6` for XGBoost. The probability-space transform
accumulates small floating-point error differently across estimator families.
The pack reports signed and absolute errors, mean and maximum absolute error,
and passing/failing row counts.

Passing additivity demonstrates internal numerical consistency in the declared
output space. It does not prove that:

- the model is accurate or calibrated;
- the technical features are economically meaningful;
- the background is the only reasonable reference population;
- a portfolio based on the model is profitable;
- the historical association persists; or
- the model is safe for live decisions.

An additivity failure invalidates the official pack; it must not be waived or
hidden.

## Local explanations

Read one prediction as a complete decomposition, not as isolated causal
effects. The base value is the expected fitted-model output under the declared
100-row training background. The feature contributions then reconstruct that
row's direct output.

The local absolute-contribution rank answers “which attribution had the largest
magnitude in this decomposition?” It does not answer “which real-world factor
was most important to the market?” A large positive value means a contribution
toward class 1 relative to the background; a large negative value means a
contribution away from class 1.

`actual_class` is an evaluation-only field. It was not supplied to the
explainer and must not be described as an explanatory feature. Whether a row is
a true/false positive/negative changes the audit category, not the SHAP
calculation.

## Global explanations

Mean absolute SHAP value summarizes typical attribution magnitude for one
feature and model over the 184 rows. It deliberately discards sign. Signed mean
shows net direction, but opposing local effects can cancel. Median and maximum
absolute values distinguish typical from extreme behavior. Positive and
negative counts show how often the sign occurred, not how often the feature
“caused” a class.

The share of total absolute attribution is model- and population-relative. It
is not variance explained, accuracy contribution, economic value, or a
guarantee that removing the feature would reduce performance by that share.

Impurity-based Random Forest importance and SHAP importance are different
quantities. Do not use the pre-existing `feature_importance.csv` as a substitute
for verified pack `XAI-20261024-140320-OGDC-tree-v1` or claim the two should
have identical ranks.

## Cross-model comparison

Schema 1.0 uses class-1 probability space for all four models, so the units are
labelled consistently. Even so, retain each model's magnitude separately. The
models have different fitted tree structures and may distribute credit among
correlated indicators differently.

Prefer rank agreement, rank range, and top-feature overlap when comparing
patterns. A pairwise rank correlation describes similarity of feature order;
it does not show that the models learned the same function, have equal quality,
or will behave similarly outside these dates. Do not average the four raw SHAP
magnitudes into a universal importance score.

## Correlated technical indicators

The feature set intentionally contains related prices, lags, moving averages,
returns, volatility, RSI, and MACD terms. Attribution can be shared or shifted
among correlated inputs. A lower-ranked correlated feature may still contain
overlapping information, and a high rank does not isolate an independent
economic mechanism.

The interventional background makes the reference policy explicit but does not
make correlated observational inputs causal. Any sensitivity analysis with a
different background is a new explanation configuration and must receive a new
immutable run identity.

## Deterministic audit examples

Audit rows are selected mechanically: highest-confidence prediction in each
class, each available confusion-matrix category, and largest absolute
additivity error. These are diagnostic examples, not success stories. If a
category is absent, report it as unavailable. Do not replace it with an
attractive or easier-to-explain row.

When showing an audit example, include at least the stable model ID,
feature/target dates, predicted and actual classes, both probabilities, output
space, expected value, direct output, reconstruction, additivity error, and top
contributors.

## Relationship to prediction and portfolio results

SHAP explains the fitted classifier output. It does not explain portfolio
return, trade timing, fees, or buy-and-hold differences. Those outcomes also
depend on the decision thresholds, current holdings, position sizing,
next-session-open execution, whole-share rounding, fees, and close valuation.

Do not use a feature attribution to claim that a feature produced a trade or
profit unless the separate policy and portfolio state are traced explicitly;
even then, describe the software decision chain rather than economic causality.

The 184 dates are the reused shared project holdout. Explanations on these rows
describe already-exposed comparative evidence, not a fresh independent test.

## Prohibited claims

Do not write or imply that an attribution:

- caused the stock to rise;
- caused a market movement;
- guarantees an increase or decrease;
- establishes economic causality;
- proves profitability;
- validates a trading recommendation;
- identifies a universal market driver;
- establishes general superiority across stocks or regimes; or
- constitutes financial advice.

Avoid anthropomorphic shorthand such as “the model knows” or “the feature
wanted the price to rise.” Name the fitted output and the observed attribution.

## Required caveats

Any result summary must disclose:

- OGDC-only, daily-only scope;
- four retained price-only tree configurations;
- one technical-feature family and historical period;
- the reused 184-date shared project holdout;
- the declared training-background policy;
- class-1 probability output space;
- correlated-feature attribution limits;
- no causal or future-performance claim; and
- the exact academic disclaimer: “Academic simulation; not financial advice.”

## Accepted evidence reference

The accepted local schema-1.0 pack is
`XAI-20261024-140320-OGDC-tree-v1`, produced from clean commit
`2cd4145e5e01459d13d186a6f3b697e979821a9d`. Its exact 24-file inventory and
quality report are `PASS`; the ZIP SHA-256 is
`7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`.
Its independently committed `artifact_manifest.json` SHA-256 trust root is
`cd43f76a208bed6b549d6dbfb3b4e468dd91db021f4543f1cfefdbbc7a661fda`.
Strict authenticated directory, ZIP, and extracted-directory verification all
passed. The ZIP sidecar separately checks transport integrity and is not a
provenance credential; PNG checks are structural rather than a rerender from
CSV evidence, while the authenticated manifest binds the accepted plot bytes.
An internal `PASS` without `authentication_status: AUTHENTICATED` is not an
official provenance claim. Cite results from that authenticated pack, not from
impurity-based importance or a synthetic/unit fixture. The certified direct
runtime versions are pinned with `constraints-explainability.txt`, and a
dedicated Python 3.13.7 CI job exercises the real SHAP library against small
Random Forest and XGBoost estimators without retained artifacts.

For M-001 through M-004 respectively, the maximum/mean absolute additivity
errors were `5.78484978308591e-06` / `5.781760213738571e-06`,
`4.302598055616613e-09` / `7.864803434113863e-10`,
`2.7466758000471714e-05` / `2.7464541568790406e-05`, and
`1.5920226559273942e-07` / `3.5509845686483396e-08`. Every one of 184 rows per
model passed the applicable `5e-5` Random Forest or `1e-6` XGBoost tolerance,
and all seven audit categories were available for every model.

The ordered top five features are:

- M-001: `close_lag_1`, `return_volatility_5`, `close`, `close_lag_5`, `ma_20`;
- M-002: `return_volatility_5`, `close_lag_5`, `close`, `ma_20`, `rsi_14`;
- M-003: `close`, `ma_20`, `close_lag_1`, `close_lag_5`,
  `return_volatility_5`; and
- M-004: `ma_20`, `close`, `close_lag_5`, `return_volatility_5`, `open`.

The reported Spearman/top-five-overlap agreement pairs are M-001/M-002
`0.8228070175438595`/4, M-001/M-003 `0.8210526315789475`/5, and M-002/M-003
`0.7543859649122807`/4. The disagreement pairs are M-001/M-004
`0.5964912280701754`/4, M-003/M-004 `0.6175438596491227`/4, and M-002/M-004
`0.7`/4. Interpret these only as rank-pattern evidence for one stock, one
period, one feature family, and four fitted tree configurations on a previously
exposed shared holdout. They do not establish causality, profitability, or
future results. Academic simulation; not financial advice.

## Review checklist

Before publishing an explanation claim, verify that:

1. the directory or ZIP passes with `--require-authenticated` and reports
   `authentication_status: AUTHENTICATED`;
2. for a non-allowlisted run, an independently retained
   `--expected-manifest-sha256` was supplied;
3. any ZIP sidecar is described only as transport-integrity evidence, and plot
   checks are described as structural rather than semantic rerendering;
4. the cited model and run IDs match the authenticated artifact;
5. the class and probability output space are stated;
6. the sign is described relative to class 1 and the expected value;
7. the claim uses the correct local or global population;
8. cross-model magnitudes have not been pooled;
9. prediction, attribution, policy, and portfolio conclusions remain separate;
10. shared-holdout reuse and limitations are visible; and
11. causal, guaranteed-performance, profitability, and advice wording is absent.
