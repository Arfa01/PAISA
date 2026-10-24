# PAISA Vertical-Slice Results Interpretation

## Scope and evidence

This interpretation applies to the completed retained run
`VSL-20261020-OGDC-main-02` and its referenced training run
`TRN-20261020-191815-624662-nogit`. The vertical-slice manifest reports
`PASS`; the run uses daily OGDC observations and one common chronological
train/validation/test contract.

The primary evidence is retained locally under:

- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/final_summary.json`
- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/model_metrics.csv`
- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/portfolio_metrics.csv`
- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/combined_leaderboard.csv`
- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/experiments/`
- `artifacts/model_training/TRN-20261020-191815-624662-nogit/split_manifest.json`
- `artifacts/model_training/TRN-20261020-191815-624662-nogit/reports/validation_leaderboard.csv`
- `artifacts/model_training/TRN-20261020-191815-624662-nogit/reports/test_leaderboard.csv`

These generated artifacts are intentionally ignored by Git and must not be
committed. Paths in this document are repository-relative so that the evidence
can be located on another checkout containing the retained run bundle.

All returns, trades, and portfolio values below are from an academic
simulation. They are not financial advice and are not evidence of future
performance.

## Keep the two questions separate

The run answers two related but different questions:

1. **Prediction evaluation:** how well does a model classify whether the next
   session's close is above the feature-date close?
2. **Portfolio simulation:** what happens when prediction probabilities pass a
   fixed decision policy, become stateful buy/sell/hold signals, execute at the
   next session's open, incur fees, and are valued at daily closes?

Accordingly, PAISA keeps the criteria separate. `M-001` is the model selected
from validation prediction evidence, while `M-002` has the best test-period
simulated model-portfolio result. Both statements are true and refer to
different measurements.

## Prediction results

All four variants used the same 853 training rows, 182 validation rows, and 184
test rows. Model selection used validation balanced accuracy first, then
validation F1, validation accuracy, and model ID as deterministic tie-breakers.
The held-out test metrics were not consulted when selecting the prediction
model.

| Variant | Algorithm/configuration | Validation accuracy | Validation balanced accuracy | Validation F1 | Test accuracy | Test balanced accuracy | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| `M-001` | Random Forest, PAISA v0 baseline | 0.532967 | **0.530204** | 0.458599 | 0.516304 | 0.512289 | 0.226087 |
| `M-002` | Random Forest, conservative | 0.500000 | 0.500846 | 0.513369 | **0.527174** | **0.524459** | 0.364964 |
| `M-003` | Random Forest, flexible | 0.516484 | 0.513350 | 0.428571 | 0.510870 | 0.506794 | 0.210526 |
| `M-004` | XGBoost, CPU histogram | 0.494505 | 0.494503 | 0.488889 | 0.494565 | 0.492910 | **0.400000** |

`M-001` is therefore the **validation-selected prediction model**. Its
validation balanced accuracy is `0.530204`, the highest of the four variants.
Although `M-002` has the highest test accuracy (`0.527174`) and test balanced
accuracy (`0.524459`), choosing it because of those held-out results would
change the pre-declared selection rule and use the test set for model
selection. The test values should remain descriptive evidence.

These scores are close to chance-level classification and should be presented
as modest, stock- and period-specific research results rather than a strong
predictive breakthrough.

## Simulated portfolio results

Every strategy started with PKR 100,000 in an independent portfolio. The common
policy used a 0.60 buy threshold, a 0.40 sell threshold, a 50% maximum
allocation, whole-share rounding, 10 basis-point proportional transaction
costs, zero modeled slippage, next-session-open execution, and close valuation.

| Series | Ending value (PKR) | Total return | Maximum drawdown | Sharpe ratio | Trades | Fees (PKR) | Return minus buy-and-hold |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M-001` strategy | 99,298.24 | -0.70176% | 2.19966% | -0.320854 | 2 | 99.06 | -6.94965 pp |
| `M-002` strategy | **105,987.04** | **5.98704%** | 5.66996% | **1.139943** | 3 | 160.95 | -0.26085 pp |
| `M-003` strategy | 99,166.07 | -0.83393% | **1.18467%** | -0.783988 | 2 | 98.93 | -7.08182 pp |
| `M-004` strategy | 100,190.05 | 0.19005% | 7.12784% | 0.079675 | **34** | **1,751.06** | -6.05784 pp |
| Cash-only | 100,000.00 | 0.00000% | 0.00000% | undefined | 0 | 0.00 | -6.24789 pp |
| Controlled buy-and-hold | **106,247.89** | **6.24789%** | 12.79045% | 0.553179 | 1 | 49.75 | 0.00000 pp |

Here, `pp` means percentage points. The buy-and-hold comparator uses the same
50% maximum-allocation constraint as the model portfolios; it is not a
conventional 100%-invested buy-and-hold calculation.

### What the portfolio table supports

- `M-002` is the **best simulated model portfolio** for this test period. It
  ends at PKR 105,987.04 for a 5.98704% return, with the highest simulated
  model-portfolio Sharpe ratio.
- `M-003` has the **lowest model-portfolio maximum drawdown**, 1.18467%. That
  does not make it the best overall portfolio: it was in the market for very
  little of the period and ended with a negative 0.83393% return.
- `M-004` **overtrades relative to the other variants under this policy**. It
  generates 34 trades, compared with two or three for each Random Forest
  portfolio, and pays PKR 1,751.06 in fees. Its 0.19005% net return does not
  compensate for the much higher activity and costs in this run.
- **No model portfolio beats the controlled buy-and-hold benchmark in total
  return.** Buy-and-hold returns 6.24789%; `M-002`, the nearest model strategy,
  returns 5.98704% and trails by 0.26085 percentage points (PKR 260.85 from the
  common starting value).
- Buy-and-hold's higher return comes with higher path risk in this sample. Its
  12.79045% maximum drawdown is 7.12049 percentage points above `M-002`'s
  5.66996% maximum drawdown. This is a return/drawdown trade-off, not proof that
  either approach is generally superior.

## Why accuracy and profitability differ

Prediction accuracy gives equal weight to every classified test row. Portfolio
profitability depends on the size and sequence of price movements and on
whether a probability crosses a trading threshold while the portfolio is
already holding shares or cash. It is also affected by the next-open execution
price, whole-share sizing, the 50% allocation cap, fees, and the duration of
each position.

The target and executable return interval are also different:

```text
prediction target: feature-date close -> next-session close
new trade return:  next-session open  -> later executable exit / daily close valuation
```

The overnight movement from the feature-date close to the next-session open is
part of the classification interval but cannot be earned by a new position
opened at that next-session open. Consequently, a model can have slightly
better classification metrics yet produce a worse simulated portfolio, or the
reverse. `M-001` being the validation-selected predictor while `M-002` produces
the best model-portfolio result is an example of this distinction, not a
contradiction.

## Defensible conclusion

For this retained run, `M-001` is the correctly selected prediction model under
the declared validation rule, `M-002` is the strongest simulated model
portfolio, `M-003` has the lowest model-portfolio drawdown, and `M-004` trades
far more often and incurs far higher fees. None of the model portfolios exceeds
the controlled buy-and-hold total return.

These conclusions apply only to **OGDC**, the recorded historical period, the
daily technical feature set, the four fitted configurations, the chronological
split, the fixed probability policy, the 50% allocation rule, and the stated
execution and cost assumptions. They do not establish general performance
across stocks, periods, feature sets, policies, costs, market regimes, or live
trading.

## Unified M-001 through M-006 comparison

The authoritative immutable comparison pack
`CMP-20261024-OGDC-price-only-02` extends the price-only review to the two LSTM
variants without retraining a model, invoking inference, or calling the
production backtest runner. `CMP-20261024-OGDC-price-only-01` is a superseded,
non-portable historical draft and is not current publication evidence. The
final `-02` ZIP SHA-256 is
`8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`,
and its verified total file count is `65`.

The hardened pack recomputes prediction metrics from persisted rows and uses
two raw market files plus six signal files to replay policy, execution, trades,
accounting, return, drawdown, and fees with exact `Decimal` arithmetic. Its
vertical-slice integrity root is represented by a digest-allowlisted redacted
certificate, and portable-content hygiene scans all text recursively. The
metrics expose exactly two logical benchmark rows, cash-only and controlled
buy-and-hold, each with six independent ledger-provenance records. All six
models share 182 validation dates and the exact 184 dates from `2025-12-16`
through `2026-09-15` in the **shared project holdout**.
That period and its M-001 through M-004 results had been exposed previously, so
the shared project holdout is comparative evidence rather than a fresh
independent performance estimate.

| Evidence question | Verified outcome |
|---|---|
| Validation-selected model across M-001--M-006 | `M-001`; validation balanced accuracy `0.5302041802585478` |
| LSTM-family validation preference | `M-005`; validation balanced accuracy `0.4792799323426362` |
| Best model shared project holdout balanced accuracy | `M-006`; `0.544251447477254` |
| Best model shared project holdout F1 | `M-005`; `0.6396761133603239` |
| Majority-class baseline on the shared project holdout | Balanced accuracy `0.5`; F1 `0.0` |
| Previous-direction baseline on the shared project holdout | Balanced accuracy `0.5271771239513174`; F1 `0.5245901639344263` |
| Best simulated model portfolio by total return | `M-002`; ending value PKR 105,987.04; return `5.98704%` |
| Lowest-drawdown model portfolio | `M-003`; maximum drawdown `1.18467%` |
| Lowest-fee model portfolio | `M-003`; fees PKR 98.93 |
| Cash-only benchmark | Ending value PKR 100,000.00; zero return, drawdown, trades, and fees |
| Controlled buy-and-hold benchmark | Ending value PKR 106,247.89; return `6.24789%` |
| Did any model beat buy-and-hold in total return? | No; `M-002`, the closest model, trails by `0.26085` percentage points |

These are separate findings, not inputs to one cross-criterion ranking.
Validation keeps `M-001` selected even though `M-006` has the highest shared
project holdout balanced accuracy, `M-005` has the highest shared project
holdout F1, `M-002` has the highest simulated model-portfolio return, and
`M-003` has the lowest model-portfolio drawdown and fees. The controlled
buy-and-hold comparator uses the same 50% allocation constraint as the model
portfolios; it is not a fully invested benchmark.

This is a price-only, tabular/report-only comparison. It does not integrate
sentiment, contains no plots, and does not import Matplotlib. CLI results use
relative artifact names; default failures omit tracebacks and opt-in debug
diagnostics are sanitized.

See [the unified comparison contract](UNIFIED_PRICE_MODEL_COMPARISON.md) for
the authoritative source bindings, complete schemas, verification commands,
rankings, limitations, and portable checksum workflow. All predictions,
trades, returns, and portfolio values are academic simulations, are not
financial advice, and are not evidence of future performance.

## Tree-model SHAP interpretation status

Official retained pack `XAI-20261024-140320-OGDC-tree-v1` was produced from
clean commit `2cd4145e5e01459d13d186a6f3b697e979821a9d`, passed directory and
ZIP verification, and records quality status `PASS`. Its ZIP SHA-256 is
`7876ed3dba620c3c6fcb6213302001ac4a42518cf0ef3d8c951b2c2fc2af13a1`.
It explains each fitted model's class-1 probability over the exact `184 × 19`
shared project holdout matrix relative to 100
deterministically selected chronological training rows. A positive SHAP value
increases the reconstructed `probability_up` relative to the expected value; a
negative value decreases it. Every direct probability must equal the expected
value plus its feature contributions within the declared estimator-family
tolerance (`5e-5` for Random Forest, `1e-6` for XGBoost).

| Model | Top five global features, in rank order | Maximum / mean absolute additivity error |
|---|---|---:|
| `M-001` | `close_lag_1`, `return_volatility_5`, `close`, `close_lag_5`, `ma_20` | `5.78484978308591e-06` / `5.781760213738571e-06` |
| `M-002` | `return_volatility_5`, `close_lag_5`, `close`, `ma_20`, `rsi_14` | `4.302598055616613e-09` / `7.864803434113863e-10` |
| `M-003` | `close`, `ma_20`, `close_lag_1`, `close_lag_5`, `return_volatility_5` | `2.7466758000471714e-05` / `2.7464541568790406e-05` |
| `M-004` | `ma_20`, `close`, `close_lag_5`, `return_volatility_5`, `open` | `1.5920226559273942e-07` / `3.5509845686483396e-08` |

All 184 rows pass tolerance for every model, and all seven audit categories are
available: both highest-confidence classes, correct positive, correct negative,
false positive, false negative, and largest absolute additivity error.

The Random-Forest pairings are the recorded rank agreements:
`M-001`/`M-002` has Spearman correlation `0.8228070175438595` and `4/5`
top-feature overlap; `M-001`/`M-003` has `0.8210526315789475` and `5/5`; and
`M-002`/`M-003` has `0.7543859649122807` and `4/5`. The `M-004` comparisons
are the recorded disagreements: `M-001`/`M-004` is `0.5964912280701754`,
`M-003`/`M-004` is `0.6175438596491227`, and `M-002`/`M-004` is `0.7`; each
shares four of the top five features.

The recorded environment is Python `3.13.7`, SHAP `0.54.1`, NumPy `2.3.5`,
pandas `2.3.3`, scikit-learn `1.9.1`, XGBoost `3.4.1`, joblib `1.6.0`, and
Matplotlib `3.11.2`. Final validation completed with
`1339 passed, 2 skipped, 6 warnings` in the backend and `132 passed` across 10
frontend test files, plus passing frontend lint and build.

That numerical reconstruction does not establish model accuracy, calibration,
economic causality, profitability, or future performance. `actual_class` is an
evaluation-only field and never enters the explainer matrix. Global mean
absolute attribution measures model-specific magnitude over these 184 rows; it
is not variance explained or economic importance. Correlated lags, moving
averages, returns, volatility, RSI, and MACD terms may share attribution.

All four models use probability space, but raw magnitudes remain separated by
model and are not averaged into a universal importance score. Cross-model rank
agreement describes similarity of feature ordering only. It does not show that
models learned the same function or that one is more accurate or profitable.

The explanation population is the already exposed shared project holdout, not
a fresh independent evaluation. Explanations cover one stock, one historical
period, one technical-feature family, and four fitted tree configurations. See
[the explainability runbook](TREE_MODEL_SHAP_EXPLAINABILITY.md), [data
contract](EXPLAINABILITY_DATA_CONTRACT.md), and [interpretation
policy](SHAP_INTERPRETATION_POLICY.md). All explanation output is an academic
simulation and model-behavior analysis, not financial advice or evidence of
future performance.
