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

Accordingly, PAISA does not declare one universal winner. `M-001` is the model
selected from validation prediction evidence, while `M-002` has the best
test-period simulated model-portfolio result. Both statements are true and
refer to different measurements.

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
