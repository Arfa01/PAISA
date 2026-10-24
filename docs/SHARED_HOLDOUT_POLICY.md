# PAISA shared project holdout policy

This policy defines how PAISA may describe and present the common evaluation
period used by the immutable M-001 through M-006 price-only comparison. It is
an academic simulation and is not financial advice.

## Required terminology

The 184-date M-001 through M-006 comparison period is the **shared project
holdout**. It must not be described as a fresh, untouched, independent, or
unseen test set because the dates and M-001 through M-004 results were exposed
before the LSTM comparison.

The canonical contract is:

- symbol: `OGDC`;
- alignment key: `ticker`, `feature_date`, and `target_date`;
- feature dates: `2025-12-15` through `2026-09-14`;
- target dates: `2025-12-16` through `2026-09-15`;
- common target-date count: `184`;
- prediction horizon: one observed trading session; and
- target: whether the target-session close is greater than the feature-session
  close.

All six models must appear exactly once on those same ordered dates. A missing,
extra, duplicated, or differently labelled date invalidates the comparison.

## Selection and reporting

Model selection uses the separate 182-date validation period. The
validation-only selection rule is balanced accuracy, followed by F1, accuracy,
and ascending model ID. Shared project holdout metrics are descriptive and may
not change the validation-selected label.

PAISA reports these outcomes separately:

- validation-selected model;
- best shared project holdout balanced accuracy;
- best shared project holdout F1;
- best simulated model portfolio return;
- lowest simulated model-portfolio drawdown;
- lowest simulated model-portfolio fees;
- cash-only and controlled buy-and-hold benchmarks; and
- whether any model portfolio beat controlled buy-and-hold.

There is no universal winner or cross-criterion score. Classification metrics,
portfolio returns, drawdown, and fees answer different questions. Better
classification performance does not necessarily produce better simulated
portfolio performance.

## Delivery rules

API and dashboard consumers must:

- read the persisted, checksum-validated comparison pack without retraining,
  inference, backtest execution, or metric recomputation;
- retain the exact term `shared project holdout` and the academic disclaimer;
- preserve undefined metrics as JSON `null` and display them as unavailable,
  never as zero, `NaN`, or infinity;
- keep validation, shared project holdout, and portfolio rankings separate;
- expose no absolute producer paths or executable model artifacts; and
- reject incomplete, failed, checksum-invalid, path-unsafe, or identity-
  inconsistent packs.

The detailed artifact schema and source-evidence rules remain authoritative in
[Unified Price-Only Model Comparison](UNIFIED_PRICE_MODEL_COMPARISON.md) and
[Architecture Contracts](ARCHITECTURE_CONTRACTS.md).
