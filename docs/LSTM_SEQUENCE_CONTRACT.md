# PAISA LSTM Sequence Contract

Status: normative contract for the OGDC daily price-only LSTM experiments.

This contract defines how PAISA converts the existing technical-daily rows into
ordered recurrent inputs. It does not change the price target, split policy,
prediction schema, trading policy, or portfolio accounting. All outputs are
academic simulations and are not financial advice.

## Fixed research identity

Both `M-005` and `M-006` use:

- symbol `OGDC` and daily PSX sessions;
- feature set `technical-daily-v1` in its declared immutable order;
- `target_next_day_up`, where class `1` means the next observed session's close
  is greater than the feature session's close;
- a one-trading-session prediction horizon;
- a sequence length of exactly 20 feature observations; and
- chronological membership determined only by `target_date`.

Target, next-session audit, identity, evaluation, and future-information fields
are never model inputs. In particular, sequences exclude every `target_*` and
`next_*` column, `actual_class`, dates, ticker/source metadata, future prices,
future sentiment, and all test metrics.

## Window definition

For a target row whose feature date is session `t`, the input is the ordered
20-row window ending at `t`:

```text
[features(t-19), features(t-18), ..., features(t)]
```

Every member of the window therefore has `feature_date <= t`. The final member
must have exactly the target row's `feature_date`; a row after `t` is an error.
Rows retain chronological order and are never randomly shuffled.

The sequence contains features known by the close of `feature_date`. The label
belongs to the next observed trading session and is not available to feature
construction, preprocessing, inference, policy decisions, or execution.

## Split boundaries and historical context

Training, validation, and test membership follows the target-date split from
the frozen price-only experiment. Historical features immediately before a
validation or test target may be used as context. That context does not move a
target into another split and does not make an earlier label a training input.

Concretely:

- a validation sequence may include earlier training features;
- a test sequence may include earlier training or validation features; and
- no sequence may include a feature observation after its own target row's
  `feature_date`.

Validation labels are used only for validation metrics, early stopping,
learning-rate or checkpoint decisions, and validation-based reporting. Test
features and labels never influence preprocessing, fitting, epoch selection,
checkpoint selection, thresholds, or configuration choice. The test split is
evaluated only after each controlled configuration is frozen.

## Canonical sequence inventory

The retained canonical price input contains 1,239 OGDC sessions and produces
1,219 complete technical-daily target rows. Its target-date splits contain 853
training, 182 validation, and 184 test rows.

With the fixed 20-session sequence length:

| Split | Target rows | Sequences | Handling |
|---|---:|---:|---|
| Train | 853 | 834 | The first 19 training targets lack 20 rows of prior feature context and are reported as dropped. |
| Validation | 182 | 182 | Earlier training features supply context; no validation target is dropped. |
| Test | 184 | 184 | Earlier training/validation features supply context; no test target is dropped. |

The dropped training target dates span the first 19 observed training targets,
from 2021-10-14 through 2021-11-10. The first retained training sequence ends
on feature date 2021-11-10 and predicts target date 2021-11-11. Validation and
test retain the original target windows of 2025-03-24 through 2025-12-15 and
2025-12-16 through 2026-09-15 respectively.

Because every validation and test target is retained, the fair comparison with
`M-001` through `M-004` uses all 182 validation dates and all 184 test dates.
Code must nevertheless calculate and persist the date intersection rather than
assuming equality. A future dataset that drops evaluation targets must restrict
cross-algorithm results to the exact common target-date intersection and report
the excluded dates.

## Preprocessing boundary

The ordered feature list is immutable. Missing-value treatment and scaling are
learned from training rows only, then reused unchanged for validation, test,
saved-model inference, and load/reproduction checks. The fitted imputer and
scaler, their configuration, feature order, fitted-sample scope, and checksums
belong to the immutable experiment artifacts.

Calling `fit`, `fit_transform`, or any equivalent learned operation on
validation/test rows is prohibited. Input frames may contain extra columns or
present the required columns in a different outer order; the preprocessor
selects the immutable stored feature list in its recorded order. It rejects
missing required columns and non-finite transformed feature values.

## Required verification

Automated tests must prove:

- exact ordered window construction and target alignment;
- no future row in any sequence;
- historical context across split boundaries without target leakage;
- train-only imputer and scaler fitting;
- immutable feature order and equal M-005/M-006 experiment rules;
- the canonical 834/182/184 sequence counts;
- explicit dropped-date and common-date reporting;
- no test input to callbacks or tuning decisions; and
- save/load prediction reproduction within documented numeric tolerance.

TensorFlow deterministic operations are requested and Python, NumPy, and
TensorFlow seeds are recorded. Small platform-dependent numeric differences may
remain, especially across CPU, CUDA GPU, and optional Apple Metal execution.
PAISA does not promise bit-identical results across devices.
