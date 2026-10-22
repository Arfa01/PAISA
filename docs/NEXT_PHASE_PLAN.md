# PAISA Next Phase Plan

This plan extends the completed price-only OGDC vertical slice without changing
its leakage, artifact, policy, portfolio, API, or presentation contracts. All
outputs remain academic simulations and are not financial advice.

## Prioritized research sequence

### 1. Financial-news ingestion

Build a reproducible collector for dated Pakistan-market financial news. Retain
the source URL, publisher, publication and retrieval timestamps, headline,
body or permitted extract, language, content hash, and provider response status.
Keep network/provider checks outside the default offline tests and never commit
bulk article data.

Acceptance gate: the same bounded date request produces a deduplicated,
schema-validated manifest with explicit failures and immutable raw evidence.

### 2. Company and entity linking

Create a versioned alias registry for PSX tickers, legal company names,
subsidiaries, sectors, industries, places, and event actors. Preserve the raw
mention, linked identifier, method, confidence, and unresolved state; do not
force ambiguous mentions to a company.

Acceptance gate: reviewed fixtures cover OGDC aliases, ambiguous mentions,
unresolved entities, and effective-date changes without future leakage.

### 3. FinBERT sentiment

Apply a pinned FinBERT model to linked articles. Record the exact model/revision,
tokenization and truncation settings, label mapping, probabilities, runtime
versions, and content hash. Treat FinBERT output as a feature source, not as a
trading signal.

Acceptance gate: deterministic fixtures and a retained manifest prove model,
input, and output identity; unavailable or truncated text is explicit.

### 4. Daily sentiment aggregation

Aggregate article-level scores by ticker and trading date using only information
available by the declared feature cutoff. Define after-close and non-trading-day
handling, source counts, confidence weighting, missing-news indicators, and
lookback windows before running experiments.

Acceptance gate: tests prove no article published after a feature cutoff enters
that feature row and no validation/test statistics are used to fit transforms.

### 5. Price-only versus price-plus-sentiment comparison

Freeze paired configurations that differ only by the added sentiment features.
Use identical target definitions, target-date split boundaries, seeds,
evaluation rows, decision policy, execution rules, initial cash, and fees.
Select configurations on validation evidence and evaluate the untouched test
split once. Report prediction and portfolio metrics separately.

Acceptance gate: one comparison report records the ordered feature sets and
proves matched dates and assumptions; inconclusive or negative results are
reported without promotion language.

### 6. LSTM under the same split contract

Run the existing optional LSTM workflow with the frozen daily dataset contract.
Fit scaling on training data only, build sequences without crossing temporal
boundaries, register the model and scaler together, and emit the same prediction
record used by the backtester. Compare it under the same validation-selection
and untouched-test policy; do not give it a different window or cost model.

Acceptance gate: an actual TensorFlow smoke run and one retained immutable LSTM
experiment pass checksum, prediction-schema, leakage, and backtest validation.

### 7. SHAP explanations

Add model-appropriate SHAP analysis for retained model artifacts. Record the
background/reference data, explainer configuration, class/output explained, and
library versions. Explanations must describe model behavior, not causality or
financial advice.

Acceptance gate: local and aggregate explanations reconcile with the selected
model and ordered features, remain stable on fixtures, and are served through an
additive API schema before being displayed by the dashboard.

### 8. Additional stocks

Expand only after the OGDC news/sentiment comparison is reproducible. Start with
a small documented PSX set, audit date coverage and missing fields per symbol,
and keep per-model portfolios independent. Do not reuse one stock's scaler,
state, or conclusions for another without an explicit pooled-model design.

Acceptance gate: every stock has a data-quality/provenance report, chronological
split evidence, immutable experiment IDs, and stock-specific conclusions.

### 9. Sector and event relationships

Build dated company-sector-event edges from the reviewed entity-linking output.
Keep evidence, direction, confidence, and validity dates on every relationship.
Evaluate whether relationship features improve the same controlled comparison
before adding interactive graph views.

Acceptance gate: relationships are traceable to sources, temporally valid at
the feature date, and tested for ambiguous, conflicting, and expired evidence.

## Cross-cutting rules

- Preserve distinct `feature_date`, `target_date`, `signal_date`, and
  `execution_date` fields.
- Fit every learned transformation on training data only and split time series
  chronologically by target date.
- Preserve price-only baselines and use immutable experiment/model IDs; never
  overwrite `VSL-20261020-OGDC-main-02` or its referenced artifacts.
- Keep models, policies, portfolios, APIs, and dashboard presentation separate.
- Keep bulk data, fitted binaries, secrets, and run outputs out of Git.
- Document missing data, negative findings, and provider failures explicitly.

## Reinforcement learning deferral

Reinforcement learning is not part of the next phase. It remains deferred until
news ingestion, entity linking, FinBERT, daily aggregation, supervised paired
comparisons, the common LSTM contract, SHAP, multi-stock data quality, and
sector/event relationships are stable and reproducible. Before any RL work,
PAISA must also have an agreed reward definition, transaction-cost and action
constraints, walk-forward evaluation, leakage tests, and supervised/benchmark
baselines. RL must not be used to bypass unresolved data or evaluation issues.
