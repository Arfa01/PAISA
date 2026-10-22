# PAISA Known Limitations

## Status of this document

These limitations apply to the completed demonstration run
`VSL-20261020-OGDC-main-02`. They define the boundary of the current vertical
slice and should be stated during the supervisor demonstration. They are not
hidden implementation details and must not be removed from user-facing
interpretation.

PAISA currently provides an **academic simulation only**. It does not provide
financial advice, place real orders, or demonstrate future investment
performance.

## Data coverage and provenance

### One stock only

The completed vertical slice trains, evaluates, backtests, and presents
`OGDC` only. Although preserved legacy source evidence includes other symbols,
the conclusions in this run have not been tested across those stocks or a
diversified universe. Company-specific behavior may dominate the observed
results.

### Daily data only

The feature, target, signal, execution, and valuation contracts use daily
trading sessions. The system does not model intraday prices, order-book
liquidity, intraday volatility, or within-session execution choices.

### Verified legacy cache, not a fresh live endpoint response

The real run uses a checksum-bound reconstruction of a legacy PSX-DPS cache.
The migration verified row accounting, required fields, ticker/date uniqueness,
positive prices, non-negative volume, and an exact rebuild match with the
legacy processed cache. However, the attempted live endpoint returned HTTP 404,
so this run is not evidence that the current live acquisition route works.

The preserved raw files contain no provider signature, retained HTTP response
headers, request URL, or independently timestamped response hash. Provider
attribution rests on preserved legacy metadata, provider-specific filenames,
and reproduction through PAISA's cleaning logic. SHA-256 checksums detect
changes within the retained evidence chain, but they are not an external
digital signature or third-party attestation.

Evidence is under:

- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/pipeline_metadata.json`
- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/migration_manifest.json`
- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/input_snapshot/legacy_pipeline_metadata.json`
- `artifacts/vertical_slice/VSL-20261020-OGDC-main-02/manifest.json`

### High and low prices are absent

The captured PSX-DPS rows do not provide verified `high` or `low` values. The
current workflow uses verified `open`, `close`, and `volume` fields and does not
invent missing OHLC data. Features or execution models that depend on intraday
ranges cannot be evaluated from this bundle.

### Corporate-action adjustment is unknown

The source does not establish whether historical prices are fully adjusted for
splits, bonus issues, rights issues, or other corporate actions. An unmodeled
price discontinuity can affect features, labels, returns, volatility, and
drawdown.

### Dividends are not modeled

Neither the model portfolios nor the buy-and-hold benchmark include dividend
cash flows or reinvestment. Therefore, the reported buy-and-hold series is not a
total-shareholder-return benchmark, and comparisons may differ from an adjusted
total-return analysis.

## Prediction and execution scope

### Close-to-close prediction versus next-open execution

The classifier predicts whether the next session's close is above the current
feature-date close. The signal is available only after that feature-date close,
so a simulated trade executes no earlier than the next observed session's
open. The overnight close-to-open move is inside the classification interval
but is not a realizable return for a newly opened position.

This timing difference means prediction accuracy and simulated profitability
measure different things. It must not be described as if a correct
close-to-close prediction guarantees a profitable next-open trade.

### One historical split and fixed assumptions

The run uses one chronological 70/15/15 split, one feature set, fixed model
configurations, fixed 0.60/0.40 probability thresholds, a 50% maximum
allocation, whole shares, 10 basis-point proportional fees, and zero modeled
slippage. The results do not show robustness across rolling windows, alternate
regimes, policy parameters, allocation rules, transaction-cost schedules, or
non-zero slippage.

The controlled buy-and-hold comparator also uses the 50% allocation cap. Its
6.24789% return is therefore not the return of a conventional 100%-invested
buy-and-hold position.

## Research features not yet implemented

### No sentiment features yet

The current models use price-, volume-, lag-, moving-average-, volatility-,
RSI-, and MACD-derived daily features. A separate news-ingestion foundation now
captures and links permitted records, but FinBERT scoring, daily sentiment
aggregation, and price-plus-sentiment comparison have not been integrated into
the retained price/model run.

### News coverage is provider-limited

RSS history and fields depend on each configured publisher. The foundation is
an explicit command, not a scheduler, and it does not crawl linked pages for
missing bodies. Manual CSV data is imported evidence, not live acquisition.
The first-stage entity linker recognizes controlled OGDC aliases only; it is
not general NER and does not infer sectors or events. RSS acquisition is
streamed with compressed/decoded byte, XML depth/element, entry, field, and
total-time limits; DTD/entities and non-public destinations are rejected by
default. These controls reduce resource-exhaustion and SSRF risk but do not
establish a publisher's reliability, completeness, licensing, or long-term
availability. The explicit private-network source opt-in is for reviewed
deployments only.

### No knowledge graph yet

Company/entity linking and company-sector-event relationships are not yet
represented in a knowledge graph. The system cannot currently propagate or
explain relationships between an event, related entities, sectors, and model
evidence.

### Explainability is limited

The artifacts retain model configuration, feature order, metrics, and
provenance, but the completed comparison does not include SHAP explanations.
Current evidence should not be presented as a causal explanation of OGDC price
movement.

## Operational limitations

### No production database yet

The vertical slice uses immutable filesystem artifacts and read-only artifact
APIs. News ingestion adds a generated local SQLite evidence store with a
versioned schema, but this is not a production database service. PAISA does not
provide production migrations across engines, access control,
multi-user tenancy, backup/restore service, scheduled ingestion, or production
observability. Local artifact availability and integrity are prerequisites for
the API and dashboard.

News-run marker replacement is atomic and has a sealed retry journal, but the
local prototype does not `fsync` files and directories for power-loss
durability. Public markers remain pending until the database commits; a final
completion marker is published last, and recovery fails closed if the sealed
artifact set changed. A permanent storage failure can retain a pending sealed
run for an explicit retry. This is not a production transaction coordinator.

### No broker integration

There is no brokerage connection, order-management system, exchange gateway,
real-time market-data feed, or reconciliation with actual fills. The
backtesting engine creates simulated trades only. It does not account for live
latency, partial fills, rejected orders, dynamic spreads, market impact, or
broker-specific charges.

### Not production deployment evidence

The FastAPI service and React dashboard demonstrate local artifact delivery and
presentation. The run does not establish production availability, security,
scalability, disaster recovery, or regulatory compliance.

## Interpretation boundary

The reported findings apply only to one stock, the recorded period, the
available daily fields, the technical feature set, the fitted models, the
chronological split, and the stated decision, portfolio, fee, and execution
assumptions. No result should be generalized to other securities or future
periods without new, leakage-safe comparative evidence.

The dashboard, API, signals, trades, equity curves, returns, Sharpe ratios, and
drawdowns are research outputs from an academic simulation. They are **not
financial advice** and must not be used as a claim of deployable trading
performance.
