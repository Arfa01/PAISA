# PAISA Current Implementation Status

Status date: 22 October 2026

Feature base: `origin/main` at `578eeec` after PR #12

Retained demonstration run: `VSL-20261020-OGDC-main-02`

PAISA is an academic research prototype. Its predictions, signals, trades,
portfolio curves, and simulated returns are not financial advice.

## Implemented and working

| Area | Current implementation and evidence |
|---|---|
| Stock-data foundation | The legacy v0 path implements PSX acquisition through `psx-dps`, with manual-CSV and offline sample alternatives; cleans the canonical daily schema; creates backward-looking technical features and `target_next_day_up`; trains the compatibility Random Forest; and preserves the original FastAPI routes. The acquisition path is implemented, but the live endpoint returned HTTP 404 during the retained run, so fresh live ingestion is not currently demonstrated. The optional `pypsx` provider remains separate from the default installation. |
| Leakage-safe experiment data | The versioned path distinguishes `feature_date` and next-session `target_date`, splits chronologically by target date, fits learned transforms on training data only, records feature order and seeds, and preserves immutable input and model evidence. |
| Versioned model comparison | `M-001`, `M-002`, and `M-003` are Random Forest variants; `M-004` is XGBoost. All four use the same daily OGDC feature set, target, split dates, and untouched test observations. Each has a unique registered model identity and checksum-protected artifacts. |
| Decision and simulation | Predictions pass through a separate threshold policy into separate long-only portfolio ledgers. The retained run uses buy at `probability_up >= 0.60`, sell at `probability_up <= 0.40`, a 50% maximum allocation, whole shares, 10-basis-point execution fees, next-session-open execution, and close valuation. |
| Independent comparisons | Each model has its own experiment, portfolio, trades, cash, holdings, and snapshots. Cash-only and 50%-allocated buy-and-hold benchmarks use the common test window and cost assumptions. |
| Immutable vertical slice | `scripts/run_ogdc_vertical_slice.py` joins acquisition, preparation, training, predictions, policies, backtests, benchmarks, evidence validation, and reports without overwriting an existing run ID. |
| News-ingestion foundation | The isolated `paisa.news` package captures configured RSS/Atom responses or permitted manual CSV imports; preserves raw bytes and acquisition metadata; normalizes schema `1.0`; keeps publication, fetch, and defensible availability timestamps distinct; applies layered deterministic deduplication; links boundary-safe controlled OGDC aliases; persists canonical and per-ingestion evidence in local SQLite; and publishes immutable quality reports. The ingestion command does not run sentiment or alter the price/model stack. |
| Article sentiment foundation | The isolated sentiment stage reads canonical normalized articles from a completed ingestion run, applies an exact pinned `ProsusAI/finbert` model/tokenizer revision, maps the upstream class indices explicitly, records bounded text/token policy and runtime identity, persists immutable article probabilities/errors in SQLite, publishes checksum-inventoried `SENT-...` artifacts, and exposes a leakage-safe decision-cutoff filter. It does not aggregate by day or alter stock models, policies, portfolios, API, or dashboard. |
| Results API | The additive `/api/v1` FastAPI surface reads completed, checksum-valid artifacts. It lists runs and exposes run metadata, source provenance, historical prices, model metrics, portfolio metrics, curves, experiments, trades, snapshots, and benchmarks. It does not retrain models or recalculate accounting in a GET request. |
| Dashboard | The React/TypeScript/Plotly application reads only the API and shows run overview, immutable OGDC history, model comparison, portfolio comparison, experiment detail, trades, snapshots, provenance, limitations, and the academic-simulation disclaimer. It has no model, policy, or accounting business logic. |
| Verification | The full Python suite, all `57` frontend tests, frontend lint, frontend production build, `pip check`, bytecode compilation, and `git diff --check` are required branch gates. The ignored retained-artifact checkout can run one additional local integration test that a clean Git clone skips; the retained sentiment run below supplies separate real-model execution evidence. |

## Retained real-run evidence

The completed `PASS` run `VSL-20261020-OGDC-main-02` references training run
`TRN-20261020-191815-624662-nogit`. It contains 1,239 daily OGDC price rows
covering 2021-09-16 through 2026-09-15. The usable feature range is
2021-10-13 through 2026-09-14, with `853` train, `182` validation, and `184`
test observations. The test feature window is 2025-12-15 through 2026-09-14;
its corresponding target window is 2025-12-16 through 2026-09-15.

The run is real-data evidence from a verified reconstruction of the legacy PSX
cache, not a fresh live download. The attempted live PSX-DPS request returned
HTTP 404. The fallback retains the original metadata, rebuilt raw-capture
evidence, migration manifest, and SHA-256 inventory. This establishes internal
integrity and reproducibility, but not independent provider authentication.

The run reports separate conclusions:

- `M-001` is the validation-selected prediction model.
- `M-002` has the highest model-portfolio ending value: PKR 105,987.04, a
  5.98704% simulated return.
- `M-003` has the lowest model-portfolio drawdown: 1.18467%, but its simulated
  return is -0.83393%.
- `M-004` makes 34 trades and pays PKR 1,751.06 in fees; it returns 0.19005%.
- The 50%-allocated buy-and-hold benchmark ends at PKR 106,247.89, a 6.24789%
  return with a 12.79045% maximum drawdown.
- No model portfolio beats buy-and-hold in total return in this run.

### Accepted news-ingestion execution

The accepted network-acquired RSS run `NEWS-20261022-131242-5ddffd2d`
reached run/quality status `PASS`/`PASS` against one configured Dawn Business
feed. It recorded `30` raw and `30` normalized occurrences, `29` duplicates,
`1` OGDC-linked occurrence, `0` missing publication times, and a publication
range from `2026-10-19T18:56:40Z` through `2026-10-22T13:07:31Z`. The source
response was acquired at `2026-10-22T13:12:43.088883Z`, after the run start at
`2026-10-22T13:12:42.772512Z`, and its declared and observed SHA-256 values
match. Evidence is under
`artifacts/news_ingestion/NEWS-20261022-131242-5ddffd2d/` and is ignored by
Git.

The `29` duplicates match canonical evidence retained from earlier acquisitions;
one newly observed feed article became new canonical evidence. The accepted run
still retains every raw response and normalized occurrence. The earlier
`NEWS-20261022-115232-03d0fc4c` run remains immutable but is superseded because
its acquisition timestamp was taken too early. The intermediate corrected run
`NEWS-20261022-120012-31d811b1` is also retained but superseded because its
manifest reported an obsolete database schema version. The later
`NEWS-20261022-120133-43a51981` run remains immutable but is superseded by the
storage-integrity hardening, while `NEWS-20261022-130023-470cb7a7` predates the
final time-evidence, secret-redaction, and recoverable-finalization hardening.
The accepted manifest reports finalization state `COMMITTED` and SQLite schema
version `3`. Its ingestion artifact tree remains historical: it predates the
schema-4 sealing fingerprint, the schema-5 sentiment tables, the current
schema-6 certification contract, and the
completion/deduplication-context files, and must not be described as passing
the current news verifier. Before sentiment execution the working
`data/news/paisa_news.sqlite3` also used schema version `3`; it is now schema
version `5` for the retained reference execution. Current code migrates a
verified database onward to schema `6`, which adds hash-bound model files,
post-truncation evidence, explicit run associations, and certified
finalization. An ignored, uncommitted pre-migration copy at
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3` was verified as
schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check. The accepted run demonstrates only
one publisher's current feed snapshot; it does not establish historical
completeness, stable future availability, licensing beyond the publisher's
feed terms, sentiment quality, or predictive/model value.

### FinBERT article-sentiment implementation

The implemented sentiment configuration pins both model and tokenizer to
`ProsusAI/finbert` revision
`0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`. The recorded model/weight license
is CC BY-NC-SA 4.0, so this PAISA configuration is limited to non-commercial
academic use. The model's original index order is `0=positive`, `1=negative`,
and `2=neutral`; PAISA maps it explicitly into the stable probability order
`negative`, `neutral`, `positive`.

The accepted source run represents 30 unique canonical articles, one linked to
OGDC, and all carry `en-us` language evidence. Twenty-nine canonical articles
were first available in an earlier retained run; one was first available in the
accepted run. Sentiment inherits each canonical article's original
`published_at`, `available_at`, and `fetched_at`; the inference timestamp does
not replace those values.

The retained real execution completed as follows. These are generated local
artifacts and database records, not committed repository files.

| Real-run result | Measured value |
|---|---|
| Sentiment run / source run | `SENT-20261022-172211-real0002` / `NEWS-20261022-131242-5ddffd2d` |
| Status | `PASS` |
| Requested / scored / unique canonical | `30 / 30 / 30` |
| Skipped / failed / any articles failed | `0 / 0 / false` |
| OGDC-linked articles scored | `1` |
| Negative / neutral / positive counts | `11 / 9 / 10` |
| Truncated inputs / title-only inputs | `17 / 0` |
| Language evidence | `en-us=30` |
| Availability range | `2026-10-22T11:52:32Z` to `2026-10-22T13:12:43.088883Z` |
| Device / duration | `cpu / 7.116756208240986 seconds` |
| Model and tokenizer revision | `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` |
| Runtime | Python `3.13.7`; PyTorch `2.10.0`; Transformers `5.20.2`; Tokenizers `0.23.3`; Hugging Face Hub `1.32.0`; Safetensors `0.8.0` |
| Database schema for retained reference run / current code | `5 / 6` |
| Artifact directory | `artifacts/sentiment_inference/SENT-20261022-172211-real0002` |

Sentiment outputs were generated, but classification accuracy was not evaluated in this run.

The retained sentiment evidence covers October 19--22, 2026, but the retained
OGDC price evidence ends on September 15, 2026. These sources have **zero
overlapping dates**. The current evidence therefore cannot form even one
price-plus-sentiment observation and cannot support a combined experiment or
portfolio conclusion. That work requires an overlapping historical news
corpus.

The narrow legacy compatibility path validates the exact recognized schema-3
signature and affected rows, transactionally restores the later constraints,
and applies migrations 4 and 5. It does not recreate missing artifacts or
retroactively validate the retained news pack. The working database is now
schema version `5` (and is upgraded to `6` when opened by current code), while
the ignored pre-migration copy above preserves the
verified schema-3 state. Even after a successful execution, this single short
RSS snapshot will not be sufficient for stock-model integration,
classifier-accuracy claims, or price-plus-sentiment comparison.

## Current research prototypes

- The four price-only classifiers are controlled comparative experiments, not
  evidence of performance on other stocks, periods, horizons, or regimes.
- The policy thresholds and portfolio simulator are research assumptions, not a
  recommendation engine or execution system.
- The dashboard and artifact API are demonstration and analysis surfaces, not a
  production deployment or current market-data service.
- The optional LSTM code and Colab notebook implement the common artifact and
  chronological-split contract. TensorFlow is intentionally outside the default
  runtime, and LSTM is not one of the four models in the retained demonstration
  run.
- Stored Random Forest feature importance is preliminary explainability. SHAP
  explanations have not been implemented.
- News entity linking is a deterministic first stage for controlled OGDC
  aliases only. It is not general NER or event reasoning.
- FinBERT output is article-level classifier evidence, not a stock prediction,
  causal explanation, trading signal, or accuracy result. No human-labelled
  evaluation set, daily aggregation, or stock-feature join is included.
- The verified-cache migration is a provenance-preserving continuity mechanism;
  it does not turn the retained data into a fresh or cryptographically signed
  provider response.

## Progress against the approved proposal

| Approved objective | Status |
|---|---|
| Collect, preprocess, and structure historical stock data | Partly complete. The reusable pipeline supports multiple symbols, while the retained vertical slice and its conclusions are OGDC-only and daily-only. |
| Continuously update financial news, economic reports, and global events | Partly complete. Reproducible explicit RSS/Atom and imported-CSV runs now preserve news evidence, timing, quality, and failures. Scheduling, broad economic/global source coverage, and production operations are not implemented. |
| Train and compare multiple prediction models | Implemented for three Random Forest configurations and one XGBoost configuration on one frozen OGDC split. The optional LSTM workflow is not promoted into this run. |
| Compare models with and without sentiment | Not implemented. Article-level inference is implemented, but daily aggregation and the controlled stock-feature comparison remain. The current stock comparison is price-only. |
| Deploy the best-performing model as the prediction engine | Not complete. The v1 API deliberately serves retained results and does not load `M-001` for live inference; the legacy `/predict/{ticker}` route remains the v0 baseline compatibility path. |
| Add event-driven company/sector reasoning | Partly complete only at the ingestion boundary: reviewed OGDC aliases are linked deterministically with evidence. General entities, sectors, events, graphs, and propagation are not implemented. |
| Deliver predictions, visualizations, explanations, and current feeds | Partly complete. Historical results and provenance are visible in FastAPI and React; sentiment, event explanations, SHAP, and current feeds remain. |

## Remaining work

The next research sequence is broader reviewed entity/source coverage, review
of the retained FinBERT execution and audit sample, daily sentiment aggregation,
price-only versus price-plus-sentiment comparison, LSTM under the same split contract, SHAP,
additional stocks, and sector/event relationships. Production storage,
authentication, scheduled ingestion/retraining, cloud deployment, broker
integration, and live trading are also absent. Reinforcement learning is
explicitly deferred until the data, sentiment, supervised-model, evaluation,
and explanation components are stable.

## Documentation inconsistencies to keep visible

- `PROJECT_CONTEXT.md` contains an older baseline-only status and an older
  multi-symbol date range. It is useful proposal history, but it is not the
  current implementation inventory or the evidence source for this run.
- `docs/DEFINITION_OF_DONE.md` still has unchecked acceptance boxes and requires
  a successful live provider fetch plus an executed TensorFlow LSTM smoke run.
  The retained run instead records a failed live request, a verified legacy
  cache, and no LSTM model. Those two strict checks must not be represented as
  complete.
- The proposal's “best-performing model” is not a single result here: `M-001`
  wins the validation-based prediction selection, while `M-002` produces the
  best model-portfolio simulation. The system correctly preserves that
  distinction.
