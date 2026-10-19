# PAISA OGDC Sprint Definition of Done

The 19–24 October 2026 sprint is complete only when **every** item below is true
and backed by reproducible evidence. Partial demonstrations, uncommitted local
output, or screenshots without retained model/experiment IDs do not qualify.

All portfolio output is an academic simulation and is not financial advice.

## Required acceptance checks

- [ ] **Existing tests continue to pass.** The unchanged v0 behaviors and API
  routes remain covered, all new tests pass, and the final report gives the exact
  command, pass count, failures, warnings, and environment.
- [ ] **OGDC data can be fetched and prepared.** A documented command
  successfully obtains daily `OGDC` data through the primary PSX provider,
  validates it, and records actual source/date coverage. The manual CSV path is
  a continuity fallback, not a substitute for this acceptance check; if an
  upstream outage prevents fetch evidence, the sprint remains blocked. The
  offline sample path also remains usable.
- [ ] **At least two model algorithms can be trained.** Random Forest and
  XGBoost train through the common experiment interface on the leakage-safe
  chronological dataset. The Colab-compatible LSTM notebook completes a clean
  smoke run that trains, evaluates chronologically, and exports a
  contract-conformant manifest/artifact; its execution steps are documented.
- [ ] **Every trained variant has a unique model ID.** Its immutable manifest
  records algorithm, full configuration, ordered features, data provenance,
  date boundaries, target, horizon, seed, metrics, code/dependency versions,
  artifact path, and checksum.
- [ ] **At least three independent portfolio experiments can run.** Each
  configuration has a unique `experiment_id` and `portfolio_id`, fresh initial
  state, separate trades/snapshots, and reproducible policy, sizing, fee, and
  execution settings. A test proves that running one portfolio cannot mutate
  another.
- [ ] **Cash-only and buy-and-hold benchmarks exist.** They use the same initial
  cash, evaluation dates, price availability, valuation convention, and stated
  transaction-cost assumptions as the experiments they accompany.
- [ ] **Portfolio results can be retrieved through FastAPI.** Read-only endpoints
  list registered models/experiments and return a selected experiment's config,
  prediction metrics, portfolio metrics, curve, and trades with tested success,
  validation, and not-found responses. Existing v0 endpoints still work.
- [ ] **A simple dashboard displays price, model metrics, and portfolio curves.**
  The React/Plotly UI reads the API, shows OGDC history, compares model metrics,
  plots each independent experiment with both benchmarks, identifies the active
  configuration/window, and displays a prominent simulation/not-financial-
  advice disclaimer.
- [ ] **The complete workflow has documented commands.** From a fresh clone, the
  runbook covers supported Python/runtime versions, dependency installation,
  OGDC preparation, Random Forest/XGBoost training, LSTM Colab execution,
  experiments, tests, API, dashboard, artifact locations, and the supervisor
  demonstration.
- [ ] **No model or experiment silently overwrites another.** Re-running an
  existing ID fails clearly or requires a new ID; it never replaces a completed
  directory. Tests exercise model and experiment path collisions. Existing v0
  fixed paths are compatibility interfaces, not canonical storage for new runs.
  Every new model is retained under its immutable ID before an explicitly
  documented compatibility alias/copy may be updated, and no alias operation
  may replace an earlier run's canonical artifact.
- [ ] **No future information is used in training or backtesting.** Features use
  information available by `feature_date`; learned transforms fit on training
  only; split membership is based on `target_date` or uses a sufficient purge;
  predictions are produced without labels; signals occur after the feature-date
  close; and trades execute no earlier than the next observed trading session.

## Evidence required for sign-off

The final sprint handoff must include:

1. Git commit and branch identifiers for each merged feature.
2. Exact clean-install and test commands with complete results.
3. The OGDC dataset manifest and actual date coverage.
4. A table of retained `model_id`, `algorithm`, configuration, and artifact path.
5. A table of at least three retained `experiment_id`/`portfolio_id` pairs and
   their model/configuration references.
6. Prediction and portfolio comparison metrics, including both benchmarks.
7. API and dashboard launch commands plus the routes/views used in the demo.
8. Automated-test evidence for leakage boundaries, artifact collisions,
   accounting reconciliation, portfolio independence, and benchmark alignment.
9. Known limitations, assumptions, non-blocking warnings, and remaining risks.

## Sign-off rule

If any required check lacks reproducible evidence, mark the sprint incomplete
and name the owner, blocker, and next action. Do not weaken a contract or label a
simulated result as financial advice to make the demonstration appear complete.
