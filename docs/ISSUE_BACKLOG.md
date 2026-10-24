# PAISA Issue Backlog

Status date: 24 October 2026

This file contains paste-ready GitHub issue drafts for the remaining PAISA
roadmap. Keep the stable ID in the issue title even after GitHub assigns an
issue number. Suggested labels may be created or mapped to equivalent repository
labels.

All model predictions, signals, trades, portfolios, and returns discussed by
these issues are academic simulations and are not financial advice.

## Shared completion rules

Every engineering issue below inherits these requirements unless its body says
it is a research/operations issue:

- Preserve existing PAISA v0 commands, schemas, and API behavior.
- Use chronological `target_date` splits and train-only learned transforms.
- Never overwrite an artifact or reuse a model, run, experiment, or portfolio
  ID. New outputs use fresh collision-refusing IDs.
- Keep model training, predictions, policies, portfolios, APIs, and dashboard
  presentation separate.
- Commit only small approved fixtures. Do not commit raw corpora, private data,
  databases, model weights, generated run directories, ZIPs, secrets, or local
  environment files.
- Keep the default suite offline. Live providers and real model downloads are
  explicit operator steps, never CI prerequisites.
- Run the affected backend and frontend tests, `pip check`, `compileall`,
  documentation-contract tests, and `git diff --check`.
- Update documentation for every public command, schema, configuration,
  artifact layout, dependency, or user-visible limitation changed.
- Include the academic-simulation/not-financial-advice statement in every new
  user-facing result surface.

The authoritative dependency and gate definitions are in
[`REMAINING_ROADMAP.md`](REMAINING_ROADMAP.md).

---

## PAISA-RM-001 — Add fixture-backed read-only LSTM artifact integration

**Suggested title:** `[PAISA-RM-001] Add fixture-backed read-only LSTM artifact integration`

**Suggested labels:** `priority:must`, `type:feature`, `area:lstm`, `area:api`, `offline-ci`

**Depends on:** None. This is the first unblocked engineering issue.

**Gate:** `G0-INTEGRITY`. Completion establishes `G1-LSTM-READ`.

### Scope

Add a separate, read-only repository boundary and additive v1 API for completed
LSTM experiment summaries. Commit a minimal synthetic or sanitized,
checksum-valid fixture that preserves the actual per-file contracts:
`run_manifest.json` schema `1.0` and authoritative `final_summary.json` schema
`1.1` with final `PASS` status. Include M-005/M-006 metadata, metrics, and
portfolio summaries. The reader must verify identifiers, safe relative paths,
completion state, exact inventory, nested checksums, finite values, and the
academic disclaimer before returning data.

### Out of scope

- TensorFlow installation, training, checkpoint loading, or metric recomputation.
- Copying the local `LSTM-20261024-OGDC-price-only-02` directory or weights into Git.
- Changing baseline vertical-slice endpoints or declaring a six-model champion.

### Acceptance checklist

- [ ] Clean-clone tests use only the committed fixture and require no TensorFlow.
- [ ] The API lists and returns completed verified LSTM runs read-only.
- [ ] Missing, incomplete, unsafe-path, checksum-mismatched, duplicate-inventory,
      and non-finite fixtures fail with clear 4xx responses.
- [ ] The response keeps validation selection, post-freeze test evidence, and
      portfolio results distinct and identifies the shared exposed holdout.
- [ ] No model binary is opened or deserialized.
- [ ] API schema, environment configuration, fixture provenance, and restore
      behavior are documented.
- [ ] Focused repository/API tests and the full offline backend suite pass.

---

## PAISA-RM-002 — Show verified LSTM results in the dashboard

**Suggested title:** `[PAISA-RM-002] Add a read-only LSTM results dashboard view`

**Suggested labels:** `priority:must`, `type:feature`, `area:lstm`, `area:frontend`

**Depends on:** `PAISA-RM-001`

**Gate:** `G1-LSTM-READ`

### Scope

Add typed client schemas and a dashboard view for a selected verified LSTM run.
Show variant identity, validation-only preference, post-freeze test metrics,
portfolio outcomes, source-run references, and limitations exactly as returned
by the API.

### Out of scope

- Recomputing metrics, selecting a model, loading weights, or joining baseline
  and LSTM data in the browser.
- Presenting test-leading M-006 as the validation-selected variant.

### Acceptance checklist

- [ ] Zod schemas reject malformed, non-finite, or disclaimer-free responses.
- [ ] Loading, empty, not-configured, integrity-error, and success states are tested.
- [ ] The view labels M-005 validation preference separately from M-006's
      post-freeze shared-holdout result.
- [ ] Portfolio values are described as simulated and not financial advice.
- [ ] Frontend tests, lint, and production build pass.
- [ ] Dashboard and API startup documentation identifies the optional LSTM root.

---

## PAISA-RM-003 — Build an immutable unified six-model comparison artifact

**Suggested title:** `[PAISA-RM-003] Build an immutable M-001 through M-006 comparison artifact`

**Suggested labels:** `priority:must`, `type:feature`, `area:experiments`, `area:lstm`

**Depends on:** `PAISA-RM-001`

**Gate:** `G1-LSTM-READ`

### Scope

Create a deterministic derived-artifact builder that consumes verified baseline
and LSTM integrity roots, proves the exact common target-date intersection, and
publishes a new collision-refusing comparison directory. Persist source root
digests, dropped-date evidence, ordered metrics, validation selections,
post-freeze test rankings, portfolio rankings, assumptions, and limitations.

### Out of scope

- Retraining, backtesting again, copying model binaries, or mutating either source run.
- Combining validation, test, and simulated-return evidence into one champion score.

### Acceptance checklist

- [ ] Source roots are revalidated before and after comparison generation.
- [ ] Fixture output proves identical dates or records every dropped date by model.
- [ ] The artifact states that the 184-date test window was previously exposed.
- [ ] Validation preference, test ranking, and portfolio ranking remain separate.
- [ ] Any mismatch in target, split, policy, fee, execution, ticker, or date
      assumptions fails closed.
- [ ] Existing comparison IDs and destinations are never replaced.
- [ ] Contract, command, layout, and verification documentation are added.
- [ ] Offline builder, corruption, collision, and leakage-regression tests pass.

---

## PAISA-RM-004 — Expose unified comparisons through a read-only API

**Suggested title:** `[PAISA-RM-004] Expose verified unified model comparisons through v1 API`

**Suggested labels:** `priority:must`, `type:feature`, `area:api`, `area:experiments`

**Depends on:** `PAISA-RM-003`

**Gate:** `G0-INTEGRITY`

### Scope

Add typed, additive endpoints that list and return completed unified comparison
artifacts. Serve persisted source references, common-date evidence, separate
rankings, and limitations without recalculation.

### Out of scope

- Training, model deserialization, backtesting, or accepting client-selected files.
- Breaking or repurposing existing vertical-slice routes.

### Acceptance checklist

- [ ] Only checksum-valid, completed comparison artifacts are listed.
- [ ] Identifier traversal, symlinks, unexpected files, absolute paths, and
      changed source roots are rejected.
- [ ] Response models prohibit non-finite metrics and require the disclaimer.
- [ ] Pagination and not-found/integrity-error behavior are documented and tested.
- [ ] Existing API compatibility tests and the full offline backend suite pass.

---

## PAISA-RM-005 — Add unified comparison dashboard and export

**Suggested title:** `[PAISA-RM-005] Present the verified six-model comparison in the dashboard`

**Suggested labels:** `priority:must`, `type:feature`, `area:frontend`, `area:experiments`

**Depends on:** `PAISA-RM-002`, `PAISA-RM-004`

**Gate:** `G0-INTEGRITY`

### Scope

Add a comparison selection/view that renders all six models on persisted common
dates. Display validation selection, post-freeze test metrics, and simulated
portfolio results as separate tables or panels, with a download of the verified
comparison payload.

### Out of scope

- Client-side joins, metric calculation, model ranking logic, or advice language.
- Claiming general superiority from one stock, period, or exposed holdout.

### Acceptance checklist

- [ ] Typed schemas reconcile six unique variants and their registered IDs.
- [ ] The UI names M-001 and M-005 validation selections independently.
- [ ] Shared-holdout and portfolio-assumption warnings are visible without interaction.
- [ ] Downloaded data comes from the verified API response, not reconstructed rows.
- [ ] Loading, error, malformed-data, and responsive rendering tests pass.
- [ ] Frontend tests, lint, build, and relevant documentation pass.

---

## PAISA-RM-006 — Approve a historical-news source and permission contract

**Suggested title:** `[PAISA-RM-006] Record a permission decision for overlapping historical OGDC news`

**Suggested labels:** `priority:must`, `type:research`, `area:news`, `gate:permission`, `needs-human-review`

**Depends on:** None; this human-governance task may run beside `PAISA-RM-001`.

**Gate:** Completion must establish `G2-NEWS-PERMISSION` before acquisition.

### Scope

Evaluate a source capable of covering the frozen OGDC price period. Record the
exact provider, fields, query scope, time semantics, academic-use basis,
attribution, automation rights, retention, local backup, publication/export,
redistribution, and revocation/re-review conditions in a checksum-bound decision.

### Out of scope

- Fetching or importing records before permission is `PERMITTED`.
- Treating metadata access as permission to store publisher bodies.
- Changing current evidence timestamps or allowlisting a future unspecified run.

### Acceptance checklist

- [ ] A human-reviewed source assessment cites current authoritative terms.
- [ ] The machine-readable decision binds the exact assessment checksum and controls.
- [ ] Required date range and defensible historical `available_at` evidence are stated.
- [ ] Prohibited fields and uses are explicit.
- [ ] `REVIEW_REQUIRED` or uncertainty leaves `PAISA-RM-007` blocked.
- [ ] No corpus, credentials, or generated artifacts are committed.

---

## PAISA-RM-007 — Acquire or import approved overlapping historical evidence

**Suggested title:** `[PAISA-RM-007] Create an immutable approved historical OGDC news run`

**Suggested labels:** `priority:must`, `type:data-operation`, `area:news`, `blocked:permission`

**Depends on:** `PAISA-RM-006`

**Gate:** `G2-NEWS-PERMISSION`

### Scope

Using the approved controls, execute one bounded acquisition or licensed import
under a fresh `NEWS-...` or evidence run ID. Preserve raw/import evidence,
request/attempt context, publication, source-observation, fetch/import, and
defensible availability times; normalize and link OGDC without forced matches.

### Out of scope

- Backdating import/fetch time, inventing publication times, bypassing network
  boundaries, or following publisher pages outside the permission decision.
- Committing bulk data or replacing any retained run.

### Acceptance checklist

- [ ] Run and quality status, provider errors, rejected rows, and completeness are explicit.
- [ ] Every retained row has provenance and the current availability policy is enforced.
- [ ] Exact permission/config references and checksums are sealed into the run.
- [ ] Verification replays against source/database evidence at the strongest available basis.
- [ ] Artifacts remain private/local as required and use a new immutable ID.
- [ ] Operator commands and negative outcomes are recorded without calling the run complete if it is partial.

---

## PAISA-RM-008 — Audit current-schema historical news/price overlap

**Suggested title:** `[PAISA-RM-008] Run the current historical news overlap gate`

**Suggested labels:** `priority:must`, `type:data-operation`, `area:news`, `gate:coverage`

**Depends on:** `PAISA-RM-007`

**Gate:** A passing result establishes `G3-NEWS-OVERLAP`; any other result blocks downstream work.

### Scope

Run a fresh current-contract, source-evidence-bound coverage audit against the
immutable OGDC stock/split evidence. Use a new `COVERAGE-...` or `ECOV-...` ID
and the configured thresholds. Report occurrence/canonical populations,
OGDC linkage, split/month coverage, cutoff eligibility, completeness,
permission, database/source provenance, and missing intervals.

### Out of scope

- Reusing or modifying historical v2 coverage packs.
- Lowering thresholds after observing results or calling nonzero coverage statistically sufficient.

### Acceptance checklist

- [ ] Verification reopens the exact source and stock inputs and checks their roots.
- [ ] Train, validation, and test denominators use exact persisted target-date arrays.
- [ ] Unsupported populations remain `N/A`, not zero.
- [ ] The result is `READY_FOR_AGGREGATION` only if every configured prerequisite passes.
- [ ] A negative result is retained and documented as a valid blocker.
- [ ] `PAISA-RM-009` through `PAISA-RM-013` remain blocked unless `G3` passes.

---

## PAISA-RM-009 — Score the approved overlap corpus with pinned FinBERT

**Suggested title:** `[PAISA-RM-009] Run pinned FinBERT on the approved overlapping corpus`

**Suggested labels:** `priority:must`, `type:data-operation`, `area:sentiment`, `blocked:coverage`

**Depends on:** `PAISA-RM-008`

**Gate:** `G3-NEWS-OVERLAP`

### Scope

Run the existing exact FinBERT model/tokenizer revision and text policy over the
approved completed ingestion population under a new immutable `SENT-...` ID.
Review the audit sample, inherited timing, truncation, failures, language
evidence, and model-file hashes before accepting the run.

### Out of scope

- Committing weights or outputs, changing the model revision mid-run, daily
  aggregation, trading signals, or classification-accuracy claims without labels.

### Acceptance checklist

- [ ] Model, tokenizer, file hashes, license boundary, runtime, and device are recorded.
- [ ] Requested, scored, skipped, failed, linked, and truncated counts reconcile.
- [ ] Article availability is inherited and never replaced by inference time.
- [ ] Output/database identities are collision-refusing and verified.
- [ ] Human audit findings and limitations are documented.
- [ ] Real execution is separate from deterministic fixture-only CI tests.

---

## PAISA-RM-010 — Define the daily sentiment contract and fixture oracle

**Suggested title:** `[PAISA-RM-010] Define leakage-safe daily sentiment aggregation`

**Suggested labels:** `priority:must`, `type:design`, `area:sentiment`, `gate:coverage`

**Depends on:** `PAISA-RM-009`

**Gate:** `G3-NEWS-OVERLAP`

### Scope

Specify the immutable input/output schemas and a small hand-calculated fixture
oracle for ticker/session aggregation. Define decision cutoff, timezone,
after-close mapping, weekends/holidays, date-only evidence, source/unique-article
counts, duplicate handling, confidence or neutral weighting, missing-news rows,
lookbacks, and train-only learned transforms.

### Out of scope

- Implementing a production scheduler or choosing rules from test outcomes.
- Treating missing news as neutral without an explicit separate indicator.

### Acceptance checklist

- [ ] The contract distinguishes article date, `available_at`, feature date, and decision cutoff.
- [ ] Fixture cases cover before/at/after cutoff, non-trading days, duplicates,
      missing news, unresolved entities, and split boundaries.
- [ ] No validation/test statistic is needed to create training rows.
- [ ] Feature order, null/zero semantics, and aggregation-version identity are explicit.
- [ ] Documentation-contract and fixture-oracle tests fail on future leakage.

---

## PAISA-RM-011 — Implement immutable daily sentiment aggregation

**Suggested title:** `[PAISA-RM-011] Implement and verify daily sentiment feature artifacts`

**Suggested labels:** `priority:must`, `type:feature`, `area:sentiment`, `area:artifacts`

**Depends on:** `PAISA-RM-010`

**Gate:** `G3-NEWS-OVERLAP`. Completion establishes `G4-DAILY-SENTIMENT`.

### Scope

Implement the pure aggregator, CLI, collision-refusing writer, manifest,
inventory, verification command, and quality report from the frozen contract.
Inputs must be verified article-sentiment and stock-calendar evidence; outputs
must retain the contributing article identities and timing evidence needed for
replay.

### Out of scope

- Model training, policy signals, portfolio simulation, API/UI work, or live-provider calls.

### Acceptance checklist

- [ ] Output matches the hand-calculated oracle exactly and is deterministic.
- [ ] Rows exclude every article unavailable by the declared cutoff.
- [ ] Missing-news and zero-score semantics remain distinct.
- [ ] Input root digests are checked before and after generation.
- [ ] Existing IDs, linked/special files, unexpected files, and checksum changes fail closed.
- [ ] CLI/help, artifact tree, verification, and recovery behavior are documented.
- [ ] Offline unit, property/boundary, corruption, and collision tests pass.

---

## PAISA-RM-012 — Add a fixture-backed paired experiment runner

**Suggested title:** `[PAISA-RM-012] Add matched price-only versus price-plus-sentiment experiments`

**Suggested labels:** `priority:must`, `type:feature`, `area:experiments`, `area:sentiment`

**Depends on:** `PAISA-RM-011`

**Gate:** `G4-DAILY-SENTIMENT`

### Scope

Implement paired configurations and a fixture-backed runner in which the only
intentional feature difference is the ordered sentiment feature block. Intersect
eligible target dates before splitting, fit every learned transform on training
only, freeze selection on validation, and pass predictions to independent
policy/portfolio ledgers with identical assumptions.

### Out of scope

- Running the official corpus, changing the target, tuning on test data, or
  sharing cash/holdings/trades between variants.

### Acceptance checklist

- [ ] Paired manifests prove identical rows, target, split dates, seed, policy,
      execution, initial cash, fees, and benchmarks.
- [ ] Ordered feature sets differ only by declared sentiment columns.
- [ ] Boundary and transform tests detect look-ahead leakage.
- [ ] Validation selection is persisted before test evaluation.
- [ ] Predictions, model IDs, experiment IDs, portfolio IDs, and artifacts are independent.
- [ ] Negative/equal fixture outcomes are reported without promotion language.
- [ ] Runner, contract, and interpretation documentation are complete.

---

## PAISA-RM-013 — Execute and report the official paired OGDC comparison

**Suggested title:** `[PAISA-RM-013] Execute the official OGDC price-plus-sentiment gate`

**Suggested labels:** `priority:must`, `type:experiment`, `area:experiments`, `gate:paired-comparison`

**Depends on:** `PAISA-RM-012`

**Gate:** `G4-DAILY-SENTIMENT`; completion must satisfy `G5-PAIRED-COMPARISON`.

### Scope

Execute the frozen paired configurations from a clean source commit with fresh
immutable run/model/experiment/portfolio IDs. Verify artifacts, produce one
comparison report, and record prediction and portfolio outcomes separately.

### Out of scope

- Retuning after seeing test results, overwriting price-only evidence, or
  describing an improvement as causal or as financial advice.

### Acceptance checklist

- [ ] Source commit, dirty-tree state, runtime, seeds, feature order, date ranges,
      target, horizon, and split counts are recorded.
- [ ] The exact matched-date proof and all gate assumptions pass.
- [ ] Shared-holdout reuse is disclosed if the existing test window is used.
- [ ] Result can be positive, equal, negative, or inconclusive without changing acceptance.
- [ ] Prediction selection and simulated-return comparisons remain separate.
- [ ] Artifacts stay outside Git and verification commands reproduce every published table.
- [ ] The report includes limitations and the academic disclaimer.

---

## PAISA-RM-014 — Generate versioned SHAP explanation artifacts

**Suggested title:** `[PAISA-RM-014] Add versioned SHAP explanation artifacts`

**Suggested labels:** `priority:should`, `type:feature`, `area:explainability`, `offline-ci`

**Depends on:** `PAISA-RM-003`

**Gate:** `G0-INTEGRITY`

**Implementation status:** Completed on
`feature/tree-model-shap-explainability`, pending merge. The accepted local
pack is `XAI-20261024-140320-OGDC-tree-v1`; its independently committed
`artifact_manifest.json` SHA-256 is
`cd43f76a208bed6b549d6dbfb3b4e468dd91db021f4543f1cfefdbbc7a661fda`.

### Scope

Define and implement model-appropriate SHAP generation for one retained model
type at a time. Persist the exact model root, ordered features, background data
identity, explainer, class/output explained, sample rows, library versions,
expected value, local values, aggregate values, and checksums.

### Out of scope

- Causal claims, advice, silently mixing SHAP output semantics, or explanations
  for unverified model binaries.

### Acceptance checklist

- [x] Deterministic fixtures exercise the contract offline, and dedicated
      Python 3.13.7 CI runs real `shap.TreeExplainer` calls for small Random
      Forest and XGBoost estimators under the certified direct pins in
      `constraints-explainability.txt`; that file is not a standalone or fully
      hashed lock.
- [x] Additivity/reconciliation is checked within documented family-specific
      tolerances.
- [x] Feature order and explained class/output are unambiguous.
- [x] Unsupported algorithms fail clearly rather than switching explainers silently.
- [x] Existing explanation IDs are collision-refusing and source roots are revalidated.
- [x] Verification distinguishes internal `PASS`, sidecar transport integrity,
      and manifest authentication. Official use requires
      `--require-authenticated`; a non-allowlisted run also requires an
      independently retained `--expected-manifest-sha256`.
- [x] PNG checks are structural rather than a semantic rerender; authenticated
      manifest bytes bind the accepted plots.
- [x] Contract, limitations, constrained dependencies, and strict verification
      commands are documented.

---

## PAISA-RM-015 — Serve and display verified SHAP evidence

**Suggested title:** `[PAISA-RM-015] Add read-only SHAP API and dashboard explanations`

**Suggested labels:** `priority:should`, `type:feature`, `area:explainability`, `area:api`, `area:frontend`

**Depends on:** `PAISA-RM-014`

**Gate:** `G0-INTEGRITY`

### Scope

Expose additive read-only explanation endpoints and a dashboard view for local
and aggregate verified SHAP evidence. Present feature contribution direction,
reference value, model output, and caveats from persisted artifacts.

### Out of scope

- Recomputing SHAP in the API/browser or translating association into causal explanation.

### Acceptance checklist

- [ ] API rejects changed model/explanation roots and unsafe paths.
- [ ] Frontend schemas require model/output/feature/background identities.
- [ ] Explanation text says model behavior, not real-world cause or advice.
- [ ] Loading, pagination, malformed-data, and integrity-error tests pass.
- [ ] Backend tests plus frontend tests/lint/build pass.
- [ ] API and dashboard documentation includes interpretation limits.

---

## PAISA-RM-016 — Onboard one additional PSX stock with data-quality evidence

**Suggested title:** `[PAISA-RM-016] Add a provenance-first multi-stock onboarding pilot`

**Suggested labels:** `priority:should`, `type:feature`, `area:data`, `area:multi-stock`

**Depends on:** `PAISA-RM-013`

**Gate:** `G5-PAIRED-COMPARISON`

### Scope

Choose at least one additional PSX symbol from the documented universe and
produce a fixture-tested data-quality/provenance assessment: provider evidence,
coverage, duplicates, gaps, missing fields, corporate-action/adjustment status,
feature readiness, and chronological split feasibility.

### Out of scope

- Generalizing OGDC results, silently substituting sample data, or training a
  pooled model before its design is approved.

### Acceptance checklist

- [ ] Selection rationale and exact symbol metadata are documented.
- [ ] Data-quality output is immutable, checksum-bound, and stock-specific.
- [ ] Missing/high/low/adjustment/dividend limitations remain explicit.
- [ ] Offline fixtures cover symbol isolation and inconsistent calendars.
- [ ] A clear READY/BLOCKED decision precedes model training.
- [ ] No raw market dataset is committed.

---

## PAISA-RM-017 — Scale experiment orchestration across stocks

**Suggested title:** `[PAISA-RM-017] Add isolated multi-stock experiment orchestration`

**Suggested labels:** `priority:stretch`, `type:feature`, `area:multi-stock`, `area:experiments`

**Depends on:** `PAISA-RM-016`

**Gate:** `G0-INTEGRITY`

### Scope

Generalize orchestration to an explicit list of READY symbols while retaining
stock-specific datasets, splits, transforms, models, predictions, experiments,
portfolios, and conclusions. Any pooled model requires a separate declared
design and symbol-aware split policy.

### Out of scope

- Reusing one stock's scaler or portfolio state, comparing unmatched periods
  without disclosure, or implying diversified live performance.

### Acceptance checklist

- [ ] Fixture tests use different calendars and catch cross-symbol leakage/state reuse.
- [ ] Every output identity includes or references its exact symbol universe.
- [ ] Per-stock and aggregate metrics are separate and denominators are explicit.
- [ ] Failures for one symbol cannot publish a falsely complete aggregate run.
- [ ] CLI, artifact layout, and interpretation documentation are updated.

---

## PAISA-RM-018 — Create a versioned entity and relationship registry

**Suggested title:** `[PAISA-RM-018] Add a versioned PSX entity and relationship registry`

**Suggested labels:** `priority:should`, `type:feature`, `area:entities`, `area:event-graph`

**Depends on:** None for the fixture-backed registry contract. Real-source
population additionally depends on `PAISA-RM-006`.

**Gate:** `G0-INTEGRITY` for fixture design. Real-source population separately
requires `G2-NEWS-PERMISSION`.

### Scope

Define immutable, effective-dated records for tickers, legal companies,
subsidiaries, sectors, industries, places, event actors, aliases, and unresolved
mentions. Preserve raw mention, linked identity, method, confidence, reviewer,
evidence source, and validity interval.

### Out of scope

- Forced fuzzy matches, timeless relationships, feature aggregation, model input,
  or interactive graph presentation.

### Acceptance checklist

- [ ] Fixtures cover OGDC aliases, ambiguous names, unresolved mentions,
      conflicting evidence, renames, and expired relationships.
- [ ] Lookups are evaluated as of a requested feature date.
- [ ] Source permissions and evidence checksums travel with real records.
- [ ] Registry versions and IDs are immutable and collision-refusing.
- [ ] Schema, review workflow, and non-causal interpretation are documented.

---

## PAISA-RM-019 — Build a dated event-graph artifact and API

**Suggested title:** `[PAISA-RM-019] Add immutable dated event-graph artifacts and read-only API`

**Suggested labels:** `priority:should`, `type:feature`, `area:event-graph`, `area:api`

**Depends on:** `PAISA-RM-008`, `PAISA-RM-018`

**Gate:** Real research claims require `G3-NEWS-OVERLAP`.

### Scope

Build an immutable graph projection with explicit node/edge types, direction,
source evidence, confidence, event/validity times, extraction method, and
unresolved/conflicting status. Add read-only endpoints that filter by entity and
as-of date without inventing or recomputing edges.

### Out of scope

- Price prediction, causal propagation, browser-created relationships, or
  treating co-mention as a verified business relationship.

### Acceptance checklist

- [ ] Fixture graph tests cover ambiguous, conflicting, expired, future, and missing evidence.
- [ ] An as-of query cannot return an edge unavailable at that date.
- [ ] Every edge resolves to permitted checksum-bound source evidence.
- [ ] Graph and registry roots are revalidated; unsafe paths and collisions fail closed.
- [ ] API responses require confidence/method/timing fields and the disclaimer.
- [ ] Contract, verification, and interpretation documentation are complete.

---

## PAISA-RM-020 — Add an interactive event-graph view

**Suggested title:** `[PAISA-RM-020] Add a read-only interactive event-graph dashboard`

**Suggested labels:** `priority:stretch`, `type:feature`, `area:event-graph`, `area:frontend`

**Depends on:** `PAISA-RM-019`

**Gate:** `G0-INTEGRITY`

### Scope

Visualize API-provided nodes and edges with as-of-date filtering, provenance,
confidence, unresolved/conflicting markers, and accessible table fallback.

### Out of scope

- Creating edges client-side, graph-based trading signals, or causal/advice language.

### Acceptance checklist

- [ ] Client validation rejects missing temporal/provenance fields.
- [ ] Date changes request filtered server evidence rather than hiding future edges only in CSS.
- [ ] Ambiguous and inferred edges are visually distinct from observed facts.
- [ ] Keyboard/accessibility, loading, empty, and malformed-data states are tested.
- [ ] Frontend tests, lint, build, and user documentation pass.

---

## PAISA-RM-021 — Generate the reproducible final report

**Suggested title:** `[PAISA-RM-021] Generate the final PAISA research report from verified artifacts`

**Suggested labels:** `priority:must`, `type:documentation`, `area:report`, `release`

**Depends on:** `PAISA-RM-005`, `PAISA-RM-008`; additionally
`PAISA-RM-013` only when `G3-NEWS-OVERLAP` opens the paired-experiment lane.

**Gate:** The historical-evidence lane must have a verified disposition. A
paired-result claim requires `G5-PAIRED-COMPARISON`; a verified blocking result
does not.

### Scope

Create the final report and a deterministic report-data export from verified
artifacts. Cover problem statement, architecture/contracts, data provenance,
leakage controls, experiments, prediction metrics, simulated portfolios,
LSTM, news coverage, sentiment and the paired comparison when its gates open,
limitations, reproducibility, and the disposition of every SHOULD/STRETCH
issue. If the overlap gate remains closed, report that verified negative result
and explain why daily aggregation and paired modeling were not executed.

### Out of scope

- Hand-editing metrics, selecting only favorable results, causal claims, or
  representing simulations as financial advice.

### Acceptance checklist

- [ ] Every table/figure records its artifact/run IDs and can be regenerated.
- [ ] Validation decisions, shared-holdout test results, and portfolio outcomes are separate.
- [ ] Negative, blocked, and inconclusive results remain visible.
- [ ] A verified `NO_OVERLAP` or other blocking coverage result is sufficient
      to close the evidence lane; it is not rewritten as a paired experiment.
- [ ] When `G3` opens, `PAISA-RM-013` and `G5` are complete before any paired
      result appears in the report.
- [ ] One-stock/daily-data, provider, timing, adjustment, dividend, fee,
      execution, and generalization limits are explicit.
- [ ] SHAP, multi-stock, and event-graph work is marked completed or explicitly deferred.
- [ ] References, licenses/attribution, appendices, commands, and test evidence are complete.
- [ ] The report prominently says academic simulation and not financial advice.

---

## PAISA-RM-022 — Prepare and rehearse the final demo

**Suggested title:** `[PAISA-RM-022] Prepare the final demo pack and release rehearsal`

**Suggested labels:** `priority:must`, `type:release`, `area:demo`, `documentation`

**Depends on:** `PAISA-RM-021`

**Gate:** Release acceptance after all required checks pass.

### Scope

Prepare a time-boxed supervisor demo with exact startup/preflight commands,
fixture-only clean-clone mode, locally restored official-artifact mode, API and
dashboard checkpoints, expected IDs/results, failure recovery, screenshot list,
and a fallback narrative for missing optional artifacts or offline services.

### Out of scope

- Live trading, live-provider dependence, downloading weights during the demo,
  hiding blockers, or merging/tagging without explicit release approval.

### Acceptance checklist

- [ ] Clean-clone setup and all offline CI-equivalent checks pass from documented commands.
- [ ] Official artifacts are restored and verified without overwriting existing IDs.
- [ ] API and dashboard show baseline, LSTM, and unified results plus either the
      gated paired result or the verified reason that paired work could not run.
- [ ] Demo explicitly distinguishes validation selection, test evidence, and simulated return.
- [ ] A no-network/no-private-artifact fallback remains demonstrable with fixtures.
- [ ] Expected warnings, known blockers, timing, and recovery steps are rehearsed.
- [ ] Release notes identify the source commit, tags to create, artifacts not in Git,
      and all deferred risks; merge/tag actions remain manual.

---

## Backlog maintenance

When an issue is created, replace its plain dependency IDs in the GitHub body
with links while retaining the stable IDs. When an issue closes, add the merged
PR, source commit, test evidence, and any generated artifact IDs to the issue.
If a gate fails, close the evidence-producing issue with the negative result and
mark downstream issues blocked; do not rewrite this dependency graph to imply
that the gate passed.
