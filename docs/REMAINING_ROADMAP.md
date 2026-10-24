# PAISA Remaining Roadmap

Status date: 24 October 2026

Base: `origin/main` at `a80594d80a884dcef129f2d4a9c896217c6793c5`
after merged PR #17.

PAISA is an academic simulation. Predictions, signals, trades, portfolio curves,
and simulated returns are not financial advice.

## How to use this roadmap

Each row is intended to fit in one focused pull request or one explicitly
documented research/operations issue. Use the stable `PAISA-RM-NNN` identifier
in the branch name, commit message, pull request, and retained evidence. Do not
combine rows merely to reduce issue count.

Priorities have the following meanings:

- **MUST**: required for the remaining controlled OGDC research comparison or
  final handoff.
- **SHOULD**: valuable proposal work that may proceed after its dependencies,
  but must not delay the gated OGDC comparison.
- **STRETCH**: useful expansion only after the MUST path is reproducible.

When AI-agent usage is limited, take the first unblocked issue, complete its
local acceptance commands, merge it, and only then start its dependent. Human
source-permission work may run in parallel with engineering, but it cannot be
replaced by code or an agent assertion.

## Requested topic index

The issue map below covers LSTM integration, unified model comparison, SHAP,
multi-stock scaling, an event graph, historical-news overlap, the daily
sentiment aggregation gate, the price-plus-sentiment comparison gate, the final
report, and the final demo. The issue table is the concise source for
priorities, dependencies, and exit criteria; the paste-ready backlog expands
each row into scope, exclusions, acceptance checks, tests, and documentation.

## Non-negotiable gates

| Gate | Requirement | Opens |
|---|---|---|
| `G0-INTEGRITY` | Offline tests use only small committed fixtures. Generated artifacts, raw corpora, model weights, secrets, and local databases stay out of Git. Readers validate immutable IDs, checksums, schemas, paths, and completion state before returning data. | Every engineering issue. |
| `G1-LSTM-READ` | A small checksum-valid fixture proves that the API can read the LSTM schema without TensorFlow, deserializing a model, retraining, or requiring the certified local run. | Unified six-model comparison work. |
| `G2-NEWS-PERMISSION` | The exact source, fields, date range, academic-use basis, attribution, retention, redistribution, and automation rights have a recorded `PERMITTED` decision. `REVIEW_REQUIRED` does not pass. | Historical acquisition/import. |
| `G3-NEWS-OVERLAP` | A fresh current-schema, source-evidence-bound audit uses new immutable IDs, passes source permission and ingestion completeness, and satisfies configured split-coverage thresholds with defensible cutoff-time evidence. A historical `NO_OVERLAP` pack cannot be relabelled or overwritten. | Sentiment inference for the comparison corpus and daily aggregation. |
| `G4-DAILY-SENTIMENT` | Daily features are derived only from articles whose `available_at` is on or before the declared decision cutoff. After-close and non-trading-day rules, missing-news rows, source counts, and train-only transforms are verified. | Price-plus-sentiment experiments. |
| `G5-PAIRED-COMPARISON` | Price-only and price-plus-sentiment configurations use identical eligible target dates, target, chronological splits, seeds, selection rule, policy, execution, fees, and independent portfolios. Validation freezes the choice before disclosed shared project holdout evaluation. | Official paired result and the paired-result sections of the final report and demo. |

If `G2` or `G3` fails, record the negative result and leave all downstream
issues blocked. Never weaken a threshold, invent a historical timestamp, use a
later fetch time as an earlier availability time, or reuse an old run ID to make
the gate pass.

## Ordered issue map

| ID | Priority | PR-sized outcome | Depends on | Required gate at start | Exit evidence |
|---|---|---|---|---|---|
| `PAISA-RM-001` | MUST | Fixture-backed, read-only LSTM artifact repository and API | None | `G0-INTEGRITY` | Offline API tests accept a sealed fixture and reject corrupt/incomplete packs. |
| `PAISA-RM-002` | MUST | LSTM result view in the dashboard | `PAISA-RM-001` | `G1-LSTM-READ` | Frontend schema, component, loading/error, and disclaimer tests pass. |
| `PAISA-RM-003` | MUST | Immutable unified M-001--M-006 comparison artifact | `PAISA-RM-001` | `G1-LSTM-READ` | Builder proves the exact 184-date intersection; the offline verifier reconstructs prediction and full `Decimal` portfolio semantics from portable source evidence; and validation, shared project holdout, and portfolio outcomes remain separate. |
| `PAISA-RM-004` | MUST | Read-only unified comparison API | `PAISA-RM-003` | `G0-INTEGRITY` | Additive typed endpoint returns only verified persisted comparison evidence. |
| `PAISA-RM-005` | MUST | Unified comparison dashboard and export | `PAISA-RM-002`, `PAISA-RM-004` | `G0-INTEGRITY` | Six models render with selection/holdout caveats and no client-side metric recomputation. |
| `PAISA-RM-006` | MUST | Historical-news permission dossier and source decision | None | Human review | Exact permissions and prohibited uses are checksum-bound; no acquisition occurs in this issue. |
| `PAISA-RM-007` | MUST | Approved historical OGDC acquisition/import run | `PAISA-RM-006` | `G2-NEWS-PERMISSION` | New collision-refusing run is complete, verified, private where required, and outside Git. |
| `PAISA-RM-008` | MUST | Current-schema historical news/price overlap audit | `PAISA-RM-007` | `G2-NEWS-PERMISSION` | New audit records `READY_FOR_AGGREGATION` or an explicit blocking result without altering old packs. |
| `PAISA-RM-009` | MUST | Pinned FinBERT inference over the approved overlap corpus | `PAISA-RM-008` | `G3-NEWS-OVERLAP` | New immutable sentiment run passes verification; audit sample is reviewed; no accuracy claim is made without labels. |
| `PAISA-RM-010` | MUST | Daily sentiment feature contract and fixture oracle | `PAISA-RM-009` | `G3-NEWS-OVERLAP` | Contract tests cover cutoffs, after-close, weekends/holidays, missing news, duplicates, and split boundaries. |
| `PAISA-RM-011` | MUST | Daily sentiment aggregation and immutable artifact writer | `PAISA-RM-010` | `G3-NEWS-OVERLAP` | Aggregator reproduces the oracle, refuses collisions, and passes leakage and checksum verification. |
| `PAISA-RM-012` | MUST | Fixture-backed paired price/sentiment experiment runner | `PAISA-RM-011` | `G4-DAILY-SENTIMENT` | Tests prove matched rows and assumptions, train-only transforms, validation-only selection, and separate ledgers. |
| `PAISA-RM-013` | MUST | Official OGDC paired comparison run and result report | `PAISA-RM-012` | `G4-DAILY-SENTIMENT` | Fresh immutable runs satisfy `G5-PAIRED-COMPARISON`; negative or inconclusive results are retained. |
| `PAISA-RM-014` | SHOULD | Versioned tree-SHAP artifact generator and portable verifier; implementation present, official real pack pending | `PAISA-RM-003` | `G0-INTEGRITY` | Fixture explanations bind model, features, background data, output/class, versions, and checksums; a real four-model directory and ZIP pass offline verification under a new immutable ID. |
| `PAISA-RM-015` | SHOULD | Read-only SHAP API and dashboard explanation view | `PAISA-RM-014` | `G0-INTEGRITY` | Additive API/UI present behavior, never causality or advice, with offline contract tests. |
| `PAISA-RM-016` | SHOULD | One-stock-at-a-time PSX data-quality onboarding pilot | `PAISA-RM-013` | `G5-PAIRED-COMPARISON` | At least one additional approved symbol has provenance, coverage, missing-field, and split-readiness evidence. |
| `PAISA-RM-017` | STRETCH | Multi-stock experiment orchestration | `PAISA-RM-016` | `G0-INTEGRITY` | Per-stock or explicitly pooled identities, transforms, portfolios, and conclusions remain isolated. |
| `PAISA-RM-018` | SHOULD | Versioned entity and relationship registry | None for fixture contract | `G0-INTEGRITY`; `G2-NEWS-PERMISSION` before real-source population | Fixtures cover aliases, ambiguity, unresolved entities, effective dates, and evidence provenance. |
| `PAISA-RM-019` | SHOULD | Dated event-graph artifact and read-only API | `PAISA-RM-008`, `PAISA-RM-018` | `G3-NEWS-OVERLAP` for research claims | Every edge is temporal, directional, confidence-scored, source-traceable, and immutable. |
| `PAISA-RM-020` | STRETCH | Interactive event-graph dashboard | `PAISA-RM-019` | `G0-INTEGRITY` | UI reads the API only and clearly separates observed evidence from inferred relationships. |
| `PAISA-RM-021` | MUST | Reproducible final report | `PAISA-RM-005`, `PAISA-RM-008`; `PAISA-RM-013` only if `G3` opens | Verified evidence-lane disposition; `G5` for paired claims | Tables are generated from verified artifacts; a blocking overlap result or gated paired result, limitations, and all deferred SHOULD/STRETCH work are explicit. |
| `PAISA-RM-022` | MUST | Final demo pack and release rehearsal | `PAISA-RM-021` | Release acceptance | Clean-clone fixture workflow, artifact-restored official workflow, API, dashboard, and fallback narrative are rehearsed. |

The `PAISA-RM-003` evidence-only implementation is present on
`feature/unified-price-model-comparison`, with canonical local pack
`CMP-20261024-OGDC-price-only-02`. Its final ZIP SHA-256 is
`8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe`,
and its verified total file count is `65`. The earlier
`CMP-20261024-OGDC-price-only-01` is a superseded, non-portable historical
draft. The hardened pack uses a digest-allowlisted redacted vertical
certificate, recursive portable hygiene, two raw market files, six signal
files, and full `Decimal` policy, execution, accounting, and benchmark replay.
It exposes exactly two logical benchmark metric rows, each with six independent
ledger-provenance records. It was explicitly authorized before `PAISA-RM-001`;
it does not clear `G1-LSTM-READ`, provide a committed fixture, or change the
remaining API/dashboard dependency order. This artifact is price-only and
tabular/report-only: it does not integrate sentiment or import Matplotlib.

The `PAISA-RM-014` producer and offline verifier are implemented on
`feature/tree-model-shap-explainability`. They are fixed to M-001 through M-004,
`shap==0.54.1`, class-1 probability space, the exact 184-date shared project
holdout, a deterministic 100-row training background, and strict additivity
gates of `5e-5` for Random Forest and `1e-6` for XGBoost. The dependency pin
was smoke-tested under the authoritative Python 3.13.7, NumPy 2.3.5,
scikit-learn 1.9.1, and XGBoost 3.4.1 stack. RM-014 is not treated as closed
until an official real directory and ZIP verify; no run ID, ranking, top
feature, or checksum is yet recorded. RM-015 remains a separate API/UI task.

## Recommended execution sequence

### Track A: integrate the completed LSTM evidence

Start with `PAISA-RM-001`. It is the first unblocked engineering issue because
the certified LSTM run already exists, but its generated artifacts must not be
required by a clean clone or CI. The PR should commit only a small synthetic or
sanitized checksum-valid fixture representing the public result schema. It must
not commit TensorFlow weights, checkpoints, private inputs, or a copy of the
local certified run.

Then complete `PAISA-RM-002` through `PAISA-RM-005` sequentially. A unified
comparison is a new immutable derived artifact referencing the baseline and
LSTM integrity roots. It is not a merged model run, must not rewrite either
source run, and must disclose that all six models share an already exposed
184-date test window. Validation-selected models, post-freeze test rankings,
and simulated portfolio rankings remain separate concepts.

### Track B: clear the historical-news gate

`PAISA-RM-006` is a human-governance task and may proceed in parallel. It is
not permission to start `PAISA-RM-007` until the machine-readable decision is
`PERMITTED`. Existing Dawn, GDELT DOC, GKG, PSX-announcement, coverage, and
sentiment packs remain immutable historical evidence; none currently clears
the overlap gate.

`PAISA-RM-008` may legitimately finish with `NO_OVERLAP`, `INSUFFICIENT_OVERLAP`,
or another blocking result. That is a valid research outcome, but it leaves
`PAISA-RM-009` through `PAISA-RM-013` blocked. Only a fresh current-contract
audit satisfying the configured thresholds opens daily aggregation.

### Track C: aggregate and compare only after the gate

`PAISA-RM-009` through `PAISA-RM-013` are deliberately sequential. Keep the
article-level model identity, daily feature artifact, stock model, decision
policy, and portfolio as separate components. The paired experiment must add
sentiment features without changing the price-only target or evaluation
assumptions. Reusing the 2025-12-16 through 2026-09-15 test window must be
reported as repeated evaluation on the shared project holdout, not as a fresh
independent evaluation.

### Track D: explainability, scaling, and events

The tree-SHAP artifact/verifier implementation is present because it does not
depend on historical news. Complete its official real execution and acceptance
before starting the separate read-only SHAP API/dashboard issue. Multi-stock work begins only after the OGDC paired
comparison is reproducible. The event-registry contract may start with
deterministic fixtures before source review, but real registry population and
event-graph claims require temporally valid, permitted evidence. Interactive
graph presentation remains stretch scope and must not create relationships in
the browser.

### Track E: report and demonstrate

The final report depends on the completed LSTM/unified path and a verified
disposition of the historical-evidence lane, not on a positive result or a gate
being forced open. If current coverage remains blocking, the report and demo
must present that negative evidence and mark aggregation/paired work not
executed. If the gate opens, the paired path through `PAISA-RM-013` becomes a
conditional prerequisite. The report must document every SHOULD and STRETCH
task, including explicit deferrals. The final demo must work in two modes:
clean-clone fixture evidence and locally restored, checksum-verified official
artifacts. Neither mode may download private data, real model weights, or call
live providers during automated acceptance.

## Definition of exit for every pull request

- The branch contains only the issue's scope and preserves PAISA v0 interfaces.
- New IDs and directories are immutable and collision-refusing; old artifacts
  are never overwritten, upgraded in place, or relabelled.
- Tests are deterministic and offline by default. Provider/network checks are
  explicit opt-ins outside the normal suite.
- Learned transforms are fitted on training data only; time-series splits are
  chronological by `target_date`, with boundary labels purged where required.
- Model, policy, portfolio, API, and dashboard responsibilities remain separate.
- Backend tests, frontend tests/lint/build when affected, `pip check`,
  `compileall`, documentation-contract tests, and `git diff --check` pass.
- Public schemas, commands, artifact layouts, or dependencies changed by the
  PR are documented in the same PR.
- User-facing text states that results are academic simulations and not
  financial advice.

The paste-ready issue bodies for these rows are in
[`ISSUE_BACKLOG.md`](ISSUE_BACKLOG.md).
