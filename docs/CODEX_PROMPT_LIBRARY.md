# PAISA Codex Prompt Library

Status date: 24 October 2026

These prompts are deliberately narrow. Replace every `<placeholder>`, attach the
relevant issue or review findings, and give an agent one PR-sized outcome at a
time. PAISA predictions, signals, trades, curves, and simulated returns are
academic simulations and are not financial advice.

## Common context block

Paste this block above any task-specific prompt:

```text
Repository: Arfa01/PAISA
Working agreement: read and follow AGENTS.md before acting.
Base: latest origin/main. Never edit or commit on main.
Branch: <branch-name>
Issue: <issue-number-or-backlog-id>

Preserve PAISA v0 behavior and all existing schemas unless the issue explicitly
approves a migration. Keep model prediction, decision policy, portfolio state,
API delivery, and dashboard presentation separate. Use daily PSX data and OGDC
for the first vertical slice. Prevent look-ahead leakage; keep feature_date,
target_date, signal_date, and execution_date distinct; fit learned transforms on
training data only. Never overwrite an artifact ID or reuse portfolio state.

Do not run live providers, download model weights, or require private generated
artifacts unless the issue explicitly authorizes a local evidence run. Keep the
default test suite offline. User-facing results must say that they are academic
simulations and not financial advice.

Before changing code, run the relevant existing tests. Add regression tests for
every behavior change. At handoff, report changed files, commands and results,
assumptions, and remaining risks. Do not merge automatically.
```

## Implementation

```text
<paste the common context block>

Implement only this outcome:
<one concrete outcome>

In scope:
- <file/component boundary>
- <required contract or fixture>
- <required tests and documentation>

Out of scope:
- <adjacent feature>
- <real provider/model execution>
- <schema or artifact migration not approved by the issue>

Acceptance criteria:
- <observable criterion 1>
- <observable criterion 2>
- <exact validation commands>

First inspect the existing contracts and identify any ambiguity that would make
the result unsafe. Otherwise proceed, keep the diff focused, validate it, commit,
push, and open a PR. Do not merge it.
```

## Tree-SHAP artifact execution

```text
<paste the common context block>

Operate the existing tree-model SHAP workflow; do not redesign it or retrain a
model. Read docs/TREE_MODEL_SHAP_EXPLAINABILITY.md,
docs/EXPLAINABILITY_DATA_CONTRACT.md, and
docs/SHAP_INTERPRETATION_POLICY.md first.

Use only these restored, checksum-valid source roots:
- artifacts/model_training/TRN-20261020-191815-624662-nogit
- artifacts/model_comparisons/CMP-20261024-OGDC-price-only-02
- artifacts/vertical_slice/VSL-20261020-OGDC-main-02

Create a separate Python 3.12+ / NumPy 2+ environment and install the exact
requirements-explainability.txt pin. Confirm `shap==0.54.1` and run `pip check`.
Explain only M-001 through M-004 over the exact 184-date shared project
holdout. Do not explain M-005/M-006, refit preprocessing, weaken additivity,
load an unverified model path, or modify a source artifact.

Run scripts/run_tree_shap_explainability.py with the three fixed inputs and
artifacts/explainability as output. Use a fresh immutable XAI ID. Verify both
the published directory and its ZIP/sidecar with
scripts/verify_explainability_run.py. Do not commit the generated pack.

Report the observed run ID, package versions, registered model IDs, output
space, row/feature counts, maximum and mean additivity errors, top features,
rank agreements/disagreements, audit categories, quality status, relative
artifact name, and ZIP SHA-256 directly from the verified pack. If execution
fails, preserve the source artifacts, report the exact failing gate, and do not
publish or describe a false PASS. Use model-behavior language only; do not make
causal, profitability, future-performance, or financial-advice claims.
```

## Read-only review

```text
Repository: Arfa01/PAISA
Review target: <PR URL, branch, or commit range>

Perform a read-only review. Do not edit files, push commits, change the PR, or
merge. Read AGENTS.md, inspect the complete diff and relevant surrounding code,
and run non-mutating checks where useful.

Prioritize correctness, time-series leakage, split-boundary contamination,
artifact immutability, path/checksum trust boundaries, independent portfolio
state, API compatibility, fixture-only CI, missing tests, and misleading
research claims. Verify that results are described as academic simulations and
not financial advice.

Return only actionable findings ordered by severity. For each finding include
the file and line, the failing scenario, why it matters, and the smallest safe
fix. If there are no findings, say so and list the checks performed and any
residual risks.
```

## Fix review findings

```text
<paste the common context block>

The branch already has an open PR. Fix only the attached review findings:
<paste exact findings>

For each finding, reproduce or prove the problem first, add or update a
regression test, implement the smallest compatible fix, and rerun the focused
test plus the full affected validation set. Do not refactor unrelated code or
change public contracts unless the finding requires it. Re-check the full diff,
commit the fixes, push the existing branch, and summarize how each finding was
resolved. Do not merge.
```

## Post-merge acceptance

```text
Repository: Arfa01/PAISA
Merged PR: <PR URL or number>

Work read-only from a clean checkout of the latest origin/main. Do not create a
feature branch or change files. Confirm that the merge commit is present, then
run the validation commands promised by the PR using only fixture/offline inputs.
Check pip consistency, Python compilation, backend tests, documentation
contracts, frontend tests/lint/build, and git diff/status. Do not run live
providers or download model weights.

Return PASS or FAIL with the exact origin/main commit, commands, test counts,
skips and reasons, warnings, clean-tree result, and any follow-up issue needed.
Do not tag or publish a release.
```

## Error triage

```text
Repository: Arfa01/PAISA
Failure context: <local, CI, Colab, API, dashboard, or artifact verification>
Exact command: <command>
Observed output: <paste complete relevant output>

Diagnose first; do not implement a fix. Read AGENTS.md and inspect the smallest
relevant code, configuration, environment versions, and recent diff. Reproduce
with fixture/offline inputs when possible. Do not delete, overwrite, regenerate,
or weaken validation of an immutable artifact. Do not use a live provider or
download weights merely to make the error disappear.

Return: likely root cause with evidence, whether this is code/environment/data,
the smallest reproduction, safe recovery steps, and the exact proposed fix and
tests. Clearly identify any missing evidence or permission needed before work
can continue.
```

## Release and tagging

```text
Repository: Arfa01/PAISA
Candidate origin/main commit: <full commit>
Proposed tag: <tag>
Release scope: <merged PR range and milestone>

Prepare a release candidate; do not tag, push a tag, publish, or merge until the
final evidence is reviewed. Read AGENTS.md and docs/REPRODUCTION_COMMANDS.md.
Confirm the commit is on origin/main, the worktree is clean, required PRs are
merged, tests/compile/pip/frontend/doc contracts pass, and no generated data,
weights, secrets, or private artifacts are tracked. Verify release notes against
immutable final summaries and identify ignored artifacts that a custodian must
preserve separately.

Return a proposed annotated-tag command, release title and notes, validation
evidence, artifact/checksum references, migration/compatibility notes, known
limitations, and a GO/NO-GO decision. State that all reported predictions and
returns are academic simulations and not financial advice. Wait for explicit
approval before any tag or release mutation.
```

## Report generation

```text
Repository: Arfa01/PAISA
Report type and audience: <supervisor, technical appendix, milestone, or final>
Evidence cutoff: <date and origin/main commit>
Allowed artifact summaries: <exact immutable run IDs and files>

Generate a factual report from committed documentation and the listed verified
artifact summaries only. Do not retrain, recalculate, infer missing metrics, or
silently combine incompatible runs. Separate validation selection from test
evaluation, prediction metrics from portfolio metrics, and historical verifier
authority from current verifier authority. Call out shared-holdout reuse,
coverage gates, source permission, missing evidence, and superseded artifacts.

Include methods, provenance, measured results, limitations, reproducibility
commands, and citations to repository files/run IDs. Label predictions, trades,
and returns as academic simulations and not financial advice. Return the draft
plus a short list of every numerical claim and its source for independent
checking. Do not commit unless explicitly asked.
```

## Using the prompts economically

- Put durable facts in an issue or repository document; do not spend a new agent
  turn rediscovering them.
- Give implementation, review, and acceptance to separate turns so each has a
  single purpose and a short context.
- Paste only the exact failing output or review findings needed for the next
  step. Link to `docs/PROJECT_STATE_2026-10-24.md` for the broader snapshot.
- Prefer a focused test while iterating, then the full affected matrix once
  before handoff. The canonical commands are in
  [`REPRODUCTION_COMMANDS.md`](REPRODUCTION_COMMANDS.md).
