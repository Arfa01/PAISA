# PAISA Workflow for Limited Agent Usage

Status date: 24 October 2026

PAISA can progress with a small number of deliberate agent turns if repository
state, issue scope, and validation evidence remain durable outside the chat.
This workflow keeps expensive discovery and training separate from small code
changes. All model outputs and simulated portfolio results are academic
simulations and are not financial advice.

## The operating rule: one issue, one branch, one observable outcome

Choose the first dependency-ready item in
[`ISSUE_BACKLOG.md`](ISSUE_BACKLOG.md). A useful issue should fit in one narrow
PR, name its inputs and outputs, exclude adjacent work, and have commands that
prove completion. If an issue needs both a new artifact contract and dashboard
work, split it: establish the validated read path first, then expose it through
the API, then present it.

Never spend an implementation turn on an item whose external gate is still
closed. In particular, daily sentiment aggregation and price-plus-sentiment
experiments remain blocked until permitted historical evidence passes the
current overlap gate.

## Sequential loop

1. **Select one ready issue.** Copy its backlog entry into a GitHub issue and
   record dependencies, acceptance criteria, and the exact artifact IDs it may
   read. Do not bundle cleanup or a second roadmap item.
2. **Start from the current integration point.** Fetch `origin`, confirm the
   expected commit, create the issue branch, and never work directly on `main`.
3. **Diagnose locally first.** Run `python scripts/doctor.py`, the relevant
   focused tests, and `git status --short`. Put any reproducible error and its
   exact command in the issue.
4. **Use one implementation turn.** Give the agent the common context and
   implementation template from
   [`CODEX_PROMPT_LIBRARY.md`](CODEX_PROMPT_LIBRARY.md). Ask for the smallest
   compatible change, tests, documentation, commit, push, and PR—not a broad
   roadmap sweep.
5. **Validate on the local machine.** Run focused tests during iteration, then
   the complete affected matrix from
   [`REPRODUCTION_COMMANDS.md`](REPRODUCTION_COMMANDS.md). Generated artifacts
   and model caches stay ignored.
6. **Use a separate read-only review turn.** A reviewer should inspect the diff,
   leakage boundaries, artifact trust boundary, compatibility, and missing
   regressions without editing the branch.
7. **Use one fix turn only when needed.** Paste the exact findings into the fix
   prompt. Re-run focused and affected tests, then push the same branch.
8. **Merge manually after checks and review.** Do not allow an agent to merge
   automatically. After merge, use the post-merge acceptance prompt on a clean
   latest `origin/main` checkout.
9. **Persist the baton.** Update the issue, project-state document when facts
   changed, and roadmap dependency status. The next turn should not need the
   previous chat transcript.

## Spend agent turns where they add the most value

| Work | Prefer | Reason |
|---|---|---|
| Listing files, running tests, checking versions | Local commands and `doctor.py` | Deterministic and cheap; paste only anomalies. |
| Defining a new contract or security boundary | One implementation agent turn | Benefits from repository-wide reasoning. |
| Reviewing a PR | Separate read-only agent turn | Avoids author confirmation bias. |
| Formatting, rerunning a known test, checking CI | Local commands/GitHub Actions | Repeated validation should be automated. |
| Long CPU/GPU training | Colab or the approved local experiment environment | Keeps agent time away from waiting and preserves explicit runtime evidence. |
| Live-provider diagnosis | A narrowly authorized local run | CI must remain offline and fixture-only. |
| Status/report writing | Repository state plus immutable summaries | Facts remain auditable and reusable. |

## Local validation and CI

Use the project virtual environment rather than the system interpreter. Run the
doctor before asking an agent to debug installation state:

```bash
source .venv/bin/activate
python scripts/doctor.py
```

During implementation, run the smallest relevant test file. Before PR handoff,
run all backend tests, `pip check`, `compileall`, and frontend test/lint/build.
GitHub Actions repeats the same offline checks in independent jobs and uses only
committed fixtures. It intentionally does not install TensorFlow, FinBERT,
`pypsx-toolkit`, model weights, or private artifacts and does not contact live
data providers.

If CI and local results differ, capture Python/Node versions, the failing step,
and the complete error in the issue. Use the error-triage prompt before changing
dependencies or weakening a test.

## Colab for optional experiments

Use Colab only for a versioned notebook or CLI whose contract already exists,
such as the price-only LSTM experiment. Before starting:

- check out the exact reviewed commit;
- install the documented optional requirements in a fresh runtime;
- restore checksum-verified input artifacts through the approved custodian
  path; never substitute a convenient dataset under an old ID;
- choose a fresh immutable run ID and record Python, package, device, seed,
  feature order, split dates, target definition, and command;
- do not describe GPU and CPU output as bit-identical.

After the run, download the complete generated pack. The LSTM runner does not
create a `.sha256` sidecar automatically: compute and record the archive
SHA-256 using the custodian procedure in [`TEAM_HANDOFF.md`](TEAM_HANDOFF.md),
then transfer the archive and sidecar together. Verify them locally before
using the summary in a report. Do not commit archives, fitted models, caches,
credentials, or raw large datasets. Only small approved fixtures and metadata
belong in Git.

## Artifact continuity

Git contains code, contracts, and small fixtures; it does not contain the
retained real runs. Treat these as two different backups:

- **Git/GitHub:** source, tests, configuration, notebooks, issue history, and
  documentation.
- **Artifact custodian storage:** checksum-inventoried vertical-slice, training,
  news, sentiment, coverage, evidence, and LSTM packs plus approved databases
  and transfer sidecars.

Never overwrite or “refresh” a retained run. A changed input, dependency,
configuration, or code revision gets a new ID. Restoration and verification
instructions live in [`TEAM_HANDOFF.md`](TEAM_HANDOFF.md); current canonical IDs
and their authority are listed in
[`PROJECT_STATE_2026-10-24.md`](PROJECT_STATE_2026-10-24.md).

## GitHub issues as durable memory

Each issue should contain:

- one result and one owning component;
- dependency issue IDs and gate state;
- explicit in-scope and out-of-scope lists;
- input artifact/schema versions and output contract;
- tests and exact exit commands;
- documentation and disclaimer requirements;
- links to the PR, review findings, CI run, and post-merge acceptance comment.

Close an issue only after its PR is on `main` and post-merge acceptance passes.
If evidence or permission blocks it, label it blocked and open the enabling issue
instead of asking an agent to improvise around the boundary.

## A low-usage weekly cadence

- **Session 1:** choose/refine one ready issue and run local baseline checks.
- **Session 2:** implementation and focused validation.
- **Session 3:** read-only review; use a fourth only if findings need a fix.
- **No-agent work:** CI, Colab execution, checksum transfer, manual demo
  rehearsal, and issue/status updates.

This cadence is intentionally sequential. Parallel feature branches that change
the same artifact or API contracts create merge and reasoning costs that exceed
the small amount of calendar time they save.
