# PAISA Agent Working Agreement

These instructions apply to the entire repository. Read them before making a
change.

## Git workflow

- Never edit or commit directly to `main`.
- Use one branch per feature or narrowly scoped task.
- Start from the intended integration base, keep commits focused, and do not mix
  unrelated cleanup into a feature branch.

## Compatibility and scope

- Preserve existing PAISA v0 functionality, commands, file schemas, and API
  behavior unless a migration is explicitly approved and documented.
- Use `OGDC` as the first stock for the vertical slice. Do not remove the
  existing ability to process other symbols.
- Use daily PSX data first. Higher-frequency data and other markets are outside
  the initial vertical slice.
- Treat model training, prediction, decision policies, portfolio simulation,
  API delivery, and dashboard presentation as separate components with explicit
  contracts between them.

## Data and ML correctness

- Avoid look-ahead leakage. A feature may use only information available on or
  before its declared `feature_date`.
- Use chronological train/validation/test splits; never randomly shuffle a time
  series split.
- Keep `feature_date`, `target_date`, `signal_date`, and `execution_date`
  distinct. Purge or otherwise prevent next-day labels from crossing split
  boundaries.
- Fit scalers, imputers, feature selection, and every other learned transform on
  training data only.
- Record random seeds, feature order, target definition, prediction horizon,
  date ranges, configuration, and metrics for every trained model.

## Tests and documentation

- Run the existing test suite before and after every change.
- Add automated tests for every new component and every corrected regression.
- Update documentation whenever a public interface, command, configuration,
  schema, artifact layout, or dependency changes.
- Preserve offline tests; keep network/provider checks separate from the default
  unit test suite.

## Artifacts and repository hygiene

- Never overwrite old model or experiment artifacts. Use immutable, unique
  model and experiment IDs and fail on an ID/path collision.
- Keep models, decision policies, portfolios, and dashboards separate. A model
  emits predictions; a policy turns predictions into signals; a portfolio owns
  cash, holdings, trades, and valuation; a dashboard only presents results.
- Give every model/configuration its own portfolio state. Never reuse cash,
  shares, trades, or snapshots across experiments.
- Do not commit raw large datasets, generated model binaries, secrets, or local
  environment files. Commit only small fixtures or metadata that are necessary
  for tests and explicitly approved for version control.

## Safety and task handoff

- Never describe predictions, signals, trades, portfolio curves, or simulated
  returns as financial advice. User-facing output must state that it is an
  academic simulation and not financial advice.
- At the end of every task, return a concise summary of changed files, tests run
  and their results, assumptions made, and remaining risks or blockers.
