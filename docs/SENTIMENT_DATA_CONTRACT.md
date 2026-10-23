# PAISA sentiment data contract

Schema version: `1.0`

This contract covers immutable article-level classifier output. It does not
define a daily stock feature, prediction, signal, trade, portfolio, or financial
advice.

## Model identity

Every result references one immutable model-revision record containing:

- model repository `ProsusAI/finbert`;
- model revision `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031`;
- tokenizer repository `ProsusAI/finbert` and the same immutable revision;
- recorded CC BY-NC-SA 4.0 weight/model license and non-commercial academic-use
  restriction;
- the upstream index labels `0=positive`, `1=negative`, `2=neutral`;
- the explicit mapping to PAISA's canonical `negative`, `neutral`, `positive`
  fields; and
- a SHA-256 inventory of the reviewed `config.json`, safetensors weights,
  tokenizer configuration, vocabulary, and special-token map, plus one
  `backend_identity_sha256` binding all of that metadata; and
- exact runtime library versions observed during inference.

A floating branch, tag, or `latest` revision is not a valid production input.
The production entry point does not accept an injected classifier. The reviewed
snapshot is hashed before load and again after load; changed, missing,
non-regular, or cache-escaping bytes fail closed. A changed model, tokenizer,
or reviewed file digest creates a new identity and never replaces old output.

## Deterministic inference input

The source is one completed news-ingestion run or its SQLite representation.
For a run-backed database query, PAISA joins that run's article occurrences to
their canonical `normalized_articles` rows and scores each distinct canonical
article at most once. It uses canonical title, summary/body, language, and
timestamps rather than a later duplicate occurrence's text or availability.

Base input policy `title-summary-body-fallback-v1` selects:

1. `title` and `summary` when both are usable;
2. `title` and the permitted `body` excerpt when summary is unavailable;
3. `title` only when no other permitted text is available.

Selected fields are joined deterministically with a blank-line separator.
Whitespace-normalized empty values are unavailable. Missing title is an error.
The configured character bound is applied before tokenization, and the pinned
tokenizer then applies the configured sequence bound. The policy records hashes
and counts, not another copy of the selected full text. The effective persisted
policy ID appends `-chars<limit>-tokens<limit>` so any setting that changes the
selected/model input also changes the idempotency identity.

## Article sentiment record

`article_sentiment.csv` and the corresponding SQLite row carry the stable
contract below. JSON-valued fields use sorted compact JSON in CSV.

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Exactly `1.0`. |
| `sentiment_output_id` | string | Immutable deterministic output identity. |
| `sentiment_run_id` | string | Owning `SENT-...` inference run in the CSV (`inference_run_id` is the equivalent SQLite foreign-key column). |
| `article_id` | string | Canonical normalized article identity. |
| `source_ingestion_run_id` | string | Completed `NEWS-...` run selected as input. |
| `input_policy_id` | string | Versioned deterministic text policy, including settings that affect selected/truncated input identity. |
| `selected_fields` | JSON array | Ordered source fields used, for example `['title','summary']`. |
| `input_text_sha256` | string | SHA-256 of the exact bounded text passed toward tokenization; full text is not persisted again. |
| `character_count` | integer | Character count after the configured deterministic character bound. |
| `original_character_count` | integer | Character count before the configured bound. |
| `tokenizer_token_count` | integer | Tokenizer count before truncation; it may therefore exceed the declared maximum sequence length. |
| `final_token_count` | integer | Non-padding attention-mask count after the sequence limit; positive, no greater than the pre-truncation count, and no greater than the configured limit. |
| `truncation_side` | string | Explicitly `right`; the discarded end is never inferred from a tokenizer default. |
| `truncation_applied` | boolean | True when either configured character or tokenizer sequence truncation changed the selected input. |
| `character_truncation_applied` | boolean | Whether the character bound shortened the selected source fields. |
| `token_truncation_applied` | boolean | Whether the tokenizer count exceeded the maximum sequence length. |
| `maximum_input_characters` | integer | Positive pre-tokenization character limit. |
| `maximum_sequence_length` | integer | Positive tokenizer limit used for this inference. |
| `original_language` | string or null | Language evidence inherited from the canonical article; never inferred silently. |
| `model_repository` | string | `ProsusAI/finbert` for the default configuration. |
| `model_revision` | string | Exact immutable model commit. |
| `tokenizer_revision` | string | Exact immutable tokenizer commit. |
| `class_labels` | JSON array | Exact canonical order `['negative','neutral','positive']`; the original index labels and explicit mapping live in `model_manifest.json` and `sentiment_model_revisions`. |
| `negative_probability` | number | Mapped probability in `[0,1]`. |
| `neutral_probability` | number | Mapped probability in `[0,1]`. |
| `positive_probability` | number | Mapped probability in `[0,1]`. |
| `predicted_label` | enum | `negative`, `neutral`, or `positive`. |
| `predicted_probability` | number | Probability corresponding exactly to `predicted_label`. |
| `published_at` | UTC timestamp or null | Inherited canonical publication evidence. |
| `available_at` | UTC timestamp | Inherited canonical availability; never replaced by inference time. |
| `fetched_at` | UTC timestamp | Inherited canonical acquisition evidence. |
| `inference_timestamp` | UTC timestamp | When model inference occurred; provenance only. |
| `device` | string | Actual selected device such as `cpu`, `cuda`, or `mps`. |
| `batch_size` | integer | Configured batch size for the run. |
| `library_versions` | JSON object | Exact relevant runtime package versions. |
| `warning_codes` | JSON array | Structured non-fatal conditions such as character or token truncation. |
| `error_code` / `error_message` | nullable strings | Both null for a successful output; structured failures live in the error table/file rather than containing source text. |

The three canonical probabilities must be finite, individually bounded, and
sum to one within the documented floating-point tolerance. The predicted label
and probability must agree with their mapped fields. The model's original
tensor positions are never used as implicit PAISA field order.

English (`en` and `en-*`) is the supported language family for this pinned
model. Missing or unsupported language evidence is retained as a structured
skip/error and is not translated or relabelled.

## Immutable run artifacts

One successful, partial, or failed invocation owns:

```text
artifacts/sentiment_inference/<run_id>/
├── run_manifest.json
├── model_manifest.json
├── input_manifest.json
├── article_sentiment.csv
├── inference_errors.json
├── audit_sample.csv
├── quality_report.json
└── final_summary.json
```

- `run_manifest.json` records status, configuration, timing, source run,
  portable storage labels, and the sorted SHA-256/size
  inventory of every other generated file. It excludes only itself to avoid a
  self-referential digest. The exact manifest contract and the inventory of
  all non-summary payloads are independently SHA-256 sealed in the immutable
  SQLite-bound final summary, so recovery does not trust a mutable manifest as
  its own root of integrity.
- `model_manifest.json` records the repository, exact revisions, license,
  intended task, upstream labels, canonical mapping, and runtime identity.
- `input_manifest.json` records the completed source run, ordered canonical
  article references/fingerprint, input-policy identity, and resource bounds;
  it contains no full input body.
- `article_sentiment.csv` contains stable successful article-level records.
- `inference_errors.json` contains sanitized structured skips and failures.
- `audit_sample.csv` is a deterministic small review sample with references,
  permitted titles, probabilities, label, OGDC linkage, timestamps, selected
  fields, character/token limits, before/after counts, explicit right-side
  truncation, flags, and warning codes.
- `quality_report.json` contains coverage/distribution/resource statistics and
  does not claim accuracy from unlabelled data.
- `final_summary.json` is the body-free CLI/reporting summary and carries the
  database-bound payload-inventory and run-manifest-contract seals.

Creation fails on any existing run path or ID; no file is overwritten. Sealing
opens only stable regular files without following symbolic links. Publication
uses an operating-system no-replace rename, so an empty directory appearing in
the check/use gap is not replaced. Artifact paths are relative and portable.
Credentials, full selected article text, and absolute cache paths are
prohibited. Downloaded weights and generated outputs remain ignored by Git.

An interrupted `RUNNING` owner is never a global idempotency source. Recovery
of a hidden staged pack is allowed only after the matching SQLite run is
finalized and both seals validate. Otherwise retrying the exact run ID requires
the explicit `--discard-stale-running` option; that transaction removes only
the named unfinished run and its unfinished output. Finalized evidence is never
eligible for discard.

## Quality report

At minimum the report records:

- articles requested, successfully scored, skipped, and failed;
- negative, neutral, and positive predicted-label counts;
- mean negative, neutral, and positive probabilities;
- truncated-input and title-only counts;
- source language counts;
- OGDC-linked scored count;
- earliest and latest inherited `available_at`;
- model repository and exact revision;
- actual device and processing duration; and
- the statement `Sentiment outputs were generated, but classification accuracy
  was not evaluated in this run.` when no human-labelled ground truth exists.

The report describes output distribution and operational coverage only.
Unlabelled articles cannot establish classification accuracy.

## SQLite persistence and migration

News database migration `5`, `financial_sentiment_foundation`, created the
original sentiment tables. Migration `6`, `sentiment_run_certification`, adds
hash-bound model identity, post-truncation evidence, explicit run-input/output
associations, recovery seals, and database-enforced completion checks:

Migration `7`, `source_observation_provenance`, then adds nullable
`source_observed_at` columns to news canonical/occurrence storage. It does not
alter sentiment ownership or make source observation time eligible input.

| Table | Ownership |
|---|---|
| `sentiment_model_revisions` | Immutable repository/tokenizer revision, license, label mapping, and model metadata. |
| `sentiment_inference_runs` | Run configuration, source ingestion run, status, timing, runtime, portable artifact reference, and summary. |
| `article_sentiment_outputs` | Successful immutable article/model/policy results and inherited time evidence. |
| `sentiment_inference_errors` | Sanitized run/article failures and skips. |
| `sentiment_run_articles` | Ordered input evidence and exactly one terminal `GENERATED`, `REUSED`, `SKIPPED`, or `FAILED` association per completed input. |

Foreign keys preserve source article, source run, model revision, and inference
run identity. A uniqueness constraint makes the same canonical article, exact
model revision, tokenizer revision, and input-policy identity idempotent. A new
model revision creates a separate result rather than updating the old one.
Finalized run rows and their outputs/errors are immutable. Summary schema
`2.0` counts are derived from SQLite rather than trusted from a caller. `PASS`
and `PARTIAL` require nonempty input evidence, no pending input, a terminal
association for every input, database-consistent generated/reused/error counts,
label and truncation counts, and both artifact recovery seals. `PASS` also
forbids retained errors. Direct SQL therefore cannot certify a zero-output or
fabricated-count `PASS`.

The accepted PR #12 ingestion manifest and artifact pack record schema version
`3`. Their historical base-table evidence predates the schema-4 sealing-layer
fingerprints introduced by schema 4. Before sentiment execution, the
working `data/news/paisa_news.sqlite3` also used schema version `3`. A narrow
compatibility repair accepted only its exact known signature, proved that
existing rows satisfied the later time/raw-evidence/status constraints,
transactionally rebuilt the three affected tables without changing their
values, verified integrity, and applied migrations 4 and 5. That retained run
was produced at schema version `5`. Current code then advances a verified
database through schema version `6` sentiment certification to current schema
version `7`, which adds nullable audit-only `source_observed_at`; unknown drift
and invalid legacy rows fail closed, and migration validation occurs only after
the SQLite write lock is held.

Migration does not invent file hashes for a version-5 model row. That legacy
row and every run that references it remain explicitly unverified. The first
current-code execution registers a separate hash-bound model identity—even
when the declared repository and revisions are the same—while preserving the
legacy identifier and its foreign-key history unchanged.

A version-5 `RUNNING` owner and its output remain explicitly discardable and
are not backfilled as completed associations. Migration fails closed if a
version-5 `FAIL` owner still contains output: that globally unique result cannot
be certified for reuse or deleted without rewriting finalized historical
evidence. Preserve such a database for audit and start a clean current-schema
working database rather than weakening immutability.

The ignored, uncommitted pre-migration copy at
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3` was verified at
schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check. This database repair did not create the old pack's
missing completion/deduplication-context artifacts or certify it under the
current news verifier. Preserve the backup and historical artifact pack.

## Consumer contract

The reusable cutoff filter may expose article sentiment records to later code,
but it does not aggregate them. Its sole inclusion rule is:

```text
available_at <= decision_cutoff
```

Later daily aggregation must preserve article counts and missing-news evidence,
declare market-close/non-trading-day rules, and produce a new versioned feature
contract. It must not mutate this article-level record.
