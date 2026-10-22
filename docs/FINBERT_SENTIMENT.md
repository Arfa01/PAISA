# PAISA FinBERT sentiment inference

PAISA's sentiment stage assigns reproducible article-level financial sentiment
probabilities to normalized news evidence. It is a separate component from news
acquisition, daily feature aggregation, stock prediction, decision policies,
portfolio simulation, the API, and the dashboard.

The output is an academic research artifact. It is not financial advice, a
buy/sell signal, or evidence that sentiment predicts OGDC returns.

## Pinned model and permitted use

The only default model is the following immutable Hugging Face snapshot:

| Setting | Pinned value |
|---|---|
| Repository | [`ProsusAI/finbert`](https://huggingface.co/ProsusAI/finbert/tree/0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031) |
| Model revision | `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` |
| Tokenizer revision | `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` |
| Intended task | Three-class financial-text sentiment classification |
| Weight/model license recorded for this snapshot | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |
| PAISA use | Non-commercial academic research only |

The non-commercial, attribution, and share-alike conditions are material. This
configuration must not be represented as licensed for commercial deployment.
A different model is never substituted silently when the pinned snapshot is
unavailable. Anyone changing the repository or revision must review its license
and intended use, add a new immutable model-revision record, and review the
result as a distinct experiment.

The upstream model's original class order is not alphabetical:

| Original class index | Upstream label | PAISA field |
|---:|---|---|
| `0` | `positive` | `positive_probability` |
| `1` | `negative` | `negative_probability` |
| `2` | `neutral` | `neutral_probability` |

PAISA validates and maps those labels explicitly. It persists its canonical
probability order as `negative`, `neutral`, `positive`; it never assumes that
the tensor index has that order. The predicted label is the canonical label
with the greatest mapped probability, and the predicted probability is that
same value.

The production loader also requires the reviewed SHA-256 values for
`config.json`, `model.safetensors`, `tokenizer_config.json`, `vocab.txt`, and
`special_tokens_map.json`. It verifies the resolved regular files, copies only
those five files into a private clean directory, and loads both model and
tokenizer from that directory so extra cache files (for example,
`added_tokens.json`) cannot affect inference. It verifies the isolated bytes
again after construction and binds the full inventory into the model-revision
ID and sealed run evidence.

## Installation

The transformer runtime is deliberately separate from the default PAISA
installation because it downloads large model dependencies and weights:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-sentiment.txt
python -m pip install -e .
```

Downloaded weights are runtime data and must remain outside Git. The CLI uses a
configurable cache and records only a portable cache label, never a credential
or a host-specific absolute cache path, in public artifacts.

## Input boundary and deterministic text policy

Inference reads canonical normalized articles represented by one completed
news-ingestion run in SQLite. It does not fetch URLs, reread publisher pages, or
accept arbitrary free text. The `title-summary-body-fallback-v1` base policy selects
text in this exact order:

1. normalized title plus normalized summary, separated by one blank line;
2. title plus the permitted body excerpt when summary is unavailable;
3. title only when neither summary nor permitted body is available.

An article without a usable title is rejected. English language tags (`en` and
`en-*`) are supported by this model configuration; missing or unsupported
language evidence is recorded as a structured skip/error rather than guessed.
The policy applies the configured maximum character limit deterministically,
then the pinned tokenizer applies the configured maximum token length. Selected
field names, the input SHA-256, character counts, pre-truncation token count,
post-limit non-padding attention-mask count, explicit `right` truncation side,
limits, flags, and warnings are persisted. The selected text itself is not duplicated in
the sentiment artifacts or SQLite. The persisted effective policy identity also
contains both bounds—for example,
`title-summary-body-fallback-v1-chars4000-tokens512`—so changing either limit
creates a distinct idempotency key.

See [SENTIMENT_DATA_CONTRACT.md](SENTIMENT_DATA_CONTRACT.md) for the complete
record and artifact schema.

## Running inference

First create or restore a completed news-ingestion run and its SQLite evidence.
Then inspect the supported resource controls:

```bash
python scripts/run_sentiment_inference.py --help
```

The accepted PR #12 input run is:

```text
NEWS-20261022-131242-5ddffd2d
```

Run the pinned model on CPU with bounded input and batching:

```bash
python scripts/run_sentiment_inference.py \
  --source-run-id NEWS-20261022-131242-5ddffd2d \
  --database data/news/paisa_news.sqlite3 \
  --output-dir artifacts/sentiment_inference \
  --model-cache data/sentiment_model_cache \
  --device cpu \
  --batch-size 8 \
  --max-input-characters 4000 \
  --max-token-length 512 \
  --max-articles 500
```

CPU is the default. CUDA and Apple Metal may be selected only when the runtime
reports them available; the actual selected device is recorded in every run.
Batch size, maximum input characters, maximum tokenizer length, maximum article
count, and the available timeout/watchdog setting are bounded command options.
Inputs are iterated in stable batches rather than loaded without limit.

An existing `SENT-...` destination is an error and is never overwritten. A
model download/load failure returns non-zero, records a structured failure, and
does not fall back to another classifier. The command prints only a body-free
structured summary; it never prints article bodies or selected full input text.

If a process was interrupted before finalization, its `RUNNING` output is not
eligible for reuse. After confirming that no process still owns it, retry the
same ID explicitly:

```bash
python scripts/run_sentiment_inference.py \
  --source-run-id NEWS-20261022-131242-5ddffd2d \
  --run-id SENT-YYYYMMDD-HHMMSS-xxxxxxxx \
  --discard-stale-running
```

The discard is an atomic SQLite operation scoped to that exact unfinished ID;
it cannot delete a finalized run. A different run ID fails rather than treating
an unfinished owner's rows as trusted idempotent output.

## Run artifacts

Each invocation publishes an immutable directory:

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

The run manifest inventories the other generated files by relative path, byte
size, and SHA-256. The manifest cannot hash itself without a recursive digest;
that sole exclusion is explicit. Instead, the SQLite-bound final summary seals
both the deterministic payload inventory and every non-inventory run-manifest
field. This prevents a private staged file and its mutable manifest checksum
from being changed together during recovery. Handled failures retain `FAIL`
evidence and never publish a false `PASS`. SQLite reaches a final status before
the staged directory is atomically exposed; an interruption before database
finalization therefore cannot expose a completed `PASS` directory. If
publication itself fails after the SQLite commit, the checksum-bound pack
remains at the private, deterministic `.SENT-....staging` path. Repeating the
command with that exact run ID validates the two database-bound seals, full
file inventory, database identity, status, completion timestamp, provenance,
configuration, input/output associations, and safety flags, then publishes
without rerunning inference.
Any mismatch refuses publication. The audit sample is a deterministic, small
selection of article references, permitted titles, probabilities, labels,
OGDC linkage, timing, selected fields, limits, before/after counts, explicit
truncation side, flags, and warnings. It is an inspection aid, not labelled
ground truth.

The quality report includes requested, scored, skipped, and failed counts;
canonical label counts; mean probabilities; truncation and title-only counts;
language counts; OGDC-linked scored count; availability range; pinned model and
revision; device; and duration. In the absence of a human-labelled evaluation
set it states exactly:

> Sentiment outputs were generated, but classification accuracy was not
> evaluated in this run.

## Timing and leakage boundary

Every sentiment observation inherits `published_at`, `available_at`, and
`fetched_at` from its canonical source article. `inference_timestamp` records
when the classifier ran; it never replaces or moves `available_at`.

A later feature builder may consider an observation only when:

```text
article.available_at <= stock_decision_cutoff
```

The reusable cutoff filter enforces that predicate but performs no daily
aggregation. See [SENTIMENT_TIME_ALIGNMENT.md](SENTIMENT_TIME_ALIGNMENT.md).

## Accepted input and real-run status

The accepted news run represents 30 distinct canonical articles, including one
OGDC-linked article, from one short Dawn Business RSS snapshot. All 30 retained
canonical inputs carry `en-us` language evidence. The accepted ingestion
manifest and artifact pack record the schema version at ingestion completion:
version `3`. Those historical artifacts are unchanged and still predate the
schema-4 sealing evidence, the schema-5 sentiment tables, and the current
schema-6 certification contract.

Before sentiment execution, the working `data/news/paisa_news.sqlite3` also
used schema version `3`. The migration recognized only that exact legacy
signature, verified that every retained row met the later constraints,
transactionally rebuilt the three affected base tables, and advanced through
schema 4 to schema 5. The retained reference run below was produced at schema
version `5`. Current code advances a verified database to schema version `6`
for model-file identity, truncation evidence, run associations, and certified
finalization; migration preconditions are evaluated while holding the write
lock. Any other drift or invalid row is rejected. An ignored, uncommitted
pre-migration copy remains at
`data/news/backups/paisa_news_schema3_before_sentiment.sqlite3`; it was verified
at schema version `3` with `31` canonical articles, `150` article occurrences,
and an `ok` integrity check.

The database compatibility repair did not recreate missing artifact evidence
or retroactively certify the accepted ingestion pack under the current
verifier. The working database, backup, model cache, and generated artifacts
remain outside Git.

The retained real CPU run completed with the following measured evidence. Its
generated database state, model cache, and artifact directory remain outside
Git and are not committed.

| Measurement | Retained value |
|---|---|
| Sentiment run | `SENT-20261022-172211-real0002` |
| Source news run | `NEWS-20261022-131242-5ddffd2d` |
| Status | `PASS` |
| Requested / scored / unique canonical | `30 / 30 / 30` |
| Skipped / failed / any articles failed | `0 / 0 / false` |
| OGDC-linked articles scored | `1` |
| Negative / neutral / positive labels | `11 / 9 / 10` |
| Truncated inputs / title-only inputs | `17 / 0` |
| Language evidence | `en-us=30` |
| Canonical availability range | `2026-10-22T11:52:32Z` to `2026-10-22T13:12:43.088883Z` |
| Model and tokenizer revision | `0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031` |
| Device / duration | `cpu / 7.116756208240986 seconds` |
| Runtime | Python `3.13.7`; PyTorch `2.10.0`; Transformers `5.20.2`; Tokenizers `0.23.3`; Hugging Face Hub `1.32.0`; Safetensors `0.8.0` |
| Database schema for the retained reference run | `5` (current code target: `6`) |
| Artifact directory | `artifacts/sentiment_inference/SENT-20261022-172211-real0002` |

Sentiment outputs were generated, but classification accuracy was not evaluated in this run.

The retained sentiment evidence covers October 19--22, 2026, while the retained
OGDC price evidence ends on September 15, 2026: there are **zero overlapping
dates**. No joined price-plus-sentiment observation can currently be
constructed, so this run cannot support a combined experiment or portfolio
conclusion. An overlapping historical news corpus is required before any
price-plus-sentiment research stage can run.

This short, single-publisher snapshot is not sufficient for stock-model
integration or a price-only versus price-plus-sentiment comparison. It cannot
measure classifier accuracy without human labels, temporal coverage, source
diversity, predictive value, or robustness across regimes.

## Explicitly out of scope

This foundation does not implement daily aggregation, stock-model retraining,
price-plus-sentiment comparison, portfolio-policy changes, knowledge graphs,
event propagation, SHAP, dashboard changes, reinforcement learning, or
financial advice. A later daily feature stage will consume these immutable
article records only after defining trading-session attribution and applying
the availability cutoff before aggregation.
