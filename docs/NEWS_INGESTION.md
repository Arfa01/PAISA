# Reproducible news ingestion

PAISA's news foundation captures permitted records, preserves raw evidence,
normalizes one versioned schema, applies deterministic duplicate rules, links
controlled OGDC aliases, persists local SQLite evidence, and publishes an
immutable quality report. It is an academic data-engineering component, not
financial advice.

It does **not** run FinBERT, score sentiment, retrain a stock model, create
signals, change a portfolio, update the API/dashboard, build a knowledge graph,
or perform trading.

## Run an RSS/Atom acquisition

The RSS adapter requests explicitly configured feed URLs and rejects HTTP
redirects. It parses RSS 2.0 and Atom entries, retains the response bytes and
required HTTP metadata, and does not fetch linked article pages. Relative entry
links are resolved against the configured feed URL, while a query-redacted
version of the unresolved link remains in the raw record payload.

Acquisition is fail-closed. By default every DNS result and the connected peer
must be a public IP address; this blocks loopback, private, link-local, and
metadata-service destinations as well as DNS rebinding. A reviewed on-premises
feed may set `allow_private_network: true` on that source, but this weakens the
default network boundary and should not be used for untrusted configuration.
Redirects remain disabled in either mode. URL user information and
credential-like query keys are rejected, all persisted URL query values are
redacted, and PAISA's own HTTP session does not use environment/netrc
credentials or cookies.

The adapter streams and bounds both compressed and decoded content, rejects
DTD/entity declarations, and caps entry count, XML depth/elements, and field
length. Defaults are 2 MiB compressed, 8 MiB decoded, 500 entries, depth 48,
20,000 XML elements, and 1 MiB per field. Connect, read, and total deadlines
default to at most 10, 15, and 30 seconds respectively. The top-level JSON
configuration can override `connect_timeout_seconds`, `read_timeout_seconds`,
`total_timeout_seconds`, `max_compressed_bytes`, `max_decoded_bytes`,
`max_entries`, `max_xml_depth`, `max_xml_elements`, and
`max_field_characters`. Rejections are retained as deterministic, redacted
provider-error evidence. The core requirements install `defusedxml`, which is
part of this XML safety boundary; a partial dependency installation is not a
supported RSS runtime.

For Atom entries, `<published>` is the only publication field. `<updated>` is
retained as source-availability evidence and is never substituted into
`published_at`; a missing `<published>` value remains missing and produces an
explicit warning.

```bash
python scripts/run_news_pipeline.py \
  --provider rss \
  --config configs/news/ogdc_sources.example.json \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

Network/provider checks are intentionally outside the default offline test
suite. One source can fail without fixture substitution; the source error is
retained and the run cannot be represented as a clean `PASS`.

## Run a bounded historical GDELT metadata acquisition

The historical adapter uses only the fixed public GDELT DOC 2.0 ArticleList
endpoint. It issues sequential, explicitly bounded UTC-window queries, rejects
redirects and unsafe destinations, bounds compressed/decoded responses, and
deterministically splits saturated windows or fails closed. It never follows a
returned publisher URL or stores publisher article bodies/images.
DNS resolution, connection, response headers, and payload reading share one
bounded request deadline. In new schema-`1.1` runs, every provider error identifies the exact capture and
attempt, split path, target/request bounds, and sanitized query/request
fingerprints even when no response arrives. Valid `Retry-After` delays are
honoured up to the configured wait ceiling; a larger requested delay fails the
window without an early retry. Captures and terminal errors record parsed
requested and actually applied delay evidence without retaining the raw header.

Review [`HISTORICAL_NEWS_SOURCE_ASSESSMENT.md`](HISTORICAL_NEWS_SOURCE_ASSESSMENT.md)
and edit the query windows before every run:

```bash
python scripts/run_news_pipeline.py \
  --provider gdelt-doc \
  --config configs/news/gdelt_ogdc_history.example.json \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

GDELT `seendate` is retained as first-class `source_observed_at` audit evidence,
never as `published_at` or eligibility evidence under
`news-availability-v1`. Missing, malformed, and out-of-window values quarantine
the historical response. Normalized
`available_at` uses the PAISA fetch time for these records, so a present-day
archive query cannot be backdated into a historical model cutoff. The configuration
must retain the reviewed permission basis, licensing/usage note, and required
GDELT attribution; it contains no credentials.

## Import a permitted CSV

```bash
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /absolute/path/to/permitted_news.csv \
  --licensing-usage-note "Licence and permitted stored fields reviewed on YYYY-MM-DD" \
  --storage-root . \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

Accepted case-insensitive column families include:

| Meaning | Accepted examples |
|---|---|
| record identity | `article_id`, `record_id`, `id`, `guid` |
| URL | `source_url`, `article_url`, `url`, `link` |
| title | `title`, `headline` |
| summary | `summary`, `description`, `abstract` |
| body | `body`, `content`, `article_text`, `text` |
| publication time | `published_at`, `published`, `publication_time`, `published_date`, `pub_date`, `date` |
| original timezone | `published_timezone`, `original_timezone`, `timezone` |
| source availability | `available_at`, `availability_time` |
| source | `source_name`, `publisher`, `publication`, `source` |
| source type | `source_type` |
| language | `language`, `lang` |

Rows may omit body, title, URL, or publication time; quality reporting makes
those gaps explicit. A row with no title, summary, or body is rejected with its
exact reason. Imported rows are always labelled `imported`, never live news.
Use `--source-name` to supply the fallback source name for rows that omit one;
without it, the input filename stem is used. Use `--default-language` to supply
a fallback language tag for rows that omit one. A missing `source_type`
defaults to `manual_csv`. Immutable evidence stores a logical source label,
not the host's absolute input path.

`--licensing-usage-note` is mandatory in the CLI and is copied into sanitized
provider configuration and every raw row. It records the operator's permission
review but does not create or broaden a licence. The programmatic constructor
retains compatibility by defaulting to `LICENSING_REVIEW_REQUIRED`; real-mode
coverage rejects that marker until a reviewed note is supplied.

## CLI result and exit behavior

After valid argument parsing, the command prints one structured summary
containing the run ID, status, counts, logical database label, and logical
artifact directory. It never prints full bodies, host paths, or configuration
secrets. Conventional `--help` and invalid-argument diagnostics are emitted by
`argparse`. `PASS` returns zero. `FAIL` and retained `PARTIAL` results return
non-zero so automation cannot silently accept incomplete acquisition; pass
`--allow-partial` only when a reviewed workflow intentionally accepts
`PARTIAL`. `--run-id` is optional and must be a new safe `NEWS-...` identity.
Existing database or artifact identities are refused.

`--storage-root` defaults to the repository working directory. Artifact and
database paths must resolve inside that trusted root, including through
symlinks. `--allow-external-storage` is an explicit opt-in for a reviewed
external location; even then, summaries retain only non-host-identifying
logical labels.

## Immutable output

Each run is published to `artifacts/news_ingestion/<run_id>/` only after its
staged files are complete:

```text
run_manifest.json
deduplication_context.json
provider_config.json
raw/
  index.json
  capture-...
  record-....json
normalized_articles.csv
article_entity_links.csv
duplicates.csv
provider_errors.json
quality_report.json
final_summary.json
completion.json
```

The pipeline creates `raw/` with mode `0700` and every file beneath it with
mode `0600`, independently of the caller's umask. These permissions protect
locally retained licensed response bytes and unredacted acquisition evidence;
they do not change the checksum-bound contents or confer reuse rights.

`run_manifest.json` contains status, timestamps, provider classification,
counts, limitations, and a sorted checksum/size inventory. Public manifest and
summary markers remain `PENDING` while the expected immutable file set is
sealed in SQLite. The run becomes consumable only after SQLite commits the
matching final status and `completion.json` is atomically published last with
the bound manifest and inventory identities. Readers must require all three:
the completion marker, its matching final manifest, and the matching committed
SQLite row. A manifest or summary that looks final without this proof is not a
completed run.

A temporary `.finalization-journal.json` makes an interrupted finalization
retryable by rerunning the exact run ID without fetching the provider again.
Recovery checks the pre-sealed file names, sizes, and hashes using `lstat`,
rejects missing, extra, changed, symlinked, special, or out-of-root content,
and reconciles artifact records with SQLite before it can commit. It never
rebuilds a new inventory from changed files. Successful finalization removes
the journal. A regular `.run.lock` can remain as an inert concurrency control
after crash recovery; it is excluded from the evidence inventory, and linked or
special lock files are rejected. An unsealed stale reservation can be discarded
and safely retried; any contradictory or already completed destination is an
error.

`provider_config.json` records the sanitized effective provider options and
enabled sources used by the run. Failed acquisitions publish `FAIL` evidence
when the provider returns or raises a controlled failure; they are never marked
`PASS`. Generated artifacts and SQLite files remain ignored by Git.

## Quality policy

Quality-report schema `2.0` distinguishes normalized occurrences, new
canonical articles, matches to existing canonicals, within-run duplicates,
historical duplicates, represented canonicals, and per-rule totals.
`duplicate_count` remains a compatibility alias for the number of duplicate
relationships and must not be interpreted as only within-feed collapses.
`duplicates.csv` records `duplicate_scope` and `canonical_owner_run_id` for
every relationship. The report also includes missing title, publication time,
and body counts, invalid URLs, language/provider counts, daily counts,
OGDC-linked count, publication/availability ranges, rejections with exact
reasons, and provider errors. Coverage artifacts additionally retain the
`source_observed_at` range and row-level provenance with an explicit audit-only,
non-eligibility semantic.

- `FAIL`: no raw records or no normalized records, or persistence failed.
- `WARN`: usable records exist but provider errors, rejected rows, missing
  title/publication time, or invalid URLs require review.
- `PASS`: the minimum checks passed. Missing bodies alone are informational
  because many lawful feeds provide only summaries.

A quality `PASS` is not an assertion that coverage is complete or that a source
is independently correct.

## Reproducibility

Provider response/import bytes, timestamps, checksums, normalized fields,
deduplication decisions, entity links, effective provider options, enabled
source definitions, and quality metrics are retained together.
`deduplication_context.json` freezes the ordered rules, pre-run canonical
index, its digest, owner-run/manifest dependencies, and code/source revision
identity used by that run. Re-normalizing the same captured `RawNewsRecord`
with the same run identity and frozen context produces the same article
identity and deduplication result. A new run remains a new immutable occurrence
and may point to an existing canonical article in SQLite.

Verify or replay a retained pack offline, without network access or mutation,
with:

```bash
python scripts/verify_news_run.py \
  --run-dir artifacts/news_ingestion/<run_id> \
  --database data/news/paisa_news.sqlite3
```

The verifier checks the completion/manifest binding, exact artifact inventory,
raw-capture checksums, frozen deduplication context, normalization, and recorded
deduplication decisions. The database is opened read-only and must carry the
matching final status and immutable seal. Omit `--database` only for portable
artifact replay; that mode reports `ARTIFACT_REPLAY_ONLY` and is not proof that
SQLite committed the run. Live SQLite is never used as a substitute for the
frozen historical context.

Run the offline validation suite with:

```bash
python -m pytest
python -m pip check
git diff --check origin/main...HEAD
```

See `NEWS_DATA_CONTRACT.md` for field/database semantics and
`NEWS_TIME_ALIGNMENT.md` for the leakage boundary.

## Legal and provider limits

- Configure only publisher-provided feeds or datasets whose use is permitted.
- Do not add page scraping, authentication bypass, paywall/CAPTCHA handling, or
  rate-limit circumvention.
- RSS and archive availability, history, metadata completeness, and licensing
  vary by provider/publisher and can change.
- Resource limits make malformed or unexpectedly large feeds fail closed; they
  do not make an untrusted feed or private-network opt-in inherently safe.
- Body text is stored only when the feed or permitted import supplies it.
- Retained URLs and metadata do not transfer republication rights.
