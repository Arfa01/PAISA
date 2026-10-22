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
links are resolved against the configured feed URL, while the unresolved link
remains in the raw record payload.

For Atom entries, `<published>` is the only publication field. `<updated>` is
retained as source-availability evidence and is never substituted into
`published_at`; a missing `<published>` value remains missing and produces an
explicit warning.

```bash
python scripts/run_news_pipeline.py \
  --provider rss \
  --config configs/news/ogdc_sources.example.json \
  --output-dir artifacts/news_ingestion \
  --database data/news/paisa_news.sqlite3
```

Network/provider checks are intentionally outside the default offline test
suite. One source can fail without fixture substitution; the source error is
retained and the run cannot be represented as a clean `PASS`.

## Import a permitted CSV

```bash
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /absolute/path/to/permitted_news.csv \
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
defaults to `manual_csv`.

## CLI result and exit behavior

After valid argument parsing, the command prints one structured summary
containing the run ID, status, counts, database location, and artifact
directory. It never prints full bodies or configuration secrets. Conventional
`--help` and invalid-argument diagnostics are emitted by `argparse`. `FAIL`
returns non-zero; `PASS` and retained `PARTIAL` results return zero. `--run-id`
is optional and must be a new safe `NEWS-...` identity. Existing database or
artifact identities are refused.

## Immutable output

Each run is published to `artifacts/news_ingestion/<run_id>/` only after its
staged files are complete:

```text
run_manifest.json
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
```

`run_manifest.json` contains status, timestamps, provider classification,
counts, limitations, and a sorted checksum/size inventory. Final summary and
manifest markers are atomically replaced while the SQLite row is still
`RUNNING`; SQLite is finalized only after both markers exist, with no artifact
writes required after a successful database finalization. A temporary
`.finalization-journal.json` makes an interrupted marker write retryable by
rerunning the exact run ID without fetching the provider again. Successful
finalization removes the journal. Any other destination collision is an error.
`provider_config.json` records the sanitized effective provider options and
enabled sources used by the run. Failed acquisitions publish `FAIL` evidence
when the provider returns or raises a controlled failure; they are never marked
`PASS`. Generated artifacts and SQLite files remain ignored by Git.

## Quality policy

The report includes raw and normalized totals, duplicate count, missing title,
publication time, and body counts, invalid URLs, language/provider counts,
daily counts, OGDC-linked count, publication/availability ranges, rejections
with exact reasons, and provider errors.

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
source definitions, and quality metrics are retained together. Re-normalizing
the same captured `RawNewsRecord` with the same run identity produces the same
article identity and content hash. A new run remains a new immutable occurrence
and may point to an existing canonical article in SQLite.

Run the offline validation suite with:

```bash
python -m pytest
python -m pip check
git diff --check
```

See `NEWS_DATA_CONTRACT.md` for field/database semantics and
`NEWS_TIME_ALIGNMENT.md` for the leakage boundary.

## Legal and provider limits

- Configure only publisher-provided feeds or datasets whose use is permitted.
- Do not add page scraping, authentication bypass, paywall/CAPTCHA handling, or
  rate-limit circumvention.
- RSS availability, history, metadata completeness, and licensing vary by
  publisher and can change.
- The current RSS adapter has a request timeout but no explicit response-byte
  cap before buffering; configure only reviewed feed endpoints.
- Body text is stored only when the feed or permitted import supplies it.
- Retained URLs and metadata do not transfer republication rights.
