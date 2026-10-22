# PAISA news data contract

Schema version: `1.0`

This contract describes acquisition evidence for later NLP research. It does
not contain sentiment, predictions, signals, trades, or financial advice.

## Normalized article record

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | Exactly `1.0` for this implementation. |
| `article_id` | string | Immutable occurrence identity derived from the run, provider, and raw evidence identity. |
| `provider` | string | Adapter identifier such as `rss` or `manual-csv`. |
| `source_name` | string | Publisher/source display name supplied by configuration or input. |
| `source_type` | string | Source class such as `rss`, `atom`, or `manual_csv`. |
| `source_url` | string or null | Provider-extracted URL after surrounding whitespace is removed. RSS/Atom entry links may be relative and are resolved against the configured feed URL; the unresolved link remains in the raw provider payload. |
| `canonical_url` | string or null | Conservative HTTP(S) canonical form used for deduplication. Invalid or absent URLs remain null and are reported; the source evidence is retained. |
| `title` | string or null | Unicode- and whitespace-normalized plain text. A missing title is not invented. |
| `summary` | string or null | Permitted feed/import summary normalized to plain text. |
| `body` | string or null | Body only when it was present in the permitted feed/import. PAISA does not fetch linked article pages. |
| `published_at` | UTC timestamp or null | Complete source publication time converted to UTC. Missing, invalid, date-only, or unjustifiably timezone-naive values remain null. |
| `published_timezone` | string or null | Original offset/name evidence from the record or the configured source timezone applied to a naive clock time. |
| `available_at` | UTC timestamp | Earliest defensible PAISA availability under the rule in `NEWS_TIME_ALIGNMENT.md`. |
| `fetched_at` | UTC timestamp | When PAISA acquired or imported the record. |
| `language` | string or null | Lowercase BCP-47-like source evidence, when supplied; no language detection is performed. |
| `mentioned_tickers` | JSON list | Deterministic linked tickers. This slice can add `OGDC` only. |
| `mentioned_companies` | JSON list | Deterministic linked company names. |
| `mentioned_sectors` | JSON list | Reserved for later reviewed entity data; no sector is inferred in this slice. |
| `content_sha256` | string | SHA-256 of deterministic normalized title, summary, and body JSON. |
| `duplicate_of` | string or null | Canonical article identity when this occurrence is a duplicate. |
| `ingestion_run_id` | string | Immutable `NEWS-...` run identity. |
| `raw_artifact_reference` | string | Safe POSIX path relative to the run directory. |
| `normalization_warnings` | JSON list | Explicit missing/invalid evidence; values are never silently fabricated. |

The CSV also carries `raw_record_id` and `data_class` (`real`, `imported`, or
`fixture`) so every normalized occurrence remains traceable and test data can
never be mistaken for acquired news.

## Raw acquisition contract

Every provider result records the provider ID, configured source, acquisition
time, HTTP status when applicable, response content type, response SHA-256,
warnings/errors, and data class. Response bytes or imported CSV bytes are
retained below the immutable run's `raw/` directory. Each extracted row also
has a JSON evidence file; normalization never replaces that evidence.

`real` means bytes obtained from a configured publisher feed during that run.
`imported` means PAISA read a user-supplied permitted dataset. `fixture` is
reserved for deterministic automated tests and is not accepted by the CLI.

## Canonical URL and content identity

URL canonicalization is deliberately conservative:

- accept only HTTP(S), with no embedded username/password;
- lowercase scheme and IDNA host;
- remove default ports, fragments, duplicate/trailing path separators, and
  known tracking parameters;
- retain and deterministically sort semantic query parameters.

Content hashing first converts permitted HTML fragments to normalized plain
text, applies Unicode NFKC and whitespace normalization, then hashes stable
UTF-8 JSON containing title, summary, and body.

## Deduplication contract

Rules are evaluated in this order, and the first match wins:

1. exact non-null canonical URL;
2. exact normalized content SHA-256;
3. exact normalized title and source with publication timestamps in the same
   UTC minute. This rule is disabled when publication time is missing.

The first-seen canonical record is retained. Every later raw and normalized
occurrence is also retained, with `duplicate_of` and the exact rule written to
`duplicates.csv` and SQLite. There is no fuzzy similarity merge.

## Deterministic OGDC linking

The controlled aliases are `OGDC`, `OGDCL`, and the `Oil and/& Gas Development
Company` names with and without `Limited`. Matching is case-insensitive and
boundary-aware, searches title, summary, then body, and records the alias,
method (`exact` or `normalized_exact`), location, and confidence. It does not
perform NER, fuzzy matching, or forced linking.

## SQLite persistence

The development database is configurable and defaults to
`data/news/paisa_news.sqlite3`. It is generated and ignored by Git. The schema
separates canonical articles from per-run occurrences so repeated acquisition
does not erase evidence or weaken canonical URL uniqueness. It stores:

- ingestion runs;
- raw source records;
- canonical normalized articles and article-ingestion occurrences;
- article-company links;
- deduplication relationships;
- provider errors; and
- quality reports.

Foreign keys, run/article identities, canonical URL uniqueness, content-hash
indexes, and deterministic link identities are database-enforced. SQLite is a
local development implementation, not a production claim; the field meanings
are independent of the future database engine.

## Consumer boundary for future FinBERT work

A later sentiment stage may consume the normalized article record and raw
checksum reference. It must preserve `article_id`, `content_sha256`, model
revision, tokenization settings, and its own output identity. It may use a
record for a feature cutoff only when `available_at <= decision_cutoff`. This
branch deliberately emits no sentiment field or score.
