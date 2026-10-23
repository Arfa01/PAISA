# PAISA news data contract

Schema version: `1.1` (news-run read compatibility retained for `1.0`; this is
separate from the coverage-audit schema)

This contract describes acquisition evidence for later NLP research. It does
not contain sentiment, predictions, signals, trades, or financial advice.

New ingestion artifacts bind `news-availability-v1` at the top level of
`provider_config.json`, `run_manifest.json`, and `completion.json`. Coverage
artifacts repeat it for every selected input run and in the top-level quality
report and final summary, so verification can reject a policy mismatch instead
of inferring eligibility rules from the data-schema number.

## Normalized article record

| Field | Type | Contract |
|---|---|---|
| `schema_version` | string | `1.1` for newly generated artifacts; the news-ingestion verifier (`verify_news_run`) retains explicit compatibility for sealed `1.0` news runs. This does not make schema-`1.0` coverage-audit packs acceptable to the current coverage verifier. |
| `article_id` | string | Immutable occurrence identity derived from the run, provider, and raw evidence identity. |
| `provider` | string | Adapter identifier such as `rss`, `gdelt-doc`, or `manual-csv`. |
| `source_name` | string | Publisher/source display name supplied by configuration or input. |
| `source_type` | string | Source class such as `rss`, `atom`, `gdelt_doc_archive`, or `manual_csv`. |
| `source_url` | string or null | Provider-extracted URL after surrounding whitespace is removed. RSS/Atom entry links may be relative and are resolved against the configured feed URL. Persisted RSS URL identities redact every query value. |
| `canonical_url` | string or null | Conservative HTTP(S) canonical form used for deduplication. Invalid or absent URLs remain null and are reported; the source evidence is retained. |
| `title` | string or null | Unicode- and whitespace-normalized plain text. A missing title is not invented. |
| `summary` | string or null | Permitted feed/import summary normalized to plain text. |
| `body` | string or null | Body only when it was present in the permitted feed/import. PAISA does not fetch linked article pages. |
| `published_at` | UTC timestamp or null | Complete source publication time converted to UTC. Missing, invalid, date-only, or unjustifiably timezone-naive values remain null. |
| `published_timezone` | string or null | Original offset/name evidence from the record or the configured source timezone applied to a naive clock time. |
| `source_observed_at` | UTC timestamp or null | When an upstream index/provider reports observing the item, such as GDELT `seendate`. It is audit provenance only and is never cutoff eligibility under availability policy `news-availability-v1`; using it for eligibility requires a new reviewed policy version. Schema `1.0` records must omit this field; they cannot carry it with either a value or an explicit null. |
| `available_at` | UTC timestamp | Earliest defensible PAISA availability under the rule in `NEWS_TIME_ALIGNMENT.md`. |
| `fetched_at` | UTC timestamp | When PAISA acquired or imported the record. |
| `language` | string or null | Lowercase BCP-47-like source evidence, when supplied; no language detection is performed. |
| `mentioned_tickers` | JSON list | Deterministic linked tickers. This slice can add `OGDC` only. |
| `mentioned_companies` | JSON list | Deterministic linked company names. |
| `mentioned_sectors` | JSON list | Reserved for later reviewed entity data; no sector is inferred in this slice. |
| `content_sha256` | string | SHA-256 of deterministic normalized text JSON. A title-only record is additionally scoped by stable provider/source and publication-or-record context, preventing unrelated generic headlines from sharing a global hash. |
| `duplicate_of` | string or null | Canonical article identity when this occurrence is a duplicate. |
| `ingestion_run_id` | string | Immutable `NEWS-...` run identity. |
| `raw_artifact_reference` | string | Safe POSIX path relative to the run directory. |
| `normalization_warnings` | JSON list | Explicit missing/invalid evidence; values are never silently fabricated. |

Every normalized article's `schema_version` must equal the schema declared by
its owning ingestion run. Verification and SQLite insertion fail closed on a
mismatch; a record cannot opt itself into fields that its run schema does not
own. The CSV also carries `raw_record_id` and `data_class` (`real`, `imported`, or
`fixture`) so every normalized occurrence remains traceable and test data can
never be mistaken for acquired news.

## Raw acquisition contract

Every provider result records the provider ID, configured source, acquisition
time, HTTP status when applicable, response content type, response SHA-256,
warnings/errors, and data class. Response bytes or imported CSV bytes are
retained below the immutable run's `raw/` directory. Each extracted row also
has a JSON evidence file; normalization never replaces that evidence. The raw
directory is owner-only (`0700`) and every retained capture/index/record file
is owner-readable and owner-writable only (`0600`); checksums bind content,
while these filesystem modes limit accidental local disclosure.

GDELT provider errors carry a structured request context: capture/attempt ID,
split path, target and request bounds, attempt number, and SHA-256 fingerprints
of the sanitized query and effective request parameters. Retry responses also
record parsed requested and actually applied wait durations; raw
`Retry-After` text is not persisted.

Historical API/manual-import raw evidence also carries the reviewed provider
subtype and licensing/usage note. GDELT evidence carries the permission basis,
required attribution, exact bounded query/window, and its original `seendate`;
manual rows carry the CLI-supplied import note. These fields document the
reviewed use and provenance but do not grant rights beyond the underlying
provider or publisher terms.

`real` means bytes obtained from a configured network provider during that run,
including publisher RSS/Atom feeds and the metadata-only GDELT API. It describes
the acquisition path, not a licence finding. `imported` means PAISA read a
user-supplied permitted dataset. `fixture` is reserved for deterministic
automated tests and is not accepted by the CLI.

## Canonical URL and content identity

URL canonicalization is deliberately conservative:

- accept only HTTP(S), with no embedded username/password;
- lowercase scheme and IDNA host;
- remove default ports, fragments, duplicate/trailing path separators, and
  known tracking parameters;
- retain and deterministically sort semantic query parameters.

Content hashing first converts permitted HTML fragments to normalized plain
text, applies Unicode NFKC and whitespace normalization, then hashes stable
UTF-8 JSON containing title, summary, and body. When both summary and body are
absent, the hash payload also includes a digest of stable provider/source
identity and publication or record context. Such records carry the
`content_hash_scoped_to_source_context` warning; their hash is not a
source-independent content identity.

## Deduplication contract

Rules are evaluated in this order, and the first match wins:

1. exact non-null canonical URL;
2. exact normalized content SHA-256;
3. exact normalized title and source with publication timestamps in the same
   UTC minute. This rule is disabled when publication time is missing.

For each rule, both finalized persisted canonicals and canonicals already seen
within the current run are considered before the next rule is attempted. This
is global rule precedence: a same-run URL match therefore outranks a historical
hash match. Canonicals owned by a `RUNNING` or failed run are never candidates.

The first-seen canonical record is retained. Every later raw and normalized
occurrence is also retained, with `duplicate_of`, the exact rule,
`duplicate_scope` (`within_run` or `historical`), and
`canonical_owner_run_id` written to `duplicates.csv` and SQLite. There is no
fuzzy similarity merge.

Quality-report schema `2.0` reports `normalized_occurrence_count`,
`new_canonical_article_count`, `matched_existing_canonical_count`,
`within_run_duplicate_count`, `historical_duplicate_count`,
`represented_canonical_count`, and `deduplication_rule_counts`.
`duplicate_count` is retained only as a compatibility alias for the number of
relationship rows.

## Deterministic OGDC linking

The controlled aliases are `OGDC`, `OGDCL`, and the `Oil and/& Gas Development
Company` names with and without `Limited`. Matching is case-insensitive and
boundary-aware, searches title, summary, then body, and records the alias,
method (`exact` or `normalized_exact`), location, and confidence. It does not
perform NER, fuzzy matching, or forced linking.

Each deterministic relationship is persisted in checksum-bound
`article_entity_links.csv`. Coverage consumers reconcile every normalized
article's `mentioned_tickers` exactly against those rows, reject incomplete,
duplicate, or unknown-article link evidence, and retain the exact link rows in
their OGDC-specific audit output rather than reconstructing an explanation
from the normalized ticker alone.

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

Database schema version 7 is advanced one version at a time in transactions.
Version 7 adds nullable `source_observed_at` columns to canonical and occurrence
storage; legacy rows migrate with null values rather than inferring observation
time from a different timestamp. Every open
verifies `user_version`, the ordered migration ledger, and the expected table,
index, and trigger definitions; malformed, unversioned nonempty, or newer
databases are refused rather than stamped as current. Only one `RUNNING` run is
allowed, so canonical matching and insertion are serialized. Deduplication
uses finalized canonical owners, canonical rows cannot themselves point to a
duplicate, and canonical evidence becomes append-only once a finalized run
references it.

Foreign keys, run/article identities, canonical URL/hash/title identities,
occurrence-to-relationship bijection, quality counts derived from SQLite, and
deterministic link identities are database-enforced. A successful or partial
run must first be sealed to one exact artifact inventory and manifest/
completion identity. Direct inserts cannot start in a final status. SQLite is
a local development implementation, not a production claim; the field
meanings are independent of the future database engine.

## Completed-run and replay contract

`run_manifest.json` and `final_summary.json` remain pending until SQLite has
committed the matching final status. `completion.json` is published last and
binds the final manifest and the exact sealed evidence inventory. A consumer
must reject a run unless the completion marker, final manifest, inventory, and
SQLite seal all agree. Recovery validates the original inventory with `lstat`,
root containment, hashes, and artifact/SQLite reconciliation; it cannot bless
a rebuilt or modified file set.

`deduplication_context.json` is part of that sealed inventory. It contains the
ordered algorithm version, frozen pre-run canonical index and digest,
canonical-owner run and manifest identities, and code/source revision identity.
This makes historical deduplication replay independent of a later-mutated live
database. Use `python scripts/verify_news_run.py --run-dir <run-dir> --database
<database>` for read-only, database-bound completion verification. Omitting
`--database` performs artifact-only offline replay and is deliberately labelled
as non-authoritative completion evidence.

## Consumer boundary for future FinBERT work

A later sentiment stage may consume the normalized article record and raw
checksum reference. It must preserve `article_id`, `content_sha256`, model
revision, tokenization settings, and its own output identity. It may use a
record for a feature cutoff only when `available_at <= decision_cutoff`. This
branch deliberately emits no sentiment field or score.
