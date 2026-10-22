# News provider configuration

PAISA's first news-ingestion slice supports publisher-provided RSS/Atom feeds
and manually imported CSV files. It does not crawl article pages, bypass access
controls, or silently replace failed providers with fixtures.

## RSS/Atom

Copy `ogdc_sources.example.json` to an untracked local configuration when you
need to change sources. The top-level fields are:

- `schema_version`: optional, defaults to `1.0`, and must be `1.0` when set;
- `timeout_seconds`: optional positive HTTP timeout, default `30` seconds;
- `user_agent`: optional non-empty, single-line HTTP user agent; the adapter's
  PAISA research identifier is used when it is omitted; and
- `sources`: a non-empty array containing at least one enabled source.

Each source supports:

- `source_id`: required stable identifier, unique among enabled sources;
- `name` or `source_name`: required display name;
- `url`: required absolute HTTP(S) feed URL;
- `enabled`: optional boolean, default `true`; disabled entries are not fetched
  or included in the run's configured-source count;
- `language`: optional fallback language evidence when the entry/feed omits it;
- `default_timezone`: optional IANA timezone, `UTC`/`GMT`/`Z`, or UTC offset
  applied only to a timezone-naive clock value; and
- `source_type`: optional `rss` or `atom` hint. PAISA detects the actual format
  from the response and records a warning if it differs from the hint.

PAISA does not turn a date-only value into an invented time. Relative article
links in a feed are resolved against the configured feed URL, while their raw
values remain in the record evidence. HTTP redirects are rejected so a
configured endpoint cannot silently move acquisition to another URL.

Only configure feeds whose publisher permits automated feed access. The
committed example uses Dawn's publisher-provided business feed and records only
the fields present in the feed; PAISA does not follow article links to obtain
additional body text. Review provider terms before every retained acquisition
because permissions and endpoints can change.

```bash
python scripts/run_news_pipeline.py \
  --provider rss \
  --config configs/news/ogdc_sources.example.json \
  --output-dir artifacts/news_ingestion
```

## Manual CSV

Manual input is explicitly labelled `imported`, never `real` or `live`.
Supported columns and aliases are documented in `docs/NEWS_INGESTION.md`.
Keep bulk or licensed datasets outside Git. `--source-name` provides a fallback
publisher name (the input filename stem is the default), and
`--default-language` provides a fallback language tag. Per-row values take
precedence over both defaults.

```bash
python scripts/run_news_pipeline.py \
  --provider manual-csv \
  --input /permitted/path/news.csv \
  --output-dir artifacts/news_ingestion
```

The current adapters require no credentials. A future authenticated provider
must read its secret from an environment variable; credentials must never be
placed in this JSON file, committed, printed, or copied into run artifacts.
