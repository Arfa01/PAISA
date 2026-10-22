# Sentiment time alignment and leakage prevention

Article sentiment adds an inference event to the news timeline; it does not
change when the underlying information became available.

All timestamps used for comparison are timezone-aware UTC values.

| Timestamp | Owner | Meaning |
|---|---|---|
| `published_at` | Canonical news article | Source-supplied publication evidence; nullable when no defensible complete time exists. |
| `fetched_at` | Canonical news article | When PAISA first acquired the retained canonical evidence. |
| `available_at` | Canonical news article | Earliest defensible time this PAISA evidence could be used, calculated by the news contract. |
| `inference_timestamp` | Sentiment run | When the pinned classifier processed the selected input. It is provenance, not source availability. |
| `decision_cutoff` | Future stock-feature consumer | Latest information time allowed for one decision or feature row. |

## Inheritance rule

Each article sentiment result copies `published_at`, `fetched_at`, and
`available_at` from the canonical normalized article identified by
`article_id`. It must not replace them with timestamps from a later duplicate
occurrence or with the inference timestamp.

For example, if a canonical article was first available at 11:52 and appeared
again in a feed fetched at 13:12, sentiment for that canonical article retains
11:52. The 13:12 occurrence remains ingestion evidence but does not move the
canonical's availability forward. Conversely, a materially corrected article
at the same canonical URL does not currently become a new canonical record.
Canonical-URL deduplication keeps it as duplicate occurrence evidence tied to
the earlier canonical article; it is not independently scored and cannot enter
an earlier or later decision as a correction.

Missing or invalid source timing is never repaired from model execution time.
If required `available_at`/`fetched_at` evidence is missing, invalid, or
timezone-naive at the sentiment boundary, the article is rejected or skipped
with a structured error. A nullable `published_at` stays null.

## Decision-cutoff filter

The reusable article-level filter applies exactly:

```text
include sentiment only if sentiment.available_at <= decision_cutoff
```

The boundary is inclusive. Filtering occurs before any future daily or ticker
aggregation. Ordering, inference timestamp, filesystem modification time, and
the time at which a database query is executed cannot make a later observation
eligible for an earlier cutoff.

Examples:

| Article availability | Inference time | Decision cutoff | Result |
|---|---|---|---|
| 09:10 UTC | 14:00 UTC | 09:05 UTC | Excluded. Inference cannot backdate the article. |
| 09:10 UTC | next day | 09:10 UTC | Included at the exact inclusive boundary. |
| Hypothetical independently versioned correction available at 13:12 UTC | 13:13 UTC | 12:00 UTC | Excluded. A future correction contract must preserve its own `available_at`. |
| Missing/naive availability | any | any | Rejected/skipped; no timestamp is invented. |

Same-URL corrections are currently unsupported. Because canonical URL has
deduplication priority, changed content later observed at the same URL is
ignored as an independent sentiment input: it receives neither a versioned
correction identity nor its own sentiment result. This is conservative against
look-ahead, but it can leave the retained canonical content stale. A future
correction contract must create versioned correction identities with their own
`available_at` before corrected content can be independently scored. Re-running
inference does not alter an old result or retroactively introduce a correction.

## Source-run versus canonical time

`source_ingestion_run_id` records which completed `NEWS-...` run selected the
article for inference. It is a lineage reference, not an availability clock.
When that run contains duplicates, PAISA resolves them to distinct canonical
article rows and inherits those canonical rows' time evidence. This distinction
is necessary for the accepted run `NEWS-20261022-131242-5ddffd2d`: 29 of its 30
canonical inputs were first retained by an earlier completed acquisition, while
one was first retained by the accepted run.

## What this stage does not decide

The sentiment foundation does not assign a trading date, align after-close
news, shift weekend/holiday articles, define lookback windows, combine multiple
articles, or join prices. Those choices belong to a later versioned daily
aggregation contract. That stage must:

1. declare the exchange calendar and decision cutoff for every feature row;
2. apply the article cutoff before aggregation;
3. retain article/source counts and missing-news indicators;
4. keep `feature_date`, `target_date`, `signal_date`, and `execution_date`
   distinct; and
5. fit any learned weighting or imputation on training data only.

No stock-model integration is justified by one short RSS snapshot. The
sentiment output remains an article-level academic research artifact and is not
financial advice.
