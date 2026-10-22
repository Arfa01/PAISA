# News time alignment and leakage prevention

News timestamps answer different questions and must never be collapsed into one
date.

| Timestamp | Meaning |
|---|---|
| `published_at` | When the source says it published the record. It is null if a complete defensible time is unavailable. |
| `fetched_at` | When PAISA obtained the feed response or imported the file. |
| source `available_at` evidence | Optional time supplied by a provider for when that record became available. |
| normalized `available_at` | Earliest defensible time at which this PAISA instance could have used the captured record. |

All comparisons use timezone-aware UTC values. `published_timezone` retains the
original offset/name evidence so UTC conversion is auditable.

## Availability rule

For a successfully normalized record:

```text
available_at = max(fetched_at, valid published_at, valid source availability)
```

Only present, valid candidates participate. This conservative maximum is
intentional: PAISA cannot use a record before acquiring it, and a source record
dated in the future must not become available merely because it was fetched
early. Missing publication time falls back only to the real acquisition time;
it is not copied into `published_at`.

## Timestamp parsing rules

- ISO-8601 and RFC-2822 clock timestamps with offsets are converted to UTC.
- A naive clock timestamp is accepted only when its row or reviewed source
  configuration supplies timezone evidence.
- A date-only value is not treated as midnight; that would invent a time.
- Missing, invalid, date-only, or timezone-ambiguous publication values remain
  null and add a normalization warning.
- For Atom, `<published>` is publication evidence. `<updated>` is retained as
  source-availability evidence only; it is never copied into `published_at`
  when `<published>` is absent. That case records the warning
  `atom_published_missing_used_updated_as_availability`.
- The raw string and original timezone evidence remain in the raw artifact.

## Feature-cutoff rule

Any future daily sentiment or price-plus-sentiment feature builder must apply:

```text
include article only if article.available_at <= decision_cutoff
```

It must apply the comparison before aggregation and must not use
`target_date`, later retrieval results, or validation/test statistics to repair
missing news. Trading-calendar, after-close, weekend, and lookback-window rules
belong to the later aggregation contract; this ingestion branch does not guess
them.

Examples:

| Source evidence | Fetch | Result |
|---|---|---|
| Published 09:00 UTC, fetched 09:10 UTC | 09:10 | `available_at` 09:10; unavailable to a 09:05 decision. |
| Publication missing, fetched 12:00 UTC | 12:00 | `published_at` null, `available_at` 12:00, missing-time warning. |
| Source says 23 Oct 09:00 UTC but fetched 22 Oct 12:00 UTC | 22 Oct 12:00 | `available_at` 23 Oct 09:00; it cannot enter a 22 Oct feature. |
| Date only (`2026-10-22`) | 22 Oct 12:00 | publication time remains null; PAISA does not invent midnight. |

Automated tests cover timezone conversion, missing and naive timestamps,
future-dated publication evidence, and the inclusive cutoff boundary.
