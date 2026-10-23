# Contemporaneous evidence time policy

Policy version: `contemporaneous-evidence-time-v1`

This policy adds a narrow, source-specific historical availability contract.
It does not weaken `news-availability-v1` for ordinary publisher news.

## Separate timestamp evidence

| Evidence type | Publication evidence | Availability evidence | PAISA acquisition evidence |
|---|---|---|---|
| `publisher_news` | Publisher timestamp when valid | Existing conservative news contract | `fetched_at`/`imported_at` |
| `gdelt_metadata` | GKG `V2.1DATE` | GKG batch encoded in `GKGRECORDID` | Archive download/query time |
| `exchange_announcement` | Authorized PSX posting timestamp, or source date only | Exact posting timestamp when authorized; otherwise conservative next-session mapping | Manual import time |

`source_published_at`, `available_at`, `fetched_at`, and `imported_at` answer
different questions and are never collapsed. A historical archive imported
today may defensibly show that GDELT had the metadata in a historical batch;
ordinary publisher content fetched today cannot use that exception.

## GDELT rule

For supported GKG 2.1 web rows, `available_at` equals the UTC batch timestamp
encoded in `GKGRECORDID`, which must also equal the archive filename batch.
Nonzero valid `V2.1DATE` is stored separately as `source_published_at`.
Publication later than the batch beyond the documented configured tolerance is
quarantined. Neither a URL date nor current PAISA acquisition moves
availability earlier or later than the preserved historical batch.

## PSX rule

An exchange posting instant is used only when an authorized source supplies
the clock value, precision, and defensible timezone. `available_at` then equals
that posting instant. If only a date is present, both timestamp fields remain
null, `time_precision=date_only`, and coverage maps strictly to the first
observed trading session after that date. No same-day credit is given.

## Coverage and leakage boundary

Timed records map to the first observed session whose configured market cutoff
is at or after `available_at`; after-cutoff, weekend, and holiday arrivals move
forward and never backward. Date-only exchange records use the stricter
next-session rule above. Evidence outside the observed stock window remains
unmapped. Split membership comes only from the immutable model
`split_manifest.json`; a record cannot be reassigned across a split boundary to
improve coverage.

Coverage is reported independently for GKG OGDC metadata, PSX OGDC
announcements, their combined OGDC union, and general publisher news when
available. Evidence types remain visible in every combined count. Official
announcement overlap alone can never produce `READY_FOR_AGGREGATION`.

This policy supports an academic data-coverage audit only. It is not financial
advice and makes no claim of model improvement, causality, or trading value.
