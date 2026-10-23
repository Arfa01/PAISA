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
Equality is accepted, but any `V2.1DATE` later than the batch is rejected as
`publication_after_batch`; there is no configuration that can admit that row.
Neither a URL date nor current PAISA acquisition moves availability earlier or
later than the preserved historical batch. A rejected future-publication row
cannot enter coverage or map to a session.

## PSX rule

An exchange posting instant is used only when an authorized source supplies
the clock value, precision, and defensible timezone. `available_at` then equals
that posting instant. Source precision is retained rather than silently filled:

- a second-precision value denotes that exact instant;
- a minute-precision value denotes the whole closed minute for cutoff
  eligibility; and
- if only a date is present, both timestamp fields remain null,
  `time_precision=date_only`, and coverage maps strictly to the first observed
  trading session after that date.

For a 15:30:00 market cutoff, second-precision `15:30:00` is eligible for that
session, while `15:30:01` moves forward. Minute-precision `15:30` covers through
`15:30:59.999999` for eligibility and therefore moves forward; the stored
source timestamp remains `15:30` with minute precision and does not invent
seconds. A minute-precision `15:29` remains eligible at that cutoff. Date-only
evidence receives no same-day credit.

The current coverage command does not yet accept an optional authorized PSX
EVID input. It emits a permission-blocked PSX reference while the source
decision remains `REVIEW_REQUIRED`. Adding an explicit, attested PSX input to
that CLI is deferred; public-page or undocumented-endpoint automation remains
prohibited.

## Provider contract

Provider identity, provider mode, evidence type, data class, and timestamp
precision form one bidirectional contract. The admitted combinations are:

| Provider and mode | Evidence type | Data class | Allowed precision |
|---|---|---|---|
| `gdelt-gkg-2.1/raw_archive` | `gdelt_metadata` | `real` | `second` |
| `psx-announcement/manual_import` | `exchange_announcement` | `imported` | `second`, `minute`, `date_only` |
| `gdelt-doc/real`, `rss/real` | `publisher_news` | `real` | `second`, `unknown` |
| `manual-csv/imported` | `publisher_news` | `imported` | `second`, `unknown` |

An unknown provider/mode pair, a known provider claiming another type or data
class, or a disallowed precision is rejected during record reconstruction. GKG
and PSX records then undergo their additional source-specific provenance
checks; a label alone cannot opt a record into a more permissive profile.

## Coverage and leakage boundary

Timed records map to the first observed session whose configured market cutoff
is at or after the end of the timestamp's declared precision interval.
After-cutoff, weekend, and holiday arrivals move to the next observed trading
session and never backward. Date-only exchange records use the stricter
next-session rule above. Evidence outside the observed stock window remains
unmapped. Split membership comes only from the immutable model
`split_manifest.json`; a record cannot be reassigned across a split boundary to
improve coverage.

An ECOV pack seals canonical normalized records and the exact stock calendar.
Its portable verifier reconstructs every record (thereby reapplying the
provider contract), derives coverage profiles from verified record semantics,
recomputes session and split mappings from the sealed calendar and cutoff, and
recalculates split/monthly coverage, missing intervals, and readiness. It
rejects persisted profile labels, mappings, or cutoff eligibility that differ
from that recomputation, including any record mapped earlier than its precision
interval permits. This ECOV check is record-and-calendar replay; the source ZIP
semantic replay is performed offline by `verify_evidence_run` when it verifies
the upstream GKG EVID pack.

Coverage is reported independently for GKG OGDC metadata, PSX OGDC
announcements, their combined OGDC union, and general publisher news when
available. Evidence types remain visible in every combined count. Official
announcement overlap alone can never produce `READY_FOR_AGGREGATION`.

Readiness thresholds are currently supplied by a checksum-bound configuration
and do not include non-configurable minimum floors. A reported readiness status
therefore means only **caller-defined gate satisfaction**, not statistical
sufficiency, representative coverage, predictive value, or financial merit.
Adding independent non-configurable readiness floors remains deferred.

EVID and ECOV packs are created in private staging directories, verified before
publication, and atomically renamed without replacement. Run directories use
mode `0700`; every pack artifact must be a regular file with mode `0600`.
Verification failure removes only the private staging tree, publishes no final
run, and leaves the requested run ID reusable.

This policy supports an academic data-coverage audit only. It is not financial
advice and makes no claim of model improvement, causality, or trading value.
