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

Profile membership is derived only after those checks. A GKG row is fixed to
the controlled OGDC ticker and company identity and its source record ID must
agree with the preserved GKG identity. A PSX row must retain the exact PSX
source identity, an official `psx.com.pk` URL, the controlled OGDC company and
ticker, and matching source/contemporaneous IDs. Publisher records use only
`GENERAL` or `OGDC`: the latter requires a positive OGDC occurrence count and
complete deterministic entity-link evidence, while a general record must have
neither. The representative publisher occurrence and raw reference must name
the same checksum-verified ingestion run. When a canonical article first
appears as general news and only a later duplicate carries an OGDC entity link,
the OGDC record uses the earliest OGDC-linked occurrence for its source,
content, permission, and availability timestamps; it cannot borrow the earlier
general-news timestamp. Coverage rejects a duplicate
`(provider_id, provider_mode, source_record_id)` even if it is assigned a new
PAISA `evidence_id`.

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

For every non-empty EVID source, the coverage input reference also carries an
ordered, checksum-bound record-identity inventory. Each entry records its
source ordinal and the digest of the full canonical normalized row. Publication
and portable verification reconstruct the complete canonical JSONL in source
order and require its digest to equal the upstream EVID
`normalized_evidence_sha256`; changing even a non-identity field such as a title
therefore breaks the binding. A zero-record GKG or PSX source must always use
the SHA-256 of the empty byte stream, whether or not it includes an explicit
empty identity inventory. Publisher inputs carry the complete canonical
NEWS artifact inventory whose digest must equal the upstream
`artifact_inventory_sha256`, plus bounded base64url snapshots of the exact
`normalized_articles.csv` and `article_entity_links.csv` bytes. Each snapshot's
size and SHA-256 must match its inventory entry, and the normalized-article
snapshot must also match `normalized_articles_sha256`. The reference also
embeds the exact bounded `run_manifest.json` bytes: their digest must equal
`run_manifest_sha256`, and the manifest's run, provider, mode, status, committed
state, inventory, and inventory digest must equal the reference. Consequently,
a resealer cannot replace both CSV snapshots and their inventory while retaining
the claimed upstream manifest identity.

The portable verifier parses those exact source snapshots, reconstructs NEWS
occurrences, reruns the shared publisher adapter, and requires the resulting
records to equal the sealed ECOV publisher records. Representatives are also
reconciled to their referenced NEWS run, permission decision and occurrence
counts. Their source metadata includes an ordered identity for every canonical
occurrence, and every entity link must name the referenced ingestion run and
article for an OGDC-linked occurrence. This permits legitimate cross-run
canonical duplicates without allowing a synthetic link, title, or label to opt
into OGDC coverage even if an attacker reseals the adapted-record inventory.
The retained demonstration pack predates these bindings; portable verification
upgrades only its exact known run/manifest hash in memory. All newly published
packs must carry the strengthened contract. These checks are portable
identity/provenance replay, not a replacement for verifying the upstream raw
NEWS artifact at initial ingestion.

These larger portable snapshots are opt-in at the NEWS loader boundary and are
enabled by the contemporaneous-evidence audit only. Ordinary news coverage
continues to use its existing run-reference schema, avoiding an unrelated
artifact-schema change for existing coverage packs.

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
