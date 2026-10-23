# GDELT GKG 2.1 historical evidence

PAISA's GKG path consumes official 15-minute **GDELT GKG 2.1 raw archives**.
It is separate from the existing GDELT DOC provider and does not reinterpret a
later DOC query as contemporaneous availability.

Authoritative references:

- [GKG 2.1 codebook](https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf)
- [GDELT 2.0 raw-file access](https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/)
- [GDELT master file list](https://data.gdeltproject.org/gdeltv2/masterfilelist.txt)
- [GDELT timestamping policy](https://blog.gdeltproject.org/a-behind-the-scenes-look-at-how-we-think-about-master-file-formats-and-timestamping/)
- [GDELT Terms of Use](https://gdeltproject.org/about.html#termsofuse)

GDELT permits use of the datasets it releases, including academic use and
redistribution with a GDELT citation and link. That grant does not confer any
right to publisher pages. This provider retains only GDELT metadata and never
dereferences `V2DOCUMENTIDENTIFIER`, downloads a publisher body or image, or
attempts to bypass a paywall.

## Record and time contract

Raw `.gkg.csv.zip` payloads are tab-delimited and have no header row. PAISA
parses against the pinned GKG 2.1 column order and preserves:

- `GKGRECORDID`;
- `V2.1DATE`;
- `V2SOURCECOLLECTIONIDENTIFIER`;
- `V2SOURCECOMMONNAME`;
- `V2DOCUMENTIDENTIFIER`;
- `V1ORGANIZATIONS` and the actual GKG 2.1
  `V2ENHANCEDORGANIZATIONS` field;
- available enhanced-theme and tone metadata;
- the exact archive filename and member, official index byte count and MD5,
  and PAISA SHA-256; and
- the archive row number and PAISA acquisition time.

PAISA does not invent or emit an alternate synthetic organization-field name.
OGDC organization matching uses `V2ENHANCEDORGANIZATIONS`;
`V1ORGANIZATIONS` is retained only as separate source provenance.

For the supported web collection (`V2SOURCECOLLECTIONIDENTIFIER=1`):

```text
source_published_at = valid, nonzero V2.1DATE
gdelt_batch_at      = GKGRECORDID timestamp prefix
available_at        = gdelt_batch_at
fetched_at          = time PAISA acquired the historical archive
```

The `GKGRECORDID` timestamp must equal the filename batch. `V2.1DATE` is
retained as publication metadata and can never move `available_at` before the
batch. Equality is valid; **any** `V2.1DATE` later than the batch is rejected as
`publication_after_batch`. No configuration can admit a future publication
timestamp. A rejected row cannot be mapped to a trading session because it is
never normalized into evidence.

The evidence result records `publication_delay = batch - publication` and
`batch_delay = fetched - batch`. Timestamp-valid counts and delay bounds are
computed after the row's publication, batch, requested-window, and acquisition
ordering checks but before OGDC entity filtering. They therefore describe all
timestamp-valid rows, not only retained OGDC matches. All instants are stored
in UTC at second precision.

Rows also fail closed for malformed or zero required timestamps, invalid row
shape or encoding, cross-batch identity, an out-of-window publication or batch,
an unsupported source collection, a duplicate source identity, an acquisition
time before the batch, or invalid normalized provenance. Translation and
English archive identities are not silently mixed.

## Exact OGDC matching and provenance

The controlled aliases are:

- `OGDC`
- `OGDCL`
- `Oil and Gas Development Company`
- `Oil & Gas Development Company`
- the two names above with `Limited`

`V2ENHANCEDORGANIZATIONS` entries are split structurally, their terminal
numeric offsets are removed, and normalized aliases are compared exactly.
`V2DOCUMENTIDENTIFIER` and `V2SOURCECOMMONNAME` may provide additional
boundary-aware evidence, but unrestricted substring matching is forbidden.
Strings such as `DOGDCX` do not match.

Every retained row binds its match `field`, controlled `alias`, `match_method`,
and `matched_value` to the corresponding preserved source field. The record
also binds `GKGRECORDID`, batch, archive identity and checksum, archive row,
source collection, document identifier, common name, fetch/import times, and
permission-policy identity. Model validation rejects missing, fabricated, or
contradictory GKG provenance before an evidence result can be constructed.

## Bounded raw acquisition

Acquisition requires an explicit UTC interval and partition plan. It enforces
maximum archive count, per-file and cumulative compressed bytes, per-file and
cumulative decompressed bytes, row bytes, field characters, parsed records,
and retained error details. Official index size and MD5 claims are validated,
and PAISA computes SHA-256 over the received archive.

The archive host and HTTPS scheme are fixed. Before each request, DNS must
complete within its configured bound and every answer must be a globally
routable address. Redirects are disabled. TLS certificate and hostname
verification remain enabled, and the connected peer must itself be global and
belong to the approved preflight resolution set. Connect/read bounds are
combined with one monotonic end-to-end deadline covering DNS, connection,
headers, and streamed body reads. Failure details are sanitized and do not
expose credentials. Environment authentication, session credentials, and
publisher-page fetching are disabled.

ZIP handling accepts exactly one expected member whose Unix type is either
regular or unspecified. Absolute paths, parent traversal, links and special
file types, duplicate or extra members, basename mismatches, encrypted members,
unsupported compression, and decompression-limit violations fail closed. A
validated archive may be checkpointed for bounded, checksum-verified resume.

## Immutable pack and replay contract

An EVID pack retains the exact downloaded ZIP bytes, capture metadata,
normalized JSONL and CSV, partition plan, provider configuration, timestamp
validation, provider errors, quality report, permission-policy reference,
checksums, and completion seals. For a GKG pack, `verify_evidence_run` performs
offline raw-ZIP semantic replay: it reopens every ZIP with the same safe-member
rules and acquisition parser, then recomputes all rows and the scanned,
timestamp-valid, rejected-by-reason, unmatched, matched, duplicate, byte, and
delay statistics. Verification requires those replayed records, errors,
counters, and timestamp statistics to agree with the sealed normalized JSONL
and CSV, provider errors, quality report, timestamp report, summary, and
manifest. A resealed but contradictory claim fails verification.

EVID and ECOV publishers build a private direct-child staging tree, apply exact
`0700` directory and `0600` regular-file modes, and run structural, semantic,
permission, checksum, and mode verification before a no-replace atomic rename.
If staging verification fails, the staging tree is removed, no final run is
published, and the same run ID remains available for a corrected attempt.
Completed-pack verification rechecks the private modes. For repository-local
outputs, symlink components, tracked locations, and locations that are not
Git-ignored are rejected.

The committed example configuration intentionally selects only three English
partitions from 4 September 2026 inside the model test interval. It is a
bounded proof, not a completeness claim for the requested 2025-12-16 through
2026-09-15 test window and not a backward expansion into validation or train.

This output is academic evidence engineering, not financial advice. It makes
no claim that GKG coverage is complete, representative, sentiment-bearing, or
predictively useful.
