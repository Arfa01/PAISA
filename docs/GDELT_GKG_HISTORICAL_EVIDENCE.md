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

Raw `.gkg.csv.zip` payloads are tab-delimited and have no header row. The
provider models and preserves:

- `GKGRECORDID`;
- `V2.1DATE`;
- `V2SOURCECOLLECTIONIDENTIFIER`;
- `V2SOURCECOMMONNAME`;
- `V2DOCUMENTIDENTIFIER`;
- organization metadata (`V1ORGANIZATIONS` and the GKG 2.1 enhanced
  organization field);
- available theme and tone metadata;
- exact archive filename, official index byte count/MD5, and PAISA SHA-256;
  and
- the PAISA acquisition time.

For the supported web collection (`V2SOURCECOLLECTIONIDENTIFIER=1`):

```text
source_published_at = valid, nonzero V2.1DATE
gdelt_batch_at      = GKGRECORDID timestamp prefix
available_at        = gdelt_batch_at
fetched_at          = time PAISA acquired the historical archive
```

The record-ID timestamp must equal the filename batch. `V2.1DATE` is retained
as publication metadata and can never move `available_at` before the batch.
The evidence pack records `publication_delay = batch - publication` and
`acquisition_delay = fetched - batch`. All instants are stored in UTC at
second precision.

Rows are rejected or quarantined for a malformed record ID, a required zero or
malformed publication time, unsupported collection semantics, cross-batch
identity, out-of-window time, or publication later than the batch by more than
the configured tolerance. Translation and English archive identities are not
silently mixed.

## Exact OGDC matching

The controlled aliases are:

- `OGDC`
- `OGDCL`
- `Oil and Gas Development Company`
- `Oil & Gas Development Company`
- the two names above with `Limited`

Organization entries are split structurally, their terminal numeric offsets
are removed, and normalized aliases are compared exactly. URL/document
metadata may provide additional boundary-aware evidence, but unrestricted
substring matching is forbidden. Every retained row records the matched
alias, field, and source value. Strings such as `DOGDCX` do not match.

## Bounded raw acquisition

Acquisition requires an explicit date/time interval and partition plan. It
enforces maximum archive count, per-file and cumulative compressed bytes,
cumulative decompressed bytes, and parsed records. The downloader streams
bytes, validates the official index size and MD5, calculates SHA-256, rejects
redirects and unsafe network destinations, and writes a resumable checkpoint
after each completed partition.

ZIP handling accepts exactly one expected regular member. Absolute paths,
parent traversal, links, duplicate/extra members, basename mismatches, and
decompression-limit violations fail closed. A completed evidence pack retains
the exact downloaded archives, capture metadata, normalized metadata rows,
partition plan, timestamp validation, errors, checksums, and permission-policy
reference. Packs are private, immutable, collision-refusing, and ignored by
Git.

The committed example configuration intentionally selects only three English
partitions from 4 September 2026 inside the model test interval. It is a
bounded proof, not a completeness claim for the requested 2025-12-16 through
2026-09-15 test window and not a backward expansion into validation or train.

This output is academic evidence engineering, not financial advice. It makes
no claim that GKG coverage is complete, representative, sentiment-bearing, or
predictively useful.
