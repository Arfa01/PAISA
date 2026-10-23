# PSX announcement data contract

Schema version: `1.0`

PSX announcements are `exchange_announcement` evidence. They are not relabeled
as publisher news or GDELT metadata, and this contract contains no sentiment,
prediction, signal, trade, or portfolio field.

## Required input

The supported provider mode is an offline `manual_import`. A local CSV row
must supply the exact `OGDC` ticker, company name, announcement ID and type,
displayed title, official source URL, event category, source precision, and
either an exchange posting timestamp with timezone evidence or a source date.
The import invocation also supplies an authorization attestation and reference.
The importer never performs network access.

Deterministic event categories are:

`financial_results`, `board_meeting`, `material_information`,
`production_update`, `dividend`, `annual_report`, `AGM_EOGM`,
`corporate_action`, and `other`.

## Normalized record

Every record carries the unified evidence fields:

- `evidence_id`, `evidence_type`, `provider_id`, `provider_mode`, and
  `data_class`;
- source identity, URL, ticker, company, announcement ID/type, title, and
  deterministic event category;
- `source_published_at` or `source_published_date`, `available_at`,
  `fetched_at`, `imported_at`, `time_precision`, and
  `historical_availability_basis`;
- `contemporaneous_source_id`, `contemporaneous_batch_at`, and
  `permission_policy_id`;
- official PDF URL/reference and an optional operator-supplied SHA-256 whose
  verification status is explicitly recorded;
- raw-evidence reference, normalized-content checksum, warnings, and
  source-specific metadata.

Exact ticker equality is required after trimming and case normalization.
Substring or company-name inference cannot turn another issuer into `OGDC`.
Unknown categories, duplicate IDs, inconsistent timestamp fields, unsafe raw
references, or a missing authorization attestation fail closed.

## Availability rules

For an authorized source with an exchange posting clock and timezone evidence:

```text
source_published_at = official exchange posting instant
available_at        = source_published_at
historical_availability_basis = official_exchange_posted_timestamp
```

The recorded precision is `minute` or `second`; missing seconds are not
invented. Coverage compares the instant with the configured Asia/Karachi market
cutoff and moves an after-cutoff item to the next observed session.

When only a source date exists:

```text
source_published_date = displayed source date
source_published_at   = null
available_at          = null
time_precision        = date_only
historical_availability_basis = official_exchange_date_conservative_next_session
```

Coverage then maps the item strictly to the first observed trading session
after the source date. Weekends and exchange holidays naturally carry forward.
The PDF's internal meeting/report dates do not replace exchange posting
evidence.

## Evidence and storage

The immutable run captures the supplied CSV bytes exactly and retains their
checksum in a private ignored artifact. A local PDF is not required and PDF
bytes are never read by this importer. A non-null CSV `pdf_sha256` is retained
only as an operator-supplied claim, with
`pdf_checksum_verification=operator_supplied_unverified_claim` and a matching
normalization warning; it is not represented as a PAISA-verified file digest.
Otherwise `pdf_sha256` is null. Public documentation and committed fixtures
contain no PSX announcement corpus or PDF content.

Read `PSX_ANNOUNCEMENT_SOURCE_ASSESSMENT.md` before every import. A sealed pack
proves integrity and recorded provenance; it does not prove the operator owns
the upstream rights. This is an academic simulation and not financial advice.
