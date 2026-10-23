# PSX OGDC announcement source assessment

Assessment date: **23 October 2026**

Scope: official Pakistan Stock Exchange (PSX) announcements for `OGDC` during
the PAISA stock period. This is an engineering and source-rights assessment,
not legal advice. It does not authorize collection by itself.

## Decision

PAISA must not automate the public PSX website or undocumented endpoints under
the evidence reviewed for this branch. The supported path is a controlled,
offline import of an operator-supplied export only after the operator attests
that the intended acquisition, storage, academic analysis, and publication of
derived metadata are authorized. No such licensed export was supplied for this
work, so the retained real result is **blocked with zero announcements**.

The machine-readable companion policy is
`configs/evidence/source_permissions.json`. It permits GDELT raw-archive
automation, prohibits the PSX web-automation mode, and marks the PSX manual
mode `REVIEW_REQUIRED`. A future authorized dataset needs a reviewed replacement
decision that checksum-binds the exact authorization attestation and reference.
Changing this assessment requires a new assessment checksum and policy review.

## Official sources reviewed

- [OGDC company page](https://dps.psx.com.pk/company/OGDC)
- [Company-announcement search](https://dps.psx.com.pk/announcements/companies)
- [PSX Terms of Use](https://www.psx.com.pk/psx/terms-of-use)
- [PSX Data Services and Vending](https://www.psx.com.pk/psx/product-and-services/data-services-vending)
- [PUCARS portal description](https://pucars.psx.com.pk/pucars/user/login-form)
- [PSX trading and PUCARS hours](https://www.psx.com.pk/psx/exchange/general/trading-hours)

The company page separates Financial Results, Board Meetings, and other
announcements and exposes a date, title, and PDF link. It appears to expose
only a recent subset and is not a documented complete export. The central
search exposes company, type, keyword, and date filters and can display a time,
title, and PDF link. Public pages reviewed for this assessment do not provide a
documented bulk API, completeness guarantee, stable export contract, or clear
timezone statement for every displayed clock value.

## Rights and access conclusion

The PSX terms reserve the site's content and, without prior written
permission, prohibit systematic retrieval used to create a collection,
database, or directory and prohibit automated browsers, spiders, robots,
agents, or similar search/navigation mechanisms. They also restrict caching,
copying, storage, reproduction, and redistribution. The Data Services page
identifies PUCARS corporate announcements as PSX data whose use in feeds,
websites, applications, or derived products requires applicable rights or
approval. Requests for data authorization can be directed to the contact
published on that page.

Therefore PAISA does not:

- scrape the company page or announcement search;
- call an undocumented AJAX endpoint;
- follow or download announcement PDFs automatically;
- treat attribution as a substitute for permission; or
- commit PSX PDFs or a systematically collected PSX metadata database.

## Controlled manual-import boundary

The importer reads only a local CSV supplied by the operator. It never opens a
source or PDF URL. Each import requires:

- the reviewed `permission_policy_id`;
- acquisition mode `manual_import`;
- an explicit authorization attestation and authorization reference;
- source URL, observation/import time, source precision, and exact ticker;
- a stable announcement identifier, displayed title, type, and event category;
- the official PDF URL may be retained as a reference; a CSV-supplied PDF
  checksum is labelled as an **operator-supplied, unverified claim** because
  this metadata-only importer never reads PDF bytes; and
- raw input retained only in an ignored, private immutable evidence pack.

The operator's attestation is provenance evidence, not a licence manufactured
by PAISA. Before importing a new dataset, written permission or a licensed
export should explicitly cover the historical range and fields, local
retention, academic/derived use, public repository disclosures, attribution,
and redistribution restrictions.

## Timestamp precision

`available_at` equals the official exchange posting timestamp only when an
authorized source supplies that timestamp, its precision, and defensible
timezone evidence. A company-page date alone is stored as `date_only`; PAISA
does not invent midnight, market open, market close, or seconds. Coverage maps
date-only evidence strictly to the **next observed eligible trading session**,
not the same date. A meeting date or clock printed inside an issuer PDF is not
evidence of when PSX posted the announcement.

Titles and official metadata are sufficient for this branch. It does not run
FinBERT over PDFs and does not assign positive or negative sentiment.
