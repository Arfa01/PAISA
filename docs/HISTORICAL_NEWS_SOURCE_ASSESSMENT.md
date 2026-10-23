# Historical News Source Assessment

Assessment date: **2026-10-23**

Scope: historical financial-news evidence overlapping the OGDC daily-price
history and configured GDELT query window from 2021-09-16 through 2026-09-15.
The exact model target dates remain those in the immutable split manifest.

This is an engineering assessment, not legal advice. Provider terms, plan
limits, and source coverage can change. Re-check the linked primary sources
before every future acquisition run and retain a dated copy or checksum of the
terms and plan evidence used for the run. The current machine-readable decision
is deliberately limited to the already captured run named below; a future run
requires a new review and an explicit policy allowlist update.

## Decision

- **Selected automated provider: GDELT DOC 2.0 metadata only.** GDELT gives an
  explicit unrestricted-use grant for its released datasets and permits
  redistribution with a GDELT citation. The selected integration may retain
  the title, source URL, GDELT observation timestamp, domain, language, source
  country, request evidence, and PAISA-computed checksums of returned GDELT
  responses. It must not fetch or retain publisher article bodies, images, or
  paywalled content.
- **Selected import path: licensed manual CSV.** A legitimately obtained
  dataset may be imported only when its provenance and field-level storage
  permissions are recorded. The importer must not treat an import as live
  retrieval or silently broaden the supplied licence.
- **Retained Dawn Business RSS run: existing evidence with
  `LICENSING_REVIEW_REQUIRED`.** Dawn published the feed endpoint, and its
  terms contain a narrow exception for a user's own personal, non-commercial
  use. That is the only published basis found for provisionally retaining the
  already-captured snapshot as private academic-audit evidence. Dawn does not
  expressly grant RSS archiving, research-corpus storage, team/institutional
  use, ML reuse, or redistribution rights. Keep the existing run local,
  ignored, access-limited, and unchanged while permission is reviewed; do not
  publish it, commit it, export its content, or use Dawn for a new automated
  acquisition without written permission.
- **No credentials are available or used in this branch.** Credentialed or
  paid providers remain configurable candidates, not selected acquisition
  sources.
- PSX, OGDCL, Business Recorder, Google News, and other publisher pages are
  reference/discovery candidates only unless written permission or a licence
  authorises the intended automated retrieval and storage.

The decision is deliberately narrower than “publicly visible means reusable.”
Where a provider does not expressly grant storage, redistribution, academic
use, or automation rights, this assessment says **unknown / contract review
required**.

## Machine-readable permission decisions

The executable companion policy is
`configs/news/historical_source_permissions.json`. It binds this document's
path and SHA-256, then records a decision ID, provider/acquisition mode, status,
scope, any allowlisted run IDs, and any permission-relevant provider controls
or licensing-note checksum. Coverage artifacts preserve the policy fingerprint
and the exact decision applied to each selected ingestion run; free-form prose
or a non-empty licence note alone cannot turn an unresolved source into
permitted evidence.

The current policy marks only the metadata-only GDELT run
`NEWS-20261023-060000-gdelt-ogdc2` `PERMITTED`, provided its
permission-relevant controls match. Those controls disable article-page
fetching and environment authentication and bind the official endpoint,
historical-API mode, and exact reviewed licensing note; they are not a claim
that every operational configuration field has legal significance. Any other
GDELT run fails the allowlist until this assessment and policy are reviewed and
updated. Existing RSS evidence and manual CSV imports are `REVIEW_REQUIRED`; a
separately obtained dataset needs a new, dataset-specific reviewed decision
before it can support `READY_FOR_AGGREGATION`. Updating this assessment requires
regenerating the companion policy's `assessment_sha256` before another audit.

## Candidate matrix

| Provider | Access and historical coverage | Search and OGDC support | Limits, key, and cost | Storage and licensing assessment | Academic use, attribution, and automation | Decision and limitations |
|---|---|---|---|---|---|---|
| **GDELT DOC 2.0** | Public REST API and downloadable GDELT datasets. DOC 2.0 search covers 2017-01-01 onward; the broader GDELT backfile should not be confused with the searchable DOC article list. Each request considers at most its last three months, so longer configured ranges must be tiled proactively. | Full-text keyword/keyphrase, Boolean `OR`, domain, language, country, and date filters. It can search exact company names and aliases such as `OGDCL`; results still require deterministic OGDC-link evidence because `OGDC` alone can be ambiguous. | No API key or fee. Article-list default is 75 and maximum is 250 records per request. GDELT applies rate limiting but publishes no fixed DOC request quota; high-volume users are directed to downloadable datasets. | GDELT says released datasets are available for unrestricted academic, commercial, and governmental use and may be redistributed with citation. Store returned title, URL, and GDELT metadata. Summary/body is not returned by DOC ArticleList and publisher body rights are not granted by this selection; do not fetch or store bodies or images. | Academic use is expressly allowed. Cite and link GDELT in every use or redistribution. Programmatic API/download access is documented; respect 429 responses, use backoff, and do not parallelise to evade limits. | **Selected for metadata-only automated retrieval.** Coverage is not exhaustive, Pakistan-source coverage is not guaranteed, there is no pagination, and the observation timestamp is not reliable proof of the publisher's publication time. |
| **Licensed manual CSV** | Offline import of a dataset obtained directly from a provider, library, research partner, or owner. Coverage is whatever the signed licence and evidence manifest declare. | Dataset-dependent. OGDC linkage must be supplied or derived deterministically from preserved company-name/ticker evidence; general financial news remains separate. | No runtime key. Cost and limits depend on the upstream agreement. | Titles, summaries, URLs, and bodies may be retained **only field by field as the upstream licence permits**. Import requires source identity, licence/usage note, provenance, raw evidence reference, and checksums. | Academic use, required attribution, redistribution, retention period, and automation must be recorded from the agreement. | **Selected import path.** Reject missing provenance, test/fixture/synthetic data in real mode, fabricated timestamps, and any dataset whose intended use is not reasonably permitted. |
| **Dawn Business RSS (retained 2026 snapshot)** | Publisher-supplied public RSS endpoint. The retained fetch is a current-feed snapshot, not a historical search API: its 30 items span 2026-10-19 through 2026-10-22 and do not establish archive coverage back to 2021. | The feed has no search or OGDC entity contract. PAISA's deterministic linker marked 1 of the 30 retained records as OGDC-linked; this is project evidence, not a coverage guarantee. | No key or fee. No public API quota was found. The captured channel declared a 60-minute RSS TTL, but that is not a licence or an assurance that indefinite polling is permitted. | The official feed supplied title, URL, description, `content:encoded`, author, and publication metadata; some description/content fields contain article-length HTML. Dawn's terms reserve all website-content rights and permit use only for one's own personal, non-commercial use unless prior written permission is obtained. No explicit field-level storage, automated archival, academic-corpus, team-sharing, or redistribution grant was found. | The public endpoint supports the fact of publisher-provided access; it does **not** by itself grant copyright reuse. The personal/non-commercial exception is the only narrow basis found for a private individual audit. PAISA should preserve source and copyright evidence as provenance, but attribution does not replace permission. | **Existing evidence only; `LICENSING_REVIEW_REQUIRED`.** Provisionally quarantine the already-retained ignored run for private audit, with no public or repository redistribution. Do not schedule new collection, backfill, training-corpus use, or content export until Dawn gives written permission covering the intended fields and users. |
| **NewsAPI.org** | Authenticated REST API. The Developer plan searches only about one month; paid Business/Advanced plans advertise up to five years of history. | `/v2/everything` supports keywords, exact phrases, Boolean expressions, title/description/content search, domains, sources, languages, and date bounds. OGDC is keyword-based, not a verified PSX entity mapping. | Key required. Developer: 100 requests/day, delayed results, development/testing only. Business: USD 449/month and 250,000 requests/month. Advanced: USD 1,749/month and 2,000,000 requests/month (prices observed on assessment date). | Returns source, author, title, description, URL, UTC `publishedAt`, image URL, and at most a truncated content fragment. Full text is not provided. Terms prohibit reproducing copyrighted material and warn that third-party content needs its own licence; public disclosure/storage rights for a research corpus are not clearly granted. Do not store publisher bodies. | No express academic-use grant was found. Source/author/legal notices must not be falsified or removed. API automation is supported, but the free plan is only for development/testing. | **Not selected.** Requires a key/paid plan for the target period, publisher coverage must be tested, historical `available_at` evidence is absent, and corpus retention/publication needs written clarification. |
| **Financial Modeling Prep (FMP) Stock News** | Authenticated REST API. FMP states that stock news begins in 2019 for most stocks, with some deeper histories. | Stock-news search accepts symbols and date bounds and returns publisher, title, text/snippet, URL, and publication date. Worldwide company search exists, but OGDC/PSX symbol coverage has not been verified with an entitled account. | Key required. Basic: 250 calls/day. Paid published limits include 300/minute (Starter), 750/minute (Premium), and 3,000/minute (Ultimate). Global coverage is advertised at the Ultimate tier. | Terms restrict copying/distribution and say display or redistribution needs a specific Data Display and Licensing Agreement. Storage locations are subject to account/security terms. Whether titles/snippets can be kept in a public academic artifact is **unknown / contract review required**. | No general academic-use grant was found; a student discount is advertised. API automation is supported within plan limits. Attribution/display terms depend on the agreement. | **Not selected.** Key/entitlement required, OGDC availability is unverified, display/redistribution needs an agreement, and `available_at` history is not documented. |
| **Alpha Vantage Market News & Sentiment** | Authenticated premium `NEWS_SENTIMENT` API with `time_from`/`time_to`; the documentation calls it live and historical but does not state a dependable earliest news date. | Filters by supported ticker(s), topics (including energy/transportation), time, and order. PSX ticker `OGDC` support is unverified; company-name free-text search is not documented for this endpoint. | Key required. The standard service limit is 25 API requests/day; `NEWS_SENTIMENT` is marked premium and premium limits are plan-specific. | The published licence covers personal, non-commercial access unless otherwise agreed. It does not clearly grant public redistribution or long-term corpus storage of third-party article titles/summaries/URLs/body. Body storage is therefore not permitted by this assessment. | Private individual research/testing fits the stated non-commercial categories, but an institutional/public FYP dataset needs provider confirmation. No specific attribution rule for news records was found. API automation is documented. | **Not selected.** Premium credential needed, OGDC/PSX coverage and historical depth are unknown, and storage/redistribution rights need written clarification. |
| **NewsAPI.ai / Event Registry** | Authenticated API, archive advertised from January 2014, with REST/Python/Node access. | Keyword, concept/entity, category, source, language, date, sentiment, event, and duplicate filters. Company concepts may support OGDCL, but entity coverage must be tested. | Key required. Free access is limited to recent evaluation data. Published plans charge more tokens for historical years; results are up to 100 articles/search. Paid historical access starts at published plan pricing and an academic discount can be requested. | Provider pages advertise article title/body/URL/date and dataset creation, but the public terms could not be reliably reviewed in static form and publisher-body reuse rights are not established. Field retention and redistribution are **unknown / contract review required**, especially for full text. | Research use and academic discounts are advertised. Exact attribution and redistribution duties require the operative subscription terms/order form. API automation is supported. | **Not selected.** No credential; contract review is required before retaining data, and Pakistani/OGDC coverage needs a sample audit. |
| **NewsData.io** | Authenticated REST API. Paid archive depth is advertised as 6 months (Basic), 2 years (Professional), or 10 years (Corporate). | Advanced keyword/title/meta search, domain, country, language, dates, symbol, duplicate, and other filters. OGDC/PSX symbol coverage is unverified. | Key required. Free: 200 credits/day and 30 credits/15 minutes. Paid: 1,800 credits/15 minutes; an Archive request costs 5 credits. Corporate historical coverage was USD 1,299.99/month on the assessment date. | Terms grant API access but say third-party “Unstructured Raw Data” carries no granted use right. The provider FAQ says title, short description, publisher, author, date, and time may be published with credit, while full content/images may not; that tension requires contract review before corpus storage. | Research use is described on the provider site. “Powered by NewsData.io API” attribution is required unless otherwise authorised. API automation is supported within the plan. | **Not selected.** No credential, target depth requires a costly plan, OGDC coverage is unverified, and raw/full-text rights are expressly unresolved. |
| **Google News / Google News RSS results** | Consumer news discovery/search and RSS-like result feeds; no supported public historical-news archive API with a documented completeness or retention contract was identified. | User-facing keyword/company search is possible, but result ranking and availability change and there is no stable OGDC entity contract. | No documented archive API quota/key because no supported archive API was found. | Google results link to third-party publisher content and do not grant a licence to store publisher titles, summaries, URLs, or bodies as a research corpus. Publisher rights still apply. | No academic archive grant or required corpus-attribution scheme was found. Google terms require automated access to follow machine-readable restrictions; an undocumented endpoint is not a defensible bulk-ingestion contract. | **Not selected.** Discovery only. No historical completeness, stable API, publication/availability provenance, or storage licence. |
| **Pakistan Stock Exchange (PSX) announcements** | Public Data Portal company-announcement pages, dated PDFs, daily announcement downloads, and symbol/date search. Exact earliest announcement completeness is not documented. | Native company symbol `OGDC`, announcement category, keyword, and date UI filters give strong issuer-link evidence. | No public archive API or automation quota was identified. | PSX terms allow a single unaltered copy for personal non-commercial use but prohibit systematic retrieval/databases and prohibit spiders/robots or other navigation/search mechanisms without prior written permission. Content/PDF redistribution is not granted. | No express academic automation grant. Attribution/proprietary notices must remain intact. Automated retrieval requires prior written PSX permission. | **Not selected for automation.** Excellent manual verification/reference source; use only licensed/manual evidence or obtain written permission. Do not scrape the portal. |
| **OGDCL official news and reports** | Public issuer news pages and report/filing downloads. The current site exposes a paginated news archive and financial reports (the report page visibly reaches at least FY2020), but exact completeness and earliest news date are not contractually documented. | Native OGDCL content and site search provide strong issuer-link evidence. | No public archive API, rate limit, or API key was identified. | No clear site-wide reuse licence or automation permission was found. Public availability is not a grant to build a corpus. Store links and manually obtained evidence only when the applicable issuer/document permission supports it; body/PDF republication is otherwise unresolved. | Academic use, attribution wording, bulk retention, and automated retrieval are **unknown / written permission required**. | **Not selected for automation.** Useful authoritative cross-check and possible licensed manual-import source; publication time may be date-only and must not be fabricated. |
| **Business Recorder** | Public Pakistan-focused financial-news site with searchable historical articles; examples confirm OGDCL material well before the target period, but no public archive API/completeness guarantee was found. | Site search and article text support OGDCL/company-name discovery. It is not a stable machine API or entity-link contract. | No documented public API or retrieval quota. | Terms reserve copyright, prohibit copying/reproduction/republishing/distribution, and permit downloading portions only for personal non-commercial use with notices intact. They do not authorise building or publishing a news corpus. | No express academic corpus grant. Attribution does not replace permission. Automated bulk retrieval is not reasonably permitted under the published access method and restrictions. | **Not selected.** Potential licensed manual provider if Business Recorder grants permission; otherwise reference links only, no scraping or body storage. |

## GDELT metadata-only provider contract

The bounded provider must use the documented DOC 2.0 endpoint:

```text
GET https://api.gdeltproject.org/api/v2/doc/doc
```

Required and allowed request parameters are:

| Parameter | Required behaviour |
|---|---|
| `query` | Configuration-supplied full-text query. Use exact quoted company names and narrowly scoped aliases. Preserve the literal query as link evidence. If more than one population is acquired, run general-financial, energy-sector, and OGDC queries separately so a broad match is never relabelled as OGDC-specific. The reviewed example and retained run intentionally query OGDC only; no energy-sector query or deterministic sector evidence was captured, so energy-sector coverage remains `UNSUPPORTED`. |
| `mode=artlist` | The only selected result mode. |
| `format=json` | Selected machine format. Article-list JSON exposes richer provenance than JSONFeed. |
| `format=jsonfeed` | Allowed by GDELT only for ArticleList and documented as a list of matching URLs and titles. It is insufficient for PAISA ingestion because it omits needed provenance fields; it may be used only for diagnostics, not as the canonical captured evidence. |
| `startdatetime`, `enddatetime` | Explicit, exclusive UTC bounds in `YYYYMMDDHHMMSS` (articles must be after the start and before the end). Do not rely on a moving `timespan`. PAISA treats configured window bounds as inclusive target seconds, expands each outbound bound by one second, and records target and request bounds in the raw evidence manifest. The current DOC search backfile starts at 2017-01-01. Since ArticleList considers at most the last three months of a request, PAISA uses overlapping request tiles of no more than three calendar months. |
| `maxrecords` | Integer from 1 through 250. GDELT defaults to 75 and caps ArticleList at 250. PAISA must configure the value and never assume a capped response is complete. |
| `sort` | `datedesc` or `dateasc`. Record the value. Ordering is not pagination and does not make a saturated response complete. |

DOC ArticleList has no documented page cursor. Therefore the provider must:

1. tile every configured inclusive range into chronological, overlapping
   request windows of no more than three calendar months, even when results are
   sparse; expand outer exclusive API bounds by one second so target endpoints
   remain queryable, and deterministically deduplicate repeated boundary rows;
2. query each bounded tile sequentially;
3. treat a response with `len(articles) == maxrecords` as potentially
   truncated;
4. split the tile deterministically and retry until every leaf window is
   below the cap, or fail closed at the configured minimum window size;
5. issue requests sequentially with a configurable delay;
6. honour HTTP 429 and 5xx responses with bounded exponential backoff; honour
   valid `Retry-After` within the approved wait ceiling and fail the tile when
   the requested wait exceeds that ceiling;
7. quarantine any response whose `seendate` is missing, malformed, or outside
   its inclusive target tile; and
8. never rotate addresses, credentials, or queries to evade service limits.

The normal `format=json` ArticleList response contains an `articles` list. The
documented/current fields used by PAISA are mapped as follows:

| GDELT field | PAISA meaning |
|---|---|
| `url` | Source URL and provider record identity input. Preserve the exact value and also compute the normalised canonical URL used for deduplication. |
| `url_mobile` | Optional alternate source URL. Preserve only as metadata; do not follow it to scrape content. |
| `title` | Article title as released in the GDELT dataset. Preserve verbatim plus a checksum. |
| `seendate` | **GDELT/source observation evidence only.** Preserve it as first-class `source_observed_at` and as the original UTC literal. Do not map it to `published_at` or use it for cutoff eligibility under the current policy. GDELT explains elsewhere that only a minority of articles expose an exact publication timestamp and otherwise its article date is when GDELT first saw the article. |
| `domain` | Source identity/domain. It does not prove publisher ownership or licence. |
| `language` | Provider-reported article language. Preserve rather than infer. |
| `sourcecountry` | Provider-reported source country. It is source metadata, not a statement about the event/company location. |
| `socialimage` | Never download or normalize the referenced image. The checksum-bound raw GDELT response is retained verbatim for replay and may therefore contain the provider-returned field or image URL; normalized row evidence records only that the field was omitted by policy. |

Additional PAISA fields must be populated without inventing source facts:

- `published_at`: `null` unless independent captured source evidence provides a
  publisher publication timestamp. Never copy `seendate` here merely to satisfy
  a non-null constraint.
- `source_observed_at`: parsed GDELT `seendate`, together with the original
  literal value and its GDELT-observation semantics. This is audit provenance,
  never an eligibility input without a new reviewed policy version.
- `available_at`: the conservative PAISA eligibility time,
  `max(fetched_at, published_at, reviewed source-availability evidence)` over
  eligible timestamps that are present. GDELT `seendate` is not such evidence
  under the current policy. An archive record cannot be backdated to
  `source_observed_at` for model features.
- `fetched_at` / `imported_at`: PAISA acquisition/import time, not a historical
  availability time.
- `summary` / `body`: `null`. The selected provider does not fetch them.
- `original_timezone_evidence`: the unmodified GDELT timestamp representation;
  never infer the publisher's local timezone.
- provider subtype: raw/config evidence records `historical_api` for GDELT and
  `manual_import` for a licensed CSV. The normalized top-level ingestion
  `provider_mode` / `data_class` remains `real` for network-acquired GDELT
  metadata and `imported` for CSV evidence; imports are never relabelled live.
- `raw_evidence_reference`: immutable captured response/request reference with
  a SHA-256 checksum.
- `content_checksum`: checksum of the canonical stored metadata payload. It is
  not a checksum of an article body that PAISA never acquired.
- `article_id`: deterministic ID derived from the provider identity and
  canonical source identifier under the news data contract.
- OGDC link evidence: preserve the exact query/alias and the provider-full-text
  match mode. A GDELT search match does not justify relabelling broad financial
  news as OGDC-specific.
- duplicate relationship: calculate from canonical URL and stored metadata
  according to the news contract; retain every occurrence and point it to the
  canonical article rather than deleting provenance.

## Retained Dawn Business RSS evidence

This assessment does not reclassify the existing RSS run as licensed historical
coverage. It records what is already present so the audit trail is not mistaken
for permission:

- Run `NEWS-20261022-131242-5ddffd2d` fetched the official Dawn Business feed at
  `2026-10-22T13:12:43.088883Z`, without credentials.
- The immutable capture reported HTTP 200 and
  `application/xml; charset=UTF-8`; its channel identified `Dawn - Business`,
  declared `Copyright 2026`, and included a 60-minute TTL.
- The run retained 30 raw and 30 normalized records whose publisher-supplied
  publication timestamps span `2026-10-19T18:56:40Z` through
  `2026-10-22T13:07:31Z`. It is only a recent-feed snapshot, not an archive.
- PAISA did not follow article links for this run. The title, URL, description,
  body HTML, author, and time metadata came in the RSS response itself. This
  explains acquisition provenance; it does not expand Dawn's copyright terms.
- The raw XML, per-record JSON, normalized CSV, and local database are ignored
  by Git. Aggregate counts, hashes, and provenance may be documented without
  reproducing article text, but the underlying content must not be committed,
  published, shared beyond the permitted user, or exported as a dataset while
  permission remains unresolved.

### Permission finding

Access and reuse are separate questions. Dawn intentionally exposes the public
feed, so direct feed access is publisher-provided rather than article-page
scraping. The feed itself contains no reuse licence. Dawn's published terms say
that all website-content intellectual-property rights remain with Dawn or its
licensors, prohibit copying/downloading/republishing and similar uses except for
the user's own personal, non-commercial use, and require prior written
permission for any other use.

Accordingly, the personal/non-commercial exception is a narrow, conditional
basis for leaving this already-captured snapshot quarantined as an individual's
private FYP audit evidence. It is **not** an explicit academic-research licence
and cannot be assumed to cover a university, project team, public examiner
bundle, repository artifact, model-training corpus, indefinite automated
archive, or any redistribution. No published Dawn term found assigns separate
rights to titles, URLs, summaries, or RSS-supplied bodies. Those field-level
storage rights are therefore **unknown / needs written permission** outside the
narrow exception. The run remains `LICENSING_REVIEW_REQUIRED`; no new Dawn
acquisition is approved by this assessment. Dawn's terms identify the site
administrator for permissions, and its official contact page provides current
editorial/web contact routes.

## Manual-import permission gate

Before a real manual import, the operator must prepare and retain a sidecar or
licence manifest that records:

- provider and dataset identity;
- acquisition date and method;
- licence/terms URL or agreement reference and checksum;
- whether academic analysis is allowed;
- whether automation was involved and permitted;
- retention and redistribution limits;
- whether each of title, summary, URL, and body may be stored;
- attribution text;
- date coverage and timezone semantics;
- whether `published_at` and `available_at` are source-observed or inferred;
- raw evidence location and checksum; and
- a declaration that the input is real, not fixture, mock, synthetic, or test
  data.

The current CLI requires an explicit `--licensing-usage-note` that identifies
this review and carries it into provider configuration and every raw row. The
CLI cannot adjudicate the underlying agreement: the operator must not run the
import if any required permission is unknown. The compatibility constructor
uses `LICENSING_REVIEW_REQUIRED` when no note is supplied, and real-mode
coverage rejects that marker. Missing publisher timestamps remain missing;
they are not reconstructed from file dates, row order, fetch time, or trading
dates.

## Primary-source references

Accessed 2026-10-23:

- GDELT [Terms of Use](https://gdeltproject.org/about.html#termsofuse),
  [DOC 2.0 documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/),
  [search backfile update](https://blog.gdeltproject.org/doc-2-0-updates-1-5-year-searching-and-updated-mobile-interface/),
  [API rate-limiting guidance](https://blog.gdeltproject.org/ukraine-api-rate-limiting-web-ngrams-3-0/),
  and [article date semantics](https://blog.gdeltproject.org/announcing-the-gdelt-article-list-rss-feed/).
- NewsAPI.org [Everything endpoint](https://newsapi.org/docs/endpoints/everything),
  [pricing](https://newsapi.org/pricing), and
  [terms](https://newsapi.org/terms).
- Financial Modeling Prep [Stock News API](https://site.financialmodelingprep.com/developer/docs/stable/stock-news),
  [pricing](https://site.financialmodelingprep.com/pricing-plans), and
  [terms](https://site.financialmodelingprep.com/terms-of-service).
- Alpha Vantage [NEWS_SENTIMENT documentation](https://www.alphavantage.co/documentation/#news-sentiment),
  [support/standard limits](https://www.alphavantage.co/support/),
  [premium access](https://www.alphavantage.co/premium/), and
  [terms](https://www.alphavantage.co/terms_of_service/).
- NewsAPI.ai / Event Registry [product and archive](https://newsapi.ai/),
  [plans](https://newsapi.ai/plans), and
  [research/data-mining use](https://newsapi.ai/data-mining).
- NewsData.io [documentation](https://newsdata.io/documentation),
  [pricing](https://newsdata.io/pricing), and
  [terms](https://newsdata.io/terms).
- Google [Terms of Service](https://policies.google.com/terms) and
  [Google News discovery documentation](https://support.google.com/news/publisher-center/answer/9606634).
- PSX [OGDC company page](https://dps.psx.com.pk/company/OGDC),
  [daily downloads](https://dps.psx.com.pk/downloads), and
  [Terms of Use](https://www.psx.com.pk/psx/terms-of-use).
- OGDCL [news archive](https://ogdcl.com/media/news) and
  [financial-report archive](https://ogdcl.com/investors/financial-reports).
- Business Recorder [Terms of Use](https://www.brecorder.com/term-of-use/).
- Dawn [Business RSS feed](https://www.dawn.com/feeds/business),
  [Terms and Conditions of Use](https://www.dawn.com/terms/), and
  [contact page](https://www.dawn.com/contact/).
