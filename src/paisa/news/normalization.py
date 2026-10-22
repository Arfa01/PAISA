from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from paisa.news.models import (
    NEWS_SCHEMA_VERSION,
    NormalizedNewsArticle,
    RawNewsRecord,
    utc_iso,
)


_TRACKING_PARAMETERS = {
    "fbclid",
    "gclid",
    "dclid",
    "msclkid",
    "mc_cid",
    "mc_eid",
}
_OFFSET = re.compile(r"^[+-](\d{2}):?(\d{2})$")
_ISO_OFFSET_AT_END = re.compile(r"([+-]\d{2}:?\d{2}|Z)$", re.IGNORECASE)
_CLOCK_COMPONENT = re.compile(
    r"(?:[T\s]\d{1,2}:\d{2}|T\d{4}(?:\d{2})?)",
    re.IGNORECASE,
)


class NormalizationError(ValueError):
    """A raw record cannot satisfy the minimum news-record contract."""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() in {"script", "style"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def normalize_text(value: str | None) -> str | None:
    """Return Unicode-normalized plain text with deterministic whitespace."""

    if value is None:
        return None
    parser = _TextExtractor()
    try:
        parser.feed(unescape(str(value)))
        text = " ".join(parser.parts)
    except Exception:
        text = unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text or None


def _timezone_from_evidence(value: str) -> timezone | ZoneInfo:
    evidence = value.strip()
    if evidence.casefold() in {"z", "utc", "gmt"}:
        return timezone.utc
    match = _OFFSET.match(evidence)
    if match:
        hours, minutes = (int(part) for part in match.groups())
        if hours > 23 or minutes > 59:
            raise ValueError("Invalid UTC offset.")
        sign = -1 if evidence.startswith("-") else 1
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    try:
        return ZoneInfo(evidence)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown timezone: {value}") from exc


def infer_timezone_evidence(value: str | None) -> str | None:
    if not value:
        return None
    stripped = value.strip()
    match = _ISO_OFFSET_AT_END.search(stripped)
    if match:
        found = match.group(1)
        return "UTC" if found.upper() == "Z" else found
    tokens = stripped.rsplit(maxsplit=1)
    if len(tokens) == 2 and (
        tokens[-1].upper() in {"UTC", "GMT"} or _OFFSET.match(tokens[-1])
    ):
        return tokens[-1]
    return None


def _localize_iana_wall_time(
    value: datetime, zone: ZoneInfo
) -> tuple[datetime | None, str | None]:
    """Resolve a naive wall clock only when it identifies one real instant."""

    wall_time = value.replace(tzinfo=None)
    utc_candidates: set[datetime] = set()
    for fold in (0, 1):
        candidate = wall_time.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        round_trip = candidate.astimezone(zone).replace(tzinfo=None)
        if round_trip == wall_time:
            utc_candidates.add(candidate)

    if not utc_candidates:
        return None, "nonexistent_local_time"
    if len(utc_candidates) > 1:
        return None, "ambiguous_local_time"
    return utc_candidates.pop(), None


def parse_timestamp(
    value: str | datetime | None,
    *,
    timezone_evidence: str | None = None,
    field_name: str = "timestamp",
) -> tuple[datetime | None, tuple[str, ...]]:
    """Parse ISO/RFC timestamps without inventing a missing clock time or zone."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None, (f"{field_name}_missing",)

    parsed: datetime | None = None
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = value.strip()
        # ``datetime.fromisoformat`` accepts some date-plus-offset strings by
        # interpreting the offset as a clock time (for example,
        # ``2026-10-22+05:00``).  A publication timestamp is defensible only
        # when the source supplied an explicit clock component.
        if not _CLOCK_COMPONENT.search(raw):
            return None, (f"{field_name}_date_only",)
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(raw)
            except (TypeError, ValueError, OverflowError):
                return None, (f"{field_name}_invalid",)

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        if not timezone_evidence:
            return None, (f"{field_name}_missing_timezone",)
        try:
            evidence_zone = _timezone_from_evidence(timezone_evidence)
        except ValueError:
            return None, (f"{field_name}_invalid_timezone",)
        if isinstance(evidence_zone, ZoneInfo):
            localized, issue = _localize_iana_wall_time(parsed, evidence_zone)
            if issue is not None:
                return None, (f"{field_name}_{issue}",)
            assert localized is not None
            return localized, ()
        parsed = parsed.replace(tzinfo=evidence_zone)
    elif timezone_evidence:
        try:
            evidence_zone = _timezone_from_evidence(timezone_evidence)
        except ValueError:
            return None, (f"{field_name}_invalid_timezone",)
        instant = parsed.astimezone(timezone.utc)
        evidence_offset = instant.astimezone(evidence_zone).utcoffset()
        if evidence_offset != parsed.utcoffset():
            return None, (f"{field_name}_timezone_conflict",)
    return parsed.astimezone(timezone.utc), ()


def canonicalize_url(value: str | None) -> str | None:
    """Canonicalize HTTP(S) URLs conservatively for deterministic deduplication."""

    if value is None or not value.strip():
        return None
    try:
        parts = urlsplit(value.strip())
        if parts.scheme.casefold() not in {"http", "https"} or not parts.hostname:
            return None
        if parts.username is not None or parts.password is not None:
            return None
        scheme = parts.scheme.casefold()
        host = parts.hostname.encode("idna").decode("ascii").casefold()
        port = parts.port
        if port is not None and not (
            (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        ):
            host = f"{host}:{port}"
        path = re.sub(r"/{2,}", "/", parts.path or "/")
        if path != "/":
            path = path.rstrip("/") or "/"
        query_items = [
            (key, val)
            for key, val in parse_qsl(parts.query, keep_blank_values=True)
            if not key.casefold().startswith("utm_")
            and key.casefold() not in _TRACKING_PARAMETERS
        ]
        query = urlencode(sorted(query_items), doseq=True)
        return urlunsplit((scheme, host, path, query, ""))
    except (UnicodeError, ValueError):
        return None


def content_sha256(
    *,
    title: str | None,
    summary: str | None,
    body: str | None,
    source_identity: str | None = None,
    publication_context: str | None = None,
    record_identity: str | None = None,
) -> str:
    normalized_title = normalize_text(title) or ""
    normalized_summary = normalize_text(summary) or ""
    normalized_body = normalize_text(body) or ""
    payload = {
        "body": normalized_body,
        "summary": normalized_summary,
        "title": normalized_title,
    }
    if not normalized_summary and not normalized_body:
        stable_source = unicodedata.normalize(
            "NFKC", str(source_identity or "")
        ).strip()
        stable_context = unicodedata.normalize(
            "NFKC", str(publication_context or record_identity or "")
        ).strip()
        if not stable_source or not stable_context:
            raise ValueError(
                "Title-only content hashes require stable source and publication "
                "or record context."
            )
        payload["sparse_scope"] = {
            "context": stable_context,
            "source_identity_sha256": hashlib.sha256(
                stable_source.encode("utf-8")
            ).hexdigest(),
        }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_article_id(run_id: str, record: RawNewsRecord) -> str:
    identity = "\0".join((run_id, record.provider_id, record.raw_record_id))
    return f"NEWSART-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:24]}"


def normalize_record(
    record: RawNewsRecord,
    *,
    ingestion_run_id: str,
    raw_artifact_reference: str,
) -> NormalizedNewsArticle:
    title = normalize_text(record.title)
    summary = normalize_text(record.summary)
    body = normalize_text(record.body)
    if not any((title, summary, body)):
        raise NormalizationError("record_has_no_text_content")

    warnings = list(record.provider_warnings)
    timezone_evidence = record.published_timezone or infer_timezone_evidence(
        record.published_at_raw
    )
    published_at, published_warnings = parse_timestamp(
        record.published_at_raw,
        timezone_evidence=timezone_evidence,
        field_name="published_at",
    )
    warnings.extend(published_warnings)

    source_available_at: datetime | None = None
    if record.available_at_raw:
        available_timezone_evidence = (
            infer_timezone_evidence(record.available_at_raw) or timezone_evidence
        )
        source_available_at, available_warnings = parse_timestamp(
            record.available_at_raw,
            timezone_evidence=available_timezone_evidence,
            field_name="source_available_at",
        )
        warnings.extend(available_warnings)

    fetched_at = record.fetched_at.astimezone(timezone.utc)
    availability_candidates = [fetched_at]
    if published_at is not None:
        availability_candidates.append(published_at)
    if source_available_at is not None:
        availability_candidates.append(source_available_at)
    available_at = max(availability_candidates)

    canonical_url = canonicalize_url(record.source_url)
    if record.source_url and canonical_url is None:
        warnings.append("source_url_invalid")
    elif not record.source_url:
        warnings.append("source_url_missing")
    if title is None:
        warnings.append("title_missing")
    if body is None:
        warnings.append("body_missing")
    if title is not None and summary is None and body is None:
        warnings.append("content_hash_scoped_to_source_context")

    language = normalize_text(record.language)
    if language is not None:
        language = language.casefold().replace("_", "-")

    return NormalizedNewsArticle(
        schema_version=NEWS_SCHEMA_VERSION,
        article_id=make_article_id(ingestion_run_id, record),
        raw_record_id=record.raw_record_id,
        provider=record.provider_id,
        source_name=normalize_text(record.source_name) or record.provider_id,
        source_type=normalize_text(record.source_type) or "unknown",
        source_url=record.source_url.strip() if record.source_url else None,
        canonical_url=canonical_url,
        title=title,
        summary=summary,
        body=body,
        published_at=published_at,
        published_timezone=timezone_evidence,
        available_at=available_at,
        fetched_at=fetched_at,
        language=language,
        mentioned_tickers=(),
        mentioned_companies=(),
        mentioned_sectors=(),
        content_sha256=content_sha256(
            title=title,
            summary=summary,
            body=body,
            source_identity="\0".join(
                (record.provider_id.strip(), record.configured_source.strip())
            ),
            publication_context=utc_iso(published_at),
            record_identity=record.provider_record_id or record.raw_record_id,
        ),
        duplicate_of=None,
        ingestion_run_id=ingestion_run_id,
        raw_artifact_reference=raw_artifact_reference,
        normalization_warnings=tuple(dict.fromkeys(warnings)),
        data_class=record.data_class,
    )


def available_by(
    article: NormalizedNewsArticle, decision_cutoff: datetime
) -> bool:
    """Return whether the article was defensibly available by a model cutoff."""

    if decision_cutoff.tzinfo is None or decision_cutoff.utcoffset() is None:
        raise ValueError("decision_cutoff must be timezone-aware.")
    return article.available_at <= decision_cutoff.astimezone(timezone.utc)


def filter_available_articles(
    articles: list[NormalizedNewsArticle] | tuple[NormalizedNewsArticle, ...],
    decision_cutoff: datetime,
) -> list[NormalizedNewsArticle]:
    return [article for article in articles if available_by(article, decision_cutoff)]
