"""Controlled offline import of official PSX company-announcement metadata.

The importer deliberately has no HTTP client.  It reads exactly one operator-
supplied CSV after a checksum-bound permission policy has authorised the
``psx-announcement/manual_import`` path.  URLs are retained only as official
source references; they are never dereferenced, and PDF bytes or text are
never read by this component.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import csv
from datetime import date, datetime, timezone
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import re
import stat
from typing import Any
from urllib.parse import parse_qsl, urlsplit
from zoneinfo import ZoneInfo

from paisa.evidence.models import (
    EvidenceCapture,
    EvidenceFetchResult,
    EvidenceRecord,
)
from paisa.evidence.policy import (
    EvidencePermissionError,
    EvidencePermissionPolicy,
)


PROVIDER_ID = "psx-announcement"
PROVIDER_MODE = "manual_import"
SOURCE_NAME = "Pakistan Stock Exchange"
COMPANY_NAME = "Oil and Gas Development Company Limited"
CAPTURE_FILENAME = "psx_announcements.csv"
PUBLISHED_TIMEZONE = "Asia/Karachi"

_RAW_REFERENCE = f"capture_0001_{CAPTURE_FILENAME}"
_DATE_ONLY = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_TIMESTAMP_MINUTE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_TIMESTAMP_SECOND = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:Z|[+-][0-9]{2}:[0-9]{2})$"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_QUERY_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?(?:key|token)|token|secret|"
    r"password|credential|authorization|auth|signature|signed|session|"
    r"cookie|jwt)(?:$|[_-])",
    re.IGNORECASE,
)
_MAX_FIELD_LENGTH = 4_000
_DEFAULT_MAX_BYTES = 5 * 1024 * 1024
_DEFAULT_MAX_RECORDS = 10_000

_REQUIRED_COLUMNS = frozenset(
    {
        "announcement_id",
        "ticker",
        "company_name",
        "announcement_type",
        "title",
        "exchange_posted_at",
        "official_source_url",
    }
)
_OPTIONAL_COLUMNS = frozenset(
    {
        "event_category",
        "language",
        "pdf_reference",
        "pdf_url",
        "pdf_sha256",
    }
)
_ALLOWED_COLUMNS = _REQUIRED_COLUMNS | _OPTIONAL_COLUMNS


class PsxAnnouncementImportError(ValueError):
    """The supplied manual PSX evidence is incomplete or contradictory."""


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise PsxAnnouncementImportError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _column_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().casefold()).strip("_")


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_text(value: Any, *, field_name: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise PsxAnnouncementImportError(f"{field_name} cannot be blank")
    if "\r" in text or "\n" in text:
        raise PsxAnnouncementImportError(f"{field_name} must be one line")
    if len(text) > _MAX_FIELD_LENGTH:
        raise PsxAnnouncementImportError(f"{field_name} is too long")
    return text


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _official_psx_url(value: Any, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if any(character.isspace() or ord(character) < 0x20 for character in text):
        raise PsxAnnouncementImportError(
            f"{field_name} must not contain whitespace or control characters"
        )
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise PsxAnnouncementImportError(f"{field_name} is not a valid URL") from exc
    scheme = parsed.scheme.casefold()
    host = (parsed.hostname or "").casefold().rstrip(".")
    if scheme not in {"http", "https"} or not host:
        raise PsxAnnouncementImportError(
            f"{field_name} must be an official HTTP(S) PSX URL"
        )
    if host != "psx.com.pk" and not host.endswith(".psx.com.pk"):
        raise PsxAnnouncementImportError(
            f"{field_name} must use the official psx.com.pk domain"
        )
    if parsed.username is not None or parsed.password is not None:
        raise PsxAnnouncementImportError(f"{field_name} must not contain credentials")
    if port is not None and port not in {80, 443}:
        raise PsxAnnouncementImportError(f"{field_name} uses an unsupported port")
    if parsed.fragment:
        raise PsxAnnouncementImportError(f"{field_name} must not contain a fragment")
    if any(_SENSITIVE_QUERY_KEY.search(key) for key, _ in parse_qsl(parsed.query)):
        raise PsxAnnouncementImportError(
            f"{field_name} must not contain credential-like query parameters"
        )
    return text


def _normalized_words(value: str) -> str:
    normalized = value.casefold().replace("&", " and ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", normalized).split())


_EXACT_TYPE_CATEGORIES: dict[str, str] = {
    "financial results": "financial_results",
    "board meeting": "board_meeting",
    "material information": "material_information",
    "production update": "production_update",
    "dividend": "dividend",
    "annual report": "annual_report",
    "agm eogm": "AGM_EOGM",
    "agm": "AGM_EOGM",
    "eogm": "AGM_EOGM",
    "corporate action": "corporate_action",
}

_CATEGORY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "annual_report",
        ("annual report", "annual financial report"),
    ),
    (
        "board_meeting",
        ("board meeting", "meeting of the board of directors"),
    ),
    (
        "financial_results",
        (
            "financial result",
            "quarterly accounts",
            "half yearly accounts",
            "annual accounts",
            "financial statements",
        ),
    ),
    (
        "dividend",
        ("cash dividend", "interim dividend", "final dividend", "dividend"),
    ),
    (
        "AGM_EOGM",
        (
            "annual general meeting",
            "extraordinary general meeting",
            "agm",
            "eogm",
        ),
    ),
    (
        "production_update",
        (
            "production update",
            "operational update",
            "commencement of production",
            "oil discovery",
            "gas discovery",
            "hydrocarbon discovery",
            "well result",
        ),
    ),
    (
        "corporate_action",
        (
            "corporate action",
            "book closure",
            "right issue",
            "rights issue",
            "bonus share",
            "share split",
            "merger",
            "demerger",
        ),
    ),
    ("material_information", ("material information", "material event")),
)


def categorize_announcement(announcement_type: str, title: str) -> str:
    """Return one deterministic, non-sentiment event category."""

    normalized_type = _normalized_words(
        _required_text(announcement_type, field_name="announcement_type")
    )
    normalized_title = _normalized_words(_required_text(title, field_name="title"))
    exact = _EXACT_TYPE_CATEGORIES.get(normalized_type)
    if exact is not None:
        return exact
    combined = f"{normalized_type} {normalized_title}"
    for category, phrases in _CATEGORY_PATTERNS:
        if any(
            re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", combined)
            for phrase in phrases
        ):
            return category
    return "other"


def _parse_exchange_posted_at(
    value: Any,
    *,
    imported_at: datetime,
) -> tuple[datetime | None, date | None, str, str, tuple[str, ...]]:
    raw = _required_text(value, field_name="exchange_posted_at")
    if _DATE_ONLY.fullmatch(raw):
        try:
            source_date = date.fromisoformat(raw)
        except ValueError as exc:
            raise PsxAnnouncementImportError(
                "exchange_posted_at date is invalid"
            ) from exc
        import_date = imported_at.astimezone(ZoneInfo(PUBLISHED_TIMEZONE)).date()
        if source_date > import_date:
            raise PsxAnnouncementImportError(
                "exchange_posted_at date cannot be later than the import date"
            )
        return (
            None,
            source_date,
            "date_only",
            "official_exchange_date_conservative_next_session",
            ("date_only_exchange_posting_conservative_next_session",),
        )

    minute_match = _TIMESTAMP_MINUTE.fullmatch(raw)
    second_match = _TIMESTAMP_SECOND.fullmatch(raw)
    if minute_match is None and second_match is None:
        raise PsxAnnouncementImportError(
            "exchange_posted_at timestamp requires an explicit UTC offset and "
            "exact YYYY-MM-DDTHH:MM[:SS](Z|+/-HH:MM) syntax"
        )
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PsxAnnouncementImportError(
            "exchange_posted_at must be YYYY-MM-DD or an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PsxAnnouncementImportError(
            "exchange_posted_at timestamp requires an explicit UTC offset"
        )
    posted_at = parsed.astimezone(timezone.utc)
    if posted_at > imported_at:
        raise PsxAnnouncementImportError(
            "exchange_posted_at cannot be later than imported_at"
        )
    if second_match is not None:
        precision = "second"
        warnings: tuple[str, ...] = ()
    else:
        # Preserve the source's declared minute precision without inventing
        # seconds; the parsed datetime's zero seconds are not source evidence.
        precision = "minute"
        warnings = ()
    return (
        posted_at,
        None,
        precision,
        "official_exchange_posted_timestamp",
        warnings,
    )


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return the stable identity and content-relevant metadata for one file."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_nonsymlink(path: Path, *, max_bytes: int) -> bytes:
    """Read one path without following links and reject replacement or mutation."""

    try:
        path_before = path.lstat()
    except OSError as exc:
        raise PsxAnnouncementImportError("PSX announcement CSV is unreadable") from exc
    if not stat.S_ISREG(path_before.st_mode):
        raise PsxAnnouncementImportError(
            "PSX announcement CSV must be a regular non-symlink file"
        )
    if path_before.st_size > max_bytes:
        raise PsxAnnouncementImportError("PSX announcement CSV exceeds max_bytes")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PsxAnnouncementImportError("PSX announcement CSV is unreadable") from exc
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise PsxAnnouncementImportError(
                "PSX announcement CSV must be a regular non-symlink file"
            )
        if _file_identity(opened_before) != _file_identity(path_before):
            raise PsxAnnouncementImportError(
                "PSX announcement CSV changed before it was imported"
            )
        if opened_before.st_size > max_bytes:
            raise PsxAnnouncementImportError("PSX announcement CSV exceeds max_bytes")

        chunks: list[bytes] = []
        total = 0
        while total <= max_bytes:
            try:
                chunk = os.read(descriptor, min(64 * 1024, max_bytes + 1 - total))
            except OSError as exc:
                raise PsxAnnouncementImportError(
                    "PSX announcement CSV is unreadable"
                ) from exc
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > max_bytes:
            raise PsxAnnouncementImportError("PSX announcement CSV exceeds max_bytes")
        payload = b"".join(chunks)

        opened_after = os.fstat(descriptor)
        try:
            path_after = path.lstat()
        except OSError as exc:
            raise PsxAnnouncementImportError(
                "PSX announcement CSV changed while it was being imported"
            ) from exc
        if (
            not stat.S_ISREG(path_after.st_mode)
            or _file_identity(opened_after) != _file_identity(opened_before)
            or _file_identity(path_after) != _file_identity(opened_before)
            or len(payload) != opened_before.st_size
        ):
            raise PsxAnnouncementImportError(
                "PSX announcement CSV changed while it was being imported"
            )
        return payload
    finally:
        os.close(descriptor)


class PsxAnnouncementManualImporter:
    """Import permission-cleared PSX announcement metadata without networking."""

    provider_id = PROVIDER_ID
    provider_mode = PROVIDER_MODE
    data_class = "imported"

    def __init__(
        self,
        csv_path: str | Path,
        *,
        permission_policy: EvidencePermissionPolicy,
        authorization_attestation: str,
        authorization_reference: str,
        clock: Callable[[], datetime] = _utc_now,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_records: int = _DEFAULT_MAX_RECORDS,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(max_records, bool)
            or not isinstance(max_records, int)
            or max_records <= 0
        ):
            raise ValueError("max_records must be a positive integer")
        decision = permission_policy.require(
            PROVIDER_ID,
            PROVIDER_MODE,
            automated=False,
        )
        if decision.get("automated_access") is not False:
            raise EvidencePermissionError(
                "The PSX manual-import decision must explicitly deny automated access"
            )
        try:
            decision_id = _required_text(
                decision.get("decision_id"),
                field_name="permission decision_id",
            )
        except PsxAnnouncementImportError as exc:
            raise EvidencePermissionError(
                "The reviewed PSX permission decision requires a decision_id"
            ) from exc

        supplied_authorization: dict[str, str] = {}
        for field_name, supplied_value in (
            ("authorization_attestation", authorization_attestation),
            ("authorization_reference", authorization_reference),
        ):
            try:
                supplied_text = _required_text(
                    supplied_value,
                    field_name=field_name,
                )
            except PsxAnnouncementImportError as exc:
                raise EvidencePermissionError(
                    f"An explicit {field_name.replace('_', ' ')} is required"
                ) from exc
            if supplied_text.casefold() in {
                "licensing_review_required",
                "review_required",
                "permission_pending",
            }:
                raise EvidencePermissionError(
                    "A pending-review marker is not an authorization value"
                )
            supplied_authorization[field_name] = hashlib.sha256(
                supplied_text.encode("utf-8")
            ).hexdigest()

        for field_name, supplied_sha256 in supplied_authorization.items():
            policy_field = f"{field_name}_sha256"
            expected_sha256 = decision.get(policy_field)
            if not isinstance(expected_sha256, str) or not _SHA256.fullmatch(
                expected_sha256
            ):
                raise EvidencePermissionError(
                    "The reviewed PSX permission decision requires a valid "
                    f"{policy_field}"
                )
            if not hmac.compare_digest(expected_sha256, supplied_sha256):
                raise EvidencePermissionError(
                    f"{field_name.replace('_', ' ').capitalize()} does not match "
                    "the reviewed decision"
                )

        attestation_sha256 = supplied_authorization["authorization_attestation"]
        reference_sha256 = supplied_authorization["authorization_reference"]

        self.csv_path = Path(csv_path)
        self.permission_policy = permission_policy
        self.permission_policy_id = permission_policy.policy_id
        self.permission_status = str(decision["status"])
        self.permission_decision_id = decision_id
        self.authorization_attestation_sha256 = attestation_sha256
        self.authorization_reference_sha256 = reference_sha256
        self.clock = clock
        self.max_bytes = max_bytes
        self.max_records = max_records
        self.configured_source = f"manual-csv:{self.csv_path.name}"

    def fetch(self, *, fetched_at: datetime | None = None) -> EvidenceFetchResult:
        """Read the supplied CSV once and return typed, checksum-bound evidence."""

        payload = _read_regular_nonsymlink(self.csv_path, max_bytes=self.max_bytes)

        imported_at = _aware_utc(
            fetched_at or self.clock(), field_name="fetched_at"
        )
        input_sha256 = hashlib.sha256(payload).hexdigest()
        records = self._parse_records(payload, imported_at=imported_at)
        capture = EvidenceCapture(
            capture_id="psx-announcements-input",
            provider_id=PROVIDER_ID,
            configured_source=self.configured_source,
            filename=CAPTURE_FILENAME,
            fetched_at=imported_at,
            content=payload,
            source_url=None,
            content_type="text/csv",
            sha256=input_sha256,
            warnings=(
                "manual_import_only_no_network_or_pdf_retrieval",
            ),
        )
        date_only_count = sum(
            record.time_precision == "date_only" for record in records
        )
        timed_count = len(records) - date_only_count
        configured_source = {
            "configured_source": self.configured_source,
            "source_name": SOURCE_NAME,
            "evidence_type": "exchange_announcement",
            "provider_mode": PROVIDER_MODE,
            "automated_access": False,
            "permission_policy_id": self.permission_policy_id,
            "permission_decision_id": self.permission_decision_id,
            "permission_status": self.permission_status,
            "manual_permission_attestation_sha256": (
                self.authorization_attestation_sha256
            ),
            "manual_permission_reference_sha256": (
                self.authorization_reference_sha256
            ),
            "input_sha256": input_sha256,
            "input_size_bytes": len(payload),
            "pdf_handling": "reference_and_supplied_checksum_only_no_retrieval",
        }
        return EvidenceFetchResult(
            provider_id=PROVIDER_ID,
            provider_mode=PROVIDER_MODE,
            data_class="imported",
            fetched_at=imported_at,
            permission_policy_id=self.permission_policy_id,
            configured_sources=(configured_source,),
            partition_plan=(
                {
                    "partition_id": "manual-csv-input",
                    "acquisition_mode": PROVIDER_MODE,
                    "automated_access": False,
                    "input_sha256": input_sha256,
                    "maximum_bytes": self.max_bytes,
                    "maximum_records": self.max_records,
                },
            ),
            records=tuple(records),
            captures=(capture,),
            errors=(),
            warnings=(
                "Offline metadata import only; source URLs and PDF references were not fetched.",
                "Date-only announcements require conservative next-session mapping.",
                "Academic simulation; not financial advice.",
            ),
            counters={
                "input_rows": len(records),
                "accepted_records": len(records),
                "ogdc_records": len(records),
                "timed_records": timed_count,
                "date_only_records": date_only_count,
                "pdf_reference_records": sum(
                    record.pdf_reference is not None for record in records
                ),
            },
            timestamp_validation={
                "policy": (
                    "official timestamp maps by exact available_at; date-only "
                    "evidence has no invented timestamp and maps to the first "
                    "observed session strictly after its source date"
                ),
                "timezone": PUBLISHED_TIMEZONE,
                "timed_records": timed_count,
                "date_only_records": date_only_count,
                "future_timestamp_rejections": 0,
            },
            effective_config={
                "automated_access": False,
                "max_bytes": self.max_bytes,
                "max_records": self.max_records,
                "pdf_retrieval": "disabled",
                "source_url_retrieval": "disabled",
                "manual_permission_attestation_sha256": (
                    self.authorization_attestation_sha256
                ),
                "manual_permission_reference_sha256": (
                    self.authorization_reference_sha256
                ),
            },
        )

    def _parse_records(
        self,
        payload: bytes,
        *,
        imported_at: datetime,
    ) -> list[EvidenceRecord]:
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PsxAnnouncementImportError(
                "PSX announcement CSV must be UTF-8"
            ) from exc
        try:
            reader = csv.DictReader(
                io.StringIO(text, newline=""),
                restkey="__extra_columns__",
                strict=True,
            )
            if reader.fieldnames is None:
                raise PsxAnnouncementImportError(
                    "PSX announcement CSV is missing its header"
                )
            columns = [_column_key(name) for name in reader.fieldnames]
            if any(not column for column in columns):
                raise PsxAnnouncementImportError(
                    "PSX announcement CSV contains a blank column"
                )
            if len(columns) != len(set(columns)):
                raise PsxAnnouncementImportError(
                    "PSX announcement CSV columns collide after normalization"
                )
            missing = sorted(_REQUIRED_COLUMNS - set(columns))
            unsupported = sorted(set(columns) - _ALLOWED_COLUMNS)
            if missing:
                raise PsxAnnouncementImportError(
                    "PSX announcement CSV is missing required columns: "
                    + ", ".join(missing)
                )
            if unsupported:
                raise PsxAnnouncementImportError(
                    "PSX announcement CSV has unsupported columns: "
                    + ", ".join(unsupported)
                )

            records: list[EvidenceRecord] = []
            seen_ids: set[str] = set()
            for source_index, source_row in enumerate(reader, start=2):
                if source_row.get("__extra_columns__"):
                    raise PsxAnnouncementImportError(
                        f"PSX announcement CSV has extra values at row {source_index}"
                    )
                raw_row = {
                    columns[index]: source_row.get(original)
                    for index, original in enumerate(reader.fieldnames)
                }
                input_row_sha256 = _canonical_json_sha256(raw_row)
                row = {
                    columns[index]: _optional_text(source_row.get(original))
                    for index, original in enumerate(reader.fieldnames)
                }
                if not any(row.values()):
                    continue
                if len(records) >= self.max_records:
                    raise PsxAnnouncementImportError(
                        "PSX announcement CSV exceeds max_records"
                    )
                announcement_id = _required_text(
                    row.get("announcement_id"),
                    field_name=f"announcement_id row {source_index}",
                )
                identity_key = announcement_id.casefold()
                if identity_key in seen_ids:
                    raise PsxAnnouncementImportError(
                        f"announcement_id is duplicated at row {source_index}"
                    )
                seen_ids.add(identity_key)
                ticker = _required_text(
                    row.get("ticker"), field_name=f"ticker row {source_index}"
                ).upper()
                if ticker != "OGDC":
                    raise PsxAnnouncementImportError(
                        f"ticker row {source_index} must exactly equal OGDC"
                    )
                company_name = _required_text(
                    row.get("company_name"),
                    field_name=f"company_name row {source_index}",
                )
                announcement_type = _required_text(
                    row.get("announcement_type"),
                    field_name=f"announcement_type row {source_index}",
                )
                title = _required_text(
                    row.get("title"), field_name=f"title row {source_index}"
                )
                source_url = _official_psx_url(
                    row.get("official_source_url"),
                    field_name=f"official_source_url row {source_index}",
                )
                pdf_url_value = row.get("pdf_url")
                pdf_reference_value = row.get("pdf_reference")
                if (
                    pdf_url_value is not None
                    and pdf_reference_value is not None
                    and pdf_url_value != pdf_reference_value
                ):
                    raise PsxAnnouncementImportError(
                        f"PDF URL/reference disagree at row {source_index}"
                    )
                raw_pdf_reference = pdf_url_value or pdf_reference_value
                pdf_reference = (
                    _official_psx_url(
                        raw_pdf_reference,
                        field_name=f"pdf_reference row {source_index}",
                    )
                    if raw_pdf_reference is not None
                    else None
                )
                pdf_sha256 = row.get("pdf_sha256")
                if pdf_sha256 is not None:
                    pdf_sha256 = pdf_sha256.casefold()
                    if not _SHA256.fullmatch(pdf_sha256):
                        raise PsxAnnouncementImportError(
                            f"pdf_sha256 is invalid at row {source_index}"
                        )
                    if pdf_reference is None:
                        raise PsxAnnouncementImportError(
                            f"pdf_sha256 requires a PDF reference at row {source_index}"
                        )
                pdf_checksum_verification = (
                    "operator_supplied_unverified_claim"
                    if pdf_sha256 is not None
                    else "not_supplied"
                )
                pdf_warnings = (
                    ("pdf_sha256_operator_supplied_unverified_claim",)
                    if pdf_sha256 is not None
                    else ()
                )

                (
                    posted_at,
                    posted_date,
                    time_precision,
                    availability_basis,
                    time_warnings,
                ) = _parse_exchange_posted_at(
                    row.get("exchange_posted_at"), imported_at=imported_at
                )
                category = categorize_announcement(announcement_type, title)
                supplied_category = row.get("event_category")
                if supplied_category is not None and supplied_category != category:
                    raise PsxAnnouncementImportError(
                        f"event_category disagrees with deterministic mapping at row {source_index}"
                    )
                posted_evidence = (
                    posted_at.isoformat().replace("+00:00", "Z")
                    if posted_at is not None
                    else posted_date.isoformat() if posted_date is not None else None
                )
                content_payload = {
                    "announcement_id": announcement_id,
                    "ticker": ticker,
                    "company_name": company_name,
                    "announcement_type": announcement_type,
                    "event_category": category,
                    "title": title,
                    "exchange_posted": posted_evidence,
                    "exchange_posted_at_raw": row["exchange_posted_at"],
                    "time_precision": time_precision,
                    "official_source_url": source_url,
                    "pdf_reference": pdf_reference,
                    "pdf_sha256": pdf_sha256,
                    "input_row_sha256": input_row_sha256,
                }
                evidence_digest = hashlib.sha256(
                    f"{PROVIDER_ID}\0{announcement_id}".encode("utf-8")
                ).hexdigest()[:24]
                records.append(
                    EvidenceRecord(
                        evidence_id=f"EVIDPSX-{evidence_digest}",
                        evidence_type="exchange_announcement",
                        provider_id=PROVIDER_ID,
                        provider_mode=PROVIDER_MODE,
                        data_class="imported",
                        source_name=SOURCE_NAME,
                        source_url=source_url,
                        source_record_id=announcement_id,
                        ticker=ticker,
                        company_name=company_name,
                        title=title,
                        source_published_at=posted_at,
                        source_published_date=posted_date,
                        available_at=posted_at,
                        fetched_at=imported_at,
                        imported_at=imported_at,
                        historical_availability_basis=availability_basis,
                        time_precision=time_precision,  # type: ignore[arg-type]
                        contemporaneous_source_id=announcement_id,
                        contemporaneous_batch_at=None,
                        permission_policy_id=self.permission_policy_id,
                        raw_evidence_reference=(
                            f"{_RAW_REFERENCE}#row={source_index}"
                        ),
                        content_sha256=_canonical_json_sha256(content_payload),
                        language=(row.get("language") or "en").casefold(),
                        event_category=category,
                        announcement_type=announcement_type,
                        pdf_reference=pdf_reference,
                        pdf_sha256=pdf_sha256,
                        normalization_warnings=(*time_warnings, *pdf_warnings),
                        entity_match_evidence=(
                            {
                                "field": "ticker",
                                "matched_value": "OGDC",
                                "match_method": "official_exact_ticker",
                            },
                        ),
                        source_metadata={
                            "source_mode": PROVIDER_MODE,
                            "automated_access": False,
                            "exchange_posted_at_raw": row[
                                "exchange_posted_at"
                            ],
                            "permission_decision_id": self.permission_decision_id,
                            "manual_permission_attestation_sha256": (
                                self.authorization_attestation_sha256
                            ),
                            "manual_permission_reference_sha256": (
                                self.authorization_reference_sha256
                            ),
                            "input_sha256": hashlib.sha256(payload).hexdigest(),
                            "input_row_number": source_index,
                            "input_row_sha256": input_row_sha256,
                            "pdf_handling": (
                                "reference_and_supplied_checksum_only_no_retrieval"
                            ),
                            "pdf_checksum_verification": (
                                pdf_checksum_verification
                            ),
                        },
                    )
                )
        except csv.Error as exc:
            raise PsxAnnouncementImportError(
                "PSX announcement CSV could not be parsed"
            ) from exc
        if not records:
            raise PsxAnnouncementImportError(
                "PSX announcement CSV contains no evidence records"
            )
        return records


def replay_psx_announcement_capture(
    content: bytes,
    *,
    imported_at: datetime,
    permission_policy_id: str,
    permission_decision_id: str,
    authorization_attestation_sha256: str,
    authorization_reference_sha256: str,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_records: int = _DEFAULT_MAX_RECORDS,
) -> tuple[EvidenceRecord, ...]:
    """Reparse one sealed PSX CSV using the importer normalization contract.

    This helper performs no filesystem or network access.  Its explicit policy
    arguments are expected to come from the independently sealed provider
    configuration (and, during publication, the reloaded repository policy).
    Equality with its returned records therefore binds every normalized field
    to the captured CSV bytes, cited row, timestamp grammar, and permission
    provenance.
    """

    if not isinstance(content, bytes):
        raise TypeError("PSX replay content must be bytes")
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if len(content) > max_bytes:
        raise PsxAnnouncementImportError("PSX announcement CSV exceeds max_bytes")
    if (
        isinstance(max_records, bool)
        or not isinstance(max_records, int)
        or max_records <= 0
    ):
        raise ValueError("max_records must be a positive integer")
    policy_id = _required_text(
        permission_policy_id, field_name="permission_policy_id"
    )
    decision_id = _required_text(
        permission_decision_id, field_name="permission_decision_id"
    )
    for field_name, digest in (
        ("authorization_attestation_sha256", authorization_attestation_sha256),
        ("authorization_reference_sha256", authorization_reference_sha256),
    ):
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise PsxAnnouncementImportError(f"{field_name} is invalid")

    # Reuse the sole normalization implementation without re-running operator
    # authorization or touching the original path.  Only the attributes read
    # by _parse_records are populated.
    parser = object.__new__(PsxAnnouncementManualImporter)
    parser.max_records = max_records
    parser.permission_policy_id = policy_id
    parser.permission_decision_id = decision_id
    parser.authorization_attestation_sha256 = authorization_attestation_sha256
    parser.authorization_reference_sha256 = authorization_reference_sha256
    return tuple(
        parser._parse_records(
            content,
            imported_at=_aware_utc(imported_at, field_name="imported_at"),
        )
    )


__all__ = [
    "CAPTURE_FILENAME",
    "COMPANY_NAME",
    "PROVIDER_ID",
    "PROVIDER_MODE",
    "PsxAnnouncementImportError",
    "PsxAnnouncementManualImporter",
    "categorize_announcement",
    "replay_psx_announcement_capture",
]
