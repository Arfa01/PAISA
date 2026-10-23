"""Bounded acquisition of metadata-only GDELT GKG 2.1 raw archives.

This provider is intentionally independent from the GDELT DOC adapter.  It
downloads only explicitly configured official GKG archive files, never follows
``V2DOCUMENTIDENTIFIER`` values, and treats the timestamp embedded in a valid
``GKGRECORDID`` as the contemporaneous availability evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import unicodedata
from typing import Any
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit, urlunsplit
from uuid import uuid4
import zipfile

import requests

from paisa.evidence.models import (
    EvidenceCapture,
    EvidenceError,
    EvidenceFetchResult,
    EvidenceRecord,
    utc_iso,
)
from paisa.evidence.policy import (
    EvidencePermissionError,
    EvidencePermissionPolicy,
)


GDELT_GKG_INDEX_URL = "https://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
GDELT_GKG_ARCHIVE_HOST = "data.gdeltproject.org"
GDELT_GKG_ARCHIVE_PREFIX = "/gdeltv2/"
GDELT_GKG_PROVIDER_ID = "gdelt-gkg-2.1"
GDELT_GKG_PROVIDER_MODE = "raw_archive"
GDELT_GKG_HISTORICAL_BASIS = "gdelt_gkg_batch_timestamp"
GDELT_WEB_SOURCE_COLLECTION_ID = "1"

OGDC_TICKER = "OGDC"
OGDC_COMPANY_NAME = "Oil and Gas Development Company Limited"
OGDC_ALIASES = (
    "Oil and Gas Development Company Limited",
    "Oil & Gas Development Company Limited",
    "Oil and Gas Development Company",
    "Oil & Gas Development Company",
    "OGDCL",
    "OGDC",
)

GKG_V21_COLUMNS = (
    "GKGRECORDID",
    "V2.1DATE",
    "V2SOURCECOLLECTIONIDENTIFIER",
    "V2SOURCECOMMONNAME",
    "V2DOCUMENTIDENTIFIER",
    "V1COUNTS",
    "V2.1COUNTS",
    "V1THEMES",
    "V2ENHANCEDTHEMES",
    "V1LOCATIONS",
    "V2ENHANCEDLOCATIONS",
    "V1PERSONS",
    "V2ENHANCEDPERSONS",
    "V1ORGANIZATIONS",
    "V2ENHANCEDORGANIZATIONS",
    "V1.5TONE",
    "V2.1ENHANCEDDATES",
    "V2GCAM",
    "V2.1SHARINGIMAGE",
    "V2.1RELATEDIMAGES",
    "V2.1SOCIALIMAGEEMBEDS",
    "V2.1SOCIALVIDEOEMBEDS",
    "V2.1QUOTATIONS",
    "V2.1ALLNAMES",
    "V2.1AMOUNTS",
    "V2.1TRANSLATIONINFO",
    "V2EXTRASXML",
)

_GKG_RECORD_ID = re.compile(r"^(?P<batch>[0-9]{14})-(?P<sequence>[0-9]+)$")
_ARCHIVE_NAME = re.compile(r"^(?P<batch>[0-9]{14})\.gkg\.csv\.zip$")
_MEMBER_NAME = re.compile(r"^(?P<batch>[0-9]{14})\.gkg\.csv$")
_PARTITION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?(?:key|token)|token|secret|password|"
    r"credential|authorization|auth|signature|signed|session|cookie|jwt)(?:$|[_-])",
    re.IGNORECASE,
)
_TRACKING_QUERY_KEYS = {
    "dclid",
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}
_DEFAULT_USER_AGENT = (
    "PAISA-FYP/0.1 academic GDELT GKG metadata research; contact repository owner"
)
_READ_CHUNK_BYTES = 64 * 1024
_MAX_FILES_HARD = 512
_MAX_COMPRESSED_BYTES_HARD = 2 * 1024 * 1024 * 1024
_MAX_DECOMPRESSED_BYTES_HARD = 8 * 1024 * 1024 * 1024
_MAX_RECORDS_HARD = 10_000_000
_MAX_ROW_BYTES_HARD = 16 * 1024 * 1024
_MAX_FIELD_CHARACTERS_HARD = 4 * 1024 * 1024
_CHECKPOINT_SCHEMA_VERSION = "1.0"


class _GkgRejected(ValueError):
    """Deterministic provider rejection safe for persisted error evidence."""

    def __init__(self, error_type: str, message: str):
        super().__init__(message)
        self.error_type = error_type
        self.safe_message = message


def _validated_gdelt_permission(
    policy: EvidencePermissionPolicy,
) -> Mapping[str, Any]:
    """Require a checksum-bound policy and its exact automated GKG decision."""

    if not isinstance(policy, EvidencePermissionPolicy):
        raise EvidencePermissionError(
            "GDELT GKG acquisition requires a validated evidence permission policy"
        )
    if (
        policy.schema_version != "1.0"
        or not _PARTITION_ID.fullmatch(str(policy.policy_id))
        or not _SHA256.fullmatch(str(policy.policy_sha256))
        or not policy.assessment_references
    ):
        raise EvidencePermissionError(
            "GDELT GKG permission policy is not checksum-bound"
        )
    for reference in policy.assessment_references:
        reference_path = (
            str(reference.get("path") or "")
            if isinstance(reference, Mapping)
            else ""
        )
        parsed_path = PurePosixPath(reference_path)
        if (
            not isinstance(reference, Mapping)
            or not reference_path
            or parsed_path.is_absolute()
            or ".." in parsed_path.parts
            or "\\" in reference_path
            or not _SHA256.fullmatch(str(reference.get("sha256") or ""))
        ):
            raise EvidencePermissionError(
                "GDELT GKG permission assessment is not checksum-bound"
            )
    try:
        decision = policy.require(
            GDELT_GKG_PROVIDER_ID,
            GDELT_GKG_PROVIDER_MODE,
            automated=True,
        )
    except EvidencePermissionError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidencePermissionError(
            "GDELT GKG permission decision is invalid"
        ) from exc
    decision_id = str(decision.get("decision_id") or "").strip()
    if (
        decision.get("provider_id") != GDELT_GKG_PROVIDER_ID
        or decision.get("provider_mode") != GDELT_GKG_PROVIDER_MODE
        or decision.get("status") != "PERMITTED"
        or decision.get("automated_access") is not True
        or decision.get("publisher_page_fetching") is not False
        or not decision_id
        or "\n" in decision_id
        or "\r" in decision_id
    ):
        raise EvidencePermissionError(
            "GDELT GKG requires the exact permitted metadata-only decision"
        )
    return decision


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_timestamp14(value: str, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{14}", value):
        raise ValueError(f"{field_name} must contain exactly 14 decimal digits")
    try:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ValueError(f"{field_name} contains an invalid UTC timestamp") from exc


def parse_gkg_record_id_timestamp(value: str) -> datetime:
    """Parse the archive-batch timestamp from an exact GKG 2.1 record ID."""

    if not isinstance(value, str):
        raise ValueError("GKGRECORDID must be text")
    match = _GKG_RECORD_ID.fullmatch(value)
    if match is None:
        raise ValueError("GKGRECORDID is malformed")
    return _parse_timestamp14(match.group("batch"), field_name="GKGRECORDID")


def parse_v21_date(value: str) -> datetime:
    """Parse a nonzero GKG ``V2.1DATE`` UTC timestamp."""

    if not isinstance(value, str):
        raise ValueError("V2.1DATE must be text")
    text = value
    if text in {"", "0", "00000000000000"}:
        raise ValueError("V2.1DATE is zero or missing")
    return _parse_timestamp14(text, field_name="V2.1DATE")


def _archive_filename(url: str) -> tuple[str, datetime]:
    try:
        parsed = urlsplit(str(url).strip())
        parsed.port
    except ValueError as exc:
        raise ValueError("GDELT GKG archive URL is invalid") from exc
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").casefold() != GDELT_GKG_ARCHIVE_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("GDELT GKG archive URL must use the fixed official HTTPS host")
    if not parsed.path.startswith(GDELT_GKG_ARCHIVE_PREFIX):
        raise ValueError("GDELT GKG archive path is outside the official archive root")
    filename = PurePosixPath(parsed.path).name
    expected_path = f"{GDELT_GKG_ARCHIVE_PREFIX}{filename}"
    if parsed.path != expected_path:
        raise ValueError("GDELT GKG archive path must identify one root archive file")
    match = _ARCHIVE_NAME.fullmatch(filename)
    if match is None:
        raise ValueError("GDELT GKG archive filename is invalid")
    return filename, _parse_timestamp14(
        match.group("batch"), field_name="archive filename"
    )


def _bounded_positive_integer(value: Any, *, field_name: str, maximum: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
        or value > maximum
    ):
        raise ValueError(f"{field_name} must be an integer in [1, {maximum}]")
    return value


def _bounded_nonnegative_number(
    value: Any, *, field_name: str, maximum: float
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be numeric")
    selected = float(value)
    if not math.isfinite(selected) or selected < 0 or selected > maximum:
        raise ValueError(f"{field_name} must be finite and in [0, {maximum}]")
    return selected


@dataclass(frozen=True, slots=True)
class GdeltGkgPartition:
    """One explicit official 15-minute GKG archive partition."""

    partition_id: str
    archive_url: str
    expected_size_bytes: int | None = None
    expected_md5: str | None = None
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        if not _PARTITION_ID.fullmatch(self.partition_id):
            raise ValueError("GKG partition_id is invalid")
        _archive_filename(self.archive_url)
        if self.expected_size_bytes is not None:
            _bounded_positive_integer(
                self.expected_size_bytes,
                field_name="expected_size_bytes",
                maximum=_MAX_COMPRESSED_BYTES_HARD,
            )
        if self.expected_md5 is not None and not _MD5.fullmatch(self.expected_md5):
            raise ValueError("expected_md5 must be lowercase hexadecimal")
        if self.expected_sha256 is not None and not _SHA256.fullmatch(
            self.expected_sha256
        ):
            raise ValueError("expected_sha256 must be lowercase hexadecimal")

    @property
    def filename(self) -> str:
        return _archive_filename(self.archive_url)[0]

    @property
    def batch_at(self) -> datetime:
        return _archive_filename(self.archive_url)[1]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GdeltGkgPartition":
        allowed = {
            "partition_id",
            "archive_url",
            "expected_size_bytes",
            "expected_md5",
            "expected_sha256",
            "enabled",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(
                "Unsupported GKG partition fields: "
                + ", ".join(sorted(str(item) for item in unexpected))
            )
        return cls(
            partition_id=str(value.get("partition_id") or "").strip(),
            archive_url=str(value.get("archive_url") or "").strip(),
            expected_size_bytes=value.get("expected_size_bytes"),
            expected_md5=(
                str(value["expected_md5"]).strip()
                if value.get("expected_md5") is not None
                else None
            ),
            expected_sha256=(
                str(value["expected_sha256"]).strip()
                if value.get("expected_sha256") is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition_id": self.partition_id,
            "archive_url": self.archive_url,
            "archive_filename": self.filename,
            "batch_at": utc_iso(self.batch_at),
            "expected_size_bytes": self.expected_size_bytes,
            "expected_md5": self.expected_md5,
            "expected_sha256": self.expected_sha256,
        }


def _normalized_phrase(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _organization_names(value: str) -> tuple[str, ...]:
    names: list[str] = []
    for raw_item in value.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        head, separator, tail = item.rpartition(",")
        if separator and tail.strip().isdigit() and head.strip():
            item = head.strip()
        names.append(item)
    return tuple(names)


def _alias_match(value: str) -> tuple[str, str] | None:
    normalized = _normalized_phrase(value)
    for alias in sorted(OGDC_ALIASES, key=len, reverse=True):
        if normalized == _normalized_phrase(alias):
            method = "exact" if value.casefold() == alias.casefold() else "normalized_exact"
            return alias, method
    return None


def _url_alias_matches(value: str) -> list[dict[str, str]]:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return []
    searchable = _normalized_phrase(
        " ".join(
            (
                parsed.hostname or "",
                unquote(parsed.path),
                unquote(parsed.query),
            )
        )
    )
    matches: list[dict[str, str]] = []
    for alias in sorted(OGDC_ALIASES, key=len, reverse=True):
        normalized_alias = _normalized_phrase(alias)
        pattern = rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])"
        if re.search(pattern, searchable):
            matches.append(
                {
                    "field": "V2DOCUMENTIDENTIFIER",
                    "alias": alias,
                    "matched_value": normalized_alias,
                    "match_method": "normalized_token_boundary",
                }
            )
            break
    return matches


def match_ogdc_aliases(
    *,
    v2_organizations: str,
    document_identifier: str | None = None,
    source_common_name: str | None = None,
) -> tuple[Mapping[str, str], ...]:
    """Return exact, auditable OGDC matches without substring matching."""

    matches: list[dict[str, str]] = []
    for organization in _organization_names(v2_organizations):
        matched = _alias_match(organization)
        if matched is None:
            continue
        alias, method = matched
        matches.append(
            {
                "field": "V2ORGANIZATIONS",
                "alias": alias,
                "matched_value": organization,
                "match_method": method,
            }
        )
    if document_identifier:
        matches.extend(_url_alias_matches(document_identifier))
    if source_common_name:
        matched = _alias_match(source_common_name)
        if matched is not None:
            alias, method = matched
            matches.append(
                {
                    "field": "V2SOURCECOMMONNAME",
                    "alias": alias,
                    "matched_value": source_common_name,
                    "match_method": method,
                }
            )
    unique: list[Mapping[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for item in matches:
        key = (
            item["field"],
            item["alias"],
            item["matched_value"],
            item["match_method"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(unique)


def _safe_document_url(value: str) -> str | None:
    try:
        parsed = urlsplit(value.strip())
        parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    host = parsed.hostname.casefold()
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    query = urlencode(
        [
            (key, item)
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
            if not _SENSITIVE_KEY.search(key)
            and not key.casefold().startswith("utm_")
            and key.casefold() not in _TRACKING_QUERY_KEYS
        ]
    )
    return urlunsplit((parsed.scheme.casefold(), host, parsed.path, query, ""))


def _published_capture_name(ordinal: int, filename: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not safe:
        safe = "capture.bin"
    return f"capture_{ordinal:04d}_{safe[:180]}"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _wire_chunks(response: Any) -> Iterable[bytes]:
    raw = getattr(response, "raw", None)
    stream = getattr(raw, "stream", None)
    if callable(stream):
        yield from stream(_READ_CHUNK_BYTES, decode_content=False)
        return
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        yield from iterator(chunk_size=_READ_CHUNK_BYTES)
        return
    yield bytes(response.content)


def _read_bounded_response(
    response: Any,
    *,
    per_file_limit: int,
    remaining_total_limit: int,
) -> bytes:
    headers = {str(key).casefold(): value for key, value in response.headers.items()}
    encoding = str(headers.get("content-encoding") or "identity").strip().casefold()
    if encoding not in {"", "identity"}:
        raise _GkgRejected(
            "unsupported_content_encoding",
            "GDELT GKG archive used an unsupported HTTP content encoding",
        )
    declared = headers.get("content-length")
    if declared is not None:
        try:
            declared_size = int(str(declared))
        except ValueError as exc:
            raise _GkgRejected(
                "invalid_response_metadata",
                "GDELT GKG Content-Length is invalid",
            ) from exc
        if declared_size < 0 or declared_size > min(
            per_file_limit, remaining_total_limit
        ):
            raise _GkgRejected(
                "compressed_byte_limit_exceeded",
                "GDELT GKG archive exceeds the configured compressed-byte limit",
            )
    chunks: list[bytes] = []
    size = 0
    for value in _wire_chunks(response):
        chunk = bytes(value)
        size += len(chunk)
        if size > per_file_limit or size > remaining_total_limit:
            raise _GkgRejected(
                "compressed_byte_limit_exceeded",
                "GDELT GKG archive exceeds the configured compressed-byte limit",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _validate_archive_checksums(
    payload: bytes, partition: GdeltGkgPartition
) -> tuple[str, str]:
    observed_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        partition.expected_size_bytes is not None
        and len(payload) != partition.expected_size_bytes
    ):
        raise _GkgRejected(
            "archive_size_mismatch",
            "GDELT GKG archive size differs from the partition plan",
        )
    if partition.expected_md5 is not None and observed_md5 != partition.expected_md5:
        raise _GkgRejected(
            "archive_checksum_mismatch",
            "GDELT GKG archive MD5 differs from the partition plan",
        )
    if (
        partition.expected_sha256 is not None
        and observed_sha256 != partition.expected_sha256
    ):
        raise _GkgRejected(
            "archive_checksum_mismatch",
            "GDELT GKG archive SHA-256 differs from the partition plan",
        )
    return observed_md5, observed_sha256


def _read_safe_zip_member(
    payload: bytes,
    *,
    partition: GdeltGkgPartition,
    per_file_limit: int,
    remaining_total_limit: int,
) -> tuple[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            members = archive.infolist()
            if len(members) != 1:
                raise _GkgRejected(
                    "unsafe_archive",
                    "GDELT GKG ZIP must contain exactly one regular member",
                )
            member = members[0]
            name = member.filename
            path = PurePosixPath(name)
            expected_name = partition.filename.removesuffix(".zip")
            match = _MEMBER_NAME.fullmatch(name)
            unix_mode = member.external_attr >> 16
            if (
                path.name != name
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in name
                or member.is_dir()
                or stat.S_ISLNK(unix_mode)
                or member.flag_bits & 0x1
                or match is None
                or name != expected_name
            ):
                raise _GkgRejected(
                    "unsafe_archive",
                    "GDELT GKG ZIP member identity is unsafe or unexpected",
                )
            if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise _GkgRejected(
                    "unsafe_archive",
                    "GDELT GKG ZIP compression method is unsupported",
                )
            if member.file_size > per_file_limit or member.file_size > remaining_total_limit:
                raise _GkgRejected(
                    "decompressed_byte_limit_exceeded",
                    "GDELT GKG archive exceeds the configured decompressed-byte limit",
                )
            output: list[bytes] = []
            size = 0
            with archive.open(member, "r") as source:
                while True:
                    chunk = source.read(_READ_CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > per_file_limit or size > remaining_total_limit:
                        raise _GkgRejected(
                            "decompressed_byte_limit_exceeded",
                            "GDELT GKG archive exceeds the configured decompressed-byte limit",
                        )
                    output.append(chunk)
            if size != member.file_size:
                raise _GkgRejected(
                    "invalid_archive",
                    "GDELT GKG ZIP member size differs after decompression",
                )
            return name, b"".join(output)
    except _GkgRejected:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, EOFError) as exc:
        raise _GkgRejected(
            "invalid_archive", "GDELT GKG response is not a valid ZIP archive"
        ) from exc


def _bounded_field(value: str, *, field_name: str, maximum: int) -> str:
    if len(value) > maximum:
        raise _GkgRejected(
            "field_limit_exceeded",
            f"GDELT GKG field exceeds the configured limit: {field_name}",
        )
    return value


def _parse_themes(value: str) -> tuple[str, ...]:
    themes: list[str] = []
    for item in value.split(";"):
        text = item.strip()
        if not text:
            continue
        theme = text.split(",", 1)[0].strip()
        if theme:
            themes.append(theme)
    return tuple(dict.fromkeys(themes))


def _parse_tone(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    numbers: list[float] = []
    for item in value.split(","):
        try:
            number = float(item)
        except ValueError as exc:
            raise _GkgRejected(
                "invalid_tone", "GDELT GKG tone metadata is invalid"
            ) from exc
        if not math.isfinite(number):
            raise _GkgRejected(
                "invalid_tone", "GDELT GKG tone metadata is invalid"
            )
        numbers.append(number)
    return tuple(numbers)


@dataclass(frozen=True, slots=True)
class _ParsedPartition:
    records: tuple[EvidenceRecord, ...]
    errors: tuple[EvidenceError, ...]
    scanned_count: int
    unmatched_count: int
    rejected_count: int
    publication_delays: tuple[float, ...]
    batch_delays: tuple[float, ...]


def _record_error(
    *,
    provider_id: str,
    partition: GdeltGkgPartition,
    occurred_at: datetime,
    row_number: int,
    error_type: str,
    message: str,
    row: bytes,
) -> EvidenceError:
    return EvidenceError(
        provider_id=provider_id,
        configured_source=f"gdelt-gkg:{partition.partition_id}",
        error_type=error_type,
        message=message,
        occurred_at=occurred_at,
        partition_id=partition.partition_id,
        details={
            "row_number": row_number,
            "row_sha256": hashlib.sha256(row).hexdigest(),
            "archive_filename": partition.filename,
        },
    )


def _parse_partition_rows(
    payload: bytes,
    *,
    partition: GdeltGkgPartition,
    archive_sha256: str,
    archive_md5: str,
    archive_size_bytes: int,
    archive_member: str,
    fetched_at: datetime,
    requested_start: datetime,
    requested_end: datetime,
    publication_after_batch_tolerance_seconds: float,
    max_records_remaining: int,
    max_row_bytes: int,
    max_field_characters: int,
    permission_policy_id: str,
    capture_reference: str,
    seen_source_ids: set[str],
    maximum_errors: int,
) -> _ParsedPartition:
    records: list[EvidenceRecord] = []
    errors: list[EvidenceError] = []
    scanned = unmatched = rejected = 0
    publication_delays: list[float] = []
    batch_delays: list[float] = []
    for row_number, line in enumerate(io.BytesIO(payload), start=1):
        raw_line = line.rstrip(b"\r\n")
        if not raw_line:
            continue
        if scanned >= max_records_remaining:
            raise _GkgRejected(
                "record_limit_exceeded",
                "GDELT GKG archive exceeds the configured record limit",
            )
        scanned += 1
        if len(raw_line) > max_row_bytes:
            rejected += 1
            if len(errors) < maximum_errors:
                errors.append(
                    _record_error(
                        provider_id=GDELT_GKG_PROVIDER_ID,
                        partition=partition,
                        occurred_at=fetched_at,
                        row_number=row_number,
                        error_type="row_limit_exceeded",
                        message="GDELT GKG row exceeds the configured byte limit",
                        row=raw_line,
                    )
                )
            continue
        try:
            text = raw_line.decode("utf-8")
        except UnicodeDecodeError:
            rejected += 1
            if len(errors) < maximum_errors:
                errors.append(
                    _record_error(
                        provider_id=GDELT_GKG_PROVIDER_ID,
                        partition=partition,
                        occurred_at=fetched_at,
                        row_number=row_number,
                        error_type="invalid_text_encoding",
                        message="GDELT GKG row is not valid UTF-8",
                        row=raw_line,
                    )
                )
            continue
        fields = text.split("\t")
        if len(fields) != len(GKG_V21_COLUMNS):
            rejected += 1
            if len(errors) < maximum_errors:
                errors.append(
                    _record_error(
                        provider_id=GDELT_GKG_PROVIDER_ID,
                        partition=partition,
                        occurred_at=fetched_at,
                        row_number=row_number,
                        error_type="invalid_column_count",
                        message="GDELT GKG row does not match the pinned 2.1 schema",
                        row=raw_line,
                    )
                )
            continue
        row = dict(zip(GKG_V21_COLUMNS, fields, strict=True))
        try:
            for field_name in (
                "GKGRECORDID",
                "V2.1DATE",
                "V2SOURCECOLLECTIONIDENTIFIER",
                "V2SOURCECOMMONNAME",
                "V2DOCUMENTIDENTIFIER",
                "V1ORGANIZATIONS",
                "V2ENHANCEDORGANIZATIONS",
                "V2ENHANCEDTHEMES",
                "V1.5TONE",
            ):
                _bounded_field(
                    row[field_name],
                    field_name=field_name,
                    maximum=max_field_characters,
                )
            source_id = row["GKGRECORDID"]
            batch_at = parse_gkg_record_id_timestamp(source_id)
            published_at = parse_v21_date(row["V2.1DATE"])
            if batch_at != partition.batch_at:
                raise _GkgRejected(
                    "archive_batch_mismatch",
                    "GKGRECORDID timestamp conflicts with its archive filename",
                )
            if not (requested_start <= batch_at <= requested_end) or not (
                requested_start <= published_at <= requested_end
            ):
                raise _GkgRejected(
                    "record_outside_requested_interval",
                    "GDELT GKG record falls outside the requested interval",
                )
            if row["V2SOURCECOLLECTIONIDENTIFIER"] != GDELT_WEB_SOURCE_COLLECTION_ID:
                raise _GkgRejected(
                    "unsupported_source_collection",
                    "GDELT GKG record is not from the supported web collection",
                )
            publication_delay = (batch_at - published_at).total_seconds()
            if publication_delay < -publication_after_batch_tolerance_seconds:
                raise _GkgRejected(
                    "publication_after_batch_tolerance",
                    "GDELT publication time is implausibly later than its batch",
                )
            batch_delay = (fetched_at - batch_at).total_seconds()
            if batch_delay < 0:
                raise _GkgRejected(
                    "batch_after_fetch_time",
                    "GDELT batch timestamp is later than PAISA fetched_at",
                )
            if source_id in seen_source_ids:
                raise _GkgRejected(
                    "duplicate_source_record",
                    "GDELT GKG record identity is duplicated across partitions",
                )
            source_common_name = row["V2SOURCECOMMONNAME"].strip()
            document_identifier = row["V2DOCUMENTIDENTIFIER"].strip()
            safe_url = _safe_document_url(document_identifier)
            if not source_common_name:
                if safe_url is None or not urlsplit(safe_url).hostname:
                    raise _GkgRejected(
                        "missing_source_identity",
                        "GDELT GKG web record has no defensible source identity",
                    )
                source_common_name = str(urlsplit(safe_url).hostname)
            organizations = row["V2ENHANCEDORGANIZATIONS"]
            matches = match_ogdc_aliases(
                v2_organizations=organizations,
                document_identifier=safe_url,
                source_common_name=source_common_name,
            )
            if not matches:
                unmatched += 1
                seen_source_ids.add(source_id)
                continue
            themes = _parse_themes(row["V2ENHANCEDTHEMES"])
            tone = _parse_tone(row["V1.5TONE"])
            content_identity = {
                "gkg_record_id": source_id,
                "source_published_at": utc_iso(published_at),
                "source_common_name": source_common_name,
                "document_identifier": safe_url,
                "v2_organizations": organizations,
                "themes": themes,
                "tone": tone,
                "archive_sha256": archive_sha256,
            }
            warnings: list[str] = ["publisher_body_not_retrieved"]
            if safe_url is None:
                warnings.append("document_identifier_not_a_safe_public_url")
            evidence_id = f"EVID-GKG-{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:24]}"
            record = EvidenceRecord(
                evidence_id=evidence_id,
                evidence_type="gdelt_metadata",
                provider_id=GDELT_GKG_PROVIDER_ID,
                provider_mode=GDELT_GKG_PROVIDER_MODE,
                data_class="real",
                source_name=source_common_name,
                source_url=safe_url,
                source_record_id=source_id,
                ticker=OGDC_TICKER,
                company_name=OGDC_COMPANY_NAME,
                title=None,
                source_published_at=published_at,
                source_published_date=None,
                available_at=batch_at,
                fetched_at=fetched_at,
                imported_at=fetched_at,
                historical_availability_basis=GDELT_GKG_HISTORICAL_BASIS,
                time_precision="second",
                contemporaneous_source_id=source_id,
                contemporaneous_batch_at=batch_at,
                permission_policy_id=permission_policy_id,
                raw_evidence_reference=f"{capture_reference}#row={row_number}",
                content_sha256=_canonical_sha256(content_identity),
                language=None,
                normalization_warnings=tuple(warnings),
                entity_match_evidence=matches,
                source_metadata={
                    "GKGRECORDID": source_id,
                    "V2.1DATE": row["V2.1DATE"],
                    "V2SOURCECOLLECTIONIDENTIFIER": row[
                        "V2SOURCECOLLECTIONIDENTIFIER"
                    ].strip(),
                    "V2SOURCECOMMONNAME": source_common_name,
                    "V2DOCUMENTIDENTIFIER": safe_url,
                    "v2_document_identifier_sha256": hashlib.sha256(
                        document_identifier.encode("utf-8")
                    ).hexdigest(),
                    "V1ORGANIZATIONS": row["V1ORGANIZATIONS"],
                    "V2ORGANIZATIONS": organizations,
                    "V2ENHANCEDORGANIZATIONS": organizations,
                    "V2ENHANCEDTHEMES": list(themes),
                    "V1.5TONE": list(tone),
                    "archive_filename": partition.filename,
                    "archive_member": archive_member,
                    "archive_size_bytes": archive_size_bytes,
                    "archive_md5": archive_md5,
                    "archive_sha256": archive_sha256,
                    "archive_row_number": row_number,
                    "publication_delay_seconds": publication_delay,
                    "batch_delay_seconds": batch_delay,
                    "publisher_body_retrieved": False,
                },
            )
            records.append(record)
            publication_delays.append(publication_delay)
            batch_delays.append(batch_delay)
            seen_source_ids.add(source_id)
        except ValueError as exc:
            rejected += 1
            if isinstance(exc, _GkgRejected):
                error_type = exc.error_type
                message = exc.safe_message
            else:
                message_text = str(exc)
                if message_text.startswith("GKGRECORDID"):
                    error_type = "malformed_gkg_record_id"
                    message = "GDELT GKG record ID is malformed"
                elif message_text.startswith("V2.1DATE"):
                    error_type = "invalid_publication_time"
                    message = "GDELT GKG publication timestamp is missing or invalid"
                else:
                    error_type = "record_validation_error"
                    message = "GDELT GKG record failed validation"
            if len(errors) < maximum_errors:
                errors.append(
                    _record_error(
                        provider_id=GDELT_GKG_PROVIDER_ID,
                        partition=partition,
                        occurred_at=fetched_at,
                        row_number=row_number,
                        error_type=error_type,
                        message=message,
                        row=raw_line,
                    )
                )
    return _ParsedPartition(
        records=tuple(records),
        errors=tuple(errors),
        scanned_count=scanned,
        unmatched_count=unmatched,
        rejected_count=rejected,
        publication_delays=tuple(publication_delays),
        batch_delays=tuple(batch_delays),
    )


class _CheckpointStore:
    def __init__(self, directory: Path, *, plan_sha256: str) -> None:
        self.directory = directory
        self.plan_sha256 = plan_sha256
        try:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory_metadata = directory.lstat()
        except OSError as exc:
            raise _GkgRejected(
                "unsafe_checkpoint", "GDELT GKG checkpoint root is unavailable"
            ) from exc
        if not stat.S_ISDIR(directory_metadata.st_mode):
            raise _GkgRejected(
                "unsafe_checkpoint",
                "GDELT GKG checkpoint root must be a regular directory",
            )
        try:
            os.chmod(directory, 0o700)
        except OSError as exc:
            raise _GkgRejected(
                "unsafe_checkpoint", "GDELT GKG checkpoint permissions failed"
            ) from exc
        self.path = directory / "checkpoint.json"
        self.entries: dict[str, dict[str, Any]] = {}
        if os.path.lexists(self.path):
            if self.path.is_symlink() or not self.path.is_file():
                raise _GkgRejected(
                    "unsafe_checkpoint", "GDELT GKG checkpoint must be a regular file"
                )
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise _GkgRejected(
                    "invalid_checkpoint", "GDELT GKG checkpoint is invalid"
                ) from exc
            if (
                not isinstance(value, Mapping)
                or value.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION
                or value.get("plan_sha256") != plan_sha256
                or not isinstance(value.get("completed"), Mapping)
            ):
                raise _GkgRejected(
                    "checkpoint_plan_mismatch",
                    "GDELT GKG checkpoint does not match the current partition plan",
                )
            self.entries = {
                str(key): dict(item)
                for key, item in value["completed"].items()
                if isinstance(item, Mapping)
            }

    def cached(
        self,
        partition: GdeltGkgPartition,
        *,
        per_file_limit: int,
        remaining_total_limit: int,
    ) -> tuple[bytes, datetime, str, str] | None:
        entry = self.entries.get(partition.partition_id)
        if entry is None:
            return None
        filename = str(entry.get("cache_filename") or "")
        if PurePosixPath(filename).name != filename or not filename:
            raise _GkgRejected(
                "invalid_checkpoint", "GDELT GKG checkpoint cache identity is unsafe"
            )
        path = self.directory / filename
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise _GkgRejected(
                "checkpoint_cache_invalid",
                "GDELT GKG checkpoint cache file is missing",
            ) from exc
        if not stat.S_ISREG(metadata.st_mode):
            raise _GkgRejected(
                "checkpoint_cache_invalid",
                "GDELT GKG checkpoint cache file is unsafe",
            )
        if metadata.st_size > per_file_limit or metadata.st_size > remaining_total_limit:
            raise _GkgRejected(
                "compressed_byte_limit_exceeded",
                "GDELT GKG checkpoint cache exceeds the compressed-byte limit",
            )
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise _GkgRejected(
                "checkpoint_cache_invalid",
                "GDELT GKG checkpoint cache file cannot be read",
            ) from exc
        observed_md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if (
            entry.get("archive_url") != partition.archive_url
            or entry.get("archive_filename") != partition.filename
            or entry.get("size_bytes") != len(payload)
            or entry.get("md5") != observed_md5
            or entry.get("sha256") != observed_sha256
        ):
            raise _GkgRejected(
                "checkpoint_cache_invalid",
                "GDELT GKG checkpoint cache checksum differs",
            )
        _validate_archive_checksums(payload, partition)
        fetched_text = str(entry.get("fetched_at") or "")
        try:
            fetched_at = datetime.fromisoformat(fetched_text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise _GkgRejected(
                "invalid_checkpoint", "GDELT GKG checkpoint timestamp is invalid"
            ) from exc
        return payload, _aware_utc(fetched_at, field_name="checkpoint fetched_at"), observed_md5, observed_sha256

    def record(
        self,
        partition: GdeltGkgPartition,
        *,
        payload: bytes,
        fetched_at: datetime,
        md5: str,
        sha256: str,
    ) -> None:
        cache_filename = f"{partition.partition_id}-{partition.filename}"
        cache_path = self.directory / cache_filename
        if os.path.lexists(cache_path):
            try:
                metadata = cache_path.lstat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise _GkgRejected(
                        "checkpoint_cache_collision",
                        "GDELT GKG checkpoint cache destination is unsafe",
                    )
                existing = cache_path.read_bytes()
            except _GkgRejected:
                raise
            except OSError as exc:
                raise _GkgRejected(
                    "checkpoint_cache_collision",
                    "GDELT GKG checkpoint cache destination cannot be read",
                ) from exc
            if hashlib.sha256(existing).hexdigest() != sha256:
                raise _GkgRejected(
                    "checkpoint_cache_collision",
                    "GDELT GKG checkpoint cache destination already differs",
                )
        else:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(cache_path, flags, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            except BaseException:
                cache_path.unlink(missing_ok=True)
                raise
        self.entries[partition.partition_id] = {
            "archive_url": partition.archive_url,
            "archive_filename": partition.filename,
            "cache_filename": cache_filename,
            "size_bytes": len(payload),
            "md5": md5,
            "sha256": sha256,
            "fetched_at": utc_iso(fetched_at),
        }
        document = {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "plan_sha256": self.plan_sha256,
            "completed": dict(sorted(self.entries.items())),
        }
        encoded = (
            json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        temporary = self.directory / f".checkpoint-{os.getpid()}-{uuid4().hex}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary, flags, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


class GdeltGkgArchiveProvider:
    """Acquire explicit official GKG archives under strict resource bounds."""

    provider_id = GDELT_GKG_PROVIDER_ID
    provider_mode = GDELT_GKG_PROVIDER_MODE
    data_class = "real"

    def __init__(
        self,
        partitions: Sequence[GdeltGkgPartition],
        *,
        requested_start: datetime,
        requested_end: datetime,
        permission_policy: EvidencePermissionPolicy,
        max_files: int = 32,
        max_compressed_bytes_per_file: int = 64 * 1024 * 1024,
        max_compressed_bytes: int = 256 * 1024 * 1024,
        max_decompressed_bytes_per_file: int = 256 * 1024 * 1024,
        max_decompressed_bytes: int = 1024 * 1024 * 1024,
        max_records: int = 1_000_000,
        max_row_bytes: int = 2 * 1024 * 1024,
        max_field_characters: int = 1_000_000,
        maximum_errors: int = 1_000,
        publication_after_batch_tolerance_seconds: float = 24 * 60 * 60,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 60.0,
        user_agent: str = _DEFAULT_USER_AGENT,
        checkpoint_directory: str | Path | None = None,
        session: Any | None = None,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        permission_decision = _validated_gdelt_permission(permission_policy)
        if not partitions:
            raise ValueError("GDELT GKG requires at least one explicit partition")
        selected_max_files = _bounded_positive_integer(
            max_files, field_name="max_files", maximum=_MAX_FILES_HARD
        )
        if len(partitions) > selected_max_files:
            raise ValueError("GDELT GKG partition plan exceeds max_files")
        selected = tuple(sorted(partitions, key=lambda item: item.batch_at))
        if len({item.partition_id for item in selected}) != len(selected):
            raise ValueError("GDELT GKG partition IDs must be unique")
        if len({item.archive_url for item in selected}) != len(selected):
            raise ValueError("GDELT GKG archive URLs must be unique")
        if len({item.batch_at for item in selected}) != len(selected):
            raise ValueError("GDELT GKG archive batches must be unique")
        if any(
            item.expected_size_bytes is None or item.expected_md5 is None
            for item in selected
        ):
            raise ValueError(
                "Every GDELT GKG partition requires official size and MD5 claims"
            )
        start = _aware_utc(requested_start, field_name="requested_start")
        end = _aware_utc(requested_end, field_name="requested_end")
        if start > end:
            raise ValueError("requested_start must not follow requested_end")
        if any(not start <= item.batch_at <= end for item in selected):
            raise ValueError("Every GDELT GKG partition must be inside the requested interval")
        self.partitions = selected
        self.requested_start = start
        self.requested_end = end
        self.permission_policy = permission_policy
        self.permission_policy_id = permission_policy.policy_id
        self.permission_policy_sha256 = permission_policy.policy_sha256
        self.permission_decision_id = str(permission_decision["decision_id"])
        self.permission_status = str(permission_decision["status"])
        self.max_files = selected_max_files
        self.max_compressed_bytes_per_file = _bounded_positive_integer(
            max_compressed_bytes_per_file,
            field_name="max_compressed_bytes_per_file",
            maximum=_MAX_COMPRESSED_BYTES_HARD,
        )
        self.max_compressed_bytes = _bounded_positive_integer(
            max_compressed_bytes,
            field_name="max_compressed_bytes",
            maximum=_MAX_COMPRESSED_BYTES_HARD,
        )
        expected_sizes = tuple(
            int(item.expected_size_bytes) for item in selected
            if item.expected_size_bytes is not None
        )
        if any(
            size > self.max_compressed_bytes_per_file for size in expected_sizes
        ):
            raise ValueError(
                "GDELT GKG partition plan exceeds the per-file compressed-byte cap"
            )
        if sum(expected_sizes) > self.max_compressed_bytes:
            raise ValueError(
                "GDELT GKG partition plan exceeds the total compressed-byte cap"
            )
        self.max_decompressed_bytes_per_file = _bounded_positive_integer(
            max_decompressed_bytes_per_file,
            field_name="max_decompressed_bytes_per_file",
            maximum=_MAX_DECOMPRESSED_BYTES_HARD,
        )
        self.max_decompressed_bytes = _bounded_positive_integer(
            max_decompressed_bytes,
            field_name="max_decompressed_bytes",
            maximum=_MAX_DECOMPRESSED_BYTES_HARD,
        )
        self.max_records = _bounded_positive_integer(
            max_records, field_name="max_records", maximum=_MAX_RECORDS_HARD
        )
        self.max_row_bytes = _bounded_positive_integer(
            max_row_bytes, field_name="max_row_bytes", maximum=_MAX_ROW_BYTES_HARD
        )
        self.max_field_characters = _bounded_positive_integer(
            max_field_characters,
            field_name="max_field_characters",
            maximum=_MAX_FIELD_CHARACTERS_HARD,
        )
        self.maximum_errors = _bounded_positive_integer(
            maximum_errors, field_name="maximum_errors", maximum=100_000
        )
        self.publication_after_batch_tolerance_seconds = _bounded_nonnegative_number(
            publication_after_batch_tolerance_seconds,
            field_name="publication_after_batch_tolerance_seconds",
            maximum=7 * 24 * 60 * 60,
        )
        self.connect_timeout_seconds = _bounded_nonnegative_number(
            connect_timeout_seconds,
            field_name="connect_timeout_seconds",
            maximum=300,
        )
        self.read_timeout_seconds = _bounded_nonnegative_number(
            read_timeout_seconds,
            field_name="read_timeout_seconds",
            maximum=300,
        )
        if not self.connect_timeout_seconds or not self.read_timeout_seconds:
            raise ValueError("Network timeouts must be strictly positive")
        self.user_agent = str(user_agent).strip()
        if not self.user_agent or "\n" in self.user_agent or "\r" in self.user_agent:
            raise ValueError("user_agent must be a non-empty single line")
        self.checkpoint_directory = (
            Path(checkpoint_directory) if checkpoint_directory is not None else None
        )
        self.clock = clock
        self._owns_session = session is None
        self.session = session or requests.Session()
        if hasattr(self.session, "trust_env"):
            self.session.trust_env = False
        if hasattr(self.session, "auth"):
            self.session.auth = None
        if hasattr(self.session, "cert"):
            self.session.cert = None
        for attribute in ("params", "proxies"):
            value = getattr(self.session, attribute, None)
            clear = getattr(value, "clear", None)
            if callable(clear):
                clear()
        cookies = getattr(self.session, "cookies", None)
        clear_cookies = getattr(cookies, "clear", None)
        if callable(clear_cookies):
            clear_cookies()
        headers = getattr(self.session, "headers", None)
        if headers is not None:
            for key in tuple(headers):
                if _SENSITIVE_KEY.search(str(key)):
                    headers.pop(key, None)
        self.configured_sources = tuple(
            {
                "configured_source": f"gdelt-gkg:{item.partition_id}",
                **item.to_dict(),
                "provider_mode": self.provider_mode,
                "source_collection_policy": "web_only_identifier_1",
                "permission_policy_id": self.permission_policy_id,
                "permission_policy_sha256": self.permission_policy_sha256,
                "permission_decision_id": self.permission_decision_id,
                "permission_status": self.permission_status,
            }
            for item in self.partitions
        )
        self.partition_plan = tuple(
            {
                "sequence": index,
                **item.to_dict(),
                "requested_start": utc_iso(self.requested_start),
                "requested_end": utc_iso(self.requested_end),
            }
            for index, item in enumerate(self.partitions, start=1)
        )
        self.effective_config = {
            "provider_id": self.provider_id,
            "provider_mode": self.provider_mode,
            "permission_policy_id": self.permission_policy_id,
            "permission_policy_sha256": self.permission_policy_sha256,
            "permission_decision_id": self.permission_decision_id,
            "permission_status": self.permission_status,
            "archive_host": GDELT_GKG_ARCHIVE_HOST,
            "requested_start": utc_iso(self.requested_start),
            "requested_end": utc_iso(self.requested_end),
            "max_files": self.max_files,
            "max_compressed_bytes_per_file": self.max_compressed_bytes_per_file,
            "max_compressed_bytes": self.max_compressed_bytes,
            "max_decompressed_bytes_per_file": self.max_decompressed_bytes_per_file,
            "max_decompressed_bytes": self.max_decompressed_bytes,
            "max_records": self.max_records,
            "max_row_bytes": self.max_row_bytes,
            "max_field_characters": self.max_field_characters,
            "publication_after_batch_tolerance_seconds": (
                self.publication_after_batch_tolerance_seconds
            ),
            "source_collection_policy": "web_only_identifier_1",
            "redirect_policy": "reject",
            "environment_authentication": "disabled",
            "publisher_page_fetching": "disabled",
            "checkpointing": "validated_archive_cache" if self.checkpoint_directory else "disabled",
        }

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        permission_policy: EvidencePermissionPolicy,
        **overrides: Any,
    ) -> "GdeltGkgArchiveProvider":
        document = _load_config_document(path)
        configured_policy_id = str(document.get("permission_policy_id") or "")
        if configured_policy_id != permission_policy.policy_id:
            raise EvidencePermissionError(
                "GDELT configuration permission_policy_id differs from the "
                "validated policy"
            )
        partitions = _partitions_from_document(document)
        options = {
            key: value
            for key, value in document.items()
            if key
            not in {
                "schema_version",
                "provider",
                "partitions",
                "archive_index_url",
                "permission_policy_id",
            }
        }
        options.update(overrides)
        for field_name in ("requested_start", "requested_end"):
            value = options.get(field_name)
            if isinstance(value, str):
                try:
                    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(
                        f"{field_name} must be an ISO-8601 timestamp"
                    ) from exc
                options[field_name] = _aware_utc(parsed, field_name=field_name)
        return cls(
            partitions,
            permission_policy=permission_policy,
            **options,
        )

    def _download(
        self,
        partition: GdeltGkgPartition,
        *,
        remaining_total_bytes: int,
        fetched_at: datetime | None,
    ) -> tuple[bytes, datetime]:
        self._assert_permission_current()
        response: Any | None = None
        try:
            response = self.session.get(
                partition.archive_url,
                timeout=(self.connect_timeout_seconds, self.read_timeout_seconds),
                allow_redirects=False,
                stream=True,
                proxies={},
                headers={
                    "User-Agent": self.user_agent,
                    "Accept": "application/zip, application/octet-stream",
                    "Accept-Encoding": "identity",
                },
            )
            status = int(response.status_code)
            if status != 200:
                raise _GkgRejected(
                    "http_error", "GDELT GKG archive request did not return HTTP 200"
                )
            payload = _read_bounded_response(
                response,
                per_file_limit=self.max_compressed_bytes_per_file,
                remaining_total_limit=remaining_total_bytes,
            )
            acquisition_time = _aware_utc(
                fetched_at or self.clock(), field_name="fetched_at"
            )
            return payload, acquisition_time
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            cookies = getattr(self.session, "cookies", None)
            clear = getattr(cookies, "clear", None)
            if callable(clear):
                clear()

    def _assert_permission_current(self) -> None:
        permission_decision = _validated_gdelt_permission(self.permission_policy)
        if (
            str(permission_decision.get("decision_id") or "")
            != self.permission_decision_id
            or self.permission_policy.policy_id != self.permission_policy_id
            or self.permission_policy.policy_sha256 != self.permission_policy_sha256
        ):
            raise EvidencePermissionError(
                "GDELT GKG permission policy changed after provider construction"
            )

    def fetch(self, *, fetched_at: datetime | None = None) -> EvidenceFetchResult:
        self._assert_permission_current()
        plan_sha256 = _canonical_sha256(
            {
                "partitions": self.partition_plan,
                "permission_policy_reference": self.permission_policy.reference(),
                "permission_decision_id": self.permission_decision_id,
                "effective_config": self.effective_config,
            }
        )
        checkpoint: _CheckpointStore | None = None
        try:
            if self.checkpoint_directory is not None:
                checkpoint = _CheckpointStore(
                    self.checkpoint_directory, plan_sha256=plan_sha256
                )
        except _GkgRejected as exc:
            now = _aware_utc(fetched_at or self.clock(), field_name="fetched_at")
            return EvidenceFetchResult(
                provider_id=self.provider_id,
                provider_mode=self.provider_mode,
                data_class="real",
                fetched_at=now,
                permission_policy_id=self.permission_policy_id,
                configured_sources=self.configured_sources,
                partition_plan=self.partition_plan,
                errors=(
                    EvidenceError(
                        provider_id=self.provider_id,
                        configured_source="gdelt-gkg:checkpoint",
                        error_type=exc.error_type,
                        message=exc.safe_message,
                        occurred_at=now,
                    ),
                ),
                counters={"partitions_configured": len(self.partitions)},
                timestamp_validation={
                    "publication_after_batch_tolerance_seconds": (
                        self.publication_after_batch_tolerance_seconds
                    )
                },
                effective_config=self.effective_config,
            )

        captures: list[EvidenceCapture] = []
        records: list[EvidenceRecord] = []
        errors: list[EvidenceError] = []
        seen_source_ids: set[str] = set()
        publication_delays: list[float] = []
        batch_delays: list[float] = []
        compressed_total = decompressed_total = scanned_total = 0
        unmatched_total = rejected_total = 0
        downloaded_count = resumed_count = completed_count = 0
        observed_fetch_times: list[datetime] = []

        for partition in self.partitions:
            configured_source = f"gdelt-gkg:{partition.partition_id}"
            if scanned_total >= self.max_records:
                occurred_at = _aware_utc(
                    fetched_at or self.clock(), field_name="fetched_at"
                )
                errors.append(
                    EvidenceError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type="record_limit_exceeded",
                        message=(
                            "GDELT GKG run reached the configured record limit"
                        ),
                        occurred_at=occurred_at,
                        partition_id=partition.partition_id,
                        details={"archive_filename": partition.filename},
                    )
                )
                observed_fetch_times.append(occurred_at)
                break
            try:
                cached = (
                    checkpoint.cached(
                        partition,
                        per_file_limit=self.max_compressed_bytes_per_file,
                        remaining_total_limit=(
                            self.max_compressed_bytes - compressed_total
                        ),
                    )
                    if checkpoint is not None
                    else None
                )
                if cached is None:
                    payload, acquisition_time = self._download(
                        partition,
                        remaining_total_bytes=self.max_compressed_bytes
                        - compressed_total,
                        fetched_at=fetched_at,
                    )
                    observed_md5, observed_sha256 = _validate_archive_checksums(
                        payload, partition
                    )
                    downloaded_count += 1
                else:
                    payload, acquisition_time, observed_md5, observed_sha256 = cached
                    resumed_count += 1
                if compressed_total + len(payload) > self.max_compressed_bytes:
                    raise _GkgRejected(
                        "compressed_byte_limit_exceeded",
                        "GDELT GKG run exceeds the configured compressed-byte limit",
                    )
                compressed_total += len(payload)
                capture_ordinal = len(captures) + 1
                capture_reference = _published_capture_name(
                    capture_ordinal, partition.filename
                )
                archive_member, decompressed = _read_safe_zip_member(
                    payload,
                    partition=partition,
                    per_file_limit=self.max_decompressed_bytes_per_file,
                    remaining_total_limit=self.max_decompressed_bytes
                    - decompressed_total,
                )
                decompressed_total += len(decompressed)
                parsed = _parse_partition_rows(
                    decompressed,
                    partition=partition,
                    archive_sha256=observed_sha256,
                    archive_md5=observed_md5,
                    archive_size_bytes=len(payload),
                    archive_member=archive_member,
                    fetched_at=acquisition_time,
                    requested_start=self.requested_start,
                    requested_end=self.requested_end,
                    publication_after_batch_tolerance_seconds=(
                        self.publication_after_batch_tolerance_seconds
                    ),
                    max_records_remaining=self.max_records - scanned_total,
                    max_row_bytes=self.max_row_bytes,
                    max_field_characters=self.max_field_characters,
                    permission_policy_id=self.permission_policy_id,
                    capture_reference=capture_reference,
                    seen_source_ids=seen_source_ids,
                    maximum_errors=max(self.maximum_errors - len(errors), 1),
                )
                captures.append(
                    EvidenceCapture(
                        capture_id=f"gkg-{partition.partition_id}",
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        filename=partition.filename,
                        fetched_at=acquisition_time,
                        content=payload,
                        source_url=partition.archive_url,
                        content_type="application/zip",
                        sha256=observed_sha256,
                        warnings=(
                            ("upstream_checksum_not_supplied",)
                            if partition.expected_md5 is None
                            and partition.expected_sha256 is None
                            else ()
                        ),
                    )
                )
                records.extend(parsed.records)
                errors.extend(parsed.errors[: max(self.maximum_errors - len(errors), 0)])
                scanned_total += parsed.scanned_count
                unmatched_total += parsed.unmatched_count
                rejected_total += parsed.rejected_count
                publication_delays.extend(parsed.publication_delays)
                batch_delays.extend(parsed.batch_delays)
                observed_fetch_times.append(acquisition_time)
                completed_count += 1
                if checkpoint is not None and cached is None:
                    checkpoint.record(
                        partition,
                        payload=payload,
                        fetched_at=acquisition_time,
                        md5=observed_md5,
                        sha256=observed_sha256,
                    )
            except _GkgRejected as exc:
                occurred_at = _aware_utc(
                    fetched_at or self.clock(), field_name="fetched_at"
                )
                errors.append(
                    EvidenceError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type=exc.error_type,
                        message=exc.safe_message,
                        occurred_at=occurred_at,
                        partition_id=partition.partition_id,
                        details={"archive_filename": partition.filename},
                    )
                )
                observed_fetch_times.append(occurred_at)
                if exc.error_type in {
                    "compressed_byte_limit_exceeded",
                    "decompressed_byte_limit_exceeded",
                    "record_limit_exceeded",
                }:
                    break
            except (OSError, requests.RequestException) as exc:
                occurred_at = _aware_utc(
                    fetched_at or self.clock(), field_name="fetched_at"
                )
                errors.append(
                    EvidenceError(
                        provider_id=self.provider_id,
                        configured_source=configured_source,
                        error_type="network_error",
                        message=(
                            "GDELT GKG archive could not be acquired "
                            f"({type(exc).__name__})"
                        ),
                        occurred_at=occurred_at,
                        partition_id=partition.partition_id,
                        details={"archive_filename": partition.filename},
                    )
                )
                observed_fetch_times.append(occurred_at)

        result_fetched_at = max(observed_fetch_times) if observed_fetch_times else _aware_utc(
            fetched_at or self.clock(), field_name="fetched_at"
        )
        warnings: list[str] = [
            "metadata_only_publisher_pages_bodies_images_not_fetched",
            "source_collection_limited_to_gdelt_web_identifier_1",
        ]
        if len(errors) >= self.maximum_errors and rejected_total > len(errors):
            warnings.append("provider_error_details_truncated")
        if any(
            item.expected_md5 is None and item.expected_sha256 is None
            for item in self.partitions
        ):
            warnings.append("one_or_more_partitions_lack_upstream_checksum")

        def bounds(values: Sequence[float]) -> dict[str, float | None]:
            return {
                "minimum_seconds": min(values) if values else None,
                "maximum_seconds": max(values) if values else None,
            }

        return EvidenceFetchResult(
            provider_id=self.provider_id,
            provider_mode=self.provider_mode,
            data_class="real",
            fetched_at=result_fetched_at,
            permission_policy_id=self.permission_policy_id,
            configured_sources=self.configured_sources,
            partition_plan=self.partition_plan,
            records=tuple(records),
            captures=tuple(captures),
            errors=tuple(errors[: self.maximum_errors]),
            warnings=tuple(warnings),
            counters={
                "partitions_configured": len(self.partitions),
                "partitions_downloaded": downloaded_count,
                "partitions_resumed": resumed_count,
                "partitions_completed": completed_count,
                "compressed_bytes": compressed_total,
                "decompressed_bytes": decompressed_total,
                "records_scanned": scanned_total,
                "records_unmatched": unmatched_total,
                "records_rejected": rejected_total,
                "records_retained": len(records),
            },
            timestamp_validation={
                "publication_after_batch_tolerance_seconds": (
                    self.publication_after_batch_tolerance_seconds
                ),
                "publication_delay": bounds(publication_delays),
                "batch_delay": bounds(batch_delays),
                "negative_publication_delay_count": sum(
                    value < 0 for value in publication_delays
                ),
                "validated_record_count": len(records),
                "rejected_record_count": rejected_total,
            },
            effective_config=self.effective_config,
        )


_CONFIG_KEYS = {
    "schema_version",
    "provider",
    "archive_index_url",
    "requested_start",
    "requested_end",
    "permission_policy_id",
    "max_files",
    "max_compressed_bytes_per_file",
    "max_compressed_bytes",
    "max_decompressed_bytes_per_file",
    "max_decompressed_bytes",
    "max_records",
    "max_row_bytes",
    "max_field_characters",
    "maximum_errors",
    "publication_after_batch_tolerance_seconds",
    "connect_timeout_seconds",
    "read_timeout_seconds",
    "user_agent",
    "checkpoint_directory",
    "partitions",
}


def _load_config_document(path: str | Path) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("GDELT GKG configuration is not valid UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError("GDELT GKG configuration must be a JSON object")
    unexpected = set(value) - _CONFIG_KEYS
    if unexpected:
        raise ValueError(
            "Unsupported GDELT GKG configuration fields: "
            + ", ".join(sorted(str(item) for item in unexpected))
        )
    if any(_SENSITIVE_KEY.search(str(key)) for key in value):
        raise ValueError("GDELT GKG configuration must not contain credentials")
    if value.get("schema_version", "1.0") != "1.0":
        raise ValueError("GDELT GKG configuration schema_version must be '1.0'")
    if value.get("provider", GDELT_GKG_PROVIDER_ID) != GDELT_GKG_PROVIDER_ID:
        raise ValueError(f"GDELT GKG provider must be '{GDELT_GKG_PROVIDER_ID}'")
    if value.get("archive_index_url", GDELT_GKG_INDEX_URL) != GDELT_GKG_INDEX_URL:
        raise ValueError("GDELT GKG archive index URL must use the official endpoint")
    return value


def _partitions_from_document(
    document: Mapping[str, Any]
) -> tuple[GdeltGkgPartition, ...]:
    raw = document.get("partitions")
    if not isinstance(raw, list) or not raw:
        raise ValueError("GDELT GKG configuration requires explicit partitions")
    partitions: list[GdeltGkgPartition] = []
    for value in raw:
        if not isinstance(value, Mapping):
            raise ValueError("Every GDELT GKG partition must be an object")
        enabled = value.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("GDELT GKG partition enabled must be true or false")
        if enabled:
            partitions.append(GdeltGkgPartition.from_mapping(value))
    if not partitions:
        raise ValueError("GDELT GKG configuration has no enabled partitions")
    return tuple(partitions)


def load_gdelt_gkg_config(path: str | Path) -> tuple[GdeltGkgPartition, ...]:
    """Load enabled explicit partitions without performing network access."""

    return _partitions_from_document(_load_config_document(path))


__all__ = [
    "GDELT_GKG_ARCHIVE_HOST",
    "GDELT_GKG_INDEX_URL",
    "GDELT_GKG_PROVIDER_ID",
    "GDELT_WEB_SOURCE_COLLECTION_ID",
    "GKG_V21_COLUMNS",
    "GdeltGkgArchiveProvider",
    "GdeltGkgPartition",
    "OGDC_ALIASES",
    "load_gdelt_gkg_config",
    "match_ogdc_aliases",
    "parse_gkg_record_id_timestamp",
    "parse_v21_date",
]
