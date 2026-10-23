from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import stat
from typing import Any
from urllib.parse import urlsplit
import zipfile

import pytest

from paisa.evidence.policy import EvidencePermissionError, EvidencePermissionPolicy
from paisa.evidence.gdelt_gkg import (
    GDELT_GKG_PROVIDER_ID,
    GKG_V21_COLUMNS,
    GdeltGkgArchiveProvider,
    GdeltGkgPartition,
    load_gdelt_gkg_config,
    match_ogdc_aliases,
    parse_gkg_record_id_timestamp,
    parse_v21_date,
)


UTC = timezone.utc
BATCH = "20260904120000"
ARCHIVE_URL = f"https://data.gdeltproject.org/gdeltv2/{BATCH}.gkg.csv.zip"
FETCHED_AT = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
REQUESTED_START = datetime(2025, 12, 16, tzinfo=UTC)
REQUESTED_END = datetime(2026, 9, 15, 23, 59, 59, tzinfo=UTC)
POLICY_ID = "test-gdelt-gkg-policy"
_COLUMN = {name: index for index, name in enumerate(GKG_V21_COLUMNS)}


def _policy(
    *,
    status: str = "PERMITTED",
    automated_access: bool = True,
    publisher_page_fetching: bool = False,
    policy_id: str = POLICY_ID,
    policy_sha256: str = "a" * 64,
) -> EvidencePermissionPolicy:
    return EvidencePermissionPolicy(
        policy_id=policy_id,
        schema_version="1.0",
        policy_path="configs/evidence/source_permissions.json",
        policy_sha256=policy_sha256,
        assessment_references=(
            {
                "path": "docs/GDELT_GKG_HISTORICAL_EVIDENCE.md",
                "sha256": "b" * 64,
            },
        ),
        decisions={
            (GDELT_GKG_PROVIDER_ID, "raw_archive"): {
                "decision_id": "gdelt-gkg-raw-metadata-test-v1",
                "provider_id": GDELT_GKG_PROVIDER_ID,
                "provider_mode": "raw_archive",
                "status": status,
                "automated_access": automated_access,
                "publisher_page_fetching": publisher_page_fetching,
            }
        },
    )


class _RawBody:
    def __init__(self, chunks: Iterable[bytes]) -> None:
        self._chunks = tuple(chunks)

    def stream(self, chunk_size: int, *, decode_content: bool) -> Iterable[bytes]:
        assert chunk_size > 0
        assert decode_content is False
        yield from self._chunks


class _Response:
    def __init__(
        self,
        content: bytes,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: Sequence[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {"Content-Length": str(len(content))}
        self.raw = _RawBody(chunks or (content,))
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Session:
    def __init__(self, responses: Sequence[_Response]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.trust_env = True
        self.auth: Any = ("ambient-user", "ambient-password")
        self.cert: Any = "/ambient/client.pem"
        self.params = {"token": "ambient-token"}
        self.proxies = {"https": "https://ambient-proxy.invalid"}
        self.cookies = {"session": "ambient-cookie"}
        self.headers = {
            "Authorization": "Bearer ambient-secret",
            "X-Harmless": "retained",
        }

    def get(self, url: str, **kwargs: Any) -> _Response:
        self.calls.append((url, kwargs))
        if not self._responses:
            raise AssertionError(f"Unexpected network request: {url}")
        return self._responses.pop(0)


def _row(
    *,
    record_id: str = f"{BATCH}-1",
    published: str = "20260904115900",
    collection: str = "1",
    source_name: str = "publisher.example",
    document_identifier: str = (
        "https://publisher.example/markets/OGDC?utm_source=test&token=secret&story=1#top"
    ),
    organizations: str = "Oil & Gas Development Company Limited,22",
    themes: str = "ECON_STOCKMARKET,10;ENERGY,20",
    tone: str = "1.25,2,3,4,5,6,7",
) -> bytes:
    values = [""] * len(GKG_V21_COLUMNS)
    values[_COLUMN["GKGRECORDID"]] = record_id
    values[_COLUMN["V2.1DATE"]] = published
    values[_COLUMN["V2SOURCECOLLECTIONIDENTIFIER"]] = collection
    values[_COLUMN["V2SOURCECOMMONNAME"]] = source_name
    values[_COLUMN["V2DOCUMENTIDENTIFIER"]] = document_identifier
    values[_COLUMN["V1ORGANIZATIONS"]] = organizations
    values[_COLUMN["V2ENHANCEDORGANIZATIONS"]] = organizations
    values[_COLUMN["V2ENHANCEDTHEMES"]] = themes
    values[_COLUMN["V1.5TONE"]] = tone
    return "\t".join(values).encode("utf-8")


def _zip_payload(
    rows: Sequence[bytes],
    *,
    batch: str = BATCH,
    member_name: str | None = None,
    extra_members: Sequence[tuple[str, bytes]] = (),
    symlink: bool = False,
) -> bytes:
    output = io.BytesIO()
    name = member_name or f"{batch}.gkg.csv"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        content = b"\n".join(rows) + (b"\n" if rows else b"")
        if symlink:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, content)
        else:
            archive.writestr(name, content)
        for extra_name, extra_content in extra_members:
            archive.writestr(extra_name, extra_content)
    return output.getvalue()


def _partition(
    payload: bytes,
    *,
    batch: str = BATCH,
    partition_id: str = "gkg-test-001",
    expected_size_bytes: int | None = None,
    expected_md5: str | None = None,
    expected_sha256: str | None = None,
) -> GdeltGkgPartition:
    return GdeltGkgPartition(
        partition_id=partition_id,
        archive_url=(
            f"https://data.gdeltproject.org/gdeltv2/{batch}.gkg.csv.zip"
        ),
        expected_size_bytes=(
            len(payload) if expected_size_bytes is None else expected_size_bytes
        ),
        expected_md5=(
            hashlib.md5(payload, usedforsecurity=False).hexdigest()
            if expected_md5 is None
            else expected_md5
        ),
        expected_sha256=(
            hashlib.sha256(payload).hexdigest()
            if expected_sha256 is None
            else expected_sha256
        ),
    )


def _provider(
    partition: GdeltGkgPartition,
    session: _Session,
    **overrides: Any,
) -> GdeltGkgArchiveProvider:
    options: dict[str, Any] = {
        "requested_start": REQUESTED_START,
        "requested_end": REQUESTED_END,
        "permission_policy": _policy(),
        "session": session,
        "clock": lambda: FETCHED_AT,
    }
    options.update(overrides)
    return GdeltGkgArchiveProvider((partition,), **options)


def _error_types(result: Any) -> set[str]:
    return {error.error_type for error in result.errors}


def test_strict_gkg_timestamp_parsers() -> None:
    expected = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    assert parse_gkg_record_id_timestamp(f"{BATCH}-123") == expected
    assert parse_v21_date(BATCH) == expected

    for value in (
        f" {BATCH}-1",
        f"{BATCH}-1 ",
        f"{BATCH}",
        f"{BATCH}-",
        "20260230120000-1",
        "2026090412000a-1",
    ):
        with pytest.raises(ValueError):
            parse_gkg_record_id_timestamp(value)

    for value in ("", "0", "00000000000000", f" {BATCH}", "20260230120000"):
        with pytest.raises(ValueError):
            parse_v21_date(value)
    with pytest.raises(ValueError):
        parse_v21_date(20260904120000)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "url",
    [
        f"http://data.gdeltproject.org/gdeltv2/{BATCH}.gkg.csv.zip",
        f"https://evil.example/gdeltv2/{BATCH}.gkg.csv.zip",
        f"https://user@data.gdeltproject.org/gdeltv2/{BATCH}.gkg.csv.zip",
        f"https://data.gdeltproject.org/gdeltv2/nested/{BATCH}.gkg.csv.zip",
        f"https://data.gdeltproject.org/gdeltv2/{BATCH}.gkg.csv.zip?token=x",
        "https://data.gdeltproject.org/gdeltv2/latest.gkg.csv.zip",
    ],
)
def test_partition_rejects_any_noncanonical_archive_url(url: str) -> None:
    with pytest.raises(ValueError):
        GdeltGkgPartition(partition_id="p1", archive_url=url)


def test_provider_requires_official_size_and_md5_partition_claims() -> None:
    session = _Session(())
    partition = GdeltGkgPartition(
        partition_id="missing-index-claims",
        archive_url=ARCHIVE_URL,
    )
    with pytest.raises(ValueError, match="official size and MD5"):
        _provider(partition, session)


@pytest.mark.parametrize(
    "policy",
    [
        _policy(status="REVIEW_REQUIRED"),
        _policy(automated_access=False),
        _policy(publisher_page_fetching=True),
        EvidencePermissionPolicy(
            policy_id=POLICY_ID,
            schema_version="1.0",
            policy_path="configs/evidence/source_permissions.json",
            policy_sha256="not-a-checksum",
            assessment_references=(
                {
                    "path": "docs/GDELT_GKG_HISTORICAL_EVIDENCE.md",
                    "sha256": "b" * 64,
                },
            ),
            decisions=_policy().decisions,
        ),
    ],
)
def test_provider_rejects_unpermitted_or_unbound_policy_before_fetch(
    policy: EvidencePermissionPolicy,
) -> None:
    payload = _zip_payload((_row(),))
    session = _Session((_Response(payload),))

    with pytest.raises(EvidencePermissionError):
        _provider(_partition(payload), session, permission_policy=policy)

    assert session.calls == []


def test_provider_rechecks_permission_immediately_before_network() -> None:
    payload = _zip_payload((_row(),))
    session = _Session((_Response(payload),))
    policy = _policy()
    provider = _provider(_partition(payload), session, permission_policy=policy)
    decision = policy.decisions[(GDELT_GKG_PROVIDER_ID, "raw_archive")]
    assert isinstance(decision, dict)
    decision["status"] = "REVIEW_REQUIRED"

    with pytest.raises(EvidencePermissionError, match="REVIEW_REQUIRED"):
        provider.fetch(fetched_at=FETCHED_AT)

    assert session.calls == []


def test_ogdc_alias_matching_is_exact_and_prevents_substrings() -> None:
    matches = match_ogdc_aliases(
        v2_organizations=(
            "Other Company,1;Oil & Gas Development Company Limited,42;OGDCL,77"
        ),
        document_identifier="https://example.test/company/ogdc/results",
        source_common_name="Unrelated Publisher",
    )
    assert {item["field"] for item in matches} == {
        "V2ORGANIZATIONS",
        "V2DOCUMENTIDENTIFIER",
    }
    assert any(item["matched_value"] == "OGDCL" for item in matches)
    assert not match_ogdc_aliases(
        v2_organizations="DOGDCX,1;MOGDCLNEWS,2",
        document_identifier="https://example.test/company/dogdcx",
        source_common_name="MOGDCLNEWS",
    )


def test_valid_fetch_returns_metadata_only_evidence_and_hardened_request() -> None:
    payload = _zip_payload((_row(),))
    partition = _partition(payload)
    response = _Response(payload)
    session = _Session((response,))
    provider = _provider(partition, session)

    assert provider.session.trust_env is False
    assert provider.session.auth is None
    assert provider.session.cert is None
    assert provider.session.params == {}
    assert provider.session.proxies == {}
    assert provider.session.cookies == {}
    assert "Authorization" not in provider.session.headers

    result = provider.fetch(fetched_at=FETCHED_AT)

    assert result.provider_id == GDELT_GKG_PROVIDER_ID
    assert result.permission_policy_id == POLICY_ID
    assert result.effective_config["permission_policy_sha256"] == "a" * 64
    assert result.effective_config["permission_decision_id"] == (
        "gdelt-gkg-raw-metadata-test-v1"
    )
    assert result.effective_config["permission_status"] == "PERMITTED"
    assert result.errors == ()
    assert result.counters == {
        "partitions_configured": 1,
        "partitions_downloaded": 1,
        "partitions_resumed": 0,
        "partitions_completed": 1,
        "compressed_bytes": len(payload),
        "decompressed_bytes": len(_row()) + 1,
        "records_scanned": 1,
        "records_unmatched": 0,
        "records_rejected": 0,
        "records_retained": 1,
    }
    assert len(result.records) == len(result.captures) == 1
    record = result.records[0]
    capture = result.captures[0]
    assert record.evidence_type == "gdelt_metadata"
    assert record.provider_mode == "raw_archive"
    assert record.title is None
    assert record.available_at == datetime(2026, 9, 4, 12, 0, tzinfo=UTC)
    assert record.contemporaneous_batch_at == record.available_at
    assert record.source_published_at == datetime(
        2026, 9, 4, 11, 59, tzinfo=UTC
    )
    assert record.fetched_at == FETCHED_AT
    assert record.raw_evidence_reference == (
        f"capture_0001_{BATCH}.gkg.csv.zip#row=1"
    )
    assert record.source_url == "https://publisher.example/markets/OGDC?story=1"
    assert record.source_metadata["publisher_body_retrieved"] is False
    assert record.source_metadata["archive_size_bytes"] == len(payload)
    assert record.source_metadata["archive_md5"] == hashlib.md5(
        payload, usedforsecurity=False
    ).hexdigest()
    assert record.source_metadata["archive_sha256"] == hashlib.sha256(
        payload
    ).hexdigest()
    assert "publisher_body_not_retrieved" in record.normalization_warnings
    assert capture.filename == f"{BATCH}.gkg.csv.zip"
    assert capture.content == payload
    assert capture.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.timestamp_validation["publication_delay"] == {
        "minimum_seconds": 60.0,
        "maximum_seconds": 60.0,
    }
    assert response.closed

    assert len(session.calls) == 1
    called_url, kwargs = session.calls[0]
    assert called_url == ARCHIVE_URL
    assert urlsplit(called_url).hostname == "data.gdeltproject.org"
    assert kwargs["allow_redirects"] is False
    assert kwargs["stream"] is True
    assert kwargs["proxies"] == {}
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["headers"]["Accept-Encoding"] == "identity"
    assert "publisher.example" not in called_url


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (
            _Response(
                b"",
                status_code=302,
                headers={"Location": "https://publisher.example/article"},
            ),
            "http_error",
        ),
        (
            _Response(b"not-wire-zip", headers={"Content-Encoding": "gzip"}),
            "unsupported_content_encoding",
        ),
    ],
)
def test_redirects_and_http_content_decoding_are_rejected(
    response: _Response, error_type: str
) -> None:
    payload = _zip_payload((_row(),))
    session = _Session((response,))
    result = _provider(_partition(payload), session).fetch(fetched_at=FETCHED_AT)
    assert result.records == ()
    assert result.captures == ()
    assert error_type in _error_types(result)
    assert len(session.calls) == 1
    assert session.calls[0][1]["allow_redirects"] is False


def test_publication_may_follow_batch_only_within_explicit_tolerance() -> None:
    payload = _zip_payload((_row(published="20260904130000"),))
    result = _provider(
        _partition(payload),
        _Session((_Response(payload),)),
        publication_after_batch_tolerance_seconds=3600,
    ).fetch(fetched_at=FETCHED_AT)
    assert len(result.records) == 1
    assert result.timestamp_validation["negative_publication_delay_count"] == 1
    assert result.records[0].source_metadata["publication_delay_seconds"] == -3600


@pytest.mark.parametrize(
    ("row", "error_type"),
    [
        (_row(collection="2"), "unsupported_source_collection"),
        (
            _row(record_id="20260904121500-1"),
            "archive_batch_mismatch",
        ),
        (_row(published="00000000000000"), "invalid_publication_time"),
        (
            _row(published="20260905120001"),
            "publication_after_batch_tolerance",
        ),
        (
            _row(published="20251215000000"),
            "record_outside_requested_interval",
        ),
    ],
)
def test_unsupported_or_incoherent_rows_are_quarantined(
    row: bytes, error_type: str
) -> None:
    payload = _zip_payload((row,))
    result = _provider(
        _partition(payload), _Session((_Response(payload),))
    ).fetch(fetched_at=FETCHED_AT)
    assert result.records == ()
    assert result.counters["records_rejected"] == 1
    assert error_type in _error_types(result)
    assert len(result.captures) == 1


@pytest.mark.parametrize(
    ("row", "options", "error_type"),
    [
        (b"only\ttwo", {}, "invalid_column_count"),
        (b"\xff\xfe", {}, "invalid_text_encoding"),
        (_row(), {"max_row_bytes": len(_row()) - 1}, "row_limit_exceeded"),
        (_row(), {"max_field_characters": 15}, "field_limit_exceeded"),
    ],
)
def test_tsv_schema_encoding_row_and_field_bounds_are_enforced(
    row: bytes, options: dict[str, int], error_type: str
) -> None:
    payload = _zip_payload((row,))
    result = _provider(
        _partition(payload),
        _Session((_Response(payload),)),
        **options,
    ).fetch(fetched_at=FETCHED_AT)
    assert result.records == ()
    assert error_type in _error_types(result)


@pytest.mark.parametrize(
    "partition",
    [
        lambda payload: _partition(payload, expected_size_bytes=len(payload) + 1),
        lambda payload: _partition(payload, expected_md5="0" * 32),
        lambda payload: _partition(payload, expected_sha256="0" * 64),
    ],
)
def test_size_md5_and_sha256_claims_are_all_enforced(partition: Any) -> None:
    payload = _zip_payload((_row(),))
    result = _provider(
        partition(payload), _Session((_Response(payload),))
    ).fetch(fetched_at=FETCHED_AT)
    assert result.records == ()
    assert result.captures == ()
    assert _error_types(result) & {"archive_size_mismatch", "archive_checksum_mismatch"}


@pytest.mark.parametrize(
    "payload_factory",
    [
        lambda: _zip_payload(
            (_row(),), extra_members=(("unexpected.txt", b"extra"),)
        ),
        lambda: _zip_payload(
            (_row(),), member_name=f"../{BATCH}.gkg.csv"
        ),
        lambda: _zip_payload((_row(),), symlink=True),
    ],
)
def test_zip_requires_one_expected_regular_member(payload_factory: Any) -> None:
    payload = payload_factory()
    result = _provider(
        _partition(payload), _Session((_Response(payload),))
    ).fetch(fetched_at=FETCHED_AT)
    assert result.records == ()
    assert result.captures == ()
    assert "unsafe_archive" in _error_types(result)


def test_compressed_and_decompressed_byte_limits_fail_closed() -> None:
    payload = _zip_payload((_row(),))
    compressed_response = _Response(payload)
    compressed = _provider(
        _partition(payload, expected_size_bytes=len(payload) - 1),
        _Session((compressed_response,)),
        max_compressed_bytes_per_file=len(payload) - 1,
    ).fetch(fetched_at=FETCHED_AT)
    assert "compressed_byte_limit_exceeded" in _error_types(compressed)
    assert compressed.captures == ()
    assert compressed_response.closed

    decompressed = _provider(
        _partition(payload),
        _Session((_Response(payload),)),
        max_decompressed_bytes_per_file=len(_row()),
    ).fetch(fetched_at=FETCHED_AT)
    assert "decompressed_byte_limit_exceeded" in _error_types(decompressed)
    assert decompressed.captures == ()


def test_streamed_bytes_override_a_false_small_content_length() -> None:
    payload = _zip_payload((_row(),))
    response = _Response(
        payload,
        headers={"Content-Length": "1"},
        chunks=(payload[:10], payload[10:]),
    )
    result = _provider(
        _partition(payload, expected_size_bytes=len(payload) - 1),
        _Session((response,)),
        max_compressed_bytes_per_file=len(payload) - 1,
    ).fetch(fetched_at=FETCHED_AT)
    assert "compressed_byte_limit_exceeded" in _error_types(result)


def test_record_cap_stops_before_requesting_later_partitions() -> None:
    first_payload = _zip_payload((_row(), _row(record_id=f"{BATCH}-2")))
    next_batch = "20260904121500"
    second_payload = _zip_payload(
        (_row(record_id=f"{next_batch}-1", published="20260904121400"),),
        batch=next_batch,
    )
    partitions = (
        _partition(first_payload, partition_id="first"),
        _partition(second_payload, batch=next_batch, partition_id="second"),
    )
    session = _Session((_Response(first_payload), _Response(second_payload)))
    provider = GdeltGkgArchiveProvider(
        partitions,
        requested_start=REQUESTED_START,
        requested_end=REQUESTED_END,
        permission_policy=_policy(),
        max_records=1,
        session=session,
        clock=lambda: FETCHED_AT,
    )
    result = provider.fetch(fetched_at=FETCHED_AT)
    assert "record_limit_exceeded" in _error_types(result)
    assert result.records == ()
    assert result.captures == ()
    assert len(session.calls) == 1


def test_validated_checkpoint_resumes_without_network_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    payload = _zip_payload((_row(),))
    partition = _partition(payload)
    checkpoint = tmp_path / "checkpoint"

    first_session = _Session((_Response(payload),))
    first = _provider(
        partition,
        first_session,
        checkpoint_directory=checkpoint,
    ).fetch(fetched_at=FETCHED_AT)
    assert first.counters["partitions_downloaded"] == 1
    assert first.counters["partitions_resumed"] == 0

    second_session = _Session(())
    second = _provider(
        partition,
        second_session,
        checkpoint_directory=checkpoint,
    ).fetch(fetched_at=FETCHED_AT)
    assert second_session.calls == []
    assert second.counters["partitions_downloaded"] == 0
    assert second.counters["partitions_resumed"] == 1
    assert second.records == first.records
    assert second.captures[0].content == payload

    cache_path = checkpoint / f"{partition.partition_id}-{partition.filename}"
    cache_path.write_bytes(b"tampered")
    third_session = _Session(())
    third = _provider(
        partition,
        third_session,
        checkpoint_directory=checkpoint,
    ).fetch(fetched_at=FETCHED_AT)
    assert third_session.calls == []
    assert third.records == ()
    assert third.captures == ()
    assert "checkpoint_cache_invalid" in _error_types(third)


def test_checkpoint_plan_is_bound_to_permission_policy_checksum(
    tmp_path: Path,
) -> None:
    payload = _zip_payload((_row(),))
    partition = _partition(payload)
    checkpoint = tmp_path / "checkpoint-policy"

    first = _provider(
        partition,
        _Session((_Response(payload),)),
        checkpoint_directory=checkpoint,
        permission_policy=_policy(policy_sha256="a" * 64),
    ).fetch(fetched_at=FETCHED_AT)
    assert first.counters["partitions_downloaded"] == 1

    second_session = _Session(())
    second = _provider(
        partition,
        second_session,
        checkpoint_directory=checkpoint,
        permission_policy=_policy(policy_sha256="c" * 64),
    ).fetch(fetched_at=FETCHED_AT)

    assert second_session.calls == []
    assert second.records == ()
    assert "checkpoint_plan_mismatch" in _error_types(second)


def test_config_loader_requires_explicit_partitions_and_parses_utc_range(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gkg.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "provider": GDELT_GKG_PROVIDER_ID,
                "requested_start": "2025-12-16T00:00:00Z",
                "requested_end": "2026-09-15T23:59:59Z",
                "permission_policy_id": POLICY_ID,
                "partitions": [
                    {
                        "partition_id": "enabled",
                        "archive_url": ARCHIVE_URL,
                        "expected_size_bytes": 1,
                        "expected_md5": "0" * 32,
                    },
                    {
                        "partition_id": "disabled",
                        "archive_url": (
                            "https://data.gdeltproject.org/gdeltv2/"
                            "20260904121500.gkg.csv.zip"
                        ),
                        "expected_size_bytes": 1,
                        "expected_md5": "0" * 32,
                        "enabled": False,
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    partitions = load_gdelt_gkg_config(path)
    assert [item.partition_id for item in partitions] == ["enabled"]
    provider = GdeltGkgArchiveProvider.from_config(
        path,
        permission_policy=_policy(),
        session=_Session(()),
    )
    assert provider.requested_start == REQUESTED_START
    assert provider.requested_end == REQUESTED_END
    assert provider.partitions == partitions

    with pytest.raises(EvidencePermissionError, match="differs"):
        GdeltGkgArchiveProvider.from_config(
            path,
            permission_policy=_policy(policy_id="different-policy"),
            session=_Session(()),
        )
