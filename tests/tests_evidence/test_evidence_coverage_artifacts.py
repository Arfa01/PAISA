from __future__ import annotations

import base64
import csv
from dataclasses import replace
from datetime import date, datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

import paisa.evidence.coverage_artifacts as coverage_artifacts
from paisa.evidence.artifacts import sha256_file
from paisa.evidence.coverage import compute_evidence_coverage
from paisa.evidence.coverage_artifacts import (
    EvidenceCoverageRunCollisionError,
    EvidenceCoverageVerificationError,
    _git_provenance,
    adapt_publisher_occurrences,
    make_record_identity_inventory,
    publish_evidence_coverage_run,
    verify_evidence_coverage_pack,
)
from paisa.evidence.models import EvidenceRecord
from paisa.news.coverage import CoverageConfig, NewsOccurrence, StockEvidence
from paisa.sentiment.artifacts import regular_file_inventory


UTC = timezone.utc
RUN_ID = "ECOV-20261023-120000-deadbeef"
PAYLOAD_FILES = (
    "evidence_session_mappings.csv",
    "final_summary.json",
    "input_evidence_reference.json",
    "missing_intervals.csv",
    "monthly_coverage.csv",
    "normalized_evidence.jsonl",
    "quality_report.json",
    "split_coverage.csv",
    "stock_artifact_reference.json",
    "stock_calendar.json",
)
ALL_FILES = set(PAYLOAD_FILES) | {"run_manifest.json", "completion.json"}


def _at(day: str, clock: str = "09:00:00") -> datetime:
    return datetime.fromisoformat(f"{day}T{clock}+00:00")


def _stock() -> StockEvidence:
    dates = tuple(
        date.fromisoformat(value)
        for value in (
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        )
    )
    return StockEvidence(
        symbol="OGDC",
        trading_dates=dates,
        split_dates={
            "train": dates[:2],
            "validation": dates[2:4],
            "test": dates[4:],
        },
        reference={
            "symbol": "OGDC",
            "vertical_slice_run_id": "VSL-20261020-OGDC-test-01",
            "vertical_manifest_sha256": "1" * 64,
            "price_artifact_reference": (
                "vertical_slice/VSL-20261020-OGDC-test-01/"
                "input_snapshot/psx_prices.csv"
            ),
            "price_artifact_sha256": "2" * 64,
            "price_date_range": {
                "start": "2024-01-01",
                "end": "2024-01-08",
            },
            "price_trading_day_count": 6,
            "model_training_run_id": "TRN-20261020-test0001",
            "training_manifest_sha256": "3" * 64,
            "training_data_manifest_sha256": "4" * 64,
            "training_input_snapshot_sha256": "2" * 64,
            "split_manifest_sha256": "5" * 64,
            "split_method": "chronological_target_date_70_15_15_no_shuffle",
            "split_counts": {"train": 2, "validation": 2, "test": 2},
            "split_ranges": {
                "train": {"start": "2024-01-01", "end": "2024-01-02"},
                "validation": {
                    "start": "2024-01-03",
                    "end": "2024-01-04",
                },
                "test": {"start": "2024-01-05", "end": "2024-01-08"},
            },
            "acquisition_mode": "test_verified_input",
            "provider": "test-provider",
            "is_real_data": True,
        },
    )


def _permission_reference() -> dict[str, object]:
    return {
        "policy_id": "test-evidence-policy",
        "schema_version": "1.0",
        "path": "configs/evidence/test-policy.json",
        "sha256": "6" * 64,
        "assessment_references": [
            {"path": "docs/test-assessment.md", "sha256": "7" * 64}
        ],
    }


def _input_references(
    *, include_sensitive_extras: bool = False, gdelt_record_count: int = 3
):
    if gdelt_record_count == 0:
        identity_records: tuple[EvidenceRecord, ...] = ()
    else:
        identity_records = _computation().source_records
        if gdelt_record_count == 4:
            identity_records = (
                *identity_records,
                _gkg("gkg-extra-reference", _at("2024-01-08")),
            )
        elif gdelt_record_count != 3:
            raise AssertionError("Unsupported fixture GDELT record count")
    identity_inventory = make_record_identity_inventory(identity_records)
    gdelt = {
        "input_kind": "gdelt_gkg",
        "run_id": "EVID-20261023-110000-feedface",
        "provider_id": "gdelt-gkg-2.1",
        "provider_mode": "raw_archive",
        "data_class": "real",
        "status": "PASS",
        "record_count": gdelt_record_count,
        "capture_count": 1,
        "manifest_sha256": "8" * 64,
        "payload_inventory_sha256": "9" * 64,
        "normalized_evidence_sha256": identity_inventory[
            "canonical_jsonl_sha256"
        ],
        "artifact_reference": "evidence/EVID-20261023-110000-feedface",
        "permission_policy_reference": _permission_reference(),
        "record_identity_inventory": identity_inventory,
    }
    if include_sensitive_extras:
        gdelt.update(
            {
                "path": "/Users/example/private/evidence",
                "file_uri": "file:/Users/example/private/evidence.jsonl",
                "network_file_uri": "file://server/private/evidence.jsonl",
                "posix_network_path": "//server/share/evidence.jsonl",
                "unc_path": r"\\server\share\evidence.jsonl",
                "safe_path_url": (
                    "https://www.investegate.co.uk/announcement/rns/"
                    "orcadian-energy--orca/-500-000-equity-financing-tvr-/7947964"
                ),
                "source_url": (
                    "https://example.com/archive?token=secret-value&part=1"
                ),
                "api_key": "do-not-retain",
            }
        )
    return (
        gdelt,
        {
            "input_kind": "psx_announcement_evidence",
            "provider_id": "psx-announcement",
            "provider_mode": "manual_import",
            "status": "BLOCKED_PERMISSION_REVIEW",
            "record_count": 0,
            "run_reference": None,
            "decision_id": "psx-review-v1",
            "permission_status": "REVIEW_REQUIRED",
            "automated_access": False,
            "permission_policy_reference": _permission_reference(),
        },
        {
            "input_kind": "publisher_news",
            "status": "NOT_SUPPLIED",
            "record_count": 0,
            "reason": "No publisher-news input supplied for this test.",
        },
        {
            "input_kind": "readiness_configuration",
            "schema_version": "1.0",
            "artifact_reference": "configs/evidence/default_readiness.json",
            "sha256": "b" * 64,
        },
    )


def _gkg(identifier: str, available_at: datetime) -> EvidenceRecord:
    record_id = f"{available_at:%Y%m%d%H%M%S}-1"
    archive_filename = f"{available_at:%Y%m%d%H%M%S}.gkg.csv.zip"
    return EvidenceRecord(
        evidence_id=identifier,
        evidence_type="gdelt_metadata",
        provider_id="gdelt-gkg-2.1",
        provider_mode="raw_archive",
        data_class="real",
        source_name="=example.com",
        source_url="https://example.com/story",
        source_record_id=record_id,
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        title=None,
        source_published_at=available_at,
        source_published_date=None,
        available_at=available_at,
        fetched_at=_at("2026-10-23"),
        imported_at=_at("2026-10-23"),
        historical_availability_basis="gdelt_gkg_batch_timestamp",
        time_precision="second",
        contemporaneous_source_id=record_id,
        contemporaneous_batch_at=available_at,
        permission_policy_id="test-evidence-policy",
        raw_evidence_reference=f"capture.zip#{identifier}",
        content_sha256="a" * 64,
        entity_match_evidence=(
            {
                "field": "V2ENHANCEDORGANIZATIONS",
                "alias": "OGDC",
                "matched_value": "OGDC",
                "match_method": "exact",
            },
        ),
        source_metadata={
            "GKGRECORDID": record_id,
            "V2.1DATE": f"{available_at:%Y%m%d%H%M%S}",
            "V2SOURCECOLLECTIONIDENTIFIER": "1",
            "V2SOURCECOMMONNAME": "=example.com",
            "V2DOCUMENTIDENTIFIER": "https://example.com/story",
            "V2ENHANCEDORGANIZATIONS": "OGDC,1",
            "archive_filename": archive_filename,
        },
    )


def _general_publisher(identifier: str, available_at: datetime) -> EvidenceRecord:
    run_id = "NEWS-20240101-000000-coverage"
    article_id = f"ARTICLE-{identifier}"
    evidence_identity = hashlib.sha256(
        (identifier + "\0" + f"{run_id}:{article_id}").encode("utf-8")
    ).hexdigest()[:24]
    return EvidenceRecord(
        evidence_id=f"EVID-NEWS-{evidence_identity}",
        evidence_type="publisher_news",
        provider_id="rss",
        provider_mode="real",
        data_class="real",
        source_name="Example News",
        source_url="https://example.com/news",
        source_record_id=identifier,
        ticker="GENERAL",
        company_name="General financial news",
        title="Financial news",
        source_published_at=available_at,
        source_published_date=None,
        available_at=available_at,
        fetched_at=available_at,
        imported_at=available_at,
        historical_availability_basis="news-availability-v1-conservative-acquisition",
        time_precision="second",
        contemporaneous_source_id=identifier,
        contemporaneous_batch_at=None,
        permission_policy_id="publisher-permission-v1",
        raw_evidence_reference=f"news_ingestion/{run_id}/feed.xml#{identifier}",
        content_sha256="d" * 64,
        source_metadata={
            "canonical_article_id": identifier,
            "representative_occurrence_key": f"{run_id}:{article_id}",
            "occurrence_count": 1,
            "ogdc_occurrence_count": 0,
            "occurrence_identities": [
                {
                    "occurrence_ordinal": 0,
                    "occurrence_key": f"{run_id}:{article_id}",
                    "ingestion_run_id": run_id,
                    "article_id": article_id,
                    "canonical_article_id": identifier,
                    "content_sha256": "d" * 64,
                    "is_ogdc": False,
                }
            ],
            "source_observed_at": None,
        },
    )


def _publisher_input_reference(
    records: tuple[EvidenceRecord, ...] | None = None,
    *,
    source_occurrences: tuple[NewsOccurrence, ...] = (),
) -> dict[str, object]:
    adapted_records = records or (
        _general_publisher("publisher-general", _at("2024-01-05")),
    )
    inventory = make_record_identity_inventory(adapted_records)
    sources_by_key = {item.occurrence_key: item for item in source_occurrences}
    occurrences_by_run: dict[
        str,
        list[tuple[EvidenceRecord, dict[str, object], NewsOccurrence | None]],
    ] = {}
    for record in adapted_records:
        for occurrence in record.source_metadata["occurrence_identities"]:
            run_id = str(occurrence["ingestion_run_id"])
            key = str(occurrence["occurrence_key"])
            source = sources_by_key.get(key)
            if source_occurrences and source is None:
                raise AssertionError("Missing source occurrence for publisher fixture")
            occurrences_by_run.setdefault(run_id, []).append(
                (record, occurrence, source)
            )

    article_fields = (
        "schema_version",
        "article_id",
        "raw_record_id",
        "provider",
        "source_name",
        "source_type",
        "source_url",
        "canonical_url",
        "title",
        "summary",
        "body",
        "published_at",
        "published_timezone",
        "source_observed_at",
        "available_at",
        "fetched_at",
        "language",
        "mentioned_tickers",
        "mentioned_companies",
        "mentioned_sectors",
        "content_sha256",
        "duplicate_of",
        "ingestion_run_id",
        "raw_artifact_reference",
        "normalization_warnings",
        "data_class",
    )
    link_fields = (
        "link_id",
        "article_id",
        "ticker",
        "company_name",
        "matched_alias",
        "match_method",
        "match_location",
        "confidence",
    )

    def csv_payload(fields, rows):
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        return stream.getvalue().encode("utf-8")

    run_references: list[dict[str, object]] = []
    for run_id, occurrence_records in sorted(occurrences_by_run.items()):
        article_rows = []
        link_rows = []
        raw_references: set[str] = set()
        for record, occurrence, source in occurrence_records:
            article_id = str(occurrence["article_id"])
            if source is not None:
                row_links = [dict(link) for link in source.entity_links]
            else:
                row_links = [
                    dict(link)
                    for link in record.entity_match_evidence
                    if link.get("ingestion_run_id") == run_id
                    and link.get("article_id") == article_id
                ]
            link_rows.extend(
                {field: link[field] for field in link_fields} for link in row_links
            )
            prefix = f"news_ingestion/{run_id}/"
            if source is not None:
                raw_reference = source.raw_artifact_reference
            elif record.raw_evidence_reference.startswith(prefix):
                raw_reference = record.raw_evidence_reference[len(prefix) :]
            else:
                raw_reference = f"raw/{article_id}.json"
            raw_references.add(raw_reference)
            article_rows.append(
                {
                    "schema_version": "1.1",
                    "article_id": article_id,
                    "raw_record_id": f"RAW-{article_id}",
                    "provider": source.provider if source else record.provider_id,
                    "source_name": source.source_name if source else record.source_name,
                    "source_type": "publisher",
                    "source_url": (
                        source.source_url if source else record.source_url
                    ) or "",
                    "canonical_url": (
                        source.source_url if source else record.source_url
                    ) or "",
                    "title": (source.title if source else record.title) or "",
                    "summary": "",
                    "body": "",
                    "published_at": (
                        (source.published_at if source else record.source_published_at).isoformat()
                        if (source.published_at if source else record.source_published_at)
                        else ""
                    ),
                    "published_timezone": (
                        source.published_timezone if source else "UTC"
                    ) or "",
                    "source_observed_at": (
                        source.source_observed_at.isoformat()
                        if source is not None and source.source_observed_at
                        else record.source_metadata.get("source_observed_at") or ""
                    ),
                    "available_at": (
                        (source.available_at if source else record.available_at).isoformat()
                        if (source.available_at if source else record.available_at)
                        else ""
                    ),
                    "fetched_at": (
                        source.fetched_at.isoformat()
                        if source is not None and source.fetched_at
                        else record.fetched_at.isoformat()
                    ),
                    "language": (source.language if source else record.language) or "",
                    "mentioned_tickers": json.dumps(
                        sorted(source.tickers)
                        if source is not None
                        else sorted({link["ticker"] for link in row_links})
                    ),
                    "mentioned_companies": json.dumps(
                        sorted(source.companies)
                        if source is not None
                        else sorted({link["company_name"] for link in row_links})
                    ),
                    "mentioned_sectors": json.dumps(
                        sorted(source.sectors) if source is not None else []
                    ),
                    "content_sha256": (
                        source.content_sha256
                        if source is not None
                        else str(occurrence["content_sha256"])
                    ),
                    "duplicate_of": (
                        source.duplicate_of
                        if source is not None
                        else str(occurrence["canonical_article_id"])
                    ) or "",
                    "ingestion_run_id": run_id,
                    "raw_artifact_reference": raw_reference,
                    "normalization_warnings": "[]",
                    "data_class": source.data_class if source else record.data_class,
                }
            )
        article_payload = csv_payload(article_fields, article_rows)
        link_payload = csv_payload(link_fields, link_rows)
        payloads = {
            "article_entity_links.csv": link_payload,
            "normalized_articles.csv": article_payload,
            **{raw: b"fixture raw evidence\n" for raw in raw_references},
        }
        artifact_inventory = [
            {
                "path": path,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
            for path, payload in sorted(payloads.items())
        ]
        inventory_sha256 = hashlib.sha256(
            json.dumps(
                artifact_inventory,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        source_snapshots = [
            {
                "path": path,
                "encoding": "base64url",
                "sha256": hashlib.sha256(payloads[path]).hexdigest(),
                "size_bytes": len(payloads[path]),
                "content_base64": base64.urlsafe_b64encode(payloads[path]).decode(
                    "ascii"
                ),
            }
            for path in ("article_entity_links.csv", "normalized_articles.csv")
        ]
        record = occurrence_records[0][0]
        run_manifest_payload = (
            json.dumps(
                {
                    "schema_version": "1.0",
                    "run_id": run_id,
                    "status": "PASS",
                    "provider": record.provider_id,
                    "provider_mode": record.provider_mode,
                    "finalization_state": "COMMITTED",
                    "completion_marker": "completion.json",
                    "artifact_inventory": artifact_inventory,
                    "artifact_inventory_sha256": inventory_sha256,
                },
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        run_manifest_sha256 = hashlib.sha256(run_manifest_payload).hexdigest()
        run_references.append(
            {
                "run_id": run_id,
                "status": "PASS",
                "provider": record.provider_id,
                "provider_mode": record.provider_mode,
                "run_manifest_sha256": run_manifest_sha256,
                "artifact_inventory_sha256": inventory_sha256,
                "normalized_articles_sha256": hashlib.sha256(
                    article_payload
                ).hexdigest(),
                "occurrence_count": len(occurrence_records),
                "artifact_reference": f"news_ingestion/{run_id}",
                "artifact_inventory": artifact_inventory,
                "source_snapshots": source_snapshots,
                "run_manifest_snapshot": {
                    "path": "run_manifest.json",
                    "encoding": "base64url",
                    "sha256": run_manifest_sha256,
                    "size_bytes": len(run_manifest_payload),
                    "content_base64": base64.urlsafe_b64encode(
                        run_manifest_payload
                    ).decode("ascii"),
                },
                "source_permission_decision": {
                    "passed": True,
                    "status": "PERMITTED",
                    "decision_id": record.permission_policy_id,
                },
            }
        )
    return {
        "input_kind": "publisher_news",
        "status": "LOADED",
        "record_count": len(adapted_records),
        "run_references": run_references,
        "database_fingerprints": {"provided": False, "files": []},
        "permission_policy_reference": _permission_reference(),
        "adapted_records_sha256": inventory["canonical_jsonl_sha256"],
        "adapted_record_inventory": inventory,
    }


def _computation():
    records = (
        _gkg("gkg-train", _at("2024-01-01")),
        _gkg("gkg-validation", _at("2024-01-03")),
        _gkg("gkg-test", _at("2024-01-05")),
    )
    return compute_evidence_coverage(records, _stock(), CoverageConfig())


def _publish(tmp_path: Path, *, run_id: str = RUN_ID) -> Path:
    return publish_evidence_coverage_run(
        _computation(),
        output_root=tmp_path / "coverage",
        input_references=_input_references(include_sensitive_extras=True),
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id=run_id,
        created_at=datetime(2026, 10, 23, 12, 0, tzinfo=UTC),
    )


def _json_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rewrite_json(path: Path, value) -> None:
    path.write_bytes(_json_bytes(value))
    os.chmod(path, 0o600)


def _reseal_payload(pack: Path) -> None:
    normalized_sha = sha256_file(pack / "normalized_evidence.jsonl")
    calendar_sha = sha256_file(pack / "stock_calendar.json")
    summary_path = pack / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["normalized_evidence_sha256"] = normalized_sha
    summary["stock_calendar_sha256"] = calendar_sha
    _rewrite_json(summary_path, summary)

    inventory = regular_file_inventory(pack, files=PAYLOAD_FILES)
    inventory_digest = hashlib.sha256(
        json.dumps(
            inventory,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = pack / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_artifact_inventory"] = inventory
    manifest["payload_artifact_inventory_sha256"] = inventory_digest
    manifest["normalized_evidence_sha256"] = normalized_sha
    manifest["stock_calendar_sha256"] = calendar_sha
    _rewrite_json(manifest_path, manifest)

    completion_path = pack / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["manifest_sha256"] = sha256_file(manifest_path)
    completion["manifest_size_bytes"] = manifest_path.stat().st_size
    completion["summary_sha256"] = sha256_file(summary_path)
    completion["summary_size_bytes"] = summary_path.stat().st_size
    completion["payload_artifact_inventory_sha256"] = inventory_digest
    _rewrite_json(completion_path, completion)


def _reseal_input_reference(pack: Path) -> None:
    reference_path = pack / "input_evidence_reference.json"
    stock_path = pack / "stock_artifact_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    stock = json.loads(stock_path.read_text(encoding="utf-8"))
    reference_sha256 = sha256_file(reference_path)
    fingerprint_sha256 = hashlib.sha256(
        json.dumps(
            {
                "input_references": reference,
                "stock_reference": stock,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    summary_path = pack / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["input_evidence_reference_sha256"] = reference_sha256
    _rewrite_json(summary_path, summary)
    manifest_path = pack / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_evidence_reference_sha256"] = reference_sha256
    manifest["input_fingerprint_sha256"] = fingerprint_sha256
    _rewrite_json(manifest_path, manifest)


def test_publish_verify_private_flat_pack_and_redact_references(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path)

    verification = verify_evidence_coverage_pack(pack)

    assert verification.run_id == RUN_ID
    assert verification.mapping_count == 3
    assert verification.artifact_count == len(PAYLOAD_FILES)
    assert not verification.to_dict()["source_evidence_bound"]
    assert {item.name for item in pack.iterdir()} == ALL_FILES
    assert pack.stat().st_mode & 0o077 == 0
    assert all(item.stat().st_mode & 0o077 == 0 for item in pack.iterdir())

    retained = b"\n".join(item.read_bytes() for item in pack.iterdir())
    assert b"secret-value" not in retained
    assert b"do-not-retain" not in retained
    assert b"/Users/example/private" not in retained
    references = json.loads(
        (pack / "input_evidence_reference.json").read_text(encoding="utf-8")
    )
    assert references["inputs"][0]["path"] == "[REDACTED_ABSOLUTE_PATH]"
    assert references["inputs"][0]["file_uri"] == "[REDACTED_FILE_URI]"
    assert references["inputs"][0]["network_file_uri"] == (
        "[REDACTED_FILE_URI]"
    )
    assert references["inputs"][0]["posix_network_path"] == (
        "[REDACTED_ABSOLUTE_PATH]"
    )
    assert references["inputs"][0]["unc_path"] == "[REDACTED_ABSOLUTE_PATH]"
    assert references["inputs"][0]["safe_path_url"].endswith(
        "equity-financing-tvr-/7947964"
    )
    assert references["inputs"][0]["api_key"] == "[REDACTED]"

    with (pack / "evidence_session_mappings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["provider_id"] == "gdelt-gkg-2.1"


def test_coverage_git_provenance_reports_clean_and_dirty_states(
    tmp_path: Path,
) -> None:
    subprocess.run(
        ["git", "init", "--quiet"], cwd=tmp_path, check=True, timeout=10
    )

    assert _git_provenance(tmp_path)["dirty_tree"] is False

    (tmp_path / "untracked.txt").write_text("change\n", encoding="utf-8")
    assert _git_provenance(tmp_path)["dirty_tree"] is True


def test_publication_refuses_collision_and_dangling_destination(
    tmp_path: Path,
) -> None:
    _publish(tmp_path)
    with pytest.raises(EvidenceCoverageRunCollisionError):
        _publish(tmp_path)

    dangling_id = "ECOV-20261023-120001-cafebabe"
    dangling = tmp_path / "coverage" / dangling_id
    dangling.symlink_to(tmp_path / "does-not-exist", target_is_directory=True)
    with pytest.raises(EvidenceCoverageRunCollisionError):
        _publish(tmp_path, run_id=dangling_id)


def test_checksum_tamper_and_extra_artifact_are_rejected(tmp_path: Path) -> None:
    pack = _publish(tmp_path)
    with (pack / "quality_report.json").open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(EvidenceCoverageVerificationError, match="inventory"):
        verify_evidence_coverage_pack(pack)

    second = _publish(
        tmp_path,
        run_id="ECOV-20261023-120002-00ddf00d",
    )
    extra = second / "unexpected.txt"
    extra.write_text("unexpected\n", encoding="utf-8")
    os.chmod(extra, 0o600)
    with pytest.raises(EvidenceCoverageVerificationError, match="artifact set"):
        verify_evidence_coverage_pack(second)


def test_resealed_split_contradiction_is_rejected_semantically(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path)
    split_path = pack / "split_coverage.csv"
    with split_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    target = next(
        row
        for row in rows
        if row["profile"] == "gdelt_gkg_ogdc" and row["split"] == "train"
    )
    target["record_count"] = str(int(target["record_count"]) + 1)
    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(split_path, 0o600)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="contradictory|disagree|recomputation",
    ):
        verify_evidence_coverage_pack(pack)


def test_resealed_source_reference_without_checksum_is_rejected(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path)
    reference_path = pack / "input_evidence_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    reference["inputs"][0].pop("manifest_sha256")
    _rewrite_json(reference_path, reference)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="manifest_sha256",
    ):
        verify_evidence_coverage_pack(pack)


def test_publication_rejects_unbound_upstream_record_identity_inventory(
    tmp_path: Path,
) -> None:
    references = list(_input_references())
    gdelt = dict(references[0])
    empty_records: list[dict[str, str]] = []
    gdelt["record_identity_inventory"] = {
        "schema_version": "1.0",
        "algorithm": "sha256",
        "records": empty_records,
        "sha256": hashlib.sha256(
            json.dumps(
                empty_records,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    references[0] = gdelt

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="record identity|identity inventory|record_identity_inventory",
    ):
        publish_evidence_coverage_run(
            _computation(),
            output_root=tmp_path / "coverage",
            input_references=references,
            stock_reference=_stock().reference,
            repository_root=tmp_path,
            run_id="ECOV-20261023-120032-unbound1",
        )


@pytest.mark.parametrize("input_kind", ("gdelt_gkg", "psx_announcement_evidence"))
@pytest.mark.parametrize("inventory_present", (False, True))
def test_zero_record_source_requires_empty_jsonl_digest_even_with_inventory(
    tmp_path: Path,
    input_kind: str,
    inventory_present: bool,
) -> None:
    record = _general_publisher("publisher-empty-source", _at("2024-01-05"))
    references = list(_input_references(gdelt_record_count=0))
    references[2] = _publisher_input_reference((record,))
    forged_empty_inventory = make_record_identity_inventory(())
    forged_empty_inventory["canonical_jsonl_sha256"] = "a" * 64
    inventory_document = {
        key: value
        for key, value in forged_empty_inventory.items()
        if key != "sha256"
    }
    forged_empty_inventory["sha256"] = hashlib.sha256(
        json.dumps(
            inventory_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    if input_kind == "gdelt_gkg":
        source = dict(references[0])
        if inventory_present:
            source["record_identity_inventory"] = forged_empty_inventory
        else:
            source.pop("record_identity_inventory")
        source["normalized_evidence_sha256"] = "a" * 64
        references[0] = source
    else:
        psx = dict(references[1])
        psx.update(
            {
                "status": "LOADED",
                "record_count": 0,
                "run_reference": {
                    "run_id": "EVID-20261023-110001-psxempty",
                    "provider_id": "psx-announcement",
                    "provider_mode": "manual_import",
                    "status": "PASS",
                    "record_count": 0,
                    "capture_count": 1,
                    "manifest_sha256": "8" * 64,
                    "payload_inventory_sha256": "9" * 64,
                    "normalized_evidence_sha256": "a" * 64,
                    "artifact_reference": (
                        "evidence/EVID-20261023-110001-psxempty"
                    ),
                    **(
                        {"record_identity_inventory": forged_empty_inventory}
                        if inventory_present
                        else {}
                    ),
                },
            }
        )
        references[1] = psx

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="empty source",
    ):
        publish_evidence_coverage_run(
            compute_evidence_coverage(
                (record,),
                _stock(),
                CoverageConfig(),
            ),
            output_root=tmp_path / "coverage",
            input_references=references,
            stock_reference=_stock().reference,
            repository_root=tmp_path,
            run_id=(
                "ECOV-20261023-120036-emptygkg"
                if input_kind == "gdelt_gkg"
                else "ECOV-20261023-120037-emptypsx"
            ),
        )


def test_resealed_title_only_mutation_breaks_upstream_jsonl_binding(
    tmp_path: Path,
) -> None:
    pack = _publish(
        tmp_path,
        run_id="ECOV-20261023-120034-title001",
    )
    evidence_path = pack / "normalized_evidence.jsonl"
    rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["title"] = "A title absent from the upstream EVID artifact"
    evidence_path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.chmod(evidence_path, 0o600)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="identity inventory|checksum-bound source JSONL",
    ):
        verify_evidence_coverage_pack(pack)


def test_resealed_record_and_inventory_cannot_rebind_upstream_jsonl(
    tmp_path: Path,
) -> None:
    pack = _publish(
        tmp_path,
        run_id="ECOV-20261023-120035-rebind01",
    )
    evidence_path = pack / "normalized_evidence.jsonl"
    rows = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    old_id = rows[0]["evidence_id"]
    rows[0]["evidence_id"] = f"{old_id}-forged"
    evidence_path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.chmod(evidence_path, 0o600)

    reference_path = pack / "input_evidence_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    inventory = reference["inputs"][0]["record_identity_inventory"]
    entry = next(
        item for item in inventory["records"] if item["evidence_id"] == old_id
    )
    entry["evidence_id"] = rows[0]["evidence_id"]
    canonical_row = json.dumps(
        rows[0],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    entry["normalized_record_sha256"] = hashlib.sha256(
        canonical_row
    ).hexdigest()
    inventory_document = {
        key: value for key, value in inventory.items() if key != "sha256"
    }
    inventory["sha256"] = hashlib.sha256(
        json.dumps(
            inventory_document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    _rewrite_json(reference_path, reference)
    _reseal_input_reference(pack)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="checksum-bound source JSONL",
    ):
        verify_evidence_coverage_pack(pack)


def test_publication_rejects_publisher_link_from_unreferenced_run(
    tmp_path: Path,
) -> None:
    general = _general_publisher("publisher-linked", _at("2024-01-05"))
    article_id = "ARTICLE-linked"
    link_id = "NEWSLINK-" + hashlib.sha256(
        "\0".join((article_id, "OGDC", "OGDC", "title", "exact")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    linked = replace(
        general,
        ticker="OGDC",
        company_name="Oil and Gas Development Company Limited",
        raw_evidence_reference=(
            "news_ingestion/NEWS-20240103-000000-unreferenced/"
            "feed.xml#publisher-linked"
        ),
        entity_match_evidence=(
            {
                "ingestion_run_id": "NEWS-20240103-000000-unreferenced",
                "link_id": link_id,
                "article_id": article_id,
                "ticker": "OGDC",
                "company_name": "Oil and Gas Development Company Limited",
                "matched_alias": "OGDC",
                "match_method": "exact",
                "match_location": "title",
                "confidence": "exact",
            },
        ),
        source_metadata={
            **general.source_metadata,
            "representative_occurrence_key": (
                "NEWS-20240103-000000-unreferenced:ARTICLE-linked"
            ),
            "occurrence_count": 2,
            "ogdc_occurrence_count": 1,
            "occurrence_identities": [
                {
                    "occurrence_ordinal": 0,
                    "occurrence_key": (
                        "NEWS-20240103-000000-unreferenced:ARTICLE-linked"
                    ),
                    "ingestion_run_id": (
                        "NEWS-20240103-000000-unreferenced"
                    ),
                    "article_id": article_id,
                    "canonical_article_id": "publisher-linked",
                    "content_sha256": "d" * 64,
                    "is_ogdc": True,
                },
                {
                    "occurrence_ordinal": 1,
                    "occurrence_key": (
                        "NEWS-20240101-000000-coverage:"
                        "ARTICLE-publisher-linked"
                    ),
                    "ingestion_run_id": "NEWS-20240101-000000-coverage",
                    "article_id": "ARTICLE-publisher-linked",
                    "canonical_article_id": "publisher-linked",
                    "content_sha256": "d" * 64,
                    "is_ogdc": False,
                },
            ],
        },
    )
    publisher = _publisher_input_reference((linked,))
    publisher["run_references"] = [
        run
        for run in publisher["run_references"]
        if run["run_id"] != "NEWS-20240103-000000-unreferenced"
    ]
    references = list(_input_references(gdelt_record_count=0))
    references[2] = publisher

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="checksum-bound NEWS snapshots|upstream run identity",
    ):
        publish_evidence_coverage_run(
            compute_evidence_coverage(
                (linked,),
                _stock(),
                CoverageConfig(),
            ),
            output_root=tmp_path / "coverage",
            input_references=references,
            stock_reference=_stock().reference,
            repository_root=tmp_path,
            run_id="ECOV-20261023-120033-linkrun1",
        )


def test_resealed_embedded_private_path_is_rejected(tmp_path: Path) -> None:
    pack = _publish(
        tmp_path,
        run_id="ECOV-20261023-120031-pathsafe",
    )
    summary_path = pack / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["extra_note"] = "stored at //server/share/private.csv"
    _rewrite_json(summary_path, summary)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="absolute host path",
    ):
        verify_evidence_coverage_pack(pack)


def test_symlink_artifact_and_non_private_mode_are_rejected(tmp_path: Path) -> None:
    pack = _publish(tmp_path)
    target = pack / "quality_report.json"
    saved = tmp_path / "saved-quality.json"
    saved.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(saved)
    with pytest.raises(
        EvidenceCoverageVerificationError, match="only regular files"
    ):
        verify_evidence_coverage_pack(pack)

    second = _publish(
        tmp_path,
        run_id="ECOV-20261023-120003-abcd1234",
    )
    os.chmod(second / "quality_report.json", 0o644)
    with pytest.raises(
        EvidenceCoverageVerificationError, match="not private|mode must be"
    ):
        verify_evidence_coverage_pack(second)


def test_copied_coverage_pack_with_public_modes_is_rejected(tmp_path: Path) -> None:
    pack = _publish(tmp_path)
    copied = tmp_path / "copied" / RUN_ID
    copied.parent.mkdir()
    shutil.copytree(pack, copied)

    os.chmod(copied, 0o755)
    with pytest.raises(EvidenceCoverageVerificationError, match="0700"):
        verify_evidence_coverage_pack(copied)

    os.chmod(copied, 0o700)
    os.chmod(copied / "quality_report.json", 0o644)
    with pytest.raises(EvidenceCoverageVerificationError, match="0600"):
        verify_evidence_coverage_pack(copied)


def test_coverage_verifier_rejects_symlinked_parent_component(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path)
    alias = tmp_path / "coverage-alias"
    alias.symlink_to(pack.parent, target_is_directory=True)

    with pytest.raises(
        EvidenceCoverageVerificationError, match="symlink components"
    ):
        verify_evidence_coverage_pack(alias / pack.name)


def test_zero_match_evidence_audit_still_produces_a_valid_pack(
    tmp_path: Path,
) -> None:
    computation = compute_evidence_coverage((), _stock(), CoverageConfig())
    pack = publish_evidence_coverage_run(
        computation,
        output_root=tmp_path / "coverage",
        input_references=_input_references(gdelt_record_count=0),
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id="ECOV-20261023-120004-zero0000",
        created_at=datetime(2026, 10, 23, 12, 4, tzinfo=UTC),
    )

    verification = verify_evidence_coverage_pack(pack)

    assert verification.mapping_count == 0
    assert set(verification.readiness_status_by_profile.values()) == {
        "NO_OVERLAP"
    }


def test_publication_rejects_untyped_or_checksum_free_source_references(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="source reference is invalid",
    ):
        publish_evidence_coverage_run(
            _computation(),
            output_root=tmp_path / "coverage",
            input_references={"run_id": "fabricated"},
            stock_reference=_stock().reference,
            repository_root=tmp_path,
            run_id="ECOV-20261023-120005-badref00",
        )

    invalid_stock = dict(_stock().reference)
    invalid_stock.pop("split_manifest_sha256")
    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="split_manifest_sha256",
    ):
        publish_evidence_coverage_run(
            _computation(),
            output_root=tmp_path / "coverage",
            input_references=_input_references(),
            stock_reference=invalid_stock,
            repository_root=tmp_path,
            run_id="ECOV-20261023-120006-badstock",
        )


def test_publication_verifies_staging_before_commit_and_run_id_is_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_verify = coverage_artifacts._verify_root

    def fail_staging(*args, **kwargs):
        del args, kwargs
        raise EvidenceCoverageVerificationError("injected staging failure")

    monkeypatch.setattr(coverage_artifacts, "_verify_root", fail_staging)
    with pytest.raises(
        EvidenceCoverageVerificationError, match="injected staging failure"
    ):
        _publish(tmp_path)

    output = tmp_path / "coverage"
    assert not (output / RUN_ID).exists()
    assert not list(output.glob(f".{RUN_ID}.staging-*"))

    monkeypatch.setattr(coverage_artifacts, "_verify_root", original_verify)
    assert _publish(tmp_path).name == RUN_ID


def test_duplicate_evidence_id_failure_is_unpublished_and_run_id_is_reusable(
    tmp_path: Path,
) -> None:
    run_id = "ECOV-20261023-120007-duplicate"
    computation = _computation()
    duplicated = replace(
        computation,
        source_records=(
            *computation.source_records,
            computation.source_records[0],
        ),
        summary={**computation.summary, "record_count_input": 4},
    )
    output = tmp_path / "coverage"

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match=(
            "identities are not canonical and unique|record identity inventory|"
            "identities are duplicated"
        ),
    ):
        publish_evidence_coverage_run(
            duplicated,
            output_root=output,
            input_references=_input_references(gdelt_record_count=4),
            stock_reference=_stock().reference,
            repository_root=tmp_path,
            run_id=run_id,
            created_at=datetime(2026, 10, 23, 12, 7, tzinfo=UTC),
        )

    assert not os.path.lexists(output / run_id)
    assert not tuple(output.glob(f".{run_id}.staging-*"))

    corrected = _publish(tmp_path, run_id=run_id)
    assert corrected == output / run_id
    assert verify_evidence_coverage_pack(corrected).mapping_count == 3


@pytest.mark.parametrize(
    ("field", "replacement", "run_id"),
    [
        (
            "profiles",
            '["gdelt_gkg_ogdc","combined_ogdc_evidence",'
            '"psx_ogdc_announcements"]',
            "ECOV-20261023-120010-profile1",
        ),
        ("assigned_session", "2024-01-02", "ECOV-20261023-120011-session1"),
        (
            "decision_cutoff_utc",
            "2024-01-01T09:00:01Z",
            "ECOV-20261023-120012-cutoff01",
        ),
        (
            "available_at",
            "2024-01-01T10:31:00Z",
            "ECOV-20261023-120013-late0001",
        ),
    ],
)
def test_resealed_mapping_labels_cannot_override_record_calendar_recomputation(
    tmp_path: Path,
    field: str,
    replacement: str,
    run_id: str,
) -> None:
    pack = _publish(tmp_path, run_id=run_id)
    mapping_path = pack / "evidence_session_mappings.csv"
    with mapping_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[0][field] = replacement
    with mapping_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(mapping_path, 0o600)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="recomputation|later than its claimed cutoff",
    ):
        verify_evidence_coverage_pack(pack)


@pytest.mark.parametrize(
    ("field", "replacement", "run_id"),
    [
        (
            "provider_id",
            "totally-fake-provider",
            "ECOV-20261023-120020-provider",
        ),
        (
            "evidence_type",
            "exchange_announcement",
            "ECOV-20261023-120021-type0001",
        ),
        ("data_class", "imported", "ECOV-20261023-120022-class001"),
        ("time_precision", "minute", "ECOV-20261023-120023-precis01"),
        ("ticker", "GENERAL", "ECOV-20261023-120024-ticker01"),
        (
            "contemporaneous_source_id",
            "20240101090000-999",
            "ECOV-20261023-120025-source01",
        ),
    ],
)
def test_resealed_unknown_or_mismatched_provider_contract_is_rejected(
    tmp_path: Path,
    field: str,
    replacement: str,
    run_id: str,
) -> None:
    pack = _publish(tmp_path, run_id=run_id)
    evidence_path = pack / "normalized_evidence.jsonl"
    rows = [json.loads(line) for line in evidence_path.read_text("utf-8").splitlines()]
    rows[0][field] = replacement
    evidence_path.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    os.chmod(evidence_path, 0o600)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError, match="evidence contract failed"
    ):
        verify_evidence_coverage_pack(pack)


def test_resealed_general_publisher_label_cannot_spoof_ogdc_membership(
    tmp_path: Path,
) -> None:
    record = _general_publisher("publisher-general", _at("2024-01-05"))
    computation = compute_evidence_coverage(
        (record,),
        _stock(),
        CoverageConfig(),
    )
    references = list(_input_references(gdelt_record_count=0))
    references[2] = _publisher_input_reference()
    pack = publish_evidence_coverage_run(
        computation,
        output_root=tmp_path / "coverage",
        input_references=references,
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id="ECOV-20261023-120026-label001",
    )
    assert verify_evidence_coverage_pack(pack).mapping_count == 1

    evidence_path = pack / "normalized_evidence.jsonl"
    row = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert row["source_metadata"]["ogdc_occurrence_count"] == 0
    assert row["entity_match_evidence"] == []
    row["ticker"] = "OGDC"
    row["company_name"] = "Oil and Gas Development Company Limited"
    evidence_path.write_text(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(evidence_path, 0o600)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="evidence contract failed",
    ):
        verify_evidence_coverage_pack(pack)


def test_fully_resealed_publisher_link_cannot_rebind_news_source_snapshots(
    tmp_path: Path,
) -> None:
    record = _general_publisher("publisher-general", _at("2024-01-05"))
    computation = compute_evidence_coverage(
        (record,),
        _stock(),
        CoverageConfig(),
    )
    references = list(_input_references(gdelt_record_count=0))
    references[2] = _publisher_input_reference((record,))
    pack = publish_evidence_coverage_run(
        computation,
        output_root=tmp_path / "coverage",
        input_references=references,
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id="ECOV-20261023-120027-linkseal",
    )

    evidence_path = pack / "normalized_evidence.jsonl"
    row = json.loads(evidence_path.read_text(encoding="utf-8"))
    occurrence = row["source_metadata"]["occurrence_identities"][0]
    article_id = occurrence["article_id"]
    run_id = occurrence["ingestion_run_id"]
    link_id = "NEWSLINK-" + hashlib.sha256(
        "\0".join((article_id, "OGDC", "OGDC", "title", "exact")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    row["ticker"] = "OGDC"
    row["company_name"] = "Oil and Gas Development Company Limited"
    row["source_metadata"]["ogdc_occurrence_count"] = 1
    occurrence["is_ogdc"] = True
    row["entity_match_evidence"] = [
        {
            "ingestion_run_id": run_id,
            "link_id": link_id,
            "article_id": article_id,
            "ticker": "OGDC",
            "company_name": "Oil and Gas Development Company Limited",
            "matched_alias": "OGDC",
            "match_method": "exact",
            "match_location": "title",
            "confidence": "exact",
        }
    ]
    evidence_path.write_text(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.chmod(evidence_path, 0o600)

    reference_path = pack / "input_evidence_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    publisher = next(
        item
        for item in reference["inputs"]
        if item["input_kind"] == "publisher_news"
    )
    forged_inventory = make_record_identity_inventory(
        (EvidenceRecord.from_dict(row),)
    )
    publisher["adapted_records_sha256"] = forged_inventory[
        "canonical_jsonl_sha256"
    ]
    publisher["adapted_record_inventory"] = forged_inventory
    _rewrite_json(reference_path, reference)
    _reseal_input_reference(pack)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="checksum-bound NEWS snapshots",
    ):
        verify_evidence_coverage_pack(pack)


def test_resealed_news_snapshots_cannot_rebind_original_run_manifest(
    tmp_path: Path,
) -> None:
    record = _general_publisher("publisher-general", _at("2024-01-05"))
    references = list(_input_references(gdelt_record_count=0))
    references[2] = _publisher_input_reference((record,))
    pack = publish_evidence_coverage_run(
        compute_evidence_coverage((record,), _stock(), CoverageConfig()),
        output_root=tmp_path / "coverage",
        input_references=references,
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id="ECOV-20261023-120038-manifest",
    )

    reference_path = pack / "input_evidence_reference.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    publisher = next(
        item
        for item in reference["inputs"]
        if item["input_kind"] == "publisher_news"
    )
    run = publisher["run_references"][0]
    original_manifest_identity = run["run_manifest_sha256"]
    original_manifest_snapshot = dict(run["run_manifest_snapshot"])
    snapshot = next(
        item
        for item in run["source_snapshots"]
        if item["path"] == "normalized_articles.csv"
    )
    source = base64.urlsafe_b64decode(snapshot["content_base64"])
    with io.StringIO(source.decode("utf-8"), newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    rows[0]["title"] = "Forged title outside the retained NEWS manifest"
    changed = io.StringIO(newline="")
    writer = csv.DictWriter(changed, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    changed_payload = changed.getvalue().encode("utf-8")
    changed_sha = hashlib.sha256(changed_payload).hexdigest()
    snapshot.update(
        {
            "sha256": changed_sha,
            "size_bytes": len(changed_payload),
            "content_base64": base64.urlsafe_b64encode(changed_payload).decode(
                "ascii"
            ),
        }
    )
    inventory_entry = next(
        item
        for item in run["artifact_inventory"]
        if item["path"] == "normalized_articles.csv"
    )
    inventory_entry.update(
        {"sha256": changed_sha, "size_bytes": len(changed_payload)}
    )
    run["normalized_articles_sha256"] = changed_sha
    run["artifact_inventory_sha256"] = hashlib.sha256(
        json.dumps(
            run["artifact_inventory"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert run["run_manifest_sha256"] == original_manifest_identity
    assert run["run_manifest_snapshot"] == original_manifest_snapshot
    _rewrite_json(reference_path, reference)
    _reseal_input_reference(pack)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="not bound to its run manifest",
    ):
        verify_evidence_coverage_pack(pack)


def test_portable_publisher_link_uses_first_linked_availability_session(
    tmp_path: Path,
) -> None:
    canonical_id = "CANONICAL-DELAYED-OGDC"
    general_run = "NEWS-20240101-000000-general"
    linked_run = "NEWS-20240103-000000-linked"
    linked_article = "ARTICLE-LINKED"
    link_id = "NEWSLINK-" + hashlib.sha256(
        "\0".join(
            (linked_article, "OGDC", "OGDC", "title", "exact")
        ).encode("utf-8")
    ).hexdigest()[:24]
    general = NewsOccurrence(
        occurrence_key=f"{general_run}:ARTICLE-GENERAL",
        ingestion_run_id=general_run,
        article_id="ARTICLE-GENERAL",
        canonical_article_id=canonical_id,
        provider="rss",
        provider_mode="real",
        source_name="Example News",
        source_url="https://example.com/general",
        title="General market story",
        published_at=_at("2024-01-01"),
        available_at=_at("2024-01-01"),
        fetched_at=_at("2024-01-01"),
        published_timezone="UTC",
        language="english",
        tickers=(),
        companies=(),
        sectors=(),
        content_sha256="d" * 64,
        raw_artifact_reference="raw/general.json",
        data_class="real",
        duplicate_of=canonical_id,
        entity_links=(),
    )
    linked = NewsOccurrence(
        occurrence_key=f"{linked_run}:{linked_article}",
        ingestion_run_id=linked_run,
        article_id=linked_article,
        canonical_article_id=canonical_id,
        provider="rss",
        provider_mode="real",
        source_name="Example News",
        source_url="https://example.com/ogdc",
        title="OGDC company update",
        published_at=_at("2024-01-03"),
        available_at=_at("2024-01-03"),
        fetched_at=_at("2024-01-03"),
        published_timezone="UTC",
        language="english",
        tickers=("OGDC",),
        companies=("Oil and Gas Development Company Limited",),
        sectors=(),
        content_sha256="e" * 64,
        raw_artifact_reference="raw/linked.json",
        data_class="real",
        duplicate_of=canonical_id,
        entity_links=(
            {
                "link_id": link_id,
                "article_id": linked_article,
                "ticker": "OGDC",
                "company_name": "Oil and Gas Development Company Limited",
                "matched_alias": "OGDC",
                "match_method": "exact",
                "match_location": "title",
                "confidence": "exact",
            },
        ),
    )
    permissions = {
        general_run: "publisher-permission-v1",
        linked_run: "publisher-permission-v1",
    }
    record = adapt_publisher_occurrences((general, linked), permissions)[0]
    assert record.available_at == _at("2024-01-03")
    assert record.raw_evidence_reference == (
        f"news_ingestion/{linked_run}/raw/linked.json"
    )

    references = list(_input_references(gdelt_record_count=0))
    references[2] = _publisher_input_reference(
        (record,),
        source_occurrences=(general, linked),
    )
    pack = publish_evidence_coverage_run(
        compute_evidence_coverage((record,), _stock(), CoverageConfig()),
        output_root=tmp_path / "coverage",
        input_references=references,
        stock_reference=_stock().reference,
        repository_root=tmp_path,
        run_id="ECOV-20261023-120039-linktime",
    )

    assert verify_evidence_coverage_pack(pack).mapping_count == 1
    with (pack / "evidence_session_mappings.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        mapping = next(csv.DictReader(handle))
    assert mapping["ticker"] == "OGDC"
    assert mapping["assigned_session"] == "2024-01-03"


def test_resealed_stock_calendar_cannot_redirect_session_mapping(
    tmp_path: Path,
) -> None:
    pack = _publish(tmp_path, run_id="ECOV-20261023-120030-calendar")
    calendar_path = pack / "stock_calendar.json"
    calendar = json.loads(calendar_path.read_text(encoding="utf-8"))
    calendar["decision_cutoff"] = "15:29:59"
    _rewrite_json(calendar_path, calendar)
    _reseal_payload(pack)

    with pytest.raises(
        EvidenceCoverageVerificationError,
        match="recomputation|calendar",
    ):
        verify_evidence_coverage_pack(pack)
