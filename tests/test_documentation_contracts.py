from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEMO_RUN_ID = "VSL-20261020-OGDC-main-02"
TRAINING_RUN_ID = "TRN-20261020-191815-624662-nogit"
DEMO_PACK_DOCUMENTS = (
    "docs/ARCHITECTURE_OVERVIEW.md",
    "docs/CURRENT_IMPLEMENTATION_STATUS.md",
    "docs/KNOWN_LIMITATIONS.md",
    "docs/NEXT_PHASE_PLAN.md",
    "docs/PROGRESS_REPORT_OCTOBER_2026.md",
    "docs/RESULTS_INTERPRETATION.md",
    "docs/SUPERVISOR_DEMO_SCRIPT.md",
    "docs/TEAM_HANDOFF.md",
)


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return re.sub(r"[\s>]+", " ", value).strip().lower()


def _artifact_gate() -> str:
    document = _read("docs/SUPERVISOR_DEMO_SCRIPT.md")
    match = re.search(
        r"<!-- BEGIN DEMO_ARTIFACT_GATE -->\s*"
        r"```bash\n(?P<commands>.*?)\n```\s*"
        r"<!-- END DEMO_ARTIFACT_GATE -->",
        document,
        flags=re.DOTALL,
    )
    assert match is not None, "demo documentation must expose one testable artifact gate"
    return match.group("commands")


def _artifact_restore() -> str:
    document = _read("docs/TEAM_HANDOFF.md")
    match = re.search(
        r"<!-- BEGIN DEMO_ARTIFACT_RESTORE -->\s*"
        r"```bash\n(?P<commands>.*?)\n```\s*"
        r"<!-- END DEMO_ARTIFACT_RESTORE -->",
        document,
        flags=re.DOTALL,
    )
    assert match is not None, "team handoff must expose one testable restore block"
    return match.group("commands")


def _artifact_export() -> str:
    document = _read("docs/TEAM_HANDOFF.md")
    match = re.search(
        r"<!-- BEGIN DEMO_ARTIFACT_EXPORT -->\s*"
        r"```bash\n(?P<commands>.*?)\n```\s*"
        r"<!-- END DEMO_ARTIFACT_EXPORT -->",
        document,
        flags=re.DOTALL,
    )
    assert match is not None, "team handoff must expose one testable export block"
    return match.group("commands")


def _news_artifact_commands(kind: str) -> str:
    document = _read("docs/TEAM_HANDOFF.md")
    label = kind.upper()
    match = re.search(
        rf"<!-- BEGIN NEWS_ARTIFACT_{label} -->\s*"
        rf"```bash\n(?P<commands>.*?)\n```\s*"
        rf"<!-- END NEWS_ARTIFACT_{label} -->",
        document,
        flags=re.DOTALL,
    )
    assert match is not None, f"team handoff must expose news {kind} commands"
    return match.group("commands")


def _bash_blocks(relative_path: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r"```bash\n(.*?)\n```", _read(relative_path), flags=re.DOTALL)
    )


def test_demo_artifact_gate_fails_visibly_when_required_pair_is_missing(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        ["bash", "-c", _artifact_gate()],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ERROR: required demo artifact is missing" in completed.stderr
    assert "docs/TEAM_HANDOFF.md" in completed.stderr
    assert f"artifacts/vertical_slice/{DEMO_RUN_ID}/manifest.json" in completed.stderr
    assert f"artifacts/vertical_slice/{DEMO_RUN_ID}/final_summary.json" in completed.stderr
    assert (
        f"artifacts/model_training/{TRAINING_RUN_ID}/run_manifest.json"
        in completed.stderr
    )


def test_demo_artifact_gate_succeeds_only_with_both_required_roots(
    tmp_path: Path,
) -> None:
    required_files = (
        tmp_path / "artifacts" / "vertical_slice" / DEMO_RUN_ID / "manifest.json",
        tmp_path
        / "artifacts"
        / "vertical_slice"
        / DEMO_RUN_ID
        / "final_summary.json",
        tmp_path
        / "artifacts"
        / "model_training"
        / TRAINING_RUN_ID
        / "run_manifest.json",
    )
    for path in required_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    completed = subprocess.run(
        ["bash", "-c", _artifact_gate()],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "OK: required demo artifact pair is present" in completed.stdout


def test_demo_artifact_restore_refuses_dangling_symlink_collision(
    tmp_path: Path,
) -> None:
    restore_target = (
        tmp_path / "artifacts" / "vertical_slice" / DEMO_RUN_ID
    )
    restore_target.parent.mkdir(parents=True)
    restore_target.symlink_to(tmp_path / "missing-demo-artifact-target")

    completed = subprocess.run(
        ["bash", "-c", _artifact_restore()],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ERROR: refusing to overwrite existing demo artifact path" in completed.stderr
    assert str(restore_target.relative_to(tmp_path)) in completed.stderr
    assert restore_target.is_symlink()


def test_demo_artifact_checksum_remains_verifiable_after_transfer(
    tmp_path: Path,
) -> None:
    sender_repository = tmp_path / "sender-repository"
    sender_staging = tmp_path / "sender-staging"
    receiver_repository = tmp_path / "receiver-repository"
    receiver_staging = tmp_path / "receiver-staging"
    for directory in (
        sender_repository,
        sender_staging,
        receiver_repository,
        receiver_staging,
    ):
        directory.mkdir()

    source_files = (
        sender_repository
        / "artifacts"
        / "vertical_slice"
        / DEMO_RUN_ID
        / "manifest.json",
        sender_repository
        / "artifacts"
        / "vertical_slice"
        / DEMO_RUN_ID
        / "final_summary.json",
        sender_repository
        / "artifacts"
        / "model_training"
        / TRAINING_RUN_ID
        / "run_manifest.json",
    )
    for path in source_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    export_commands = _artifact_export().replace(
        'PAISA_DEMO_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"',
        f'PAISA_DEMO_TRANSFER_DIR="{sender_staging}"',
    )
    exported = subprocess.run(
        ["bash", "-c", export_commands],
        cwd=sender_repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert exported.returncode == 0, exported.stderr

    archive_name = "PAISA-demo-VSL-20261020-OGDC-main-02.tar.gz"
    for suffix in ("", ".sha256"):
        shutil.copy2(sender_staging / f"{archive_name}{suffix}", receiver_staging)
    shutil.rmtree(sender_staging)

    restore_commands = _artifact_restore().replace(
        'PAISA_DEMO_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"',
        f'PAISA_DEMO_TRANSFER_DIR="{receiver_staging}"',
    )
    restored = subprocess.run(
        ["bash", "-c", restore_commands],
        cwd=receiver_repository,
        text=True,
        capture_output=True,
        check=False,
    )

    assert restored.returncode == 0, restored.stderr
    assert (
        receiver_repository
        / "artifacts"
        / "vertical_slice"
        / DEMO_RUN_ID
        / "manifest.json"
    ).is_file()
    assert (
        receiver_repository
        / "artifacts"
        / "model_training"
        / TRAINING_RUN_ID
        / "run_manifest.json"
    ).is_file()


def test_team_handoff_documents_artifact_restore_and_new_run_regeneration() -> None:
    handoff = _read("docs/TEAM_HANDOFF.md")
    handoff_lower = _normalized(handoff)

    assert "project lead/artifact custodian" in handoff_lower
    assert f"artifacts/vertical_slice/{DEMO_RUN_ID}/" in handoff
    assert f"artifacts/model_training/{TRAINING_RUN_ID}/" in handoff
    assert "PAISA-demo-VSL-20261020-OGDC-main-02.tar.gz" in handoff
    assert "shasum -a 256" in handoff
    assert "tar -xzf" in handoff
    assert "scripts/migrate_legacy_psx_cache.py" in handoff
    assert "scripts/run_ogdc_vertical_slice.py" in handoff
    assert "--run-id \"$PAISA_NEW_VERTICAL_RUN_ID\"" in handoff
    assert "cannot reproduce the exact historical demo" in handoff_lower


def test_readme_indexes_every_demo_pack_document() -> None:
    readme = _read("README.md")

    assert "## Project documentation" in readme
    for relative_path in DEMO_PACK_DOCUMENTS:
        assert f"]({relative_path})" in readme
    assert "canonical current implementation" in readme.lower()
    assert "](PROJECT_CONTEXT.md)" in readme
    assert "historical" in readme.lower()


def test_project_context_prominently_marks_superseded_status() -> None:
    context = _read("PROJECT_CONTEXT.md")
    opening = _normalized("\n".join(context.splitlines()[:15]))

    assert "historical" in opening
    assert "not the current implementation inventory" in opening
    assert "](docs/CURRENT_IMPLEMENTATION_STATUS.md)" in context
    assert "## Historical implementation status: PAISA v0 (superseded)" in context
    assert "## Current implementation status: PAISA v0" not in context


def test_sentiment_status_docs_disclose_zero_price_date_overlap() -> None:
    for relative_path in (
        "docs/FINBERT_SENTIMENT.md",
        "docs/CURRENT_IMPLEMENTATION_STATUS.md",
        "RUN_FROM_GITHUB.md",
    ):
        document = _normalized(_read(relative_path))

        assert "october 19--22, 2026" in document
        assert "september 15, 2026" in document
        assert "zero overlapping dates" in document
        assert "combined" in document
        assert "portfolio conclusion" in document
        assert "overlapping historical news corpus" in document


def test_retained_coverage_packs_are_not_presented_as_current_verifier_passes() -> None:
    for relative_path in (
        "README.md",
        "RUN_FROM_GITHUB.md",
        "docs/CURRENT_IMPLEMENTATION_STATUS.md",
        "docs/HISTORICAL_NEWS_COVERAGE.md",
        "docs/NEWS_PRICE_OVERLAP_CONTRACT.md",
        "docs/NEXT_PHASE_PLAN.md",
        "docs/TEAM_HANDOFF.md",
    ):
        document = _normalized(_read(relative_path))

        assert "news-price-coverage-v2" in document
        assert "news-price-coverage-v3" in document
        assert "producing revision" in document
        assert "schema `1.0`" in document or "schema-`1.0`" in document
        assert (
            "schema `1.1`" in document
            or "schema-`1.1`" in document
            or "coverage artifact schema: `1.1`" in document
        )


def test_historical_test_matrices_name_both_real_skip_reasons() -> None:
    for relative_path in (
        "RUN_FROM_GITHUB.md",
        "docs/CURRENT_IMPLEMENTATION_STATUS.md",
        "docs/PROGRESS_REPORT_OCTOBER_2026.md",
        "docs/TEAM_HANDOFF.md",
    ):
        document = _normalized(_read(relative_path))

        assert "609 passed, 1 skipped" in document
        assert "608 passed, 2 skipped" in document
        assert "optional tensorflow" in document
        assert "absent local-real-run" in document
        assert "historical" in document


def test_fresh_clone_docs_disclose_news_artifact_restore_source() -> None:
    runbook = _normalized(_read("RUN_FROM_GITHUB.md"))
    handoff = _normalized(_read("docs/TEAM_HANDOFF.md"))
    readme = _normalized(_read("README.md"))
    coverage_runbook = _normalized(_read("docs/HISTORICAL_NEWS_COVERAGE.md"))

    assert "artifact-present" in runbook
    assert "artifact-present retained example" in readme
    assert "artifact-present retained example" in coverage_runbook
    assert "checksum-verified custodian export" in coverage_runbook
    assert "the repository does not publish a download url" in runbook
    assert "restore or regenerate ignored news evidence" in handoff
    assert "project lead/artifact custodian" in handoff
    assert "there is no project-managed public download url" in handoff
    assert "paisa-news-evidence-20261023.tar.gz" in handoff
    assert "artifacts/news_ingestion" in handoff
    assert "artifacts/news_coverage" in handoff
    assert "data/news/paisa_news.sqlite3" in handoff
    assert "shasum -a 256" in handoff
    assert "new `coverage-...` id" in handoff


def test_news_evidence_checksum_remains_verifiable_after_transfer(
    tmp_path: Path,
) -> None:
    sender_repository = tmp_path / "news-sender-repository"
    sender_staging = tmp_path / "news-sender-staging"
    receiver_repository = tmp_path / "news-receiver-repository"
    receiver_staging = tmp_path / "news-receiver-staging"
    for directory in (
        sender_repository,
        sender_staging,
        receiver_repository,
        receiver_staging,
    ):
        directory.mkdir()

    source_files = (
        sender_repository
        / "artifacts/news_ingestion/NEWS-20261023-060000-gdelt-ogdc2/completion.json",
        sender_repository
        / "artifacts/news_coverage/COVERAGE-20261023-120000-real0002/completion.json",
        sender_repository
        / "artifacts/news_coverage/COVERAGE-20261023-130000-gdelt0001/completion.json",
        sender_repository / "data/news/paisa_news.sqlite3",
    )
    for path in source_files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")

    export_commands = _news_artifact_commands("export").replace(
        'PAISA_NEWS_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"',
        f'PAISA_NEWS_TRANSFER_DIR="{sender_staging}"',
    )
    exported = subprocess.run(
        ["bash", "-c", export_commands],
        cwd=sender_repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert exported.returncode == 0, exported.stderr

    archive_name = "PAISA-news-evidence-20261023.tar.gz"
    for suffix in ("", ".sha256"):
        shutil.copy2(sender_staging / f"{archive_name}{suffix}", receiver_staging)

    restore_commands = _news_artifact_commands("restore").replace(
        'PAISA_NEWS_TRANSFER_DIR="/absolute/path/to/approved-transfer-staging"',
        f'PAISA_NEWS_TRANSFER_DIR="{receiver_staging}"',
    )
    restored = subprocess.run(
        ["bash", "-c", restore_commands],
        cwd=receiver_repository,
        text=True,
        capture_output=True,
        check=False,
    )
    assert restored.returncode == 0, restored.stderr
    for source_path in source_files:
        relative = source_path.relative_to(sender_repository)
        assert (receiver_repository / relative).is_file()


def test_coverage_docs_distinguish_source_bound_and_pack_only_authority() -> None:
    for relative_path in (
        "README.md",
        "RUN_FROM_GITHUB.md",
        "docs/CURRENT_IMPLEMENTATION_STATUS.md",
        "docs/HISTORICAL_NEWS_COVERAGE.md",
        "docs/NEWS_PRICE_OVERLAP_CONTRACT.md",
        "docs/NEXT_PHASE_PLAN.md",
        "docs/TEAM_HANDOFF.md",
    ):
        document = _normalized(_read(relative_path))

        assert "source_evidence_bound" in document
        assert "pack_semantic" in document
        assert "pack-only" in document
        assert "weaker" in document


def test_coverage_verification_examples_are_valid_shell_workflows() -> None:
    for relative_path in (
        "RUN_FROM_GITHUB.md",
        "docs/HISTORICAL_NEWS_COVERAGE.md",
        "docs/NEWS_PRICE_OVERLAP_CONTRACT.md",
    ):
        verification_blocks = tuple(
            block
            for block in _bash_blocks(relative_path)
            if "verify_news_coverage.py" in block
        )
        assert verification_blocks, f"{relative_path} must document verification"
        for block in verification_blocks:
            parsed = subprocess.run(
                ["bash", "-n"],
                input=block,
                text=True,
                capture_output=True,
                check=False,
            )
            assert parsed.returncode == 0, f"{relative_path}: {parsed.stderr}"

    handoff_blocks = tuple(
        block
        for block in _bash_blocks("docs/TEAM_HANDOFF.md")
        if "audit_news_price_overlap.py" in block
        and "verify_news_coverage.py" in block
    )
    assert len(handoff_blocks) == 1
    handoff = handoff_blocks[0]
    parsed = subprocess.run(
        ["bash", "-n"],
        input=handoff,
        text=True,
        capture_output=True,
        check=False,
    )
    assert parsed.returncode == 0, parsed.stderr
    assert handoff.index("audit_news_price_overlap.py") < handoff.index(
        "verify_news_coverage.py"
    )
    assert '--run-id "$PAISA_COVERAGE_RUN_ID"' in handoff
    assert handoff.count("--allow-legacy-unsealed") >= 2
    assert re.search(
        r'PAISA_COVERAGE_RUN_ID="COVERAGE-\d{8}-\d{6}-[A-Za-z0-9._-]+"',
        handoff,
    )


def test_time_alignment_documents_current_same_url_correction_limit() -> None:
    document = _normalized(_read("docs/SENTIMENT_TIME_ALIGNMENT.md"))

    assert "canonical url has deduplication priority" in document
    assert "same-url corrections are currently unsupported" in document
    assert "ignored as an independent sentiment input" in document
    assert "neither a versioned correction identity nor its own sentiment result" in document
    assert "with their own `available_at`" in document


def test_news_docs_separate_record_schema_from_availability_policy() -> None:
    data_contract = _normalized(_read("docs/NEWS_DATA_CONTRACT.md"))
    time_contract = _normalized(_read("docs/NEWS_TIME_ALIGNMENT.md"))

    for document in (data_contract, time_contract):
        assert "news-availability-v1" in document
        assert "policy `1.1`" not in document

    for relative_path in (
        "README.md",
        "RUN_FROM_GITHUB.md",
        "docs/CURRENT_IMPLEMENTATION_STATUS.md",
        "docs/HISTORICAL_NEWS_COVERAGE.md",
        "docs/NEWS_PRICE_OVERLAP_CONTRACT.md",
        "docs/NEXT_PHASE_PLAN.md",
        "docs/NEWS_INGESTION.md",
        "docs/TEAM_HANDOFF.md",
    ):
        assert "news-availability-v1" in _normalized(_read(relative_path))

    assert "schema `1.0` records must omit this field" in data_contract
    assert "owning ingestion run" in data_contract
    assert "legacy rows migrate with null values" in data_contract
    for artifact_name in (
        "`provider_config.json`",
        "`run_manifest.json`",
        "`completion.json`",
    ):
        assert artifact_name in data_contract
    assert "top-level quality report and final summary" in data_contract
    assert "including an explicit null field" in time_contract
    assert (
        "gdelt_seendate_is_source_observation_not_publication_or_eligibility"
        in time_contract
    )


def test_sentiment_docs_distinguish_retained_schema_from_current_schema() -> None:
    for relative_path in (
        "README.md",
        "docs/NEXT_PHASE_PLAN.md",
        "docs/SENTIMENT_DATA_CONTRACT.md",
    ):
        document = _normalized(_read(relative_path))

        assert "schema `7`" in document or "schema version `7`" in document
        assert "current code targets schema `6`" not in document
        assert "upgraded to `6` by current code" not in document


def test_operator_commands_do_not_overwrite_frontend_environment() -> None:
    for relative_path in (
        "README.md",
        "RUN_FROM_GITHUB.md",
        "docs/SUPERVISOR_DEMO_SCRIPT.md",
        "docs/TEAM_HANDOFF.md",
    ):
        document = _read(relative_path)

        assert "cp .env.example .env.local" in document
        assert (
            "[ ! -e .env.local ]" in document
            or "[ -e .env.local ] || cp .env.example .env.local" in document
        )
        assert "\ncp .env.example .env.local\n" not in document
