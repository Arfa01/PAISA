from __future__ import annotations

from collections import Counter
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import yaml


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
CONTINUATION_DOCUMENTS = (
    "docs/PROJECT_STATE_2026-10-24.md",
    "docs/REMAINING_ROADMAP.md",
    "docs/ISSUE_BACKLOG.md",
    "docs/REPRODUCTION_COMMANDS.md",
    "docs/UNIFIED_PRICE_MODEL_COMPARISON.md",
    "docs/FREE_PLAN_WORKFLOW.md",
    "docs/CODEX_PROMPT_LIBRARY.md",
)
CONTINUATION_LINK_DOCUMENTS = ("README.md", *CONTINUATION_DOCUMENTS)
CERTIFIED_LSTM_RUN_ID = "LSTM-20261024-OGDC-price-only-02"
HISTORICAL_LSTM_RUN_ID = "LSTM-20261024-OGDC-price-only-01"
UNIFIED_COMPARISON_ID = "CMP-20261024-OGDC-price-only-02"
HISTORICAL_UNIFIED_COMPARISON_ID = "CMP-20261024-OGDC-price-only-01"
UNIFIED_COMPARISON_ZIP_SHA256 = (
    "8d3bd49f3f018d9134a245ee9c58eebe26a1c9d939851976702c86b66e385fbe"
)
UNIFIED_COMPARISON_FILE_COUNT = 65
ALLOWED_ACTIONS = {
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1": 3,
    "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97": 2,
    "actions/setup-node@949feb2413d6458794dcd2491c4babbbce0c15c1": 1,
}


def _read(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def _normalized(value: str) -> str:
    return re.sub(r"[\s>]+", " ", value).strip().lower()


def _markdown_section(document: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        document,
        flags=re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"missing Markdown section: {heading}"
    return match.group("body")


def _markdown_prose(document: str) -> str:
    """Return Markdown with fenced code blocks and HTML comments removed."""
    prose: list[str] = []
    fence: tuple[str, int] | None = None
    in_comment = False
    for raw_line in document.splitlines():
        if fence is not None:
            closing = re.match(r"^\s*(`{3,}|~{3,})\s*$", raw_line)
            if (
                closing
                and closing.group(1)[0] == fence[0]
                and len(closing.group(1)) >= fence[1]
            ):
                fence = None
            continue

        line = ""
        remainder = raw_line
        while remainder:
            if in_comment:
                comment_end = remainder.find("-->")
                if comment_end < 0:
                    remainder = ""
                    break
                remainder = remainder[comment_end + 3 :]
                in_comment = False
                continue
            comment_start = remainder.find("<!--")
            if comment_start < 0:
                line += remainder
                break
            line += remainder[:comment_start]
            remainder = remainder[comment_start + 4 :]
            in_comment = True

        opening = re.match(r"^\s*(`{3,}|~{3,})", line)
        if opening:
            token = opening.group(1)
            fence = (token[0], len(token))
            continue
        if fence is None:
            prose.append(line)
    return "\n".join(prose)


def _github_heading_anchors(document: str) -> set[str]:
    anchors: set[str] = set()
    occurrences: dict[str, int] = {}
    prose = _markdown_prose(document)
    for match in re.finditer(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose, re.MULTILINE):
        title = re.sub(r"<[^>]+>", "", match.group(1))
        title = re.sub(r"[`*_~]", "", title).lower()
        base = re.sub(r"[^\w\s-]", "", title)
        base = re.sub(r"\s+", "-", base.strip())
        suffix = occurrences.get(base, 0)
        occurrences[base] = suffix + 1
        anchors.add(base if suffix == 0 else f"{base}-{suffix}")
    return anchors


def _local_markdown_targets(relative_path: str) -> tuple[tuple[Path, str], ...]:
    source = REPOSITORY_ROOT / relative_path
    targets: list[tuple[Path, str]] = []
    pattern = re.compile(
        r"(?<!!)\[[^\]]+\]\("
        r"(?P<target><[^>]+>|[^\s)]+)"
        r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\)"
    )
    document = _markdown_prose(source.read_text(encoding="utf-8"))
    for match in pattern.finditer(document):
        destination = match.group("target").strip("<>")
        if destination.startswith(("http://", "https://", "mailto:", "tel:")):
            continue
        path_text, separator, fragment = destination.partition("#")
        if not path_text:
            target = source.resolve()
        else:
            decoded = unquote(path_text)
            assert not Path(decoded).is_absolute(), (
                f"{relative_path} uses an absolute local link: {destination}"
            )
            target = (source.parent / decoded).resolve()
        try:
            target.relative_to(REPOSITORY_ROOT.resolve())
        except ValueError as error:
            raise AssertionError(
                f"{relative_path} link escapes the repository: {destination}"
            ) from error
        targets.append((target, unquote(fragment) if separator else ""))
    return tuple(targets)


def _has_prohibited_git_directive(document: str) -> bool:
    patterns = (
        r"\b(?:work|edit|commit|push)\s+(?:directly\s+)?(?:on|to)\s+`?main`?\b",
        r"\b(?:work|edit|commit|push)\s+`?main`?\s+directly\b",
        r"\bautomatically\s+merge(?:\s+(?:it|the\s+pr|this\s+pr))?\b",
        r"\bmerge(?:\s+(?:it|the\s+pr|this\s+pr))?\s+automatically\b",
        r"\benable\s+auto-merge\b",
        r"\bgit\s+(?:switch|checkout)\b[^\n.;]*\bmain\b",
        r"\bgit\s+push\b[^\n.;]*\bmain\b",
        r"\bgh\s+pr\s+merge\b",
    )
    safe_qualifiers = ("do not", "never", "must not")
    # Prompt-library instructions intentionally live in fenced text blocks, so
    # scan the complete document rather than the prose-only link surface.
    normalized = _normalized(document).replace("*", "")
    clauses = re.split(
        r"(?<=[.!?;])\s+|\s*[;,]\s*(?:but|however)\s+|\s+but\s+|"
        r"\s*,?\s+(?:and\s+)?then\s+|\s+yet\s+",
        normalized,
    )
    for clause in clauses:
        for pattern in patterns:
            for match in re.finditer(pattern, clause):
                prefix = clause[: match.start()]
                if not any(qualifier in prefix for qualifier in safe_qualifiers):
                    return True
    return False


def _has_unsafe_authority_claim(document: str, artifact_id: str) -> bool:
    normalized = _normalized(_markdown_prose(document)).replace("`", "").replace("*", "")
    escaped_id = re.escape(artifact_id.lower())
    patterns = (
        rf"{escaped_id}(?:\s+(?:run|result|artifact))?\s+"
        rf"(?:is|remains|became|becomes)\s+(?:now\s+)?(?:the\s+)?"
        rf"(?:authoritative|certified|current)\b",
        rf"\b(?:authoritative|certified|current)(?:\s+(?:run|result|artifact))?"
        rf"\s*(?::|=|-|\bis\b)\s*{escaped_id}\b",
        rf"{escaped_id}[^.!?]*[.!?]\s*"
        rf"(?:it|this|that)(?:\s+(?:run|result|artifact))?\s+"
        rf"(?:is|remains|became|becomes)\s+(?:now\s+)?(?:the\s+)?"
        rf"(?:authoritative|certified|current)\b",
    )
    if any(re.search(pattern, normalized) for pattern in patterns):
        return True

    # Also reject looser claims later in the same sentence, such as
    # "..., but is also authoritative". A locally negated classification
    # ("not the certified result") remains valid historical wording.
    for artifact_match in re.finditer(escaped_id, normalized):
        sentence_tail = re.split(r"[.!?]", normalized[artifact_match.end() :], maxsplit=1)[0]
        for classification in re.finditer(
            r"\b(?:authoritative|certified|current)\b", sentence_tail
        ):
            prefix = sentence_tail[: classification.start()]
            locally_negated = re.search(
                r"\b(?:not|never)\s+(?:(?:the|a|an)\s+)?$", prefix
            )
            if locally_negated is None:
                return True
    return False


def _contains_secret_interpolation(value: str) -> bool:
    return (
        re.search(
            r"\$\{\{[^}]*\b(?:secrets\s*(?:\.|\[)|"
            r"github\s*(?:\.token\b|\[\s*['\"]token['\"]\s*\]))",
            value,
            re.IGNORECASE,
        )
        is not None
    )


def _has_prohibited_safety_claim(document: str) -> bool:
    normalized = _normalized(document).replace("*", "")
    patterns = (
        r"\b(?:is|are|constitutes?|provides?)\s+"
        r"(?:personalized\s+|professional\s+)?financial advice\b",
        r"\b(?:may|can|should|must)\s+(?:safely\s+)?"
        r"(?:overwrite|reuse)\b[^.;]*\b(?:existing\s+)?(?:artifact|run)\s+ids?\b",
        r"\b(?:existing\s+)?(?:artifact|run)\s+ids?\s+"
        r"(?:may|can|should|must)\s+be\s+(?:overwritten|reused)\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, normalized):
            prefix = normalized[: match.start()]
            if re.search(r"\b(?:not|never)\s+$", prefix):
                continue
            return True
    return False


def _workflow_run_lines(job: dict[str, object]) -> set[str]:
    commands: set[str] = set()
    for step in job.get("steps", []):
        if not isinstance(step, dict) or "run" not in step:
            continue
        commands.update(
            line.strip()
            for line in str(step["run"]).splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return commands


def _nested_yaml_keys(value: object) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_nested_yaml_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_nested_yaml_keys(nested))
    return tuple(keys)


def _nested_yaml_strings(value: object) -> tuple[str, ...]:
    strings: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            strings.append(str(key))
            strings.extend(_nested_yaml_strings(nested))
    elif isinstance(value, list):
        for nested in value:
            strings.extend(_nested_yaml_strings(nested))
    elif value is not None:
        strings.append(str(value))
    return tuple(strings)


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
    restore_commands = restore_commands.replace(
        ".venv/bin/python scripts/restore_artifact_bundle.py",
        f'"{sys.executable}" '
        f'"{REPOSITORY_ROOT / "scripts" / "restore_artifact_bundle.py"}"',
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
    assert "scripts/restore_artifact_bundle.py" in handoff
    assert "scripts/migrate_legacy_psx_cache.py" in handoff
    assert "scripts/run_ogdc_vertical_slice.py" in handoff
    assert "--run-id \"$PAISA_NEW_VERTICAL_RUN_ID\"" in handoff
    assert "cannot reproduce the exact historical demo" in handoff_lower


def test_evidence_bundle_restore_never_extracts_untrusted_tar_into_checkout() -> None:
    handoff = _read("docs/TEAM_HANDOFF.md")
    evidence_restore = handoff.split(
        "Preserve the canonical and retained superseded pairs together.", 1
    )[1].split("### Restore the exact supervisor demo pair", 1)[0]

    assert "scripts/restore_evidence_bundle.py" in evidence_restore
    assert 'tar -xzf "$PAIR_ARCHIVE" -C .' not in evidence_restore
    normalized_restore = _normalized(evidence_restore)
    assert "private staging" in normalized_restore
    assert "without replacement" in normalized_restore
    assert "COPYFILE_DISABLE=1 tar -czf" in evidence_restore


def test_all_handoff_restore_examples_use_private_allowlisted_staging() -> None:
    handoff = _read("docs/TEAM_HANDOFF.md")

    assert 'tar -xzf "$DEMO_ARCHIVE" -C .' not in handoff
    assert 'tar -xzf "$NEWS_ARCHIVE" -C .' not in handoff
    assert handoff.count("scripts/restore_artifact_bundle.py") == 2
    assert "--allow-directory artifacts/vertical_slice/" in handoff
    assert "--allow-directory artifacts/news_ingestion" in handoff
    assert "--allow-file data/news/paisa_news.sqlite3" in handoff


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
    restore_commands = restore_commands.replace(
        ".venv/bin/python scripts/restore_artifact_bundle.py",
        f'"{sys.executable}" '
        f'"{REPOSITORY_ROOT / "scripts" / "restore_artifact_bundle.py"}"',
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


def test_continuation_documents_are_discoverable_and_cover_the_handoff() -> None:
    readme = _read("README.md")
    for relative_path in CONTINUATION_DOCUMENTS:
        assert (REPOSITORY_ROOT / relative_path).is_file()
        assert f"]({relative_path})" in readme

    state = _normalized(_read("docs/PROJECT_STATE_2026-10-24.md"))
    assert "d7015cb" in state
    assert "pr #16" in state
    assert "lstm-20261024-ogdc-price-only-02" in state
    assert "vsl-20261020-ogdc-main-02" in state
    assert "exact next safe task" in state
    assert "not financial advice" in state

    roadmap_and_backlog = _normalized(
        _read("docs/REMAINING_ROADMAP.md") + _read("docs/ISSUE_BACKLOG.md")
    )
    for topic in (
        "lstm integration",
        "unified model comparison",
        "shap",
        "multi-stock",
        "event graph",
        "historical-news overlap",
        "daily sentiment aggregation",
        "price-plus-sentiment",
        "final report",
        "final demo",
    ):
        assert topic in roadmap_and_backlog
    for priority in ("must", "should", "stretch"):
        assert priority in roadmap_and_backlog
    assert "dependencies" in roadmap_and_backlog
    assert "exit criteria" in roadmap_and_backlog

    backlog = _read("docs/ISSUE_BACKLOG.md")
    permission_run = backlog.split("## PAISA-RM-007", 1)[1].split(
        "## PAISA-RM-008", 1
    )[0]
    registry = backlog.split("## PAISA-RM-018", 1)[1].split(
        "## PAISA-RM-019", 1
    )[0]
    final_report = backlog.split("## PAISA-RM-021", 1)[1].split(
        "## PAISA-RM-022", 1
    )[0]
    assert "**Depends on:** `PAISA-RM-006`" in permission_run
    assert "None for the fixture-backed registry contract" in registry
    assert "`PAISA-RM-013` only when `G3-NEWS-OVERLAP` opens" in final_report


def test_continuation_markdown_links_and_fragments_resolve() -> None:
    checked_targets = 0
    checked_fragments = 0
    for relative_path in CONTINUATION_LINK_DOCUMENTS:
        targets = _local_markdown_targets(relative_path)
        for target, fragment in targets:
            checked_targets += 1
            assert target.exists(), f"{relative_path} links to missing {target}"
            if fragment:
                checked_fragments += 1
                assert target.is_file(), (
                    f"{relative_path} fragment target is not a file: {target}"
                )
                anchors = _github_heading_anchors(target.read_text(encoding="utf-8"))
                assert fragment.lower() in anchors, (
                    f"{relative_path} links to missing fragment #{fragment} in {target}"
                )

    assert checked_targets >= 10
    assert checked_fragments >= 1
    assert _github_heading_anchors(
        "```bash\n# fake anchor\n```\n<!-- ## hidden anchor -->\n## Real anchor\n"
    ) == {"real-anchor"}
    assert _github_heading_anchors(
        "````text\n# fake anchor\n```\n## still fake\n````\n## Real anchor\n"
    ) == {"real-anchor"}
    assert _github_heading_anchors(
        "~~~~text\n# fake anchor\n~~~\n## still fake\n~~~~\n## Real anchor\n"
    ) == {"real-anchor"}
    assert _github_heading_anchors(
        "```html\n<!--\n```\n## Live heading\n-->\n"
    ) == {"live-heading"}


def test_project_state_preserves_canonical_research_boundaries() -> None:
    raw_state = _read("docs/PROJECT_STATE_2026-10-24.md")
    state = _normalized(raw_state)
    merged = _normalized(_markdown_section(raw_state, "Merged milestone pull requests"))
    active = _normalized(_markdown_section(raw_state, "Active and open work"))

    assert CERTIFIED_LSTM_RUN_ID.lower() in state
    assert "certified `pass` result" in state
    assert state.count(HISTORICAL_LSTM_RUN_ID.lower()) == 1
    assert "immutable pre-hardening history, not the certified result" in state
    assert not _has_unsafe_authority_claim(raw_state, HISTORICAL_LSTM_RUN_ID)
    assert _has_unsafe_authority_claim(
        f"`{HISTORICAL_LSTM_RUN_ID}` run is authoritative.",
        HISTORICAL_LSTM_RUN_ID,
    )
    assert _has_unsafe_authority_claim(
        f"Authoritative result: `{HISTORICAL_LSTM_RUN_ID}`.",
        HISTORICAL_LSTM_RUN_ID,
    )
    assert _has_unsafe_authority_claim(
        f"The authoritative result is `{HISTORICAL_LSTM_RUN_ID}`.",
        HISTORICAL_LSTM_RUN_ID,
    )
    assert _has_unsafe_authority_claim(
        f"`{HISTORICAL_LSTM_RUN_ID}` remains immutable pre-hardening history, "
        "not the certified result, but is also authoritative.",
        HISTORICAL_LSTM_RUN_ID,
    )
    assert _has_unsafe_authority_claim(
        f"`{HISTORICAL_LSTM_RUN_ID}` is immutable history, "
        "not certified and authoritative.",
        HISTORICAL_LSTM_RUN_ID,
    )
    assert _has_unsafe_authority_claim(
        f"`{HISTORICAL_LSTM_RUN_ID}` is historical. It is now authoritative.",
        HISTORICAL_LSTM_RUN_ID,
    )

    assert "common 184-date test intersection" in state
    assert "shared project holdout" in state
    assert "globally untouched" not in state
    assert UNIFIED_COMPARISON_ID.lower() in state
    assert HISTORICAL_UNIFIED_COMPARISON_ID.lower() in state
    assert not _has_unsafe_authority_claim(
        raw_state, HISTORICAL_UNIFIED_COMPARISON_ID
    )
    assert UNIFIED_COMPARISON_ZIP_SHA256 in raw_state
    assert f"`{UNIFIED_COMPARISON_FILE_COUNT}`" in raw_state

    raw_unified = _read("docs/UNIFIED_PRICE_MODEL_COMPARISON.md")
    unified = _normalized(raw_unified)
    assert UNIFIED_COMPARISON_ID.lower() in unified
    assert HISTORICAL_UNIFIED_COMPARISON_ID.lower() in unified
    assert not _has_unsafe_authority_claim(
        raw_unified, HISTORICAL_UNIFIED_COMPARISON_ID
    )
    assert UNIFIED_COMPARISON_ZIP_SHA256 in raw_unified
    assert f"total file count is `{UNIFIED_COMPARISON_FILE_COUNT}`" in raw_unified
    assert "shared project holdout" in unified
    assert "globally untouched" not in unified
    assert "universal winner" not in unified
    assert "does not integrate or evaluate sentiment features" in unified
    assert "tabular/report-only" in unified
    assert "exactly two logical rows" in unified
    assert "six `ledger_provenance` records" in unified

    custody_documents = (
        "docs/PROJECT_STATE_2026-10-24.md",
        "docs/CURRENT_IMPLEMENTATION_STATUS.md",
        "docs/REMAINING_ROADMAP.md",
        "docs/REPRODUCTION_COMMANDS.md",
        "docs/RESULTS_INTERPRETATION.md",
        "docs/UNIFIED_PRICE_MODEL_COMPARISON.md",
    )
    for relative_path in custody_documents:
        document = _read(relative_path)
        assert "<TO_BE_REGENERATED>" not in document
        assert UNIFIED_COMPARISON_ZIP_SHA256 in document
        assert f"`{UNIFIED_COMPARISON_FILE_COUNT}`" in document

    assert (
        "| ogdc vertical slice | `vsl-20261020-ogdc-main-02` | completed `pass`"
        in state
    )
    assert "vsl-20261020-ogdc-main-01" not in state

    assert "| #15 | `cf897e1` | contemporaneous historical evidence |" in merged
    assert (
        "| #16 | `d7015cb` | price-only lstm experiments and certified evidence |"
        in merged
    )
    assert "pr #15" not in active
    assert "pr #16 is merged" in active
    assert "pr #16 is open" not in active
    assert "pr #16" not in active.replace("pr #16 is merged", "")

    assert "no model portfolio beats buy-and-hold" in state
    assert "neither beats `m-002` or buy-and-hold" in state
    for contradiction in (
        "m-002 beats buy-and-hold",
        "m-005 beats buy-and-hold",
        "m-006 beats buy-and-hold",
        "model portfolios outperform buy-and-hold",
    ):
        assert contradiction not in state

    assert "every canonical evidence-coverage profile reports zero eligible sessions" in state
    assert "article sentiment therefore cannot yet be joined" in state
    assert "daily aggregation and price-plus-sentiment work must stay blocked" in state
    assert "daily sentiment aggregation is unblocked" not in state
    assert "price-plus-sentiment work is ready" not in state
    assert "historical coverage-v2 packs passed the verifier at their producing revision" in state
    assert "current schema-`1.1` / coverage-v3 code intentionally does not certify them" in state
    assert "coverage-v2 packs are current-verifier passes" not in state
    for historical_coverage_id in (
        "COVERAGE-20261023-120000-real0002",
        "COVERAGE-20261023-130000-gdelt0001",
    ):
        assert not _has_unsafe_authority_claim(raw_state, historical_coverage_id)
        expected_historical_row = (
            f"`{historical_coverage_id}` | immutable coverage-v2 `no_overlap`; "
            "passed only at its producing revision."
        ).lower()
        assert expected_historical_row in raw_state.lower()

    assert "a read-only `/api/v1` validates immutable vertical-slice artifacts" in state
    assert "the react/typescript/plotly dashboard presents history" in state
    assert "the certified lstm run remains separate from the" in state
    assert "there is no fixture-backed read-only lstm" in state
    assert "api currently serves the certified lstm run" not in state
    assert "dashboard currently presents the unified six-model" not in state


def test_continuation_docs_preserve_immutability_and_safe_git_instructions() -> None:
    for relative_path in CONTINUATION_DOCUMENTS:
        document = _read(relative_path)
        normalized = _normalized(document)
        assert "not financial advice" in normalized, relative_path
        assert "immutable" in normalized, relative_path
        assert not _has_prohibited_git_directive(document), relative_path
        assert not _has_prohibited_safety_claim(document), relative_path

    prompts = _normalized(_read("docs/CODEX_PROMPT_LIBRARY.md"))
    limited_workflow = _normalized(_read("docs/FREE_PLAN_WORKFLOW.md"))
    reproduction = _normalized(_read("docs/REPRODUCTION_COMMANDS.md"))
    assert "never edit or commit on main" in prompts
    assert "do not merge automatically" in prompts
    assert "never work directly on `main`" in limited_workflow
    assert "merge manually after checks and review" in limited_workflow
    assert "a collision must fail; do not delete, overwrite, reseal, or relabel" in reproduction

    assert _has_prohibited_git_directive("Work directly on main.")
    assert _has_prohibited_git_directive("Merge automatically.")
    assert _has_prohibited_git_directive(
        "Never work directly on main. Work directly on main."
    )
    assert _has_prohibited_git_directive(
        "Never work directly on main, but work directly on main."
    )
    assert _has_prohibited_git_directive("Edit main directly.")
    assert _has_prohibited_git_directive("Automatically merge the PR.")
    assert _has_prohibited_git_directive("git switch main")
    assert _has_prohibited_git_directive("git push origin main")
    assert _has_prohibited_git_directive("gh pr merge --auto")
    assert _has_prohibited_git_directive(
        "```text\nEdit main directly.\ngh pr merge --auto\n```"
    )
    assert _has_prohibited_git_directive(
        "Never work directly on main and then git switch main."
    )
    assert _has_prohibited_git_directive(
        "Do not merge automatically and then gh pr merge --auto."
    )
    assert _has_prohibited_git_directive(
        "Complete the manual checklist and then gh pr merge --auto."
    )
    assert not _has_prohibited_git_directive(
        "Never edit or commit on main. Do not merge automatically."
    )
    assert _has_prohibited_git_directive(
        "Do not merge automatically. Merge automatically."
    )
    assert _has_prohibited_safety_claim("PAISA results are financial advice.")
    assert _has_prohibited_safety_claim(
        "Existing artifact IDs may be overwritten."
    )
    assert _has_prohibited_safety_claim("You can reuse existing run IDs.")
    assert not _has_prohibited_safety_claim(
        "PAISA results are not financial advice. Do not provide financial advice. "
        "Never overwrite or reuse existing artifact IDs."
    )


def test_prompt_library_and_limited_usage_workflow_cover_every_stage() -> None:
    prompts = _normalized(_read("docs/CODEX_PROMPT_LIBRARY.md"))
    for heading in (
        "implementation",
        "read-only review",
        "fix review findings",
        "post-merge acceptance",
        "error triage",
        "release and tagging",
        "report generation",
    ):
        assert heading in prompts

    workflow = _normalized(_read("docs/FREE_PLAN_WORKFLOW.md"))
    for concept in (
        "small number",
        "one issue, one branch",
        "local validation",
        "github actions",
        "colab",
        "artifact custodian",
        "github issues",
    ):
        assert concept in workflow


def test_reproduction_runbook_covers_required_commands_and_parses_as_shell() -> None:
    document = _read("docs/REPRODUCTION_COMMANDS.md")
    normalized = _normalized(document)
    for command in (
        "python scripts/doctor.py",
        "python -m pytest",
        "python -m pip check",
        "python -m compileall",
        "npm ci",
        "npm run test",
        "npm run lint",
        "npm run build",
        "uvicorn paisa.api.main:app",
        "npm run dev",
        "run_ogdc_vertical_slice.py",
        "run_news_pipeline.py",
        "run_sentiment_inference.py",
        "audit_news_price_overlap.py",
        "verify_news_coverage.py",
        "run_lstm_price_only_experiments.py",
    ):
        assert command in normalized
    assert "new immutable" in normalized
    assert "not financial advice" in normalized

    blocks = _bash_blocks("docs/REPRODUCTION_COMMANDS.md")
    assert blocks
    for block in blocks:
        parsed = subprocess.run(
            ["bash", "-n"],
            input=block,
            text=True,
            capture_output=True,
            check=False,
        )
        assert parsed.returncode == 0, parsed.stderr


def test_ci_is_pinned_cached_offline_and_fixture_only() -> None:
    workflow_text = _read(".github/workflows/ci.yml")
    workflow = yaml.load(workflow_text, Loader=yaml.BaseLoader)
    assert isinstance(workflow, dict)

    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    assert set(triggers) == {"pull_request", "push", "workflow_dispatch"}
    assert triggers["pull_request"] == ""
    assert triggers["workflow_dispatch"] == ""
    assert triggers["push"] == {"branches": ["main"]}
    assert "pull_request_target" not in triggers

    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)
    assert set(jobs) == {"python", "documentation-contracts", "frontend"}
    assert all("permissions" not in job for job in jobs.values())

    action_references = Counter(
        step["uses"]
        for job in jobs.values()
        for step in job["steps"]
        if "uses" in step
    )
    assert action_references == Counter(ALLOWED_ACTIONS)

    checkout_reference = next(
        action for action in ALLOWED_ACTIONS if action.startswith("actions/checkout@")
    )
    for job_name, job in jobs.items():
        checkout_steps = [
            step
            for step in job["steps"]
            if step.get("uses") == checkout_reference
        ]
        assert len(checkout_steps) == 1, f"{job_name} must have exactly one checkout"
        assert checkout_steps[0].get("with") == {"persist-credentials": "false"}

    python_job = jobs["python"]
    docs_job = jobs["documentation-contracts"]
    frontend_job = jobs["frontend"]
    assert python_job["env"]["HF_HUB_OFFLINE"] == "1"
    assert python_job["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert python_job["env"]["MPLCONFIGDIR"] == "/tmp/paisa-matplotlib"
    assert docs_job["env"]["HF_HUB_OFFLINE"] == "1"
    assert docs_job["env"]["TRANSFORMERS_OFFLINE"] == "1"

    python_setup = next(
        step
        for step in python_job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    docs_setup = next(
        step
        for step in docs_job["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    node_setup = next(
        step
        for step in frontend_job["steps"]
        if step.get("uses", "").startswith("actions/setup-node@")
    )
    assert python_setup["with"]["cache"] == "pip"
    assert docs_setup["with"]["cache"] == "pip"
    assert node_setup["with"]["cache"] == "npm"

    required_commands = {
        "python": {
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            "python -m pip install --no-deps -e .",
            "python -m pip check",
            "python -m compileall -q src scripts tests",
            "python -m pytest --ignore=tests/test_documentation_contracts.py",
            "git diff --check",
            "git diff --exit-code",
        },
        "documentation-contracts": {
            "python -m pip install -r requirements-dev.txt",
            "python -m pytest tests/test_documentation_contracts.py",
            "git diff --check",
            "git diff --exit-code",
        },
        "frontend": {
            "npm ci",
            "npm run test",
            "npm run lint",
            "npm run build",
            "git diff --check",
            "git diff --exit-code",
        },
    }
    all_commands: set[str] = set()
    for job_name, expected in required_commands.items():
        actual = _workflow_run_lines(jobs[job_name])
        assert actual == expected, (
            f"{job_name} command drift: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
        all_commands.update(actual)

    nested_keys = _nested_yaml_keys(workflow)
    nested_strings = _nested_yaml_strings(workflow)
    assert "continue-on-error" not in nested_keys
    assert "if" not in nested_keys
    assert "shell" not in nested_keys
    assert "pull_request_target" not in nested_keys
    assert not any(_contains_secret_interpolation(value) for value in nested_strings)
    assert _contains_secret_interpolation("${{secrets.PRIVATE_TOKEN}}")
    assert _contains_secret_interpolation("${{ secrets.PRIVATE_TOKEN }}")
    assert _contains_secret_interpolation("${{ secrets['PRIVATE_TOKEN'] }}")
    assert _contains_secret_interpolation("${{ github.token }}")
    assert _contains_secret_interpolation("${{ github['token'] }}")
    assert not any(
        value.lower() in {"write", "write-all"} for value in nested_strings
    )
    assert "${{ runner.temp }}" not in nested_strings

    for forbidden in (
        "requirements-lstm.txt",
        "requirements-sentiment.txt",
        "requirements-provider-pypsx.txt",
        "run_news_pipeline.py",
        "run_sentiment_inference.py",
        "run_ogdc_vertical_slice.py",
        "run_lstm_price_only_experiments.py",
    ):
        assert not any(forbidden in command for command in all_commands)
