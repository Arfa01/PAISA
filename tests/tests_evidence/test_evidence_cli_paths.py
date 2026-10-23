from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    REPOSITORY_ROOT / "scripts" / "run_gdelt_gkg_evidence.py",
    REPOSITORY_ROOT / "scripts" / "import_psx_announcements.py",
)
GKG_SCRIPT = REPOSITORY_ROOT / "scripts" / "run_gdelt_gkg_evidence.py"
VERIFY_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_evidence_run.py"


def _load_script(path: Path):
    spec = importlib.util.spec_from_file_location(
        f"test_{path.stem}",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", SCRIPTS)
def test_evidence_cli_artifact_labels_never_expose_absolute_paths(
    script: Path,
    tmp_path: Path,
) -> None:
    module = _load_script(script)
    run_id = "EVID-20261023-235959-deadbeef"

    internal = module._artifact_directory_label(
        REPOSITORY_ROOT / "artifacts" / "contemporaneous_evidence" / run_id
    )
    external = module._artifact_directory_label(tmp_path / "private" / run_id)

    assert internal == f"artifacts/contemporaneous_evidence/{run_id}"
    assert external == f"external-evidence/{run_id}"
    assert not Path(internal).is_absolute()
    assert not Path(external).is_absolute()
    assert str(tmp_path) not in external


def test_gkg_checkpoint_cli_rejects_repository_hygiene_violations(
    tmp_path: Path,
) -> None:
    module = _load_script(GKG_SCRIPT)
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet"], cwd=repository, check=True, timeout=10
    )

    with pytest.raises(ValueError, match="Git-ignored"):
        module._prepare_checkpoint_directory(
            repository / "unignored",
            repository_root=repository,
        )
    assert not (repository / "unignored").exists()

    (repository / ".gitignore").write_text(
        "/safe-checkpoint/\n/tracked-checkpoint/\n",
        encoding="utf-8",
    )
    safe = module._prepare_checkpoint_directory(
        repository / "safe-checkpoint",
        repository_root=repository,
    )
    assert safe == (repository / "safe-checkpoint").resolve()

    tracked = repository / "tracked-checkpoint"
    tracked.mkdir()
    (tracked / "checkpoint.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", "tracked-checkpoint/checkpoint.json"],
        cwd=repository,
        check=True,
        timeout=10,
    )
    with pytest.raises(ValueError, match="Git-tracked"):
        module._prepare_checkpoint_directory(
            tracked,
            repository_root=repository,
        )


def test_evidence_verification_cli_does_not_leak_invalid_host_path(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "private" / "secret-evidence" / "EVID-SECRET"
    completed = subprocess.run(
        [sys.executable, str(VERIFY_SCRIPT), "--run-dir", str(missing)],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["status"] == "FAIL"
    assert payload["verification"] == "FAIL"
    assert payload["error_type"] == "EvidenceVerificationError"
    assert payload["academic_disclaimer"] == (
        "Academic simulation; not financial advice."
    )
    combined = completed.stdout + completed.stderr
    assert str(tmp_path) not in combined
    assert str(REPOSITORY_ROOT) not in combined
    assert "Traceback" not in combined
