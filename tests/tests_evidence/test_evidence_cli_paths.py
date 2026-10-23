from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = (
    REPOSITORY_ROOT / "scripts" / "run_gdelt_gkg_evidence.py",
    REPOSITORY_ROOT / "scripts" / "import_psx_announcements.py",
)


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
