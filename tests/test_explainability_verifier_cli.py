from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import paisa.explainability.tree_shap as tree_shap


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VERIFY_SCRIPT = REPOSITORY_ROOT / "scripts" / "verify_explainability_run.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_explainability_verifier_cli", VERIFY_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_parses_independent_digest_and_authentication_requirement() -> None:
    module = _load_script()
    digest = "a" * 64

    args = module.parse_args(
        [
            "pack.zip",
            "--checksum",
            "pack.zip.sha256",
            "--expected-manifest-sha256",
            digest,
            "--require-authenticated",
        ]
    )

    assert args.pack == "pack.zip"
    assert args.checksum == "pack.zip.sha256"
    assert args.expected_manifest_sha256 == digest
    assert args.require_authenticated is True


def test_cli_forwards_authentication_controls_and_prints_trust_result(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script()
    digest = "b" * 64
    observed: dict[str, Any] = {}

    def fake_verify(pack: str, **kwargs: Any) -> dict[str, Any]:
        observed["pack"] = pack
        observed.update(kwargs)
        return {
            "status": "PASS",
            "integrity_status": "PASS",
            "authentication_status": "AUTHENTICATED",
            "official_artifact": False,
        }

    monkeypatch.setattr(tree_shap, "verify_explainability_pack", fake_verify)

    assert (
        module.main(
            [
                "pack-directory",
                "--expected-manifest-sha256",
                digest,
                "--require-authenticated",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["authentication_status"] == "AUTHENTICATED"
    assert observed == {
        "pack": "pack-directory",
        "checksum": None,
        "expected_manifest_sha256": digest,
        "require_authenticated": True,
    }
