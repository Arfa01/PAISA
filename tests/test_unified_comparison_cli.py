from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_CASES = (
    (
        "build",
        REPOSITORY_ROOT / "scripts/build_unified_price_model_comparison.py",
        "build_unified_price_model_comparison",
        [
            "--comparison-id",
            "CMP-cli-test",
            "--output-dir",
            "output",
            "--training-run",
            "training",
            "--vertical-slice",
            "vertical",
            "--lstm-run",
            "lstm",
        ],
    ),
    (
        "verify",
        REPOSITORY_ROOT / "scripts/verify_unified_price_model_comparison.py",
        "verify_unified_comparison_pack",
        ["comparison-pack"],
    ),
)


def _load_script(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"test_unified_cli_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("name", "script", "callable_name", "argv"), SCRIPT_CASES
)
def test_default_failure_is_redacted_and_omits_traceback(
    name: str,
    script: Path,
    callable_name: str,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(name, script)
    capsys.readouterr()
    sensitive = (
        'failed at "/Users/alice/Secret Project/repository/file.py"; '
        "linux=/home/bob/private/file.py; root=/root/repository/file.py; "
        'windows="C:\\Users\\Carol\\Repository Name\\file.py"; '
        "forward=C:/Users/Dana/private/file.py; "
        "unc=\\\\server\\share\\private\\file.csv; "
        "forward_unc=//server/share/Forward Secret/file.csv; "
        "colon_home=path:/Users/Eve/Colon Home/file.py; "
        "tmp=/private/tmp/comparison-secret.csv; framework=/Library/Secret/file.py; "
        "url=https://alice:swordfish@example.test/private; "
        "token_url=https://opaque-token@example.test/private; "
        "password=hunter2; API_TOKEN='tok-123'; private_key=key-private; "
        "AWS_SECRET_ACCESS_KEY=aws-secret; "
        "Authorization: Bearer bearer-secret-value; "
        "standalone=ghp_1234567890abcdef; openai=sk-proj-1234567890abcdef; "
        "jwt=eyJabcdefghijk.eyJabcdefghijk.abcdefghijk"
    )

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(sensitive)

    monkeypatch.setattr(module, callable_name, fail)
    assert module.main(argv) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert set(payload) == {"disclaimer", "error", "error_type", "status"}
    assert payload["status"] == "FAIL"
    assert payload["error_type"] == "RuntimeError"
    serialized = json.dumps(payload)
    for secret in (
        "/Users/alice",
        "/home/bob",
        "/root/repository",
        "C:\\Users\\Carol",
        "C:/Users/Dana",
        "\\\\server\\share",
        "//server/share",
        "/Users/Eve",
        "/private/tmp",
        "/Library/Secret",
        "alice:swordfish",
        "opaque-token",
        "hunter2",
        "tok-123",
        "key-private",
        "aws-secret",
        "bearer-secret-value",
        "ghp_1234567890abcdef",
        "sk-proj-1234567890abcdef",
        "eyJabcdefghijk",
    ):
        assert secret not in serialized
    assert "<redacted-path>" in payload["error"]
    assert "https://<redacted-credentials>@example.test/private" in payload["error"]
    assert "<redacted-credential>" in payload["error"]
    for leaked_suffix in (
        "Secret Project/repository/file.py",
        "Repository Name\\file.py",
        "Forward Secret/file.csv",
    ):
        assert leaked_suffix not in payload["error"]


@pytest.mark.parametrize(
    ("name", "script", "callable_name", "argv"), SCRIPT_CASES
)
def test_debug_failure_includes_only_a_sanitized_traceback(
    name: str,
    script: Path,
    callable_name: str,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(name, script)
    capsys.readouterr()

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise ValueError(
            "/Users/debug-user/private/file.py "
            "C:\\Users\\Debug\\private.txt "
            "https://debug:password@example.test token=debug-token"
        )

    monkeypatch.setattr(module, callable_name, fail)
    assert module.main([*argv, "--debug"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert set(payload) == {
        "disclaimer",
        "error",
        "error_type",
        "status",
        "traceback",
    }
    assert payload["traceback"].startswith("Traceback (most recent call last):")
    assert "<redacted-path>" in payload["traceback"]
    serialized = json.dumps(payload)
    for secret in (
        "/Users/debug-user",
        "C:\\Users\\Debug",
        "debug:password",
        "debug-token",
        str(REPOSITORY_ROOT),
    ):
        assert secret not in serialized


@pytest.mark.parametrize(
    ("name", "script", "callable_name", "argv"), SCRIPT_CASES
)
def test_success_json_remains_relative_and_contains_no_host_path(
    name: str,
    script: Path,
    callable_name: str,
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(name, script)
    capsys.readouterr()
    expected = {
        "status": "PASS",
        "pack": "CMP-cli-test.zip",
        "path_base": "input_parent",
    }

    def succeed(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return expected

    monkeypatch.setattr(module, callable_name, succeed)
    assert module.main(argv) == 0

    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == expected
    assert "/Users/" not in captured.out


@pytest.mark.parametrize(("name", "script", "_callable_name", "argv"), SCRIPT_CASES)
def test_argument_failure_is_sanitized_json_without_raw_argparse_stderr(
    name: str,
    script: Path,
    _callable_name: str,
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(name, script)
    capsys.readouterr()
    sensitive = "path:/Users/argparse-user/Secret Project/private.txt"

    assert module.main([*argv, "--unknown-option", sensitive]) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert set(payload) == {"disclaimer", "error", "error_type", "status"}
    assert payload["status"] == "FAIL"
    assert payload["error_type"] == "ArgumentParsingError"
    assert "--unknown-option" in payload["error"]
    assert sensitive not in captured.err
    assert "/Users/argparse-user" not in captured.err
    assert "Secret Project/private.txt" not in captured.err
    assert "usage:" not in captured.err


@pytest.mark.parametrize(("_name", "script", "_callable_name", "argv"), SCRIPT_CASES)
def test_argument_failure_is_sanitized_end_to_end_in_a_subprocess(
    _name: str,
    script: Path,
    _callable_name: str,
    argv: list[str],
) -> None:
    sensitive = "path:/Users/subprocess-user/Secret Project/private.txt"

    completed = subprocess.run(
        [sys.executable, str(script), *argv, "--unknown-option", sensitive],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    payload = json.loads(completed.stderr)
    assert payload["status"] == "FAIL"
    assert payload["error_type"] == "ArgumentParsingError"
    assert "--unknown-option" in payload["error"]
    assert sensitive not in completed.stderr
    assert "/Users/subprocess-user" not in completed.stderr
    assert "Secret Project/private.txt" not in completed.stderr
    assert "usage:" not in completed.stderr


@pytest.mark.parametrize(("name", "script", "_callable_name", "_argv"), SCRIPT_CASES)
def test_help_preserves_argparse_success_behavior(
    name: str,
    script: Path,
    _callable_name: str,
    _argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script(name, script)
    capsys.readouterr()

    with pytest.raises(SystemExit) as raised:
        module.main(["--help"])

    assert raised.value.code == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.startswith("usage:")
