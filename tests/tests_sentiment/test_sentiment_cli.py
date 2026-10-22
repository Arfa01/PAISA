from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_sentiment_inference.py"


def _load_cli():
    spec = importlib.util.spec_from_file_location("paisa_sentiment_cli", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arguments(*extra: str) -> list[str]:
    return [
        "--source-run-id",
        "NEWS-20261022-131242-5ddffd2d",
        "--run-id",
        "SENT-20261022-170700-test0001",
        *extra,
    ]


def test_cli_failure_is_nonzero_and_redacts_backend_details(monkeypatch, capsys):
    cli = _load_cli()

    def fail(**_kwargs):
        raise RuntimeError("DO-NOT-LOG-ARTICLE-BODY /private/cache HF_TOKEN=secret")

    monkeypatch.setattr(cli, "run_sentiment_inference", fail)

    assert cli.main(_arguments()) == 1
    output = capsys.readouterr().out
    assert '"status": "FAIL"' in output
    assert "DO-NOT-LOG-ARTICLE-BODY" not in output
    assert "/private/cache" not in output
    assert "HF_TOKEN" not in output


@pytest.mark.parametrize(
    ("status", "extra", "expected"),
    [
        ("PASS", (), 0),
        ("PARTIAL", (), 1),
        ("PARTIAL", ("--allow-partial",), 0),
        ("FAIL", (), 1),
    ],
)
def test_cli_exit_status_policy(monkeypatch, capsys, status, extra, expected):
    cli = _load_cli()
    summary = SimpleNamespace(
        status=status,
        to_dict=lambda: {
            "schema_version": "1.0",
            "status": status,
            "academic_use_only": True,
            "financial_advice": False,
        },
    )
    monkeypatch.setattr(cli, "run_sentiment_inference", lambda **_kwargs: summary)

    assert cli.main(_arguments(*extra)) == expected
    assert f'"status": "{status}"' in capsys.readouterr().out


def test_cli_rejects_token_lengths_above_model_limit():
    cli = _load_cli()

    with pytest.raises(SystemExit) as error:
        cli.parse_args(_arguments("--max-token-length", "513"))

    assert error.value.code == 2


def test_cli_requires_exact_run_id_for_stale_discard():
    cli = _load_cli()

    with pytest.raises(SystemExit) as error:
        cli.parse_args(
            [
                "--source-run-id",
                "NEWS-20261022-131242-5ddffd2d",
                "--discard-stale-running",
            ]
        )

    assert error.value.code == 2


def test_cli_passes_explicit_stale_discard_to_pipeline(monkeypatch, capsys):
    cli = _load_cli()
    observed = {}
    summary = SimpleNamespace(
        status="PASS",
        to_dict=lambda: {"schema_version": "1.0", "status": "PASS"},
    )

    def run(**kwargs):
        observed.update(kwargs)
        return summary

    monkeypatch.setattr(cli, "run_sentiment_inference", run)
    assert cli.main(_arguments("--discard-stale-running")) == 0
    assert observed["discard_stale_running"] is True
    capsys.readouterr()
