from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from paisa.news.models import PipelineSummary


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "run_news_pipeline.py"


def _load_cli_module():
    spec = importlib.util.spec_from_file_location("paisa_news_cli_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(*, status: str = "PASS") -> PipelineSummary:
    return PipelineSummary(
        schema_version="1.0",
        run_id="NEWS-20261022-093000-cli",
        run_status=status,
        quality_status="PASS" if status != "FAIL" else "FAIL",
        provider="manual-csv",
        provider_mode="imported",
        source_count=1,
        raw_record_count=1,
        normalized_record_count=1,
        duplicate_count=0,
        ogdc_linked_article_count=1,
        missing_publication_time_count=0,
        date_range={"start": "2026-10-21T05:00:00Z", "end": "2026-10-21T05:00:00Z"},
        artifact_directory="/tmp/news/NEWS-20261022-093000-cli",
        database_path="/tmp/news.sqlite3",
    )


@pytest.mark.parametrize(
    "argv,expected_message",
    [
        (["--provider", "rss"], "requires --config"),
        (["--provider", "manual-csv"], "requires --input"),
        (
            ["--provider", "rss", "--config", "feeds.json", "--input", "news.csv"],
            "--input is only valid",
        ),
        (
            ["--provider", "fixture", "--input", "news.csv"],
            "invalid choice",
        ),
    ],
)
def test_cli_validates_provider_specific_arguments(argv, expected_message, capsys):
    module = _load_cli_module()

    with pytest.raises(SystemExit) as raised:
        module.parse_args(argv)

    assert raised.value.code == 2
    assert expected_message in capsys.readouterr().err


def test_cli_prints_exactly_one_body_free_summary_and_rejects_partial_by_default(
    monkeypatch, capsys
):
    module = _load_cli_module()
    monkeypatch.setattr(module, "_build_provider", lambda args: object())
    monkeypatch.setattr(
        module,
        "run_news_pipeline",
        lambda *args, **kwargs: _summary(status="PARTIAL"),
    )

    exit_code = module.main(["--provider", "manual-csv", "--input", "news.csv"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["run_id"] == "NEWS-20261022-093000-cli"
    assert payload["artifact_directory"].endswith("NEWS-20261022-093000-cli")
    assert payload["database_path"].endswith("news.sqlite3")
    assert "body" not in captured.out.lower()
    assert captured.err == ""


def test_cli_allows_partial_only_with_explicit_opt_in(monkeypatch, capsys):
    module = _load_cli_module()
    monkeypatch.setattr(module, "_build_provider", lambda args: object())
    monkeypatch.setattr(
        module,
        "run_news_pipeline",
        lambda *args, **kwargs: _summary(status="PARTIAL"),
    )

    exit_code = module.main(
        [
            "--provider",
            "manual-csv",
            "--input",
            "news.csv",
            "--allow-partial",
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["run_status"] == "PARTIAL"


def test_cli_returns_nonzero_for_failed_summary(monkeypatch, capsys):
    module = _load_cli_module()
    monkeypatch.setattr(module, "_build_provider", lambda args: object())
    monkeypatch.setattr(
        module,
        "run_news_pipeline",
        lambda *args, **kwargs: _summary(status="FAIL"),
    )

    exit_code = module.main(["--provider", "manual-csv", "--input", "news.csv"])

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["run_status"] == "FAIL"


def test_cli_exception_output_does_not_echo_credentials(monkeypatch, capsys):
    module = _load_cli_module()
    monkeypatch.setattr(module, "_build_provider", lambda args: object())

    def fail(*args, **kwargs):
        raise RuntimeError("request failed with api_key=DO-NOT-PRINT")

    monkeypatch.setattr(module, "run_news_pipeline", fail)

    exit_code = module.main(["--provider", "manual-csv", "--input", "news.csv"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "FAIL"
    assert payload["run_id"].startswith("NEWS-")
    assert payload["error_type"] == "RuntimeError"
    assert "DO-NOT-PRINT" not in captured.out
    assert captured.err == ""


def test_manual_csv_cli_end_to_end_emits_summary_and_artifact_locations(tmp_path):
    imported = tmp_path / "permitted-news.csv"
    imported.write_text(
        "article_id,title,summary,body,source_url,published_at,source_name,language\n"
        "real-1,OGDC announces discovery,Summary,Permitted dataset body,"
        "https://example.test/business/real-1,2026-10-21T10:00:00+05:00,"
        "Example Publisher,en\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "artifacts"
    database = tmp_path / "news.sqlite3"
    run_id = "NEWS-20261022-093000-cli-e2e"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--provider",
            "manual-csv",
            "--input",
            str(imported),
            "--output-dir",
            str(output_root),
            "--database",
            str(database),
            "--storage-root",
            str(tmp_path),
            "--run-id",
            run_id,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["run_id"] == run_id
    assert payload["provider"] == "manual-csv"
    assert payload["provider_mode"] == "imported"
    assert payload["raw_record_count"] == 1
    assert payload["normalized_record_count"] == 1
    assert payload["ogdc_linked_article_count"] == 1
    assert payload["artifact_directory"] == f"artifacts/{run_id}"
    assert payload["database_path"] == "news.sqlite3"
    assert "Permitted dataset body" not in completed.stdout
    assert completed.stderr == ""
    assert (output_root / run_id / "run_manifest.json").is_file()
    assert (output_root / run_id / "final_summary.json").is_file()
