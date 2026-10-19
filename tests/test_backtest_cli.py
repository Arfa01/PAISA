from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from test_backtest_runner import _inputs


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_backtest_cli_smoke_emits_json_stdout(tmp_path: Path):
    predictions, prices, config = _inputs(tmp_path / "inputs", experiment_id="cli-ogdc-run")
    runs_root = tmp_path / "cli-runs"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY_ROOT / "scripts" / "run_backtest.py"),
            "--predictions",
            str(predictions),
            "--prices",
            str(prices),
            "--config",
            str(config),
            "--runs-dir",
            str(runs_root),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    output = json.loads(completed.stdout)
    assert output["experiment_id"] == "cli-ogdc-run"
    assert output["artifact_path"] == str(runs_root / "cli-ogdc-run")
    assert output["ending_equity"]
    assert "not financial advice" in output["disclaimer"].lower()
    assert completed.stderr == ""
    assert (runs_root / "cli-ogdc-run" / "result.json").is_file()
