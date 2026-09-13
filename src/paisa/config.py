from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    model_dir: Path


def project_root() -> Path:
    """Return repository root when running from the source tree."""
    return Path(__file__).resolve().parents[2]


def get_paths(data_dir: str | Path | None = None, model_dir: str | Path | None = None) -> ProjectPaths:
    root = project_root()
    data = Path(data_dir or os.getenv("PAISA_DATA_DIR", root / "data")).resolve()
    models = Path(model_dir or os.getenv("PAISA_MODEL_DIR", root / "models")).resolve()
    return ProjectPaths(
        root=root,
        data_dir=data,
        raw_dir=data / "raw" / "psx",
        processed_dir=data / "processed",
        model_dir=models,
    )
