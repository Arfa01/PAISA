from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def make_run_id(repo_root: Path, prefix: str = "TRN") -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    sha = get_git_commit_sha(repo_root, short=True) or "nogit"
    return f"{prefix}-{timestamp}-{sha}"


def get_git_commit_sha(repo_root: Path, short: bool = False) -> str | None:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        result = subprocess.run(args, cwd=repo_root, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def detect_runtime_environment() -> dict[str, Any]:
    env = "linux"
    if "COLAB_RELEASE_TAG" in os.environ or "google.colab" in sys.modules:
        env = "colab"
    elif "KAGGLE_URL_BASE" in os.environ:
        env = "kaggle"

    cpu_info = platform.processor() or platform.machine()
    ram_gb = None
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = round(pages * page_size / (1024**3), 2)
    except Exception:
        ram_gb = None

    gpu_available = False
    gpu_note = "No GPU detected through torch/cuda; RandomForestClassifier uses CPU for this run."
    try:
        import torch  # type: ignore

        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            gpu_note = f"GPU detected by torch: {torch.cuda.get_device_name(0)}. RandomForestClassifier still uses CPU in this run."
    except Exception:
        pass

    return {
        "runtime_environment": env,
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_info,
        "ram_gb": ram_gb,
        "gpu_available": gpu_available,
        "gpu_note": gpu_note,
        "random_forest_compute_note": "scikit-learn RandomForestClassifier uses CPU in this run.",
    }


def package_versions() -> dict[str, str | None]:
    packages = ["numpy", "pandas", "sklearn", "joblib", "matplotlib"]
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            module = __import__(package)
            versions[package] = getattr(module, "__version__", None)
        except Exception:
            versions[package] = None
    return versions


def make_zip_from_directory(source_dir: Path, zip_path: Path) -> Path:
    ensure_dir(zip_path.parent)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in sorted(source_dir.rglob("*")):
            if item.is_file():
                archive.write(item, arcname=item.relative_to(source_dir.parent))
    return zip_path


def file_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 3)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Expected file does not exist: {src}")
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def split_rows_by_dates(frame: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    date_set = set(dates)
    data = frame.copy()
    data["_date_str"] = pd.to_datetime(data["date"]).dt.strftime("%Y-%m-%d")
    result = data[data["_date_str"].isin(date_set)].drop(columns=["_date_str"]).copy()
    return result.sort_values("date").reset_index(drop=True)
