"""Trusted immutable registry for versioned model artifacts.

This registry is intentionally separate from ``portfolio_registry``: model
artifacts own fitted estimators and manifests, while portfolio experiments own
cash, holdings, trades, and snapshots.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from paisa.experiments.artifacts import (
    MANIFEST_FILENAME,
    load_model_artifact,
    write_model_artifact,
)


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class ModelArtifactRegistry:
    """Resolve, register, and checksum-verify artifacts beneath one root."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    @staticmethod
    def _validate_model_id(model_id: str) -> str:
        identifier = str(model_id).strip()
        if not _SAFE_ID.fullmatch(identifier):
            raise ValueError(
                "model_id must start with a letter or digit and contain only letters, "
                "digits, '.', '_', and '-'"
            )
        return identifier

    def model_dir(self, model_id: str) -> Path:
        return self.root / self._validate_model_id(model_id)

    def contains(self, model_id: str) -> bool:
        directory = self.model_dir(model_id)
        return directory.is_dir() and (directory / MANIFEST_FILENAME).is_file()

    def register(
        self,
        model_id: str,
        artifact: Any,
        manifest: Mapping[str, Any],
        *,
        artifact_filename: str = "model.joblib",
        extra_files: Mapping[str, Any] | None = None,
    ) -> Path:
        return write_model_artifact(
            self.root,
            self._validate_model_id(model_id),
            artifact,
            manifest,
            artifact_filename=artifact_filename,
            extra_files=extra_files,
        )

    def load(self, model_id: str) -> tuple[Any, dict[str, Any]]:
        return load_model_artifact(self.model_dir(model_id))

    def manifest(self, model_id: str) -> dict[str, Any]:
        path = self.model_dir(model_id) / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Registered model manifest is missing: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Model manifest must be a JSON object")
        if value.get("model_id") != self._validate_model_id(model_id):
            raise ValueError("Manifest model_id does not match the registry key")
        return value

    def list_model_ids(self) -> tuple[str, ...]:
        if not self.root.exists():
            return ()
        return tuple(
            sorted(
                child.name
                for child in self.root.iterdir()
                if child.is_dir()
                and _SAFE_ID.fullmatch(child.name)
                and (child / MANIFEST_FILENAME).is_file()
            )
        )


# Concise compatibility name for callers that already use ``ModelRegistry``.
ModelRegistry = ModelArtifactRegistry


__all__ = ["ModelArtifactRegistry", "ModelRegistry"]
