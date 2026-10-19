"""Trusted immutable registry for versioned model artifacts.

This registry is intentionally separate from ``portfolio_registry``: model
artifacts own fitted estimators and manifests, while portfolio experiments own
cash, holdings, trades, and snapshots.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from paisa.experiments.artifacts import (
    MANIFEST_FILENAME,
    load_model_artifact,
    read_model_manifest,
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
        directory = self.root / self._validate_model_id(model_id)
        if directory.is_symlink():
            raise ValueError(
                f"Registered model directory must not be a symlink: {directory}"
            )
        resolved = directory.resolve(strict=False)
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(
                "Registered model directory resolves outside the trusted registry root"
            ) from exc
        return directory

    def contains(self, model_id: str) -> bool:
        directory = self.model_dir(model_id)
        manifest = directory / MANIFEST_FILENAME
        return (
            directory.is_dir()
            and not manifest.is_symlink()
            and manifest.is_file()
        )

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
        return load_model_artifact(
            self.model_dir(model_id),
            trusted_root=self.root,
        )

    def manifest(self, model_id: str) -> dict[str, Any]:
        value = read_model_manifest(
            self.model_dir(model_id),
            trusted_root=self.root,
        )
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
                if not child.is_symlink()
                and child.is_dir()
                and _SAFE_ID.fullmatch(child.name)
                and not (child / MANIFEST_FILENAME).is_symlink()
                and (child / MANIFEST_FILENAME).is_file()
            )
        )


# Concise compatibility name for callers that already use ``ModelRegistry``.
ModelRegistry = ModelArtifactRegistry


__all__ = ["ModelArtifactRegistry", "ModelRegistry"]
