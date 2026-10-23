"""Checksum-bound permission policy for contemporaneous evidence providers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_POLICY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_STATUSES = frozenset({"PERMITTED", "MANUAL_ONLY", "REVIEW_REQUIRED", "PROHIBITED"})


class EvidencePermissionError(ValueError):
    """A provider is outside the reviewed permission boundary."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: Any, *, field_name: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or text == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in text
    ):
        raise EvidencePermissionError(f"{field_name} must be a safe relative path")
    return path.as_posix()


@dataclass(frozen=True, slots=True)
class EvidencePermissionPolicy:
    policy_id: str
    schema_version: str
    policy_path: str
    policy_sha256: str
    assessment_references: tuple[Mapping[str, str], ...]
    decisions: Mapping[tuple[str, str], Mapping[str, Any]]

    def decision(self, provider_id: str, provider_mode: str) -> Mapping[str, Any]:
        selected = self.decisions.get((provider_id, provider_mode))
        if selected is None:
            raise EvidencePermissionError(
                f"No reviewed permission decision for {provider_id}/{provider_mode}"
            )
        return selected

    def require(
        self,
        provider_id: str,
        provider_mode: str,
        *,
        automated: bool,
    ) -> Mapping[str, Any]:
        selected = self.decision(provider_id, provider_mode)
        status = selected["status"]
        if status in {"PROHIBITED", "REVIEW_REQUIRED"}:
            raise EvidencePermissionError(
                f"{provider_id}/{provider_mode} permission status is {status}"
            )
        if automated and status != "PERMITTED":
            raise EvidencePermissionError(
                f"{provider_id}/{provider_mode} is not permitted for automation"
            )
        if automated and selected.get("automated_access") is not True:
            raise EvidencePermissionError(
                f"{provider_id}/{provider_mode} forbids automated access"
            )
        if not automated and status not in {"PERMITTED", "MANUAL_ONLY"}:
            raise EvidencePermissionError(
                f"{provider_id}/{provider_mode} is not permitted for manual import"
            )
        return selected

    def reference(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "schema_version": self.schema_version,
            "path": self.policy_path,
            "sha256": self.policy_sha256,
            "assessment_references": [
                dict(item) for item in self.assessment_references
            ],
        }


def load_evidence_permission_policy(
    path: str | Path,
    *,
    repository_root: str | Path,
) -> EvidencePermissionPolicy:
    root = Path(repository_root).resolve()
    candidate = Path(path)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise EvidencePermissionError(
                "Permission policy must be inside the repository"
            ) from exc
    else:
        relative = _safe_relative(candidate.as_posix(), field_name="policy path")
        resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise EvidencePermissionError(
            "Permission policy escaped the repository"
        ) from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise EvidencePermissionError("Permission policy must be a regular file")
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidencePermissionError("Permission policy is unreadable") from exc
    if not isinstance(raw, Mapping) or raw.get("schema_version") != "1.0":
        raise EvidencePermissionError("Permission policy schema is unsupported")
    policy_id = str(raw.get("policy_id") or "")
    if not _POLICY_ID.fullmatch(policy_id):
        raise EvidencePermissionError("Permission policy ID is invalid")

    raw_assessments = raw.get("assessment_references")
    if not isinstance(raw_assessments, list) or not raw_assessments:
        raise EvidencePermissionError("Permission policy assessments are missing")
    assessments: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(raw_assessments):
        if not isinstance(item, Mapping):
            raise EvidencePermissionError("Assessment reference is invalid")
        assessment_relative = _safe_relative(
            item.get("path"), field_name=f"assessment_references[{index}].path"
        )
        if assessment_relative in seen_paths:
            raise EvidencePermissionError("Assessment reference is duplicated")
        seen_paths.add(assessment_relative)
        expected = str(item.get("sha256") or "")
        if not _SHA256.fullmatch(expected):
            raise EvidencePermissionError("Assessment SHA-256 is invalid")
        assessment_path = (root / assessment_relative).resolve()
        try:
            assessment_path.relative_to(root)
        except ValueError as exc:
            raise EvidencePermissionError(
                "Assessment reference escaped the repository"
            ) from exc
        if assessment_path.is_symlink() or not assessment_path.is_file():
            raise EvidencePermissionError("Assessment must be a regular file")
        if sha256_file(assessment_path) != expected:
            raise EvidencePermissionError("Assessment checksum does not match policy")
        assessments.append({"path": assessment_relative, "sha256": expected})

    raw_decisions = raw.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise EvidencePermissionError("Permission decisions are missing")
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for item in raw_decisions:
        if not isinstance(item, Mapping):
            raise EvidencePermissionError("Permission decision is invalid")
        provider_id = str(item.get("provider_id") or "").strip()
        provider_mode = str(item.get("provider_mode") or "").strip()
        status = str(item.get("status") or "")
        if not provider_id or not provider_mode or status not in _STATUSES:
            raise EvidencePermissionError("Permission decision fields are invalid")
        key = (provider_id, provider_mode)
        if key in decisions:
            raise EvidencePermissionError("Permission decision is duplicated")
        decision = dict(item)
        decision["provider_id"] = provider_id
        decision["provider_mode"] = provider_mode
        decision["status"] = status
        decisions[key] = decision

    return EvidencePermissionPolicy(
        policy_id=policy_id,
        schema_version="1.0",
        policy_path=relative,
        policy_sha256=sha256_file(resolved),
        assessment_references=tuple(assessments),
        decisions=decisions,
    )


__all__ = [
    "EvidencePermissionError",
    "EvidencePermissionPolicy",
    "load_evidence_permission_policy",
    "sha256_file",
]
