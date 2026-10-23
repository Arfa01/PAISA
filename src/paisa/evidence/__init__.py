"""Contemporaneous historical evidence contracts for PAISA.

This package is deliberately separate from :mod:`paisa.news`.  Publisher news,
GDELT metadata, and exchange announcements have different provenance and time
semantics and must not be relabelled as one another.
"""

from paisa.evidence.models import (
    EVIDENCE_SCHEMA_VERSION,
    EVENT_CATEGORIES,
    EvidenceCapture,
    EvidenceError,
    EvidenceFetchResult,
    EvidenceRecord,
)

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "EVENT_CATEGORIES",
    "EvidenceCapture",
    "EvidenceError",
    "EvidenceFetchResult",
    "EvidenceRecord",
]
