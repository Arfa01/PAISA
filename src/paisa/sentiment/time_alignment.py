"""Leakage-safe sentiment availability filtering for later feature builders."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, TypeVar


Observation = TypeVar("Observation")


class SentimentTimingEvidenceError(ValueError):
    """Raised instead of inventing absent or timezone-naive availability."""


def _available_at(observation: object) -> object:
    if isinstance(observation, Mapping):
        return observation.get("available_at")
    return getattr(observation, "available_at", None)


def filter_sentiment_by_decision_cutoff(
    observations: Iterable[Observation],
    decision_cutoff: datetime,
) -> tuple[Observation, ...]:
    """Return observations known by cutoff; never consult inference time."""

    if decision_cutoff.tzinfo is None or decision_cutoff.utcoffset() is None:
        raise SentimentTimingEvidenceError(
            "Decision cutoff must be timezone-aware."
        )
    eligible: list[Observation] = []
    for observation in observations:
        available_at = _available_at(observation)
        if not isinstance(available_at, datetime):
            raise SentimentTimingEvidenceError(
                "Sentiment observation is missing datetime available_at evidence."
            )
        if available_at.tzinfo is None or available_at.utcoffset() is None:
            raise SentimentTimingEvidenceError(
                "Sentiment available_at evidence must be timezone-aware."
            )
        if available_at <= decision_cutoff:
            eligible.append(observation)
    return tuple(eligible)
