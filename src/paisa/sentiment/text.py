"""Deterministic source-text selection without duplicating article evidence."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from paisa.sentiment.config import (
    DEFAULT_INPUT_POLICY_ID,
    DEFAULT_MAXIMUM_INPUT_CHARACTERS,
)
from paisa.sentiment.models import TextSelection


_WHITESPACE = re.compile(r"\s+")


class ArticleText(Protocol):
    title: str | None
    summary: str | None
    body: str | None


class MissingArticleTextError(ValueError):
    """Raised when an article cannot satisfy the declared input policy."""


def _normalized_field(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _WHITESPACE.sub(" ", value).strip()
    return normalized or None


def select_article_text(
    article: ArticleText,
    *,
    maximum_input_characters: int = DEFAULT_MAXIMUM_INPUT_CHARACTERS,
    policy_id: str = DEFAULT_INPUT_POLICY_ID,
) -> TextSelection:
    """Apply title+summary, title+body, then title-only selection in order."""

    if maximum_input_characters <= 0:
        raise ValueError("maximum_input_characters must be positive.")

    title = _normalized_field(article.title)
    summary = _normalized_field(article.summary)
    body = _normalized_field(article.body)
    if title is None:
        raise MissingArticleTextError(
            "Article has no usable title for the configured input policy."
        )

    if summary is not None:
        selected_fields = ("title", "summary")
        original_text = f"{title}\n\n{summary}"
    elif body is not None:
        selected_fields = ("title", "body")
        original_text = f"{title}\n\n{body}"
    else:
        selected_fields = ("title",)
        original_text = title

    input_text = original_text[:maximum_input_characters]
    return TextSelection(
        policy_id=policy_id,
        selected_fields=selected_fields,
        input_text=input_text,
        input_text_sha256=hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
        character_count=len(input_text),
        original_character_count=len(original_text),
        character_truncation_applied=len(input_text) < len(original_text),
        maximum_input_characters=maximum_input_characters,
    )
