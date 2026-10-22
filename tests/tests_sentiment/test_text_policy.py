from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from paisa.sentiment.text import MissingArticleTextError, select_article_text


def test_text_policy_is_deterministic_and_prefers_summary(article):
    first = select_article_text(article)
    second = select_article_text(article)

    assert first == second
    assert first.selected_fields == ("title", "summary")
    assert first.input_text == (
        "OGDC reports higher production\n\n"
        "The company reported an increase in output."
    )
    assert first.input_text_sha256 == hashlib.sha256(
        first.input_text.encode("utf-8")
    ).hexdigest()
    assert "input_text" not in first.metadata_dict()


def test_policy_uses_permitted_body_only_when_summary_is_unavailable(article):
    selected = select_article_text(replace(article, summary="  \n "))

    assert selected.selected_fields == ("title", "body")
    assert selected.input_text.endswith("additional details.")


def test_policy_falls_back_to_title_only(article):
    selected = select_article_text(
        replace(article, summary=None, body=None, title="  OGDC   update  ")
    )

    assert selected.selected_fields == ("title",)
    assert selected.input_text == "OGDC update"


@pytest.mark.parametrize("title", [None, "", " \n\t "])
def test_policy_rejects_missing_title_instead_of_inventing_text(article, title):
    with pytest.raises(MissingArticleTextError):
        select_article_text(replace(article, title=title))


def test_character_cap_records_truncation_and_hashes_actual_input(article):
    selected = select_article_text(article, maximum_input_characters=20)

    assert selected.character_count == 20
    assert selected.original_character_count > 20
    assert selected.character_truncation_applied is True
    assert selected.input_text_sha256 == hashlib.sha256(
        selected.input_text.encode("utf-8")
    ).hexdigest()
