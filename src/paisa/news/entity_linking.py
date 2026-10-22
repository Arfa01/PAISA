from __future__ import annotations

from dataclasses import replace
import hashlib
import re
import unicodedata
from typing import Iterable

from paisa.news.models import ArticleEntityLink, NormalizedNewsArticle


OGDC_TICKER = "OGDC"
OGDC_COMPANY_NAME = "Oil and Gas Development Company Limited"
OGDC_ALIASES = (
    "Oil and Gas Development Company Limited",
    "Oil & Gas Development Company Limited",
    "Oil and Gas Development Company",
    "Oil & Gas Development Company",
    "OGDCL",
    "OGDC",
)
_LOCATIONS = ("title", "summary", "body")


def _normalized_phrase(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _exact_alias(text: str, alias: str) -> bool:
    pattern = rf"(?<![A-Za-z0-9]){re.escape(alias)}(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def _normalized_alias(text: str, alias: str) -> bool:
    normalized_text = _normalized_phrase(text)
    normalized_alias = _normalized_phrase(alias)
    pattern = rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _make_link_id(
    article_id: str, alias: str, location: str, method: str
) -> str:
    payload = "\0".join((article_id, OGDC_TICKER, alias, location, method))
    return f"NEWSLINK-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def link_ogdc(
    article: NormalizedNewsArticle,
) -> tuple[NormalizedNewsArticle, list[ArticleEntityLink]]:
    """Link boundary-safe controlled OGDC aliases without probabilistic NER."""

    links: list[ArticleEntityLink] = []
    aliases = sorted(OGDC_ALIASES, key=len, reverse=True)
    for location in _LOCATIONS:
        text = getattr(article, location)
        if not text:
            continue
        match: tuple[str, str] | None = None
        for alias in aliases:
            if _exact_alias(text, alias):
                match = (alias, "exact")
                break
        if match is None:
            for alias in aliases:
                if _normalized_alias(text, alias):
                    match = (alias, "normalized_exact")
                    break
        if match is None:
            continue
        alias, method = match
        links.append(
            ArticleEntityLink(
                link_id=_make_link_id(article.article_id, alias, location, method),
                article_id=article.article_id,
                ticker=OGDC_TICKER,
                company_name=OGDC_COMPANY_NAME,
                matched_alias=alias,
                match_method=method,  # type: ignore[arg-type]
                match_location=location,  # type: ignore[arg-type]
                confidence=method,  # type: ignore[arg-type]
            )
        )

    if not links:
        return article, []
    linked = replace(
        article,
        mentioned_tickers=tuple(
            dict.fromkeys((*article.mentioned_tickers, OGDC_TICKER))
        ),
        mentioned_companies=tuple(
            dict.fromkeys((*article.mentioned_companies, OGDC_COMPANY_NAME))
        ),
    )
    return linked, links


def link_ogdc_articles(
    articles: Iterable[NormalizedNewsArticle],
) -> tuple[list[NormalizedNewsArticle], list[ArticleEntityLink]]:
    linked_articles: list[NormalizedNewsArticle] = []
    links: list[ArticleEntityLink] = []
    for article in articles:
        linked, article_links = link_ogdc(article)
        linked_articles.append(linked)
        links.extend(article_links)
    return linked_articles, links
