"""
Article normalizer: converts a RawArticle into a NormalizedArticle.
Strips HTML, removes URLs, collapses whitespace, and extracts the Reddit score
from raw_metadata when the source is reddit.
"""
from __future__ import annotations

import re

from shared.models import NormalizedArticle, RawArticle
from shared.utils import strip_html

# Remove bare URLs from body text
_URL_RE = re.compile(r"https?://\S+")

# Collapse runs of whitespace (including \n, \t) to a single space
_WHITESPACE_RE = re.compile(r"\s+")


def normalize(article: RawArticle) -> NormalizedArticle:
    """Convert a RawArticle to a NormalizedArticle."""
    # 1. Strip HTML tags
    text = strip_html(article.body)
    # 2. Remove raw URLs
    text = _URL_RE.sub("", text)
    # 3. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text).strip()

    reddit_score = 0
    if article.source == "reddit":
        reddit_score = int(article.raw_metadata.get("reddit_score", 0))

    return NormalizedArticle(
        id=article.id,
        source=article.source,
        url=article.url,
        title=article.title,
        cleaned_body=text,
        published_at=article.published_at,
        fetched_at=article.fetched_at,
        word_count=len(text.split()),
        reddit_score=reddit_score,
    )
