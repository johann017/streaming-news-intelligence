"""
Article filters — boolean predicates applied before normalization.
Keeping these separate from the normalizer makes them individually testable.
"""
from __future__ import annotations

from datetime import timedelta

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None  # type: ignore[assignment]
    LangDetectException = Exception  # type: ignore[assignment,misc]

import shared.config as cfg
from shared.models import RawArticle
from shared.utils import utcnow


def is_recent(article: RawArticle) -> bool:
    """Return True if the article was published within MAX_ARTICLE_AGE_HOURS."""
    from datetime import timezone
    now = utcnow()
    cutoff = now - timedelta(hours=cfg.MAX_ARTICLE_AGE_HOURS)
    pub = article.published_at
    # Ensure both are timezone-aware for comparison
    if pub.tzinfo is None:
        pub = pub.replace(tzinfo=timezone.utc)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=timezone.utc)
    return pub >= cutoff


def has_sufficient_body(article: RawArticle) -> bool:
    """Return True if the article body has at least MIN_BODY_WORDS words."""
    word_count = len(article.body.split())
    return word_count >= cfg.MIN_BODY_WORDS


def is_english(article: RawArticle) -> bool:
    """
    Return True if the article appears to be in English.
    Skips language detection for very short texts (< 30 words) — assumed English
    since our RSS sources are English-language feeds.
    """
    body = article.body.strip()
    words = body.split()
    if len(words) < 30:
        return True  # too short to detect reliably; assume English
    if detect is None:
        return True  # langdetect not installed; assume English
    try:
        return detect(body) == "en"
    except Exception:
        return True  # on detection failure, keep the article


def passes_all_filters(article: RawArticle) -> bool:
    """Return True if the article passes all filters."""
    return is_recent(article) and has_sufficient_body(article) and is_english(article)
