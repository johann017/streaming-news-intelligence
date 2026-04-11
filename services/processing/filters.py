"""
Article filters — boolean predicates applied before normalization.
Keeping these separate from the normalizer makes them individually testable.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import timedelta

from nltk.stem import PorterStemmer

try:
    from langdetect import detect, LangDetectException
except ImportError:
    detect = None  # type: ignore[assignment]
    LangDetectException = Exception  # type: ignore[assignment,misc]

import shared.config as cfg
from shared.models import RawArticle
from shared.utils import utcnow

# ---------------------------------------------------------------------------
# Off-topic blocklist — titles containing these are dropped
# ---------------------------------------------------------------------------
_BLOCKLIST = re.compile(
    r"\b("
    r"cat|cats|kitten|kittens|puppy|puppies|dog|dogs|pet|pets|"
    r"celebrity|celebrities|kardashian|taylor swift|beyoncé|"
    r"nfl|nba|nhl|mlb|premier league|soccer|football score|"
    r"recipe|cooking|baking|restaurant review|"
    r"gaming|video game|minecraft|fortnite|twitch|"
    r"meme|viral video|tiktok trend|"
    r"horoscope|zodiac|astrology|"
    r"mugshot|mug shot|true crime"
    r")\b",
    re.IGNORECASE,
)

# GDELT URL-derived titles are often pure alphanumeric slugs with no spaces.
# Require at least 3 words and no runs of random characters.
_GARBAGE_TITLE = re.compile(r"^[a-z0-9]{8,}$", re.IGNORECASE)

# Taxonomy / machine-code token pattern: all-caps words with digits and underscores,
# e.g. TAX_FNCACT, CRISISLEX_T03_DEAD, SOC_POINTSOFINTEREST, USPEC_POLITICS_GENERAL1
_MACHINE_CODE_TOKEN = re.compile(r"^[A-Z][A-Z0-9_]{3,}$")

# PorterStemmer reduces inflected forms to a common root so we can detect when
# multiple surface forms of the same word appear in the same text — a reliable
# signal of SEO keyword stuffing (e.g. "murder murdered kill killed arrest arrested").
# PorterStemmer is a pure algorithm; no NLTK corpus download is required.
_stemmer = PorterStemmer()
_WORD_RE = re.compile(r"\b[a-z]{4,}\b")


def _stuffed_stem_count(text: str) -> int:
    """
    Count how many word stems have 2+ distinct surface forms in the text.
    Natural prose rarely repeats a word alongside its own inflected variants;
    keyword-stuffed slugs do this systematically.
    """
    stem_forms: dict[str, set[str]] = defaultdict(set)
    for word in _WORD_RE.findall(text.lower()):
        stem_forms[_stemmer.stem(word)].add(word)
    return sum(1 for forms in stem_forms.values() if len(forms) >= 2)


def is_recent(article: RawArticle) -> bool:
    """Return True if the article was published within MAX_ARTICLE_AGE_HOURS."""
    from datetime import timezone

    now = utcnow()
    cutoff = now - timedelta(hours=cfg.MAX_ARTICLE_AGE_HOURS)
    pub = article.published_at
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
    Skips language detection for very short texts (< 30 words).
    """
    body = article.body.strip()
    words = body.split()
    if len(words) < 30:
        return True
    if detect is None:
        return True
    try:
        return detect(body) == "en"
    except Exception:
        return True


def is_relevant(article: RawArticle) -> bool:
    """
    Return True if the article is on-topic (world news).

    Drops:
    - Titles matching the off-topic blocklist (cats, sports scores, etc.)
    - GDELT articles with garbage URL-slug titles (no spaces, pure alphanumeric)
    """
    title = article.title.strip()

    # Must have at least 2 words — single-word slug titles are useless
    if len(title.split()) < 2:
        return False

    # GDELT-specific: drop pure alphanumeric slugs like "ab12cd34ef56"
    if article.source == "gdelt" and _GARBAGE_TITLE.match(title):
        return False

    # Drop off-topic content
    if _BLOCKLIST.search(title):
        return False

    return True


def is_natural_language_body(article: RawArticle) -> bool:
    """
    Return True if the article body is prose, not machine-generated codes or
    SEO keyword dumps.

    Rejects:
    - Bodies where >40% of tokens are taxonomy-style labels
      (e.g. GDELT GKG theme codes like TAX_FNCACT, CRISISLEX_T03_DEAD)
    - Bodies where 3+ words appear in both base and inflected form
      (e.g. "murder murdered kill killed arrest arrested"), which is the
      fingerprint of tabloid/true-crime keyword-stuffed URL slugs.
    """
    tokens = article.body.split()
    if not tokens:
        return False
    machine_count = sum(1 for t in tokens if _MACHINE_CODE_TOKEN.match(t))
    if (machine_count / len(tokens)) > 0.4:
        return False
    if _stuffed_stem_count(article.body) >= 3:
        return False
    return True


def passes_all_filters(article: RawArticle) -> bool:
    """Return True if the article passes all filters."""
    return (
        is_recent(article)
        and has_sufficient_body(article)
        and is_english(article)
        and is_relevant(article)
        and is_natural_language_body(article)
    )
