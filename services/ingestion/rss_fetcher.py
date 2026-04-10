"""
RSS feed fetcher.
Uses feedparser to pull articles from configured RSS URLs.
Filters to articles newer than the source's cursor timestamp.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from shared.config import RSS_FEEDS
from shared.models import RawArticle
from shared.utils import compute_hash, get_logger, strip_html, utcnow
from services.ingestion.cursor_store import get_cursor, set_cursor

logger = get_logger(__name__)

# Simple boilerplate patterns to strip from RSS bodies
_BOILERPLATE_RE = re.compile(
    r"(Read more|Continue reading|Click here|Subscribe now|Sign up for|"
    r"Copyright \d{4}|All rights reserved|Terms of use|Privacy policy)",
    re.IGNORECASE,
)


def _parse_published(entry: feedparser.FeedParserDict) -> datetime:
    """Extract and normalise the published date from a feed entry."""
    for attr in ("published", "updated"):
        raw = getattr(entry, attr, None)
        if raw:
            try:
                dt = parsedate_to_datetime(raw)
                # Ensure timezone-aware
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
    # Fall back to now so the article isn't filtered out
    return utcnow()


def _extract_body(entry: feedparser.FeedParserDict) -> str:
    """Extract the best available body text from a feed entry."""
    # Some feeds include full content
    if hasattr(entry, "content") and entry.content:
        raw = entry.content[0].get("value", "")
    elif hasattr(entry, "summary"):
        raw = entry.summary
    else:
        raw = ""
    cleaned = strip_html(raw)
    # Remove boilerplate phrases
    cleaned = _BOILERPLATE_RE.sub("", cleaned)
    return " ".join(cleaned.split())  # collapse whitespace


def fetch_rss(feed_url: str) -> list[RawArticle]:
    """
    Fetch articles from a single RSS feed URL.
    Returns only articles published after the stored cursor for this feed.
    """
    cursor = get_cursor(feed_url)
    logger.info("Fetching RSS: %s (since %s)", feed_url, cursor.isoformat())

    try:
        feed = feedparser.parse(feed_url)
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", feed_url, exc)
        return []

    articles: list[RawArticle] = []
    newest_ts = cursor

    for entry in feed.entries:
        published_at = _parse_published(entry)
        if published_at <= cursor:
            continue  # already seen

        title = getattr(entry, "title", "").strip()
        url = getattr(entry, "link", "").strip()
        if not title or not url:
            continue

        body = _extract_body(entry)
        article_id = compute_hash(url + title)

        article = RawArticle(
            id=article_id,
            source="rss",
            url=url,
            title=title,
            body=body,
            published_at=published_at,
            fetched_at=utcnow(),
            raw_metadata={"feed_url": feed_url},
        )
        articles.append(article)

        if published_at > newest_ts:
            newest_ts = published_at

    if articles:
        set_cursor(feed_url, newest_ts)
        logger.info("RSS %s: fetched %d new articles", feed_url, len(articles))
    else:
        logger.info("RSS %s: no new articles", feed_url)

    return articles


def fetch_all_rss() -> list[RawArticle]:
    """Fetch from all configured RSS feeds and merge results."""
    all_articles: list[RawArticle] = []
    for url in RSS_FEEDS:
        all_articles.extend(fetch_rss(url))
    return all_articles
