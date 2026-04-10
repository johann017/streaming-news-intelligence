"""
Reddit fetcher (unauthenticated).
Uses the public Reddit JSON API — no OAuth or credentials required.
Fetches hot posts from configured subreddits.
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests

from shared.config import REDDIT_POST_LIMIT, REDDIT_SUBREDDITS
from shared.models import RawArticle
from shared.utils import compute_hash, get_logger, utcnow
from services.ingestion.cursor_store import get_cursor, set_cursor

logger = get_logger(__name__)

_REDDIT_HOT_URL = "https://www.reddit.com/r/{subreddit}/hot.json?limit={limit}"
_HEADERS = {
    # Reddit requires a descriptive User-Agent for the JSON API
    "User-Agent": "streaming-news-intelligence/0.1 (pipeline; contact via GitHub)"
}
_TIMEOUT = 15  # seconds


def fetch_subreddit(subreddit: str) -> list[RawArticle]:
    """
    Fetch hot posts from a single subreddit.
    Returns only posts newer than the stored cursor.
    """
    cursor_key = f"reddit:{subreddit}"
    cursor = get_cursor(cursor_key)
    logger.info("Fetching r/%s (since %s)", subreddit, cursor.isoformat())

    url = _REDDIT_HOT_URL.format(subreddit=subreddit, limit=REDDIT_POST_LIMIT)
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch r/%s: %s", subreddit, exc)
        return []

    posts = data.get("data", {}).get("children", [])
    articles: list[RawArticle] = []
    newest_ts = cursor

    for post in posts:
        p = post.get("data", {})

        # Convert Unix timestamp to aware datetime
        created_utc = p.get("created_utc", 0)
        published_at = datetime.fromtimestamp(created_utc, tz=timezone.utc)

        if published_at <= cursor:
            continue

        title = p.get("title", "").strip()
        url_val = p.get("url", "").strip()
        if not title or not url_val:
            continue

        # Use the Reddit post text if available, otherwise just the title
        selftext = p.get("selftext", "").strip()
        body = selftext if selftext and selftext != "[removed]" else title

        article_id = compute_hash(url_val + title)
        article = RawArticle(
            id=article_id,
            source="reddit",
            url=url_val,
            title=title,
            body=body,
            published_at=published_at,
            fetched_at=utcnow(),
            raw_metadata={
                "subreddit": subreddit,
                "reddit_score": p.get("score", 0),
                "num_comments": p.get("num_comments", 0),
                "reddit_url": f"https://reddit.com{p.get('permalink', '')}",
            },
        )
        articles.append(article)

        if published_at > newest_ts:
            newest_ts = published_at

    if articles:
        set_cursor(cursor_key, newest_ts)
        logger.info("r/%s: fetched %d new posts", subreddit, len(articles))
    else:
        logger.info("r/%s: no new posts", subreddit)

    return articles


def fetch_all_reddit() -> list[RawArticle]:
    """Fetch from all configured subreddits and merge results."""
    all_articles: list[RawArticle] = []
    for sub in REDDIT_SUBREDDITS:
        all_articles.extend(fetch_subreddit(sub))
    return all_articles
