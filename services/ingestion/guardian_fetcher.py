"""
The Guardian news fetcher.
Uses The Guardian's free Content API to fetch recent articles.
Requires a free API key: https://open-platform.theguardian.com/access/

Free tier limits: 1 request/second, 5,000 requests/day — well within budget
at the current 7-minute pipeline interval.
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests

from shared.config import GUARDIAN_API_KEY, GUARDIAN_SECTIONS
from shared.models import RawArticle
from shared.utils import compute_hash, get_logger, utcnow
from services.ingestion.cursor_store import get_cursor, set_cursor

logger = get_logger(__name__)

_GUARDIAN_URL = "https://content.guardianapis.com/search"
_CURSOR_KEY = "guardian"
_TIMEOUT = 15  # seconds
_PAGE_SIZE = 50


def fetch_guardian(max_articles: int = 100) -> list[RawArticle]:
    """
    Fetch recent articles from The Guardian Content API.
    Only fetches articles published after the stored cursor.
    Returns an empty list if GUARDIAN_API_KEY is not configured.
    """
    if not GUARDIAN_API_KEY:
        logger.warning("GUARDIAN_API_KEY not set — skipping Guardian fetch")
        return []

    cursor = get_cursor(_CURSOR_KEY)

    params = {
        "api-key": GUARDIAN_API_KEY,
        "section": GUARDIAN_SECTIONS,
        "show-fields": "bodyText,headline",
        "order-by": "newest",
        "page-size": _PAGE_SIZE,
        # Pass the full cursor datetime so the API filters server-side.
        # The Guardian accepts ISO 8601 with time for from-date.
        "from-date": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    try:
        resp = requests.get(_GUARDIAN_URL, params=params, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Failed to fetch Guardian articles: %s", exc)
        return []

    results = data.get("response", {}).get("results", [])

    articles: list[RawArticle] = []
    newest_ts = cursor

    for item in results:
        if len(articles) >= max_articles:
            break

        pub_str = item.get("webPublicationDate", "")
        try:
            published_at = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published_at = utcnow()

        # Client-side guard against duplicates (API from-date is inclusive)
        if published_at <= cursor:
            continue

        url = item.get("webUrl", "").strip()
        if not url:
            continue

        fields = item.get("fields", {})
        title = (fields.get("headline") or item.get("webTitle", "")).strip()
        body = (fields.get("bodyText") or title).strip()

        if not title:
            continue

        article_id = compute_hash(url + title)
        articles.append(
            RawArticle(
                id=article_id,
                source="guardian",
                url=url,
                title=title,
                body=body,
                published_at=published_at,
                fetched_at=utcnow(),
                raw_metadata={"section": item.get("sectionName", "")},
            )
        )

        if published_at > newest_ts:
            newest_ts = published_at

    if articles:
        set_cursor(_CURSOR_KEY, newest_ts)
        logger.info("Guardian: fetched %d new articles", len(articles))
    else:
        logger.info("Guardian: no new articles since %s", cursor.isoformat())

    return articles
