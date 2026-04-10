"""
Ingestion service entry point.
Fetches from all configured sources, merges results, deduplicates by article ID,
and writes to data/raw_articles.json.
"""
from __future__ import annotations

import json
import os
import sys

# Allow running as `python services/ingestion/run.py` from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import shared.config as cfg
from shared.models import RawArticle
from shared.utils import get_logger

from services.ingestion.rss_fetcher import fetch_all_rss
from services.ingestion.reddit_fetcher import fetch_all_reddit
from services.ingestion.gdelt_fetcher import fetch_gdelt

logger = get_logger(__name__)


def run() -> list[RawArticle]:
    """
    Run all ingestion sources and write merged, deduplicated results to disk.
    Returns the list of new RawArticles fetched this run.
    """
    logger.info("=== Ingestion started ===")

    rss_articles = fetch_all_rss()
    reddit_articles = fetch_all_reddit()
    gdelt_articles = fetch_gdelt()

    all_articles = rss_articles + reddit_articles + gdelt_articles

    # Deduplicate within this batch (same article from multiple sources)
    seen: set[str] = set()
    unique: list[RawArticle] = []
    for article in all_articles:
        if article.id not in seen:
            seen.add(article.id)
            unique.append(article)

    logger.info(
        "Ingestion complete: %d RSS + %d Reddit + %d GDELT = %d total, %d unique",
        len(rss_articles),
        len(reddit_articles),
        len(gdelt_articles),
        len(all_articles),
        len(unique),
    )

    os.makedirs(os.path.dirname(cfg.RAW_ARTICLES_PATH) or ".", exist_ok=True)
    with open(cfg.RAW_ARTICLES_PATH, "w") as f:
        json.dump([a.to_dict() for a in unique], f, indent=2)

    logger.info("Wrote %d articles to %s", len(unique), cfg.RAW_ARTICLES_PATH)
    return unique


if __name__ == "__main__":
    run()
