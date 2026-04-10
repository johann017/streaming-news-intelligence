"""
Processing service entry point.
Reads raw_articles.json → deduplicates → filters → normalizes →
writes normalized_articles.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import shared.config as cfg
from shared.models import NormalizedArticle, RawArticle
from shared.utils import get_logger

from services.processing.deduplicator import deduplicate
from services.processing.filters import passes_all_filters
from services.processing.normalizer import normalize

logger = get_logger(__name__)


def run() -> list[NormalizedArticle]:
    """
    Process raw articles into normalized form.
    Returns the list of NormalizedArticles written to disk.
    """
    logger.info("=== Processing started ===")

    if not os.path.exists(cfg.RAW_ARTICLES_PATH):
        logger.warning("No raw articles file found at %s — skipping", cfg.RAW_ARTICLES_PATH)
        return []

    with open(cfg.RAW_ARTICLES_PATH) as f:
        raw_dicts = json.load(f)

    raw_articles = [RawArticle.from_dict(d) for d in raw_dicts]
    logger.info("Loaded %d raw articles", len(raw_articles))

    # Step 1: Cross-run deduplication
    unique = deduplicate(raw_articles)

    # Step 2: Filter (recency, body length, language)
    passed = [a for a in unique if passes_all_filters(a)]
    filtered_count = len(unique) - len(passed)
    if filtered_count:
        logger.info("Filters dropped %d articles (%d remain)", filtered_count, len(passed))

    # Step 3: Normalize
    normalized = [normalize(a) for a in passed]

    os.makedirs(os.path.dirname(cfg.NORMALIZED_ARTICLES_PATH) or ".", exist_ok=True)
    with open(cfg.NORMALIZED_ARTICLES_PATH, "w") as f:
        json.dump([a.to_dict() for a in normalized], f, indent=2)

    logger.info(
        "Processing complete: %d raw → %d normalized, written to %s",
        len(raw_articles),
        len(normalized),
        cfg.NORMALIZED_ARTICLES_PATH,
    )
    return normalized


if __name__ == "__main__":
    run()
