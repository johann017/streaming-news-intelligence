"""
Cross-run deduplication.
Maintains a FIFO set of seen article IDs persisted to disk so that articles
already processed in previous pipeline runs are dropped immediately.
"""
from __future__ import annotations

import json
import os
from collections import deque

import shared.config as cfg
from shared.models import RawArticle
from shared.utils import get_logger

logger = get_logger(__name__)


def _load_seen_ids() -> deque[str]:
    if os.path.exists(cfg.SEEN_IDS_PATH):
        with open(cfg.SEEN_IDS_PATH) as f:
            return deque(json.load(f), maxlen=cfg.SEEN_IDS_MAX_SIZE)
    return deque(maxlen=cfg.SEEN_IDS_MAX_SIZE)


def _save_seen_ids(seen: deque[str]) -> None:
    os.makedirs(os.path.dirname(cfg.SEEN_IDS_PATH) or ".", exist_ok=True)
    with open(cfg.SEEN_IDS_PATH, "w") as f:
        json.dump(list(seen), f)


def deduplicate(articles: list[RawArticle]) -> list[RawArticle]:
    """
    Filter out articles whose IDs have been seen in previous pipeline runs.
    Updates the persisted seen-IDs set with newly encountered IDs.

    Uses a deque capped at SEEN_IDS_MAX_SIZE for bounded memory.
    When the cap is reached, oldest entries are evicted (FIFO).
    """
    seen = _load_seen_ids()
    seen_set: set[str] = set(seen)

    new_articles: list[RawArticle] = []
    for article in articles:
        if article.id in seen_set:
            continue
        new_articles.append(article)
        seen.append(article.id)
        seen_set.add(article.id)

    _save_seen_ids(seen)

    dropped = len(articles) - len(new_articles)
    if dropped:
        logger.info("Deduplicator: dropped %d already-seen articles", dropped)
    logger.info("Deduplicator: %d new articles pass through", len(new_articles))

    return new_articles
