"""
Cursor store: persists the last-seen timestamp per source so each pipeline
run only fetches new articles (incremental ingestion).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

import shared.config as cfg
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)


def _load() -> dict[str, str]:
    if os.path.exists(cfg.CURSORS_PATH):
        with open(cfg.CURSORS_PATH) as f:
            return json.load(f)
    return {}


def _save(cursors: dict[str, str]) -> None:
    os.makedirs(os.path.dirname(cfg.CURSORS_PATH) or ".", exist_ok=True)
    with open(cfg.CURSORS_PATH, "w") as f:
        json.dump(cursors, f, indent=2)


def get_cursor(source: str) -> datetime:
    """
    Return the last-seen datetime for a source.
    Defaults to `INGESTION_LOOKBACK_HOURS` ago if no cursor exists.
    """
    cursors = _load()
    if source in cursors:
        return datetime.fromisoformat(cursors[source])
    fallback = utcnow().replace(second=0, microsecond=0)
    return fallback - timedelta(hours=cfg.INGESTION_LOOKBACK_HOURS)


def set_cursor(source: str, timestamp: datetime) -> None:
    """Update the cursor for a source to `timestamp`."""
    cursors = _load()
    cursors[source] = timestamp.isoformat()
    _save(cursors)
    logger.debug("Cursor updated: %s → %s", source, timestamp.isoformat())
