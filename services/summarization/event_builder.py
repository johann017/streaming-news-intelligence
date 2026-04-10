"""
Builds Event objects from RankedClusters + summaries.
Sets TTL, extracts geo tags from titles using the geo_lookup.json dictionary.
"""
from __future__ import annotations

import json
import os
from datetime import timedelta

import shared.config as cfg
from shared.models import Event, NormalizedArticle, RankedCluster
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)

_geo_lookup: dict[str, list[str]] | None = None


def _load_geo_lookup() -> dict[str, list[str]]:
    global _geo_lookup
    if _geo_lookup is not None:
        return _geo_lookup
    geo_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "shared", "geo_lookup.json"
    )
    try:
        with open(geo_path) as f:
            data = json.load(f)
        _geo_lookup = {
            "countries": data.get("countries", []),
            "cities": data.get("cities", []),
        }
    except Exception as exc:
        logger.warning("Failed to load geo_lookup.json: %s", exc)
        _geo_lookup = {"countries": [], "cities": []}
    return _geo_lookup


def extract_geo_tags(text: str) -> list[str]:
    """
    Find country/city names mentioned in `text`.
    Simple substring match (case-insensitive) against geo_lookup.json.
    Returns deduplicated list of matched names.
    """
    lookup = _load_geo_lookup()
    text_lower = text.lower()
    found: set[str] = set()
    for name in lookup["countries"] + lookup["cities"]:
        if name.lower() in text_lower:
            found.add(name)
    return sorted(found)


def build_event(
    cluster: RankedCluster,
    summary: str,
    key_points: list[str],
    articles_by_id: dict[str, NormalizedArticle],
) -> Event:
    """Construct an Event from a RankedCluster and its generated summary."""
    now = utcnow()

    ttl_hours = cfg.TOP_EVENT_TTL_HOURS if cluster.is_top_event else cfg.EVENT_TTL_HOURS
    expires_at = now + timedelta(hours=ttl_hours)

    # Collect source URLs (up to 5 to keep Firestore document small)
    sources = [
        articles_by_id[aid].url
        for aid in cluster.article_ids
        if aid in articles_by_id
    ][:5]

    # Geo tags from representative title + summary
    combined_text = f"{cluster.representative_title} {summary}"
    geo_tags = extract_geo_tags(combined_text)

    return Event(
        event_id=cluster.cluster_id,
        cluster_id=cluster.cluster_id,
        summary=summary or cluster.representative_title,
        key_points=key_points,
        sources=sources,
        representative_title=cluster.representative_title,
        score=cluster.score,
        is_top_event=cluster.is_top_event,
        geo_tags=geo_tags,
        created_at=now,
        expires_at=expires_at,
    )
