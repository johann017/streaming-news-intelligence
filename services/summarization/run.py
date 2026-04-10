"""
Summarization service entry point.
Reads ranked_clusters.json + normalized_articles.json → summarizes → builds
Events → writes events.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import shared.config as cfg
from shared.models import NormalizedArticle, RankedCluster
from shared.utils import get_logger

from services.summarization.summarizer import summarize
from services.summarization.key_points import extract_key_points
from services.summarization.event_builder import build_event

logger = get_logger(__name__)


def run() -> list:
    """
    Run the summarization stage.
    Returns the list of Event objects written to disk.
    """
    logger.info("=== Summarization started ===")

    if not os.path.exists(cfg.RANKED_CLUSTERS_PATH):
        logger.warning("No ranked clusters at %s — skipping", cfg.RANKED_CLUSTERS_PATH)
        return []

    with open(cfg.RANKED_CLUSTERS_PATH) as f:
        ranked_dicts = json.load(f)
    clusters = [RankedCluster.from_dict(d) for d in ranked_dicts]

    articles_by_id: dict[str, NormalizedArticle] = {}
    if os.path.exists(cfg.NORMALIZED_ARTICLES_PATH):
        with open(cfg.NORMALIZED_ARTICLES_PATH) as f:
            for d in json.load(f):
                a = NormalizedArticle.from_dict(d)
                articles_by_id[a.id] = a

    logger.info("Loaded %d ranked clusters, %d articles", len(clusters), len(articles_by_id))

    events = []
    summarized_count = 0
    extractive_count = 0

    for cluster in clusters:
        # Build text to summarize: combine representative article body
        if cluster.score >= cfg.MIN_SCORE_FOR_SUMMARIZATION:
            text_parts = []
            for aid in cluster.article_ids[:3]:  # use up to 3 articles
                a = articles_by_id.get(aid)
                if a and a.cleaned_body:
                    text_parts.append(a.cleaned_body)
            text = " ".join(text_parts)

            summary_text = summarize(text)
            summarized_count += 1
        else:
            # Below threshold: use representative title as bare summary
            summary_text = cluster.representative_title
            extractive_count += 1

        key_points = extract_key_points(summary_text)
        event = build_event(cluster, summary_text, key_points, articles_by_id)
        events.append(event)

    logger.info(
        "Summarization: %d abstractive, %d extractive (threshold=%.2f)",
        summarized_count,
        extractive_count,
        cfg.MIN_SCORE_FOR_SUMMARIZATION,
    )

    os.makedirs(os.path.dirname(cfg.EVENTS_PATH) or ".", exist_ok=True)
    with open(cfg.EVENTS_PATH, "w") as f:
        json.dump([e.to_dict() for e in events], f, indent=2)

    logger.info(
        "Summarization complete: %d events written to %s", len(events), cfg.EVENTS_PATH
    )
    return events


if __name__ == "__main__":
    run()
