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
    gdelt_skipped = 0

    for cluster in clusters:
        # Skip clusters where the representative article is GDELT. This only
        # occurs when every article in the cluster is from GDELT, because the
        # cluster_builder already prefers non-GDELT sources as the representative.
        # GDELT titles are URL-slug derivatives and can semantically invert the
        # article's meaning (e.g. "Autism Ice Agents Detain…" instead of
        # "ICE agents detain man with autism"). GDELT's role is corroboration
        # during clustering, not as a display source.
        rep = articles_by_id.get(cluster.representative_id)
        if rep is None or rep.source == "gdelt":
            gdelt_skipped += 1
            continue
        if cluster.score >= cfg.MIN_SCORE_FOR_SUMMARIZATION:
            # Always summarize from the representative article so the generated
            # text is guaranteed to match the title. Mixing bodies from multiple
            # cluster members caused title/body mismatches when the centroid sat
            # between unrelated articles.
            rep = articles_by_id.get(cluster.representative_id)
            text = (rep.cleaned_body if rep else "") or cluster.representative_title

            summary_text = summarize(text)
            summarized_count += 1
        else:
            summary_text = cluster.representative_title
            extractive_count += 1

        key_points = extract_key_points(summary_text)
        event = build_event(cluster, summary_text, key_points, articles_by_id)
        events.append(event)

    logger.info(
        "Summarization: %d abstractive, %d extractive, %d GDELT-only skipped (threshold=%.2f)",
        summarized_count,
        extractive_count,
        gdelt_skipped,
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
