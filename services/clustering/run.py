"""
Clustering service entry point.
Reads normalized_articles.json → embeds → clusters → writes clusters.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import shared.config as cfg
from shared.models import NormalizedArticle
from shared.utils import get_logger

from services.clustering.embedder import embed_articles
from services.clustering.clusterer import cluster_embeddings
from services.clustering.cluster_builder import build_clusters

logger = get_logger(__name__)


def run() -> list:
    """
    Run the embedding + clustering pipeline stage.
    Returns the list of Cluster objects written to disk.
    """
    logger.info("=== Clustering started ===")

    if not os.path.exists(cfg.NORMALIZED_ARTICLES_PATH):
        logger.warning("No normalized articles at %s — skipping", cfg.NORMALIZED_ARTICLES_PATH)
        return []

    with open(cfg.NORMALIZED_ARTICLES_PATH) as f:
        normalized_dicts = json.load(f)

    articles = [NormalizedArticle.from_dict(d) for d in normalized_dicts]
    logger.info("Loaded %d normalized articles", len(articles))

    if not articles:
        return []

    articles_by_id = {a.id: a for a in articles}

    # Step 1: Embed
    embeddings = embed_articles(articles)

    # Step 2: Cluster
    groups = cluster_embeddings(embeddings)

    # Step 3: Build Cluster objects
    clusters = build_clusters(groups, embeddings, articles_by_id)

    os.makedirs(os.path.dirname(cfg.CLUSTERS_PATH) or ".", exist_ok=True)
    with open(cfg.CLUSTERS_PATH, "w") as f:
        json.dump([c.to_dict() for c in clusters], f, indent=2)

    logger.info(
        "Clustering complete: %d clusters written to %s", len(clusters), cfg.CLUSTERS_PATH
    )
    return clusters


if __name__ == "__main__":
    run()
