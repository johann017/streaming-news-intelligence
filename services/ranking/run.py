"""
Ranking service entry point.
Reads clusters.json + normalized_articles.json → scores + ranks →
writes ranked_clusters.json.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import shared.config as cfg
from shared.models import Cluster, NormalizedArticle
from shared.utils import get_logger

from services.ranking.ranker import rank_clusters

logger = get_logger(__name__)


def run() -> list:
    """
    Run the ranking stage.
    Returns the list of RankedCluster objects written to disk.
    """
    logger.info("=== Ranking started ===")

    if not os.path.exists(cfg.CLUSTERS_PATH):
        logger.warning("No clusters file at %s — skipping", cfg.CLUSTERS_PATH)
        return []

    with open(cfg.CLUSTERS_PATH) as f:
        cluster_dicts = json.load(f)
    clusters = [Cluster.from_dict(d) for d in cluster_dicts]

    # Load normalized articles for scoring context (reddit scores etc.)
    articles_by_id: dict[str, NormalizedArticle] = {}
    if os.path.exists(cfg.NORMALIZED_ARTICLES_PATH):
        with open(cfg.NORMALIZED_ARTICLES_PATH) as f:
            for d in json.load(f):
                a = NormalizedArticle.from_dict(d)
                articles_by_id[a.id] = a

    logger.info("Loaded %d clusters, %d articles", len(clusters), len(articles_by_id))

    ranked = rank_clusters(clusters, articles_by_id)

    os.makedirs(os.path.dirname(cfg.RANKED_CLUSTERS_PATH) or ".", exist_ok=True)
    with open(cfg.RANKED_CLUSTERS_PATH, "w") as f:
        json.dump([r.to_dict() for r in ranked], f, indent=2)

    logger.info(
        "Ranking complete: %d ranked clusters written to %s",
        len(ranked),
        cfg.RANKED_CLUSTERS_PATH,
    )
    return ranked


if __name__ == "__main__":
    run()
