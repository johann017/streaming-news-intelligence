"""
Constructs Cluster objects from article groups produced by the clusterer.
Picks the representative article (closest to centroid) and computes cluster ID.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import numpy as np

from shared.models import Cluster, NormalizedArticle
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)


def _cluster_id_from_article_ids(article_ids: list[str]) -> str:
    """Stable cluster ID: SHA-256 of sorted article IDs joined by '|'."""
    key = "|".join(sorted(article_ids))
    return hashlib.sha256(key.encode()).hexdigest()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_clusters(
    article_groups: list[list[str]],
    embeddings: dict[str, np.ndarray],
    articles_by_id: dict[str, NormalizedArticle],
    dbscan_labels: list[int] | None = None,
) -> list[Cluster]:
    """
    Build Cluster objects from article groups.

    Args:
        article_groups: output of cluster_embeddings()
        embeddings: dict of article_id → embedding vector
        articles_by_id: dict of article_id → NormalizedArticle
        dbscan_labels: optional DBSCAN labels (used to tag noise as singletons)

    Returns:
        List of Cluster objects
    """
    clusters: list[Cluster] = []
    now = utcnow()

    for group_ids in article_groups:
        if not group_ids:
            continue

        is_singleton = len(group_ids) == 1

        # Compute centroid
        vecs = np.stack([embeddings[aid] for aid in group_ids if aid in embeddings])
        centroid = vecs.mean(axis=0)

        # Pick representative: article whose embedding is closest to centroid
        best_id = group_ids[0]
        best_sim = -1.0
        for aid in group_ids:
            if aid not in embeddings:
                continue
            sim = _cosine_similarity(embeddings[aid], centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = aid

        rep_article = articles_by_id.get(best_id)
        if rep_article is None:
            continue

        unique_sources = list({articles_by_id[aid].source for aid in group_ids if aid in articles_by_id})

        cluster = Cluster(
            cluster_id=_cluster_id_from_article_ids(group_ids),
            article_ids=group_ids,
            representative_title=rep_article.title,
            representative_url=rep_article.url,
            sources=unique_sources,
            created_at=now,
            centroid_embedding=centroid.tolist(),
            is_singleton=is_singleton,
        )
        clusters.append(cluster)

    logger.info("Built %d clusters (%d non-singleton)", len(clusters),
                sum(1 for c in clusters if not c.is_singleton))
    return clusters
