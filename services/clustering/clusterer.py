"""
DBSCAN-based clustering of article embeddings using cosine distance.
Returns lists of article-ID groups (one group per cluster).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize as l2_normalize

import shared.config as cfg
from shared.utils import get_logger

logger = get_logger(__name__)


def cluster_embeddings(
    embeddings: dict[str, np.ndarray],
) -> list[list[str]]:
    """
    Cluster article embeddings using DBSCAN with cosine distance.

    Returns a list of groups, where each group is a list of article IDs.
    Noise points (DBSCAN label = -1) are returned as singleton groups.

    Args:
        embeddings: dict mapping article_id → embedding vector

    Returns:
        List of article-ID groups (each group = one candidate cluster)
    """
    if not embeddings:
        return []

    ids = list(embeddings.keys())
    matrix = np.stack([embeddings[aid] for aid in ids])  # shape (N, D)

    if len(ids) == 1:
        # Nothing to cluster
        return [[ids[0]]]

    # L2-normalise so that Euclidean distance ≈ cosine distance
    # (cosine_distance = 1 - cosine_similarity = ||a-b||^2 / 2 for unit vectors)
    matrix_norm = l2_normalize(matrix, norm="l2")

    # DBSCAN with Euclidean distance on L2-normalised vectors.
    # eps=0.25 in cosine distance ≈ eps=sqrt(2*0.25)≈0.707 in L2 on unit sphere,
    # but we use the configured DBSCAN_EPS directly on L2 distance here.
    db = DBSCAN(
        eps=cfg.DBSCAN_EPS,
        min_samples=cfg.DBSCAN_MIN_SAMPLES,
        metric="cosine",
        algorithm="brute",
        n_jobs=1,
    )
    labels = db.fit_predict(matrix_norm)

    # Group article IDs by cluster label
    groups: dict[int, list[str]] = {}
    for article_id, label in zip(ids, labels):
        groups.setdefault(label, []).append(article_id)

    result: list[list[str]] = []
    noise_count = 0
    for label, group_ids in groups.items():
        result.append(group_ids)
        if label == -1:
            noise_count += len(group_ids)

    n_clusters = len([l for l in groups if l != -1])
    logger.info(
        "Clustering: %d articles → %d clusters, %d noise (singleton) points",
        len(ids),
        n_clusters,
        noise_count,
    )

    return result
