"""
Ranker: sorts scored clusters by importance and selects the top-N.
Flags the top TOP_EVENT_COUNT as `is_top_event`.
"""
from __future__ import annotations

import shared.config as cfg
from shared.models import Cluster, NormalizedArticle, RankedCluster
from shared.utils import get_logger
from services.ranking.scorer import score_cluster

logger = get_logger(__name__)


def rank_clusters(
    clusters: list[Cluster],
    articles_by_id: dict[str, NormalizedArticle],
) -> list[RankedCluster]:
    """
    Score all clusters, sort descending by score, keep top TOP_CLUSTERS_KEPT,
    and flag top TOP_EVENT_COUNT as is_top_event.

    Returns a list of RankedCluster sorted by score descending.
    """
    scored: list[tuple[float, dict, Cluster]] = []
    for cluster in clusters:
        score, breakdown = score_cluster(cluster, articles_by_id)
        reddit_engagement = sum(
            articles_by_id[aid].reddit_score
            for aid in cluster.article_ids
            if aid in articles_by_id
        )
        scored.append((score, breakdown, cluster))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Take top-N
    top = scored[: cfg.TOP_CLUSTERS_KEPT]

    ranked: list[RankedCluster] = []
    for rank_idx, (score, breakdown, cluster) in enumerate(top):
        is_top = rank_idx < cfg.TOP_EVENT_COUNT
        reddit_engagement = sum(
            articles_by_id[aid].reddit_score
            for aid in cluster.article_ids
            if aid in articles_by_id
        )
        ranked.append(
            RankedCluster(
                cluster_id=cluster.cluster_id,
                article_ids=cluster.article_ids,
                representative_title=cluster.representative_title,
                representative_url=cluster.representative_url,
                sources=cluster.sources,
                created_at=cluster.created_at,
                centroid_embedding=cluster.centroid_embedding,
                is_singleton=cluster.is_singleton,
                score=score,
                score_breakdown=breakdown,
                is_top_event=is_top,
                reddit_engagement=reddit_engagement,
            )
        )

    n_top = sum(1 for r in ranked if r.is_top_event)
    logger.info(
        "Ranking: %d clusters → top %d kept, %d flagged as top events",
        len(clusters),
        len(ranked),
        n_top,
    )
    return ranked
