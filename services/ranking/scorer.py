"""
Cluster scorer: computes a composite importance score for each cluster.

Score components:
  source_diversity  — unique source count / article count (0–1)
  recency           — exponential decay from cluster creation time
  article_count     — log-normalised count (diminishing returns)
  reddit_engagement — log-normalised sum of Reddit upvotes

Weights are configured in shared/config.py.
"""
from __future__ import annotations

import math
from datetime import timedelta

import shared.config as cfg
from shared.models import Cluster, NormalizedArticle
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)


def _source_diversity_score(cluster: Cluster) -> float:
    """Ratio of unique sources to total articles. Ranges 0–1."""
    if not cluster.article_ids:
        return 0.0
    return min(len(set(cluster.sources)) / len(cluster.article_ids), 1.0)


def _recency_score(cluster: Cluster) -> float:
    """
    Exponential decay: score = exp(-t / half_life) where t is age in hours.
    A cluster created now scores 1.0; one created RECENCY_HALF_LIFE_HOURS ago scores ~0.5.
    """
    now = utcnow()
    created = cluster.created_at
    if created.tzinfo is None:
        from datetime import timezone
        created = created.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        from datetime import timezone
        now = now.replace(tzinfo=timezone.utc)

    age_hours = max((now - created).total_seconds() / 3600.0, 0.0)
    half_life = max(cfg.RECENCY_HALF_LIFE_HOURS, 0.1)
    return math.exp(-age_hours * math.log(2) / half_life)


def _article_count_score(cluster: Cluster) -> float:
    """Log-normalised article count. log2(1)=0, log2(2)=1, log2(10)≈3.3 → capped at 1.0."""
    count = len(cluster.article_ids)
    if count <= 1:
        return 0.0
    return min(math.log2(count) / 5.0, 1.0)  # 2^5=32 articles → score=1.0


def _reddit_engagement_score(
    cluster: Cluster,
    articles_by_id: dict[str, NormalizedArticle],
) -> float:
    """Log-normalised total Reddit upvotes across all cluster articles."""
    total_score = sum(
        articles_by_id[aid].reddit_score
        for aid in cluster.article_ids
        if aid in articles_by_id
    )
    if total_score <= 0:
        return 0.0
    # log10(100)=2, log10(10000)=4; normalise to cap at ~1.0 for 10k upvotes
    return min(math.log10(total_score + 1) / 4.0, 1.0)


def score_cluster(
    cluster: Cluster,
    articles_by_id: dict[str, NormalizedArticle],
) -> tuple[float, dict[str, float]]:
    """
    Compute the composite importance score for a cluster.

    Returns:
        (score, breakdown) where breakdown maps component name → component score
    """
    w = {
        "source_diversity": cfg.SCORE_WEIGHT_SOURCE_DIVERSITY,
        "recency": cfg.SCORE_WEIGHT_RECENCY,
        "article_count": cfg.SCORE_WEIGHT_ARTICLE_COUNT,
        "reddit_engagement": cfg.SCORE_WEIGHT_REDDIT,
    }

    components = {
        "source_diversity": _source_diversity_score(cluster),
        "recency": _recency_score(cluster),
        "article_count": _article_count_score(cluster),
        "reddit_engagement": _reddit_engagement_score(cluster, articles_by_id),
    }

    # Penalise singleton clusters (noise from DBSCAN) with a 0.8 multiplier.
    # 0.5 was too aggressive — it suppressed real breaking news that hadn't
    # yet been picked up by multiple sources.
    singleton_penalty = 0.8 if cluster.is_singleton else 1.0

    composite = sum(w[k] * v for k, v in components.items()) * singleton_penalty
    return round(composite, 4), components
