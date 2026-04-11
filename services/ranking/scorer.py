"""
Cluster scorer: computes a composite importance score for each cluster.

Score components:
  source_diversity  — unique source count / 3 (capped at 1.0)
  article_count     — log-normalised count (diminishing returns)
  recency           — exponential decay from the newest article's published_at

Weights are configured in shared/config.py.
"""
from __future__ import annotations

import math
from datetime import timezone

import shared.config as cfg
from shared.models import Cluster, NormalizedArticle
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)


def _source_diversity_score(cluster: Cluster) -> float:
    """
    Score based on absolute number of unique sources.
    1 source → 0.33, 2 sources → 0.67, 3+ sources → 1.0.

    Using a ratio (unique/total) was wrong: a singleton always scored 1.0
    because 1 unique / 1 article = 100%. This made every fresh article
    look equally important regardless of how many outlets covered it.
    """
    if not cluster.article_ids:
        return 0.0
    unique_sources = len(set(cluster.sources))
    return min(unique_sources / 3.0, 1.0)


def _article_count_score(cluster: Cluster) -> float:
    """Log-normalised article count. log2(1)=0, log2(2)=1, log2(10)≈3.3 → capped at 1.0."""
    count = len(cluster.article_ids)
    if count <= 1:
        return 0.0
    return min(math.log2(count) / 5.0, 1.0)  # 2^5=32 articles → score=1.0


def _recency_score(
    cluster: Cluster,
    articles_by_id: dict[str, NormalizedArticle],
) -> float:
    """
    Exponential decay based on the newest article's published_at.
    score = exp(-age_hours * ln(2) / half_life)

    A cluster whose newest article was just published scores 1.0; one whose
    newest article is RECENCY_HALF_LIFE_HOURS old scores ~0.5.

    Using published_at rather than cluster.created_at is correct: cluster
    objects are rebuilt from scratch each run, so created_at is always
    approximately 'now' and gives no useful signal.
    """
    now = utcnow()
    timestamps = [
        articles_by_id[aid].published_at
        for aid in cluster.article_ids
        if aid in articles_by_id
    ]
    if not timestamps:
        return 0.5  # no article data available — neutral score
    newest = max(timestamps)
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    age_hours = max((now - newest).total_seconds() / 3600.0, 0.0)
    half_life = max(cfg.RECENCY_HALF_LIFE_HOURS, 0.1)
    return math.exp(-age_hours * math.log(2) / half_life)


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
        "article_count": cfg.SCORE_WEIGHT_ARTICLE_COUNT,
        "recency": cfg.SCORE_WEIGHT_RECENCY,
    }

    components = {
        "source_diversity": _source_diversity_score(cluster),
        "article_count": _article_count_score(cluster),
        "recency": _recency_score(cluster, articles_by_id),
    }

    # Penalise singleton clusters (noise from DBSCAN) with a 0.8 multiplier.
    # 0.5 was too aggressive — it suppressed real breaking news that hadn't
    # yet been picked up by multiple sources.
    singleton_penalty = 0.8 if cluster.is_singleton else 1.0

    composite = sum(w[k] * v for k, v in components.items()) * singleton_penalty
    return round(composite, 4), components
