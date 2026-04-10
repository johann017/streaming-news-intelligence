"""Tests for the ranking service (scorer + ranker)."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
RECENT = FIXED_NOW - timedelta(minutes=30)
OLD = FIXED_NOW - timedelta(hours=12)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "CLUSTERS_PATH", str(tmp_path / "clusters.json"))
    monkeypatch.setattr(cfg, "RANKED_CLUSTERS_PATH", str(tmp_path / "ranked_clusters.json"))
    monkeypatch.setattr(cfg, "NORMALIZED_ARTICLES_PATH", str(tmp_path / "normalized_articles.json"))
    yield tmp_path


def _make_cluster(
    cluster_id: str = "clu1",
    article_ids: list[str] | None = None,
    sources: list[str] | None = None,
    created_at: datetime = RECENT,
    is_singleton: bool = False,
) -> "Cluster":
    from shared.models import Cluster
    return Cluster(
        cluster_id=cluster_id,
        article_ids=article_ids or ["a1"],
        representative_title="Test headline",
        representative_url="https://example.com/1",
        sources=sources or ["rss"],
        created_at=created_at,
        centroid_embedding=[],
        is_singleton=is_singleton,
    )


def _make_article(
    article_id: str = "a1",
    source: str = "rss",
    reddit_score: int = 0,
) -> "NormalizedArticle":
    from shared.models import NormalizedArticle
    return NormalizedArticle(
        id=article_id,
        source=source,
        url=f"https://example.com/{article_id}",
        title=f"Article {article_id}",
        cleaned_body="body text",
        published_at=RECENT,
        fetched_at=FIXED_NOW,
        word_count=10,
        reddit_score=reddit_score,
    )


# ---------------------------------------------------------------------------
# scorer tests
# ---------------------------------------------------------------------------

def test_recency_score_fresh_cluster():
    from services.ranking.scorer import _recency_score
    cluster = _make_cluster(created_at=FIXED_NOW)
    with patch("services.ranking.scorer.utcnow", return_value=FIXED_NOW):
        score = _recency_score(cluster)
    assert score == pytest.approx(1.0, abs=0.01)


def test_recency_score_old_cluster():
    from services.ranking.scorer import _recency_score
    cluster = _make_cluster(created_at=OLD)
    with patch("services.ranking.scorer.utcnow", return_value=FIXED_NOW):
        score = _recency_score(cluster)
    # 12h old with 6h half-life → exp(-12 * ln2 / 6) = exp(-2*ln2) ≈ 0.25
    assert score == pytest.approx(0.25, abs=0.02)


def test_source_diversity_single_source():
    from services.ranking.scorer import _source_diversity_score
    cluster = _make_cluster(article_ids=["a1", "a2"], sources=["rss", "rss"])
    score = _source_diversity_score(cluster)
    assert score == pytest.approx(0.5)  # 1 unique / 2 articles


def test_source_diversity_multiple_sources():
    from services.ranking.scorer import _source_diversity_score
    cluster = _make_cluster(article_ids=["a1", "a2"], sources=["rss", "reddit"])
    score = _source_diversity_score(cluster)
    assert score == pytest.approx(1.0)  # 2 unique / 2 articles = 1.0


def test_article_count_score_one_article():
    from services.ranking.scorer import _article_count_score
    cluster = _make_cluster(article_ids=["a1"])
    assert _article_count_score(cluster) == 0.0


def test_article_count_score_multiple():
    from services.ranking.scorer import _article_count_score
    cluster = _make_cluster(article_ids=["a1", "a2"])
    assert _article_count_score(cluster) > 0.0


def test_reddit_engagement_score_zero():
    from services.ranking.scorer import _reddit_engagement_score
    cluster = _make_cluster(article_ids=["a1"])
    articles = {"a1": _make_article(reddit_score=0)}
    assert _reddit_engagement_score(cluster, articles) == 0.0


def test_reddit_engagement_score_positive():
    from services.ranking.scorer import _reddit_engagement_score
    cluster = _make_cluster(article_ids=["a1"])
    articles = {"a1": _make_article(source="reddit", reddit_score=1000)}
    score = _reddit_engagement_score(cluster, articles)
    assert score > 0.0


def test_singleton_penalty_applied():
    from services.ranking.scorer import score_cluster
    cluster_single = _make_cluster(is_singleton=True)
    cluster_multi = _make_cluster(is_singleton=False)
    articles: dict = {}

    with patch("services.ranking.scorer.utcnow", return_value=FIXED_NOW):
        score_s, _ = score_cluster(cluster_single, articles)
        score_m, _ = score_cluster(cluster_multi, articles)

    assert score_s < score_m  # singleton is penalised


# ---------------------------------------------------------------------------
# ranker tests
# ---------------------------------------------------------------------------

def test_ranker_sorts_by_score():
    from services.ranking.ranker import rank_clusters

    # Cluster A: multi-source, recent → higher score
    # Cluster B: single-source, old → lower score
    cluster_a = _make_cluster("A", article_ids=["a1", "a2"], sources=["rss", "reddit"], created_at=RECENT)
    cluster_b = _make_cluster("B", article_ids=["b1"], sources=["rss"], created_at=OLD, is_singleton=True)
    articles = {
        "a1": _make_article("a1", source="rss"),
        "a2": _make_article("a2", source="reddit", reddit_score=500),
        "b1": _make_article("b1"),
    }

    with patch("services.ranking.scorer.utcnow", return_value=FIXED_NOW):
        ranked = rank_clusters([cluster_b, cluster_a], articles)

    assert ranked[0].cluster_id == "A"
    assert ranked[1].cluster_id == "B"


def test_ranker_flags_top_events(monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "TOP_CLUSTERS_KEPT", 5)
    monkeypatch.setattr(cfg, "TOP_EVENT_COUNT", 2)

    from services.ranking.ranker import rank_clusters

    clusters = [
        _make_cluster(f"c{i}", article_ids=[f"a{i}"], sources=["rss"], created_at=RECENT)
        for i in range(5)
    ]
    articles = {f"a{i}": _make_article(f"a{i}") for i in range(5)}

    with patch("services.ranking.scorer.utcnow", return_value=FIXED_NOW):
        ranked = rank_clusters(clusters, articles)

    top_event_count = sum(1 for r in ranked if r.is_top_event)
    assert top_event_count == 2


def test_ranker_respects_top_clusters_kept(monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "TOP_CLUSTERS_KEPT", 3)
    monkeypatch.setattr(cfg, "TOP_EVENT_COUNT", 1)

    from services.ranking.ranker import rank_clusters

    clusters = [
        _make_cluster(f"c{i}", article_ids=[f"a{i}"], sources=["rss"], created_at=RECENT)
        for i in range(10)
    ]
    articles = {f"a{i}": _make_article(f"a{i}") for i in range(10)}

    with patch("services.ranking.scorer.utcnow", return_value=FIXED_NOW):
        ranked = rank_clusters(clusters, articles)

    assert len(ranked) == 3


def test_ranker_handles_empty_input():
    from services.ranking.ranker import rank_clusters
    result = rank_clusters([], {})
    assert result == []


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

def test_ranking_run_produces_output(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "TOP_CLUSTERS_KEPT", 5)
    monkeypatch.setattr(cfg, "TOP_EVENT_COUNT", 2)

    from services.ranking import run as ranking_run

    clusters = [
        _make_cluster(f"c{i}", article_ids=[f"a{i}"], sources=["rss"], created_at=RECENT)
        for i in range(3)
    ]
    articles = [_make_article(f"a{i}") for i in range(3)]

    clusters_path = tmp_data_dir / "clusters.json"
    clusters_path.write_text(json.dumps([c.to_dict() for c in clusters]))
    norm_path = tmp_data_dir / "normalized_articles.json"
    norm_path.write_text(json.dumps([a.to_dict() for a in articles]))

    with patch("services.ranking.scorer.utcnow", return_value=FIXED_NOW):
        result = ranking_run.run()

    assert len(result) == 3
    out = tmp_data_dir / "ranked_clusters.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data[0]["score"] >= data[-1]["score"]  # sorted descending


def test_ranking_run_handles_missing_clusters(tmp_data_dir):
    from services.ranking import run as ranking_run
    result = ranking_run.run()
    assert result == []
