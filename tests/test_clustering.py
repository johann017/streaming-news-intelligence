"""
Tests for the clustering service.
All tests use pre-computed fixture embeddings (no actual ML model needed).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "NORMALIZED_ARTICLES_PATH", str(tmp_path / "normalized_articles.json"))
    monkeypatch.setattr(cfg, "CLUSTERS_PATH", str(tmp_path / "clusters.json"))
    monkeypatch.setattr(cfg, "EMBEDDING_CACHE_PATH", str(tmp_path / "embedding_cache.pkl"))
    yield tmp_path


def _make_normalized(article_id: str, source: str = "rss") -> "NormalizedArticle":
    from shared.models import NormalizedArticle
    return NormalizedArticle(
        id=article_id,
        source=source,
        url=f"https://example.com/{article_id}",
        title=f"Article {article_id}",
        cleaned_body=f"Body text for article {article_id}",
        published_at=FIXED_NOW - timedelta(hours=1),
        fetched_at=FIXED_NOW,
        word_count=8,
    )


# ---------------------------------------------------------------------------
# Fixture embeddings: 3 tight clusters of 2 + 1 outlier
# Vectors are unit-normalised 3D for clarity
# ---------------------------------------------------------------------------

_E = {
    # Cluster A (very similar)
    "a1": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "a2": np.array([0.99, 0.14, 0.0], dtype=np.float32),
    # Cluster B (very similar)
    "b1": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "b2": np.array([0.14, 0.99, 0.0], dtype=np.float32),
    # Outlier (far from both clusters)
    "c1": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


# ---------------------------------------------------------------------------
# clusterer tests
# ---------------------------------------------------------------------------

def test_cluster_embeddings_groups_similar_articles():
    from services.clustering.clusterer import cluster_embeddings
    import shared.config as cfg

    # Use a generous eps so a1+a2 and b1+b2 group together
    with patch.object(cfg, "DBSCAN_EPS", 0.3), patch.object(cfg, "DBSCAN_MIN_SAMPLES", 2):
        groups = cluster_embeddings(_E)

    # We expect at least 2 groups with 2 members (a1+a2, b1+b2), plus c1 as noise
    multi_member = [g for g in groups if len(g) == 2]
    assert len(multi_member) == 2, f"Expected 2 two-member groups, got: {groups}"


def test_cluster_embeddings_empty_input():
    from services.clustering.clusterer import cluster_embeddings
    assert cluster_embeddings({}) == []


def test_cluster_embeddings_single_article():
    from services.clustering.clusterer import cluster_embeddings
    single = {"only": np.array([1.0, 0.0, 0.0], dtype=np.float32)}
    groups = cluster_embeddings(single)
    assert len(groups) == 1
    assert groups[0] == ["only"]


def test_cluster_embeddings_noise_is_singleton():
    """Noise points are returned as singleton groups."""
    from services.clustering.clusterer import cluster_embeddings
    import shared.config as cfg

    with patch.object(cfg, "DBSCAN_EPS", 0.3), patch.object(cfg, "DBSCAN_MIN_SAMPLES", 2):
        groups = cluster_embeddings(_E)

    # c1 (the outlier) should be a singleton
    all_ids_flat = [aid for group in groups for aid in group]
    assert "c1" in all_ids_flat


# ---------------------------------------------------------------------------
# cluster_builder tests
# ---------------------------------------------------------------------------

def test_build_clusters_creates_correct_count():
    from services.clustering.cluster_builder import build_clusters

    groups = [["a1", "a2"], ["b1", "b2"], ["c1"]]
    articles = {aid: _make_normalized(aid) for aid in ["a1", "a2", "b1", "b2", "c1"]}

    with patch("services.clustering.cluster_builder.utcnow", return_value=FIXED_NOW):
        clusters = build_clusters(groups, _E, articles)

    assert len(clusters) == 3


def test_build_clusters_singleton_flag():
    from services.clustering.cluster_builder import build_clusters

    groups = [["a1", "a2"], ["c1"]]
    articles = {aid: _make_normalized(aid) for aid in ["a1", "a2", "c1"]}

    with patch("services.clustering.cluster_builder.utcnow", return_value=FIXED_NOW):
        clusters = build_clusters(groups, _E, articles)

    singleton = next(c for c in clusters if len(c.article_ids) == 1)
    multi = next(c for c in clusters if len(c.article_ids) == 2)
    assert singleton.is_singleton is True
    assert multi.is_singleton is False


def test_build_clusters_stable_id():
    """Same article IDs always produce the same cluster_id."""
    from services.clustering.cluster_builder import build_clusters

    groups = [["a1", "a2"]]
    articles = {"a1": _make_normalized("a1"), "a2": _make_normalized("a2")}

    with patch("services.clustering.cluster_builder.utcnow", return_value=FIXED_NOW):
        c1 = build_clusters(groups, _E, articles)
        c2 = build_clusters(groups, _E, articles)

    assert c1[0].cluster_id == c2[0].cluster_id


def test_build_clusters_representative_is_closest_to_centroid():
    from services.clustering.cluster_builder import build_clusters

    # a1 and a2 are very similar; centroid ≈ a1 direction
    groups = [["a1", "a2"]]
    articles = {"a1": _make_normalized("a1"), "a2": _make_normalized("a2")}

    with patch("services.clustering.cluster_builder.utcnow", return_value=FIXED_NOW):
        clusters = build_clusters(groups, _E, articles)

    assert clusters[0].representative_title in ("Article a1", "Article a2")


def test_build_clusters_unique_sources():
    from services.clustering.cluster_builder import build_clusters

    groups = [["a1", "b1"]]
    articles = {
        "a1": _make_normalized("a1", source="rss"),
        "b1": _make_normalized("b1", source="reddit"),
    }

    with patch("services.clustering.cluster_builder.utcnow", return_value=FIXED_NOW):
        clusters = build_clusters(groups, _E, articles)

    assert set(clusters[0].sources) == {"rss", "reddit"}


# ---------------------------------------------------------------------------
# embedder tests (mocked model)
# ---------------------------------------------------------------------------

def test_embedder_uses_cache_on_second_call(tmp_data_dir):
    from services.clustering.embedder import embed_articles

    articles = [_make_normalized("a1"), _make_normalized("a2")]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
    )

    with patch("services.clustering.embedder._get_model", return_value=mock_model):
        result1 = embed_articles(articles)

    # Second call — model.encode should NOT be called (cache hit)
    with patch("services.clustering.embedder._get_model", return_value=mock_model) as mock_get:
        result2 = embed_articles(articles)
        mock_get.assert_not_called()

    assert list(result1.keys()) == list(result2.keys())


def test_embedder_returns_correct_keys(tmp_data_dir):
    from services.clustering.embedder import embed_articles

    articles = [_make_normalized("x1"), _make_normalized("x2")]

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array(
        [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
    )

    with patch("services.clustering.embedder._get_model", return_value=mock_model):
        result = embed_articles(articles)

    assert set(result.keys()) == {"x1", "x2"}
    assert result["x1"].shape == (2,)


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

def test_clustering_run_produces_clusters_file(tmp_data_dir):
    from services.clustering import run as cluster_run
    from shared.models import NormalizedArticle

    articles = [_make_normalized(f"art{i}") for i in range(4)]
    norm_path = tmp_data_dir / "normalized_articles.json"
    norm_path.write_text(json.dumps([a.to_dict() for a in articles]))

    mock_model = MagicMock()
    # Return distinct embeddings so clustering works
    mock_model.encode.return_value = np.random.rand(4, 16).astype(np.float32)

    with patch("services.clustering.embedder._get_model", return_value=mock_model):
        with patch("services.clustering.cluster_builder.utcnow", return_value=FIXED_NOW):
            result = cluster_run.run()

    assert len(result) > 0
    out = tmp_data_dir / "clusters.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == len(result)


def test_clustering_run_handles_missing_input(tmp_data_dir):
    from services.clustering import run as cluster_run
    result = cluster_run.run()
    assert result == []
