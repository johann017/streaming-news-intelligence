"""Tests for the summarization service."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "RANKED_CLUSTERS_PATH", str(tmp_path / "ranked_clusters.json"))
    monkeypatch.setattr(cfg, "NORMALIZED_ARTICLES_PATH", str(tmp_path / "normalized_articles.json"))
    monkeypatch.setattr(cfg, "EVENTS_PATH", str(tmp_path / "events.json"))
    yield tmp_path


def _make_ranked_cluster(
    cluster_id: str = "clu1",
    article_ids: list[str] | None = None,
    score: float = 0.5,
    is_top_event: bool = False,
    title: str = "Global summit reaches climate agreement",
) -> "RankedCluster":
    ids = article_ids or ["a1"]
    from shared.models import RankedCluster
    return RankedCluster(
        cluster_id=cluster_id,
        article_ids=ids,
        representative_title=title,
        representative_url="https://example.com/1",
        representative_id=ids[0],
        sources=["rss"],
        created_at=FIXED_NOW,
        centroid_embedding=[],
        is_singleton=False,
        score=score,
        is_top_event=is_top_event,
    )


def _make_normalized(article_id: str = "a1", body: str = "Test body.") -> "NormalizedArticle":
    from shared.models import NormalizedArticle
    return NormalizedArticle(
        id=article_id,
        source="rss",
        url=f"https://example.com/{article_id}",
        title=f"Article {article_id}",
        cleaned_body=body,
        published_at=FIXED_NOW - timedelta(hours=1),
        fetched_at=FIXED_NOW,
        word_count=len(body.split()),
    )


# ---------------------------------------------------------------------------
# summarizer tests
# ---------------------------------------------------------------------------

def test_summarizer_uses_pipeline_when_available():
    import services.summarization.summarizer as mod
    mock_pipe = MagicMock(return_value=[{"summary_text": "A concise summary."}])

    # Reset cached state
    mod._pipeline = None
    mod._pipeline_failed = False

    with patch("services.summarization.summarizer._get_pipeline", return_value=mock_pipe):
        result = mod.summarize("Some long article text here.")

    assert result == "A concise summary."


def test_summarizer_falls_back_to_extractive():
    import services.summarization.summarizer as mod
    mod._pipeline = None
    mod._pipeline_failed = False

    with patch("services.summarization.summarizer._get_pipeline", return_value=None):
        result = mod.summarize("First sentence. Second sentence. Third sentence.")

    assert "First sentence" in result


def test_summarizer_handles_empty_text():
    import services.summarization.summarizer as mod
    with patch("services.summarization.summarizer._get_pipeline", return_value=None):
        result = mod.summarize("")
    assert result == ""


def test_extractive_summary_returns_n_sentences():
    from services.summarization.summarizer import _extractive_summary
    text = "Sentence one. Sentence two. Sentence three. Sentence four."
    result = _extractive_summary(text, num_sentences=2)
    assert "Sentence one" in result
    assert "Sentence three" not in result


# ---------------------------------------------------------------------------
# key_points tests
# ---------------------------------------------------------------------------

def test_extract_key_points_splits_summary():
    from services.summarization.key_points import extract_key_points
    summary = "Leaders met in Geneva. They agreed on targets. Implementation starts next year."
    points = extract_key_points(summary, max_points=3)
    assert len(points) == 3
    assert any("Geneva" in p for p in points)


def test_extract_key_points_respects_max():
    from services.summarization.key_points import extract_key_points
    summary = "Point one. Point two. Point three. Point four. Point five."
    points = extract_key_points(summary, max_points=2)
    assert len(points) == 2


def test_extract_key_points_handles_empty():
    from services.summarization.key_points import extract_key_points
    assert extract_key_points("") == []


# ---------------------------------------------------------------------------
# event_builder tests
# ---------------------------------------------------------------------------

def test_extract_geo_tags_finds_country():
    from services.summarization.event_builder import extract_geo_tags
    tags = extract_geo_tags("Leaders in France and Germany met today.")
    assert "France" in tags
    assert "Germany" in tags


def test_extract_geo_tags_finds_city():
    from services.summarization.event_builder import extract_geo_tags
    tags = extract_geo_tags("The summit was held in Paris.")
    assert "Paris" in tags


def test_extract_geo_tags_no_match():
    from services.summarization.event_builder import extract_geo_tags
    tags = extract_geo_tags("Scientists discovered a new method.")
    assert tags == []


def test_build_event_sets_ttl_for_top_event(monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "TOP_EVENT_TTL_HOURS", 72)
    monkeypatch.setattr(cfg, "EVENT_TTL_HOURS", 48)

    from services.summarization.event_builder import build_event

    cluster = _make_ranked_cluster(is_top_event=True)
    articles = {"a1": _make_normalized("a1")}

    with patch("services.summarization.event_builder.utcnow", return_value=FIXED_NOW):
        event = build_event(cluster, "Some summary.", ["Point 1."], articles)

    expected_expires = FIXED_NOW + timedelta(hours=72)
    assert event.expires_at == expected_expires


def test_build_event_sets_ttl_for_normal_event(monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "TOP_EVENT_TTL_HOURS", 72)
    monkeypatch.setattr(cfg, "EVENT_TTL_HOURS", 48)

    from services.summarization.event_builder import build_event

    cluster = _make_ranked_cluster(is_top_event=False)
    articles = {"a1": _make_normalized("a1")}

    with patch("services.summarization.event_builder.utcnow", return_value=FIXED_NOW):
        event = build_event(cluster, "Some summary.", [], articles)

    expected_expires = FIXED_NOW + timedelta(hours=48)
    assert event.expires_at == expected_expires


def test_build_event_limits_sources_to_five():
    from services.summarization.event_builder import build_event

    cluster = _make_ranked_cluster(
        article_ids=[f"a{i}" for i in range(10)]
    )
    articles = {f"a{i}": _make_normalized(f"a{i}") for i in range(10)}

    with patch("services.summarization.event_builder.utcnow", return_value=FIXED_NOW):
        event = build_event(cluster, "Summary.", [], articles)

    assert len(event.sources) <= 5


def test_build_event_uses_title_as_fallback_summary():
    from services.summarization.event_builder import build_event

    cluster = _make_ranked_cluster(title="Important world event happens")
    articles: dict = {}

    with patch("services.summarization.event_builder.utcnow", return_value=FIXED_NOW):
        event = build_event(cluster, "", [], articles)

    assert event.summary == "Important world event happens"


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

def test_summarization_run_produces_events(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "MIN_SCORE_FOR_SUMMARIZATION", 0.3)

    from services.summarization import run as summ_run
    import services.summarization.summarizer as mod

    # One above threshold, one below
    clusters = [
        _make_ranked_cluster("c1", score=0.8, is_top_event=True, article_ids=["a1"]),
        _make_ranked_cluster("c2", score=0.1, is_top_event=False, article_ids=["a2"]),
    ]
    articles = [
        _make_normalized("a1", body="Scientists discovered breakthrough technology in renewable energy."),
        _make_normalized("a2", body="Short article."),
    ]

    (tmp_data_dir / "ranked_clusters.json").write_text(
        json.dumps([c.to_dict() for c in clusters])
    )
    (tmp_data_dir / "normalized_articles.json").write_text(
        json.dumps([a.to_dict() for a in articles])
    )

    mod._pipeline = None
    mod._pipeline_failed = False

    mock_pipe = MagicMock(return_value=[{"summary_text": "Scientists made a major discovery."}])

    with patch("services.summarization.summarizer._get_pipeline", return_value=mock_pipe):
        with patch("services.summarization.event_builder.utcnow", return_value=FIXED_NOW):
            result = summ_run.run()

    assert len(result) == 2
    assert (tmp_data_dir / "events.json").exists()


def test_summarization_run_handles_missing_input(tmp_data_dir):
    from services.summarization import run as summ_run
    result = summ_run.run()
    assert result == []
