"""Tests for the processing service (deduplication, filters, normalization)."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

# Body with 35 words — passes MIN_BODY_WORDS=5
_LONG_BODY = (
    "Scientists have discovered a new approach to renewable energy that could "
    "significantly reduce carbon emissions worldwide over the next two decades "
    "according to researchers at several leading universities around the globe."
)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "SEEN_IDS_PATH", str(tmp_path / "seen_ids.json"))
    monkeypatch.setattr(cfg, "RAW_ARTICLES_PATH", str(tmp_path / "raw_articles.json"))
    monkeypatch.setattr(cfg, "NORMALIZED_ARTICLES_PATH", str(tmp_path / "normalized_articles.json"))
    yield tmp_path


def _make_raw(
    article_id: str = "abc123",
    source: str = "rss",
    title: str = "World leaders meet in Geneva",
    body: str = _LONG_BODY,
    hours_ago: float = 1.0,
) -> "RawArticle":
    from shared.models import RawArticle
    published = FIXED_NOW - timedelta(hours=hours_ago)
    return RawArticle(
        id=article_id,
        source=source,
        url=f"https://example.com/{article_id}",
        title=title,
        body=body,
        published_at=published,
        fetched_at=FIXED_NOW,
        raw_metadata={},
    )


# ---------------------------------------------------------------------------
# deduplicator tests
# ---------------------------------------------------------------------------

def test_deduplicator_passes_new_articles(tmp_data_dir):
    from services.processing.deduplicator import deduplicate
    articles = [_make_raw("id1"), _make_raw("id2")]
    result = deduplicate(articles)
    assert len(result) == 2


def test_deduplicator_drops_seen_on_second_run(tmp_data_dir):
    from services.processing.deduplicator import deduplicate
    articles = [_make_raw("id1"), _make_raw("id2")]
    deduplicate(articles)           # first run: both new
    result = deduplicate(articles)  # second run: both seen
    assert len(result) == 0


def test_deduplicator_partial_overlap(tmp_data_dir):
    from services.processing.deduplicator import deduplicate
    deduplicate([_make_raw("id1")])
    result = deduplicate([_make_raw("id1"), _make_raw("id2")])
    assert len(result) == 1
    assert result[0].id == "id2"


def test_deduplicator_idempotent_starting_from_empty(tmp_data_dir):
    """Same input from a clean state produces the same output."""
    from services.processing.deduplicator import deduplicate
    articles = [_make_raw("id1"), _make_raw("id2"), _make_raw("id3")]

    r1 = deduplicate(articles)

    seen_path = tmp_data_dir / "seen_ids.json"
    seen_path.unlink(missing_ok=True)

    r2 = deduplicate(articles)
    assert [a.id for a in r1] == [a.id for a in r2]


def test_deduplicator_fifo_eviction(tmp_data_dir, monkeypatch):
    """When cap is hit, oldest entries are evicted and re-ingest is possible."""
    import shared.config as cfg
    monkeypatch.setattr(cfg, "SEEN_IDS_MAX_SIZE", 3)

    from services.processing.deduplicator import deduplicate
    deduplicate([_make_raw("id1"), _make_raw("id2"), _make_raw("id3")])
    deduplicate([_make_raw("id4")])
    result = deduplicate([_make_raw("id1")])
    assert len(result) == 1


# ---------------------------------------------------------------------------
# filter tests
# ---------------------------------------------------------------------------

def test_is_recent_passes_fresh_article():
    from services.processing.filters import is_recent
    with patch("services.processing.filters.utcnow", return_value=FIXED_NOW):
        assert is_recent(_make_raw(hours_ago=1.0)) is True


def test_is_recent_drops_old_article():
    from services.processing.filters import is_recent
    with patch("services.processing.filters.utcnow", return_value=FIXED_NOW):
        assert is_recent(_make_raw(hours_ago=48.0)) is False


def test_has_sufficient_body_passes():
    from services.processing.filters import has_sufficient_body
    assert has_sufficient_body(_make_raw()) is True


def test_has_sufficient_body_drops_short(monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "MIN_BODY_WORDS", 100)
    from services.processing.filters import has_sufficient_body
    assert has_sufficient_body(_make_raw()) is False


def test_is_english_passes_english():
    from services.processing.filters import is_english
    article = _make_raw(body=" ".join(["word"] * 40))
    with patch("services.processing.filters.detect", return_value="en"):
        assert is_english(article) is True


def test_is_english_drops_non_english():
    from services.processing.filters import is_english
    body = " ".join(["palabra"] * 40)
    article = _make_raw(body=body)
    with patch("services.processing.filters.detect", return_value="es"):
        assert is_english(article) is False


def test_is_english_skips_detection_for_short_body():
    from services.processing.filters import is_english
    article = _make_raw(body="Short body.")
    with patch("services.processing.filters.detect", side_effect=AssertionError("should not call")):
        assert is_english(article) is True


def test_is_relevant_passes_world_news():
    from services.processing.filters import is_relevant
    article = _make_raw(title="UN Security Council votes on ceasefire resolution")
    assert is_relevant(article) is True


def test_is_relevant_drops_blocklisted_title():
    from services.processing.filters import is_relevant
    assert is_relevant(_make_raw(title="Cute cats go viral on TikTok")) is False
    assert is_relevant(_make_raw(title="NFL scores from last night")) is False
    assert is_relevant(_make_raw(title="Horoscope for Aries today")) is False


def test_is_relevant_drops_short_title():
    from services.processing.filters import is_relevant
    assert is_relevant(_make_raw(title="Breaking")) is False


def test_is_relevant_drops_gdelt_garbage_slug():
    from services.processing.filters import is_relevant
    article = _make_raw(source="gdelt", title="ab12cd34ef56")
    assert is_relevant(article) is False


def test_is_relevant_allows_gdelt_normal_title():
    from services.processing.filters import is_relevant
    article = _make_raw(source="gdelt", title="Earthquake strikes coastal region overnight")
    assert is_relevant(article) is True


# ---------------------------------------------------------------------------
# normalizer tests
# ---------------------------------------------------------------------------

def test_normalizer_strips_html():
    from services.processing.normalizer import normalize
    article = _make_raw(body="<p>Hello <b>world</b></p>")
    result = normalize(article)
    assert "<" not in result.cleaned_body
    assert "Hello" in result.cleaned_body
    assert "world" in result.cleaned_body


def test_normalizer_removes_urls():
    from services.processing.normalizer import normalize
    article = _make_raw(body="Read more at https://example.com/article for details.")
    result = normalize(article)
    assert "https://" not in result.cleaned_body


def test_normalizer_collapses_whitespace():
    from services.processing.normalizer import normalize
    article = _make_raw(body="Word1   \n\n  Word2\t\tWord3")
    result = normalize(article)
    assert "  " not in result.cleaned_body


def test_normalizer_preserves_ids():
    from services.processing.normalizer import normalize
    article = _make_raw(article_id="myid123")
    result = normalize(article)
    assert result.id == "myid123"


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

def test_processing_run_produces_normalized_file(tmp_data_dir):
    from services.processing import run as proc_run

    articles = [_make_raw("a1"), _make_raw("a2")]
    raw_path = tmp_data_dir / "raw_articles.json"
    raw_path.write_text(json.dumps([a.to_dict() for a in articles]))

    with patch("services.processing.filters.utcnow", return_value=FIXED_NOW):
        result = proc_run.run()

    assert len(result) == 2
    out = tmp_data_dir / "normalized_articles.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert len(data) == 2


def test_processing_run_handles_missing_raw_file(tmp_data_dir):
    from services.processing import run as proc_run
    result = proc_run.run()
    assert result == []
