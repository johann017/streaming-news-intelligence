"""Tests for the ingestion service (RSS + Reddit fetchers)."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
OLD_TIME = FIXED_NOW - timedelta(hours=2)
NEW_TIME = FIXED_NOW - timedelta(minutes=10)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect all data/ paths to a temp directory for test isolation."""
    import shared.config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(cfg, "CURSORS_PATH", str(tmp_path / "cursors.json"))
    monkeypatch.setattr(cfg, "RAW_ARTICLES_PATH", str(tmp_path / "raw_articles.json"))
    yield tmp_path


# ---------------------------------------------------------------------------
# cursor_store tests
# ---------------------------------------------------------------------------

def test_get_cursor_no_file_returns_fallback(tmp_data_dir):
    from services.ingestion.cursor_store import get_cursor
    with patch("services.ingestion.cursor_store.utcnow", return_value=FIXED_NOW):
        cursor = get_cursor("some-source")
    # Should be INGESTION_LOOKBACK_HOURS before now
    from shared.config import INGESTION_LOOKBACK_HOURS
    expected = FIXED_NOW - timedelta(hours=INGESTION_LOOKBACK_HOURS)
    # Allow 1-second tolerance for replace(second=0)
    assert abs((cursor - expected).total_seconds()) < 60


def test_set_and_get_cursor_roundtrip(tmp_data_dir):
    from services.ingestion.cursor_store import get_cursor, set_cursor
    ts = datetime(2024, 5, 31, 8, 0, 0, tzinfo=timezone.utc)
    set_cursor("bbc", ts)
    assert get_cursor("bbc") == ts


# ---------------------------------------------------------------------------
# rss_fetcher tests
# ---------------------------------------------------------------------------

def _make_feed_entry(title: str, url: str, published: datetime, summary: str = "") -> MagicMock:
    entry = MagicMock()
    entry.title = title
    entry.link = url
    entry.published = published.strftime("%a, %d %b %Y %H:%M:%S +0000")
    entry.summary = summary
    entry.content = None
    # Make hasattr work for 'content'
    del entry.content
    return entry


def test_rss_fetcher_returns_new_articles(tmp_data_dir):
    from services.ingestion.cursor_store import set_cursor
    from services.ingestion.rss_fetcher import fetch_rss

    feed_url = "http://example.com/rss.xml"
    set_cursor(feed_url, OLD_TIME)

    mock_feed = MagicMock()
    mock_feed.entries = [
        _make_feed_entry("New Article", "http://example.com/1", NEW_TIME, "Some content here"),
        _make_feed_entry("Old Article", "http://example.com/2", OLD_TIME - timedelta(hours=1), "Old"),
    ]

    with patch("services.ingestion.rss_fetcher.feedparser.parse", return_value=mock_feed):
        with patch("services.ingestion.rss_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_rss(feed_url)

    assert len(articles) == 1
    assert articles[0].title == "New Article"
    assert articles[0].source == "rss"


def test_rss_fetcher_deduplicates_by_hash(tmp_data_dir):
    from services.ingestion.cursor_store import set_cursor
    from services.ingestion.rss_fetcher import fetch_rss

    feed_url = "http://example.com/rss.xml"
    set_cursor(feed_url, OLD_TIME)

    # Two entries with same URL+title → same hash
    entry = _make_feed_entry("Duplicate", "http://example.com/same", NEW_TIME, "body")
    mock_feed = MagicMock()
    mock_feed.entries = [entry, entry]

    with patch("services.ingestion.rss_fetcher.feedparser.parse", return_value=mock_feed):
        with patch("services.ingestion.rss_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_rss(feed_url)

    # fetch_rss itself doesn't dedup — that's processing; but IDs should be equal
    assert articles[0].id == articles[1].id


def test_rss_fetcher_handles_network_error(tmp_data_dir):
    from services.ingestion.rss_fetcher import fetch_rss
    with patch("services.ingestion.rss_fetcher.feedparser.parse", side_effect=Exception("timeout")):
        articles = fetch_rss("http://bad-url.example.com/rss.xml")
    assert articles == []


# ---------------------------------------------------------------------------
# reddit_fetcher tests
# ---------------------------------------------------------------------------

def _mock_reddit_response(posts: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "data": {
            "children": [{"data": p} for p in posts]
        }
    }
    return resp


def test_reddit_fetcher_returns_new_posts(tmp_data_dir):
    from services.ingestion.cursor_store import set_cursor
    from services.ingestion.reddit_fetcher import fetch_subreddit

    set_cursor("reddit:worldnews", OLD_TIME)

    posts = [
        {
            "title": "Breaking News",
            "url": "https://example.com/news",
            "selftext": "Details here",
            "created_utc": NEW_TIME.timestamp(),
            "score": 1500,
            "num_comments": 200,
            "permalink": "/r/worldnews/comments/abc",
        },
        {
            "title": "Old News",
            "url": "https://example.com/old",
            "selftext": "",
            "created_utc": (OLD_TIME - timedelta(hours=1)).timestamp(),
            "score": 50,
            "num_comments": 5,
            "permalink": "/r/worldnews/comments/def",
        },
    ]

    with patch("services.ingestion.reddit_fetcher.requests.get",
               return_value=_mock_reddit_response(posts)):
        with patch("services.ingestion.reddit_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_subreddit("worldnews")

    assert len(articles) == 1
    assert articles[0].title == "Breaking News"
    assert articles[0].source == "reddit"
    assert articles[0].raw_metadata["reddit_score"] == 1500


def test_reddit_fetcher_handles_http_error(tmp_data_dir):
    from services.ingestion.reddit_fetcher import fetch_subreddit

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("403 Forbidden")

    with patch("services.ingestion.reddit_fetcher.requests.get", return_value=mock_resp):
        articles = fetch_subreddit("worldnews")

    assert articles == []


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

def test_ingestion_run_writes_json(tmp_data_dir):
    from services.ingestion import run as ingestion_run

    mock_rss = [
        MagicMock(
            id="aaa",
            to_dict=lambda: {"id": "aaa", "title": "RSS story", "source": "rss"},
        )
    ]
    mock_reddit = [
        MagicMock(
            id="bbb",
            to_dict=lambda: {"id": "bbb", "title": "Reddit story", "source": "reddit"},
        )
    ]

    with patch("services.ingestion.run.fetch_all_rss", return_value=mock_rss):
        with patch("services.ingestion.run.fetch_all_reddit", return_value=mock_reddit):
            result = ingestion_run.run()

    assert len(result) == 2
    out_path = tmp_data_dir / "raw_articles.json"
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert len(data) == 2


def test_ingestion_run_deduplicates_within_batch(tmp_data_dir):
    from services.ingestion import run as ingestion_run

    shared_article = MagicMock(id="same", to_dict=lambda: {"id": "same", "title": "dup"})

    with patch("services.ingestion.run.fetch_all_rss", return_value=[shared_article]):
        with patch("services.ingestion.run.fetch_all_reddit", return_value=[shared_article]):
            result = ingestion_run.run()

    assert len(result) == 1
