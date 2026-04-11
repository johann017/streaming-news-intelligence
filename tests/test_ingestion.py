"""Tests for the ingestion service (RSS + Guardian fetchers)."""
from __future__ import annotations

import json
import os
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
    from shared.config import INGESTION_LOOKBACK_HOURS
    expected = FIXED_NOW - timedelta(hours=INGESTION_LOOKBACK_HOURS)
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

    entry = _make_feed_entry("Duplicate", "http://example.com/same", NEW_TIME, "body")
    mock_feed = MagicMock()
    mock_feed.entries = [entry, entry]

    with patch("services.ingestion.rss_fetcher.feedparser.parse", return_value=mock_feed):
        with patch("services.ingestion.rss_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_rss(feed_url)

    assert articles[0].id == articles[1].id


def test_rss_fetcher_handles_network_error(tmp_data_dir):
    from services.ingestion.rss_fetcher import fetch_rss
    with patch("services.ingestion.rss_fetcher.feedparser.parse", side_effect=Exception("timeout")):
        articles = fetch_rss("http://bad-url.example.com/rss.xml")
    assert articles == []


# ---------------------------------------------------------------------------
# guardian_fetcher tests
# ---------------------------------------------------------------------------

def _mock_guardian_response(items: list[dict]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": {"results": items}}
    return resp


def _make_guardian_item(title: str, url: str, published: datetime, body: str = "") -> dict:
    return {
        "webUrl": url,
        "webPublicationDate": published.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "webTitle": title,
        "fields": {
            "headline": title,
            "bodyText": body or f"Full body text for {title}.",
        },
        "sectionName": "World news",
    }


def test_guardian_fetcher_returns_new_articles(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "GUARDIAN_API_KEY", "test-key")

    from services.ingestion.cursor_store import set_cursor
    from services.ingestion.guardian_fetcher import fetch_guardian

    set_cursor("guardian", OLD_TIME)

    items = [
        _make_guardian_item("New World Story", "https://theguardian.com/1", NEW_TIME),
        _make_guardian_item("Old World Story", "https://theguardian.com/2", OLD_TIME - timedelta(hours=1)),
    ]

    with patch("services.ingestion.guardian_fetcher.requests.get",
               return_value=_mock_guardian_response(items)):
        with patch("services.ingestion.guardian_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_guardian()

    assert len(articles) == 1
    assert articles[0].title == "New World Story"
    assert articles[0].source == "guardian"


def test_guardian_fetcher_skips_when_no_api_key(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "GUARDIAN_API_KEY", "")

    from services.ingestion.guardian_fetcher import fetch_guardian
    articles = fetch_guardian()
    assert articles == []


def test_guardian_fetcher_handles_http_error(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "GUARDIAN_API_KEY", "test-key")

    from services.ingestion.guardian_fetcher import fetch_guardian

    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("503 Service Unavailable")

    with patch("services.ingestion.guardian_fetcher.requests.get", return_value=mock_resp):
        articles = fetch_guardian()

    assert articles == []


def test_guardian_fetcher_advances_cursor(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "GUARDIAN_API_KEY", "test-key")

    from services.ingestion.cursor_store import get_cursor, set_cursor
    from services.ingestion.guardian_fetcher import fetch_guardian

    set_cursor("guardian", OLD_TIME)

    items = [_make_guardian_item("Story", "https://theguardian.com/1", NEW_TIME)]

    with patch("services.ingestion.guardian_fetcher.requests.get",
               return_value=_mock_guardian_response(items)):
        with patch("services.ingestion.guardian_fetcher.utcnow", return_value=FIXED_NOW):
            fetch_guardian()

    assert get_cursor("guardian") == NEW_TIME


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

def test_ingestion_run_writes_json(tmp_data_dir):
    from services.ingestion import run as ingestion_run

    mock_rss = [
        MagicMock(id="aaa", to_dict=lambda: {"id": "aaa", "title": "RSS story", "source": "rss"})
    ]
    mock_guardian = [
        MagicMock(id="bbb", to_dict=lambda: {"id": "bbb", "title": "Guardian story", "source": "guardian"})
    ]

    with patch("services.ingestion.run.fetch_all_rss", return_value=mock_rss):
        with patch("services.ingestion.run.fetch_guardian", return_value=mock_guardian):
            with patch("services.ingestion.run.fetch_gdelt", return_value=[]):
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
        with patch("services.ingestion.run.fetch_guardian", return_value=[shared_article]):
            with patch("services.ingestion.run.fetch_gdelt", return_value=[]):
                result = ingestion_run.run()

    assert len(result) == 1
