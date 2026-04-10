"""Tests for the GDELT fetcher (mocked HTTP)."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
NEW_BATCH_TS = "20240601120000"
NEW_BATCH_DT = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "CURSORS_PATH", str(tmp_path / "cursors.json"))
    yield tmp_path


def _make_zip_csv(rows: list[list[str]]) -> bytes:
    """Create an in-memory zip containing a tab-separated CSV."""
    tsv = "\n".join("\t".join(row) for row in rows)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("20240601120000.gkg.csv", tsv)
    return buf.getvalue()


def _gdelt_row(
    url: str = "https://bbc.com/news/world-article",
    ts: str = NEW_BATCH_TS,
    themes: str = "PROTEST;ELECTION",
) -> list[str]:
    """Build a minimal GDELT GKG row (tab-separated, 15 columns)."""
    row = [""] * 15
    row[0] = ts         # DATE
    row[4] = url        # SOURCEURL
    row[7] = themes     # THEMES
    row[9] = "France"   # LOCATIONS
    return row


def _mock_lastupdate(url: str = f"http://data.gdeltproject.org/gdeltv2/{NEW_BATCH_TS}.gkg.csv.zip") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = f"123 456 {url}\n"
    return resp


def _mock_csv_resp(rows: list[list[str]]) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.content = _make_zip_csv(rows)
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_gdelt_fetcher_returns_articles(tmp_data_dir):
    from services.ingestion.gdelt_fetcher import fetch_gdelt
    from services.ingestion.cursor_store import set_cursor

    # Cursor is before the batch
    set_cursor("gdelt", FIXED_NOW - timedelta(hours=1))

    rows = [_gdelt_row(), _gdelt_row(url="https://reuters.com/article/2")]

    with patch("services.ingestion.gdelt_fetcher.requests.get") as mock_get:
        mock_get.side_effect = [_mock_lastupdate(), _mock_csv_resp(rows)]
        with patch("services.ingestion.gdelt_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_gdelt()

    assert len(articles) == 2
    assert articles[0].source == "gdelt"
    assert articles[0].url.startswith("https://")


def test_gdelt_fetcher_skips_if_cursor_is_newer(tmp_data_dir):
    from services.ingestion.gdelt_fetcher import fetch_gdelt
    from services.ingestion.cursor_store import set_cursor

    # Cursor is AFTER the batch → already seen
    set_cursor("gdelt", FIXED_NOW + timedelta(hours=1))

    with patch("services.ingestion.gdelt_fetcher.requests.get") as mock_get:
        mock_get.return_value = _mock_lastupdate()
        articles = fetch_gdelt()

    assert articles == []
    # Should only have called GET once (for lastupdate.txt, not the CSV)
    assert mock_get.call_count == 1


def test_gdelt_fetcher_handles_lastupdate_failure(tmp_data_dir):
    from services.ingestion.gdelt_fetcher import fetch_gdelt
    with patch("services.ingestion.gdelt_fetcher.requests.get", side_effect=Exception("timeout")):
        articles = fetch_gdelt()
    assert articles == []


def test_gdelt_fetcher_handles_csv_download_failure(tmp_data_dir):
    from services.ingestion.gdelt_fetcher import fetch_gdelt
    from services.ingestion.cursor_store import set_cursor
    set_cursor("gdelt", FIXED_NOW - timedelta(hours=1))

    bad_resp = MagicMock()
    bad_resp.raise_for_status.side_effect = Exception("500")

    with patch("services.ingestion.gdelt_fetcher.requests.get") as mock_get:
        mock_get.side_effect = [_mock_lastupdate(), bad_resp]
        articles = fetch_gdelt()

    assert articles == []


def test_gdelt_fetcher_respects_max_articles(tmp_data_dir):
    from services.ingestion.gdelt_fetcher import fetch_gdelt
    from services.ingestion.cursor_store import set_cursor
    set_cursor("gdelt", FIXED_NOW - timedelta(hours=1))

    rows = [_gdelt_row(url=f"https://example.com/{i}") for i in range(20)]

    with patch("services.ingestion.gdelt_fetcher.requests.get") as mock_get:
        mock_get.side_effect = [_mock_lastupdate(), _mock_csv_resp(rows)]
        with patch("services.ingestion.gdelt_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_gdelt(max_articles=5)

    assert len(articles) == 5


def test_gdelt_fetcher_skips_invalid_urls(tmp_data_dir):
    from services.ingestion.gdelt_fetcher import fetch_gdelt
    from services.ingestion.cursor_store import set_cursor
    set_cursor("gdelt", FIXED_NOW - timedelta(hours=1))

    rows = [
        _gdelt_row(url=""),                         # empty URL
        _gdelt_row(url="not-a-url"),                # no scheme
        _gdelt_row(url="https://valid.com/article"), # valid
    ]

    with patch("services.ingestion.gdelt_fetcher.requests.get") as mock_get:
        mock_get.side_effect = [_mock_lastupdate(), _mock_csv_resp(rows)]
        with patch("services.ingestion.gdelt_fetcher.utcnow", return_value=FIXED_NOW):
            articles = fetch_gdelt()

    assert len(articles) == 1
    assert articles[0].url == "https://valid.com/article"
