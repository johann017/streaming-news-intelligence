"""
GDELT v2 GKG (Global Knowledge Graph) fetcher.
GDELT publishes a new 15-minute CSV every 15 minutes at a known URL.
We download the latest available file and parse relevant columns.

No API key required. GDELT is a free public dataset.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from datetime import datetime, timezone
from urllib.parse import urlparse, unquote

import requests

from shared.models import RawArticle
from shared.utils import compute_hash, get_logger, utcnow
from services.ingestion.cursor_store import get_cursor, set_cursor

logger = get_logger(__name__)

_GDELT_LASTUPDATE_URL = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
_CURSOR_KEY = "gdelt"
_TIMEOUT = 30  # seconds

# GDELT GKG columns we care about (0-indexed)
# Full schema: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
_COL_DATE = 0  # YYYYMMDDHHMMSS
_COL_SOURCE_URL = 4  # article URL
_COL_THEMES = 7  # semicolon-separated themes (e.g., "TERROR;PROTEST;ELECTION")
_COL_LOCATIONS = 9  # semicolon-separated location info
_COL_TITLE = -1  # not a real column; we derive from URL

# UUID path segments look like cbb83379-13bc-50bb-935e-0457b8a2dd3d.
# They're meaningless as titles and must be excluded before picking the slug.
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$",
    re.IGNORECASE,
)


def _parse_gdelt_timestamp(ts_str: str) -> datetime | None:
    """Parse GDELT timestamp format YYYYMMDDHHMMSS into a timezone-aware datetime."""
    try:
        dt = datetime.strptime(ts_str.strip(), "%Y%m%d%H%M%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _get_latest_csv_url() -> tuple[str, datetime] | None:
    """
    Fetch GDELT's lastupdate.txt to find the most recent GKG CSV URL.
    Returns (url, published_at) or None on failure.
    """
    try:
        resp = requests.get(_GDELT_LASTUPDATE_URL, timeout=_TIMEOUT)
        resp.raise_for_status()
        for line in resp.text.strip().splitlines():
            parts = line.strip().split()
            if len(parts) >= 3 and "gkg.csv.zip" in parts[2]:
                url = parts[2]
                # Extract timestamp from filename: 20240601120000.gkg.csv.zip
                filename = url.split("/")[-1]
                ts_str = filename.split(".")[0]
                published_at = _parse_gdelt_timestamp(ts_str)
                return url, published_at or utcnow()
    except Exception as exc:
        logger.warning("Failed to fetch GDELT lastupdate.txt: %s", exc)
    return None


def fetch_gdelt(max_articles: int = 50) -> list[RawArticle]:
    """
    Fetch the latest GDELT GKG batch and return new articles.
    Only fetches if the batch is newer than the stored cursor.
    """
    cursor = get_cursor(_CURSOR_KEY)

    result = _get_latest_csv_url()
    if result is None:
        return []

    csv_url, batch_published_at = result

    if batch_published_at <= cursor:
        logger.info("GDELT: no new batch (latest: %s)", batch_published_at.isoformat())
        return []

    logger.info("GDELT: fetching batch from %s", batch_published_at.isoformat())

    try:
        resp = requests.get(csv_url, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("Failed to download GDELT CSV: %s", exc)
        return []

    # GDELT files are zip-compressed CSVs
    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_filename = zf.namelist()[0]
            with zf.open(csv_filename) as csv_file:
                content = csv_file.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Failed to decompress GDELT CSV: %s", exc)
        return []

    articles: list[RawArticle] = []
    # GDELT theme/location columns can exceed the default 128 KB field limit.
    csv.field_size_limit(10 * 1024 * 1024)
    reader = csv.reader(io.StringIO(content), delimiter="\t")

    for row in reader:
        if len(row) < 10:
            continue
        if len(articles) >= max_articles:
            break

        url = row[_COL_SOURCE_URL].strip()
        if not url or not url.startswith("http"):
            continue

        ts_str = row[_COL_DATE].strip()
        published_at = _parse_gdelt_timestamp(ts_str) or batch_published_at

        themes = row[_COL_THEMES].strip() if len(row) > _COL_THEMES else ""
        # Derive a pseudo-title from the URL path — prefer longer path components
        # as they tend to contain readable slugs (e.g. "ukraine-ceasefire-talks").
        # urlparse cleanly separates path from query string; unquote decodes
        # percent-encoded characters (e.g. %20 → space, %27 → apostrophe).
        parsed_url = urlparse(url)
        path_parts = [
            p for p in parsed_url.path.rstrip("/").split("/")
            if p and not _UUID_RE.match(p)
        ]
        path_part = max(path_parts, key=len) if path_parts else ""
        title = unquote(path_part).replace("-", " ").replace("_", " ")[:200].strip()
        # Strip leading date prefix (e.g. "2026 04 10 ") that appears when the
        # publication date is embedded in the URL slug.
        title = re.sub(r"^\d{4}\s+\d{1,2}\s+\d{1,2}\s+", "", title).strip()
        # Strip leading numeric ID prefix (e.g. "26012577." or "12345 ") that
        # appears when a publisher embeds a record ID at the start of the URL slug.
        title = re.sub(r"^\d+[.\s]+(?=[a-zA-Z])", "", title).strip()
        # URL slugs are lowercase; convert to title case for readability.
        title = title.title()
        if not title:
            title = url[:100]

        # GDELT provides no article body. Use the title as the body so the
        # article passes has_sufficient_body. Theme codes are kept only in
        # raw_metadata — including them in the body corrupts embeddings because
        # the sentence transformer treats codes like WB_137_WATER as noise,
        # pulling semantically unrelated articles together.
        body = title
        article_id = compute_hash(url + ts_str)

        articles.append(
            RawArticle(
                id=article_id,
                source="gdelt",
                url=url,
                title=title,
                body=body,
                published_at=published_at,
                fetched_at=utcnow(),
                raw_metadata={
                    "gdelt_themes": themes,
                    "batch_published_at": batch_published_at.isoformat(),
                },
            )
        )

    # Always advance the cursor once a batch is successfully downloaded,
    # so we never re-fetch the same batch on the next run even if no
    # valid articles were found in it.
    set_cursor(_CURSOR_KEY, batch_published_at)
    logger.info("GDELT: parsed %d articles from batch", len(articles))

    return articles
