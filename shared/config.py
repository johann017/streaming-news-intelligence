"""
Central configuration for the pipeline.
All values read from environment variables with sensible defaults.
Set overrides in .env (local) or GitHub Actions secrets.
"""

from __future__ import annotations

import os


def _int(key: str, default: int) -> int:
    val = os.getenv(key, "").strip()
    return int(val) if val else default


def _float(key: str, default: float) -> float:
    val = os.getenv(key, "").strip()
    return float(val) if val else default


def _str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _list(key: str, default: list[str]) -> list[str]:
    raw = os.getenv(key)
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return default


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

RSS_FEEDS: list[str] = _list(
    "RSS_FEEDS",
    [
        # World news
        "http://feeds.bbci.co.uk/news/world/rss.xml",
        "https://feeds.reuters.com/reuters/worldNews",
        "https://feeds.reuters.com/reuters/topNews",
        "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
        "https://www.aljazeera.com/xml/rss/all.xml",
        # US news
        "https://rss.nytimes.com/services/xml/rss/nyt/US.xml",
        "https://feeds.npr.org/1001/rss.xml",
        "https://feeds.apnews.com/rss/apf-topnews",
        "https://feeds.apnews.com/rss/apf-usnews",
    ],
)

# The Guardian Content API — free key at https://open-platform.theguardian.com/access/
GUARDIAN_API_KEY: str = _str("GUARDIAN_API_KEY", "")
GUARDIAN_SECTIONS: str = _str(
    "GUARDIAN_SECTIONS",
    "world,us-news,politics,business,science,environment,technology",
)

# How far back to look for new articles (hours) when no cursor exists
INGESTION_LOOKBACK_HOURS: int = _int("INGESTION_LOOKBACK_HOURS", 6)

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

MIN_BODY_WORDS: int = _int("MIN_BODY_WORDS", 5)
MAX_ARTICLE_AGE_HOURS: int = _int("MAX_ARTICLE_AGE_HOURS", 24)
SEEN_IDS_MAX_SIZE: int = _int("SEEN_IDS_MAX_SIZE", 10_000)

# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

EMBEDDING_MODEL: str = _str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
# eps=0.12 requires ≥88% cosine similarity — tight enough that only articles
# about the same specific event cluster together, not just the same broad topic.
DBSCAN_EPS: float = _float("DBSCAN_EPS", 0.12)
DBSCAN_MIN_SAMPLES: int = _int("DBSCAN_MIN_SAMPLES", 2)
EMBEDDING_CACHE_MAX_SIZE: int = _int("EMBEDDING_CACHE_MAX_SIZE", 10_000)

# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

# Weights for composite score (must sum to 1.0)
# Source diversity is the dominant signal: a story covered independently by
# multiple outlets is verifiably important. Article count provides diminishing-
# returns depth. Recency is a light tiebreaker — at 7-minute polling intervals
# most articles are already recent.
SCORE_WEIGHT_SOURCE_DIVERSITY: float = _float("SCORE_WEIGHT_SOURCE_DIVERSITY", 0.60)
SCORE_WEIGHT_ARTICLE_COUNT: float = _float("SCORE_WEIGHT_ARTICLE_COUNT", 0.25)
SCORE_WEIGHT_RECENCY: float = _float("SCORE_WEIGHT_RECENCY", 0.15)

RECENCY_HALF_LIFE_HOURS: float = _float("RECENCY_HALF_LIFE_HOURS", 6.0)
TOP_CLUSTERS_KEPT: int = _int("TOP_CLUSTERS_KEPT", 50)
TOP_EVENT_COUNT: int = _int("TOP_EVENT_COUNT", 10)

# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

SUMMARIZATION_MODEL: str = _str("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
MIN_SCORE_FOR_SUMMARIZATION: float = _float("MIN_SCORE_FOR_SUMMARIZATION", 0.3)
SUMMARY_MAX_LENGTH: int = _int("SUMMARY_MAX_LENGTH", 130)
SUMMARY_MIN_LENGTH: int = _int("SUMMARY_MIN_LENGTH", 30)

# ---------------------------------------------------------------------------
# Storage / Firestore
# ---------------------------------------------------------------------------

FIRESTORE_COLLECTION_EVENTS: str = _str("FIRESTORE_COLLECTION_EVENTS", "events")
FIRESTORE_MAX_DOCUMENTS: int = _int("FIRESTORE_MAX_DOCUMENTS", 500)

EVENT_TTL_HOURS: int = _int("EVENT_TTL_HOURS", 24)
TOP_EVENT_TTL_HOURS: int = _int("TOP_EVENT_TTL_HOURS", 24)

# ---------------------------------------------------------------------------
# Notifications (ntfy.sh)
# ---------------------------------------------------------------------------

NTFY_TOPIC: str = _str("NTFY_TOPIC", "")
NTFY_BASE_URL: str = _str("NTFY_BASE_URL", "https://ntfy.sh")
NOTIFICATION_SCORE_THRESHOLD: float = _float("NOTIFICATION_SCORE_THRESHOLD", 0.45)
NOTIFIED_IDS_MAX_SIZE: int = _int("NOTIFIED_IDS_MAX_SIZE", 1_000)
NOTIFIABLE_EVENTS_MAX: int = _int(
    "NOTIFIABLE_EVENTS_MAX", 5
)  # max new notifications per run

# ---------------------------------------------------------------------------
# Data paths (relative to repo root; pipeline.py sets cwd to repo root)
# ---------------------------------------------------------------------------

DATA_DIR: str = _str("DATA_DIR", "data")
RAW_ARTICLES_PATH: str = f"{DATA_DIR}/raw_articles.json"
NORMALIZED_ARTICLES_PATH: str = f"{DATA_DIR}/normalized_articles.json"
CLUSTERS_PATH: str = f"{DATA_DIR}/clusters.json"
RANKED_CLUSTERS_PATH: str = f"{DATA_DIR}/ranked_clusters.json"
EVENTS_PATH: str = f"{DATA_DIR}/events.json"
CURSORS_PATH: str = f"{DATA_DIR}/cursors.json"
SEEN_IDS_PATH: str = f"{DATA_DIR}/seen_ids.json"
NOTIFIED_IDS_PATH: str = f"{DATA_DIR}/notified_ids.json"
EMBEDDING_CACHE_PATH: str = f"{DATA_DIR}/embedding_cache.pkl"
PIPELINE_LOG_PATH: str = f"{DATA_DIR}/pipeline_run_log.json"
