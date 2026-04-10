"""
Canonical data models for the streaming-news-intelligence pipeline.
All services import from here — do not define models elsewhere.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class RawArticle:
    """A freshly fetched article before any processing."""
    id: str                          # SHA-256 of url+title (dedup key)
    source: str                      # e.g. "bbc-rss", "reuters-rss", "reddit", "gdelt"
    url: str
    title: str
    body: str                        # raw HTML or plain text
    published_at: datetime
    fetched_at: datetime
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "body": self.body,
            "published_at": self.published_at.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "raw_metadata": self.raw_metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RawArticle:
        return cls(
            id=d["id"],
            source=d["source"],
            url=d["url"],
            title=d["title"],
            body=d["body"],
            published_at=datetime.fromisoformat(d["published_at"]),
            fetched_at=datetime.fromisoformat(d["fetched_at"]),
            raw_metadata=d.get("raw_metadata", {}),
        )


@dataclass
class NormalizedArticle:
    """A cleaned, filtered article ready for embedding."""
    id: str                          # same SHA-256 as RawArticle.id
    source: str
    url: str
    title: str
    cleaned_body: str                # HTML stripped, boilerplate removed
    published_at: datetime
    fetched_at: datetime
    word_count: int = 0
    reddit_score: int = 0            # upvotes if source == "reddit", else 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "cleaned_body": self.cleaned_body,
            "published_at": self.published_at.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "word_count": self.word_count,
            "reddit_score": self.reddit_score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NormalizedArticle:
        return cls(
            id=d["id"],
            source=d["source"],
            url=d["url"],
            title=d["title"],
            cleaned_body=d["cleaned_body"],
            published_at=datetime.fromisoformat(d["published_at"]),
            fetched_at=datetime.fromisoformat(d["fetched_at"]),
            word_count=d.get("word_count", 0),
            reddit_score=d.get("reddit_score", 0),
        )


@dataclass
class Cluster:
    """A group of articles about the same real-world event."""
    cluster_id: str                          # SHA-256 of sorted article IDs
    article_ids: list[str]
    representative_title: str               # title of article closest to centroid
    representative_url: str
    sources: list[str]                      # unique source names in this cluster
    created_at: datetime
    centroid_embedding: list[float] = field(default_factory=list)
    is_singleton: bool = False              # True if DBSCAN labeled as noise

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "article_ids": self.article_ids,
            "representative_title": self.representative_title,
            "representative_url": self.representative_url,
            "sources": self.sources,
            "created_at": self.created_at.isoformat(),
            "centroid_embedding": self.centroid_embedding,
            "is_singleton": self.is_singleton,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Cluster:
        return cls(
            cluster_id=d["cluster_id"],
            article_ids=d["article_ids"],
            representative_title=d["representative_title"],
            representative_url=d["representative_url"],
            sources=d["sources"],
            created_at=datetime.fromisoformat(d["created_at"]),
            centroid_embedding=d.get("centroid_embedding", []),
            is_singleton=d.get("is_singleton", False),
        )


@dataclass
class RankedCluster(Cluster):
    """A Cluster with an importance score attached."""
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    is_top_event: bool = False
    reddit_engagement: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "score": self.score,
            "score_breakdown": self.score_breakdown,
            "is_top_event": self.is_top_event,
            "reddit_engagement": self.reddit_engagement,
        })
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RankedCluster:  # type: ignore[override]
        base = Cluster.from_dict(d)
        return cls(
            cluster_id=base.cluster_id,
            article_ids=base.article_ids,
            representative_title=base.representative_title,
            representative_url=base.representative_url,
            sources=base.sources,
            created_at=base.created_at,
            centroid_embedding=base.centroid_embedding,
            is_singleton=base.is_singleton,
            score=d.get("score", 0.0),
            score_breakdown=d.get("score_breakdown", {}),
            is_top_event=d.get("is_top_event", False),
            reddit_engagement=d.get("reddit_engagement", 0),
        )


@dataclass
class Event:
    """
    A fully processed, summarized event ready to write to Firestore.
    This is the final output of the pipeline.
    """
    event_id: str                           # same as cluster_id
    cluster_id: str
    summary: str                            # 2-3 sentence summary
    key_points: list[str]                   # up to 3 bullet points
    sources: list[str]                      # source URLs (not domains)
    representative_title: str
    score: float
    is_top_event: bool
    geo_tags: list[str]                     # country/city names detected
    created_at: datetime
    expires_at: datetime                    # Firestore TTL field

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "cluster_id": self.cluster_id,
            "summary": self.summary,
            "key_points": self.key_points,
            "sources": self.sources,
            "representative_title": self.representative_title,
            "score": self.score,
            "is_top_event": self.is_top_event,
            "geo_tags": self.geo_tags,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Event:
        return cls(
            event_id=d["event_id"],
            cluster_id=d["cluster_id"],
            summary=d["summary"],
            key_points=d.get("key_points", []),
            sources=d.get("sources", []),
            representative_title=d["representative_title"],
            score=d.get("score", 0.0),
            is_top_event=d.get("is_top_event", False),
            geo_tags=d.get("geo_tags", []),
            created_at=datetime.fromisoformat(d["created_at"]),
            expires_at=datetime.fromisoformat(d["expires_at"]),
        )


def new_event_id() -> str:
    """Generate a unique event ID (used when cluster_id is not yet known)."""
    return str(uuid.uuid4())
