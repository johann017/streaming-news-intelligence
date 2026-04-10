"""
Article embedder using sentence-transformers.
Generates dense vector representations of article text for clustering.
Maintains a disk cache to avoid re-embedding articles across pipeline runs.
"""
from __future__ import annotations

import os
import pickle
from typing import TYPE_CHECKING

import numpy as np

import shared.config as cfg
from shared.models import NormalizedArticle
from shared.utils import get_logger, truncate_text

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", cfg.EMBEDDING_MODEL)
        _model = SentenceTransformer(cfg.EMBEDDING_MODEL)
        logger.info("Model loaded.")
    return _model


def _load_cache() -> dict[str, list[float]]:
    if os.path.exists(cfg.EMBEDDING_CACHE_PATH):
        with open(cfg.EMBEDDING_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict[str, list[float]]) -> None:
    # Evict oldest entries if over cap (dict preserves insertion order in Py3.7+)
    if len(cache) > cfg.EMBEDDING_CACHE_MAX_SIZE:
        evict_count = len(cache) - cfg.EMBEDDING_CACHE_MAX_SIZE
        keys_to_evict = list(cache.keys())[:evict_count]
        for k in keys_to_evict:
            del cache[k]
    os.makedirs(os.path.dirname(cfg.EMBEDDING_CACHE_PATH) or ".", exist_ok=True)
    with open(cfg.EMBEDDING_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def embed_articles(
    articles: list[NormalizedArticle],
) -> dict[str, np.ndarray]:
    """
    Compute embeddings for a list of articles.
    Returns a dict mapping article_id → embedding vector.
    Uses disk cache; only articles not in cache are sent to the model.
    """
    cache = _load_cache()

    # Separate cached vs. uncached
    cached_ids: list[str] = []
    uncached: list[NormalizedArticle] = []
    for article in articles:
        if article.id in cache:
            cached_ids.append(article.id)
        else:
            uncached.append(article)

    logger.info(
        "Embedding: %d cached, %d to compute", len(cached_ids), len(uncached)
    )

    new_embeddings: dict[str, np.ndarray] = {}
    if uncached:
        model = _get_model()
        texts = [
            truncate_text(f"{a.title}. {a.cleaned_body}", max_chars=512)
            for a in uncached
        ]
        vectors = model.encode(texts, batch_size=32, show_progress_bar=False)
        for article, vec in zip(uncached, vectors):
            new_embeddings[article.id] = vec
            cache[article.id] = vec.tolist()  # store as list for pickle

    _save_cache(cache)

    # Build result dict: cached from pickle (as numpy), new ones direct
    result: dict[str, np.ndarray] = {}
    for article in articles:
        if article.id in new_embeddings:
            result[article.id] = new_embeddings[article.id]
        else:
            result[article.id] = np.array(cache[article.id], dtype=np.float32)

    return result
