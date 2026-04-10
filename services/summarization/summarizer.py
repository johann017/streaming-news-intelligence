"""
Article summarizer using HuggingFace transformers.
Primary model: sshleifer/distilbart-cnn-12-6 (~490MB, ~20s load time on CPU).
Falls back to extractive summarization (first 2 sentences) if the model
fails to load or the pipeline times out.
"""
from __future__ import annotations

import re

import shared.config as cfg
from shared.utils import get_logger, truncate_text

logger = get_logger(__name__)

_pipeline = None
_pipeline_failed = False


def _get_pipeline():
    global _pipeline, _pipeline_failed
    if _pipeline is not None:
        return _pipeline
    if _pipeline_failed:
        return None
    try:
        from transformers import pipeline
        logger.info("Loading summarization model: %s", cfg.SUMMARIZATION_MODEL)
        _pipeline = pipeline(
            "summarization",
            model=cfg.SUMMARIZATION_MODEL,
            device=-1,  # CPU only
        )
        logger.info("Summarization model loaded.")
        return _pipeline
    except Exception as exc:
        logger.warning("Failed to load summarization model: %s — using extractive fallback", exc)
        _pipeline_failed = True
        return None


def _extractive_summary(text: str, num_sentences: int = 2) -> str:
    """Simple extractive fallback: return the first N sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:num_sentences])


def summarize(text: str) -> str:
    """
    Generate a 2-3 sentence abstractive summary of the given text.
    Falls back to extractive (first 2 sentences) if the model is unavailable.
    """
    if not text.strip():
        return ""

    pipe = _get_pipeline()
    if pipe is None:
        return _extractive_summary(text)

    # Truncate to model's max input length
    truncated = truncate_text(text, max_chars=1024)

    try:
        result = pipe(
            truncated,
            max_length=cfg.SUMMARY_MAX_LENGTH,
            min_length=cfg.SUMMARY_MIN_LENGTH,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"].strip()
    except Exception as exc:
        logger.warning("Summarization failed: %s — using extractive fallback", exc)
        return _extractive_summary(text)
