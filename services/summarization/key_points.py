"""
Extractive key-point extraction.
Splits the summary into sentences and returns the top N as bullet points.
No secondary ML model needed — keeps dependencies light.
"""
from __future__ import annotations

import nltk

nltk.download("punkt_tab", quiet=True)


def extract_key_points(summary: str, max_points: int = 3) -> list[str]:
    """
    Split the summary into sentences and return up to `max_points` as bullet points.
    Returns an empty list if the summary is empty.
    """
    if not summary.strip():
        return []
    sentences = nltk.sent_tokenize(summary.strip())
    points = [s.strip() for s in sentences if len(s.strip()) > 10]
    return points[:max_points]
