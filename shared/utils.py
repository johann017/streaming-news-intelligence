"""Shared utility functions used across all pipeline services."""
from __future__ import annotations

import hashlib
import logging
import sys
from datetime import datetime, timezone


def compute_hash(text: str) -> str:
    """Return a stable SHA-256 hex digest of the given text (UTF-8 encoded)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(tz=timezone.utc)


def get_logger(name: str) -> logging.Logger:
    """
    Return a consistently configured logger.
    Logs to stdout so GitHub Actions captures output in the run log.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%SZ",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def strip_html(html: str) -> str:
    """
    Remove HTML tags from a string using the stdlib html.parser.
    Faster and dependency-free compared to BeautifulSoup for simple stripping.
    """
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self._parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self._parts.append(data)

        def get_text(self) -> str:
            return " ".join(self._parts)

    stripper = _Stripper()
    stripper.feed(html)
    return stripper.get_text()


def truncate_text(text: str, max_chars: int = 512) -> str:
    """Truncate text to max_chars, breaking at word boundary where possible."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 0 else truncated
