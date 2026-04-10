"""
Manual cleanup script for emergency use.
Deletes all expired Firestore events and enforces the document cap.
Run this once after initial Firebase setup or when storage gets unexpectedly large.

Usage:
    python backend/workers/cleanup.py

Do NOT run this on a cron — Firestore TTL policies handle routine cleanup server-side.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from shared.utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    from backend.workers.firestore_writer import delete_expired_events, enforce_document_cap

    logger.info("=== Manual cleanup started ===")

    deleted_expired = delete_expired_events()
    deleted_cap = enforce_document_cap()

    logger.info(
        "Cleanup complete: %d expired deleted, %d cap-enforcement deleted",
        deleted_expired,
        deleted_cap,
    )


if __name__ == "__main__":
    main()
