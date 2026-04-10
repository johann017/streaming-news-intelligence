"""
Firestore writer: upserts Events, enforces document caps, and cleans up expired events.
Uses Firebase Admin SDK (bypasses Firestore security rules).
"""
from __future__ import annotations

import json
import os

import shared.config as cfg
from shared.models import Event
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)


def _get_db():
    from backend.workers.firestore_client import get_firestore_client
    return get_firestore_client()


def get_existing_event_ids() -> set[str]:
    """Return the set of event_ids currently in Firestore."""
    db = _get_db()
    col = db.collection(cfg.FIRESTORE_COLLECTION_EVENTS)
    docs = col.select(["event_id"]).stream()
    return {doc.id for doc in docs}


def upsert_events(events: list[Event]) -> int:
    """
    Write events to Firestore using set(merge=True).
    Skips events whose IDs are already in Firestore (no-op upsert).
    Returns the number of documents written.
    """
    if not events:
        return 0

    db = _get_db()
    existing_ids = get_existing_event_ids()

    new_events = [e for e in events if e.event_id not in existing_ids]
    if not new_events:
        logger.info("Firestore: all %d events already exist — no writes needed", len(events))
        return 0

    col = db.collection(cfg.FIRESTORE_COLLECTION_EVENTS)
    batch = db.batch()
    write_count = 0

    for event in new_events:
        doc_ref = col.document(event.event_id)
        doc_data = event.to_dict()

        # Convert expires_at to a Firestore Timestamp for TTL to work
        from google.cloud.firestore_v1 import SERVER_TIMESTAMP
        from datetime import timezone
        expires_dt = event.expires_at
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        doc_data["expires_at"] = expires_dt

        # Same for created_at
        created_dt = event.created_at
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        doc_data["created_at"] = created_dt

        batch.set(doc_ref, doc_data, merge=True)
        write_count += 1

        if write_count % 400 == 0:
            # Firestore batches are limited to 500 operations
            batch.commit()
            batch = db.batch()

    if write_count % 400 != 0:
        batch.commit()

    logger.info("Firestore: wrote %d new events", write_count)
    return write_count


def delete_expired_events() -> int:
    """
    Delete documents whose expires_at is in the past.
    Returns the number of documents deleted.

    Note: Firestore TTL policies handle this server-side automatically,
    but this function provides manual cleanup when needed.
    """
    from datetime import timezone
    db = _get_db()
    now = utcnow()
    col = db.collection(cfg.FIRESTORE_COLLECTION_EVENTS)

    expired = col.where("expires_at", "<", now).stream()
    batch = db.batch()
    delete_count = 0

    for doc in expired:
        batch.delete(doc.reference)
        delete_count += 1
        if delete_count % 400 == 0:
            batch.commit()
            batch = db.batch()

    if delete_count % 400 != 0 and delete_count > 0:
        batch.commit()

    logger.info("Firestore: deleted %d expired events", delete_count)
    return delete_count


def enforce_document_cap() -> int:
    """
    If the events collection exceeds FIRESTORE_MAX_DOCUMENTS, delete the
    lowest-scoring events until we're back under the cap.
    Returns the number of documents deleted.
    """
    db = _get_db()
    col = db.collection(cfg.FIRESTORE_COLLECTION_EVENTS)

    total = len(list(col.select([]).stream()))
    if total <= cfg.FIRESTORE_MAX_DOCUMENTS:
        return 0

    excess = total - cfg.FIRESTORE_MAX_DOCUMENTS
    # Get the lowest-scored events
    to_delete = col.order_by("score").limit(excess).stream()
    batch = db.batch()
    delete_count = 0

    for doc in to_delete:
        batch.delete(doc.reference)
        delete_count += 1

    if delete_count > 0:
        batch.commit()

    logger.info(
        "Firestore cap enforced: deleted %d lowest-scoring events (cap=%d)",
        delete_count,
        cfg.FIRESTORE_MAX_DOCUMENTS,
    )
    return delete_count


def write_events_from_file(events_path: str | None = None) -> int:
    """
    Read events.json and write to Firestore.
    Entry point called by the pipeline orchestrator.
    """
    path = events_path or cfg.EVENTS_PATH
    if not os.path.exists(path):
        logger.warning("No events file at %s — skipping Firestore write", path)
        return 0

    with open(path) as f:
        event_dicts = json.load(f)
    events = [Event.from_dict(d) for d in event_dicts]

    written = upsert_events(events)
    delete_expired_events()
    enforce_document_cap()
    return written
