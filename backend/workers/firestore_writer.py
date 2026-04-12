"""
Firestore writer: upserts Events and enforces a document cap.
Uses Firebase Admin SDK (bypasses Firestore security rules).

Read budget per pipeline run (target: near zero):
  - upsert_events: 0 reads (set/merge is a pure write)
  - enforce_document_cap: 1 read (COUNT aggregation)
  - delete_expired_events: not called from the pipeline;
    Firestore TTL handles routine expiry server-side at no read cost.
"""
from __future__ import annotations

import json
import os
from datetime import timezone

import shared.config as cfg
from shared.models import Event
from shared.utils import get_logger, utcnow

logger = get_logger(__name__)


def _get_db():
    from backend.workers.firestore_client import get_firestore_client
    return get_firestore_client()


def upsert_events(events: list[Event]) -> int:
    """
    Write events to Firestore using set(merge=True).
    Idempotent: re-writing an existing event_id just overwrites with fresh data.
    Returns the number of documents written.

    No pre-fetch of existing IDs — set(merge=True) is an upsert by nature,
    so checking first would only cost N reads for no benefit.
    """
    if not events:
        return 0

    db = _get_db()
    col = db.collection(cfg.FIRESTORE_COLLECTION_EVENTS)
    batch = db.batch()
    write_count = 0

    for event in events:
        doc_ref = col.document(event.event_id)
        doc_data = event.to_dict()

        # Convert expires_at / created_at to native datetimes so Firestore TTL works
        expires_dt = event.expires_at
        if expires_dt.tzinfo is None:
            expires_dt = expires_dt.replace(tzinfo=timezone.utc)
        doc_data["expires_at"] = expires_dt

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

    logger.info("Firestore: wrote %d events", write_count)
    return write_count


def enforce_document_cap() -> int:
    """
    If the events collection exceeds FIRESTORE_MAX_DOCUMENTS, delete the
    lowest-scoring events until we're back under the cap.
    Uses COUNT aggregation (1 read) instead of streaming all documents.
    Returns the number of documents deleted.
    """
    db = _get_db()
    col = db.collection(cfg.FIRESTORE_COLLECTION_EVENTS)

    # COUNT aggregation costs exactly 1 read regardless of collection size
    count_result = col.count().get()
    total = count_result[0][0].value

    if total <= cfg.FIRESTORE_MAX_DOCUMENTS:
        return 0

    excess = total - cfg.FIRESTORE_MAX_DOCUMENTS
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


def delete_expired_events() -> int:
    """
    Manually delete documents whose expires_at is in the past.
    NOT called from the pipeline — Firestore TTL handles routine expiry
    server-side at no read cost. Only call this via cleanup.py for emergency use.
    Returns the number of documents deleted.
    """
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
    enforce_document_cap()
    return written
