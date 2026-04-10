"""
Tests for the Firestore writer.
All Firestore interactions are mocked — no real Firebase connection needed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "EVENTS_PATH", str(tmp_path / "events.json"))
    monkeypatch.setattr(cfg, "FIRESTORE_COLLECTION_EVENTS", "events")
    monkeypatch.setattr(cfg, "FIRESTORE_MAX_DOCUMENTS", 5)
    yield tmp_path


def _make_event(
    event_id: str = "evt1",
    score: float = 0.5,
    is_top_event: bool = False,
    hours_until_expiry: float = 48.0,
) -> "Event":
    from shared.models import Event
    return Event(
        event_id=event_id,
        cluster_id=event_id,
        summary="Test summary sentence here.",
        key_points=["Point one.", "Point two."],
        sources=["https://example.com/article"],
        representative_title="Test event headline",
        score=score,
        is_top_event=is_top_event,
        geo_tags=["France"],
        created_at=FIXED_NOW,
        expires_at=FIXED_NOW + timedelta(hours=hours_until_expiry),
    )


def _make_mock_db(existing_doc_ids: list[str] | None = None):
    """Create a mock Firestore db object."""
    db = MagicMock()

    # Mock collection().select().stream() to return existing doc IDs
    existing_ids = existing_doc_ids or []
    mock_docs = [MagicMock(id=doc_id) for doc_id in existing_ids]

    col = MagicMock()
    col.select.return_value.stream.return_value = iter(mock_docs)
    col.where.return_value.stream.return_value = iter([])
    col.order_by.return_value.limit.return_value.stream.return_value = iter([])
    col.select.return_value.stream.return_value = iter(mock_docs)
    col.document.return_value = MagicMock()

    db.collection.return_value = col
    db.batch.return_value = MagicMock()

    return db


# ---------------------------------------------------------------------------
# upsert_events tests
# ---------------------------------------------------------------------------

def test_upsert_events_writes_new_events():
    from backend.workers import firestore_writer

    events = [_make_event("e1"), _make_event("e2")]
    mock_db = _make_mock_db(existing_doc_ids=[])

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        written = firestore_writer.upsert_events(events)

    assert written == 2
    assert mock_db.batch.return_value.set.call_count == 2


def test_upsert_events_skips_existing():
    from backend.workers import firestore_writer

    events = [_make_event("e1"), _make_event("e2")]
    # e1 already exists
    mock_db = _make_mock_db(existing_doc_ids=["e1"])

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        written = firestore_writer.upsert_events(events)

    assert written == 1  # only e2 written


def test_upsert_events_empty_list():
    from backend.workers import firestore_writer
    mock_db = _make_mock_db()
    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        written = firestore_writer.upsert_events([])
    assert written == 0


def test_upsert_events_document_structure():
    """Verify the document written to Firestore has the correct fields."""
    from backend.workers import firestore_writer

    event = _make_event("e1", score=0.75, is_top_event=True)
    mock_db = _make_mock_db(existing_doc_ids=[])

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        firestore_writer.upsert_events([event])

    batch = mock_db.batch.return_value
    assert batch.set.called
    _, call_args, _ = batch.set.mock_calls[0]
    doc_ref, doc_data = call_args[0], call_args[1]
    assert doc_data["event_id"] == "e1"
    assert doc_data["score"] == 0.75
    assert doc_data["is_top_event"] is True


# ---------------------------------------------------------------------------
# delete_expired_events tests
# ---------------------------------------------------------------------------

def test_delete_expired_events_deletes_expired_docs():
    from backend.workers import firestore_writer

    mock_db = _make_mock_db()
    expired_doc = MagicMock()
    mock_db.collection.return_value.where.return_value.stream.return_value = iter([expired_doc])

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        with patch("backend.workers.firestore_writer.utcnow", return_value=FIXED_NOW):
            count = firestore_writer.delete_expired_events()

    assert count == 1
    mock_db.batch.return_value.delete.assert_called_once_with(expired_doc.reference)


def test_delete_expired_events_none_expired():
    from backend.workers import firestore_writer

    mock_db = _make_mock_db()
    mock_db.collection.return_value.where.return_value.stream.return_value = iter([])

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        with patch("backend.workers.firestore_writer.utcnow", return_value=FIXED_NOW):
            count = firestore_writer.delete_expired_events()

    assert count == 0


# ---------------------------------------------------------------------------
# enforce_document_cap tests
# ---------------------------------------------------------------------------

def test_enforce_cap_deletes_lowest_scored_when_over():
    from backend.workers import firestore_writer
    import shared.config as cfg

    mock_db = _make_mock_db()
    # 7 docs in collection, cap is 5 → should delete 2
    mock_db.collection.return_value.select.return_value.stream.return_value = iter(
        [MagicMock() for _ in range(7)]
    )
    low_score_docs = [MagicMock(), MagicMock()]
    mock_db.collection.return_value.order_by.return_value.limit.return_value.stream.return_value = iter(
        low_score_docs
    )

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        count = firestore_writer.enforce_document_cap()

    assert count == 2


def test_enforce_cap_no_op_when_under():
    from backend.workers import firestore_writer

    mock_db = _make_mock_db()
    # 3 docs, cap is 5 → no deletion
    mock_db.collection.return_value.select.return_value.stream.return_value = iter(
        [MagicMock() for _ in range(3)]
    )

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        count = firestore_writer.enforce_document_cap()

    assert count == 0


# ---------------------------------------------------------------------------
# write_events_from_file tests
# ---------------------------------------------------------------------------

def test_write_events_from_file(tmp_data_dir):
    from backend.workers import firestore_writer

    events = [_make_event("e1"), _make_event("e2")]
    events_path = tmp_data_dir / "events.json"
    events_path.write_text(json.dumps([e.to_dict() for e in events]))

    mock_db = _make_mock_db(existing_doc_ids=[])

    with patch("backend.workers.firestore_writer._get_db", return_value=mock_db):
        with patch("backend.workers.firestore_writer.utcnow", return_value=FIXED_NOW):
            written = firestore_writer.write_events_from_file(str(events_path))

    assert written == 2


def test_write_events_from_file_handles_missing():
    from backend.workers import firestore_writer
    written = firestore_writer.write_events_from_file("/nonexistent/path.json")
    assert written == 0
