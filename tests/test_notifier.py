"""Tests for the ntfy.sh notifier."""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

FIXED_NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def tmp_data_dir(tmp_path, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "NOTIFIED_IDS_PATH", str(tmp_path / "notified_ids.json"))
    monkeypatch.setattr(cfg, "EVENTS_PATH", str(tmp_path / "events.json"))
    monkeypatch.setattr(cfg, "NTFY_TOPIC", "test-secret-topic")
    monkeypatch.setattr(cfg, "NTFY_BASE_URL", "https://ntfy.sh")
    monkeypatch.setattr(cfg, "NOTIFICATION_SCORE_THRESHOLD", 0.7)
    monkeypatch.setattr(cfg, "NOTIFIED_IDS_MAX_SIZE", 100)
    yield tmp_path


def _make_event(
    event_id: str = "evt1",
    score: float = 0.8,
    is_top_event: bool = True,
    summary: str = "Leaders met to discuss climate. Agreement was reached.",
    title: str = "World leaders agree on climate deal",
) -> "Event":
    from shared.models import Event
    return Event(
        event_id=event_id,
        cluster_id=event_id,
        summary=summary,
        key_points=["Leaders met.", "Agreement reached."],
        sources=["https://example.com/1"],
        representative_title=title,
        score=score,
        is_top_event=is_top_event,
        geo_tags=["France"],
        created_at=FIXED_NOW,
        expires_at=FIXED_NOW + timedelta(hours=48),
    )


# ---------------------------------------------------------------------------
# Dedup (notified_ids) tests
# ---------------------------------------------------------------------------

def test_has_been_notified_false_initially(tmp_data_dir):
    from backend.workers.notifier import has_been_notified
    assert has_been_notified("evt1") is False


def test_mark_notified_and_check(tmp_data_dir):
    from backend.workers.notifier import has_been_notified, mark_notified
    mark_notified("evt1")
    assert has_been_notified("evt1") is True


def test_mark_notified_does_not_affect_others(tmp_data_dir):
    from backend.workers.notifier import has_been_notified, mark_notified
    mark_notified("evt1")
    assert has_been_notified("evt2") is False


def test_notified_ids_fifo_cap(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "NOTIFIED_IDS_MAX_SIZE", 3)
    from backend.workers.notifier import has_been_notified, mark_notified
    mark_notified("e1")
    mark_notified("e2")
    mark_notified("e3")
    mark_notified("e4")  # evicts e1
    assert has_been_notified("e1") is False
    assert has_been_notified("e4") is True


# ---------------------------------------------------------------------------
# get_notifiable_events tests
# ---------------------------------------------------------------------------

def test_get_notifiable_filters_by_is_top_event(tmp_data_dir):
    """Only events with is_top_event=True qualify, regardless of score."""
    from backend.workers.notifier import get_notifiable_events
    events = [
        _make_event("e1", score=0.8, is_top_event=True),
        _make_event("e2", score=0.5, is_top_event=True),
    ]
    notifiable = get_notifiable_events(events)
    assert len(notifiable) == 2
    # Results are sorted by score descending
    assert notifiable[0].event_id == "e1"
    assert notifiable[1].event_id == "e2"


def test_get_notifiable_excludes_non_top_events(tmp_data_dir):
    """Events with is_top_event=False are excluded even if score is high."""
    from backend.workers.notifier import get_notifiable_events
    events = [
        _make_event("e1", score=0.9, is_top_event=False),
        _make_event("e2", score=0.8, is_top_event=True),
    ]
    notifiable = get_notifiable_events(events)
    assert len(notifiable) == 1
    assert notifiable[0].event_id == "e2"


def test_get_notifiable_filters_already_notified(tmp_data_dir):
    from backend.workers.notifier import get_notifiable_events, mark_notified
    mark_notified("e1")
    events = [_make_event("e1", score=0.9, is_top_event=True)]
    notifiable = get_notifiable_events(events)
    assert notifiable == []


# ---------------------------------------------------------------------------
# send_notification tests
# ---------------------------------------------------------------------------

def test_send_notification_success(tmp_data_dir):
    from backend.workers.notifier import send_notification

    event = _make_event()
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.workers.notifier.requests.post", return_value=mock_resp) as mock_post:
        result = send_notification(event)

    assert result is True
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args
    assert "test-secret-topic" in call_kwargs[0][0]  # URL contains topic


def test_send_notification_includes_correct_headers(tmp_data_dir):
    from backend.workers.notifier import send_notification

    event = _make_event(title="Big news today")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.workers.notifier.requests.post", return_value=mock_resp) as mock_post:
        send_notification(event)

    headers = mock_post.call_args[1]["headers"]
    assert "Breaking:" in headers["Title"]
    assert "Big news today" in headers["Title"]
    assert headers["Priority"] in ("urgent", "high")


def test_send_notification_returns_false_on_http_error(tmp_data_dir):
    from backend.workers.notifier import send_notification

    event = _make_event()
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = Exception("404")

    with patch("backend.workers.notifier.requests.post", return_value=mock_resp):
        result = send_notification(event)

    assert result is False


def test_send_notification_skips_when_no_topic(tmp_data_dir, monkeypatch):
    import shared.config as cfg
    monkeypatch.setattr(cfg, "NTFY_TOPIC", "")
    from backend.workers.notifier import send_notification

    with patch("backend.workers.notifier.requests.post") as mock_post:
        result = send_notification(_make_event())

    assert result is False
    mock_post.assert_not_called()


def test_send_notification_urgent_for_high_score(tmp_data_dir):
    from backend.workers.notifier import send_notification

    event = _make_event(score=0.90)
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.workers.notifier.requests.post", return_value=mock_resp) as mock_post:
        send_notification(event)

    headers = mock_post.call_args[1]["headers"]
    assert headers["Priority"] == "urgent"


# ---------------------------------------------------------------------------
# run() integration test
# ---------------------------------------------------------------------------

def test_notifier_run_sends_qualifying_events(tmp_data_dir):
    from backend.workers import notifier

    events = [
        _make_event("e1", score=0.9, is_top_event=True),
        _make_event("e2", score=0.4, is_top_event=True),   # low score but is_top_event → qualifies
        _make_event("e3", score=0.9, is_top_event=False),  # high score but not top → excluded
    ]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.workers.notifier.requests.post", return_value=mock_resp):
        sent = notifier.run(events=events)

    assert sent == 2  # e1 and e2 qualify (is_top_event=True); e3 excluded


def test_notifier_run_no_duplicates(tmp_data_dir):
    from backend.workers import notifier

    events = [_make_event("e1", score=0.9, is_top_event=True)]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()

    with patch("backend.workers.notifier.requests.post", return_value=mock_resp):
        sent1 = notifier.run(events=events)
        sent2 = notifier.run(events=events)  # second run — already notified

    assert sent1 == 1
    assert sent2 == 0


def test_notifier_run_handles_missing_events_file(tmp_data_dir):
    from backend.workers import notifier
    sent = notifier.run()
    assert sent == 0
