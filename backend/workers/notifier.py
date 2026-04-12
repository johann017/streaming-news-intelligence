"""
ntfy.sh push notifier.
Sends phone notifications for high-impact events via the free ntfy.sh HTTP API.
No account, no device token registration, no cost.

Setup:
  1. Install the ntfy app on your phone (iOS/Android — free)
  2. Subscribe to your private topic (e.g. a UUID: ntfy.sh/<your-topic>)
  3. Set NTFY_TOPIC env var / GitHub Actions secret to that topic name

Security note: anyone who knows your topic name can send you notifications.
Choose an unguessable topic (e.g. a UUID or random string).
"""
from __future__ import annotations

import json
import os
from collections import deque

import requests

import shared.config as cfg
from shared.models import Event
from shared.utils import get_logger

logger = get_logger(__name__)

_NTFY_TIMEOUT = 10  # seconds


def _load_notified_ids() -> deque[str]:
    if os.path.exists(cfg.NOTIFIED_IDS_PATH):
        with open(cfg.NOTIFIED_IDS_PATH) as f:
            return deque(json.load(f), maxlen=cfg.NOTIFIED_IDS_MAX_SIZE)
    return deque(maxlen=cfg.NOTIFIED_IDS_MAX_SIZE)


def _save_notified_ids(ids: deque[str]) -> None:
    os.makedirs(os.path.dirname(cfg.NOTIFIED_IDS_PATH) or ".", exist_ok=True)
    with open(cfg.NOTIFIED_IDS_PATH, "w") as f:
        json.dump(list(ids), f)


def has_been_notified(event_id: str) -> bool:
    """Return True if we already sent a notification for this event."""
    notified = _load_notified_ids()
    return event_id in set(notified)


def mark_notified(event_id: str) -> None:
    """Record that we notified for this event (prevents re-notification)."""
    notified = _load_notified_ids()
    notified.append(event_id)
    _save_notified_ids(notified)


def get_notifiable_events(events: list[Event]) -> list[Event]:
    """
    Filter events to those that warrant a phone notification:
    - must be flagged as a top event (is_top_event=True)
    - must not have been notified already in a previous run
    Returns events sorted by score descending.

    Using is_top_event rather than a raw score threshold ensures that every
    event the pipeline designates as important gets a notification, not just
    those that happen to clear an arbitrary score cutoff.
    """
    qualifying = [
        e for e in events
        if e.is_top_event
        and not has_been_notified(e.event_id)
    ]
    qualifying.sort(key=lambda e: e.score, reverse=True)
    return qualifying


def send_notification(event: Event) -> bool:
    """
    Send a single push notification via ntfy.sh.
    Returns True on success, False on failure.

    Notification format:
      Title:    Breaking: <first 80 chars of headline>
      Body:     First sentence of summary (or full title if no summary)
      Priority: urgent for top events, high otherwise
      Tags:     news, globe_showing_europe-africa (world emoji in app)
    """
    topic = cfg.NTFY_TOPIC
    if not topic:
        logger.warning("NTFY_TOPIC not set — skipping notification")
        return False

    url = f"{cfg.NTFY_BASE_URL}/{topic}"

    # Build notification content
    title = f"Breaking: {event.representative_title[:80]}"
    if event.summary:
        import re
        sentences = re.split(r"(?<=[.!?])\s+", event.summary.strip())
        body = " ".join(sentences[:2]) if len(sentences) >= 2 else sentences[0]
    else:
        body = event.representative_title

    priority = "urgent" if event.score >= 0.85 else "high"

    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": "newspaper,globe_showing_europe-africa",
        "Content-Type": "text/plain",
    }

    try:
        resp = requests.post(url, data=body.encode("utf-8"), headers=headers, timeout=_NTFY_TIMEOUT)
        resp.raise_for_status()
        logger.info(
            "Notification sent: event=%s score=%.3f title=%s",
            event.event_id[:8],
            event.score,
            event.representative_title[:60],
        )
        return True
    except Exception as exc:
        logger.warning("Failed to send notification for %s: %s", event.event_id[:8], exc)
        return False


def run(events: list[Event] | None = None) -> int:
    """
    Send notifications for all qualifying events.
    If `events` is None, reads from cfg.EVENTS_PATH.
    Returns the number of notifications sent.
    """
    if events is None:
        if not os.path.exists(cfg.EVENTS_PATH):
            logger.info("No events file — skipping notifications")
            return 0
        with open(cfg.EVENTS_PATH) as f:
            events = [Event.from_dict(d) for d in json.load(f)]

    notifiable = get_notifiable_events(events)
    logger.info(
        "Notifier: %d events total, %d qualify for notification",
        len(events),
        len(notifiable),
    )

    sent = 0
    for event in notifiable:
        if send_notification(event):
            mark_notified(event.event_id)
            sent += 1

    if sent:
        logger.info("Notifier: sent %d notifications", sent)
    return sent
