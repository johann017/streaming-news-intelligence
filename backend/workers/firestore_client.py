"""
Firebase Admin SDK client initializer.
Supports two credential sources:
  1. GOOGLE_APPLICATION_CREDENTIALS env var pointing to a JSON key file (local dev)
  2. FIREBASE_SERVICE_ACCOUNT_JSON env var containing the full JSON content (GitHub Actions)

Only initializes once per process (Firebase Admin SDK singleton pattern).
"""
from __future__ import annotations

import json
import os
import tempfile

from shared.utils import get_logger

logger = get_logger(__name__)

_initialized = False


def get_firestore_client():
    """
    Return an initialized Firestore client.
    Raises RuntimeError if credentials are not configured.
    """
    global _initialized

    import firebase_admin
    from firebase_admin import credentials, firestore

    if not _initialized:
        cred = _build_credentials()
        firebase_admin.initialize_app(cred)
        _initialized = True
        logger.info("Firebase Admin SDK initialized.")

    return firestore.client()


def _build_credentials():
    """Build Firebase credentials from available environment variables."""
    import firebase_admin
    from firebase_admin import credentials

    # Option 1: full JSON content in env var (GitHub Actions)
    json_content = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    if json_content:
        try:
            cred_dict = json.loads(json_content)
            return credentials.Certificate(cred_dict)
        except Exception as exc:
            logger.warning("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", exc)

    # Option 2: path to JSON file (local dev)
    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if key_path and os.path.exists(key_path):
        return credentials.Certificate(key_path)

    raise RuntimeError(
        "Firebase credentials not found. Set FIREBASE_SERVICE_ACCOUNT_JSON "
        "(GitHub Actions) or GOOGLE_APPLICATION_CREDENTIALS (local dev)."
    )


def reset_for_testing() -> None:
    """Reset the singleton for testing. Do not call in production."""
    global _initialized
    try:
        import firebase_admin
        firebase_admin.delete_app(firebase_admin.get_app())
    except Exception:
        pass
    _initialized = False
