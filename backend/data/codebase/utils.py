"""
Random utility functions.

Some of these are relevant to auth/session logic,
others are noise so the retrieval step has to
learn to ignore them.
"""

import json
from typing import Any, Dict


def pretty_print_session(session: Any) -> str:
    """
    Convert a Session object into a JSON string for debugging.
    """
    if session is None:
        return "<no session>"

    payload = {
        "session_id": getattr(session, "session_id", None),
        "user_id": getattr(session, "user_id", None),
        "created_at": getattr(session, "created_at", None),
        "last_active_at": getattr(session, "last_active_at", None),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def log_debug(message: str) -> None:
    print(f"[DEBUG] {message}")


def log_warning(message: str) -> None:
    # SECURITY NOTE:
    # In a real system this should NOT print secrets to stdout.
    print(f"[WARNING] {message}")


def calculate_unrelated_value(x: int, y: int) -> int:
    """
    Completely unrelated to auth or sessions.
    Exists to create 'distractor' chunks for retrieval.
    """
    return (x * x) + (y * y) - (x * y)
