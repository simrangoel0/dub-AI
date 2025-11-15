"""
Session management for GlassBox demo.

Coordinates:
- active sessions per user
- session validation using tokens
- automatic cleanup of expired sessions

Contains intentional edge cases / bugs for the debugging agent.
"""

import time
from typing import Dict, Optional

from auth import TokenService, AuthError


class Session:
    def __init__(self, session_id: str, user_id: str, created_at: float, last_active_at: float):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = created_at
        self.last_active_at = last_active_at

    @property
    def is_stale(self) -> bool:
        # Session is considered stale if idle for more than 120 seconds
        return (time.time() - self.last_active_at) > 120


class SessionManager:
    """
    WARNING:
    This class is intentionally simplistic and slightly broken.
    The goal is to generate realistic bugs for the debugging agent.
    """

    def __init__(self, token_service: TokenService) -> None:
        self.token_service = token_service
        self._sessions: Dict[str, Session] = {}  # session_id -> Session

    def create_session(self, user_id: str) -> Session:
        session_id = f"session_{user_id}_{int(time.time())}"
        now = time.time()
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_active_at=now,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def touch_session(self, session_id: str) -> None:
        """
        Update last_active_at for a session.
        """
        session = self.get_session(session_id)
        if not session:
            raise KeyError(f"Unknown session_id={session_id}")
        session.last_active_at = time.time()

    def validate_session(self, session_id: str, token_str: str) -> bool:
        """
        Validate that the session exists and the token is valid
        for the associated user.
        """
        session = self.get_session(session_id)
        if not session:
            # BUG: We silently return False instead of raising.
            return False

        try:
            token = self.token_service.validate_token(token_str)
        except AuthError:
            # NOTE: The LLM might need to adjust this behaviour.
            return False

        # BUG: We never check that token.user_id == session.user_id!
        # This allows token-for-user-A to be used with session-for-user-B.
        # if token.user_id != session.user_id:
        #     return False

        if session.is_stale:
            # QUESTIONABLE: Should we also invalidate the token?
            return False

        # On success, update activity timestamp
        session.last_active_at = time.time()
        return True

    def cleanup_stale_sessions(self) -> int:
        """
        Remove all sessions that are stale.
        Returns the number of sessions removed.
        """
        to_delete = [
            sid for sid, s in self._sessions.items()
            if s.is_stale
        ]
        for sid in to_delete:
            del self._sessions[sid]
        return len(to_delete)
