"""
Authentication module for GlassBox demo.

Simulates a realistic auth flow with:
- login
- token refresh
- basic in-memory user store

NOTE:
This file intentionally contains a few subtle bugs
so that the debugging agent has something to fix.
"""

import time
from typing import Optional, Dict


class AuthError(Exception):
    pass


class User:
    def __init__(self, user_id: str, password_hash: str):
        self.user_id = user_id
        self.password_hash = password_hash
        self.created_at = time.time()


class InMemoryUserStore:
    """
    Very naive user store.
    In a real system this would be a database.
    """

    def __init__(self) -> None:
        self._users: Dict[str, User] = {}

    def add_user(self, user_id: str, password_hash: str) -> None:
        if user_id in self._users:
            raise AuthError(f"user {user_id} already exists")
        self._users[user_id] = User(user_id, password_hash)

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)


class Token:
    def __init__(self, user_id: str, expires_at: float):
        self.user_id = user_id
        self.expires_at = expires_at

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class TokenService:
    """
    WARNING: This implementation is NOT secure.
    It is only for demo purposes.
    """

    def __init__(self, ttl_seconds: int = 60) -> None:
        self.ttl_seconds = ttl_seconds
        self._tokens: Dict[str, Token] = {}

    def issue_token(self, user_id: str) -> str:
        expires_at = time.time() + self.ttl_seconds
        token_str = f"{user_id}:{int(expires_at)}"
        self._tokens[token_str] = Token(user_id=user_id, expires_at=expires_at)
        return token_str

    def validate_token(self, token_str: str) -> Token:
        token = self._tokens.get(token_str)
        if token is None:
            raise AuthError("Unknown token")

        # BUG: expired tokens should be rejected, but we forgot to check!
        # if token.is_expired:
        #     raise AuthError("Token has expired")

        return token

    def refresh_token(self, token_str: str) -> str:
        """
        Refresh a token. If the original token is expired,
        we should NOT refresh it.
        """
        token = self.validate_token(token_str)

        # INTENDED BEHAVIOUR:
        # if token.is_expired:
        #     raise AuthError("Cannot refresh expired token")

        # BUG: We refresh even if it's expired.
        return self.issue_token(token.user_id)


def hash_password(raw: str) -> str:
    # NOTE: Not a real hash, intentionally bad.
    return "hash_" + raw


def verify_password(raw: str, password_hash: str) -> bool:
    return hash_password(raw) == password_hash


def login(user_store: InMemoryUserStore, token_service: TokenService,
          user_id: str, raw_password: str) -> str:
    user = user_store.get_user(user_id)
    if user is None:
        raise AuthError("User does not exist")

    if not verify_password(raw_password, user.password_hash):
        raise AuthError("Invalid credentials")

    return token_service.issue_token(user_id)
