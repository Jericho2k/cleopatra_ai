"""
services/fansly_session_store.py

Stores encrypted Fansly sessions in your existing Supabase DB.
Tracks per-account health (green/yellow/red).
Fires Telegram alerts when sessions die.
"""

import logging
import os
import time
from typing import Optional, Callable

from cryptography.fernet import Fernet

from .fansly_client import FanslyClient, SessionExpiredError

logger = logging.getLogger(__name__)


class SessionStore:
    """
    Manages Fansly session credentials for all model accounts.

    Plug into your existing FastAPI app:

        # In main.py or wherever you init services:
        from services.fansly_session_store import SessionStore
        session_store = SessionStore(
            supabase=supabase,   # your existing supabase client
            encryption_key=os.environ["FANSLY_SESSION_KEY"],
            alert_fn=send_telegram_alert,  # optional
        )

    Generate FANSLY_SESSION_KEY once and add to Railway:
        python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    """

    def __init__(
        self,
        supabase,
        encryption_key: str,
        alert_fn: Optional[Callable] = None,
    ):
        self._supabase = supabase
        self._fernet = Fernet(encryption_key.encode())
        self._alert_fn = alert_fn
        # In-memory cache: account_id → FanslyClient kwargs
        self._cache: dict[str, dict] = {}
        # Health: account_id → consecutive failure count
        self._failures: dict[str, int] = {}

    # ── Encryption ─────────────────────────────────────────────────────────

    def _enc(self, val: str) -> str:
        return self._fernet.encrypt(val.encode()).decode()

    def _dec(self, val: str) -> str:
        return self._fernet.decrypt(val.encode()).decode()

    # ── Registration ───────────────────────────────────────────────────────

    async def register(
        self,
        account_id: str,
        username: str,
        creator_id: str,       # your internal creator/model ID in Cleopatra
        auth_token: str,
        client_id: str,
        client_check: str,
        session_cookie: str,
        proxy_url: str,
    ):
        """
        Store a new model account session.
        Call this from your onboarding endpoint when an agency connects a model.
        """
        row = {
            "account_id": account_id,
            "username": username,
            "creator_id": creator_id,
            "auth_token_enc": self._enc(auth_token),
            "client_id": client_id,           # not sensitive
            "client_check_enc": self._enc(client_check),
            "session_cookie_enc": self._enc(session_cookie),
            "proxy_url_enc": self._enc(proxy_url),
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._supabase.table("fansly_sessions").upsert(row).execute()
        self._cache[account_id] = self._row_to_kwargs(row)
        self._failures[account_id] = 0
        logger.info(f"Registered Fansly session for {username} ({account_id})")

    # ── Client factory ─────────────────────────────────────────────────────

    def get_client(self, account_id: str) -> FanslyClient:
        """
        Get a ready-to-use FanslyClient for an account.

        Usage:
            async with session_store.get_client(account_id) as client:
                messages = await client.get_messages(group_id)
        """
        if account_id not in self._cache:
            raise ValueError(f"No session for account {account_id}. Register first.")
        kwargs = self._cache[account_id]
        return FanslyClient(**kwargs)

    async def load_all(self):
        """
        Load all sessions from DB into memory cache.
        Call this on app startup.
        """
        result = self._supabase.table("fansly_sessions").select("*").execute()
        for row in (result.data or []):
            self._cache[row["account_id"]] = self._row_to_kwargs(row)
            self._failures[row["account_id"]] = 0
        logger.info(f"Loaded {len(self._cache)} Fansly sessions")

    # ── Health tracking ─────────────────────────────────────────────────────

    def record_success(self, account_id: str):
        self._failures[account_id] = 0

    async def record_failure(self, account_id: str, error: str):
        self._failures[account_id] = self._failures.get(account_id, 0) + 1
        count = self._failures[account_id]
        logger.warning(f"[{account_id}] failure #{count}: {error}")

        if count >= 3:
            msg = (
                f"🔴 Fansly session DEAD for account {account_id}\n"
                f"Error: {error}\n"
                f"Action needed: re-auth this account in Cleopatra."
            )
            if self._alert_fn:
                await self._alert_fn(msg)

    def get_health(self) -> dict:
        return {
            aid: {
                "status": "red" if f >= 3 else "yellow" if f >= 1 else "green",
                "failures": f,
            }
            for aid, f in self._failures.items()
        }

    def update_cookie(self, account_id: str, new_cookie: str):
        """Called automatically by FanslyClient when server rotates the cookie."""
        if account_id in self._cache:
            self._cache[account_id]["session_cookie"] = new_cookie
        self._supabase.table("fansly_sessions").update(
            {"session_cookie_enc": self._enc(new_cookie), "updated_at": time.time()}
        ).eq("account_id", account_id).execute()

    # ── Internal ────────────────────────────────────────────────────────────

    def _row_to_kwargs(self, row: dict) -> dict:
        return {
            "account_id": row["account_id"],
            "auth_token": self._dec(row["auth_token_enc"]),
            "client_id": row["client_id"],
            "client_check": self._dec(row["client_check_enc"]),
            "session_cookie": self._dec(row["session_cookie_enc"]),
            "proxy_url": self._dec(row["proxy_url_enc"]),
        }