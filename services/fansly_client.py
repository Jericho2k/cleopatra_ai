"""
services/fansly_client.py

Fansly private API client — reverse engineered from browser traffic.
Base URL: https://apiv3.fansly.com/api/v1/
Auth: Bearer token + 3 custom headers captured from browser session.
"""

import asyncio
import logging
import os
import random
import time
from typing import Optional
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://apiv3.fansly.com/api/v1"

# Full header profile from browser intercept (Safari on macOS)
# Missing any of these increases ban risk
BASE_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://fansly.com",
    "Referer": "https://fansly.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Priority": "u=3, i",
}

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.3 Safari/605.1.15",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
]


def _is_fansly_host(value: str) -> bool:
    """Whether a URL points at an HTTPS Fansly-controlled host.

    Kept local rather than imported from ``services.apifansly`` so the direct
    transport does not depend on the provider module it is meant to replace.
    """
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    host = str(parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host == "fansly.com" or host.endswith(".fansly.com")
    )


class SessionExpiredError(Exception):
    pass


class FanslyAPIError(Exception):
    pass


class FanslyClient:
    """
    HTTP client for one Fansly model account.

    How to get the 4 required values:
      1. Log into Fansly in Chrome/Safari
      2. DevTools → Network → any /api/v1/ request → Headers tab
      3. Copy: Authorization header value
      4. Copy: fansly-client-id header value
      5. Copy: fansly-client-check header value
      6. Copy from Cookie header: the f-s-c=... value (just the value, not 'f-s-c=')

    These are stable for weeks. Store encrypted in DB.
    You get a 401 when they expire → trigger re-auth alert.
    """

    def __init__(
        self,
        account_id: str,
        auth_token: str,       # Authorization header value (the full string)
        client_id: str,        # fansly-client-id header value
        client_check: str,     # fansly-client-check header value
        session_cookie: str,   # f-s-c cookie value
        proxy_url: Optional[str] = None,  # "http://user:pass@host:port"
        user_agent: Optional[str] = None,
    ):
        self.account_id = account_id
        self.auth_token = auth_token
        self.client_id = client_id
        self.client_check = client_check
        self.session_cookie = session_cookie
        self.proxy_url = proxy_url
        self.user_agent = user_agent or random.choice(USER_AGENTS)
        self._client: Optional[httpx.AsyncClient] = None
        self._last_request_at = 0.0

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            proxy=self.proxy_url,
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    def _auth_headers(self) -> dict:
        return {
            **BASE_HEADERS,
            "Authorization": self.auth_token,
            "fansly-client-id": self.client_id,
            "fansly-client-ts": str(int(time.time() * 1000)),
            "fansly-client-check": self.client_check,
            "Cookie": (
                f"fansly-d={self.client_id}; "
                f"f-d={self.client_id}; "
                f"f-s-c={self.session_cookie}"
            ),
            "User-Agent": self.user_agent,
        }

    async def _jitter(self):
        """Human-like delay between requests: 0.5–2s base + random jitter."""
        now = time.monotonic()
        since_last = now - self._last_request_at
        base_interval = random.uniform(0.5, 2.0)
        wait = max(0, base_interval - since_last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_at = time.monotonic()

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        json: Optional[dict] = None,
        retries: int = 3,
    ) -> dict:
        params = params or {}
        params["ngsw-bypass"] = "true"  # required PWA bypass param
        url = f"{BASE_URL}{path}"

        for attempt in range(retries):
            await self._jitter()
            try:
                resp = await self._client.request(
                    method, url,
                    headers=self._auth_headers(),
                    params=params,
                    json=json,
                )

                # Server may rotate the session cookie
                if "set-cookie" in resp.headers:
                    self._maybe_update_cookie(resp.headers["set-cookie"])

                if resp.status_code == 401:
                    raise SessionExpiredError(
                        f"Session expired for account {self.account_id}"
                    )

                if resp.status_code == 429:
                    wait = 60 * (2 ** attempt)
                    logger.warning(f"[{self.account_id}] Rate limited, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if not data.get("success"):
                    raise FanslyAPIError(f"success=false: {data}")

                return data["response"]

            except (httpx.ConnectError, httpx.TimeoutException):
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(5 * (2 ** attempt))

        raise FanslyAPIError("All retries exhausted")

    def _maybe_update_cookie(self, set_cookie: str):
        if "f-s-c=" in set_cookie:
            try:
                start = set_cookie.index("f-s-c=") + 6
                end = set_cookie.index(";", start)
                self.session_cookie = set_cookie[start:end]
            except ValueError:
                pass

    # ── Account ────────────────────────────────────────────────────────────

    async def get_me(self) -> dict:
        """Verify session is alive. Returns account info."""
        return await self._request("GET", "/account/me")

    # ── Messages ───────────────────────────────────────────────────────────

    async def get_chat_groups(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """Get inbox — list of all conversations."""
        data = await self._request(
            "GET", "/messenger/groups",
            params={"limit": limit, "offset": offset}
        )
        return data.get("groups", data if isinstance(data, list) else [])

    async def get_messages(
        self,
        group_id: str,
        limit: int = 20,
        before_id: Optional[str] = None,
        after_id: Optional[str] = None,
    ) -> dict:
        """
        Fetch messages for a conversation.
        - before_id: paginate backwards (load older messages)
        - after_id: poll for new messages since last seen ID
        """
        params: dict = {"limit": limit}
        if before_id:
            params["before"] = before_id
        if after_id:
            params["after"] = after_id
        return await self._request(
            "GET", f"/messenger/groups/{group_id}/messages", params=params
        )

    async def send_message(self, group_id: str, content: str) -> dict:
        """Send a message to a fan conversation."""
        # Small pre-send delay to mimic typing
        await asyncio.sleep(random.uniform(0.3, 1.2))
        payload = {
            "groupId": group_id,
            "content": content,
            "correlationId": self._new_correlation_id(),
        }
        return await self._request("POST", "/messenger/message", json=payload)

    async def get_subscribers(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """Get active subscribers list."""
        data = await self._request(
            "GET", "/subscriptions",
            params={"limit": limit, "offset": offset, "status": "active"}
        )
        return data.get("subscriptions", [])

    async def poll_new_messages(self, group_id: str, since_id: str) -> list[dict]:
        """Lightweight poll: only returns messages newer than since_id."""
        data = await self.get_messages(group_id, after_id=since_id, limit=50)
        return data.get("messages", [])

    async def ingest_full_history(self, group_id: str, batch_size: int = 50):
        """
        Async generator — yields all messages oldest-first.
        Use for initial RAG ingestion of a conversation.
        """
        before_id = None
        batches = []

        while True:
            data = await self.get_messages(group_id, limit=batch_size, before_id=before_id)
            messages = data.get("messages", [])
            if not messages:
                break
            batches.append(messages)
            before_id = messages[-1]["id"]
            if len(messages) < batch_size:
                break
            await asyncio.sleep(random.uniform(1.0, 2.0))

        for batch in reversed(batches):
            for msg in reversed(batch):
                yield msg

    # ── Media ──────────────────────────────────────────────────────────────
    #
    # These two exist so vault classification can read a protected asset
    # without paying the provider's per-megabyte media proxy. They are the
    # only methods here that move bytes rather than JSON, so they deliberately
    # bypass ``_request``: its JSON envelope handling, its ``success`` check
    # and its ngsw-bypass parameter all describe the API, not the CDN.

    async def get_account_media(self, media_ids: list[str]) -> dict:
        """Metadata for account-media items, including FRESH signed locations.

        This is the free way to recover from an expired signed URL: ask the API
        for the item again and it answers with a newly signed location that the
        CDN will serve to anyone. Fetching that location costs nothing, which
        is the whole reason this method exists.

        The path is overridable because it is observed from browser traffic
        rather than published, and an upstream rename must be a configuration
        change rather than a deploy — the same reason
        ``APIFANSLY_LISTS_PATH`` is overridable in ``services/apifansly.py``.
        """
        ids = [str(value).strip() for value in media_ids if str(value or "").strip()]
        if not ids:
            return {}
        path = str(
            os.environ.get("FANSLY_DIRECT_MEDIA_PATH") or "/account/media"
        )
        if not path.startswith("/"):
            path = "/" + path
        return await self._request("GET", path, params={"ids": ",".join(ids)})

    async def download_asset(
        self,
        url: str,
        *,
        timeout: float = 45.0,
        authenticated: bool = True,
    ) -> httpx.Response:
        """GET one CDN asset as bytes, optionally carrying the session.

        ``authenticated`` sends the account's session headers. That is only
        ever safe against a Fansly-controlled host, so the caller must have
        checked the host first; this method re-checks rather than trusting it,
        because the cost of being wrong is leaking a live session token to
        whatever host ended up in a stored URL.
        """
        headers = dict(BASE_HEADERS)
        headers["Accept"] = "*/*"
        headers["User-Agent"] = self.user_agent
        if authenticated:
            if not _is_fansly_host(url):
                raise FanslyAPIError(
                    "refusing to send session credentials to a non-Fansly host"
                )
            headers = self._auth_headers()
            headers["Accept"] = "*/*"
        await self._jitter()
        return await self._client.get(
            url,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )

    @staticmethod
    def _new_correlation_id() -> str:
        ts = int(time.time() * 1000)
        rand = random.randint(0, 0xFFFF)
        return str((ts << 16) | rand)