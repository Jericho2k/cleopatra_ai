"""Shared API Fansly client contract.

Every call to the managed Fansly API goes through this module so payload shapes,
response unwrapping, authentication, and account-access errors remain identical
across chat sync, vault sync, PPV delivery, and purchase reconciliation.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Iterable
from urllib.parse import urlparse

import httpx


DEFAULT_BASE_URL = "https://v1.apifansly.com/api/fansly"
_USAGE_WINDOW_SECONDS = 24 * 60 * 60
_USAGE_EVENTS: deque[dict[str, Any]] = deque(maxlen=50_000)
_USAGE_LOCK = threading.Lock()
_USAGE_STARTED_AT = time.time()


class ApiFanslyConfigurationError(RuntimeError):
    """The deployment does not have a usable API Fansly configuration."""


class ApiFanslyAccountAccessError(RuntimeError):
    """The configured key cannot access the creator's stored account connection."""


class ApiFanslyProtocolError(RuntimeError):
    """The upstream response was successful HTTP but not the documented shape."""


class ApiFanslyTransientError(RuntimeError):
    """Upstream refused this request for a reason that is expected to pass.

    Raised only when the server answered with a rate-limit or unavailability
    status, which means the request was definitively *not* processed. That is a
    different fact from an authentication failure (which will fail identically
    until a human reconnects the account) and from a network timeout (where the
    platform may have accepted the request and we cannot know).

    Callers on an exactly-once path must treat a timeout as ambiguous even
    though it is also "transient"; only this class states that nothing
    happened upstream.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.retry_after_seconds = retry_after_seconds


# Statuses where the server answered and told us it did not process the request.
TRANSIENT_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

# ---------------------------------------------------------------------------
# PERF-006 — one connection pool for the whole process.
#
# Every helper below used to fall back to ``httpx.AsyncClient()`` when no client
# was injected, so a single ``list_chat_messages`` or ``send_message`` paid for
# a fresh TCP connection and TLS handshake and then threw the pool away. The
# client is now created once and reused, which is safe because nothing about it
# is per-creator: authentication is a request header built by ``headers()`` from
# one deployment-wide key, and httpx carries no cross-request state between
# concurrent callers.
#
# ``follow_redirects`` stays False, exactly as the per-call clients had it, so a
# redirect cannot carry the x-api-key header to another host. The two helpers
# that legitimately follow redirects ask for it per request.
# ---------------------------------------------------------------------------

_SHARED_CLIENT_LOCK = threading.Lock()
_shared_client: httpx.AsyncClient | None = None

# Sized for the vault and chat reconciliation fan-out without letting one
# process open an unbounded number of sockets against the provider.
SHARED_CLIENT_LIMITS = httpx.Limits(
    max_connections=64,
    max_keepalive_connections=32,
    keepalive_expiry=60.0,
)


def shared_client() -> httpx.AsyncClient:
    """Return the process-wide pooled client, creating it on first use."""

    global _shared_client
    client = _shared_client
    if client is not None and not client.is_closed:
        return client
    with _SHARED_CLIENT_LOCK:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.AsyncClient(
                limits=SHARED_CLIENT_LIMITS,
                follow_redirects=False,
            )
        return _shared_client


def set_shared_client(client: httpx.AsyncClient | None) -> None:
    """Install a client for tests, or clear it so the next call rebuilds one.

    Does not close whatever was there: the caller owns anything it installed.
    """

    global _shared_client
    with _SHARED_CLIENT_LOCK:
        _shared_client = client


async def close_shared_client() -> None:
    """Close the pooled client. Called once from the application's shutdown."""

    global _shared_client
    with _SHARED_CLIENT_LOCK:
        client = _shared_client
        _shared_client = None
    if client is not None and not client.is_closed:
        await client.aclose()


@asynccontextmanager
async def client_scope() -> AsyncIterator[httpx.AsyncClient]:
    """Yield the shared client for a block that used to own a private one.

    Deliberately does not close on exit — the pool outlives the request.
    """

    yield shared_client()


def _record_usage(
    response: httpx.Response,
    *,
    operation: str,
    account_id: str | None,
) -> None:
    """Record a bounded, secret-free API usage event for diagnostics."""
    now = time.time()
    try:
        request = response.request
        method = str(request.method or "")
        path = str(request.url.path or "")
    except RuntimeError:
        method = ""
        path = ""
    event = {
        "at": now,
        "operation": str(operation or "unknown"),
        "account_id": str(account_id or ""),
        "method": method,
        "path": path,
        "status": int(response.status_code),
        "response_bytes": len(response.content or b""),
    }
    with _USAGE_LOCK:
        _USAGE_EVENTS.append(event)
        cutoff = now - _USAGE_WINDOW_SECONDS
        while _USAGE_EVENTS and _USAGE_EVENTS[0]["at"] < cutoff:
            _USAGE_EVENTS.popleft()
        total_24h = len(_USAGE_EVENTS)
    print(
        f"[APIFANSLY USAGE] operation={event['operation']} "
        f"account={event['account_id'] or 'none'} method={event['method']} "
        f"status={event['status']} bytes={event['response_bytes']} "
        f"calls_24h={total_24h}"
    )


def usage_snapshot() -> dict[str, Any]:
    """Return the current process's rolling 24-hour API usage summary."""
    now = time.time()
    cutoff = now - _USAGE_WINDOW_SECONDS
    with _USAGE_LOCK:
        while _USAGE_EVENTS and _USAGE_EVENTS[0]["at"] < cutoff:
            _USAGE_EVENTS.popleft()
        events = list(_USAGE_EVENTS)
    by_operation = Counter(event["operation"] for event in events)
    by_account = Counter(
        event["account_id"] or "unbound"
        for event in events
    )
    by_status = Counter(str(event["status"]) for event in events)
    return {
        "window_hours": 24,
        "process_started_at": _USAGE_STARTED_AT,
        "calls": len(events),
        "response_bytes": sum(event["response_bytes"] for event in events),
        "by_operation": dict(sorted(by_operation.items())),
        "by_account": dict(sorted(by_account.items())),
        "by_status": dict(sorted(by_status.items())),
        "note": (
            "This counts HTTP calls observed by the current backend process. "
            "Provider credits may be higher for large payloads."
        ),
    }


def api_key() -> str:
    value = str(os.environ.get("APIFANSLY_API_KEY") or "").strip()
    if not value:
        raise ApiFanslyConfigurationError("APIFANSLY_API_KEY is not configured")
    return value


def base_url() -> str:
    value = str(os.environ.get("APIFANSLY_BASE_URL") or DEFAULT_BASE_URL).strip()
    value = value.rstrip("/")
    if not value.startswith("https://") or not value.endswith("/api/fansly"):
        raise ApiFanslyConfigurationError(
            "APIFANSLY_BASE_URL must end with /api/fansly"
        )
    return value


def url(path: str = "") -> str:
    suffix = str(path or "").strip("/")
    return f"{base_url()}/{suffix}" if suffix else base_url()


def headers(*, json_content: bool = False) -> dict[str, str]:
    result = {"x-api-key": api_key()}
    if json_content:
        result["Content-Type"] = "application/json"
    return result


def is_fansly_cdn_url(value: str) -> bool:
    """Return whether a URL is an HTTPS Fansly CDN asset.

    Vault locations are signed and may be rejected when fetched directly from
    application infrastructure.  Only known Fansly hosts may be sent to the
    managed media-download proxy.
    """
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    host = str(parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and (host == "fansly.com" or host.endswith(".fansly.com"))
    )


def response_message(response: httpx.Response) -> str:
    try:
        payload: Any = response.json()
    except Exception:
        payload = None
    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if value:
                return str(value)
    return response.text[:200] or f"HTTP {response.status_code}"


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Read a Retry-After header, seconds form only."""

    raw = str(response.headers.get("retry-after") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        # HTTP-date form. Honouring it needs a clock comparison we do not need
        # here; the caller's own backoff is a safe substitute.
        return None
    return seconds if seconds >= 0 else None


def raise_for_response(
    response: httpx.Response,
    *,
    operation: str,
    account_id: str | None = None,
) -> None:
    _record_usage(
        response,
        operation=operation,
        account_id=account_id,
    )
    if response.is_success:
        return
    message = response_message(response)
    if response.status_code in {401, 403}:
        target = f" for account {account_id}" if account_id else ""
        raise ApiFanslyAccountAccessError(
            f"API Fansly access denied{target} during {operation}: {message}. "
            "Reconnect this creator under the current APIFANSLY_API_KEY."
        )
    if response.status_code in TRANSIENT_STATUS_CODES:
        # The server answered, so the request was definitively not processed.
        # Distinguishing this from an invalid key matters: a rate limit clears
        # by itself, while a disconnected account never does, and treating them
        # alike is what freezes a fan for a passing 503.
        raise ApiFanslyTransientError(
            f"API Fansly {operation} is temporarily unavailable "
            f"(HTTP {response.status_code}): {message}",
            status_code=response.status_code,
            retry_after_seconds=_retry_after_seconds(response),
        )
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            f"API Fansly {operation} failed: {message}",
            request=exc.request,
            response=exc.response,
        ) from exc


def response_data(payload: Any) -> Any:
    """Return the documented ``data.data.response`` value."""
    if not isinstance(payload, dict):
        return None
    outer = payload.get("data")
    if not isinstance(outer, dict):
        return None
    inner = outer.get("data")
    if not isinstance(inner, dict):
        return None
    return inner.get("response")


def response_cursor(payload: Any, *, response: Any | None = None) -> str | None:
    """Normalize cursors used by chat, message, follower, and vault endpoints."""
    if isinstance(response, dict):
        for key in ("cursor", "nextCursor"):
            if response.get(key) is not None and response.get(key) != "":
                return str(response[key])
    if not isinstance(payload, dict):
        return None
    outer = payload.get("data")
    if not isinstance(outer, dict):
        return None
    value = outer.get("nextCursor")
    return str(value) if value is not None and value != "" else None


def media_references(
    media_ids: Iterable[str],
    *,
    preview_ids: dict[str, str | None] | None = None,
) -> list[dict[str, str | None]]:
    """Build the documented media attachment objects for message sends."""
    seen: set[str] = set()
    references: list[dict[str, str | None]] = []
    previews = preview_ids or {}
    for raw in media_ids:
        media_id = str(raw or "").strip()
        if not media_id or media_id in seen:
            continue
        seen.add(media_id)
        references.append({
            "mediaId": media_id,
            "previewId": (
                str(previews[media_id])
                if previews.get(media_id)
                else None
            ),
        })
    return references


def message_payload(
    *,
    content: str,
    media_ids: Iterable[str] = (),
    preview_ids: dict[str, str | None] | None = None,
    price_dollars: float | None = None,
) -> dict[str, Any]:
    """Build a free-text, free-media, or locked-PPV message payload."""
    payload: dict[str, Any] = {"content": str(content or "")}
    media = media_references(media_ids, preview_ids=preview_ids)
    if media:
        payload["mediaIds"] = media
    if price_dollars is not None:
        price = round(float(price_dollars), 2)
        if not media:
            raise ValueError("PPV messages require at least one media item")
        if price < 1 or price > 500:
            raise ValueError("PPV price must be between $1 and $500")
        payload["access_type"] = ["ppv"]
        payload["price"] = price
    return payload


def sent_message_id(payload: Any) -> str | None:
    """Extract the message ID from the documented send-message response."""
    response = response_data(payload)
    if not isinstance(response, dict):
        return None
    value = response.get("id")
    return str(value) if value is not None and value != "" else None


def sent_attachment_ids(payload: Any) -> list[str]:
    """Extract the account-media bundle IDs returned by a successful send."""
    response = response_data(payload)
    if not isinstance(response, dict):
        return []
    result: list[str] = []
    for attachment in response.get("attachments") or []:
        if not isinstance(attachment, dict):
            continue
        content_id = str(attachment.get("contentId") or "").strip()
        if content_id and content_id not in result:
            result.append(content_id)
    return result


def account_media_prices(row: dict[str, Any]) -> list[float]:
    """Return every price Fansly exposes for an account-media item.

    API Fansly responses are not consistent about where the PPV price lives:
    some put it on ``accountMedia.price`` while others put it in the nested
    permission flags (occasionally inside JSON-encoded metadata).
    """
    prices: list[float] = []

    def collect(value: Any, *, price_key: bool = False) -> None:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0:1] in {"{", "["}:
                try:
                    collect(json.loads(stripped))
                    return
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass
            if price_key:
                try:
                    prices.append(float(stripped))
                except (TypeError, ValueError):
                    pass
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if price_key:
                prices.append(float(value))
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                collect(nested, price_key=str(key).lower() == "price")
            return
        if isinstance(value, list):
            for nested in value:
                collect(nested)

    collect({"price": row.get("price")})
    collect(row.get("permissions") or {})
    unique: list[float] = []
    for price in prices:
        if price not in unique:
            unique.append(price)
    return unique


def ppv_delivery_evidence(
    messages: Iterable[dict[str, Any]],
    account_media: Iterable[dict[str, Any]],
    *,
    message_id: str,
    expected_media_ids: Iterable[str],
    expected_price_cents: int,
) -> dict[str, Any]:
    """Validate that a sent message is actually locked by Fansly.

    A successful send response only proves that Fansly accepted the message.
    The documented chat-message read response is the authoritative place where
    the resulting ``accountMedia`` price and original ``mediaId`` are exposed.
    Prices have appeared as both dollars and cents across API Fansly responses,
    so comparison accepts either representation but never accepts zero.
    """
    sent = next(
        (
            row
            for row in messages
            if str(row.get("id") or "") == str(message_id)
        ),
        None,
    )
    if not sent:
        return {"verified": False, "reason": "message_not_visible"}

    attachment_ids = {
        str(attachment.get("contentId") or "")
        for attachment in (sent.get("attachments") or [])
        if isinstance(attachment, dict) and attachment.get("contentId")
    }
    if not attachment_ids:
        return {"verified": False, "reason": "message_has_no_media"}

    matched = [
        row
        for row in account_media
        if isinstance(row, dict)
        and (
            str(row.get("id") or "") in attachment_ids
            or str(row.get("mediaId") or "") in attachment_ids
        )
    ]
    if not matched:
        return {
            "verified": False,
            "reason": "account_media_not_visible",
            "attachment_ids": sorted(attachment_ids),
            "attachment_count": len(attachment_ids),
        }

    expected_ids = {
        str(value).strip()
        for value in expected_media_ids
        if str(value).strip()
    }
    actual_ids = {
        str(row.get("mediaId") or row.get("id") or "").strip()
        for row in matched
        if row.get("mediaId") or row.get("id")
    }
    if expected_ids and not expected_ids.issubset(actual_ids):
        return {
            "verified": False,
            "reason": "media_mismatch",
            "actual_media_ids": sorted(actual_ids),
        }

    expected_cents = int(expected_price_cents)
    required_rows = [
        row
        for row in matched
        if str(row.get("mediaId") or row.get("id") or "").strip() in expected_ids
    ] if expected_ids else matched
    raw_prices = [
        price
        for row in required_rows
        for price in account_media_prices(row)
    ]
    positive_prices_by_media = [
        [price for price in account_media_prices(row) if price > 0]
        for row in required_rows
    ]
    if (
        not required_rows
        or not positive_prices_by_media
        or any(not prices for prices in positive_prices_by_media)
    ):
        return {
            "verified": False,
            "reason": "media_is_not_payment_gated",
            "raw_prices": raw_prices,
        }

    def _matches_expected(raw_price: float) -> bool:
        return (
            abs(raw_price - expected_cents) < 0.01
            or abs((raw_price * 100) - expected_cents) < 0.01
        )

    if any(
        not any(_matches_expected(price) for price in prices)
        for prices in positive_prices_by_media
    ):
        return {
            "verified": False,
            "reason": "price_mismatch",
            "raw_prices": raw_prices,
        }

    return {
        "verified": True,
        "reason": "locked_ppv_confirmed",
        "actual_media_ids": sorted(actual_ids),
        "raw_prices": raw_prices,
    }


# A read may be repeated freely; a write may not. Automatic retry is therefore
# decided by the HTTP method, never by how transient the failure looked.
IDEMPOTENT_METHODS = frozenset({"GET", "HEAD"})

# Deliberately small. This exists to ride out a rate limit or a single bad
# gateway, not to become a generic retry framework, and every attempt is a real
# provider call that costs credits.
MAX_READ_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 0.5
_RETRY_MAX_SECONDS = 8.0


def _retry_delay(attempt: int, advertised: float | None) -> float:
    if advertised is not None:
        return min(max(advertised, 0.0), _RETRY_MAX_SECONDS)
    ceiling = min(_RETRY_BASE_SECONDS * (2 ** attempt), _RETRY_MAX_SECONDS)
    return random.uniform(0.0, ceiling)


async def request(
    method: str,
    path: str,
    *,
    operation: str,
    account_id: str | None = None,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
    files: Any = None,
    timeout: float = 30,
    client: httpx.AsyncClient | None = None,
    follow_redirects: bool | None = None,
    retry_idempotent: bool = True,
) -> dict[str, Any]:
    """Execute one API Fansly request with uniform errors and JSON validation.

    Uses the process-wide connection pool unless a client is injected, so a
    single call no longer pays for its own TLS handshake (PERF-006).

    A GET or HEAD is repeated after a transient refusal or a network failure,
    because repeating a read cannot have a side effect. Every other method is
    attempted exactly once and its error is raised to the caller, which is the
    only layer that knows whether the platform may already have accepted the
    write. Timeouts on a send are ambiguous by nature and must go through the
    delivery journal, not through a retry here.
    """

    active_client = client if client is not None else shared_client()
    upper_method = str(method or "").upper()
    attempts = (
        MAX_READ_ATTEMPTS
        if retry_idempotent and upper_method in IDEMPOTENT_METHODS
        else 1
    )

    for attempt in range(attempts):
        try:
            response = await active_client.request(
                method,
                url(path),
                headers=headers(json_content=json is not None),
                params=params,
                json=json,
                files=files,
                timeout=timeout,
                # None means "whatever this client was built with", so an
                # injected client keeps its own redirect policy.
                follow_redirects=(
                    httpx.USE_CLIENT_DEFAULT
                    if follow_redirects is None
                    else follow_redirects
                ),
            )
            raise_for_response(
                response,
                operation=operation,
                account_id=account_id,
            )
        except (ApiFanslyTransientError, httpx.TransportError) as exc:
            if attempt + 1 >= attempts:
                raise
            advertised = getattr(exc, "retry_after_seconds", None)
            delay = _retry_delay(attempt, advertised)
            print(
                f"[APIFANSLY RETRY] operation={operation} "
                f"attempt={attempt + 1}/{attempts} in {delay:.2f}s: {exc}"
            )
            await asyncio.sleep(delay)
            continue

        try:
            payload = response.json()
        except Exception as exc:
            raise ApiFanslyProtocolError(
                f"API Fansly {operation} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ApiFanslyProtocolError(
                f"API Fansly {operation} returned a non-object response"
            )
        return payload

    # Unreachable: the loop either returns or re-raises on its last attempt.
    raise ApiFanslyProtocolError(f"API Fansly {operation} produced no response")


async def download_media(
    cdn_url: str,
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 45,
) -> bytes:
    """Download a protected Fansly CDN asset through the documented proxy.

    Unlike the regular API helpers, this endpoint returns binary content rather
    than the usual JSON envelope.
    """
    if not is_fansly_cdn_url(cdn_url):
        raise ValueError("media download requires an HTTPS Fansly CDN URL")

    active_client = client if client is not None else shared_client()
    # No try/finally: the pool is process-wide and deliberately outlives this
    # call, and an injected client belongs to whoever injected it.
    response = await active_client.post(
        url("media/download"),
        headers=headers(json_content=True),
        json={"cdnUrl": cdn_url},
        timeout=timeout,
        follow_redirects=True,
    )
    raise_for_response(response, operation="protected media download")
    content_type = str(response.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        raise ApiFanslyProtocolError(
            "API Fansly media download returned JSON instead of media: "
            + response_message(response)
        )
    if len(response.content) <= 1000:
        raise ApiFanslyProtocolError(
            "API Fansly media download returned an empty or truncated file"
        )
    return bytes(response.content)


async def send_message(
    account_id: str,
    chat_id: str,
    *,
    content: str,
    media_ids: Iterable[str] = (),
    preview_ids: dict[str, str | None] | None = None,
    price_dollars: float | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Send one text/media/PPV message using the documented request contract."""
    return await request(
        "POST",
        f"{account_id}/chats/{chat_id}/messages",
        operation="message delivery",
        account_id=account_id,
        json=message_payload(
            content=content,
            media_ids=media_ids,
            preview_ids=preview_ids,
            price_dollars=price_dollars,
        ),
        timeout=15,
        client=client,
    )


async def delete_message(
    account_id: str,
    message_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> bool:
    """Delete a sent message through the documented compensation endpoint."""
    await request(
        "DELETE",
        f"{account_id}/messages/{message_id}",
        operation="message deletion",
        account_id=account_id,
        timeout=15,
        client=client,
    )
    return True


async def list_chats(
    account_id: str,
    *,
    cursor: str | None = None,
    filter: str = "all",
    sort: str = "newest",
    search: str | None = None,
    subscription_tier_id: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"filter": filter, "sort": sort}
    if cursor:
        params["cursor"] = cursor
    if search:
        params["search"] = search
    if subscription_tier_id:
        params["subscriptionTierId"] = subscription_tier_id
    payload = await request(
        "GET",
        f"{account_id}/chats",
        operation="chat listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError("API Fansly chat listing response is invalid")
    chats = response.get("data")
    aggregation = response.get("aggregationData")
    return (
        chats if isinstance(chats, list) else [],
        aggregation.get("accounts", [])
        if isinstance(aggregation, dict)
        and isinstance(aggregation.get("accounts"), list)
        else [],
        response_cursor(payload, response=response),
    )


async def list_chat_messages(
    account_id: str,
    chat_id: str,
    *,
    cursor: str | None = None,
    limit: int = 10,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"limit": max(1, min(10, int(limit)))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/chats/{chat_id}/messages",
        operation="chat message listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError(
            "API Fansly chat message listing response is invalid"
        )
    messages = response.get("messages")
    account_media = response.get("accountMedia")
    return (
        messages if isinstance(messages, list) else [],
        account_media if isinstance(account_media, list) else [],
        response_cursor(payload, response=response),
    )


async def list_vault_albums(
    account_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    payload = await request(
        "GET",
        f"{account_id}/vault/albums",
        operation="vault album listing",
        account_id=account_id,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError(
            "API Fansly vault album listing response is invalid"
        )
    albums = response.get("albums")
    return albums if isinstance(albums, list) else []


async def list_vault_album_media(
    account_id: str,
    album_id: str,
    *,
    cursor: str | None = None,
    limit: int = 50,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/vault/albums/{album_id}/media",
        operation="vault album media listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if isinstance(response, list):
        items = response
    elif isinstance(response, dict):
        raw = response.get("data") or response.get("media") or []
        items = raw if isinstance(raw, list) else []
    else:
        raise ApiFanslyProtocolError(
            "API Fansly vault album media response is invalid"
        )
    return items, response_cursor(payload, response=response)


# --- Account lists -----------------------------------------------------------
#
# Agencies build Lists (VIP, Whales, Buyers, Re-engage, ...) directly on the
# creator's Fansly account. Cleopatra mirrors them read-only; it never creates,
# renames or deletes a remote list.
#
# The path segments follow the same {accountId}/<resource> shape as every other
# endpoint in this module, and are overridable without a deploy so a
# documentation change does not require one.
_LISTS_PATH = "lists"
_LIST_ITEMS_PATH = "items"

# Remote payloads vary in which key carries the collection and which carries the
# identifier, exactly as they do for vault albums and followers above. Parsing
# stays tolerant rather than asserting one shape.
_LIST_COLLECTION_KEYS = ("lists", "items", "data", "accountLists")
_LIST_ID_KEYS = ("id", "listId", "_id")
_LIST_NAME_KEYS = ("label", "name", "title")
_LIST_ITEM_COLLECTION_KEYS = ("items", "listItems", "accounts", "data", "members")
_LIST_MEMBER_ID_KEYS = ("accountId", "itemId", "id", "userId", "followerId")


def _lists_path() -> str:
    return str(os.environ.get("APIFANSLY_LISTS_PATH") or _LISTS_PATH).strip("/")


def _list_items_path() -> str:
    return str(
        os.environ.get("APIFANSLY_LIST_ITEMS_PATH") or _LIST_ITEMS_PATH
    ).strip("/")


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _collection(response: Any, keys: Iterable[str]) -> list[Any]:
    """Return the collection from a list-shaped or object-shaped response."""
    if isinstance(response, list):
        return response
    if not isinstance(response, dict):
        return []
    for key in keys:
        value = response.get(key)
        if isinstance(value, list):
            return value
    return []


def parse_account_lists(response: Any) -> list[dict[str, Any]]:
    """Normalize one page of remote lists into id/name/count records.

    A list with no usable remote identifier is dropped rather than mirrored:
    the remote id is the only stable mapping key, and a mirror without one would
    duplicate itself on the next sync.
    """
    parsed: list[dict[str, Any]] = []
    for row in _collection(response, _LIST_COLLECTION_KEYS):
        if not isinstance(row, dict):
            continue
        external_id = _first_value(row, _LIST_ID_KEYS)
        if not external_id:
            continue
        name = _first_value(row, _LIST_NAME_KEYS) or f"Fansly list {external_id}"
        try:
            item_count = int(row.get("itemCount") or row.get("count") or 0)
        except (TypeError, ValueError):
            item_count = 0
        parsed.append(
            {
                "external_list_id": external_id,
                "name": name,
                "item_count": max(item_count, 0),
            }
        )
    return parsed


def parse_list_member_ids(response: Any) -> list[str]:
    """Normalize one page of list membership into platform account IDs.

    Members are identified only by their Fansly account ID. Usernames and
    display names are deliberately ignored: they are mutable and not unique.
    """
    member_ids: list[str] = []
    seen: set[str] = set()
    for row in _collection(response, _LIST_ITEM_COLLECTION_KEYS):
        if isinstance(row, (str, int)):
            account_id = str(row).strip()
        elif isinstance(row, dict):
            account_id = _first_value(row, _LIST_MEMBER_ID_KEYS) or ""
        else:
            continue
        if account_id and account_id not in seen:
            seen.add(account_id)
            member_ids.append(account_id)
    return member_ids


async def list_account_lists(
    account_id: str,
    *,
    cursor: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """Return one page of the creator's own Fansly lists."""
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/{_lists_path()}",
        operation="account list listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if response is None:
        raise ApiFanslyProtocolError("API Fansly account list response is invalid")
    return parse_account_lists(response), response_cursor(payload, response=response)


async def list_account_list_members(
    account_id: str,
    list_id: str,
    *,
    cursor: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> tuple[list[str], str | None]:
    """Return one page of Fansly account IDs belonging to a remote list."""
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/{_lists_path()}/{list_id}/{_list_items_path()}",
        operation="account list member listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    if response is None:
        raise ApiFanslyProtocolError(
            "API Fansly account list member response is invalid"
        )
    return parse_list_member_ids(response), response_cursor(
        payload,
        response=response,
    )


async def list_followers(
    account_id: str,
    *,
    cursor: str | None = None,
    limit: int = 100,
    client: httpx.AsyncClient | None = None,
) -> tuple[Any, str | None]:
    params: dict[str, Any] = {"limit": max(1, int(limit))}
    if cursor:
        params["cursor"] = cursor
    payload = await request(
        "GET",
        f"{account_id}/followers",
        operation="follower listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    return response, response_cursor(payload, response=response)


async def list_subscribers(
    account_id: str,
    *,
    status: str = "all",
    cursor: str | None = None,
    limit: int = 100,
    search: str | None = None,
    subscription_tier_ids: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> tuple[Any, str | None]:
    if status not in {"all", "active", "expired"}:
        raise ValueError("subscriber status must be all, active, or expired")
    params: dict[str, Any] = {
        "status": status,
        "limit": max(1, int(limit)),
    }
    if cursor:
        params["cursor"] = cursor
    if search:
        params["search"] = search
    if subscription_tier_ids:
        params["subscriptionTierIds"] = subscription_tier_ids
    payload = await request(
        "GET",
        f"{account_id}/subscribers",
        operation="subscriber listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    response = response_data(payload)
    return response, response_cursor(payload, response=response)


async def top_supporters(
    account_id: str,
    *,
    before_ms: int | None = None,
    after_ms: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> Any:
    params: dict[str, Any] = {}
    if before_ms is not None:
        params["before"] = int(before_ms)
    if after_ms is not None:
        params["after"] = int(after_ms)
    payload = await request(
        "GET",
        f"{account_id}/top-supporters",
        operation="top supporter listing",
        account_id=account_id,
        params=params,
        client=client,
    )
    return response_data(payload)


async def current_account(
    account_id: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    payload = await request(
        "GET",
        f"{account_id}/me",
        operation="account profile load",
        account_id=account_id,
        client=client,
    )
    response = response_data(payload)
    if not isinstance(response, dict):
        raise ApiFanslyProtocolError("API Fansly account response is invalid")
    return response
