"""Serving platform operations ourselves, using the creator's own session.

What this is
------------
The first piece of the migration off the metered API Fansly provider. It serves
one operation today — reading a protected media asset — because that is the one
the provider charges most for and the one we can serve with the least risk.

The provider meters media transfer at 2 credits per megabyte. A 250 MB clip is
~500 credits on a single call, and vault classification reaches for exactly
those clips. Nothing about moving those bytes is hard; what the provider sells
is a live Fansly session to move them with. We already hold one per creator, in
``services.fansly_session_store``, so the bytes can be ours for the price of
bandwidth.

Two routes, cheapest first
--------------------------
1. **Authenticated CDN read.** Fetch the stored location carrying the account's
   session. This is the common case: the URL is still valid and the asset is
   merely protected.
2. **Signed-URL refresh.** Ask the API for the media item again, take the fresh
   signed location out of the answer, and fetch *that* with no credentials at
   all. This is the expired-URL case, and it is the route the paid proxy is
   really selling: the signature, not the bytes.

Both routes cost zero provider credits. What they cost instead is a request
against the creator's own session, which is the trade this module exists to
make explicit.

Failure is not an outage
------------------------
Every failure here raises ``DirectTransportError``, and the call site is
expected to fall back to the provider (see ``core.transport_policy``). A direct
transport that is having a bad day must degrade into a slightly larger invoice,
never into a fan waiting on a reply or a vault item that cannot be classified.
That is why nothing in this module retries aggressively and nothing swallows an
error into an empty result: a clean, fast failure is what makes the fallback
work.

Savings are recorded, not assumed
---------------------------------
Every successful direct transfer records what the provider *would* have billed.
Without that, a migration's whole justification stays a theory: the provider's
usage dashboard would simply show a smaller number, with no way to attribute
the difference to this code rather than to a quiet week. The ledger here is
deliberately separate from ``services.apifansly``'s — that one means "what we
were billed" and must keep meaning exactly that.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Protocol
from urllib.parse import urlparse


# Same rule the provider bills on, restated rather than imported so the savings
# ledger does not depend on the module it exists to retire.
CREDIT_MEDIA_CREDITS_PER_MB = 2.0
CREDIT_BYTES_PER_MB = 1024 * 1024

_SAVINGS_WINDOW_SECONDS = 24 * 60 * 60
_SAVINGS_EVENTS: deque[dict[str, Any]] = deque(maxlen=50_000)
_SAVINGS_LOCK = threading.Lock()

# A downloaded asset is held whole in memory and then handed to a classifier,
# so "free of provider credits" is not the same as "free". This ceiling is the
# memory-and-time bound that survives the credit bound going away.
DEFAULT_MAX_DIRECT_DOWNLOAD_MB = 512.0

# Below this, the response is an error page or a truncated transfer rather than
# an asset. Mirrors the provider client's own floor.
MIN_PLAUSIBLE_ASSET_BYTES = 1000


class DirectTransportError(RuntimeError):
    """A direct attempt failed. The caller should fall back to the provider."""


class DirectTransportUnavailable(DirectTransportError):
    """No usable session for this account, so no direct attempt was possible.

    Separate from a failed attempt because it means something different to an
    operator: an account that has never been connected, or whose session died,
    is a configuration problem they can fix, not a transport that is misbehaving.
    """


class SessionProvider(Protocol):
    """The slice of ``SessionStore`` this module needs.

    A Protocol rather than an import, so the transport does not reach back into
    the application that owns the store, and so a test can pass a fake without
    a database.
    """

    def get_client(self, account_id: str) -> Any: ...


_session_provider: SessionProvider | None = None


def set_session_provider(provider: SessionProvider | None) -> None:
    """Install the session source. Called once, at application startup."""
    global _session_provider
    _session_provider = provider


def session_provider() -> SessionProvider | None:
    return _session_provider


def client_for(account_id: str) -> Any:
    """A ready-to-use direct client for this account.

    Raises ``DirectTransportUnavailable`` rather than returning None, because
    every caller of this function is about to fall back to the paid provider
    and the reason belongs in the logs.
    """
    provider = _session_provider
    if provider is None:
        raise DirectTransportUnavailable(
            "direct transport has no session provider installed"
        )
    account = str(account_id or "").strip()
    if not account:
        raise DirectTransportUnavailable(
            "direct transport requires an account to borrow a session from"
        )
    try:
        return provider.get_client(account)
    except Exception as exc:
        raise DirectTransportUnavailable(
            f"no usable Fansly session for account {account}: {exc}"
        ) from exc


def is_fansly_host(value: str) -> bool:
    """Whether a URL points at an HTTPS Fansly-controlled host."""
    try:
        parsed = urlparse(str(value or "").strip())
    except ValueError:
        return False
    host = str(parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        host == "fansly.com" or host.endswith(".fansly.com")
    )


# --- Savings ledger ---------------------------------------------------------


def provider_credits_for_bytes(media_bytes: int | float | None) -> float:
    """What API Fansly would have billed to move this many bytes."""
    try:
        size = float(media_bytes or 0)
    except (TypeError, ValueError):
        return 0.0
    if size <= 0:
        return 0.0
    return max(1.0, CREDIT_MEDIA_CREDITS_PER_MB * size / float(CREDIT_BYTES_PER_MB))


def _prune_locked(now: float) -> None:
    cutoff = now - _SAVINGS_WINDOW_SECONDS
    while _SAVINGS_EVENTS and _SAVINGS_EVENTS[0]["at"] < cutoff:
        _SAVINGS_EVENTS.popleft()


def record_saving(
    *,
    operation: str,
    account_id: str | None,
    media_bytes: int,
    route: str,
) -> dict[str, Any]:
    """Record one transfer the provider did not bill for."""
    now = time.time()
    event = {
        "at": now,
        "operation": str(operation or "unknown"),
        "account_id": str(account_id or ""),
        "route": str(route or "unknown"),
        "media_bytes": max(0, int(media_bytes or 0)),
        "credits_saved": provider_credits_for_bytes(media_bytes),
    }
    with _SAVINGS_LOCK:
        _SAVINGS_EVENTS.append(event)
        _prune_locked(now)
    print(
        f"[DIRECT TRANSPORT] operation={event['operation']} "
        f"account={event['account_id'] or 'none'} route={event['route']} "
        f"bytes={event['media_bytes']} "
        f"credits_saved={event['credits_saved']:.2f}"
    )
    return event


def savings_snapshot() -> dict[str, Any]:
    """What the direct transport has saved in the last 24 hours."""
    now = time.time()
    with _SAVINGS_LOCK:
        _prune_locked(now)
        events = list(_SAVINGS_EVENTS)
    by_route: dict[str, float] = {}
    by_account: dict[str, float] = {}
    for event in events:
        by_route[event["route"]] = by_route.get(event["route"], 0.0) + event[
            "credits_saved"
        ]
        key = event["account_id"] or "unattributed"
        by_account[key] = by_account.get(key, 0.0) + event["credits_saved"]
    return {
        "window_hours": _SAVINGS_WINDOW_SECONDS / 3600,
        "transfers": len(events),
        "media_bytes": sum(event["media_bytes"] for event in events),
        "credits_saved": round(
            sum(event["credits_saved"] for event in events), 2
        ),
        "by_route": {key: round(value, 2) for key, value in by_route.items()},
        "by_account": {key: round(value, 2) for key, value in by_account.items()},
    }


def reset_savings_for_tests() -> None:
    with _SAVINGS_LOCK:
        _SAVINGS_EVENTS.clear()


# --- Media ------------------------------------------------------------------

ROUTE_AUTHENTICATED_CDN = "authenticated_cdn"
ROUTE_REFRESHED_URL = "refreshed_signed_url"


def _extract_media_locations(payload: Any) -> list[str]:
    """Every plausible asset location in an account-media response.

    Tolerant by design, exactly as the provider client's parsing is: the shape
    of this payload is observed from browser traffic rather than published, and
    a rename upstream should cost a missed refresh and a fallback, not an
    exception in a background loop.
    """
    found: list[str] = []
    seen: set[str] = set()

    def visit(node: Any, key_hint: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                visit(value, str(key))
            return
        if isinstance(node, list):
            for value in node:
                visit(value, key_hint)
            return
        if not isinstance(node, str):
            return
        hint = key_hint.lower()
        if hint not in {"location", "url", "src", "locationurl"}:
            return
        candidate = node.strip()
        if candidate and candidate not in seen and is_fansly_host(candidate):
            seen.add(candidate)
            found.append(candidate)

    visit(payload)
    return found


async def refresh_media_url(
    media_id: str,
    *,
    account_id: str,
    client: Any = None,
) -> str:
    """A freshly signed location for one media item.

    The expensive problem this solves cheaply: an expired signature is why a
    free CDN read fails, and re-signing is the only thing the metered proxy
    does that we cannot do with bytes alone. Asking the API for the item again
    costs one ordinary request against the creator's session and yields a URL
    that needs no credentials at all.
    """
    media = str(media_id or "").strip()
    if not media:
        raise DirectTransportError("a signed-URL refresh needs a media id")

    async def _fetch(active: Any) -> Any:
        return await active.get_account_media([media])

    try:
        if client is not None:
            payload = await _fetch(client)
        else:
            async with client_for(account_id) as active:
                payload = await _fetch(active)
    except DirectTransportError:
        raise
    except Exception as exc:
        raise DirectTransportError(
            f"signed-URL refresh failed for media {media}: {type(exc).__name__}"
        ) from exc

    locations = _extract_media_locations(payload)
    if not locations:
        raise DirectTransportError(
            f"signed-URL refresh returned no usable location for media {media}"
        )
    return locations[0]


async def download_media(
    cdn_url: str,
    *,
    account_id: str,
    media_id: str = "",
    timeout: float = 45.0,
    max_megabytes: float = DEFAULT_MAX_DIRECT_DOWNLOAD_MB,
    operation: str = "protected media download",
    client: Any = None,
) -> tuple[bytes, str]:
    """Read one protected asset without paying the provider. Returns (bytes, route).

    Tries the authenticated CDN read first and a signed-URL refresh second,
    because the first costs one request and the second costs two. Raises
    ``DirectTransportError`` on any failure so the caller can fall back.

    ``max_megabytes`` is a memory-and-time bound, not a cost bound. The
    provider's credit ceiling (``services.media_cost_guard``) does not apply to
    a transfer the provider is not billing, but "we are not paying per
    megabyte" is not a reason to pull an unbounded file into this process.
    """
    if not is_fansly_host(cdn_url):
        raise DirectTransportError(
            "direct media download requires an HTTPS Fansly CDN URL"
        )
    max_bytes = max(0.0, float(max_megabytes)) * CREDIT_BYTES_PER_MB

    def _validate(response: Any, route: str) -> bytes:
        status = int(getattr(response, "status_code", 0) or 0)
        if status != 200:
            raise DirectTransportError(
                f"direct media download returned HTTP {status} via {route}"
            )
        content_type = str(
            (getattr(response, "headers", {}) or {}).get("content-type") or ""
        ).lower()
        if "application/json" in content_type or "text/html" in content_type:
            raise DirectTransportError(
                f"direct media download returned {content_type} instead of "
                f"media via {route}"
            )
        content = bytes(getattr(response, "content", b"") or b"")
        if len(content) <= MIN_PLAUSIBLE_ASSET_BYTES:
            raise DirectTransportError(
                f"direct media download returned an empty or truncated file "
                f"via {route}"
            )
        if max_bytes and len(content) > max_bytes:
            raise DirectTransportError(
                f"direct media download exceeded the {max_megabytes:.0f} MB "
                f"in-process ceiling via {route}"
            )
        return content

    async def _attempt(active: Any) -> tuple[bytes, str]:
        first_error: Exception | None = None
        try:
            response = await active.download_asset(
                cdn_url, timeout=timeout, authenticated=True
            )
            return _validate(response, ROUTE_AUTHENTICATED_CDN), (
                ROUTE_AUTHENTICATED_CDN
            )
        except Exception as exc:
            first_error = exc

        if not str(media_id or "").strip():
            raise DirectTransportError(
                "direct media download failed and no media id was available "
                f"for a signed-URL refresh: {first_error}"
            ) from first_error

        fresh_url = await refresh_media_url(
            media_id, account_id=account_id, client=active
        )
        response = await active.download_asset(
            fresh_url, timeout=timeout, authenticated=False
        )
        return _validate(response, ROUTE_REFRESHED_URL), ROUTE_REFRESHED_URL

    try:
        if client is not None:
            content, route = await _attempt(client)
        else:
            async with client_for(account_id) as active:
                content, route = await _attempt(active)
    except DirectTransportError:
        raise
    except Exception as exc:
        raise DirectTransportError(
            f"direct media download failed: {type(exc).__name__}: {exc}"
        ) from exc

    record_saving(
        operation=operation,
        account_id=account_id,
        media_bytes=len(content),
        route=route,
    )
    return content, route
