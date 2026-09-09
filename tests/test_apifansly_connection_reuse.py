"""PERF-006 — one connection pool per process, not one per call.

Every helper in services/apifansly.py used to fall back to a private
``httpx.AsyncClient()`` and close it in a ``finally``, so a single
``list_chat_messages`` or ``send_message`` paid for its own TCP connection and
TLS handshake and then discarded the pool. These tests pin the reuse, the
shutdown, the injectability, and — the part that makes sharing safe — that two
concurrent creators cannot see each other's request state.

They also pin the Part 15 failure split: a read is retried after a transient
refusal, a write never is.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from services import apifansly
from services.apifansly import (
    ApiFanslyAccountAccessError,
    ApiFanslyTransientError,
    MAX_READ_ATTEMPTS,
    close_shared_client,
    request as apifansly_request,
    send_message,
    set_shared_client,
    shared_client,
)


@pytest.fixture(autouse=True)
def api_env(monkeypatch):
    monkeypatch.setenv("APIFANSLY_API_KEY", "test-key")
    monkeypatch.delenv("APIFANSLY_BASE_URL", raising=False)
    yield
    set_shared_client(None)


def _envelope(response=None):
    return {"data": {"data": {"response": response or {}}}}


# --- pooling ----------------------------------------------------------------


def test_repeated_calls_reuse_one_client_instance():
    set_shared_client(None)
    first = shared_client()
    second = shared_client()

    assert first is second
    assert first.is_closed is False


def test_the_pool_has_explicit_bounds():
    limits = apifansly.SHARED_CLIENT_LIMITS

    assert limits.max_connections == 64
    assert limits.max_keepalive_connections == 32


def test_a_call_with_no_injected_client_uses_the_shared_pool():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_envelope())

    pooled = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    set_shared_client(pooled)

    async def run():
        for _ in range(5):
            await apifansly_request("GET", "acct/chats", operation="chat listing")

    asyncio.run(run())

    assert len(seen) == 5
    # The shared client is still open: no helper closed the pool behind us.
    assert pooled.is_closed is False
    asyncio.run(pooled.aclose())


def test_an_injected_client_is_used_and_not_closed():
    set_shared_client(None)
    calls = {"count": 0}

    def handler(request):
        calls["count"] += 1
        return httpx.Response(200, json=_envelope())

    injected = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        await apifansly_request(
            "GET", "acct/chats", operation="chat listing", client=injected
        )

    asyncio.run(run())

    assert calls["count"] == 1
    assert injected.is_closed is False
    # And the injected client did not become the process-wide one.
    assert apifansly._shared_client is None
    asyncio.run(injected.aclose())


def test_shutdown_closes_the_pool_and_the_next_call_rebuilds_it():
    set_shared_client(None)
    first = shared_client()

    asyncio.run(close_shared_client())

    assert first.is_closed is True
    assert apifansly._shared_client is None

    rebuilt = shared_client()
    assert rebuilt is not first
    assert rebuilt.is_closed is False
    asyncio.run(close_shared_client())


def test_closing_twice_is_safe():
    set_shared_client(None)
    shared_client()
    asyncio.run(close_shared_client())
    asyncio.run(close_shared_client())


# --- isolation --------------------------------------------------------------


def test_concurrent_creators_do_not_share_request_state():
    """The client may be shared; the request context may not.

    Authentication is a per-request header and the account is a per-request
    path, so two creators running through one pool must still produce two
    distinct, correctly-addressed requests.
    """

    observed: list[tuple[str, str]] = []

    async def handler(request):
        # Yield inside the handler so the two requests genuinely interleave.
        await asyncio.sleep(0)
        observed.append((request.url.path, request.headers["x-api-key"]))
        return httpx.Response(200, json=_envelope())

    pooled = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    set_shared_client(pooled)

    async def run():
        await asyncio.gather(
            apifansly_request(
                "GET",
                "creator-a/chats",
                operation="chat listing",
                account_id="creator-a",
            ),
            apifansly_request(
                "GET",
                "creator-b/chats",
                operation="chat listing",
                account_id="creator-b",
            ),
        )

    asyncio.run(run())

    paths = sorted(path for path, _ in observed)
    assert paths == [
        "/api/fansly/creator-a/chats",
        "/api/fansly/creator-b/chats",
    ]
    assert {key for _, key in observed} == {"test-key"}
    asyncio.run(pooled.aclose())


def test_request_timeouts_are_preserved_through_the_shared_pool():
    seen: list[httpx.Timeout] = []

    def handler(request):
        seen.append(request.extensions.get("timeout"))
        return httpx.Response(200, json=_envelope())

    pooled = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    set_shared_client(pooled)

    async def run():
        await apifansly_request(
            "GET", "acct/chats", operation="chat listing", timeout=17
        )
        # send_message pins 15s; the shared pool must not widen it.
        await send_message("acct", "group", content="hi")

    asyncio.run(run())

    assert seen[0]["read"] == 17
    assert seen[1]["read"] == 15
    asyncio.run(pooled.aclose())


def test_an_injected_clients_redirect_policy_is_not_overridden():
    """Call sites that deliberately follow redirects keep doing so."""

    hops: list[str] = []

    def handler(request):
        hops.append(request.url.path)
        if len(hops) == 1:
            return httpx.Response(302, headers={"location": "/api/fansly/moved"})
        return httpx.Response(200, json=_envelope())

    redirecting = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    )

    async def run():
        await apifansly_request(
            "GET", "acct/chats", operation="chat listing", client=redirecting
        )

    asyncio.run(run())

    assert hops == ["/api/fansly/acct/chats", "/api/fansly/moved"]
    asyncio.run(redirecting.aclose())


# --- Part 15: transient vs permanent ----------------------------------------


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_transient_statuses_are_retried_for_reads(status):
    set_shared_client(None)
    attempts = {"count": 0}

    def handler(request):
        attempts["count"] += 1
        if attempts["count"] < MAX_READ_ATTEMPTS:
            return httpx.Response(status, json={"error": "later"})
        return httpx.Response(200, json=_envelope({"ok": True}))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        return await apifansly_request(
            "GET", "acct/chats", operation="chat listing", client=client
        )

    payload = asyncio.run(run())

    assert attempts["count"] == MAX_READ_ATTEMPTS
    assert payload["data"]["data"]["response"] == {"ok": True}
    asyncio.run(client.aclose())


def test_a_network_failure_is_retried_for_reads():
    attempts = {"count": 0}

    def handler(request):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(200, json=_envelope())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        await apifansly_request(
            "GET", "acct/chats", operation="chat listing", client=client
        )

    asyncio.run(run())

    assert attempts["count"] == 2
    asyncio.run(client.aclose())


def test_a_send_is_never_retried_after_a_timeout():
    """An ambiguous send outcome must reach the delivery journal, not a retry.

    The platform may have accepted a request that timed out, so repeating it
    here could deliver the same paid message twice.
    """

    attempts = {"count": 0}

    def handler(request):
        attempts["count"] += 1
        raise httpx.ReadTimeout("no answer", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        await send_message("acct", "group", content="hi", client=client)

    with pytest.raises(httpx.ReadTimeout):
        asyncio.run(run())

    assert attempts["count"] == 1
    asyncio.run(client.aclose())


def test_authentication_failure_is_not_treated_as_transient():
    """A rejected key fails identically forever; retrying it is pure cost."""

    attempts = {"count": 0}

    def handler(request):
        attempts["count"] += 1
        return httpx.Response(403, json={"error": "no access"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        await apifansly_request(
            "GET",
            "acct/chats",
            operation="chat listing",
            account_id="acct",
            client=client,
        )

    with pytest.raises(ApiFanslyAccountAccessError):
        asyncio.run(run())

    assert attempts["count"] == 1
    asyncio.run(client.aclose())


def test_a_permanent_client_error_is_not_retried():
    attempts = {"count": 0}

    def handler(request):
        attempts["count"] += 1
        return httpx.Response(400, json={"error": "malformed cursor"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        await apifansly_request(
            "GET", "acct/chats", operation="chat listing", client=client
        )

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())

    assert attempts["count"] == 1
    asyncio.run(client.aclose())


def test_retry_after_is_honoured_over_the_default_backoff(monkeypatch):
    delays: list[float] = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(apifansly.asyncio, "sleep", fake_sleep)

    attempts = {"count": 0}

    def handler(request):
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "2"},
                json={"error": "slow down"},
            )
        return httpx.Response(200, json=_envelope())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        await apifansly_request(
            "GET", "acct/chats", operation="chat listing", client=client
        )

    asyncio.run(run())

    assert delays == [2.0]
    asyncio.run(client.aclose())


def test_the_transient_error_carries_its_status_for_callers():
    def handler(request):
        return httpx.Response(503, json={"error": "busy"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        await send_message("acct", "group", content="hi", client=client)

    with pytest.raises(ApiFanslyTransientError) as exc:
        asyncio.run(run())

    assert exc.value.status_code == 503
    asyncio.run(client.aclose())
