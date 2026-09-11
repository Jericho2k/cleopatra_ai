"""The PostgREST transport incident: HTTP/2 GOAWAY taking the process with it.

Production symptom: four unrelated background loops and an inbound
``GET /my-creators`` all failing in the same instant with one message —

    <ConnectionTerminated error_code:0, last_stream_id:3, additional_data:None>

The first three tests here are not about our code at all. They run a real HTTP/2
server, a real ``h2`` state machine and a real socket, and establish the two
facts the fix rests on:

* a GOAWAY delivered while requests are in flight fails EVERY multiplexed
  stream on that connection, and httpx does not retry any of them;
* HTTP/1.1 does not have that failure mode, because httpcore checks an idle
  socket before reusing it and because one connection carries one request.

Everything after that tests our own behaviour on top of that transport.
"""
from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from core import supabase as supabase_module
from services import db_reliability
from tests.h2_probe_server import (
    h2_server_that_goaways_in_flight,
    h2_server_that_holds_until,
    http11_server_that_recycles_connections,
)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def clean_transport():
    """Never leave a built client behind for another test module."""
    supabase_module.close_supabase_client()
    yield
    supabase_module.close_supabase_client()


# ---------------------------------------------------------------------------
# 1. The diagnosis, against a real HTTP/2 peer.
# ---------------------------------------------------------------------------


def test_http2_goaway_on_an_in_flight_stream_reaches_the_caller():
    """The production error string, produced by h2 rather than by a fixture."""
    server = h2_server_that_goaways_in_flight(streams_before_goaway=2)
    client = httpx.Client(http1=False, http2=True)
    try:
        first = client.get(f"{server.base_url}/rest/v1/creators")
        assert first.http_version == "HTTP/2"

        with pytest.raises(httpx.RemoteProtocolError) as caught:
            client.get(f"{server.base_url}/rest/v1/creators")

        message = str(caught.value)
        assert "ConnectionTerminated" in message
        assert "error_code:0" in message
        assert "last_stream_id:3" in message
        # And this is what nothing in supabase-py, postgrest or httpx retries.
        assert db_reliability.is_transient_db_error(caught.value)

        # The pool itself does recover: the next request opens a new connection.
        # So the outage was never "the client is poisoned" — it was "one
        # round trip died and nobody asked again".
        third = client.get(f"{server.base_url}/rest/v1/creators")
        assert third.json() == [{"connection": 2}]
    finally:
        client.close()
        server.close()


def test_http2_goaway_fails_every_concurrent_request_on_the_connection():
    """Why four unrelated loops broke at the same instant.

    HTTP/2 multiplexes, so one process-wide client means one connection, and one
    GOAWAY is every caller's problem simultaneously.
    """
    gate = threading.Event()
    server = h2_server_that_holds_until(gate)
    client = httpx.Client(http1=False, http2=True)
    results: dict[int, str] = {}

    def call(index: int) -> None:
        try:
            client.get(f"{server.base_url}/rest/v1/scheduled_actions")
            results[index] = "ok"
        except Exception as exc:  # noqa: BLE001 - the point of the test
            results[index] = type(exc).__name__

    try:
        threads = [threading.Thread(target=call, args=(i,)) for i in range(6)]
        for thread in threads:
            thread.start()
        # Give every request time to land on the one shared connection.
        threading.Event().wait(0.6)
        gate.set()
        for thread in threads:
            thread.join(timeout=20)

        assert len(results) == 6
        assert set(results.values()) == {"RemoteProtocolError"}
    finally:
        gate.set()
        client.close()
        server.close()


def test_http11_transport_absorbs_the_same_peer_recycling():
    """The fix, against a peer that recycles connections just as aggressively.

    Same server behaviour, HTTP/1.1 instead: every request succeeds. httpcore
    detects that an idle socket has been closed before handing it out
    (``http11.py``, ``has_expired``) — the check the HTTP/2 path does not have.
    """
    server = http11_server_that_recycles_connections()
    client = supabase_module.build_http_client()
    try:
        assert client._transport._pool._http2 is False
        for _ in range(6):
            threading.Event().wait(0.12)
            response = client.get(f"{server.base_url}/rest/v1/creators")
            assert response.status_code == 200
            assert response.http_version == "HTTP/1.1"
    finally:
        client.close()
        server.close()


def test_supabase_client_is_built_on_the_configured_transport():
    """supabase-py's own default is ``http2=True``; ours is not, and it is shared.

    Pinning this matters because the default lives three dependencies down
    (``postgrest/_sync/client.py``) where an upgrade can silently change it back.
    """
    client = supabase_module.get_supabase()
    session = client.postgrest.session
    pool = session._transport._pool

    assert pool._http2 is False
    assert pool._http1 is True
    # Redirect-following has to be preserved: postgrest sets it on the session
    # it builds itself, and an injected client replaces that session wholesale.
    assert session.follow_redirects is True
    # One transport for the whole SDK, not one per sub-client.
    assert client.auth._http_client is session

    limits = supabase_module.transport_limits()
    assert limits.max_connections == supabase_module.DEFAULT_MAX_CONNECTIONS
    assert limits.keepalive_expiry == supabase_module.DEFAULT_KEEPALIVE_EXPIRY_SECONDS


def test_http2_can_be_turned_back_on_by_an_operator(monkeypatch):
    """The knob exists so reverting needs an env var, not a deploy."""
    monkeypatch.setenv("SUPABASE_HTTP2", "1")
    client = supabase_module.build_http_client()
    try:
        assert client._transport._pool._http2 is True
    finally:
        client.close()


def test_connect_retries_are_configured_and_cannot_replay_a_write():
    """``retries`` is safe for writes because it only wraps connection setup.

    httpcore catches ConnectError/ConnectTimeout inside ``_connect``, before any
    request bytes are written, so a retried attempt cannot have been applied.
    """
    client = supabase_module.build_http_client()
    try:
        connection_pool = client._transport._pool
        assert connection_pool._retries == supabase_module.DEFAULT_CONNECT_RETRIES
    finally:
        client.close()
