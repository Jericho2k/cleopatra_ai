"""Supabase client — one process-wide instance over a transport we control.

Why this file is no longer three lines
--------------------------------------

It used to be ``@lru_cache`` around ``create_client(...)``, which meant the
transport underneath was whatever ``supabase-py`` builds by default. That
default is not neutral. ``postgrest`` constructs its session as::

    self.session = http_client or Client(..., http2=True)

— HTTP/2 is hardcoded on, and ``h2`` is installed here as a transitive
dependency, so it was genuinely negotiated in production.

HTTP/2 multiplexes every concurrent request onto ONE TCP connection. That is
normally the point of it. Here it was the blast radius.

The failure, measured
---------------------

Supabase's PostgREST edge recycles connections with a graceful
``GOAWAY`` (``ConnectionTerminated error_code:0`` — NO_ERROR, an orderly
shutdown, not a fault). ``httpcore`` handles that in two very different ways
depending on when it lands, and the difference is the whole bug
(``httpcore/_sync/http2.py``, ``_receive_events``)::

    if stream_id and last_stream_id and stream_id > last_stream_id:
        raise ConnectionNotAvailable()      # pool transparently retries
    raise RemoteProtocolError(self._connection_terminated)   # caller sees this

A GOAWAY seen while the connection is *idle* self-heals: our stream id is above
``last_stream_id``, the pool discards the connection and re-dispatches. A GOAWAY
that arrives while requests are *in flight* does not — and because HTTP/2
multiplexes, EVERY concurrent stream on that connection fails at once, with the
identical message. That is exactly the production signature: the scheduled-action
loop, PPV sweep, chat reconciliation, vault autosync and an inbound
``GET /my-creators`` all reporting one shared
``<ConnectionTerminated error_code:0, last_stream_id:3>`` in the same instant.
Reproduced against a real h2 server in tests/test_postgrest_transport.py: six
concurrent requests, one GOAWAY, six ``RemoteProtocolError``.

Nothing above us retried it. ``postgrest``'s own ``send_with_retry`` only retries
*HTTP status* 503/520 on GET — it never sees a transport exception. ``httpx``'s
``retries=`` only covers connection establishment. So a routine connection
recycle became an application-visible 500.

HTTP/1.1 does not have this failure mode, and not by luck: ``httpcore``'s
HTTP/1.1 connection checks whether an idle socket has become readable before
reusing it (``http11.py``, ``has_expired``), which is precisely the
server-initiated-disconnect detection the HTTP/2 path lacks. And without
multiplexing, a peer close can cost at most the one request on that connection
instead of all of them.

So the transport is built here, explicitly, with HTTP/2 off. Set
``SUPABASE_HTTP2=1`` to put it back — the knob exists so the decision is
reversible in an environment variable rather than a deploy, not because it is
expected to be used.

Everything else here is the same idea: make the numbers explicit instead of
inheriting them.
"""
from __future__ import annotations

import os
import threading

import httpx
from supabase import Client, ClientOptions, create_client

from core.config import get_settings

# Sized against the database thread pool (core/db_executor.py, 32 threads by
# default), which is the real ceiling on concurrent Supabase calls. Connections
# comfortably above that; keepalives at it, so a steady workload reuses sockets
# instead of handshaking.
DEFAULT_MAX_CONNECTIONS = 64
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 32

# httpx defaults to 5s, which under a bursty workload means paying for a TLS
# handshake several times a minute. 30s is well inside any sane proxy idle
# timeout, and on HTTP/1.1 a connection the peer closed early is detected before
# reuse rather than surfacing as an error.
DEFAULT_KEEPALIVE_EXPIRY_SECONDS = 30.0

# postgrest's default is a flat 120s for every phase. A handshake that has not
# completed in 10s is not going to, and holding a database-executor thread for
# two minutes on a dead socket is how one incident becomes a stall. The read and
# write budgets are deliberately left at postgrest's 120s: tightening those
# changes which queries succeed, which is not this fix's business.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10.0
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_POOL_TIMEOUT_SECONDS = 30.0

# Retries inside httpcore's ``_connect`` only, which runs BEFORE any request
# bytes are written (httpcore/_sync/connection.py catches ConnectError and
# ConnectTimeout and nothing else). That makes it safe for writes as well as
# reads: a request that was never sent cannot have been applied.
DEFAULT_CONNECT_RETRIES = 2

# A retired client is not closed while another thread may still be reading a
# response from it. It is closed after this grace period instead, which is
# longer than any single request's budget short of the 120s ceiling.
RETIRE_GRACE_SECONDS = 130.0

_LOCK = threading.Lock()
_client: Client | None = None
_http_client: httpx.Client | None = None
_generation: int = 0


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def http2_enabled() -> bool:
    """HTTP/2 is off unless an operator turns it back on deliberately."""
    return os.getenv("SUPABASE_HTTP2", "").strip().lower() in {"1", "true", "yes", "on"}


def transport_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=_env_int("SUPABASE_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS),
        max_keepalive_connections=_env_int(
            "SUPABASE_MAX_KEEPALIVE_CONNECTIONS", DEFAULT_MAX_KEEPALIVE_CONNECTIONS
        ),
        keepalive_expiry=_env_float(
            "SUPABASE_KEEPALIVE_EXPIRY_SECONDS", DEFAULT_KEEPALIVE_EXPIRY_SECONDS
        ),
    )


def transport_timeout() -> httpx.Timeout:
    request_timeout = _env_float(
        "SUPABASE_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    return httpx.Timeout(
        connect=_env_float(
            "SUPABASE_CONNECT_TIMEOUT_SECONDS", DEFAULT_CONNECT_TIMEOUT_SECONDS
        ),
        read=request_timeout,
        write=request_timeout,
        pool=_env_float("SUPABASE_POOL_TIMEOUT_SECONDS", DEFAULT_POOL_TIMEOUT_SECONDS),
    )


def build_http_client() -> httpx.Client:
    """The HTTPX client every PostgREST (and auth) call runs on.

    Built here rather than left to supabase-py so that HTTP/2, the pool size and
    the timeouts are decisions with names, not defaults three dependencies down.
    """
    limits = transport_limits()
    http2 = http2_enabled()
    transport = httpx.HTTPTransport(
        http1=True,
        http2=http2,
        limits=limits,
        retries=_env_int("SUPABASE_CONNECT_RETRIES", DEFAULT_CONNECT_RETRIES),
    )
    # postgrest builds its own session with follow_redirects=True; an injected
    # client has to keep that or a 3xx from the edge stops being followed.
    return httpx.Client(
        transport=transport,
        timeout=transport_timeout(),
        follow_redirects=True,
    )


def describe() -> str:
    limits = transport_limits()
    return (
        f"supabase transport: http2={http2_enabled()} "
        f"max_connections={limits.max_connections} "
        f"max_keepalive={limits.max_keepalive_connections} "
        f"keepalive_expiry={limits.keepalive_expiry}s "
        f"(SUPABASE_HTTP2, SUPABASE_MAX_CONNECTIONS, ...)"
    )


def snapshot() -> dict:
    """Transport facts for the health document. No URLs, no keys."""
    limits = transport_limits()
    return {
        "http2": http2_enabled(),
        "generation": _generation,
        "built": _client is not None,
        "max_connections": limits.max_connections,
        "max_keepalive_connections": limits.max_keepalive_connections,
        "keepalive_expiry_seconds": limits.keepalive_expiry,
    }


def _build_locked() -> Client:
    """Caller holds ``_LOCK``."""
    global _client, _http_client

    settings = get_settings()
    http_client = build_http_client()
    client = create_client(
        settings.SUPABASE_URL,
        settings.SUPABASE_SERVICE_KEY,
        options=ClientOptions(httpx_client=http_client),
    )
    _http_client = http_client
    _client = client
    print(f"[DB TRANSPORT] built generation={_generation} {describe()}")
    return client


def get_supabase() -> Client:
    """The process-wide Supabase client.

    Reused in the healthy case — this is not a per-request client — but no
    longer an ``lru_cache`` that cannot be invalidated.
    """
    client = _client
    if client is not None:
        return client
    with _LOCK:
        if _client is not None:
            return _client
        return _build_locked()


def supabase_generation() -> int:
    """Which build of the client is current.

    A caller reads this BEFORE a database operation and hands it back to
    ``reset_supabase_client``. That is what stops a hundred callers who all hit
    the same dead connection from building a hundred clients: the first reset
    bumps the generation, and every later request carrying the old number is a
    no-op because the rebuild it is asking for has already happened.
    """
    return _generation


def _retire(client: httpx.Client | None) -> None:
    """Close a replaced transport, but not out from under a live request.

    ``httpx.Client.close()`` closes the whole connection pool, including
    connections another thread is mid-response on. A reset happens precisely
    when several threads are in flight, so closing synchronously would convert
    one transport failure into several. The retired pool stops receiving new
    requests immediately (nothing holds a reference to it any more) and its
    sockets are released once the in-flight work has had time to finish.
    """
    if client is None or client.is_closed:
        return

    def _close() -> None:
        try:
            client.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"[DB TRANSPORT] retired client close failed: {type(exc).__name__}")

    timer = threading.Timer(RETIRE_GRACE_SECONDS, _close)
    timer.daemon = True
    timer.start()


def reset_supabase_client(*, reason: str, generation: int | None = None) -> bool:
    """Discard the cached client so the next call builds a fresh transport.

    Returns whether this call is the one that did it. ``generation`` is the
    value the caller observed before its failed operation; if the client has
    already been rebuilt since, this is a no-op and returns False.

    This is a last resort, not the recovery path. httpx's pool already evicts
    connections that the peer terminated — measured, see
    tests/test_postgrest_transport.py — so an ordinary GOAWAY needs nothing
    here. This exists for the case the pool itself cannot recover from, and it
    is deliberately hard to trigger repeatedly.
    """
    global _client, _http_client, _generation

    with _LOCK:
        if generation is not None and generation != _generation:
            return False
        if _client is None and _http_client is None:
            # Nothing built yet: still advance the generation so a concurrent
            # caller holding the old number does not reset the client that is
            # about to be built.
            _generation += 1
            return True
        retired, _http_client = _http_client, None
        _client = None
        previous = _generation
        _generation += 1
        print(
            f"[DB TRANSPORT RESET] reason={reason} "
            f"generation={previous}->{_generation}"
        )
    _retire(retired)
    return True


def close_supabase_client() -> None:
    """Release the transport. Called once from application shutdown."""
    global _client, _http_client

    with _LOCK:
        client, _http_client = _http_client, None
        _client = None
    if client is not None and not client.is_closed:
        try:
            client.close()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            print(f"[DB TRANSPORT] close failed: {type(exc).__name__}")
