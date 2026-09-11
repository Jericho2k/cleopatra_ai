"""Retry helpers for short-lived database transport failures.

Only idempotent reads and writes belong here. External delivery calls must never
be retried through this module because their outcome may already be visible to a
fan even when the HTTP response was lost.

What counts as retryable, and why the list is not longer
-------------------------------------------------------

Every name and marker below describes a failure of the *transport*: a socket
that went away, a pool that timed out, a connection the peer terminated. None of
them can mean "the statement ran and the answer was lost" for a read, so a read
can be repeated freely.

For a write, they can. ``RemoteProtocolError`` after the request bytes are on the
wire is ambiguous by construction — the server may well have committed. That is
why this module is a helper a caller opts into per operation and not a wrapper
around the client: the safety argument lives at the call site, where somebody can
see whether there is a unique key, an idempotency key, a compare-and-set, or an
upsert contract underneath. There is no automatic write retry here and there
must not be one. Ambiguous writes belong to the durable action queue and the
reconciliation passes, which were built for exactly that.

Transport reset
---------------

``reset_after_attempt`` exists for the case httpx's own pool cannot recover
from. It is not the normal path: a PostgREST ``GOAWAY`` evicts the dead
connection from the pool by itself and the very next attempt gets a fresh one
(measured — tests/test_postgrest_transport.py). The reset is generation-guarded
in core/supabase.py, so a hundred callers failing together produce one rebuild,
not a hundred.
"""
from __future__ import annotations

import asyncio
import os
import random
from collections.abc import Awaitable, Callable
from typing import Any


TRANSIENT_ERROR_NAMES = {
    "ConnectError",
    "ConnectTimeout",
    "ConnectionTerminated",
    "PoolTimeout",
    "ReadError",
    "ReadTimeout",
    "RemoteProtocolError",
    "WriteError",
    "WriteTimeout",
}
TRANSIENT_ERROR_MARKERS = (
    "connection terminated",
    "connection reset",
    "connection refused",
    "connection closed",
    "server disconnected",
    "temporarily unavailable",
)

# Retry delays are jittered so that the callers a single GOAWAY knocked over
# together do not come back in lockstep and collide again.
JITTER_RATIO = 0.5
MAX_DELAY_SECONDS = 2.0

# How much of an exception message is safe to print. The type name carries most
# of the diagnostic value; the message is truncated and scrubbed because httpx
# errors can quote a request URL.
_MAX_DETAIL_CHARS = 200


def _secrets() -> list[str]:
    values = []
    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_KEY"):
        value = os.getenv(name, "").strip()
        if len(value) > 8:
            values.append(value)
    return values


def redact(text: str) -> str:
    """A log-safe rendering of an exception message.

    Never the service key, never the project URL, never unbounded length.
    """
    cleaned = " ".join(str(text).split())
    for secret in _secrets():
        cleaned = cleaned.replace(secret, "[redacted]")
    if len(cleaned) > _MAX_DETAIL_CHARS:
        cleaned = cleaned[:_MAX_DETAIL_CHARS] + "..."
    return cleaned


def describe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {redact(str(exc))}"


def is_transient_db_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if type(current).__name__ in TRANSIENT_ERROR_NAMES:
            return True
        message = str(current).lower()
        if any(marker in message for marker in TRANSIENT_ERROR_MARKERS):
            return True
        current = current.__cause__ or current.__context__
    return False


def _backoff(delay_seconds: float, attempt: int) -> float:
    base = min(delay_seconds * attempt, MAX_DELAY_SECONDS)
    return base + random.uniform(0.0, base * JITTER_RATIO)


async def retry_transient_db_operation(
    operation: Callable[[], Awaitable[Any]],
    *,
    label: str,
    attempts: int = 3,
    delay_seconds: float = 0.15,
    log_prefix: str = "DB RETRY",
    reset_after_attempt: int | None = 2,
) -> Any:
    """Retry an idempotent database operation after transport disconnects.

    ``attempts`` is a hard budget, not a target: a persistently failing
    operation raises the original exception rather than spinning.

    ``reset_after_attempt`` asks core.supabase to rebuild the transport once the
    named attempt has failed transiently, on the theory that a pool which has
    not healed itself by then is not going to. Pass ``None`` to never reset — do
    that where the operation is itself the health probe, so that diagnosing an
    outage cannot churn the client.
    """
    from core.supabase import reset_supabase_client, supabase_generation

    for attempt in range(1, attempts + 1):
        generation = supabase_generation()
        try:
            result = await operation()
        except Exception as exc:
            if not is_transient_db_error(exc):
                raise
            if attempt >= attempts:
                print(
                    f"[{log_prefix}] exhausted label={label} "
                    f"attempts={attempts} error={describe_error(exc)}"
                )
                raise
            print(
                f"[{log_prefix}] transient label={label} "
                f"attempt={attempt}/{attempts} error={describe_error(exc)}"
            )
            if reset_after_attempt is not None and attempt >= reset_after_attempt:
                reset_supabase_client(
                    reason=f"{label}:{type(exc).__name__}", generation=generation
                )
            await asyncio.sleep(_backoff(delay_seconds, attempt))
        else:
            if attempt > 1:
                print(
                    f"[{log_prefix}] recovered label={label} "
                    f"attempt={attempt}/{attempts}"
                )
            return result
    raise RuntimeError(f"{label} unavailable")


async def retry_db_read(
    read: Callable[[], Any],
    *,
    label: str,
    attempts: int = 3,
    delay_seconds: float = 0.15,
    reset_after_attempt: int | None = 2,
) -> Any:
    """Run a BLOCKING Supabase read off the event loop, with bounded retry.

    The one-liner for the call sites that were ``await asyncio.to_thread(...)``
    around a ``.select(...).execute()``. Reads only: repeating a select cannot
    apply anything twice. ``read`` must be a plain callable, not a coroutine, so
    that each attempt submits a fresh job to the database executor.
    """
    return await retry_transient_db_operation(
        lambda: asyncio.to_thread(read),
        label=label,
        attempts=attempts,
        delay_seconds=delay_seconds,
        reset_after_attempt=reset_after_attempt,
    )
