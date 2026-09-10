"""The thread pool every Supabase call actually runs on.

The Supabase Python client is synchronous, so every database call in this
codebase reaches it through ``asyncio.to_thread``. That helper submits to the
event loop's DEFAULT executor, and nothing here ever configured one — so the
ceiling on concurrent database work was whatever ``ThreadPoolExecutor()`` picks
by itself: ``min(32, (os.cpu_count() or 1) + 4)``.

On a 4-vCPU container that is EIGHT concurrent database calls, for the whole
process. Not eight per worker — eight in total, shared between the
scheduled-action worker's slots, the chat reconciliation pass, vault sync, the
health probe, and every inbound webhook.

That is a real capacity limit and it was invisible: it appears in no
configuration, no log line, and no health document, and it moves when the
container size changes. A burst of inbound webhooks and a full worker batch
compete for the same eight threads, so ingestion latency rises for a reason an
operator cannot see anywhere.

This makes the number explicit, logged at boot, and tunable. It does not remove
the limit — a synchronous client on a thread pool has to have one — it just
stops it being an accident of the container's CPU count.

Sizing
------
Above the worker's action concurrency (8 by default) plus headroom for
ingestion and the schedulers, and below the Supabase pooler's connection
allowance. 32 is comfortably inside both. These are HTTP requests to PostgREST
rather than direct PostgreSQL connections, so the pooler multiplexes them and
the thread count is not a connection count.

Raising this does not make the database faster. It stops unrelated work
queueing behind the worker, which is a different thing and the one that was
hurting inbound latency under burst.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MAX_WORKERS = 32

_executor: ThreadPoolExecutor | None = None


def configured_max_workers() -> int:
    raw = os.getenv("DB_EXECUTOR_MAX_WORKERS", "").strip()
    if not raw:
        return DEFAULT_MAX_WORKERS
    try:
        # Floor of 4: below the worker's own concurrency this would serialise
        # the product rather than protect anything.
        return max(4, min(128, int(raw)))
    except ValueError:
        return DEFAULT_MAX_WORKERS


def install(loop: asyncio.AbstractEventLoop | None = None) -> ThreadPoolExecutor:
    """Make the database thread pool explicit for this process.

    Idempotent: calling it twice returns the same executor rather than leaking
    a second pool.
    """
    global _executor
    if _executor is not None:
        return _executor

    max_workers = configured_max_workers()
    _executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="cleo-db",
    )
    (loop or asyncio.get_event_loop()).set_default_executor(_executor)
    return _executor


def shutdown() -> None:
    """Release the pool. Threads are not daemons, so a lingering pool keeps the
    process alive after the event loop stops."""
    global _executor
    if _executor is None:
        return
    # wait=False: shutdown runs during lifespan teardown and must not block on
    # a database call that is itself waiting on a timeout.
    _executor.shutdown(wait=False, cancel_futures=False)
    _executor = None


def describe() -> str:
    return (
        f"database thread pool: max_workers={configured_max_workers()} "
        f"(DB_EXECUTOR_MAX_WORKERS), the ceiling on CONCURRENT Supabase calls "
        f"for this whole process"
    )


def snapshot() -> dict:
    """What the health surface reports.

    ``queued`` is the number of database calls waiting for a thread. A
    persistently non-zero value is the signal that this limit — rather than the
    database, the model gate, or the provider — is what is slowing the product
    down.
    """
    limit = configured_max_workers()
    if _executor is None:
        return {"configured": False, "max_workers": limit, "queued": 0}
    try:
        queued = _executor._work_queue.qsize()  # noqa: SLF001
    except Exception:  # pragma: no cover - diagnostics only
        queued = 0
    return {
        "configured": True,
        "max_workers": limit,
        "queued": queued,
        "threads": len(getattr(_executor, "_threads", ()) or ()),
    }
