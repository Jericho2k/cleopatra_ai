"""Global admission control for outbound model calls.

Every paid generation in Cleopatra — the Kimi writer, the DeepSeek commercial
route, the situation analyzer, the background extractor — travels through
``ai.model_providers.complete``. Until now nothing bounded how many of those
could be in flight at once; the sequential scheduled-actions worker happened to
hold Full Auto to one, and the Assisted path had no limit at all.

Concurrent action execution removes that accident, so the limit has to become
explicit. This module is that limit: one process-wide semaphore, a conservative
default, and enough telemetry to tell "the provider was slow" apart from "we
waited for a local slot".

Two properties matter more than the number itself:

* **Queueing, not shedding.** A burst larger than the limit waits. It is never
  dropped, never retried into the provider, and never allowed to open more
  upstream connections than the limit allows.
* **Cancellation safety.** The wait is a plain ``await`` on an ``asyncio``
  primitive, so cancelling the surrounding action cancels the wait, and the slot
  is released on success, exception, timeout, and cancellation alike.

The HTTP connection pool is deliberately *not* the policy. ``AsyncOpenAI``
defaults to ``max_connections=1000``; that is a transport ceiling, not a
statement about how much concurrent inference this deployment should buy.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections import deque
from dataclasses import dataclass

DEFAULT_MODEL_MAX_CONCURRENCY = 8

# Enough samples to see a representative recent wait without retaining history.
_WAIT_SAMPLE_SIZE = 64


def configured_model_concurrency() -> int:
    """Read ``MODEL_MAX_CONCURRENCY`` with a conservative, always-valid default."""
    raw = os.getenv("MODEL_MAX_CONCURRENCY", "").strip()
    if not raw:
        return DEFAULT_MODEL_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MODEL_MAX_CONCURRENCY
    # A limit of zero would deadlock every generation; clamp instead of failing
    # a deploy on a typo.
    return max(1, min(value, 256))


@dataclass
class _GateState:
    limit: int
    semaphore: asyncio.Semaphore
    inflight: int = 0
    waiting: int = 0
    total_acquired: int = 0
    total_wait_ms: float = 0.0
    max_wait_ms: float = 0.0
    max_inflight: int = 0


class ModelGate:
    """Process-wide concurrency gate for provider calls.

    The semaphore is created lazily per event loop. ``asyncio`` primitives bind
    to the first loop that awaits them, and this process legitimately runs more
    than one loop over its lifetime (tests, ``asyncio.run`` entry points), so a
    module-level semaphore constructed at import time would raise once a second
    loop touched it.
    """

    def __init__(self) -> None:
        self._states: dict[int, _GateState] = {}
        self._waits: deque[float] = deque(maxlen=_WAIT_SAMPLE_SIZE)

    def _state(self) -> _GateState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        key = id(loop)
        state = self._states.get(key)
        limit = configured_model_concurrency()
        if state is None:
            state = _GateState(limit=limit, semaphore=asyncio.Semaphore(limit))
            # Only the live loop's state is retained; a finished loop's entry is
            # dead weight and its semaphore can never be awaited again.
            self._states = {key: state}
        elif state.limit != limit and state.inflight == 0 and state.waiting == 0:
            # Allow an operator to change the limit without a restart, but only
            # while the gate is completely idle so no slot accounting is lost.
            state.limit = limit
            state.semaphore = asyncio.Semaphore(limit)
        return state

    @contextlib.asynccontextmanager
    async def acquire(self, *, feature: str = "model"):
        """Hold one model slot for the duration of the block.

        Yields the milliseconds spent waiting for the slot so the caller can
        report provider latency and admission latency separately.
        """
        state = self._state()
        started = time.perf_counter()
        state.waiting += 1
        try:
            await state.semaphore.acquire()
        finally:
            state.waiting -= 1
        wait_ms = (time.perf_counter() - started) * 1000.0
        state.inflight += 1
        state.total_acquired += 1
        state.total_wait_ms += wait_ms
        state.max_wait_ms = max(state.max_wait_ms, wait_ms)
        state.max_inflight = max(state.max_inflight, state.inflight)
        self._waits.append(wait_ms)
        try:
            yield wait_ms
        finally:
            # Success, exception, timeout, and cancellation all land here.
            state.inflight -= 1
            state.semaphore.release()

    def snapshot(self) -> dict:
        """Return non-identifying gate metrics for the health surface."""
        state = self._states.get(id(_current_loop()))
        if state is None:
            return {
                "limit": configured_model_concurrency(),
                "inflight": 0,
                "waiting": 0,
                "max_inflight": 0,
                "acquired_total": 0,
                "avg_wait_ms": 0,
                "max_wait_ms": 0,
                "recent_wait_ms": 0,
            }
        recent = list(self._waits)
        return {
            "limit": state.limit,
            "inflight": state.inflight,
            "waiting": state.waiting,
            "max_inflight": state.max_inflight,
            "acquired_total": state.total_acquired,
            "avg_wait_ms": int(
                state.total_wait_ms / state.total_acquired
            ) if state.total_acquired else 0,
            "max_wait_ms": int(state.max_wait_ms),
            "recent_wait_ms": int(sorted(recent)[len(recent) // 2]) if recent else 0,
        }

    def reset(self) -> None:
        """Drop all gate state. Test-support only."""
        self._states = {}
        self._waits.clear()


def _current_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


MODEL_GATE = ModelGate()
