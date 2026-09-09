"""Global admission control for creator-level vault jobs.

VAULT-001 — ``vault_autosync_scheduler`` looped over every due creator and
``await``-ed ``sync_vault_start``, which only *spawns* the run and returns. The
await therefore provided no serialisation whatsoever. Creators are typically
connected in batches, so their 24-hour anniversaries cluster: 50 crossing the
interval in one hourly pass started 50 concurrent vault syncs, each of which
then ran its own media categorisation at ``VAULT_CATEGORIZATION_CONCURRENCY``.
That fan-out happens inside the same process that serves chat, and it competes
for the same CPU, thread pool and vision endpoint.

This gate is the creator-level bound. It is deliberately separate from the
per-media concurrency inside one run: the two answer different questions.

    creator-level slots = 2      (this gate)
    per-media concurrency        (VAULT_CATEGORIZATION_CONCURRENCY, inside a run)

    creator A   holds a slot, runs up to its media concurrency
    creator B   holds a slot, runs up to its media concurrency
    creator C   waits

It is also separate from ``core.model_gate``, which bounds paid inference.
Vault classification and chat generation both pass through that gate; this one
additionally stops the vault from queueing an unbounded amount of work behind
it.

Waiting is safe and lossless. A queued job is a live asyncio task parked on a
semaphore, so nothing is dropped, nothing is rescheduled into the provider, and
cancelling the surrounding task cancels the wait and releases nothing it never
held. Restart behaviour is unchanged: an interrupted run leaves
``last_vault_sync_at`` unstamped and the hourly scheduler picks it up again.

The structure mirrors ``core.model_gate`` — lazily created per event loop,
because asyncio primitives bind to the first loop that awaits them and this
process legitimately runs more than one loop over its lifetime.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass, field

# Conservative on purpose. Two creator-level jobs already multiply out to
# 2 x VAULT_CATEGORIZATION_CONCURRENCY media items in flight.
DEFAULT_VAULT_SYNC_MAX_CONCURRENCY = 2


def configured_vault_concurrency() -> int:
    """Read ``VAULT_SYNC_MAX_CONCURRENCY`` with an always-valid default."""

    raw = os.getenv("VAULT_SYNC_MAX_CONCURRENCY", "").strip()
    if not raw:
        return DEFAULT_VAULT_SYNC_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_VAULT_SYNC_MAX_CONCURRENCY
    # Zero would deadlock every vault job; clamp rather than fail a deploy on a
    # typo. The ceiling is a sanity bound, not a recommendation.
    return max(1, min(value, 32))


@dataclass
class _Job:
    creator_id: str
    kind: str
    queued_at: float
    started_at: float | None = None


@dataclass
class _GateState:
    limit: int
    semaphore: asyncio.Semaphore
    active: dict[int, _Job] = field(default_factory=dict)
    waiting: dict[int, _Job] = field(default_factory=dict)
    total_started: int = 0
    total_wait_seconds: float = 0.0
    max_wait_seconds: float = 0.0
    max_active: int = 0


class VaultGate:
    """Process-wide bound on how many creators sync or categorise at once."""

    def __init__(self) -> None:
        self._states: dict[int, _GateState] = {}
        self._sequence = 0

    def _state(self) -> _GateState:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        key = id(loop)
        state = self._states.get(key)
        limit = configured_vault_concurrency()
        if state is None:
            state = _GateState(limit=limit, semaphore=asyncio.Semaphore(limit))
            # Only the live loop's state is retained; a finished loop's
            # semaphore can never be awaited again.
            self._states = {key: state}
        elif state.limit != limit and not state.active and not state.waiting:
            # Let an operator change the limit without a restart, but only while
            # the gate is idle so no slot accounting is lost.
            state.limit = limit
            state.semaphore = asyncio.Semaphore(limit)
        return state

    @contextlib.asynccontextmanager
    async def acquire(self, *, creator_id: str, kind: str = "vault_sync"):
        """Hold one creator-level vault slot for the duration of the block.

        Yields the seconds spent queued, so a caller can report "we waited for a
        slot" separately from "the provider was slow".
        """

        state = self._state()
        self._sequence += 1
        token = self._sequence
        job = _Job(creator_id=str(creator_id), kind=kind, queued_at=time.time())
        state.waiting[token] = job
        started = time.perf_counter()
        try:
            await state.semaphore.acquire()
        finally:
            state.waiting.pop(token, None)

        waited = time.perf_counter() - started
        job.started_at = time.time()
        state.active[token] = job
        state.total_started += 1
        state.total_wait_seconds += waited
        state.max_wait_seconds = max(state.max_wait_seconds, waited)
        state.max_active = max(state.max_active, len(state.active))
        if waited >= 1.0:
            print(
                f"[VAULT GATE] creator={creator_id} kind={kind} "
                f"waited={waited:.1f}s for one of {state.limit} slots"
            )
        try:
            yield waited
        finally:
            # Success, exception, and cancellation all land here.
            state.active.pop(token, None)
            state.semaphore.release()

    def snapshot(self) -> dict:
        """Non-identifying gate metrics for the operator health surface."""

        state = self._states.get(id(_current_loop()))
        if state is None:
            return {
                "limit": configured_vault_concurrency(),
                "active": 0,
                "waiting": 0,
                "max_active": 0,
                "started_total": 0,
                "avg_wait_seconds": 0,
                "max_wait_seconds": 0,
                "oldest_active_seconds": 0,
                "oldest_waiting_seconds": 0,
            }

        now = time.time()
        oldest_active = max(
            (now - (job.started_at or job.queued_at) for job in state.active.values()),
            default=0.0,
        )
        oldest_waiting = max(
            (now - job.queued_at for job in state.waiting.values()),
            default=0.0,
        )
        return {
            "limit": state.limit,
            "active": len(state.active),
            "waiting": len(state.waiting),
            "max_active": state.max_active,
            "started_total": state.total_started,
            "avg_wait_seconds": round(
                state.total_wait_seconds / state.total_started, 1
            ) if state.total_started else 0,
            "max_wait_seconds": round(state.max_wait_seconds, 1),
            "oldest_active_seconds": int(oldest_active),
            "oldest_waiting_seconds": int(oldest_waiting),
        }

    def reset(self) -> None:
        """Drop all gate state. Test-support only."""

        self._states = {}
        self._sequence = 0


def _current_loop():
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


VAULT_GATE = VaultGate()
