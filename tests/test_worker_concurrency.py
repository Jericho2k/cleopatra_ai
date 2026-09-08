"""SCALE-001: bounded concurrent scheduled-action execution.

Every test here is written so that it FAILS against the previous sequential
implementation and passes against the bounded pool — that is the point of the
change, so it has to be the thing under assertion.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from tests.worker_harness import (
    ConcurrencyProbe,
    FakeQueue,
    install,
    make_actions,
    stub_handler,
)
from workers import scheduled_actions as worker


def run(coro):
    return asyncio.run(coro)


def test_independent_fans_make_parallel_progress(monkeypatch):
    """Eight fans, 100 ms each. Sequential would need ~800 ms; bounded needs ~100."""
    actions = make_actions(8)
    queue = FakeQueue(actions)
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.1))

    started = time.perf_counter()
    result = run(worker.process_cycle(concurrency=8, limit=24))
    duration = time.perf_counter() - started

    assert result.processed == 8
    assert result.sent == 8
    assert len(queue.completed) == 8
    # The whole point: real simultaneity across independent conversations.
    assert probe.max_inflight == 8
    assert result.max_concurrency == 8
    # Generous bound; sequentially this cannot finish under 800 ms.
    assert duration < 0.5


def test_concurrency_is_strictly_bounded(monkeypatch):
    """Fifty due actions must never exceed the configured ceiling."""
    queue = FakeQueue(make_actions(50))
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.02))

    result = run(worker.process_cycle(concurrency=6, limit=50))

    assert result.processed == 50
    assert probe.max_inflight <= 6
    assert result.max_concurrency <= 6


def test_same_fan_actions_never_run_simultaneously(monkeypatch):
    """Six actions across two fans: parallel across fans, serial within one."""
    queue = FakeQueue(make_actions(6, fans=2))
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.03))

    result = run(worker.process_cycle(concurrency=6, limit=24))

    assert result.processed == 6
    # The invariant that makes concurrency safe for a live conversation.
    assert probe.same_fan_overlaps == 0
    # And it is genuinely concurrent across the two fans.
    assert probe.max_inflight == 2


def test_same_fan_actions_run_in_claim_order(monkeypatch):
    queue = FakeQueue(make_actions(4, fans=1))
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.005))

    run(worker.process_cycle(concurrency=8, limit=24))

    assert probe.completed == ["action-0", "action-1", "action-2", "action-3"]
    assert probe.max_inflight == 1


def test_grouping_keeps_fanless_actions_independent():
    chains = worker.group_actions_by_fan(
        [
            {"id": "a", "fan_id": "fan-1"},
            {"id": "b", "fan_id": "fan-2"},
            {"id": "c", "fan_id": "fan-1"},
            {"id": "d", "fan_id": None},
            {"id": "e", "fan_id": None},
        ]
    )
    assert [[a["id"] for a in chain] for chain in chains] == [
        ["a", "c"],
        ["b"],
        ["d"],
        ["e"],
    ]


def test_one_failing_action_does_not_abandon_the_batch(monkeypatch):
    queue = FakeQueue(make_actions(5))
    probe = ConcurrencyProbe()

    async def handler(action):
        if action["id"] == "action-2":
            raise RuntimeError("temporary platform failure")
        probe.enter(str(action["fan_id"]))
        try:
            await asyncio.sleep(0.01)
            return worker.HandlerResult(sent_message=True)
        finally:
            probe.exit(str(action["fan_id"]))

    install(monkeypatch, queue=queue, handler=handler)
    result = run(worker.process_cycle(concurrency=5, limit=24))

    assert result.processed == 5
    assert result.errors == 1
    assert result.sent == 4
    assert [action_id for action_id, _ in queue.failed] == ["action-2"]


def test_full_batch_is_reported_so_the_loop_repolls(monkeypatch):
    queue = FakeQueue(make_actions(10))
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.001))

    full = run(worker.process_cycle(concurrency=4, limit=10))
    assert full.batch_full is True

    queue2 = FakeQueue(make_actions(3))
    install(monkeypatch, queue=queue2, handler=stub_handler(probe, seconds=0.001))
    short = run(worker.process_cycle(concurrency=4, limit=10))
    assert short.batch_full is False


def test_loop_repolls_immediately_on_a_full_batch(monkeypatch):
    """A backlog drains without waiting out the idle poll between batches."""
    cycles = []
    sleeps = []

    async def fake_cycle(**kwargs):
        cycles.append(kwargs)
        if len(cycles) >= 4:
            raise asyncio.CancelledError
        return worker.CycleResult(batch_full=True, claimed=24)

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(worker, "process_cycle", fake_cycle)
    monkeypatch.setattr(worker.asyncio, "sleep", fake_sleep)

    with pytest.raises(asyncio.CancelledError):
        run(worker.scheduled_actions_loop())

    assert len(cycles) == 4
    # Every gap is the short busy poll, never the 60 s the old loop always took.
    assert sleeps == [worker.BUSY_POLL_SECONDS] * 3
    assert all(gap < 1.0 for gap in sleeps)


def test_loop_idles_on_a_short_batch_instead_of_busy_looping(monkeypatch):
    waits = []

    async def fake_cycle(**_kwargs):
        if len(waits) >= 2:
            raise asyncio.CancelledError
        return worker.CycleResult(batch_full=False)

    async def fake_wait_for(awaitable, timeout):
        waits.append(timeout)
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(worker, "process_cycle", fake_cycle)
    monkeypatch.setattr(worker.asyncio, "wait_for", fake_wait_for)

    with pytest.raises(asyncio.CancelledError):
        run(worker.scheduled_actions_loop())

    assert waits == [worker.poll_seconds(), worker.poll_seconds()]


def test_busy_polling_is_capped_so_it_cannot_spin_forever(monkeypatch):
    """An unproductive full claim must eventually fall back to the idle poll."""
    idle_waits = []
    cycles = []

    async def fake_cycle(**_kwargs):
        cycles.append(1)
        if len(cycles) > worker.MAX_CONSECUTIVE_BUSY_CYCLES + 1:
            raise asyncio.CancelledError
        return worker.CycleResult(batch_full=True)

    async def fake_sleep(_seconds):
        return None

    async def fake_wait_for(awaitable, timeout):
        idle_waits.append(timeout)
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(worker, "process_cycle", fake_cycle)
    monkeypatch.setattr(worker.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(worker.asyncio, "wait_for", fake_wait_for)

    with pytest.raises(asyncio.CancelledError):
        run(worker.scheduled_actions_loop())

    assert idle_waits, "busy polling must yield to an idle poll eventually"


def test_configuration_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("SCHEDULED_ACTION_CONCURRENCY", raising=False)
    monkeypatch.delenv("SCHEDULED_ACTION_CLAIM_LIMIT", raising=False)
    assert worker.action_concurrency() == 8
    assert worker.claim_limit() == 24
    # Claim batch must stay comfortably larger than the pool so slots stay fed.
    assert worker.claim_limit() >= 2 * worker.action_concurrency()

    monkeypatch.setenv("SCHEDULED_ACTION_CONCURRENCY", "12")
    monkeypatch.setenv("SCHEDULED_ACTION_CLAIM_LIMIT", "40")
    assert worker.action_concurrency() == 12
    assert worker.claim_limit() == 40

    # A nonsense value must not take the worker down on deploy.
    monkeypatch.setenv("SCHEDULED_ACTION_CONCURRENCY", "banana")
    assert worker.action_concurrency() == 8
    monkeypatch.setenv("SCHEDULED_ACTION_CONCURRENCY", "0")
    assert worker.action_concurrency() == 1


def test_claim_uses_the_configured_batch_size(monkeypatch):
    seen = {}

    async def claim(limit=20, **_kwargs):
        seen["limit"] = limit
        return []

    monkeypatch.setattr(worker, "repair_followup_obligations", _zero)
    monkeypatch.setattr(worker, "claim_due_actions", claim)
    monkeypatch.setenv("SCHEDULED_ACTION_CLAIM_LIMIT", "33")

    run(worker.process_cycle())
    assert seen["limit"] == 33


def test_health_snapshot_reports_the_last_cycle(monkeypatch):
    queue = FakeQueue(make_actions(4))
    probe = ConcurrencyProbe()
    install(monkeypatch, queue=queue, handler=stub_handler(probe, seconds=0.005))

    run(worker.process_cycle(concurrency=4, limit=10))
    snapshot = worker.worker_health_snapshot()

    assert snapshot["last_claimed"] == 4
    assert snapshot["last_sent"] == 4
    assert snapshot["last_max_action_concurrency"] >= 1
    assert snapshot["seconds_since_last_cycle"] is not None
    assert snapshot["action_concurrency_limit"] >= 1
    assert snapshot["last_error"] is None


async def _zero(**_kwargs) -> int:
    return 0
