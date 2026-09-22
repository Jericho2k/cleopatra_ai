"""Per-fan ownership across worker processes, not just inside one.

``group_actions_by_fan`` is still the cheap path and still correct — inside one
process. ``docs/full_auto_capacity.md`` was explicit that it described a single
Uvicorn process, and this sprint's correctness must not depend on that. So the
question these tests ask is the one grouping cannot answer: when two DIFFERENT
processes each hold a due action for the same conversation, does exactly one of
them get to send?
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from db import outbound_queries as store
from tests.fake_supabase import FakeSupabase
from workers import scheduled_actions as worker


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _reset():
    store._reset_availability_for_tests()
    yield
    store._reset_availability_for_tests()


@pytest.fixture
def leases(monkeypatch):
    fake = FakeSupabase({"fan_execution_leases": []})
    monkeypatch.setattr(store, "get_supabase", lambda: fake)
    return fake


def acquire(owner: str, *, fan_id: str = "fan-1", ttl: int = 120) -> bool:
    return run(
        store.acquire_fan_lease(
            fan_id=fan_id,
            creator_id="creator-1",
            owner_token=owner,
            ttl_seconds=ttl,
            purpose="AUTO_REPLY",
        )
    )


def test_one_worker_wins_a_contested_fan(leases):
    assert acquire("worker-a") is True
    assert acquire("worker-b") is False


def test_the_owner_may_reacquire_its_own_lease(leases):
    assert acquire("worker-a") is True
    assert acquire("worker-a") is True


def test_releasing_hands_the_fan_to_the_next_worker(leases):
    acquire("worker-a")
    run(store.release_fan_lease(fan_id="fan-1", owner_token="worker-a"))

    assert acquire("worker-b") is True


def test_releasing_with_the_wrong_token_does_not_free_the_fan(leases):
    acquire("worker-a")
    run(store.release_fan_lease(fan_id="fan-1", owner_token="worker-b"))

    assert acquire("worker-b") is False


def test_a_crashed_worker_lease_expires_and_is_recoverable(leases):
    """Crash recovery. Nothing else releases a lease a dead process held."""
    acquire("worker-a")
    leases.tables["fan_execution_leases"][0]["expires_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=1)
    ).isoformat()

    assert acquire("worker-b") is True


def test_unrelated_fans_stay_independent(leases):
    assert acquire("worker-a", fan_id="fan-1") is True
    assert acquire("worker-b", fan_id="fan-2") is True


def test_the_lease_is_not_a_global_lock(leases):
    """A hundred fans are a hundred independent claims, never one queue."""
    owners = [acquire(f"worker-{i}", fan_id=f"fan-{i}") for i in range(100)]
    assert all(owners)


def test_a_missing_table_falls_back_to_in_process_grouping(monkeypatch):
    """Pre-migration rollout: the previous guarantee, not a weaker one."""

    class Broken:
        def table(self, _name):
            raise RuntimeError('relation "fan_execution_leases" does not exist')

    monkeypatch.setattr(store, "get_supabase", lambda: Broken())
    assert acquire("worker-a") is True


# --- the worker's use of it -------------------------------------------------


def _action(action_id: str, fan_id: str = "fan-1") -> dict:
    return {
        "id": action_id,
        "creator_id": "creator-1",
        "fan_id": fan_id,
        "action_type": "AUTO_REPLY",
        "payload": {},
        "attempts": 0,
    }


def test_the_worker_defers_rather_than_dropping_a_busy_fan(leases, monkeypatch):
    """A fan owned elsewhere is rescheduled, never skipped and never failed."""
    acquire("someone-else")

    rescheduled: list = []

    async def _reschedule(action_id, execute_at, payload=None):
        rescheduled.append((action_id, execute_at))

    async def _handler(_action):
        raise AssertionError("the handler must not run for a fan we do not own")

    monkeypatch.setattr(worker, "reschedule_action", _reschedule)
    monkeypatch.setattr(worker, "repair_followup_obligations", _zero)
    monkeypatch.setattr(worker, "_connector_blocks_delivery", _false)
    monkeypatch.setitem(worker.HANDLERS, "AUTO_REPLY", _handler)

    outcome = run(worker._resolve_action(_action("action-1"), sent_counter=[0]))

    assert outcome == "fan_busy"
    assert [row[0] for row in rescheduled] == ["action-1"]


def test_the_worker_releases_the_fan_when_the_action_finishes(leases, monkeypatch):
    async def _handler(_action):
        return worker.HandlerResult(sent_message=True, reason="stub")

    monkeypatch.setattr(worker, "repair_followup_obligations", _zero)
    monkeypatch.setattr(worker, "complete_action", _noop)
    monkeypatch.setattr(worker, "_record_message_action_resolution", _noop)
    monkeypatch.setattr(worker, "_should_still_send", _ok)
    monkeypatch.setattr(worker, "_connector_blocks_delivery", _false)
    monkeypatch.setitem(worker.HANDLERS, "AUTO_REPLY", _handler)

    run(worker._resolve_action(_action("action-1"), sent_counter=[0]))

    # Another worker can take the fan immediately afterwards.
    assert acquire("worker-b") is True


def test_the_fan_is_released_even_when_the_handler_raises(leases, monkeypatch):
    async def _handler(_action):
        raise RuntimeError("provider exploded")

    monkeypatch.setattr(worker, "repair_followup_obligations", _zero)
    monkeypatch.setattr(worker, "fail_action", _noop)
    monkeypatch.setattr(worker, "_should_still_send", _ok)
    monkeypatch.setattr(worker, "_connector_blocks_delivery", _false)
    monkeypatch.setitem(worker.HANDLERS, "AUTO_REPLY", _handler)

    run(worker._resolve_action(_action("action-1"), sent_counter=[0]))

    assert acquire("worker-b") is True


async def _zero(**_kwargs) -> int:
    return 0


async def _noop(*_args, **_kwargs):
    return None


async def _ok(_action=None):
    return worker.ActionCheck(True)


async def _false(_action):
    return False
