"""The atomic claim's rollout affordance, and its limits.

db/scheduled_action_claim_v1.sql has to be applied by a human in Supabase —
there is no migration runner — so the code that calls it can reach production
first. It falls back to the previous per-row compare-and-swap in exactly one
case: the function is not there. Any other failure must surface, because
silently downgrading the claim path on, say, a deadlock would hide a real
problem behind a slower query.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import db.commercial_queries as commercial


class _Rpc:
    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.calls: list[tuple[str, dict]] = []

    def rpc(self, name, params):
        self.calls.append((name, params))
        return SimpleNamespace(execute=lambda: self._behaviour())

    def table(self, _name):
        raise AssertionError("the CAS fallback must not run in this test")


@pytest.fixture(autouse=True)
def reset_flag():
    commercial._ATOMIC_CLAIM_AVAILABLE = True
    yield
    commercial._ATOMIC_CLAIM_AVAILABLE = True


def test_the_atomic_claim_is_one_round_trip(monkeypatch):
    rows = [{"id": "action-1"}, {"id": "action-2"}]
    db = _Rpc(lambda: SimpleNamespace(data=rows))
    monkeypatch.setattr(commercial, "get_supabase", lambda: db)

    claimed = asyncio.run(commercial.claim_due_actions(limit=20, stale_minutes=10))

    assert claimed == rows
    assert db.calls == [
        ("claim_due_actions", {"p_limit": 20, "p_stale_minutes": 10})
    ]


def test_a_missing_function_falls_back_once_and_then_stops_trying(monkeypatch):
    """A rolling deploy must not pay a failed RPC on every poll."""

    attempts = {"count": 0}

    def missing():
        attempts["count"] += 1
        raise RuntimeError(
            "PGRST202 Could not find the function public.claim_due_actions"
        )

    db = _Rpc(missing)
    monkeypatch.setattr(commercial, "get_supabase", lambda: db)

    fallback_calls = {"count": 0}

    async def fake_cas(limit, stale_minutes):
        fallback_calls["count"] += 1
        return [{"id": "action-1"}]

    monkeypatch.setattr(commercial, "_claim_due_actions_by_cas", fake_cas)

    first = asyncio.run(commercial.claim_due_actions())
    second = asyncio.run(commercial.claim_due_actions())

    assert first == second == [{"id": "action-1"}]
    assert fallback_calls["count"] == 2
    # The RPC was attempted once, not once per poll.
    assert attempts["count"] == 1


def test_a_real_failure_inside_the_function_surfaces(monkeypatch):
    """A deadlock or constraint violation is not a rollout problem."""

    def deadlock():
        raise RuntimeError("deadlock detected")

    db = _Rpc(deadlock)
    monkeypatch.setattr(commercial, "get_supabase", lambda: db)

    async def fake_cas(limit, stale_minutes):
        raise AssertionError("must not fall back on a genuine failure")

    monkeypatch.setattr(commercial, "_claim_due_actions_by_cas", fake_cas)

    with pytest.raises(RuntimeError, match="deadlock"):
        asyncio.run(commercial.claim_due_actions())

    assert commercial._ATOMIC_CLAIM_AVAILABLE is True


def test_the_fallback_still_only_lets_one_worker_own_a_row(monkeypatch):
    """The CAS path is unchanged, and it was correct on its own."""

    rows = [{"id": "action-1", "status": "PENDING", "locked_at": None}]
    updates: list[dict] = []

    class _Table:
        def __init__(self, name):
            self.name = name
            self._filters: dict = {}
            self._op = None
            self._payload = None

        def select(self, *_args, **_kwargs):
            self._op = "select"
            return self

        def eq(self, column, value):
            self._filters[column] = value
            return self

        def lte(self, *_args, **_kwargs):
            return self

        def lt(self, *_args, **_kwargs):
            self._op = "select_stale"
            return self

        def order(self, *_args, **_kwargs):
            return self

        def limit(self, *_args, **_kwargs):
            return self

        def update(self, payload):
            self._op = "update"
            self._payload = payload
            return self

        def execute(self):
            if self._op == "update":
                updates.append(dict(self._filters))
                # The row's status already moved, so the guard finds nothing:
                # this is the losing worker.
                won = self._filters.get("status") == "PENDING" and len(updates) == 1
                return SimpleNamespace(data=rows if won else [])
            if self._op == "select_stale":
                return SimpleNamespace(data=[])
            if self._filters.get("status") == "PENDING":
                return SimpleNamespace(data=list(rows))
            return SimpleNamespace(data=[])

    db = SimpleNamespace(table=lambda name: _Table(name))
    monkeypatch.setattr(commercial, "get_supabase", lambda: db)

    first = asyncio.run(commercial._claim_due_actions_by_cas(20, 10))
    second = asyncio.run(commercial._claim_due_actions_by_cas(20, 10))

    assert len(first) == 1
    assert second == []
    # Every claim asserted the status it observed.
    assert all(update.get("status") == "PENDING" for update in updates)
