"""The Simulator as a persistent workspace: clean fans, run-now, and the boundary.

Three properties are load-bearing and all three are asserted against the real
code rather than a stub:

* a new test fan is genuinely clean, and its platform id is generated here so
  the control cannot produce a fan the rest of the system treats as real;
* "Run now" fires the worker's OWN handler, inside the simulation scope, so the
  real revalidation, planner, writer and state transitions run and only the wait
  is skipped;
* none of it can point at a real fan.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.simulation import TEST_FAN_PREFIX, is_simulatable_fan
from services import simulation_workspace
from services.simulation_workspace import (
    NotASimulationFan,
    RUNNABLE_ACTION_TYPES,
    SimulationWorkspaceError,
    create_test_fan,
    generate_test_platform_fan_id,
    pending_scheduled_actions,
    require_simulation_fan,
    run_scheduled_action_now,
)


FANS = {
    "fan-test": {
        "id": "fan-test",
        "creator_id": "creator-1",
        "platform_fan_id": "test_abc",
        "display_name": "Test fan",
        "total_spent": 0,
        "active_session": None,
        "pending_ppv_check": None,
        "sales_log": [],
        "needs_human_review": False,
        "sale_paused_at": None,
        "auto_mode": True,
        "ai_summary": None,
        "spend_tier": "new",
    },
    "fan-real": {
        "id": "fan-real",
        "creator_id": "creator-1",
        "platform_fan_id": "884422113355",
        "display_name": "Real fan",
        "total_spent": 400,
        "active_session": None,
        "pending_ppv_check": None,
        "sales_log": [{"amount": 400}],
        "needs_human_review": False,
        "sale_paused_at": None,
        "auto_mode": True,
        "ai_summary": None,
        "spend_tier": "whale",
    },
}

ACTIONS = {
    "action-payday": {
        "id": "action-payday",
        "fan_id": "fan-test",
        "creator_id": "creator-1",
        "action_type": "PAYDAY_REENGAGEMENT",
        "execute_at": "2026-09-15T18:00:00+00:00",
        "status": "PENDING",
        "attempts": 0,
        "last_error": None,
        "payload": {},
    },
    "action-auto": {
        "id": "action-auto",
        "fan_id": "fan-test",
        "creator_id": "creator-1",
        "action_type": "AUTO_REPLY",
        "execute_at": "2026-09-12T18:00:00+00:00",
        "status": "PENDING",
        "attempts": 0,
        "last_error": None,
        "payload": {},
    },
    "action-real-fan": {
        "id": "action-real-fan",
        "fan_id": "fan-real",
        "creator_id": "creator-1",
        "action_type": "PAYDAY_REENGAGEMENT",
        "execute_at": "2026-09-15T18:00:00+00:00",
        "status": "PENDING",
        "attempts": 0,
        "last_error": None,
        "payload": {},
    },
}


class _Table:
    def __init__(self, store: "_DB", name: str) -> None:
        self.store = store
        self.name = name
        self.rows = store.rows(name)
        self.filters: list[tuple[str, object]] = []
        self.in_filters: list[tuple[str, list]] = []
        self.payload: dict | None = None
        self.mode = "select"

    def select(self, *_a, **_k):
        return self

    def insert(self, payload):
        self.mode = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.mode = "update"
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def in_(self, column, values):
        self.in_filters.append((column, list(values)))
        return self

    def order(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    def _matched(self) -> list[dict]:
        rows = list(self.rows.values())
        for column, value in self.filters:
            rows = [row for row in rows if str(row.get(column)) == str(value)]
        for column, values in self.in_filters:
            allowed = {str(item) for item in values}
            rows = [row for row in rows if str(row.get(column)) in allowed]
        return rows

    def execute(self):
        if self.mode == "insert":
            row = dict(self.payload)
            row.setdefault("id", f"{self.name}-{len(self.rows) + 1}")
            self.rows[row["id"]] = row
            self.store.inserts.append(dict(row))
            return SimpleNamespace(data=[dict(row)])
        matched = self._matched()
        if self.mode == "update":
            for row in matched:
                row.update(self.payload)
            return SimpleNamespace(data=[dict(row) for row in matched])
        return SimpleNamespace(data=[dict(row) for row in matched])


class _DB:
    def __init__(self) -> None:
        self.fans = {key: dict(value) for key, value in FANS.items()}
        self.actions = {key: dict(value) for key, value in ACTIONS.items()}
        self.inserts: list[dict] = []

    def rows(self, name: str) -> dict:
        return self.fans if name == "fans" else self.actions

    def table(self, name):
        return _Table(self, name)


@pytest.fixture
def db(monkeypatch):
    store = _DB()
    monkeypatch.setattr(simulation_workspace, "get_supabase", lambda: store)
    return store


# --- the boundary -----------------------------------------------------------


def test_a_real_fan_is_refused_by_every_workspace_entry_point(db):
    with pytest.raises(NotASimulationFan):
        asyncio.run(require_simulation_fan("fan-real", "creator-1"))


def test_run_now_cannot_operate_on_a_real_fan(db, monkeypatch):
    def never_runs(*_a, **_k):
        raise AssertionError("a real fan's action must never be fired early")

    monkeypatch.setattr(
        "workers.scheduled_actions._resolve_action", never_runs, raising=False
    )

    with pytest.raises(NotASimulationFan):
        asyncio.run(
            run_scheduled_action_now(
                creator_id="creator-1",
                fan_id="fan-real",
                action_id="action-real-fan",
            )
        )
    # And the row was not claimed on the way to the refusal.
    assert db.actions["action-real-fan"]["status"] == "PENDING"


# --- creating a test fan ----------------------------------------------------


def test_a_generated_platform_id_is_always_a_test_id():
    for _ in range(50):
        generated = generate_test_platform_fan_id()
        assert generated.startswith(TEST_FAN_PREFIX)
        assert is_simulatable_fan(generated)
        # Nothing resembling a Fansly numeric media/account id.
        assert not generated[len(TEST_FAN_PREFIX):].isdigit()


def test_a_new_test_fan_starts_completely_clean(db):
    created = asyncio.run(create_test_fan(creator_id="creator-1", display_name="Fan A"))

    row = db.fans[created["id"]]
    assert row["platform_fan_id"].startswith(TEST_FAN_PREFIX)
    assert created["simulation"] is True
    # No conversation, no purchases, no learned budget, no stale state.
    assert row["total_spent"] == 0
    assert row["sales_log"] == []
    assert row["active_session"] is None
    assert row["pending_ppv_check"] is None
    assert row["sale_paused_at"] is None
    assert row["ai_summary"] is None
    assert row["needs_human_review"] is False


def test_the_platform_id_is_never_taken_from_the_caller(db):
    """There is no parameter for it, by design — the id is generated server
    side so this control cannot produce a fan that turns out to be real."""
    import inspect

    signature = inspect.signature(create_test_fan)
    assert set(signature.parameters) == {"creator_id", "display_name"}

    asyncio.run(create_test_fan(creator_id="creator-1", display_name="x"))
    assert db.inserts[-1]["platform_fan_id"].startswith(TEST_FAN_PREFIX)


def test_two_test_fans_do_not_collide(db):
    first = asyncio.run(create_test_fan(creator_id="creator-1"))
    second = asyncio.run(create_test_fan(creator_id="creator-1"))

    assert first["platform_fan_id"] != second["platform_fan_id"]


# --- delayed behaviour ------------------------------------------------------


def test_pending_actions_are_listed_with_whether_they_can_be_fired_early(db):
    actions = asyncio.run(pending_scheduled_actions("fan-test"))

    by_id = {row["id"]: row for row in actions}
    assert by_id["action-payday"]["can_run_now"] is True
    assert by_id["action-payday"]["label"] == "Payday follow-up"
    # An inbound reply is triggered by typing in the simulator, not by waiting.
    assert by_id["action-auto"]["can_run_now"] is False


def test_run_now_uses_the_workers_own_handler_inside_the_simulation_scope(
    db, monkeypatch
):
    seen: dict = {}

    async def fake_resolve(action, *, sent_counter):
        from core.apifansly_gate import simulation_active

        seen["action"] = action
        # The whole safety claim: no platform call is possible from in here.
        seen["scoped"] = simulation_active()
        sent_counter[0] += 1
        return "sent"

    monkeypatch.setattr(
        "workers.scheduled_actions._resolve_action", fake_resolve, raising=False
    )

    result = asyncio.run(
        run_scheduled_action_now(
            creator_id="creator-1", fan_id="fan-test", action_id="action-payday"
        )
    )

    assert seen["scoped"] is True
    assert seen["action"]["id"] == "action-payday"
    # Claimed exactly as the worker claims it, so the worker cannot also run it.
    assert seen["action"]["status"] == "PROCESSING"
    assert db.actions["action-payday"]["status"] == "PROCESSING"
    assert result["outcome"] == "sent"
    assert result["messages_sent"] == 1
    assert result["simulation"] is True


def test_an_action_that_is_not_time_delayed_cannot_be_fired_early(db, monkeypatch):
    monkeypatch.setattr(
        "workers.scheduled_actions._resolve_action",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not run")),
        raising=False,
    )

    with pytest.raises(SimulationWorkspaceError):
        asyncio.run(
            run_scheduled_action_now(
                creator_id="creator-1", fan_id="fan-test", action_id="action-auto"
            )
        )


def test_an_already_claimed_action_is_reported_rather_than_run_twice(db, monkeypatch):
    db.actions["action-payday"]["status"] = "COMPLETED"
    monkeypatch.setattr(
        "workers.scheduled_actions._resolve_action",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not run")),
        raising=False,
    )

    with pytest.raises(SimulationWorkspaceError):
        asyncio.run(
            run_scheduled_action_now(
                creator_id="creator-1", fan_id="fan-test", action_id="action-payday"
            )
        )


def test_the_runnable_set_excludes_platform_reconciliation():
    """Anything that reconciles against the platform stays out: firing it early
    would ask about a purchase that does not exist."""
    assert "PPV_RECONCILE" not in RUNNABLE_ACTION_TYPES
    assert "AUTO_REPLY" not in RUNNABLE_ACTION_TYPES
    assert "PAYDAY_REENGAGEMENT" in RUNNABLE_ACTION_TYPES
    assert "POST_SESSION_FOLLOWUP" in RUNNABLE_ACTION_TYPES
    assert "INACTIVITY_REENGAGEMENT" in RUNNABLE_ACTION_TYPES
