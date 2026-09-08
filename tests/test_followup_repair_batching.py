"""SCALE-002: follow-up obligation repair must stay a safety net, cheaply.

The invariant is unchanged — if commercial state says a follow-up is owed and
its durable action is missing or terminal, recreate it. What changes is the
cost: a due window instead of a full scan, and three round trips instead of two
per outstanding obligation.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from db import commercial_queries
from workers import scheduled_actions as worker

NOW = datetime.now(timezone.utc)


def run(coro):
    return asyncio.run(coro)


def obligation(fan: str, *, due_in_seconds: int, key: str | None = None) -> dict:
    return {
        "creator_id": "creator-1",
        "fan_id": fan,
        "next_followup_at": (NOW + timedelta(seconds=due_in_seconds)).isoformat(),
        "next_followup_type": "POST_SESSION_FOLLOWUP",
        "next_followup_payload": {"experience": "shower"},
        "next_followup_dedupe_key": key or f"post-session:{fan}",
    }


class RepairSpy:
    """Counts round trips the way the audit measured them."""

    def __init__(self, rows: list[dict], existing: dict[str, dict] | None = None):
        self.rows = rows
        self.existing = existing or {}
        self.round_trips = 0
        self.horizons: list[datetime | None] = []
        self.upserted: list[dict] = []
        self.lookup_batches: list[list[str]] = []

    async def get_obligations(self, page_size: int = 500, *, due_before=None):
        self.round_trips += 1
        self.horizons.append(due_before)
        if due_before is None:
            return list(self.rows)
        return [
            row for row in self.rows
            if not row.get("next_followup_at")
            or datetime.fromisoformat(row["next_followup_at"]) <= due_before
        ]

    async def get_states(self, keys):
        self.round_trips += 1
        self.lookup_batches.append(list(keys))
        return {k: v for k, v in self.existing.items() if k in set(keys)}

    async def bulk_upsert(self, rows):
        self.round_trips += 1
        self.upserted.extend(rows)
        return len(rows)


def install(monkeypatch, spy: RepairSpy) -> None:
    monkeypatch.setattr(worker, "get_followup_obligations", spy.get_obligations)
    monkeypatch.setattr(worker, "get_action_states_by_dedupe_key", spy.get_states)
    monkeypatch.setattr(worker, "bulk_upsert_pending_actions", spy.bulk_upsert)


def test_obligation_due_next_week_is_not_scanned(monkeypatch):
    spy = RepairSpy([obligation("fan-1", due_in_seconds=7 * 24 * 3600)])
    install(monkeypatch, spy)

    assert run(worker.repair_followup_obligations()) == 0
    # It never even reached the action lookup, let alone a write.
    assert spy.round_trips == 1
    assert spy.upserted == []
    # And the horizon really is a short upcoming window.
    horizon = spy.horizons[0]
    assert horizon is not None
    assert timedelta(seconds=0) < horizon - NOW <= timedelta(minutes=10)


def test_obligation_entering_the_horizon_becomes_durable(monkeypatch):
    spy = RepairSpy([obligation("fan-1", due_in_seconds=120)])
    install(monkeypatch, spy)

    assert run(worker.repair_followup_obligations()) == 1
    assert [row["fan_id"] for row in spy.upserted] == ["fan-1"]
    assert spy.upserted[0]["status"] == "PENDING"
    assert spy.upserted[0]["action_type"] == "POST_SESSION_FOLLOWUP"


def test_missing_action_is_repaired_and_a_correct_pending_one_is_left_alone(monkeypatch):
    spy = RepairSpy(
        [
            obligation("fan-missing", due_in_seconds=60, key="k-missing"),
            obligation("fan-pending", due_in_seconds=60, key="k-pending"),
            obligation("fan-processing", due_in_seconds=60, key="k-processing"),
        ],
        existing={
            "k-pending": {"id": "a1", "status": "PENDING", "last_error": None},
            "k-processing": {"id": "a2", "status": "PROCESSING", "last_error": None},
        },
    )
    install(monkeypatch, spy)

    assert run(worker.repair_followup_obligations()) == 1
    assert [row["dedupe_key"] for row in spy.upserted] == ["k-missing"]


def test_completed_action_for_a_still_current_obligation_is_recreated(monkeypatch):
    spy = RepairSpy(
        [obligation("fan-1", due_in_seconds=60, key="k1")],
        existing={"k1": {"id": "a1", "status": "COMPLETED", "last_error": None}},
    )
    install(monkeypatch, spy)

    assert run(worker.repair_followup_obligations()) == 1
    assert spy.upserted[0]["dedupe_key"] == "k1"
    assert spy.upserted[0]["attempts"] == 0
    assert spy.upserted[0]["locked_at"] is None


def test_obsolete_obligation_rows_are_not_resurrected(monkeypatch):
    """A row with no type or no dedupe key is not an obligation. Skip it."""
    rows = [
        {
            "creator_id": "c",
            "fan_id": "fan-a",
            "next_followup_at": NOW.isoformat(),
            "next_followup_type": None,
            "next_followup_payload": {},
            "next_followup_dedupe_key": "k",
        },
        {
            "creator_id": "c",
            "fan_id": "fan-b",
            "next_followup_at": NOW.isoformat(),
            "next_followup_type": "POST_SESSION_FOLLOWUP",
            "next_followup_payload": {},
            "next_followup_dedupe_key": "",
        },
        {
            "creator_id": "c",
            "fan_id": "fan-c",
            "next_followup_at": None,
            "next_followup_type": "POST_SESSION_FOLLOWUP",
            "next_followup_payload": {},
            "next_followup_dedupe_key": "k2",
        },
    ]
    spy = RepairSpy(rows)
    install(monkeypatch, spy)

    assert run(worker.repair_followup_obligations()) == 0
    assert spy.upserted == []


def test_cancelled_or_failed_actions_are_left_alone(monkeypatch):
    spy = RepairSpy(
        [
            obligation("fan-cancelled", due_in_seconds=60, key="k-cancelled"),
            obligation("fan-failed", due_in_seconds=60, key="k-failed"),
        ],
        existing={
            "k-cancelled": {"id": "a1", "status": "CANCELLED", "last_error": None},
            "k-failed": {"id": "a2", "status": "FAILED", "last_error": "send refused"},
        },
    )
    install(monkeypatch, spy)

    assert run(worker.repair_followup_obligations()) == 0


def test_rolling_deploy_compatibility_failure_is_still_recovered(monkeypatch):
    """Preserved from the previous sprint: this exact failure must self-heal."""
    spy = RepairSpy(
        [obligation("fan-1", due_in_seconds=60, key="k1")],
        existing={
            "k1": {
                "id": "a1",
                "status": "FAILED",
                "last_error": "module has no attribute get_creator_auto_mode_default",
            }
        },
    )
    install(monkeypatch, spy)
    assert run(worker.repair_followup_obligations()) == 1


def test_two_hundred_obligations_do_not_cost_hundreds_of_round_trips(monkeypatch):
    """Round-trip regression guard. The audit measured 401 for 200 rows."""
    rows = [
        obligation(f"fan-{i}", due_in_seconds=60, key=f"k-{i}")
        for i in range(200)
    ]
    spy = RepairSpy(rows)
    install(monkeypatch, spy)

    assert run(worker.repair_followup_obligations()) == 200
    # obligations page + one key lookup + one bulk upsert.
    assert spy.round_trips == 3
    assert spy.round_trips < 10
    assert len(spy.upserted) == 200


def test_key_lookups_are_chunked_for_url_safety(monkeypatch):
    rows = [
        obligation(f"fan-{i}", due_in_seconds=60, key=f"k-{i}")
        for i in range(450)
    ]
    spy = RepairSpy(rows)
    install(monkeypatch, spy)
    run(worker.repair_followup_obligations())
    # The worker hands the whole key list to the query layer, which chunks it.
    assert len(spy.lookup_batches[0]) == 450
    assert commercial_queries._DEDUPE_KEY_CHUNK <= 200


def test_query_layer_chunks_key_lookups(monkeypatch):
    executed: list[int] = []

    class Query:
        def table(self, _name):
            return self

        def select(self, _cols):
            return self

        def in_(self, _col, values):
            executed.append(len(values))
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    monkeypatch.setattr(commercial_queries, "get_supabase", lambda: Query())
    run(commercial_queries.get_action_states_by_dedupe_key([f"k-{i}" for i in range(450)]))
    assert executed == [200, 200, 50]


def test_due_window_reaches_the_database_query(monkeypatch):
    """The filter must be applied server-side, not after fetching everything."""
    filters: list[tuple[str, str]] = []

    class Query:
        def table(self, _name):
            return self

        def select(self, _cols):
            return self

        @property
        def not_(self):
            return self

        def is_(self, col, value):
            return self

        def lte(self, col, value):
            filters.append((col, value))
            return self

        def order(self, _col):
            return self

        def range(self, _start, _end):
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    monkeypatch.setattr(commercial_queries, "get_supabase", lambda: Query())
    horizon = NOW + timedelta(minutes=5)
    run(commercial_queries.get_followup_obligations(due_before=horizon))
    assert filters == [("next_followup_at", horizon.isoformat())]


def test_repair_runs_on_its_own_cadence_not_every_claim_poll(monkeypatch):
    """A 5-second claim poll must not multiply repair cost by twelve."""
    calls = []

    async def repair(**kwargs):
        calls.append(kwargs)
        return 0

    async def claim(**_kwargs):
        return []

    monkeypatch.setattr(worker, "repair_followup_obligations", repair)
    monkeypatch.setattr(worker, "claim_due_actions", claim)

    run(worker.process_cycle(run_repair=False))
    assert calls == []
    run(worker.process_cycle(run_repair=True))
    assert len(calls) == 1
    assert worker.REPAIR_INTERVAL_SECONDS >= 60
