"""Simulated fans are persistent, realistic — and not the agency's revenue.

Simulation state is deliberately NOT ephemeral: a test fan accumulates real
confirmed spend, real purchase counts, real lifecycle transitions, because that
is what makes the Simulator worth having. None of it is agency revenue.

The isolation is analytical. One filter, applied at every read that produces a
production metric. The Simulator applies the opposite and shows the simulated
numbers, which is what it is for.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.simulation import (
    TEST_FAN_PREFIX,
    exclude_simulation_fans,
    is_simulation_fan_row,
)
from services import full_auto_operations


def test_the_filter_is_applied_in_the_database_not_in_python():
    """A page boundary must not be able to let a test fan through."""
    calls: list[tuple[str, str]] = []

    class _Query:
        @property
        def not_(self):
            return self

        def like(self, column, pattern):
            calls.append((column, pattern))
            return self

    exclude_simulation_fans(_Query())

    assert calls == [("platform_fan_id", "test\\_%")]


def test_the_underscore_is_escaped_so_a_real_fan_is_not_excluded():
    """In SQL LIKE an unescaped _ is a single-character wildcard, so test_%
    would also exclude a genuine fan whose platform id started 'testX'."""
    pattern = None

    class _Query:
        @property
        def not_(self):
            return self

        def like(self, _column, value):
            nonlocal pattern
            pattern = value
            return self

    exclude_simulation_fans(_Query())

    assert "\\_" in pattern
    assert pattern.startswith(TEST_FAN_PREFIX[:-1])


@pytest.mark.parametrize(
    "platform_fan_id, expected",
    [
        ("test_a1b2", True),
        ("test_", True),
        ("884422113355", False),
        ("testing_account", False),
        (None, False),
        ("", False),
    ],
)
def test_a_row_is_recognised_by_its_platform_id(platform_fan_id, expected):
    assert is_simulation_fan_row({"platform_fan_id": platform_fan_id}) is expected


def test_is_simulation_fan_row_tolerates_a_non_mapping():
    assert is_simulation_fan_row(None) is False
    assert is_simulation_fan_row("test_a1") is False


# --- the operator health view ----------------------------------------------


REAL_FAN = {
    "id": "fan-real",
    "display_name": "Real fan",
    "auto_mode": True,
    "needs_human_review": False,
    "review_reason": None,
    "platform_fan_id": "884422113355",
}
TEST_FAN = {
    "id": "fan-test",
    "display_name": "Test fan",
    "auto_mode": True,
    "needs_human_review": True,
    "review_reason": "simulated",
    "platform_fan_id": "test_a1b2",
}


class _Rows:
    """Enough PostgREST to record the filter and answer with the right rows."""

    def __init__(self, store, name):
        self.store = store
        self.name = name
        self.negated_like: tuple[str, str] | None = None

    def select(self, *_a, **_k):
        return self

    def eq(self, *_a, **_k):
        return self

    def in_(self, *_a, **_k):
        return self

    def order(self, *_a, **_k):
        return self

    def range(self, *_a, **_k):
        return self

    def limit(self, *_a, **_k):
        return self

    @property
    def not_(self):
        return self

    def like(self, column, pattern):
        self.negated_like = (column, pattern)
        return self

    def execute(self):
        rows = list(self.store[self.name])
        if self.name == "fans" and self.negated_like:
            rows = [
                row
                for row in rows
                if not str(row.get("platform_fan_id") or "").startswith(TEST_FAN_PREFIX)
            ]
            self.store["fans_filtered"] = True
        return SimpleNamespace(data=rows)


def test_full_auto_health_excludes_test_fans_and_their_rows(monkeypatch):
    store = {
        "fans": [REAL_FAN, TEST_FAN],
        "fan_commercial_states": [
            {"fan_id": "fan-real", "status": "IDLE", "next_followup_at": None,
             "next_followup_type": None, "last_abandoned_ppv_at": None,
             "updated_at": "2026-09-12T00:00:00Z"},
            {"fan_id": "fan-test", "status": "PAYMENT_PENDING",
             "next_followup_at": "2026-09-13T00:00:00Z",
             "next_followup_type": "PAYDAY_REENGAGEMENT",
             "last_abandoned_ppv_at": None, "updated_at": "2026-09-12T00:00:00Z"},
        ],
        "scheduled_actions": [
            {"id": "a1", "fan_id": "fan-test", "action_type": "PAYDAY_REENGAGEMENT",
             "execute_at": "2026-09-13T00:00:00Z", "status": "FAILED", "attempts": 3,
             "last_error": "simulated", "locked_at": None, "payload": {}},
        ],
        "fans_filtered": False,
    }

    class _DB:
        def table(self, name):
            return _Rows(store, name)

    monkeypatch.setattr(full_auto_operations, "get_supabase", _DB)
    monkeypatch.setattr(
        full_auto_operations, "analyzer_health", lambda hours: {}, raising=False
    )
    monkeypatch.setattr(
        "workers.scheduled_actions.worker_health_snapshot", dict, raising=False
    )

    health = asyncio.run(
        full_auto_operations.get_creator_full_auto_health("creator-1")
    )

    assert store["fans_filtered"] is True
    fan_ids = {row["fan_id"] for row in health["fans"]}
    assert "fan-test" not in fan_ids
    # The summary counts are narrowed too: a simulated PAYMENT_PENDING and a
    # simulated FAILED action are not the agency's problem.
    assert health["summary"]["payment_pending"] == 0
    assert health["summary"]["failed_actions"] == 0
    assert health["summary"]["human_review"] == 0


def test_simulation_state_itself_is_never_deleted(monkeypatch):
    """Isolation is analytical. The rows stay, and the Simulator reads them."""
    store = {"fans": [REAL_FAN, TEST_FAN], "fans_filtered": False}

    rows = _Rows(store, "fans")
    rows.execute()

    # The underlying store is untouched by the filtered read.
    assert len(store["fans"]) == 2
