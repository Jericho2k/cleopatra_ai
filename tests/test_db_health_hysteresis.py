"""A blip is not an outage, and an outage is not a blip.

The dashboard kept showing "Cleopatra cannot reach its database. Messages are
not being processed." while every ordinary endpoint immediately before and after
worked and ``/health`` answered 200. One recycled PostgREST connection under a
probe was enough: the probe failed once, ``evaluate`` called that fatal, and the
banner then sat there until the next sixty-second poll.

Both halves of that are wrong, and both are fixed here:

* a single failed probe no longer produces an operator notice at all;
* recovery is immediate and unconditional — the first success clears everything,
  with no cooling-off timer for an operator to wait out.

Real outages are still reported. They just have to be confirmed first, by count
AND by duration, so neither a burst of fast probes nor two failures an hour apart
can manufacture one.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.db_health_state import (
    DatabaseHealthState,
    DatabaseHealthTracker,
    sustained_seconds,
    unavailable_after_failures,
    unstable_after_failures,
)
from services import operational_health


class FakeClock:
    def __init__(self) -> None:
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def tracker(monkeypatch):
    monkeypatch.setenv("HEALTH_DB_UNSTABLE_AFTER_FAILURES", "2")
    monkeypatch.setenv("HEALTH_DB_UNAVAILABLE_AFTER_FAILURES", "4")
    monkeypatch.setenv("HEALTH_DB_SUSTAINED_SECONDS", "30")
    clock = FakeClock()
    return DatabaseHealthTracker(clock=clock), clock


# ---------------------------------------------------------------------------
# The ladder
# ---------------------------------------------------------------------------


def test_a_single_failure_is_recorded_and_shown_to_nobody(tracker):
    health, _clock = tracker
    verdict = health.observe(reachable=False, error="RemoteProtocolError")

    assert verdict.state is DatabaseHealthState.UNCONFIRMED
    assert verdict.fatal is False
    assert verdict.degraded is False
    assert verdict.reason.startswith("database_probe_failed_unconfirmed")


def test_repeated_failures_are_unstable_not_fatal(tracker):
    health, clock = tracker
    health.observe(reachable=False, error="timeout")
    clock.advance(5)
    verdict = health.observe(reachable=False, error="timeout")

    assert verdict.state is DatabaseHealthState.UNSTABLE
    assert verdict.fatal is False
    assert verdict.degraded is True
    assert verdict.reason == "database_unstable:timeout"


def test_a_confirmed_sustained_outage_is_fatal(tracker):
    health, clock = tracker
    for _ in range(4):
        clock.advance(10)
        verdict = health.observe(reachable=False, error="timeout")

    assert verdict.state is DatabaseHealthState.UNAVAILABLE
    assert verdict.fatal is True
    assert verdict.reason == "database_unavailable:timeout"
    assert verdict.consecutive_failures == 4


def test_count_alone_cannot_declare_an_outage(tracker):
    """A dashboard polling in a tight loop must not manufacture one."""
    health, _clock = tracker  # clock never advances
    for _ in range(20):
        verdict = health.observe(reachable=False, error="timeout")

    assert verdict.consecutive_failures == 20
    assert verdict.state is DatabaseHealthState.UNSTABLE
    assert verdict.fatal is False


def test_duration_alone_cannot_declare_an_outage(tracker):
    """Two failures an hour apart are not an outage either."""
    health, clock = tracker
    health.observe(reachable=False, error="timeout")
    clock.advance(3600)
    verdict = health.observe(reachable=False, error="timeout")

    assert verdict.failing_for_seconds >= 3600
    assert verdict.state is DatabaseHealthState.UNSTABLE
    assert verdict.fatal is False


def test_recovery_is_immediate_and_unconditional(tracker):
    health, clock = tracker
    for _ in range(6):
        clock.advance(30)
        health.observe(reachable=False, error="timeout")
    assert health.snapshot().fatal is True

    verdict = health.observe(reachable=True)

    assert verdict.state is DatabaseHealthState.HEALTHY
    assert verdict.fatal is False
    assert verdict.consecutive_failures == 0
    assert verdict.reason is None
    # And the run genuinely restarts rather than resuming where it left off.
    assert health.observe(reachable=False, error="timeout").state is (
        DatabaseHealthState.UNCONFIRMED
    )


def test_a_success_in_the_middle_of_a_run_resets_the_count(tracker):
    health, clock = tracker
    health.observe(reachable=False, error="timeout")
    clock.advance(10)
    health.observe(reachable=False, error="timeout")
    health.observe(reachable=True)
    clock.advance(10)

    verdict = health.observe(reachable=False, error="timeout")
    assert verdict.consecutive_failures == 1
    assert verdict.state is DatabaseHealthState.UNCONFIRMED


def test_snapshot_does_not_advance_the_ladder(tracker):
    health, _clock = tracker
    health.observe(reachable=False, error="timeout")
    for _ in range(5):
        assert health.snapshot().consecutive_failures == 1


def test_thresholds_are_configurable_and_ordered(monkeypatch):
    monkeypatch.setenv("HEALTH_DB_UNSTABLE_AFTER_FAILURES", "5")
    monkeypatch.setenv("HEALTH_DB_UNAVAILABLE_AFTER_FAILURES", "2")
    # A misconfiguration must not make "unavailable" easier to reach than
    # "unstable"; the ladder is clamped rather than inverted.
    assert unavailable_after_failures() >= unstable_after_failures()
    assert sustained_seconds() >= 0


# ---------------------------------------------------------------------------
# What the health document publishes
# ---------------------------------------------------------------------------


def _evaluate(database: dict) -> dict:
    return operational_health.evaluate(
        database=database,
        queue={"available": True, "pending": 0, "oldest_pending_age_seconds": 0},
        scheduler={
            "poll_seconds": 5,
            "seconds_since_last_cycle": 1.0,
            "cycles_completed": 2,
        },
        model_gate={},
        model_availability={"status": "ok"},
    )


def _failed_probe(state: DatabaseHealthState, error: str = "timeout") -> dict:
    return {
        "reachable": False,
        "latency_ms": 1500,
        "error": error,
        "confirmation": {"state": state.value, "consecutive_failures": 1},
    }


def test_an_unconfirmed_failure_produces_no_operator_notice():
    verdict = _evaluate(_failed_probe(DatabaseHealthState.UNCONFIRMED))

    assert verdict["fatal_reasons"] == []
    assert verdict["status"] == "degraded"
    assert verdict["degraded_reasons"] == ["database_probe_failed_unconfirmed:timeout"]


def test_an_unstable_database_is_degraded_never_fatal():
    verdict = _evaluate(_failed_probe(DatabaseHealthState.UNSTABLE))

    assert verdict["fatal_reasons"] == []
    assert "database_unstable:timeout" in verdict["degraded_reasons"]


def test_a_confirmed_outage_is_fatal():
    verdict = _evaluate(_failed_probe(DatabaseHealthState.UNAVAILABLE))

    assert verdict["status"] == "unhealthy"
    assert "database_unavailable:timeout" in verdict["fatal_reasons"]


def test_a_probe_with_no_confirmation_record_still_fails_closed():
    """An injected result, or a caller that built the dict by hand. Leniency
    here would be a way to hide a real outage."""
    verdict = _evaluate({"reachable": False, "latency_ms": 10, "error": "timeout"})

    assert verdict["status"] == "unhealthy"
    assert "database_unavailable:timeout" in verdict["fatal_reasons"]


def test_a_reachable_database_publishes_nothing_about_itself():
    verdict = _evaluate({"reachable": True, "latency_ms": 8, "error": None})

    assert verdict["fatal_reasons"] == []
    assert not any("database" in reason for reason in verdict["degraded_reasons"])


def test_the_probe_records_into_the_process_tracker(monkeypatch):
    """End to end: a failing probe walks the ladder rather than jumping it."""
    from types import SimpleNamespace

    state = {"fail": True}

    class _Probe:
        def table(self, _name):
            return self

        def select(self, *_a, **_k):
            return self

        def limit(self, _v):
            return self

        def execute(self):
            if state["fail"]:
                raise RuntimeError("connection terminated")
            return SimpleNamespace(data=[{"id": "creator-1"}])

    monkeypatch.setattr("core.supabase.get_supabase", lambda: _Probe())
    monkeypatch.setenv("HEALTH_DB_UNSTABLE_AFTER_FAILURES", "2")
    monkeypatch.setenv("HEALTH_DB_UNAVAILABLE_AFTER_FAILURES", "3")
    monkeypatch.setenv("HEALTH_DB_SUSTAINED_SECONDS", "0")
    operational_health.reset_cache()

    first = asyncio.run(operational_health.probe_database())
    assert first["confirmation"]["state"] == DatabaseHealthState.UNCONFIRMED.value
    assert _evaluate(first)["fatal_reasons"] == []

    second = asyncio.run(operational_health.probe_database())
    assert second["confirmation"]["state"] == DatabaseHealthState.UNSTABLE.value
    assert _evaluate(second)["fatal_reasons"] == []

    third = asyncio.run(operational_health.probe_database())
    assert third["confirmation"]["state"] == DatabaseHealthState.UNAVAILABLE.value
    assert _evaluate(third)["status"] == "unhealthy"

    state["fail"] = False
    recovered = asyncio.run(operational_health.probe_database())
    assert recovered["reachable"] is True
    assert _evaluate(recovered)["status"] == "ok"
    operational_health.reset_cache()


def test_the_pending_message_identity_migration_is_published(monkeypatch):
    """Ingestion still works without the unique index, which is exactly why its
    absence has to be visible rather than logged once into Railway."""
    import db.queries as queries

    monkeypatch.setattr(queries, "_message_identity_index_missing", True)
    verdict = _evaluate({"reachable": True, "latency_ms": 8, "error": None})

    assert "message_identity_index_missing" in verdict["degraded_reasons"]
    assert verdict["fatal_reasons"] == [], "a pending migration is not an outage"
