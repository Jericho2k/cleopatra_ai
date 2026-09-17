"""Simulated time, and the fact that it cannot happen in production.

The continuation brief asks the harness for "a clock injected through expiry,
scheduling and continuity". Before this, `Disturbance.days_since_previous` was
a number nobody read: `run_trajectory` called `advance_clock` only when a
caller supplied one, and no caller did. The trajectory labelled "he comes back
a day later, then a week later" ran its four turns back to back while the label
said a week passed.

Two things have to hold, and they pull against each other:

  * an evaluation can move time far enough that a queued follow-up becomes due
    and a thread expires — otherwise three of review §5's rows are untestable;
  * a production process cannot move time at all, because an offset clock there
    fires real scheduled work at the wrong moment for real customers.

The second is the one with teeth, so it is tested first and tested hardest.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from core import clock


@pytest.fixture(autouse=True)
def _wall_clock():
    """Never leave an offset behind for another test."""
    clock.reset()
    yield
    clock.reset()


@pytest.fixture
def movable(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv(clock.EVAL_CLOCK_FLAG, "1")
    return clock


# ===========================================================================
# 1. It cannot happen in production
# ===========================================================================


def test_production_cannot_move_its_clock_even_with_the_flag_set(monkeypatch):
    """The flag is not enough. Two conditions, and neither is the default."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv(clock.EVAL_CLOCK_FLAG, "1")

    assert clock.movable() is False
    with pytest.raises(clock.ClockNotMovable):
        clock.advance(7)
    assert clock.offset() == timedelta(0)


def test_a_non_production_process_without_the_flag_cannot_move_either(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.delenv(clock.EVAL_CLOCK_FLAG, raising=False)

    assert clock.movable() is False
    with pytest.raises(clock.ClockNotMovable):
        clock.advance(1)


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
def test_the_flag_is_not_set_by_anything_that_merely_looks_set(monkeypatch, value):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv(clock.EVAL_CLOCK_FLAG, value)

    assert clock.movable() is False


def test_refusing_raises_rather_than_doing_nothing(monkeypatch):
    """The bug this module replaces, one level up.

    A harness that quietly declined to advance reported a week as having
    passed when it had not. Refusing loudly is what lets the caller report the
    run as uncovered instead.
    """
    monkeypatch.setenv("APP_ENV", "production")

    with pytest.raises(clock.ClockNotMovable) as refused:
        clock.advance(7)

    assert clock.EVAL_CLOCK_FLAG in str(refused.value)


def test_an_unmoved_clock_is_the_wall_clock(monkeypatch):
    """Every deployment, by default. Identical to datetime.now(timezone.utc)."""
    monkeypatch.delenv(clock.EVAL_CLOCK_FLAG, raising=False)

    before = datetime.now(timezone.utc)
    value = clock.now()
    after = datetime.now(timezone.utc)

    assert before <= value <= after
    assert clock.offset_seconds() == 0.0
    assert clock.simulated_now_for_sql() is None


# ===========================================================================
# 2. When it is allowed, it actually moves
# ===========================================================================


def test_advancing_moves_now_forward(movable):
    before = clock.now()
    movable.advance(7)

    assert clock.now() - before >= timedelta(days=7)
    assert movable.offset() == timedelta(days=7)


def test_advances_accumulate(movable):
    movable.advance(1)
    movable.advance(7)

    assert movable.offset() == timedelta(days=8)


def test_advancing_by_nothing_is_allowed_everywhere(monkeypatch):
    """A trajectory turn with no gap must not need the flag to run."""
    monkeypatch.setenv("APP_ENV", "production")

    clock.advance(0)  # no raise

    assert clock.offset() == timedelta(0)


def test_reset_returns_to_the_wall_clock(movable):
    movable.advance(30)
    movable.reset()

    assert movable.offset() == timedelta(0)
    assert movable.simulated_now_for_sql() is None


def test_reset_is_allowed_in_production(monkeypatch):
    """Moving BACK to real time cannot make a deployment act early."""
    monkeypatch.setenv("APP_ENV", "production")

    clock.reset()  # no raise


def test_sql_is_only_told_the_time_while_time_is_simulated(movable):
    assert clock.simulated_now_for_sql() is None

    movable.advance(2)
    supplied = clock.simulated_now_for_sql()

    assert supplied is not None
    assert datetime.fromisoformat(supplied) > datetime.now(timezone.utc)


# ===========================================================================
# 3. Continuity reads it
# ===========================================================================


def test_thread_expiry_sees_the_simulated_time(movable):
    """An unanswered question has to be able to age.

    services/conversation_continuity._now is the module's single clock seam,
    so this is the whole of "continuity reads the eval clock".
    """
    from services import conversation_continuity

    before = conversation_continuity._now()
    movable.advance(50)

    assert conversation_continuity._now() - before >= timedelta(days=50)


# ===========================================================================
# 4. Scheduling reads it — against real PostgreSQL
# ===========================================================================
#
# This is the half a Python clock cannot do on its own. Whether a scheduled
# action is due is decided inside claim_due_actions, by SQL's own now(), so
# without db/eval_clock_claim_v1.sql the harness could move every clock in the
# process and the one thing "a queued follow-up becomes due" is about would not
# move at all.

psycopg = pytest.importorskip("psycopg", reason="psycopg is required for schema tests")

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()

schema_only = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; claim tests need a real PostgreSQL",
)

DB = Path(__file__).resolve().parents[1] / "db"


def _order() -> list[str]:
    return [
        line.strip()
        for line in (DB / "migration_order.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture(scope="module")
def scheduled():
    """A creator, a fan, and one action that is not due for three days."""
    name = f"cleo_clock_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)

    def scoped(sql: str) -> str:
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            cursor.execute((DB / "ci_supabase_stubs.sql").read_text())
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(scoped((DB / "ci_baseline_schema.sql").read_text()))
            for filename in _order():
                cursor.execute(scoped((DB / filename).read_text()))

            cursor.execute(
                f'insert into "{name}".creators (name) values (%s) returning id',
                ("Agency",),
            )
            creator_id = cursor.fetchone()[0]
            cursor.execute(
                f'insert into "{name}".fans (creator_id, display_name, platform_fan_id) '
                "values (%s, %s, %s) returning id",
                (creator_id, "Fan", "test_p1"),
            )
            fan_id = cursor.fetchone()[0]
        yield connection, name, {"creator_id": creator_id, "fan_id": fan_id}
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _queue_followup(cursor, schema, data, *, days_out: float) -> str:
    cursor.execute(
        f'insert into "{schema}".scheduled_actions '
        "(fan_id, creator_id, action_type, status, execute_at) "
        "values (%s, %s, 'AUTO_REPLY', 'PENDING', now() + make_interval(secs => %s)) "
        "returning id",
        (data["fan_id"], data["creator_id"], days_out * 86400),
    )
    return cursor.fetchone()[0]


@schema_only
def test_a_follow_up_three_days_out_is_not_due_yet(scheduled):
    connection, name, data = scheduled

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            _queue_followup(cursor, name, data, days_out=3)
            cursor.execute(f'select count(*) from "{name}".claim_due_actions(20, 10)')
            assert cursor.fetchone()[0] == 0
        finally:
            cursor.execute("rollback")


@schema_only
def test_telling_the_claim_it_is_next_week_makes_it_due(scheduled):
    """The row §5 calls "a previously queued follow-up becomes due".

    Untestable before this: the harness could not move SQL's now(), so the
    action stayed three days out however far the process advanced its own
    clock.
    """
    connection, name, data = scheduled

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            action_id = _queue_followup(cursor, name, data, days_out=3)
            cursor.execute(
                f'select id from "{name}".claim_due_actions(20, 10, now() + interval \'7 days\')'
            )
            claimed = [row[0] for row in cursor.fetchall()]
            assert claimed == [action_id]

            cursor.execute(
                f'select status from "{name}".scheduled_actions where id = %s',
                (action_id,),
            )
            assert cursor.fetchone()[0] == "PROCESSING"
        finally:
            cursor.execute("rollback")


@schema_only
def test_omitting_the_time_still_uses_the_database_clock(scheduled):
    """Every existing caller. The new parameter must change nothing for them."""
    connection, name, data = scheduled

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            overdue = _queue_followup(cursor, name, data, days_out=-1)
            _queue_followup(cursor, name, data, days_out=3)
            cursor.execute(f'select id from "{name}".claim_due_actions(20, 10)')
            assert [row[0] for row in cursor.fetchall()] == [overdue]
        finally:
            cursor.execute("rollback")


@schema_only
def test_there_is_exactly_one_claim_function(scheduled):
    """An ambiguous overload set is worse than either member of it.

    scheduled_action_claim_v1 created a two-argument version; PostgreSQL would
    happily keep both, and which one a two-argument call resolves to would stop
    being obvious.
    """
    connection, name, _ = scheduled

    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
            "where n.nspname = %s and p.proname = 'claim_due_actions'",
            (name,),
        )
        assert cursor.fetchone()[0] == 1


@schema_only
def test_the_browser_roles_cannot_call_it_at_all(scheduled):
    """Least privilege survives the signature change."""
    connection, name, _ = scheduled

    with connection.cursor() as cursor:
        for role in ("anon", "authenticated"):
            cursor.execute(
                "select has_function_privilege(%s, p.oid, 'execute') "
                "from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
                "where n.nspname = %s and p.proname = 'claim_due_actions'",
                (role, name),
            )
            assert cursor.fetchone()[0] is False, role


# ===========================================================================
# 5. A clock that refuses does not take the conversation down with it
# ===========================================================================


def test_a_refusing_clock_is_reported_and_the_trajectory_still_runs():
    """Running the harness in a process that may not simulate time.

    The turns are still worth running; what must not happen is the report
    claiming the week passed. So the refusal is a finding, the elapsed-time
    claim goes uncovered, and every turn still executes.
    """
    from services.trajectory_eval import (
        Disturbance,
        Trajectory,
        run_trajectory,
    )

    sent: list[str] = []

    async def send_turn(message: str) -> dict:
        sent.append(message)
        return {"outcome": "replied", "creator_messages": [{"content": "ok"}]}

    def refusing(_days: float) -> None:
        raise clock.ClockNotMovable("not here")

    trajectory = Trajectory(
        name="t",
        disturbances=(
            Disturbance(message="hey"),
            Disturbance(message="back after a week", days_since_previous=7.0),
        ),
        requires_elapsed_days=7,
    )

    report = asyncio.run(
        run_trajectory(trajectory, send_turn=send_turn, advance_clock=refusing)
    )

    assert sent == ["hey", "back after a week"]
    assert "clock_did_not_advance" in [f.kind for f in report.findings]
    assert report.elapsed_days == 0.0
    assert [gap.claim for gap in report.coverage_gaps] == ["elapsed time"]


def test_a_working_clock_covers_the_elapsed_time_claim(movable):
    """The other half: with the clock on, the claim is actually covered."""
    from services.trajectory_eval import (
        Disturbance,
        Trajectory,
        run_trajectory,
    )

    async def send_turn(_message: str) -> dict:
        return {"outcome": "replied", "creator_messages": [{"content": "ok"}]}

    trajectory = Trajectory(
        name="t",
        disturbances=(
            Disturbance(message="hey"),
            Disturbance(message="a day later", days_since_previous=1.0),
            Disturbance(message="a week later", days_since_previous=7.0),
        ),
        requires_elapsed_days=8,
    )

    report = asyncio.run(
        run_trajectory(trajectory, send_turn=send_turn, advance_clock=movable.advance)
    )

    assert report.clock_injected is True
    assert report.elapsed_days == 8.0
    assert report.coverage_gaps == []
    assert movable.offset() == timedelta(days=8)
