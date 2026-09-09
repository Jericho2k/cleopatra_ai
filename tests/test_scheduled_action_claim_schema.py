"""The atomic claim must keep every ownership guarantee the CAS gave us.

db/scheduled_action_claim_v1.sql replaces two selects plus one compare-and-swap
UPDATE per row with a single statement. That is only worth doing if it is at
least as safe, so these run against a real PostgreSQL: an in-memory double
cannot demonstrate that two concurrent claimers do not both win a row, and that
is the entire property.

Skipped unless TEST_DATABASE_URL points at a disposable PostgreSQL.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg", reason="psycopg is required for schema tests")

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; schema tests need a real PostgreSQL",
)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db"


def _order() -> list[str]:
    lines = (DB / "migration_order.txt").read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture
def schema():
    """A disposable schema with the pipeline applied, plus one creator and fan."""

    name = f"cleo_claim_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)

    def scoped(sql: str) -> str:
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            cursor.execute((DB / "ci_supabase_stubs.sql").read_text(encoding="utf-8"))
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(
                scoped((DB / "ci_baseline_schema.sql").read_text(encoding="utf-8"))
            )
            for filename in _order():
                cursor.execute(scoped((DB / filename).read_text(encoding="utf-8")))

            cursor.execute(
                f'insert into "{name}".creators (id, name) '
                "values (gen_random_uuid(), 'Creator') returning id"
            )
            creator_id = cursor.fetchone()[0]
            cursor.execute(
                f'insert into "{name}".fans (id, creator_id, display_name) '
                "values (gen_random_uuid(), %s, 'Fan') returning id",
                (creator_id,),
            )
            fan_id = cursor.fetchone()[0]
        yield connection, name, creator_id, fan_id
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f'drop schema if exists "{name}" cascade')
        connection.close()


def _insert(connection, schema_name, creator_id, fan_id, **overrides):
    row = {
        "action_type": "AUTO_REPLY",
        "status": "PENDING",
        "execute_at": datetime.now(timezone.utc) - timedelta(seconds=5),
        "locked_at": None,
        "dedupe_key": uuid.uuid4().hex,
    }
    row.update(overrides)
    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{schema_name}".scheduled_actions '
            "(id, creator_id, fan_id, action_type, status, execute_at, locked_at, dedupe_key) "
            "values (gen_random_uuid(), %s, %s, %s, %s, %s, %s, %s) returning id",
            (
                creator_id,
                fan_id,
                row["action_type"],
                row["status"],
                row["execute_at"],
                row["locked_at"],
                row["dedupe_key"],
            ),
        )
        return cursor.fetchone()[0]


def _claim(connection, schema_name, limit=20, stale_minutes=10):
    with connection.cursor() as cursor:
        cursor.execute(
            f'select id, status, locked_at from "{schema_name}".claim_due_actions(%s, %s)',
            (limit, stale_minutes),
        )
        return cursor.fetchall()


def test_a_due_action_is_claimed_and_stamped(schema):
    connection, name, creator_id, fan_id = schema
    action_id = _insert(connection, name, creator_id, fan_id)

    claimed = _claim(connection, name)

    assert [row[0] for row in claimed] == [action_id]
    assert claimed[0][1] == "PROCESSING"
    assert claimed[0][2] is not None


def test_an_action_that_is_not_due_yet_is_left_alone(schema):
    connection, name, creator_id, fan_id = schema
    _insert(
        connection,
        name,
        creator_id,
        fan_id,
        execute_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    assert _claim(connection, name) == []


def test_a_freshly_locked_action_is_not_reclaimed(schema):
    connection, name, creator_id, fan_id = schema
    _insert(
        connection,
        name,
        creator_id,
        fan_id,
        status="PROCESSING",
        locked_at=datetime.now(timezone.utc),
    )

    assert _claim(connection, name) == []


def test_a_stale_locked_action_is_reclaimed(schema):
    """The crash-recovery path, still keyed on locked_at."""

    connection, name, creator_id, fan_id = schema
    action_id = _insert(
        connection,
        name,
        creator_id,
        fan_id,
        status="PROCESSING",
        locked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
    )

    claimed = _claim(connection, name, stale_minutes=10)

    assert [row[0] for row in claimed] == [action_id]
    # Re-stamped, so the next worker's stale window starts from now.
    assert claimed[0][2] > datetime.now(timezone.utc) - timedelta(seconds=30)


def test_a_completed_action_is_never_claimed(schema):
    connection, name, creator_id, fan_id = schema
    _insert(connection, name, creator_id, fan_id, status="COMPLETED")
    _insert(connection, name, creator_id, fan_id, status="FAILED")

    assert _claim(connection, name) == []


def test_only_one_of_two_concurrent_workers_owns_an_action(schema):
    """The property the compare-and-swap existed to provide.

    Two real connections claim at the same time against one due row. With
    FOR UPDATE SKIP LOCKED the second does not merely lose a compare-and-swap —
    it never selects the row at all.
    """

    connection, name, creator_id, fan_id = schema
    action_id = _insert(connection, name, creator_id, fan_id)

    first = psycopg.connect(DATABASE_URL)
    second = psycopg.connect(DATABASE_URL)
    try:
        with first.cursor() as cursor_a, second.cursor() as cursor_b:
            cursor_a.execute(f'set search_path to "{name}", public')
            cursor_b.execute(f'set search_path to "{name}", public')

            # A holds an open transaction with the row locked and uncommitted.
            cursor_a.execute(f'select id from "{name}".claim_due_actions(20, 10)')
            claimed_a = [row[0] for row in cursor_a.fetchall()]

            # B runs while A is still open. SKIP LOCKED means it sees nothing.
            cursor_b.execute(f'select id from "{name}".claim_due_actions(20, 10)')
            claimed_b = [row[0] for row in cursor_b.fetchall()]

            first.commit()
            second.commit()

        assert claimed_a == [action_id]
        assert claimed_b == []
    finally:
        first.close()
        second.close()


def test_a_second_claim_after_the_first_commits_still_sees_nothing(schema):
    connection, name, creator_id, fan_id = schema
    _insert(connection, name, creator_id, fan_id)

    assert len(_claim(connection, name)) == 1
    assert _claim(connection, name) == []


def test_the_limit_applies_to_each_set(schema):
    """Preserved from the Python implementation: up to limit due AND limit stale."""

    connection, name, creator_id, fan_id = schema
    for _ in range(3):
        _insert(connection, name, creator_id, fan_id)
    for _ in range(3):
        _insert(
            connection,
            name,
            creator_id,
            fan_id,
            status="PROCESSING",
            locked_at=datetime.now(timezone.utc) - timedelta(minutes=30),
        )

    claimed = _claim(connection, name, limit=2)

    assert len(claimed) == 4


def test_the_oldest_due_actions_are_claimed_first(schema):
    connection, name, creator_id, fan_id = schema
    now = datetime.now(timezone.utc)
    oldest = _insert(
        connection, name, creator_id, fan_id, execute_at=now - timedelta(hours=2)
    )
    _insert(connection, name, creator_id, fan_id, execute_at=now - timedelta(minutes=1))

    claimed = _claim(connection, name, limit=1)

    assert [row[0] for row in claimed] == [oldest]
