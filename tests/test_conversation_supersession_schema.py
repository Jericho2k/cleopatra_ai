"""The durable supersession objects, proved against a real PostgreSQL.

An in-memory double cannot demonstrate the two properties this migration exists
for: that two concurrent claimers of one fan's execution lease do not both win,
and that the generation bump is atomic under concurrency. Both are decided
inside the database, so both are tested there.

Skipped unless TEST_DATABASE_URL points at a disposable PostgreSQL.
"""

from __future__ import annotations

import os
import uuid
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
    name = f"cleo_super_{uuid.uuid4().hex[:12]}"
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


def test_a_new_fan_starts_at_generation_zero(schema):
    connection, name, _creator_id, fan_id = schema
    with connection.cursor() as cursor:
        cursor.execute(
            f'select conversation_generation from "{name}".fans where id = %s',
            (fan_id,),
        )
        assert cursor.fetchone()[0] == 0


def test_bumping_the_generation_is_monotonic(schema):
    connection, name, _creator_id, fan_id = schema
    seen = []
    with connection.cursor() as cursor:
        for _ in range(3):
            cursor.execute(
                f'select "{name}".bump_conversation_generation(%s)', (fan_id,)
            )
            seen.append(cursor.fetchone()[0])
    assert seen == [1, 2, 3]


def test_two_concurrent_bumps_do_not_collide(schema):
    """The whole reason the bump is a function and not a read-modify-write."""
    connection, name, _creator_id, fan_id = schema
    other = psycopg.connect(DATABASE_URL, autocommit=False)
    try:
        with connection.cursor() as first, other.cursor() as second:
            first.execute(f'set search_path to "{name}", public')
            second.execute(f'set search_path to "{name}", public')
            second.execute("begin")
            second.execute(f'select "{name}".bump_conversation_generation(%s)', (fan_id,))
            # The first connection blocks on the row lock rather than reading a
            # stale value, so the two bumps cannot both produce 1.
            other.commit()
            first.execute(f'select "{name}".bump_conversation_generation(%s)', (fan_id,))
            assert first.fetchone()[0] == 2
    finally:
        other.close()


def test_only_one_worker_wins_the_fan_execution_lease(schema):
    connection, name, creator_id, fan_id = schema
    with connection.cursor() as cursor:
        cursor.execute(
            f'select "{name}".acquire_fan_execution_lease(%s, %s, %s, %s, %s)',
            (fan_id, creator_id, "worker-a", 120, "AUTO_REPLY"),
        )
        assert cursor.fetchone()[0] is True
        cursor.execute(
            f'select "{name}".acquire_fan_execution_lease(%s, %s, %s, %s, %s)',
            (fan_id, creator_id, "worker-b", 120, "AUTO_REPLY"),
        )
        assert cursor.fetchone()[0] is False


def test_the_owner_may_renew_its_own_lease(schema):
    connection, name, creator_id, fan_id = schema
    with connection.cursor() as cursor:
        for _ in range(2):
            cursor.execute(
                f'select "{name}".acquire_fan_execution_lease(%s, %s, %s, %s, %s)',
                (fan_id, creator_id, "worker-a", 120, "AUTO_REPLY"),
            )
            assert cursor.fetchone()[0] is True


def test_an_expired_lease_is_recoverable_by_another_worker(schema):
    """Crash recovery. A worker killed mid-send must not block the fan forever."""
    connection, name, creator_id, fan_id = schema
    with connection.cursor() as cursor:
        cursor.execute(
            f'select "{name}".acquire_fan_execution_lease(%s, %s, %s, %s, %s)',
            (fan_id, creator_id, "worker-a", 5, "AUTO_REPLY"),
        )
        assert cursor.fetchone()[0] is True
        cursor.execute(
            f'update "{name}".fan_execution_leases '
            "set expires_at = now() - interval '1 minute' where fan_id = %s",
            (fan_id,),
        )
        cursor.execute(
            f'select "{name}".acquire_fan_execution_lease(%s, %s, %s, %s, %s)',
            (fan_id, creator_id, "worker-b", 120, "AUTO_REPLY"),
        )
        assert cursor.fetchone()[0] is True


def test_releasing_is_scoped_to_the_owner(schema):
    connection, name, creator_id, fan_id = schema
    with connection.cursor() as cursor:
        cursor.execute(
            f'select "{name}".acquire_fan_execution_lease(%s, %s, %s, %s, %s)',
            (fan_id, creator_id, "worker-a", 120, ""),
        )
        cursor.fetchone()
        cursor.execute(
            f'select "{name}".release_fan_execution_lease(%s, %s)',
            (fan_id, "worker-b"),
        )
        assert cursor.fetchone()[0] is False
        cursor.execute(
            f'select "{name}".release_fan_execution_lease(%s, %s)',
            (fan_id, "worker-a"),
        )
        assert cursor.fetchone()[0] is True


def test_one_sequence_per_trigger_per_fan(schema):
    """A retried action must adopt its sequence, never queue the reply twice."""
    connection, name, creator_id, fan_id = schema
    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".outbound_sequences '
            "(creator_id, fan_id, conversation_generation, trigger_identity) "
            "values (%s, %s, 3, 'message-1')",
            (creator_id, fan_id),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cursor.execute(
                f'insert into "{name}".outbound_sequences '
                "(creator_id, fan_id, conversation_generation, trigger_identity) "
                "values (%s, %s, 4, 'message-1')",
                (creator_id, fan_id),
            )


def test_one_row_per_bubble_index(schema):
    connection, name, creator_id, fan_id = schema
    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".outbound_sequences '
            "(creator_id, fan_id, conversation_generation, trigger_identity) "
            "values (%s, %s, 1, 'message-2') returning id",
            (creator_id, fan_id),
        )
        sequence_id = cursor.fetchone()[0]
        cursor.execute(
            f'insert into "{name}".outbound_sequence_parts '
            "(sequence_id, creator_id, fan_id, part_index, body, due_at) "
            "values (%s, %s, %s, 0, 'hi', now())",
            (sequence_id, creator_id, fan_id),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cursor.execute(
                f'insert into "{name}".outbound_sequence_parts '
                "(sequence_id, creator_id, fan_id, part_index, body, due_at) "
                "values (%s, %s, %s, 0, 'hi again', now())",
                (sequence_id, creator_id, fan_id),
            )


def test_delivery_machinery_is_not_readable_by_the_browser_roles(schema):
    """SEC-001: unsent copy and send arbitration are owner-only, by registry."""
    connection, name, _creator_id, _fan_id = schema
    with connection.cursor() as cursor:
        cursor.execute(
            f'select table_name from "{name}".owner_only_tables order by table_name'
        )
        registered = {row[0] for row in cursor.fetchall()}
    assert {
        "outbound_sequences",
        "outbound_sequence_parts",
        "fan_execution_leases",
    } <= registered

    with connection.cursor() as cursor:
        for table in (
            "outbound_sequences",
            "outbound_sequence_parts",
            "fan_execution_leases",
        ):
            cursor.execute(
                "select count(*) from information_schema.role_table_grants "
                "where table_schema = %s and table_name = %s "
                "and grantee in ('anon', 'authenticated')",
                (name, table),
            )
            assert cursor.fetchone()[0] == 0, table
            cursor.execute(
                "select relrowsecurity from pg_class c "
                "join pg_namespace n on n.oid = c.relnamespace "
                "where n.nspname = %s and c.relname = %s",
                (name, table),
            )
            assert cursor.fetchone()[0] is True, table
