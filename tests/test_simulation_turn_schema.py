"""The two guarantees a durable Simulator turn gets from the DATABASE.

The service asks for both correctly — tests/test_simulation_turn_durability.py
proves that — but "asks correctly" is not the same as "cannot happen". Both of
these are read-then-act shapes at the application layer: two requests can pass
the same check before either writes. So the guarantee has to be a constraint,
and a constraint is only real if it actually rejects.

  1. ONE TURN PER SUBMISSION. ``unique (fan_id, idempotency_key)``. A
     double-clicked Send, a POST retried after a dropped connection, and a
     browser that reconnects and resubmits all resolve to one row — and
     therefore to one generation, one creator reply, one set of commercial
     state transitions.

  2. ONE ACTIVE TURN PER FAN. A partial unique index over the active statuses.
     This is what makes a disabled Send button a fact rather than a courtesy:
     a client that ignores it is refused here, not talked out of it.

Run against a real PostgreSQL, because the question is what PostgreSQL does.
Skipped unless TEST_DATABASE_URL points at a disposable one; CI provides it.
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
STUBS = DB / "ci_supabase_stubs.sql"
BASE_SCHEMA = DB / "ci_baseline_schema.sql"
MIGRATION = DB / "simulation_turn_durability_v1.sql"


@pytest.fixture(scope="module")
def schema():
    """A throwaway schema with the base tables and this migration applied."""
    name = f"cleo_sim_turns_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)

    def scoped(sql: str) -> str:
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            cursor.execute(STUBS.read_text(encoding="utf-8"))
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(scoped(BASE_SCHEMA.read_text(encoding="utf-8")))
            cursor.execute(scoped(MIGRATION.read_text(encoding="utf-8")))
        yield connection, name
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


@pytest.fixture
def conversation(schema):
    """One creator and one simulated fan, fresh for each test."""
    connection, name = schema
    creator_id = str(uuid.uuid4())
    fan_id = str(uuid.uuid4())
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        cursor.execute(
            f'insert into "{name}".creators (id, platform_username) values (%s, %s)',
            (creator_id, f"sim-{creator_id[:8]}"),
        )
        cursor.execute(
            f'insert into "{name}".fans (id, creator_id, platform_fan_id, display_name) '
            "values (%s, %s, %s, %s)",
            (fan_id, creator_id, f"test_{fan_id[:8]}", "Test fan"),
        )
    return connection, name, creator_id, fan_id


def _insert(cursor, name, creator_id, fan_id, key, status="accepted"):
    cursor.execute(
        f'insert into "{name}".simulation_turns '
        "(creator_id, fan_id, idempotency_key, status, fan_message) "
        "values (%s, %s, %s, %s, %s) returning id",
        (creator_id, fan_id, key, status, "hii"),
    )
    return cursor.fetchone()[0]


def test_the_migration_is_idempotent(schema):
    """Re-running a migration against a live database must be safe."""
    connection, name = schema
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        cursor.execute(
            MIGRATION.read_text(encoding="utf-8")
            .replace("public.", f'"{name}".')
            .replace("'public'", f"'{name}'")
        )


def test_one_submission_cannot_become_two_turns(conversation):
    """The double-clicked Send, refused by the database rather than by luck."""
    connection, name, creator_id, fan_id = conversation
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        _insert(cursor, name, creator_id, fan_id, "send-1")

        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert(cursor, name, creator_id, fan_id, "send-1")


def test_two_fans_may_use_the_same_idempotency_key(conversation, schema):
    """The key is unique per conversation, not globally.

    Two operators pressing Send at the same moment in different conversations
    is ordinary use, and a globally unique key would make one of them fail.
    """
    connection, name, creator_id, fan_id = conversation
    other_fan = str(uuid.uuid4())
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        cursor.execute(
            f'insert into "{name}".fans (id, creator_id, platform_fan_id, display_name) '
            "values (%s, %s, %s, %s)",
            (other_fan, creator_id, f"test_{other_fan[:8]}", "Second test fan"),
        )
        _insert(cursor, name, creator_id, fan_id, "send-1")
        _insert(cursor, name, creator_id, other_fan, "send-1")


def test_a_fan_cannot_have_two_turns_running_at_once(conversation):
    """Simple serialisation per simulated fan, enforced where it cannot race."""
    connection, name, creator_id, fan_id = conversation
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        _insert(cursor, name, creator_id, fan_id, "send-1", status="processing")

        with pytest.raises(psycopg.errors.UniqueViolation):
            _insert(cursor, name, creator_id, fan_id, "send-2", status="accepted")


def test_a_new_turn_is_allowed_once_the_previous_one_is_terminal(conversation):
    """The index is partial: finished turns accumulate as history."""
    connection, name, creator_id, fan_id = conversation
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        first = _insert(cursor, name, creator_id, fan_id, "send-1", status="processing")
        cursor.execute(
            f'update "{name}".simulation_turns set status = %s where id = %s',
            ("completed", first),
        )
        second = _insert(cursor, name, creator_id, fan_id, "send-2")
        cursor.execute(
            f'update "{name}".simulation_turns set status = %s where id = %s',
            ("failed", second),
        )
        _insert(cursor, name, creator_id, fan_id, "send-3")

        cursor.execute(
            f'select count(*) from "{name}".simulation_turns where fan_id = %s',
            (fan_id,),
        )
        assert cursor.fetchone()[0] == 3


def test_only_the_four_lifecycle_statuses_are_accepted(conversation):
    """A typo in a status is a turn that no poll can ever resolve."""
    connection, name, creator_id, fan_id = conversation
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        with pytest.raises(psycopg.errors.CheckViolation):
            _insert(cursor, name, creator_id, fan_id, "send-1", status="done")


def test_turns_disappear_with_the_fan_they_belong_to(conversation):
    """A deleted test fan must not leave orphan turns behind it."""
    connection, name, creator_id, fan_id = conversation
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        _insert(cursor, name, creator_id, fan_id, "send-1")
        cursor.execute(f'delete from "{name}".fans where id = %s', (fan_id,))
        cursor.execute(
            f'select count(*) from "{name}".simulation_turns where fan_id = %s',
            (fan_id,),
        )
        assert cursor.fetchone()[0] == 0


def test_row_level_security_is_on(schema):
    """Discovered by tenant_isolation_v1 like every other creator-owned table."""
    connection, name = schema
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select relrowsecurity
              from pg_class c
              join pg_namespace n on n.oid = c.relnamespace
             where n.nspname = %s and c.relname = 'simulation_turns'
            """,
            (name,),
        )
        assert cursor.fetchone()[0] is True
