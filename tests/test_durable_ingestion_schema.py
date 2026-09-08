"""REL-006 — prove the constraints webhook redelivery depends on actually bite.

The webhook returns 2xx as soon as the message is persisted and a processing
obligation exists, which makes platform redelivery both likely and safe — but
only if "one platform message, one row" and "one event, one obligation" are real
database guarantees rather than hopeful read-then-write code. A text assertion on
a migration cannot show that, so this applies the whole pipeline and tries the
duplicate.

REL-002's db/message_platform_identity_v1.sql provides the first guarantee;
this file's db/durable_ingestion_v1.sql provides the second. Both are exercised
here because REL-006 depends on both holding together.

Skipped unless TEST_DATABASE_URL points at a throwaway database; CI provides one
through the postgres service in .github/workflows/ci.yml.
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


def _migration_order() -> list[str]:
    lines = (DB / "migration_order.txt").read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture
def schema():
    """A disposable schema with the base schema and every migration applied.

    Runs the full pipeline in db/migration_order.txt rather than only this
    sprint's own migration, because REL-006's guarantees are the composition of
    two files (message identity + durable ingestion) and a copy of just one of
    them would not prove what production actually applies.
    """
    name = f"cleo_ingest_{uuid.uuid4().hex[:12]}"
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
            for filename in _migration_order():
                sql = (DB / filename).read_text(encoding="utf-8")
                try:
                    cursor.execute(scoped(sql))
                except Exception as exc:  # pragma: no cover - failure detail
                    raise AssertionError(
                        f"migration {filename} failed against a fresh database: {exc}"
                    ) from exc
        yield connection, name
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _creator_and_fan(connection, schema_name):
    with connection.cursor() as cursor:
        cursor.execute(f'insert into "{schema_name}".creators default values returning id')
        creator = cursor.fetchone()[0]
        cursor.execute(
            f'insert into "{schema_name}".fans (creator_id) values (%s) returning id',
            (creator,),
        )
        return creator, cursor.fetchone()[0]


def _insert_message(connection, schema_name, creator, fan, platform_id):
    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{schema_name}".messages '
            "(fan_id, creator_id, role, content, fansly_message_id) "
            "values (%s, %s, 'fan', 'hi', %s)",
            (fan, creator, platform_id),
        )


def test_pipeline_creates_the_indexes_durable_ingestion_relies_on(schema):
    connection, name = schema
    with connection.cursor() as cursor:
        cursor.execute(
            "select indexname from pg_indexes where schemaname = %s", (name,)
        )
        indexes = {row[0] for row in cursor.fetchall()}

    assert "scheduled_actions_status_execute_at_idx" in indexes


def test_dedupe_key_is_unique_however_the_constraint_was_declared(schema):
    """The migration skips its own index when the column is already unique.

    What matters to REL-006 is that the column IS unique, not which object
    provides it, so assert the property rather than an index name.
    """
    connection, name = schema
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select count(*)
              from pg_index i
              join pg_class t on t.oid = i.indrelid
              join pg_namespace n on n.oid = t.relnamespace
             where n.nspname = %s
               and t.relname = 'scheduled_actions'
               and i.indisunique
               and i.indnatts = 1
               and i.indkey[0] = (
                   select attnum from pg_attribute
                    where attrelid = t.oid and attname = 'dedupe_key'
               )
            """,
            (name,),
        )
        assert cursor.fetchone()[0] >= 1


def test_a_redelivered_platform_message_cannot_create_a_second_row(schema):
    connection, name = schema
    creator, fan = _creator_and_fan(connection, name)

    _insert_message(connection, name, creator, fan, "platform-msg-redelivery")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_message(connection, name, creator, fan, "platform-msg-redelivery")


def test_a_second_creator_may_carry_the_same_platform_message_id(schema):
    """The key is (creator_id, fansly_message_id), not the id alone.

    A global key would silently drop a second creator's legitimately distinct
    message if platform ids turn out to be unique only per account. Neither
    this migration nor message_platform_identity_v1 adds one.
    """
    connection, name = schema
    creator_a, fan_a = _creator_and_fan(connection, name)
    creator_b, fan_b = _creator_and_fan(connection, name)

    _insert_message(connection, name, creator_a, fan_a, "shared-platform-id")
    _insert_message(connection, name, creator_b, fan_b, "shared-platform-id")

    with connection.cursor() as cursor:
        cursor.execute(
            f'select count(*) from "{name}".messages '
            "where fansly_message_id = 'shared-platform-id'",
        )
        assert cursor.fetchone()[0] == 2


def test_locally_originated_messages_are_unconstrained(schema):
    """The index is partial: many rows legitimately have no platform id."""
    connection, name = schema
    creator, fan = _creator_and_fan(connection, name)

    for _ in range(5):
        _insert_message(connection, name, creator, fan, None)

    with connection.cursor() as cursor:
        cursor.execute(
            f'select count(*) from "{name}".messages '
            "where fan_id = %s and fansly_message_id is null",
            (fan,),
        )
        assert cursor.fetchone()[0] == 5


def test_one_event_cannot_create_two_processing_obligations(schema):
    connection, name = schema
    creator, fan = _creator_and_fan(connection, name)
    key = "inbound-message:fan-x:platform-msg-1"

    def insert_action():
        with connection.cursor() as cursor:
            cursor.execute(
                f'insert into "{name}".scheduled_actions '
                "(creator_id, fan_id, action_type, execute_at, dedupe_key) "
                "values (%s, %s, 'PROCESS_INBOUND_MESSAGE', now(), %s)",
                (creator, fan, key),
            )

    insert_action()
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_action()


def test_ignore_duplicates_upsert_leaves_a_completed_obligation_alone(schema):
    """The exact shape schedule_action(replace_existing=False) sends.

    A redelivery of an already-processed event must not resurrect its action.
    """
    connection, name = schema
    creator, fan = _creator_and_fan(connection, name)
    key = "inbound-message:fan-y:platform-msg-2"

    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".scheduled_actions '
            "(creator_id, fan_id, action_type, execute_at, dedupe_key, status) "
            "values (%s, %s, 'PROCESS_INBOUND_MESSAGE', now(), %s, 'COMPLETED')",
            (creator, fan, key),
        )
        cursor.execute(
            f'insert into "{name}".scheduled_actions '
            "(creator_id, fan_id, action_type, execute_at, dedupe_key, status) "
            "values (%s, %s, 'PROCESS_INBOUND_MESSAGE', now(), %s, 'PENDING') "
            "on conflict (dedupe_key) do nothing",
            (creator, fan, key),
        )
        cursor.execute(
            f'select status, count(*) from "{name}".scheduled_actions '
            "where dedupe_key = %s group by status",
            (key,),
        )
        rows = cursor.fetchall()

    assert rows == [("COMPLETED", 1)]
