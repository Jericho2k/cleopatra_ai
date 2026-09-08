"""Apply db/durable_ingestion_v1.sql to a real PostgreSQL and prove it bites.

REL-006 decouples webhook acknowledgement from processing, which makes platform
redelivery both likely and safe — but only if "one platform message, one row"
and "one event, one obligation" are real database guarantees rather than
hopeful read-then-write code. A text assertion on the migration cannot show
that, so this runs the DDL and tries the duplicate.

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
MIGRATION = ROOT / "db" / "durable_ingestion_v1.sql"

# Pre-migration shape of the tables the migration constrains. Both are created
# out of band in Supabase, so the columns Cleopatra actually writes are
# reproduced here.
BASELINE = """
create table public.creators (
    id uuid primary key default gen_random_uuid()
);

create table public.fans (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade
);

create table public.messages (
    id uuid primary key default gen_random_uuid(),
    fan_id uuid not null references public.fans(id) on delete cascade,
    creator_id uuid not null references public.creators(id) on delete cascade,
    role text not null,
    content text not null default '',
    was_ai_suggested boolean not null default false,
    fansly_message_id text null,
    media_context jsonb null,
    sent_at timestamptz not null default now()
);

create table public.scheduled_actions (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    action_type text not null,
    execute_at timestamptz not null,
    payload jsonb not null default '{}'::jsonb,
    dedupe_key text not null,
    status text not null default 'PENDING',
    attempts integer not null default 0,
    locked_at timestamptz null,
    last_error text null
);
"""


@pytest.fixture
def schema():
    name = f"cleo_test_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            cursor.execute(BASELINE.replace("public.", f'"{name}".'))
            cursor.execute(MIGRATION.read_text().replace("public.", f'"{name}".'))
        yield connection, name
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _fan(connection, name):
    with connection.cursor() as cursor:
        cursor.execute(f'insert into "{name}".creators default values returning id')
        creator = cursor.fetchone()[0]
        cursor.execute(
            f'insert into "{name}".fans (creator_id) values (%s) returning id',
            (creator,),
        )
        return creator, cursor.fetchone()[0]


def _insert_message(connection, name, creator, fan, platform_id):
    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".messages '
            "(fan_id, creator_id, role, content, fansly_message_id) "
            "values (%s, %s, 'fan', 'hi', %s)",
            (fan, creator, platform_id),
        )


def test_migration_applies_and_is_idempotent(schema):
    connection, name = schema
    with connection.cursor() as cursor:
        # It will be deployed by re-running it; that must be a no-op.
        cursor.execute(MIGRATION.read_text().replace("public.", f'"{name}".'))
        cursor.execute(
            "select indexname from pg_indexes where schemaname = %s", (name,)
        )
        indexes = {row[0] for row in cursor.fetchall()}

    assert "messages_fansly_message_id_key" in indexes
    assert "scheduled_actions_dedupe_key_key" in indexes
    assert "scheduled_actions_status_execute_at_idx" in indexes


def test_a_redelivered_platform_message_cannot_create_a_second_row(schema):
    connection, name = schema
    creator, fan = _fan(connection, name)

    _insert_message(connection, name, creator, fan, "platform-msg-1")
    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_message(connection, name, creator, fan, "platform-msg-1")


def test_locally_originated_messages_are_unconstrained(schema):
    """The index is partial: many rows legitimately have no platform id."""
    connection, name = schema
    creator, fan = _fan(connection, name)

    for _ in range(5):
        _insert_message(connection, name, creator, fan, None)

    with connection.cursor() as cursor:
        cursor.execute(
            f'select count(*) from "{name}".messages where fansly_message_id is null'
        )
        assert cursor.fetchone()[0] == 5


def test_one_event_cannot_create_two_processing_obligations(schema):
    connection, name = schema
    creator, fan = _fan(connection, name)

    def insert_action():
        with connection.cursor() as cursor:
            cursor.execute(
                f'insert into "{name}".scheduled_actions '
                "(creator_id, fan_id, action_type, execute_at, dedupe_key) "
                "values (%s, %s, 'PROCESS_INBOUND_MESSAGE', now(), %s)",
                (creator, fan, "inbound-message:fan-1:platform-msg-1"),
            )

    insert_action()
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_action()


def test_ignore_duplicates_upsert_leaves_a_completed_obligation_alone(schema):
    """The exact shape schedule_action(replace_existing=False) sends.

    A redelivery of an already-processed event must not resurrect its action.
    """
    connection, name = schema
    creator, fan = _fan(connection, name)
    key = "inbound-message:fan-1:platform-msg-1"

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
