"""Apply db/fansly_lists_v1.sql to a real PostgreSQL and exercise its invariants.

Text assertions on a migration cannot prove that a unique index actually rejects
a duplicate mirror, so this runs the real DDL. It is skipped unless
TEST_DATABASE_URL points at a throwaway database; CI provides one through the
postgres service in .github/workflows/backend-ci.yml.

    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \\
        pytest tests/test_fansly_lists_schema.py
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
MIGRATION = ROOT / "db" / "fansly_lists_v1.sql"

# The pre-migration shape of the two tables. They are created out of band in
# Supabase, so the columns Cleopatra actually reads are reproduced here.
BASELINE = """
create table public.creators (
    id uuid primary key default gen_random_uuid(),
    apifansly_account_id text null
);

create table public.fans (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    platform_fan_id text null
);

create table public.fan_lists (
    id uuid primary key default gen_random_uuid(),
    creator_id uuid not null references public.creators(id) on delete cascade,
    name text not null,
    color text null,
    exclude_from_auto boolean not null default false
);

create table public.fan_list_members (
    list_id uuid not null references public.fan_lists(id) on delete cascade,
    fan_id uuid not null references public.fans(id) on delete cascade,
    primary key (list_id, fan_id)
);
"""


@pytest.fixture
def schema():
    """A disposable schema with the baseline tables and the migration applied.

    The migration is written against ``public.*``. Rewriting that prefix onto a
    throwaway schema keeps concurrent runs isolated without editing the file
    that actually ships.
    """
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


def _creator(connection, schema_name):
    with connection.cursor() as cursor:
        cursor.execute(f'insert into "{schema_name}".creators default values returning id')
        return cursor.fetchone()[0]


def test_migration_applies_and_is_idempotent(schema):
    connection, name = schema

    with connection.cursor() as cursor:
        # Re-running must be a no-op, which is how it will be deployed.
        cursor.execute(MIGRATION.read_text().replace("public.", f'"{name}".'))
        cursor.execute(
            """
            select column_name
              from information_schema.columns
             where table_schema = %s and table_name = 'fan_lists'
            """,
            (name,),
        )
        columns = {row[0] for row in cursor.fetchall()}

    assert {
        "source",
        "external_list_id",
        "external_synced_at",
        "external_archived_at",
        "external_item_count",
    } <= columns


def test_existing_rows_are_backfilled_as_local(schema):
    connection, name = schema
    creator = _creator(connection, name)

    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".fan_lists (creator_id, name) values (%s, %s) '
            "returning source, external_list_id",
            (creator, "VIP"),
        )
        source, external_id = cursor.fetchone()

    assert source == "local"
    assert external_id is None


def test_duplicate_mirror_for_one_remote_list_is_rejected(schema):
    """The invariant repeated syncs depend on."""
    connection, name = schema
    creator = _creator(connection, name)

    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".fan_lists '
            "(creator_id, name, source, external_list_id) values (%s, %s, %s, %s)",
            (creator, "VIP", "fansly", "1001"),
        )

    with pytest.raises(psycopg.errors.UniqueViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f'insert into "{name}".fan_lists '
                "(creator_id, name, source, external_list_id) values (%s, %s, %s, %s)",
                (creator, "VIP Buyers", "fansly", "1001"),
            )


def test_two_creators_may_mirror_the_same_remote_id(schema):
    connection, name = schema
    first = _creator(connection, name)
    second = _creator(connection, name)

    with connection.cursor() as cursor:
        for creator in (first, second):
            cursor.execute(
                f'insert into "{name}".fan_lists '
                "(creator_id, name, source, external_list_id) values (%s, %s, %s, %s)",
                (creator, "VIP", "fansly", "1001"),
            )
        cursor.execute(
            f'select count(*) from "{name}".fan_lists where external_list_id = %s',
            ("1001",),
        )
        assert cursor.fetchone()[0] == 2


def test_many_local_lists_may_share_a_name_and_a_null_external_id(schema):
    connection, name = schema
    creator = _creator(connection, name)

    with connection.cursor() as cursor:
        for _ in range(3):
            cursor.execute(
                f'insert into "{name}".fan_lists (creator_id, name) values (%s, %s)',
                (creator, "VIP"),
            )
        cursor.execute(
            f'select count(*) from "{name}".fan_lists where creator_id = %s', (creator,)
        )
        assert cursor.fetchone()[0] == 3


def test_a_fansly_list_without_a_remote_id_is_rejected(schema):
    connection, name = schema
    creator = _creator(connection, name)

    with pytest.raises(psycopg.errors.CheckViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f'insert into "{name}".fan_lists (creator_id, name, source) '
                "values (%s, %s, %s)",
                (creator, "VIP", "fansly"),
            )


def test_a_local_list_may_not_carry_a_remote_id(schema):
    connection, name = schema
    creator = _creator(connection, name)

    with pytest.raises(psycopg.errors.CheckViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f'insert into "{name}".fan_lists '
                "(creator_id, name, source, external_list_id) values (%s, %s, %s, %s)",
                (creator, "VIP", "local", "1001"),
            )


def test_an_unknown_source_is_rejected(schema):
    connection, name = schema
    creator = _creator(connection, name)

    with pytest.raises(psycopg.errors.CheckViolation):
        with connection.cursor() as cursor:
            cursor.execute(
                f'insert into "{name}".fan_lists (creator_id, name, source) '
                "values (%s, %s, %s)",
                (creator, "VIP", "onlyfans"),
            )


def test_membership_rows_carry_provenance(schema):
    connection, name = schema
    creator = _creator(connection, name)

    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".fan_lists '
            "(creator_id, name, source, external_list_id) values (%s, %s, %s, %s) "
            "returning id",
            (creator, "VIP", "fansly", "1001"),
        )
        list_id = cursor.fetchone()[0]
        cursor.execute(
            f'insert into "{name}".fans (creator_id, platform_fan_id) '
            "values (%s, %s) returning id",
            (creator, "p-a"),
        )
        fan_id = cursor.fetchone()[0]

        cursor.execute(
            f'insert into "{name}".fan_list_members (list_id, fan_id, source) '
            "values (%s, %s, %s)",
            (list_id, fan_id, "fansly"),
        )
        # A hand-added membership defaults to local, which is what protects it
        # from remote reconciliation.
        cursor.execute(
            f'select source from "{name}".fan_list_members '
            "where list_id = %s and fan_id = %s",
            (list_id, fan_id),
        )
        assert cursor.fetchone()[0] == "fansly"

        cursor.execute(
            f'select count(*) from "{name}".fan_list_members '
            "where list_id = %s and source = 'fansly'",
            (list_id,),
        )
        assert cursor.fetchone()[0] == 1


def test_creator_sync_state_columns_exist(schema):
    connection, name = schema

    with connection.cursor() as cursor:
        cursor.execute(
            """
            select column_name
              from information_schema.columns
             where table_schema = %s and table_name = 'creators'
            """,
            (name,),
        )
        columns = {row[0] for row in cursor.fetchall()}

    assert {
        "last_fansly_lists_sync_at",
        "fansly_lists_sync_error",
        "fansly_lists_sync_failed_at",
    } <= columns
