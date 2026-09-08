"""SEC-003 — fan_conversation_summaries must respect the caller's tenancy.

The dashboard reads this view straight from the browser (app/page.tsx), but
tenant_isolation_v1's discovery loops filter on `table_type = 'BASE TABLE'`, so
no policy was ever created for it. A PostgreSQL view executes with its owner's
privileges unless it is security_invoker — which means an authenticated operator
who deletes the client-side .eq('creator_id', ...) reads every agency's
conversations.

These run against a real PostgreSQL with real roles and real RLS. A mock cannot
establish this: the whole question is what Postgres does when a non-owning role
selects from the view.

NOTE ON SCOPE: this proves the mechanism against the CI baseline view. Whether
the LIVE view is security_invoker cannot be checked from this repository — see
db/MIGRATIONS.md § Open questions.
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
    reason="TEST_DATABASE_URL is not set; RLS tests need a real PostgreSQL",
)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db"

OPERATOR_A = "11111111-1111-1111-1111-111111111111"
OPERATOR_B = "22222222-2222-2222-2222-222222222222"


def _order() -> list[str]:
    return [
        line.strip()
        for line in (DB / "migration_order.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture
def tenancy():
    """Two agencies, RLS enabled, in a disposable schema."""
    name = f"cleo_rls_{uuid.uuid4().hex[:12]}"
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

            cursor.execute(f'grant usage on schema "{name}" to authenticated')
            cursor.execute(
                f'grant select on all tables in schema "{name}" to authenticated'
            )

            creators = {}
            for label, operator in (("A", OPERATOR_A), ("B", OPERATOR_B)):
                cursor.execute(
                    f'insert into "{name}".creators (name) values (%s) returning id',
                    (f"Agency {label}",),
                )
                creator_id = cursor.fetchone()[0]
                creators[label] = creator_id
                cursor.execute(
                    f'insert into "{name}".chatter_creators (chatter_id, creator_id) '
                    "values (%s, %s)",
                    (operator, creator_id),
                )
                cursor.execute(
                    f'insert into "{name}".fans (creator_id, display_name, '
                    "platform_fan_id) values (%s, %s, %s) returning id",
                    (creator_id, f"Fan of {label}", f"p-{label}"),
                )
                fan_id = cursor.fetchone()[0]
                cursor.execute(
                    f'insert into "{name}".messages (fan_id, creator_id, role, content) '
                    "values (%s, %s, %s, %s)",
                    (fan_id, creator_id, "fan", f"secret message for {label}"),
                )
        yield connection, name, creators
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _as_operator(connection, schema, operator, sql, params=None):
    """Run one query as the authenticated role impersonating an operator."""
    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{schema}", public')
            cursor.execute("set local role authenticated")
            cursor.execute(f"set local request.jwt.claim.sub = '{operator}'")
            cursor.execute(sql, params or ())
            return cursor.fetchall()
        finally:
            cursor.execute("rollback")


# --- the view is configured correctly ---------------------------------------


def test_the_view_is_security_invoker(tenancy):
    connection, name, _ = tenancy
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select o.option_value
              from pg_class c
              join pg_namespace n on n.oid = c.relnamespace,
                   pg_options_to_table(c.reloptions) o
             where n.nspname = %s
               and c.relname = 'fan_conversation_summaries'
               and lower(o.option_name) = 'security_invoker'
            """,
            (name,),
        )
        row = cursor.fetchone()

    assert row is not None, "the view is definer-rights and bypasses RLS"
    assert row[0].lower() in {"on", "true", "yes", "1"}


def test_the_migration_repairs_a_definer_rights_view(tenancy):
    """The production case: the setting is absent until the migration runs."""
    connection, name, _ = tenancy
    with connection.cursor() as cursor:
        cursor.execute(
            f'alter view "{name}".fan_conversation_summaries '
            "reset (security_invoker)"
        )
        cursor.execute(f'set search_path to "{name}", public')
        cursor.execute(
            (DB / "summaries_security_invoker_v1.sql")
            .read_text()
            .replace("public.", f'"{name}".')
            .replace("'public'", f"'{name}'")
        )
        cursor.execute(
            """
            select o.option_value
              from pg_class c
              join pg_namespace n on n.oid = c.relnamespace,
                   pg_options_to_table(c.reloptions) o
             where n.nspname = %s
               and c.relname = 'fan_conversation_summaries'
               and lower(o.option_name) = 'security_invoker'
            """,
            (name,),
        )
        assert cursor.fetchone()[0].lower() in {"on", "true"}


# --- each agency sees only its own ------------------------------------------


def test_agency_a_sees_only_its_own_conversations(tenancy):
    connection, name, creators = tenancy

    rows = _as_operator(
        connection,
        name,
        OPERATOR_A,
        "select creator_id, display_name from fan_conversation_summaries",
    )

    assert len(rows) == 1
    assert rows[0][0] == creators["A"]
    assert rows[0][1] == "Fan of A"


def test_agency_b_sees_only_its_own_conversations(tenancy):
    connection, name, creators = tenancy

    rows = _as_operator(
        connection,
        name,
        OPERATOR_B,
        "select creator_id, display_name from fan_conversation_summaries",
    )

    assert len(rows) == 1
    assert rows[0][0] == creators["B"]


def test_a_cannot_read_b_by_removing_the_client_side_filter(tenancy):
    """The exact attack: edit the .eq('creator_id', ...) out in the console."""
    connection, name, creators = tenancy

    unfiltered = _as_operator(
        connection, name, OPERATOR_A, "select creator_id from fan_conversation_summaries"
    )
    assert {row[0] for row in unfiltered} == {creators["A"]}

    # And asking for B's id explicitly returns nothing rather than B's rows.
    targeted = _as_operator(
        connection,
        name,
        OPERATOR_A,
        "select creator_id from fan_conversation_summaries where creator_id = %s",
        (creators["B"],),
    )
    assert targeted == []


def test_a_cannot_read_b_message_content_through_the_view(tenancy):
    """The view exposes last_message; the leak would be content, not just ids."""
    connection, name, _ = tenancy

    rows = _as_operator(
        connection,
        name,
        OPERATOR_A,
        "select last_message from fan_conversation_summaries",
    )

    contents = {row[0] for row in rows}
    assert "secret message for B" not in contents
    assert contents == {"secret message for A"}


def test_an_operator_with_no_assignment_sees_nothing(tenancy):
    connection, name, _ = tenancy

    rows = _as_operator(
        connection,
        name,
        "33333333-3333-3333-3333-333333333333",
        "select creator_id from fan_conversation_summaries",
    )

    assert rows == []


def test_a_definer_rights_view_would_leak(tenancy):
    """Proves the test would catch the regression rather than passing anyway."""
    connection, name, creators = tenancy
    with connection.cursor() as cursor:
        cursor.execute(
            f'alter view "{name}".fan_conversation_summaries '
            "reset (security_invoker)"
        )

    leaked = _as_operator(
        connection, name, OPERATOR_A, "select creator_id from fan_conversation_summaries"
    )

    assert {row[0] for row in leaked} == {creators["A"], creators["B"]}, (
        "expected the definer-rights view to leak; if it does not, these tests "
        "are not actually exercising RLS"
    )
