"""The continuity tables must reject a record that would mislead a conversation.

``db/conversation_continuity_v1.sql`` carries constraints that are the whole
reason the tables are worth having, and an in-memory double cannot demonstrate
any of them: a CHECK that a resolved thread says how and when it closed, a
CHECK that only a superseded thread points at a successor, and a uniqueness
scoped to (creator, fan) rather than to the fan alone.

Those are not stylistic. A thread marked ``fulfilled`` with no ``resolved_by``
is an obligation nobody can audit; a thread pointing at a successor while still
``open`` is the two-simultaneously-authoritative-facts state
``docs/autonomy_architecture_review.md`` §4 rules out; and a uniqueness scoped
to the fan alone would let one creator's conversation collide with another's.

Skipped unless TEST_DATABASE_URL points at a disposable PostgreSQL, exactly like
the other schema tests.
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
NOW = datetime.now(timezone.utc)


def _order() -> list[str]:
    lines = (DB / "migration_order.txt").read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture
def schema():
    """A disposable schema with the pipeline applied, plus two creators and two fans."""
    name = f"cleo_continuity_{uuid.uuid4().hex[:12]}"
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

            creators = []
            for label in ("Creator A", "Creator B"):
                cursor.execute(
                    f'insert into "{name}".creators (id, name) '
                    "values (gen_random_uuid(), %s) returning id",
                    (label,),
                )
                creators.append(cursor.fetchone()[0])
            fans = []
            for label in ("Fan One", "Fan Two"):
                cursor.execute(
                    f'insert into "{name}".fans (id, creator_id, display_name) '
                    "values (gen_random_uuid(), %s, %s) returning id",
                    (creators[0], label),
                )
                fans.append(cursor.fetchone()[0])
        yield connection, name, creators, fans
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f'drop schema if exists "{name}" cascade')
        connection.close()


def _insert_thread(connection, name, creator_id, fan_id, **overrides):
    row = {
        "kind": "question",
        "raised_by": "fan",
        "summary": "whether you ever visit Chicago",
        "status": "open",
        "evidence_type": "stated",
        "confidence": 1.0,
        "dedupe_key": uuid.uuid4().hex,
        "resolved_at": None,
        "resolved_by": None,
        "superseded_by": None,
    }
    row.update(overrides)
    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".conversation_open_threads '
            "(creator_id, fan_id, kind, raised_by, summary, status, evidence_type, "
            " confidence, dedupe_key, resolved_at, resolved_by, superseded_by) "
            "values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) returning id",
            (
                creator_id,
                fan_id,
                row["kind"],
                row["raised_by"],
                row["summary"],
                row["status"],
                row["evidence_type"],
                row["confidence"],
                row["dedupe_key"],
                row["resolved_at"],
                row["resolved_by"],
                row["superseded_by"],
            ),
        )
        return cursor.fetchone()[0]


# --- the tables exist and are policed like every other creator-owned table --


def test_both_tables_exist_after_the_pipeline(schema):
    connection, name, _creators, _fans = schema
    with connection.cursor() as cursor:
        cursor.execute(
            "select table_name from information_schema.tables "
            "where table_schema = %s and table_name in "
            "('conversation_open_threads', 'conversation_episodes')",
            (name,),
        )
        found = {row[0] for row in cursor.fetchall()}
    assert found == {"conversation_open_threads", "conversation_episodes"}


def test_tenant_isolation_reached_the_new_tables(schema):
    """They are created before tenant_isolation_v1, which discovers at run time.

    A creator-owned table added after that migration would silently get no
    policy at all, which is the whole reason migration_order.txt exists.
    """
    connection, name, _creators, _fans = schema
    with connection.cursor() as cursor:
        cursor.execute(
            "select tablename, count(*) from pg_policies where schemaname = %s "
            "and tablename in ('conversation_open_threads', 'conversation_episodes') "
            "group by tablename",
            (name,),
        )
        policies = dict(cursor.fetchall())
    assert policies.get("conversation_open_threads", 0) > 0
    assert policies.get("conversation_episodes", 0) > 0


# --- a resolved thread must be auditable ------------------------------------


def test_a_resolved_thread_must_say_how_and_when(schema):
    connection, name, creators, fans = schema
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_thread(
            connection, name, creators[0], fans[0], status="fulfilled"
        )


def test_an_open_thread_may_not_claim_to_have_been_resolved(schema):
    connection, name, creators, fans = schema
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_thread(
            connection,
            name,
            creators[0],
            fans[0],
            status="open",
            resolved_at=NOW,
            resolved_by="operator",
        )


def test_a_properly_resolved_thread_is_accepted(schema):
    connection, name, creators, fans = schema
    assert _insert_thread(
        connection,
        name,
        creators[0],
        fans[0],
        status="fulfilled",
        resolved_at=NOW,
        resolved_by="creator_reply",
    )


# --- supersession is the only way one thread replaces another ---------------


def test_only_a_superseded_thread_may_point_at_a_successor(schema):
    """§4: a correction supersedes; it does not sit beside what it corrects."""
    connection, name, creators, fans = schema
    successor = _insert_thread(connection, name, creators[0], fans[0])

    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_thread(
            connection,
            name,
            creators[0],
            fans[0],
            status="open",
            superseded_by=successor,
        )


def test_a_superseded_thread_keeps_the_link_to_what_replaced_it(schema):
    connection, name, creators, fans = schema
    successor = _insert_thread(connection, name, creators[0], fans[0])
    original = _insert_thread(
        connection,
        name,
        creators[0],
        fans[0],
        status="superseded",
        resolved_at=NOW,
        resolved_by="supersession",
        superseded_by=successor,
    )

    with connection.cursor() as cursor:
        cursor.execute(
            f'select superseded_by from "{name}".conversation_open_threads where id = %s',
            (original,),
        )
        assert cursor.fetchone()[0] == successor


# --- scoping ----------------------------------------------------------------


def test_one_dedupe_key_may_exist_once_per_conversation(schema):
    connection, name, creators, fans = schema
    key = "question:shared"
    _insert_thread(connection, name, creators[0], fans[0], dedupe_key=key)

    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_thread(connection, name, creators[0], fans[0], dedupe_key=key)


def test_the_same_key_in_another_conversation_is_a_different_obligation(schema):
    """Uniqueness scoped to the fan alone would collide across creators."""
    connection, name, creators, fans = schema
    key = "question:shared"
    _insert_thread(connection, name, creators[0], fans[0], dedupe_key=key)

    assert _insert_thread(connection, name, creators[0], fans[1], dedupe_key=key)
    assert _insert_thread(connection, name, creators[1], fans[0], dedupe_key=key)


def test_deleting_a_fan_takes_their_threads_with_them(schema):
    connection, name, creators, fans = schema
    _insert_thread(connection, name, creators[0], fans[0])

    with connection.cursor() as cursor:
        cursor.execute(f'delete from "{name}".fans where id = %s', (fans[0],))
        cursor.execute(
            f'select count(*) from "{name}".conversation_open_threads where fan_id = %s',
            (fans[0],),
        )
        assert cursor.fetchone()[0] == 0


# --- an episode is never a receipt ------------------------------------------


def test_an_episode_has_no_column_that_could_hold_money(schema):
    """The review's constraint, enforced by the schema rather than by habit."""
    connection, name, _creators, _fans = schema
    with connection.cursor() as cursor:
        cursor.execute(
            "select column_name from information_schema.columns "
            "where table_schema = %s and table_name = 'conversation_episodes'",
            (name,),
        )
        columns = {row[0] for row in cursor.fetchall()}

    for money in (
        "amount",
        "amount_cents",
        "price",
        "price_cents",
        "purchased",
        "purchased_at",
        "order_id",
        "platform_order_id",
    ):
        assert money not in columns, (
            f"conversation_episodes.{money} would let a summary be read as a "
            "receipt; ppv_deliveries is the authority on money"
        )


def test_an_episode_range_must_run_forwards(schema):
    connection, name, creators, fans = schema
    with connection.cursor() as cursor, pytest.raises(psycopg.errors.CheckViolation):
        cursor.execute(
            f'insert into "{name}".conversation_episodes '
            "(creator_id, fan_id, summary, first_message_at, last_message_at, dedupe_key) "
            "values (%s,%s,%s,%s,%s,%s)",
            (
                creators[0],
                fans[0],
                "a stretch of conversation",
                NOW,
                NOW - timedelta(hours=1),
                uuid.uuid4().hex,
            ),
        )


def test_an_unknown_thread_kind_is_refused_by_the_database(schema):
    """The vocabulary is fixed on purpose; a new kind is a schema decision."""
    connection, name, creators, fans = schema
    with pytest.raises(psycopg.errors.CheckViolation):
        _insert_thread(connection, name, creators[0], fans[0], kind="vibe")
