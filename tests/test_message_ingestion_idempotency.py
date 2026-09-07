"""REL-002 — message ingestion must be idempotent under concurrency.

Four paths write fan messages — the webhook, the poller, reconciliation, and
save_message — and every one did a read followed by a non-atomic write. Whether
duplicates were actually possible depended entirely on a unique constraint that
no migration in this repository created.

The database half runs against a real PostgreSQL with real concurrent
transactions: a mock cannot demonstrate what two writers racing on one key
actually do. The client half uses the PostgREST double to check that
save_message_result reports which caller inserted.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg", reason="psycopg is required for schema tests")

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()

pytestmark = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; concurrency tests need a real PostgreSQL",
)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db"


def _order() -> list[str]:
    return [
        line.strip()
        for line in (DB / "migration_order.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture
def schema():
    """Two creators, each with a fan, on the full migrated schema."""
    name = f"cleo_msg_{uuid.uuid4().hex[:12]}"
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

            ids = {}
            for label in ("A", "B"):
                cursor.execute(
                    f'insert into "{name}".creators (name) values (%s) returning id',
                    (label,),
                )
                creator_id = cursor.fetchone()[0]
                cursor.execute(
                    f'insert into "{name}".fans (creator_id, platform_fan_id) '
                    "values (%s, %s) returning id",
                    (creator_id, f"p-{label}"),
                )
                ids[label] = (creator_id, cursor.fetchone()[0])
        yield connection, name, ids
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _upsert(cursor, schema, creator_id, fan_id, platform_id, content):
    """The atomic write save_message_result performs."""
    cursor.execute(
        f'insert into "{schema}".messages '
        "(fan_id, creator_id, role, content, fansly_message_id) "
        "values (%s, %s, 'fan', %s, %s) "
        "on conflict (creator_id, fansly_message_id) "
        "where fansly_message_id is not null do nothing "
        "returning id",
        (fan_id, creator_id, content, platform_id),
    )
    return cursor.fetchone()


def _count(connection, schema, platform_id):
    with connection.cursor() as cursor:
        cursor.execute(
            f'select count(*) from "{schema}".messages where fansly_message_id = %s',
            (platform_id,),
        )
        return cursor.fetchone()[0]


# --- the constraint exists and has the right shape --------------------------


def test_the_unique_index_exists_on_the_composite_key(schema):
    connection, name, _ = schema
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select indexdef from pg_indexes
             where schemaname = %s and indexname = %s
            """,
            (name, "messages_creator_platform_identity_idx"),
        )
        row = cursor.fetchone()

    assert row is not None, "REL-002's unique index was not created"
    definition = row[0].lower()
    assert "unique" in definition
    assert "creator_id" in definition and "fansly_message_id" in definition
    assert "where (fansly_message_id is not null)" in definition


def test_a_duplicate_is_rejected_by_the_database(schema):
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        cursor.execute(
            f'insert into "{name}".messages '
            "(fan_id, creator_id, role, content, fansly_message_id) "
            "values (%s, %s, 'fan', 'hi', 'm-1')",
            (fan_id, creator_id),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            cursor.execute(
                f'insert into "{name}".messages '
                "(fan_id, creator_id, role, content, fansly_message_id) "
                "values (%s, %s, 'fan', 'hi again', 'm-1')",
                (fan_id, creator_id),
            )


# --- concurrency -------------------------------------------------------------


def test_two_simultaneous_writers_produce_exactly_one_row(schema):
    """Both transactions open before either commits — the real race."""
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]

    first = psycopg.connect(DATABASE_URL)
    second = psycopg.connect(DATABASE_URL)
    try:
        c1, c2 = first.cursor(), second.cursor()
        c1.execute(f'set search_path to "{name}", public')
        c2.execute(f'set search_path to "{name}", public')

        # Writer 1 inserts but does not commit.
        inserted_1 = _upsert(c1, name, creator_id, fan_id, "race-1", "from webhook")
        assert inserted_1 is not None

        # Writer 2 arrives on the same key. Under the old check-then-insert both
        # would have read "absent" and both would have inserted.
        first.commit()
        inserted_2 = _upsert(c2, name, creator_id, fan_id, "race-1", "from poller")
        second.commit()

        assert inserted_2 is None, "the second writer created a second row"
    finally:
        first.close()
        second.close()

    assert _count(connection, name, "race-1") == 1


def test_blocked_concurrent_writer_still_yields_one_row(schema):
    """Writer 2 blocks on writer 1's uncommitted row, then finds it committed."""
    import threading

    connection, name, ids = schema
    creator_id, fan_id = ids["A"]

    first = psycopg.connect(DATABASE_URL)
    second = psycopg.connect(DATABASE_URL)
    outcome = {}

    try:
        c1 = first.cursor()
        c1.execute(f'set search_path to "{name}", public')
        _upsert(c1, name, creator_id, fan_id, "race-2", "writer one")

        def racer():
            c2 = second.cursor()
            c2.execute(f'set search_path to "{name}", public')
            # Blocks on the uncommitted duplicate key until writer 1 commits.
            outcome["row"] = _upsert(
                c2, name, creator_id, fan_id, "race-2", "writer two"
            )
            second.commit()

        thread = threading.Thread(target=racer)
        thread.start()
        thread.join(timeout=2)
        assert thread.is_alive(), "writer 2 should be blocked on the duplicate key"

        first.commit()
        thread.join(timeout=10)
        assert not thread.is_alive()
    finally:
        first.close()
        second.close()

    assert outcome["row"] is None, "writer 2 inserted a duplicate"
    assert _count(connection, name, "race-2") == 1


def test_webhook_redelivery_is_idempotent(schema):
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        first = _upsert(cursor, name, creator_id, fan_id, "redeliver", "hello")
        for _ in range(4):
            again = _upsert(cursor, name, creator_id, fan_id, "redeliver", "hello")
            assert again is None

    assert first is not None
    assert _count(connection, name, "redeliver") == 1


# --- legitimate messages are never collapsed --------------------------------


def test_different_messages_are_not_collapsed(schema):
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        for index in range(5):
            assert _upsert(
                cursor, name, creator_id, fan_id, f"distinct-{index}", "hi"
            ) is not None

    with connection.cursor() as cursor:
        cursor.execute(f'select count(*) from "{name}".messages')
        assert cursor.fetchone()[0] == 5


def test_the_same_platform_id_under_two_creators_is_two_rows(schema):
    """The reason the key is composite rather than global.

    If Fansly ids turn out to be unique per account rather than platform-wide, a
    global unique index would silently reject the second creator's legitimately
    distinct message. The composite key still dedupes every real race, because
    racing writers for one message always share a creator_id.
    """
    connection, name, ids = schema
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        for label in ("A", "B"):
            creator_id, fan_id = ids[label]
            assert _upsert(
                cursor, name, creator_id, fan_id, "shared-id", f"message for {label}"
            ) is not None

    assert _count(connection, name, "shared-id") == 2


def test_locally_originated_messages_never_collide(schema):
    """A null fansly_message_id is excluded from the partial index."""
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        for _ in range(3):
            cursor.execute(
                f'insert into "{name}".messages (fan_id, creator_id, role, content) '
                "values (%s, %s, 'creator', 'sent before confirmation')",
                (fan_id, creator_id),
            )
        cursor.execute(
            f'select count(*) from "{name}".messages where fansly_message_id is null'
        )
        assert cursor.fetchone()[0] == 3


# --- client behaviour: who inserted? ----------------------------------------
#
# These do not need PostgreSQL, but they live here because they are the other
# half of the same finding: the constraint above makes duplicates impossible,
# and this makes the *caller* able to tell whether it was the one that inserted.


class _UpsertDB:
    """A Supabase double whose upsert honours the composite unique key."""

    def __init__(self):
        self.rows: list[dict] = []
        self.upserts = 0

    def table(self, _name):
        return _UpsertQuery(self)


class _UpsertQuery:
    def __init__(self, db):
        self._db = db
        self._op = None
        self._payload = None
        self._filters = {}
        self._conflict = None

    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict=None, ignore_duplicates=False, **_k):
        self._op = "upsert"
        self._payload = payload
        self._conflict = on_conflict
        return self

    def eq(self, column, value):
        self._filters[column] = value
        return self

    def limit(self, _n):
        return self

    def execute(self):
        from types import SimpleNamespace

        if self._op == "upsert":
            self._db.upserts += 1
            assert self._conflict == "creator_id,fansly_message_id"
            key = (
                self._payload.get("creator_id"),
                self._payload.get("fansly_message_id"),
            )
            if any(
                (row.get("creator_id"), row.get("fansly_message_id")) == key
                for row in self._db.rows
            ):
                # ON CONFLICT DO NOTHING returns no representation.
                return SimpleNamespace(data=[])
            row = {**self._payload, "id": f"msg-{len(self._db.rows) + 1}"}
            self._db.rows.append(row)
            return SimpleNamespace(data=[row])

        if self._op == "insert":
            row = {**self._payload, "id": f"msg-{len(self._db.rows) + 1}"}
            self._db.rows.append(row)
            return SimpleNamespace(data=[row])

        matches = [
            row
            for row in self._db.rows
            if all(str(row.get(k)) == str(v) for k, v in self._filters.items())
        ]
        return SimpleNamespace(data=matches[:1])


@pytest.fixture
def upsert_db(monkeypatch):
    from db import queries

    db = _UpsertDB()
    monkeypatch.setattr(queries, "get_supabase", lambda: db)
    return db


def _save(**kwargs):
    from db.queries import save_message_result

    return asyncio.run(
        save_message_result("fan-1", "creator-1", "fan", "hello", **kwargs)
    )


def test_first_write_reports_inserted(upsert_db):
    result = _save(fansly_message_id="m-1")

    assert result.inserted is True
    assert result.message_id == "msg-1"


def test_redelivery_reports_not_inserted_but_still_resolves_the_id(upsert_db):
    """The caller needs both: do not re-run the pipeline, but do keep provenance."""
    first = _save(fansly_message_id="m-1")
    second = _save(fansly_message_id="m-1")

    assert second.inserted is False
    assert second.message_id == first.message_id
    assert len(upsert_db.rows) == 1


def test_a_single_round_trip_is_used_for_the_common_case(upsert_db):
    """The old path was SELECT then INSERT; the new one is one upsert."""
    _save(fansly_message_id="m-1")

    assert upsert_db.upserts == 1


def test_messages_without_a_platform_id_are_always_inserted(upsert_db):
    """Locally originated sends must each get a row, never dedupe together."""
    first = _save()
    second = _save()

    assert first.inserted is second.inserted is True
    assert first.message_id != second.message_id
    assert len(upsert_db.rows) == 2


def test_save_message_still_returns_just_the_id(upsert_db):
    """The existing callers' contract is unchanged."""
    from db.queries import save_message

    message_id = asyncio.run(
        save_message("fan-1", "creator-1", "fan", "hello", fansly_message_id="m-9")
    )
    assert message_id == "msg-1"


def test_missing_unique_index_falls_back_instead_of_failing_ingestion(monkeypatch):
    """The code may ship before the migration is applied; do not drop messages."""
    from db import queries

    db = _UpsertDB()
    original = db.table

    def table(name):
        query = original(name)
        real_upsert = query.upsert

        def upsert(*args, **kwargs):
            raise RuntimeError(
                'there is no unique or exclusion constraint matching the ON '
                'CONFLICT specification (42P10)'
            )

        query.upsert = upsert
        _ = real_upsert
        return query

    db.table = table
    monkeypatch.setattr(queries, "get_supabase", lambda: db)

    result = _save(fansly_message_id="m-1")

    assert result.inserted is True
    assert result.message_id == "msg-1"


def test_an_unexpected_database_error_is_not_swallowed(monkeypatch):
    """Only the missing-constraint case falls back; real failures must surface."""
    from db import queries

    db = _UpsertDB()

    def table(_name):
        query = _UpsertQuery(db)

        def upsert(*_a, **_k):
            raise RuntimeError("connection reset by peer")

        query.upsert = upsert
        return query

    db.table = table
    monkeypatch.setattr(queries, "get_supabase", lambda: db)

    with pytest.raises(RuntimeError, match="connection reset"):
        _save(fansly_message_id="m-1")
