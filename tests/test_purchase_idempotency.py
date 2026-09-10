"""REL-003 — one platform order produces one purchase, under concurrency.

The webhook used to decide "have I seen this order" by scanning fans.sales_log
in Python and then writing. Two concurrent deliveries of the same order both
read a log that did not contain it, both concluded "new", and both applied
spend, lifecycle, PPV purchase and follow-up cancellation. A double-counted sale
does not just look wrong on a receipt — it corrupts spend tiers, affordability
and price learning downstream.

Identity now lives in a unique index, so PostgreSQL decides who is first. These
tests run against a real PostgreSQL with real concurrent transactions, because a
mock cannot show what two writers racing on one key actually do — which is
exactly the thing that was broken.
"""

from __future__ import annotations

import os
import threading
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
    name = f"cleo_buy_{uuid.uuid4().hex[:12]}"
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


def _claim(cursor, schema, creator_id, order_id, fan_id=None):
    cursor.execute(
        f'select "{schema}".claim_platform_purchase(%s, %s, %s)',
        (creator_id, order_id, fan_id),
    )
    return cursor.fetchone()[0]


def _count(connection, schema, order_id):
    with connection.cursor() as cursor:
        cursor.execute(
            f'select count(*) from "{schema}".platform_purchase_events '
            "where platform_order_id = %s",
            (order_id,),
        )
        return cursor.fetchone()[0]


# --- the constraint exists and has the right shape --------------------------


def test_the_unique_index_exists_on_the_composite_key(schema):
    connection, name, _ = schema
    with connection.cursor() as cursor:
        cursor.execute(
            "select indexdef from pg_indexes "
            " where schemaname = %s and indexname = %s",
            (name, "platform_purchase_events_identity_key"),
        )
        row = cursor.fetchone()

    assert row is not None, "REL-003's unique index was not created"
    definition = row[0].lower()
    assert "unique" in definition
    assert "creator_id" in definition and "platform_order_id" in definition


def test_the_claim_functions_are_not_reachable_from_the_browser(schema):
    """Ingestion identity is backend-only; an authenticated session must not be
    able to assert that an order was already handled."""
    connection, name, _ = schema
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select p.proname, r.rolname
              from pg_proc p
              join pg_namespace n on n.oid = p.pronamespace
              left join lateral aclexplode(p.proacl) a on true
              left join pg_roles r on r.oid = a.grantee
             where n.nspname = %s
               and p.proname in (
                   'claim_platform_purchase',
                   'complete_platform_purchase',
                   'release_platform_purchase'
               )
            """,
            (name,),
        )
        grants = cursor.fetchall()

    granted = {(proname, rolname) for proname, rolname in grants if rolname}
    for name_ in (
        "claim_platform_purchase",
        "complete_platform_purchase",
        "release_platform_purchase",
    ):
        assert (name_, "service_role") in granted, f"{name_} not granted to service_role"
        assert (name_, "authenticated") not in granted
        assert (name_, "anon") not in granted


# --- the behaviour ----------------------------------------------------------


def test_a_single_delivery_claims_the_order(schema):
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "claimed"
    assert _count(connection, name, "order-1") == 1


def test_the_same_order_redelivered_later_is_idempotent(schema):
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "claimed"
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "duplicate"
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "duplicate"
    assert _count(connection, name, "order-1") == 1


def test_two_simultaneous_deliveries_produce_one_effective_purchase(schema):
    """Both transactions open before either commits — the real race.

    Under the old sales_log scan both would have read "absent" and both would
    have applied spend, lifecycle and follow-up cancellation.
    """
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]

    first = psycopg.connect(DATABASE_URL)
    second = psycopg.connect(DATABASE_URL)
    try:
        c1, c2 = first.cursor(), second.cursor()
        c1.execute(f'set search_path to "{name}", public')
        c2.execute(f'set search_path to "{name}", public')

        outcome_1 = _claim(c1, name, creator_id, "race-1", fan_id)
        assert outcome_1 == "claimed"

        first.commit()
        outcome_2 = _claim(c2, name, creator_id, "race-1", fan_id)
        second.commit()

        assert outcome_2 == "duplicate", "the second delivery also claimed the order"
    finally:
        first.close()
        second.close()

    assert _count(connection, name, "race-1") == 1


def test_a_blocked_concurrent_delivery_still_yields_one_purchase(schema):
    """Delivery 2 blocks on delivery 1's uncommitted row, then finds it
    committed. This is the interleaving a plain 'insert, catch the error' would
    get wrong if it retried."""
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]

    first = psycopg.connect(DATABASE_URL)
    second = psycopg.connect(DATABASE_URL)
    outcome: dict[str, str] = {}

    try:
        c1 = first.cursor()
        c1.execute(f'set search_path to "{name}", public')
        assert _claim(c1, name, creator_id, "race-2", fan_id) == "claimed"

        def racer():
            c2 = second.cursor()
            c2.execute(f'set search_path to "{name}", public')
            outcome["result"] = _claim(c2, name, creator_id, "race-2", fan_id)
            second.commit()

        thread = threading.Thread(target=racer)
        thread.start()
        thread.join(timeout=2)
        assert thread.is_alive(), "delivery 2 should block on the duplicate key"

        first.commit()
        thread.join(timeout=10)
        assert not thread.is_alive()
    finally:
        first.close()
        second.close()

    assert outcome["result"] == "duplicate"
    assert _count(connection, name, "race-2") == 1


def test_identity_is_scoped_to_the_creator(schema):
    """A different creator's identically-named order is a DIFFERENT order.

    If Fansly ids turn out to be unique only per account, a global key would
    silently reject this second creator's real sale.
    """
    connection, name, ids = schema
    creator_a, fan_a = ids["A"]
    creator_b, fan_b = ids["B"]

    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_a, "shared-id", fan_a) == "claimed"
        assert _claim(cursor, name, creator_b, "shared-id", fan_b) == "claimed"

    assert _count(connection, name, "shared-id") == 2


def test_different_real_orders_both_apply(schema):
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "claimed"
        assert _claim(cursor, name, creator_id, "order-2", fan_id) == "claimed"
    assert _count(connection, name, "order-1") == 1
    assert _count(connection, name, "order-2") == 1


def test_an_event_without_an_order_id_reports_no_identity(schema):
    """Never invent a key for an event the platform did not identify."""
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id, "", fan_id) == "no_identity"
        assert _claim(cursor, name, creator_id, "   ", fan_id) == "no_identity"
        assert _claim(cursor, name, creator_id, None, fan_id) == "no_identity"
        cursor.execute(
            f'select count(*) from "{name}".platform_purchase_events'
        )
        assert cursor.fetchone()[0] == 0


# --- releasing a claim that did not become a purchase -----------------------


def test_a_released_claim_can_be_retried(schema):
    """The failure direction that matters: a claim taken for work that did not
    happen must not turn a legitimate redelivery into a silent no-op."""
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "claimed"
        cursor.execute(
            f'select "{name}".release_platform_purchase(%s, %s)',
            (creator_id, "order-1"),
        )
        assert _count(connection, name, "order-1") == 0
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "claimed"


def test_a_processed_purchase_cannot_be_released(schema):
    """A late release from a crashed handler must not un-record a real sale."""
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "claimed"
        cursor.execute(
            f'select "{name}".complete_platform_purchase(%s, %s)',
            (creator_id, "order-1"),
        )
        cursor.execute(
            f'select "{name}".release_platform_purchase(%s, %s)',
            (creator_id, "order-1"),
        )

    assert _count(connection, name, "order-1") == 1
    with connection.cursor() as cursor:
        cursor.execute(
            f'select status, processed_at from "{name}".platform_purchase_events '
            "where platform_order_id = %s",
            ("order-1",),
        )
        status, processed_at = cursor.fetchone()
    assert status == "processed"
    assert processed_at is not None


def test_deleting_a_fan_does_not_erase_the_order_identity(schema):
    """Losing the identity would make a redelivery look like a new sale."""
    connection, name, ids = schema
    creator_id, fan_id = ids["A"]
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id, "order-1", fan_id) == "claimed"
        cursor.execute(f'delete from "{name}".fans where id = %s', (fan_id,))
        assert _claim(cursor, name, creator_id, "order-1", None) == "duplicate"

    assert _count(connection, name, "order-1") == 1
