"""SEC-001 — tenancy is not authorization.

tenant_isolation_v1 closed the tenant boundary but gave every creator-owned
table `FOR ALL TO authenticated`. A legitimate operator for agency A, holding
nothing more exotic than the browser Supabase client and their own JWT, could
therefore INSERT, UPDATE and DELETE rows on tables the product never writes from
a browser — durable scheduled work, the PPV ledger, the approval queue,
commercial state — bypassing every rule the backend enforces.

These run against a real PostgreSQL with real roles, real RLS and real GRANTs,
because the question is precisely what PostgreSQL does when the `authenticated`
role attempts a write. A mock would be asserting our own beliefs back at us.

Four things are proved:

  1. the tenant boundary still holds (A cannot read or write B);
  2. sensitive A-owned tables are no longer mutable from a browser at all;
  3. the writes the dashboard genuinely performs still work — a security change
     that breaks the product is not a fix;
  4. service_role, which every background job uses, is untouched.

The fixture deliberately does NOT hand out blanket grants. Privileges come only
from the migrations, which is the thing under test.
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


@pytest.fixture(scope="module")
def tenancy():
    """Two agencies with a fan, a message and a list each."""
    name = f"cleo_priv_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)

    def scoped(sql: str) -> str:
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            cursor.execute((DB / "ci_supabase_stubs.sql").read_text())
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(scoped((DB / "ci_baseline_schema.sql").read_text()))

            # Model Supabase, not a clean-room PostgreSQL.
            #
            # A real Supabase project ships broad grants on the public schema to
            # anon and authenticated, and default privileges so tables created
            # later get them too. Without this the fixture would deny every
            # browser write for the mundane reason that no grant was ever made,
            # and the denial tests below would pass whether or not SEC-001 had
            # been fixed. They have to start permissive for the narrowing to be
            # what closes them.
            cursor.execute(f'grant usage on schema "{name}" to anon, authenticated')
            cursor.execute(f'grant usage on schema "{name}" to service_role')
            cursor.execute(
                f'grant all on all tables in schema "{name}" to anon, authenticated'
            )
            cursor.execute(
                f'alter default privileges in schema "{name}" '
                "grant all on tables to anon, authenticated"
            )

            for filename in _order():
                cursor.execute(scoped((DB / filename).read_text()))

            data: dict[str, dict] = {}
            for label, operator in (("A", OPERATOR_A), ("B", OPERATOR_B)):
                cursor.execute(
                    f'insert into "{name}".creators (name) values (%s) returning id',
                    (f"Agency {label}",),
                )
                creator_id = cursor.fetchone()[0]
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
                    "values (%s, %s, %s, %s) returning id",
                    (fan_id, creator_id, "fan", f"secret for {label}"),
                )
                message_id = cursor.fetchone()[0]
                cursor.execute(
                    f'insert into "{name}".fan_lists (creator_id, name, source) '
                    "values (%s, %s, 'local') returning id",
                    (creator_id, f"Local list {label}"),
                )
                local_list_id = cursor.fetchone()[0]
                cursor.execute(
                    f'insert into "{name}".fan_lists '
                    "(creator_id, name, source, external_list_id) "
                    "values (%s, %s, 'fansly', %s) returning id",
                    (creator_id, f"Mirror {label}", f"ext-{label}"),
                )
                mirror_list_id = cursor.fetchone()[0]
                data[label] = {
                    "operator": operator,
                    "creator_id": creator_id,
                    "fan_id": fan_id,
                    "message_id": message_id,
                    "local_list_id": local_list_id,
                    "mirror_list_id": mirror_list_id,
                }
        yield connection, name, data
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _as_operator(connection, schema, operator, sql, params=None, *, fetch=False):
    """Run one statement as `authenticated`, impersonating an operator.

    Always rolled back: these tests must not leave writes behind for each other.
    Returns rows when asked, otherwise the affected row count.
    """
    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{schema}", public')
            cursor.execute("set local role authenticated")
            cursor.execute(f"set local request.jwt.claim.sub = '{operator}'")
            cursor.execute(sql, params or ())
            return cursor.fetchall() if fetch else cursor.rowcount
        finally:
            cursor.execute("rollback")


def _denied(connection, schema, operator, sql, params=None) -> bool:
    """Whether a write is refused, by ANY of the mechanisms that can refuse it.

    Three outcomes all mean "the browser cannot do this", and which one applies
    depends on whether the block came from a missing GRANT, a missing policy, or
    a policy that matched no rows. The test is about the outcome, so it accepts
    all three rather than pinning the mechanism.
    """
    try:
        affected = _as_operator(connection, schema, operator, sql, params)
    except psycopg.errors.InsufficientPrivilege:
        return True
    except psycopg.Error as exc:
        # "new row violates row-level security policy"
        return "row-level security" in str(exc).lower()
    return affected == 0


# ===========================================================================
# 1. The tenant boundary still holds
# ===========================================================================


def test_operator_a_cannot_read_operator_b(tenancy):
    connection, name, data = tenancy

    rows = _as_operator(
        connection, name, OPERATOR_A,
        f'select id from "{name}".messages where creator_id = %s',
        (data["B"]["creator_id"],),
        fetch=True,
    )

    assert rows == []


def test_operator_a_sees_their_own_rows(tenancy):
    connection, name, data = tenancy

    rows = _as_operator(
        connection, name, OPERATOR_A,
        f'select id from "{name}".messages where creator_id = %s',
        (data["A"]["creator_id"],),
        fetch=True,
    )

    assert len(rows) == 1


def test_operator_a_cannot_write_operator_b(tenancy):
    connection, name, data = tenancy

    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fans set hobbies = %s where id = %s',
        ("hijacked", data["B"]["fan_id"]),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'delete from "{name}".fan_lists where id = %s',
        (data["B"]["local_list_id"],),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'insert into "{name}".blocked_words (creator_id, word) values (%s, %s)',
        (data["B"]["creator_id"], "planted"),
    )


# ===========================================================================
# 2. Sensitive A-owned state is no longer mutable from a browser
#
# These are all rows operator A legitimately OWNS. Tenancy permits them; the
# product does not.
# ===========================================================================


@pytest.mark.parametrize("table", [
    "scheduled_actions",
    "ppv_deliveries",
    "ppv_approval_requests",
    "platform_purchase_events",
])
def test_durable_and_commercial_tables_reject_browser_inserts(tenancy, table):
    connection, name, data = tenancy
    if not _table_exists(connection, name, table):
        pytest.skip(f"{table} is not in the CI baseline")

    assert _denied(
        connection, name, OPERATOR_A,
        f'insert into "{name}".{table} (creator_id) values (%s)',
        (data["A"]["creator_id"],),
    )


@pytest.mark.parametrize("table", [
    "scheduled_actions",
    "ppv_deliveries",
    "ppv_approval_requests",
    "platform_purchase_events",
])
def test_durable_and_commercial_tables_reject_browser_deletes(tenancy, table):
    connection, name, data = tenancy
    if not _table_exists(connection, name, table):
        pytest.skip(f"{table} is not in the CI baseline")

    assert _denied(
        connection, name, OPERATOR_A,
        f'delete from "{name}".{table} where creator_id = %s',
        (data["A"]["creator_id"],),
    )


def test_messages_are_not_writable_from_the_browser(tenancy):
    """Conversation history is evidence. Sends go through the backend, which
    records identity and applies the delivery rules."""
    connection, name, data = tenancy

    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".messages set content = %s where id = %s',
        ("rewritten history", data["A"]["message_id"]),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'delete from "{name}".messages where id = %s',
        (data["A"]["message_id"],),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'insert into "{name}".messages (fan_id, creator_id, role, content) '
        "values (%s, %s, 'creator', 'forged')",
        (data["A"]["fan_id"], data["A"]["creator_id"]),
    )


def test_a_fans_commercial_state_is_not_browser_editable(tenancy):
    """The column half of SEC-001. The operator may edit this fan's hobbies;
    that must not also mean they may edit what the fan has spent."""
    connection, name, data = tenancy

    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fans set total_spent = 999999 where id = %s',
        (data["A"]["fan_id"],),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fans set spend_tier = %s where id = %s',
        ("whale", data["A"]["fan_id"]),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fans set needs_human_review = false where id = %s',
        (data["A"]["fan_id"],),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fans set sales_log = %s where id = %s',
        ("[]", data["A"]["fan_id"]),
    )


def test_a_creators_platform_binding_is_not_browser_editable(tenancy):
    """Repointing a creator at a different API Fansly account from the browser
    would redirect every message that creator sends."""
    connection, name, data = tenancy

    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".creators set apifansly_account_id = %s where id = %s',
        ("attacker-account", data["A"]["creator_id"]),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".creators set fansly_account_id = %s where id = %s',
        ("attacker-account", data["A"]["creator_id"]),
    )


def test_creators_cannot_be_created_or_deleted_from_the_browser(tenancy):
    connection, name, data = tenancy

    assert _denied(
        connection, name, OPERATOR_A,
        f'insert into "{name}".creators (name) values (%s)', ("Smuggled",),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'delete from "{name}".creators where id = %s', (data["A"]["creator_id"],),
    )


def test_a_fansly_mirror_is_not_operator_editable(tenancy):
    """Reconciliation owns mirrored lists. The UI already treats them as
    read-only (isEditableList); this makes that an actual guarantee."""
    connection, name, data = tenancy

    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fan_lists set name = %s where id = %s',
        ("renamed mirror", data["A"]["mirror_list_id"]),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'delete from "{name}".fan_lists where id = %s',
        (data["A"]["mirror_list_id"],),
    )


def test_a_local_list_cannot_be_disguised_as_a_mirror(tenancy):
    """Writing `source` directly would forge a Fansly-owned list, which then
    stops being operator-editable and starts being reconciled."""
    connection, name, data = tenancy

    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fan_lists set source = %s where id = %s',
        ("fansly", data["A"]["local_list_id"]),
    )
    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".fan_lists set external_list_id = %s where id = %s',
        ("ext-hijack", data["A"]["local_list_id"]),
    )


def test_vault_media_platform_identity_is_not_browser_editable(tenancy):
    """An operator may correct what the classifier decided, not rewrite the
    vault's record of what exists on the platform."""
    connection, name, data = tenancy
    if not _table_exists(connection, name, "creator_vault_media"):
        pytest.skip("creator_vault_media is not in the CI baseline")

    assert _denied(
        connection, name, OPERATOR_A,
        f'update "{name}".creator_vault_media set url = %s where creator_id = %s',
        ("https://attacker/", data["A"]["creator_id"]),
    )


# ===========================================================================
# 3. The dashboard still works
#
# Every one of these corresponds to a real .from(...).insert/update/delete in
# the dashboard. A security change that breaks the product is not a fix.
# ===========================================================================


def _allowed(connection, schema, operator, sql, params=None) -> bool:
    try:
        return _as_operator(connection, schema, operator, sql, params) == 1
    except psycopg.Error as exc:  # pragma: no cover - failure detail
        raise AssertionError(f"a legitimate dashboard write was refused: {exc}") from exc


def test_fan_details_are_editable(tenancy):
    """components/FanPanel.tsx — FAN DETAILS."""
    connection, name, data = tenancy

    for column in ("age", "payday", "hobbies", "relationship_status"):
        assert _allowed(
            connection, name, OPERATOR_A,
            f'update "{name}".fans set {column} = %s where id = %s',
            ("edited", data["A"]["fan_id"]),
        ), column


def test_creator_settings_are_editable(tenancy):
    """app/settings/page.tsx — persona, sleep hours, spend caps."""
    connection, name, data = tenancy

    assert _allowed(
        connection, name, OPERATOR_A,
        f'update "{name}".creators set sleep_hours_start = 1, sleep_hours_end = 7 '
        "where id = %s",
        (data["A"]["creator_id"],),
    )
    assert _allowed(
        connection, name, OPERATOR_A,
        f'update "{name}".creators set caps_enabled = true where id = %s',
        (data["A"]["creator_id"],),
    )


def test_blocked_words_can_be_added_and_removed(tenancy):
    """app/settings/page.tsx — blocked words."""
    connection, name, data = tenancy

    assert _allowed(
        connection, name, OPERATOR_A,
        f'insert into "{name}".blocked_words (creator_id, word) values (%s, %s)',
        (data["A"]["creator_id"], "nope"),
    )


def test_local_lists_can_be_created_renamed_and_deleted(tenancy):
    """app/page.tsx — Sidebar list management."""
    connection, name, data = tenancy

    assert _allowed(
        connection, name, OPERATOR_A,
        f'insert into "{name}".fan_lists (creator_id, name) values (%s, %s)',
        (data["A"]["creator_id"], "New list"),
    )
    assert _allowed(
        connection, name, OPERATOR_A,
        f'update "{name}".fan_lists set name = %s where id = %s',
        ("Renamed", data["A"]["local_list_id"]),
    )
    assert _allowed(
        connection, name, OPERATOR_A,
        f'delete from "{name}".fan_lists where id = %s',
        (data["A"]["local_list_id"],),
    )


def test_list_membership_can_be_managed(tenancy):
    """app/page.tsx — adding and removing a fan from a list."""
    connection, name, data = tenancy

    assert _allowed(
        connection, name, OPERATOR_A,
        f'insert into "{name}".fan_list_members (list_id, fan_id) values (%s, %s)',
        (data["A"]["local_list_id"], data["A"]["fan_id"]),
    )


def test_vault_sets_can_be_curated(tenancy):
    """app/scripts/page.tsx — manual set curation."""
    connection, name, data = tenancy
    if not _table_exists(connection, name, "vault_sets"):
        pytest.skip("vault_sets is not in the CI baseline")

    assert _allowed(
        connection, name, OPERATOR_A,
        f'insert into "{name}".vault_sets (creator_id, title) values (%s, %s)',
        (data["A"]["creator_id"], "New set"),
    )


def test_vault_media_classification_can_be_corrected(tenancy):
    """app/vault/page.tsx — the preview panel's edit form."""
    connection, name, data = tenancy
    if not _table_exists(connection, name, "creator_vault_media"):
        pytest.skip("creator_vault_media is not in the CI baseline")

    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        cursor.execute(
            f'insert into "{name}".creator_vault_media '
            "(creator_id, media_id, fansly_media_id) values (%s, %s, %s) "
            "returning id",
            (data["A"]["creator_id"], "m-1", "m-1"),
        )
        media_id = cursor.fetchone()[0]

    try:
        assert _allowed(
            connection, name, OPERATOR_A,
            f'update "{name}".creator_vault_media set content_category = %s '
            "where id = %s",
            ("solo", media_id),
        )
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(
                f'delete from "{name}".creator_vault_media where id = %s', (media_id,)
            )


def test_realtime_select_visibility_is_preserved(tenancy):
    """Realtime subscriptions are SELECTs. Narrowing reads would silently break
    live conversation updates."""
    connection, name, data = tenancy

    for table in ("fans", "messages", "suggestions", "fan_lists"):
        rows = _as_operator(
            connection, name, OPERATOR_A,
            f'select count(*) from "{name}".{table}',
            fetch=True,
        )
        assert rows, f"{table} is not readable from the browser"


# ===========================================================================
# 4. service_role is untouched
# ===========================================================================


def test_service_role_still_writes_everything(tenancy):
    """Every background job runs as service_role. If this fails, the product
    is down."""
    connection, name, data = tenancy

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            cursor.execute("set local role service_role")
            cursor.execute(
                f'update "{name}".fans set total_spent = 42 where id = %s',
                (data["A"]["fan_id"],),
            )
            assert cursor.rowcount == 1
            cursor.execute(
                f'insert into "{name}".messages (fan_id, creator_id, role, content) '
                "values (%s, %s, 'creator', 'from the backend')",
                (data["A"]["fan_id"], data["A"]["creator_id"]),
            )
            assert cursor.rowcount == 1
            cursor.execute(
                f'update "{name}".creators set apifansly_account_id = %s where id = %s',
                ("legitimately-reconnected", data["A"]["creator_id"]),
            )
            assert cursor.rowcount == 1
        finally:
            cursor.execute("rollback")


def test_service_role_crosses_creators(tenancy):
    """The schedulers are deliberately cross-creator."""
    connection, name, _ = tenancy

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            cursor.execute("set local role service_role")
            cursor.execute(f'select count(*) from "{name}".creators')
            assert cursor.fetchone()[0] >= 2
        finally:
            cursor.execute("rollback")


# ===========================================================================
# 5. The policies themselves have the expected shape
# ===========================================================================


def test_no_creator_owned_table_still_has_a_for_all_policy(tenancy):
    """The finding, asserted structurally. A new table added later that only
    gets tenant_isolation_v1's FOR ALL policy fails here."""
    connection, name, _ = tenancy

    with connection.cursor() as cursor:
        cursor.execute(
            "select tablename, policyname from pg_policies "
            " where schemaname = %s and cmd = 'ALL' and %s = any(roles)",
            (name, "authenticated"),
        )
        offenders = cursor.fetchall()

    assert offenders == [], (
        "these tables still grant every operation to the browser: "
        f"{offenders}"
    )


def test_anon_has_no_access_to_creator_data(tenancy):
    """Nothing in the product reads before login."""
    connection, name, _ = tenancy

    with connection.cursor() as cursor:
        cursor.execute(
            """
            select table_name, privilege_type
              from information_schema.role_table_grants
             where table_schema = %s and grantee = 'anon'
            """,
            (name,),
        )
        grants = cursor.fetchall()

    assert grants == [], f"anon can still reach: {grants}"


def _table_exists(connection, schema, table) -> bool:
    with connection.cursor() as cursor:
        cursor.execute(
            "select 1 from information_schema.tables "
            " where table_schema = %s and table_name = %s",
            (schema, table),
        )
        return cursor.fetchone() is not None
