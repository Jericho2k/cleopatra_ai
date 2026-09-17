"""An agency operator's browser must not be able to read model routing.

THE FINDING THIS CLOSES
-----------------------
services/ai_stack_visibility.py redacts routing out of the responses main.py
serves, and it is careful and correct about that. The dashboard also does this:

    app/simulator/page.tsx:205   supabase.from('messages').select('*')

which is the operator's own JWT against Supabase, returning media_context
verbatim, on a table db/browser_least_privilege_v1.sql grants `authenticated`
SELECT over. So `media_context.reply_provenance.writer.actual.model`, the whole
`writer.attempts` fallback ladder, and `media_context.ai_stack.provider` were
in every agency browser regardless of what the routes redacted.

Two halves, tested separately because they fail separately:

  * the SPLIT — services/message_diagnostics.py, pure and unit-testable, which
    decides what leaves the row;
  * the GRANT — db/owner_only_diagnostics_v1.sql, which decides whether the
    browser could read the table it lands in. That half runs against a real
    PostgreSQL with real roles, RLS and GRANTs, for the reason
    tests/test_browser_least_privilege.py gives: the question is what
    PostgreSQL does when `authenticated` selects, and a mock would be asserting
    our own beliefs back at us.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest

from services.ai_stack_visibility import public_media_context, public_message_rows
from services.message_diagnostics import (
    PUBLIC_MEDIA_CONTEXT_KEYS,
    diagnostics_row,
    split_media_context,
    unrecognised_keys,
)

# The document as the pipeline actually assembles it: services/suggestions.py
# merges the ai_stack marker and the provenance record into the PPV metadata.
LEAKY_CONTEXT = {
    "ppv": {"media_ids": ["111"], "price_cents": 2500},
    "ai_stack": {
        "profile": "cleo_v3",
        "route": "commercial_complex",
        "prompt_version": "writer_v2",
        "provider": "openrouter",
        "model": "moonshotai/kimi-k2",
    },
    "reply_provenance": {
        "turn_id": "turn-1",
        "part": 1,
        "mode": "auto",
        "writer": {
            "requested": {"provider": "together", "model": "Qwen/Qwen3-235B"},
            "actual": {"provider": "openrouter", "model": "moonshotai/kimi-k2"},
            "attempts": [
                {"provider": "together", "model": "Qwen/Qwen3-235B", "ok": False},
                {"provider": "openrouter", "model": "moonshotai/kimi-k2", "ok": True},
            ],
        },
        "build": {"sha": "4a1683a", "flags_digest": "2ccb799c"},
    },
}

# Every string that names the supply chain. Asserted as a set against the
# serialized document, so a reshaping of the record cannot slip one through by
# moving it to a different key.
SUPPLY_CHAIN = (
    "openrouter",
    "together",
    "moonshotai/kimi-k2",
    "Qwen/Qwen3-235B",
    "commercial_complex",
    "writer_v2",
)


def _names_the_supply_chain(document) -> list[str]:
    blob = json.dumps(document, default=str)
    return [needle for needle in SUPPLY_CHAIN if needle in blob]


# ===========================================================================
# 1. The split
# ===========================================================================


def test_the_split_leaves_no_routing_on_the_row():
    public, owner_only = split_media_context(LEAKY_CONTEXT)

    assert _names_the_supply_chain(public) == []
    # And it is not merely absent from the top level.
    assert "reply_provenance" not in public
    assert public["ai_stack"] == {"profile": "cleo_v3"}


def test_the_split_keeps_what_the_product_needs():
    """A security change that breaks the product is not a fix."""
    public, _ = split_media_context(LEAKY_CONTEXT)

    assert public["ppv"] == {"media_ids": ["111"], "price_cents": 2500}
    # lib/aiStack.ts renders the profile, and "which stack answered" is the one
    # product-level fact the marker carries.
    assert public["ai_stack"]["profile"] == "cleo_v3"


def test_the_split_loses_nothing():
    """Owner-only is not a synonym for deleted.

    The record still has to answer "which model produced this reply" months
    later — that is what services/reply_provenance.py exists for. It simply has
    to answer it to the platform owner.
    """
    _, owner_only = split_media_context(LEAKY_CONTEXT)

    assert set(_names_the_supply_chain(owner_only)) == set(SUPPLY_CHAIN)
    writer = owner_only["reply_provenance"]["writer"]
    assert writer["actual"]["model"] == "moonshotai/kimi-k2"
    assert len(writer["attempts"]) == 2
    assert owner_only["ai_stack"]["provider"] == "openrouter"


def test_an_unreviewed_key_is_owner_only_rather_than_public():
    """The rule that reply_provenance itself broke.

    It was added next to ai_stack, under the rule that already governed
    ai_stack, and the denial named ai_stack alone. An allowlist is the only
    shape where the NEXT such key is safe before anyone notices it exists.
    """
    context = {"ppv": {"price_cents": 100}, "some_future_diagnostic": {"model": "x"}}

    public, owner_only = split_media_context(context)

    assert "some_future_diagnostic" not in public
    assert owner_only["some_future_diagnostic"] == {"model": "x"}
    # Diverted, never destroyed, and said out loud so a product field that
    # stops rendering is traceable to this decision.
    assert unrecognised_keys(context) == ["some_future_diagnostic"]


def test_a_recognised_key_is_not_reported_as_unrecognised():
    assert unrecognised_keys(LEAKY_CONTEXT) == []
    for key in PUBLIC_MEDIA_CONTEXT_KEYS:
        assert unrecognised_keys({key: {}}) == []


@pytest.mark.parametrize("value", [None, "", [], 0, "a string"])
def test_the_split_never_raises_on_a_shape_it_did_not_expect(value):
    """It runs in the send path. A recorder that raises stops a reply."""
    public, owner_only = split_media_context(value)
    assert public == value
    assert owner_only == {}


def test_the_row_denormalises_the_turn_it_belongs_to():
    _, owner_only = split_media_context(LEAKY_CONTEXT)
    row = diagnostics_row(
        message_id="m-1", creator_id="c-1", fan_id="f-1", record=owner_only
    )

    assert row["turn_id"] == "turn-1"
    assert row["part"] == 1
    assert row["record"] == owner_only


def test_a_record_without_a_turn_still_produces_a_row():
    """An ai_stack-only record, as the backfill of an older message produces."""
    row = diagnostics_row(
        message_id="m-1",
        creator_id="c-1",
        fan_id="f-1",
        record={"ai_stack": {"model": "kimi"}},
    )

    assert row["turn_id"] is None
    assert row["part"] == 0


# ===========================================================================
# 2. The response boundary, for deployments whose migration has not run
# ===========================================================================


def test_the_response_redaction_covers_provenance_too():
    """The originally reported finding, directly.

    public_media_context redacted ai_stack and returned every sibling key
    untouched, so reply_provenance passed straight through it.
    """
    assert _names_the_supply_chain(public_media_context(LEAKY_CONTEXT)) == []


def test_the_response_redaction_covers_message_rows():
    rows = public_message_rows([{"id": "m-1", "media_context": LEAKY_CONTEXT}])
    assert _names_the_supply_chain(rows) == []


def test_the_response_redaction_keeps_the_product_fields():
    public = public_media_context(LEAKY_CONTEXT)
    assert public["ppv"]["price_cents"] == 2500
    assert public["ai_stack"] == {"profile": "cleo_v3"}


# ===========================================================================
# 3. The grant, against a real PostgreSQL
# ===========================================================================

psycopg = pytest.importorskip("psycopg", reason="psycopg is required for schema tests")

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()

schema_only = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is not set; grant tests need a real PostgreSQL",
)

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db"

OPERATOR = "33333333-3333-3333-3333-333333333333"


def _order() -> list[str]:
    return [
        line.strip()
        for line in (DB / "migration_order.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


@pytest.fixture(scope="module")
def deployed():
    """One agency, one fan, and a message whose media_context leaks.

    The message is inserted BEFORE the migrations run, carrying the full
    pre-fix document. That is what makes the backfill testable: a fix that only
    applies to new messages leaves every existing conversation readable.
    """
    name = f"cleo_diag_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)

    def scoped(sql: str) -> str:
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            cursor.execute((DB / "ci_supabase_stubs.sql").read_text())
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(scoped((DB / "ci_baseline_schema.sql").read_text()))

            # Model Supabase, not a clean-room PostgreSQL: a real project ships
            # broad grants, so the denials below have to be what CLOSES them
            # rather than an absence of any grant ever having been made.
            cursor.execute(f'grant usage on schema "{name}" to anon, authenticated')
            cursor.execute(f'grant usage on schema "{name}" to service_role')
            cursor.execute(
                f'grant all on all tables in schema "{name}" to anon, authenticated'
            )
            cursor.execute(
                f'alter default privileges in schema "{name}" '
                "grant all on tables to anon, authenticated"
            )

            cursor.execute(
                f'insert into "{name}".creators (name) values (%s) returning id',
                ("Agency",),
            )
            creator_id = cursor.fetchone()[0]
            cursor.execute(
                f'insert into "{name}".chatter_creators (chatter_id, creator_id) '
                "values (%s, %s)",
                (OPERATOR, creator_id),
            )
            cursor.execute(
                f'insert into "{name}".fans (creator_id, display_name, platform_fan_id) '
                "values (%s, %s, %s) returning id",
                (creator_id, "Fan", "p-1"),
            )
            fan_id = cursor.fetchone()[0]
            cursor.execute(
                f'insert into "{name}".messages '
                "(fan_id, creator_id, role, content, media_context) "
                "values (%s, %s, 'creator', 'hey', %s) returning id",
                (fan_id, creator_id, json.dumps(LEAKY_CONTEXT)),
            )
            message_id = cursor.fetchone()[0]

            for filename in _order():
                cursor.execute(scoped((DB / filename).read_text()))

        yield connection, name, {
            "creator_id": creator_id,
            "fan_id": fan_id,
            "message_id": message_id,
        }
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _as_operator(connection, schema, sql, params=None):
    """Run one statement as `authenticated`, impersonating the operator."""
    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{schema}", public')
            cursor.execute("set local role authenticated")
            cursor.execute(f"set local request.jwt.claim.sub = '{OPERATOR}'")
            cursor.execute(sql, params or ())
            return cursor.fetchall()
        finally:
            cursor.execute("rollback")


@schema_only
def test_the_backfill_moved_the_record_off_the_existing_message(deployed):
    """The message predates the migration. Its routing still has to leave."""
    connection, name, data = deployed

    with connection.cursor() as cursor:
        cursor.execute(
            f'select media_context from "{name}".messages where id = %s',
            (data["message_id"],),
        )
        media_context = cursor.fetchone()[0]

    assert _names_the_supply_chain(media_context) == []
    assert "reply_provenance" not in media_context
    assert media_context["ai_stack"] == {"profile": "cleo_v3"}
    # Untouched. The backfill is a security change, not a scrubber.
    assert media_context["ppv"]["price_cents"] == 2500


@schema_only
def test_the_backfill_kept_the_record(deployed):
    connection, name, data = deployed

    with connection.cursor() as cursor:
        cursor.execute(
            f'select turn_id, part, record from "{name}".message_diagnostics '
            "where message_id = %s",
            (data["message_id"],),
        )
        turn_id, part, record = cursor.fetchone()

    assert turn_id == "turn-1"
    assert part == 1
    assert set(_names_the_supply_chain(record)) == set(SUPPLY_CHAIN)


@schema_only
def test_an_operator_cannot_read_the_diagnostics_table(deployed):
    """The whole point. This operator owns the creator, the fan and the message."""
    connection, name, _ = deployed

    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _as_operator(connection, name, f'select * from "{name}".message_diagnostics')


@schema_only
def test_an_operator_cannot_read_the_registry_either(deployed):
    """Otherwise the browser is handed the list of tables worth attacking."""
    connection, name, _ = deployed

    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        _as_operator(connection, name, f'select * from "{name}".owner_only_tables')


@schema_only
def test_the_operator_can_still_read_their_own_messages(deployed):
    """The narrowing must not have cost them the conversation."""
    connection, name, data = deployed

    rows = _as_operator(
        connection, name,
        f'select content, media_context from "{name}".messages where id = %s',
        (data["message_id"],),
    )

    assert rows[0][0] == "hey"
    assert _names_the_supply_chain(rows[0][1]) == []


@schema_only
def test_the_discovery_migrations_did_not_re_grant_it(deployed):
    """The trap this registry exists for.

    tenant_isolation_v1 and browser_least_privilege_v1 find creator-owned
    tables by looking for a creator_id or fan_id column. message_diagnostics
    has both. Without the registry they would have granted `authenticated`
    SELECT on it automatically, at the next run, silently — re-opening the hole
    from the same two files that are supposed to close things.
    """
    connection, name, _ = deployed

    with connection.cursor() as cursor:
        cursor.execute(
            "select count(*) from information_schema.role_table_grants "
            "where table_schema = %s and table_name = 'message_diagnostics' "
            "and grantee in ('anon', 'authenticated')",
            (name,),
        )
        assert cursor.fetchone()[0] == 0

        cursor.execute(
            "select count(*) from pg_policies "
            "where schemaname = %s and tablename = 'message_diagnostics' "
            "and 'authenticated' = any(roles)",
            (name,),
        )
        assert cursor.fetchone()[0] == 0

        # And it is registered, so the next owner-only table gets this for free.
        cursor.execute(
            f'select reason from "{name}".owner_only_tables '
            "where table_name = 'message_diagnostics'"
        )
        assert cursor.fetchone() is not None


@schema_only
def test_service_role_still_reads_it(deployed):
    """Every background job and the owner trace endpoint run as service_role."""
    connection, name, data = deployed

    with connection.cursor() as cursor:
        cursor.execute("begin")
        try:
            cursor.execute(f'set local search_path to "{name}", public')
            cursor.execute("set local role service_role")
            cursor.execute(
                f'select record from "{name}".message_diagnostics where message_id = %s',
                (data["message_id"],),
            )
            record = cursor.fetchone()[0]
        finally:
            cursor.execute("rollback")

    assert record["reply_provenance"]["writer"]["actual"]["model"] == "moonshotai/kimi-k2"
