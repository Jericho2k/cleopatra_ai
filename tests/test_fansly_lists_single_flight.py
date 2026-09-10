"""One Fansly Lists reconciliation per creator at a time.

sync_fansly_lists had no guard. The staleness check it did have answers "should
this run", not "am I the one running it", so two operators pressing Refresh — or
a manual refresh landing while the automatic pass fires — produced two
concurrent reconciliations diffing the same remote lists against the same local
mirror, each seeing the other's half-applied state.

The claim is tested twice: against a real PostgreSQL, because the guarantee is
an atomic conditional UPDATE and a double cannot demonstrate that; and through
the wrapper, because a correct claim function that nothing calls guards nothing.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import services.fansly_lists as fansly_lists

psycopg = pytest.importorskip("psycopg", reason="psycopg is required for schema tests")

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
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
    if not DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    name = f"cleo_lists_{uuid.uuid4().hex[:12]}"
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
            cursor.execute(
                f'insert into "{name}".creators (name) values (%s) returning id',
                ("A",),
            )
            creator_id = cursor.fetchone()[0]
            cursor.execute(
                f'insert into "{name}".creators (name) values (%s) returning id',
                ("B",),
            )
            other_id = cursor.fetchone()[0]
        yield connection, name, creator_id, other_id
    finally:
        try:
            with connection.cursor() as cursor:
                cursor.execute(f'drop schema if exists "{name}" cascade')
        finally:
            connection.close()


def _claim(cursor, schema, creator_id, stale_minutes=15):
    cursor.execute(
        f'select "{schema}".claim_fansly_lists_sync(%s, %s)',
        (creator_id, stale_minutes),
    )
    return cursor.fetchone()[0]


def _release(cursor, schema, creator_id):
    cursor.execute(
        f'select "{schema}".release_fansly_lists_sync(%s)', (creator_id,)
    )


# --- the database guarantee -------------------------------------------------


def test_only_one_of_two_claims_wins(schema):
    connection, name, creator_id, _ = schema
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id) is True
        assert _claim(cursor, name, creator_id) is False


def test_two_simultaneous_operators_do_not_both_start(schema):
    """The manual + manual race, with both transactions genuinely open."""
    _connection, name, creator_id, _ = schema

    first = psycopg.connect(DATABASE_URL)
    second = psycopg.connect(DATABASE_URL)
    outcomes = []
    try:
        c1, c2 = first.cursor(), second.cursor()
        c1.execute(f'set search_path to "{name}", public')
        c2.execute(f'set search_path to "{name}", public')

        outcomes.append(_claim(c1, name, creator_id))
        first.commit()
        outcomes.append(_claim(c2, name, creator_id))
        second.commit()
    finally:
        first.close()
        second.close()

    assert outcomes.count(True) == 1, "both operators started a reconciliation"
    assert outcomes.count(False) == 1


def test_releasing_lets_the_next_run_start(schema):
    connection, name, creator_id, _ = schema
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id) is True
        _release(cursor, name, creator_id)
        assert _claim(cursor, name, creator_id) is True


def test_different_creators_sync_concurrently(schema):
    """Per creator, not global: one agency's large refresh must not block
    everyone else's."""
    connection, name, creator_id, other_id = schema
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id) is True
        assert _claim(cursor, name, other_id) is True


def test_a_crashed_process_does_not_wedge_the_creator_forever(schema):
    """The claim is a timestamp, so it becomes reclaimable rather than needing
    an operator to clear it."""
    connection, name, creator_id, _ = schema
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id) is True
        # The claiming process dies here: no release ever happens.
        cursor.execute(
            f'update "{name}".creators '
            "set fansly_lists_sync_claimed_at = now() - interval '1 hour' "
            "where id = %s",
            (creator_id,),
        )
        assert _claim(cursor, name, creator_id) is True


def test_a_long_running_sync_is_not_reclaimed_underneath_itself(schema):
    connection, name, creator_id, _ = schema
    with connection.cursor() as cursor:
        assert _claim(cursor, name, creator_id) is True
        cursor.execute(
            f'update "{name}".creators '
            "set fansly_lists_sync_claimed_at = now() - interval '5 minutes' "
            "where id = %s",
            (creator_id,),
        )
        assert _claim(cursor, name, creator_id, stale_minutes=15) is False


def test_the_claim_does_not_look_like_a_completed_sync(schema):
    """Conflating the claim with last_fansly_lists_sync_at would make claiming
    suppress the next real refresh."""
    connection, name, creator_id, _ = schema
    with connection.cursor() as cursor:
        _claim(cursor, name, creator_id)
        cursor.execute(
            f'select last_fansly_lists_sync_at from "{name}".creators where id = %s',
            (creator_id,),
        )
        assert cursor.fetchone()[0] is None


def test_the_claim_functions_are_backend_only(schema):
    connection, name, _creator_id, _ = schema
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
                   'claim_fansly_lists_sync', 'release_fansly_lists_sync'
               )
            """,
            (name,),
        )
        granted = {(fn, role) for fn, role in cursor.fetchall() if role}

    for fn in ("claim_fansly_lists_sync", "release_fansly_lists_sync"):
        assert (fn, "service_role") in granted
        assert (fn, "authenticated") not in granted
        assert (fn, "anon") not in granted


# --- the wrapper actually uses it ------------------------------------------


@pytest.fixture
def wrapper(monkeypatch):
    state = {"claims": [], "releases": [], "runs": [], "grant": True, "raise": False}

    async def fake_claim(creator_id):
        state["claims"].append(creator_id)
        return state["grant"]

    async def fake_release(creator_id):
        state["releases"].append(creator_id)

    async def fake_sync(creator_id, account_id, **_kwargs):
        state["runs"].append(creator_id)
        if state["raise"]:
            raise RuntimeError("API Fansly 503")
        return {"status": "ok", "remote_lists": 2}

    monkeypatch.setattr(fansly_lists, "_claim_lists_sync", fake_claim)
    monkeypatch.setattr(fansly_lists, "_release_lists_sync", fake_release)
    monkeypatch.setattr(fansly_lists, "sync_fansly_lists", fake_sync)
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")
    return state


def _run(creator_id="creator-1"):
    return asyncio.run(
        fansly_lists.sync_fansly_lists_single_flight(creator_id, "acct-1")
    )


def test_the_winner_runs_and_releases(wrapper):
    result = _run()

    assert result["status"] == "ok"
    assert wrapper["runs"] == ["creator-1"]
    assert wrapper["releases"] == ["creator-1"]


def test_the_loser_reports_already_syncing_without_running(wrapper):
    wrapper["grant"] = False

    result = _run()

    assert result["status"] == "already_syncing"
    assert wrapper["runs"] == [], "a second reconciliation started anyway"
    assert wrapper["releases"] == [], "the loser released someone else's claim"


def test_a_failed_sync_still_releases(wrapper):
    """Otherwise one transient API error blocks synchronisation for the whole
    stale window."""
    wrapper["raise"] = True

    with pytest.raises(RuntimeError):
        _run()

    assert wrapper["releases"] == ["creator-1"]


def test_a_disabled_feature_never_claims(wrapper, monkeypatch):
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "false")

    assert _run()["status"] == "disabled"
    assert wrapper["claims"] == []


def test_a_deployment_without_the_migration_still_syncs(monkeypatch):
    """A missing claim function must not be able to stop list synchronisation."""

    class Boom:
        def rpc(self, *_args, **_kwargs):
            raise RuntimeError('function claim_fansly_lists_sync does not exist')

    monkeypatch.setattr(fansly_lists, "get_supabase", lambda: Boom())

    assert asyncio.run(fansly_lists._claim_lists_sync("creator-1")) is True


def test_a_present_claim_function_is_honoured(monkeypatch):
    class Db:
        def rpc(self, name, params):
            assert name == "claim_fansly_lists_sync"
            assert params["p_creator_id"] == "creator-1"
            return SimpleNamespace(execute=lambda: SimpleNamespace(data=False))

    monkeypatch.setattr(fansly_lists, "get_supabase", lambda: Db())

    assert asyncio.run(fansly_lists._claim_lists_sync("creator-1")) is False
