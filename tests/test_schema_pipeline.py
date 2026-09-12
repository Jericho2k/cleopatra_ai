"""DB-000 — a fresh PostgreSQL must accept the whole schema, in order.

The repository holds 18 additive migrations and no CREATE TABLE for any of the
objects they alter, so CI could previously only apply migrations to nothing.
This applies the full pipeline — Supabase stubs, base schema, then every
migration in db/migration_order.txt — to a throwaway database, and asserts the
invariants the additive migrations depend on.

Skipped unless TEST_DATABASE_URL points at a disposable PostgreSQL; CI provides
one through the postgres service in .github/workflows/ci.yml.

    TEST_DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres \\
        pytest tests/test_schema_pipeline.py
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
STUBS = DB / "ci_supabase_stubs.sql"
ORDER_FILE = DB / "migration_order.txt"

# Swap this for db/000_base_schema.sql once the production dump lands — see
# db/MIGRATIONS.md § Switching CI to the real base schema.
BASE_SCHEMA = DB / "ci_baseline_schema.sql"


def migration_order() -> list[str]:
    lines = ORDER_FILE.read_text(encoding="utf-8").splitlines()
    return [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def test_every_migration_has_a_declared_position():
    """A new migration cannot be merged without choosing where it runs."""
    on_disk = {
        path.name
        for path in DB.glob("*.sql")
        if not path.name.startswith("ci_") and path.name != "000_base_schema.sql"
    }
    declared = set(migration_order())

    assert on_disk - declared == set(), (
        "these migrations are not listed in db/migration_order.txt"
    )
    assert declared - on_disk == set(), (
        "db/migration_order.txt lists files that do not exist"
    )


def test_migration_order_has_no_duplicates():
    order = migration_order()
    assert len(order) == len(set(order))


def test_tenant_isolation_and_least_privilege_run_last_as_a_pair():
    """Both discover creator-owned objects at run time, and their order matters.

    tenant_isolation_v1 must come after every migration that creates a
    creator-owned table, or that table gets no policy at all.

    browser_least_privilege_v1 must come immediately after tenant_isolation_v1,
    because tenant_isolation_v1 DROPS every policy on each table it discovers
    before creating its own FOR ALL policy. Anything that narrows those policies
    has to run afterwards or it is silently undone — which is also why the two
    have to be applied together to production, not one at a time (SEC-001).
    """
    order = migration_order()

    assert order[-2:] == [
        "tenant_isolation_v1.sql",
        "browser_least_privilege_v1.sql",
    ], (
        "tenant_isolation_v1 and browser_least_privilege_v1 must be the last "
        "two migrations, in that order: the first grants FOR ALL to "
        "authenticated and the second narrows it (SEC-001)."
    )


@pytest.fixture(scope="module")
def pipeline():
    """A disposable schema with the base schema and every migration applied."""
    name = f"cleo_pipeline_{uuid.uuid4().hex[:12]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)

    def scoped(sql: str) -> str:
        """Retarget a migration at the throwaway schema.

        Two rewrites, not one. The `public.` prefix covers DDL, but
        tenant_isolation_v1 also *queries* the catalog with a literal
        `schemaname = 'public'` / `table_schema = 'public'` to discover which
        objects to police. Without rewriting those too it would enumerate the
        real public schema and try to drop its policies on our tables.
        """
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    try:
        with connection.cursor() as cursor:
            cursor.execute(f'create schema "{name}"')
            # The role/auth stubs are genuinely global; they are idempotent.
            cursor.execute(STUBS.read_text(encoding="utf-8"))
            cursor.execute(f'set search_path to "{name}", public')
            cursor.execute(scoped(BASE_SCHEMA.read_text(encoding="utf-8")))
            for filename in migration_order():
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


def _objects(connection, schema, kind):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select table_name
              from information_schema.tables
             where table_schema = %s and table_type = %s
            """,
            (schema, kind),
        )
        return {row[0] for row in cursor.fetchall()}


def test_the_whole_pipeline_applies_to_a_fresh_database(pipeline):
    connection, name = pipeline
    tables = _objects(connection, name, "BASE TABLE")

    for required in (
        "creators",
        "fans",
        "messages",
        "suggestions",
        "chatter_creators",
        "fan_lists",
        "fan_list_members",
        "scheduled_actions",
        "ppv_offers",
        "vault_sets",
        "creator_vault_media",
        "fan_commercial_states",
        "ppv_deliveries",
        "model_usage_events",
        "scripts",
        "blocked_words",
        "reengagement_log",
        "reengagement_settings",
    ):
        assert required in tables, f"{required} is missing after the full pipeline"


def test_the_summaries_view_exists(pipeline):
    connection, name = pipeline
    assert "fan_conversation_summaries" in _objects(connection, name, "VIEW")


def test_migrations_are_idempotent(pipeline):
    """They are deployed by re-running them; a second pass must be a no-op."""
    connection, name = pipeline
    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        for filename in migration_order():
            sql = (DB / filename).read_text(encoding="utf-8")
            try:
                cursor.execute(
                    sql.replace("public.", f'"{name}".').replace(
                        "'public'", f"'{name}'"
                    )
                )
            except Exception as exc:  # pragma: no cover - failure detail
                raise AssertionError(
                    f"migration {filename} is not idempotent: {exc}"
                ) from exc


def _columns(connection, schema, table) -> set[str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            select column_name
              from information_schema.columns
             where table_schema = %s and table_name = %s
            """,
            (schema, table),
        )
        return {row[0] for row in cursor.fetchall()}


def test_the_simulation_catalog_columns_exist_and_default_to_live(pipeline):
    """The live/simulation boundary is a column, so it has to actually be there.

    Defaulting to ``false`` matters as much as existing: every row that predates
    the migration, and every row a future writer inserts without thinking about
    this feature, is live inventory. Test content is the thing that has to be
    declared.
    """
    connection, name = pipeline

    for table in ("vault_sets", "creator_vault_media"):
        columns = _columns(connection, name, table)
        assert "simulation_only" in columns, f"{table} has no live/test boundary"
        assert "source_creator_id" in columns, f"{table} keeps no provenance"

        with connection.cursor() as cursor:
            cursor.execute(
                """
                select column_default, is_nullable
                  from information_schema.columns
                 where table_schema = %s and table_name = %s
                   and column_name = 'simulation_only'
                """,
                (name, table),
            )
            default, nullable = cursor.fetchone()
            assert default == "false", f"{table}.simulation_only must default to live"
            assert nullable == "NO"

    assert "source_set_id" in _columns(connection, name, "vault_sets")
    assert "source_media_id" in _columns(connection, name, "creator_vault_media")


def test_a_mirrored_row_cannot_be_left_unmarked(pipeline):
    """Provenance without the flag would be sellable test content."""
    connection, name = pipeline

    with connection.cursor() as cursor:
        cursor.execute(f'set search_path to "{name}", public')
        cursor.execute(
            """
            insert into creators (id) values (gen_random_uuid()) returning id
            """
        )
        creator_id = cursor.fetchone()[0]

        # The control: correctly marked, it inserts. Without this the negative
        # case below could be passing for some unrelated NOT NULL violation.
        cursor.execute(
            """
            insert into vault_sets (creator_id, source_creator_id, simulation_only)
            values (%s, %s, true)
            """,
            (creator_id, creator_id),
        )

        with pytest.raises(Exception) as failure:
            cursor.execute(
                """
                insert into vault_sets (creator_id, source_creator_id, simulation_only)
                values (%s, %s, false)
                """,
                (creator_id, creator_id),
            )
        assert "vault_sets_mirror_is_simulation_only" in str(failure.value)


def test_the_mirror_identity_is_unique_per_target_and_source(pipeline):
    """A refresh updates rather than duplicating, which the index is what makes
    true rather than the code remembering to delete first."""
    connection, name = pipeline

    with connection.cursor() as cursor:
        cursor.execute(
            """
            select indexname
              from pg_indexes
             where schemaname = %s and tablename in ('vault_sets', 'creator_vault_media')
            """,
            (name,),
        )
        indexes = {row[0] for row in cursor.fetchall()}

    assert "vault_sets_simulation_mirror_idx" in indexes
    assert "creator_vault_media_simulation_mirror_idx" in indexes
