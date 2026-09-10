"""The preflight must be right about a database, and must never write to one.

Two claims are worth testing, and they are the two the tool would be worthless
without:

  * it agrees with reality — a fully migrated schema passes, a schema missing
    the migrations fails and names what to apply;
  * it cannot mutate anything. It is pointed at production by design, so
    "read-only" has to be a property, not an intention.

The CI-fixture guard is tested without a database at all, because it is the
check that would have caught a CI fixture nearly being applied to the live
project, and it should keep working even when nobody has a connection string.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg", reason="psycopg is required for schema tests")

from scripts import production_preflight as preflight  # noqa: E402

DATABASE_URL = os.getenv("TEST_DATABASE_URL", "").strip()
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "db"

needs_db = pytest.mark.skipif(
    not DATABASE_URL,
    reason="TEST_DATABASE_URL is not set",
)


def _order() -> list[str]:
    return [
        line.strip()
        for line in (DB / "migration_order.txt").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _build(connection, name: str, *, migrated: bool) -> None:
    def scoped(sql: str) -> str:
        return sql.replace("public.", f'"{name}".').replace("'public'", f"'{name}'")

    with connection.cursor() as cursor:
        cursor.execute(f'create schema "{name}"')
        cursor.execute((DB / "ci_supabase_stubs.sql").read_text())
        cursor.execute(f'set search_path to "{name}", public')
        cursor.execute(f'grant usage on schema "{name}" to anon, authenticated')
        cursor.execute(scoped((DB / "ci_baseline_schema.sql").read_text()))
        # Supabase's default posture, so "anon has no access" is something the
        # migration achieves rather than something the fixture never granted.
        cursor.execute(
            f'grant all on all tables in schema "{name}" to anon, authenticated'
        )
        cursor.execute(
            f'alter default privileges in schema "{name}" '
            "grant all on tables to anon, authenticated"
        )
        if migrated:
            for filename in _order():
                cursor.execute(scoped((DB / filename).read_text()))


@pytest.fixture
def migrated_schema():
    name = f"cleo_pf_ok_{uuid.uuid4().hex[:10]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        _build(connection, name, migrated=True)
        yield connection, name
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f'drop schema if exists "{name}" cascade')
        connection.close()


@pytest.fixture
def bare_schema():
    name = f"cleo_pf_bare_{uuid.uuid4().hex[:10]}"
    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        _build(connection, name, migrated=False)
        yield connection, name
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f'drop schema if exists "{name}" cascade')
        connection.close()


def _run(schema: str) -> tuple[int, preflight.Report]:
    report = preflight.Report()
    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        catalog = preflight.Catalog(connection, schema)
        preflight.check_schema(catalog, report)
        preflight.check_security(catalog, report)
    finally:
        connection.close()
    failures = [r for r in report.results if r.status == preflight.FAIL]
    return len(failures), report


# --- it agrees with reality -------------------------------------------------


@needs_db
def test_a_fully_migrated_schema_passes(migrated_schema):
    _connection, name = migrated_schema

    failures, report = _run(name)

    assert failures == 0, [
        (r.name, r.detail) for r in report.results if r.status == preflight.FAIL
    ]


@needs_db
def test_an_unmigrated_schema_fails_and_names_what_to_apply(bare_schema):
    _connection, name = bare_schema

    failures, report = _run(name)

    assert failures > 0
    failed = {r.name: r.detail for r in report.results if r.status == preflight.FAIL}
    assert "API-001 chat sync checkpoint" in failed
    assert "REL-003 purchase identity" in failed
    assert "VAULT-003 interruption state" in failed
    # A failure that does not say what to do about it is barely a failure.
    for name_, detail in failed.items():
        assert detail, f"{name_} failed with no remediation detail"


@needs_db
def test_missing_tenant_isolation_is_reported_as_a_security_failure(bare_schema):
    _connection, name = bare_schema

    _failures, report = _run(name)

    failed = {r.name for r in report.results if r.status == preflight.FAIL}
    assert "RLS enabled on creators" in failed
    assert "SEC-001 least-privilege policies" in failed


@needs_db
def test_an_absence_of_policies_is_not_reported_as_least_privilege(bare_schema):
    """The subtle false pass: no FOR ALL policies because there are no policies
    at all is the tenancy failure, not a narrowed one."""
    _connection, name = bare_schema

    _failures, report = _run(name)

    sec = next(
        r for r in report.results if r.name == "SEC-001 least-privilege policies"
    )
    assert sec.status == preflight.FAIL
    assert "no RLS policies" in sec.detail


# --- it cannot write --------------------------------------------------------


@needs_db
def test_the_preflight_mutates_nothing(migrated_schema):
    """Pointed at production by design, so read-only has to be a property."""
    connection, name = migrated_schema

    def snapshot() -> tuple:
        with connection.cursor() as cursor:
            cursor.execute(
                "select count(*) from information_schema.tables "
                " where table_schema = %s", (name,),
            )
            tables = cursor.fetchone()[0]
            cursor.execute(
                "select count(*) from pg_policies where schemaname = %s", (name,)
            )
            policies = cursor.fetchone()[0]
            cursor.execute(
                "select count(*) from information_schema.role_table_grants "
                " where table_schema = %s", (name,),
            )
            grants = cursor.fetchone()[0]
            cursor.execute(
                "select count(*) from pg_indexes where schemaname = %s", (name,)
            )
            indexes = cursor.fetchone()[0]
            cursor.execute(f'select count(*) from "{name}".creators')
            creators = cursor.fetchone()[0]
        return tables, policies, grants, indexes, creators

    before = snapshot()
    _run(name)
    assert snapshot() == before


@needs_db
def test_a_read_only_connection_is_enough_to_run_it(migrated_schema):
    """If any check tried to write, this would raise instead of reporting."""
    _connection, name = migrated_schema

    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        with connection.cursor() as cursor:
            cursor.execute("set default_transaction_read_only = on")
        report = preflight.Report()
        catalog = preflight.Catalog(connection, name)
        preflight.check_schema(catalog, report)
        preflight.check_security(catalog, report)
    finally:
        connection.close()

    assert report.results


# --- the guard that needs no database --------------------------------------


def test_ci_fixtures_are_not_listed_as_migrations():
    """The check that would have caught the near-miss."""
    report = preflight.Report()

    preflight.check_ci_fixtures_are_not_migrations(report)

    result = next(
        r for r in report.results
        if r.name == "CI fixtures excluded from migration order"
    )
    assert result.status == preflight.PASS


def test_a_ci_fixture_in_the_migration_order_is_a_failure(tmp_path, monkeypatch):
    order = tmp_path / "migration_order.txt"
    order.write_text("some_migration_v1.sql\nci_baseline_schema.sql\n")
    monkeypatch.setattr(preflight, "DB_DIR", tmp_path)

    report = preflight.Report()
    preflight.check_ci_fixtures_are_not_migrations(report)

    result = next(
        r for r in report.results
        if r.name == "CI fixtures excluded from migration order"
    )
    assert result.status == preflight.FAIL
    assert "never be applied to production" in result.detail


def test_db_000_status_is_reported_honestly():
    report = preflight.Report()

    preflight.check_base_schema_status(report)

    result = report.results[0]
    if (DB / "000_base_schema.sql").exists():
        assert result.status == preflight.PASS
    else:
        assert result.status == preflight.WARN
        assert "dump_base_schema.sh" in result.detail


# --- the environment checks never print a secret ---------------------------


def test_environment_checks_never_reveal_a_value(monkeypatch):
    secret = "sk-super-secret-value-9876543210"
    for name in (
        "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "APIFANSLY_API_KEY",
        "FANSLY_SESSION_KEY", "DASHBOARD_API_SECRET", "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY", "WEBHOOK_SECRET",
    ):
        monkeypatch.setenv(name, secret)
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("WRITER_DEFAULT_PROVIDER", "openrouter")
    monkeypatch.setenv("ANALYZER_PROVIDER", "anthropic")

    report = preflight.Report()
    preflight.check_environment(report)

    rendered = "\n".join(f"{r.name} {r.detail}" for r in report.results)
    assert secret not in rendered
    assert secret[:8] not in rendered
    assert secret[-8:] not in rendered


def test_a_development_app_env_is_a_failure(monkeypatch):
    monkeypatch.setenv("APP_ENV", "development")

    report = preflight.Report()
    preflight.check_environment(report)

    result = next(r for r in report.results if r.name == "APP_ENV")
    assert result.status == preflight.FAIL


def test_a_production_app_env_passes(monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")

    report = preflight.Report()
    preflight.check_environment(report)

    assert next(r for r in report.results if r.name == "APP_ENV").status == (
        preflight.PASS
    )


@needs_db
def test_the_lists_flag_is_checked_against_the_schema(bare_schema, monkeypatch):
    """A feature flag on without its schema fails every sync pass."""
    _connection, name = bare_schema
    monkeypatch.setenv("FANSLY_LISTS_SYNC_ENABLED", "true")

    connection = psycopg.connect(DATABASE_URL, autocommit=True)
    try:
        report = preflight.Report()
        preflight.check_lists_flag_against_schema(
            preflight.Catalog(connection, name), report
        )
    finally:
        connection.close()

    result = next(
        r for r in report.results if r.name == "Fansly lists flag matches schema"
    )
    assert result.status == preflight.FAIL


def test_the_report_exit_status_follows_failures(capsys):
    report = preflight.Report()
    report.ok("fine")
    assert report.render() == 0

    report = preflight.Report()
    report.ok("fine")
    report.warn("noticed")
    assert report.render() == 0, "a warning must not fail the run"

    report = preflight.Report()
    report.fail("broken", "apply something")
    assert report.render() == 1
    capsys.readouterr()
