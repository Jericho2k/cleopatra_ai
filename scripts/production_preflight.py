#!/usr/bin/env python3
"""Read-only production preflight.

Audit reference: DB-000, and the Sprint 4 deployment-safety work.

WHY THIS EXISTS
---------------
Migrations are applied to Supabase by hand. That is a deliberate choice — there
is no automatic runner and this script does not become one — but it means the
question "is production actually in the state the code expects" has, until now,
only been answerable by remembering. It was error-prone enough that a CI-only
test fixture was very nearly executed against the live project.

So this verifies BY EFFECT rather than by trusting a ledger. It asks the
database what exists: is the index there, does the function exist, is the view
security_invoker, does the browser still have write access it should not have.
An effect that is present is present regardless of whether anyone recorded
applying it, and an effect that is missing is missing regardless of what a
migrations table claims. For a project whose history was never tracked, that is
the only honest question to ask.

IT NEVER WRITES
---------------
Every statement is a catalog read. The connection is opened read-only and each
check runs inside a transaction that is rolled back. There is no code path here
that creates, alters, drops or updates anything, and the fixture guard below
refuses to let this script be the thing that runs CI SQL at production.

USAGE
-----
    SUPABASE_DB_URL='postgresql://...' python scripts/production_preflight.py

    # schema checks only, no environment checks
    SUPABASE_DB_URL='postgresql://...' python scripts/production_preflight.py --schema-only

    # environment checks only, no database needed
    python scripts/production_preflight.py --env-only

Exit status is 0 when nothing FAILED, 1 otherwise. WARN never fails the run: a
warning means "this shell cannot see that value", which is normal when checking
the database from a laptop while the key lives in Railway.

NEVER PRINTS SECRETS
--------------------
Environment checks report only whether a value is set and whether it is
plausibly shaped. No value, prefix or suffix is ever written to output.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_DIR = ROOT / "db"

# Files that are CI scaffolding and must NEVER reach a real database.
#
# Named here as well as in db/MIGRATIONS.md because a comment in a file is only
# read by someone who already opened it. This list is asserted against
# migration_order.txt below, so the two cannot drift.
CI_ONLY_SQL = ("ci_baseline_schema.sql", "ci_supabase_stubs.sql")

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Result:
    status: str
    name: str
    detail: str = ""


class Report:
    def __init__(self) -> None:
        self.results: list[Result] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.results.append(Result(status, name, detail))

    def ok(self, name: str, detail: str = "") -> None:
        self.add(PASS, name, detail)

    def fail(self, name: str, detail: str = "") -> None:
        self.add(FAIL, name, detail)

    def warn(self, name: str, detail: str = "") -> None:
        self.add(WARN, name, detail)

    def skip(self, name: str, detail: str = "") -> None:
        self.add(SKIP, name, detail)

    def render(self) -> int:
        width = max((len(r.name) for r in self.results), default=0)
        for result in self.results:
            line = f"{result.status:<4}  {result.name.ljust(width)}"
            if result.detail:
                line += f"  {result.detail}"
            print(line)

        failures = sum(1 for r in self.results if r.status == FAIL)
        warnings = sum(1 for r in self.results if r.status == WARN)
        passes = sum(1 for r in self.results if r.status == PASS)
        skipped = sum(1 for r in self.results if r.status == SKIP)
        print()
        print(
            f"{passes} passed, {failures} failed, {warnings} warnings, "
            f"{skipped} skipped"
        )
        if failures:
            print()
            print("FAILED checks mean production is not in the state this code "
                  "expects. See db/MIGRATIONS.md for what to apply.")
        return 1 if failures else 0


# ---------------------------------------------------------------------------
# Guard: the CI fixtures are not migrations
# ---------------------------------------------------------------------------


def check_ci_fixtures_are_not_migrations(report: Report) -> None:
    """Assert the repository still keeps CI scaffolding out of the apply order.

    This is the check that would have caught the near-miss. It costs nothing and
    it runs before anything touches a database.
    """
    order_file = DB_DIR / "migration_order.txt"
    if not order_file.exists():
        report.fail("migration order file present", "db/migration_order.txt missing")
        return

    listed = {
        line.strip()
        for line in order_file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    leaked = sorted(listed & set(CI_ONLY_SQL))
    if leaked:
        report.fail(
            "CI fixtures excluded from migration order",
            f"{', '.join(leaked)} is listed as a migration and must never be "
            "applied to production",
        )
    else:
        report.ok(
            "CI fixtures excluded from migration order",
            f"{', '.join(CI_ONLY_SQL)} are test-only",
        )

    missing = [name for name in CI_ONLY_SQL if not (DB_DIR / name).exists()]
    if missing:
        report.warn(
            "CI fixtures present in repository",
            f"missing: {', '.join(missing)}",
        )


def check_base_schema_status(report: Report) -> None:
    """DB-000 — is the authoritative schema in version control yet?"""
    if (DB_DIR / "000_base_schema.sql").exists():
        report.ok("DB-000 base schema committed", "db/000_base_schema.sql")
    else:
        report.warn(
            "DB-000 base schema committed",
            "not dumped yet; run "
            "SUPABASE_DB_URL=... scripts/dump_base_schema.sh",
        )


# ---------------------------------------------------------------------------
# Database checks. Every one is a catalog read.
# ---------------------------------------------------------------------------


class Catalog:
    """Read-only accessor for the checks below."""

    def __init__(self, connection, schema: str = "public") -> None:
        self._connection = connection
        self.schema = schema

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self._connection.cursor() as cursor:
            cursor.execute("begin read only")
            try:
                cursor.execute(sql, params)
                return cursor.fetchall()
            finally:
                cursor.execute("rollback")

    def table_exists(self, table: str) -> bool:
        return bool(self._query(
            "select 1 from information_schema.tables "
            " where table_schema = %s and table_name = %s",
            (self.schema, table),
        ))

    def column_exists(self, table: str, column: str) -> bool:
        return bool(self._query(
            "select 1 from information_schema.columns "
            " where table_schema = %s and table_name = %s and column_name = %s",
            (self.schema, table, column),
        ))

    def index_definition(self, table: str, predicate: str) -> str | None:
        rows = self._query(
            "select indexdef from pg_indexes "
            " where schemaname = %s and tablename = %s",
            (self.schema, table),
        )
        for (definition,) in rows:
            if predicate in definition.lower():
                return definition
        return None

    def unique_index_on(self, table: str, columns: list[str]) -> str | None:
        rows = self._query(
            "select indexdef from pg_indexes "
            " where schemaname = %s and tablename = %s",
            (self.schema, table),
        )
        for (definition,) in rows:
            lowered = definition.lower()
            if "unique" in lowered and all(c in lowered for c in columns):
                return definition
        return None

    def function_exists(self, name: str) -> bool:
        return bool(self._query(
            "select 1 from pg_proc p join pg_namespace n on n.oid = p.pronamespace "
            " where n.nspname = %s and p.proname = %s",
            (self.schema, name),
        ))

    def view_is_security_invoker(self, view: str) -> bool | None:
        rows = self._query(
            """
            select lower(o.option_value)
              from pg_class c
              join pg_namespace n on n.oid = c.relnamespace,
                   pg_options_to_table(c.reloptions) o
             where n.nspname = %s and c.relname = %s
               and lower(o.option_name) = 'security_invoker'
            """,
            (self.schema, view),
        )
        if not self._query(
            "select 1 from information_schema.views "
            " where table_schema = %s and table_name = %s",
            (self.schema, view),
        ):
            return None
        return bool(rows) and rows[0][0] in ("on", "true", "yes", "1")

    def rls_enabled(self, table: str) -> bool | None:
        rows = self._query(
            "select c.relrowsecurity from pg_class c "
            " join pg_namespace n on n.oid = c.relnamespace "
            " where n.nspname = %s and c.relname = %s",
            (self.schema, table),
        )
        return bool(rows[0][0]) if rows else None

    def for_all_policies(self) -> list[tuple]:
        return self._query(
            "select tablename, policyname from pg_policies "
            " where schemaname = %s and cmd = 'ALL' and 'authenticated' = any(roles) "
            " order by tablename",
            (self.schema,),
        )

    def authenticated_policy_count(self) -> int:
        rows = self._query(
            "select count(*) from pg_policies "
            " where schemaname = %s and 'authenticated' = any(roles)",
            (self.schema,),
        )
        return int(rows[0][0]) if rows else 0

    def policy_names(self, table: str) -> set[str]:
        return {
            row[0]
            for row in self._query(
                "select policyname from pg_policies "
                " where schemaname = %s and tablename = %s",
                (self.schema, table),
            )
        }

    def grants_to(self, role: str) -> list[tuple]:
        return self._query(
            "select table_name, privilege_type "
            "  from information_schema.role_table_grants "
            " where table_schema = %s and grantee = %s "
            " order by table_name, privilege_type",
            (self.schema, role),
        )

    def column_grants(self, role: str, table: str, privilege: str) -> set[str]:
        return {
            row[0]
            for row in self._query(
                "select column_name from information_schema.column_privileges "
                " where table_schema = %s and table_name = %s "
                "   and grantee = %s and privilege_type = %s",
                (self.schema, table, role, privilege.upper()),
            )
        }


def check_schema(catalog: Catalog, report: Report) -> None:
    """Verify the effects the application actually depends on."""

    # --- REL-002: message platform identity --------------------------------
    definition = catalog.unique_index_on(
        "messages", ["creator_id", "fansly_message_id"]
    )
    if definition:
        report.ok("message platform identity", "unique (creator_id, fansly_message_id)")
    else:
        report.fail(
            "message platform identity",
            "no unique index on (creator_id, fansly_message_id); "
            "apply db/message_platform_identity_v1.sql",
        )

    # --- Owner-only simulation catalog -------------------------------------
    #
    # A warning rather than a failure: without these columns no mirrored row can
    # exist, so live planning is correct by construction and the deployment is
    # fine. It is reported because the owner simulator's test catalog does not
    # work until the migration is applied.
    missing_catalog = [
        f"{table}.{column}"
        for table, column in (
            ("vault_sets", "simulation_only"),
            ("vault_sets", "source_creator_id"),
            ("creator_vault_media", "simulation_only"),
            ("creator_vault_media", "source_creator_id"),
        )
        if not catalog.column_exists(table, column)
    ]
    if missing_catalog:
        report.warn(
            "simulation catalog boundary",
            "missing " + ", ".join(missing_catalog)
            + "; apply db/simulation_catalog_v1.sql to use the owner simulator's "
            "test catalog",
        )
    else:
        report.ok(
            "simulation catalog boundary",
            "simulation_only and provenance columns present",
        )

    # --- The hottest read in the product -----------------------------------
    if catalog.index_definition("messages", "fan_id"):
        report.ok("messages conversation index", "an index on fan_id exists")
    else:
        report.warn(
            "messages conversation index",
            "no index containing fan_id; conversation history is the hottest "
            "query in the product",
        )

    # --- REL-006: durable ingestion ----------------------------------------
    if catalog.unique_index_on("scheduled_actions", ["dedupe_key"]):
        report.ok("scheduled action dedupe uniqueness", "unique (dedupe_key)")
    else:
        report.fail(
            "scheduled action dedupe uniqueness",
            "webhook redelivery can create duplicate obligations; "
            "apply db/durable_ingestion_v1.sql",
        )

    for name, detail in (
        ("claim_due_actions", "db/scheduled_action_claim_v1.sql"),
        ("claim_chat_reconciliation", "db/agency_operability_v1.sql"),
        ("vault_album_summary", "db/vault_album_summary_v1.sql"),
        ("attach_pending_ppv", "db/ppv_delivery_ledger_v1.sql"),
    ):
        if catalog.function_exists(name):
            report.ok(f"function {name}")
        else:
            report.fail(f"function {name}", f"apply {detail}")

    for index_table, fragment, label, source in (
        ("scheduled_actions", "status", "scheduled action claim indexes",
         "db/scheduled_action_claim_v1.sql"),
    ):
        if catalog.index_definition(index_table, fragment):
            report.ok(label)
        else:
            report.warn(label, f"consider {source}")

    # --- SEC-003: the summaries view ---------------------------------------
    invoker = catalog.view_is_security_invoker("fan_conversation_summaries")
    if invoker is None:
        report.warn(
            "summaries view security_invoker",
            "fan_conversation_summaries does not exist in this database",
        )
    elif invoker:
        report.ok("summaries view security_invoker")
    else:
        report.fail(
            "summaries view security_invoker",
            "the view bypasses RLS on fans and messages; "
            "apply db/tenant_isolation_v1.sql",
        )

    # --- Fansly Lists -------------------------------------------------------
    lists_columns = all(
        catalog.column_exists("fan_lists", column)
        for column in ("source", "external_list_id", "external_archived_at")
    )
    if lists_columns:
        report.ok("fan_lists Fansly source columns")
    else:
        report.fail(
            "fan_lists Fansly source columns",
            "apply db/fansly_lists_v1.sql before enabling "
            "FANSLY_LISTS_SYNC_ENABLED",
        )

    # --- Sprint 4 effects ---------------------------------------------------
    if catalog.column_exists("fans", "chat_last_message_id"):
        report.ok("API-001 chat sync checkpoint", "fans.chat_last_message_id")
    else:
        report.fail(
            "API-001 chat sync checkpoint",
            "every restart cold-syncs every chat; "
            "apply db/chat_sync_checkpoint_v1.sql",
        )

    if catalog.table_exists("platform_purchase_events"):
        if catalog.unique_index_on(
            "platform_purchase_events", ["creator_id", "platform_order_id"]
        ):
            report.ok("REL-003 purchase identity", "unique (creator_id, platform_order_id)")
        else:
            report.fail(
                "REL-003 purchase identity",
                "table exists but the unique index does not; re-apply "
                "db/purchase_identity_v1.sql",
            )
    else:
        report.fail(
            "REL-003 purchase identity",
            "concurrent purchase webhooks can double-count a sale; "
            "apply db/purchase_identity_v1.sql",
        )

    for name, source in (
        ("claim_platform_purchase", "db/purchase_identity_v1.sql"),
        ("claim_fansly_lists_sync", "db/fansly_lists_single_flight_v1.sql"),
    ):
        if catalog.function_exists(name):
            report.ok(f"function {name}")
        else:
            report.fail(f"function {name}", f"apply {source}")

    if catalog.column_exists("creators", "vault_sync_owner"):
        report.ok("VAULT-003 interruption state", "creators.vault_sync_owner")
    else:
        report.fail(
            "VAULT-003 interruption state",
            "an interrupted vault sync reports idle; "
            "apply db/vault_sync_interruption_v1.sql",
        )


def check_security(catalog: Catalog, report: Report) -> None:
    """SEC-001 and the tenancy boundary."""

    for table in ("creators", "fans", "messages", "chatter_creators"):
        enabled = catalog.rls_enabled(table)
        if enabled is None:
            report.warn(f"RLS enabled on {table}", "table not found")
        elif enabled:
            report.ok(f"RLS enabled on {table}")
        else:
            report.fail(
                f"RLS enabled on {table}",
                "every agency can read every other agency; "
                "apply db/tenant_isolation_v1.sql",
            )

    total_policies = catalog.authenticated_policy_count()
    offenders = catalog.for_all_policies()
    if total_policies == 0:
        # Not a pass. An absence of FOR ALL policies because there are no
        # policies at all is the tenancy failure above, not least privilege.
        report.fail(
            "SEC-001 least-privilege policies",
            "no RLS policies exist for authenticated; apply "
            "db/tenant_isolation_v1.sql then db/browser_least_privilege_v1.sql",
        )
    elif offenders:
        listed = ", ".join(sorted({table for table, _ in offenders})[:6])
        report.fail(
            "SEC-001 least-privilege policies",
            f"{len(offenders)} table(s) still grant every operation to the "
            f"browser ({listed}...); apply db/browser_least_privilege_v1.sql "
            "AFTER db/tenant_isolation_v1.sql",
        )
    else:
        report.ok(
            "SEC-001 least-privilege policies",
            "no creator-owned table grants FOR ALL to authenticated",
        )

    anon_grants = catalog.grants_to("anon")
    if anon_grants:
        report.fail(
            "anon has no table access",
            f"{len(anon_grants)} grant(s) to anon remain; "
            "apply db/browser_least_privilege_v1.sql",
        )
    else:
        report.ok("anon has no table access")

    # The column half of SEC-001: an operator may edit a fan's notes, not their
    # commercial state.
    if catalog.table_exists("fans"):
        updatable = catalog.column_grants("authenticated", "fans", "UPDATE")
        forbidden = {
            "total_spent", "spend_tier", "sales_log", "needs_human_review",
            "auto_mode", "pending_ppv_check", "active_session",
        } & updatable
        if forbidden:
            report.fail(
                "fan commercial state is backend-only",
                f"browser can update: {', '.join(sorted(forbidden))}",
            )
        elif updatable:
            report.ok(
                "fan commercial state is backend-only",
                f"browser may update {len(updatable)} note column(s) only",
            )
        else:
            report.warn(
                "fan commercial state is backend-only",
                "the browser cannot update any fan column; the FAN DETAILS "
                "form will not save",
            )

    if catalog.table_exists("creators"):
        updatable = catalog.column_grants("authenticated", "creators", "UPDATE")
        forbidden = {"apifansly_account_id", "fansly_account_id"} & updatable
        if forbidden:
            report.fail(
                "creator platform binding is backend-only",
                f"browser can update: {', '.join(sorted(forbidden))}",
            )
        else:
            report.ok("creator platform binding is backend-only")


# ---------------------------------------------------------------------------
# Environment. Presence and shape only — never a value.
# ---------------------------------------------------------------------------


def _is_set(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def check_environment(report: Report) -> None:
    app_env = os.environ.get("APP_ENV", "").strip()
    if app_env == "production":
        report.ok("APP_ENV", "production")
    elif not app_env:
        report.warn(
            "APP_ENV",
            "unset; core/environment.py resolves this to production, so auth "
            "fails closed — set it explicitly",
        )
    elif app_env == "development":
        report.fail(
            "APP_ENV",
            "development disables authentication fail-closed behaviour; "
            "never use it for a real deployment",
        )
    else:
        report.warn("APP_ENV", f"unrecognised value (treated as production)")

    required = [
        ("SUPABASE_URL", "the database"),
        ("SUPABASE_SERVICE_KEY", "the database"),
        ("APIFANSLY_API_KEY", "sending and receiving messages"),
        ("FANSLY_SESSION_KEY", "encrypting stored creator sessions"),
        ("DASHBOARD_API_SECRET", "authenticating operator requests"),
    ]
    for name, why in required:
        if _is_set(name):
            report.ok(f"{name} configured")
        else:
            report.warn(
                f"{name} configured",
                f"not visible to this shell; required in the deployment for {why}",
            )

    webhook_secret = _is_set("APIFANSLY_WEBHOOK_SECRET") or _is_set("WEBHOOK_SECRET")
    if webhook_secret:
        report.ok("webhook signing secret configured")
    else:
        report.warn(
            "webhook signing secret configured",
            "APIFANSLY_WEBHOOK_SECRET or WEBHOOK_SECRET must be set or the "
            "webhook refuses every delivery outside development",
        )

    # Model providers: which ones matter depends on how the routes are
    # configured, so this reports what the CURRENT configuration implies rather
    # than demanding all of them.
    writer_provider = os.environ.get(
        "WRITER_DEFAULT_PROVIDER", os.environ.get("CHAT_PROVIDER", "")
    ).strip().lower()
    analyzer_provider = os.environ.get("ANALYZER_PROVIDER", "").strip().lower()

    provider_keys = {
        "openrouter": "OPENROUTER_API_KEY",
        "together": "TOGETHER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }

    for role, provider in (("writer", writer_provider), ("analyzer", analyzer_provider)):
        if not provider:
            report.warn(f"{role} provider configured", "provider not set in this shell")
            continue
        key = provider_keys.get(provider)
        if key is None:
            report.warn(f"{role} provider configured", f"unrecognised provider")
            continue
        if _is_set(key):
            report.ok(f"{role} provider configured", f"{provider} key present")
        else:
            report.warn(
                f"{role} provider configured",
                f"{provider} selected but {key} is not visible to this shell",
            )

    # A feature flag that is on without its schema fails every pass.
    lists_enabled = os.environ.get(
        "FANSLY_LISTS_SYNC_ENABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    report.add(
        PASS if not lists_enabled else WARN,
        "Fansly lists flag",
        "off" if not lists_enabled
        else "on — the schema check above must also pass or every sync fails",
    )


def check_lists_flag_against_schema(catalog: Catalog | None, report: Report) -> None:
    """The one cross-check between configuration and schema."""
    enabled = os.environ.get(
        "FANSLY_LISTS_SYNC_ENABLED", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled or catalog is None:
        return
    if catalog.column_exists("fan_lists", "external_list_id"):
        report.ok("Fansly lists flag matches schema")
    else:
        report.fail(
            "Fansly lists flag matches schema",
            "FANSLY_LISTS_SYNC_ENABLED is on but db/fansly_lists_v1.sql has "
            "not been applied; every sync pass will fail",
        )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only verification that a database and environment are "
                    "in the state this code expects. Never writes.",
    )
    parser.add_argument(
        "--schema-only", action="store_true",
        help="skip environment checks",
    )
    parser.add_argument(
        "--env-only", action="store_true",
        help="skip database checks (no connection needed)",
    )
    parser.add_argument(
        "--database-url", default="",
        help="connection string; defaults to $SUPABASE_DB_URL or $TEST_DATABASE_URL",
    )
    parser.add_argument(
        "--schema", default="public",
        help="schema to inspect (default: public)",
    )
    args = parser.parse_args(argv)

    report = Report()

    # Repository-level guards first: they need no database and they are the
    # checks that would have caught the near-miss.
    check_ci_fixtures_are_not_migrations(report)
    check_base_schema_status(report)

    catalog: Catalog | None = None
    connection = None

    if not args.env_only:
        url = (
            args.database_url
            or os.environ.get("SUPABASE_DB_URL", "")
            or os.environ.get("TEST_DATABASE_URL", "")
        ).strip()
        if not url:
            report.skip(
                "database checks",
                "set SUPABASE_DB_URL (or pass --database-url) to verify schema",
            )
        else:
            try:
                import psycopg
            except ImportError:
                report.skip("database checks", "psycopg is not installed")
            else:
                try:
                    connection = psycopg.connect(url, autocommit=True)
                except Exception as exc:
                    report.fail("database reachable", type(exc).__name__)
                else:
                    report.ok("database reachable")
                    catalog = Catalog(connection, args.schema)
                    check_schema(catalog, report)
                    check_security(catalog, report)

    if not args.schema_only:
        check_environment(report)
        check_lists_flag_against_schema(catalog, report)

    if connection is not None:
        connection.close()

    return report.render()


if __name__ == "__main__":
    sys.exit(main())
