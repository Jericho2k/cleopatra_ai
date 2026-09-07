# Database schema and migration discipline

Audit reference: **DB-000**, **SEC-003**, **REL-002**.

## The problem this documents

`db/` contains 18 additive migrations — `ALTER TABLE` and `CREATE INDEX`. It
contains no `CREATE TABLE` for `creators`, `fans`, `messages`, `suggestions`,
`chatter_creators`, `scheduled_actions`, `fan_lists`, `fan_list_members`,
`ppv_offers`, `vault_sets`, `creator_vault_media`, or the
`fan_conversation_summaries` view. Those objects were created out of band in
Supabase and exist only there.

Consequence: their primary keys, foreign keys, unique constraints, cascade
behaviour, nullability and indexes cannot be reviewed, reproduced, or tested
from this repository. A new environment cannot be built from the repo, and CI
could only ever test the additive migrations against nothing.

## Files

| File | What it is | Apply to production? |
|---|---|---|
| `db/000_base_schema.sql` | The authoritative `pg_dump --schema-only` of the production `public` schema. **Does not exist yet.** | It *is* production — never re-apply |
| `db/ci_baseline_schema.sql` | A **test fixture** reproducing enough shape for CI. Not authoritative. | **Never** |
| `db/ci_supabase_stubs.sql` | CI-only stand-ins for Supabase's `anon`/`authenticated`/`service_role` roles and `auth.uid()`. | **Never** |
| `db/migration_order.txt` | The deterministic order migrations are applied in. | n/a |
| `db/*_v1.sql`, `db/*_v2.sql` | The additive migrations. Idempotent; safe to re-run. | Yes, in the order above |

`ci_baseline_schema.sql` is deliberately **not** named `000_base_schema.sql`. It
was written from what the migrations and application code require, not dumped
from production, and must never be mistaken for the real thing.

## The CI pipeline

`tests/test_schema_pipeline.py` runs, against a fresh PostgreSQL:

```
ci_supabase_stubs.sql  ->  base schema  ->  migrations in migration_order.txt  ->  schema tests
```

It also asserts that every `db/*.sql` migration appears in `migration_order.txt`,
so a new migration cannot be merged without an explicit position in the order.

CI never touches production. There is no automatic migration step — applying a
migration to Supabase is a deliberate manual action.

## Migration ordering

The order in `migration_order.txt` is not alphabetical. Three real dependencies
force it:

1. `full_auto_lifecycle_v1` adds `creator_commercial_policies.ppv_payment_window_hours`;
   `operator_ppv_concurrency_v1` then alters that column.
2. `ppv_delivery_ledger_v1` creates `ppv_deliveries`; `operator_ppv_concurrency_v1`
   indexes it.
3. `tenant_isolation_v1` **discovers creator-owned objects at run time**, so it
   must run last. A table created by a later migration would otherwise get no
   RLS policy — which is how `fan_conversation_summaries` came to have none
   (SEC-003), except that it is a view and the discovery loop only ever looked
   at base tables.

**Re-run `tenant_isolation_v1` after adding any creator-owned table or view.**
It is idempotent by design.

## Switching CI to the real base schema

Once someone with production read access has run `scripts/dump_base_schema.sh`:

1. Commit the resulting `db/000_base_schema.sql`. Verify first that it contains
   no rows (`INSERT INTO` / `COPY`), no credentials, and nothing from the
   Supabase-managed `auth`, `storage`, `realtime` or `vault` schemas.
2. Point `BASE_SCHEMA` in `tests/test_schema_pipeline.py` at it.
3. Delete `db/ci_baseline_schema.sql`. Keep `db/ci_supabase_stubs.sql` — a
   plain PostgreSQL container still has no Supabase roles or `auth.uid()`.
4. Re-run the suite. Differences that surface at this point are real: the fixture
   is a guess and production is the truth.

## Open questions — blocked until the real dump exists

These cannot be answered from this repository, and none of them should be
"fixed" by guessing. **Do not add an index because the audit suspected it was
missing.** Run the query, look at the result, then decide.

```sql
-- Which indexes actually exist on the hot tables?
select tablename, indexname, indexdef
  from pg_indexes
 where schemaname = 'public'
   and tablename in (
        'messages', 'fans', 'creators', 'chatter_creators',
        'fan_list_members', 'fan_lists', 'scheduled_actions',
        'creator_vault_media'
   )
 order by tablename, indexname;
```

Specifically to confirm:

| Table | Access path | Why it matters |
|---|---|---|
| `messages` | `(fan_id, sent_at desc)` | The hottest query in the product — conversation history, several times per message |
| `messages` | `fansly_message_id` | Ingestion dedupe on every webhook and poll (REL-002) |
| `chatter_creators` | `(chatter_id)` | Every authenticated request **and** every RLS policy evaluation |
| `creators` | `apifansly_account_id`, `fansly_account_id` | Webhook creator resolution |
| `fans` | `(creator_id, platform_fan_id)` | Fan resolution on every inbound message |
| `fan_list_members` | `fan_id` | Auto Audience and list reconciliation |
| `scheduled_actions` | `(status, execute_at)`, `dedupe_key` | The durable queue |

### Before applying `message_platform_identity_v1.sql` (REL-002)

Run this against production first. The migration refuses to create the index if
it finds duplicates, reports how many and which, and changes nothing — but
knowing the answer beforehand is better than learning it from a failed
migration.

```sql
select creator_id, fansly_message_id, count(*) as copies,
       min(sent_at) as first_seen, max(sent_at) as last_seen
  from public.messages
 where fansly_message_id is not null
 group by creator_id, fansly_message_id
having count(*) > 1
 order by copies desc;
```

An empty result means the migration applies cleanly. A non-empty result is
**production history** — investigate which ingestion path produced each pair and
design an explicit cleanup. Do not delete rows to make the migration pass.

The key is `(creator_id, fansly_message_id)`, not `fansly_message_id` alone.
Fansly ids look globally unique — the webhook has always looked them up with no
creator filter and production has not mis-deduplicated — but the two choices
fail asymmetrically. If ids really are global, the composite key still catches
every race, because writers racing on one message always share a `creator_id`.
If ids turn out to be per-account, a global key silently rejects a second
creator's legitimately distinct message. Composite is correct under both.

And for SEC-003:

```sql
-- Is the summaries view security_invoker?
select c.relname, c.reloptions
  from pg_class c
  join pg_namespace n on n.oid = c.relnamespace
 where n.nspname = 'public'
   and c.relname = 'fan_conversation_summaries';
```
