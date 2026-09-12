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

`scripts/production_preflight.py` verifies, read-only, that the effects of these
migrations are actually present in a target database. Run it after applying
anything, and before trusting a deployment. It never writes.

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

**Re-run `tenant_isolation_v1` AND `browser_least_privilege_v1`, in that order,
after adding any creator-owned table or view.** Both are idempotent by design.

They are a pair and must never be applied singly. `tenant_isolation_v1` drops
every policy on each table it discovers before creating its own
`FOR ALL TO authenticated` policy, so running it alone silently reopens SEC-001:
every creator-owned table becomes fully writable from the browser again.
`browser_least_privilege_v1` then narrows those policies to the operations the
dashboard actually performs, and revokes the broad Supabase grants that RLS
alone cannot constrain (PostgreSQL has no per-column RLS, so column-level access
is expressed as `GRANT ... (column)`).

`tests/test_schema_pipeline.py` asserts they are the last two entries of
`migration_order.txt`, in that order, and
`tests/test_browser_least_privilege.py` asserts no creator-owned table is left
with a `FOR ALL` policy for `authenticated`.

## Applying a migration to production

There is no automatic runner and this sprint deliberately did not build one.
The workflow is four steps, and the third is the one that was missing:

1. **Write the migration.** Additive where possible, idempotent where practical,
   listed in `migration_order.txt` (CI fails otherwise), and tested against a
   real PostgreSQL by `tests/test_schema_pipeline.py`.
2. **Apply it intentionally**, by hand, in the Supabase SQL editor, in the order
   `migration_order.txt` gives. Never `ci_baseline_schema.sql` or
   `ci_supabase_stubs.sql` — those are CI fixtures and applying them to
   production would be destructive.
3. **Verify by effect**:
   ```
   SUPABASE_DB_URL='postgresql://...' python scripts/production_preflight.py
   ```
   It is read-only. It reports PASS/FAIL per effect and names the file to apply
   for anything missing. Exit status is non-zero if anything FAILED.
4. **Deploy the code.** Every migration in this sprint degrades to the previous
   behaviour when absent, so the order of steps 2 and 4 is not load-bearing —
   but verifying before deploying means a missing migration is a report rather
   than an incident.

### Why verify-by-effect and not a migrations ledger

A ledger table records what someone *told* it was applied. For a project whose
base schema was created out of band and whose history was never tracked, the
first thing a new ledger would have to do is assert something about the past
that nobody can check. Marking the existing migrations "applied" because their
files exist would be fabricating history, and marking them un-applied would be
false too.

Verify-by-effect has no such problem: an index that exists, exists. It is also
strictly more useful, because it catches the case a ledger cannot — a migration
recorded as applied whose effect was later dropped, renamed, or never committed
because the transaction failed halfway.

**If a ledger is introduced later**, it should record only migrations applied
*from that point forward*, and the preflight should remain the authority on
whether an effect is actually present. The two answer different questions:
"was this run" and "is this true". Only the second one matters at 3am.

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

### Applying `message_platform_identity_v1.sql` (REL-002)

Production is currently logging:

```
[MESSAGE IDENTITY] unique index missing — falling back to check-then-insert.
Apply db/message_platform_identity_v1.sql (creator=...)
```

Ingestion still works in that state, which is exactly why it must not be left
alone: the fallback is a read followed by a non-atomic write, so two writers
racing on one platform message (webhook redelivery, webhook racing the poller,
two workers) can each decide the row is absent and each insert it. The unique
index is what makes ingestion idempotent; without it, idempotency is a
coincidence. The process also publishes `message_identity_index_missing` in
`/health` → `degraded_reasons` once it has taken that fallback, and the
dashboard surfaces it as a pending migration.

The file is unchanged and still correct: the key is
`(creator_id, fansly_message_id)`, partial on `fansly_message_id is not null`,
and it is what `save_message_result` upserts against
(`_PLATFORM_IDENTITY_CONFLICT` in `db/queries.py`). It is idempotent and safe to
re-run.

**Step 1 — duplicate diagnostic.** Run this against production first. The
migration refuses to create the index if it finds duplicates, reports how many
and which, and changes nothing — but knowing the answer beforehand is better
than learning it from a failed migration.

```sql
select creator_id, fansly_message_id, count(*) as copies,
       min(sent_at) as first_seen, max(sent_at) as last_seen
  from public.messages
 where fansly_message_id is not null
 group by creator_id, fansly_message_id
having count(*) > 1
 order by copies desc;
```

**Step 2 — expected clean result.** Zero rows:

```
 creator_id | fansly_message_id | copies | first_seen | last_seen
------------+-------------------+--------+------------+-----------
(0 rows)
```

Anything else is **production history**. Do not delete or merge it to make the
migration pass. Investigate which ingestion path produced each pair, decide
deliberately which row is canonical, and design an explicit cleanup; the
migration is intentionally refusing rather than repairing.

**Step 3 — apply.** Either paste the file into the Supabase SQL editor and run
it, or from a shell with `psql`:

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f db/message_platform_identity_v1.sql
```

`ON_ERROR_STOP=1` matters: the file's guard and its `CREATE UNIQUE INDEX` share
one `DO` block precisely so a refusal aborts both, but a `psql` run without it
would still continue to the second block.

**Step 4 — verify.** The index exists, is unique, and is partial:

```sql
select indexname, indexdef
  from pg_indexes
 where schemaname = 'public'
   and tablename = 'messages'
   and indexname = 'messages_creator_platform_identity_idx';
```

Expected — one row, reading:

```
create unique index messages_creator_platform_identity_idx
    on public.messages using btree (creator_id, fansly_message_id)
    where (fansly_message_id is not null)
```

Then confirm the application agrees: restart is not required, but the next
`save_message` with a platform id stops logging `[MESSAGE IDENTITY] unique index
missing`, and `/health` stops reporting `message_identity_index_missing` in
`degraded_reasons`. `scripts/production_preflight.py` also checks for it,
read-only.

An empty diagnostic result means the migration applies cleanly. A non-empty result is
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

## Applying `simulation_catalog_v1.sql` (owner-only simulation catalog)

Purely additive: two boolean/uuid/text column sets, four indexes, two `NOT VALID`
check constraints. It creates no table, drops nothing, and moves no data.

```bash
psql "$SUPABASE_DB_URL" -v ON_ERROR_STOP=1 -f db/simulation_catalog_v1.sql
```

Verify:

```sql
select table_name, column_name, column_default, is_nullable
  from information_schema.columns
 where table_schema = 'public'
   and column_name in ('simulation_only', 'source_creator_id',
                       'source_set_id', 'source_media_id')
 order by table_name, column_name;
```

Expected — `simulation_only` on both `vault_sets` and `creator_vault_media`,
`not null` with default `false`, plus the provenance columns. Defaulting to
`false` is the safety property: every pre-existing row, and every row a future
writer inserts without knowing about this feature, is live inventory. Test
content is the thing that has to be declared.

The two check constraints (`..._mirror_is_simulation_only`) are created `NOT
VALID` so the migration does not scan a large vault. They apply to every write
from the moment they exist. Validate them whenever convenient — both are no-ops
on a database that has never had a mirror:

```sql
alter table public.vault_sets
  validate constraint vault_sets_mirror_is_simulation_only;
alter table public.creator_vault_media
  validate constraint creator_vault_media_mirror_is_simulation_only;
```

**Deploy order.** The backend tolerates the columns being absent — the live
catalog filter retries once without it and logs
`[SIMULATION CATALOG] simulation_only column missing` — so the code may ship
first. The mirror endpoints will not work until the migration is applied, which
is correct: with no columns there can be no mirrored rows, and therefore nothing
for live planning to exclude.
