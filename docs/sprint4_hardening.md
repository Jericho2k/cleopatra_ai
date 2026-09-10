# Sprint 4 — final pre-agency-beta hardening

Reference: `docs/ENGINEERING_AUDIT_2026-09.md`, `docs/sprint3_optimization.md`.

The question this sprint had to answer was not "is Cleopatra well built". It was:

> Would I put 10–20 real creator accounts from an OFM agency onto this
> deployment and let them use Assisted and Full Auto?

**Answer: GO WITH CONDITIONS.** The conditions are in §19 and they are all
deployment steps, not engineering work.

---

## 1. Starting SHAs

Fetched at sprint start; both matched the prompt.

| Repository | Starting SHA |
|---|---|
| `Jericho2k/cleopatra_ai` | `da81a88b7c2dabc3dadf2d18f746925468b69410` |
| `Jericho2k/cleopatra-dashboard` | `6b90f6e9f343bfac0cc9241cd3dcea0aa8bd0259` |

Baselines before any change: backend **812 passed**; dashboard **79 passed**,
`tsc` clean, 107 lint warnings / 0 errors.

## 2. Final branches

Both repositories: `claude/cleopatra-sprint-4-hardening-lua5ii`.

| Repository | Commits |
|---|---|
| backend | 7 |
| dashboard | 2 |

## 3. Findings already fixed before Sprint 4

Re-checked against current `main`, not assumed:

- REL-002 message platform identity — `message_platform_identity_v1`, unique
  `(creator_id, fansly_message_id)`.
- REL-006 durable webhook ingestion — `durable_ingestion_v1`, dedupe uniqueness.
- SEC-003 summaries view — `security_invoker` via `tenant_isolation_v1`.
- SEC-002 / API-003 / FE-006 — creator-filtered reads, paginated fan reads,
  write-only-on-change.
- VAULT-001 vault load control; PERF-006 API Fansly connection pool.
- Scheduled-action concurrency with admission control; `claim_due_actions` RPC.
- Tenant isolation itself — an operator for agency A genuinely cannot reach
  agency B's rows.

## 4. Findings fixed now

| ID | Finding | Fix |
|---|---|---|
| **API-001** | Reconciliation checkpoint in process memory; every restart cold-synced every chat | Durable checkpoint on `fans.chat_last_message_id` |
| **SEC-001** | `FOR ALL TO authenticated` on every creator-owned table | Per-operation policies + column GRANTs |
| **REL-003** | Purchase dedupe by scanning `fans.sales_log` JSON, then writing | `platform_purchase_events`, unique `(creator_id, platform_order_id)` |
| **REL-004** | `_processed_messages` cleared wholesale; retry maps never pruned | Bounded FIFO; pruning on the existing reconcile pass |
| **REL-005** | Every failure retried 8× regardless of cause | Permanent / transient / writer-quality classification |
| **VAULT-003** | Interrupted vault sync reported "idle" | Durable run ownership; reports `interrupted` |
| **Lists** | `sync_fansly_lists` had no single-flight guard | Per-creator DB claim |
| **FE-007** | Effect depended on state it mutated; partial response looped | Requested-ID ref + backfill of omissions |
| *(new)* | DB thread pool capped at 8 on a 4-vCPU box, invisibly | Explicit, logged, tunable, in `/health` |

## 5. Files changed

Backend — 33 files, +6,709 / −103. Dashboard — 11 files, +456 / −46.
Full list in `git diff --stat origin/main..HEAD` on each branch.

New backend modules: `core/action_failures.py`, `core/bounded_state.py`,
`core/db_executor.py`, `scripts/production_preflight.py`,
`scripts/model_cache_report.py`, `scripts/load_test_sprint4.py`.

New dashboard module: `lib/ppvMedia.ts`.

## 6. Migrations added

Five, all additive and idempotent, all applied in
`tests/test_schema_pipeline.py` against real PostgreSQL:

| File | Effect |
|---|---|
| `db/chat_sync_checkpoint_v1.sql` | `fans.chat_last_message_id`, `chat_last_synced_at` |
| `db/purchase_identity_v1.sql` | `platform_purchase_events` + 3 claim RPCs |
| `db/vault_sync_interruption_v1.sql` | `creators.vault_sync_started_at/finished_at/owner` |
| `db/fansly_lists_single_flight_v1.sql` | `creators.fansly_lists_sync_claimed_at` + 2 RPCs |
| `db/browser_least_privilege_v1.sql` | SEC-001 policies and column grants |

## 7. Manual production migrations required

In this order. `browser_least_privilege_v1` **must** be preceded by
`tenant_isolation_v1` in the same session — see §26.

## 8. DB-000 status

**Externally blocked, unchanged.** `db/000_base_schema.sql` does not exist and
was deliberately not invented.

One command unblocks it:

```
SUPABASE_DB_URL='postgresql://postgres:...@db.<ref>.supabase.co:5432/postgres' \
    scripts/dump_base_schema.sh
```

Then follow `db/MIGRATIONS.md § Switching CI to the real base schema`.

What was done instead of guessing: `scripts/production_preflight.py` verifies
the effects that matter **by asking the live database**, which answers the
operational half of DB-000 (is production in the expected state) without
answering the archival half (what is the authoritative schema).

The CI fixture was extended with columns the dashboard demonstrably reads and
writes — `fans.age/payday/hobbies/relationship_status`, the creator settings
columns, the vault classification columns — because without them CI could not
test the SEC-001 column grants at all. This is still a fixture and still not
authoritative.

## 9. Production indexes discovered

**None.** Requires the schema dump. The preflight checks the ones the code
depends on and reports what it finds; the open questions in
`db/MIGRATIONS.md § Open questions` are unchanged. No speculative index was
added.

## 10. SEC-001 final browser permission matrix

Built by inventorying every `.from(...)` call in the dashboard, not by guessing.

| Table | SELECT | INSERT | UPDATE | DELETE |
|---|---|---|---|---|
| `creators` | ✅ | ❌ | 8 settings columns only | ❌ |
| `fans` | ✅ | ❌ | 7 note columns only | ❌ |
| `blocked_words` | ✅ | ✅ | ❌ | ✅ |
| `fan_lists` | ✅ | ✅ (local only) | name/color/exclude, local only | ✅ local only |
| `fan_list_members` | ✅ | ✅ local only | ✅ local only (upsert) | ✅ local only |
| `vault_sets` | ✅ | ✅ | ✅ | ✅ |
| `creator_vault_media` | ✅ | ❌ | 12 classification columns | ❌ |
| `chatter_creators` | ✅ | ❌ | ❌ | ❌ |
| `messages`, `suggestions`, `scripts`, summaries view | ✅ | ❌ | ❌ | ❌ |
| **every other creator-owned table** | ✅ | ❌ | ❌ | ❌ |

Explicitly **not** browser-writable: `scheduled_actions`, `ppv_deliveries`,
`ppv_approval_requests`, `platform_purchase_events`, all commercial and
lifecycle tables.

Explicitly **not** browser-updatable columns: `creators.apifansly_account_id`,
`creators.fansly_account_id`, `fans.total_spent`, `fans.spend_tier`,
`fans.sales_log`, `fans.needs_human_review`, `fans.auto_mode`,
`fans.pending_ppv_check`, `fans.active_session`, `fan_lists.source`,
`fan_lists.external_list_id`, `creator_vault_media.url`,
`creator_vault_media.fansly_media_id`.

`anon` has **no** table access. `service_role` is unchanged (BYPASSRLS).

**The tests are not vacuous.** With `browser_least_privilege_v1` removed from
the order, 14 of 30 fail and name the vulnerability: inserts into
`scheduled_actions` and the PPV ledger, rewritten message history, edited
`total_spent`, a repointed creator binding, and 308 grants to `anon`.

## 11. API-001 restart call reduction

MEASURED by `tests/test_chat_sync_restart.py`, which counts real
`list_chat_messages` calls through the real `sync_chats`.

| | Before | After |
|---|---|---|
| 2,000 chats, cold start | 2,000 | 2,000 |
| 2,000 chats, **after restart, unchanged** | **2,000** | **0** |
| 20 creators × 2,000 chats, after restart | **40,000** | **0** |
| One chat's marker moved | 1 | 1 |
| One new chat | 1 | 1 |

A deploy no longer costs 40,000 provider calls. Calls are suppressed **only**
when the remote marker is byte-identical to one stored after a successful sync
of that same chat, so this cannot lose a message; a deleted newest message moves
`lastMessageId` and therefore triggers a sync.

## 12. Purchase idempotency design

`platform_purchase_events`, unique on `(creator_id, platform_order_id)`.

Composite, not global, for the same reason message identity is: if Fansly order
ids are unique only per account, a global key would silently reject a second
creator's real sale. Composite is correct under both readings.

Flow: claim → work → settle. Every path that returns **without** recording a
purchase releases the claim, because a claim left behind for work that never
happened would answer a legitimate redelivery "duplicate" and lose the sale. A
row already marked `processed` cannot be released, so a late release from a
crashed handler cannot un-record a real purchase.

`sales_log` is not rewritten or deleted. It is still scanned *after* the claim
so pre-migration orders keep deduplicating; it simply stops being the
concurrency authority, which it was never able to be.

Proven against real PostgreSQL with genuinely interleaved transactions,
including the blocking interleaving.

## 13. REL-005 final retry semantics

| Class | Example | Budget |
|---|---|---|
| TRANSIENT (default) | OpenRouter 429/503, API Fansly 503, DB blip | 8 |
| PERMANENT | Creator not connected, creator missing | **1**, FAILED with a code |
| WRITER QUALITY | No usable candidate | 2 |
| OBSOLETE | Fan replied, human sent, Auto off | 0 — completed, not failed |
| PPV_RECONCILE | Still pending | 50 (unchanged) |

Anything unrecognised stays TRANSIENT: guessing "permanent" wrongly drops a real
message, which is the more expensive mistake.

The saving is not the retries, it is the pipelines. `_run_auto_reply` now
establishes that a delivery route can exist **before** spending an analyzer and
writer run. A disconnected creator with 20 queued fans previously cost
**160 model pipelines** (20 × 8) to discover; it now costs 20 indexed reads.

Operator visibility: `/health` counts terminal failures per cause, so
`actions_blocked_creator_not_connected:20` is distinguishable from
`model_unavailable`.

## 14. In-memory state remaining

| Structure | Bound | Notes |
|---|---|---|
| `_processed_messages` | 5,000 (FIFO, tunable) | Was: cleared wholesale at 1,000 |
| `_active_chat_binding_retry_after` | Fans in backoff *at once* | Was: unbounded |
| `_vault_sync_retry_after` | Active creators | Pruned on reconcile |
| `_chat_reconcile_due_at` | Active creators | Already pruned |
| `_vault_sync_state` | Creators | Bounded by creator count |
| `_chat_last_message_ids` | **removed** | Superseded by API-001 |

Verified bounded at 10,000 messages and 10,000 fans in
`tests/test_bounded_state.py`. None is a correctness boundary — message identity
is a unique index, purchase identity is a unique index — so losing all of it on
restart costs a little duplicate work and never correctness.

## 15. Fansly Lists single-flight

Per-creator atomic claim (`claim_fansly_lists_sync`), same shape as
`claim_chat_reconciliation`. A second caller gets `{"status": "already_syncing"}`
rather than starting a second reconciliation of the same membership.

Per creator, not global: different creators share no state, and serialising them
would make one agency's large refresh block everyone else's for no correctness
benefit.

The claim is a timestamp with a 15-minute stale window, so a crashed process
cannot wedge a creator; a long legitimate sync is not reclaimed underneath
itself. Released on **failure** as well as success, or one transient API error
would block synchronisation for the whole window.

The manual route still runs to completion before responding — "synced" means the
reconciliation finished, not that a background task was created that a restart
could discard.

## 16. Vault interruption behaviour

| Situation | Before | After |
|---|---|---|
| Never synced | idle | idle |
| Running, this process | running | running |
| **Interrupted by restart** | **idle** | **interrupted, recoverable** |
| Completed, then restart | idle | idle |
| Failed, then restart | idle | idle |

Recovery is unchanged and was already correct: an interrupted sync never stamps
`last_vault_sync_at`, so the cooldown never starts and the scheduler picks it up.
Deliberately not reported as `failed` — a run cut off by a deploy has not failed,
and saying so invites intervention where nothing is wrong. Nothing for an
operator to clear.

## 17. production_preflight output and usage

```
SUPABASE_DB_URL='postgresql://...' python scripts/production_preflight.py
```

Against a fully migrated schema: **25 passed, 0 failed, 1 warning** (the warning
is DB-000). Against a schema with the migrations missing: **16 failed**, each
naming the file to apply.

It never writes: every statement is a catalog read inside a rolled-back
read-only transaction, and a test asserts both that nothing changes and that the
whole run completes with `default_transaction_read_only = on`.

It refuses to let the CI fixtures be mistaken for migrations — the first check
needs no database and now runs in CI.

Environment checks report presence and shape only. A test sets every secret to a
known value and asserts it appears nowhere in the output, not even truncated.

## 18. Model cache report

`scripts/model_cache_report.py --window {1h,24h,7d} --by {provider,model,upstream,feature}`

**Result today: NO DATA.** There is no production traffic in this environment,
so no provider has reported a cache read. That is the honest answer and a better
one than a theoretical cacheability figure presented as a measurement.

Verified against synthetic rows: it reports calls, input tokens, cached tokens,
cache-read percentage, cost, p50/p95 latency and failures, grouped four ways.
The denominator is fresh input **plus** cache reads — dividing by input alone
reports above 100% on a good hit.

Run it after the first week of real traffic. Do not touch prompts again until it
shows something.

## 19. Test, lint and build results

| Gate | Before | After |
|---|---|---|
| Backend pytest | 812 | **951** |
| Backend Ruff (fatal subset) | pass | pass |
| Backend compileall | *(not gated)* | pass, now gated |
| Backend preflight fixture guard | *(did not exist)* | pass, now gated |
| Real-Postgres schema pipeline | pass | pass |
| Dashboard vitest | 79 | **101** |
| Dashboard `tsc --noEmit` | clean | clean, **now gated** |
| Dashboard eslint | 0 errors, 107 warnings | **0 errors, 91 warnings** |
| Dashboard build | pass | pass |

No test was weakened. `test_tenant_isolation_runs_last` was replaced by a
**stronger** assertion (the pair must be last, in order).

Remaining 91 dashboard warnings: 63 `no-explicit-any`, 17 `no-img-element`,
9 `exhaustive-deps` (the intentional "key on `fan?.id`" pattern), 2
`set-state-in-effect`. `--max-warnings 0` is deliberately not enforced — it
would fail every unrelated change without catching anything.

## 20. Load-test methodology

Two harnesses, both with stubbed providers and configurable latency. Neither
touches production OpenRouter, production API Fansly, or a live account. The
code under test is the **real** worker cycle, the real model gate, the real
checkpoint decision and the real retry classification.

- `scripts/load_test_scheduled_actions.py` (Sprint 3) — worker drain.
- `scripts/load_test_sprint4.py` (new) — inbound absorption, restart recovery,
  failure injection.

The absorption model performs the four sequential round trips `main.py` actually
performs (counted from the code: creator resolve, fan resolve, message upsert,
obligation upsert) and gates them on the real DB thread-pool size. Without that
gate an earlier version reported 19,000 messages/second, which was measuring
asyncio rather than the product.

**Intentional human delay is separated from architecture delay throughout.** A
reply arriving ~40 s after a fan messages is the product working. `queue_wait_ms`
is the capacity signal.

## 21. Load-test results

### Full Auto drain (concurrency 8, model latency 2 s, 2 model calls/action)

| Actions | Drain | Actions/min | p50 | p95 | Dupes | Errors |
|---|---|---|---|---|---|---|
| 10 | 9.4 s | 64.1 | 4.66 s | 4.66 s | 0 | 0 |
| 50 | 33.3 s | 90.2 | 4.66 s | 4.66 s | 0 | 0 |
| 100 | 61.8 s | 97.0 | 4.66 s | 4.66 s | 0 | 0 |
| 250 | 152.3 s | 98.5 | 4.66 s | 4.67 s | 0 | 0 |
| 500 | 299.8 s | 100.1 | 4.66 s | 4.66 s | 0 | 0 |

Converges to **~100 actions/min**, matching theory (8 slots ÷ 4.7 s). Max action
concurrency 8, max model concurrency 8 — both gates hold exactly. Zero duplicate
sends, zero same-fan overlaps, 30 DB calls per action.

### Inbound absorption

At 25 ms DB latency (realistic Supabase-from-Railway):

| Burst | Absorbed/s | Accept p50 | p95 | p99 | Dup rows | Dup obligations |
|---|---|---|---|---|---|---|
| 10 in 1 s | 10.0 | 102 ms | 102 ms | 102 ms | 0 | 0 |
| 50 in 5 s | 10.0 | 102 ms | 102 ms | 102 ms | 0 | 0 |
| 100 in 10 s | 10.0 | 102 ms | 105 ms | 148 ms | 0 | 0 |
| 500 in 60 s | 8.3 | 102 ms | 102 ms | 104 ms | 0 | 0 |
| 100 unpaced | 298 | 284 ms | 335 ms | 335 ms | 0 | 0 |
| 500 unpaced | 306 | 1,422 ms | 1,606 ms | 1,631 ms | 0 | 0 |

Saturation scales exactly as `32 / (4 × latency)`: **~1,700/s at 4 ms, ~300/s at
25 ms, ~155/s at 50 ms**. Zero duplicate rows and zero duplicate obligations at
every size, after redelivering 10% of each burst.

### Restart recovery (100 conversations, 50 pending actions)

| Measure | Result |
|---|---|
| Message-list calls after restart, unchanged chats | **0** |
| ... when one chat moved | 1 |
| ... when one chat is new | 1 |
| Durable actions recovered | 50 / 50 |
| Obligations lost | 0 |
| Inbound messages lost | 0 |
| Duplicate sends | 0 |
| Vault state | `interrupted` |
| 20-creator resync cost | 2,000 → **0** |

### Failure injection

Eleven faults through the real `_resolve_action`. **All bounded**, correctly
classified, zero duplicate sends, no unbounded task creation:

OpenRouter 429/503, Anthropic timeout, API Fansly 429/503, ambiguous send,
Supabase latency spike, database failure, worker exception → transient, budget 8.
Creator disconnected → permanent, budget **1**. Writer produced nothing →
writer-quality, budget 2.

## 22. Current safe operating envelope

Every value labelled. **MEASURED** = observed in a harness this sprint.
**CALCULATED** = derived from a measured constant. **INFERRED** = reasoned from
code. **UNKNOWN** = cannot be answered without production data.

| Metric | Safe estimate | Evidence | First failure mode | Alert threshold |
|---|---|---|---|---|
| Concurrent dashboard operators | 25 | INFERRED | Realtime event fan-out per creator | p95 page load > 3 s |
| Concurrent active fan conversations | 250 | MEASURED (drain) | Queue age grows | oldest pending > 300 s |
| Inbound messages/sec (sustained) | 50 | CALCULATED (32 threads ÷ 4 trips ÷ 25 ms, 60% headroom) | DB thread pool saturates | `db_executor.queued` > 0 for 60 s |
| Inbound messages/sec (burst) | 300 | MEASURED | Accept latency → 1.4 s | accept p95 > 1 s |
| Inbound messages/min | 3,000 | CALCULATED | as above | — |
| Full Auto replies/min | **100** | MEASURED | Queue grows unboundedly | pending > 500 |
| Concurrent model calls | 8 | MEASURED (`MODEL_MAX_CONCURRENCY`) | Gate saturates, queue grows | gate waiting ≥ limit for 60 s |
| Scheduled-action queue depth | 500 | MEASURED | 5-min drain tail | `HEALTH_QUEUE_MAX_DEPTH` (500) |
| Scheduled-action drain rate | 100/min | MEASURED | — | — |
| DB ops/sec | 1,280 | CALCULATED (32 ÷ 25 ms) | Thread pool, then pooler | `db_executor.queued` > 0 |
| DB ops per fan message | 4 ingest + ~30 process | MEASURED | — | — |
| API Fansly calls per fan message | 1–2 | INFERRED | — | — |
| Creators per deployment | **20** | MEASURED (restart model) | Reconcile pass duration | reconcile pass > 5 min |
| Agencies per deployment | 5–10 | INFERRED | RLS `chatter_creators` evaluation | — |
| Fans per creator | 2,000 | MEASURED (sync_chats scale) | Chat-list pagination | sync duration > 10 min |
| Conversation length | 50 messages loaded | INFERRED | `(fan_id, sent_at)` index — **UNVERIFIED** (DB-000) | p95 conversation load > 2 s |
| Vault media per creator | UNKNOWN | — | Vault gate queues (by design) | — |
| Concurrent vault syncs | 2 | MEASURED (`VAULT_SYNC_MAX_CONCURRENCY`) | Gate queues — intended | — |
| API Fansly calls/hour | ~1,200 at 20 creators | CALCULATED (20 creators × 2,000 chats ÷ 10-min active interval, with API-001) | Provider rate limit | 429 rate > 1% |
| **Restart cost** | **0 extra provider calls** | MEASURED | — | Burst of `list_chat_messages` after deploy |

The binding constraint at 10–20 creators is **Full Auto replies/min (100)**, not
ingestion. Ingestion absorbs roughly 30× what the worker drains — which is
correct, because the queue is durable and the fan sees the drain rate.

## 23. Sign-off

### GO WITH CONDITIONS

For a 10–20 creator production beta.

Every remaining item is a deployment step, not engineering work. There is no
security hole, no data-corruption path, no duplicate-send path, no message-loss
path, and no capacity wall below the beta target that I know of and have not
fixed.

What changed the answer from the audit's position:

- an operator can no longer bypass backend business rules from the browser
  (SEC-001, demonstrated);
- two concurrent purchase webhooks can no longer double-count a sale (REL-003,
  demonstrated on real PostgreSQL);
- a deploy no longer costs 40,000 provider calls (API-001, measured);
- a broken creator binding no longer burns 160 model pipelines to discover
  (REL-005, measured);
- eleven injected failures are all bounded with zero duplicate sends.

## 24. Exact user actions required before the first real agency

1. **Apply the five migrations** (§26), in order, in the Supabase SQL editor.
2. **Re-run `tenant_isolation_v1` then `browser_least_privilege_v1` as a pair.**
   Running the first alone silently reopens SEC-001.
3. **Run the preflight** (§17). Every check PASS except the DB-000 warning.
4. **Set the environment variables** in §25.
5. **Run the one-creator smoke test**: `docs/production_smoke_checklist.md`.
6. **Watch the first deploy after step 1**: no burst of `list_chat_messages`.
7. *(Recommended, unblocks DB-000)* run `scripts/dump_base_schema.sh`.

Do **not** enable `FANSLY_LISTS_SYNC_ENABLED` before `fansly_lists_v1` and
`fansly_lists_single_flight_v1` are applied. The preflight cross-checks this.

## 25. Railway environment variables

**Required** (deployment fails closed or misbehaves without them):

```
APP_ENV=production
SUPABASE_URL, SUPABASE_SERVICE_KEY
APIFANSLY_API_KEY, APIFANSLY_BASE_URL
FANSLY_SESSION_KEY
DASHBOARD_API_SECRET
APIFANSLY_WEBHOOK_SECRET   (or WEBHOOK_SECRET)
CORS_ALLOW_ORIGINS=https://<your-dashboard-domain>
```

**Provider keys** — whichever the configured routes use:

```
OPENROUTER_API_KEY          writer (Kimi via OpenRouter)
ANTHROPIC_API_KEY           analyzer, if ANALYZER_PROVIDER=anthropic
TOGETHER_API_KEY            only if a Together route is configured
```

**New in Sprint 4** (optional, sensible defaults):

```
DB_EXECUTOR_MAX_WORKERS=32          ceiling on concurrent Supabase calls
PROCESSED_MESSAGE_CACHE_SIZE=5000   dedupe window
FANSLY_LISTS_SYNC_STALE_MINUTES=15  list single-flight reclaim window
```

**Recommended for beta**:

```
SCHEDULED_ACTION_CONCURRENCY=8
MODEL_MAX_CONCURRENCY=8
HEALTH_QUEUE_MAX_AGE_SECONDS=300
HEALTH_QUEUE_MAX_DEPTH=500
```

Dashboard: `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY`,
`NEXT_PUBLIC_API_URL`, `NEXT_PUBLIC_API_KEY`.

## 26. Supabase SQL files to apply

In this order:

```
1. db/chat_sync_checkpoint_v1.sql
2. db/purchase_identity_v1.sql
3. db/vault_sync_interruption_v1.sql
4. db/fansly_lists_single_flight_v1.sql
5. db/tenant_isolation_v1.sql            <- re-run
6. db/browser_least_privilege_v1.sql     <- MUST follow 5
```

Steps 5 and 6 are a pair. `tenant_isolation_v1` drops every policy on each table
it discovers before creating its own `FOR ALL` policy, so running it without
step 6 leaves every creator-owned table fully writable from the browser.

**NEVER apply** `db/ci_baseline_schema.sql` or `db/ci_supabase_stubs.sql`. They
are CI fixtures. The preflight and CI both now assert they are absent from
`migration_order.txt`.

If a migration reports pre-existing conflicting data, it stops and reports
rather than deleting history. Investigate before proceeding.

## 27. One-creator smoke checklist

`docs/production_smoke_checklist.md` — 21 steps plus a read-only preflight,
covering login, tenant isolation, webhook acceptance, single persistence, single
obligation, Assisted, Full Auto, analyzer, writer, upstream verification, single
delivery, duplicate-webhook harmlessness, transient API failure, analyzer
failure sending nothing, Auto-off cancellation, list refresh with a deliberate
double-click, vault summary, and health.

PPV purchase is marked optional and test-account-only.

## 28. Rollback strategy

**Code**: redeploy the previous Railway image. Every Sprint 4 feature degrades
to its previous behaviour when its migration is absent, and every migration is
tolerated by the previous code (all additive). A rolling deploy is safe in
either order, and rolling back code without rolling back schema is safe.

**Schema**: do **not** roll back. All five migrations are additive — new columns,
new tables, new functions. Rolling them back would destroy data
(`platform_purchase_events`) and reopen SEC-001. If a migration must be undone,
undo the smallest thing: e.g. `alter table fans drop column chat_last_message_id`
returns API-001 to its old behaviour and nothing else.

**SEC-001 specifically**: re-running `tenant_isolation_v1` alone restores the
previous permissive policies. That is the rollback, and it is also the accident
to avoid.

**Per-feature kill switches, no deploy needed**:

| Feature | Disable |
|---|---|
| Fansly Lists sync | `FANSLY_LISTS_SYNC_ENABLED=false` |
| Auto replies | Creator-level Auto toggle |
| Worker throughput | `SCHEDULED_ACTION_CONCURRENCY=1` |
| Model spend | `MODEL_MAX_CONCURRENCY=1` |
| Vault sync | `VAULT_SYNC_INTERVAL_HOURS=8760` |

## 29. Technical debt that is NOT a beta blocker

1. **DB-000** — the authoritative schema is still not in version control. One
   command from the user. Preflight covers the operational half.
2. **Unverified indexes** — `messages(fan_id, sent_at desc)` and friends. At
   2,000 fans/creator this is unlikely to bite; at 10,000 it might.
3. **`main.py` is 231 KB.** Explicitly out of scope. It is a comprehension
   problem, not a correctness one.
4. **63 `no-explicit-any` + 17 `no-img-element`** dashboard warnings.
5. **2 `set-state-in-effect` warnings** in the message-loading path. State
   converges; fixing them means restructuring conversation loading.
6. **9 `exhaustive-deps` warnings** — the intentional `fan?.id` pattern. Acting
   on them would be the regression.
7. **`_vault_sync_state` is still per-process.** Interruption is now visible;
   live progress across processes is not, and does not need to be.
8. **Purchase claim release depends on the process surviving.** A kill between
   claim and release leaves a `claimed` row that blocks that one order's
   redelivery. Visible in `platform_purchase_events_status_claimed_idx`; a
   sweeper is a follow-up, not a blocker.
9. **No legacy code was removed.** Everything checked (`services/fansly_client`,
   `fansly_poller`, every service module) is referenced and reachable. Nothing
   met the removal bar, so nothing was removed.

## 30. When the web/worker/scheduler/media split becomes necessary

Not now, and not at 20 creators. Recalculated against **current** code:

| Trigger | Threshold | Why |
|---|---|---|
| **Full Auto replies/min sustained > 100** | ~50 creators with 30% Auto | The measured drain ceiling. First real wall. |
| Scheduled-action queue age > 300 s consistently | ~50–75 creators | Worker cannot keep up; a separate worker process is the fix |
| `db_executor.queued` persistently > 0 at 32 threads | ~100 creators | Synchronous Supabase client on a thread pool; asyncpg or a worker split |
| Event-loop lag > 100 ms | ~75–100 creators | Web and background work sharing one loop |
| Chat reconcile pass > 5 min | ~100 creators (200,000 chats) | Needs a scheduler process with per-creator sharding |
| Vault syncs queued > 1 h | ~30 creators with large vaults | Media workers |
| One agency starving another | 3+ agencies with very uneven size | Per-agency queue fairness |

**Approximate order:** worker split first (~50 creators), then scheduler/leader
(~75), then media workers (~100, sooner with big vaults), then per-agency
fairness (only when a large agency starts starving a small one).

Redis is not needed for any of these. Postgres already provides the claim
semantics a multi-process split requires — `claim_due_actions` uses
`FOR UPDATE SKIP LOCKED` and is already correct for multiple workers.

---

## Stop condition

This is the last audit-driven engineering sprint before beta. The next phase is
agency outreach, a controlled production beta, and fixing only what real usage
proves is broken.

Run `scripts/model_cache_report.py` after the first week of real traffic. Do not
touch prompts again until it shows something.
