# Sprint 3 — latency, cost, API efficiency, database efficiency, dashboard performance, vault load control

Reference audit: `docs/ENGINEERING_AUDIT_2026-09.md`, performed against an older
snapshot. Every finding below was re-verified against current `main` before any
change; none of Parts 1–15 had been solved by Sprint 1 or Sprint 2.

Starting SHAs:

| Repository | SHA |
|---|---|
| `Jericho2k/cleopatra_ai` | `dbd87fcd925f114b57e7d4f3221451557c566d56` |
| `Jericho2k/cleopatra-dashboard` | `19c00598af2b183ee5439001f43d78629a38ee25` |

---

## What this sprint did not touch

Full Auto concurrency and backpressure, durable webhook ACK semantics, analyzer
fail-closed behaviour, message idempotency, Fansly list reconciliation, Kimi
K2.6 OpenRouter routing and pinned upstreams, prompt session affinity, the
DeepSeek complex/commercial route, human-like composition delays, PPV
single-flight, post-send reconciliation, creator tenancy, Auto Audience
semantics, and vault categorisation result semantics are all unchanged. The
changes here remove work; they do not move a decision.

---

## Measurement harnesses

All four are runnable and are the source of every number in this document.

| Harness | What it measures | Kind |
|---|---|---|
| `scripts/measure_prompt_cache.py` | longest identical prefix between two consecutive turns | exact character comparison; token figures are chars/4 estimates |
| `scripts/bench_apifansly_pool.py` | pooled vs per-call client against a local TLS server | measured, loopback only |
| `scripts/bench_vault_categorization.py` | worker pool vs batch barrier | modelled durations, exact write counts |
| `cleopatra-dashboard/scripts/bench-dedupe.mjs` | dedupe cost by conversation length | measured |
| `cleopatra-dashboard/scripts/bench-sidebar-filter.mjs` | conversation filter cost | measured |
| `cleopatra-dashboard/scripts/bench-vault-loading.mjs` | rows, requests and JSON for the vault page | modelled from the selected columns |
| `cleopatra-dashboard/scripts/bench-apifansly-calls.mjs` | provider calls per hour per open tab | modelled; intervals exact, operator behaviour estimated |

### Theoretical prefix reuse vs actual provider cache reads

`measure_prompt_cache.py` reports **theoretical prefix reuse only**. It renders
two turns of one synthetic conversation and measures the longest byte-identical
prefix. It is not evidence that a provider cached anything.

**Actual provider cache reads** are recorded per call in
`model_usage_events.cache_read_tokens` and `metadata.cache_hit_ratio` by
`services/model_telemetry.py`. That instrumentation is unchanged and is now
covered by a test. Query that table before acting on any cache number.

One caveat worth stating plainly: Anthropic will not cache a block below its
per-model minimum — 1,024 tokens for Sonnet and Opus, 2,048 for Haiku. The
analyzer's system block is ~960 estimated tokens, so on the documented default
(`ANALYZER_PROVIDER=anthropic`, Haiku) Anthropic's *explicit* cache will not
engage for it even now. `cacheable_system_blocks` therefore declines to mark a
block that short rather than sending a marker that cannot do anything. The
analyzer's gain is real but it comes from implicit prefix caching on
OpenAI-compatible providers and from the prompt now having a stable prefix at
all.

---

## Prompt caching (COST-002)

| Prompt | System tokens | Total tokens | Shared prefix | Reusable | |
|---|---|---|---|---|---|
| writer, flags off | 1600 → 1600 | 2856 → 2858 | 2032 → 2032 | 71.2% → 71.1% | unchanged, deliberately |
| writer, intelligence flags on | 1600 → 1600 | 3420 → 3422 | 1668 → 1878 | **48.8% → 54.9%** | |
| analyzer | 15 → 960 | 1066 → 1066 | 46 → 966 | **4.4% → 90.7%** | |

The enriched writer baseline of 48.8% reproduces the audit's measured 49%, so
the harness agrees with the audit before the change.

The flags-off writer case is deliberately flat. Expression calibration is
constant when no director or session strategy is configured — today's default —
so it stays in front of the transcript there and moves behind it only once it
actually varies. Moving it unconditionally would have cost the default
deployment ~700 characters of prefix to help a configuration that is switched
off.

The analyzer's system block is now byte-identical across unrelated
conversations, so it is reusable deployment-wide rather than only between
consecutive turns of one chat.

---

## API Fansly connection reuse (PERF-006)

Architecture:

- one `httpx.AsyncClient` per process, created lazily by
  `services.apifansly.shared_client()`;
- explicit `Limits(max_connections=64, max_keepalive_connections=32,
  keepalive_expiry=60s)`;
- `follow_redirects=False`, matching the per-call clients it replaces, so a
  redirect cannot carry `x-api-key` to another host. `request()` takes an
  explicit override and otherwise passes `httpx.USE_CLIENT_DEFAULT`, so an
  injected client keeps its own policy;
- per-request timeouts unchanged (`send_message` still pins 15 s);
- `set_shared_client()` installs a test transport; `close_shared_client()` runs
  once from `lifespan` shutdown and the pool rebuilds if used again;
- authentication is a per-request header built from one deployment-wide key, so
  nothing per-creator lives on the client. A test drives two creators through
  one pool concurrently and asserts both requests are correctly addressed and
  authenticated.

Converted: eleven call sites in `main.py` (through an `apifansly_client_scope()`
context manager, so no block changed shape), Fansly list reconciliation, the
audience sync, and the typing indicator in `services/suggestions.py` — that last
one sits inside the human-like composition delay on the live reply path, where a
handshake was pure latency before the fan sees anything.

Left alone: the vault album-media loop, the classifier image fetcher and the
categorisation client. Those are per-batch clients already amortised over many
requests with tuned limits, not the per-call pattern the audit found.

Measured, `scripts/bench_apifansly_pool.py`, 200 requests against a local TLS
server: **4.84 ms/request pooled vs 5.67 ms/request per-call — 0.83 ms saved**.
That is loopback, so it excludes network round-trip time and is the **floor**:
each avoided handshake also costs two extra round trips on a real link. No
internet-level saving is claimed because none was measured.

---

## API Fansly failure semantics (Part 15)

`raise_for_response` now raises `ApiFanslyTransientError` for
408/425/429/500/502/503/504, carrying the status and any `Retry-After`. Three
facts that were previously indistinguishable are now distinct:

| Fact | Class | Behaviour |
|---|---|---|
| the server says it did not process this | `ApiFanslyTransientError` | retried for GET/HEAD; on a send, the claim is released without freezing the fan |
| the server may have processed it and we lost the answer | `httpx.TimeoutException` etc. | never retried on a write; a send still freezes the fan for review |
| this key will fail identically forever | `ApiFanslyAccountAccessError` | never retried |

Retry is decided by the **HTTP method**, not by how transient a failure looked:
`GET`/`HEAD` up to three attempts with jittered backoff honouring `Retry-After`,
everything else exactly once. Exactly-once send caution is unchanged — an
ambiguous send goes through the delivery journal, not through a retry.

---

## API Fansly calls per hour from the dashboard (API-002)

| Scenario | Before | After |
|---|---|---|
| active conversation, one tab | 80/hour | 23/hour |
| idle conversation, one tab | 20/hour | 4/hour |
| background (hidden) tab | 20/hour | 0/hour |
| 25 active operator tabs | 2,000/hour | 575/hour |

The 45-second heartbeat is gone. What remains on an active tab is mostly the
operator opening conversations, which is the "did we miss anything" check they
actually want. Reconciliation now runs on: conversation opened, realtime
reconnect, tab visibility restored, a backend-reported pending chat binding, and
a 15-minute safety interval — with a per-fan floor of one call per minute so
several reasons firing together still cost one call.

Behavioural assumptions (12 conversations opened/hour, 1 reconnect, 6 refocuses)
are estimates. The intervals and the rate limit are exact.

---

## Conversation dedupe (FE-001)

Cost of appending one realtime message (the realtime handler runs it **twice**
per message):

| Messages | Before | After |
|---|---|---|
| 50 | 0.38 ms | 0.20 ms |
| 200 | 3.93 ms | 0.52 ms |
| 500 | 27.0 ms | 1.00 ms |
| 1,000 | 118.2 ms | 2.01 ms |
| 2,000 | 428.7 ms | 4.32 ms |
| 5,000 | 2,826 ms | 13.81 ms |
| 10,000 | 12,024 ms | 33.96 ms |

Quadratic growth is gone: 20× the messages costs ~17×, not ~300×.

Duplicate semantics were pinned first, in tests written against the **original**
implementation, and pass unchanged against the replacement.

Retained history is bounded at 1,000 messages, applied only where messages
arrive unasked (realtime insert, reconnect catch-up). "Load more" is the
operator deliberately reading history and is never trimmed; a thread they have
scrolled back into is recorded and left alone until it is reloaded. Nothing is
deleted or made unreachable.

---

## Vault page data loading (FE-002, FE-003)

Opening the page:

| Vault size | Rows fetched (before → after) | Requests | JSON |
|---|---|---|---|
| 100 | 100 → 1 | 1 → 1 | 0.14 MB → ~0 |
| 1,000 | 1,000 → 4 | 1 → 1 | 1.43 MB → ~0 |
| 10,000 | 10,000 → 40 | 10 → 1 | 14.31 MB → ~0 |
| 50,000 | 50,000 → 40 | 50 → 1 | 71.53 MB → ~0 |

"After" rows on open are album summary rows, not media. Opening an album then
costs **one request and 200 rows at every vault size**, and the retained set is
that page rather than the vault.

Images for one album view: **~600 MB of originals before, ~5 MB of thumbnails
after**.

One realtime row change: the whole vault was refetched before; one row is
patched in place now, and `patchLoadedRow` returns the same array reference when
the row is not on the current page, so React skips the render. Inserts and
deletes stay debounced and reload the counts plus the open album only.

---

## Sidebar (FE-004)

Filter cost per render:

| Conversations | List members | Before | After |
|---|---|---|---|
| 200 | 100 | 0.042 ms | 0.018 ms |
| 1,000 | 500 | 0.960 ms | 0.045 ms |
| 5,000 | 2,500 | 19.639 ms | 0.369 ms |

It also runs far less often: memoised, hover moved to CSS, and the minute tick
re-renders only the `<RelativeTime>` spans.

**Virtualisation was not added.** The audit's own threshold was "when operators
are routinely working past ~1,000 rendered rows", and 1,000 conversations now
filter in 0.045 ms. Adding a virtualisation dependency before that is real would
be cost without a reason.

---

## Vault processing (VAULT-001, VAULT-002)

Simulated with mocked classifiers and modelled DB latency, one video in five,
concurrency 12:

| Items | Before | After | Speed-up | Writes before | Writes after |
|---|---|---|---|---|---|
| 100 | 345.7 s | 93.9 s | 3.68× | 100 | 1 |
| 1,000 | 3,244.1 s | 781.6 s | 4.15× | 1,000 | 10 |

Durations are modelled. **The write counts and the concurrency ceiling are
exact**, and the ceiling is unchanged: exactly
`VAULT_CATEGORIZATION_CONCURRENCY` workers exist, so the same budget is spent
with no idle slots.

**Max concurrent creator-level vault syncs: 2** (`VAULT_SYNC_MAX_CONCURRENCY`,
clamped to 1–32). A queued creator is a live task parked on a semaphore —
nothing is dropped, no due job is lost, and the slot is released on success,
exception and cancellation alike. The categorisation run started from inside a
vault sync deliberately does not take a second slot; at a limit of 1 that would
deadlock.

Telemetry: `VAULT_GATE.snapshot()` (limit, active, waiting, max active, started
total, average and max wait, oldest active, oldest waiting) appears under
`vault.gate` in the Sprint 2 health document and in the per-creator vault
overview. It is informational and never part of the health verdict — a queue
here is the gate working as intended.

---

## Database round trips

| Path | Before | After |
|---|---|---|
| `sync_chats`, 200 chats, nothing changed | 200 `select("*")` + 200 UPDATEs | **1 paginated read, 0 writes** |
| `sync_chats`, 200 chats, one display name changed | 200 + 200 | 1 read, **1 write** |
| `claim_due_actions(20)` | 2 selects + 20 CAS UPDATEs = 22 | **1** |
| vault categorisation, 1,000 items | 1,000 sequential UPDATEs | **10 batched upserts** |
| vault categorisation, 10,000 items | 10,000 | **~100** |
| vault page open, 10,000 items | 10 paged selects of 26 columns | **1 aggregate** |

---

## Remaining >1,000-row completeness risks

Every unbounded select on a large table in current `main` was classified:

**Fixed (correctness depended on completeness):** `get_sent_ppv`,
`mark_ppv_purchased`, `sent_set_ids` in `get_offerable_packages`,
`sweep_stale_ppv_checks`, `sync_fansly_audience`, `sync_chats`'s known-platform
ids, `load_fan_history`'s already-imported set, `delete_creator`'s fan list, and
the three creator health reads in `services/full_auto_operations.py`.

**Fixed (belonged as an aggregate):** the chat reconciliation scheduler's
deployment-wide read of every auto-mode fan, now a bounded existence check per
due creator.

**Accepted as safe:** single-row reads by unique filter (`get_fan_state`,
count/head reads); naturally bounded domains (chunked `in_()` lookups, one fan's
facts, one fan's list memberships, id lists from a single API page); and
display-only reads where a 1,000-row cap is acceptable — a creator's approved
vault sets (`get_offerable_packages`, `_load_approved_sets`), a creator's manual
PPV offers, and one fan's delivery ledger.

**Residual risk:** a creator with **more than 1,000 approved vault sets** would
have package selection drawn from the first 1,000. That degrades which offer is
chosen; it does not re-offer purchased content, because `sent_set_ids` is now
complete. No creator is near that number, and paginating it would pull the whole
approved catalogue into memory on every offer build, so it was left as a
conscious trade rather than fixed blindly.

---

## `claim_due_actions` — decision

**Changed, because the new form is safer, not merely cheaper.**

The client-side compare-and-swap was correct: it re-asserted the observed status
and, for a stale reclaim, the observed `locked_at`. What it could not do is
avoid the read-then-write window, and it paid a round trip per row to narrow it.

`db/scheduled_action_claim_v1.sql` does the selection and the update in one
statement with `FOR UPDATE SKIP LOCKED`, so two workers never select the same
row — the race the CAS existed to lose is not entered at all. Selection and
UPDATE commit together, so there is no window where a row is chosen but not yet
owned.

Preserved: PENDING claimable once `execute_at` has passed; PROCESSING
reclaimable once `locked_at` is older than the stale window; `locked_at`
re-stamped at claim time; the limit applied to each of the two sets, so one call
can still return up to `2 × limit` rows.

The guarantees are tested against real PostgreSQL, including two concurrent
connections racing for one row.

**Rollout:** there is no migration runner, so the code can reach production
before the function. `claim_due_actions` falls back to the previous per-row CAS
in exactly one case — the function is absent — and sets a process flag so a
rolling deploy pays one failed RPC rather than one per poll. Any other error
surfaces; silently downgrading the claim path on a deadlock would hide a real
problem behind a slower query.

---

## Configuration

One new environment variable:

```
VAULT_SYNC_MAX_CONCURRENCY=2
```

How many creators may sync or categorise their vault at once, process-wide.
Separate from, and multiplied by, `VAULT_CATEGORIZATION_CONCURRENCY`. Clamped to
1–32; an unset or unparseable value resolves to 2.

The classification write batch (100 rows) and flush interval (5 s) are internal
constants, not environment variables. An env var per batch size is configuration
sprawl, not tuning.

---

## Migrations

Two, both additive and idempotent, both listed in `db/migration_order.txt`
before `tenant_isolation_v1.sql`:

| File | What it adds | Required before |
|---|---|---|
| `db/scheduled_action_claim_v1.sql` | `claim_due_actions(int, int)` plus two partial indexes | backend benefits; backend falls back without it |
| `db/vault_album_summary_v1.sql` | `vault_album_summary(uuid)` (SECURITY INVOKER) plus `(creator_id, album_title, id)` index | dashboard benefits; dashboard falls back without it |

`vault_album_summary` is SECURITY **INVOKER** on purpose: it runs with the
caller's own privileges, so the row-level security already on
`creator_vault_media` applies exactly as it did to the direct select it
replaces. It grants the browser nothing it did not already have and deliberately
does not paper over SEC-001.

`db/ci_baseline_schema.sql` gains `fans.avatar_url` and
`creator_vault_media.album_title`. Both columns exist only in the live project
and are written or read by shipping code, so without them the CI fixture could
not host migrations the product depends on. This does not resolve DB-000.

---

## Deployment order

1. **Apply both migrations in Supabase.** Neither is required for a deploy to
   succeed — both callers fall back — but applying first means no fallback is
   ever exercised.
2. **Deploy the backend.** Set `VAULT_SYNC_MAX_CONCURRENCY=2` (or leave it
   unset; the default is the same). Watch the boot line for the resolved
   environment and `[VAULT GATE]` lines for queueing.
3. **Deploy the dashboard.** It is independent of the backend deploy: nothing in
   it requires a backend change, and `POST /sync-fan-messages` is unchanged —
   the dashboard simply calls it far less.

Order 2 and 3 can be swapped. Order 1 can happen at any point.

## Rollback

- **Dashboard:** revert the deploy. It holds no state and no migration.
- **Backend:** revert the deploy. Both new functions are additive; leaving them
  in the database is harmless, because the reverted code never calls them.
- **Migrations:** no rollback needed. Neither drops or alters an existing
  column, and both are `create or replace` plus `create index if not exists`.
  If you must, `drop function public.claim_due_actions(integer, integer)` and
  `drop function public.vault_album_summary(uuid)` after the code that calls
  them is gone.
- **Partial rollback:** `VAULT_SYNC_MAX_CONCURRENCY` can be raised at runtime to
  restore the old fan-out without a deploy. The gate applies a changed limit as
  soon as it is idle.

There is no data migration in this sprint and no destructive statement, so
rolling backwards does not lose anything.
