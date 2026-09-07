# Cleopatra AI — Engineering Audit

**Audited commits (fetched from `main` at audit time, not from any prior document):**

| Repository | SHA | Date | Head commit |
|---|---|---|---|
| `Jericho2k/cleopatra_ai` | `3f4867bcef96e42e2c5fdb0ffe84484c46477f71` | 2026-09-07T19:57:11+03:00 | Add Fansly list mirroring and OpenRouter Kimi K2.6 writer support (#23) |
| `Jericho2k/cleopatra-dashboard` | `5c9a0a6c22d7cdde84b97ed7c22666bdae94cb49` | 2026-09-07T19:57:47+03:00 | Show imported Fansly lists alongside Cleopatra's own (#15) |

Both repositories include the OpenRouter Kimi K2.6 routing/pinning/caching sprint and the
Fansly Lists synchronisation sprint. Both are fully covered below.

**Scale of the audited system:** backend 43,373 lines of Python across 181 files
(`main.py` alone is 5,799 lines), 18 SQL migration files, 69 registered HTTP routes;
dashboard 9,914 lines of TypeScript/TSX across 21 files, 8 pages, 6 components.

**Evidence labels used throughout:**
`MEASURED` = produced by instrumentation run against this exact code during the audit ·
`CALCULATED` = arithmetic on measured components ·
`INFERRED` = read from code with no runtime measurement ·
`UNKNOWN` = cannot be established from the repository alone.

---

## 1. EXECUTIVE SUMMARY

### What worries me most about putting Cleopatra on real agency accounts

**The whole Full Auto product runs through one sequential `for` loop.**
`workers/scheduled_actions.py::process_once` claims up to 20 due actions and then
processes them **one at a time, awaiting each to completion**, including the
`AUTO_REPLY` handler — which is the entire Full Auto reply pipeline: ~43 database
round trips (MEASURED), 2 sequential LLM calls, an API Fansly send, and a
**6.8-second simulated-typing sleep** (MEASURED) that the worker sits through.
Then it `asyncio.sleep(60)` unconditionally, backlog or not. This is not a
per-creator or per-agency limit; it is the ceiling for the entire deployment.
My estimate is **3–6 Full Auto replies per minute across all agencies combined**
(CALCULATED). One agency with 40 active conversations will consume the entire
deployment's Full Auto capacity and delay every other agency's fans behind it.

**Second: the deployment silently truncates at 1,000 rows in at least six places.**
Supabase/PostgREST caps unfiltered selects at 1,000 rows. The codebase clearly knows
this — `_vault_existing_media_ids` paginates with the comment *"Supabase caps ordinary
selects at 1,000"* — but six other hot paths do not. The consequences are not slow
queries, they are **wrong answers and destructive writes**: the Fansly Lists
reconciliation deletes correct list memberships for any creator with more than 1,000
fans (SCALE-003), the auto-audience preview reports the wrong eligible count and
silently loses exclusion rules (SEC-002), the analytics page reports revenue summed
over an arbitrary 1,000 fans, and `sweep_stale_ppv_checks` permanently stops repairing
PPV reconciliation past 1,000 pending rows.

**Third: an LLM outage does not stop Full Auto — it makes it send blind.**
When the situation analyzer fails (transport error, malformed JSON, provider 429),
`ai/situation_analyzer.py::_fallback_result` returns a **plausible-looking neutral
analysis** — `purchase_signal: "none"`, `crisis_signal: "none"`, `wants_media: "false"` —
and the pipeline continues to the writer and sends. A fan saying *"I'll take the $50 one"*
during a provider incident is classified as an ordinary chat turn and answered as one.
There is no degraded-mode gate, no telemetry distinction between "analyzed" and
"guessed", and no operator-visible signal.

**Fourth: the writer has an unbounded, un-backed-off retry that triples cost on a
predictable input.** `ai/generator.py::generate_replies` makes three attempts with
**zero delay between them**, and retries not only on transport failure but on
*validation* failure. Validation rejects any reply containing the substring
`"yourself"`, `"interesting"`, `"noted"`, or `"got it"` — substrings that are extremely
common in this product's domain ("touch yourself", "by yourself"). A rejected batch
costs three full generations and then **returns `[]`, sending nothing at all**.
Combined with `OPENROUTER_ALLOW_FALLBACKS=false` pinned to a single upstream
(`Inceptron`), a provider rate-limit event turns into an immediate 3× request
amplification against the exact provider that is throttling.

### What is genuinely good

This is not a badly built system. The PPV delivery ledger with its partial unique
index, the proactive-message delivery journal with crash reconciliation, the RLS
tenancy migration, the `claim_chat_reconciliation` database-level lock, and the
"platform acceptance is authoritative, freeze rather than duplicate" delivery
semantics are all careful, correct work that most systems at this stage do not have.
Section 20 lists them explicitly. The problems above are almost all **scale
problems in code that is correct at one test account** — which is exactly what this
audit was asked to find.

---

## 2. TOP 10 PROBLEMS

Ranked by expected production impact on real agency accounts.

| # | ID | Problem | Impact |
|---|---|---|---|
| 1 | SCALE-001 | Full Auto replies are processed one at a time in a single sequential loop with an unconditional 60 s sleep | Deployment-wide ceiling of ~3–6 auto replies/min; latency grows linearly with concurrent conversations |
| 2 | SCALE-002 | `repair_followup_obligations` spends 2 DB round trips per outstanding obligation, every 60 s, sequentially | 2,000 round trips/min at 1,000 obligations; worker stops keeping up entirely around ~5,000 |
| 3 | SCALE-003 | Fansly Lists reconciliation reads `fans` unpaginated (1,000-row cap) and then **deletes** memberships it cannot map | Silent, repeating destruction of Auto Audience targeting for any creator with >1,000 fans |
| 4 | REL-001 | Analyzer failure returns a fabricated neutral analysis and Full Auto keeps sending | Wrong commercial decisions during any provider incident, with no signal |
| 5 | FE-001 | `dedupeMessages` is O(n²) and runs twice per incoming realtime message | 667 ms freeze at 1,000 loaded messages, 15 s at 5,000, 63 s at 10,000 (MEASURED) |
| 6 | SEC-002 | `preview_auto_audience` reads the **entire** `fan_list_members` table with no creator filter, under service-role credentials | Cross-tenant read; exclusion rules silently stop working past 1,000 global rows |
| 7 | COST-001 | Writer retries 3× with no backoff, including on validation failure, against a single pinned provider | 3× token cost on a common input; retry amplification into a throttling provider |
| 8 | COST-002 | `cache_control` is built by the prompt builder and then discarded before transport; the analyzer's 1,570 static tokens sit *after* the volatile conversation | Anthropic caching is completely inert (PROVEN); ~1,570 analyzer tokens/message are needlessly uncached |
| 9 | SEC-001 | RLS policies are `for all` to `authenticated` — browsers can INSERT/UPDATE/DELETE every creator-owned table | Any operator can bypass approval gates, PPV single-flight, and commercial state directly from the browser console |
| 10 | FE-002 | Vault grid renders up to 200 **full-resolution originals** as 100×100 thumbnails while `thumbnail_url` sits unused | Hundreds of MB of image downloads per album view |

---

## 3. USER-VISIBLE LATENCY CRITICAL PATH

### 3a. Assisted suggestion (`POST /suggestions`) — MEASURED

Instrumented with a counting Supabase double and a recording model transport, against
this exact code.

```
POST /suggestions
 ├─ middleware: authenticated_dashboard_user      1 Supabase auth call (60 s cached)
 ├─ require_creator_fan_access                    2 DB (chatter_creators, fans)  [per request, uncached]
 ├─ asyncio.gather of 11 reads                    PARALLEL  ← good
 │    history(40) · fan · intelligence · lifecycle · affordability ·
 │    price_learning · persona · legend · ppv_offers · sent_ppv · session
 ├─ classify_stage()                              pure Python
 ├─ analyze_situation()                    ★ LLM #1  (ANALYZER target)
 ├─ refresh_affordability_from_situation()        deterministic + DB
 ├─ refresh_fan_lifecycle()                       deterministic + DB
 ├─ refresh_price_learning()                      deterministic + DB
 ├─ direct_conversation()                         deterministic + DB
 ├─ plan_next_action()                            deterministic + DB
 ├─ select_writer_route()                         pure Python
 ├─ build_prompt()                                pure Python
 ├─ generate_replies()                     ★ LLM #2  (writer, up to 3 attempts)
 ├─ save_message()                                1 DB
 └─ spawn(learn_from_fan_message)          ☆ LLM #3 background (EXTRACTOR)
```

| Configuration | DB round trips | LLM calls on critical path |
|---|---|---|
| Intelligence flags **off** (repo defaults) | **9** MEASURED | **2** MEASURED |
| Intelligence flags **on** | **34** MEASURED | **2** MEASURED (+1 background) |

**The good news is significant and worth stating plainly:** the commercial layer —
affordability, price learning, buyer lifecycle, conversation director, adaptive session
planner, session planner, commercial orchestrator — is **entirely deterministic Python,
not LLM calls** (verified by import analysis across all nine services). There are only
two sequential model calls before an operator sees a suggestion, not the six or seven
the module layout suggests. This is good architecture and it should not be changed.

**Where the time actually goes** (INFERRED from structure; latency not measured against
live providers):

| Stage | Est. contribution |
|---|---|
| Tenancy check (2 uncached DB reads) | 40–80 ms |
| 11-way parallel read (bounded by slowest, `get_ppv_offers` does 2 serial queries) | 60–150 ms |
| **Analyzer LLM** (~1,600 input tokens, `max_tokens=650`, `temperature=0`) | **1,000–2,500 ms** |
| 15 serial DB round trips across the 5 deterministic refreshers (flags on) | 375–750 ms |
| **Writer LLM** (~3,500 input tokens, `max_tokens=1000`) | **2,000–5,000 ms** |
| `save_message` + telemetry inserts | 50–100 ms |
| **Total** | **~3.5–8.5 s** |

**The single biggest structural inefficiency on this path:** the five deterministic
refreshers run **strictly sequentially** and produce 15 DB round trips (MEASURED), even
though `refresh_affordability`, `refresh_fan_lifecycle` and `refresh_price_learning`
have no data dependency on each other in the assisted path. See PERF-004.

### 3b. Full Auto reply — MEASURED

This path has two halves and the second half is the problem.

```
── HALF 1: ingestion (inside the webhook HTTP request) ──────────────────
POST /webhook/fansly  (messages.received)
 ├─ HMAC verify                                   pure
 ├─ creators lookup by apifansly_account_id       1 DB
 ├─ get_fan / create_fan                          1–3 DB
 ├─ messages duplicate check                      1 DB
 ├─ fans.update(group_id)                         1 DB
 ├─ [if attachments] list_chat_messages(limit=10) 1 API FANSLY CALL, in-request
 ├─ save_message                                  2 DB (dedupe read + insert)
 └─ process_incoming_fan_message()
      ├─ cancel_pending_ppv_approvals             1–2 DB
      ├─ acknowledge_fan_return                   1–3 DB
      ├─ get_conversation_history + get_fan_by_id 2 DB (SEQUENTIAL — should be gathered)
      ├─ spawn(learn_from_fan_message)            ☆ background LLM
      ├─ gather(audience_policy, memberships, auto_availability)  3 DB, parallel
      └─ schedule_auto_reply()
           ├─ get_fan_session + get_creator_sleep_hours   2 DB
           ├─ cancel_actions_for_fan("AUTO_REPLY")        1 DB
           └─ schedule_action(execute_at = now + 7–12 s + availability delay)  1 DB
      → HTTP 200 returned here.  ~17 DB round trips.
── HALF 2: delivery (scheduled_actions worker, up to 60 s later) ────────
process_once()
 ├─ repair_followup_obligations()   2 × N_obligations DB round trips   ← SCALE-002
 ├─ claim_due_actions(limit=20)     22 DB round trips for 20 due       MEASURED
 └─ for action in claimed:          ★★ STRICTLY SEQUENTIAL ★★
      ├─ _should_still_send()       4 DB (fan, auto_availability, history(10), sleep_hours)
      └─ _run_auto_reply()
           ├─ sync_recent_fan_messages (only on stale reclaim)   1 API FANSLY + 3–5 DB
           ├─ get_conversation_history(10)                        1 DB
           └─ _debounced_auto_reply()            43 DB + 2 LLM     MEASURED
                 ├─ analyze_situation()          ★ LLM #1
                 ├─ 5 deterministic refreshers   ~24 DB (price_learning runs TWICE)
                 ├─ generate_replies()           ★ LLM #2
                 ├─ _sleep_while_current(composition_delay)   6.8 s MEASURED
                 ├─ send_apifansly_message()     1 API FANSLY (fresh TLS handshake)
                 └─ save_message()               2 DB
           └─ get_conversation_history(10)                        1 DB
 └─ asyncio.sleep(60)   ← unconditional, even with a full backlog
```

**Total per Full Auto reply: ~50 DB round trips, 2 LLM calls, 1–2 API Fansly calls,
6.8 s of deliberate sleep** (MEASURED components; total CALCULATED).

**End-to-end fan-visible latency** (CALCULATED):
`0–60 s` scheduler poll wait (avg 30 s) `+ 7–12 s` scheduled jitter
`+ availability delay` (human-realism, `build_availability_delay`)
`+ ~3–8 s` generation `+ 6.8 s` typing simulation
= **~45–90 s minimum, before any queueing**, and it degrades linearly once more than
a handful of actions are due in the same cycle (see §17).

### 3c. Operator PPV send (`POST /fan/{id}/operator-ppv`)

Synchronous in-request: ledger claim (atomic, protected by the partial unique index —
good), API Fansly send, receipt persist, reconciliation schedule. Roughly 12–18 DB
round trips and one API Fansly call. Latency is dominated by the API Fansly send on a
**freshly established TLS connection** (`services/apifansly.py::request` creates a new
`httpx.AsyncClient` per call — PERF-006), so ~300–700 ms of avoidable handshake.

### 3d. Conversation opening (dashboard)

```
browser
 ├─ proxy.ts middleware: supabase.auth.getSession()          (cookie read)
 ├─ hydrate 1.0 MB of client JS (MEASURED, 23 chunks, largest 222 KB)
 ├─ supabase.auth.getUser()                                  1 round trip
 ├─ chatter_creators + creators join                         1 round trip
 ├─ fan_conversation_summaries.select('*')  NO LIMIT         1 round trip, up to 1,000 rows, all columns
 ├─ messages.select('*').limit(50)                           1 round trip
 ├─ getLatestSuggestions (direct Supabase)                   1 round trip
 ├─ apiFetch /creator/{id}/auto-availability                 1 round trip
 ├─ apiFetch /fan/{id}/full-auto-status                      1 round trip
 ├─ scripts + blocked_words                                  2 round trips
 ├─ apiFetch /vault-media-urls (if any PPV in history)       1 round trip
 └─ apiFetch /sync-fan-messages  → backend → API FANSLY      1 round trip + 1 API Fansly call
```

**Nine to eleven sequential-ish client round trips**, every one of them after hydration
because **every page and component in the dashboard is `'use client'`** (verified: all
12 files). There are no server components and no server-side data fetching, so
time-to-content is `hydration + waterfall`.

---

## 4. BACKGROUND AUTOMATION RISK MAP

Complete inventory of every long-lived task, loop, and in-memory structure.

### 4a. Schedulers (all started in `main.py::lifespan`, all in the web process)

| Loop | Interval | Durable? | Restart behaviour | Can it run twice? |
|---|---|---|---|---|
| `ppv_sweep_scheduler` | 15 min | Obligation lives in `fans.pending_ppv_check` | Resumes; sweep is a repair pass | Yes if >1 worker |
| `_scheduled_actions_scheduler` | 60 s | **Yes** — `scheduled_actions` table | In-flight action left `PROCESSING`, reclaimed after 10 min | Yes if >1 worker; stale-reclaim protected by status-guarded update |
| `vault_autosync_scheduler` | 1 h | Partially — `creators.last_vault_sync_at` | In-flight sync **lost silently**, dashboard shows `idle` | Guarded only by an **in-process dict** |
| `chat_reconciliation_scheduler` | 5 min tick | `claim_chat_reconciliation` RPC (9 min DB lock) — **good** | Cursor map lost → full message re-sync for every chat | Protected by the DB claim |
| `model_availability_scheduler` | 6 h | No | Resumes | Harmless |
| `FanslyPoller._poll_loop` (per account) | 8–300 s adaptive | No | Cursors reseeded | See DEAD-004 — effectively inert |

**Every one of these runs inside the single Uvicorn web process.** There is no separate
worker process and no leader election. Adding `--workers N` to the Procfile would
start N copies of all six.

### 4b. In-memory state — growth analysis

| Structure | File | Keyed by | Pruned? | Size at 100 creators / 100k fans |
|---|---|---|---|---|
| `_processed_messages` | `main.py:129` | message id | **`.clear()` at 1,000** — drops all dedup state at once | bounded, but see REL-004 |
| `_chat_last_message_ids` | `main.py:668` | (creator, group) | **Never** | **1 entry per chat ever seen** — 100k+ entries, ~15 MB |
| `_chat_reconcile_due_at` | `main.py:667` | creator | Yes (removed when creator disappears) | fine |
| `_chat_reconcile_denied_bindings` | `main.py:666` | (creator, account) | Partially | fine |
| `_active_chat_binding_retry_after` | `main.py:132` | fan | **Only on success** | one float per fan that ever failed binding — unbounded |
| `_active_chat_binding_tasks` | `main.py:133` | fan | Yes (finally block) | fine |
| `_vault_sync_state`, `_vault_sync_retry_after` | `main.py:130–131` | creator | Never | bounded by creators |
| `_categorize_state` | `main.py:3321` | creator | Never | bounded by creators |
| `_pending_auto_replies` | `services/suggestions.py:106` | fan | Yes (`_release_auto_reply_slot`) | fine |
| `_USER_CACHE` | `core/auth.py:26` | token hash | Expired-only prune above 512 | grows with token rotation |
| `_USAGE_EVENTS` | `services/apifansly.py:22` | — | `deque(maxlen=50_000)` + 24 h window — **good** | bounded ~10 MB |
| `_pending_writes` | `services/model_telemetry.py:19` | — | Hard cap 500 — **good** | bounded |
| `lru_cache` clients | `ai/model_providers.py:125,132` | (base_url, key, timeout) | maxsize 8/16 — **good** | bounded |

Only two of these are real leaks at agency scale: `_chat_last_message_ids` and
`_active_chat_binding_retry_after`. Both grow per-fan and are never pruned.

### 4c. Concurrency gates

Four semaphores exist, **none of them on LLM calls**:

```
main.py:135                    _protected_video_download_gate = Semaphore(1)
services/vault_classifier.py:81  _VISION_GATE                = Semaphore(2)
services/vault_semantics.py:21   _SEMANTIC_GATE              = Semaphore(32)
services/video_frames.py:18      _VIDEO_EXTRACTION_GATE      = Semaphore(2)
```

There is **no concurrency limiter on writer or analyzer calls anywhere in the
codebase** (verified by exhaustive grep). See §17E.

---

## 5. DATABASE PERFORMANCE REPORT

### 5a. Schema authority — the most important database finding

**There is no authoritative schema in the repository.** The 18 files in `db/*.sql` are
all `ALTER TABLE ... ADD COLUMN` / `CREATE INDEX` additions. **Not one of them creates
`creators`, `fans`, `messages`, `suggestions`, `chatter_creators`, `ppv_offers`,
`vault_sets`, `creator_vault_media`, `fan_lists`, `fan_list_members`, `scheduled_actions`,
`reengagement_log`, `scripts`, `blocked_words`, or the `fan_conversation_summaries`
view.** Those objects exist only in the live Supabase project.

Consequences, all of them real:

- **The primary keys, foreign keys, unique constraints, and indexes on the hottest
  tables in the system cannot be verified, reviewed, or reproduced.** The single
  most-executed query in the product — `messages WHERE fan_id = ? ORDER BY sent_at DESC
  LIMIT 40` — has no index definition anywhere in version control.
- **Whether `messages.fansly_message_id` has a unique constraint is UNKNOWN**, and
  the entire webhook/poller/reconciler duplicate-message story depends on it
  (see REL-002).
- A new environment cannot be created from the repository.
- CI's Postgres schema tests (`tests/test_*_schema.py`, 10 tests, currently skipped
  locally) can only test the *additive* migrations, never the base schema.

### 5b. Index audit — what exists in version control

| Table | Indexes present in `db/*.sql` | Hot query | Verdict |
|---|---|---|---|
| `scheduled_actions` | `(status, execute_at)`, `(fan_id, status)` | `status='PENDING' AND execute_at <= now() ORDER BY execute_at` | ✅ correct |
| `ppv_deliveries` | `(fan_id, claimed_at desc)`, `(creator_id, status, claimed_at desc)`, gin(`media_ids`), partial unique on `(fan_id) WHERE status IN (claimed, delivered_pending) AND source <> 'operator'` | single-flight claim | ✅ excellent |
| `ppv_approval_requests` | partial unique one-pending-per-fan, `(creator_id, status, created_at)` | pending list | ✅ correct |
| `fan_lists` | unique `(creator_id, external_list_id) WHERE external_list_id IS NOT NULL`, `(creator_id, source)` | mirror upsert | ✅ correct |
| `fan_list_members` | `(list_id, source)` | reconciliation | ⚠️ **no index on `fan_id`** — `preview_auto_audience` and `process_incoming_fan_message` both filter by `fan_id` |
| `fans` | `(creator_id, subscription_status, is_follower, fansly_lifetime_spend_cents desc)` | audience preview | ⚠️ narrow; nothing for `(creator_id, platform_fan_id)` which `get_fan` uses on **every inbound message** |
| `fan_facts` | `(fan_id, ...)` active partial, `(creator_id)`, unique `(fan_id, fact_key, normalized_value)` | context load | ✅ correct |
| `model_usage_events` | `created_at`, `creator_id`, `model`, `feature` (four separate single-column) | "cost per creator last 30 days" | ⚠️ needs composite `(creator_id, created_at desc)` |
| **`messages`** | **none in repo** | `(fan_id, sent_at desc)`, `(fansly_message_id)`, `(creator_id, role)` | ❓ **UNKNOWN — verify immediately** |
| **`creators`** | **none in repo** | `(apifansly_account_id)`, `(fansly_account_id)` — used on **every webhook** | ❓ **UNKNOWN** |
| **`chatter_creators`** | **none in repo** | `(chatter_id)` — used on **every authenticated request** and inside **every RLS policy evaluation** | ❓ **UNKNOWN — highest-frequency lookup in the system** |

### 5c. Measured query amplification

| Operation | DB round trips | Source |
|---|---|---|
| Assisted suggestion, flags off | **9** | MEASURED |
| Assisted suggestion, flags on | **34** | MEASURED |
| Full Auto reply generation+delivery | **43** | MEASURED |
| Full Auto reply, end to end incl. worker overhead | **~50** | CALCULATED |
| `claim_due_actions` with 20 due | **22** (2 selects + 20 individual UPDATEs) | MEASURED |
| `repair_followup_obligations` | **2.00 per outstanding obligation, every 60 s** | MEASURED (401 for 200) |
| Inbound webhook ingestion (before delivery) | **~17** | CALCULATED |

### 5d. N+1 patterns and one-row-at-a-time writes

| Location | Pattern | Cost at scale |
|---|---|---|
| `db/commercial_queries.py::claim_due_actions` | 2 selects then **one UPDATE per claimed row** | 22 round trips per cycle; should be a single `UPDATE ... WHERE id = ANY(...) AND status = ...  RETURNING *` |
| `workers/scheduled_actions.py::repair_followup_obligations` | `ensure_action_pending` per obligation = select + conditional write | 2 round trips × every obligation × every minute |
| `main.py::_run_vault_categorization` | `for result in results: await update(...)` after each concurrency batch | **one UPDATE per media item** — 10,000 sequential UPDATEs for a 10,000-item vault |
| `main.py::sync_chats` | `get_fan` + `fans.update` **per chat**, sequential | 2 round trips × every chat × every reconcile pass |
| `services/fansly_lists.py::_reconcile` | one upsert per added member, one delete per removed member | 5,000 sequential writes for a 5,000-member list |
| `services/suggestions.py::sweep_stale_ppv_checks` | `ensure_action_pending` per stale fan, sequential | 2 round trips per pending PPV, every 15 min |
| `db/queries.py::mark_ppv_purchased` | reads **all** creator messages with `media_context`, loops in Python, one UPDATE | unbounded read |

### 5e. Unpaginated selects on unbounded tables (the 1,000-row cap)

Static analysis found **43 selects on large tables with no `.limit()`, `.range()`,
`.single()` or `count=`**. These are the ones where truncation changes behaviour:

| Location | Table | Filter | What breaks at >1,000 rows |
|---|---|---|---|
| `main.py:4843` | `fan_list_members` | **none at all** | Cross-tenant read; exclusion rules silently lost |
| `main.py:4837` | `fans` | creator | Auto-audience preview under-reports total/eligible |
| `main.py:4848` | `messages` | creator+role | `is_new_fan` wrong for most fans |
| `main.py:1793` | `fans` | creator | `sync_chats` incremental early-break never triggers → full pagination every 10 min |
| `main.py:799` | `fans` | `auto_mode=true` | Creators with auto fans past row 1,000 downgraded to the idle reconcile interval |
| `services/fansly_lists.py:132` | `fans` | creator | **Destructive** — see SCALE-003 |
| `services/fansly_lists.py:147` | `fan_list_members` | mirror ids | Incomplete `current_fan_ids` → missed removals |
| `services/suggestions.py:1758` | `fans` | `pending_ppv_check not null` | PPV reconciliation repair silently stops |
| `services/full_auto_operations.py:297,306,312` | states/fans/actions | creator | Full-auto health page under-reports |
| `db/queries.py:192` | `messages` | fan+role | `get_sent_ppv` misses old PPVs |
| `db/commercial_queries.py:178` | `messages` | fan+role | `sent_set_ids` incomplete → previously-sent content can be re-offered |

### 5f. Other schema observations

- **Money is stored two ways.** `ppv_deliveries.price_cents` and `fan_commercial_states.*_cents`
  use integer cents (correct), but `fans.total_spent` is used as whole dollars
  (`fan_total_spent >= 500 → "whale"`), and `pending_ppv_check` carries **both**
  `price` (dollars, float) and `price_cents`. `messages.media_context.ppv.price` is a
  float. Conversions appear in at least six places (`float(price_cents)/100`,
  `int(round(price*100))`). This is a latent rounding/comparison bug surface.
- **JSONB used as relational data**: `fans.sales_log`, `fans.not_sold_log`,
  `fans.pending_ppv_check`, `fans.ai_summary`, `fans.preferences`,
  `messages.media_context`. `sales_log` is scanned in Python for duplicate order IDs in
  the purchase webhook (`main.py:4166`) — a check-then-write with no uniqueness
  constraint, so two concurrent purchase webhooks for the same order can both pass.
- **`db/queries.py::get_fan` and `get_fan_by_id` use `select("*")`** and are called on
  every inbound message, pulling every JSONB column including `sales_log` and
  `ai_summary` when only ~12 scalar fields are used.

### 5g. Migration discipline

There is **no single authoritative migration mechanism**. The 18 SQL files are
idempotent and reasonably ordered by convention (`agency_operability` → `ppv_delivery_ledger`
→ `operator_ppv_concurrency`), and header comments state ordering
("Apply after ppv_delivery_ledger_v1.sql"), but:

- There is **no migration runner, no ordering table, and no applied-state tracking**.
  Application is manual, in Supabase, by a human reading comments.
- **CI does not verify that the migrations were applied**, only that they parse and
  enforce constraints against a fresh Postgres.
- `db/tenant_isolation_v1.sql` **auto-discovers tables** and must be **re-run after any
  new table is added**. `db/fansly_lists_v1.sql` documents this requirement in a
  comment. Any new creator-owned table shipped without re-running it lands with **RLS
  disabled**, which is a cross-tenant read from the browser.
- Runtime code compensates for un-applied migrations in at least two places:
  `workers/scheduled_actions.py::_creator_auto_mode_default` has an explicit
  "rolling deployment / missing symbol" fallback, and `ensure_action_pending` has a
  special case keyed on the literal string `"get_creator_auto_mode_default"` appearing
  in a stored `last_error`. Both are compensation for deployment races.
- Six major features are gated off by default (`FAN_INTELLIGENCE_ENABLED`,
  `FAN_LIFECYCLE_ENABLED`, `AFFORDABILITY_ENABLED`, `PRICE_LEARNING_ENABLED`,
  `ADAPTIVE_SESSION_PLANNER_ENABLED`, `CONVERSATION_DIRECTOR_ENABLED`,
  `FANSLY_LISTS_SYNC_ENABLED`) precisely because their migrations may not be applied.
  **`COMMERCIAL_LAYER_ENABLED` is read by `services/suggestions.py` but appears nowhere
  in `.env.example`** — the entire commercial state machine is behind an undocumented flag.

---

## 6. LLM / TOKEN / CACHE REPORT

### 6a. Model call inventory

| Call site | Target env prefix | Default | Runs on | Telemetry? |
|---|---|---|---|---|
| `ai/situation_analyzer.py::analyze_situation` | `ANALYZER_*` | `anthropic:claude-haiku-4-5-20251001` | **every message** | ✅ |
| `ai/generator.py::generate_replies` | writer router | `openrouter:moonshotai/kimi-k2.6` → `together:deepseek-ai/DeepSeek-V4-Pro` | **every message** | ✅ |
| `services/fan_intelligence.py` extractor | `EXTRACTOR_*` | `together:openai/gpt-oss-120b` | every message (flag-gated) | ✅ |
| `services/suggestions.py::_update_fan_memory` | **hardcoded** | `together:meta-llama/Llama-3.3-70B-Instruct-Turbo` | every 10th fan message | ❌ **none** |
| `services/suggestions.py::_update_fan_ai_summary` | **hardcoded** | `together:meta-llama/Llama-3.3-70B-Instruct-Turbo` | every 10th fan message | ❌ **none** |
| `persona/extractor.py::extract_persona` | hardcoded | `together` | **never — zero callers** | ❌ |
| `ai/rag.py::get_embedding` | hardcoded | `openai:text-embedding-3-small` | **never — `enabled=False` at all 3 call sites** | ❌ |
| `services/vault_classifier.py` | Qwen3-VL via Modal | — | vault categorisation | partial |

**Two of the five live model calls bypass the entire provider abstraction, the
writer router, model-availability tracking, and telemetry.** `_update_fan_memory` and
`_update_fan_ai_summary` construct their own `AsyncOpenAI` client at module import
against Together with a hardcoded Llama-3.3-70B model and **no timeout** (the OpenAI
SDK default is 600 s). Their cost never reaches `model_usage_events`, so
**"what did this month cost" is systematically under-reported**.

These two also **overlap heavily with each other and with `learn_from_fan_message`**:
all three extract fan facts (age, location, payday, kinks, preferences) from the same
recent conversation, into three different storage shapes (`fans.member_note`,
`fans.ai_summary`, `fan_facts`). Three extraction models on the same input is a
genuine cost and consistency problem — see COST-005.

### 6b. Prompt size — MEASURED

Measured by rendering `build_prompt` against a realistic context and counting
characters ÷ 3.6.

| Conversation length | System block | User block | Total | Byte-identical prefix vs. next turn | Reusable % |
|---|---|---|---|---|---|
| 10 messages | 1,778 | 1,579 | 3,357 | 2,372 | 70.7% |
| 16 messages | 1,778 | 1,716 | 3,494 | 2,149 | 61.5% |
| 40 messages | 1,778 | 1,719 | 3,497 | 2,149 | 61.4% |
| 100 messages | 1,778 | 1,719 | 3,497 | 2,149 | 61.4% |
| 40, with director + strategy blocks (flags on) | 1,778 | 1,887 | **3,665** | **1,811** | **49.4%** |

The writer prompt is stable at ~3,500 tokens regardless of conversation length,
because the transcript is capped at `conversation_history[-16:]`. That is good.

### 6c. Prompt caching — three concrete defects

**COST-002a — `cache_control` never reaches any provider. PROVEN.**
`ai/prompt_builder.py` returns the system message as content blocks with
`{"type": "text", "text": stable_system, "cache_control": {"type": "ephemeral"}}`.
`ai/generator.py::generate_replies` then calls
`flatten_message_content(prompt_messages[0]["content"])`, which **joins the block text
into a plain string and discards `cache_control`**. I verified this by execution:

```
system content type from build_prompt: list
first block keys: ['type', 'text', 'cache_control']
after flatten_message_content -> type: str
cache_control survives to transport? False
```

`_complete_anthropic` then passes `"system": <plain string>`. **Anthropic prompt
caching is therefore completely inert** — `cache_read_input_tokens` will always be 0
for any Anthropic target. Since `.env.example` documents `ANALYZER_PROVIDER=anthropic`
and the analyzer runs on every single message, this is an ongoing, silent cost.
For the OpenAI-compatible path (OpenRouter/Kimi, Together/DeepSeek) the flattening
is *correct and necessary* — those providers use implicit prefix caching — so the fix
is to keep the blocks for Anthropic only, not to remove the flattening.

**COST-002b — the analyzer's static rules are placed where they can never be cached.**
`ai/situation_analyzer.py` builds one user message shaped:

```
[ instruction line ] [ THE CONVERSATION ] [ latest message ] [ ~1,570 TOKENS OF STATIC RULES ]
                          ↑ changes every turn
```

with `system="Return only the requested JSON object. Do not add commentary."` (~12 tokens).
The **entire reusable prefix of the analyzer prompt is 12 tokens**. Moving the
1,570 tokens of JSON schema + COMMERCIAL INTERPRETATION RULES + SAFETY into the
`system` argument, and leaving only the conversation in the user turn, would make
~1,570 tokens cacheable on every message with **zero behavioural change**. On the
default Haiku target this is the single highest-leverage token change available.
Additionally, `analyze_situation` passes **no `session_id`**, so if the analyzer is
ever moved to OpenRouter it gets no sticky routing either.

**COST-002c — volatile blocks sit ahead of durable ones inside the user turn.**
`build_prompt` documents the intent correctly — *"the durable fan profile and the
transcript are appended to, not rewritten, between turns, so they belong in front of
the values that change on every single message"* — but `fan_context`, which is placed
first, is assembled as:

```python
fan_context_parts = [
    learned_intelligence_block,      # changes as facts are learned
    affordability_block,             # recomputed every message
    price_learning_block,            # recomputed every message (TWICE in the auto path)
    conversation_director_block,     # recomputed every message
    session_strategy_block,          # recomputed every message
    expression_guidance_block,
    notes, member_note, model_note, kinks, reengagement_triggers,   # durable
    buyer_lifecycle,                 # changes on purchase
]
```

The five per-message blocks are **first**, ahead of the durable profile and the
transcript. That is what drops the reusable prefix from 61% to 49% when the
intelligence flags are on (MEASURED). Reordering `fan_context_parts` to put the durable
items first costs nothing and recovers ~340 tokens of cacheable prefix per call.

A secondary effect: the transcript is a **sliding** 16-message window
(`conversation_history[-16:]`), so past turn 16 the first transcript line changes every
turn and the transcript can never be part of a shared prefix. A growing window pinned
to a stable start (e.g. anchored every N turns) would preserve it; this is a real
trade-off against prompt size and I am not recommending it without measurement.

**What is genuinely well done here:** `ai/session_affinity.py` derives OpenRouter's
`session_id` from `sha256(creator_id + fan_id)` only — no timestamps, no counters, no
randomness — and `ai/openrouter_routing.py` pins `provider.only` with
`allow_fallbacks=false`. That is exactly right for cache affinity, and
`tests/test_prompt_cache_structure.py` pins the ordering so a future edit cannot
silently break it. The measured 49–61% reusable prefix is a real number that provider
caching will actually exploit.

**Theoretical vs. actual:** theoretical reusable prefix is **1,811–2,372 tokens per
writer call** (MEASURED). Actual provider-reported cached tokens are **UNKNOWN** — they
are recorded in `model_usage_events.cache_read_tokens` and
`metadata.cache_hit_ratio` by `services/model_telemetry.py`, which is correct
instrumentation, but no data was available to this audit. **Query that table before
acting on any cache recommendation.**

### 6d. Retry and fallback behaviour

`generate_replies` builds `attempt_targets = [primary, primary, fallback or primary]`
and loops with **no sleep between attempts**. It `continue`s to the next attempt on:

1. transport exception, **and**
2. `parse_reply_candidates()` returning `[]`.

Condition (2) is the problem. `parse_reply_candidates` rejects a reply if
`any(phrase in reply.lower() for phrase in bot_phrases)`, and `bot_phrases` includes
the bare substrings **`"yourself"`, `"interesting"`, `"noted"`, `"got it"`,
`"i like that"`, `"that's nice"`**. In an adult-chat product, `"yourself"` matches
"touch yourself", "by yourself", "enjoy yourself" — all legitimate. It then requires
**≥3 surviving candidates** (or ≥2 plus padding to exactly 3), else returns `[]`.

So a single ordinary reply containing "yourself" can cost **three full writer
generations (~10,500 input tokens) and then send nothing at all** — `generate_replies`
returns `[]` and `_debounced_auto_reply` returns silently. In Full Auto this means the
fan gets no reply, the action is marked failed, and it is **retried up to 8 times by
`fail_action`, re-running the entire pipeline each time**: worst case
**8 × 3 = 24 writer generations plus 8 analyzer calls for one message that is never
delivered** (COST-001 / REL-005).

The same three-attempt burst is what turns an OpenRouter 429 into 3× load on the
pinned `Inceptron` provider, because `allow_fallbacks=false` means the retry cannot
route elsewhere.

---

## 7. API FANSLY EFFICIENCY REPORT

### 7a. Operation table

| Operation | Endpoint | Trigger | Frequency | Pagination | Retry | Cache/reuse | On failure |
|---|---|---|---|---|---|---|---|
| `list_chats` | `GET {acct}/chats` | `sync_chats` | every 5–30 min per creator, + on connect | cursor, **all pages** unless early-break fires | **none** | none | 409 → binding suppressed for process |
| `list_chat_messages` | `GET {acct}/chats/{g}/messages` | per changed chat; webhook media enrich; proactive reconcile | **hard cap `limit=10`** | cursor | **none** | `_chat_last_message_ids` (in-memory) | logged, chat skipped |
| `send_message` | `POST {acct}/chats/{g}/messages` | every reply/PPV | per message | n/a | **none** | **new TLS connection every call** | raise → freeze fan |
| `delete_message` | `DELETE ...` | PPV compensation | rare | n/a | none | — | logged |
| `typing` indicator | `POST {acct}/chats/{g}/typing` | every auto reply | per auto reply | n/a | none | new client, `timeout=5` | swallowed |
| `list_vault_albums` | `GET {acct}/vault/albums` | vault sync | ≤1/creator/24 h | none | none | — | access-denied backoff 24 h |
| `list_vault_album_media` | `GET {acct}/vault/albums/{a}/media` | vault sync | per album per page (`limit=50`) | cursor + `should_stop_album_scan` early stop — **good** | none | `existing_ids` set | as above |
| `media/download` | `POST media/download` | protected video frames | per unreadable video | n/a | none | `Semaphore(1)` | raise |
| `current_account`, `account_media_prices` | various | health/ppv options | per call | — | none | — | logged |
| `list_account_lists` | `GET {acct}/lists` | chat sync when mirror >6 h old, or operator button | ≤4/creator/day | cursor, **`_MAX_LIST_PAGES=50`** | none | `creators.last_fansly_lists_sync_at` | recorded on creator row |
| `list_account_list_members` | `GET {acct}/lists/{id}/items` | **per list, sequentially** | per list per sync | cursor, **`_MAX_MEMBER_PAGES=200`** | none | none | recorded |
| `connect` / `verify-2fa` | `POST connect` | operator onboarding | rare | n/a | none | — | returns HTTP 200 with `success:false` |

**There is no retry, no backoff, and no rate-limit handling anywhere in
`services/apifansly.py::request`.** A 429 or 502 raises immediately. Different callers
then behave differently: `sync_chats` converts 409 to an HTTP exception and suppresses
the binding for the process; `_sync_recent_fan_messages` logs and skips the chat;
`send_fansly_message` returns `None` which the auto path treats as "platform rejected"
and freezes the fan. That inconsistency means a transient provider blip produces
frozen fans rather than a retry.

### 7b. The three biggest API Fansly cost drivers

**API-001 — every deploy triggers a full message re-sync of every chat.**
`_chat_message_sync_needed` returns `True` whenever
`_chat_last_message_ids.get((creator_id, group_id))` is `None`, which is the case for
**every chat after every process restart**. The docstring says this is intentional
("a durable safety reconciliation without a new database column") and at one test
account it is free. At 100 creators × 2,000 chats it is **200,000 `list_chat_messages`
calls immediately after every Railway redeploy** — and Railway redeploys on every push.
This is the clearest example in the codebase of a deliberate, correct decision at
1 account that becomes a bill at 100.

**API-002 — the dashboard polls API Fansly through the backend.**
`app/page.tsx` runs `syncActiveFanMessages` every **45 s** (recent activity) or **3 min**
(idle) for the open conversation, and `POST /sync-fan-messages/{c}/{f}` makes a real
`list_chat_messages` call every time. That is ~80 API Fansly calls/hour **per open
browser tab**, entirely duplicating the backend's own 5–30 min reconciliation and the
webhook. At 25 concurrent operators: **~2,000 API Fansly calls/hour from browser
polling alone.**

**API-003 — big creators lose the incremental early-break.**
`sync_chats` builds `existing_platform_ids` from an **unpaginated** `fans` select
(1,000-row cap). The early-break condition is
`page_ids.issubset(existing_platform_ids)`. For a creator with >1,000 fans this is
almost never true, so **every incremental pass paginates the entire chat list**, every
5–30 minutes, forever.

### 7c. Fansly Lists sync — API cost

`sync_fansly_lists` fetches lists (≤50 pages), then for **each list sequentially**
fetches all members (≤200 pages). A creator with 30 lists averaging 3 pages each is
~90 sequential calls per sync, at up to 4 syncs/day. Worst case within the coded
bounds is `50 + 30×200 = 6,050` calls per creator per sync. There is **no concurrency
across lists** and **no single-flight guard** — two operators clicking
`POST /creator/{id}/sync-fansly-lists` start two concurrent full reconciliations of
the same creator, and that endpoint runs the whole thing **synchronously inside the
HTTP request**.

### 7d. Usage telemetry — assessment

`services/apifansly.py` records every response into a bounded 24-hour `deque(maxlen=50_000)`
with per-operation and per-status breakdowns, and exposes it at `GET /apifansly-usage`.
The design is sound (bounded, secret-free, no PII). Three gaps:

1. **It is process-local and lost on every restart.** On Railway, that is every deploy.
   Credit accounting across a month is impossible from this source.
2. **It is not broken down by creator or agency**, so per-tenant credit attribution —
   which is what an agency business needs — is not answerable.
3. **`GET /apifansly-usage` has no tenancy dependency** (it is the only route besides
   `/health` and `/model-runtime-health` without one), so any authenticated operator
   from any agency sees deployment-wide provider usage.

---

## 8. VAULT / MEDIA PERFORMANCE REPORT

### 8a. What is done well

- Incremental sync: `_vault_existing_media_ids` **correctly paginates** past the
  1,000-row cap (with an explicit comment about it), and `should_stop_album_scan` uses
  `lastItemId` plus a conservative three-page fallback to avoid rescanning whole albums.
- Media rows are **batch-upserted in groups of 50** with `on_conflict="creator_id,media_id"`.
- All CPU work is correctly off the event loop: `_prepare_classifier_image` (Pillow),
  `build_contact_sheet` (Pillow), and `_detect` (NudeNet/ONNX) all go through
  `asyncio.to_thread`; ffmpeg/ffprobe use `asyncio.create_subprocess_exec` with
  `wait_for` timeouts and `process.kill()` on timeout.
- Temp files are removed in `finally` blocks; `httpx` clients are closed in `finally`.
- `_VIDEO_EXTRACTION_GATE(2)`, `_VISION_GATE(2)`, `_protected_video_download_gate(1)`
  bound the heaviest operations.

### 8b. VAULT-001 — the autosync scheduler starts every due creator at once (P1)

```python
for c in (creators.data or []):
    ...
    res = await sync_vault_start(cid)      # spawns _run_vault_sync and returns immediately
```

`sync_vault_start` only *spawns* the task. The `await` therefore provides **no
serialisation whatsoever**. With 100 creators whose `last_vault_sync_at` crosses 24 h
in the same hourly pass, this starts **100 concurrent vault syncs**, each of which runs
`_run_vault_categorization` at `VAULT_CATEGORIZATION_CONCURRENCY` (default 12 with a
semantic endpoint configured) → **up to 1,200 concurrent media classifications** inside
the web process. There is no global cap. Memory, CPU, thread pool, and the Modal
vision endpoint all take it simultaneously. Because creators are typically connected in
batches, their 24-hour anniversaries naturally cluster.

### 8c. VAULT-002 — batch-barrier categorisation and per-item writes (P2)

```python
for i in range(0, total, batch_size):
    batch = all_items[i:i + batch_size]
    results = await asyncio.gather(*[_categorize_single_item_with_retry(item, ...) for item in batch])
    for result in results:
        await retry_transient_db_operation(lambda: db.table(...).update(...).eq("id", r["id"]).execute())
```

Two problems. First, the fixed-window `gather` is a **barrier**: a batch of 12 finishes
only when the slowest item finishes, and a video requiring ffmpeg frame extraction can
take 35 s while 11 images take 1 s each. A worker-pool (`Semaphore` + `as_completed`)
would use the same concurrency budget 2–5× more efficiently. Second, the results are
persisted with **one UPDATE per item, sequentially** — 10,000 sequential round trips for
a 10,000-item vault (~250–500 s of pure DB latency).

### 8d. VAULT-003 — in-flight sync/categorisation is lost on restart with no record (P2)

`_vault_sync_state` and `_categorize_state` are plain module dicts. On a Railway
restart mid-sync, the spawned task dies, the state vanishes, and
`GET /sync-vault-status/{id}` returns `{"status": "idle"}` — the dashboard shows the job
as not running rather than as interrupted. Classified rows are persisted so progress is
not lost, and because `_stamp_vault_op("last_vault_sync_at")` runs only at the *end*,
the hourly scheduler will retry within the hour. So the work recovers, but the
**operator-visible state is wrong** and there is no durable record that a job was
interrupted.

### 8e. Worst case for a very large vault (CALCULATED)

For one creator with 20,000 media (15,000 images, 5,000 videos), full initial
categorisation at concurrency 12:

| Stage | Cost |
|---|---|
| `_vault_existing_media_ids` | 20 paged selects |
| Album enumeration + media pages (`limit=50`) | ~400 API Fansly calls |
| Row upserts (batch 50) | ~400 writes |
| Image classification (download + Pillow + NudeNet + semantic) | 15,000 × ~2–4 s ÷ 12 ≈ **42–83 min** |
| Video classification (ffprobe + 4× ffmpeg + contact sheet), gated at 2 concurrent | 5,000 × ~10–35 s ÷ 2 ≈ **7–24 hours** |
| Per-item UPDATE writes | 20,000 sequential ≈ 8–17 min |
| Peak memory | `all_items` list ~8 MB + 12 in-flight images ×~2 MB + Pillow buffers ≈ **150–300 MB** |

**Videos are the wall.** `_VIDEO_EXTRACTION_GATE = Semaphore(2)` is global to the
process, so five creators syncing simultaneously do not get 5× the video throughput —
they *share* the two slots, and the batch barrier means each creator's image batches
stall behind their own videos. A 5,000-video creator is a multi-day job that also
degrades every other creator's sync and, because it shares the default thread pool and
the CPU with request handling, degrades chat latency for the whole deployment.

---

## 9. DASHBOARD PERFORMANCE REPORT

### 9a. FE-001 — `dedupeMessages` is O(n²) and runs twice per incoming message (P1, MEASURED)

`lib/messages.ts::dedupeMessages` does `result.findIndex(...)` for every message, and
each comparison calls `normalizedContent()` on **both** sides — allocating two new
strings via `.trim().replace(/\s+/g,' ')` per comparison, with no memoisation.

Measured on Node 22 (a faster environment than a browser main thread carrying React):

| Loaded messages | Full dedupe | **Cost of appending one realtime message** |
|---|---|---|
| 50 | 4.4 ms | **2.9 ms** |
| 200 | 26.1 ms | **31.1 ms** |
| 500 | 159.7 ms | **154.9 ms** |
| 1,000 | 677.4 ms | **666.7 ms** |
| 2,000 | 2,477 ms | **2,368 ms** |
| 5,000 | 15,276 ms | **15,397 ms** |
| 10,000 | 63,709 ms | **63,030 ms** |

In `app/page.tsx`'s realtime INSERT handler it is called **twice** per message — once
for `messagesCache.current[fan_id]` and once for `tab.messages` — inside a `setTabs`
updater, **synchronously on the main thread**. Double the right-hand column.

The initial load is 50 messages, so this is invisible at first. It becomes a problem the
moment an operator scrolls back: each "load more" adds 50 messages with **no upper
bound**, and `loadMoreMessages` itself runs `dedupeMessages` on the combined array. Ten
scroll-backs (550 messages) already means a ~190 ms freeze per incoming message; twenty
(1,050) means ~1.3 s.

### 9b. FE-002 — the vault grid downloads full-resolution originals as thumbnails (P1)

`app/vault/page.tsx:799`:

```tsx
<img src={item.url} alt="" loading="lazy" style={{ width: 100, height: 100, objectFit: 'cover' }} />
```

`item.url` is the **original** Fansly CDN asset. `thumbnail_url` is selected in the
query (line 260) and populated by `main.py::_vault_media_visual_urls`, which walks
`media.variants` for a real image variant and only falls back to the original when no
variant exists — but the grid never uses it. With `vaultVisibleLimit = 200`, opening an
album renders 200 full-resolution originals into 100×100 boxes. At a typical 2–4 MB per
original that is **400–800 MB of image transfer** for one album view. `loading="lazy"`
only defers off-screen images; everything above the fold loads immediately.

Also on this line: the `onError` handler does
`(e.target as HTMLImageElement).parentElement!.innerHTML = '...'`, mutating DOM that
React owns. Subsequent reconciliation of that subtree can throw
`NotFoundError: The node to be removed is not a child of this node`.

### 9c. FE-003 — the whole vault is loaded into browser state (P1)

`loadVaultMedia` pages the **entire** `creator_vault_media` table for a creator into
`vaultAlbums` state, selecting **26 columns** including `ai_description`, `tags`,
`good_for`, and both URLs (signed Fansly CDN URLs are 200–400 characters each).

| Vault size | JSON transferred (est. 1–2 KB/row) | Sequential requests (`pageSize=1000`, awaited in a `while` loop) | Retained JS heap (≈3–5× JSON) |
|---|---|---|---|
| 1,000 | ~1.5 MB | 1 | ~5 MB |
| 10,000 | **~15 MB** | 10 | **~50–75 MB** |
| 50,000 | **~75 MB** | 50 | **~250–375 MB** |

At 50,000 items the requests alone are ~25–50 s of sequential waiting and the tab is a
plausible OOM candidate. Worse, the realtime handler `refreshVaultSoon` re-runs
**the entire load** 750 ms after any `creator_vault_media` change — so during a
categorisation run the page repeatedly re-downloads the full vault.

`selectedVaultItems` is also recomputed on **every render** with
`Object.values(vaultAlbums).flat()` (no `useMemo`) — a 10,000-element array allocation
per keystroke in the preview modal.

### 9d. FE-004 — the conversation list has no memoisation and no virtualisation (P2)

`components/Sidebar.tsx`:

- **`Sidebar` is not wrapped in `React.memo`** (unlike `ConversationView`, which is).
  Every realtime event in `app/page.tsx` re-renders it in full.
- **There is no `useMemo` anywhere in the file** (0 occurrences). The filter runs inline
  in JSX and does `fanLists.find(l => l.id === activeListId)?.member_fan_ids.includes(c.fan.id)`
  **per conversation** — an O(conversations × members) scan on every render.
  MEASURED: 0.11 ms @ 200 conversations / 100 members, 2.32 ms @ 1,000/500,
  47.75 ms @ 5,000/2,500.
- **`onMouseEnter={() => setHoveredFanId(c.fan.id)}` and `onMouseLeave={() => setHoveredFanId(null)}`
  on every row.** Moving the mouse down the list fires a state update per row, each
  re-running the filter and re-rendering every row.
- `setInterval(() => setNow(Date.now()), 60_000)` forces a full list re-render every
  minute for relative timestamps.
- `filtered.map(...)` renders every row — ~15 DOM nodes each. At the PostgREST cap of
  1,000 conversations that is **~15,000 DOM nodes**.

### 9e. FE-005 — background creator tabs go permanently stale (P2)

The realtime channel in `app/page.tsx` is subscribed with
`filter: creator_id=eq.${activeTab.creatorId}` — **one creator only** — yet the handler
maps over *all* tabs looking for `tab.creatorId === msg.creator_id`. Tabs for other
creators therefore receive no events at all. When the operator switches back, the
conversation-load effect short-circuits on `if (activeTab.conversations.length > 0) return`
and `conversationsCache` still holds the stale list. There is no catch-up fetch on
creator switch (the `recoveryTick` catch-up effect only refreshes the *open thread's*
messages). A multi-creator operator sees stale conversation lists and unread counts
indefinitely.

### 9f. FE-006 — backend write amplification becomes a dashboard realtime storm (P2)

`main.py::sync_chats` executes, for **every chat on every pass**:

```python
await asyncio.to_thread(lambda: db.table("fans").update({"fansly_group_id": group_id, "display_name": fan_name}).eq("id", fan.id).execute())
```

unconditionally — no comparison against the current values. Every one of those UPDATEs
is delivered to every subscribed dashboard via the `fans` UPDATE channel, each one
triggering a `setTabs` that rebuilds the conversations array and re-renders the
un-memoised `Sidebar`. For a creator with 2,000 chats that is **2,000 no-op UPDATEs
every 10 minutes**, arriving as a burst.

### 9g. FE-007 — self-referential fetch effect can loop (P3)

`components/ConversationView.tsx:314` has `ppvMediaMap` in its dependency array **and**
calls `setPpvMediaMap` inside. It terminates only because
`POST /vault-media-urls/{creator_id}` fills `null` entries for every requested id. It
does **not** terminate if the id the client sends differs from the key the server
returns — `normalize_media_ids` strips whitespace and drops empty strings, so a
`media_ids` array containing `""`, `null`, or a padded id produces an id that is never
added to the map, and the effect refires on every render, POSTing forever. The error
path already guards against this by filling nulls; the success path does not.

### 9h. Conversation view at scale — simulation

| Fan history | Fetched initially | Held in React | DOM nodes rendered | Dedupe cost per incoming message | Verdict |
|---|---|---|---|---|---|
| 5 | 5 | 5 | ~50 | <1 ms | fine |
| 100 | 50 | 50 (+50/scroll-back) | ~500 | 2.9 ms | fine |
| 1,000 | 50 | 50 unless scrolled | ~500 | 2.9 ms | **fine until the operator scrolls back** |
| 1,000, fully scrolled back | 50 + 19×50 | 1,000 | ~10,000 | **~1.3 s (2 passes)** | unusable |
| 10,000, fully scrolled back | 10,000 | 10,000 | ~100,000 | **~126 s** | tab hangs |

**Virtualisation is not the first thing to fix.** The initial 50-message window means a
long conversation is fine on open; the cliff comes from unbounded `loadMoreMessages`
combined with O(n²) dedupe. **Fix the dedupe (make it a `Map` keyed on
`id`/`fansly_message_id` with a bounded time-window bucket for the reconciliation case)
and cap retained history at ~500 messages first.** Introduce virtualisation only when
operators are routinely working threads past ~1,000 rendered rows — that is the real
threshold, and today's default of 50 means most will not reach it.

### 9i. Vault page at scale — simulation

| Vault size | Transferred | Requests | Retained heap | `<img>` elements | Full-res bytes above the fold |
|---|---|---|---|---|---|
| 100 | ~150 KB | 1 | ~0.5 MB | 100 | ~200–400 MB |
| 1,000 | ~1.5 MB | 1 | ~5 MB | 200 (capped) | ~400–800 MB |
| 10,000 | ~15 MB | 10 sequential | ~50–75 MB | 200 | ~400–800 MB |
| 50,000 | ~75 MB | 50 sequential | ~250–375 MB | 200 | ~400–800 MB |

The image bytes dominate at *every* size and are fixed by using `thumbnail_url`. The
JSON payload becomes the problem past ~10,000 items and is fixed by fetching per-album
with a projection instead of loading the whole vault.

### 9j. Bundle and build — MEASURED

`npm run build` succeeds (Next 16.2.12 + Turbopack, TypeScript clean).
Client JS: **1.0 MB across 23 chunks**, largest 222 KB and 205 KB.
All 8 routes prerender as static shells; **every page and component is `'use client'`**,
so the shells contain no data and everything waterfalls after hydration.

`recharts` (^3.8.0) is declared in `dependencies` but **imported by zero files** —
confirmed by grep across `app/` and `components/`. It is not in the bundle, but it is
in `npm ci` install time and `node_modules`.

---

## 10. DEAD / OBSOLETE CODE REPORT

Every entry below was checked against imports, dynamic imports, FastAPI route
registration, worker dispatch, string-based action names (`HANDLERS`), callbacks, test
usage, migration dependencies, and dashboard call sites. **Nothing was deleted.**

### DEFINITELY DEAD — SAFE TO DELETE

| Item | Evidence |
|---|---|
| `persona/extractor.py` (entire module, 132 lines) | `extract_persona` has **zero references** anywhere — no route, no service, no test, no script. `persona/__init__.py` is empty and nothing imports `persona.extractor`, so the module is never even loaded. It is the only caller of `db.queries.save_embedding`. |
| `db/queries.py::save_embedding` | Only caller is `persona/extractor.py`. |
| `core/auth.py::require_dashboard` | **0 references.** Superseded by `api_auth_middleware`. |
| `core/auth.py::require_webhook` | **0 references.** Superseded by the middleware's `_WEBHOOK_PATHS` branch. |
| `db/queries.py::update_fan_spend` | **0 references.** |
| `db/queries.py::increment_fan_total_spent` | **0 references.** (The `increment_fan_spent` RPC it calls may also be orphaned in Postgres.) |
| `db/queries.py::get_creator_fansly_account_id` | **0 references.** |
| `ai/generator.py::BOT_PHRASES` | Defined at module level, **never read**; an identical list is duplicated inline inside `parse_reply_candidates`. |
| `redis==8.0.1` in `requirements.txt` | No `import redis` anywhere. `UPSTASH_REDIS_URL` / `UPSTASH_REDIS_TOKEN` are **required** `Settings` fields read by nothing. |
| `recharts` in dashboard `package.json` | Imported by zero files. |
| Unused imports: `main.py` `timedelta`, `EventSourceResponse`; `ai/situation_analyzer.py` `re`; `services/fan_lifecycle.py` `json`; `services/affordability.py` `AffordabilityState`; `services/price_learning.py` `get_price_learning_policy`; `services/fansly_poller.py` `Optional`; `services/fansly_session_store.py` `os`, `SessionExpiredError`; `scripts/run_model_eval.py` `os` | Ruff `F401`, verified |
| `db/queries.py:357` `location`, `services/suggestions.py:699` `fan_tier`, `db/queries.py:397` `fan_kinks`, `services/fansly_client.py:167` `e` | Ruff `F841` / vulture 100% |

### PROBABLY DEAD — needs one confirmation before deletion

| Item | Evidence | What to confirm |
|---|---|---|
| **`routes/fansly.py` (all 4 routes) + `services/fansly_client.py` + `services/fansly_session_store.py` + `services/fansly_poller.py`** | See DEAD-004 below — the router is **registered and reachable but structurally broken**. | That no agency has live rows in `fansly_sessions`. |
| `ai/rag.py` (`get_embedding`, `find_similar_exchanges`) | Called at 3 sites, **all with `enabled=False`** — it returns `[]` before doing anything. Constructs an `AsyncOpenAI` client at import against the required `OPENAI_API_KEY`. | That RAG is not planned for re-enablement this quarter. |
| `db/queries.py::get_similar_exchanges` | Only caller is `ai/rag.find_similar_exchanges`. | Same. |
| `message_embeddings` table | Only written by `save_embedding` (dead) and read by `get_similar_exchanges` (dead). | Same. |
| `db/price_learning_queries.py::get_price_learning_policy` | Only reference is the unused import in `services/price_learning.py`; the live path uses `get_effective_price_learning_policy`. | Nothing. |
| `POST /sync-vault/{creator_id}` | Superseded by `/sync-vault-start`; the dashboard calls only `/sync-vault-start`. | No external caller. |
| `GET /apifansly-usage` | **Not called by the dashboard.** Also the only tenanted-data route with no tenancy dependency. | Whether it is used by an ops runbook. |
| `GET /debug-scenes/{id}`, `GET /debug-shoot-clusters/{id}` | Debug endpoints, not called by the dashboard, live in production. | Whether they are used for support. |
| `core/config.py` `USE_ANTHROPIC`, `PRIMARY_MODEL`, `FALLBACK_MODEL` | Module-level constants naming `claude-sonnet-4-20250514` and `openai/gpt-oss-20b`; **0 references**. Stale from the pre-router era. | Nothing. |
| `Settings.TOGETHER_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` as **required** fields | All three are read from `os.environ` by name at the provider layer, not through `Settings`. Only `TOGETHER_API_KEY` is read via `get_settings()` — by `services/suggestions.py:103`'s `together_client` (see COST-005) and `ai/rag.py` (dead). | Which providers production actually uses. |

### KEEP FOR COMPATIBILITY — do not delete

| Item | Why |
|---|---|
| `main.py:4331` webhook fallback lookup by `fansly_account_id` | Explicitly handles older webhook deliveries lacking the top-level API account id. |
| `core/webhooks.py` bare-hex vs `sha256=` signature handling | Providers send both forms. |
| `main.py` `legacy_secret` / `x-webhook-secret` path on `/webhook/fansly` | Documented as retained for local tooling. |
| `workers/scheduled_actions.py::_creator_auto_mode_default` getattr fallback | Deliberate rolling-deployment guard. |
| `ensure_action_pending`'s `"get_creator_auto_mode_default" in last_error` special case | Deliberate recovery from the above. |
| `main.py::process_incoming_fan_message` `legacy_excluded_ids` from `fan_lists.exclude_from_auto` | Bridges the pre-`AutoAudiencePolicy` exclusion model. |
| `db/operator_ppv_concurrency_v1.sql` `drop index ppv_deliveries_one_active_per_fan_idx` | Migration dependency on the superseded index. |
| `ai/model_migrations.py` `("together","moonshotai/Kimi-K2.6") → "moonshotai/Kimi-K3"` | Live redirect for a retired model id. |

### UNKNOWN / NEEDS RUNTIME EVIDENCE

| Item | Question |
|---|---|
| `deploy/modal_qwen_vl.py` (522 lines) | Deployed to Modal separately? Not imported by the app; `modal` is a dev-only dependency. |
| `scripts/` (8 files, ~1,700 lines) | `run_model_eval.py`, `build_blind_review.py`, `ingest_chat_logs.py`, `augment_chat_logs.py`, `backfill_*.py`, `openrouter_smoke.py`, `test_fansly_login.py` — operational tooling; some backfills are presumably one-shot and already run. |
| `reengagement_log`, `reengagement_settings` tables | Referenced once each (`delete_creator` cascade, and one read). Is re-engagement configured through them or through `AutoAudiencePolicy`? |
| `tests/test_stage_classifier.py`, `tests/test_prompt_builder.py` | **Both are 0-byte files.** Placeholders or deleted content. |

---

## 11. RELIABILITY / RACE CONDITION REPORT

### State-machine traces

I traced each of the requested scenarios against the actual code.

| Scenario | What actually happens | Verdict |
|---|---|---|
| **Fan replies while a delayed auto message is queued** | `schedule_auto_reply` cancels the in-memory task **and** calls `cancel_actions_for_fan(fan_id, "AUTO_REPLY")`, which for AUTO_REPLY cancels `PENDING`, `FAILED` **and `PROCESSING`** rows. The in-flight generation is cancelled via `_pending_auto_replies`; `_debounced_auto_reply` also re-checks `expected_trigger_at` and re-reads history post-generation. `fail_action` is `.eq("status","PROCESSING")`-guarded so it no-ops on the now-CANCELLED row. | ✅ **Correctly handled.** Three independent guards. |
| **Creator switches Auto off while an action is pending** | `_should_still_send` re-reads `fan.auto_mode`, falls back to `creator.auto_mode`, and re-checks `_creator_auto_availability`. | ✅ Handled. |
| **Fan purchases while a followup is pending** | `_should_still_send` compares `state.next_followup_type`, `next_followup_dedupe_key`, and per-type snapshots (`payday_at`, `session_completed_at`, `last_abandoned_media_id`, `last_offer_at`). | ✅ Handled — this is careful work. |
| **Two purchase webhooks for the same order** | `main.py:4166` scans `fan_row["sales_log"]` in Python for a matching `platform_order_id` and returns `duplicate`. **Read-then-write with no uniqueness constraint.** Two concurrent deliveries both read the pre-write log and both proceed. | ⚠️ **REL-003 — race window is real.** Single-worker + sequential webhook handling makes it narrow today. |
| **Webhook and poller ingest the same message** | The `_processed_messages` set is **one-directional**: `handle_new_fan_message` (poller) and `/generate-suggestions` check *and* add; `/webhook/fansly` does **neither**. The actual protection is `save_message`'s check-then-insert on `(fan_id, creator_id, fansly_message_id)` plus the webhook's own pre-check. Both are non-atomic. | ⚠️ **REL-002 — depends entirely on an unverified DB unique constraint.** |
| **Railway restarts after an external send but before the local DB save** | *Proactive path:* `services/proactive.py` writes a delivery journal (`payload._delivery` with `text` + `started_at`) **before** sending, and on retry calls `_reconcile_ambiguous_delivery` to find the already-accepted message by exact normalised text within the last 5 pages. No duplicate. *Auto path:* `deliver_scheduled_auto_reply` detects `status == "PROCESSING"`, runs `sync_recent_fan_messages` to import the sent message, then checks for a creator message newer than the trigger. No duplicate — **unless more than 10 messages have passed**, since `list_chat_messages` is capped at `limit=10`. | ✅ **Both handled**, with a narrow edge on the auto path. Genuinely good engineering. |
| **Two workers process the same action** | `claim_due_actions` uses a compare-and-swap UPDATE (`.eq("status", row["status"])`, plus `.eq("locked_at", ...)` for stale reclaims) and only treats a row as claimed if the update returns data. | ✅ Correct optimistic lock. |
| **Creator disconnected mid-send** | `send_fansly_message` requires a platform message id; absence raises. Auto path freezes the fan (`ppv_send_failed`) for PPV, returns silently for text. | ⚠️ Text failures are silent — see REL-005. |
| **Vault item removed while a PPV action references it** | `plan[idx]["media_ids"]` is authoritative and the writer cannot alter it, but nothing revalidates that the media still exists before sending. API Fansly rejects → freeze. | ✅ Degrades to freeze, not to a wrong send. |
| **Fan list changes while Auto eligibility is queued** | `process_incoming_fan_message` re-reads `fan_list_members` at evaluation time. | ✅ Handled. |
| **Human sends a message while an Auto reply is scheduled** | `_should_still_send` for AUTO_REPLY rejects on *any* message newer than `trigger_sent_at` (not just fan messages), and `_debounced_auto_reply` re-checks post-generation. | ✅ Handled. |

### REL-001 — analyzer failure fabricates a neutral analysis (P1)

Covered in §1. `ai/situation_analyzer.py` catches both transport and parse failures and
returns `_fallback_result()`, which is a complete, well-formed, **wrong** analysis. The
only backstop is the `_looks_like_self_harm` regex, applied after. Downstream,
`_debounced_auto_reply` uses `situation["purchase_signal"]`, `crisis_signal`,
`resend_requested`, and `strategic_move` to set decline locks, route the writer, and
decide whether to sell. The commercial orchestrator's own failure path is correct
(`"Full Auto must fail closed ... auto reply aborted"`) — the analyzer's is not.

### REL-004 — `_processed_messages.clear()` drops all dedupe state at once (P3)

```python
_processed_messages.add(message_id)
if len(_processed_messages) > 1000:
    _processed_messages.clear()
```

At the 1,001st message the entire dedupe set is discarded, so a poller message the
webhook handled moments earlier can be reprocessed. `save_message`'s DB check covers
the persistence side; the exposure is a second `process_incoming_fan_message` run
(a duplicate analyzer call and a duplicate `schedule_auto_reply` — the latter is
idempotent via `cancel_actions_for_fan` + dedupe key). Low impact, trivially fixed
with a bounded FIFO.

### REL-005 — an undeliverable fan burns 8 full pipeline runs (P2)

If `_debounced_auto_reply` returns without sending for a non-PPV reason — no
`group_id`, no `apifansly_account_id`, or `generate_replies` returning `[]` — nothing
is recorded. `deliver_scheduled_auto_reply` then finds no newer creator message,
returns `False`, `_run_auto_reply` raises `RuntimeError`, and `fail_action` requeues
with `max_attempts=8` and exponential backoff capped at 60 min. Each retry re-runs the
**entire** pipeline: analyzer + writer (up to 3 attempts) + ~43 DB round trips. Worst
case for one message that is never delivered: **8 analyzer calls, up to 24 writer
generations, ~350 DB round trips.**

### Error-handling classification

Ruff reports **128 `except Exception` blocks**. Classified by intent:

- **(A) Intentionally best-effort — correct as written (~70):** telemetry writes,
  legend merge, typing indicator, `find_similar_exchanges`, `_record_failure`,
  history lookup inside `_should_still_send`, inactivity scheduling after a confirmed send.
  Several carry explicit comments explaining why. Good.
- **(B) Should emit structured telemetry but continue (~35):** every
  `print(f"[X ERROR] {e}")` in the schedulers and vault pipeline. These are the only
  record that anything failed, and they are unstructured stdout with no counters.
- **(C) Should retry (~8):** `[SYNC MESSAGES ERROR]` skips the chat until the next
  pass; `[FANSLY LISTS ERROR]` returns `{"status":"error"}` that no caller inspects.
- **(D) Should surface degraded state (~10):** `_crisis_freezes_chat`'s
  `except: caps = {}` silently defaults `crisis_policy` to `"continue"` when the
  creator-caps read fails — a creator who configured `freeze` gets `continue` during a
  DB blip. `_within_daily_caps` returns `(True, "")` on config-read failure with the
  comment "never block selling on a config-read failure" — a defensible but explicit
  fail-open on a spend cap.
- **(E) Dangerous silent failure (~5):** `_fallback_result` (REL-001);
  `tips.received` returning `{"status":"queued_for_reconciliation"}` when **nothing is
  queued** — no action row, no table write, just a `print`; `connect_creator` returning
  HTTP **200** with `{"success": false}`; `_debounced_auto_reply`'s outer
  `except Exception: print + traceback` swallowing a mid-delivery failure.

### Where Cleopatra reports success without it being true

| Reported | Reality |
|---|---|
| `POST /webhook/fansly` → `{"status": "queued_for_reconciliation"}` for `tips.received` | Nothing is queued. The tip is dropped. |
| `POST /connect-creator` → HTTP 200 `{"success": false, ...}` | Failure delivered as success at the HTTP layer. |
| `POST /reply` → 200 after a Fansly send but before `save_message` returns | If `save_message` raises, the operator sees a 500 **after the message was already sent**, and a retry sends it twice (the second time `save_message`'s dedupe read would catch it only if a platform id was returned — which `/reply` does not require, unlike `send_fansly_message`). |
| `GET /health` → `{"status": "ok"}` | Static. Never touches the database, the schedulers, the queue, or any provider. |
| `sync_chats` → `{"status": "ok", "lists": {"status": "error", ...}}` | Nested failure that no caller inspects. |

---

## 12. SECURITY / TENANCY REPORT

### What is right

`db/tenant_isolation_v1.sql` is the strongest piece of security work in the codebase.
It defines `can_access_creator` / `can_access_fan` / `can_access_fan_list` as
`security definer` functions, revokes them from `public`, and then **auto-discovers**
every base table with a `creator_id` (or a `fan_id` without a `creator_id`) and applies
a policy — rather than maintaining a hand-written list. `fan_list_members` requires
**both** fan and list access so a guessed list UUID cannot cross tenants. Route-level,
`core/tenancy.py` provides seven `require_*` helpers with per-request caching of the
operator's creator set, and **66 of the 69 routes carry one**.

### SEC-001 — RLS grants browsers full write access to every creator-owned table (P1)

Every generated policy is:

```sql
create policy tenant_creator_membership on public.<table>
for all to authenticated
using (public.can_access_creator(creator_id::text))
with check (public.can_access_creator(creator_id::text));
```

`for all` is SELECT **and** INSERT, UPDATE, DELETE. The dashboard authenticates as
`authenticated` with the anon key plus the operator's JWT. Therefore any operator, from
the browser console, can directly:

- `UPDATE fans SET total_spent = ..., needs_human_review = false, pending_ppv_check = ...`
- `INSERT INTO messages ...` (fabricating conversation history that the writer will use)
- `UPDATE ppv_deliveries SET status = 'purchased'` (bypassing the ledger's single-flight)
- `UPDATE scheduled_actions SET status = 'CANCELLED'` or insert new ones
- `UPDATE creators SET auto_mode = true, persona = ..., auto_audience_policy = ...`
- `DELETE FROM ppv_approval_requests` (bypassing the operator approval gate)
- `INSERT INTO model_usage_events` (poisoning cost accounting)

Every business rule the backend enforces — approval gates, PPV single-flight, commercial
state transitions, daily caps — is enforced **only in Python**, and the browser has a
direct path around it. The dashboard needs SELECT on ~12 tables plus narrow writes on
`fan_lists` / `fan_list_members`. It should not have `for all` on the other ~30.

### SEC-002 — `preview_auto_audience` reads every agency's list memberships (P1)

```python
asyncio.to_thread(
    lambda: db.table("fan_list_members")
    .select("fan_id, list_id, fan_lists(exclude_from_auto, creator_id)")
    .execute()          # ← no .eq(), no .limit(), service-role credentials
)
```

No creator filter at all. Under the service role, RLS does not apply. The rows are
filtered in Python afterwards (`if str(joined.get("creator_id")) != str(creator_id): continue`),
so nothing cross-tenant is *returned* — but every agency's membership data is read into
this request, and the correctness consequence is worse than the privacy one: PostgREST
caps the result at 1,000 rows **globally**, so past ~1,000 total memberships across all
agencies the requesting creator's own rows are likely **absent**, `memberships` is empty,
`legacy_exclusions` is empty, and **the exclusion policy silently stops applying** in
the preview shown to the operator. This endpoint is called on every load of both
`/analytics` and `/settings`.

### SEC-003 — `fan_conversation_summaries` is a view and the RLS migration skips views (P1, needs verification)

`app/page.tsx` reads the sidebar directly from
`supabase.from('fan_conversation_summaries')`. The tenancy migration's discovery loop
filters on `t.table_type = 'BASE TABLE'`, so **no policy is ever created for this view**.
A Postgres view executes with the **definer's** privileges unless created with
`WITH (security_invoker = on)` (PG15+), which would mean RLS on the underlying `fans`
and `messages` is not applied and any authenticated operator could read every agency's
conversation summaries by changing one filter. **The view's definition is not in the
repository, so this cannot be confirmed here — verify it directly and immediately:**

```sql
SELECT relname, reloptions FROM pg_class WHERE relname = 'fan_conversation_summaries';
-- expect reloptions to contain security_invoker=true
```

### SEC-004 — the auth default is fail-open (P2)

`core/auth.py::_is_dev()` is `os.environ.get("APP_ENV", "development") == "development"`.
The **default** is development. If `APP_ENV` is ever missing or misspelled in the Railway
environment:

- `api_auth_middleware`: unconfigured `DASHBOARD_API_SECRET` → **request allowed**.
- `authenticated_dashboard_user`: no bearer token → returns `None` instead of raising.
- `require_creator_access`: `if not user_id and _is_dev(): return` → **all tenancy checks
  become no-ops**.
- `/webhook/fansly`: no signing secret → signature check skipped entirely.

This is a single environment variable between a correct deployment and a completely open
one, with the unsafe value as the default. `.env.example` does set `APP_ENV=production`,
and the header comment documents the fail-closed intent — the intent is right, the
default is inverted.

### SEC-005 — `/test/*` endpoints are live in production (P2)

Neither `POST /test/simulate-ppv-purchase` nor `POST /test/inject-message` is guarded by
`_is_dev()`. Both are tenancy-checked, so this is an *operator* capability, not an
anonymous one — but:

- `simulate_ppv_purchase` lets any operator forge a purchase: it writes `total_spent`,
  `spend_tier`, and `sales_log`, feeding price learning, lifecycle, writer routing
  (`high_value_fan`), and revenue reporting.
- **It also destroys data.** It selects `"pending_ppv_check, total_spent, active_session, ai_summary"`
  and then reads `fan_data.get("sales_log")` — a column it never selected — which is
  always `[]`. It then writes `sales_log` back as a **single-element list**, wiping the
  fan's entire sales history. (`main.py:5459` vs `5481`.) This is a confirmed data-loss
  bug on a production-reachable endpoint.
- `test_inject_message` calls `process_incoming_fan_message(..., auto_mode=True, ...)`
  with `auto_mode` **hardcoded True**, so it can trigger a real Full Auto send to a real
  fan for a creator whose Auto mode is off.

### SEC-006 — `NEXT_PUBLIC_API_KEY` ships the "dashboard secret" to every browser (P2)

`lib/api.ts` reads `process.env.NEXT_PUBLIC_API_KEY`. Next.js inlines `NEXT_PUBLIC_*`
into the client bundle, so `DASHBOARD_API_SECRET` is readable by anyone who fetches the
Vercel JS. The code comment is honest about this — *"Deployment identifier retained for
defense in depth. Creator authorization is enforced by the signed-in Supabase access
token"* — and that is accurate: the real boundary is the JWT, which the middleware
validates. The finding is that the variable's **name implies a secret it is not**, and
rotating it requires a dashboard redeploy. Rename it, or drop the header and rely on
the JWT alone.

### SEC-007 — middleware uses `getSession()` rather than `getUser()` (P3)

`proxy.ts` gates navigation on `supabase.auth.getSession()`, which decodes the cookie
without server-side verification. A forged cookie passes the redirect gate. It does not
grant data access (RLS and the backend both validate the JWT), so the impact is limited
to reaching a shell that then fails every query. `getUser()` is the documented correct
call for server code.

### Secrets and PII in logs

Reviewed all `print()` statements on the credential paths. **No API keys, passwords,
tokens, or session cookies are logged** — `connect_creator` logs only name/country,
`raise_for_response` truncates provider bodies to 200 characters, and
`services/apifansly.py::_record_usage` is documented and implemented as secret-free.

However, **fan message content is logged at INFO level in at least four places**:

```
main.py:4033   [WEBHOOK] ... content={message_content[:30]}
main.py:4348   [WEBHOOK] ... content={message_content[:50]}
main.py:530    [POLLER] ... content={content[:80]}
services/suggestions.py:1376  [AUTO REPLY] Sent part {i+1}: {text_out[:50]}
```

For an adult-content product these are sensitive user communications flowing into
Railway's log retention. Prompts themselves are not logged (good), and
`model_usage_events` stores token counts, not text (good).

---

## 13. CI / TEST COVERAGE GAPS

### Measured baseline

| | Backend | Dashboard |
|---|---|---|
| Tests | **457 passed, 10 skipped, 6.73 s** | **10 passed, 0.5 s** |
| Test files | 68 | **1** (`lib/__tests__/fanLists.test.ts`) |
| Lint | `ruff` → **339 findings** (128 blind-except, 67 unsorted-import, 5 loop-variable-capture); with `--select E,F` → **713** (681 line-length) | `eslint` → **111 warnings, 0 errors** |
| Type check | **none configured** | `tsc --noEmit` clean (run inside `next build`) |
| CI runs | `pytest -q` **only** | `lint` + `test` + `build` |

### The gap between test count and coverage

**457 backend tests in 6.7 seconds means essentially no integration coverage.** The
suite is monkeypatch-based at function boundaries — `tests/test_durable_auto_reply.py`
replaces `cancel_actions_for_fan`, `schedule_action`, `get_fan_session`,
`get_creator_sleep_hours`, and `build_availability_delay` before asserting on a dict.
That verifies orchestration wiring, which is worth having, but it cannot catch:

- **A wrong PostgREST query shape.** Every one of the 43 unpaginated selects in §5e
  passes its tests, because no test issues a real query. `services/fansly_lists.py`'s
  destructive truncation bug is fully test-covered at the reconciliation-logic level
  (`tests/test_fansly_lists.py`, 622 lines) and completely invisible to it.
- **`select("...")` / `row.get("...")` mismatches.** The `simulate_ppv_purchase`
  `sales_log` data-loss bug is exactly this class.
- **Missing indexes or constraints on the base tables**, which have no SQL in the repo.

### Invariant-to-test mapping

| Production invariant | Protected by a test? |
|---|---|
| One Auto reply per fan message | ✅ `test_durable_auto_reply.py` (mocked) |
| No duplicate proactive send across a crash | ✅ `test_proactive_delivery_recovery.py` |
| PPV single-flight per fan | ✅ constraint + `test_operator_ppv_contract.py` |
| Fan return cancels pending approvals | ✅ `test_pending_offer_continuity.py` |
| Tenant isolation on routes | ✅ `test_tenant_isolation.py` (89 lines) |
| Writer route selection | ✅ `test_writer_router.py`, `test_openrouter_writer_route.py` |
| Prompt cache prefix ordering | ✅ `test_prompt_cache_structure.py` — **but it does not assert that `cache_control` survives to the transport**, which is why COST-002a went unnoticed |
| **Webhook + poller duplicate ingestion** | ❌ **none** |
| **Two concurrent purchase webhooks** | ❌ **none** |
| **Concurrent claim of the same scheduled action** | ❌ **none** |
| **Restart mid-`_run_vault_sync` / mid-categorisation** | ❌ **none** |
| **PostgREST 1,000-row truncation on any path** | ❌ **none** |
| **RLS actually blocking a cross-tenant browser read** | ❌ **none** (route-level only) |
| **Provider response-shape changes** (OpenRouter usage fields, API Fansly envelopes) | ⚠️ partial — `test_model_telemetry_openrouter.py` covers usage parsing; API Fansly shapes are covered by `test_apifansly_integration.py` against fixtures |
| **Any dashboard race condition, effect, or realtime behaviour** | ❌ **none — 9,900 lines of TSX, 0 component tests** |
| **`dedupeMessages`** (the measured hot spot) | ❌ **none** |

### CI configuration gaps

**Backend CI runs `pytest -q` and nothing else.** `ruff==0.16.0` is installed via
`requirements-dev.txt` and **never invoked**. There is no `python -m compileall`, no
type checker, and no check that `db/*.sql` has been applied. The Postgres service is
provisioned and `TEST_DATABASE_URL` is set — good — but only 10 tests use it.

**Dashboard CI is stronger** (lint + test + production build) but `npm run lint`
passes with 111 warnings because nothing is configured as an error, and `npm test`
covers one 55-line file.

**Version drift:** CI builds the backend on Python 3.12; `requirements.txt` pins no
Python version and Railpack's default may differ. There is no `runtime.txt` or
`.python-version`.

---

## 14. OBSERVABILITY GAPS

The five questions from the brief, answered honestly against the current code.

### "Why did this reply take 11 seconds?" — **cannot be answered**

`model_usage_events` records `latency_ms` per model call. That covers 2 of the ~8
contributors. There is **no instrumentation at all** for:

| Contributor | Instrumented? |
|---|---|
| Scheduler poll wait (0–60 s, often the largest single term) | ❌ |
| `_should_still_send` revalidation (4–7 DB round trips) | ❌ |
| History/context load (11-way gather) | ❌ |
| Deterministic refresher chain (15 DB round trips) | ❌ |
| Analyzer model call | ✅ `latency_ms` |
| Writer model call | ✅ `latency_ms` (whole-call; **no TTFT**, even though `target.stream` exists) |
| Simulated typing delay (**6.8 s MEASURED**) | ❌ (printed to stdout, not stored) |
| API Fansly send (incl. fresh TLS handshake) | ❌ (`_USAGE_EVENTS` is in-memory and per-process) |
| DB persistence | ❌ |

An 11-second reply is most likely ~7 s of *deliberate* typing simulation plus ~4 s of
model time — and there is no way to tell that from the data, which means the first
instinct will be to optimise the wrong thing.

### "Why wasn't a message sent?" — **partially**

`scheduled_actions.last_error` and `status` are durable and queryable, and
`GET /creator/{id}/full-auto-health` surfaces them. But a skip taken by
`_should_still_send` is only `print`ed (`[SCHEDULED] skip ... : {reason}`) — the row is
marked `COMPLETED` and the reason is lost. So "the follow-up never went out" is
answerable; "**why** it never went out" is not.

### "Why was a PPV duplicated?" — **yes**

`ppv_deliveries` is a proper durable ledger with `reference`, `source`, `claimed_at`,
`delivered_at`, `platform_message_id`, `last_error`, and `metadata`, plus the partial
unique index. This is the best-instrumented subsystem in the product.

### "Why did vault sync stop?" — **no**

`_vault_sync_state` and `_categorize_state` are in-process dicts, lost on restart.
`GET /sync-vault-status/{id}` returns `{"status": "idle"}` for a job that was killed
mid-run. There is no durable job record, no start/stop/error history, and no counter.

### "Why did OpenRouter fall back?" — **yes, and this part is well built**

`ModelResult.upstream_provider` is parsed from the response and stored in
`model_usage_events.metadata.upstream_provider`, alongside `writer_route`,
`writer_route_reason`, `writer_attempt`, `writer_attempt_role`, `writer_fallback_used`,
`cost_source`, `cached_input_tokens`, and `cache_hit_ratio`. That is exactly the right
metadata set and it answers both "did the pin hold" and "did caching work".

### "Why is one creator slow?" — **no**

`creator_id` is on `model_usage_events`, but the four single-column indexes on that
table make a per-creator time-range query a scan. Nothing else in the system is
tagged per creator: no per-creator queue depth, no per-creator API Fansly usage, no
per-creator error rate.

### The three highest-value instrumentation additions

1. **A durable per-reply timing record.** One row per Full Auto/Assisted reply with
   `queue_wait_ms`, `revalidate_ms`, `context_ms`, `analyzer_ms`, `refreshers_ms`,
   `writer_ms`, `typing_ms`, `send_ms`, `persist_ms`, `db_round_trips`. Everything
   needed is already measured internally and thrown away.
2. **A real `/health`.** Today it is a static dict. It should report: DB reachable,
   seconds since each scheduler's last completed cycle (`worker_health_snapshot()`
   already exists and is not exposed), `scheduled_actions` PENDING count and oldest
   `execute_at`, and model-availability status. Railway's healthcheck currently cannot
   detect a dead scheduler or an unreachable database.
3. **Structured logging.** 128 `print()` calls with ad-hoc `[TAG]` prefixes are the
   only record of most failures. A `logging` call with `extra={"creator_id":..., "fan_id":...}`
   would make them queryable — and would let the four message-content log lines be
   dropped or redacted.

---

## 15. CAPACITY ENVELOPE

Every value is labelled. Nothing here is presented as a benchmark that was not run.

| Metric | Current safe estimate | Label | Failure mode when exceeded |
|---|---|---|---|
| **Concurrent AI generations** | **1** | **MEASURED** (code structure) | Not a degradation — an architectural fact. `process_once` awaits each action in a `for` loop. |
| **Full Auto replies / minute (deployment-wide)** | **3–6** | CALCULATED from 43 DB round trips + 2 LLM + 6.8 s typing, MEASURED | Queue grows; per-reply latency grows linearly |
| **Full Auto reply latency (light load)** | **45–90 s** end to end | CALCULATED | — |
| **Full Auto reply latency (20 due)** | **5–7 min** | CALCULATED | — |
| **Concurrent HTTP requests** | **50–200** for I/O-bound routes | INFERRED | Single Uvicorn worker; async I/O handles this fine until a blocking call lands |
| **Concurrent blocking DB calls** | **`min(32, cpu_count+4)`** | CALCULATED | Default asyncio executor; **274 `to_thread` call sites, 0 custom executors** (MEASURED). On Railway `os.cpu_count()` reports host cores, so likely **32** — but the CPU *quota* is the real limit |
| **DB ops / second** | **~30–60 sustained** from this process | INFERRED (32 threads ÷ ~25–50 ms/round trip, minus CPU contention) | Thread-pool queueing; every `await` on a DB call blocks behind it |
| **DB ops per Full Auto reply** | **~50** | MEASURED (43) + CALCULATED (worker overhead) | — |
| **DB ops per Assisted suggestion** | **9** (flags off) / **34** (flags on) | **MEASURED** | — |
| **Fan messages / minute absorbed (ingestion only)** | **60–150** | CALCULATED (17 DB round trips each) | Webhook latency rises; API Fansly retries |
| **Fan messages / minute answered by Full Auto** | **3–6** | CALCULATED | **This is the binding constraint** |
| **Fan messages / second** | **~1–2.5 ingest, ~0.1 answered** | CALCULATED | — |
| **Active Full Auto conversations** | **~40–60** at a ~5 min reply SLA | CALCULATED | Beyond this, replies queue behind each other |
| **Creators (connected)** | **20–40** | INFERRED | Chat reconciliation is sequential across creators; head-of-line blocking |
| **Agencies** | **2–5** | INFERRED | No per-tenant isolation of the shared worker |
| **Scheduled actions / minute processed** | **3–6** (AUTO_REPLY) / **10–20** (cheap types) | CALCULATED | — |
| **Queue drain: 100 pending** | **~17–33 min** | CALCULATED | — |
| **Queue drain: 1,000 pending** | **~3–6 hours** | CALCULATED | — |
| **Queue drain: 10,000 pending** | **~28–56 hours** | CALCULATED | Effectively never — new work arrives faster |
| **Outstanding follow-up obligations** | **≤ ~1,000** | MEASURED (2.00 DB round trips each per 60 s) | At 5,000 the repair pass alone exceeds the cycle; durable follow-ups stop firing |
| **Vault size per creator (sync)** | **~20,000 images** | CALCULATED | Videos are the wall: ~7–24 h for 5,000 videos at `Semaphore(2)` |
| **Concurrent vault syncs** | **1–2 safely; unbounded in code** | INFERRED | `sync_vault_start` spawns and returns — 100 due creators = 100 concurrent syncs |
| **Conversation length (backend)** | **unbounded** — history capped at 40 fetched / 16 in prompt | MEASURED (prompt flat at ~3,500 tokens from 16 messages up) | — |
| **Conversation length (dashboard, unscrolled)** | **unbounded** — 50-message window | MEASURED | — |
| **Conversation length (dashboard, scrolled back)** | **~500 messages** | **MEASURED** | 155 ms freeze/message @ 500; 667 ms @ 1,000; 15 s @ 5,000 |
| **Conversations in sidebar** | **1,000** (PostgREST cap) | INFERRED | Truncation, ~15,000 DOM nodes, un-memoised filter |
| **Vault items in dashboard** | **~10,000** | CALCULATED | ~15 MB JSON, ~50–75 MB heap, 10 sequential requests |
| **Dashboard concurrent users** | **50–100** | INFERRED | Not the binding constraint — but each costs ~80 API Fansly calls/hour |
| **Realtime channels** | **4 per open dashboard** | MEASURED (code) | `messages-realtime`, `suggestions-{fan}`, `fan-{fan}`, `creator-legend-{creator}` |
| **API Fansly calls / creator / hour (idle)** | **~2–12** | CALCULATED | — |
| **API Fansly calls / creator / hour (active + 1 operator watching)** | **~100–200** | CALCULATED | Dominated by the dashboard's 45 s poll |
| **API Fansly calls per deploy** | **~2 × total chats** | CALCULATED | `_chat_last_message_ids` cold start (API-001) |
| **Concurrent LLM requests (own limit)** | **none** | **MEASURED** (exhaustive grep: 4 semaphores, none on model calls) | Bursts pass straight through to the provider |
| **Peak Railway memory (20 creators, no vault sync)** | **~400–700 MB** | INFERRED | — |
| **Peak Railway memory (during vault categorisation)** | **~800 MB–1.5 GB** | INFERRED | 12 concurrent images + Pillow + ONNX + `all_items` |

### CURRENT SAFE OPERATING ENVELOPE

> **With the architecture as audited, I would be comfortable putting approximately
> 15–25 creators, across 1–2 agencies, with roughly 30–50 simultaneously active Full
> Auto fan conversations, on this deployment — provided that:**
>
> 1. **Full Auto reply latency of 1–3 minutes is acceptable to the agency.** It is not
>    a fast product today, by design (the typing simulation is deliberate) and by
>    accident (the 60 s poll and the sequential worker are not).
> 2. **No creator exceeds ~1,000 fans**, or the six unpaginated selects in §5e begin
>    returning wrong answers — including the Fansly Lists sync **deleting correct
>    memberships** (SCALE-003), which is destructive and self-repeating.
> 3. **`FANSLY_LISTS_SYNC_ENABLED` stays off** until SCALE-003 is fixed.
> 4. **`APP_ENV=production` is verified set** in Railway (SEC-004), and
>    `fan_conversation_summaries` is verified as `security_invoker` (SEC-003).
> 5. **Vault categorisation is run deliberately, off-peak, one creator at a time** —
>    it shares CPU and the thread pool with live chat and there is no global cap
>    (VAULT-001).
> 6. **Operators do not routinely scroll back more than ~10 pages** in a conversation
>    (FE-001), and vaults shown in the dashboard stay under ~10,000 items (FE-003).
> 7. **Someone watches the `scheduled_actions` PENDING count**, because nothing in the
>    product surfaces a growing backlog today.

### FIRST BOTTLENECK

**The sequential scheduled-actions worker (SCALE-001).** It is reached at roughly
**40–60 simultaneously active Full Auto conversations**, well before any database,
provider, or HTTP limit. It is a pure code-structure limit — the machine is idle while
it waits — and it is deployment-wide, so it is also the mechanism by which one agency
degrades another.

### SECOND BOTTLENECK

**`repair_followup_obligations` (SCALE-002).** At 2.00 DB round trips per outstanding
obligation **every 60 seconds** (MEASURED), 1,000 obligations is 2,000 round trips per
minute of pure overhead before any work is done, and around 5,000 obligations the
repair pass alone exceeds the cycle time — at which point durable follow-ups
effectively stop firing. This arrives at roughly **50–100 creators** with normal
follow-up usage.

### THIRD BOTTLENECK

**The PostgREST 1,000-row cap**, which is not a performance limit at all — it is a
correctness cliff that is hit **per creator**, not per deployment, and produces silent
wrong answers and destructive writes rather than errors.

### WHAT MUST CHANGE BEFORE 5× SCALE (~100 creators, ~250 active conversations)

1. **Make the scheduled-actions worker concurrent with a bounded pool.** Claim a
   larger batch and run N actions through `asyncio.gather` behind a
   `Semaphore(N)` (start at 8–12), and **poll immediately again when the previous batch
   was full** instead of always sleeping 60 s. This alone moves the ceiling from ~5 to
   ~50 replies/minute.
2. **Add an explicit LLM concurrency limiter** (§17E) so (1) cannot open 50
   simultaneous paid model requests.
3. **Fix `repair_followup_obligations`**: filter to `next_followup_at <= now() + interval '5 minutes'`,
   and replace the per-row `ensure_action_pending` with a single bulk upsert.
4. **Paginate or aggregate all six truncating queries** in §5e, and add a lint rule or
   review checklist item for `.select(...).execute()` without a bound.
5. **Batch `claim_due_actions`** into one CAS UPDATE with `RETURNING` instead of 20.
6. **Cap concurrent vault syncs** with a global semaphore, and batch the per-item
   classification writes.
7. **Fix `dedupeMessages`** and cap retained dashboard history.
8. **Use `thumbnail_url` in the vault grid**, and fetch vault media per album.
9. **Add a real `/health`** exposing queue depth and scheduler liveness, plus the
   per-reply timing record — you cannot operate at 5× without them.
10. **Reuse one `httpx.AsyncClient`** for API Fansly instead of creating one per call.

### WHAT MUST CHANGE BEFORE 20× SCALE (~500 creators, ~1,000+ active conversations)

1. **Split the process.** `web` (FastAPI, N workers, no schedulers) / `worker`
   (scheduled actions, horizontally scalable — the `claim_due_actions` CAS already
   supports multiple claimers) / `scheduler` (single leader for the cron loops, or a
   DB advisory lock). **This is the change that removes the single-worker constraint
   entirely, and none of the current schedulers are safe to duplicate without it.**
2. **Move media processing out of the web process** — a separate service or a Modal
   job. Vault work and chat latency must stop sharing CPU and a thread pool.
3. **Move all in-process state to Postgres or Redis**: `_chat_last_message_ids`,
   `_vault_sync_state`, `_categorize_state`, `_processed_messages`,
   `_active_chat_binding_retry_after`, `_pending_auto_replies`. (`redis` is already a
   declared dependency and Upstash credentials are already required env vars — both
   currently unused.)
4. **Introduce per-agency fairness** in the action queue. A single global FIFO means
   one agency's burst delays every other agency's fans.
5. **Consider a proper async Postgres driver.** 274 `asyncio.to_thread` call sites
   through a shared thread pool is the wrong shape at this scale; `asyncpg` or
   `psycopg3` async would remove the thread ceiling entirely.
6. **Partition or archive `messages`**, and add a materialised conversation-summary
   table refreshed on write rather than a view read live by every browser.
7. **Per-tenant API Fansly budgeting**, with usage persisted rather than in-process.

---

## 16. SCALING LIMITS BY TIER

Active-fan assumptions: 5% of a creator's fans messaging in any given hour, 20% of
those in Full Auto — deliberately conservative, and **stated as an assumption, not a
measurement**.

### 1 agency / 20 creators — **likely safe**

~2,000 fans/creator → ~100 active/hour/creator → ~2,000 messages/hour deployment-wide
→ ~33/min ingest, ~7/min needing a Full Auto reply.

- **Ingestion**: fine (~560 DB ops/min at 17 each).
- **Full Auto**: **at the edge.** 7/min against a 3–6/min ceiling means a slowly
  growing backlog during peak hours that drains overnight.
- **Chat reconciliation**: 20 creators × sequential `sync_chats` — fine if each takes
  <15 s.
- **Expected failure mode**: reply latency drifts from ~1 min to ~5 min at peak.
- **Change before this tier**: nothing structural. Verify `APP_ENV`, verify the
  summaries view, keep Fansly Lists off.

### 5 agencies / ~100 creators — **first hard bottleneck**

~10,000 messages/hour → ~167/min ingest, ~33/min Full Auto.

- **Full Auto**: **6× over capacity.** Backlog grows ~27 actions/min and never drains.
- **`repair_followup_obligations`**: ~1,000–2,000 obligations → 2,000–4,000 DB round
  trips/min of pure overhead.
- **Chat reconciliation**: 100 creators sequentially; if any creator's `sync_chats`
  is slow (large chat list, or an inline `process_incoming_fan_message` with 2 LLM
  calls — which it does), the pass exceeds the tick and creators are skipped.
- **Expected failure mode**: Full Auto replies arrive hours late or not at all;
  operators see "Auto is on" with nothing happening; no alert fires because
  `/health` is static.
- **Change before this tier**: all ten items in the 5× list, especially (1), (2), (3).

### 10 agencies / ~200 creators — **needs the process split**

~20,000 messages/hour → ~333/min ingest, ~67/min Full Auto.

- **DB ops**: ~5,700/min ingestion alone ≈ 95/s, against an inferred ~30–60/s ceiling
  from one process's thread pool. **The database access layer becomes the second wall.**
- **Thread pool**: sustained saturation; every `to_thread` queues, so p95 latency on
  *every* route rises together.
- **Vault**: 200 creators × 24 h anniversaries ≈ 8 concurrent syncs/hour average, with
  clustering spikes far higher and no cap.
- **Expected failure mode**: correlated latency across all routes, followed by API
  Fansly webhook timeouts and redelivery — which arrives as *more* load.
- **Change before this tier**: the web/worker/scheduler split, media processing moved
  out, state moved out of process.

### 25 agencies / ~500 creators — **needs the 20× architecture**

~50,000 messages/hour → ~833/min ingest, ~167/min Full Auto.

- Single-process operation is not viable at any tuning.
- **API Fansly** becomes a first-order cost: ~500 creators × 2,000 chats ≈ **1,000,000
  `list_chat_messages` calls after every deploy** from API-001 alone.
- **`fan_list_members` and `messages` need real index review**, which cannot happen
  until the base schema is in version control.
- **Expected failure mode**: provider rate limits and credit exhaustion, before CPU.

### Materially larger

Requires per-tenant sharding or queue partitioning, a materialised conversation
summary table, `messages` partitioning by time, and per-agency provider budgets. Not
worth planning in detail until the 20× architecture is in place and measured.

---

## 17. BURST / OVERLOAD BEHAVIOUR

### A. Sudden fan-message spike (50 messages in 5 s)

**Ingestion**: 50 concurrent webhook requests, each ~17 DB round trips = 850 blocking
calls queued into a `min(32, cpu+4)`-thread pool. Webhook latency rises to
seconds. API Fansly's webhook timeout is **UNKNOWN**; if it is exceeded, deliveries are
retried, and the retry passes the `messages.fansly_message_id` pre-check only if the
first attempt already inserted — **a check-then-insert race under exactly the
conditions that create it**. Whether this produces duplicate rows depends on the
unverified unique constraint (REL-002).

**Delivery**: 50 `AUTO_REPLY` rows land. The worker claims 20, processes them
sequentially over ~4–6 minutes, sleeps 60 s, claims the next 20. **The 50th fan waits
~12–15 minutes.** Work is not dropped — it queues durably, which is the right failure
mode — but there is no backpressure signal anywhere: no queue-depth metric, no
health degradation, no operator warning.

### B. LLM provider throttling

`OPENROUTER_ALLOW_FALLBACKS=false` with `provider.only=["Inceptron"]` means a 429 or
503 is returned rather than rerouted. `generate_replies` then retries **twice more
immediately with no delay**, against the same throttled provider. **3× amplification
into a provider that is already shedding load.** After three failures it returns `[]`,
`_debounced_auto_reply` returns silently, and `fail_action` requeues with exponential
backoff (5, 10, 20, 40, 60, 60, 60, 60 min) up to 8 attempts — each of which repeats
the 3× burst. `record_model_transport_failure` feeds
`services/model_availability.py`, which surfaces in `/model-runtime-health` and the
dashboard's `SystemHealthBanner`, so the *operator* sees it. The *system* does not
back off.

### C. Database slowdown

Every DB call is a blocking client call in `asyncio.to_thread`. When Supabase p99 rises,
the thread pool fills, and because it is **shared with NudeNet inference, Pillow
processing, and temp-file I/O**, all of them contend. There are no timeouts on Supabase
calls, so a hung connection holds a thread until the client's own default fires. The
event loop stays responsive — FastAPI keeps accepting requests — which makes it *worse*:
requests accumulate behind a saturated pool with no admission control.

### D. API Fansly slowdown

Sends have `timeout=30` (the `request()` default), chat listing the same. A slow API
Fansly stalls the sequential `_reconcile_chat_creators_once` loop — one slow creator
delays every other creator's reconciliation, because the loop is sequential and
awaits `sync_chats` per creator. In the auto path, a slow send holds the entire
scheduled-actions worker.

### E. Worker backlog

`claim_due_actions` reclaims `PROCESSING` rows whose `locked_at` is older than
**10 minutes**. A single action that legitimately takes longer than 10 minutes — a slow
provider plus a long typing simulation plus a slow send is not far off — is
**reclaimed and re-run while still executing**. For `AUTO_REPLY` this is caught by
`deliver_scheduled_auto_reply`'s `status == "PROCESSING"` reconciliation path; for
`PAYDAY_REENGAGEMENT` and the other proactive types it is caught by the delivery
journal. Both defences exist and both are good — but they are the *only* thing
between a slow provider and a duplicate message to a real fan.

### F. Railway restart during a backlog

- `scheduled_actions`: durable. `PROCESSING` rows are reclaimed after 10 min. ✅
- `fan_commercial_states` obligations: repaired by `repair_followup_obligations`. ✅
- In-flight `_run_vault_sync` / `_run_vault_categorization`: **lost, with the
  dashboard reporting `idle`.** Recovers on the next hourly pass. ⚠️
- `_chat_last_message_ids`: lost → **full message re-sync of every chat** (API-001). ⚠️
- `_processed_messages`: lost → webhook/poller dedupe falls back to the DB check. ⚠️
- In-flight `spawn()`ed tasks (`learn_from_fan_message`, `_update_fan_memory`,
  `_enrich_fan_profile`, `sync_chats_background`, telemetry writes): **silently
  dropped**, no record, no retry. ⚠️

### Cascading-failure assessment

**The good news: Cleopatra's dominant failure mode is a slow queue, not a cascade.**
The durable `scheduled_actions` table, the compare-and-swap claim, the delivery
journals, and the "freeze rather than duplicate" persistence semantics mean that under
overload work **backs up rather than multiplying**. That is the right shape and it was
clearly designed for.

**Three real cascade paths exist:**

1. **Writer retry into a throttled provider** (§17B): 3× amplification with zero delay,
   repeated across up to 8 durable retries, against a deliberately un-fallback-able
   pinned provider. This is the most likely cascade.
2. **Webhook timeout → provider redelivery → duplicate processing**: because the entire
   pipeline (including two LLM calls in the Assisted path) runs **inside the webhook
   HTTP request**, a slow model makes the webhook slow, which makes the provider retry,
   which creates more load and a duplicate-ingestion race. **Returning 200 immediately
   and doing the work in a durable action would eliminate this entirely** and is the
   single highest-value reliability change available.
3. **Unbounded vault sync fan-out** (VAULT-001): 100 creators' anniversaries landing in
   one hourly pass starts 100 concurrent syncs at 12-way concurrency each, saturating
   CPU, memory, and the thread pool — which slows the DB calls of every chat request,
   which slows the webhook, which triggers path (2).

**Missing across all three: admission control.** There is no concurrency limit on
generations, no queue-depth-aware degradation, no shed-load path, and no health signal
that reflects backlog. The system will always try to do everything.

---

## 18. RECOMMENDED FIX ORDER

### SPRINT A — must fix before agency beta

| ID | Fix | Risk | Benefit |
|---|---|---|---|
| SCALE-003 | Paginate `fans` / `fan_list_members` in `services/fansly_lists.py::_load_state`, or keep `FANSLY_LISTS_SYNC_ENABLED=false` | Low | Prevents destructive, self-repeating membership loss |
| SEC-003 | Verify `fan_conversation_summaries` is `security_invoker=true`; add views to the tenancy migration | Low | Closes a possible cross-tenant read |
| SEC-004 | Make `APP_ENV` required (fail closed on absence) rather than defaulting to `development` | Low | Removes a one-variable path to a wide-open deployment |
| SEC-005 | Gate `/test/*` behind `_is_dev()`; fix the `sales_log` data-loss bug | Low | Stops forged purchases and history destruction |
| SEC-002 | Add `.eq()` on the creator and paginate in `preview_auto_audience` | Low | Removes the cross-tenant scan; makes the preview correct |
| REL-001 | Mark analyzer fallback results (`"analysis_degraded": true`), suppress commercial actions and Full Auto sends when set | **Medium** | Stops blind selling during provider incidents |
| COST-001 | Add jittered backoff between writer attempts; only retry transport failures, not validation failures; remove the substring bans that match legitimate copy (`"yourself"`, `"interesting"`, `"noted"`, `"got it"`) | Low | Removes 3× cost amplification and the "sends nothing" outcome |
| DB-000 | Dump the base schema (`creators`, `fans`, `messages`, `suggestions`, `chatter_creators`, the summaries view, …) into `db/000_base_schema.sql`; verify `messages.fansly_message_id` is unique and `messages(fan_id, sent_at desc)` and `chatter_creators(chatter_id)` are indexed | Low | Everything else in this report depends on being able to see the schema |
| REL-002 | Make `save_message` an idempotent upsert on the unique constraint once it is confirmed | Low | Removes the check-then-insert race under retry |

### SPRINT B — fix during beta

| ID | Fix | Risk | Benefit |
|---|---|---|---|
| SCALE-001 | Bounded-concurrency scheduled-actions worker + immediate re-poll on a full batch | **Medium** | 5–10× Full Auto throughput |
| SCALE-004 | Global `Semaphore` on writer + analyzer calls, sized to the provider budget | Low | Prevents a burst opening dozens of paid requests |
| SCALE-002 | Filter `get_followup_obligations` by due window; bulk-upsert the repairs | Low | Removes 2,000+ round trips/min at 1,000 obligations |
| REL-006 | Return 200 from `/webhook/fansly` immediately; do ingestion + generation in a durable action | **Medium** | Eliminates the timeout→redelivery→duplicate cascade |
| FE-001 | Rewrite `dedupeMessages` with a `Map` index; cap retained history at ~500 | Low | Removes the measured 0.7–63 s main-thread freeze |
| FE-002 | Use `thumbnail_url` in the vault grid; replace the `innerHTML` `onError` with state | Low | Hundreds of MB less transfer per album view |
| COST-002 | Keep content blocks for the Anthropic transport; move the analyzer's 1,570 static tokens into `system`; reorder `fan_context_parts` durable-first | Low | Real, measurable token reduction on every message |
| PERF-006 | One shared `httpx.AsyncClient` for API Fansly (module-level, closed in `lifespan`) | Low | 100–300 ms off every external call |
| SEC-001 | Split RLS into `for select` + narrow write policies | **Medium** | Removes the browser's path around every business rule |
| OBS-001 | Real `/health` (DB, scheduler liveness, queue depth); per-reply timing record | Low | Makes everything above operable |

### SPRINT C — before scaling past ~100 creators

- Split web / worker / scheduler processes; make schedulers leader-elected.
- Move media processing out of the web process.
- Move `_chat_last_message_ids`, `_vault_sync_state`, `_categorize_state`,
  `_processed_messages` into Postgres or Redis.
- Paginate or aggregate the remaining queries in §5e; add an aggregate endpoint for
  the analytics page instead of pulling every fan's `sales_log` into the browser.
- Global cap on concurrent vault syncs; batch the classification writes.
- Per-album vault fetching in the dashboard.
- Per-agency fairness in the action queue.
- Compound index on `model_usage_events(creator_id, created_at desc)`; index on
  `fan_list_members(fan_id)`.

### LATER — cleanup and optimisation

- Delete the confirmed-dead code in §10 (one PR per group, with the `/fansly/*` router
  and the RAG chain each getting their own).
- Deduplicate `connect_creator` / `connect_creator_2fa` (~60 duplicated lines).
- Extract the vault pipeline, the webhook, and the chat-sync code out of `main.py`
  (5,799 lines) — **specifically** because `main.py` currently owns the six background
  schedulers *and* the webhook *and* the vault pipeline, which is why
  `workers/scheduled_actions.py` has to `from main import _creator_auto_availability`
  and `services/suggestions.py` has to `from main import send_fansly_message` and
  `from main import sync_recent_fan_messages` — three circular imports resolved by
  function-local `import` statements. That circularity is the concrete cost of the file
  size, not the line count itself.
- Structured logging; drop or redact the four message-content log lines.
- Consolidate `_update_fan_memory` / `_update_fan_ai_summary` / `learn_from_fan_message`
  into one extraction pass through the provider abstraction, with telemetry.
- Normalise money to integer cents everywhere.
- Consider virtualisation for the conversation list and vault grid **only if** operators
  are measured working past ~1,000 rendered rows.

---

## 19. LOW-RISK QUICK WINS

Disproportionate value, minimal risk, each independently shippable.

| Fix | Effort | Why it pays |
|---|---|---|
| Move the analyzer's static rules into the `system` argument | ~10 lines | ~1,570 cacheable tokens per message, zero behaviour change |
| Reorder `fan_context_parts` durable-first | ~10 lines | Recovers ~340 tokens of prefix (MEASURED 49% → ~61%) |
| Drop `"yourself"`, `"interesting"`, `"noted"`, `"got it"` from the writer's substring bans | 4 lines | Removes a common cause of 3× retries and silent non-delivery |
| Add `asyncio.sleep` jitter between writer attempts | 3 lines | Removes the 3× instant burst into a throttling provider |
| `if not (fan.display_name == name and fan.fansly_group_id == group_id)` before the `fans.update` in `sync_chats` | 3 lines | Eliminates 2,000 no-op UPDATEs / 10 min / creator and the resulting dashboard realtime storm |
| One shared `httpx.AsyncClient` for API Fansly | ~15 lines | 100–300 ms off every send |
| `thumbnail_url` in the vault grid | 1 line | Hundreds of MB less image transfer |
| `React.memo(Sidebar)` + `useMemo` on the filter + a `Set` for list membership | ~15 lines | Removes the per-mouse-move full-list re-render |
| Bounded FIFO for `_processed_messages` instead of `.clear()` | ~5 lines | Removes REL-004 |
| Prune `_chat_last_message_ids` and `_active_chat_binding_retry_after` on the reconcile pass | ~10 lines | Closes the two real memory leaks |
| Run `ruff check` in backend CI | 1 line | Would have caught 5 loop-variable captures, 10 unused imports, 3 unused variables |
| Add `--max-warnings 0` to the dashboard lint (after fixing the 111) | 1 line | Makes `exhaustive-deps` a gate — the exact class of bug in FE-005/FE-007 |
| `run_once` guard on `POST /creator/{id}/sync-fansly-lists` | ~10 lines | Prevents two concurrent reconciliations racing on the same memberships |
| Expose `worker_health_snapshot()` (already written, unused) on `/health` | ~5 lines | Immediate scheduler-liveness visibility |
| Delete `redis` and `recharts` | 2 lines | Smaller install, one less required env pair |

---

## 20. THINGS I EXAMINED THAT ARE ACTUALLY FINE

This section matters. Several things that look alarming from the outside are already
correctly protected, and the report would be misleading without saying so.

**Concurrency and correctness**

- **`ppv_deliveries` single-flight is genuinely correct.** The partial unique index
  `(fan_id) WHERE status IN ('claimed','delivered_pending') AND source <> 'operator'`
  enforces at-most-one automated PPV per fan **in the database**, not in Python, while
  deliberately allowing operators to send multiple manual offers. The
  `attach_pending_ppv` function uses `SELECT ... FOR UPDATE` so a fast purchase webhook
  cannot be overwritten back to pending. This is the right design.
- **The proactive delivery journal is a correct at-most-once implementation across a
  crash.** `payload._delivery` is written *before* the send with a stable `text` and
  `started_at`; on retry `_reconcile_ambiguous_delivery` searches the platform for that
  exact normalised text since that timestamp. Because the text is reused rather than
  regenerated, the match is reliable. I tried to break this and could not.
- **`claim_chat_reconciliation`** is a real database-level claim with a 9-minute
  interval — the correct answer, not an in-memory guard.
- **`claim_due_actions`** uses a proper compare-and-swap (`.eq("status", row["status"])`
  plus `.eq("locked_at", ...)` for stale reclaims) and only counts a row as claimed if
  the update returns data. Multi-worker safe today.
- **`cancel_actions_for_fan` cancels `PROCESSING` rows for `AUTO_REPLY`**, and
  `fail_action` / `complete_action` / `reschedule_action` are all `.eq("status", ...)`-guarded,
  so a cancelled-mid-flight action cannot be resurrected by its own failure handler.
  I initially flagged this as a spurious-retry bug; on inspection it is correct.
- **`_should_still_send` is thorough.** It revalidates fan existence, human-review
  freeze, fan and creator auto mode, approved-set availability, newer conversation
  activity, per-type state snapshots, dedupe-key replacement, recent-activity
  suppression, and creator sleep hours — and postpones rather than dropping when the
  reason is temporal. This is the most carefully written function in the codebase.
- **Delivery semantics are correct.** `send_fansly_message` requires a platform message
  id and treats a 2xx without one as a failure; a persistence failure after a confirmed
  send calls `freeze_fan_for_review` rather than retrying. "Freeze rather than
  duplicate" is the right trade for this product.

**Architecture**

- **The commercial layer is deterministic Python, not LLM calls.** Affordability, price
  learning, buyer lifecycle, conversation director, adaptive session planner, session
  planner, and the orchestrator make zero model calls (verified across all nine
  modules). Only two model calls sit on the user-visible path. This is the single best
  architectural decision in the system and it should be protected.
- **`_debounced_auto_reply`'s stale-generation guards are layered correctly**: an
  `expected_trigger_at` check before generation, a `_pending_auto_replies` ownership
  check before *and* after generation, and a fresh DB history comparison after
  generation. Three independent checks, each catching a different window.
- **The writer router is deterministic and never asks a model which business action to
  take.** It routes on already-computed state (crisis, commercial action, active
  session, purchase signal, stage, lifecycle, spend). It also deliberately puts the
  commercial writer on a *different provider* so an OpenRouter incident cannot take it
  down — that reasoning is correct and documented.
- **OpenRouter integration is well done.** The `session_id` affinity key is derived
  purely from `(creator_id, fan_id)` via SHA-256 — no timestamps, no counters, no
  randomness — which is exactly what sticky routing needs; `provider.only` +
  `allow_fallbacks=false` is the right call for predictable cost and cache locality;
  `upstream_provider` is parsed back out and stored; and the cache-write accounting in
  `_complete_openai_compatible` correctly handles providers that report cache writes
  additively versus inclusively. `tests/test_prompt_cache_structure.py` pins the
  ordering so a future edit cannot silently regress it.

**Media pipeline**

- **No CPU work runs on the event loop.** Pillow (`_prepare_classifier_image`,
  `build_contact_sheet`), NudeNet/ONNX (`_detect`), and temp-file writes all go through
  `asyncio.to_thread`; ffmpeg and ffprobe use `asyncio.create_subprocess_exec` with
  `wait_for` timeouts and explicit `process.kill()`. I specifically looked for blocking
  media work in request handlers and found none.
- **No temp-file or subprocess leaks.** Every temp path is removed in a `finally`, every
  `httpx.AsyncClient` created inside a function is closed in a `finally`, and every
  subprocess is killed on timeout.
- **Vault sync is properly incremental**: `_vault_existing_media_ids` paginates past the
  1,000-row cap (with a comment about it), `should_stop_album_scan` uses `lastItemId`
  with a conservative three-page fallback, and media rows are batch-upserted in groups
  of 50 with `on_conflict`. Only the *classification write-back* is one-at-a-time.

**Security**

- **The RLS migration's auto-discovery approach is better than a hand-maintained list**,
  and `fan_list_members` correctly requires *both* fan and list access so a guessed list
  UUID cannot cross tenants.
- **66 of 69 routes carry a tenancy dependency**, and the tenancy helpers deliberately
  return 404 rather than 403 so another agency's resource existence is not revealed.
- **No secrets are logged anywhere** on the credential paths, and
  `services/apifansly.py::_record_usage` is explicitly and correctly secret-free.
- **Webhook signature verification is correct**: raw-body HMAC-SHA256 with
  `hmac.compare_digest`, accepting both the bare digest and the `sha256=` prefix.

**Operations**

- **`services/model_telemetry.py` is well designed**: bounded queue (500) with an
  explicit drop, off the reply path via `spawn`, and it records `cache_read_tokens`,
  `cache_write_tokens`, `cache_hit_ratio`, `upstream_provider`, and
  `cost_source: provider_reported | catalog_estimate`. That last distinction —
  knowing whether a cost figure is real or estimated — is a detail most teams miss.
- **`core/tasks.spawn` exists precisely because bare `create_task` swallows exceptions**,
  and the module docstring explains why. The right instinct, correctly implemented.
- **`services/apifansly.py::_USAGE_EVENTS` is bounded** (`deque(maxlen=50_000)` plus a
  rolling 24-hour cutoff) — I checked specifically for an unbounded telemetry buffer
  and this is not one.
- **`ai/model_providers.py` caches clients via `lru_cache`** on
  `(base_url, api_key, timeout)`, so HTTP connection pools are reused across model
  calls. (The API Fansly layer does not do this — that is PERF-006 — but the model
  layer does.)
- **CI does provision a real PostgreSQL and apply `db/*.sql` against it**, with the
  stated reasoning that "a unique index or check constraint that would not actually
  reject bad data cannot pass". That is the right idea; it just needs the base schema
  and more than 10 tests using it.

---

## APPENDIX — AUDIT METHOD AND REPRODUCTION

All measurements were produced against the audited SHAs during this audit.
Instrumentation was written to a scratch directory and **not committed**; it is
described here so any number can be reproduced or challenged.

| Measurement | Method |
|---|---|
| DB round trips per flow | A counting double for the Supabase client recording every `.table(...).<op>(...).execute()`, substituted into `core.supabase.get_supabase` and every module that imported it; `ai.model_providers.complete` replaced with a recorder returning valid JSON. `get_suggestions` and `_debounced_auto_reply` were then invoked directly, with and without the intelligence feature flags. |
| LLM calls and prompt sizes | Same harness; the recorder logged provider, model, system/user character counts, and `session_id`. |
| Reusable prompt prefix | `build_prompt` rendered for turn *N* and turn *N+1* of the same conversation, flattened through `ai.generator.flatten_message_content`, then a longest-common-prefix scan over the concatenated system+user text. Tokens estimated at chars ÷ 3.6. |
| `cache_control` survival | Direct execution: inspected `build_prompt` output block keys, then the type and content of `flatten_message_content`'s result. |
| Worker overhead | `repair_followup_obligations` and `claim_due_actions` run against the counting double with synthetic obligation and action sets. |
| `dedupeMessages` | The function copied verbatim from `lib/messages.ts` into a Node 22 harness, timed with `process.hrtime.bigint()` over synthetic conversations at 50–10,000 messages, measuring both a full dedupe and the append-one-message case. |
| Sidebar filter | The filter predicate copied verbatim from `components/Sidebar.tsx`, averaged over 20 iterations. |
| Route inventory | `app.openapi()["paths"]` from the real application object. |
| `routes/fansly.py` None binding | Imported `main` and `routes.fansly` and printed both module attributes. |
| Test / lint / build baselines | `pytest -q`, `ruff check --statistics`, `npm run lint`, `npx tsc --noEmit`, `npm test`, `npm run build`, plus `du`/`find` over `.next/static`. |
| Unpaginated-select inventory | A regex/AST scan for `.table("X").select(...)...execute()` chains lacking `.limit(`, `.range(`, `.single(`, `.maybe_single(`, or `count=`, restricted to the ten highest-volume tables. |
| Dead-code candidates | `ruff` (`F401`/`F811`/`F841`), `vulture` at 80% and 60% confidence, then **manual verification of every candidate** against imports, dynamic imports, FastAPI registration, the `HANDLERS` string-dispatch table, worker dispatch, test usage, migration references, and dashboard call sites. Nothing was classified as dead on tooling output alone. |

### Second-pass review of P0/P1 findings

Each P1 was re-examined with the explicit goal of disproving it. Two were **downgraded**
and one was **withdrawn**:

- **Withdrawn:** "a new fan message during generation causes a spurious failed retry of
  the superseded action." Disproved — `cancel_actions_for_fan` includes `PROCESSING` for
  `AUTO_REPLY`, and `fail_action` is `.eq("status","PROCESSING")`-guarded, so it no-ops
  on the already-cancelled row.
- **Downgraded P1 → P2:** API-001 (full message re-sync after every deploy). The
  docstring shows this is a **deliberate** durability trade-off, not an oversight. The
  finding stands as a cost issue at scale, not a defect.
- **Downgraded P1 → P2:** SEC-006 (`NEXT_PUBLIC_API_KEY`). The code comment correctly
  identifies the Supabase JWT as the real boundary, and the middleware does enforce it.
  The issue is naming and rotation friction, not an open door.
- **Revised:** the thread-pool ceiling was initially calculated as 6 threads on a
  2-vCPU container. On Railway, `os.cpu_count()` typically reports **host** cores rather
  than the container's quota, so the ceiling is more likely the `min(32, ...)` cap of
  **32**, with CPU quota — not thread count — as the real constraint. The number in §15
  reflects the corrected reasoning and is labelled `CALCULATED`, not `MEASURED`.

Findings that **survived** the second pass with evidence intact: SCALE-001 (worker
`for` loop read directly; `AUTO_REPLY` confirmed in `HANDLERS`; Procfile confirms a
single Uvicorn worker), SCALE-002 (measured at 2.00 round trips per obligation, and
`get_followup_obligations` confirmed to have no due-window filter), SCALE-003 (the
truncated `fans` read feeds `desired_fan_ids`, and the `current - desired` set drives
`DELETE` — and PostgREST returns an *unordered* 1,000, so the truncated set differs
between runs, producing membership flapping rather than a stable subset), SEC-002
(no filter present, service role bypasses RLS), SEC-001 (`for all` read directly from
the migration), REL-001 (`_fallback_result` returns a complete valid analysis with no
degraded marker anywhere downstream), COST-001 (`attempt_targets` and the `continue`
on empty `replies` read directly), COST-002 (proven by execution), FE-001 and FE-002
(measured / read directly), and VAULT-001 (`sync_vault_start` spawns and returns —
the `await` in the scheduler loop provides no serialisation).

**Explicitly not established by this audit:** actual provider-reported cache hit rates,
real end-to-end latency against live providers, real Supabase query plans and index
usage, API Fansly's documented rate limits and webhook timeout, the base table schema,
and any load test against a running deployment. Section 15 labels every one of these as
`INFERRED` or `UNKNOWN` rather than presenting a guess as a benchmark.

---

# FINDINGS REGISTER

The findings that warrant a full record, in the requested format. P0/P1 entries carry the
result of the second-pass disproof attempt (see the Appendix). A handful of P2/P3 items are
described in full in their body section rather than repeated here — FE-004 (Sidebar has no
memoisation or virtualisation) in §9d, FE-007 (self-referential fetch effect) in §9g, and
the error-handling classification in §11.

---

## SCALE-001 — Full Auto delivery is a single sequential loop with an unconditional sleep

- **Severity:** P0 Critical
- **Category:** Capacity / Async
- **Confidence:** Confirmed
- **Where:** `workers/scheduled_actions.py::process_once` (lines 519–611), `scheduled_actions_loop` (613–620); `HANDLERS["AUTO_REPLY"]` → `services/suggestions.py::deliver_scheduled_auto_reply` (1874) → `_debounced_auto_reply` (630); started by `main.py::_scheduled_actions_scheduler` (609).
- **Current behavior:** `claim_due_actions(limit=20)` returns up to 20 due plus up to 20 stale actions. `process_once` then iterates them in a plain `for` loop, `await`ing each to completion before starting the next. For `AUTO_REPLY` that means the full reply pipeline — 43 DB round trips (MEASURED), 2 sequential LLM calls, an API Fansly send, and a 6.8 s `_sleep_while_current` typing simulation (MEASURED) — runs to completion before the next fan is touched. After the batch, `await asyncio.sleep(POLL_SECONDS)` sleeps a further 60 seconds regardless of remaining backlog.
- **Why this matters:** This is the throughput ceiling for Full Auto across the entire deployment, not per creator or per agency. At an estimated 12–18 s per `AUTO_REPLY`, a full batch of 20 takes 4–6 minutes, plus the 60 s sleep — roughly **3–6 replies per minute for all agencies combined** (CALCULATED). One agency's burst delays every other agency's fans behind it in a single global FIFO. The machine is idle while this happens: the constraint is code structure, not resources.
- **Trigger:** More than ~5 fans in Full Auto messaging within the same minute, anywhere in the deployment.
- **Scale sensitivity:** Invisible at 1 creator. At 20 creators with normal activity the queue is at the edge. At 100 creators the backlog grows continuously and never drains.
- **Suggested fix:** Raise the claim limit, then run the claimed batch through `asyncio.gather` behind a `Semaphore(N)` (start at 8–12). Replace the unconditional `sleep(60)` with: sleep only when the previous claim returned fewer rows than the limit. Keep `_should_still_send` inside each concurrent unit so revalidation stays per-action.
- **Risk of fix:** Medium — concurrency changes the ordering guarantees the current loop implicitly provides, and per-fan single-flight must still hold (`_pending_auto_replies` and the `AUTO_REPLY` dedupe key already provide it, but that should be verified under concurrency).
- **Expected benefit:** Latency, reliability, capacity. 5–10× Full Auto throughput with no new infrastructure.
- **Tests required:** A test that N concurrently claimed `AUTO_REPLY` actions for N distinct fans all complete; a test that two actions for the *same* fan cannot both send; a test that a full batch triggers an immediate re-poll rather than a 60 s sleep.
- **Second pass:** Attempted disproof by looking for any parallelism, a second worker, or an alternate `AUTO_REPLY` path. `Procfile` is `uvicorn main:app` with no `--workers`, so exactly one scheduler task exists. `HANDLERS` confirms `AUTO_REPLY` is routed here. `schedule_auto_reply` writes `action_type="AUTO_REPLY"` and there is no in-process delayed-send fallback. **Finding survives.**

---

## SCALE-002 — `repair_followup_obligations` costs 2 DB round trips per obligation, every 60 seconds

- **Severity:** P1 High
- **Category:** Capacity / DB
- **Confidence:** Confirmed (MEASURED)
- **Where:** `workers/scheduled_actions.py::repair_followup_obligations` (498–516); `db/commercial_queries.py::get_followup_obligations` (507–536) and `ensure_action_pending` (261–305).
- **Current behavior:** Called at the top of **every** `process_once` cycle. `get_followup_obligations` pages through **every** `fan_commercial_states` row where `next_followup_at IS NOT NULL AND next_followup_type IS NOT NULL` — with **no due-time filter**, so a follow-up scheduled for next Friday is fetched every minute for a week. Each row is then passed to `ensure_action_pending`, which does a `SELECT` plus a conditional `INSERT`/`UPDATE`, **sequentially**. Measured at **2.00 DB round trips per obligation**: 401 round trips for 200 obligations.
- **Why this matters:** Pure overhead that scales with the number of *outstanding* obligations rather than the number of *due* ones, and it runs before any real work. At 1,000 obligations that is ~2,000 round trips per minute; at 5,000 (~33 per creator at 150 creators) the repair pass alone exceeds the 60 s cycle, at which point the worker is permanently behind and **durable follow-ups stop firing altogether** — silently, because nothing surfaces cycle duration.
- **Trigger:** Steady-state accumulation of scheduled follow-ups. No burst required.
- **Scale sensitivity:** Negligible at 1 creator. Noticeable around 20–30 creators. Becomes the second hard bottleneck around 50–100 creators.
- **Suggested fix:** Add `.lte("next_followup_at", now + 5 minutes)` to `get_followup_obligations`. Replace the per-row `ensure_action_pending` with a single bulk `upsert` on `dedupe_key` for the rows that need repair, after one batched read of the existing actions.
- **Risk of fix:** Low — the semantics of "repair a missing durable action" are unchanged; only rows that could fire soon are considered.
- **Expected benefit:** Capacity, latency. Removes thousands of round trips per minute and frees the cycle for actual delivery.
- **Tests required:** A test that an obligation due far in the future is not repaired on this pass but *is* repaired as its time approaches; a test that a `COMPLETED` action for a still-current obligation is recreated; a round-trip-count assertion so the N+1 cannot return.

---

## SCALE-003 — Fansly Lists reconciliation deletes correct memberships past 1,000 fans

- **Severity:** P1 High (P0 if `FANSLY_LISTS_SYNC_ENABLED` is turned on before it is fixed)
- **Category:** DB / Reliability / Data loss
- **Confidence:** Confirmed
- **Where:** `services/fansly_lists.py::_load_state` (121–158), consumed by `_reconcile` (161–272).
- **Current behavior:**
  ```python
  fans = (db.table("fans").select("id, platform_fan_id").eq("creator_id", creator_id).execute()).data or []
  fan_by_platform_id = {str(r["platform_fan_id"]): str(r["id"]) for r in fans ...}
  ```
  No `.range()`, no `.limit()`, no `.order()`. PostgREST caps this at 1,000 rows and — because there is no `ORDER BY` — the particular 1,000 returned is not stable between calls. `_reconcile` then builds `desired_fan_ids` by mapping remote members through this dictionary, counting anything unmapped as `unmapped_members` and skipping it, and finally executes:
  ```python
  for fan_id in sorted(current_fan_ids - desired_fan_ids):
      db.table("fan_list_members").delete().eq("list_id", list_id).eq("fan_id", fan_id).eq("source", "fansly").execute()
  ```
  A fan whose row was inside the 1,000 on a previous sync (so the membership exists) but outside it on this sync (so it cannot be mapped) lands in `current - desired` and **its membership is deleted**. On the next sync the ordering may differ again and it is re-added.
- **Why this matters:** Silent, repeating, destructive corruption of exactly the data agencies care about — VIP / Whales / Buyers lists that drive Auto Audience targeting and re-engagement. Fans flap in and out of Auto eligibility every 6 hours with no error, no log line beyond an `unmapped` counter, and no operator-visible signal. `fan_list_members` at line 147 has the same unpaginated problem in the other direction (incomplete `current_fan_ids` → missed removals).
- **Trigger:** Any creator with more than 1,000 fans, on any lists sync.
- **Scale sensitivity:** Zero impact below 1,000 fans per creator. Total corruption of that creator's list targeting above it. This is a **per-creator** threshold, not a deployment one — a single successful creator crosses it.
- **Suggested fix:** Paginate `fans` and `fan_list_members` in `_load_state` with `.order(...).range(offset, offset+size-1)` exactly as `main.py::_vault_existing_media_ids` already does. Additionally: if the fan map is ever incomplete for any reason, **skip the deletion phase** rather than deleting — removals should require positive evidence, not absence of evidence.
- **Risk of fix:** Low. The pagination pattern already exists in the codebase.
- **Expected benefit:** Reliability, correctness. Prevents destructive data loss.
- **Tests required:** A test with 1,500 synthetic fans asserting that every remote member maps and that no membership is deleted; a test that a partial/failed fan load aborts the removal phase rather than deleting.
- **Second pass:** Attempted disproof by checking whether Supabase's `db-max-rows` might be raised for this project. It cannot be confirmed from the repository — but `main.py::_vault_existing_media_ids` carries the comment *"Load every existing media ID; Supabase caps ordinary selects at 1,000"* and paginates accordingly, which is strong internal evidence the cap is active. Also checked whether the delete is constrained to prevent this: it is scoped to `source='fansly'` and to the mirror, which correctly protects operator-created memberships, but does nothing about this case. **Finding survives.**

---

## SCALE-004 — No concurrency limiter on model calls anywhere in the system

- **Severity:** P1 High
- **Category:** Capacity / Cost
- **Confidence:** Confirmed
- **Where:** absence across `ai/model_providers.py`, `ai/generator.py`, `ai/situation_analyzer.py`, `services/fan_intelligence.py`, `services/suggestions.py`.
- **Current behavior:** Exhaustive grep finds four `asyncio.Semaphore` instances — `_protected_video_download_gate(1)`, `_VISION_GATE(2)`, `_SEMANTIC_GATE(32)`, `_VIDEO_EXTRACTION_GATE(2)` — **none on writer or analyzer calls**. The `AsyncOpenAI` clients are `lru_cache`d (good, pools are reused) but carry the SDK's default `httpx` limits (`max_connections=1000`), so there is no client-side cap either. Today SCALE-001's sequential worker accidentally provides a limit of 1; the Assisted path has no limit at all, and fixing SCALE-001 removes the accidental one.
- **Why this matters:** The brief's exact concern — one agency spike opening hundreds of simultaneous paid model requests. Concretely: 50 operators clicking Regenerate, or a concurrent Full Auto worker after SCALE-001 is fixed, both go straight to the provider with no gate. Combined with COST-001's 3× retry burst, an overload becomes an amplified overload.
- **Trigger:** Any burst of concurrent generations.
- **Scale sensitivity:** Latent today because of SCALE-001. Becomes immediately load-bearing the moment SCALE-001 is fixed — **these two must ship together.**
- **Suggested fix:** A module-level `asyncio.Semaphore` in `ai/model_providers.py::complete`, sized from an env var, wrapping every provider call. Optionally a second per-creator gate if fairness becomes an issue. Emit a wait-time metric so saturation is visible.
- **Risk of fix:** Low.
- **Expected benefit:** Cost, reliability, latency predictability.
- **Tests required:** A test that N+5 concurrent `complete()` calls never exceed N in flight; a test that the semaphore does not deadlock when a call raises.

---

## REL-001 — Analyzer failure fabricates a neutral analysis and Full Auto keeps sending

- **Severity:** P1 High
- **Category:** Reliability / Silent failure
- **Confidence:** Confirmed
- **Where:** `ai/situation_analyzer.py::analyze_situation` (112–150) and `_fallback_result` (152–178).
- **Current behavior:** Both the transport `except Exception` and the JSON `except Exception` assign `result = _fallback_result()`, which returns a **complete, well-formed** analysis: `purchase_signal: "none"`, `offer_response: "none"`, `crisis_signal: "none"`, `wants_media: "false"`, `cannot_afford_any_offer_now: "false"`, `strategic_move: "mirror_warmth"`. There is no marker distinguishing this from a real analysis. Downstream, `_debounced_auto_reply` uses `situation["purchase_signal"]` to set and clear decline locks, `crisis_signal` to decide freezing, `resend_requested` to resend PPVs, and passes the whole dict to `select_writer_route` and `build_prompt`. The reply is then generated and sent.
- **Why this matters:** During any analyzer-provider incident Full Auto does not stop — it **degrades to guessing and keeps selling**. A fan saying "I'll take the $50 one" is classified as ordinary chat. A fan expressing distress in language the `_looks_like_self_harm` regex does not cover is classified as no crisis. The contrast is instructive: the commercial orchestrator's own failure path is explicitly correct — `"Full Auto must fail closed. Silently reverting to the legacy planner can send content after a pause or at the wrong price"` — the analyzer's is not.
- **Trigger:** Any analyzer transport error, timeout, rate limit, or malformed JSON response.
- **Scale sensitivity:** Independent of scale, but the probability of hitting it rises with volume, and the blast radius is every fan in Full Auto simultaneously.
- **Suggested fix:** Add `"analysis_degraded": True` to `_fallback_result`. In `_debounced_auto_reply`, treat it as a hard stop for Full Auto (skip the send, leave the action to retry) and as a visible banner in the Assisted path. Record it as a distinct telemetry outcome so incidents are countable.
- **Risk of fix:** Medium — it converts a currently-silent degradation into a visible non-delivery, which will surface as "Auto stopped working" during incidents. That is the correct trade, but it changes observable behaviour and should ship with the health surface from OBS-001.
- **Expected benefit:** Reliability, safety, trust.
- **Tests required:** A test that a raising analyzer produces `analysis_degraded` and that `_debounced_auto_reply` sends nothing; a test that the Assisted path still returns suggestions but flags the degradation; a test that the self-harm regex backstop still applies.
- **Second pass:** Attempted disproof by looking for a downstream degraded-mode gate. `_crisis_freezes_chat` only fires on a *positive* crisis signal. The commercial orchestrator is `COMMERCIAL_LAYER_ENABLED`-gated and, when off, `should_plan` falls back to `_fan_wants_content(latest_message, situation)` which consults the same fabricated dict. No gate exists. **Finding survives.**

---

## REL-002 — Duplicate message ingestion relies on an unverified unique constraint

- **Severity:** P1 High
- **Category:** Race / DB
- **Confidence:** High confidence (the race is confirmed; the mitigating constraint is UNKNOWN)
- **Where:** `db/queries.py::save_message` (231–285); `main.py::fansly_webhook` (4363–4373); `main.py::handle_new_fan_message` (492–595); `main.py::_sync_recent_fan_messages` (1513–1517); `main.py::_processed_messages` (129).
- **Current behavior:** Three independent ingestion paths write fan messages. The dedupe story is inconsistent:
  - `handle_new_fan_message` (poller) and `/generate-suggestions` both **check and add** to the in-memory `_processed_messages` set.
  - `/webhook/fansly` does **neither** — it performs its own `SELECT id FROM messages WHERE fansly_message_id = ?` pre-check instead.
  - `save_message` performs a third check-then-insert on `(fan_id, creator_id, fansly_message_id)`.
  - `_sync_recent_fan_messages` bulk-`insert`s rows after an `in_` existence check.

  Every one of these is a read followed by a non-atomic write. Whether concurrent writers can actually create duplicate rows therefore depends entirely on a unique constraint on `messages.fansly_message_id` — **and no migration in the repository creates the `messages` table or any constraint on it.**
- **Why this matters:** A duplicate fan message row means the analyzer runs twice, `schedule_auto_reply` runs twice (idempotent via the dedupe key, so probably survivable), and the conversation history the writer sees contains the fan's message twice. Under webhook redelivery — which is exactly what a slow webhook causes (§17A) — this is the likely path.
- **Trigger:** Webhook redelivery after a timeout; webhook and poller ingesting the same message; `_processed_messages.clear()` at the 1,000 boundary (REL-004).
- **Scale sensitivity:** Probability rises with message volume and with webhook latency, which itself rises with load — so it is self-reinforcing under exactly the conditions that create it.
- **Suggested fix:** Confirm or add `CREATE UNIQUE INDEX ... ON messages (fansly_message_id) WHERE fansly_message_id IS NOT NULL`, put it in `db/000_base_schema.sql` (DB-000), then convert `save_message` from check-then-insert to an `upsert` on that constraint and delete the redundant pre-checks in the webhook and reconciler.
- **Risk of fix:** Low, once the constraint is confirmed. Adding the index may fail if duplicates already exist — check first.
- **Expected benefit:** Reliability, correctness, and three fewer round trips per ingestion.
- **Tests required:** A schema test (against the CI Postgres) asserting the constraint rejects a duplicate; a concurrency test issuing two simultaneous `save_message` calls with the same platform id and asserting exactly one row.

---

## REL-005 — An undeliverable fan burns eight full pipeline runs

- **Severity:** P2 Medium
- **Category:** Cost / Reliability
- **Confidence:** Confirmed
- **Where:** `services/suggestions.py::_debounced_auto_reply` (1248–1254, 1421–1428), `deliver_scheduled_auto_reply` (1874–1930), `workers/scheduled_actions.py::_run_auto_reply` (363–384) and `fail_action` (395–419).
- **Current behavior:** When `_debounced_auto_reply` returns without sending for a non-PPV reason — no `group_id`, no `apifansly_account_id`, `generate_replies` returning `[]`, or the outer `except Exception` — nothing is recorded. `deliver_scheduled_auto_reply` then finds no creator message newer than the trigger and returns `False`; `_run_auto_reply` checks for a pending PPV approval and, finding none, raises `RuntimeError("Auto reply completed without a confirmed message")`; `fail_action` requeues with `max_attempts=8` and backoff capped at 60 minutes. Each retry re-runs the **entire** pipeline.
- **Why this matters:** Worst case for one message that is never delivered: **8 analyzer calls, up to 24 writer generations (COST-001's 3× applies per attempt), and roughly 350 DB round trips.** The failure is deterministic in the common case (a fan with no `fansly_group_id` will still have none an hour later), so all eight attempts are guaranteed waste. Note the PPV path is handled correctly — it calls `freeze_fan_for_review` — only the plain-text path is not.
- **Trigger:** Any fan without a resolvable delivery route, or a persistent writer-validation failure.
- **Scale sensitivity:** Grows with the number of unbound fans, which grows with fan count.
- **Suggested fix:** Have `_debounced_auto_reply` return a typed outcome rather than `None`. Map deterministic non-delivery reasons (no route, validation exhausted) to `complete_action` with a recorded reason instead of `fail_action`; reserve retries for genuinely transient failures.
- **Risk of fix:** Low.
- **Expected benefit:** Cost, capacity (the retries occupy the sequential worker), observability.
- **Tests required:** A test that a fan with no `fansly_group_id` produces exactly one attempt and a recorded reason, not eight.

---

## COST-001 — Writer retries three times with no backoff, including on validation failure

- **Severity:** P1 High
- **Category:** Cost / LLM / Reliability
- **Confidence:** Confirmed
- **Where:** `ai/generator.py::generate_replies` (295–404), specifically `attempt_targets` (315–316), the `continue` at 372, and the implicit loop continuation at 382; `parse_reply_candidates` (89–210); `ai/openrouter_routing.py::provider_preferences` (85–107).
- **Current behavior:** `attempt_targets = [primary, primary, fallback or primary]` — three attempts with **no `sleep` between them**. The loop advances not only on transport exception but also when `parse_reply_candidates` returns `[]`. That function rejects a candidate if `any(phrase in reply.lower() for phrase in bot_phrases)`, where `bot_phrases` contains the bare substrings `"yourself"`, `"interesting"`, `"noted"`, `"got it"`, `"i like that"`, `"that's nice"`, and then requires **three** surviving candidates (or two plus padding to exactly three) or it returns `[]`. Meanwhile `provider_preferences` sets `only: ["Inceptron"]` with `allow_fallbacks: False`, so a retry cannot route around a throttled provider.
- **Why this matters:** Two distinct problems.
  1. **Cost and quality:** `"yourself"` is a normal word in this product's domain ("touch yourself", "by yourself"). One such reply costs three full generations (~10,500 input tokens) and then **sends nothing** — `generate_replies` returns `[]`, and in Full Auto that cascades into REL-005's eight retries.
  2. **Retry amplification:** A 429 from the pinned provider produces three immediate requests to that same provider, repeated across up to eight durable retries. This is the most likely cascading-failure path in the system (§17B).
- **Trigger:** (1) any reply containing a banned substring — common; (2) any provider throttling event.
- **Scale sensitivity:** (1) is volume-proportional. (2) becomes dangerous exactly when the provider is already under load, i.e. at peak.
- **Suggested fix:** Separate the two retry reasons. Retry transport failures with jittered exponential backoff (e.g. 0.5 s, 2 s) and route the second retry to the fallback target rather than the primary. Do **not** retry validation failures on the same target — instead, remove the substring bans that match legitimate copy (`"yourself"`, `"interesting"`, `"noted"`, `"got it"`) and relax the "must have three" requirement to "must have at least one".
- **Risk of fix:** Low for the retry change. Editing `bot_phrases` changes writer output quality and should be reviewed by whoever owns the voice — but the current list demonstrably rejects valid copy.
- **Expected benefit:** Cost (up to 3× on affected turns), reliability, and fewer silent non-deliveries.
- **Tests required:** A test that a transport failure sleeps before retrying and that the second retry uses the fallback target; a test that "touch yourself" is not rejected; a test that a single valid candidate is returned rather than `[]`.
- **Second pass:** Attempted disproof by checking whether backoff exists elsewhere in the chain. `fail_action`'s exponential backoff applies between *durable action* retries, not between the three in-process attempts. `record_model_transport_failure` feeds availability tracking but does not gate. **Finding survives.**

---

## COST-002 — Prompt caching is defeated in three distinct ways

- **Severity:** P1 High
- **Category:** Cost / LLM
- **Confidence:** Confirmed (a) by execution, (b) and (c) by measurement
- **Where:** (a) `ai/prompt_builder.py::build_prompt` (1040–1053) vs `ai/generator.py::flatten_message_content` (212–235) and `ai/model_providers.py::_complete_anthropic` (194–230). (b) `ai/situation_analyzer.py::analyze_situation` (32–119). (c) `ai/prompt_builder.py` `fan_context_parts` (696–719).
- **Current behavior:**
  **(a)** `build_prompt` emits the system message as content blocks carrying `cache_control: {"type": "ephemeral"}`. `generate_replies` immediately calls `flatten_message_content` on it, producing a plain string and discarding the directive. Verified by execution:
  ```
  first block keys: ['type', 'text', 'cache_control']
  after flatten_message_content -> type: str
  cache_control survives to transport? False
  ```
  `_complete_anthropic` therefore receives `system=<str>` and Anthropic prompt caching never engages — `cache_read_input_tokens` is structurally always 0. (The flattening is *correct* for the OpenAI-compatible providers, which use implicit prefix caching; the defect is that there is no Anthropic-specific branch.)
  **(b)** The analyzer places ~1,570 tokens of static JSON schema, COMMERCIAL INTERPRETATION RULES and SAFETY text **after** the conversation, with a 12-token `system`. Its reusable prefix is therefore 12 tokens.
  **(c)** `fan_context_parts` puts the five per-message blocks (learned intelligence, affordability, price learning, conversation director, session strategy) **before** the durable profile and the transcript, dropping the measured byte-identical prefix from 2,149 tokens (61%) to 1,811 (49%).
- **Why this matters:** (b) is the largest single line item — ~1,570 tokens per message on the default Haiku analyzer that could be cached with a pure code move and zero behavioural change. (a) means any Anthropic-routed traffic pays full input price on every call. (c) costs ~340 tokens of prefix per writer call.
- **Trigger:** Every message, on every path.
- **Scale sensitivity:** Linear in message volume — this is a pure per-message tax.
- **Suggested fix:** (a) Pass the block list through to `_complete_anthropic` and flatten only for OpenAI-compatible providers. (b) Move the static rules block into the `system` argument and leave only the conversation and latest message in the user turn. (c) Reorder `fan_context_parts` so durable items come first.
- **Risk of fix:** Low for all three — none changes the tokens the model sees, only their order and transport encoding.
- **Expected benefit:** Cost. Directly measurable in `model_usage_events.metadata.cache_hit_ratio`, which is already instrumented.
- **Tests required:** Extend `tests/test_prompt_cache_structure.py` to assert `cache_control` survives to the Anthropic transport (this is the gap that let (a) exist); a test pinning the analyzer's system/user split; a test asserting the durable-first ordering of `fan_context_parts`.
- **Second pass:** Attempted disproof of (a) by checking whether Anthropic accepts a string system with caching applied implicitly — it does not; `cache_control` breakpoints are required. Confirmed by direct execution of the flattening. **Finding survives.**

---

## COST-005 — Three overlapping extraction models, two of them invisible to telemetry

- **Severity:** P2 Medium
- **Category:** Cost / Observability / Duplicated logic
- **Confidence:** Confirmed
- **Where:** `services/suggestions.py::_update_fan_memory` (461–565), `_update_fan_ai_summary` (567–628), module-level `together_client` (101–104); `services/fan_intelligence.py::learn_from_fan_message`.
- **Current behavior:** `_update_fan_memory` and `_update_fan_ai_summary` each construct a prompt and call `together_client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo", ...)` directly — bypassing `ai/model_providers.complete`, the writer router, `services/model_availability`, and `services/model_telemetry` entirely. Neither has a timeout (the OpenAI SDK default is 600 s). Both fire on every tenth fan message, alongside `learn_from_fan_message`, which extracts the same categories of fact (age, location, payday, kinks, preferences) through the proper abstraction into `fan_facts`.
- **Why this matters:** Three models extract overlapping facts from the same conversation into three different stores (`fans.member_note`, `fans.ai_summary`, `fan_facts`), which is both a cost multiplier and a consistency problem — they can and will disagree. And because two of them never reach `model_usage_events`, **the answer to "what did this month cost" is systematically low**, which undermines every cost decision made from that table.
- **Trigger:** Every tenth fan message per conversation.
- **Scale sensitivity:** Linear in message volume.
- **Suggested fix:** Short term, route both through `ai.model_providers.complete` with a `get_runtime_target("EXTRACTOR")` target and a telemetry context — this alone fixes the accounting and the missing timeout. Medium term, consolidate the three extractions into one pass that writes all three shapes.
- **Risk of fix:** Low for the routing change; medium for consolidation, since prompt differences affect output quality.
- **Expected benefit:** Cost, observability, consistency.
- **Tests required:** A test that both functions record a `model_usage_events` row; a test that a hung provider is bounded by a timeout.

---

## SEC-001 — RLS grants browsers full write access to every creator-owned table

- **Severity:** P1 High
- **Category:** Security / Tenancy
- **Confidence:** Confirmed
- **Where:** `db/tenant_isolation_v1.sql` (108–147, and the `creators` policy at 82–104).
- **Current behavior:** Every generated policy is `for all to authenticated` with `using` and `with check` both set to the membership predicate. `for all` covers SELECT, INSERT, UPDATE **and** DELETE. The dashboard authenticates as `authenticated` with the anon key plus the operator's JWT.
- **Why this matters:** Every business rule the backend enforces exists only in Python, and the browser has a direct path around it. From the console an operator can `UPDATE ppv_deliveries SET status='purchased'` (bypassing the ledger's single-flight), `DELETE FROM ppv_approval_requests` (bypassing the approval gate), `INSERT INTO messages` (fabricating history the writer will then use), `UPDATE fans SET needs_human_review=false` (unfreezing a crisis-flagged fan), `UPDATE creators SET auto_mode=true`, or `INSERT INTO model_usage_events` (poisoning cost data). The tenancy boundary holds — an operator cannot touch another agency — but the *authorization* boundary within their own agency does not exist.
- **Trigger:** Any operator with a browser console. No exploit required.
- **Scale sensitivity:** Independent of scale; grows with the number of operators who have access.
- **Suggested fix:** Split into `for select` policies on the ~12 tables the dashboard reads, plus narrow `for insert/update/delete` policies only on `fan_lists` and `fan_list_members` (the two it legitimately writes). Everything else becomes service-role-only. The auto-discovery loop can generate the read policies; the write exceptions are a short explicit list.
- **Risk of fix:** Medium — any dashboard write not on the exception list breaks. Audit `app/page.tsx` (`fan_lists` insert/update/delete, `fan_list_members` upsert/delete), `app/settings/page.tsx`, and `app/scripts/page.tsx` first.
- **Expected benefit:** Security. Restores the backend as the only path to state changes.
- **Tests required:** A schema test asserting an `authenticated` role cannot `UPDATE ppv_deliveries` or `INSERT INTO messages`; a dashboard smoke test that list management still works.

---

## SEC-002 — `preview_auto_audience` scans every agency's list memberships

- **Severity:** P1 High
- **Category:** Security / Tenancy / DB
- **Confidence:** Confirmed
- **Where:** `main.py::preview_auto_audience` (4841–4852).
- **Current behavior:**
  ```python
  asyncio.to_thread(lambda: db.table("fan_list_members")
      .select("fan_id, list_id, fan_lists(exclude_from_auto, creator_id)")
      .execute())   # no .eq(), no .limit()
  ```
  No creator filter at all, executed with service-role credentials so RLS does not apply. The rows are filtered in Python afterwards. The two sibling queries in the same `gather` (`fans`, `messages`) are creator-filtered but also unpaginated.
- **Why this matters:** Two problems, and the correctness one is worse than the privacy one. **Privacy:** every agency's membership data is read into a request scoped to one creator — a Python-side filter is the only thing preventing a leak, and this is exactly the pattern the RLS migration was written to eliminate. **Correctness:** PostgREST truncates at 1,000 rows *globally*, so past ~1,000 total memberships across all agencies the requesting creator's own rows are likely absent, `memberships` is empty, `legacy_exclusions` is empty, and **the exclusion policy silently stops applying** in the preview the operator uses to decide whether to turn Auto on. The endpoint is called on every load of `/analytics` and `/settings`.
- **Trigger:** Any call to the endpoint once total `fan_list_members` rows exceed ~1,000.
- **Scale sensitivity:** Breaks at the *deployment* level, not per creator — so it breaks earlier than most of the other 1,000-row findings.
- **Suggested fix:** Add `.eq("fan_lists.creator_id", creator_id)` (or filter via a join on the creator's list ids) and paginate. Replace the `messages` scan with a `count`/`distinct` aggregate for `is_new_fan`, and paginate `fans`.
- **Risk of fix:** Low.
- **Expected benefit:** Security, correctness, latency.
- **Tests required:** A test asserting the query carries a creator constraint; a test with >1,000 global memberships asserting the requesting creator's exclusions still apply.

---

## SEC-003 — `fan_conversation_summaries` is a view and the RLS migration only covers base tables

- **Severity:** P1 High
- **Category:** Security / Tenancy
- **Confidence:** Needs runtime measurement — the view definition is not in the repository
- **Where:** `db/tenant_isolation_v1.sql` (114–120: `and t.table_type = 'BASE TABLE'`); consumed by `app/page.tsx:415` (`supabase.from('fan_conversation_summaries').select('*')`).
- **Current behavior:** The migration's discovery loop explicitly filters to `BASE TABLE`, so **no policy is ever created for this view**. A Postgres view executes with the definer's privileges unless created `WITH (security_invoker = on)`. If it was not, RLS on the underlying `fans` and `messages` is not applied when an authenticated browser queries it.
- **Why this matters:** The dashboard's entire conversation sidebar — fan names, spend, last message content, notes — reads from this view directly from the browser. If it is not `security_invoker`, changing one `.eq('creator_id', ...)` in the console reads **every agency's conversations**. This would be the most serious tenancy defect in the system.
- **Trigger:** Any authenticated operator, if the view is definer-rights.
- **Scale sensitivity:** Independent of scale; severity increases with the number of agencies.
- **Suggested fix:** Verify immediately:
  ```sql
  SELECT relname, reloptions FROM pg_class WHERE relname = 'fan_conversation_summaries';
  ```
  If `security_invoker=true` is absent, recreate the view with it. Then extend the migration's discovery loop to include `VIEW` for views carrying `creator_id`, and add the view definition to `db/000_base_schema.sql`.
- **Risk of fix:** Low if it is already `security_invoker`; medium otherwise, since enabling it will correctly start filtering rows and any code depending on the unfiltered behaviour will change.
- **Expected benefit:** Security.
- **Tests required:** A schema test that queries the view as a non-owning `authenticated` role and asserts zero rows for another creator.

---

## SEC-004 — Authentication defaults to fail-open when `APP_ENV` is absent

- **Severity:** P2 Medium (P0 if `APP_ENV` is ever unset in production)
- **Category:** Security / Deployment
- **Confidence:** Confirmed
- **Where:** `core/auth.py::_is_dev` (30–31), used at 37, 67; `core/tenancy.py::require_creator_access` (39–43); `main.py::api_auth_middleware` (933–936); `main.py::fansly_webhook` (4109–4112).
- **Current behavior:** `_is_dev()` is `os.environ.get("APP_ENV", "development") == "development"` — the **default is development**. When true: an unconfigured `DASHBOARD_API_SECRET` allows the request through; a missing bearer token returns `None` instead of raising; `require_creator_access` returns early on a `None` user, making **every tenancy check a no-op**; and a missing webhook signing secret skips signature verification.
- **Why this matters:** A single missing or misspelled environment variable turns a correctly-written multi-tenant system into an open one, and the unsafe value is the default. The intent is documented and correct (*"fails CLOSED in production ... and OPEN in development"*); the polarity of the default inverts it.
- **Trigger:** `APP_ENV` unset, misspelled, or lost during a Railway environment migration.
- **Scale sensitivity:** Independent of scale.
- **Suggested fix:** Invert the default — treat any value other than an explicit `development`/`test` as production. Or make `APP_ENV` a required `Settings` field so the process refuses to start without it.
- **Risk of fix:** Low, but it will break local development for anyone not setting `APP_ENV` — which is the point.
- **Expected benefit:** Security, deployment safety.
- **Tests required:** A test that with `APP_ENV` unset, an unauthenticated request is rejected and `require_creator_access` raises.

---

## SEC-005 — `/test/*` endpoints are live in production, and one destroys sales history

- **Severity:** P2 Medium
- **Category:** Security / Bug / Data loss
- **Confidence:** Confirmed
- **Where:** `main.py::simulate_ppv_purchase` (5447–5522) and `test_inject_message` (5525–5535).
- **Current behavior:** Neither is guarded by `_is_dev()`; both are reachable in production with an ordinary operator session (they are tenancy-checked, so this is an operator capability rather than an anonymous one). Two specific defects:
  1. **Data loss.** `simulate_ppv_purchase` selects `"pending_ppv_check, total_spent, active_session, ai_summary"` (line 5459) and then reads `fan_data.get("sales_log")` (line 5481) — a column it never selected, so the value is always `[]`. It appends one entry and writes `sales_log` back (line 5503), **replacing the fan's entire sales history with a single fabricated row.**
  2. `test_inject_message` calls `process_incoming_fan_message(..., auto_mode=True, ...)` with `auto_mode` **hardcoded**, so it can trigger a real Full Auto send to a real fan for a creator whose Auto mode is off.
- **Why this matters:** Beyond the obvious — forged `total_spent` and `spend_tier` feed price learning, buyer lifecycle, writer routing (`high_value_fan`), and revenue reporting, so a single accidental call permanently distorts a fan's commercial model. The `sales_log` wipe is unrecoverable.
- **Trigger:** Any call to either endpoint.
- **Scale sensitivity:** Independent of scale.
- **Suggested fix:** Guard both with `if not _is_dev(): raise HTTPException(404)`. Independently, fix the `sales_log` select/write mismatch and stop hardcoding `auto_mode=True`.
- **Risk of fix:** Low.
- **Expected benefit:** Security, data integrity.
- **Tests required:** A test that both return 404 when `APP_ENV=production`; a test that `simulate_ppv_purchase` appends to rather than replaces `sales_log`.

---

## PERF-004 — Five deterministic refreshers run sequentially, and price learning runs twice

- **Severity:** P2 Medium
- **Category:** Latency / DB
- **Confidence:** Confirmed (MEASURED)
- **Where:** `services/suggestions.py::get_suggestions` (293–360) and `_debounced_auto_reply` (770–1020). The duplicate call is at ~782 (`trigger_type="auto_message_pre_policy"`) and ~985 (`trigger_type="auto_message_post_lifecycle"`).
- **Current behavior:** `refresh_affordability_from_situation`, `refresh_fan_lifecycle`, `refresh_price_learning`, `direct_conversation` and `plan_next_action` are awaited one after another. Measured: **15 DB round trips in the Assisted path** and **~24 in the Auto path**, where `refresh_price_learning` executes **twice** — each run doing a policy-scope read, an events read, a profile read, a profile upsert, an audit read, and an audit insert. Three of the five (`fan_price_learning_audits`, `fan_conversation_director_audits`, `fan_session_strategy_audits`) do a `SELECT` before an `INSERT` even though each table has `dedupe_key text not null unique`.
- **Why this matters:** ~375–750 ms of serial DB latency on the user-visible Assisted path, and ~600 ms–1.2 s on the Auto path, for work that is CPU-trivial. In the Auto path it is also multiplied by the sequential worker (SCALE-001).
- **Trigger:** Every message, when the intelligence flags are on.
- **Scale sensitivity:** Linear in message volume; also consumes thread-pool slots that other requests need.
- **Suggested fix:** In the Assisted path, `gather` affordability, lifecycle and price learning (they have no mutual data dependency there), then run director and strategy. In the Auto path, remove the `auto_message_pre_policy` price-learning call unless the commercial orchestrator genuinely needs it — and if it does, pass the result forward rather than recomputing. Replace the six audit `SELECT`+`INSERT` pairs with `upsert(on_conflict="dedupe_key", ignore_duplicates=True)`.
- **Risk of fix:** Medium for the ordering change — `refresh_price_learning` currently receives `affordability` and `lifecycle`, so their dependency needs verifying before parallelising. Low for the audit upserts.
- **Expected benefit:** Latency, DB load.
- **Tests required:** A round-trip-count assertion for the Auto path; a test that the audit rows are still written exactly once under a duplicate dedupe key.

---

## PERF-006 — A new TLS connection is opened for every API Fansly call

- **Severity:** P2 Medium
- **Category:** Latency
- **Confidence:** Confirmed
- **Where:** `services/apifansly.py::request` (460–503) — `active_client = client or httpx.AsyncClient()` with `owns_client` closing it in `finally`. Callers that pass no client include `send_message`, `delete_message`, `current_account`, `account_media_prices`, and the typing-indicator POST in `services/suggestions.py:1154`.
- **Current behavior:** Roughly 20 sites across the codebase construct a fresh `httpx.AsyncClient`. `sync_chats` and `_run_vault_sync` correctly create one client and pass it down; the message-send path does not.
- **Why this matters:** A full TCP + TLS handshake (~100–300 ms) on every message send, on a path that already sits inside the sequential worker. It is also socket and file-descriptor churn under burst.
- **Trigger:** Every send.
- **Scale sensitivity:** Linear in send volume; the FD churn becomes noticeable under burst.
- **Suggested fix:** One module-level `httpx.AsyncClient` created in `lifespan` and closed on shutdown, used as the default when no client is passed. The `client=` parameter already exists everywhere, so this is a default-value change.
- **Risk of fix:** Low, provided the client is properly closed in `lifespan` and connection limits are set explicitly.
- **Expected benefit:** Latency (100–300 ms per send), resource usage.
- **Tests required:** A test that repeated `send_message` calls reuse one client instance.

---

## API-001 — Every deploy triggers a full message re-sync of every chat

- **Severity:** P2 Medium (intentional design, but the cost scales badly)
- **Category:** Cost / Capacity
- **Confidence:** Confirmed
- **Where:** `main.py::_chat_message_sync_needed` (684–705) and `_chat_last_message_ids` (668).
- **Current behavior:** The function returns `True` whenever `_chat_last_message_ids.get((creator_id, group_id))` is `None`, which is true for every chat after every process start. The docstring is explicit that this is deliberate: *"The first observation after a process restart intentionally returns true, providing a durable safety reconciliation without a new database column."*
- **Why this matters:** The reasoning is sound and the behaviour is correct — it is a genuine safety net. The problem is purely arithmetic: at 100 creators × 2,000 chats it is **200,000 `list_chat_messages` calls immediately after every Railway redeploy**, and Railway redeploys on every push. The map also never prunes, so it is one of the two real memory leaks (§4b).
- **Trigger:** Every process restart.
- **Scale sensitivity:** Free at 1 creator. At 20 creators it is a few thousand calls. At 100+ it is a bill and a thundering herd against the provider.
- **Suggested fix:** Persist the cursor. A `fans.last_synced_platform_message_id` column (or a small `chat_sync_cursors` table) keeps the safety property while making restarts free. Prune the in-memory map on the reconcile pass in the meantime.
- **Risk of fix:** Low — it strengthens the guarantee rather than weakening it.
- **Expected benefit:** Cost, capacity, memory.
- **Tests required:** A test that a restart with a persisted cursor does not re-sync unchanged chats, and that a missing cursor still does.

---

## API-002 — The dashboard polls API Fansly through the backend every 45 seconds

- **Severity:** P2 Medium
- **Category:** Cost / Frontend
- **Confidence:** Confirmed
- **Where:** `app/page.tsx` `ACTIVE_CHAT_RECENT_POLL_MS = 45_000` / `ACTIVE_CHAT_IDLE_POLL_MS = 3 * 60_000` (33–34), the `reconcile` loop (595–661), `syncActiveFanMessages` (38–67) → `POST /sync-fan-messages/{c}/{f}` → `main.py::_sync_recent_fan_messages` → `apifansly_list_chat_messages`.
- **Current behavior:** Every open conversation polls the backend every 45 s (recent activity) or 3 min (idle), and each poll makes a real API Fansly call. This runs *in addition to* the webhook and the backend's own 5–30 minute chat reconciliation — three mechanisms doing the same job.
- **Why this matters:** ~80 API Fansly calls per hour **per open browser tab**. At 25 concurrent operators that is ~2,000 calls/hour from browser polling alone, which is likely the single largest API Fansly line item during business hours. The polling is well-behaved in isolation (visibility-gated, in-flight-guarded, backs off on error) — the issue is that it duplicates two other mechanisms.
- **Trigger:** Any open conversation.
- **Scale sensitivity:** Linear in concurrent operators.
- **Suggested fix:** Rely on realtime for the open thread and reserve the API Fansly call for an explicit refresh, a realtime-recovery catch-up, or a long interval (5+ min). The `recoveryTick` catch-up effect already handles the "socket was dead" case that this polling was presumably added for.
- **Risk of fix:** Medium — this polling likely compensates for a real gap (attachments arriving without media URLs). Confirm the realtime path covers it before lengthening the interval.
- **Expected benefit:** Cost.
- **Tests required:** A dashboard test that the poll interval is respected and that realtime alone surfaces a new message.

---

## FE-001 — `dedupeMessages` is O(n²) and runs twice per incoming message

- **Severity:** P1 High
- **Category:** Frontend / Latency
- **Confidence:** Confirmed (MEASURED)
- **Where:** `lib/messages.ts::dedupeMessages` (43–61), `deliveryDuplicate` (9–32), `normalizedContent` (5–7); called at `app/page.tsx` 466 (initial load), 508 (load more), 683 and 687 (realtime INSERT — **twice per message**).
- **Current behavior:** For each message, `result.findIndex(...)` scans everything accumulated so far, and each comparison calls `normalizedContent()` on both operands — allocating two strings via `.trim().replace(/\s+/g,' ')` per comparison, with no memoisation. Measured on Node 22:

  | Loaded messages | Cost of appending one realtime message |
  |---|---|
  | 200 | 31 ms |
  | 500 | 155 ms |
  | 1,000 | 667 ms |
  | 2,000 | 2,368 ms |
  | 5,000 | 15,397 ms |
  | 10,000 | 63,030 ms |

  In the realtime handler this runs **twice** (once for `messagesCache`, once for `tab.messages`), synchronously inside a `setTabs` updater on the browser main thread.
- **Why this matters:** The browser tab freezes for the duration. At 1,000 loaded messages that is ~1.3 s of unresponsive UI **per incoming message** — and during an active conversation messages arrive in bursts. The initial 50-message window hides this; it appears the moment an operator scrolls back, and `loadMoreMessages` has no upper bound.
- **Trigger:** An operator scrolling back through history, then receiving messages.
- **Scale sensitivity:** Quadratic in retained messages. Fine below ~200; degraded at 500; unusable past ~2,000.
- **Suggested fix:** Index by `id` and `fansly_message_id` in a `Map` for O(1) exact-match dedupe, and bucket by rounded `sent_at` (15 s window) so the reconciliation case — one local row without a platform id, one imported row with it — only compares within a bucket. Separately, cap retained history at ~500 messages, dropping the oldest.
- **Risk of fix:** Low, but the reconciliation semantics in `deliveryDuplicate` are subtle and deliberately conservative (it will not collapse two rows that both lack a platform id) — preserve them exactly.
- **Expected benefit:** Frontend responsiveness. Removes a measured multi-second main-thread freeze.
- **Tests required:** Port the existing dedupe semantics into unit tests **first** (there are none today), including the "two unidentified rows are not duplicates" and "richer message wins" cases, then a performance assertion at 5,000 messages.

---

## FE-002 — The vault grid downloads full-resolution originals to render 100×100 thumbnails

- **Severity:** P1 High
- **Category:** Frontend / Cost
- **Confidence:** Confirmed
- **Where:** `app/vault/page.tsx:799`; `thumbnail_url` is selected at line 260 and populated by `main.py::_vault_media_visual_urls` (2184–2202).
- **Current behavior:** `<img src={item.url} loading="lazy" style={{width:100, height:100, objectFit:'cover'}} />`. `item.url` is the original Fansly CDN asset. `_vault_media_visual_urls` walks `media.variants` for a real image variant and only falls back to the original when none exists, so a genuine thumbnail is available for most items — it is simply not used. With `vaultVisibleLimit = 200`, opening an album renders 200 originals.
- **Why this matters:** At a typical 2–4 MB per original, **400–800 MB of image transfer for one album view**. `loading="lazy"` only defers off-screen images. This is the largest single bandwidth item in the dashboard and it is a one-line fix.
  Secondary defect on the same line: the `onError` handler sets `parentElement.innerHTML`, mutating DOM React owns; later reconciliation of that subtree can throw `NotFoundError: The node to be removed is not a child of this node`.
- **Trigger:** Opening any vault album.
- **Scale sensitivity:** Fixed cost per album view regardless of vault size (capped at 200 images), so it hurts equally at every scale.
- **Suggested fix:** `src={item.thumbnail_url || item.url}`. Replace the `onError` mutation with an error-state flag rendered by React.
- **Risk of fix:** Low. Items whose `thumbnail_url` equals `url` are unaffected.
- **Expected benefit:** Bandwidth, page load, browser memory.
- **Tests required:** A component test asserting `thumbnail_url` is preferred and that a load error renders the placeholder without DOM mutation.

---

## FE-003 — The entire vault is loaded into browser state and reloaded on every change

- **Severity:** P1 High
- **Category:** Frontend / Memory
- **Confidence:** Confirmed
- **Where:** `app/vault/page.tsx::loadVaultMedia` (252–273); the realtime `refreshVaultSoon` debounce (296–320); `selectedVaultItems` (495–499).
- **Current behavior:** `loadVaultMedia` pages the whole `creator_vault_media` table for a creator — **26 columns** including `ai_description`, `tags`, `good_for`, and both URLs — awaiting each 1,000-row page in a `while` loop, into `vaultAlbums` state. The realtime handler re-runs the **entire** load 750 ms after any change. `selectedVaultItems` recomputes `Object.values(vaultAlbums).flat()` on **every render** with no `useMemo`.
- **Why this matters:** At 10,000 items: ~15 MB of JSON, 10 sequential requests, ~50–75 MB of retained heap. At 50,000: ~75 MB, 50 sequential requests (~25–50 s of waiting), ~250–375 MB heap — a plausible tab OOM. During a categorisation run the debounced realtime handler re-downloads all of it repeatedly. And the `flat()` allocates a 10,000-element array on every keystroke in the preview modal.
- **Trigger:** Selecting a creator on the vault page; any `creator_vault_media` change while it is open.
- **Scale sensitivity:** Comfortable to ~1,000 items; degraded at 10,000; unusable at 50,000.
- **Suggested fix:** Fetch per album with a projection (id, thumbnail_url, mimetype, content_category, album_title) and load the heavy fields only for the previewed item. Make the realtime handler patch the changed rows in place instead of reloading. Wrap `selectedVaultItems` in `useMemo`.
- **Risk of fix:** Medium — the album grouping and the "all" view both assume the full set is in memory.
- **Expected benefit:** Frontend responsiveness, memory, bandwidth.
- **Tests required:** A test that a realtime update patches rather than reloads; a test that switching albums fetches only that album.

---

## FE-005 — Background creator tabs receive no realtime events and never catch up

- **Severity:** P2 Medium
- **Category:** Frontend / Stale state
- **Confidence:** Confirmed
- **Where:** `app/page.tsx` realtime effect (665–810) — `filter: creator_id=eq.${cid}` where `cid = activeTab.creatorId`, with `[activeTab?.creatorId, recoveryTick]` deps; the conversation-load effect's early return at 407.
- **Current behavior:** The channel subscribes to one creator, yet the handlers map over all tabs matching `tab.creatorId === msg.creator_id`. Tabs for other creators therefore receive nothing. On switching back, the conversation-load effect returns early because `activeTab.conversations.length > 0`, and `conversationsCache` still holds the stale list. The `recoveryTick` catch-up effect only refreshes the currently-open thread's messages, not the newly-active creator's conversation list.
- **Why this matters:** A multi-creator operator — the normal case for an agency — sees stale conversation lists, stale last-message previews, and stale unread counts for every creator except the active one, indefinitely. There is no visible indication the data is old.
- **Trigger:** Having more than one creator tab open.
- **Scale sensitivity:** Worse with more creators per operator.
- **Suggested fix:** Either subscribe one channel per open tab's creator, or subscribe without the creator filter and rely on RLS plus the existing per-tab matching. Additionally, refetch the conversation list on creator switch when the cached list is older than a threshold.
- **Risk of fix:** Low.
- **Expected benefit:** Correctness, operator trust.
- **Tests required:** A test that a message for a non-active tab's creator updates that tab's conversation list.

---

## FE-006 — Unconditional `fans` updates in `sync_chats` become a dashboard realtime storm

- **Severity:** P2 Medium
- **Category:** Frontend / DB
- **Confidence:** Confirmed
- **Where:** `main.py::sync_chats` (1885–1890); consumed by the `fans` UPDATE subscription in `app/page.tsx` (765–800) and `components/FanPanel.tsx` (277–300).
- **Current behavior:** For every chat on every reconciliation pass, `sync_chats` executes `fans.update({"fansly_group_id": ..., "display_name": ...})` with no comparison against the current values. Every one of those UPDATEs is a realtime event delivered to every subscribed dashboard, each triggering a `setTabs` that rebuilds the conversations array and re-renders the un-memoised `Sidebar`.
- **Why this matters:** For a creator with 2,000 chats that is 2,000 no-op UPDATEs every 10 minutes, arriving as a burst. Combined with FE-004 (no `React.memo`, no `useMemo` on the filter) this produces a visible periodic freeze in the dashboard, and it wastes 2,000 writes per pass on the database side.
- **Trigger:** Every chat reconciliation pass.
- **Scale sensitivity:** Linear in chats per creator.
- **Suggested fix:** Compare before writing: only update when `display_name`, `avatar_url`, or `fansly_group_id` actually changed.
- **Risk of fix:** Low.
- **Expected benefit:** DB write volume, dashboard responsiveness, realtime quota.
- **Tests required:** A test that an unchanged chat produces no `fans` update.

---

## VAULT-001 — The autosync scheduler starts every due creator's sync simultaneously

- **Severity:** P1 High
- **Category:** Capacity / Async
- **Confidence:** Confirmed
- **Where:** `main.py::vault_autosync_scheduler` (622–657), specifically `res = await sync_vault_start(cid)` inside the creator loop; `sync_vault_start` (2291–2312) ends with `spawn(_run_vault_sync(creator_id))`.
- **Current behavior:** `sync_vault_start` spawns a background task and returns immediately, so the `await` in the scheduler loop provides **no serialisation**. Every creator whose `last_vault_sync_at` has crossed 24 hours in the same hourly pass starts a concurrent sync, each of which runs `_run_vault_categorization` at `VAULT_CATEGORIZATION_CONCURRENCY` (default 12 when a semantic endpoint is configured). There is no global cap.
- **Why this matters:** 100 due creators means 100 concurrent vault syncs and up to **1,200 concurrent media classifications** inside the web process — saturating CPU, memory, the shared asyncio thread pool (which every Supabase call also uses), and the Modal vision endpoint. Because creators are typically connected in batches, their 24-hour anniversaries naturally cluster, so this is likely rather than theoretical. The knock-on effect is the cascade path in §17: slow DB calls → slow webhook → provider redelivery → more load.
- **Trigger:** Multiple creators' 24-hour vault-sync anniversaries landing in the same hourly pass.
- **Scale sensitivity:** Harmless below ~5 creators. Dangerous above ~20.
- **Suggested fix:** A module-level `asyncio.Semaphore(N)` (N = 1–2) acquired inside `_run_vault_sync`, so the scheduler may start many but only N run. Optionally stagger start times with jitter.
- **Risk of fix:** Low.
- **Expected benefit:** Capacity, stability, and protection of chat latency from vault work.
- **Tests required:** A test that N+3 concurrent `_run_vault_sync` calls never exceed N in flight.

---

## VAULT-002 — Batch-barrier categorisation with one write per item

- **Severity:** P2 Medium
- **Category:** Capacity / DB
- **Confidence:** Confirmed
- **Where:** `main.py::_run_vault_categorization` (3712–3752).
- **Current behavior:** A fixed-window `for i in range(0, total, batch_size)` loop gathers `batch_size` items and waits for **all** of them, then persists results with **one `UPDATE` per item, sequentially**.
- **Why this matters:** Two compounding inefficiencies. The barrier means a batch of 12 finishes only when its slowest member does — a video needing ffmpeg extraction can take 35 s while 11 images take 1 s — so effective concurrency is far below 12. And 10,000 items means 10,000 sequential UPDATEs (~250–500 s of pure DB latency) on top of the classification itself.
- **Trigger:** Any vault categorisation run.
- **Scale sensitivity:** Linear in vault size; the barrier effect worsens with the proportion of videos.
- **Suggested fix:** Replace the fixed batches with a worker pool (`Semaphore(batch_size)` + `asyncio.as_completed`), and accumulate results into batched `upsert` calls of ~100 rows.
- **Risk of fix:** Low–medium — the `provider_failures >= 3` abort logic assumes batch boundaries and needs rework.
- **Expected benefit:** 2–5× faster categorisation, far fewer DB round trips.
- **Tests required:** A test that concurrency stays bounded; a test that the three-failure abort still triggers; a test that every item is written exactly once.

---

## VAULT-003 — In-flight vault work is lost on restart and reported as idle

- **Severity:** P2 Medium
- **Category:** Reliability / Observability
- **Confidence:** Confirmed
- **Where:** `main.py::_vault_sync_state` (130), `_categorize_state` (3321), `sync_vault_status` (2318–2321), `categorize_vault_status` (3437–3438).
- **Current behavior:** Both are plain module dicts. On restart the spawned task dies and the state vanishes; the status endpoints return `{"status": "idle"}` for a job that was killed mid-run. Classified rows are persisted, and because `_stamp_vault_op("last_vault_sync_at")` runs only at the end, the hourly scheduler retries within the hour — so the *work* recovers.
- **Why this matters:** The operator-visible state is wrong, there is no durable record that a job was interrupted, and there is no way to answer "why did vault sync stop?" (§14). It also means the in-process single-flight guard (`if _vault_sync_state.get(cid).status in ACTIVE`) disappears on restart, so a second sync can start while the first is still finishing elsewhere — currently impossible with one worker, but it is one of the things that must move out of process before multi-worker.
- **Trigger:** Any Railway restart during a vault sync or categorisation.
- **Scale sensitivity:** Probability rises with sync duration, which rises with vault size.
- **Suggested fix:** Persist job state in a small `vault_jobs` table (creator_id, kind, status, started_at, updated_at, done, total, error) and have the status endpoints read it. Mark stale `running` rows as `interrupted` on startup.
- **Risk of fix:** Low.
- **Expected benefit:** Observability, and a prerequisite for multi-worker.
- **Tests required:** A test that a restart marks an in-flight job `interrupted` rather than reporting `idle`.

---

## DEAD-004 — The `/fansly/*` router is registered, reachable, and structurally broken

- **Severity:** P2 Medium
- **Category:** Dead Code / Bug
- **Confidence:** Confirmed (proven by execution)
- **Where:** `routes/fansly.py:8` (`from main import fansly_poller, session_store`); `main.py:586–592` (globals initialised to `None`), `main.py:5797–5799` (`include_router`), `main.py::lifespan` (837–870, where the globals are actually assigned).
- **Current behavior:** `routes/fansly.py` binds the module-level values **at import time**, when both are still `None`. `lifespan` later assigns `main.session_store` and `main.fansly_poller`, but the names in `routes.fansly` remain bound to the original `None`. Proven by execution:
  ```
  main.session_store = None
  routes.fansly.session_store = None
  ```
  and confirmed reachable — `app.openapi()["paths"]` lists `/fansly/connect`, `/fansly/health`, `/fansly/accounts/{id}/groups`, `/fansly/accounts/{id}/groups/{gid}/messages`. Three of the four call `session_store.<method>` unguarded and will raise `AttributeError` → HTTP 500. `/fansly/health` has an `if session_store else {}` guard and returns misleading empty state.
- **Why this matters:** Four live production endpoints that cannot work. They belong to the superseded direct-Fansly-session integration (auth token / client id / session cookie / proxy), replaced by API Fansly. The surrounding modules — `services/fansly_client.py`, `services/fansly_session_store.py`, `services/fansly_poller.py` — are the same generation, and `SessionStore` is still constructed in `lifespan` with a **required** `FANSLY_SESSION_KEY` env var, so this dead path also imposes a deployment requirement. It additionally imports `cryptography`, which is **not declared in `requirements.txt`** (it arrives transitively) — a transitive-dependency change would break startup.
- **Trigger:** Any call to a `/fansly/*` endpoint.
- **Scale sensitivity:** Independent of scale.
- **Suggested fix:** Confirm no agency has live rows in `fansly_sessions`, then remove the router, the three services, the `lifespan` wiring, and the `FANSLY_SESSION_KEY` requirement. If it must be kept, change the import to `import main` and reference `main.session_store` at call time.
- **Risk of fix:** Low once the `fansly_sessions` table is confirmed empty.
- **Expected benefit:** Maintainability, one fewer required env var, removal of an undeclared dependency.
- **Tests required:** None for deletion. If retained, a test that the routes work after `lifespan` has run.

---

## OBS-001 — `/health` is static and cannot detect any real failure

- **Severity:** P2 Medium
- **Category:** Observability / Deployment
- **Confidence:** Confirmed
- **Where:** `main.py::health` (5780–5788); `workers/scheduled_actions.py::worker_health_snapshot` (37–48) exists and is **never exposed**.
- **Current behavior:** Returns a fixed dict: `status: "ok"`, the classifier version, and whether a semantic URL is configured. It never touches the database, the schedulers, the queue, or any provider. It is one of only two paths in `_PUBLIC_PATHS`, so it is what Railway's healthcheck hits.
- **Why this matters:** A deploy with wrong Supabase credentials reports healthy. A crashed scheduler task reports healthy. A 10,000-action backlog reports healthy. The one function that already computes scheduler liveness (`worker_health_snapshot`, with `last_run_started_at`, `last_run_completed_at`, `last_error`, `last_sent`) is written and unused.
- **Trigger:** Any infrastructure failure that does not crash the process.
- **Scale sensitivity:** Severity rises with scale, because more things can fail silently.
- **Suggested fix:** Extend `/health` with: a cheap DB round trip, `worker_health_snapshot()`, seconds since each scheduler's last cycle, the `scheduled_actions` PENDING count and oldest `execute_at`, and `current_model_availability()`. Return a degraded status when the queue is deep or a scheduler has not completed a cycle recently.
- **Risk of fix:** Low, but the DB check must be cheap and time-bounded so the healthcheck itself cannot become a load source.
- **Expected benefit:** Observability. This is the prerequisite for operating anything above the current scale.
- **Tests required:** A test that a failing DB check produces a degraded status; a test that the endpoint stays under a latency budget.

---

## MEM-001 — Two in-memory maps grow per fan and are never pruned

- **Severity:** P3 Low
- **Category:** Memory
- **Confidence:** Confirmed
- **Where:** `main.py::_chat_last_message_ids` (668, written by `_remember_chat_message_id` at 707–715) and `_active_chat_binding_retry_after` (132, written in `_resolve_active_chat_group_id` at 194–221).
- **Current behavior:** `_chat_last_message_ids` gains one `(creator_id, group_id) → message_id` entry per chat ever observed and is never removed. `_active_chat_binding_retry_after` is popped **only on successful binding** (line 216), so every fan whose group-id lookup fails leaves a permanent float.
- **Why this matters:** At 100 creators × 1,000 chats that is ~100,000 tuple-keyed entries (~15 MB with Python object overhead) growing monotonically for the life of the process. Not fatal — Railway restarts clear it, which is also why it has not been noticed — but it is unbounded growth in a long-lived process, and `_chat_last_message_ids` is one of the structures that must move out of process before multi-worker anyway (API-001).
- **Trigger:** Steady-state operation.
- **Scale sensitivity:** Linear in total chats and in fans with failed bindings.
- **Suggested fix:** Prune both on the chat-reconciliation pass, dropping entries for creators no longer active and retry timestamps already in the past. Longer term, persist the chat cursor (API-001) and drop the in-memory map entirely.
- **Risk of fix:** Low.
- **Expected benefit:** Memory stability.
- **Tests required:** A test that the reconcile pass removes entries for a creator that no longer exists.

---

## REL-004 — `_processed_messages` discards all dedupe state at the 1,000 boundary

- **Severity:** P3 Low
- **Category:** Race
- **Confidence:** Confirmed
- **Where:** `main.py::handle_new_fan_message` (549–551) and `generate_suggestions_webhook` (4043–4045).
- **Current behavior:** `if len(_processed_messages) > 1000: _processed_messages.clear()` — the entire set is discarded at the boundary rather than evicting the oldest entries.
- **Why this matters:** Immediately after a clear, a poller message that the webhook handled moments earlier is no longer recognised as processed. `save_message`'s DB check still prevents a duplicate *row*, so the exposure is a second `process_incoming_fan_message` run — one wasted analyzer call and one redundant `schedule_auto_reply` (which is idempotent via `cancel_actions_for_fan` plus the dedupe key). Small, but it is a self-inflicted correctness hole with a trivial fix.
- **Trigger:** Every 1,000 messages.
- **Scale sensitivity:** More frequent at higher volume.
- **Suggested fix:** A bounded FIFO — `collections.deque(maxlen=1000)` plus a companion `set`, or an `OrderedDict` with `popitem(last=False)`.
- **Risk of fix:** Low.
- **Expected benefit:** Correctness, cost.
- **Tests required:** A test that the 1,001st insert evicts only the oldest entry and that the 1,000th-most-recent id is still recognised.

---

## CI-001 — Ruff is installed and never run; there is no backend type checking

- **Severity:** P2 Medium
- **Category:** Testing / CI
- **Confidence:** Confirmed
- **Where:** `.github/workflows/ci.yml` (the job runs `pytest -q` only); `requirements-dev.txt` (`ruff==0.16.0`).
- **Current behavior:** Backend CI installs `ruff` and never invokes it. There is no `compileall`, no type checker, and no verification that `db/*.sql` has been applied to the target environment. Running `ruff check` now yields **339 findings**, including 5 `B023` loop-variable captures, 10 unused imports, 3 unused variables, and 1 redefinition — every one of which CI would have caught. The dashboard's `npm run lint` passes with **111 warnings** because nothing is configured as an error, and `react-hooks/exhaustive-deps` warnings are exactly the defect class behind FE-005 and FE-007.
- **Why this matters:** The tooling to catch a meaningful subset of this report's findings is already installed and simply not wired up.
- **Trigger:** Every pull request.
- **Scale sensitivity:** Independent of scale; the cost compounds with codebase age.
- **Suggested fix:** Add `ruff check .` to backend CI (start with `--select E9,F,B` to gate on real errors while the 681 line-length findings are addressed separately). Add `--max-warnings 0` to the dashboard lint after clearing the current 111. Consider `mypy` on `ai/`, `core/`, `db/`, `models/` — the typed, self-contained modules — rather than the whole tree.
- **Risk of fix:** Low, provided the initial rule selection is narrow enough to pass.
- **Expected benefit:** Maintainability; prevents recurrence of several finding classes here.
- **Tests required:** N/A — this is the test infrastructure.

---

## DB-000 — The base schema is not in version control

- **Severity:** P1 High
- **Category:** DB / Deployment / Testing
- **Confidence:** Confirmed
- **Where:** `db/*.sql` — 18 files, all `ALTER TABLE` / `CREATE INDEX` additions. No `CREATE TABLE` for `creators`, `fans`, `messages`, `suggestions`, `chatter_creators`, `ppv_offers`, `vault_sets`, `creator_vault_media`, `fan_lists`, `fan_list_members`, `scheduled_actions`, `reengagement_log`, `reengagement_settings`, `scripts`, `blocked_words`, `message_embeddings`, `fansly_sessions`, or the `fan_conversation_summaries` view.
- **Current behavior:** These objects exist only in the live Supabase project. Their primary keys, foreign keys, unique constraints, cascade behaviour, nullability, and indexes cannot be reviewed, reproduced, or tested.
- **Why this matters:** This is the finding that limits every other database finding in this report. The hottest query in the product — `messages WHERE fan_id = ? ORDER BY sent_at DESC LIMIT 40`, executed several times per message — has no index definition anywhere in version control. `chatter_creators(chatter_id)` is consulted on every authenticated request **and** inside every RLS policy evaluation, and is likewise unverifiable. Whether `messages.fansly_message_id` is unique determines whether REL-002 is a real duplicate-row risk or merely a wasted round trip. A new environment cannot be built from the repository, and CI's Postgres service can only test the additive migrations.
- **Trigger:** Every schema question, every environment rebuild, every index review.
- **Scale sensitivity:** The cost of not knowing rises sharply with data volume.
- **Suggested fix:** `pg_dump --schema-only` the production database, commit it as `db/000_base_schema.sql`, and have CI apply it before the additive migrations so the schema tests run against the real shape. Then review indexes against the query inventory in §5b and add what is missing.
- **Risk of fix:** Low — it is a documentation and CI change, not a production one. The follow-on index work is separate.
- **Expected benefit:** Everything downstream: reviewability, reproducibility, real schema tests, and the ability to answer the index questions this audit had to mark UNKNOWN.
- **Tests required:** CI applies `000_base_schema.sql` plus all migrations to a clean Postgres and the existing schema tests pass against it.
