# Full Auto capacity: concurrency, admission control, durable ingestion

This describes the operating envelope of the **current single-process
deployment** — one Uvicorn process, no `--workers`, every background scheduler
in-process. Nothing here prepares for a web/worker split; it removes the first
practical Full Auto capacity ceiling inside the shape that exists today.

## 1. Where the ceiling was

`workers/scheduled_actions.py::process_once` claimed up to 20 due actions and
processed them in a plain `for` loop, awaiting each to completion. One
`AUTO_REPLY` is ~43 DB round trips, an analyzer call, a writer call, a
deliberate composition delay, and a platform send. After the batch, the
scheduler slept a flat 60 seconds whether or not backlog remained.

So the ceiling was deployment-wide, not per creator and not per agency: one
agency's burst sat in front of every other agency's fans in a single global
FIFO, on an otherwise idle machine.

## 2. Bounded concurrent execution

```
claim_due_actions(limit = SCHEDULED_ACTION_CLAIM_LIMIT)
            |
    group_actions_by_fan()          <- the safety mechanism
            |
   +--------+--------+--------+
   |        |        |        |
 fan A    fan B    fan C    fan D    <- run in parallel
   |        |        |        |
 a1->a2    b1       c1       d1->d2  <- serial WITHIN a fan
   |        |        |        |
   +--------+--------+--------+
            |
   asyncio.Semaphore(SCHEDULED_ACTION_CONCURRENCY)
```

Per-fan grouping is the whole per-fan safety story, and it is deliberately the
simplest thing that works: actions for one fan share one chain, so two
conflicting sends to one conversation cannot be in flight together. No lock, no
registry, no distributed coordination.

Everything that protected a single conversation before is still in force
underneath it and unchanged:

- the `AUTO_REPLY` dedupe key and `_pending_auto_replies` single-flight registry
- `cancel_actions_for_fan` (which cancels PENDING, FAILED **and** PROCESSING)
- the `PROCESSING` status compare-and-swap on claim, complete, fail and reschedule
- `_should_still_send`, including the expected-trigger-timestamp check
- the post-generation history re-check
- the PPV and proactive delivery journals
- "freeze rather than duplicate" when a send succeeded but persistence failed

### Polling

- A **full** claim is treated as evidence of backlog: the next cycle starts
  after `BUSY_POLL_SECONDS` (0.25 s), capped at `MAX_CONSECUTIVE_BUSY_CYCLES`
  (60) so an unproductive claim cannot become a busy loop.
- A **short** claim waits `SCHEDULED_ACTION_POLL_SECONDS` (5 s) on an event, so
  work enqueued inside this process (an accepted webhook) starts immediately
  rather than waiting out the interval.
- Obligation repair runs on its own 60 s cadence rather than once per claim, so
  a fast poll does not multiply repair cost.

### Reclaim window

`claim_due_actions(stale_minutes=10)` is **unchanged**. Worst-case action
duration under concurrency is bounded by: composition delay ≤22 s, up to three
inter-part delays ≤14 s each, two model calls, and gate wait ≤ roughly one model
call. That is well under three minutes against a ten-minute window, so a
legitimately slow action still cannot be reclaimed and run twice.

## 3. Model admission control

The gate lives at `ai.model_providers.complete` — the single function every
paid generation goes through — so it covers the Kimi/OpenRouter writer, the
Together commercial route, the analyzer, and the background extractor
with one budget. Wrapping `generate_replies` alone would have left the analyzer
unbounded, which is the specific mistake this placement avoids.

Before this sprint the sequential worker *accidentally* held Full Auto model
concurrency to one. Concurrency removes that accident, which is why the two ship
together.

- A burst larger than the limit **queues**. It is never dropped, never retried
  into the provider, and never opens more upstream connections than the limit.
- The wait is a plain `await` on an `asyncio` primitive, so cancelling the
  action cancels the wait and no stale request reaches the provider.
- The slot is released on success, exception, timeout and cancellation alike.
- `MODEL_MAX_CONCURRENCY` is an **application policy**. It is deliberately not
  derived from the OpenAI SDK's `max_connections`, which is a transport ceiling.

Routing is untouched: the OpenRouter provider pin, the explicit Together
fallback route, and session affinity for prefix caching all behave exactly as
before. The gate sits above routing, not inside it.

## 4. Follow-up obligation repair

The invariant is unchanged — if commercial state says a follow-up is owed and
its durable action is missing or terminal, recreate it. Only the cost changed.

| | Before | After |
|---|---|---|
| Scan | every row with `next_followup_at IS NOT NULL` | only `next_followup_at <= now + 5 min` |
| Action lookup | one `SELECT` per obligation | one query per 200 dedupe keys |
| Write | one conditional write per obligation | one bulk upsert per 200 rows |
| 200 obligations | **401 round trips**, every 60 s | **3 round trips** |

`action_needs_repair()` is shared with `ensure_action_pending`, so the batched
and single-row paths cannot drift: missing, `COMPLETED`, or `FAILED` with the
known rolling-deploy compatibility error. `PENDING`, `PROCESSING`, `CANCELLED`
and other `FAILED` states are left alone.

The 5-minute horizon is sized against the polling semantics: idle poll is 5 s
with a 60 s repair cadence, so the horizon carries roughly five repair cycles of
slack for clock skew or a stalled cycle before an action would be needed.

## 5. Durable webhook ingestion

```
POST /webhook/fansly (messages.received)
 ├─ HMAC signature verification
 ├─ minimal shape validation
 ├─ tenant resolution (creator by apifansly_account_id)
 ├─ get_or_create fan
 ├─ save_message()            <- unique index on messages(fansly_message_id)
 ├─ schedule_action(PROCESS_INBOUND_MESSAGE, replace_existing=False)
 └─ HTTP 200 {"status": "accepted"}      <<< ACK BOUNDARY

   ... later, in the scheduled-actions worker ...

PROCESS_INBOUND_MESSAGE
 ├─ media enrichment (live API Fansly call)
 └─ process_incoming_fan_message()  -> analyzer, writer, Auto scheduling
```

**The acknowledgement point** is after `save_message` and `schedule_action` have
both returned. After that, a Railway restart cannot lose the obligation to
process the message. `asyncio.create_task` would not have been durable, which is
why it is not used here.

If either write fails, the route raises **HTTP 503** and does not acknowledge —
the platform redelivers, which is now harmless.

**Idempotency** comes from two database constraints, not process memory:

- `messages(fansly_message_id)` unique → one canonical message. `save_message`
  now resolves a losing concurrent insert to the winner's row instead of
  failing, so a race produces a 200, not a 5xx that provokes another redelivery.
- `scheduled_actions(dedupe_key)` unique, written with `replace_existing=False`
  → one effective obligation. A redelivery never resets an action that is
  already `PENDING` (would run twice), `PROCESSING` (would race), or `COMPLETED`
  (would reprocess), but a crash between the message insert and the action
  insert *is* repaired by the next redelivery.

The poller fallback (`handle_new_fan_message`) shares the same acceptance path
and the same dedupe key, so webhook-and-poller delivery of one message no longer
risks a second analyzer and writer pass.

Assisted and Full Auto semantics are unchanged; both simply happen in the worker
now. The webhook waits for neither.

## 6. Health

Three questions, three answers:

| | Endpoint | Fails on |
|---|---|---|
| Liveness | `/health` (always 200) | nothing external |
| Readiness | `/health/ready` (503) | database unreachable, and only that |
| Degraded | `status` field on both | queue age/depth, stale scheduler, saturated gate, provider unavailable |

A throttled model provider produces `status: "degraded"` and **never** a
non-2xx, because restarting a container that is holding a durable queue turns a
provider incident into an outage.

Defaults: `HEALTH_QUEUE_MAX_AGE_SECONDS=900`, `HEALTH_QUEUE_MAX_DEPTH=500`,
scheduler stale after `max(120 s, 6 × poll interval)`. The queue-age threshold is
deliberately far above normal composition delays (≤22 s) so intended human-like
timing never alerts. Probes are time-bounded (1.5 s DB, 2.5 s queue) and cached
for 5 s so the healthcheck cannot itself become a load source.

The payload carries counts, ages, statuses and configured limits only — no
message content, prompts, fan names, tokens or provider keys, and database
errors are reported by exception type rather than message.

## 7. Per-action timing

One structured `[ACTION TIMING]` log line per action, no extra database writes:

```
queue_wait_ms  revalidation_ms  db_context_ms  model_gate_wait_ms
analyzer_ms    writer_ms        composition_delay_ms  availability_delay_ms
fansly_send_ms persistence_ms   total_processing_ms   model_calls
```

`model_gate_wait_ms` is also written to model telemetry metadata, so "the
provider took 5 s" and "we waited 4 s for a local slot and the provider took 1 s"
are distinguishable. `composition_delay_ms` is separate from everything else
precisely so intended human realism is never mistaken for a capacity problem.

## 8. Load harness

`scripts/load_test_scheduled_actions.py` drives the **real** worker and the
**real** model gate against stubbed externals with configurable latency. It
never touches production Fansly, live creators, or a real provider.

```
python scripts/load_test_scheduled_actions.py --sizes 10 50 100
python scripts/load_test_scheduled_actions.py --sequential   # the pre-sprint shape
```

`tests/test_scheduled_action_load_harness.py` runs the same harness at
compressed latencies on every commit.

### Measured results

Model latency 2.0 s per call (two calls per reply), composition delay 6.8 s (the
value the audit measured), API send 150 ms, DB 4 ms per round trip.

| Burst | Before: drain | Before: actions/min | After: drain | After: actions/min | Speed-up |
|---|---|---|---|---|---|
| 10 | 110.8 s | 5.4 | **22.2 s** | **27.1** | 5.0x |
| 50 | 554.7 s | 5.4 | **78.1 s** | **38.4** | 7.1x |
| 100 | 1109.4 s | 5.4 | **145.1 s** | **41.3** | 7.6x |

Peak action concurrency 8/8, peak model concurrency 8/8, zero duplicate sends,
zero same-fan overlaps, zero errors at every size.

Two things the table deliberately does not claim:

- **The "before" column is generous to the old design.** It runs the harness at
  concurrency 1 but still uses the new short inter-cycle poll. The real old loop
  slept a flat 60 s after every batch of 20, which puts it at roughly 170 s /
  3.5 per min for 10, 734 s / 4.1 for 50, and 1407 s / 4.3 for 100 — matching
  the audit's estimate of 3-6 replies per minute for the whole deployment.
- **Per-action completion time did not improve, and should not have.** p50 and
  p95 are 11.07 s in both columns, because 6.8 s of that is the deliberate
  composition delay and 4 s is provider latency. What collapsed is the time a
  fan spends waiting *behind other fans*. That is the correct outcome: parallel
  conversations, not faster robotic sends.

## 9. Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SCHEDULED_ACTION_CONCURRENCY` | 8 | Independent fans processed at once |
| `SCHEDULED_ACTION_CLAIM_LIMIT` | 24 | Actions locked per claim (3× concurrency) |
| `MODEL_MAX_CONCURRENCY` | 8 | Global simultaneous model calls |
| `SCHEDULED_ACTION_POLL_SECONDS` | 5 | Idle poll; a full batch re-polls immediately |
| `HEALTH_QUEUE_MAX_AGE_SECONDS` | 900 | Degraded above this oldest-pending age |
| `HEALTH_QUEUE_MAX_DEPTH` | 500 | Degraded above this pending depth |

Every value is safe to deploy unchanged. An unparseable value falls back to the
default rather than failing the deploy.
