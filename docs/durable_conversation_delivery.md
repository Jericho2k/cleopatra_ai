# Durable conversation delivery: supersession, timing, intentions, leases

This describes the layer that sits between "Core v1 has decided what to say" and
"the fan sees it". It replaces a design that was correct inside one Python
process and correct nowhere else.

It does **not** change Core v1. The split is unchanged:

```
raw conversation + sourced memory + working context
    -> GLM: semantic judgement / conversation decision ONLY
    -> deterministic application authority
    -> Kimi: ALL fan-facing generated wording
    -> deterministic execution / delivery / persistence
```

Everything below is the last line of that diagram.

## 1. What was wrong with the old shape

Full Auto's interruption story was `services.suggestions._pending_auto_replies`
— a dictionary of `asyncio.Task` — plus `_sleep_while_current()`, which woke
every 0.5 s to ask whether a newer task had replaced it.

The 0.5 s wake is not a database query, and a sleeping task is cheap. That was
never the problem. The problem is what the design *cannot* do:

| Property | Old | Now |
|---|---|---|
| Cancellation | process-local dictionary | durable row read at the send boundary |
| Correctness under two workers | depends which process got the event | independent of which process got the event |
| A 9 s inter-bubble pause | holds a live coroutine and a worker slot | one indexed row with a due time |
| Restart between bubble 1 and 2 | bubble 2 is gone | bubble 2 is durable and revalidated |
| Cancelling from another process | impossible | the generation moved; the gate refuses |

Separately, Core v1's delivery path had lost the timing mathematics altogether:
`services/human_delivery.py` still computed availability modes, reading time,
composition time and jittered inter-bubble pauses, and
`live_orchestration._deliver_plain_parts()` simply did not use any of it.

## 2. Conversation generation

`fans.conversation_generation` is a monotonically increasing integer.

**One place decides it moved**: `db.queries.save_message_result`, the chokepoint
every message write already funnels through. `services.conversation_generation.bumps_generation`
states the rule:

- a **newly inserted** fan message bumps it;
- a **human** creator reply (`was_ai_suggested=False`) bumps it;
- an automated bubble (`was_ai_suggested=True`) does **not** — otherwise bubble 1
  would invalidate bubble 2;
- a message that was **not inserted** does not — REL-002's `inserted` flag is
  what makes a duplicate webhook delivery harmless here, as it already was for
  the pipeline.

Every planned outbound sequence records the generation it was produced from.
Every externally visible send revalidates `sequence_generation == current`.

## 3. Outbound sequences

Kimi returns bubbles. The orchestrator then:

```
compute DeliverySchedule            services/human_delivery.py (unchanged maths)
  -> persist outbound_sequences     bound to the conversation generation
  -> persist outbound_sequence_parts  one row per bubble, each with due_at
  -> schedule one DELIVER_OUTBOUND_PART action per bubble at its due time
  -> the worker returns its slot
```

`(fan_id, trigger_identity)` is unique, so a retried action adopts the plan it
already made instead of queueing the reply twice.

The **availability** delay is deliberately not part of this: it is already spent
as the durable `AUTO_REPLY` action's own `execute_at`
(`services.suggestions.schedule_auto_reply`). Charging it again would double it.

### The send gate

`services.outbound_delivery.evaluate_send_gate` is the only place interruption is
decided, and it reads the database:

1. the sequence is still active, and this part is still `PENDING`
   (idempotent under a reclaim after a crash);
2. no earlier bubble is still unsent (bounded wait, then abandon rather than
   deliver out of order);
3. `current_generation(fan) == sequence.conversation_generation`;
4. the fan is not frozen for human review;
5. auto mode is not off.

A refusal marks the sequence `SUPERSEDED`, retires its remaining parts, and
cancels their queued actions. Already-sent bubbles stay conversation canon.

### Commercial settlement

`PRESENT_OFFER` and `CHECK_PAYMENT_CLAIM` used to settle immediately after an
inline send. With timed delivery a reply may never leave, so settlement moved to
the **first delivered bubble** (`services/outbound_settlement.py`). Committing at
plan time would leave fan state claiming a pending offer nobody was shown — which
the next turn would treat as deliverable against, and the abandoned-offer chase
would then pursue.

A locked PPV is one atomic message with its own plan/receipt invariants and is
still delivered inline. See §9.

## 4. Timing precision without per-fan polling

Inter-bubble pauses are 1–14 s. A flat 5 s idle poll turns a planned 1.5 s gap
into up to 6.5 s.

The fix is global, not per fan: after a cycle that claimed nothing, the
dispatcher asks the queue *when the next thing is due*
(`db.commercial_queries.next_due_at`, one indexed read) and sleeps until then,
bounded above by `SCHEDULED_ACTION_POLL_SECONDS` and below by a small floor.

One query per idle cycle, regardless of how many fans, how many conversations,
or how many pending bubbles. There is no timer, coroutine, or poll per fan
anywhere in the design.

## 5. Per-fan execution leases

`group_actions_by_fan` is still the cheap path and is still correct inside one
process. `fan_execution_leases` is what makes the same statement true across two.

- `acquire_fan_execution_lease(fan, creator, owner, ttl, purpose)` — insert, or
  take over only an expired lease, or renew your own. One statement.
- Taken only for `FAN_EXCLUSIVE_ACTIONS` (anything that may put words in front
  of one fan), only while work that is actually **due** is executing, and
  released in a `finally`.
- Never held across a future scheduled action: a nine-second pause releases it
  and the next bubble re-acquires when it is due.
- TTL 300 s — comfortably above a bounded turn, comfortably below the
  ten-minute stale-action reclaim, so a crashed worker frees the fan well before
  its action becomes re-claimable.
- A fan owned elsewhere is **rescheduled** a few seconds out, never skipped and
  never failed.

## 6. Scheduled conversational intentions

Cleopatra could keep specific promises (payday, post-session, abandoned offer,
abandoned PPV, inactivity). It could not keep the one the conversation itself
made: *"wait right there 😏"*.

GLM may now emit an optional `scheduled_intent`:

```json
{"kind": "short_continuation",
 "goal": "come back with the next beat if he has not answered",
 "timing": {"relative_minutes": 2},
 "source_ids": ["message-9"],
 "activity_policy": "cancel_on_activity"}
```

What makes it safe:

- **No copy is frozen.** The payload is a semantic goal on a fixed allowlist of
  keys. When it comes due it re-enters the ordinary path — current conversation,
  GLM revalidates, Kimi writes now — and may well decide to say nothing.
- **Application code owns the clock.** A relative delay is clamped to
  `[30 s, 24 h]` and jittered; a named reference (`payday`,
  `pending_offer_expiry`) resolves against evidence the application already
  parsed. A model-stated clock time is refused outright, as is a goal naming a
  price or an unknown kind.
- **One live obligation per kind per fan**, via the dedupe key
  `conv-intent:<fan>:<kind>`, so a newer beat replaces an older one.

`activity_policy`:

| Policy | Meaning | When due |
|---|---|---|
| `cancel_on_activity` | only worth keeping if he has not spoken | dropped when the generation moved |
| `revalidate_on_activity` | a real obligation his chatter does not delete | runs; GLM decides, possibly to stay quiet |

The existing specialised follow-ups are untouched and keep their own handlers.

## 7. Grounding

`EvidenceSnapshot` now states, deterministically, what the evidence supports:

- `publication_evidence` — whether any authoritative post/feed evidence exists.
  There is no feed integration today, so it reports `integration: "none"`, and
  the prompt and a narrow deterministic check both treat that as **nothing was
  posted**. Approved vault inventory is permission to offer privately; it has
  never been evidence of a post, of current clothing, or of anything findable by
  refreshing a page.
- `fan_publication_references` — posts the FAN raised. Those may be discussed as
  his context, which is why `unsupported_publication_claim` fires on the claim
  ("the bedroom set I just posted", "check my feed") and not on the noun ("that
  bikini post was a good day").
- `commercial_opportunity` — an explicit, fan-authored buying signal with the
  message ids that show it. A record of what he SAID, never an inference about
  what he can afford.
- `purchase_claim` — whether he said he paid and whether anything authoritative
  agrees.
- `voice_rhythm` — which emoji the recent creator bubbles leaned on. Soft; it
  never blocks a send.

## 8. Telemetry

`[OUTBOUND SEQUENCE]` lines are structured JSON carrying identifiers and reasons
only — never message text:

| Event | Answers |
|---|---|
| `planned` | which generation produced this sequence, how many bubbles, the timing plan |
| `sent` | sequence id, part number, planned delay, how late it actually was |
| `superseded` | why it was cancelled (`stale_generation`, `human_review_hold`, `auto_mode_off`, `newer_authorized_turn`, `fan_replied`) and how many bubbles were dropped |
| `settled` | which commercial settlement the first bubble ran, and its result |

The reply-provenance record carries `conversation_generation`, so "why was that
bubble cancelled?" starts from the message row.

The existing `[ACTION TIMING]` record still separates queue wait from deliberate
human delay; a bubble's deliberate wait now shows as queue time on its own
action, with `planned_delay_seconds` and `late_by_seconds` on the sequence line.

## 9. Deliberately unchanged

- **Locked PPV delivery** stays inline and atomic. It is one message whose plan,
  price and receipt must commit together (`_commit_locked_plan`,
  `send_locked_ppv`), and splitting it across durable parts would buy a
  composition pause at the cost of that atomicity. It gains no human-like
  composition delay in this sprint; it also loses none, because Core v1 never
  had one.
- **`semantic_v1` / `semantic_v2`** keep the previous inline delivery path
  exactly.
- **The legacy (non-semantic) core** keeps `_pending_auto_replies` and
  `_sleep_while_current`; they are its debounce, not Core v1's correctness.
