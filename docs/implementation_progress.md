# Implementation progress — autonomous long-conversation system

**Purpose of this file.** A session picking this work up should be able to read
this page and continue, without re-running the investigation that produced
`docs/autonomy_architecture_review.md`. It records what has been decided, what
has been built, what has been verified and how, and what the next sprint is.
Keep it current in the same commit as the work it describes.

Last updated: 2026-09-17.

---

## Source of the plan

The implementation brief for this programme is
[`docs/autonomy_architecture_review.md`](autonomy_architecture_review.md),
merged to `main` in
[PR #48](https://github.com/Jericho2k/cleopatra_ai/pull/48). Its §6 is the
sequence these sprints follow; §3 is the evidence; §4 is the architecture
decision; §5 is the evaluation design.

**The short failure excerpts are regression seeds, not the specification.** The
unit of success is the complete customer conversation across interruptions,
purchases, ordinary chat, problems and later visits (review §1 and the
Decision). A sprint that makes an excerpt pass without moving one of the
capabilities in §1 has not advanced this programme.

### Note for the next session on the handoff document

This session was pointed at a local file,
`Cleopatra_Claude_Code_Implementation_Handoff.md`, on the user's own machine.
That path does not exist in the remote container and the contents were not
pasted, so the sprint numbering below is derived from §6 of the in-repo review
rather than copied from that document. The mapping is stated explicitly under
each sprint. **If the handoff defines sprints differently, reconcile the two and
correct this file** — the work already landed is written against §6 and stands
on its own, but the numbering may need renaming.

---

## Repository state at the start of this work

| | |
|---|---|
| `Jericho2k/cleopatra_ai` `main` | `4c4f58f` — PR #48 merged 2026-09-17T08:31Z |
| `Jericho2k/cleopatra-dashboard` `main` | `1b8c0c1` — PR #31 |
| PR #48 | **Merged.** Everything it landed is on `main`; do not re-implement it. |

What PR #48 already did, and must not be proposed again: saved creator facts
loaded in Full Auto; the direct paid resend replaced by a persisted
`content_access_issue` review hold before commercial mutations; truthful
handoff classification in simulation and the scheduled worker; a failed
review-hold write propagating instead of pretending a handoff succeeded.

---

## Sprint status

| Sprint | Review §6 step | Status |
|---|---|---|
| 0 — ground truth for every visible reply | 1 | **Done** (this branch) |
| 1 — close execution bypasses | 2 | **Done** except confirming the live deployment |
| 2 — the general continuity packet | 3 | **Done** except semantic extraction |
| 3 — one decision owner, compared under replay | 4 | Not started |
| 4 — longitudinal evaluation harness | 5 | Not started |
| 5 — shadow real interactions | 5 (operational) | Out of scope for code alone |
| 6 — bounded supervised pilot | 6 (operational) | Out of scope for code alone |

---

## Sprint 0 — ground truth for every visible reply — DONE

> §6.1: *Link each visible reply to its triggering event, input context, state
> version, decision, actual successful model attempt, transformations, and
> delivery receipt. Confirm deployed SHA and active flags.*

Nothing else in §6 can be evaluated until this exists. Review §2 is the reason:
a transcript that cannot name the commit, the flags or the model that answered
is not evidence about any particular piece of code.

### What was built

**`core/build_info.py`** — the deployed commit and the flags actually in force.
`build_sha()` reads the platform's own variable (Railway, Render, GitHub
Actions, or an explicit `CLEOPATRA_BUILD_SHA`) and falls back to reading `.git`
directly, with no git binary. `OBSERVED_FLAGS` is an explicit allowlist of
behaviour-affecting variables; `_assert_no_secrets` runs at import so widening
it to anything named like a credential fails the suite rather than leaking into
a message row. `flags_digest()` is eight hex characters — small enough to store
on every reply, and equal digests mean two replies ran under one configuration.

**`ai/generation_trace.py` + the generator** — *finding H*. `generate_replies`
returned `list[str]`, so a success on a retry, on another upstream host, or on
the configured fallback was indistinguishable from a first-try success by the
requested model. It now takes an optional `trace=` sink (the same shape as the
existing `outcome_sink`) and fills it in on success and total failure alike.
The return type and every existing call site are unchanged.

**`services/reply_provenance.py`** — the per-turn recorder. Created at the top
of a turn, filled in as the turn decides, and emitted once per delivered bubble
into `messages.media_context` under `reply_provenance`. **No migration**: it
reuses the existing jsonb column, next to the `ai_stack` marker, for the reason
that marker gives. It stores fingerprints and counts, never message text. Parts
of one reply share a `turn_id`.

**`SuggestionProvenanceStore`** — Assisted generates in one request and sends in
another, with a human in between. `get_suggestions` stores the record and
returns an opaque `suggestion_token` on `SuggestionResponse`; the dashboard
returns it on `POST /reply`, which redeems it once, adds the operator's chosen
index and whether they edited the text, and closes the record with the platform
receipt. In-process, bounded, TTL 30 minutes; a miss means no provenance, never
a blocked send.

**Named transcript windows** — `ANALYZER_TRANSCRIPT_MESSAGES = 12`
(`ai/situation_analyzer.py`) and `WRITER_TRANSCRIPT_MESSAGES = 16`
(`ai/prompt_builder.py`) replace inline slices, so finding D's evidence
asymmetry is reported on every reply instead of being a constant in two modules.
Sprint 2 replaces both with a budgeted builder; these are the seam.

**`GET /build`** (authenticated) returns the full flag mapping — the lookup that
turns a digest on a row into the configuration that produced it. `GET /health`
(public) carries `build_sha` and `flags_digest` only, never the flag values, and
reports `build_sha` even on its degraded path.

**`message_ai_stack_metadata`** now reports the model that *answered* when a
trace is supplied, keeping `requested_model`/`requested_provider` alongside.
Without a trace it is byte-for-byte what it was.

### What a record contains

```
reply_provenance:
  turn_id, mode (auto|assisted|...), part, parts, started_at, recorded_at
  build:     { sha, env, flags_digest }
  trigger:   { kind, text_fingerprint, text_chars, sent_at, history_position }
  context:   { history_messages, analyzer_window, writer_window,
               stack_profile, writer_prompt_version, live_state_blocks[] }
  decision:  { source, action, reason, purchase_signal, crisis_signal, ... }
  writer:    { requested{...}, actual{...}, served_by_requested_model,
               attempts, elapsed_ms, outcome }
  transforms: [ inventory_repair | delivery_language_repair |
                ppv_tag_stripped | message_shape_applied |
                ppv_single_message_merge | operator_edit ]
  delivery:  { kind, platform_message_id, accepted_by_platform, reference,
               price_cents }
```

### Verification

`2016 passed, 130 skipped` — the whole backend suite, up from `1975 passed` on
`main`, i.e. 41 new tests and no existing test changed to accommodate the work.
Repository fatal Ruff subset, `production_preflight.py --env-only`,
`compileall` over the tree, and `git diff --check` all pass. Schema tests skip
without `TEST_DATABASE_URL`, exactly as on `main`.

```bash
python -m pytest -q tests/test_reply_provenance.py tests/test_full_auto_simulation.py
```

### What Sprint 0 does NOT establish

* Nothing reads a provenance record yet. It is evidence for the sprints that
  follow, not a control surface, and no dashboard surfaces it.
* The Assisted token is in-process. A restart, an eviction, or a second backend
  replica loses it, and that reply persists with no provenance. Making it
  durable means a database write per suggestion; that trade was deliberately not
  taken.
* Real deployments were not inspected. The code can now *report* the deployed
  SHA and flags; nobody has yet confirmed what the agency's deployment says.
  **That confirmation is the first thing Sprint 1 should do**, because several
  review findings turn on flag values the code default does not establish
  (§3E in particular).
* No model quality, latency or cost was measured. Provenance makes such a
  measurement possible; it is not one.

---

## Sprint 1 — close execution bypasses — DONE (one item carried forward)

> §6.2: *Land the reviewed support/identity fixes; add payment interleaving
> tests and verified entitlement/access recovery. Confirm dashboard visibility
> of review holds and access outcomes.*

The reviewed support/identity fixes are already on `main` (PR #48).

### Done — finding I, payment-state concurrency

**`transition_delivery` has an expected-prior-status predicate.** It updated
`ppv_deliveries` by reference alone while `abandon_delivery_if_active`, in the
same module, used a status predicate. Four writers reach it from different
clocks: `ppv_delivery.py` writes `delivered_pending` *after* a platform round
trip to verify the payment lock, `record_ppv_purchase` writes `purchased`,
`ppv_reconciliation.py` writes `abandoned` on expiry, `ppv_recovery.py` writes
`voided`. Unordered, a slow pending write undid a purchase that landed while it
was verifying.

Nothing may now move a row out of `purchased`. A purchase may still be recorded
from any other status, including after an expiry or a void: money arriving is a
fact, and refusing to record it hides a real payment. The function returns
whether the ledger says the target status, which is also true when it was
already there — a duplicate webhook and a retry whose response was lost mean the
same thing about the world, and neither may read as a conflict.

**The expiry sweep asks the ledger before writing commercial state.**
`_finalize_abandonment` now calls `delivery_is_paid` before touching
`not_sold_log` and `pending_ppv_check`, so a sweep running after a purchase has
nothing to abandon. For the narrow window where a purchase lands *during* that
write, the refused transition freezes the fan with `expiry_lost_to_purchase`
rather than leaving two stores quietly disagreeing.

**Purchase aggregates use compare-and-set.** `record_ppv_purchase` read
`total_spent`/`sales_log`/`not_sold_log` near the top and wrote them back after
awaiting the session, the ledger and the platform. Two events for one fan both
read $100 and both wrote $125; one $25 purchase disappeared with no exception
and no log line. `db/queries.apply_purchase_to_fan` guards the write on the
total it was computed from and re-merges against the row that actually exists,
declining when the other event already recorded the same purchase. Exhausting
the budget raises `PurchaseAggregateConflict` and freezes the fan: money that
could not be written down is an operator's problem, and silence is how it became
invisible.

A row version column would be cleaner and needs a migration; this deliberately
needs none. These are **source-level races proven by tests, not reproduced live
incidents**, and passing them does not mean every payment race in this codebase
is resolved.

### Done — verified content-access recovery (§3B)

PR #48 contained the complaint and said so: *this is containment, not autonomous
repair [...] a safe verified access-recovery workflow remains necessary.*
`services/content_access.py` is that workflow, in two halves.

**The evidence half is read-only.** `inspect_content_access` joins the delivery
ledger — the only authority on whether money arrived — to what the platform
currently shows for the message it arrived in: still listed, still carrying
media, media still resolvable. It sends nothing, clears nothing, and reports
"could not check" as its own answer rather than as "fine". A purchase older than
the newest page of the conversation is `unknown`, not `missing`: a full page may
simply not reach back that far, and that is not evidence of removal.

**The repair half is bounded by what was paid for.** `resend_paid_content`
re-sends exactly the `media_ids` the ledger row records, **unpriced**, and only
when that row says `purchased`. Three properties, each enforced rather than
assumed: a complaint is not proof of payment, a repair cannot become a second
charge, and a repair cannot quietly become a different offer. The message is
persisted as `content_access_repair` with `price_cents: 0`, so it can never be
read back as a second PPV against the same media. If the platform returns no
receipt, nothing is recorded as repaired and the hold stays — the review's rule
that a delivery claim is tied to the operation result, applied to the repair
itself.

The one customer-facing sentence is a constant, not a writer call. A model asked
to phrase an apology could promise a fix that has not happened.

**Plain "Resume AI" is now refused on an access hold.** That path put automation
back in front of an unanswered complaint, which is how the baseline came to treat
an access problem as an opening for another sale. "Not an access problem" is the
recorded way to say the analyzer misread it, and clears the hold just as fast.

Nothing here decides on its own: Full Auto still stops at the hold, and every
function is called by an operator.

### Done — dashboard visibility (cleopatra-dashboard)

`GET /fan/{fan_id}/content-access` backs a reason-specific panel in `FanPanel`.
`lib/contentAccess.ts` turns the evidence into the operator's sentence and,
critically, decides whether to offer the resend at all: offering a repair the
backend would refuse trains an operator to click through errors. Uncertainty is
shown as uncertainty.

### Carried forward

**Confirm the deployed SHA and flags** against the running deployment, using
`GET /build` from Sprint 0, and record the answer here. This needs someone with
access to the deployment; it is not something this branch can do. Several review
findings are unresolvable without it, §3E in particular — live fact extraction is
flag-gated and the code default is off, which says nothing about what production
runs.

---

## Sprint 2 — the general continuity packet — DONE (one item carried forward)

> §6.3: *Add open-thread and episode records with provenance, correction
> semantics, and creator/customer scoping. Share the builder across Assisted,
> Auto, and evaluation paths.*

Findings D and E. The existing memory is real and the review says so —
`services/fan_intelligence.py` evidence validation and merge rules,
`services/fan_history_memory.py` compaction, saved creator facts, persistent
scene state. What none of it held is the state of the interaction.

### What was built

**`db/conversation_continuity_v1.sql`** — two tables, for two different things.
`conversation_open_threads` is what is unfinished: a question nobody answered, a
promise nobody kept, a topic put off, a complaint nobody resolved, a correction.
Each persists until fulfilled, cancelled, superseded or expired — never merely
because the recent-message window rolled over it. `conversation_episodes` is
what a completed stretch was about and how it ended, with the source range, so a
summary can be read back to the messages it describes.

Constraints carry the rules rather than trusting callers: a resolved thread must
say how and when, only a superseded thread may point at a successor, and
uniqueness is scoped to (creator, fan) so one creator's conversation cannot
collide with another's. `conversation_episodes` has **no column that could hold
money** — the review's "never proof of payment", enforced by the schema, with a
test that fails if one is ever added.

**`services/conversation_continuity.py`** — recording is idempotent (a question
mentioned across four turns is one obligation), resolution is guarded on
`status = 'open'` so two workers cannot both claim to have closed it,
supersession is a link rather than a delete, and expiry runs on the read path.
`rank_threads` orders by what a good operator would deal with first — complaints,
then what he is waiting on us for, oldest within each group, because "he returns
to the earlier of two subjects" is a §1 failure precisely when the newest is
treated as the only context.

**`services/context_packet.py`** — the budgeted builder, and the answer to all
three halves of finding D:

* *Bubbles are not turns.* Consecutive messages from one speaker group into one
  turn, so a multipart reply no longer spends the window several times faster
  than a single one. The same exchange now costs the same however it was sent.
* *The analyzer saw less than the writer.* Both build from
  `STANDARD_BUDGET` now. The classifier that decides what the turn DOES can no
  longer decide on less evidence than the reply is written from.
* *Small talk evicted obligations.* Threads and episodes have their own
  allowance, taken before the transcript is measured. A conversation can push a
  question out of the recent window; it cannot push it out of the packet.

It is pure — no database, no clock, no I/O — which is what lets Sprint 4's
replay comparison build the identical packet the live path builds.

**Wired in.** Full Auto and Assisted both load threads and episodes and pass
them into both context objects, so finding A's divergence cannot recur here. A
`content_access_issue` hold now also records a `complaint` thread, and resolving
that hold closes it: clearing the freeze alone left every later reply written as
though the problem were still live. Every reply's provenance record carries the
packet fingerprint, including what was dropped — which is what makes "the model
never mentioned it" separable from "the model was never told".

`scripts/production_preflight.py` fails when the tables are missing. The
continuity layer swallows its own failures by design, so a missing table
otherwise produces no error anybody sees: every reply is simply written as
though the conversation were carrying nothing, which looks exactly like a model
that forgets.

### Verification

`2258 passed, 0 skipped` — the whole suite **including every PostgreSQL schema
test**, run against a real PostgreSQL 16 in this container rather than skipped.
Up from `2053 passed, 130 skipped` at the end of Sprint 1. The schema pipeline
applies the full migration order from scratch, and `production_preflight
--schema-only` reports `29 passed, 0 failed` against that database.

```bash
# What this session ran; CI provides the same thing via its postgres service.
TEST_DATABASE_URL=postgresql://... pytest -q
```

### Carried forward — semantic thread extraction

Threads are currently recorded from events the system knows for certain: an
access complaint, today. Noticing that *a question went unanswered* or that *a
preference was corrected* is interpretation, which review §4 assigns to semantic
reasoning rather than deterministic code — so it belongs in the analyzer.

It is deliberately not done here. Adding an output to the analyzer changes its
prompt on every turn, and the review is explicit that another authority must not
be added without measuring it: *"Test the effect by removing duplicate guidance
under replay; do not assume a shorter prompt is automatically better."* Sprint 3
builds the replay harness that can measure it; extraction lands there, with a
before/after rather than an assumption. The storage, the ranking, the budget and
the prompt wiring are all in place for it.

The same applies to tying PPV promises to threads (a pending delivery is a
promise; a purchase fulfils it; an expiry cancels it). That one is deterministic
and could land sooner; it was left out to keep this change reviewable.

---

## Sprint 3 — one decision owner, compared under replay

> §6.4: *Keep the executor fixed; compare the current controller stack with a
> single semantic decision owner. Change model routing separately.*

Finding F. The review is explicit that adding another planner on top would add
another authority, and that a shorter prompt must not be assumed better —
removal is tested under replay, not argued. Two candidates are named: one model
call returning a reply plus typed intent, versus a separate semantic planner
then a writer. The second costs more and adds a failure point; take it only if
complete-conversation evaluation shows a benefit.

Deterministic code keeps permissions, consent, money, inventory, entitlements,
idempotency and competing events. Neither candidate may authorize a charge or
declare a tool succeeded.

---

## Sprint 4 — longitudinal evaluation harness

> §6.5 and review §5.

`scripts/run_model_eval.py` compares replies; it does not run the orchestration,
state transitions, delivery or weeks of interaction. The simulator does, and
should become the basis of a separate non-explicit longitudinal evaluation.
Both fixed-prefix replay and adaptive trajectories are required. The trajectory
table in review §5 is a proposed test design, not a measured distribution.

Score complete trajectories; count critical execution failures separately, since
a high average prose score cannot cancel an unauthorized transaction. Keep the
unseen evaluation set separate from prompt tuning.

---

## Standing constraints

* **Do not re-propose what is already on `main`** — review §3G lists the parts
  of the supplied failures the current head already addresses.
* **Attribution before improvement.** The review refuses a tone instruction or a
  blanket model switch until every output is attributable. Sprint 0 is what
  makes that possible; do not spend it.
* **A record must never be able to stop a reply.** Provenance methods are total
  and never raise; keep any future evidence layer the same way.
* **No credential may reach a message row**, a log line or an API response.
* **Finite tests cannot prove an absolute zero production failure rate.** State
  what was verified and how; never report a reliability increment as a
  production-readiness certificate.
