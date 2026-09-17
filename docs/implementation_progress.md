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
| 1 — close execution bypasses | 2 | Not started |
| 2 — the general continuity packet | 3 | Not started |
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

## Sprint 1 — close execution bypasses — NEXT

> §6.2: *Land the reviewed support/identity fixes; add payment interleaving
> tests and verified entitlement/access recovery. Confirm dashboard visibility
> of review holds and access outcomes.*

The reviewed support/identity fixes are already on `main` (PR #48). What remains:

1. **Confirm the deployed SHA and flags** against the running deployment, using
   `GET /build`. Record the answer here. Several findings are unresolvable
   without it.
2. **Finding I — payment-state concurrency.**
   `services/ppv_delivery_ledger.py::transition_delivery` updates by reference
   with no expected prior status, while `abandon_delivery_if_active` in the same
   module uses a status predicate. Add the expected-status guard and race tests
   for late acknowledgements and purchase-versus-expiry interleavings: a delayed
   pending update must never reverse a confirmed purchase. Purchase recording
   also does read/modify/write on aggregate fan state and needs concurrent-event
   verification. This is a real source-level risk, not a reproduced incident —
   say so in the commit message.
3. **Verified content-access recovery.** PR #48 contains the complaint; it does
   not repair anything. A safe workflow needs entitlement inspection, URL
   refresh, and an outcome the customer is only told about after it is true.
   Never send a repair claim that has not been verified.
4. **Dashboard visibility** of review holds and access outcomes, in
   `cleopatra-dashboard`. An operator cannot clear a hold they cannot see.

---

## Sprint 2 — the general continuity packet

> §6.3: *Add open-thread and episode records with provenance, correction
> semantics, and creator/customer scoping. Share the builder across Assisted,
> Auto, and evaluation paths.*

Findings D and E. Existing memory is real — `services/fan_intelligence.py`
evidence validation and merge rules, `services/fan_history_memory.py`
compaction, saved creator facts, persistent scene state — and the review is
explicit that saying Cleopatra has no memory would be inaccurate. What is
missing is a ledger of *unfinished* things: unanswered questions, promised
actions, deferred topics, unresolved complaints, each persisting until
fulfilled, cancelled, superseded or expired.

Seams already in place: `ANALYZER_TRANSCRIPT_MESSAGES` and
`WRITER_TRANSCRIPT_MESSAGES` are the two fixed slices the budgeted builder
replaces, and the provenance `context` block already reports them, so the
before/after is measurable on real replies rather than asserted.

Required properties from review §4: every memory carries source, timestamp,
evidence type and correction behaviour; "the customer said payment succeeded"
and "the platform confirmed order X" are different kinds of fact; a correction
supersedes rather than coexisting; retrieval is scoped by creator AND customer.

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
