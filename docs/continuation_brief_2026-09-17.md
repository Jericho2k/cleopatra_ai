# Continuation brief — the current specification for this work

**This file is the brief, verbatim.** It supersedes the sprint numbering in
`docs/implementation_progress.md`, which was derived from §6 of
`docs/autonomy_architecture_review.md` by a session that did not have this
document. Where the two disagree, this one is correct.

Received 2026-09-17. Grounded at backend `4a1683a` and dashboard `9c47543`,
which were `main` in both repositories when it was written.

**On the earlier handoff.** A previous session was pointed at
`Cleopatra_Claude_Code_Implementation_Handoff.md` on a local machine and could
not read it; `docs/implementation_progress.md` says so. That file has still not
been supplied and is still unread. This brief is a different document — it was
supplied in full and is reproduced below — and it does not stand in for the
handoff. Nothing in this repository should be read as a claim that the handoff
was ever available.

**Phases, not sprints.** This brief is organised A–E with an explicit gate per
phase. The older Sprint 0–4 labels describe work that landed under a different
plan; they are not evidence that any gate here has been met, and
`docs/implementation_progress.md` now says so where it used to imply otherwise.

---

## The review this brief answers

# Cleopatra — review findings and Claude Code continuation prompt
Reviewed 2026-09-17.

Backend main: 4a1683a64b378d2696a89df5251d8f71cce2b2db
Dashboard main: 9c475439783d26dad585b5ab768b2f3edcb0124f

## Assessment for Rustam

Implementation incomplete. Pilot readiness unproven. Autonomous human replacement unproven.

Backend PRs #48, #49 and #50 and dashboard PRs #32 and #33 are merged.
Useful work includes actual writer-attempt attribution, payment transition guards,
purchase aggregate compare-and-set, access evidence/resolution UI, continuity tables,
shared transcript grouping, an offline decision comparison, and a trajectory runner.

However, docs/implementation_progress.md explicitly says the detailed original
handoff was unavailable to Claude. Its sprint numbering follows the shorter review
instead. "Done" there does not mean the original acceptance gates were completed.
It also still says Sprint 4 is unmerged, although PR #50 has since merged.

### Findings checked against code

1. Latest backend main CI failed:
   https://github.com/Jericho2k/cleopatra_ai/actions/runs/35219091193
   Result: 1 failed, 2336 passed.
   tests/test_operational_health.py searches serialized JSON for the string "4321".
   This matched the harmless timestamp T12:08:25.432182+00:00. Fix the assertion,
   retaining the actual operational-data redaction test; do not remove the gate.
   Dashboard main CI is successful at the reviewed SHA.

2. Agency redaction misses the new provenance record.
   services/ai_stack_visibility.py::public_media_context redacts ai_stack but
   preserves reply_provenance.writer.actual.model/provider. Reproduced directly.
   main.py::_simulation_turn_response uses this helper for agency callers.
   This is a model-routing information disclosure, not evidence of credential
   leakage. Audit all response paths and direct DB/realtime reads too.

3. Access repair is vulnerable to duplicate sends on retry.
   services/content_access.py sends directly through the platform adapter before
   save_message, with no durable repair claim/idempotency state.
   Reproduction using existing FakeSupabase and platform fixtures: make save_message
   raise after a successful platform response; call resolve_content_access twice.
   Two provider sends occur and the review hold remains.
   This sends two free copies, not two charges. Concurrent repair clicks and
   accepted-but-response-lost outcomes need the same treatment.
   clear_fan_review is unconditional by fan id; a repair finishing after a new
   hold is established can clear the newer hold. This last race is a source-level
   finding requiring an interleaving regression test.

4. The access panel cannot choose the affected purchase.
   components/FanPanel.tsx displays several paid items but posts only {resolution};
   the backend defaults to the most recent paid delivery. An older item's
   complaint can therefore cause a different, newer purchased item to be resent.

5. Conversation continuity is only partly populated.
   The only live record_open_thread call identified outside the service itself is
   the content-access complaint in services/suggestions.py. No live producer for
   record_episode was found. Ordinary questions, promises, deferred topics and
   corrections do not acquire the requested durable lifecycle.
   The original 40-message history fetch remains a separate upstream constraint.

6. The context builder is not a complete bounded evidence packet.
   It groups recent turns and includes strings for threads/episodes, but not a
   unified operational-fact/evidence envelope.
   A single 10,000-character fan message renders to 10,005 characters despite a
   transcript_chars budget of 6,000. Thread/episode limits count records, not tokens.
   fingerprint() returns counts/budgets rather than a content/version hash.
   The analyzer still separately appends the latest input already present in
   the recent history and does not receive all writer operational evidence.
   These are gaps against the supplied handoff; do not equate equal transcript
   windows with equal complete model inputs.

7. Decision replacement remains offline scaffolding.
   services/decision_owners.py is explicitly not wired live. CurrentStackOwner
   projects precomputed commercial/analyzer fields; replay does not compare two
   complete new conversational cores through a shared writer/executor.
   The one-call reply-plus-intent candidate is absent.
   parse_semantic_decision('{"hold":"typo","operation":"typo"}') returns NONE/NONE
   with confidence 1.0, contrary to its fail-closed documentation. Reproduced.
   _REACTION_FISHING_LINES is still used for scheduled post-purchase text.

8. The trajectory CLI overstates several coverage labels.
   scripts/run_trajectory_eval.py does not pass advance_clock; day/week labels do
   not advance the runtime clock. It returns an immediate empty result for empty
   customer messages; no due worker is run, so the queued-follow-up scenario cannot
   detect the behavior it claims to exercise.
   The default runs reuse one fan with no state reset between scenarios.
   The ordinary-chat fixture has 7 input turns; the 30+ turn deferred-question
   fixture has 5. Paid-content fixtures contain claims of payment, not seeded
   authoritative purchase events. Adaptive customer support is an injection seam,
   not an implemented adaptive runner.
   Detectors also mislabel evidence: "you dislike outdoor photos" is flagged as a
   critical reassertion after correcting "outdoor"; an answer on the very turn that
   asks about "trip" is classified as an unanswered obligation. Both reproduced.
   Missing provenance and failure outcomes without raised exceptions need explicit
   classification instead of being able to disappear into silence counts.

9. Cross-repository operator work remains.
   The access panel and Assisted suggestion token are present. An owner trace
   inspector, evidence-backed thread/episode correction workflow and browser
   end-to-end suite were not found in the reviewed changes.
   Assisted provenance tokens remain in-process, so restarts/replicas can lose
   attribution. The progress doc acknowledges this limitation.

### Validation performed in this review

Read recent commits, PRs, current source and GitHub CI logs in both repositories.
Ran 172 existing targeted backend tests successfully:
test_content_access_recovery.py, test_context_packet.py, test_decision_replay.py,
test_trajectory_eval.py, test_reply_provenance.py, test_operational_health.py.
Ran separate small reproductions of the redaction, budget, parser, detector and
repair-retry issues above. All model/platform effects in these reproductions
were isolated doubles; no customer was messaged.

No live deployment, schema state, provider quality, real payment, browser session,
latency/cost under real load, or commercial parity was verified. A passing local
subset does not overturn the red main CI result or establish readiness.

---

## The brief

Continue the existing Cleopatra implementation in BOTH Jericho2k/cleopatra_ai and
Jericho2k/cleopatra-dashboard. This is a continuation and corrective implementation,
not another architecture investigation.

You own completing the working product against observable acceptance gates.
The target is extended, adaptive customer conversation across days, interruptions,
multiple topics, corrections and operational events. Short failure excerpts are
only regression seeds. Do not preserve abstractions solely because they exist.

This prompt is self-contained. Save it into the backend repository as the current
continuation brief and reconcile docs/implementation_progress.md against it.
Do not claim an inaccessible local handoff was read. Read repository AGENTS.md
instructions and preserve concurrent user work.

The review above is grounded at backend 4a1683a and dashboard 9c47543. Check newer
diffs first and avoid duplicating fixes that have since landed. PR #50 is merged;
the old progress document's branch status is stale. Existing sprint labels do not
prove that the original gates passed.

Keep the current production path available. Work in reviewable increments with a
progress ledger. Do not enable a new core on live accounts, send customer messages,
run purchases, or change live deployment settings as part of ordinary development.
Use ordinary, non-explicit conversation fixtures for memory, interpretation and
quality evaluation. This task does not request explicit sexual content generation
or manipulative emotional-dependence tactics.

## Phase A — repair the concrete correctness gaps first

A1. Fix the flaky health redaction test. Assert the response schema and prohibited
operational fields/typed values, not arbitrary digit substrings in serialized
timestamps or SHA/digest values. Retain a regression containing the exact timestamp
that triggered the failure. Run CI on the resulting commit, including Postgres
schema tests; do not declare it green while only a local subset has finished.

A2. Enforce owner-only provenance at actual data boundaries. Agency callers must
not receive provider/model routing, attempts or internal diagnostic records through
reply_provenance, simulation polling, message history, exports, websocket/realtime
or direct database queries. Preserve useful product outcomes and existing media.
If the dashboard can read the JSON directly through Supabase, fix storage/access
architecture as needed: hiding a panel or sanitizing one REST route is insufficient.
Owners still need an authorized trace inspector. Test both positive owner access
and negative agency/cross-tenant cases, including responses with provenance and
without ai_stack.

A3. Put free access repair behind a durable operation boundary. Bind each repair
to tenant, fan, exact purchased reference and the particular review case/version.
Claim before sending. Distinguish pending, confirmed, failed and unknown outcomes.
On platform acceptance followed by DB failure, response loss, worker restart or
duplicate operator requests, reconcile instead of blindly sending again. Do not
claim exactly-once external delivery unless the platform supports it.
Use confirmed entitlements and the original media only; no new charge.
Clear only the review case that was resolved using a conditional update; never
clear a newer crisis/support hold. Test overlapping operators and review changes
during the platform call against real PostgreSQL where locking/uniqueness matters.
Avoid holding DB locks across slow external calls.

A4. Give the operator an explicit purchase selection in the access panel. Send the
selected reference and a stable repair request/case identifier. Require a selection
when ambiguous rather than silently assuming newest. Persist an auditable resolution
with actor/evidence. Reloads and retries must show the operation's actual state.
When platform checking is inconclusive, preserve uncertainty in the UI.

Gate A: demonstrate the known reproductions fail before and pass after the fixes;
publish backend/dashboard test and CI evidence. Keep deployment activation separate.

## Phase B — make the evaluation harness capable of disproving readiness

Implement eventful scenarios around the actual runtime orchestration, not an
imitation. Add a clock injected through expiry, scheduling and continuity, actual
due-worker execution, authoritative purchase/delivery fixtures, fault injection,
and isolated scenario state. Persist state within a scenario across sessions; use
fresh or reproducibly restored state between independent candidates/scenarios.
Keep all platform delivery isolated and enforce test-tenant boundaries.

Ship real 40–80 turn ordinary conversations and questions resumed after 30+ turns,
not short fixtures merely labeled with those numbers. Include two deferred topics,
corrections versus stale summaries, day/week return, ambiguity, low engagement,
no-follow-up with a queued job, takeover during generation, provider failure,
duplicate events, purchase/expiry races and concurrent creators/customers.
Implement adaptive customer behavior with reproducible seeds and reviewable rules
or model configuration; never use a model as the sole quality judge.

Repair the detectors. Same-turn answers count. Keyword occurrence is not proof of
reasserting an obsolete fact. Correction/contradiction suspicions need review unless
supported by authoritative evidence. Silence preferences have scope/expiry and can
be changed by a later customer instruction. Distinguish legitimate silence, handoff,
failed analysis, writer failure, unknown delivery and infrastructure errors even
when no exception was raised. Missing trace/state evidence is unknown, never a pass.
Test paid deliveries with missing, malformed and partial provenance.

Report transcript, event timeline, missing coverage, failures, configuration,
actual model attempts, latency and cost coverage. Human rubric dimensions are useful;
"no self-grading model" must not become a prohibition on human-rated quality measures.
Never let a prose score cancel an execution failure.

Gate B: demonstrate that injected duplicate sends, ignored no-follow-up preferences,
wrong references and lost continuity are actually detected by the runner. A complete
mock run proves wiring, not live model quality.

## Phase C — finish durable ordinary conversation memory and context

Reuse existing continuity tables and fan-fact provenance. Add real producers for
questions, commitments, deferred topics, corrections and episodes, with source event
IDs, revision/supersession, certainty and tenant/fan scoping. Validate extracted
proposals deterministically. Monetary obligations resolve only from authoritative
events; conversational topics require appropriate source evidence.

Add a processing watermark and include the unprocessed event tail so delayed
extraction cannot erase short-term continuity. Handle retries, stale/out-of-order
extraction and backfill without overriding newer corrections.
Use semantic extraction where interpretation is required, not a larger keyword list.
Creator-generated inventions must not silently become creator-approved identity.

Complete the context packet with approved creator facts, relevant customer facts,
operational state, stable evidence references, timestamps, conversation/settings
versions, open threads and selectively retrieved episodes. Build consistent
projections for understanding and writing. Include newest input exactly once.
Fetch complete turns rather than grouping a window that already cuts them in half.

Bound the complete model input by tokens, with reservations for required evidence.
Handle a single oversized message and oversized memory. Preserve necessary meaning,
record omissions and clarify/hold when required evidence cannot fit; do not silently
drop a hard constraint. Use a real content/version fingerprint, not just counts.

Add practical operator views for saved facts and threads with source/correction/
resolution controls. Test both the stored state AND actual model-bound payloads.

Gate C: two deferred subjects and an unanswered question survive 30+ intervening
turns and a return session; a correction displaces obsolete evidence across Auto,
Assisted, simulation and replay. Cross-tenant and delayed-extraction tests pass.

## Phase D — complete and compare conversational cores

The current decision_owners module is an offline candidate, not a shipped
replacement. Implement strict typed parsing: missing required fields, invalid enums,
wrong types, non-finite confidence and malformed output must result in a visible
failed/insufficient-evidence outcome, never a fabricated confident decision.
Carry supporting evidence and context/conversation versions.

Build comparable runtime candidates:
1. ordinary reply plus typed intent in one call;
2. semantic decision followed by a writer.
Use one authoritative decision owner on each candidate path. Bypass conflicting
legacy behavioral directions there; retain deterministic commercial permissions,
inventory, entitlement and delivery constraints. Avoid constructing yet another
controller above the old stack.

Compare complete replies and executed/suppressed actions with the same evidence,
executor and controlled model configurations. Merely comparing projected labels is
insufficient. Separate model-routing experiments from architecture changes.

Replace universal random post-purchase/proactive text with context-aware eligible
actions or silence. Revalidate queued work against current message/case/settings
versions, operator takeover and communication preferences.

Gate D: produce baseline/candidate trajectory evidence and inspectable disagreements.
Do not select the more elaborate candidate by assumption. If real provider access
or spending authorization is missing, complete adapters, fixtures and dry runs, then
record the exact empirical gate still pending. Do not mark selection completed.

## Phase E — finish operator flows and prepare controlled evaluation

Add an owner trace inspector showing actual successful attempts and execution state;
keep sensitive diagnostics inaccessible to agencies. Make Assisted attribution
durable across replicas/restarts, or explicitly surface missing attribution and
record this as an unmet gate rather than pretending every reply is attributable.

Add browser-level tests for actual backend contracts with isolated external
adapters: desktop/mobile, reload during generation, reconnect after completion,
out-of-order events, correct purchase/media after repair, multiple purchases,
takeover, new hold during repair, and evidence-based resumption.
Preserve distinction between accepted, generated, delivered, paid and unknown.

Prepare account-scoped new-core controls, shadow mode with zero live sends AND
zero live commercial mutations, takeover and rollback. Record migration order,
deployment compatibility, rollback limits and required schema checks.

Deliver a reproducible evaluation bundle and separate:
- implementation complete;
- pilot ready;
- human replacement/commercial parity established.

All unexecuted empirical gates remain pending. The owner and agency set release
thresholds against measured baselines; do not invent a parity claim from mock tests.

## What needs Rustam, and what does not

Continue all independent development without repeated permission questions.
If an external prerequisite is absent, finish the code and tests it does not block.

Ask Rustam only for concrete missing inputs:
- access to the actual deployed backend/dashboard and authenticated build/schema
  verification, using existing configured credentials without pasting secrets;
- a controlled test creator/catalog and provider budget for real model evaluations;
- representative permitted, anonymized agency conversations and qualified reviewers;
- agreed quality, latency, cost, rescue-rate and commercial pilot criteria;
- explicit authorization/account selection before live shadow/pilot activation.

Do not require Rustam to design missing extraction, implement the harness, invent
the UI contract or manually decide routine engineering choices.

At each checkpoint report changed behavior, commits/PRs, exact commands/results,
reproduction outcomes, pending gates and next action. Update the progress document
in the same change. Open reviewable PRs and watch their final CI result. Do not
merge or deploy without authorization. Continue until the independently actionable
development is complete, or identify a concrete external blocker.
