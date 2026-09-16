# Cleopatra: autonomy and long-conversation architecture review

Reviewed backend baseline: `77d487ffda102ef839a4ece730c475a09583cb81`.

This is an engineering assessment of general conversational continuity, truthful
execution, customer support, and production reliability. It does not design
explicit sexual generation or emotional-dependence tactics. Private supplied
transcripts and training material are not copied into this public repository.

## Decision

The supplied short conversations are regression examples, not the target product
specification. Passing them would not demonstrate autonomous replacement of a
human operator. The unit of success must be the complete customer conversation
across interruptions, purchases, ordinary chat, problems, and later visits.

**Production readiness is not established.** This review found reproducible
backend defects, plus architectural limitations that warrant a controlled
replacement of conversational decision-making and context assembly. It did not
measure live conversation quality, revenue, actual deployment configuration, or
the agency's human baseline. No claim of human parity follows from the tests.

Do not start with another tone instruction or a blanket model switch. First
make every observed output attributable to the actual code, context, decision,
model attempt, postprocessing, and delivery event that produced it.

## 1. Independent standard for an excellent operator

The following are product requirements derived independently of the code. They
are an operational standard, not an empirical claim that every human chatter
meets them. The supplied manual illustrates adaptation and conversational
continuity, but its scripts, fixed message counts, and manipulation claims are
not a validated benchmark.

| Capability | Observable behavior | What a convincing test requires |
|---|---|---|
| Understand the whole message | Respond to the actual request, correction, question, or complaint; recognize multiple intents in one message | Mixed-intent turns with competing priorities |
| Sustain ordinary conversation | Discuss a topic without forcing a question, rehearsed joke, or purchase opportunity into every reply | Extended conversations with no purchase goal |
| Maintain shared context | Remember what was discussed, promised, deferred, corrected, and already resolved | Interrupted threads resumed after the recent-message window has rolled over |
| Adapt without losing identity | Adjust warmth, directness, length, and initiative while preserving the creator's approved facts and boundaries | Several customers with different styles, then a return to the same customer |
| Notice changing needs | Treat a new request as a possible change of direction rather than a mandatory next stage | Topic changes, explicit preference changes, hesitation, and withdrawal |
| Follow through | Carry out an accepted, authorized operation once; distinguish intent, acceptance, delivery, and payment | Out-of-order confirmations, retries, failures, and resumed sessions |
| Repair mistakes | Acknowledge a misunderstanding, correct the record, and address access problems before new transactions | Complaints after successful payment and after failed delivery |
| Use judgment about silence | Respect a goodbye, an unanswered message, or a request for no follow-up | Delayed jobs that become obsolete before execution |
| Know what is uncertain | Avoid inventing personal facts, content details, tool success, or payment status | Missing evidence and conflicting records |
| Hand over usefully | Identify the unresolved issue and preserve the relevant records for an operator | Review holds, takeover during generation, and resumption after resolution |

Long-term quality is not simply recall of names and favorite things. It includes
the state of the interaction: why a topic matters, which question remains
unanswered, whether a misunderstanding was repaired, and whether a prior
invitation to continue is still current.

## 2. What the short examples establish, and what they do not

The examples show repeated confirmation requests, generic responses, delivery
claims that do not match visible outcomes, repetition after purchases, and
access complaints being treated as opportunities to send more content.

They do **not** identify the generating model, deployed commit, enabled flags,
exact media metadata, true payment state, or the cause of a blurred preview.
Repeated input in an exported transcript is not sufficient to prove an ingestion
bug. A simulation purchase is not a verified live purchase. The tree emoji could
reflect metadata, a prompt, or generation; attributing it to a particular model
without the trace would be speculation.

They also reveal nothing reliable about performance after 100 turns, a week of
absence, a changed preference, or simultaneous conversations on multiple creator
accounts. Those are separate tests, not extrapolations from a successful demo.

## 3. Findings in the inspected code

### A. Full Auto omitted saved creator facts — reproduced and fixed here

`services/suggestions.py::get_suggestions` loads `get_creator_legend`, but the
baseline `_debounced_auto_reply` did not load or pass it. The writer's
`ai/prompt_builder.py` reads `ctx.creator_legend` for canonical identity and
self-facts. Full Auto therefore could not benefit from those saved facts through
that input, even though Assisted mode could.

This branch loads the same saved creator facts in Full Auto and passes them to
both context objects. A regression test failed against the baseline and now
checks two turns with a corrected saved fact. This proves context plumbing;
model compliance and creator-fact accuracy still need evaluation. In particular,
the analyzer's current prompt does not render those facts merely because its
context object contains them.

### B. Access complaints bypassed authoritative delivery — reproduced and fixed here

The baseline resend branch in `_debounced_auto_reply` called
`send_apifansly_message` directly with the existing price. It did not go through
`send_locked_ppv`, and it ran after commercial state updates and planning. If no
pending payment existed, it fell through to normal generation. That is a poor
fit for a customer who already paid and cannot access the item.

This branch deletes the direct paid resend. An analyzer-reported access/resend
issue now establishes `needs_human_review` with reason `content_access_issue`
before commercial mutations, planning, or writing. It preserves the existing
payment snapshot and reports `human_review` in simulation. A failed hold write
propagates as an error instead of pretending a handoff succeeded. Crisis handling
retains priority. The scheduled worker recognizes persisted review holds as a
handoff rather than a writer-quality failure.

This is containment, not autonomous repair. It depends on the existing analyzer
recognizing the problem. It does not resolve media entitlements, refresh expired
URLs, inspect the dashboard renderer, or notify an operator outside the existing
review workflow. It sends no customer-facing repair claim. A safe verified
access-recovery workflow remains necessary.

### C. Some visible replies bypass the writer entirely — confirmed

`services/suggestions.py::_REACTION_FISHING_LINES` contains generic reaction
prompts matching several supplied outputs. `record_ppv_purchase` randomly picks
one and schedules `POST_PURCHASE_REACTION` with the text already populated.

The worker has staleness checks, so this is not a claim that every scheduled line
is blindly sent. But model replacement cannot improve text selected from a
literal list. Inspect all outbound paths in a quality evaluation, including
scheduled messages, operator actions, fallbacks, and postprocessing.

### D. Understanding and writing see different, short text windows — confirmed

`db/queries.py::get_conversation_history` defaults to the latest 40 messages.
`ai/situation_analyzer.py::build_analyzer_prompt` renders only the last 12;
`ai/prompt_builder.py::build_prompt` renders the last 16. These are message
bubbles, not complete conversational turns. Multipart replies consume the
window faster. Both transcript renderers use speaker and text rather than a
full event history containing authoritative payment and delivery receipts.

The analyzer prompt uses that recent text plus the newest message; it does not
receive the writer's structured live-state blocks. This creates an evidence
asymmetry: a decision-driving classifier can lack context that exists elsewhere
in the system. Increasing the writer's model size does not restore evidence
missing upstream.

### E. Durable memory exists, but facts are not the whole conversation — confirmed structure; quality impact unmeasured

There is useful existing work: fan-fact evidence validation and merge rules in
`services/fan_intelligence.py`, historical compaction in
`services/fan_history_memory.py`, saved creator facts, and persistent scene state.
It would be inaccurate to say Cleopatra has no memory.

The inspected fact extractor uses an enumerated CRM vocabulary; the scene stores
a particular interaction's progression. Neither is a general ledger of multiple
unfinished topics, questions, commitments, corrections, and their resolution
conditions. Historical compaction into the same fact schema does not by itself
preserve that missing information. Live fact extraction is also flag-gated;
the code default is off, which does not establish the live deployment value.

### F. Several systems direct the same reply — confirmed structure; causal contribution requires ablation

The current path combines a stage classifier, situation analysis, commercial
policy, conversation director, session strategy, experience director, expression
guidance, and writer instructions. Some are optional and have different defaults.
The prompt builder conditionally assembles their outputs into live-state blocks.

The problem is not the number of files. It is that several representations can
prescribe what a single conversation should do next. Another planner added on
top would introduce another authority unless ownership is explicitly simplified.
Test the effect by removing duplicate guidance under replay; do not assume a
shorter prompt is automatically better.

### G. The current head already addresses parts of the supplied failures — confirmed

`services/ppv_turn.py` separates attachment selection from writer text.
`tests/test_ppv_delivery_boundary.py` checks that ordinary text can accompany a
planned attachment and that failure cannot leave a false delivery claim.
The current commercial path has moved toward a single offer rather than the
older quick/full menu, and newer writer profiles change the output contract.

These should not be proposed again as if absent. Their existence also does not
prove the agency deployment uses them or that the complete conversation works.
This review anchors to the inspected SHA, not an assumed Railway deployment.

### H. Message-level model attribution is incomplete — confirmed

`message_ai_stack_metadata` stores `route.primary_target` as the message's model
and provider. `generate_replies` may succeed through an alternate provider or a
fallback. Recovery telemetry exists, but the message marker alone cannot prove
which attempt supplied the final text. A quality comparison needs the successful
attempt joined to the visible message, not merely the configured primary.

### I. Payment-state concurrency still needs a focused audit — source-level risk, not a reproduced live incident

`services/ppv_delivery_ledger.py::transition_delivery` updates by reference
without an expected prior status. In contrast, `abandon_delivery_if_active` uses
a status predicate. The former deserves race tests for late acknowledgments and
purchase-versus-expiry interleavings; a delayed pending update must not reverse a
confirmed purchase. Purchase recording also updates aggregate fan state through
read/modify/write operations that need concurrent-event verification.

No ledger redesign is included in this branch. Do not interpret the targeted
tests as proof that all payment races are resolved.

## 4. Architecture decision for the general conversation system

Use one explicit owner of conversational decisions, supplied with one
evidence-backed context, behind a deterministic execution boundary. Transaction
state machines remain useful for delivery and payment. A rigid phase machine
should not be the specification for human conversation.

### Context with distinct evidence types

| Layer | Contents | Authority and lifetime |
|---|---|---|
| Event history | Messages, edits, operator actions, tool results, payment events | Append-only evidence with stable IDs and timestamps |
| Creator facts | Approved identity, preferences, boundaries, availability | Creator-scoped, versioned; generated statements do not automatically become truth |
| Customer facts | Explicit preferences and corrections, with source messages | Customer-and-creator scoped; uncertainty and supersession retained |
| Conversation episodes | What a completed interaction was about and how it ended | Compact summaries with source ranges; never proof of payment |
| Open threads | Unanswered questions, promised actions, deferred topics, unresolved complaints | Persist until fulfilled, cancelled, superseded, or expired |
| Operational state | Current authorized action, exact item, amount, entitlement, delivery status | Database/tool authority; cannot be inferred from conversational wording |

Build a context packet from recent complete turns, relevant earlier episodes,
open threads, approved identity, and current operational facts. Reserve context
space for unresolved obligations; do not drop them merely because small talk
filled a 16-bubble window. Retrieval must be scoped by creator and customer.

Each memory needs a source, timestamp, confidence/evidence type, and correction
behavior. Distinguish "the customer said payment succeeded" from "the platform
confirmed order X." A later correction should supersede an old preference,
not create two simultaneously authoritative facts. Temporary mood and durable
preference need different lifetimes.

### One conversational decision

The decision object should identify the customer's active needs, the messages
supporting that interpretation, unresolved references, which questions the reply
must address, any proposed operation, and why waiting or handing off is needed.
It should not prescribe a mandatory emotional ladder or a fixed sentence shape.

Use deterministic code for permissions, consent and communication preferences,
money, inventory, entitlements, idempotency, and competing events. Use semantic
reasoning for interpretation, ambiguity, relevant recall, and response selection.
A request to fix access outranks a new commercial suggestion.

Two viable candidates should be compared: one model call returning an ordinary
reply and typed intent, versus a separate semantic planner followed by a writer.
The second costs more and adds another failure point. Select it only if complete
conversation evaluation establishes a benefit. Neither model can authorize a
charge or declare a tool succeeded.

### Transaction execution and result truth

Before an external operation, check the current conversation version, account
settings, operator takeover, accepted action, and exact operational state. Stale
work must be discarded. Persist a durable intent/idempotency key, perform the
operation through one delivery boundary, reconcile ambiguous outcomes, and then
record the authoritative receipt.

Delivery claims must be tied to the operation result or coupled to the attachment
itself. Unknown delivery outcomes must not trigger blind retries. Payment
confirmation, not a model's interpretation of "yes" or "I paid," advances paid
state. All outgoing paths must obey the same boundary.

### Verification without an endless rewrite loop

Always run deterministic checks for unauthorized operations, stale state,
invented identifiers/prices, duplicate sends, and unsupported success claims.
Evaluate an optional semantic checker for ordinary factual contradiction,
unanswered questions, unresolved references, and repetition. Bound its retries.
Do not install a generic "make this more human" critic that repeatedly rewrites
valid replies without a measurable acceptance criterion.

### What to keep, replace, and retire

| Existing area | Decision | Reason |
|---|---|---|
| Platform adapters, tenancy, simulation isolation | Keep and independently verify | Required production infrastructure |
| Durable queue and delivery receipts | Keep; strengthen concurrency tests and common boundaries | Useful execution foundations |
| Provenance validation and fan-fact merge logic | Reuse | Evidence discipline is valuable |
| Saved creator facts | Keep, with common Auto/Assisted context assembly | Identity must survive modes and sessions |
| Bounded writer recovery | Keep initially, fix actual-attempt attribution | Provider availability and voice quality are separate concerns |
| Multiple conversational progression controllers | Replace behind one decision interface after replay comparison | Reduce competing sources of behavioral instructions |
| Fixed short transcript slicing | Replace with a budgeted evidence/context builder | Long conversations need selective continuity |
| Random hardcoded proactive text | Retire as a universal response strategy | Ignores the specific interaction; evaluate any replacement in context |
| Informal read/modify/write commercial snapshots | Audit and converge on transactional authority | Concurrent events must not disagree about reality |
| Additional planners/critics by default | Do not add automatically | More calls are not evidence of better outcomes |

These are proposed decisions, not changes implemented by this patch.

## 5. Evaluation that can actually reject a weak architecture

The existing `scripts/run_model_eval.py` constructs scenario contexts and invokes
model completion. It is useful for reply comparisons, but it does not run the
complete autonomous orchestration, real state transitions, delivery, or weeks of
interaction. The simulator exercises more of the actual path and should become
the basis of a separate, non-explicit longitudinal evaluation.

Use both fixed-prefix replay and adaptive trajectories. Replay gives candidates
the same evidence; adaptive trajectories expose the consequences of earlier
choices. A scripted cooperative customer and a model grading its own text are
insufficient substitutes for expert human review.

Suggested initial coverage below is a proposed test design, not a measured
distribution or a validated production threshold:

| Trajectory | Required disturbance | Failure to catch |
|---|---|---|
| 40–80 turns of ordinary conversation | Several topic changes; no purchase request | Forced commercial pivots, repetitive questioning |
| Return after one day and one week | Resume a previously unfinished topic | Restarting introductions or inventing what happened |
| A question deferred across 30+ turns | Two unrelated topics intervene | Forgotten obligation or wrong referent |
| Preference correction | Old preference appears in retrieved history | Reasserting superseded information |
| Multiple open threads | Customer returns to the earlier of two subjects | Treating the newest subject as the only context |
| Creator fact consistency | Same question across customers and provider failover | Identity drift or contamination between customers |
| Content-access support | Complaint after payment, then a mixed request | Another charge instead of repair |
| Explicit decline or goodbye | Previously queued follow-up becomes due | Ignoring the customer's current choice |
| Operator takeover | Human acts during model generation | Stale automated message after takeover |
| Failure recovery | Timeout after remote acceptance; duplicate webhook | Duplicate delivery or false success |
| Event ordering | Purchase, pending acknowledgment, and expiry reordered | Paid state reverting or a second offer blocking access |
| Cross-account workload | Same customer identifiers on different creators | Leakage of facts or operational state |

Vary spelling, tone, language, interruptions, message batching, ambiguous
pronouns, and quiet periods. Reuse the original failure excerpts as regression
seeds, while keeping the unseen evaluation set separate from prompt tuning.

Score complete trajectories on responsiveness, factual continuity, useful
initiative, natural variation, respect for changed preferences, recovery, and
the proportion requiring operator rescue. Count critical execution failures
separately; a high average prose score cannot cancel an unauthorized transaction.
Record latency, cost per completed conversation, and tail behavior during errors.

Commercial parity requires a prospective, controlled comparison against the
agency's human baseline, with equivalent traffic and account conditions. Measure
net outcomes over a meaningful return window, including refunds, complaints,
retention, operator time, and inference cost; immediate simulated purchases do
not establish commercial value. Sample size and acceptable uncertainty must be
set from baseline variability and business tolerance, not invented here.

## 6. Concrete implementation sequence and release gates

1. **Capture ground truth.** Link each visible reply to its triggering event,
   input context, state version, decision, actual successful model attempt,
   transformations, and delivery receipt. Confirm deployed SHA and active flags.
2. **Close execution bypasses.** Land the reviewed support/identity fixes; add
   payment interleaving tests and verified entitlement/access recovery. Confirm
   dashboard visibility of review holds and access outcomes.
3. **Build the general continuity packet.** Add open-thread and episode records
   with provenance, correction semantics, and creator/customer scoping. Share
   the builder across Assisted, Auto, and evaluation paths.
4. **Compare replacement conversational cores offline.** Keep the executor fixed;
   compare the current controller stack with a single semantic decision owner.
   Change model routing separately so results remain attributable.
5. **Shadow representative real interactions.** Generate no live autonomous
   actions. Have operators review complete trajectories and disagreements.
6. **Run a bounded supervised pilot.** Expand autonomy only after agreed quality,
   reliability, and economic criteria hold. Keep takeover and rollback operable.

Required before calling this a human replacement: longitudinal human-reviewed
quality evidence; no critical unauthorized/duplicate operation in the defined
fault matrix; reliable recovery and takeover; acceptable tail latency and cost;
and a measured comparison with the human baseline. Finite tests cannot prove an
absolute zero production failure rate.

## 7. Scope of this branch and remaining uncertainty

Local validation: **176 targeted tests passed** across Full Auto simulation,
delivery boundaries, analyzer degradation, concurrency invariants, worker failure
classification, durable simulator turns/endpoints, creator-fact prompting, V3
Auto contracts, local delivery, durable replies, and payment-pending workflows.
The repository's fatal Ruff subset passed, as did compilation of changed Python
modules and `git diff --check`. The tests stub model/platform behavior; they do
not measure real model quality. Full CI and PostgreSQL-backed schema tests were
not run locally.

Reproduce the targeted test run:

```bash
python -m pytest -q \
  tests/test_full_auto_simulation.py \
  tests/test_ppv_delivery_boundary.py \
  tests/test_analyzer_degraded.py \
  tests/test_concurrency_invariants.py \
  tests/test_action_failure_classification.py \
  tests/test_simulation_turn_durability.py \
  tests/test_simulation_turn_endpoints.py \
  tests/test_prompt_builder_legend.py \
  tests/test_v3_full_auto_contract.py \
  tests/test_local_test_delivery.py \
  tests/test_durable_auto_reply.py \
  tests/test_payment_pending_human_delivery.py
```

Implemented: support-before-commercial handoff, removal of the direct paid
resend, truthful simulation/worker handoff classification, and loading saved
creator facts in Full Auto. Regression tests include live versus simulated
routing, pending versus already-resolved payments, both commercial flag states,
boolean/string analyzer signals, hold-write failure, crisis priority, and saved
fact updates across turns.

Not implemented: the replacement conversational core, episode/open-thread
memory, actual-model attribution repair, verified access repair, ledger race
remediation, or the longitudinal evaluation harness. No live model comparison,
real payment, production deployment, revenue experiment, or frontend visual
verification was performed. These omissions matter: this branch is a reviewed
reliability increment and architecture decision record, not a completed rewrite
or production-readiness certificate.
