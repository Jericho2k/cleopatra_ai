# Conversational Core v2 — Session-aware

`conversational_v2` is an **alternative** to `conversational_v1`, selectable per
test fan (and per creator) through the existing Conversation Core mechanism. It
is not a layer on v1: it has its own durable state, its own owner contract, and
v1's code paths are byte-identical to `main` (a 4-turn offer → send → purchase
scenario was diffed prompt-for-prompt against `main` during development).

## The problem it addresses

v1 decides the next turn and the next commercial operation well, but it has no
first-class representation of a longer interaction, so paid behaviour collapses
into `conversation → next unlock → reaction → next unlock`. v2 adds durable,
validated state for:

1. **Trajectory continuity** — what broader interaction is being carried out,
   what has happened, what is provisionally intended next.
2. **Immersion continuity** — the active premise, participation, tempo and
   callbacks that must survive between content events.

## Shape of a turn (one semantic brain)

```
evidence + persisted session state
   → reconcile content facts from the ledger              (deterministic)
   → GLM: next_experience_move + decision + session_delta (semantic owner)
   → validate every delta field; v1 validator on working state
   → v2 guards → shared v1 operation validator/executor   (deterministic)
   → Kimi writes, given the validated session view        (sole writer)
   → persist with revision CAS, execute through the v1 delivery path
```

No new model, planner or director. GLM remains the only semantic owner, Kimi the
only writer, and application code the only authority for inventory, prices,
purchases, deliveries, permissions and persistence.

## State (`models/conversational_session.py`, table `conversational_session_states`)

`ConversationalSessionState` (`schema_version = conversational_session_v2`):

| Field | Meaning | Written by |
|---|---|---|
| `working` | v1's `ConversationalWorkingState` (scene, flow), with v1's validator | owner, validated |
| `session.status` | `inactive → proposed / planning → active ⇄ paused → completed / abandoned` (transition table) | owner, validated |
| `experience_premise` | shared situation, `world_scope` `conversation` or `imagined_scene` | owner, validated |
| `interaction_goal`, `fan_participation`, `tempo`, `content_direction` | what this stretch is doing and how | owner, validated |
| `known_constraints` | evidence-only (fan statement / ledger / creator config); a spending limit needs the amount **in the fan's cited message** | owner, validated |
| `information_needs` | discovery goals, only when material; a known limit cannot be asked for again | owner, validated |
| `tentative_trajectory` | provisional future beats; a content beat binds an exposed candidate handle and states `media_role` | owner, validated |
| `current_beat`, `completed_beats` | current beat; append-only history (replans cannot touch it) | owner + application |
| `next_experience_move` | what happens next in the interaction — often no content | owner |
| `used_content`, `last_event`, `content_fact_seq` | **ledger mirror**: presented / accepted / sent / purchased | application only |
| `replans`, `last_replan_reason` | why the future changed | owner, validated |

### Invariants enforced in code

* **Planned ≠ presented ≠ accepted ≠ sent ≠ purchased.** Planning lives only in
  the trajectory; lifecycle facts only in `used_content`, rebuilt from the
  delivery/purchase ledgers and commercial state before the owner decides.
  `SessionDelta` has no field that can write them (attempts are refused by name).
* Lifecycle facts only advance; a truncated ledger page cannot make consumed
  content "unused". Consumed content is removed from candidates, cannot be bound
  to a beat, and cannot be presented or sent.
* A confirmed purchase never authorises another paid item: in the turn where a
  sale/delivery is first observed, a new paid operation requires the fan's own
  buying signal or media request.
* A fan-stated spending limit caps candidate pricing and every paid operation
  (purchases recorded after it was stated count against it).
* No funnel: nothing requires a session, constraint or budget before a legal
  operation. A direct request is handled directly.
* A session may be opened by the fan or PROPOSED by the creator when the
  moment genuinely suits one; a proposal is an offer awaiting his answer, never
  a routine step, and intimacy alone is not an opportunity.
* Content never advances merely because another candidate exists or the plan
  lists it next; the interaction must make that beat appropriate. There is no
  minimum number of turns between content events.
* An imagined premise must cite the fan's participation and is handed to Kimi
  explicitly as imagined. Real-world claims stay forbidden: the v1 writer
  contract (present-activity, publication, delivery, price checks) applies
  unchanged.

## Content candidates

`load_evidence(content_candidates=True)` (v2 only) calls
`db.commercial_queries.get_next_offer_with_candidates`, which reads the same
rows as the single-offer path and returns the unchanged next offer plus ≤ 3
bounded alternatives from `services.media_packages.build_candidate_offers`
(next progression rung, one alternative per unrepresented asset type, further
rungs), each priced inside its own approved bounds and under every explicit
ceiling. The owner sees opaque handles, asset type and approved description —
no prices, no catalogue. Choosing a handle in `operation_proposal` rebinds the
shared validator's `next_offer`; a pending offer is never rebound.

## Rollout and rollback

1. Apply `db/conversational_session_state_v2.sql` (additive; creates the
   owner-only table, widens the `conversation_core` CHECK constraints). It sits
   before the tenant-isolation pair in `db/migration_order.txt`. The production
   preflight warns if it is missing.
2. Owner selects **Conversational Core v2 — Session-aware** for a test fan in
   the Simulator (`PUT /creator/{c}/fan/{f}/conversation-core`). Real fans are
   never affected unless a creator-level override is set deliberately.
3. Rollback: select v1 (or clear). The resolver cache is cleared on write, so
   the next turn runs v1, which never reads or writes v2's table. Re-selecting
   v2 resumes from the persisted session.

## Evaluation

* Deterministic acceptance and a long immersion trajectory:
  `tests/test_conversational_core_v2.py`, scored by
  `services/session_immersion_eval.py` (premise continuity, no post-purchase
  sales, every content event made appropriate by the interaction rather than by
  a remaining candidate, a creator line for every fan line once media is
  removed). Turns between content events are a diagnostic only — natural pacing
  owns that gap.
* Live A/B against v1 through the real simulator:
  `python scripts/run_ab_trajectory_eval.py --baseline conversational_v1 --candidate conversational_v2 …`
  (see `docs/conversational_core_v1_evaluation.md`), followed by blind review of
  the transcripts with media cards removed.

## Experimental

* The owner and writer prompt extensions have not yet been tuned against the
  live GLM/Kimi models; the deterministic tests script both models.
* Candidate breadth (≤ 4) and the "post-event" guard window (the one turn in
  which the event is first observed) are initial values.
* Session state is written every turn (turn counters), so concurrent turns for
  one fan resolve through the revision CAS (`stale_generation`), as in v1.
