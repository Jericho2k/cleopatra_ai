# Selectable live conversation core

`semantic_v1` is a selectable replacement for the legacy conversational
controllers. It runs through the production Assisted, Full Auto, simulator and
conversational scheduled-event entry points. It is not enabled by this change.

## Ownership on the selected path

| Concern | Owner |
| --- | --- |
| Meaning, response intent, obligations, reply/silence/handoff, proposed operation | `SemanticDecisionOwner(strict_live=True)` |
| Tenant, control state, exact records, price, inventory, caps, holds and payment state | deterministic validation in `services/live_orchestration.py` |
| Natural-language expression of the approved plan | the configured writer stage |
| Delivery, duplicate prevention and reconciliation | shared delivery ledger, `send_locked_ppv` and deterministic workers |
| Whether a scheduled conversational event merits a message now | the same semantic owner and validator |

Conversation Director, Experience Director, session strategy, commercial
`decide_next_action`, situation `strategic_move`, stage guidance and the legacy
prompt builder are not called on this path. Useful persisted facts, commercial
state, inventory, continuity and delivery boundaries remain shared. Payment
reconciliation and queue repair remain deterministic workers.

## Evidence and execution

Each turn receives `live_evidence_v1`, a bounded typed snapshot with creator and
fan scope, trigger identity, state revision, creator facts and voice, recent
turns, source-linked historical facts, unresolved obligations, corrections,
approved inventory, exact pending offer, confirmed purchases and deliveries,
pending payment, explicit limits, operator constraints, memory/backfill status
and named truncation.

The semantic owner can only propose a typed operation. Validation binds the
proposal to the current authoritative record. The writer sees only the approved
execution plan and cannot choose another operation. State is re-read before
each visible send. Locked-delivery planning is a dry run during generation and
is durably committed only after the writer succeeds and the state revision is
still current.

Assisted generation has no commercial side effect. Its provenance token binds
approval to the evidence revision and exact record references; approval
revalidates before delivery. Operator edits remain attributed. Offer state is
committed only after its text is accepted by the platform, while locked items
use the same durable PPV adapter as Full Auto.

## Selection, rollout and rollback

Apply `db/conversation_core_v1.sql`. Both columns are nullable and the migration
sets no value, so existing accounts stay on legacy behavior.

Resolution order is:

1. a `test_` fan override;
2. creator override;
3. `CONVERSATION_CORE` (`legacy` or `semantic_v1`);
4. built-in `legacy` default.

Fan-level overrides are honored only for simulation test fans. Registry and
write endpoints require the platform owner; creator routes also enforce normal
tenant access. The simulator dashboard exposes the test-fan selector only to
that owner.

Rollback is immediate and state-preserving: clear the test-fan or creator
override to inherit the next level, or explicitly choose `legacy`. Do not reset
fan commercial state, sessions, pending payments or purchases. An unknown
stored/environment value is a visible configuration error and never falls back
to a different architecture mid-turn.

No live creator is activated by the migration or application code.

## Controlled evaluation

Use separate `test_` fans with identical seeded inventory and initial state so
one candidate cannot inherit the other's conversation or transaction state:

```bash
python scripts/run_trajectory_eval.py \
  --creator <creator-id> --fan <legacy-test-fan-id> \
  --core legacy --simulate-time --fail-on-uncovered

python scripts/run_trajectory_eval.py \
  --creator <creator-id> --fan <semantic-test-fan-id> \
  --core semantic_v1 --simulate-time --fail-on-uncovered
```

The shipped trajectories include 45-turn ordinary conversation, a 34-turn
deferred-question sequence, and a 40-turn day/week return sequence. The runner
uses the real simulator/Full Auto entry point, records outcomes and provenance,
and restores the fan's prior core override even when a run fails.

For the existing paid provider candidate comparison:

```bash
python scripts/run_candidate_provider_eval.py \
  --confirm-paid-provider-calls \
  --output evaluation_bundles/<run-name>
```

Provider runs require configured inference credentials and incur cost. Unit and
trajectory tests establish contracts, routing and deterministic defects; they
do not establish conversational quality. Before unattended operation, compare
multiple complete runs, inspect actual-provider/fallback provenance, latency,
cost and intervention rates, and have a human review complete transcripts and
candidate disagreements.
