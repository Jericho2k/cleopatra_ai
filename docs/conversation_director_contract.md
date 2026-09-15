# Conversation Director v1 — backend/UI contract

The Conversation Director is a persistent deterministic state machine above the
writer and below the authoritative commercial policy.

## Progression

```text
OPENING → RAPPORT → FLIRT → QUALIFY → TENSION → SOFT_OFFER
```

That is a description of where a conversation usually goes, not a queue a fan
has to be walked through. A fan who has already said plainly that he wants
content (`direct_interest`) goes straight to SOFT_OFFER on the turn he says it,
whatever his message count — the phases describe context, they are not
checkpoints.

Commercial decisions can override it:

```text
OFFER | OBJECTION | PAID_SESSION | FOLLOW_UP | PAUSED | SAFETY
```

## Current state fields

- `phase`
- `previous_phase`
- `action`
- `fan_turn_count`
- `creator_turn_count`
- `turns_in_phase`
- `same_action_streak`
- `recent_actions`
- `engagement_score`
- `qualification_complete`
- `offer_eligible`
- `question_due` — the DISCOVER_PREFERENCE objective is live. It does not mean
  the reply must contain a question: the prompt states the objective (find out
  what he wants) and leaves the method to the writer
- `must_not_ask_question`
- `direct_interest` — he has already asked for content; no warm-up is owed
- `transition_reason`
- `director_version`

A future UI may show current phase, next move, transition reason and engagement
as explainable guidance. Engagement is not a guaranteed conversion probability.

## Persistence

Every field above is a column of `fan_conversation_directors`, and that is a
requirement rather than an observation: `save_conversation_director` upserts the
whole `to_context()` dict, and PostgREST rejects the entire row when one key has
no column. `direct_interest` was missing in production for the life of the
feature — see `db/MIGRATIONS.md` § "Drift: a column the application WRITES but
the schema does not have", and
`tests/test_schema_pipeline.py::test_the_director_can_persist_every_field_it_computes`,
which asserts the rule rather than the one column.

## Relationship to the Experience Director

They are different machines and do not overlap. The Conversation Director tracks
the RELATIONSHIP's phase across the whole conversation. The Experience Director
(`docs/experience_director_contract.md`) tracks the current SCENE — the beat,
what he just unlocked, how he reacted, and whether the conversation has earned
another paid moment. Only the Experience Director can withhold an offer.
