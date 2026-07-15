# Conversation Director v1 — backend/UI contract

The Conversation Director is a persistent deterministic state machine above the
writer and below the authoritative commercial policy.

## Progression

```text
OPENING → RAPPORT → FLIRT → QUALIFY → TENSION → SOFT_OFFER
```

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
- `question_due`
- `must_not_ask_question`
- `transition_reason`
- `director_version`

A future UI may show current phase, next move, transition reason and engagement
as explainable guidance. Engagement is not a guaranteed conversion probability.
