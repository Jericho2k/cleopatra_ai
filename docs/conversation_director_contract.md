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
