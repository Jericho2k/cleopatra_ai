# Adaptive Session Planning — backend/UI contract

The planner does not replace commercial policy. It converts the authoritative
commercial decision plus lifecycle, affordability, price learning, active
session, and conversation stage into one explainable next-best action.

## Current strategy fields

- `goal`: CARE, RAPPORT, QUALIFY, WARM, TEASE, PRESENT_OFFER,
  HANDLE_OBJECTION, CLOSE, DELIVER, FOLLOW_UP, HOLD
- `phase`: human-readable current phase
- `next_action`: deterministic execution step
- `writer_goal`: concise instruction for the writer
- `writer_avoid`: forbidden tactics for this turn
- `approved_offer_ids` / `approved_offer_prices_cents`: copied only from the
  commercial decision
- `selected_offer_price_cents`: exact approved amount, when one exists
- `route_hint`: default, commercial_complex, or safety_sensitive
- `reason_codes`: explainability and audit

## Pricing hierarchy

```text
environment safety fallback
→ agency-scoped DB defaults
→ creator-scoped DB overrides
→ fan price-learning range
→ current affordability
→ approved vault-set boundaries
→ final approved package price
```

The later UI should edit `price_learning_policy_scopes.settings`; Railway values
remain emergency fallbacks.

## Vault fields

- `base_price_cents`
- `min_price_cents`
- `max_price_cents`
- `dynamic_pricing_enabled`

A learned fan target never authorizes a price outside those boundaries.
