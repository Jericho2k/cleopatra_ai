# Conversational Core v1 — split-role failure model

Core v1 now has two explicit model roles:

```text
raw conversation + sourced memory + working state
    -> GLM semantic decision
    -> deterministic operation preparation
    -> Kimi fan-facing writer
    -> deterministic validation / stale check / persistence
```

GLM never writes reply text, captions, rewrites, repair wording, or candidate
sentences. Its JSON contract contains semantic goals, must-address items,
initiative, pacing, evidence requests, an optional operation proposal, and
optional state/memory candidates. Any copy-like field makes the decision
malformed and triggers at most one bounded same-evidence GLM repair.

Kimi writes every model-generated fan-facing word. The dedicated
`conversational_writer` stage is pinned to `moonshotai/kimi-k2.6`; provider
failover may serve the same Kimi model elsewhere, but there is no GLM or
different-model fallback. Exhaustion returns `writer_failed`. When a provider
reports a served model outside the Kimi family, the output is suppressed and
recorded as a routing mismatch.

The components remain independently recoverable:

- An invalid operation is reduced to `none`; a valid conversational decision
  still reaches Kimi.
- A malformed or rejected state delta does not prevent Kimi generation.
- Unsafe writer clauses are suppressed or repaired without re-enabling an
  operation.
- An unusable GLM decision returns `owner_failed`; an unusable Kimi result
  returns `writer_failed`. Neither falls through to the legacy runtime.
- A stale conversation revision invalidates dependent output before send.

If GLM requests essential evidence not already present, application code
refreshes the typed snapshot once in the same fan turn and asks GLM for a final
decision. A second unresolved request becomes a typed insufficient-evidence
handoff; evidence is never fabricated.

The owner stability harness exercises the semantic decision boundary with
seeded malformed responses:

```bash
python scripts/run_owner_stability_eval.py --turns 500
```

Conversation quality remains a whole-trajectory human-review question. The
failure harness verifies routing, bounded recovery, isolation, and authority;
it does not claim human-like quality.
