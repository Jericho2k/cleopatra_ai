# Conversational Core v1 — the owner failure model

## The problem this replaces

Core v1 asked one model call to produce the fan-facing reply, a typed response
intent, a disposition, a proposed commercial operation, four exact operation
references, three lists of interpretive metadata, a confidence, and a
working-state delta — as one strict JSON object. It then read that object with
`parse_reply_plus_intent`, which is deliberately all-or-nothing because it
exists to make an **offline comparison** of two candidate architectures fair.

Applied to a live turn, that parser made every recoverable mistake fatal:

| what the model got wrong | what the fan got |
| --- | --- |
| `confidence: 1.5` | nothing |
| `response_intent: "vibe_check"` | nothing |
| `state_delta` truncated by the token budget | nothing |
| operation citing an offer that no longer exists | nothing (PR #65 fixed this one case) |
| `message.content` empty | nothing, after ~98 seconds |

The reply was in the response every time except the last.

The last row is the one that produced
`reason=the conversational owner did not answer usably: the response contained
no JSON object`. That message is a true statement about an empty string, and it
says nothing about why the string was empty.

## Root causes

1. **One token budget, two consumers.** On every OpenAI-compatible reasoning
   route, the request's `max_tokens` covers the hidden reasoning trace *and* the
   visible answer. The owner's budget was 4096 with reasoning mandatory and
   uncapped, so GLM-5.3-Flash could — and did — spend all of it thinking,
   returning `message.content = null` with `finish_reason = "length"`. The Kimi
   entry in `config/model_candidates.json` already carried a note about exactly
   this failure for a *non*-reasoning route; the owner route reintroduced it by
   turning reasoning on.
2. **A transport with no diagnostics.** `content = response.choices[0].message.content or ""`
   collapsed a truncation, a content filter, a refusal, a 200 carrying an
   `error` object, and a model that wrote prose into one empty string.
3. **An offline-comparison parser doing live work.** See the table above.
4. **No repair at all on the v1 path.** `semantic_v2` had three attempts;
   `conversational_v1` had one call and then `owner_failed`.
5. **An open upstream pool with no parameter requirement.** The GLM route pins
   no provider, and without `provider.require_parameters` OpenRouter may serve a
   request carrying `response_format` and `reasoning` from an upstream that
   ignores both.

## The failure model now

```
provider response
   -> ai/model_providers: structural diagnostics, never conversation text
   -> services/owner_contract: reply | operation | state_delta, read independently
   -> at most ONE bounded same-evidence repair, if and only if there is no reply
   -> services/live_orchestration.settle_conversational_v1_turn:
        deterministic authority rules on the operation
        the writer contract rules on the fan-visible copy
   -> only an answer with no reply and no stated silence is owner_failed
```

### The three components fail independently

* **Reply** — the only required output. It survives malformed optional
  metadata, a rejected operation, a discarded delta, and JSON truncated
  part-way through the object. `services/owner_contract` closes an unterminated
  object locally, and as a last resort reads the `reply` string alone.
* **Operation** — optional. Anything unreadable becomes `none`, recorded as
  discarded. An unreadable `confidence` disqualifies the operation but not the
  reply: an external effect may never rest on a number the parser invented.
  Deterministic validation in `validate_decision` remains the only authority
  that can approve one.
* **State delta** — optional and opaque to the contract. It goes to
  `validate_and_apply_delta` exactly as received, because that validator
  already refuses fields individually. A delta that cannot be read at all —
  including one that makes the validator raise — costs one turn of
  interpretation and nothing a fan sees.

Every recovery makes the answer *smaller*. None invents a reference, raises a
confidence, or promotes a proposal.

### The bounded repair

Triggered only when there is no reply and no stated reason for silence. It is
one extra call, against the same immutable evidence, asking for the minimum
object (`reply`, `response_intent`, `operation`) with reasoning capped at 512
tokens and a shortened deadline. It may not carry a `state_delta`, and any
operation reference outside `_evidenced_reference_set(loaded)` — every exact ref
the owner was actually shown — discards the operation before deterministic
validation sees it. If the repair also fails, the turn is `owner_failed` with a
named category, and no legacy controller runs. Two calls is the whole budget.

### What `owner_failed` now reports

`empty_content_truncated`, `empty_content_unexplained`, `content_filtered`,
`provider_refusal`, `provider_error`, `provider_timeout`, `transport_error`,
`no_json_object`, `invalid_json`, `json_not_an_object`, `missing_reply`,
`empty_reply` — plus, per attempt, the model, upstream provider, latency,
finish reason, response id, content length, whether content was null, reasoning
characters and tokens, completion tokens, which message fields were populated,
and what the request asked for. All of it is lengths, counts, enums and
provider-authored status. No fan or creator text is logged, and
`test_diagnostics_carry_no_conversation_text` holds that line.

## Configuration changes

| where | change | why |
| --- | --- | --- |
| `ai/stack_profiles.py` | owner `max_tokens` 4096 → 8192, `CONVERSATIONAL_OWNER_MAX_TOKENS` override | the answer needs budget after the trace |
| `config/model_candidates.json` | `reasoning_max_tokens: 1024` on the GLM owner | a hard ceiling on the trace |
| `config/model_candidates.json` | `openrouter_require_parameters: true` | route only to upstreams that honour `response_format` and `reasoning` |
| `config/model_candidates.json` | owner `timeout_seconds` 90 → 60 | a bounded trace should not take 98 seconds; the repair adds at most 45 |

`OPENROUTER_REQUIRE_PARAMETERS=false` widens the pool again without a deploy if
no eligible upstream remains for this model. The model itself was not changed:
the evidence pointed at the orchestration and the request parameters, not at
the endpoint's suitability.

## The stability gate

`python scripts/run_owner_stability_eval.py --turns 500`

`services/owner_stability_eval.py` drives many turns through
`decide_conversational_v1` and `settle_conversational_v1_turn` — the real
functions — with only the transport replaced by a seeded generator of the
response shapes production has actually produced. It reports owner-failed rate,
malformed-first-response rate, repair-attempt and repair-success rates, no-send
rate, operation-proposal and operation-rejection rates, state-delta rejection
rate, a failure-category histogram, and p50/p95 latency.

It needs no network, database or API key, so it is a pre-deploy check rather
than a report written after an incident. `--fail-over-owner-failed` makes it a
gate. It reports arithmetic and issues no verdict; conversation *quality*
remains the job of `services/ab_trajectory_eval.py` and the blind review.
