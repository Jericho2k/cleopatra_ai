# Ordinary writer: Kimi K2.6 through OpenRouter

## What changed and why

Kimi K2.6 is Cleopatra's preferred ordinary conversational writer. Together
removed the K2.6 endpoint, so PR #22 migrated the Together route to Kimi K3.
This change restores K2.6 by reaching it through OpenRouter instead, pinned to a
single upstream provider.

The Together compatibility rule is unchanged and still correct:

| Configured provider | Configured model        | Model actually used  |
| ------------------- | ----------------------- | -------------------- |
| `together`          | `moonshotai/Kimi-K2.6`  | `moonshotai/Kimi-K3` |
| `openrouter`        | `moonshotai/kimi-k2.6`  | `moonshotai/kimi-k2.6` |

The redirect in `ai/model_migrations.py` is keyed on `(provider, model)`, so it
only ever applied to Together. It was not loosened.

## Routing

```
ordinary conversation      -> openrouter / moonshotai/kimi-k2.6  (pinned upstream)
commercially complex turn  -> together   / Qwen/Qwen3.7-Plus
safety-sensitive turn      -> together   / Qwen/Qwen3.7-Plus
```

The Together routes deliberately stay on a different provider, so an OpenRouter
incident cannot take the commercial writer down with it. `ai/generator.py` still
runs the bounded plan of two primary attempts followed by one explicit fallback
attempt: for an ordinary turn that is Kimi, Kimi, Qwen3.7-Plus.

> **2026-09 correction.** The complex/fallback route was
> `deepseek-ai/DeepSeek-V4-Pro`. Together answers that handle with a live 400 for
> this account — *"Unable to access non-serverless model
> deepseek-ai/DeepSeek-V4-Pro. Please create and start a dedicated endpoint."* —
> so the fallback had never worked, and an ordinary turn whose two Kimi attempts
> failed ended with no reply at all. `Qwen/Qwen3.7-Plus` is serverless on Together
> and callable with the same `TOGETHER_API_KEY`.

## Reasoning must be off on both writer targets

Both catalog entries set `"reasoning_enabled": false`, and
`ai/model_providers.py` forwards it as `reasoning: {"enabled": false}`.

This is a correctness requirement, not tuning. Kimi K2.6 reasons by default. With
reasoning on it spends the completion budget on hidden reasoning and returns
`message.content = null`; the writer parses that as unparseable output and moves
on. The result is a total, silent writer failure on an HTTP 200: nothing is sent,
nothing is retried usefully, and the only Railway line is whatever the last model
in the ladder happened to say.

The writer never needs chain-of-thought. It needs the final JSON array of chat
replies.

Verify a route before trusting it — this spends real credits and CI never runs
it:

```
OPENROUTER_API_KEY=... python scripts/openrouter_smoke.py
TOGETHER_API_KEY=...   python scripts/openrouter_smoke.py --route complex
```

The smoke fails non-zero on empty content, on output that does not parse into
usable replies, and on an upstream other than the pinned one. A transport
success is not a pass.

## Provider pinning (fail closed)

Every OpenRouter request carries a `provider` object:

```json
{
  "provider": {
    "only": ["Inceptron"],
    "allow_fallbacks": false,
    "data_collection": "deny"
  }
}
```

* `only` — the request may use no other upstream.
* `allow_fallbacks: false` — if Inceptron is unavailable, OpenRouter returns the
  upstream error rather than silently switching provider. That error is
  recorded in telemetry and in the runtime health state, and Cleopatra's own
  explicit Together fallback then covers the turn. A random OpenRouter provider
  would change writing style, price, and cache locality with no signal at all.
* `data_collection: "deny"` — OpenRouter routes only to providers that do not
  collect user data, and errors explicitly if the pinned provider does not
  qualify.

**On privacy claims:** these are the controls OpenRouter exposes; they are not a
Zero Data Retention guarantee. `OPENROUTER_ZDR=true` additionally sets
`provider.zdr`, which asks OpenRouter to restrict routing to ZDR endpoints. It
is off by default and should only be turned on after confirming on the model's
OpenRouter provider page that a ZDR endpoint exists for it — otherwise every
request fails.

## Prompt caching

Caching matters here because Cleopatra sends a large, mostly identical context
on every message of a conversation.

**Session affinity.** Each request carries OpenRouter's documented sticky-routing
key, `session_id`, so all turns of one fan conversation reach the same upstream
and hit the same warm cache. The key is derived in `ai/session_affinity.py` as a
truncated SHA-256 of `creator_id` + `fan_id` — deterministic, with no timestamp,
counter, or random component, and hashed so internal identifiers do not leave our
infrastructure. `user` carries a second digest in a different namespace as the
pseudonymous end-user identifier OpenRouter uses for abuse isolation.

**Prompt structure.** `ai/prompt_builder.py` orders the prompt so the shared
prefix between consecutive turns is as long as possible:

1. *Stable prefix* — platform context, writer rules, creator persona and voice,
   formatting rules, boundaries, stop words, few-shot examples. Identical for the
   whole conversation, and by far the largest block.
2. *Semi-stable* — durable fan profile and memory, then the recent transcript.
   Both are appended to rather than rewritten.
3. *Volatile tail* — conversation stage, situation, commercial directive,
   transient state, the newest incoming message, and the response instruction.

The response instruction stays last on purpose: it must be the most recent thing
the model reads.

Moving stage/mood/strategy behind the transcript matters because the analyzer
recomputes them on every inbound message. Previously a single mood change
truncated the shared prefix before any conversation history.

**No fake cache markers.** `build_prompt` still emits Anthropic content blocks
with `cache_control` because the Anthropic route consumes them natively.
`flatten_message_content` in `ai/generator.py` collapses them to ordered plain
text for every OpenAI-compatible transport, so no `cache_control` marker is ever
sent to an API that does not use them. (That function also fixes a real bug: the
system message was previously stringified with `str()`, which shipped a Python
list repr as the system prompt.)

## Telemetry

Per writer generation, into the existing `model_usage_events` table via
`services/model_telemetry.py`:

| Field | Source |
| --- | --- |
| `provider` | the logical route, e.g. `openrouter` |
| `model` | e.g. `moonshotai/kimi-k2.6` |
| `feature`, `creator_id`, `fan_id` | existing telemetry context |
| `input_tokens` | `prompt_tokens` minus cached (and minus cache writes when they are inside `prompt_tokens`) |
| `cache_read_tokens` | `usage.prompt_tokens_details.cached_tokens` |
| `cache_write_tokens` | `usage.prompt_tokens_details.cache_write_tokens` |
| `output_tokens`, `latency_ms`, `retry_count`, `success` | existing |
| `estimated_cost_usd` | OpenRouter's reported `usage.cost` when present, else catalog pricing |
| `metadata.upstream_provider` | the provider that actually served the request |
| `metadata.cost_source` | `provider_reported` or `catalog_estimate` |
| `metadata.cached_input_tokens`, `metadata.cache_hit_ratio` | derived from the same counters |
| `metadata.writer_route`, `writer_fallback_used`, ... | existing route metadata |

Cached input is never double counted: `input_tokens + cache_read_tokens`
reconstructs `prompt_tokens` exactly. No prompt or conversation content is ever
logged.

To check whether caching is working:

```sql
select
  metadata->>'upstream_provider' as upstream,
  count(*)                       as calls,
  avg((metadata->>'cache_hit_ratio')::float) as avg_cache_hit_ratio,
  sum(estimated_cost_usd)        as cost_usd
from model_usage_events
where provider = 'openrouter'
  and created_at > now() - interval '1 day'
group by 1;
```

## Model health

`services/model_availability.py` checks each configured writer against its own
provider's catalog (`/models` on Together and on OpenRouter), using that
provider's credential. The previous Together-only check would have reported the
OpenRouter-hosted Kimi as unavailable simply because Together does not list it.

* A missing API key for a checked provider → `misconfigured`.
* A catalog that answers but omits the model → `degraded` / `unavailable`.
* A catalog that cannot be read → `check_failed`, and the model is reported as
  *unknown*, never as absent.
* A provider with no catalog check (Anthropic) → `available: null`.

The OpenRouter row also reports `pinned_providers` so an operator can see the
active pin in `/model-runtime-health`.

## Live smoke check

Never run in CI; it spends real credits.

```
OPENROUTER_API_KEY=... python scripts/openrouter_smoke.py
```

It issues two requests with an identical long prefix and the same affinity key,
and prints the upstream provider, token counts, cached tokens, cost, and latency
for each. Attempt 2 showing `cached > 0` is the confirmation that provider-side
prefix caching is live.

## Railway environment changes

New:

```
OPENROUTER_API_KEY=<key>
WRITER_DEFAULT_PROVIDER=openrouter
WRITER_DEFAULT_MODEL=moonshotai/kimi-k2.6
OPENROUTER_PROVIDERS=Inceptron
OPENROUTER_ALLOW_FALLBACKS=false
OPENROUTER_DATA_COLLECTION=deny
```

Unchanged (must stay set):

```
TOGETHER_API_KEY=<key>            # complex/fallback route
WRITER_COMPLEX_PROVIDER=together
```

Changed (2026-09), and this one must be edited by hand in Railway if it is set
there — an explicit variable overrides the code default:

```
WRITER_COMPLEX_MODEL=Qwen/Qwen3.7-Plus
```

Optional: `OPENROUTER_ZDR`, `OPENROUTER_BASE_URL`.

Existing deployments that do not set `WRITER_DEFAULT_*` pick up the new default
automatically and will need `OPENROUTER_API_KEY`. A deployment that pins
`WRITER_DEFAULT_PROVIDER=together` keeps its current Kimi K3 behaviour.

## Rollback

Set `WRITER_DEFAULT_PROVIDER=together` and `WRITER_DEFAULT_MODEL=moonshotai/Kimi-K3`.
No code change or migration is involved.
