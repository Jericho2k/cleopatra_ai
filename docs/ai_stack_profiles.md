# AI Stack Profiles

## What a profile is

An **AI Stack Profile** is the complete configuration of every model-powered
stage in the conversational pipeline: which provider and model each stage
targets, what it falls back to, which prompt version it uses, whether reasoning
is on, and the generation parameters that materially change what comes back.

It is not "an AI model". It is the whole brain, named and versioned, so old and
new conversational behaviour can be compared against the same fixed runtime.

The authoritative registry is `ai/stack_profiles.py`. It is the only place a
provider/model pair is written down, and no client can submit one: the API
accepts a stable profile identifier and nothing else.

## What a profile is NOT

A profile does **not** fork application behaviour. These are shared and
identical under every profile:

- simulator isolation and the `test_` fan boundary
- inventory authority
- the deterministic commercial engine: what may be sold, at what price, from
  what inventory, under which session and lifecycle constraints
- session-plan recovery
- money handling and the customer price grid
- every safety rule, including the crisis backstop

"Legacy" means **old prompts and old model routing**. It never means "restore
old bugs".

## The stages

These are the model-powered stages that actually exist in the chat pipeline.
`ai/stage_classifier.py` is deliberately absent: it is pure logic with no model
call, and listing it would be inventing a stage.

| Stage | What it does |
| --- | --- |
| `situation_analyzer` | Reads the turn and reports observations. Makes no commercial decision. |
| `writer_default` | Ordinary conversation. |
| `writer_commercial` | Commercially complex, high-value and session-active turns. |
| `writer_safety` | Crisis and human-review turns. |
| `fan_intelligence` | Durable fact extraction. Enrichment; never blocks a reply. |
| `fan_summary` | Periodic psychological profile refresh, out of band. |

Which **route** a turn takes (`ai/writer_router.py`) is shared application logic
and is the same under every profile, because it describes the conversation
rather than the brain answering it. Only which model each route points at, and
which writer voice it uses, varies.

## The two shipped profiles

### `cleo_legacy_v1`

A frozen snapshot of the AI configuration that was on main before this pass.

| Stage | Provider / model | Fallback | Prompt | Reasoning |
| --- | --- | --- | --- | --- |
| situation_analyzer | anthropic / claude-haiku-4-5-20251001 | — | analyzer_v1 | off |
| writer_default | openrouter / moonshotai/kimi-k2.6 | together / Qwen/Qwen3.7-Plus | writer_v1 | off |
| writer_commercial | together / Qwen/Qwen3.7-Plus | — | writer_v1 | off |
| writer_safety | together / Qwen/Qwen3.7-Plus | — | writer_v1 | off |
| fan_intelligence | together / openai/gpt-oss-120b | — | fan_intelligence_v1 | off |
| fan_summary | together / meta-llama/Llama-3.3-70B-Instruct-Turbo | — | fan_summary_v1 | off |

It still honours `WRITER_DEFAULT_*`, `WRITER_COMPLEX_*`, `ANALYZER_*` and
`EXTRACTOR_*`. That is not an oversight: a deployment with one of those set is
running that model *today*, and a frozen profile that ignored them would not be
frozen.

### `cleo_v2`

| Stage | Provider / model | Fallback | Prompt | Reasoning |
| --- | --- | --- | --- | --- |
| situation_analyzer | anthropic / claude-haiku-4-5-20251001 | — | analyzer_v1 | off |
| writer_default | openrouter / moonshotai/kimi-k2.6 | together / Qwen/Qwen3.7-Plus | writer_v2 | off |
| writer_commercial | openrouter / moonshotai/kimi-k2.6 | together / Qwen/Qwen3.7-Plus | writer_v2 | off |
| writer_safety | together / Qwen/Qwen3.7-Plus | — | writer_v2 | off |
| fan_intelligence | together / openai/gpt-oss-120b | — | fan_intelligence_v1 | off |
| fan_summary | together / meta-llama/Llama-3.3-70B-Instruct-Turbo | — | fan_summary_v1 | off |

Two deliberate decisions:

**One writer voice.** The deterministic engine already decides what commercial
action is allowed, at what price, from what inventory. The writer increasingly
only has to express that decision, so there is no longer a reason for a sale to
arrive in a different model's voice than the conversation around it. Qwen on
Together remains the fallback for both routes — still a different provider, so
an OpenRouter incident cannot silence the writer.

**The safety route is unchanged.** A crisis turn is not commercial expression,
and keeping it on a second provider means one incident cannot take every writer
down at once.

Analyzer, extractor and summary are **not** re-pointed. Changing a model for
symmetry is how a comparison stops being a comparison.

`cleo_v2`'s stages are pinned: the writer/analyzer/extractor environment
variables do not re-point them.

## Resolution

```
test-fan simulation override   (fans.ai_stack_profile, `test_` fans ONLY)
        ↓
creator override               (creators.ai_stack_profile)
        ↓
AI_STACK_PROFILE               (deployment default)
        ↓
cleo_legacy_v1                 (built-in default)
```

The built-in default is the frozen profile on purpose: a deployment that has not
been told which brain to run must keep behaving exactly as it did before this
code shipped. Switching every fan onto a new brain because a variable is absent
is the outcome a versioning system exists to prevent. Production sets
`AI_STACK_PROFILE=cleo_v2` explicitly.

The creator override is **persistent and creator-scoped**, not a browser or
session setting, because Full Auto answers asynchronously from a worker where no
browser session exists.

`fans.ai_stack_profile` is honoured **only** for a fan whose `platform_fan_id`
starts with `test_`. The read path re-checks the prefix rather than trusting the
column, so a simulation setting can never reach a paying customer even if
something else wrote it there.

## Where the effective profile is recorded

- **Railway logs**, once per turn, before the first model call:
  `[AI STACK] feature=full_auto creator=… fan=… profile=cleo_v2 source=environment`
  and on the writer line:
  `[WRITER ROUTE] fan=… mode=auto profile=cleo_v2 route=… primary=…`
- **Every generated creator message**, inside the existing
  `messages.media_context` jsonb (no migration), as
  `{"ai_stack": {"profile", "route", "prompt_version", "provider", "model"}}`.
  That is enough to answer "which AI stack produced this message?" months later
  from the row alone, without a telemetry join.
- **Model telemetry**, as `ai_stack_profile` on the analyzer, writer and
  extractor metadata.

## Adding a profile

1. Add it to `PROFILES` in `ai/stack_profiles.py`, with a `StageSpec` for every
   entry in `STAGE_ORDER`.
2. Add its identifier to the CHECK constraints in
   `db/ai_stack_profile_v1.sql` (a follow-up migration, since the constraints
   are `NOT VALID` and additive).

The friction is intentional. This decides what every fan is answered by.
