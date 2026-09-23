# Evaluating Conversational Core v1

This document describes the evaluation, not the runtime. The runtime is being
built separately; nothing here implements it, imports it, or assumes anything
about its internals beyond one optional provenance key described at the end.

The success criterion for this work is **not** "Core v1 wins". It is:

> We can fairly and reproducibly determine whether Core v1 wins.

Everything below follows from that. There is no overall score anywhere in the
harness, because a single number is exactly how an evaluation stops being able
to answer the question it was built for.

---

## The experiment

```
                 one scripted fan trajectory
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
         semantic_v2                conversational_v1
        (against test fan A)       (against test fan B)
              │                           │
              └─────────────┬─────────────┘
                            ▼
              paired transcripts · objective metrics
                            ▼
                 blind review (A / B, keyed apart)
```

Both arms run the *real* Full Auto turn through the simulator
(`services.suggestions.run_simulated_inbound`) — the real analyzer, the real
orchestrator, the real writer routing, the real delivery boundary. Nothing talks
to the platform: the simulator's own scope refuses every remote call, and every
arm is checked to be a `test_` fan before anything runs.

The candidate runtime is a **string**. The harness validates it against
`services.conversation_core.CORE_IDS` *at call time*, so the day the runtime
branch registers `conversational_v1`, every command below works unchanged.
Until then, baseline-only runs are first class.

---

## Commands

### Describe the suite (no backend, no database, no cost)

```bash
python scripts/run_ab_trajectory_eval.py --describe
```

### Baseline only — works today, before `conversational_v1` exists

```bash
APP_ENV=development EVAL_CLOCK_ENABLED=1 \
python scripts/run_ab_trajectory_eval.py \
    --baseline semantic_v2 \
    --creator <creator-id> \
    --provision-fans \
    --suite conversational \
    --simulate-time
```

### A/B, once the candidate runtime registers its core id

```bash
APP_ENV=development EVAL_CLOCK_ENABLED=1 \
python scripts/run_ab_trajectory_eval.py \
    --baseline semantic_v2 \
    --candidate conversational_v1 \
    --creator <creator-id> \
    --provision-fans \
    --suite conversational \
    --simulate-time \
    --seed 1729
```

### The full suite, including the commercial scenarios

```bash
APP_ENV=development EVAL_CLOCK_ENABLED=1 \
python scripts/run_ab_trajectory_eval.py \
    --baseline semantic_v2 \
    --candidate conversational_v1 \
    --creator <creator-id> \
    --provision-fans \
    --suite all \
    --simulate-time \
    --blind-review
```

One scenario, against fans that already exist:

```bash
python scripts/run_ab_trajectory_eval.py \
    --baseline semantic_v2 --candidate conversational_v1 \
    --creator <creator-id> \
    --baseline-fan <test-fan-a> --candidate-fan <test-fan-b> \
    --scenario D_shared_imagined_scene --scenario L_long_trajectory
```

### Blind review

```bash
python scripts/build_conversation_blind_review.py eval/results/<run-id>

# ... complete the review, then:
python scripts/build_conversation_blind_review.py eval/results/<run-id> --unblind
```

### Optional model judge (costs money, never automatic)

```bash
python scripts/run_conversation_judge.py eval/results/<run-id> \
    --model <candidate-name-from-config/model_candidates.json> \
    --confirm-paid-provider-calls
```

`--simulate-time` needs `APP_ENV != production` and `EVAL_CLOCK_ENABLED=1`
(`core/clock.py`). Without it the elapsed-time claims are reported as
**uncovered** rather than quietly assumed, which is the existing convention in
`scripts/run_trajectory_eval.py`.

---

## Experimental fairness

Held constant, and **checked** rather than asserted:

| Held constant | How |
| --- | --- |
| Scripted fan inputs | One trajectory object drives both arms. `paired.json` carries a digest of each arm's inputs; a mismatch is named per scenario, never averaged. |
| Creator and persona | One `--creator` for the whole run. |
| AI stack profile | Resolved once and recorded in `metadata.json`. `models_served` per arm makes a fallback visible, so "two architectures" cannot silently become "two models". |
| Starting conversation state | Seeded fixture state is cleared for every arm before every scenario, and the same `seed` block is applied to both. |
| Timing semantics | The simulated clock is reset between arms and between scenarios; `days_since_previous` is applied identically. |
| Arm order | Shuffled per scenario from the run seed and recorded, so the second-mover advantage is randomised rather than systematic. |

Adaptive trajectories are **refused**. An adaptive customer reacts to the reply,
so each arm would face a different conversation — a legitimate experiment, and
not this one.

### Isolation: separate test fans, one per arm

A conversation in this product is persistent by design: history, commercial
state, affordability, price learning, lifecycle. Two runtimes writing into one
fan is one conversation with two authors, so the harness **refuses to start**
when both arms name the same fan (`assert_arm_isolation`).

`--provision-fans` creates a fresh `test_` fan per arm for the run via
`services.simulation_workspace.create_test_fan`. That is the strongest available
form of equivalent initial conditions: both arms begin from a fan with no
conversation, no purchase history, no learned budget and no stale commercial
state. Pre-existing fans can be named with `--baseline-fan` / `--candidate-fan`
when a specific starting state is wanted.

### Safety

- Every arm's fan row is re-read from the database and must satisfy
  `core.simulation.is_simulatable_fan` (`platform_fan_id` starting `test_`).
  A real fan is refused with `UnsafeEvaluationTarget`.
- A fan belonging to another creator is refused.
- Seeded purchases go through `services/trajectory_fixtures.py`, which already
  refuses anything but a test fan and namespaces every reference `eval:`.
- Each arm's previous core selection is restored afterwards, including when the
  run raises.

**Never run this against a real customer.**

---

## Scenarios

`eval/conversational_core_scenarios.json`. Every message was written for this
file; nothing is copied from a real conversation, and
`tests/test_conversational_core_scenarios.py` asserts that no handle, link,
email or long digit run appears in any of them.

| Id | What it exercises |
| --- | --- |
| `A_ordinary_statement` | A plain statement where a reaction suffices. Specificity, contribution, unnecessary questions. |
| `B_compliment_to_intimate` | Compliment moving to a more direct register. Canned validation, contextual progression, premature commercial pivot. |
| `C_short_ambiguous_reply` | "yeah maybe" / "mm" / "idk" four turns in. Contextual interpretation, continuity, generic-question behaviour. |
| `D_shared_imagined_scene` | A hypothetical built over several turns, then referred to obliquely. Scene continuity, reference resolution, reality vs imagination. |
| `E_creator_takes_initiative` | Fan warm but bringing nothing. Creator contribution without interrogation. |
| `F_direction_change` | Explicit mid-conversation pivot. Cheap adaptation, no forced old trajectory. |
| `G_correction` | The fan corrects a stated fact. Correction honoured, old belief not resurfaced. |
| `H_delayed_return` | A day, then a week. Natural resumption, relevant callback, no invented present-world claim. |
| `I_natural_media_interest` | Interest arising from conversation. Conversational transition, no catalogue copy, no metadata leakage. |
| `J_commercial_rejection` | A decline mid-conversation. No pressure loop, no immediate re-offer, natural continuation. |
| `K_post_event_continuation` | Starts from a seeded authoritative purchase. Transaction truth, staying in the moment, no automatic next sale. |
| `L_long_trajectory` | 22 fan turns over 8 simulated days: ordinary talk, two topic changes, a shared scene, short replies, a 19-turn callback, a correction, a cooldown and a delayed return. |
| `M_paid_platform_media_request` | Repeated direct requests to see content inside the private paid chat. Same-chat affordance judgment, narration vs delivery, and imagined vs present-world action. |

Suites: `all`, `conversational` (A–H, L), `commercial` (I, J, K, M), `smoke`
(A, C). Four of thirteen scenarios are commercial; M specifically measures the
paid-platform operating model without making every intimate turn a sale.

`I_natural_media_interest` measures the *proposal and the fan-visible wording*.
Whether the full media flow executes depends on the creator's catalog fixtures;
the harness records the operation proposal and its result either way.

---

## Artifacts

```
eval/results/<run-id>/
    metadata.json        run id, timestamp, git SHA, scenario ids, seed,
                         baseline/candidate cores, creator + fan ids, stack
                         profile, requested and served models, isolation record,
                         arm order per scenario
    semantic_v2.json     the baseline arm: every conversation, every turn
    conversational_v1.json   the candidate arm, named by its core id
    paired.json          scenarios lined up across arms, with the input-digest
                         fairness check
    metrics.json         objective metrics per arm, per scenario and pooled
    blind_review.md      reviewer-facing, names no runtime
    blind_mapping.json   the key
    judge.json           optional, only if the judge was run
```

`eval/results/` is already gitignored, and
`tests/test_ab_trajectory_eval.py` asserts it with `git check-ignore` rather
than by reading the file. Generated conversations are not source.

### Turn-level record

Per turn: index, fan input, creator output (per bubble), bubble count,
conversation core (as recorded, plus as requested), latency, outcome and
classified outcome, control record (disposition, hold, handoff, freeze, error),
requested and served model, provider, operation proposal and result including
deliveries, context fingerprint, provenance turn id, applied transforms.

Optional and absent when not recorded: `tokens`, `cost_usd`, `core_state`.
Absent rather than zero — a runtime that has no concept of a field must not
contribute a value that something downstream can average.

---

## Metrics vs review

`metrics.json` contains only arithmetic over what was sent and what the
pipeline recorded:

question rate; share of creator turns ending in a question; longest consecutive
question-ending streak; question marks per turn; repeated 5-word phrases;
repeated opening fragments; verbatim repeats; reply length and word
distributions including variance; bubbles per turn; latency including the tail;
tokens and cost **with their coverage**; error, handoff, freeze and silence
counts; operation proposals and refusals; models served and the share served by
the requested model.

It explicitly does **not** claim to measure specificity, contribution,
initiative quality, scene quality, pacing, truthful presence, commercial
naturalness or overall coherence. Those are listed by name in
`metrics.json.not_measured_here` so nobody reading the file can mistake the
counts for the rubric. They are judged in the blind review.

### Blind review

`blind_review.md` shows Conversation A and Conversation B with the fan's
messages identical between them. Label assignment is drawn from the run seed
**per scenario**, so working one out reveals nothing about the next. The
document contains no core id, no `baseline`/`candidate`, no model name, no
latency and no fan id — anything of the sort would be a fingerprint grouping a
runtime's scenarios together. The build refuses to emit a document that carries
a forbidden term; terms appearing inside conversation text are redacted.

Thirteen anchored dimensions, rated separately, with no total.

### Optional judge

`scripts/run_conversation_judge.py` is off unless run, costs money, requires
`--confirm-paid-provider-calls`, receives the same blinded transcripts, and
returns a rating plus an observable reason per dimension. No total, no winner
field, no hidden reasoning requested. It does not replace the human review.

---

## Component audit

**REUSE** — unchanged, driven as-is:
`services/trajectory_eval.py` (turn classification, deterministic detectors,
coverage gaps, `run_trajectory`), `services/trajectory_fixtures.py` (seeding and
its test-fan refusal), `services/suggestions.run_simulated_inbound`,
`services/reply_provenance.py`, `services/message_diagnostics.py`,
`services/conversation_core.py` (selection and its `test_`-only override),
`services/simulation_workspace.create_test_fan`, `core/simulation.py`,
`core/clock.py`, `core/build_info.py`, `eval/trajectories.json` format,
`config/model_candidates.json`, `.gitignore`'s `eval/results/`.

**EXTEND** — two small, additive changes to existing files:

- `scripts/run_trajectory_eval.py` — `--core` no longer hard-codes
  `("legacy", "semantic_v1")`; it validates against `CORE_IDS` at run time, so a
  runtime registered on another branch works there too without an edit.
- `services/reply_provenance.py` — one optional `core_state` block and the
  total `record_core_state` recorder that writes it. Nothing reads it to decide
  anything, and no existing field changed.

**NEW**:
`services/ab_trajectory_eval.py`, `services/conversation_metrics.py`,
`services/blind_conversation_review.py`, `services/conversation_judge.py`,
`scripts/run_ab_trajectory_eval.py`,
`scripts/build_conversation_blind_review.py`,
`scripts/run_conversation_judge.py`,
`eval/conversational_core_scenarios.json`, this document, and the four test
modules.

Nothing in `services/live_orchestration.py`, `services/conversation_core.py`,
the runtime routing or any migration was touched.

---

## The one hook the runtime branch needs

**Required, and it is one line of configuration:** register the runtime's id in
`services/conversation_core.py`:

```python
CORE_CONVERSATIONAL_V1 = "conversational_v1"
CORE_IDS = (CORE_LEGACY, CORE_SEMANTIC_V1, CORE_SEMANTIC_V2, CORE_CONVERSATIONAL_V1)
```

That is the whole integration. The evaluator reads `CORE_IDS` at call time and
needs no other change: no new enum, no new branch, no new artifact shape.
`tests/test_ab_trajectory_eval.py::test_a_newly_registered_core_id_is_accepted_with_no_evaluator_change`
simulates exactly this registration and runs a complete A/B through it.

**Optional, and genuinely optional:** if Core v1 wants its working state to
appear in the run artifacts, the recorder already exists on this branch —
`ReplyProvenance.record_core_state` in `services/reply_provenance.py`, added
here so the producer and the reader agree on one key instead of inventing two:

```python
provenance.record_core_state({
    "state_before": {...},
    "proposed_delta": {...},
    "accepted_fields": [...],
    "rejected_fields": [...],
    "state_after": {...},
})
```

Any subset of those five fields is fine, any other field is carried through
unread, and it is total: it never raises, and anything that is not a mapping is
ignored. A runtime that writes none is not a gap in the run — `semantic_v2`
writes none, and the harness records the key as absent rather than empty. The
evaluator imports nothing from the runtime and has no opinion about what the
state contains.

---

## Deliberately deferred

- **Statistics across repeated runs.** One run per arm is one sample of a
  stochastic system. The seed and the artifact shape are designed to make
  repeated runs cheap to pool, but no aggregation across run directories is
  implemented, and no significance test — a confidence interval over three runs
  would be worse than reading the three.
- **Token and cost capture.** Usage is recorded by `services/model_telemetry.py`
  rather than on the reply provenance record, so most runs will report
  `turns_reporting: 0`. The fields are wired and report their own coverage
  rather than a misleading zero. Joining the telemetry rows to a run is a
  follow-up.
- **An adaptive-customer A/B.** Refused here on purpose (different inputs per
  arm). Worth building separately, against the existing
  `services/adaptive_customer.py` seam, once the scripted comparison is
  established.
- **Full media execution in `I_natural_media_interest`.** Proposal and
  fan-visible copy are measured; whether delivery completes depends on the
  creator's catalog fixtures.
- **A dashboard.** The artifacts are JSON and Markdown on purpose.
