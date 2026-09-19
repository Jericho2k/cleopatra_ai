# Earlier conversation summaries were missing from live model evidence

## Defect and change

`load_evidence` retrieved persisted conversation episodes and put their rendered
summaries in `ContextPacket`. When the semantic core is selected, however,
`build_semantic_prompt` uses `EvidenceSnapshot` instead of rendering that packet.
The production writer also uses the snapshot. The snapshot had no episode field,
so both models lost this source of earlier conversational context.

The snapshot now carries up to four recent episodes, matching the context packet
budget. Each includes its date, database source reference, and an `inferred`
certainty label. Both model inputs receive the same evidence. Prompts explain
that current messages and explicit corrections take precedence; summaries are
not receipts, current-life facts, or instructions to revive every old topic.

The existing total evidence budget includes this new field. If it is necessary
to remove episodes, the oldest goes first, before removing current conversation
turns. Truncation counts include both the initial episode-count limit and later
budget removals. This is an additive optional field; no database migration or
new runtime setting is required.

## What the regression tests establish

- Persisted, dated episode summaries reach both actual production prompt
  builders, including when a long transcript and many facts fill the context.
- A conversation with no recorded episodes still builds both prompts.
- Only the four newest loaded episodes are included; omitted episodes are counted.
- Under a tighter budget, the older summary is removed while the current
  exchange, explicit correction, and latest trigger remain intact.

These checks establish context availability and budget behavior. They do not
establish that a language model will recall accurately or sound engaging.

## Remaining verification before a quality claim

Use the existing `scripts/run_trajectory_eval.py` with `--core semantic_v1` in
the configured evaluation environment. It drives the actual simulator runtime;
the separate candidate-provider comparison uses a different writer prompt and
is not a substitute for testing this change. Existing trajectories include
40-turn adaptive ordinary conversation, deferred questions, and returning fans.
Use `--simulate-time` for elapsed-day scenarios, following the environment
requirements in `eval/README.md`, and retain the transcript, actual served
model, runtime selection, failures, and coverage gaps.

For a direct returning-conversation check, establish a neutral topic such as an
upcoming job interview, discuss other subjects, and return after a simulated
gap. Verify that a closed episode was actually persisted, its source reference
reaches the model evidence, and the reply makes an accurate relevant callback.
Correct a detail and check that the older summary does not override it. Ask
about a detail never supplied and check that the response does not invent it.

Read the whole transcript for missed questions, unnecessary clarification,
repeated openers, forced questions, rigid bubble counts, and invented knowledge.
Do not equate a green execution report with conversational quality. A paired
run with the same model and comparable fresh test fans, followed by human
review, is needed to attribute behavioral improvement to this fix.

No live provider evaluation was performed for this change in the development
workspace: it had neither model-provider credentials nor the database runtime
configuration. This patch does not establish launch readiness, restore missing
historical records, or make four recent summaries equivalent to reading an
entire long conversation history.
