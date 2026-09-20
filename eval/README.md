# Cleopatra evaluation tools

The realism review is an internal pre-release check. It is not a live-chat
ranker, it is not shown to agencies, and it never changes production routing.

## What “blind” means

The review document replaces model or prompt names with Candidate A, B, C, and
randomizes their order separately for every scenario. Reviewers judge the text
before opening the generated answer key, which reduces the tendency to favor a
model or prompt they already expect to win.

## Run a realism comparison

Generate candidates with the realism-focused scenarios:

```bash
python scripts/run_model_eval.py \
  --scenarios eval/realism_scenarios.json \
  --models <comma-separated-candidate-names>
```

Then turn the resulting JSON into a blind review sheet:

```bash
python scripts/build_blind_review.py \
  eval/results/<result-file>.json \
  --scenarios eval/realism_scenarios.json \
  --output eval/results/realism_review.md \
  --answer-key eval/results/realism_review_answer_key.json
```

Complete the review before opening the answer key. The useful signal is the
reviewer’s comparison: human believability, context use, creator voice,
commercial usefulness, and concrete AI tells. There is deliberately no
automatic “naturalness score” in the production reply path.

## Compare complete runtime trajectories

The conversation-core comparison runs through the real simulator and Full Auto
entry point. Use isolated `test_` fans with equivalent seeded state:

```bash
python scripts/run_trajectory_eval.py \
  --creator <creator-id> --fan <legacy-test-fan-id> \
  --core legacy --simulate-time --fail-on-uncovered

python scripts/run_trajectory_eval.py \
  --creator <creator-id> --fan <semantic-test-fan-id> \
  --core semantic_v1 --simulate-time --fail-on-uncovered
```

`--core` temporarily persists the fan selection so Full Auto and scheduled
work resolve the same runtime, and restores the previous override afterward.
The runner refuses real fans and mixed-fan trajectory files.

Provider-shadow comparison of the one-call and owner-plus-writer candidates is
still available separately and requires explicit acknowledgement of cost:

```bash
python scripts/run_candidate_provider_eval.py \
  --confirm-paid-provider-calls \
  --output evaluation_bundles/<run-name>
```

Neither command creates an automatic quality score. Human review of complete
conversations remains a separate requirement.

## Compare two conversational runtimes (Conversational Core v1)

The A/B harness runs one scripted fan trajectory through two runtimes, each
against its own `test_` fan, and writes both runs, the objective metrics and a
blind review into `eval/results/<run-id>/`. It works with one arm, so it is
usable before a candidate runtime exists:

```bash
python scripts/run_ab_trajectory_eval.py \
  --baseline semantic_v2 --creator <creator-id> \
  --provision-fans --suite conversational --simulate-time
```

Add `--candidate <core-id>` for the comparison, then:

```bash
python scripts/build_conversation_blind_review.py eval/results/<run-id>
```

The reviewer document names no runtime and says nothing about which one is
expected to win; the key is a separate file. Scenarios live in
`eval/conversational_core_scenarios.json` and are entirely synthetic. The full
experiment, the fairness and isolation rules, and the artifact shape are in
`docs/conversational_core_v1_evaluation.md`.

As everywhere else here, no automatic quality score is produced, and human
review of the complete conversations remains a separate requirement.
