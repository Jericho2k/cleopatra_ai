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
