# Continuation deployment and rollback runbook

This runbook covers the continuation-correctness backend change and the
owner-only reply-trace dashboard. It does not authorize a deployment, enable
Full Auto, change message authority, or select a conversational candidate.

## Release order

1. Record the backend and dashboard commit SHAs selected for release. Confirm
   the backend `/build` endpoint reports the expected SHA and flags in the
   target environment before evaluating any conversation result.
2. Apply database migrations in `db/migration_order.txt` order. For this change,
   `assisted_provenance_v1.sql` must exist before
   `assisted_provenance_atomic_consume_v1.sql`. Do not deploy the new backend
   until the atomic consume RPC is present.
3. Verify the RPC in the target database with two concurrent transactions for
   one token. Exactly one result must return the record. The other must return
   no row. Run the schema suite with `TEST_DATABASE_URL` against a disposable
   database before touching production.
4. Deploy the backend with all existing autonomy and agency-access flags left
   unchanged. Verify:
   - owner authentication can read the reply-trace route;
   - agency authentication receives `403` even if the simulator master switch
     is enabled;
   - an Assisted suggestion token can be redeemed once across two replicas;
   - an answered thread disappears from the writer packet on the same turn;
   - a correction supersedes only the exact referenced thread in the same
     creator/fan scope.
5. Deploy the dashboard. Verify the trace inspector as an owner and verify it
   is absent from an agency session. A missing trace must show its explicit
   unavailable reason rather than inventing attribution.
6. Observe Assisted traffic only. This release is not approval to enable Full
   Auto, send candidate-evaluation output, or mutate external state from the
   evaluation runner.

## Rollback

1. Roll back the dashboard first if the owner panel is unusable. This does not
   affect message generation or delivery.
2. Roll back the backend application if memory extraction or Assisted
   provenance is unhealthy. The new consume RPC and table are backward
   compatible and may remain installed during application rollback.
3. Do not drop the consume RPC or provenance table during an incident. Dropping
   them removes investigation evidence and can break a still-running replica.
   Remove schema only in a later planned migration after every old and new
   backend replica is gone.
4. If the new backend was deployed before the RPC by mistake, immediately
   restore the previous backend version; do not weaken `redeem` to trust its
   process cache.
5. Preserve logs, reply traces, the deployed SHA/flag snapshot, and the exact
   affected creator/fan/turn identifiers for the incident review.

## Candidate provider-shadow evaluation

The evaluation runner makes paid inference calls but has no database or
platform adapter. It sends zero live messages and makes zero production state
mutations.

```bash
python scripts/run_candidate_provider_eval.py \
  --confirm-paid-provider-calls \
  --scenarios eval/decision_scenarios.json \
  --output evaluation_bundles/<run-id>
```

Run from a clean commit. The command refuses a dirty source tree unless
`--allow-dirty` is explicit. The bundle contains the source SHA, active flags,
migration-order digest, scenario digest and names, model targets, complete
decisions/replies, disagreements, and provider latency/token/cost records. An
operation marked `operation_permitted_in_dry_run` was never executed.

Required external inputs before this can produce real evidence:

- provider credentials and an approved inference budget;
- an approved, non-explicit scenario/export set with no unnecessary personal
  data;
- human reviewers and selection criteria;
- a disposable Postgres URL for schema/concurrency verification;
- owner and agency test accounts for deployed browser verification;
- deployment access and explicit release approval.

No candidate is promoted automatically. Disagreements and critical failures
are evidence for human review, not a leaderboard.
