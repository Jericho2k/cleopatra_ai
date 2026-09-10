# One-creator production smoke test

The deterministic end-to-end check to run **before** the first real agency
creator is put on the deployment, and again after any migration or deploy that
touches ingestion, delivery, or the commercial path.

One controlled creator. Not bulk traffic, not a real agency account.

## Before you start

**Use a test creator and a test fan.** Steps 12–14 send a real message to a real
Fansly conversation. Step 19 reads the vault. Nothing here should be run against
a customer's account.

**Do not run a PPV purchase against a customer.** Step 17 is optional and needs
its own test account and safe media. Skip it rather than improvise — a real
charge to a real fan is not recoverable by a rollback.

Have ready:

- an operator login for the dashboard, assigned to exactly one test creator;
- a second operator login assigned to a *different* creator (step 2);
- the test creator connected to an API Fansly account;
- a fan who has messaged that creator at least once;
- `SUPABASE_DB_URL` for the preflight, and Railway logs open.

## Step 0 — preflight (read-only, no writes)

```
SUPABASE_DB_URL='postgresql://...' python scripts/production_preflight.py
```

Every check must be PASS. `DB-000 base schema committed` may be WARN. Any FAIL
stops the smoke test: fix the schema first, because several steps below cannot
be interpreted against a half-migrated database.

---

| # | Step | What to do | Pass condition |
|---|---|---|---|
| 1 | Operator login | Log in to the dashboard | Session established, creator list loads |
| 2 | Tenant isolation | Log in as the *second* operator | The test creator is **not** visible. Both operators see only their own. |
| 3 | Webhook accepted | Have the test fan send a message | Railway logs show `[FANSLY WEBHOOK] event=messages.received`, HTTP 2xx |
| 4 | Message persisted once | Query `messages` for that `fansly_message_id` | Exactly **one** row |
| 5 | Obligation created once | Query `scheduled_actions` for `action_type='PROCESS_INBOUND_MESSAGE'` and that message | Exactly **one** row |
| 6 | Assisted suggestions | Open the conversation in the dashboard | Suggestions appear within a few seconds |
| 7 | Full Auto can be enabled | Toggle Auto on for this fan | Toggle persists; no error toast |
| 8 | Auto schedules | Have the fan send another message | A `AUTO_REPLY` row appears in `scheduled_actions`, status PENDING, `execute_at` a little in the future |
| 9 | Analyzer succeeds | Watch logs | `[ACTION TIMING]` shows a non-zero `revalidation_ms`; no analyzer error in `/health` |
| 10 | Writer succeeds | Watch logs | `[ACTION TIMING]` shows `model_calls >= 1` and `outcome=sent` |
| 11 | Expected upstream used | `python scripts/model_cache_report.py --window 1h --by upstream` | The upstream is the one you configured, not an unexpected fallback |
| 12 | Delivered once | Look at the Fansly conversation | Exactly **one** creator message arrived |
| 13 | Local DB has it | Query `messages` for `role='creator'` | One row, with a non-null `fansly_message_id` |
| 14 | Duplicate webhook is harmless | Replay the step-3 webhook delivery | Response `duplicate` or `accepted`; **no** second message row, **no** second obligation, **no** second send |
| 15 | Transient API failure | Temporarily set `APIFANSLY_BASE_URL` to an unreachable host, trigger an Auto reply, then restore it | Action goes PENDING with a future `execute_at` and a `last_error`; it is **not** FAILED; it succeeds after you restore |
| 16 | Analyzer failure sends nothing | Temporarily unset the analyzer provider key, trigger an Auto reply, then restore | **No** message reaches the fan. `/health` reports degraded. |
| 17 | *(optional, test account only)* PPV purchase | Buy a PPV with safe media on a test account | One `platform_purchase_events` row; spend increments **once**; replaying the webhook returns `duplicate` and changes nothing |
| 18 | Auto OFF cancels | Turn Auto off while an `AUTO_REPLY` is PENDING | The action resolves as skipped/cancelled; **no** message is sent |
| 19 | Fansly Lists refresh | Press Refresh on lists (needs `FANSLY_LISTS_SYNC_ENABLED=true`) | Returns a completed result. Press it twice quickly: the second returns `already_syncing`, and list membership is correct — not doubled. |
| 20 | Vault album summary | Open the Vault page | Albums and counts load |
| 21 | Health | `GET /health` | `status: ok`. If `degraded`, every reason is one you caused above. `db_executor.queued` is 0. |

## Interpreting timing

`[ACTION TIMING]` separates the two kinds of delay, and they are not the same
problem:

- `queue_wait_ms` — how long the action sat before a worker picked it up. This
  is **architecture delay**. Sustained growth here is the capacity signal.
- the composition/availability delay inside `total_processing_ms` — deliberate
  human realism. A reply arriving 40 seconds after a fan messages is the
  product working, not a performance defect. Do not tune it away.

## If a step fails

| Step | Most likely cause |
|---|---|
| 2 | `tenant_isolation_v1` / `browser_least_privilege_v1` not applied as a pair |
| 3 | `APIFANSLY_WEBHOOK_SECRET` mismatch, or `APP_ENV` unset with no secret |
| 4 | `message_platform_identity_v1` not applied |
| 5, 14 | `durable_ingestion_v1` not applied (no dedupe uniqueness) |
| 8 | Creator not connected — check `/health` for `actions_blocked_creator_not_connected` |
| 11 | `OPENROUTER_PROVIDERS` routing to an unexpected upstream |
| 17 | `purchase_identity_v1` not applied |
| 19 | `fansly_lists_v1` and `fansly_lists_single_flight_v1` not applied |

## After a restart (run once per deploy)

The API-001 check, and the cheapest signal that a deploy did not become the most
expensive thing the system does that day:

1. Note the current API Fansly call volume.
2. Redeploy.
3. Watch the first chat reconciliation pass after boot.

A creator whose conversations have not changed should produce **no**
`list_chat_messages` calls. A burst of them means `chat_sync_checkpoint_v1` is
not applied — confirm with the preflight.

Also check `GET /sync-vault-status/{creator_id}` for a creator whose vault sync
was interrupted by the restart: it should report `interrupted`, not `idle`.
