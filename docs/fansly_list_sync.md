# Fansly list mirroring

## Product intent

Agencies already build and maintain Lists — VIP, Whales, Buyers, Re-engage —
directly on the creator's Fansly account. Cleopatra pulls those lists in so an
agency does not have to recreate them by hand before using list-based targeting
in Auto Audience and re-engagement.

**Read only, one direction.** Cleopatra never creates, renames, or deletes a
list on Fansly. It also never modifies a locally created Cleopatra list.

## Data model

`db/fansly_lists_v1.sql` extends the existing tables rather than adding parallel
ones, so every existing list rule, RLS policy and join keeps working.

`fan_lists`

| Column | Meaning |
| --- | --- |
| `source` | `'local'` (operator-created, the pre-existing rows) or `'fansly'` (mirror) |
| `external_list_id` | the remote Fansly list ID; the only stable mapping key |
| `external_synced_at` | last time this mirror was reconciled |
| `external_archived_at` | set when the remote list stopped being returned |
| `external_item_count` | remote member count, for the "not yet imported" gap |

`fan_list_members`

| Column | Meaning |
| --- | --- |
| `source` | `'local'` or `'fansly'` — which side owns this membership |
| `external_synced_at` | last time it was confirmed remotely |

`creators`: `last_fansly_lists_sync_at`, `fansly_lists_sync_error`,
`fansly_lists_sync_failed_at`, matching the existing
`last_fansly_audience_sync_at` convention.

Constraints that carry real weight:

* `fan_lists_creator_external_id_key` — a **partial unique index** on
  `(creator_id, external_list_id)`. This is the invariant repeated syncs depend
  on. Partial, so it never constrains local lists, and creator-scoped, so two
  agencies mirroring the same remote ID do not collide.
* `fan_lists_external_id_required` — a `'fansly'` row must have an
  `external_list_id` and a `'local'` row must not. Without it a mirror with a
  null external id would duplicate on every sync.
* `fan_lists_source_check`, `fan_list_members_source_check`.

`tests/test_fansly_lists_schema.py` applies this migration to a real PostgreSQL
and proves the constraints actually reject bad data. CI provides the database
through a `postgres:16` service; the tests skip when `TEST_DATABASE_URL` is
unset, so a local `pytest` needs no server.

RLS is unchanged: `db/tenant_isolation_v1.sql` enumerates every table with a
`creator_id` rather than a hand-maintained list, so `fan_lists` and
`fan_list_members` are already covered and the new columns inherit those
policies.

## Reconciliation

`services/fansly_lists.py` applies one full remote snapshot per run.

**Lists**

1. Page every remote list (`fetch_remote_lists`), keyed on `external_list_id`.
2. Upsert one mirror per remote ID. A **rename updates that same row**, so
   `VIP` → `VIP Buyers` never creates a second list and never loses membership.
   Two remote lists that happen to share a name stay distinct because the key is
   the ID, not the name.
3. A mirror that stops being returned is **archived, not deleted**
   (`external_archived_at`). Auto Audience include/exclude rules and
   re-engagement settings reference `fan_lists.id`; deleting the row would
   silently change which fans those rules select. Nothing is ever repointed to
   a different list. A list that reappears is unarchived in place.
4. Local lists are never read, written, or archived by any of this.

**Membership**

1. Members are mapped by `fans.platform_fan_id` only. Usernames and display
   names are mutable and not unique and are never used.
2. A remote member Cleopatra has not imported yet is **counted and skipped**
   (`unmapped_members`). No placeholder fan row is invented — a fabricated fan
   would corrupt every downstream spend, tier and lifecycle count. A later sync
   picks it up automatically once the fan exists.
3. Membership rows written by the sync carry `source = 'fansly'`. Removals are
   scoped to `list_id` + `fan_id` + `source = 'fansly'`, so an operator who
   hand-added a fan to a mirror keeps that membership even when the fan is
   removed remotely — and a local list's memberships are never in scope at all.

Everything is idempotent: a second run with the same remote snapshot performs no
writes and reports zeroes.

## When it runs

No new scheduler. List refresh rides the existing Fansly account
synchronization lifecycle in `main.py`:

| Trigger | Behaviour |
| --- | --- |
| Creator connect / reconnect | `sync_chats_background` runs a full sync, which refreshes lists |
| Full or forced `/sync-chats/{creator_id}` | always refreshes |
| Incremental chat reconciliation (the existing scheduler) | refreshes only when the mirror is older than `FANSLY_LISTS_SYNC_INTERVAL_HOURS` (default 6) |
| `POST /creator/{id}/sync-fansly-lists` | explicit operator refresh, always runs |

The dashboard never fetches remote lists on render. It reads mirrored rows from
Supabase like it already does, and calls the refresh endpoint only when an
operator clicks the button.

A list failure never fails the surrounding chat sync; it is reported in the
`lists` key of the sync result and recorded on the creator row. A 401/403
raises `ApiFanslyAccountAccessError`, the same exception every other API Fansly
call uses, so the existing reconnect/backoff semantics apply instead of retrying
a binding that will keep failing.

## Endpoints

| Route | Purpose |
| --- | --- |
| `GET /creator/{creator_id}/fansly-lists` | mirrored lists plus last synced / failure state |
| `POST /creator/{creator_id}/sync-fansly-lists` | explicit refresh |

Both use `require_creator_path_access`. `creator_id` from the client is never
trusted on its own: the dependency confirms the caller is assigned to that
creator through `chatter_creators`, and the API Fansly account ID is read from
the creator row rather than the request. A creator belonging to another agency
returns **404**, not 403, so the existence of another agency's creator is not
revealed.

## Dashboard

Imported lists appear anywhere list targeting is chosen, labelled
`Fansly · VIP` with a small source badge so operators can see at a glance that
Cleopatra does not own the remote list. Archived mirrors are shown as such and
are not offered for new targeting, but an existing rule that references one keeps
resolving.

## API Fansly contract — needs confirmation before enabling

The transport lives in `services/apifansly.py` with every other API Fansly call
and reuses its usage metering, auth, error handling and cursor normalization.
The endpoint shape follows the `{accountId}/<resource>` convention used by
`followers`, `subscribers` and `vault/albums`:

```
GET {base}/{accountId}/lists                     -> lists
GET {base}/{accountId}/lists/{listId}/items      -> members
```

Parsing is deliberately tolerant, exactly as `list_vault_album_media` already is:
the collection is accepted under `lists` / `items` / `data` / `accountLists`, the
ID under `id` / `listId` / `_id`, the name under `label` / `name` / `title`, and
member IDs under `accountId` / `itemId` / `id` / `userId` / `followerId`.

**Confirm the two paths against the current API Fansly documentation before
enabling this in production.** If they differ, set `APIFANSLY_LISTS_PATH` and
`APIFANSLY_LIST_ITEMS_PATH` — no redeploy required. If the response shape differs
in a way the tolerant parser does not cover, the only code to change is
`parse_account_lists` / `parse_list_member_ids`, which are pure functions with
direct test coverage.

## Railway environment changes

```
FANSLY_LISTS_SYNC_ENABLED=false     # flip to true AFTER applying the migration
FANSLY_LISTS_SYNC_INTERVAL_HOURS=6
# APIFANSLY_LISTS_PATH=lists        # only if the documented path differs
# APIFANSLY_LIST_ITEMS_PATH=items
```

Defaults off, matching `FAN_INTELLIGENCE_ENABLED` and `FAN_LIFECYCLE_ENABLED`:
running the sync against a database without the new columns would fail every
pass.

## Deployment order

1. Apply `db/fansly_lists_v1.sql` in Supabase (idempotent, additive, no rewrite
   of existing rows beyond backfilling `source = 'local'`).
2. Deploy the backend with `FANSLY_LISTS_SYNC_ENABLED=false`.
3. Confirm the API Fansly list paths, then set `FANSLY_LISTS_SYNC_ENABLED=true`.
4. Deploy the dashboard. It degrades safely against a backend without the
   endpoint and against a database without the columns.
