-- Restart-safe chat reconciliation checkpoint.
--
-- Audit reference: API-001. Additive and idempotent; safe to re-run.
--
-- The problem
-- -----------
-- Reconciliation decided whether a chat needed a list_chat_messages call by
-- comparing the chat-list cursor against a PROCESS-MEMORY dict keyed by
-- (creator_id, group_id). A Railway restart emptied it, so every known chat
-- looked like it had never been reconciled and every one of them cost a
-- message-list call on the next pass.
--
-- At 20 creators x ~2,000 chats that is ~40,000 provider calls per deploy.
-- Deploying should not be the most expensive thing the system does.
--
-- The checkpoint
-- --------------
-- API Fansly's list_chats response already carries `lastMessageId` per chat, so
-- no cursor has to be invented: the marker the comparison needs is remote data
-- we are already paying for. It is stored per fan, because a fan row is exactly
-- one (creator_id, fansly_group_id) binding — the same key the in-memory dict
-- used — and sync_chats already reads and writes that row on every pass. The
-- checkpoint therefore costs ZERO additional round trips: it rides the select
-- that was already loading fans, and the update that was already correcting
-- display name and group binding.
--
-- Deliberately NOT a new table: a per-chat side table would need its own read,
-- its own write, its own RLS policy and its own cleanup when a fan is deleted.
-- A column inherits the fan's cascade and the fan's tenancy for free.
--
-- Deliberately NOT stuffed into an existing jsonb column (ai_summary,
-- active_session): those are model-authored documents that get overwritten
-- wholesale, which would silently drop the checkpoint and reintroduce the cold
-- resync it exists to prevent.
--
-- Failure direction
-- -----------------
-- Absent, null or unreadable checkpoint => reconcile (the pre-existing safe
-- behaviour). The checkpoint can only ever SUPPRESS a call when the remote
-- marker is byte-identical to the one we stored after a successful sync, so a
-- stale or corrupt value fails toward synchronization, never toward message
-- loss. A deleted newest message changes lastMessageId, which triggers a sync
-- rather than suppressing one.

alter table public.fans
    -- The platform lastMessageId observed at the end of the last SUCCESSFUL
    -- reconciliation of this fan's chat. Text, not uuid: it is a foreign
    -- platform identifier and its shape is not ours to constrain.
    add column if not exists chat_last_message_id text null,
    -- When that checkpoint was written. Not used to decide whether to sync —
    -- the marker alone does that — but an operator looking at a chat that
    -- stopped updating needs to see how old the agreement is.
    add column if not exists chat_last_synced_at timestamptz null;

comment on column public.fans.chat_last_message_id is
    'API-001: platform lastMessageId at the last successful chat reconciliation. '
    'Null means reconcile. Cleared when fansly_group_id changes.';

-- Row level security is applied by db/tenant_isolation_v1.sql, which discovers
-- creator-owned tables at run time; fans is already covered and these columns
-- inherit those policies. No new policy is required.
