-- Vault synchronisation that admits when it was interrupted.
--
-- Audit reference: VAULT-003. Additive and idempotent; safe to re-run.
--
-- What was already correct
-- -----------------------
-- Recovery. An interrupted sync never stamps last_vault_sync_at, so the
-- cooldown does not start and the autosync scheduler picks the creator up on a
-- later pass. No media is lost and nothing needs unsticking.
--
-- What was wrong
-- --------------
-- What the operator was told. Run state lived only in the _vault_sync_state
-- dict, so a restart mid-sync left the status endpoint answering "idle" —
-- indistinguishable from "this creator has never been asked to sync". An
-- operator watching a large vault import saw a progress bar vanish and had no
-- way to tell whether it finished, failed, or was cut off.
--
-- The fix is three columns, not a queue. Which process a run belongs to is the
-- only fact process memory was holding that the database was not, so that is
-- the only fact added.
--
--   started_at set, finished_at null, owner is THIS process  -> running
--   started_at set, finished_at null, owner is a DEAD process -> interrupted
--   finished_at >= started_at                                 -> idle
--
-- "Interrupted" is a description of the past, not a state anything has to
-- clear: the next autosync or manual run overwrites started_at and proceeds
-- exactly as it does today. Deliberately not a 'failed' flag — a run that was
-- cut off by a deploy has not failed, and telling an operator it has invites
-- them to intervene where nothing is wrong.

alter table public.creators
    -- When the current (or most recent) vault synchronisation began.
    add column if not exists vault_sync_started_at timestamptz null,
    -- When it reached a terminal state. Null while a run is in flight.
    add column if not exists vault_sync_finished_at timestamptz null,
    -- Which process owns the in-flight run. A random per-boot identifier, not a
    -- hostname or PID: two containers on one host, or a reused PID, must not be
    -- mistaken for each other. Compared only for equality, never parsed.
    add column if not exists vault_sync_owner text null;

comment on column public.creators.vault_sync_owner is
    'VAULT-003: per-boot id of the process running the current vault sync. '
    'A value that is not the live process means the run was interrupted by a '
    'restart, which is reported as interrupted rather than idle. Recovery is '
    'unchanged: the next scheduled or manual run proceeds normally.';

-- The autosync scheduler already reads creators by id; no new index is
-- justified for three columns read alongside the row that is being fetched
-- anyway. Deliberately not adding one on speculation (DB-000).
