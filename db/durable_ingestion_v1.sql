-- Durable webhook ingestion: make platform-message dedupe a database
-- guarantee instead of a read-then-write race.
--
-- Apply BEFORE deploying the matching backend commit. Both steps are safe to
-- re-run.
--
-- The webhook now returns HTTP 2xx as soon as the fan message is persisted and
-- a PROCESS_INBOUND_MESSAGE action exists, and the platform is free to redeliver
-- the same event. Redelivery must produce exactly one canonical message row and
-- exactly one processing obligation, and that has to hold when two deliveries
-- are in flight at the same moment -- which a check-then-insert cannot promise.

-- Step 1: report any pre-existing duplicates. If this returns rows, resolve them
-- before running step 2; the index creation will otherwise fail.
--
--   select fansly_message_id, count(*)
--     from public.messages
--    where fansly_message_id is not null
--    group by fansly_message_id
--   having count(*) > 1;

-- Step 2: one platform message id, one row. Partial so the many locally
-- originated rows with a null platform id are unaffected.
create unique index if not exists messages_fansly_message_id_key
    on public.messages (fansly_message_id)
    where fansly_message_id is not null;

-- The scheduled-actions dedupe key is the authority for "one effective
-- obligation per event". schedule_action() already upserts on it, and the
-- ingestion path relies on the conflict being detected rather than inserting a
-- second row, so the constraint has to exist for real.
create unique index if not exists scheduled_actions_dedupe_key_key
    on public.scheduled_actions (dedupe_key);

-- Health reads the pending queue by status and due time on every probe.
create index if not exists scheduled_actions_status_execute_at_idx
    on public.scheduled_actions (status, execute_at);
